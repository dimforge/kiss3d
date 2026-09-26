//! iOS support: the inverted main loop.
//!
//! iOS forbids the pump model the other native platforms use. winit's
//! `run_app` calls `UIApplicationMain`, which owns the main thread and never
//! returns, and `pump_events` does not exist on the platform. So the roles
//! flip, the same way they do on the web: winit drives, and the user's
//! `async fn main` becomes a future polled once per loop turn. The renderer's
//! per-frame [`next_frame`]`().await` is the seam that makes the user's
//! `while window.render().await` loop yield back to UIKit between frames —
//! the iOS analogue of the wasm path awaiting `requestAnimationFrame`.

use super::wgpu_canvas::{push_lifecycle, LifecycleEvent};
use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::ptr;
use std::task::{Context as TaskContext, Poll, Waker};

use objc2::rc::Retained;
use objc2::{define_class, msg_send, MainThreadMarker, MainThreadOnly};
use objc2_core_foundation::CGRect;
use objc2_foundation::{
    NSDictionary, NSNotification, NSNotificationCenter, NSObject, NSObjectProtocol, NSString,
    NSValue, NSURL,
};
use objc2_ui_kit::{
    NSValueUIGeometryExtensions, UIApplicationDidFinishLaunchingNotification, UIFont,
    UIFontTextStyleBody, UIKeyInput, UIKeyboardFrameEndUserInfoKey,
    UIKeyboardWillChangeFrameNotification, UITextInputTraits, UIView,
};
use winit::application::ApplicationHandler;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

thread_local! {
    // The ActiveEventLoop of the winit callback currently on the stack. Only
    // valid inside that callback: IosApp sets it before polling the app
    // future and clears it after, and `create_window` reads it when the
    // future opens a window mid-poll.
    static ACTIVE: Cell<*const ActiveEventLoop> = const { Cell::new(ptr::null()) };
    // Turn counter that resumes stalled `next_frame` futures: a frame that
    // yielded in turn N completes its await in turn N+1.
    static FRAME: Cell<u64> = const { Cell::new(0) };
}

/// Creates a winit window from inside the running app future.
///
/// # Panics
/// Outside a `run_ios` poll — on iOS a window can only be created while
/// winit's loop is on the stack, which `#[kiss3d::main]` guarantees.
pub(crate) fn create_window(attrs: WindowAttributes) -> Window {
    ACTIVE.with(|active| {
        let ptr = active.get();
        assert!(
            !ptr.is_null(),
            "on iOS, windows can only be created inside #[kiss3d::main]'s event loop"
        );
        // Valid for the duration of the callback that set it; this is only
        // reachable from the app future, which is polled inside one.
        let event_loop = unsafe { &*ptr };
        event_loop
            .create_window(attrs)
            .expect("Failed to create window")
    })
}

/// Completes on the loop turn after its first poll. One await per rendered
/// frame is what paces the app future to the display.
pub(crate) async fn next_frame() {
    struct NextFrame(u64);

    impl Future for NextFrame {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _: &mut TaskContext<'_>) -> Poll<()> {
            FRAME.with(|frame| {
                if frame.get() == self.0 {
                    Poll::Pending
                } else {
                    Poll::Ready(())
                }
            })
        }
    }

    NextFrame(FRAME.with(Cell::get)).await;
}

struct IosApp {
    // None once the future completes; the loop is asked to exit then, but
    // UIApplicationMain does not oblige, so this also guards re-polling.
    fut: Option<Pin<Box<dyn Future<Output = ()>>>>,
    // The future first runs in `resumed` — UIKit forbids UI work before the
    // application is active — and `about_to_wait` fires earlier than that.
    started: bool,
}

impl IosApp {
    fn poll(&mut self, event_loop: &ActiveEventLoop) {
        let Some(fut) = self.fut.as_mut() else {
            return;
        };
        ACTIVE.with(|active| active.set(ptr::from_ref(event_loop)));
        // A no-op waker: the loop re-polls every turn regardless, which is
        // exactly the pacing `next_frame` encodes.
        let mut cx = TaskContext::from_waker(Waker::noop());
        let done = fut.as_mut().poll(&mut cx).is_ready();
        ACTIVE.with(|active| active.set(ptr::null()));
        if done {
            self.fut = None;
            event_loop.exit();
        }
    }
}

impl ApplicationHandler for IosApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
        if self.started {
            push_lifecycle(LifecycleEvent::Resumed);
        }
        self.started = true;
        self.poll(event_loop);
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        push_lifecycle(LifecycleEvent::Suspended);
    }

    fn memory_warning(&mut self, _event_loop: &ActiveEventLoop) {
        push_lifecycle(LifecycleEvent::LowMemory);
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: winit::event::WindowEvent,
    ) {
        super::wgpu_canvas::collect_window_event(window_id, event);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if !self.started {
            return;
        }
        FRAME.with(|frame| frame.set(frame.get() + 1));
        self.poll(event_loop);
    }
}

/// Runs a kiss3d app future under winit's iOS main loop. `#[kiss3d::main]`
/// expands to a call to this. In practice it never returns: once started,
/// `UIApplicationMain` owns the process.
pub fn run_ios(fut: impl Future<Output = ()> + 'static) {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    if let Some(mtm) = MainThreadMarker::new() {
        observe_launch_url(mtm);
        observe_keyboard(mtm);
    }
    let mut app = IosApp {
        fut: Some(Box::pin(fut)),
        started: false,
    };
    let _ = event_loop.run_app(&mut app);
}

/// Text produced by the system keyboard, drained by `poll_events` into the
/// regular event stream — typed text reaches the app the same way desktop
/// typing does.
pub(crate) enum TextEvent {
    Char(char),
    Backspace,
}

thread_local! {
    static TEXT_EVENTS: std::cell::RefCell<Vec<TextEvent>> =
        const { std::cell::RefCell::new(Vec::new()) };
    // The hidden UIKeyInput view, created on the first keyboard request and
    // kept attached: becoming/resigning first responder is what shows and
    // hides the keyboard.
    static KEY_VIEW: std::cell::RefCell<Option<Retained<KeyInputView>>> =
        const { std::cell::RefCell::new(None) };
}

pub(crate) fn take_text_events() -> Vec<TextEvent> {
    TEXT_EVENTS.with(|events| events.borrow_mut().drain(..).collect())
}

// winit has no IME support on iOS, so the keyboard is summoned the way UIKit
// intends: only a first responder that adopts UIKeyInput gets one, so a
// zero-sized subview adopts it and the system delivers typed text to
// `insertText:`. The view must not be `hidden` — a hidden view is refused
// first-responder status — but at zero size it draws nothing.
define_class!(
    #[unsafe(super(UIView))]
    #[thread_kind = MainThreadOnly]
    #[name = "Kiss3dKeyInputView"]
    pub(crate) struct KeyInputView;

    /// UIResponder override: without it the keyboard can never appear.
    impl KeyInputView {
        #[unsafe(method(canBecomeFirstResponder))]
        fn can_become_first_responder(&self) -> bool {
            true
        }
    }

    unsafe impl NSObjectProtocol for KeyInputView {}

    unsafe impl UITextInputTraits for KeyInputView {}

    unsafe impl UIKeyInput for KeyInputView {
        #[unsafe(method(hasText))]
        fn has_text(&self) -> bool {
            true
        }

        #[unsafe(method(insertText:))]
        fn insert_text(&self, text: &NSString) {
            TEXT_EVENTS.with(|events| {
                let mut events = events.borrow_mut();
                for c in text.to_string().chars() {
                    events.push(TextEvent::Char(c));
                }
            });
        }

        #[unsafe(method(deleteBackward))]
        fn delete_backward(&self) {
            TEXT_EVENTS.with(|events| events.borrow_mut().push(TextEvent::Backspace));
        }
    }
);

thread_local! {
    /// The URL the app was launched with, if it was launched from one.
    static LAUNCH_URL: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
    /// The observer is kept alive for the life of the process: the
    /// notification centre does not retain it, and a dropped observer is a
    /// dangling pointer the next notification would follow.
    static LAUNCH_OBSERVER: std::cell::RefCell<Option<Retained<LaunchObserver>>> =
        const { std::cell::RefCell::new(None) };
    /// Where the keyboard will end up, in screen coordinates, as UIKit last
    /// announced it; `None` before it has ever moved.
    static KEYBOARD_FRAME: Cell<Option<CGRect>> = const { Cell::new(None) };
    /// Kept alive for the reason `LAUNCH_OBSERVER` is.
    static KEYBOARD_OBSERVER: std::cell::RefCell<Option<Retained<KeyboardObserver>>> =
        const { std::cell::RefCell::new(None) };
}

// UIKit hands the launch URL to the application delegate, and winit owns the
// delegate. Rather than take it over, this listens for the notification the
// same launch posts: its `userInfo` carries the very dictionary the delegate
// is given, so the URL is read without owning anything.
define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "Kiss3dLaunchObserver"]
    struct LaunchObserver;

    unsafe impl NSObjectProtocol for LaunchObserver {}

    impl LaunchObserver {
        #[unsafe(method(didFinishLaunching:))]
        #[expect(
            deprecated,
            reason = "the scene lifecycle it defers to is not one a kiss3d app adopts"
        )]
        fn did_finish_launching(&self, notification: &NSNotification) {
            let Some(info): Option<Retained<NSDictionary>> = notification.userInfo() else {
                return;
            };
            let key: &NSString = unsafe { objc2_ui_kit::UIApplicationLaunchOptionsURLKey };
            let Some(object) = info.objectForKey(key) else {
                return;
            };
            // The value under that key is documented as an NSURL.
            let url: &NSURL = unsafe { &*ptr::from_ref(&*object).cast::<NSURL>() };
            if let Some(text) = unsafe { url.absoluteString() } {
                LAUNCH_URL.with(|cell| *cell.borrow_mut() = Some(text.to_string()));
            }
        }
    }
);

// Every keyboard move posts `UIKeyboardWillChangeFrameNotification` with the
// frame it is heading for, hides included: a hidden keyboard's frame sits
// below the screen, so one handler covers show, hide, rotate and resize.
define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "Kiss3dKeyboardObserver"]
    struct KeyboardObserver;

    unsafe impl NSObjectProtocol for KeyboardObserver {}

    impl KeyboardObserver {
        #[unsafe(method(keyboardWillChangeFrame:))]
        fn will_change_frame(&self, notification: &NSNotification) {
            let Some(info): Option<Retained<NSDictionary>> = notification.userInfo() else {
                return;
            };
            let key: &NSString = unsafe { UIKeyboardFrameEndUserInfoKey };
            let Some(object) = info.objectForKey(key) else {
                return;
            };
            // Documented as an NSValue wrapping a CGRect.
            let value: &NSValue = unsafe { &*ptr::from_ref(&*object).cast::<NSValue>() };
            let frame = unsafe { value.CGRectValue() };
            KEYBOARD_FRAME.with(|cell| cell.set(Some(frame)));
        }
    }
);

fn observe_keyboard(mtm: MainThreadMarker) {
    let observer: Retained<KeyboardObserver> =
        unsafe { msg_send![KeyboardObserver::alloc(mtm), init] };
    unsafe {
        NSNotificationCenter::defaultCenter().addObserver_selector_name_object(
            &observer,
            objc2::sel!(keyboardWillChangeFrame:),
            Some(UIKeyboardWillChangeFrameNotification),
            None,
        );
    }
    KEYBOARD_OBSERVER.with(|cell| *cell.borrow_mut() = Some(observer));
}

/// How far up `window` the keyboard reaches, in points; zero while it is down.
///
/// The frame is converted into the view's own coordinates, as Apple asks: a
/// screen-space frame is wrong under rotation and in a window smaller than
/// the screen.
/// The `UIView` winit put this window's content in, with the marker proving
/// the thread UIKit may be touched from.
///
/// `None` off the main thread, which winit callbacks (and so the app future)
/// never leave, or on a window that is not a UIKit one.
fn ui_view(window: &Window) -> Option<(MainThreadMarker, &UIView)> {
    use wgpu::rwh::{HasWindowHandle, RawWindowHandle};

    let mtm = MainThreadMarker::new()?;
    let handle = window.window_handle().ok()?;
    let RawWindowHandle::UiKit(ui_kit) = handle.as_raw() else {
        return None;
    };
    // Valid while `window` is alive, which the borrow guarantees.
    Some((mtm, unsafe { ui_kit.ui_view.cast().as_ref() }))
}

pub(crate) fn keyboard_height(window: &Window) -> f64 {
    let Some(frame) = KEYBOARD_FRAME.with(Cell::get) else {
        return 0.0;
    };
    let Some((_, view)) = ui_view(window) else {
        return 0.0;
    };
    let local = view.convertRect_fromView(frame, None);
    let bottom = view.bounds().size.height;
    (bottom - local.origin.y).clamp(0.0, local.size.height.max(0.0))
}

/// Start listening for the launch URL. Called before `UIApplicationMain`
/// takes the thread, which is the only moment early enough to hear the
/// notification it posts.
///
/// `UIApplicationLaunchOptionsURLKey` is deprecated in favour of the UIScene
/// lifecycle, and deliberately used anyway: the key is empty only for an app
/// that declares a `UIApplicationSceneManifest`, and a kiss3d app declares
/// none. Adopting scenes to read a URL would mean taking over the window
/// lifecycle winit owns.
fn observe_launch_url(mtm: MainThreadMarker) {
    let observer: Retained<LaunchObserver> = unsafe { msg_send![LaunchObserver::alloc(mtm), init] };
    unsafe {
        NSNotificationCenter::defaultCenter().addObserver_selector_name_object(
            &observer,
            objc2::sel!(didFinishLaunching:),
            Some(UIApplicationDidFinishLaunchingNotification),
            None,
        );
    }
    LAUNCH_OBSERVER.with(|cell| *cell.borrow_mut() = Some(observer));
}

/// The URL this app was launched with, taken once.
///
/// Empty unless the app was opened from a link. It is taken rather than read
/// so a game that asks twice does not act on the same launch twice.
pub fn take_launch_url() -> Option<String> {
    LAUNCH_URL.with(|cell| cell.borrow_mut().take())
}

/// `[left, top, right, bottom]` in points; zeros before the view is laid out.
pub(crate) fn safe_area(window: &Window) -> [f64; 4] {
    let Some((_, view)) = ui_view(window) else {
        return [0.0; 4];
    };
    let insets = view.safeAreaInsets();
    [insets.left, insets.top, insets.right, insets.bottom]
}

/// How much larger than standard the reader asked their text to be.
///
/// Dynamic Type, as the ratio of the body style's point size to the 17 points
/// it is at the default setting. 1.0 where UIKit cannot be asked.
pub(crate) fn text_scale() -> f32 {
    /// The body style's size at the default Dynamic Type setting.
    const STANDARD_BODY: f64 = 17.0;
    if MainThreadMarker::new().is_none() {
        return 1.0;
    }
    let font = unsafe { UIFont::preferredFontForTextStyle(UIFontTextStyleBody) };
    let size = unsafe { font.pointSize() };
    if size > 0.0 {
        (size / STANDARD_BODY) as f32
    } else {
        1.0
    }
}

/// Show or hide the system keyboard for `window`.
pub(crate) fn set_keyboard_visible(window: &Window, visible: bool) {
    let Some((mtm, parent)) = ui_view(window) else {
        return;
    };

    KEY_VIEW.with(|cell| {
        let mut cell = cell.borrow_mut();
        if visible {
            let view = cell.get_or_insert_with(|| {
                let view: Retained<KeyInputView> =
                    unsafe { msg_send![KeyInputView::alloc(mtm), init] };
                parent.addSubview(&view);
                view
            });
            unsafe { view.becomeFirstResponder() };
        } else if let Some(view) = cell.as_ref() {
            unsafe { view.resignFirstResponder() };
        }
    });
}
