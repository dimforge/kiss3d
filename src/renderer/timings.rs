//! Per-frame render timings: GPU timestamp queries for the render passes, plus
//! CPU wall-clock for the submit/present calls and the whole frame.
//!
//! - **GPU times** ([`RenderTimings::gpu_steps`]) come from `wgpu` timestamp
//!   queries written at the start/end of the principal render passes (shadows,
//!   opaque, transparent, tonemap for the rasterizer; trace, denoise, tonemap for
//!   the path tracer). They are the *actual* GPU execution time of those passes.
//!   They require the [`TIMESTAMP_QUERY`](wgpu::Features::TIMESTAMP_QUERY) device
//!   feature; on platforms/adapters that don't support it (e.g. WebGL2, some
//!   web/mobile GPUs) GPU timing is disabled and `gpu_steps` is `None`.
//! - **CPU times** are wall-clock ([`web_time`]) around the queue `submit` and the
//!   `present` calls — the parts of the frame that actually run on the CPU and can
//!   block — plus the `total` time of the whole `render_*` call.
//!
//! Read the latest with [`Window::render_timings`](crate::window::Window::render_timings).
//! [`RenderTimings`] implements [`Display`](std::fmt::Display); the built-in
//! inspector shows it.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use web_time::Instant;

use crate::context::Context;

/// Wall-clock duration in milliseconds.
fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Timings captured for a single rendered frame.
///
/// See the [module docs](self) for what is measured (GPU pass times via timestamp
/// queries; CPU wall-clock for submit/present and the whole frame).
#[derive(Clone, Debug, Default)]
pub struct RenderTimings {
    /// Which renderer produced these timings (`"Rasterizer"` or `"Path tracer"`).
    pub renderer: &'static str,
    /// Total CPU wall-clock time of the whole `render_*` / `raytrace_3d` call.
    pub total: Duration,
    /// CPU wall-clock time spent in the queue `submit` call.
    pub cpu_submit: Duration,
    /// CPU wall-clock time spent in the `present` call.
    pub cpu_present: Duration,
    /// GPU execution time of each timed render pass `(name, duration)`, in order.
    /// `None` when GPU timestamp queries are unsupported on this platform, or
    /// while the first results are still in flight.
    pub gpu_steps: Option<Vec<(&'static str, Duration)>>,
}

impl RenderTimings {
    /// Sum of all GPU pass times, or `None` when GPU timing is unavailable.
    pub fn gpu_total(&self) -> Option<Duration> {
        self.gpu_steps
            .as_ref()
            .map(|s| s.iter().map(|(_, d)| *d).sum())
    }
}

impl fmt::Display for RenderTimings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} — total {:.3} ms", self.renderer, ms(self.total))?;
        write!(f, "\n  cpu submit   {:>8.3} ms", ms(self.cpu_submit))?;
        write!(f, "\n  cpu present  {:>8.3} ms", ms(self.cpu_present))?;
        match &self.gpu_steps {
            Some(steps) => {
                for (name, dur) in steps {
                    write!(f, "\n  gpu {name:<9}{:>8.3} ms", ms(*dur))?;
                }
                if let Some(total) = self.gpu_total() {
                    write!(f, "\n  gpu total    {:>8.3} ms", ms(total))?;
                }
            }
            None => write!(f, "\n  gpu timing unsupported")?,
        }
        Ok(())
    }
}

/// A small CPU stopwatch for the submit/present calls and the frame total.
pub(crate) struct CpuTimer {
    start: Instant,
}

impl CpuTimer {
    pub(crate) fn start() -> CpuTimer {
        CpuTimer {
            start: Instant::now(),
        }
    }

    /// Times a single closure (used for the submit and present calls).
    pub(crate) fn time<R>(f: impl FnOnce() -> R) -> (R, Duration) {
        let t = Instant::now();
        let r = f();
        (r, t.elapsed())
    }

    pub(crate) fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

/// Maximum number of GPU timestamp scopes (begin/end query pairs) per frame.
/// Passes beyond this are simply not timed. Shadow/denoise helpers reuse a single
/// scope name across several passes (their times are summed), so this is ample.
const MAX_SCOPES: u32 = 64;
const QUERY_COUNT: u32 = MAX_SCOPES * 2;
const BYTES: u64 = QUERY_COUNT as u64 * 8;

/// One readback buffer for a frame's resolved timestamps, mapped asynchronously.
struct Readback {
    buffer: wgpu::Buffer,
    ready: Arc<AtomicBool>,
    names: Vec<&'static str>,
    pairs: u32,
    /// `true` once `map_async` has been issued and the result not yet read back.
    pending: bool,
}

/// GPU timestamp-query timer.
///
/// Holds a timestamp [`QuerySet`](wgpu::QuerySet) plus a couple of readback
/// buffers, and hands out per-pass `timestamp_writes` for the render/compute
/// passes. Results are read back asynchronously (one frame of latency, never
/// blocking the CPU). Constructed disabled (a no-op) when the device lacks
/// [`TIMESTAMP_QUERY`](wgpu::Features::TIMESTAMP_QUERY).
pub(crate) struct GpuTimer {
    enabled: bool,
    /// Nanoseconds per timestamp tick (`Queue::get_timestamp_period`).
    period_ns: f32,
    query_set: Option<wgpu::QuerySet>,
    resolve: Option<wgpu::Buffer>,
    slots: Vec<Readback>,
    frame: usize,
    /// Query pairs allocated so far this frame.
    next_pair: u32,
    /// Scope names allocated so far this frame, in order.
    names: Vec<&'static str>,
    /// Slot written this frame (awaiting `map_async` in `after_submit`).
    wrote: Option<usize>,
    /// Most recent successfully read-back GPU step times.
    last_gpu: Option<Vec<(&'static str, Duration)>>,
}

impl GpuTimer {
    /// Creates a timer, enabling GPU timing only when the device supports
    /// timestamp queries (otherwise every method is a no-op and `gpu_steps`
    /// stays `None`).
    pub(crate) fn new() -> GpuTimer {
        let ctxt = Context::get();
        let enabled = ctxt
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY);

        if !enabled {
            return GpuTimer {
                enabled: false,
                period_ns: 0.0,
                query_set: None,
                resolve: None,
                slots: Vec::new(),
                frame: 0,
                next_pair: 0,
                names: Vec::new(),
                wrote: None,
                last_gpu: None,
            };
        }

        let query_set = ctxt.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("kiss3d_gpu_timer_queries"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERY_COUNT,
        });
        let resolve = ctxt.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kiss3d_gpu_timer_resolve"),
            size: BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slots = (0..2)
            .map(|_| Readback {
                buffer: ctxt.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("kiss3d_gpu_timer_readback"),
                    size: BYTES,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                ready: Arc::new(AtomicBool::new(false)),
                names: Vec::new(),
                pairs: 0,
                pending: false,
            })
            .collect();

        GpuTimer {
            enabled: true,
            period_ns: ctxt.queue.get_timestamp_period(),
            query_set: Some(query_set),
            resolve: Some(resolve),
            slots,
            frame: 0,
            next_pair: 0,
            names: Vec::new(),
            wrote: None,
            last_gpu: None,
        }
    }

    /// The most recent GPU step times (cloned), or `None` if unsupported / not
    /// yet available.
    pub(crate) fn last(&self) -> Option<Vec<(&'static str, Duration)>> {
        self.last_gpu.clone()
    }

    /// Starts a new frame: drives async map callbacks and reads back any ready
    /// result, then resets the per-frame scope allocation.
    pub(crate) fn begin_frame(&mut self) {
        if !self.enabled {
            return;
        }
        self.next_pair = 0;
        self.names.clear();
        self.wrote = None;
        // Non-blocking: advances pending buffer mappings without stalling.
        let _ = Context::get().device.poll(wgpu::PollType::Poll);
        self.try_collect();
    }

    /// Reads back the first slot whose mapping has completed.
    fn try_collect(&mut self) {
        let period = self.period_ns as f64;
        let mut collected: Option<Vec<(&'static str, Duration)>> = None;
        for slot in &mut self.slots {
            if !slot.pending || !slot.ready.load(Ordering::Acquire) {
                continue;
            }
            {
                let view = slot.buffer.slice(..).get_mapped_range();
                let ticks: &[u64] = bytemuck::cast_slice(&view);
                let mut out: Vec<(&'static str, Duration)> = Vec::new();
                for (i, &name) in slot.names.iter().enumerate() {
                    let begin = ticks[i * 2];
                    let end = ticks[i * 2 + 1];
                    let ns = end.saturating_sub(begin) as f64 * period;
                    let dur = Duration::from_nanos(ns as u64);
                    // Several passes can share a scope name (e.g. the per-view
                    // shadow passes); sum them, preserving first-seen order.
                    if let Some(e) = out.iter_mut().find(|(n, _)| *n == name) {
                        e.1 += dur;
                    } else {
                        out.push((name, dur));
                    }
                }
                collected = Some(out);
            }
            slot.buffer.unmap();
            slot.ready.store(false, Ordering::Release);
            slot.pending = false;
            let _ = slot.pairs;
            break;
        }
        if collected.is_some() {
            self.last_gpu = collected;
        }
    }

    /// Allocates a begin/end query pair for a render pass, returning the
    /// `timestamp_writes` to put in its descriptor (or `None` when disabled or
    /// the per-frame scope budget is exhausted).
    pub(crate) fn render_scope(
        &mut self,
        name: &'static str,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let (b, e) = self.alloc_pair(name)?;
        Some(wgpu::RenderPassTimestampWrites {
            query_set: self.query_set.as_ref().unwrap(),
            beginning_of_pass_write_index: Some(b),
            end_of_pass_write_index: Some(e),
        })
    }

    /// Like [`render_scope`](Self::render_scope) but for a compute pass.
    pub(crate) fn compute_scope(
        &mut self,
        name: &'static str,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let (b, e) = self.alloc_pair(name)?;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: self.query_set.as_ref().unwrap(),
            beginning_of_pass_write_index: Some(b),
            end_of_pass_write_index: Some(e),
        })
    }

    fn alloc_pair(&mut self, name: &'static str) -> Option<(u32, u32)> {
        if !self.enabled || self.next_pair >= MAX_SCOPES {
            return None;
        }
        let b = self.next_pair * 2;
        self.next_pair += 1;
        self.names.push(name);
        Some((b, b + 1))
    }

    /// Resolves this frame's timestamp queries into a readback buffer. Call after
    /// recording all timed passes, before submitting the encoder.
    pub(crate) fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.wrote = None;
        if !self.enabled || self.next_pair == 0 {
            return;
        }
        let slot_idx = self.frame % self.slots.len();
        if self.slots[slot_idx].pending {
            // Previous use of this slot hasn't been read back yet; skip this
            // frame's GPU timing rather than clobber a mapped buffer.
            return;
        }
        let pairs = self.next_pair;
        let qs = self.query_set.as_ref().unwrap();
        let resolve = self.resolve.as_ref().unwrap();
        encoder.resolve_query_set(qs, 0..pairs * 2, resolve, 0);
        encoder.copy_buffer_to_buffer(
            resolve,
            0,
            &self.slots[slot_idx].buffer,
            0,
            pairs as u64 * 2 * 8,
        );
        let names = self.names.clone();
        let slot = &mut self.slots[slot_idx];
        slot.names = names;
        slot.pairs = pairs;
        self.wrote = Some(slot_idx);
    }

    /// Issues the async map of the resolved buffer. Call once after the encoder
    /// has been submitted (mapping requires the copy to be in flight).
    pub(crate) fn after_submit(&mut self) {
        if let Some(slot_idx) = self.wrote.take() {
            let slot = &mut self.slots[slot_idx];
            slot.ready.store(false, Ordering::Release);
            slot.pending = true;
            let ready = slot.ready.clone();
            slot.buffer.slice(..).map_async(wgpu::MapMode::Read, move |res| {
                if res.is_ok() {
                    ready.store(true, Ordering::Release);
                }
            });
        }
        self.frame = self.frame.wrapping_add(1);
    }
}
