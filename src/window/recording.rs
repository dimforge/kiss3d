//! Video recording functionality.

use std::path::Path;

use image::{ImageBuffer, Rgb};

use super::Window;

/// Configuration options for video recording.
///
/// Use this to customize recording behavior such as frame skipping.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RecordingConfig {
    /// Record every Nth frame. Set to 1 to record every frame,
    /// 2 to record every other frame, etc.
    /// Default: 1
    pub frame_skip: u32,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self { frame_skip: 1 }
    }
}

impl RecordingConfig {
    /// Creates a new recording config with default settings (every frame).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets how many frames to skip between captures.
    /// 1 = every frame, 2 = every other frame, etc.
    pub fn with_frame_skip(mut self, skip: u32) -> Self {
        self.frame_skip = skip.max(1);
        self
    }
}

/// State for video recording.
pub(crate) struct RecordingState {
    pub(crate) frames: Vec<ImageBuffer<Rgb<u8>, Vec<u8>>>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) config: RecordingConfig,
    pub(crate) paused: bool,
    pub(crate) frame_counter: u32,
}

impl Window {
    /// Starts recording frames for a screencast with default settings.
    ///
    /// After calling this method, each frame rendered will be captured and stored.
    /// Call `end_recording` to stop recording and encode the frames to an MP4 video file.
    ///
    /// **Note:** This feature requires the `recording` feature to be enabled.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// window.begin_recording();
    /// // Render some frames...
    /// # for _ in 0..60 {
    /// #     window.render().await;
    /// # }
    /// window.end_recording("output.mp4", 30).unwrap();
    /// # }
    /// ```
    pub fn begin_recording(&mut self) {
        self.begin_recording_with_config(RecordingConfig::default());
    }

    /// Starts recording frames for a screencast with custom configuration.
    ///
    /// # Arguments
    /// * `config` - Recording configuration specifying frame skip, etc.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::{Window, RecordingConfig};
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// // Record every 2nd frame (reduces file size and encoding time)
    /// let config = RecordingConfig::new()
    ///     .with_frame_skip(2);
    /// window.begin_recording_with_config(config);
    /// # for _ in 0..60 {
    /// #     window.render().await;
    /// # }
    /// window.end_recording("output.mp4", 30).unwrap();
    /// # }
    /// ```
    pub fn begin_recording_with_config(&mut self, config: RecordingConfig) {
        let (width, height) = self.canvas.size();
        self.recording = Some(RecordingState {
            frames: Vec::new(),
            width,
            height,
            config,
            paused: false,
            frame_counter: 0,
        });
    }

    /// Returns whether recording is currently active.
    ///
    /// **Note:** This feature requires the `recording` feature to be enabled.
    pub fn is_recording(&self) -> bool {
        self.recording.is_some()
    }

    /// Returns whether recording is currently paused.
    ///
    /// **Note:** This feature requires the `recording` feature to be enabled.
    pub fn is_recording_paused(&self) -> bool {
        self.recording.as_ref().is_some_and(|r| r.paused)
    }

    /// Pauses the current recording.
    ///
    /// While paused, frames will not be captured. Call `resume_recording` to continue.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// window.begin_recording();
    /// // Record some frames...
    /// # for _ in 0..30 { window.render().await; }
    /// window.pause_recording();
    /// // These frames won't be recorded
    /// # for _ in 0..30 { window.render().await; }
    /// window.resume_recording();
    /// // Continue recording...
    /// # for _ in 0..30 { window.render().await; }
    /// window.end_recording("output.mp4", 30).unwrap();
    /// # }
    /// ```
    pub fn pause_recording(&mut self) {
        if let Some(ref mut recording) = self.recording {
            recording.paused = true;
        }
    }

    /// Resumes a paused recording.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// window.begin_recording();
    /// window.pause_recording();
    /// // ... do something without recording ...
    /// window.resume_recording();
    /// # window.end_recording("output.mp4", 30).unwrap();
    /// # }
    /// ```
    pub fn resume_recording(&mut self) {
        if let Some(ref mut recording) = self.recording {
            recording.paused = false;
        }
    }

    /// Stops recording and encodes the captured frames to an MP4 video file.
    ///
    /// This method consumes all recorded frames and encodes them using H.264 codec
    /// with proper compression via FFmpeg (through the `video-rs` crate).
    ///
    /// **Note:** This feature requires the `recording` feature to be enabled and
    /// FFmpeg libraries to be installed on the system.
    ///
    /// # Arguments
    /// * `path` - The output file path for the video (should end in `.mp4`)
    /// * `fps` - The frames per second for the output video
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(String)` with an error message if encoding fails
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// window.begin_recording();
    /// for _ in 0..120 {
    ///     // Animate your scene...
    ///     window.render().await;
    /// }
    /// // Save as 30fps video (120 frames = 4 seconds)
    /// window.end_recording("animation.mp4", 30).unwrap();
    /// # }
    /// ```
    pub fn end_recording<P: AsRef<Path>>(&mut self, path: P, fps: u32) -> Result<(), String> {
        use ffmpeg::{
            codec, encoder, format, frame, software::scaling, Dictionary, Packet, Rational,
        };
        use ffmpeg_the_third as ffmpeg;

        let recording = self
            .recording
            .take()
            .ok_or_else(|| "No recording in progress".to_string())?;

        if recording.frames.is_empty() {
            return Err("No frames were recorded".to_string());
        }

        let width = recording.width;
        let height = recording.height;

        // Initialize FFmpeg (safe to call multiple times)
        ffmpeg::init().map_err(|e| format!("Failed to initialize FFmpeg: {}", e))?;

        // Create output context
        let mut octx =
            format::output(&path).map_err(|e| format!("Failed to create output context: {}", e))?;

        // Check if global header is required before borrowing octx mutably
        let global_header = octx.format().flags().contains(format::Flags::GLOBAL_HEADER);

        // Find H.264 encoder
        let codec = encoder::find(codec::Id::H264).ok_or_else(|| {
            "H.264 encoder not found. Install FFmpeg with libx264 support.".to_string()
        })?;

        // Add video stream
        let mut ost = octx
            .add_stream(Some(codec))
            .map_err(|e| format!("Failed to add stream: {}", e))?;

        let ost_index = ost.index();

        // Configure encoder
        let mut encoder_ctx = codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()
            .map_err(|e| format!("Failed to create encoder context: {}", e))?;

        encoder_ctx.set_width(width);
        encoder_ctx.set_height(height);
        encoder_ctx.set_format(format::Pixel::YUV420P);
        encoder_ctx.set_time_base(Rational::new(1, fps as i32));
        encoder_ctx.set_frame_rate(Some(Rational::new(fps as i32, 1)));

        // Set global header flag if required by container format
        if global_header {
            encoder_ctx.set_flags(codec::Flags::GLOBAL_HEADER);
        }

        // Open encoder with x264 preset
        let mut x264_opts = Dictionary::new();
        x264_opts.set("preset", "medium");
        x264_opts.set("crf", "23");
        let mut encoder = encoder_ctx
            .open_with(x264_opts)
            .map_err(|e| format!("Failed to open encoder: {}", e))?;

        // Set stream parameters from encoder
        ost.set_parameters(codec::Parameters::from(&encoder));

        // Write header
        octx.write_header()
            .map_err(|e| format!("Failed to write header: {}", e))?;

        // Create scaler to convert RGB24 to YUV420P
        let mut scaler = scaling::Context::get(
            format::Pixel::RGB24,
            width,
            height,
            format::Pixel::YUV420P,
            width,
            height,
            scaling::Flags::BILINEAR,
        )
        .map_err(|e| format!("Failed to create scaler: {}", e))?;

        let ost_time_base = octx.stream(ost_index).unwrap().time_base();

        // Encode each frame
        for (i, img_frame) in recording.frames.into_iter().enumerate() {
            // Create RGB frame from captured image
            let raw_data: Vec<u8> = img_frame.into_raw();

            let mut rgb_frame = frame::Video::new(format::Pixel::RGB24, width, height);
            rgb_frame.data_mut(0).copy_from_slice(&raw_data);

            // Scale to YUV420P
            let mut yuv_frame = frame::Video::empty();
            scaler
                .run(&rgb_frame, &mut yuv_frame)
                .map_err(|e| format!("Failed to scale frame: {}", e))?;

            // Set PTS (presentation timestamp)
            yuv_frame.set_pts(Some(i as i64));

            // Send frame to encoder
            encoder
                .send_frame(&yuv_frame)
                .map_err(|e| format!("Failed to send frame: {}", e))?;

            // Receive and write encoded packets
            let mut packet = Packet::empty();
            while encoder.receive_packet(&mut packet).is_ok() {
                packet.set_stream(ost_index);
                packet.rescale_ts(Rational::new(1, fps as i32), ost_time_base);
                packet
                    .write_interleaved(&mut octx)
                    .map_err(|e| format!("Failed to write packet: {}", e))?;
            }
        }

        // Flush encoder
        encoder
            .send_eof()
            .map_err(|e| format!("Failed to send EOF: {}", e))?;

        let mut packet = Packet::empty();
        while encoder.receive_packet(&mut packet).is_ok() {
            packet.set_stream(ost_index);
            packet.rescale_ts(Rational::new(1, fps as i32), ost_time_base);
            packet
                .write_interleaved(&mut octx)
                .map_err(|e| format!("Failed to write packet: {}", e))?;
        }

        // Write trailer
        octx.write_trailer()
            .map_err(|e| format!("Failed to write trailer: {}", e))?;

        Ok(())
    }

    /// Captures the current frame if recording is active, not paused, and frame skip allows.
    ///
    /// This is called automatically during `render()` when recording is enabled.
    pub(crate) fn capture_frame_if_recording(&mut self) {
        // Check if we should capture this frame
        let should_capture = if let Some(ref mut recording) = self.recording {
            if recording.paused {
                false
            } else {
                recording.frame_counter += 1;
                // Capture if frame_counter matches the skip interval
                (recording.frame_counter - 1) % recording.config.frame_skip == 0
            }
        } else {
            false
        };

        if should_capture {
            let frame = self.snap_image();
            let (current_width, current_height) = self.canvas.size();

            // Now we can mutably borrow recording
            if let Some(ref mut recording) = self.recording {
                // Check if window was resized during recording
                if current_width != recording.width || current_height != recording.height {
                    // For now, we'll just capture at current size
                    // A more robust solution would resize frames or fail
                    recording.width = current_width;
                    recording.height = current_height;
                }
                recording.frames.push(frame);
            }
        }
    }
}
