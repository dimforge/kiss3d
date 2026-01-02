//! Common color constants for use with kiss3d rendering functions.
//!
//! This module provides all named CSS colors as [`Color`] values with RGBA components
//! in the range [0.0, 1.0]. The colors are taken from the
//! [SVG/CSS3 named colors](https://www.w3.org/TR/css-color-3/#svg-color).
//!
//! # Example
//! ```no_run
//! # use kiss3d::window::Window;
//! # use kiss3d::color;
//! # use glamx::Vec3;
//! # #[kiss3d::main]
//! # async fn main() {
//! # let mut window = Window::new("Example").await;
//! window.draw_line(Vec3::ZERO, Vec3::X, color::RED, 2.0, false);
//! window.draw_point(Vec3::Y, color::LIME_GREEN, 5.0);
//! # }
//! ```

pub use rgb::Rgba;

/// The color type used throughout kiss3d. RGBA with f32 components in [0.0, 1.0].
pub type Color = Rgba<f32>;

// ============================================================================
// Basic Colors
// ============================================================================

/// <div style="margin:2px 0"><span style="background-color:rgb(0, 0, 0);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Black (0, 0, 0)</div>
pub const BLACK: Color = Color::new(0.0, 0.0, 0.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 255, 255);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>White (255, 255, 255)</div>
pub const WHITE: Color = Color::new(1.0, 1.0, 1.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 0, 0);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Red (255, 0, 0)</div>
pub const RED: Color = Color::new(1.0, 0.0, 0.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(0, 255, 0);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Lime (0, 255, 0) - CSS "lime", pure green</div>
pub const LIME: Color = Color::new(0.0, 1.0, 0.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(0, 0, 255);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Blue (0, 0, 255)</div>
pub const BLUE: Color = Color::new(0.0, 0.0, 1.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 255, 0);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Yellow (255, 255, 0)</div>
pub const YELLOW: Color = Color::new(1.0, 1.0, 0.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(0, 255, 255);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Cyan (0, 255, 255)</div>
pub const CYAN: Color = Color::new(0.0, 1.0, 1.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(0, 255, 255);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Aqua (0, 255, 255) - same as cyan</div>
pub const AQUA: Color = Color::new(0.0, 1.0, 1.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 0, 255);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Magenta (255, 0, 255)</div>
pub const MAGENTA: Color = Color::new(1.0, 0.0, 1.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 0, 255);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Fuchsia (255, 0, 255) - same as magenta</div>
pub const FUCHSIA: Color = Color::new(1.0, 0.0, 1.0, 1.0);

// ============================================================================
// Gray / Grey Colors
// ============================================================================

/// <div style="margin:2px 0"><span style="background-color:rgb(128, 128, 128);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Gray (128, 128, 128)</div>
pub const GRAY: Color = Color::new(0.5019608, 0.5019608, 0.5019608, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(128, 128, 128);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Grey (128, 128, 128) - same as gray</div>
pub const GREY: Color = Color::new(0.5019608, 0.5019608, 0.5019608, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(192, 192, 192);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Silver (192, 192, 192)</div>
pub const SILVER: Color = Color::new(0.7529412, 0.7529412, 0.7529412, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(169, 169, 169);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark gray (169, 169, 169)</div>
pub const DARK_GRAY: Color = Color::new(0.6627451, 0.6627451, 0.6627451, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(169, 169, 169);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark grey (169, 169, 169) - same as dark gray</div>
pub const DARK_GREY: Color = Color::new(0.6627451, 0.6627451, 0.6627451, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(211, 211, 211);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Light gray (211, 211, 211)</div>
pub const LIGHT_GRAY: Color = Color::new(0.827451, 0.827451, 0.827451, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(211, 211, 211);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Light grey (211, 211, 211) - same as light gray</div>
pub const LIGHT_GREY: Color = Color::new(0.827451, 0.827451, 0.827451, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(105, 105, 105);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dim gray (105, 105, 105)</div>
pub const DIM_GRAY: Color = Color::new(0.4117647, 0.4117647, 0.4117647, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(105, 105, 105);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dim grey (105, 105, 105) - same as dim gray</div>
pub const DIM_GREY: Color = Color::new(0.4117647, 0.4117647, 0.4117647, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(119, 136, 153);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Light slate gray (119, 136, 153)</div>
pub const LIGHT_SLATE_GRAY: Color = Color::new(0.46666667, 0.53333336, 0.6, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(119, 136, 153);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Light slate grey (119, 136, 153) - same as light slate gray</div>
pub const LIGHT_SLATE_GREY: Color = Color::new(0.46666667, 0.53333336, 0.6, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(112, 128, 144);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Slate gray (112, 128, 144)</div>
pub const SLATE_GRAY: Color = Color::new(0.4392157, 0.5019608, 0.5647059, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(112, 128, 144);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Slate grey (112, 128, 144) - same as slate gray</div>
pub const SLATE_GREY: Color = Color::new(0.4392157, 0.5019608, 0.5647059, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(47, 79, 79);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark slate gray (47, 79, 79)</div>
pub const DARK_SLATE_GRAY: Color = Color::new(0.18431373, 0.30980393, 0.30980393, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(47, 79, 79);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark slate grey (47, 79, 79) - same as dark slate gray</div>
pub const DARK_SLATE_GREY: Color = Color::new(0.18431373, 0.30980393, 0.30980393, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(220, 220, 220);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Gainsboro (220, 220, 220)</div>
pub const GAINSBORO: Color = Color::new(0.8627451, 0.8627451, 0.8627451, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(245, 245, 245);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>White smoke (245, 245, 245)</div>
pub const WHITE_SMOKE: Color = Color::new(0.9607843, 0.9607843, 0.9607843, 1.0);

// ============================================================================
// Pink Colors
// ============================================================================

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 192, 203);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Pink (255, 192, 203)</div>
pub const PINK: Color = Color::new(1.0, 0.7529412, 0.79607844, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 182, 193);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Light pink (255, 182, 193)</div>
pub const LIGHT_PINK: Color = Color::new(1.0, 0.7137255, 0.75686276, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 105, 180);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Hot pink (255, 105, 180)</div>
pub const HOT_PINK: Color = Color::new(1.0, 0.4117647, 0.7058824, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 20, 147);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Deep pink (255, 20, 147)</div>
pub const DEEP_PINK: Color = Color::new(1.0, 0.078431375, 0.5764706, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(199, 21, 133);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Medium violet red (199, 21, 133)</div>
pub const MEDIUM_VIOLET_RED: Color = Color::new(0.78039217, 0.08235294, 0.52156866, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(219, 112, 147);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Pale violet red (219, 112, 147)</div>
pub const PALE_VIOLET_RED: Color = Color::new(0.85882354, 0.4392157, 0.5764706, 1.0);

// ============================================================================
// Red Colors
// ============================================================================

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 160, 122);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Light salmon (255, 160, 122)</div>
pub const LIGHT_SALMON: Color = Color::new(1.0, 0.627451, 0.47843137, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(250, 128, 114);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Salmon (250, 128, 114)</div>
pub const SALMON: Color = Color::new(0.98039216, 0.5019608, 0.44705883, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(233, 150, 122);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark salmon (233, 150, 122)</div>
pub const DARK_SALMON: Color = Color::new(0.9137255, 0.5882353, 0.47843137, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(240, 128, 128);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Light coral (240, 128, 128)</div>
pub const LIGHT_CORAL: Color = Color::new(0.9411765, 0.5019608, 0.5019608, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(205, 92, 92);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Indian red (205, 92, 92)</div>
pub const INDIAN_RED: Color = Color::new(0.8039216, 0.36078432, 0.36078432, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(220, 20, 60);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Crimson (220, 20, 60)</div>
pub const CRIMSON: Color = Color::new(0.8627451, 0.078431375, 0.23529412, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(178, 34, 34);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Fire brick (178, 34, 34)</div>
pub const FIRE_BRICK: Color = Color::new(0.69803923, 0.13333334, 0.13333334, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(139, 0, 0);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark red (139, 0, 0)</div>
pub const DARK_RED: Color = Color::new(0.54509807, 0.0, 0.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(128, 0, 0);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Maroon (128, 0, 0)</div>
pub const MAROON: Color = Color::new(0.5019608, 0.0, 0.0, 1.0);

// ============================================================================
// Orange Colors
// ============================================================================

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 165, 0);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Orange (255, 165, 0)</div>
pub const ORANGE: Color = Color::new(1.0, 0.64705884, 0.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 140, 0);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark orange (255, 140, 0)</div>
pub const DARK_ORANGE: Color = Color::new(1.0, 0.54901963, 0.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 69, 0);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Orange red (255, 69, 0)</div>
pub const ORANGE_RED: Color = Color::new(1.0, 0.27058825, 0.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 99, 71);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Tomato (255, 99, 71)</div>
pub const TOMATO: Color = Color::new(1.0, 0.3882353, 0.2784314, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 127, 80);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Coral (255, 127, 80)</div>
pub const CORAL: Color = Color::new(1.0, 0.49803922, 0.3137255, 1.0);

// ============================================================================
// Yellow Colors
// ============================================================================

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 255, 224);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Light yellow (255, 255, 224)</div>
pub const LIGHT_YELLOW: Color = Color::new(1.0, 1.0, 0.8784314, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 250, 205);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Lemon chiffon (255, 250, 205)</div>
pub const LEMON_CHIFFON: Color = Color::new(1.0, 0.98039216, 0.8039216, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(250, 250, 210);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Light goldenrod yellow (250, 250, 210)</div>
pub const LIGHT_GOLDENROD_YELLOW: Color = Color::new(0.98039216, 0.98039216, 0.8235294, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 239, 213);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Papaya whip (255, 239, 213)</div>
pub const PAPAYA_WHIP: Color = Color::new(1.0, 0.9372549, 0.8352941, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 228, 181);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Moccasin (255, 228, 181)</div>
pub const MOCCASIN: Color = Color::new(1.0, 0.89411765, 0.70980394, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 218, 185);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Peach puff (255, 218, 185)</div>
pub const PEACH_PUFF: Color = Color::new(1.0, 0.85490197, 0.7254902, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(238, 232, 170);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Pale goldenrod (238, 232, 170)</div>
pub const PALE_GOLDENROD: Color = Color::new(0.93333334, 0.9098039, 0.6666667, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(240, 230, 140);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Khaki (240, 230, 140)</div>
pub const KHAKI: Color = Color::new(0.9411765, 0.9019608, 0.54901963, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(189, 183, 107);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark khaki (189, 183, 107)</div>
pub const DARK_KHAKI: Color = Color::new(0.7411765, 0.7176471, 0.41960785, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 215, 0);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Gold (255, 215, 0)</div>
pub const GOLD: Color = Color::new(1.0, 0.84313726, 0.0, 1.0);

// ============================================================================
// Brown Colors
// ============================================================================

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 248, 220);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Cornsilk (255, 248, 220)</div>
pub const CORNSILK: Color = Color::new(1.0, 0.972549, 0.8627451, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 235, 205);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Blanched almond (255, 235, 205)</div>
pub const BLANCHED_ALMOND: Color = Color::new(1.0, 0.92156863, 0.8039216, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 228, 196);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Bisque (255, 228, 196)</div>
pub const BISQUE: Color = Color::new(1.0, 0.89411765, 0.76862746, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 222, 173);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Navajo white (255, 222, 173)</div>
pub const NAVAJO_WHITE: Color = Color::new(1.0, 0.87058824, 0.6784314, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(245, 222, 179);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Wheat (245, 222, 179)</div>
pub const WHEAT: Color = Color::new(0.9607843, 0.87058824, 0.7019608, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(222, 184, 135);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Burly wood (222, 184, 135)</div>
pub const BURLY_WOOD: Color = Color::new(0.87058824, 0.72156864, 0.5294118, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(210, 180, 140);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Tan (210, 180, 140)</div>
pub const TAN: Color = Color::new(0.8235294, 0.7058824, 0.54901963, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(188, 143, 143);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Rosy brown (188, 143, 143)</div>
pub const ROSY_BROWN: Color = Color::new(0.7372549, 0.56078434, 0.56078434, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(244, 164, 96);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Sandy brown (244, 164, 96)</div>
pub const SANDY_BROWN: Color = Color::new(0.95686275, 0.6431373, 0.3764706, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(218, 165, 32);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Goldenrod (218, 165, 32)</div>
pub const GOLDENROD: Color = Color::new(0.85490197, 0.64705884, 0.1254902, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(184, 134, 11);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark goldenrod (184, 134, 11)</div>
pub const DARK_GOLDENROD: Color = Color::new(0.72156864, 0.5254902, 0.043137256, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(205, 133, 63);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Peru (205, 133, 63)</div>
pub const PERU: Color = Color::new(0.8039216, 0.52156866, 0.24705882, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(210, 105, 30);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Chocolate (210, 105, 30)</div>
pub const CHOCOLATE: Color = Color::new(0.8235294, 0.4117647, 0.11764706, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(139, 69, 19);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Saddle brown (139, 69, 19)</div>
pub const SADDLE_BROWN: Color = Color::new(0.54509807, 0.27058825, 0.07450981, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(160, 82, 45);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Sienna (160, 82, 45)</div>
pub const SIENNA: Color = Color::new(0.627451, 0.32156864, 0.1764706, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(165, 42, 42);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Brown (165, 42, 42)</div>
pub const BROWN: Color = Color::new(0.64705884, 0.16470589, 0.16470589, 1.0);

// ============================================================================
// Green Colors
// ============================================================================

/// <div style="margin:2px 0"><span style="background-color:rgb(0, 128, 0);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Green (0, 128, 0) - CSS "green", darker than lime</div>
pub const GREEN: Color = Color::new(0.0, 0.5019608, 0.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(0, 100, 0);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark green (0, 100, 0)</div>
pub const DARK_GREEN: Color = Color::new(0.0, 0.39215687, 0.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(85, 107, 47);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark olive green (85, 107, 47)</div>
pub const DARK_OLIVE_GREEN: Color = Color::new(0.33333334, 0.41960785, 0.18431373, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(34, 139, 34);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Forest green (34, 139, 34)</div>
pub const FOREST_GREEN: Color = Color::new(0.13333334, 0.54509807, 0.13333334, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(46, 139, 87);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Sea green (46, 139, 87)</div>
pub const SEA_GREEN: Color = Color::new(0.18039216, 0.54509807, 0.34117648, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(128, 128, 0);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Olive (128, 128, 0)</div>
pub const OLIVE: Color = Color::new(0.5019608, 0.5019608, 0.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(107, 142, 35);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Olive drab (107, 142, 35)</div>
pub const OLIVE_DRAB: Color = Color::new(0.41960785, 0.5568628, 0.13725491, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(60, 179, 113);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Medium sea green (60, 179, 113)</div>
pub const MEDIUM_SEA_GREEN: Color = Color::new(0.23529412, 0.7019608, 0.44313726, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(50, 205, 50);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Lime green (50, 205, 50)</div>
pub const LIME_GREEN: Color = Color::new(0.19607843, 0.8039216, 0.19607843, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(144, 238, 144);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Light green (144, 238, 144)</div>
pub const LIGHT_GREEN: Color = Color::new(0.5647059, 0.93333334, 0.5647059, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(152, 251, 152);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Pale green (152, 251, 152)</div>
pub const PALE_GREEN: Color = Color::new(0.59607846, 0.9843137, 0.59607846, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(143, 188, 143);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark sea green (143, 188, 143)</div>
pub const DARK_SEA_GREEN: Color = Color::new(0.56078434, 0.7372549, 0.56078434, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(0, 250, 154);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Medium spring green (0, 250, 154)</div>
pub const MEDIUM_SPRING_GREEN: Color = Color::new(0.0, 0.98039216, 0.6039216, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(0, 255, 127);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Spring green (0, 255, 127)</div>
pub const SPRING_GREEN: Color = Color::new(0.0, 1.0, 0.49803922, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(124, 252, 0);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Lawn green (124, 252, 0)</div>
pub const LAWN_GREEN: Color = Color::new(0.4862745, 0.9882353, 0.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(127, 255, 0);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Chartreuse (127, 255, 0)</div>
pub const CHARTREUSE: Color = Color::new(0.49803922, 1.0, 0.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(102, 205, 170);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Medium aquamarine (102, 205, 170)</div>
pub const MEDIUM_AQUAMARINE: Color = Color::new(0.4, 0.8039216, 0.6666667, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(173, 255, 47);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Green yellow (173, 255, 47)</div>
pub const GREEN_YELLOW: Color = Color::new(0.6784314, 1.0, 0.18431373, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(154, 205, 50);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Yellow green (154, 205, 50)</div>
pub const YELLOW_GREEN: Color = Color::new(0.6039216, 0.8039216, 0.19607843, 1.0);

// ============================================================================
// Cyan Colors
// ============================================================================

/// <div style="margin:2px 0"><span style="background-color:rgb(0, 128, 128);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Teal (0, 128, 128)</div>
pub const TEAL: Color = Color::new(0.0, 0.5019608, 0.5019608, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(0, 139, 139);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark cyan (0, 139, 139)</div>
pub const DARK_CYAN: Color = Color::new(0.0, 0.54509807, 0.54509807, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(224, 255, 255);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Light cyan (224, 255, 255)</div>
pub const LIGHT_CYAN: Color = Color::new(0.8784314, 1.0, 1.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(175, 238, 238);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Pale turquoise (175, 238, 238)</div>
pub const PALE_TURQUOISE: Color = Color::new(0.6862745, 0.93333334, 0.93333334, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(127, 255, 212);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Aquamarine (127, 255, 212)</div>
pub const AQUAMARINE: Color = Color::new(0.49803922, 1.0, 0.83137256, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(64, 224, 208);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Turquoise (64, 224, 208)</div>
pub const TURQUOISE: Color = Color::new(0.2509804, 0.8784314, 0.8156863, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(72, 209, 204);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Medium turquoise (72, 209, 204)</div>
pub const MEDIUM_TURQUOISE: Color = Color::new(0.28235295, 0.81960785, 0.8, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(0, 206, 209);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark turquoise (0, 206, 209)</div>
pub const DARK_TURQUOISE: Color = Color::new(0.0, 0.80784315, 0.81960785, 1.0);

// ============================================================================
// Blue Colors
// ============================================================================

/// <div style="margin:2px 0"><span style="background-color:rgb(0, 0, 128);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Navy (0, 0, 128)</div>
pub const NAVY: Color = Color::new(0.0, 0.0, 0.5019608, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(0, 0, 139);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark blue (0, 0, 139)</div>
pub const DARK_BLUE: Color = Color::new(0.0, 0.0, 0.54509807, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(0, 0, 205);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Medium blue (0, 0, 205)</div>
pub const MEDIUM_BLUE: Color = Color::new(0.0, 0.0, 0.8039216, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(25, 25, 112);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Midnight blue (25, 25, 112)</div>
pub const MIDNIGHT_BLUE: Color = Color::new(0.09803922, 0.09803922, 0.4392157, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(65, 105, 225);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Royal blue (65, 105, 225)</div>
pub const ROYAL_BLUE: Color = Color::new(0.25490198, 0.4117647, 0.88235295, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(70, 130, 180);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Steel blue (70, 130, 180)</div>
pub const STEEL_BLUE: Color = Color::new(0.27450982, 0.50980395, 0.7058824, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(30, 144, 255);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dodger blue (30, 144, 255)</div>
pub const DODGER_BLUE: Color = Color::new(0.11764706, 0.5647059, 1.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(0, 191, 255);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Deep sky blue (0, 191, 255)</div>
pub const DEEP_SKY_BLUE: Color = Color::new(0.0, 0.7490196, 1.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(100, 149, 237);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Cornflower blue (100, 149, 237)</div>
pub const CORNFLOWER_BLUE: Color = Color::new(0.39215687, 0.58431375, 0.92941177, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(135, 206, 235);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Sky blue (135, 206, 235)</div>
pub const SKY_BLUE: Color = Color::new(0.5294118, 0.80784315, 0.92156863, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(135, 206, 250);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Light sky blue (135, 206, 250)</div>
pub const LIGHT_SKY_BLUE: Color = Color::new(0.5294118, 0.80784315, 0.98039216, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(176, 196, 222);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Light steel blue (176, 196, 222)</div>
pub const LIGHT_STEEL_BLUE: Color = Color::new(0.6901961, 0.76862746, 0.87058824, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(173, 216, 230);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Light blue (173, 216, 230)</div>
pub const LIGHT_BLUE: Color = Color::new(0.6784314, 0.84705883, 0.9019608, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(176, 224, 230);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Powder blue (176, 224, 230)</div>
pub const POWDER_BLUE: Color = Color::new(0.6901961, 0.8784314, 0.9019608, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(95, 158, 160);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Cadet blue (95, 158, 160)</div>
pub const CADET_BLUE: Color = Color::new(0.37254903, 0.61960787, 0.627451, 1.0);

// ============================================================================
// Purple/Violet Colors
// ============================================================================

/// <div style="margin:2px 0"><span style="background-color:rgb(75, 0, 130);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Indigo (75, 0, 130)</div>
pub const INDIGO: Color = Color::new(0.29411766, 0.0, 0.50980395, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(128, 0, 128);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Purple (128, 0, 128)</div>
pub const PURPLE: Color = Color::new(0.5019608, 0.0, 0.5019608, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(139, 0, 139);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark magenta (139, 0, 139)</div>
pub const DARK_MAGENTA: Color = Color::new(0.54509807, 0.0, 0.54509807, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(148, 0, 211);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark violet (148, 0, 211)</div>
pub const DARK_VIOLET: Color = Color::new(0.5803922, 0.0, 0.827451, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(153, 50, 204);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark orchid (153, 50, 204)</div>
pub const DARK_ORCHID: Color = Color::new(0.6, 0.19607843, 0.8, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(186, 85, 211);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Medium orchid (186, 85, 211)</div>
pub const MEDIUM_ORCHID: Color = Color::new(0.7294118, 0.33333334, 0.827451, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(216, 191, 216);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Thistle (216, 191, 216)</div>
pub const THISTLE: Color = Color::new(0.84705883, 0.7490196, 0.84705883, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(221, 160, 221);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Plum (221, 160, 221)</div>
pub const PLUM: Color = Color::new(0.8666667, 0.627451, 0.8666667, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(238, 130, 238);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Violet (238, 130, 238)</div>
pub const VIOLET: Color = Color::new(0.93333334, 0.50980395, 0.93333334, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(218, 112, 214);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Orchid (218, 112, 214)</div>
pub const ORCHID: Color = Color::new(0.85490197, 0.4392157, 0.8392157, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(138, 43, 226);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Blue violet (138, 43, 226)</div>
pub const BLUE_VIOLET: Color = Color::new(0.5411765, 0.16862746, 0.8862745, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(147, 112, 219);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Medium purple (147, 112, 219)</div>
pub const MEDIUM_PURPLE: Color = Color::new(0.5764706, 0.4392157, 0.85882354, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(102, 51, 153);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Rebecca purple (102, 51, 153)</div>
pub const REBECCA_PURPLE: Color = Color::new(0.4, 0.2, 0.6, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(106, 90, 205);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Slate blue (106, 90, 205)</div>
pub const SLATE_BLUE: Color = Color::new(0.41568628, 0.3529412, 0.8039216, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(72, 61, 139);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Dark slate blue (72, 61, 139)</div>
pub const DARK_SLATE_BLUE: Color = Color::new(0.28235295, 0.23921569, 0.54509807, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(123, 104, 238);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Medium slate blue (123, 104, 238)</div>
pub const MEDIUM_SLATE_BLUE: Color = Color::new(0.48235294, 0.40784314, 0.93333334, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(230, 230, 250);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Lavender (230, 230, 250)</div>
pub const LAVENDER: Color = Color::new(0.9019608, 0.9019608, 0.98039216, 1.0);

// ============================================================================
// White/Beige Colors
// ============================================================================

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 250, 250);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Snow (255, 250, 250)</div>
pub const SNOW: Color = Color::new(1.0, 0.98039216, 0.98039216, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(240, 255, 240);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Honeydew (240, 255, 240)</div>
pub const HONEYDEW: Color = Color::new(0.9411765, 1.0, 0.9411765, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(245, 255, 250);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Mint cream (245, 255, 250)</div>
pub const MINT_CREAM: Color = Color::new(0.9607843, 1.0, 0.98039216, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(240, 255, 255);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Azure (240, 255, 255)</div>
pub const AZURE: Color = Color::new(0.9411765, 1.0, 1.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(240, 248, 255);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Alice blue (240, 248, 255)</div>
pub const ALICE_BLUE: Color = Color::new(0.9411765, 0.972549, 1.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(248, 248, 255);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Ghost white (248, 248, 255)</div>
pub const GHOST_WHITE: Color = Color::new(0.972549, 0.972549, 1.0, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 245, 238);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Seashell (255, 245, 238)</div>
pub const SEASHELL: Color = Color::new(1.0, 0.9607843, 0.93333334, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(245, 245, 220);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Beige (245, 245, 220)</div>
pub const BEIGE: Color = Color::new(0.9607843, 0.9607843, 0.8627451, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(253, 245, 230);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Old lace (253, 245, 230)</div>
pub const OLD_LACE: Color = Color::new(0.99215686, 0.9607843, 0.9019608, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 250, 240);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Floral white (255, 250, 240)</div>
pub const FLORAL_WHITE: Color = Color::new(1.0, 0.98039216, 0.9411765, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 255, 240);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Ivory (255, 255, 240)</div>
pub const IVORY: Color = Color::new(1.0, 1.0, 0.9411765, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(250, 235, 215);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Antique white (250, 235, 215)</div>
pub const ANTIQUE_WHITE: Color = Color::new(0.98039216, 0.92156863, 0.84313726, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(250, 240, 230);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Linen (250, 240, 230)</div>
pub const LINEN: Color = Color::new(0.98039216, 0.9411765, 0.9019608, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 240, 245);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Lavender blush (255, 240, 245)</div>
pub const LAVENDER_BLUSH: Color = Color::new(1.0, 0.9411765, 0.9607843, 1.0);

/// <div style="margin:2px 0"><span style="background-color:rgb(255, 228, 225);padding:0 0.7em;margin-right:0.5em;border:1px solid"></span>Misty rose (255, 228, 225)</div>
pub const MISTY_ROSE: Color = Color::new(1.0, 0.89411765, 0.88235295, 1.0);

// ============================================================================
// Special values
// ============================================================================

/// Transparent color (0, 0, 0, 0). Useful for clearing or as a default.
pub const TRANSPARENT: Color = Color::new(0.0, 0.0, 0.0, 0.0);
