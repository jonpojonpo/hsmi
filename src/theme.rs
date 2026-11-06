//! Colours and gradients.

use ratatui::style::Color;

pub type Stop = (f64, (u8, u8, u8));

/// Sample a multi-stop gradient at `t` in [0, 1].
pub fn grad(stops: &[Stop], t: f64) -> Color {
    let t = if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.0 };
    let (r, g, b) = grad_rgb(stops, t);
    Color::Rgb(r, g, b)
}

pub fn grad_rgb(stops: &[Stop], t: f64) -> (u8, u8, u8) {
    if t <= stops[0].0 {
        return stops[0].1;
    }
    for w in stops.windows(2) {
        let (a, ca) = w[0];
        let (b, cb) = w[1];
        if t <= b {
            let k = if b > a { (t - a) / (b - a) } else { 0.0 };
            let l = |x: u8, y: u8| (x as f64 + (y as f64 - x as f64) * k).round() as u8;
            return (l(ca.0, cb.0), l(ca.1, cb.1), l(ca.2, cb.2));
        }
    }
    stops[stops.len() - 1].1
}

/// SM / compute: electric blue → violet → hot pink → orange → gold.
pub const SM: &[Stop] = &[
    (0.0, (56, 120, 255)),
    (0.35, (139, 92, 246)),
    (0.6, (236, 72, 153)),
    (0.82, (249, 115, 22)),
    (1.0, (250, 214, 40)),
];
/// Memory bandwidth: teal → green → lime.
pub const MEMBW: &[Stop] = &[(0.0, (20, 184, 166)), (0.55, (34, 197, 94)), (1.0, (200, 245, 100))];
/// VRAM fill: sky → indigo → pink → red when nearly full.
pub const VRAM: &[Stop] = &[
    (0.0, (56, 189, 248)),
    (0.6, (129, 140, 248)),
    (0.85, (244, 114, 182)),
    (1.0, (239, 68, 68)),
];
/// Power: amber → orange → red.
pub const POWER: &[Stop] = &[(0.0, (253, 224, 71)), (0.6, (249, 115, 22)), (1.0, (239, 68, 68))];
/// Temperature: cold blue → green → yellow → red.
pub const TEMP: &[Stop] = &[
    (0.0, (56, 189, 248)),
    (0.45, (74, 222, 128)),
    (0.7, (250, 204, 21)),
    (0.85, (249, 115, 22)),
    (1.0, (239, 68, 68)),
];
pub const FAN: &[Stop] = &[(0.0, (165, 243, 252)), (1.0, (6, 182, 212))];
pub const CLOCK: &[Stop] = &[(0.0, (196, 181, 253)), (1.0, (217, 70, 239))];
pub const PCIE_TX: &[Stop] = &[(0.0, (147, 197, 253)), (1.0, (59, 130, 246))];
pub const PCIE_RX: &[Stop] = &[(0.0, (153, 246, 228)), (1.0, (16, 185, 129))];

pub const TRACK: Color = Color::Rgb(32, 33, 46);
pub const GRID: Color = Color::Rgb(48, 50, 68);
pub const BORDER: Color = Color::Rgb(70, 74, 104);
pub const TITLE: Color = Color::Rgb(186, 196, 255);
pub const LABEL: Color = Color::Rgb(125, 211, 252);
pub const TEXT: Color = Color::Rgb(226, 232, 240);
pub const MUTED: Color = Color::Rgb(120, 128, 158);
pub const ACCENT: Color = Color::Rgb(236, 72, 153);
pub const SELECT_BG: Color = Color::Rgb(55, 48, 107);

/// Classic Doom fire palette (index 0 is treated as transparent).
pub const FIRE: [(u8, u8, u8); 37] = [
    (0x07, 0x07, 0x07),
    (0x1F, 0x07, 0x07),
    (0x2F, 0x0F, 0x07),
    (0x47, 0x0F, 0x07),
    (0x57, 0x17, 0x07),
    (0x67, 0x1F, 0x07),
    (0x77, 0x1F, 0x07),
    (0x8F, 0x27, 0x07),
    (0x9F, 0x2F, 0x07),
    (0xAF, 0x3F, 0x07),
    (0xBF, 0x47, 0x07),
    (0xC7, 0x47, 0x07),
    (0xDF, 0x4F, 0x07),
    (0xDF, 0x57, 0x07),
    (0xDF, 0x57, 0x07),
    (0xD7, 0x5F, 0x07),
    (0xD7, 0x5F, 0x07),
    (0xD7, 0x67, 0x0F),
    (0xCF, 0x6F, 0x0F),
    (0xCF, 0x77, 0x0F),
    (0xCF, 0x7F, 0x0F),
    (0xCF, 0x87, 0x17),
    (0xC7, 0x87, 0x17),
    (0xC7, 0x8F, 0x17),
    (0xC7, 0x97, 0x1F),
    (0xBF, 0x9F, 0x1F),
    (0xBF, 0x9F, 0x1F),
    (0xBF, 0xA7, 0x27),
    (0xBF, 0xA7, 0x27),
    (0xBF, 0xAF, 0x2F),
    (0xB7, 0xAF, 0x2F),
    (0xB7, 0xB7, 0x2F),
    (0xB7, 0xB7, 0x37),
    (0xCF, 0xCF, 0x6F),
    (0xDF, 0xDF, 0x9F),
    (0xEF, 0xEF, 0xC7),
    (0xFF, 0xFF, 0xFF),
];
