//! Hand-rolled drawing primitives: gradient meters, area charts, fire, big digits.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};

use crate::source::Rng;
use crate::theme::{self, Stop};

const LEFT_EIGHTHS: [&str; 8] = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"];
const LOWER_EIGHTHS: [&str; 8] = ["", "▁", "▂", "▃", "▄", "▅", "▆", "▇"];

fn put(buf: &mut Buffer, x: u16, y: u16, sym: &str, fg: Color, bg: Option<Color>) {
    if let Some(c) = buf.cell_mut((x, y)) {
        c.set_symbol(sym);
        c.set_fg(fg);
        if let Some(bg) = bg {
            c.set_bg(bg);
        }
    }
}

/// htop-style meter: `LABEL [██████████▋        text]` with a left→right gradient
/// and sub-cell precision.
pub fn meter(buf: &mut Buffer, area: Rect, label: &str, frac: f64, text: &str, stops: &[Stop]) {
    if area.width < 10 || area.height == 0 {
        return;
    }
    let y = area.y;
    let label_w = 6u16;
    buf.set_stringn(
        area.x,
        y,
        label,
        label_w as usize,
        Style::default().fg(theme::LABEL).add_modifier(Modifier::BOLD),
    );
    let bracket = Style::default().fg(theme::MUTED);
    let bx = area.x + label_w;
    buf.set_string(bx, y, "[", bracket);
    buf.set_string(area.right() - 1, y, "]", bracket);
    let x0 = bx + 1;
    let w = area.right().saturating_sub(1).saturating_sub(x0);
    if w == 0 {
        return;
    }
    let frac = if frac.is_finite() { frac.clamp(0.0, 1.0) } else { 0.0 };
    let eighths = (frac * w as f64 * 8.0).round() as u32;
    let full = (eighths / 8) as u16;
    let rem = (eighths % 8) as usize;

    let color_at = |i: u16| theme::grad(stops, if w > 1 { i as f64 / (w - 1) as f64 } else { 1.0 });
    for i in 0..w {
        let x = x0 + i;
        if i < full {
            put(buf, x, y, "█", color_at(i), Some(theme::TRACK));
        } else if i == full && rem > 0 {
            put(buf, x, y, LEFT_EIGHTHS[rem], color_at(i), Some(theme::TRACK));
        } else {
            put(buf, x, y, " ", theme::TEXT, Some(theme::TRACK));
        }
    }
    // Right-aligned text overlay; dark on the filled part, light on the track.
    let tw = text.chars().count() as u16;
    if tw + 1 < w {
        let start = x0 + w - tw - 1;
        for (k, ch) in text.chars().enumerate() {
            let i = start - x0 + k as u16;
            let x = start + k as u16;
            if let Some(c) = buf.cell_mut((x, y)) {
                c.set_char(ch);
                if i < full {
                    c.set_fg(Color::Rgb(15, 15, 25)).set_bg(color_at(i));
                } else {
                    c.set_fg(theme::TEXT).set_bg(theme::TRACK);
                }
                c.modifier.insert(Modifier::BOLD);
            }
        }
    }
}

/// Filled area chart with eighth-block tops and a vertical gradient.
/// `data` is oldest → newest; the newest sample sits at the right edge.
pub fn area_chart(buf: &mut Buffer, area: Rect, data: &[f64], max: f64, stops: &[Stop]) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let h = area.height as usize;
    let w = area.width as usize;
    // Grid lines at 25/50/75 %.
    for p in [0.25, 0.5, 0.75] {
        let r = (h as f64 * p).round() as usize;
        if r == 0 || r >= h {
            continue;
        }
        let y = area.bottom() - 1 - r as u16;
        for x in area.left()..area.right() {
            put(buf, x, y, "┈", theme::GRID, None);
        }
    }
    let n = data.len();
    for col in 0..w {
        let idx = n as isize - w as isize + col as isize;
        if idx < 0 {
            continue;
        }
        let v = data[idx as usize];
        let v = if max > 0.0 && v.is_finite() { (v / max).clamp(0.0, 1.0) } else { 0.0 };
        let total = (v * h as f64 * 8.0).round() as usize;
        let x = area.x + col as u16;
        for r in 0..h {
            let level = total.saturating_sub(r * 8);
            if level == 0 {
                break;
            }
            let y = area.bottom() - 1 - r as u16;
            let color = theme::grad(stops, (r as f64 + 0.5) / h as f64);
            let sym = if level >= 8 { "█" } else { LOWER_EIGHTHS[level] };
            put(buf, x, y, sym, color, None);
        }
    }
}

/// Doom-style fire simulation, rendered with half blocks (2 pixels per cell).
pub struct Fire {
    w: usize,
    h: usize,
    heat: Vec<u8>,
    rng: Rng,
}

impl Fire {
    pub fn new() -> Self {
        Self {
            w: 0,
            h: 0,
            heat: Vec::new(),
            rng: Rng::new(0xF1AE),
        }
    }

    fn resize(&mut self, w: usize, h: usize) {
        if w != self.w || h != self.h {
            self.w = w;
            self.h = h;
            self.heat = vec![0; w * h];
        }
    }

    /// Advance one frame. `intensity` in [0, 1] controls the source row heat.
    pub fn step(&mut self, intensity: f64) {
        let (w, h) = (self.w, self.h);
        if w == 0 || h < 2 {
            return;
        }
        let max = (theme::FIRE.len() - 1) as f64;
        let src = (intensity.clamp(0.0, 1.0) * max).round() as u8;
        for x in 0..w {
            // A little flicker in the source row.
            let jitter = if src > 2 { (self.rng.next_u32() % 3) as u8 } else { 0 };
            self.heat[(h - 1) * w + x] = src.saturating_sub(jitter);
        }
        // Decay tuned so a full-intensity flame roughly fills the area.
        // Random decay in [0, 2·rate] so the mean is `rate` and flames flicker.
        let rate = max / (h as f64 * 0.8);
        for y in 1..h {
            for x in 0..w {
                let below = self.heat[y * w + x];
                let r = self.rng.next_u32();
                let u = (r & 0xFFFF) as f64 / 65536.0;
                let decay = (u * (2.0 * rate + 1.0)).floor() as u8;
                // Wind: drift the flame sideways.
                let shift = ((r >> 20) % 3) as isize - 1;
                let dx = (x as isize + shift).clamp(0, w as isize - 1) as usize;
                self.heat[(y - 1) * w + dx] = below.saturating_sub(decay);
            }
        }
    }

    pub fn render(&mut self, buf: &mut Buffer, area: Rect) {
        self.resize(area.width as usize, area.height as usize * 2);
        let w = self.w;
        for cy in 0..area.height as usize {
            for cx in 0..w {
                let top = self.heat[(cy * 2) * w + cx] as usize;
                let bot = self.heat[(cy * 2 + 1) * w + cx] as usize;
                let x = area.x + cx as u16;
                let y = area.y + cy as u16;
                let col = |i: usize| {
                    let (r, g, b) = theme::FIRE[i.min(theme::FIRE.len() - 1)];
                    Color::Rgb(r, g, b)
                };
                let Some(c) = buf.cell_mut((x, y)) else { continue };
                match (top > 0, bot > 0) {
                    (false, false) => {}
                    (true, false) => {
                        c.set_symbol("▀").set_fg(col(top));
                    }
                    (false, true) => {
                        c.set_symbol("▄").set_fg(col(bot));
                    }
                    (true, true) => {
                        c.set_symbol("▀").set_fg(col(top)).set_bg(col(bot));
                    }
                }
            }
        }
    }
}

/// Darken everything already drawn in `area` (a smoked-glass backdrop).
pub fn darken(buf: &mut Buffer, area: Rect, k: f64, base: Color) {
    let f = |c: Color| match c {
        Color::Rgb(r, g, b) => Color::Rgb(
            (r as f64 * k) as u8,
            (g as f64 * k) as u8,
            (b as f64 * k) as u8,
        ),
        _ => base,
    };
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            if let Some(c) = buf.cell_mut((x, y)) {
                let (fg, bg) = (c.fg, c.bg);
                c.set_fg(f(fg)).set_bg(f(bg));
            }
        }
    }
}

const DIGITS: [[&str; 5]; 11] = [
    ["███", "█ █", "█ █", "█ █", "███"],
    ["██ ", " █ ", " █ ", " █ ", "███"],
    ["███", "  █", "███", "█  ", "███"],
    ["███", "  █", "███", "  █", "███"],
    ["█ █", "█ █", "███", "  █", "  █"],
    ["███", "█  ", "███", "  █", "███"],
    ["███", "█  ", "███", "█ █", "███"],
    ["███", "  █", "  █", "  █", "  █"],
    ["███", "█ █", "███", "█ █", "███"],
    ["███", "█ █", "███", "  █", "███"],
    ["█ █", "  █", " █ ", "█  ", "█ █"], // %
];

pub fn big_text_width(s: &str) -> u16 {
    let n = s.chars().filter(|c| c.is_ascii_digit() || *c == '%').count() as u16;
    if n == 0 { 0 } else { n * 4 - 1 }
}

/// Draw digits (and '%') in a 3×5 block font with a dark drop-shadow.
pub fn big_text(buf: &mut Buffer, x: u16, y: u16, s: &str, fg: impl Fn(usize) -> Color, clip: Rect) {
    let mut cx = x;
    for ch in s.chars() {
        let g = match ch {
            '0'..='9' => &DIGITS[ch as usize - '0' as usize],
            '%' => &DIGITS[10],
            _ => continue,
        };
        for (row, line) in g.iter().enumerate() {
            for (col, px) in line.chars().enumerate() {
                if px != '█' {
                    continue;
                }
                let px_x = cx + col as u16;
                let px_y = y + row as u16;
                // shadow
                let sx = px_x + 1;
                let sy = px_y + 1;
                if sx < clip.right() && sy < clip.bottom() {
                    if let Some(c) = buf.cell_mut((sx, sy)) {
                        if c.symbol() != "█" {
                            c.set_symbol("█").set_fg(Color::Rgb(20, 8, 4));
                        }
                    }
                }
                if px_x < clip.right() && px_y < clip.bottom() {
                    put(buf, px_x, px_y, "█", fg(row), None);
                }
            }
        }
        cx += 4;
    }
}
