//! Layout and rendering.

use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Cell, Clear, Paragraph, Row, Table, Wrap};

use crate::app::{App, GpuState, Popup};
use crate::theme::{self, Stop};
use crate::widgets;

const BG: Color = Color::Rgb(13, 14, 23);
const SPINNER: [&str; 4] = ["◐", "◓", "◑", "◒"];

pub fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();
    f.buffer_mut().set_style(area, Style::default().bg(BG).fg(theme::TEXT));

    if app.gpus.is_empty() {
        let p = Paragraph::new("No NVIDIA GPUs found.  (press q to quit, or try `hsmi --demo`)")
            .alignment(Alignment::Center);
        f.render_widget(p, area);
        return;
    }
    app.sel = app.sel.min(app.gpus.len() - 1);

    let strip_h = if app.gpus.len() > 1 { app.gpus.len().min(8) as u16 } else { 0 };
    let proc_h = (app.rows.len() as u16 + 3).clamp(5, 12);
    let [header, strip, top, charts, info, procs, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(strip_h),
        Constraint::Length(11),
        Constraint::Min(6),
        Constraint::Length(1),
        Constraint::Length(proc_h),
        Constraint::Length(1),
    ])
    .areas(area);

    draw_header(f, app, header);
    if strip_h > 0 {
        draw_strip(f, app, strip);
    }
    let [meters, brr] =
        Layout::horizontal([Constraint::Percentage(62), Constraint::Percentage(38)]).areas(top);
    let frame = app.frame;
    let interval = app.interval;
    {
        let g = &mut app.gpus[app.sel];
        draw_meters(f, g, meters);
        draw_brr(f, g, brr, frame);
        draw_charts(f, g, charts, interval.as_secs_f64());
        draw_info(f, g, info);
    }
    draw_procs(f, app, procs);
    draw_footer(f, app, footer);
    draw_popup(f, app, area);
}

fn rounded(title: Line<'static>) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::BORDER))
        .title(title)
}

fn title(s: impl Into<String>) -> Span<'static> {
    Span::styled(
        format!(" {} ", s.into()),
        Style::default().fg(theme::TITLE).add_modifier(Modifier::BOLD),
    )
}

fn gradient_spans(s: &str, stops: &[Stop], fg: Color, shift: f64) -> Vec<Span<'static>> {
    let n = s.chars().count().max(2) as f64;
    s.chars()
        .enumerate()
        .map(|(i, ch)| {
            let t = ((i as f64 / (n - 1.0)) + shift).rem_euclid(1.0);
            let t = 1.0 - (2.0 * t - 1.0).abs(); // ping-pong so it loops smoothly
            Span::styled(
                ch.to_string(),
                Style::default().bg(theme::grad(stops, t)).fg(fg).add_modifier(Modifier::BOLD),
            )
        })
        .collect()
}

fn fmt_gib(b: u64) -> String {
    format!("{:.1}", b as f64 / (1u64 << 30) as f64)
}

fn fmt_rate(kbs: u32) -> String {
    let b = kbs as f64 * 1024.0;
    if b >= 1e9 {
        format!("{:.2} GB/s", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.0} MB/s", b / 1e6)
    } else {
        format!("{:.0} KB/s", b / 1e3)
    }
}

fn fmt_dur(secs: u64) -> String {
    format!("{:02}:{:02}:{:02}", secs / 3600, secs / 60 % 60, secs % 60)
}

fn mini_bar(frac: f64, width: usize, stops: &[Stop]) -> Vec<Span<'static>> {
    const E: [&str; 8] = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"];
    let frac = if frac.is_finite() { frac.clamp(0.0, 1.0) } else { 0.0 };
    let eighths = (frac * width as f64 * 8.0).round() as usize;
    (0..width)
        .map(|i| {
            let c = theme::grad(stops, if width > 1 { i as f64 / (width - 1) as f64 } else { 1.0 });
            let sym = if eighths >= (i + 1) * 8 {
                "█"
            } else if eighths > i * 8 {
                E[eighths - i * 8]
            } else {
                " "
            };
            Span::styled(sym, Style::default().fg(c).bg(theme::TRACK))
        })
        .collect()
}

// ─────────────────────────────── header ───────────────────────────────

fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    let shift = (app.frame as f64 / 90.0).fract();
    let mut left = gradient_spans(" ▲ hsmi ", theme::SM, Color::Rgb(12, 10, 20), shift);
    left.push(Span::raw(" "));
    left.push(Span::styled(app.host.clone(), Style::default().fg(theme::LABEL).bold()));
    left.push(Span::styled(
        format!("  {} GPU{}", app.gpus.len(), if app.gpus.len() == 1 { "" } else { "s" }),
        Style::default().fg(theme::MUTED),
    ));
    f.render_widget(Paragraph::new(Line::from(left)), area);

    let mut right = vec![
        Span::styled("driver ", Style::default().fg(theme::MUTED)),
        Span::styled(app.sys.driver.clone(), Style::default().fg(theme::TEXT)),
        Span::styled("  CUDA ", Style::default().fg(theme::MUTED)),
        Span::styled(app.sys.cuda.clone(), Style::default().fg(theme::TEXT)),
        Span::styled("  up ", Style::default().fg(theme::MUTED)),
        Span::styled(fmt_dur(app.started.elapsed().as_secs()), Style::default().fg(theme::TEXT)),
        Span::styled("  ⟳ ", Style::default().fg(theme::MUTED)),
        Span::styled(format!("{}ms ", app.interval.as_millis()), Style::default().fg(theme::TEXT)),
    ];
    if app.paused {
        right.push(Span::styled(
            " PAUSED ",
            Style::default().bg(Color::Rgb(250, 204, 21)).fg(Color::Black).bold(),
        ));
    }
    f.render_widget(Paragraph::new(Line::from(right)).alignment(Alignment::Right), area);
}

// ─────────────────────────────── GPU strip ─────────────────────────────

fn draw_strip(f: &mut Frame, app: &App, area: Rect) {
    for (i, g) in app.gpus.iter().enumerate().take(area.height as usize) {
        let y = area.y + i as u16;
        let row = Rect::new(area.x, y, area.width, 1);
        let sel = i == app.sel;
        let mut spans = vec![Span::styled(
            if sel { " ▶ " } else { "   " },
            Style::default().fg(theme::ACCENT).bold(),
        )];
        spans.push(Span::styled(format!("{i} "), Style::default().fg(theme::LABEL).bold()));
        let mut name = g.info.name.clone();
        name.truncate(26);
        spans.push(Span::styled(format!("{name:<27}"), Style::default().fg(theme::TEXT)));
        spans.push(Span::styled("SM ", Style::default().fg(theme::MUTED)));
        spans.extend(mini_bar(g.disp.sm, 12, theme::SM));
        spans.push(Span::raw(format!(" {:>3.0}%  ", g.disp.sm * 100.0)));
        spans.push(Span::styled("VRAM ", Style::default().fg(theme::MUTED)));
        spans.extend(mini_bar(g.disp.vram, 12, theme::VRAM));
        if let Some(s) = &g.last {
            spans.push(Span::raw(format!(
                " {:>5}/{}G  ",
                fmt_gib(s.mem_used),
                fmt_gib(s.mem_total)
            )));
            if let Some(p) = s.power_w {
                spans.push(Span::styled(
                    format!("{p:>4.0}W  "),
                    Style::default().fg(theme::grad(theme::POWER, g.disp.power)),
                ));
            }
            if let Some(t) = s.temp {
                spans.push(Span::styled(
                    format!("{t:>3}°C  "),
                    Style::default().fg(theme::grad(theme::TEMP, g.disp.temp)),
                ));
            }
        }
        spans.push(Span::styled("BRR ", Style::default().fg(theme::MUTED)));
        spans.push(Span::styled(
            format!("{:>3.0}%", g.disp.brr * 100.0),
            Style::default().fg(theme::grad(theme::SM, g.disp.brr)).bold(),
        ));
        let style = if sel { Style::default().bg(theme::SELECT_BG) } else { Style::default() };
        f.render_widget(Paragraph::new(Line::from(spans)).style(style), row);
    }
}

// ─────────────────────────────── meters ───────────────────────────────

fn draw_meters(f: &mut Frame, g: &GpuState, area: Rect) {
    let spin = SPINNER[(g.fan_phase as usize) % SPINNER.len()];
    let mut t = vec![
        Span::styled(" ", Style::default()),
        Span::styled(spin, Style::default().fg(theme::grad(theme::FAN, g.disp.fan))),
        title(format!("GPU{} · {}", g.info.index, g.info.name)),
        Span::styled("· ", Style::default().fg(theme::MUTED)),
    ];
    let mut extra = Vec::new();
    if let Some(c) = g.info.cores {
        extra.push(format!("{c} cores"));
    }
    if let Some(b) = g.info.bus_width {
        extra.push(format!("{b}-bit"));
    }
    if !extra.is_empty() {
        t.push(Span::styled(format!("{} ", extra.join(" · ")), Style::default().fg(theme::MUTED)));
    }
    let block = rounded(Line::from(t));
    let inner = block.inner(area).inner(ratatui::layout::Margin::new(1, 0));
    f.render_widget(block, area);

    let Some(s) = &g.last else { return };
    let d = &g.disp;
    let na = || "n/a".to_string();
    let rows: Vec<(&str, f64, String, &[Stop])> = vec![
        ("SM", d.sm, format!("{:.0}%", s.sm_util), theme::SM),
        ("MEM", d.mem_bw, format!("{:.0}% bw", s.mem_bw_util), theme::MEMBW),
        (
            "VRAM",
            d.vram,
            format!("{}/{} GiB", fmt_gib(s.mem_used), fmt_gib(s.mem_total)),
            theme::VRAM,
        ),
        (
            "POWER",
            d.power,
            match (s.power_w, s.power_limit_w) {
                (Some(p), Some(l)) => format!("{p:.0}/{l:.0} W"),
                (Some(p), None) => format!("{p:.0} W"),
                _ => na(),
            },
            theme::POWER,
        ),
        (
            "TEMP",
            d.temp,
            s.temp.map(|t| format!("{t}°C")).unwrap_or_else(na),
            theme::TEMP,
        ),
        (
            "FAN",
            d.fan,
            s.fan.map(|v| format!("{v}%")).unwrap_or_else(na),
            theme::FAN,
        ),
        (
            "CLOCK",
            d.clock,
            match (s.sm_clock, g.info.max_sm_clock) {
                (Some(c), Some(m)) => format!("{c}/{m} MHz"),
                (Some(c), None) => format!("{c} MHz"),
                _ => na(),
            },
            theme::CLOCK,
        ),
        (
            "PCI↑",
            d.tx,
            s.pcie_tx_kbs.map(fmt_rate).unwrap_or_else(na),
            theme::PCIE_TX,
        ),
        (
            "PCI↓",
            d.rx,
            s.pcie_rx_kbs.map(fmt_rate).unwrap_or_else(na),
            theme::PCIE_RX,
        ),
    ];
    let buf = f.buffer_mut();
    for (i, (label, frac, text, stops)) in rows.iter().enumerate().take(inner.height as usize) {
        let r = Rect::new(inner.x, inner.y + i as u16, inner.width, 1);
        widgets::meter(buf, r, label, *frac, text, stops);
    }
}

// ─────────────────────────────── BRR-O-METER ───────────────────────────

fn draw_brr(f: &mut Frame, g: &mut GpuState, area: Rect, frame: u64) {
    let real_tensor = g.last.as_ref().and_then(|s| s.tensor).is_some();
    let block = rounded(Line::from(vec![
        Span::styled(" 🔥", Style::default()),
        title(if real_tensor { "TENSOR CORES" } else { "BRR-O-METER" }),
    ]))
    .title_bottom(
        Line::from(Span::styled(
            if real_tensor { " GPM tensor-pipe activity " } else { " SM util × power/limit " },
            Style::default().fg(theme::MUTED),
        ))
        .right_aligned(),
    );
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width < 4 || inner.height < 2 {
        return;
    }
    let buf = f.buffer_mut();
    g.fire.render(buf, inner);

    let v = g.disp.brr;
    let pct = format!("{:.0}%", (v * 100.0).round());
    let w = widgets::big_text_width(&pct);
    if inner.height >= 7 && inner.width >= w + 2 {
        let x = inner.x + (inner.width - w) / 2;
        let y = inner.y + 1 + (inner.height.saturating_sub(7)) / 2;
        let back = Rect::new(x.saturating_sub(2), y.saturating_sub(1), w + 4, 7).intersection(inner);
        widgets::darken(buf, back, 0.28, BG);
        widgets::big_text(
            buf,
            x,
            y,
            &pct,
            |row| {
                let t = row as f64 / 4.0;
                let a = (255.0, 252.0, 240.0);
                let b = (255.0, 196.0, 92.0);
                Color::Rgb(
                    (a.0 + (b.0 - a.0) * t) as u8,
                    (a.1 + (b.1 - a.1) * t) as u8,
                    (a.2 + (b.2 - a.2) * t) as u8,
                )
            },
            inner,
        );
    } else {
        buf.set_string(inner.x + 1, inner.y + 1, &pct, Style::default().fg(Color::White).bold());
    }

    // Status word on the top line.
    let (word, color) = match v {
        v if v < 0.05 => ("zzz… idle", theme::MUTED),
        v if v < 0.30 => ("warming up", Color::Rgb(147, 197, 253)),
        v if v < 0.60 => ("brr", Color::Rgb(250, 204, 21)),
        v if v < 0.85 => ("BRRRR", Color::Rgb(249, 115, 22)),
        _ => ("MAXIMUM BRRRRRR", Color::White),
    };
    let wlen = word.chars().count() as u16;
    let x = inner.x + inner.width.saturating_sub(wlen) / 2;
    if v >= 0.85 {
        // Rainbow shimmer when it's really going.
        for (i, ch) in word.chars().enumerate() {
            let hue = ((i as f64 * 0.07) + frame as f64 * 0.03).fract();
            let c = hsv(hue, 0.65, 1.0);
            if let Some(cell) = buf.cell_mut((x + i as u16, inner.y)) {
                cell.set_char(ch).set_fg(c).set_bg(BG);
                cell.modifier.insert(Modifier::BOLD);
            }
        }
    } else {
        buf.set_string(x, inner.y, word, Style::default().fg(color).bg(BG).bold());
    }
}

fn hsv(h: f64, s: f64, v: f64) -> Color {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s));
    let (r, g, b) = match i as i32 % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    Color::Rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

// ─────────────────────────────── charts ───────────────────────────────

fn chart_block(name: &str, value: String, color: Color, window: String) -> Block<'static> {
    rounded(Line::from(vec![
        title(name.to_string()),
        Span::styled(format!("{value} "), Style::default().fg(color).bold()),
    ]))
    .title(Line::from(Span::styled(format!(" {window} "), Style::default().fg(theme::MUTED))).right_aligned())
}

fn draw_chart(buf: &mut Buffer, block: Block, area: Rect, data: &[f64], max: f64, stops: &[Stop]) -> Rect {
    use ratatui::widgets::Widget;
    let inner = block.inner(area);
    block.render(area, buf);
    widgets::area_chart(buf, inner, data, max, stops);
    inner
}

fn draw_charts(f: &mut Frame, g: &mut GpuState, area: Rect, interval_s: f64) {
    // Side charts need ≥4 rows each to be worth drawing; on short or narrow
    // terminals fall back to fewer of them (or just the big SM chart).
    let n_side = if area.width < 70 { 0 } else { (area.height / 4).min(3) as usize };
    let (big, side) = if n_side == 0 {
        (area, Rect::default())
    } else {
        let [a, b] =
            Layout::horizontal([Constraint::Percentage(62), Constraint::Percentage(38)]).areas(area);
        (a, b)
    };
    let window = |w: u16| {
        let s = (w.saturating_sub(2) as f64 * interval_s).round() as u64;
        if s >= 120 { format!("{}m", s / 60) } else { format!("{s}s") }
    };
    let last = g.last.clone().unwrap_or_default();
    let buf = f.buffer_mut();

    let block = chart_block(
        "SM UTILIZATION",
        format!("{:.0}%", last.sm_util),
        theme::grad(theme::SM, last.sm_util / 100.0),
        window(big.width),
    );
    draw_chart(buf, block, big, g.hist.sm.slice(), 100.0, theme::SM);
    if n_side == 0 {
        return;
    }

    let slots = Layout::vertical(vec![Constraint::Ratio(1, n_side as u32); n_side]).split(side);

    let limit = last.power_limit_w.unwrap_or(0.0).max(1.0);
    let power_max = g.hist.power.0.iter().cloned().fold(limit, f64::max);
    let block = chart_block(
        "POWER",
        last.power_w.map(|p| format!("{p:.0} W")).unwrap_or_else(|| "n/a".into()),
        theme::grad(theme::POWER, last.power_w.unwrap_or(0.0) / limit),
        window(side.width),
    );
    draw_chart(buf, block, slots[0], g.hist.power.slice(), power_max, theme::POWER);

    if n_side >= 2 {
        let block = chart_block(
            "MEM BW",
            format!("{:.0}%", last.mem_bw_util),
            theme::grad(theme::MEMBW, last.mem_bw_util / 100.0),
            window(side.width),
        );
        draw_chart(buf, block, slots[1], g.hist.mem_bw.slice(), 100.0, theme::MEMBW);
    }

    if n_side >= 3 {
        let slowdown = g.info.temp_slowdown.unwrap_or(90) as f64;
        let block = chart_block(
            "TEMP",
            last.temp.map(|t| format!("{t}°C")).unwrap_or_else(|| "n/a".into()),
            theme::grad(theme::TEMP, last.temp.unwrap_or(0) as f64 / slowdown),
            window(side.width),
        );
        draw_chart(buf, block, slots[2], g.hist.temp.slice(), slowdown.max(100.0), theme::TEMP);
    }
}

// ─────────────────────────────── info line ─────────────────────────────

fn badge(text: &str, bg: Color) -> Span<'static> {
    Span::styled(format!(" {text} "), Style::default().bg(bg).fg(Color::Rgb(12, 10, 20)).bold())
}

fn draw_info(f: &mut Frame, g: &GpuState, area: Rect) {
    let Some(s) = &g.last else { return };
    let m = Style::default().fg(theme::MUTED);
    let v = Style::default().fg(theme::TEXT);
    let mut spans = vec![Span::raw(" ")];
    if let Some(p) = s.pstate {
        let c = if p <= 2 { Color::Rgb(74, 222, 128) } else { Color::Rgb(148, 163, 184) };
        spans.push(badge(&format!("P{p}"), c));
        spans.push(Span::raw(" "));
    }
    if g.info.unlocked {
        spans.push(badge("UNLOCKED", Color::Rgb(232, 121, 249)));
        spans.push(Span::raw(" "));
    }
    spans.push(Span::raw(" "));
    if let Some(mc) = s.mem_clock {
        spans.push(Span::styled("mem ", m));
        let max = g.info.max_mem_clock.map(|m| format!("/{m}")).unwrap_or_default();
        spans.push(Span::styled(format!("{mc}{max} MHz  "), v));
    }
    if let (Some(gen_), Some(w)) = (s.pcie_gen, s.pcie_width) {
        spans.push(Span::styled("PCIe ", m));
        let max = match (g.info.pcie_max_gen, g.info.pcie_max_width) {
            (Some(mg), Some(mw)) if (mg, mw) != (gen_, w) => format!(" (max {mg}x{mw})"),
            _ => String::new(),
        };
        spans.push(Span::styled(format!("Gen{gen_} x{w}{max}  "), v));
    }
    if let (Some(e), Some(d)) = (s.enc, s.dec) {
        spans.push(Span::styled("enc/dec ", m));
        spans.push(Span::styled(format!("{e}%/{d}%  "), v));
    }
    if let (Some(now), Some(start)) = (s.energy_mj, g.energy_start_mj) {
        let wh = now.saturating_sub(start) as f64 / 1000.0 / 3600.0;
        spans.push(Span::styled("⚡ ", Style::default().fg(Color::Rgb(250, 204, 21))));
        spans.push(Span::styled(format!("{wh:.2} Wh this session  "), v));
    }
    spans.push(Span::styled("clocks: ", m));
    if s.throttle.is_empty() {
        spans.push(badge("UNTHROTTLED", Color::Rgb(74, 222, 128)));
    }
    for t in &s.throttle {
        let c = match *t {
            "PWR CAP" | "PWR BRAKE" => Color::Rgb(251, 146, 60),
            "SW THERMAL" | "HW THERMAL" | "HW SLOWDOWN" => Color::Rgb(248, 113, 113),
            "IDLE" => Color::Rgb(100, 116, 139),
            "VOLT LIMIT" | "BOARD LIMIT" => Color::Rgb(250, 204, 21),
            _ => Color::Rgb(196, 181, 253),
        };
        spans.push(badge(t, c));
        spans.push(Span::raw(" "));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

// ─────────────────────────────── processes ─────────────────────────────

fn draw_procs(f: &mut Frame, app: &mut App, area: Rect) {
    let g = &app.gpus[app.sel];
    let total = g.last.as_ref().map(|s| s.mem_total).unwrap_or(1).max(1);
    let block = rounded(Line::from(vec![
        title(format!("PROCESSES · GPU{}", g.info.index)),
        Span::styled(
            format!("{} running · sort: {} ▼ ", app.rows.len(), app.sort.label()),
            Style::default().fg(theme::MUTED),
        ),
    ]));

    if app.rows.is_empty() {
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(
            Paragraph::new("no processes on this GPU — go start a training run 🚀")
                .style(Style::default().fg(theme::MUTED))
                .alignment(Alignment::Center),
            Rect::new(inner.x, inner.y + inner.height / 2, inner.width, 1),
        );
        return;
    }

    let hdr = Style::default().fg(Color::Rgb(12, 10, 20)).bg(theme::LABEL).bold();
    let header = Row::new(["PID", "USER", "TYPE", "SM%", "MEM%", "VRAM", "COMMAND"]).style(hdr);
    let rows: Vec<Row> = app
        .rows
        .iter()
        .map(|r| {
            let sm = r.s.sm.unwrap_or(0) as f64;
            let mem = r.s.mem.unwrap_or(0) as f64;
            let vram = r.s.vram.unwrap_or(0);
            let vfrac = vram as f64 / total as f64;

            let mut sm_line = mini_bar(sm / 100.0, 6, theme::SM);
            sm_line.push(Span::styled(
                format!(" {sm:>3.0}"),
                Style::default().fg(theme::grad(theme::SM, sm / 100.0)).bold(),
            ));
            let mut vram_line = mini_bar(vfrac, 6, theme::VRAM);
            vram_line.push(Span::styled(
                match r.s.vram {
                    Some(b) if b >= 1 << 30 => format!(" {:>5.1}G", b as f64 / (1u64 << 30) as f64),
                    Some(b) => format!(" {:>5.0}M", b as f64 / (1u64 << 20) as f64),
                    None => "   n/a".into(),
                },
                Style::default().fg(theme::grad(theme::VRAM, vfrac)),
            ));
            let (exe, args) = split_cmd(&r.meta.cmd);
            Row::new(vec![
                Cell::from(Span::styled(r.s.pid.to_string(), Style::default().fg(theme::LABEL))),
                Cell::from(Span::styled(r.meta.user.clone(), Style::default().fg(theme::TEXT))),
                Cell::from(Span::styled(
                    r.s.kind.tag(),
                    Style::default().fg(match r.s.kind.tag() {
                        "G" => Color::Rgb(196, 181, 253),
                        _ => Color::Rgb(74, 222, 128),
                    }),
                )),
                Cell::from(Line::from(sm_line)),
                Cell::from(Span::styled(format!("{mem:>3.0}"), Style::default().fg(theme::grad(theme::MEMBW, mem / 100.0)))),
                Cell::from(Line::from(vram_line)),
                Cell::from(Line::from(vec![
                    Span::styled(exe, Style::default().fg(Color::White).bold()),
                    Span::styled(args, Style::default().fg(theme::MUTED)),
                ])),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(8),
        Constraint::Length(10),
        Constraint::Length(4),
        Constraint::Length(10),
        Constraint::Length(4),
        Constraint::Length(13),
        Constraint::Min(10),
    ];
    let table = Table::new(rows, widths)
        .header(header)
        .block(block)
        .column_spacing(1)
        .row_highlight_style(Style::default().bg(theme::SELECT_BG))
        .highlight_symbol(Span::styled("▶ ", Style::default().fg(theme::ACCENT).bold()));
    f.render_stateful_widget(table, area, &mut app.table);
}

/// Split a command line into "program" (basename of argv[0], plus the script for interpreters)
/// and the remaining arguments.
fn split_cmd(cmd: &str) -> (String, String) {
    let mut parts = cmd.split_whitespace();
    let Some(argv0) = parts.next() else { return (cmd.to_string(), String::new()) };
    let base = argv0.rsplit('/').next().unwrap_or(argv0).to_string();
    let rest: Vec<&str> = parts.collect();
    let is_interp = base.starts_with("python") || base == "node" || base == "torchrun";
    if is_interp {
        if let Some(first) = rest.first() {
            if !first.starts_with('-') || *first == "-m" {
                let take = if *first == "-m" { 2 } else { 1 };
                let head: Vec<&str> = rest.iter().take(take).cloned().collect();
                let tail: Vec<&str> = rest.iter().skip(take).cloned().collect();
                return (format!("{base} {}", head.join(" ")), format!(" {}", tail.join(" ")).trim_end().to_string());
            }
        }
    }
    let args = if rest.is_empty() { String::new() } else { format!(" {}", rest.join(" ")) };
    (base, args)
}

// ─────────────────────────────── footer ───────────────────────────────

fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    let keys: &[(&str, &str)] = &[
        ("?", "Help"),
        ("Tab", "GPU"),
        ("↑↓", "Select"),
        ("s", "Sort"),
        ("K", "Kill"),
        ("p", "Pause"),
        ("+-", "Rate"),
        ("q", "Quit"),
    ];
    let mut spans = Vec::new();
    for (k, l) in keys {
        spans.push(Span::styled(
            format!(" {k} "),
            Style::default().bg(theme::LABEL).fg(Color::Rgb(12, 10, 20)).bold(),
        ));
        spans.push(Span::styled(format!("{l:<7}"), Style::default().fg(theme::TEXT).bg(Color::Rgb(28, 30, 46))));
    }
    f.render_widget(
        Paragraph::new(Line::from(spans)).style(Style::default().bg(Color::Rgb(28, 30, 46))),
        area,
    );
    if app.sys.demo {
        f.render_widget(
            Paragraph::new(Line::from(badge("DEMO DATA", theme::ACCENT))).alignment(Alignment::Right),
            area,
        );
    }
}

// ─────────────────────────────── popups ───────────────────────────────

fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let w = w.min(area.width);
    let h = h.min(area.height);
    Rect::new(area.x + (area.width - w) / 2, area.y + (area.height - h) / 2, w, h)
}

fn draw_popup(f: &mut Frame, app: &App, area: Rect) {
    let (t, lines, w, h): (&str, Vec<Line>, u16, u16) = match &app.popup {
        Popup::None => return,
        Popup::Help => {
            let k = |k: &str, d: &str| {
                Line::from(vec![
                    Span::styled(format!("  {k:<14}"), Style::default().fg(theme::LABEL).bold()),
                    Span::raw(d.to_string()),
                ])
            };
            (
                "hsmi · help",
                vec![
                    Line::from(""),
                    k("Tab / ← →", "switch GPU (or press its number)"),
                    k("↑ ↓ / j k", "select process   PgUp/PgDn to jump"),
                    k("s", "cycle sort: VRAM → SM% → PID"),
                    k("K / F9 / Del", "kill selected process (asks first)"),
                    k("p / space", "pause sampling"),
                    k("+ / -", "slower / faster sampling (100ms–5s)"),
                    k("q / Esc", "quit"),
                    Line::from(""),
                    Line::from(Span::styled("  BRR-O-METER", Style::default().fg(theme::ACCENT).bold())),
                    Line::from("  SM utilisation × (power draw / power limit). Dense tensor-core"),
                    Line::from("  GEMMs pin both; memory-bound or stalled work burns less power,"),
                    Line::from("  so it reads lower. On Hopper+ GPUs with GPM support, hsmi shows"),
                    Line::from("  the real tensor-pipe activity instead."),
                    Line::from(""),
                    Line::from(Span::styled("  press any key to close", Style::default().fg(theme::MUTED))),
                ],
                72,
                19,
            )
        }
        Popup::Kill { pid, cmd } => {
            let mut c = cmd.clone();
            if c.chars().count() > 56 {
                c = c.chars().take(55).collect::<String>() + "…";
            }
            (
                "kill process",
                vec![
                    Line::from(""),
                    Line::from(vec![
                        Span::raw("  Send SIGTERM to "),
                        Span::styled(pid.to_string(), Style::default().fg(theme::LABEL).bold()),
                        Span::raw("?"),
                    ]),
                    Line::from(Span::styled(format!("  {c}"), Style::default().fg(theme::MUTED))),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled("  y", Style::default().fg(Color::Rgb(74, 222, 128)).bold()),
                        Span::raw(" SIGTERM   "),
                        Span::styled("9", Style::default().fg(Color::Rgb(248, 113, 113)).bold()),
                        Span::raw(" SIGKILL   "),
                        Span::styled("any other key", Style::default().fg(theme::MUTED)),
                        Span::raw(" cancel"),
                    ]),
                ],
                64,
                8,
            )
        }
        Popup::Message(m) => (
            "hsmi",
            vec![Line::from(""), Line::from(format!("  {m}"))],
            (m.chars().count() as u16 + 8).max(30),
            5,
        ),
    };
    let r = centered(area, w, h);
    f.render_widget(Clear, r);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::ACCENT))
        .title(title(t))
        .style(Style::default().bg(Color::Rgb(20, 21, 34)).fg(theme::TEXT));
    f.render_widget(Paragraph::new(lines).block(block).wrap(Wrap { trim: false }), r);
}
