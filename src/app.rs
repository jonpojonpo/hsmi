//! Application state, history buffers, input handling and the main loop.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use anyhow::Result;
use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::widgets::TableState;

use crate::procinfo::{self, ProcCache, ProcMeta};
use crate::source::{GpuSample, ProcSample, Source, StaticInfo, SystemInfo};
use crate::widgets::Fire;

pub const HISTORY: usize = 2048;

#[derive(Default)]
pub struct Ring(pub VecDeque<f64>);

impl Ring {
    pub fn push(&mut self, v: f64) {
        if self.0.len() == HISTORY {
            self.0.pop_front();
        }
        self.0.push_back(v);
    }
    pub fn slice(&mut self) -> &[f64] {
        self.0.make_contiguous()
    }
}

#[derive(Default)]
pub struct History {
    pub sm: Ring,
    pub mem_bw: Ring,
    pub vram: Ring,
    pub power: Ring,
    pub temp: Ring,
    pub brr: Ring,
    pub pcie: Ring,
}

/// Values that glide towards the latest sample every frame, so meters move smoothly.
#[derive(Default, Clone, Copy)]
pub struct Smooth {
    pub sm: f64,
    pub mem_bw: f64,
    pub vram: f64,
    pub power: f64,
    pub temp: f64,
    pub fan: f64,
    pub clock: f64,
    pub tx: f64,
    pub rx: f64,
    pub brr: f64,
}

impl Smooth {
    fn ease(&mut self, t: &Smooth, k: f64) {
        let e = |a: &mut f64, b: f64| *a += (b - *a) * k;
        e(&mut self.sm, t.sm);
        e(&mut self.mem_bw, t.mem_bw);
        e(&mut self.vram, t.vram);
        e(&mut self.power, t.power);
        e(&mut self.temp, t.temp);
        e(&mut self.fan, t.fan);
        e(&mut self.clock, t.clock);
        e(&mut self.tx, t.tx);
        e(&mut self.rx, t.rx);
        e(&mut self.brr, t.brr);
    }
}

pub struct GpuState {
    pub info: StaticInfo,
    pub last: Option<GpuSample>,
    pub hist: History,
    pub target: Smooth,
    pub disp: Smooth,
    pub fire: Fire,
    pub fan_phase: f64,
    pub energy_start_mj: Option<u64>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SortBy {
    Vram,
    Sm,
    Pid,
}

impl SortBy {
    pub fn label(self) -> &'static str {
        match self {
            SortBy::Vram => "VRAM",
            SortBy::Sm => "SM%",
            SortBy::Pid => "PID",
        }
    }
    fn next(self) -> Self {
        match self {
            SortBy::Vram => SortBy::Sm,
            SortBy::Sm => SortBy::Pid,
            SortBy::Pid => SortBy::Vram,
        }
    }
}

pub struct ProcRow {
    pub s: ProcSample,
    pub meta: ProcMeta,
}

pub enum Popup {
    None,
    Help,
    Kill { pid: u32, cmd: String },
    Message(String),
}

pub struct App {
    pub sys: SystemInfo,
    pub host: String,
    pub gpus: Vec<GpuState>,
    pub sel: usize,
    pub interval: Duration,
    pub paused: bool,
    pub sort: SortBy,
    pub table: TableState,
    pub rows: Vec<ProcRow>,
    pub popup: Popup,
    pub started: Instant,
    pub frame: u64,
    pub procs: ProcCache,
}

/// GEMM-intensity proxy: how busy the SMs are × how close we are to the power limit.
/// Big matmuls on tensor cores pin both; memory-bound or stalled work doesn't.
/// If GPM (Hopper+) reports real tensor-pipe activity, use that instead.
pub fn brr(s: &GpuSample) -> f64 {
    if let Some(t) = s.tensor {
        return (t / 100.0).clamp(0.0, 1.0);
    }
    let p = match (s.power_w, s.power_limit_w) {
        (Some(p), Some(l)) if l > 0.0 => (p / l).clamp(0.0, 1.0),
        _ => 1.0,
    };
    (s.sm_util / 100.0 * p).clamp(0.0, 1.0)
}

pub fn pcie_max_kbs(gen_: Option<u32>, width: Option<u32>) -> f64 {
    // Per-lane, per-direction usable bandwidth in GB/s.
    let lane = match gen_.unwrap_or(4) {
        1 => 0.25,
        2 => 0.5,
        3 => 0.985,
        4 => 1.969,
        5 => 3.938,
        _ => 7.563,
    };
    lane * width.unwrap_or(16) as f64 * 1e6
}

impl App {
    pub fn new(src: &dyn Source, interval: Duration) -> Self {
        let gpus = src
            .statics()
            .into_iter()
            .map(|info| GpuState {
                info,
                last: None,
                hist: History::default(),
                target: Smooth::default(),
                disp: Smooth::default(),
                fire: Fire::new(),
                fan_phase: 0.0,
                energy_start_mj: None,
            })
            .collect();
        let mut table = TableState::default();
        table.select(Some(0));
        Self {
            sys: src.system(),
            host: procinfo::hostname(),
            gpus,
            sel: 0,
            interval,
            paused: false,
            sort: SortBy::Vram,
            table,
            rows: Vec::new(),
            popup: Popup::None,
            started: Instant::now(),
            frame: 0,
            procs: ProcCache::new(),
        }
    }

    pub fn ingest(&mut self, samples: Vec<GpuSample>) {
        for (g, s) in self.gpus.iter_mut().zip(samples) {
            let total = s.mem_total.max(1) as f64;
            let power_frac = match (s.power_w, s.power_limit_w) {
                (Some(p), Some(l)) if l > 0.0 => p / l,
                _ => 0.0,
            };
            let slowdown = g.info.temp_slowdown.unwrap_or(90).max(1) as f64;
            let temp_frac = s.temp.map(|t| t as f64 / slowdown).unwrap_or(0.0);
            let b = brr(&s);
            let pmax = pcie_max_kbs(g.info.pcie_max_gen, g.info.pcie_max_width);
            let tx = s.pcie_tx_kbs.unwrap_or(0) as f64 / pmax;
            let rx = s.pcie_rx_kbs.unwrap_or(0) as f64 / pmax;

            g.hist.sm.push(s.sm_util);
            g.hist.mem_bw.push(s.mem_bw_util);
            g.hist.vram.push(s.mem_used as f64 / total * 100.0);
            g.hist.power.push(s.power_w.unwrap_or(0.0));
            g.hist.temp.push(s.temp.unwrap_or(0) as f64);
            g.hist.brr.push(b * 100.0);
            g.hist.pcie.push((tx + rx) * 50.0);

            g.target = Smooth {
                sm: s.sm_util / 100.0,
                mem_bw: s.mem_bw_util / 100.0,
                vram: s.mem_used as f64 / total,
                power: power_frac,
                temp: temp_frac,
                fan: s.fan.unwrap_or(0) as f64 / 100.0,
                clock: match (s.sm_clock, g.info.max_sm_clock) {
                    (Some(c), Some(m)) if m > 0 => c as f64 / m as f64,
                    _ => 0.0,
                },
                tx,
                rx,
                brr: b,
            };
            if g.energy_start_mj.is_none() {
                g.energy_start_mj = s.energy_mj;
            }
            g.last = Some(s);
        }
        self.refresh_rows();
    }

    pub fn refresh_rows(&mut self) {
        let Some(g) = self.gpus.get(self.sel) else { return };
        let selected_pid = self
            .table
            .selected()
            .and_then(|i| self.rows.get(i))
            .map(|r| r.s.pid);
        let procs = g.last.as_ref().map(|s| s.procs.clone()).unwrap_or_default();
        let alive: Vec<u32> = self
            .gpus
            .iter()
            .filter_map(|g| g.last.as_ref())
            .flat_map(|s| s.procs.iter().map(|p| p.pid))
            .collect();
        self.procs.retain(&alive);
        let mut rows: Vec<ProcRow> = procs
            .into_iter()
            .map(|s| {
                let mut meta = self.procs.get(s.pid, s.name_hint.as_deref());
                if self.sys.demo {
                    meta.user = "demo".into();
                }
                ProcRow { s, meta }
            })
            .collect();
        match self.sort {
            SortBy::Vram => rows.sort_by(|a, b| b.s.vram.cmp(&a.s.vram).then(a.s.pid.cmp(&b.s.pid))),
            SortBy::Sm => rows.sort_by(|a, b| b.s.sm.cmp(&a.s.sm).then(b.s.vram.cmp(&a.s.vram))),
            SortBy::Pid => rows.sort_by_key(|r| r.s.pid),
        }
        let idx = selected_pid
            .and_then(|pid| rows.iter().position(|r| r.s.pid == pid))
            .unwrap_or_else(|| self.table.selected().unwrap_or(0).min(rows.len().saturating_sub(1)));
        self.table.select(if rows.is_empty() { None } else { Some(idx) });
        self.rows = rows;
    }

    /// Per-frame animation step.
    pub fn tick(&mut self, dt: f64) {
        self.frame += 1;
        // Frame-rate independent easing (~150 ms time constant).
        let k = 1.0 - (-dt / 0.15).exp();
        for g in &mut self.gpus {
            let t = g.target;
            g.disp.ease(&t, k);
            g.fire.step(g.disp.brr);
            // Spinner speed follows the fan.
            g.fan_phase += dt * (0.5 + 14.0 * g.disp.fan);
        }
    }

    /// Returns false when the app should quit.
    pub fn on_key(&mut self, k: KeyEvent) -> bool {
        if k.modifiers.contains(KeyModifiers::CONTROL) && k.code == KeyCode::Char('c') {
            return false;
        }
        match &self.popup {
            Popup::Kill { pid, .. } => {
                let pid = *pid;
                self.popup = Popup::None;
                let sig = match k.code {
                    KeyCode::Char('y') | KeyCode::Char('Y') => Some(libc::SIGTERM),
                    KeyCode::Char('9') => Some(libc::SIGKILL),
                    _ => None,
                };
                if let Some(sig) = sig {
                    let r = unsafe { libc::kill(pid as i32, sig) };
                    if r != 0 {
                        let e = std::io::Error::last_os_error();
                        self.popup = Popup::Message(format!("kill {pid} failed: {e}"));
                    }
                }
                return true;
            }
            Popup::Help | Popup::Message(_) => {
                self.popup = Popup::None;
                return !matches!(k.code, KeyCode::Char('q'));
            }
            Popup::None => {}
        }
        match k.code {
            KeyCode::Char('q') | KeyCode::Esc | KeyCode::F(10) => return false,
            KeyCode::Char('?') | KeyCode::Char('h') | KeyCode::F(1) => self.popup = Popup::Help,
            KeyCode::Tab | KeyCode::Right | KeyCode::Char('l') => {
                self.sel = (self.sel + 1) % self.gpus.len().max(1);
                self.refresh_rows();
            }
            KeyCode::BackTab | KeyCode::Left => {
                let n = self.gpus.len().max(1);
                self.sel = (self.sel + n - 1) % n;
                self.refresh_rows();
            }
            KeyCode::Char(c @ '0'..='9') => {
                let i = c as usize - '0' as usize;
                if i < self.gpus.len() {
                    self.sel = i;
                    self.refresh_rows();
                }
            }
            KeyCode::Down | KeyCode::Char('j') => self.move_sel(1),
            KeyCode::Up | KeyCode::Char('k') => self.move_sel(-1),
            KeyCode::PageDown => self.move_sel(10),
            KeyCode::PageUp => self.move_sel(-10),
            KeyCode::Char('s') | KeyCode::F(6) => {
                self.sort = self.sort.next();
                self.refresh_rows();
            }
            KeyCode::Char('p') | KeyCode::Char(' ') => self.paused = !self.paused,
            KeyCode::Char('+') | KeyCode::Char('=') => {
                self.interval = (self.interval * 2).min(Duration::from_secs(5))
            }
            KeyCode::Char('-') | KeyCode::Char('_') => {
                self.interval = (self.interval / 2).max(Duration::from_millis(100))
            }
            KeyCode::Char('K') | KeyCode::F(9) | KeyCode::Delete => {
                if self.sys.demo {
                    self.popup = Popup::Message("demo mode: nothing to kill".into());
                } else if let Some(r) = self.table.selected().and_then(|i| self.rows.get(i)) {
                    self.popup = Popup::Kill {
                        pid: r.s.pid,
                        cmd: r.meta.cmd.clone(),
                    };
                }
            }
            _ => {}
        }
        true
    }

    fn move_sel(&mut self, d: isize) {
        if self.rows.is_empty() {
            return;
        }
        let cur = self.table.selected().unwrap_or(0) as isize;
        let n = self.rows.len() as isize;
        self.table.select(Some((cur + d).clamp(0, n - 1) as usize));
    }
}

pub fn run(terminal: &mut DefaultTerminal, mut src: Box<dyn Source>, interval: Duration, fps: u32) -> Result<()> {
    let mut app = App::new(src.as_ref(), interval);
    app.ingest(src.sample());
    let frame_dt = Duration::from_secs_f64(1.0 / fps.clamp(1, 120) as f64);
    let mut last_sample = Instant::now();
    let mut last_frame = Instant::now();
    loop {
        let now = Instant::now();
        if !app.paused && now.duration_since(last_sample) >= app.interval {
            last_sample = now;
            app.ingest(src.sample());
        }
        let dt = now.duration_since(last_frame).as_secs_f64();
        last_frame = now;
        app.tick(dt);
        terminal.draw(|f| crate::ui::draw(f, &mut app))?;

        let deadline = now + frame_dt;
        while let Some(left) = deadline.checked_duration_since(Instant::now()) {
            if !event::poll(left)? {
                break;
            }
            if let Event::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Press && !app.on_key(k) {
                    return Ok(());
                }
            }
        }
    }
}
