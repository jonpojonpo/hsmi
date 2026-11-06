//! Data sources: real NVML sampling and a synthetic demo "training run".

use std::collections::HashMap;

use anyhow::{Context, Result};
use nvml_wrapper::bitmasks::device::ThrottleReasons;
use nvml_wrapper::enum_wrappers::device::{
    Clock, PcieUtilCounter, PerformanceState, TemperatureSensor, TemperatureThreshold,
};
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::enums::gpm::GpmMetricId;
use nvml_wrapper::{GpmSample, Nvml};

#[derive(Clone, Debug, Default)]
pub struct SystemInfo {
    pub driver: String,
    pub cuda: String,
    pub demo: bool,
}

/// Things that don't change while we run.
#[derive(Clone, Debug, Default)]
pub struct StaticInfo {
    pub index: u32,
    pub name: String,
    pub mem_total: u64,
    pub max_sm_clock: Option<u32>,
    pub max_mem_clock: Option<u32>,
    pub pcie_max_gen: Option<u32>,
    pub pcie_max_width: Option<u32>,
    pub temp_slowdown: Option<u32>,
    pub cores: Option<u32>,
    pub bus_width: Option<u32>,
    /// GPM (Hopper+) gives real tensor-pipe activity.
    pub has_gpm: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcKind {
    Compute,
    Graphics,
    Both,
}

impl ProcKind {
    pub fn tag(self) -> &'static str {
        match self {
            ProcKind::Compute => "C",
            ProcKind::Graphics => "G",
            ProcKind::Both => "C+G",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProcSample {
    pub pid: u32,
    pub kind: ProcKind,
    pub vram: Option<u64>,
    pub sm: Option<u32>,
    pub mem: Option<u32>,
    /// Name reported by the source (used as a fallback when /proc is unreadable).
    pub name_hint: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct GpuSample {
    pub sm_util: f64,
    pub mem_bw_util: f64,
    pub mem_used: u64,
    pub mem_total: u64,
    pub temp: Option<u32>,
    pub fan: Option<u32>,
    pub power_w: Option<f64>,
    pub power_limit_w: Option<f64>,
    pub sm_clock: Option<u32>,
    pub mem_clock: Option<u32>,
    pub pstate: Option<u8>,
    pub pcie_tx_kbs: Option<u32>,
    pub pcie_rx_kbs: Option<u32>,
    pub pcie_gen: Option<u32>,
    pub pcie_width: Option<u32>,
    pub enc: Option<u32>,
    pub dec: Option<u32>,
    pub throttle: Vec<&'static str>,
    pub energy_mj: Option<u64>,
    /// Real tensor-pipe activity in percent (GPM, Hopper+ only).
    pub tensor: Option<f64>,
    pub procs: Vec<ProcSample>,
}

pub trait Source {
    fn system(&self) -> SystemInfo;
    fn statics(&self) -> Vec<StaticInfo>;
    fn sample(&mut self) -> Vec<GpuSample>;
}

// ───────────────────────────────── NVML ─────────────────────────────────

struct PerGpu {
    last_proc_ts: u64,
    /// pid -> (sm, mem, timestamp µs)
    proc_util: HashMap<u32, (u32, u32, u64)>,
    gpm_prev: Option<GpmSample<'static>>,
}

pub struct NvmlSource {
    nvml: &'static Nvml,
    sys: SystemInfo,
    statics: Vec<StaticInfo>,
    per: Vec<PerGpu>,
}

impl NvmlSource {
    pub fn new() -> Result<Self> {
        let nvml = Nvml::init().context(
            "could not initialise NVML (is the NVIDIA driver installed? try `hsmi --demo`)",
        )?;
        // GPM samples borrow the NVML handle; it lives for the whole program anyway.
        let nvml: &'static Nvml = Box::leak(Box::new(nvml));

        let driver = nvml.sys_driver_version().unwrap_or_else(|_| "?".into());
        let cuda = nvml
            .sys_cuda_driver_version()
            .map(|v| {
                format!(
                    "{}.{}",
                    nvml_wrapper::cuda_driver_version_major(v),
                    nvml_wrapper::cuda_driver_version_minor(v)
                )
            })
            .unwrap_or_else(|_| "?".into());

        let count = nvml.device_count().context("NVML: device count")?;
        let mut statics = Vec::new();
        let mut per = Vec::new();
        for i in 0..count {
            let d = nvml.device_by_index(i)?;
            let has_gpm = d.gpm_support().unwrap_or(false);
            statics.push(StaticInfo {
                index: i,
                name: d.name().unwrap_or_else(|_| format!("GPU {i}")),
                mem_total: d.memory_info().map(|m| m.total).unwrap_or(0),
                max_sm_clock: d.max_clock_info(Clock::Graphics).ok(),
                max_mem_clock: d.max_clock_info(Clock::Memory).ok(),
                pcie_max_gen: d.max_pcie_link_gen().ok(),
                pcie_max_width: d.max_pcie_link_width().ok(),
                temp_slowdown: d.temperature_threshold(TemperatureThreshold::Slowdown).ok(),
                cores: d.num_cores().ok(),
                bus_width: d.memory_bus_width().ok(),
                has_gpm,
            });
            per.push(PerGpu {
                last_proc_ts: 0,
                proc_util: HashMap::new(),
                gpm_prev: None,
            });
        }

        Ok(Self {
            nvml,
            sys: SystemInfo {
                driver,
                cuda,
                demo: false,
            },
            statics,
            per,
        })
    }
}

fn pstate_num(p: PerformanceState) -> Option<u8> {
    use PerformanceState::*;
    Some(match p {
        Zero => 0,
        One => 1,
        Two => 2,
        Three => 3,
        Four => 4,
        Five => 5,
        Six => 6,
        Seven => 7,
        Eight => 8,
        Nine => 9,
        Ten => 10,
        Eleven => 11,
        Twelve => 12,
        Thirteen => 13,
        Fourteen => 14,
        Fifteen => 15,
        Unknown => return None,
    })
}

fn throttle_labels(r: ThrottleReasons) -> Vec<&'static str> {
    let table: &[(ThrottleReasons, &str)] = &[
        (ThrottleReasons::SW_POWER_CAP, "PWR CAP"),
        (ThrottleReasons::HW_POWER_BRAKE_SLOWDOWN, "PWR BRAKE"),
        (ThrottleReasons::SW_THERMAL_SLOWDOWN, "SW THERMAL"),
        (ThrottleReasons::HW_THERMAL_SLOWDOWN, "HW THERMAL"),
        (ThrottleReasons::HW_SLOWDOWN, "HW SLOWDOWN"),
        (ThrottleReasons::SYNC_BOOST, "SYNC BOOST"),
        (ThrottleReasons::APPLICATIONS_CLOCKS_SETTING, "APP CLOCKS"),
        (ThrottleReasons::DISPLAY_CLOCK_SETTING, "DISPLAY"),
        (ThrottleReasons::GPU_IDLE, "IDLE"),
    ];
    table
        .iter()
        .filter(|(f, _)| r.contains(*f))
        .map(|(_, s)| *s)
        .collect()
}

impl Source for NvmlSource {
    fn system(&self) -> SystemInfo {
        self.sys.clone()
    }

    fn statics(&self) -> Vec<StaticInfo> {
        self.statics.clone()
    }

    #[allow(deprecated)]
    fn sample(&mut self) -> Vec<GpuSample> {
        let mut out = Vec::with_capacity(self.statics.len());
        for (i, st) in self.statics.iter().enumerate() {
            let Ok(d) = self.nvml.device_by_index(st.index) else {
                out.push(GpuSample::default());
                continue;
            };
            let per = &mut self.per[i];

            let util = d.utilization_rates().ok();
            let mem = d.memory_info().ok();
            let mut s = GpuSample {
                sm_util: util.as_ref().map(|u| u.gpu as f64).unwrap_or(0.0),
                mem_bw_util: util.as_ref().map(|u| u.memory as f64).unwrap_or(0.0),
                mem_used: mem.as_ref().map(|m| m.used).unwrap_or(0),
                mem_total: mem.as_ref().map(|m| m.total).unwrap_or(st.mem_total),
                temp: d.temperature(TemperatureSensor::Gpu).ok(),
                fan: d.fan_speed(0).ok(),
                power_w: d.power_usage().ok().map(|mw| mw as f64 / 1000.0),
                power_limit_w: d.enforced_power_limit().ok().map(|mw| mw as f64 / 1000.0),
                sm_clock: d.clock_info(Clock::Graphics).ok(),
                mem_clock: d.clock_info(Clock::Memory).ok(),
                pstate: d.performance_state().ok().and_then(pstate_num),
                pcie_tx_kbs: d.pcie_throughput(PcieUtilCounter::Send).ok(),
                pcie_rx_kbs: d.pcie_throughput(PcieUtilCounter::Receive).ok(),
                pcie_gen: d.current_pcie_link_gen().ok(),
                pcie_width: d.current_pcie_link_width().ok(),
                enc: d.encoder_utilization().ok().map(|u| u.utilization),
                dec: d.decoder_utilization().ok().map(|u| u.utilization),
                throttle: d
                    .current_throttle_reasons()
                    .map(throttle_labels)
                    .unwrap_or_default(),
                energy_mj: d.total_energy_consumption().ok(),
                tensor: None,
                procs: Vec::new(),
            };

            // Real tensor activity on GPUs that support GPM.
            if st.has_gpm {
                if let Ok(cur) = d.gpm_sample() {
                    if let Some(prev) = per.gpm_prev.take() {
                        if let Ok(res) = nvml_wrapper::gpm::gpm_metrics_get(
                            self.nvml,
                            &prev,
                            &cur,
                            &[GpmMetricId::AnyTensorUtil],
                        ) {
                            s.tensor = res.into_iter().next().and_then(|r| r.ok()).map(|r| r.value);
                        }
                        let _ = prev.free();
                    }
                    per.gpm_prev = Some(cur);
                }
            }

            // Per-process SM / memory utilisation.
            if let Ok(samples) = d.process_utilization_stats(per.last_proc_ts) {
                for ps in samples {
                    per.last_proc_ts = per.last_proc_ts.max(ps.timestamp);
                    let e = per.proc_util.entry(ps.pid).or_insert((0, 0, 0));
                    if ps.timestamp >= e.2 {
                        *e = (ps.sm_util, ps.mem_util, ps.timestamp);
                    }
                }
            }
            // Forget utilisation that hasn't been refreshed in 3 s.
            let horizon = per.last_proc_ts.saturating_sub(3_000_000);
            per.proc_util.retain(|_, v| v.2 >= horizon);

            let mut procs: HashMap<u32, ProcSample> = HashMap::new();
            let mut add = |list: Vec<nvml_wrapper::struct_wrappers::device::ProcessInfo>,
                           kind: ProcKind| {
                for p in list {
                    let vram = match p.used_gpu_memory {
                        UsedGpuMemory::Used(b) => Some(b),
                        UsedGpuMemory::Unavailable => None,
                    };
                    procs
                        .entry(p.pid)
                        .and_modify(|e| {
                            if e.kind != kind {
                                e.kind = ProcKind::Both;
                            }
                            e.vram = match (e.vram, vram) {
                                (Some(a), Some(b)) => Some(a.max(b)),
                                (a, b) => a.or(b),
                            };
                        })
                        .or_insert(ProcSample {
                            pid: p.pid,
                            kind,
                            vram,
                            sm: None,
                            mem: None,
                            name_hint: None,
                        });
                }
            };
            if let Ok(l) = d.running_compute_processes() {
                add(l, ProcKind::Compute);
            }
            if let Ok(l) = d.running_graphics_processes() {
                add(l, ProcKind::Graphics);
            }
            for p in procs.values_mut() {
                if let Some((sm, mem, _)) = per.proc_util.get(&p.pid) {
                    p.sm = Some(*sm);
                    p.mem = Some(*mem);
                } else {
                    p.sm = Some(0);
                    p.mem = Some(0);
                }
                p.name_hint = self.nvml.sys_process_name(p.pid, 256).ok();
            }
            s.procs = procs.into_values().collect();
            out.push(s);
        }
        out
    }
}

// ───────────────────────────────── demo ─────────────────────────────────

/// Tiny xorshift RNG so we don't need a dependency.
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
    pub fn f64(&mut self) -> f64 {
        self.next_u32() as f64 / u32::MAX as f64
    }
}

pub struct DemoSource {
    t0: std::time::Instant,
    rng: Rng,
    temps: [f64; 2],
    energy: [f64; 2],
    last: std::time::Instant,
}

impl DemoSource {
    pub fn new() -> Self {
        Self {
            t0: std::time::Instant::now(),
            rng: Rng::new(0x9E37_79B9_7F4A_7C15),
            temps: [38.0, 36.0],
            energy: [0.0, 0.0],
            last: std::time::Instant::now(),
        }
    }
}

const GIB: u64 = 1 << 30;

impl Source for DemoSource {
    fn system(&self) -> SystemInfo {
        SystemInfo {
            driver: "demo".into(),
            cuda: "demo".into(),
            demo: true,
        }
    }

    fn statics(&self) -> Vec<StaticInfo> {
        (0..2)
            .map(|i| StaticInfo {
                index: i,
                name: format!("Demo GPU {i} (synthetic)"),
                mem_total: 24 * GIB,
                max_sm_clock: Some(3105),
                max_mem_clock: Some(10501),
                pcie_max_gen: Some(4),
                pcie_max_width: Some(16),
                temp_slowdown: Some(90),
                cores: Some(16384),
                bus_width: Some(384),
                has_gpm: false,
            })
            .collect()
    }

    fn sample(&mut self) -> Vec<GpuSample> {
        let t = self.t0.elapsed().as_secs_f64();
        let dt = self.last.elapsed().as_secs_f64();
        self.last = std::time::Instant::now();
        let mut out = Vec::new();
        for g in 0..2usize {
            let r = &mut self.rng;
            // GPU 0: training loop. GPU 1: a sleepier inference server with bursts.
            let (sm, mem_bw, power) = if g == 0 {
                let warm = (t / 8.0).min(1.0);
                let cycle = t % 60.0;
                let step = (t * 0.9) % 1.0;
                if cycle > 50.0 {
                    // eval + checkpoint
                    if cycle > 57.0 {
                        (2.0, 1.0, 95.0)
                    } else {
                        (55.0 + 10.0 * r.f64(), 40.0, 260.0)
                    }
                } else if step < 0.82 {
                    // fwd/bwd GEMMs
                    (
                        warm * (96.0 + 4.0 * r.f64()),
                        warm * (55.0 + 15.0 * r.f64()),
                        80.0 + warm * (360.0 + 20.0 * r.f64()),
                    )
                } else {
                    // optimizer step / dataloader hiccup
                    (
                        warm * (35.0 + 20.0 * r.f64()),
                        warm * (70.0 + 10.0 * r.f64()),
                        80.0 + warm * (150.0 + 30.0 * r.f64()),
                    )
                }
            } else {
                let burst = ((t / 7.0).sin() * 0.5 + 0.5).powf(3.0);
                let noisy = burst * (0.7 + 0.3 * r.f64());
                (100.0 * noisy, 60.0 * noisy, 30.0 + 300.0 * noisy)
            };
            let limit = 450.0;
            let power = power.min(limit);
            let target_temp = 34.0 + power / limit * 42.0;
            self.temps[g] += (target_temp - self.temps[g]) * (1.0 - (-dt / 6.0).exp());
            self.energy[g] += power * dt * 1000.0;
            let temp = self.temps[g];
            let mut throttle = Vec::new();
            if power >= limit - 5.0 {
                throttle.push("PWR CAP");
            }
            if sm < 5.0 {
                throttle.push("IDLE");
            }
            let used = if g == 0 {
                ((t / 8.0).min(1.0) * 19.4 * GIB as f64) as u64 + GIB / 2
            } else {
                (11.2 * GIB as f64) as u64
            };
            let procs = if g == 0 {
                vec![
                    ProcSample {
                        pid: 41337,
                        kind: ProcKind::Compute,
                        vram: Some(used.saturating_sub(GIB / 2)),
                        sm: Some(sm as u32),
                        mem: Some(mem_bw as u32),
                        name_hint: Some(
                            "python train.py --model llama-1b --bf16 --compile --bs 32".into(),
                        ),
                    },
                    ProcSample {
                        pid: 1984,
                        kind: ProcKind::Graphics,
                        vram: Some(420 << 20),
                        sm: Some(1),
                        mem: Some(0),
                        name_hint: Some("/usr/lib/xorg/Xorg :0".into()),
                    },
                ]
            } else {
                vec![ProcSample {
                    pid: 7777,
                    kind: ProcKind::Compute,
                    vram: Some(used),
                    sm: Some(sm as u32),
                    mem: Some(mem_bw as u32),
                    name_hint: Some("vllm serve qwen-7b --port 8000".into()),
                }]
            };
            let frac = power / limit;
            out.push(GpuSample {
                sm_util: sm.clamp(0.0, 100.0),
                mem_bw_util: mem_bw.clamp(0.0, 100.0),
                mem_used: used,
                mem_total: 24 * GIB,
                temp: Some(temp as u32),
                fan: Some((30.0 + (temp - 35.0).max(0.0) * 1.6).min(100.0) as u32),
                power_w: Some(power),
                power_limit_w: Some(limit),
                sm_clock: Some((210.0 + frac * 2600.0).min(2790.0) as u32),
                mem_clock: Some(if sm > 5.0 { 10501 } else { 405 }),
                pstate: Some(if sm > 5.0 { 2 } else { 8 }),
                pcie_tx_kbs: Some((mem_bw * 900.0 * r.f64()) as u32),
                pcie_rx_kbs: Some((sm * 6000.0 * r.f64()) as u32),
                pcie_gen: Some(if sm > 5.0 { 4 } else { 1 }),
                pcie_width: Some(16),
                enc: Some(0),
                dec: Some(0),
                throttle,
                energy_mj: Some(self.energy[g] as u64),
                tensor: None,
                procs,
            });
        }
        out
    }
}
