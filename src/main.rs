//! hsmi — htop for your GPUs. Watch the GEMMs go brrr.

mod app;
mod procinfo;
mod source;
mod theme;
mod ui;
mod widgets;

use std::time::Duration;

use anyhow::Result;
use clap::Parser;

use source::{DemoSource, NvmlSource, Source};

#[derive(Parser)]
#[command(name = "hsmi", version, about = "htop for your GPUs — watch the GEMMs go brrr")]
struct Args {
    /// Sampling interval in milliseconds (change live with + / -)
    #[arg(short, long, default_value_t = 500)]
    interval: u64,

    /// Animation frame rate
    #[arg(long, default_value_t = 30)]
    fps: u32,

    /// Use synthetic data instead of NVML (no GPU needed)
    #[arg(long)]
    demo: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    // Initialise NVML before touching the terminal so errors print normally.
    let src: Box<dyn Source> = if args.demo {
        Box::new(DemoSource::new())
    } else {
        Box::new(NvmlSource::new()?)
    };
    let interval = Duration::from_millis(args.interval.clamp(100, 5000));
    let mut terminal = ratatui::init();
    let res = app::run(&mut terminal, src, interval, args.fps);
    ratatui::restore();
    res
}
