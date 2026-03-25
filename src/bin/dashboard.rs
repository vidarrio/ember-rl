//! `ember-dashboard` -- interactive run browser.
//!
//! Scans a `runs/` directory and serves a live dashboard for browsing past
//! training runs.
//!
//! # Run from source
//!
//!   cargo run --bin ember-dashboard --features dashboard
//!   cargo run --bin ember-dashboard --features dashboard -- --dir path/to/runs
//!
//! # Install globally
//!
//!   cargo install ember-rl --features dashboard
//!   ember-dashboard --dir path/to/runs

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let dir = args
        .iter()
        .position(|a| a == "--dir")
        .and_then(|p| args.get(p + 1))
        .map(String::as_str)
        .unwrap_or("runs");

    // blocks until the process is killed
    ember_rl::dashboard::serve_runs(dir, 6006);
}
