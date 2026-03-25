//! Live training dashboard served over HTTP.
//!
//! Start the `ember-dashboard` binary to browse past training runs:
//!
//!   cargo run --bin ember-dashboard --features dashboard
//!   cargo run --bin ember-dashboard --features dashboard -- --dir path/to/runs
//!
//! The dashboard reads the `train_episodes.jsonl` files written by
//! [`TrainingRun`] and serves live-updating charts. No changes to your
//! training code are required.
//!
//! [`TrainingRun`]: crate::training::TrainingRun

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// ── Public types ──────────────────────────────────────────────────────────────

/// Metadata about a single training run, returned by `/api/runs`.
#[derive(Debug, Clone, Serialize)]
pub struct RunInfo {
    /// Filesystem path to the run directory (forward-slash normalised).
    pub path: String,
    pub name: String,
    pub version: String,
    pub run_id: String,
    pub total_steps: usize,
    #[serde(default)]
    pub total_episodes: usize,
    pub started_at: String,
    pub last_updated: String,
    /// `true` if `train_episodes.jsonl` was modified within the last 30 s.
    pub is_live: bool,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Start a standalone run-browser dashboard on `port`. **Blocks until the
/// process exits** -- call this at the end of `main()`.
///
/// Scans `runs_dir` for training runs (any directory three levels deep that
/// contains a `metadata.json`) and serves them in the browser UI.
pub fn serve_runs(runs_dir: impl AsRef<Path>, port: u16) {
    let dir = runs_dir.as_ref().to_owned();

    // Resolve to an absolute path so the user can see exactly where we're looking.
    let abs = std::env::current_dir()
        .map(|cwd| cwd.join(&dir))
        .unwrap_or_else(|_| dir.clone());

    if !abs.exists() {
        eprintln!("warning: runs directory does not exist: {}", abs.display());
    }

    println!("Dashboard at http://localhost:{port}");
    println!("Scanning: {}", abs.display());
    serve(dir, port);
}

// ── HTTP server ───────────────────────────────────────────────────────────────

fn serve(runs_dir: PathBuf, port: u16) {
    let server = tiny_http::Server::http(format!("0.0.0.0:{port}"))
        .expect("failed to bind dashboard server");

    let html_ct = tiny_http::Header::from_bytes(b"Content-Type", b"text/html; charset=utf-8").unwrap();
    let json_ct = tiny_http::Header::from_bytes(b"Content-Type", b"application/json").unwrap();
    let cors    = tiny_http::Header::from_bytes(b"Access-Control-Allow-Origin", b"*").unwrap();

    for request in server.incoming_requests() {
        let url = request.url().to_owned();
        let path = url.split('?').next().unwrap_or("/");

        let _ = match path {
            "/" | "/index.html" => request.respond(
                tiny_http::Response::from_string(HTML).with_header(html_ct.clone()),
            ),
            "/api/runs" => {
                let runs = list_runs(&runs_dir);
                let json = serde_json::to_string(&runs).unwrap_or_else(|_| "[]".to_string());
                request.respond(
                    tiny_http::Response::from_string(json)
                        .with_header(json_ct.clone())
                        .with_header(cors.clone()),
                )
            }
            "/api/stats" => {
                let run_path = query_param(&url, "path")
                    .map(|p| PathBuf::from(p.replace('/', std::path::MAIN_SEPARATOR_STR)))
                    .unwrap_or_default();
                let stats = load_run_stats(&run_path);
                let json = serde_json::to_string(&stats).unwrap_or_else(|_| "[]".to_string());
                request.respond(
                    tiny_http::Response::from_string(json)
                        .with_header(json_ct.clone())
                        .with_header(cors.clone()),
                )
            }
            _ => request.respond(tiny_http::Response::empty(404)),
        };
    }
}

// ── Run discovery ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct MetadataSnapshot {
    name: String,
    version: String,
    run_id: String,
    total_steps: usize,
    #[serde(default)]
    total_episodes: usize,
    started_at: String,
    last_updated: String,
}

/// Walk `base/<name>/<version>/<run_id>/` and return a `RunInfo` for every
/// directory that contains a readable `metadata.json`.
fn list_runs(base: &Path) -> Vec<RunInfo> {
    let mut runs = Vec::new();

    let Ok(names) = std::fs::read_dir(base) else { return runs };
    for name_entry in names.flatten() {
        if !name_entry.file_type().map(|t| t.is_dir()).unwrap_or(false) { continue; }
        let Ok(versions) = std::fs::read_dir(name_entry.path()) else { continue };
        for version_entry in versions.flatten() {
            if !version_entry.file_type().map(|t| t.is_dir()).unwrap_or(false) { continue; }
            let Ok(run_dirs) = std::fs::read_dir(version_entry.path()) else { continue };
            for run_entry in run_dirs.flatten() {
                if !run_entry.file_type().map(|t| t.is_dir()).unwrap_or(false) { continue; }
                let run_path = run_entry.path();
                let Ok(content) = std::fs::read_to_string(run_path.join("metadata.json")) else { continue };
                let Ok(meta) = serde_json::from_str::<MetadataSnapshot>(&content) else { continue };
                let is_live = was_recently_modified(&run_path.join("train_episodes.jsonl"), 30);
                runs.push(RunInfo {
                    path: run_path.to_string_lossy().replace('\\', "/"),
                    name: meta.name,
                    version: meta.version,
                    run_id: meta.run_id,
                    total_steps: meta.total_steps,
                    total_episodes: meta.total_episodes,
                    started_at: meta.started_at,
                    last_updated: meta.last_updated,
                    is_live,
                });
            }
        }
    }

    runs.sort_by(|a, b| b.last_updated.cmp(&a.last_updated));
    runs
}

/// Returns `true` if `path` exists and was modified within the last `secs` seconds.
fn was_recently_modified(path: &Path, secs: u64) -> bool {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .and_then(|t| t.elapsed().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)))
        .map(|elapsed| elapsed.as_secs() < secs)
        .unwrap_or(false)
}

// ── JSONL loading ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct EpisodePoint {
    step: usize,
    reward: f64,
    length: usize,
    #[serde(flatten)]
    extras: HashMap<String, f64>,
}

#[derive(Deserialize)]
struct EpisodeRecordSnapshot {
    total_reward: f64,
    length: usize,
    #[serde(default)]
    extras: HashMap<String, f64>,
}

/// Read `<run_dir>/train_episodes.jsonl` and convert to chart data points.
/// The step axis is reconstructed by cumulative sum of episode lengths.
fn load_run_stats(run_dir: &Path) -> Vec<EpisodePoint> {
    let Ok(content) = std::fs::read_to_string(run_dir.join("train_episodes.jsonl")) else {
        return Vec::new();
    };

    let mut points = Vec::new();
    let mut cumulative_steps = 0usize;

    for line in content.lines() {
        if line.is_empty() { continue; }
        let Ok(record) = serde_json::from_str::<EpisodeRecordSnapshot>(line) else { continue };
        cumulative_steps += record.length;
        points.push(EpisodePoint {
            step: cumulative_steps,
            reward: record.total_reward,
            length: record.length,
            extras: record.extras,
        });
    }

    points
}

// ── URL helpers ───────────────────────────────────────────────────────────────

fn query_param(url: &str, key: &str) -> Option<String> {
    let query = url.split_once('?')?.1;
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        if k == key { Some(url_decode(v)) } else { None }
    })
}

fn url_decode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '%' {
            let h1 = chars.next().unwrap_or('0');
            let h2 = chars.next().unwrap_or('0');
            if let Ok(byte) = u8::from_str_radix(&format!("{h1}{h2}"), 16) {
                out.push(byte as char);
                continue;
            }
        } else if c == '+' {
            out.push(' ');
            continue;
        }
        out.push(c);
    }
    out
}

const HTML: &str = include_str!("dashboard.html");
