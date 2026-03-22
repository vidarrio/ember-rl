use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::Local;
use serde::{Deserialize, Serialize};

use crate::stats::EpisodeRecord;

// ── Metadata ──────────────────────────────────────────────────────────────────

/// Persisted metadata for a training run.
///
/// Written to `metadata.json` in the run directory. Updated after each
/// checkpoint via `TrainingRun::update_metadata`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    /// Human-readable name (e.g. `"cartpole"`).
    pub name: String,

    /// Version string (e.g. `"v1"`, `"baseline"`).
    pub version: String,

    /// Timestamp string used as the run directory name (`YYYYMMDD_HHMMSS`).
    pub run_id: String,

    /// Total environment steps at last update.
    pub total_steps: usize,

    /// Total episodes completed at last update.
    pub total_episodes: usize,

    /// ISO-8601 datetime this run was created.
    pub started_at: String,

    /// ISO-8601 datetime of the last metadata update.
    pub last_updated: String,
}

// ── EvalEntry (JSONL row for eval episodes) ────────────────────────────────────

#[derive(Serialize)]
struct EvalEntry<'a> {
    total_steps_at_eval: usize,
    #[serde(flatten)]
    record: &'a EpisodeRecord,
}

// ── TrainingRun ───────────────────────────────────────────────────────────────

/// Manages the on-disk artefacts for a single training run.
///
/// Directory layout:
/// ```text
/// runs/<name>/<version>/<YYYYMMDD_HHMMSS>/
///     metadata.json          ← name, version, step counts, timestamps
///     config.json            ← serialized hyperparams (written by caller)
///     checkpoints/
///         step_<N>.mpk       ← periodic checkpoints
///         latest.mpk         ← symlink-equivalent: overwritten each checkpoint
///         best.mpk           ← best eval-reward checkpoint
///     train_episodes.jsonl   ← one EpisodeRecord per line (training)
///     eval_episodes.jsonl    ← one tagged EpisodeRecord per line (eval)
/// ```
///
/// `TrainingRun` is **not** generic over the neural network backend. It manages
/// directories and JSON; the caller (e.g. `DqnTrainer`) handles actual
/// network serialization by using the paths returned by the checkpoint methods.
///
/// # Usage
///
/// ```rust,ignore
/// // Start a new run
/// let run = TrainingRun::create("cartpole", "v1")?;
/// run.write_config(&(&config, &encoder, &mapper))?;
///
/// // During training
/// run.log_train_episode(&episode_record)?;
/// run.update_metadata(total_steps, total_episodes)?;
/// // (save network to run.checkpoint_path(step) yourself)
///
/// // Resume
/// let run = TrainingRun::resume("runs/cartpole/v1")?; // picks latest
/// ```
pub struct TrainingRun {
    /// Root directory for this run (`.../runs/<name>/<version>/<run_id>/`).
    dir: PathBuf,

    /// Loaded/created metadata.
    pub metadata: RunMetadata,
}

impl TrainingRun {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Create a brand-new run directory under `runs/<name>/<version>/<timestamp>/`.
    ///
    /// Returns an error if the directory cannot be created or metadata cannot
    /// be written.
    pub fn create(name: impl Into<String>, version: impl Into<String>) -> std::io::Result<Self> {
        let name = name.into();
        let version = version.into();
        let run_id = Local::now().format("%Y%m%d_%H%M%S").to_string();
        let now = Local::now().to_rfc3339();

        let dir = PathBuf::from("runs")
            .join(&name)
            .join(&version)
            .join(&run_id);

        fs::create_dir_all(dir.join("checkpoints"))?;

        let metadata = RunMetadata {
            name,
            version,
            run_id,
            total_steps: 0,
            total_episodes: 0,
            started_at: now.clone(),
            last_updated: now,
        };

        let run = Self { dir, metadata };
        run.write_metadata()?;
        Ok(run)
    }

    /// Resume the most recent run found under `base_path`.
    ///
    /// `base_path` can be:
    /// - An exact run directory (`runs/cartpole/v1/20260322_120000`) — used directly.
    /// - A name/version directory (`runs/cartpole/v1`) — picks the lexicographically
    ///   latest subdirectory (timestamps sort correctly).
    /// - A name directory (`runs/cartpole`) — picks latest version, then latest run.
    ///
    /// Returns an error if no run is found or `metadata.json` is missing/corrupt.
    pub fn resume(base_path: impl AsRef<Path>) -> std::io::Result<Self> {
        let dir = Self::resolve_latest(base_path.as_ref())?;
        let metadata_path = dir.join("metadata.json");
        let raw = fs::read_to_string(&metadata_path)?;
        let metadata: RunMetadata = serde_json::from_str(&raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(Self { dir, metadata })
    }

    /// The root directory of this run.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    // ── Config ────────────────────────────────────────────────────────────────

    /// Write an arbitrary serialisable value to `config.json`.
    ///
    /// Typically called once after `create` with a tuple of
    /// `(&config, &encoder, &action_mapper)`.
    pub fn write_config<T: Serialize>(&self, config: &T) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(config)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        fs::write(self.dir.join("config.json"), json)
    }

    // ── Checkpoint paths ──────────────────────────────────────────────────────

    /// Path for a numbered checkpoint: `checkpoints/step_<N>.mpk`.
    ///
    /// Pass this to `DqnAgent::save` (or `network.save_file`).
    pub fn checkpoint_path(&self, step: usize) -> PathBuf {
        self.dir.join("checkpoints").join(format!("step_{}.mpk", step))
    }

    /// Path for the rolling "latest" checkpoint: `checkpoints/latest.mpk`.
    ///
    /// Overwrite this on every checkpoint save so users can always resume
    /// from the most recent state without knowing the step number.
    pub fn latest_checkpoint_path(&self) -> PathBuf {
        self.dir.join("checkpoints").join("latest.mpk")
    }

    /// Path for the best-eval-reward checkpoint: `checkpoints/best.mpk`.
    pub fn best_checkpoint_path(&self) -> PathBuf {
        self.dir.join("checkpoints").join("best.mpk")
    }

    /// Delete old numbered checkpoints, keeping the `keep` most recent.
    ///
    /// `latest.mpk` and `best.mpk` are never deleted.
    pub fn prune_checkpoints(&self, keep: usize) -> std::io::Result<()> {
        let ckpt_dir = self.dir.join("checkpoints");
        let mut numbered: Vec<PathBuf> = fs::read_dir(&ckpt_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("step_") && n.ends_with(".mpk"))
                    .unwrap_or(false)
            })
            .collect();

        // Sort lexicographically — step_ prefix + zero-padded or not: sort by step number
        numbered.sort_by_key(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.strip_prefix("step_"))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0)
        });

        let to_delete = numbered.len().saturating_sub(keep);
        for path in numbered.into_iter().take(to_delete) {
            fs::remove_file(path)?;
        }
        Ok(())
    }

    // ── Stats logging ─────────────────────────────────────────────────────────

    /// Append an episode record to `train_episodes.jsonl`.
    pub fn log_train_episode(&self, record: &EpisodeRecord) -> std::io::Result<()> {
        self.append_jsonl("train_episodes.jsonl", record)
    }

    /// Append an episode record (tagged with `total_steps_at_eval`) to `eval_episodes.jsonl`.
    pub fn log_eval_episode(&self, record: &EpisodeRecord, total_steps: usize) -> std::io::Result<()> {
        let entry = EvalEntry { total_steps_at_eval: total_steps, record };
        self.append_jsonl("eval_episodes.jsonl", &entry)
    }

    // ── Metadata ──────────────────────────────────────────────────────────────

    /// Update step/episode counts and `last_updated` timestamp, then flush to disk.
    pub fn update_metadata(&mut self, total_steps: usize, total_episodes: usize) -> std::io::Result<()> {
        self.metadata.total_steps = total_steps;
        self.metadata.total_episodes = total_episodes;
        self.metadata.last_updated = Local::now().to_rfc3339();
        self.write_metadata()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn write_metadata(&self) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        fs::write(self.dir.join("metadata.json"), json)
    }

    fn append_jsonl<T: Serialize>(&self, filename: &str, value: &T) -> std::io::Result<()> {
        let line = serde_json::to_string(value)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join(filename))?;
        writeln!(file, "{}", line)
    }

    /// Walk `path` downward, always picking the lexicographically latest child
    /// directory until we find one that contains `metadata.json`.
    fn resolve_latest(path: &Path) -> std::io::Result<PathBuf> {
        if path.join("metadata.json").exists() {
            return Ok(path.to_path_buf());
        }

        let latest = Self::latest_subdir(path)?;
        Self::resolve_latest(&latest)
    }

    fn latest_subdir(dir: &Path) -> std::io::Result<PathBuf> {
        let mut subdirs: Vec<PathBuf> = fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();

        if subdirs.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no subdirectories in {}", dir.display()),
            ));
        }

        subdirs.sort();
        Ok(subdirs.pop().unwrap())
    }
}
