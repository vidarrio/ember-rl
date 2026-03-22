use std::collections::HashMap;

use rl_traits::EpisodeStatus;

// ── EpisodeStatus serde ───────────────────────────────────────────────────────
// EpisodeStatus lives in rl-traits which doesn't derive serde.
// We use a string-based representation via #[serde(with)].

mod episode_status_serde {
    use rl_traits::EpisodeStatus;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(status: &EpisodeStatus, s: S) -> Result<S::Ok, S::Error> {
        let tag = match status {
            EpisodeStatus::Continuing => "Continuing",
            EpisodeStatus::Terminated => "Terminated",
            EpisodeStatus::Truncated => "Truncated",
        };
        tag.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<EpisodeStatus, D::Error> {
        let tag = String::deserialize(d)?;
        match tag.as_str() {
            "Continuing" => Ok(EpisodeStatus::Continuing),
            "Terminated" => Ok(EpisodeStatus::Terminated),
            "Truncated" => Ok(EpisodeStatus::Truncated),
            other => Err(serde::de::Error::unknown_variant(other, &["Continuing", "Terminated", "Truncated"])),
        }
    }
}

// ── Records ──────────────────────────────────────────────────────────────────

/// Stats recorded for a single completed episode.
///
/// Produced by the runner at every episode boundary and fed to `StatsTracker`.
/// The `extras` map holds any additional per-episode scalars the user wants
/// to track (e.g. custom environment metrics).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EpisodeRecord {
    /// Total undiscounted reward for the episode.
    pub total_reward: f64,

    /// Number of steps in the episode.
    pub length: usize,

    /// How the episode ended.
    #[serde(with = "episode_status_serde")]
    pub status: EpisodeStatus,

    /// Arbitrary scalar extras provided by the user or environment.
    pub extras: HashMap<String, f64>,
}

impl EpisodeRecord {
    pub fn new(total_reward: f64, length: usize, status: EpisodeStatus) -> Self {
        Self {
            total_reward,
            length,
            status,
            extras: HashMap::new(),
        }
    }

    pub fn with_extra(mut self, key: impl Into<String>, value: f64) -> Self {
        self.extras.insert(key.into(), value);
        self
    }
}

// ── Aggregator trait ─────────────────────────────────────────────────────────

/// Accumulates a stream of `f64` values into a single summary statistic.
///
/// Implement this to add custom aggregators. Built-ins: [`Mean`], [`Max`],
/// [`Min`], [`Last`], [`RollingMean`].
pub trait Aggregator: Send + Sync {
    /// Record a new value.
    fn update(&mut self, value: f64);

    /// Return the current aggregate. `f64::NAN` if no values have been seen.
    fn value(&self) -> f64;

    /// Clear all accumulated values, as if freshly constructed.
    fn reset(&mut self);
}

// ── Built-in aggregators ──────────────────────────────────────────────────────

/// Running mean over all values seen since the last reset.
#[derive(Debug, Clone, Default)]
pub struct Mean {
    sum: f64,
    count: usize,
}

impl Aggregator for Mean {
    fn update(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
    }

    fn value(&self) -> f64 {
        if self.count == 0 { f64::NAN } else { self.sum / self.count as f64 }
    }

    fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
    }
}

/// Maximum value seen since the last reset.
#[derive(Debug, Clone, Default)]
pub struct Max {
    max: Option<f64>,
}

impl Aggregator for Max {
    fn update(&mut self, value: f64) {
        self.max = Some(self.max.map_or(value, |m| m.max(value)));
    }

    fn value(&self) -> f64 {
        self.max.unwrap_or(f64::NAN)
    }

    fn reset(&mut self) {
        self.max = None;
    }
}

/// Minimum value seen since the last reset.
#[derive(Debug, Clone, Default)]
pub struct Min {
    min: Option<f64>,
}

impl Aggregator for Min {
    fn update(&mut self, value: f64) {
        self.min = Some(self.min.map_or(value, |m| m.min(value)));
    }

    fn value(&self) -> f64 {
        self.min.unwrap_or(f64::NAN)
    }

    fn reset(&mut self) {
        self.min = None;
    }
}

/// The most recent value, ignoring history.
#[derive(Debug, Clone, Default)]
pub struct Last {
    last: Option<f64>,
}

impl Aggregator for Last {
    fn update(&mut self, value: f64) {
        self.last = Some(value);
    }

    fn value(&self) -> f64 {
        self.last.unwrap_or(f64::NAN)
    }

    fn reset(&mut self) {
        self.last = None;
    }
}

/// Mean over the last `window` values (sliding window).
#[derive(Debug, Clone)]
pub struct RollingMean {
    window: usize,
    buf: std::collections::VecDeque<f64>,
}

impl RollingMean {
    pub fn new(window: usize) -> Self {
        assert!(window > 0, "window must be > 0");
        Self { window, buf: std::collections::VecDeque::with_capacity(window) }
    }
}

impl Aggregator for RollingMean {
    fn update(&mut self, value: f64) {
        if self.buf.len() == self.window {
            self.buf.pop_front();
        }
        self.buf.push_back(value);
    }

    fn value(&self) -> f64 {
        if self.buf.is_empty() {
            f64::NAN
        } else {
            self.buf.iter().sum::<f64>() / self.buf.len() as f64
        }
    }

    fn reset(&mut self) {
        self.buf.clear();
    }
}

// ── StatsTracker ──────────────────────────────────────────────────────────────

struct TrackedStat {
    name: String,
    extractor: Box<dyn Fn(&EpisodeRecord) -> f64 + Send + Sync>,
    aggregator: Box<dyn Aggregator>,
}

/// Accumulates per-episode stats and reports summary aggregates.
///
/// By default tracks `episode_reward` (mean) and `episode_length` (mean).
/// Use the builder methods to add custom stats or change aggregators.
///
/// # Usage
///
/// ```rust,ignore
/// let mut tracker = StatsTracker::new()
///     .with("episode_reward_max", StatSource::TotalReward, Max::default())
///     .with_custom("ep_len_last10", |r| r.length as f64, RollingMean::new(10));
///
/// // Feed an episode record:
/// tracker.update(&record);
///
/// // Print summary:
/// let summary = tracker.summary();
/// println!("mean reward: {:.1}", summary["episode_reward"]);
/// ```
pub struct StatsTracker {
    stats: Vec<TrackedStat>,
}

/// Predefined sources for `StatsTracker::with`.
pub enum StatSource {
    /// `EpisodeRecord::total_reward`
    TotalReward,
    /// `EpisodeRecord::length` cast to `f64`
    Length,
    /// A key from `EpisodeRecord::extras`
    Extra(String),
}

impl StatSource {
    fn into_extractor(self) -> Box<dyn Fn(&EpisodeRecord) -> f64 + Send + Sync> {
        match self {
            StatSource::TotalReward => Box::new(|r: &EpisodeRecord| r.total_reward),
            StatSource::Length => Box::new(|r: &EpisodeRecord| r.length as f64),
            StatSource::Extra(key) => Box::new(move |r: &EpisodeRecord| {
                r.extras.get(&key).copied().unwrap_or(f64::NAN)
            }),
        }
    }
}

impl StatsTracker {
    /// Create a tracker with the default stats: episode_reward (mean) and episode_length (mean).
    pub fn new() -> Self {
        let mut tracker = Self { stats: Vec::new() };
        tracker = tracker.with("episode_reward", StatSource::TotalReward, Mean::default());
        tracker = tracker.with("episode_length", StatSource::Length, Mean::default());
        tracker
    }

    /// Create a tracker with no default stats.
    pub fn empty() -> Self {
        Self { stats: Vec::new() }
    }

    /// Track a predefined stat field with the given aggregator.
    pub fn with(mut self, name: impl Into<String>, source: StatSource, aggregator: impl Aggregator + 'static) -> Self {
        self.stats.push(TrackedStat {
            name: name.into(),
            extractor: source.into_extractor(),
            aggregator: Box::new(aggregator),
        });
        self
    }

    /// Track an arbitrary derived value from each episode record.
    pub fn with_custom(
        mut self,
        name: impl Into<String>,
        f: impl Fn(&EpisodeRecord) -> f64 + Send + Sync + 'static,
        aggregator: impl Aggregator + 'static,
    ) -> Self {
        self.stats.push(TrackedStat {
            name: name.into(),
            extractor: Box::new(f),
            aggregator: Box::new(aggregator),
        });
        self
    }

    /// Feed a completed episode into all tracked stats.
    pub fn update(&mut self, record: &EpisodeRecord) {
        for stat in &mut self.stats {
            let value = (stat.extractor)(record);
            stat.aggregator.update(value);
        }
    }

    /// Snapshot of all current aggregate values.
    pub fn summary(&self) -> HashMap<String, f64> {
        self.stats.iter()
            .map(|s| (s.name.clone(), s.aggregator.value()))
            .collect()
    }

    /// Reset all aggregators (e.g. between eval runs).
    pub fn reset(&mut self) {
        for stat in &mut self.stats {
            stat.aggregator.reset();
        }
    }
}

impl Default for StatsTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── EvalReport ────────────────────────────────────────────────────────────────

/// Summary statistics from a single evaluation run.
///
/// Returned by `DqnTrainer::eval()`. Contains the aggregated stats from
/// all eval episodes plus the step count at which eval was performed.
#[derive(Debug, Clone)]
pub struct EvalReport {
    /// Total agent steps at the time of evaluation.
    pub total_steps: usize,

    /// Number of episodes evaluated.
    pub n_episodes: usize,

    /// Aggregated statistics (same keys as the `StatsTracker` that produced this).
    pub stats: HashMap<String, f64>,
}

impl EvalReport {
    pub fn new(total_steps: usize, n_episodes: usize, stats: HashMap<String, f64>) -> Self {
        Self { total_steps, n_episodes, stats }
    }

    /// Pretty-print all stats to stdout.
    pub fn print(&self) {
        println!("=== Eval @ step {} ({} episodes) ===", self.total_steps, self.n_episodes);
        let mut keys: Vec<_> = self.stats.keys().collect();
        keys.sort();
        for key in keys {
            println!("  {}: {:.3}", key, self.stats[key]);
        }
    }
}

