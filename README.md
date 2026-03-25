# ember-rl

[![crates.io](https://img.shields.io/crates/v/ember-rl.svg)](https://crates.io/crates/ember-rl)
[![docs.rs](https://docs.rs/ember-rl/badge.svg)](https://docs.rs/ember-rl)
[![CI](https://github.com/vidarrio/ember-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/vidarrio/ember-rl/actions/workflows/ci.yml)

Algorithm implementations for the Rust RL ecosystem, powered by [Burn](https://burn.dev).

`ember-rl` provides ready-to-use RL algorithms that work with any environment
implementing [`rl-traits`](https://crates.io/crates/rl-traits). It handles the
neural networks, replay buffers, and training loops — you bring the environment.

## Ecosystem

| Crate | Role |
|---|---|
| [`rl-traits`](https://crates.io/crates/rl-traits) | Shared traits and types |
| **ember-rl** | Algorithm implementations (DQN, PPO, SAC) using Burn (this crate) |
| [`bevy-gym`](https://crates.io/crates/bevy-gym) | Bevy ECS plugin for parallelised environment simulation |

## Algorithms

| Algorithm | Status | Action space |
|---|---|---|
| DQN | Stable | Discrete |
| PPO | Stable | Discrete (continuous planned) |
| SAC | Planned | Continuous |

## Usage

Add to `Cargo.toml`:

```toml
[dependencies]
ember-rl = "0.3"
burn = { version = "0.20.1", features = ["ndarray", "autodiff"] }
```

### Training with `DqnTrainer`

The simplest entry point — create an agent, wrap it in a trainer, iterate:

```rust
use burn::backend::{Autodiff, NdArray};
use ember_rl::{
    algorithms::dqn::{DqnAgent, DqnConfig},
    encoding::{UsizeActionMapper, VecEncoder},
    training::{DqnTrainer, TrainingRun},
};

type B = Autodiff<NdArray>;

let config = DqnConfig::default();
let agent = DqnAgent::<MyEnv, _, _, B>::new(
    VecEncoder::new(obs_size),
    UsizeActionMapper::new(num_actions),
    config.clone(),
    Default::default(), // device
    42,                 // seed
);

// Attach a named run for automatic checkpointing and JSONL logging
let run = TrainingRun::create("my_experiment", "v1")?;
run.write_config(&(&config, VecEncoder::new(obs_size), UsizeActionMapper::new(num_actions)))?;

let mut trainer = DqnTrainer::new(MyEnv::new(), agent)
    .with_run(run)
    .with_checkpoint_freq(10_000)
    .with_keep_checkpoints(3);

// Iterator-style — full control over the loop
for step in trainer.steps().take(100_000) {
    if step.episode_done {
        println!("ep {}  reward {:.1}  ε {:.3}",
            step.episode, step.episode_reward, step.epsilon);
    }
}

// Eval at end — saves best.mpk automatically
let report = trainer.eval(20);
report.print();
```

### `TrainingSession` — loop-agnostic coordinator

`TrainingSession` is the composable core behind `DqnTrainer`. Use it directly
when your training loop is owned externally — for example, in a Bevy ECS system:

```rust
use ember_rl::training::{TrainingSession, TrainingRun};
use ember_rl::traits::ActMode;

// Any LearningAgent implementation works here
let session = TrainingSession::new(agent)
    .with_run(TrainingRun::create("my_experiment", "v1")?)
    .with_checkpoint_freq(10_000)
    .with_keep_checkpoints(3);

// Each environment step:
let action = session.act(&obs, ActMode::Explore);
session.observe(experience);   // auto-checkpoints at milestones

// Each episode end:
session.on_episode(total_reward, steps, status, env_extras);
// → logs to JSONL, merges agent + env extras, saves best checkpoint if improved

if session.is_done() { break; }
```

### Evaluation

```rust
// Eval at the end of training — returns an EvalReport
let report = trainer.eval(20);
report.print();

// Or load a saved checkpoint for inference (no autodiff overhead)
use burn::backend::NdArray;
use ember_rl::algorithms::dqn::DqnPolicy;

let policy = DqnPolicy::<MyEnv, _, _, NdArray>::new(encoder, mapper, &config, device)
    .load("runs/my_experiment/v1")?;

let action = policy.act(&observation);
```

### Convert a trained agent to an inference policy

```rust
// into_policy() strips training state and downcasts to a plain Backend
let policy = trainer.into_agent().into_policy();
```

### Resuming training

```rust
let run = TrainingRun::resume("runs/my_experiment/v1")?; // picks latest timestamp
println!("resuming from step {}", run.metadata.total_steps);
```

### Custom replay buffers

```rust
// Swap in any ReplayBuffer implementation (e.g. PER)
let agent = DqnAgent::<MyEnv, _, _, B, MyPER>::new_with_buffer(
    encoder, mapper, config, device, seed, my_per_buffer,
);
```

## Training run directory layout

`TrainingRun` manages a versioned on-disk structure:

```
runs/<name>/<version>/<YYYYMMDD_HHMMSS>/
    metadata.json          ← name, version, step counts, timestamps
    config.json            ← serialized hyperparams, encoder, action mapper
    checkpoints/
        step_<N>.mpk       ← periodic checkpoints (pruned to keep_last n)
        latest.mpk         ← most recent checkpoint
        best.mpk           ← best eval-reward checkpoint
    train_episodes.jsonl   ← one EpisodeRecord per line (reward, length, extras)
    eval_episodes.jsonl    ← eval episodes tagged with total_steps_at_eval
```

## Stats

The `stats` module provides composable, algorithm-independent statistics tracking.
Both algorithms and environments can register the stats they want to collect:

```rust
use ember_rl::stats::{StatsTracker, StatSource, Mean, Max, Std, RollingMean};

// Default tracker: episode_reward (mean) and episode_length (mean)
let mut tracker = StatsTracker::new()
    .with("reward_max",   StatSource::TotalReward, Max::default())
    .with("reward_std",   StatSource::TotalReward, Std::default())
    .with_custom("last10_reward", |r| r.total_reward, RollingMean::new(10));

tracker.update(&episode_record);
let summary = tracker.summary(); // HashMap<String, f64>
```

Available aggregators: `Mean`, `Max`, `Min`, `Last`, `RollingMean`, `Std`.

Per-episode dynamics (e.g. training loss) are captured by the agent via its own
internal aggregators and exposed through `LearningAgent::episode_extras()`.
These are merged with environment extras (`Environment::episode_extras()` from
`rl-traits`) into each `EpisodeRecord` automatically by `TrainingSession`.

## Implementing `ObservationEncoder`

`ember-rl` bridges the generic `rl-traits` world to Burn tensors through two
traits you implement for your observation and action types:

```rust
use ember_rl::encoding::{ObservationEncoder, DiscreteActionMapper};

struct MyEncoder;
impl<B: Backend> ObservationEncoder<Vec<f32>, B> for MyEncoder {
    fn obs_size(&self) -> usize { 4 }
    fn encode(&self, obs: &Vec<f32>, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats(obs.as_slice(), device)
    }
}

struct MyMapper;
impl DiscreteActionMapper<MyAction> for MyMapper {
    fn num_actions(&self) -> usize { 2 }
    fn action_to_index(&self, action: &MyAction) -> usize { /* ... */ 0 }
    fn index_to_action(&self, index: usize) -> MyAction { /* ... */ }
}
```

Built-in `VecEncoder` and `UsizeActionMapper` cover the common `Vec<f32>` /
`usize` case without any boilerplate.

## Feature flags

| Feature | Description |
|---|---|
| `envs` | Reference environments (CartPole-v1) |
| `dashboard` | `ember-dashboard` binary for browsing training runs |

## Reference environments

Enable with `--features envs`:

```toml
ember-rl = { version = "0.3", features = ["envs"] }
```

| Environment | Description |
|---|---|
| `CartPole-v1` | Classic balance task matching the Gymnasium spec |

## Dashboard

`ember-rl` ships an `ember-dashboard` binary for browsing training runs. It reads
the `train_episodes.jsonl` files written by `TrainingRun` and serves live-updating
charts — no changes to your training code required.

```
# Browse runs/ in the current directory
cargo run --bin ember-dashboard --features dashboard

# Browse a specific directory
cargo run --bin ember-dashboard --features dashboard -- --dir path/to/runs

# Install globally
cargo install ember-rl --features dashboard
ember-dashboard
ember-dashboard --dir path/to/runs
```

Open `http://localhost:6006` in a browser. The dashboard auto-refreshes every
2 seconds and shows a pulsing indicator next to any run that is actively being
trained. Use the run selector to switch between runs.

Charts shown: episode reward, episode length, exploration rate (ε), and loss.

## Examples

| Example | Algorithm | Notes |
|---|---|---|
| [`cartpole_dqn`](docs/examples/cartpole_dqn.md) | DQN | Train + eval CartPole-v1, checkpoint resume |
| [`cartpole_ppo`](docs/examples/cartpole_ppo.md) | PPO | Train CartPole-v1 with TrainingSession |

## Algorithm notes

Detailed hyperparameter references and implementation notes:

- [DQN](docs/algorithms/dqn.md)
- [PPO](docs/algorithms/ppo.md)

## Development

This crate was developed with the assistance of AI coding tools (Claude by Anthropic).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.
