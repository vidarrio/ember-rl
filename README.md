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

| Algorithm | Status |
|---|---|
| DQN | Stable |
| PPO | Planned |
| SAC | Planned |

## Usage

Add to `Cargo.toml`:

```toml
[dependencies]
ember-rl = "0.2"
burn = { version = "0.20.1", features = ["ndarray", "autodiff"] }
```

### Training

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

// Attach a named run for checkpointing and stats
let run = TrainingRun::create("my_experiment", "v1")?;
run.write_config(&(&config, VecEncoder::new(obs_size), UsizeActionMapper::new(num_actions)))?;

let mut trainer = DqnTrainer::new(MyEnv::new(), agent, 0)
    .with_run(run)
    .with_checkpoint_freq(10_000)
    .with_keep_checkpoints(3);

// Imperative: runs n steps, saves checkpoints, logs episodes
trainer.train(100_000);

// Or iterator-style for manual control
for step in trainer.steps().take(100_000) {
    if step.episode_done {
        println!("ep {}  reward {:.1}  ε {:.3}",
            step.episode, step.episode_reward, step.epsilon);
    }
}
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

### Convert a trained agent directly to an inference policy

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
    train_episodes.jsonl   ← one EpisodeRecord per line
    eval_episodes.jsonl    ← eval episodes tagged with total_steps_at_eval
```

## Stats

The `stats` module provides composable statistics tracking:

```rust
use ember_rl::stats::{StatsTracker, StatSource, Mean, Max, RollingMean};

let mut tracker = StatsTracker::new()  // default: episode_reward (mean), episode_length (mean)
    .with("reward_max", StatSource::TotalReward, Max::default())
    .with_custom("last10_reward", |r| r.total_reward, RollingMean::new(10));

tracker.update(&episode_record);
let summary = tracker.summary(); // HashMap<String, f64>
```

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
`usize` case without any boilerplate. Both implement `serde::Serialize +
Deserialize`, so they can be written to `config.json` by `TrainingRun`.

## Reference environments

Enable with `--features envs`:

```toml
ember-rl = { version = "0.2", features = ["envs"] }
```

| Environment | Description |
|---|---|
| `CartPole-v1` | Classic balance task matching the Gymnasium spec |

## Running the CartPole example

```
# Train (saves checkpoints to runs/cartpole/v1/<timestamp>/)
cargo run --example cartpole --features envs --release

# Eval from the latest saved run
cargo run --example cartpole --features envs --release -- --eval runs/cartpole/v1
```

## DQN notes

- **Two separate RNGs.** The trainer drives ε-greedy exploration; the agent
  drives buffer sampling. These are intentionally decoupled — sharing a single
  RNG causes subtle learning instability.
- **`DqnConfig::default()`** uses conservative general-purpose hyperparameters.
  Domain-specific examples override them explicitly.
- **Epsilon decay** is linear from `epsilon_start` to `epsilon_end` over
  `epsilon_decay_steps`, then flat.
- **Target network** is a hard-copy of the online network, updated every
  `target_update_freq` steps.
- **Checkpoints** use Burn's `CompactRecorder` (MessagePack format, `.mpk`).
  Only network weights are saved — sufficient for inference. Resume training
  by calling `agent.load(path)` followed by `agent.set_total_steps(n)`.

## Development

This crate was developed with the assistance of AI coding tools (Claude by Anthropic).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.
