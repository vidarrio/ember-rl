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
| `bevy-gym` *(planned)* | Bevy ECS plugin for visualising and parallelising environments |

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
ember-rl = "*"
burn = { version = "0.20.1", features = ["ndarray", "autodiff"] }
```

### DQN on a custom environment

```rust
use burn::backend::{Autodiff, NdArray};
use ember_rl::{
    algorithms::dqn::{DqnAgent, DqnConfig},
    encoding::{UsizeActionMapper, VecEncoder},
    training::DqnRunner,
};

type B = Autodiff<NdArray>;

let config = DqnConfig::default();
let agent = DqnAgent::<MyEnv, _, _, B>::new(
    VecEncoder::new(obs_size),
    UsizeActionMapper::new(num_actions),
    config,
    Default::default(), // device
    42,                 // seed
);

let mut runner = DqnRunner::new(MyEnv::new(), agent, 0);

for step in runner.steps().take(100_000) {
    if step.episode_done {
        println!("ep {}  reward {:.1}  ε {:.3}",
            step.episode, step.episode_reward, step.epsilon);
    }
}
```

The runner exposes training as an infinite iterator. Use `.take(n)` to cap steps,
or `break` on a solved condition — no callbacks, no inversion of control.

### Implementing `ObservationEncoder`

`ember-rl` bridges the generic `rl-traits` world to Burn tensors through two
traits you implement for your observation and action types:

```rust
use ember_rl::encoding::{ObservationEncoder, DiscreteActionMapper};

// Encode a Vec<f32> observation into a 1-D Burn tensor
struct MyEncoder;
impl<B: Backend> ObservationEncoder<Vec<f32>, B> for MyEncoder {
    fn obs_size(&self) -> usize { 4 }
    fn encode(&self, obs: &Vec<f32>, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats(obs.as_slice(), device)
    }
}

// Map between usize action indices and your Action type
struct MyMapper;
impl DiscreteActionMapper<MyAction> for MyMapper {
    fn num_actions(&self) -> usize { 2 }
    fn to_index(&self, action: &MyAction) -> usize { /* ... */ }
    fn from_index(&self, index: usize) -> MyAction { /* ... */ }
}
```

Built-in `VecEncoder` and `UsizeActionMapper` cover the common `Vec<f32>` /
`usize` case without any boilerplate.

## Reference environments

Enable with `--features envs`:

```toml
ember-rl = { version = "*", features = ["envs"] }
```

| Environment | Description |
|---|---|
| `CartPole-v1` | Classic balance task matching the Gymnasium spec |

## Running the CartPole example

```
cargo run --example cartpole --features envs --release
```

Expected output: the agent reaches a reward of 500 (episode solved) within
a few hundred episodes.

## DQN notes

- **Two separate RNGs.** The runner drives ε-greedy exploration; the agent
  drives buffer sampling. These are intentionally decoupled — sharing a single
  RNG causes subtle learning instability.
- **`DqnConfig::default()`** uses conservative general-purpose hyperparameters.
  Domain-specific examples override them explicitly.
- **Epsilon decay** is linear from `epsilon_start` to `epsilon_end` over
  `epsilon_decay_steps`, then flat.
- **Target network** is a hard-copy of the online network, updated every
  `target_update_freq` steps.

## Development

This crate was developed with the assistance of AI coding tools (Claude by Anthropic).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.
