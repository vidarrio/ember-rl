# cartpole_ppo

PPO training on CartPole-v1.

## Usage

```sh
cargo run --example cartpole_ppo --features envs --release
```

## What it demonstrates

- Building a `PpoAgent` with `VecEncoder` and `UsizeActionMapper`
- Attaching a `TrainingRun` for automatic checkpointing and JSONL logging
- Driving `TrainingSession` from a plain Rust loop (no external trainer required)
