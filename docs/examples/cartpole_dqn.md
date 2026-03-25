# cartpole_dqn

DQN training and evaluation on CartPole-v1.

## Usage

```sh
# Train (saves checkpoints to runs/cartpole/v1/<timestamp>/)
cargo run --example cartpole_dqn --features envs --release

# Eval from the latest saved run
cargo run --example cartpole_dqn --features envs --release -- --eval runs/cartpole/v1
```

## What it demonstrates

- Building a `DqnAgent` with `VecEncoder` and `UsizeActionMapper`
- Attaching a `TrainingRun` for automatic checkpointing and JSONL logging
- Using `TrainingSession` as a loop-agnostic coordinator
- Loading a saved checkpoint into a `DqnPolicy` for inference-only evaluation
- Resuming training from a previous run via `TrainingRun::resume`
