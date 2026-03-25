# DQN

Deep Q-Network with experience replay and a target network.

## Key properties

- **Off-policy** -- experiences are stored in a replay buffer and reused across many updates.
- **Discrete actions only** -- the Q-network outputs one value per action; argmax selects the best.
- **Two networks** -- an online network updated every step and a target network synced periodically, which stabilises training.

## Hyperparameters (`DqnConfig`)

| Field | Default | Notes |
|---|---|---|
| `gamma` | `0.99` | Discount factor |
| `learning_rate` | `1e-3` | Adam learning rate |
| `batch_size` | `64` | Transitions sampled per update |
| `buffer_capacity` | `10_000` | Maximum replay buffer size |
| `min_replay_size` | `1_000` | Steps before training begins |
| `target_update_freq` | `200` | Steps between target network syncs |
| `hidden_sizes` | `[64, 64]` | Trunk layer widths |
| `epsilon_start` | `1.0` | Initial exploration rate |
| `epsilon_end` | `0.01` | Final exploration rate |
| `epsilon_decay_steps` | `10_000` | Steps to anneal epsilon |

## Implementation notes

- **Two separate RNGs.** The agent uses independent RNGs for epsilon-greedy exploration and replay buffer sampling. Sharing a single RNG causes subtle learning instability.
- **Epsilon decay** is linear from `epsilon_start` to `epsilon_end` over `epsilon_decay_steps`, then flat.
- **Target network** is a hard copy of the online network, updated every `target_update_freq` steps.
- **Checkpoints** use Burn's `CompactRecorder` (MessagePack format, `.mpk`). Only network weights are saved -- sufficient for inference. Resume training by calling `agent.load(path)` followed by `agent.set_total_steps(n)`.
- **Custom replay buffers** -- swap in any `ReplayBuffer` implementation via `DqnAgent::new_with_buffer()`.

## Known behaviour

- CartPole-v1 solves consistently across seeds within 100k steps with default config.
- Seed-sensitive: variance between runs is normal for DQN.
