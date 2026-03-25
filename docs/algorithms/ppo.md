# PPO

Proximal Policy Optimization with GAE advantage estimation and a shared actor-critic network.

## Key properties

- **On-policy** -- experience is collected into a rollout buffer, used for `n_epochs` gradient passes, then discarded. There is no replay buffer.
- **Discrete actions** -- the actor head outputs logits over a categorical distribution. Continuous action support is planned.
- **Actor-critic** -- a shared trunk feeds both a policy head (action logits) and a value head (state-value estimate).
- **Parallel environments** -- set `n_envs` to match the number of environments feeding the agent. All envs contribute to the same rollout; the update fires after `n_steps` ticks. This is the primary reason to use PPO with `bevy-gym`.

## Hyperparameters (`PpoConfig`)

| Field | Default | Notes |
|---|---|---|
| `n_steps` | `128` | Steps collected per env before each update |
| `n_envs` | `1` | Number of parallel environments |
| `n_epochs` | `4` | Gradient epochs per rollout |
| `batch_size` | `64` | Minibatch size within each epoch |
| `learning_rate` | `2.5e-4` | Adam learning rate |
| `clip_epsilon` | `0.2` | PPO probability ratio clipping range |
| `value_loss_coef` | `0.5` | Weight on the value function loss |
| `entropy_coef` | `0.01` | Weight on the entropy bonus |
| `gamma` | `0.99` | Discount factor |
| `gae_lambda` | `0.95` | GAE smoothing: `1.0` = Monte Carlo, `0.0` = TD(0) |
| `hidden_sizes` | `[64, 64]` | Shared trunk layer widths |
| `max_grad_norm` | `0.5` | Gradient norm clipping (0.0 = disabled) |

## Implementation notes

- **GAE** -- advantages are computed with Generalized Advantage Estimation, then normalized to zero mean and unit variance before each update.
- **act/observe pairing** -- `PpoAgent` caches `(log_prob, value)` from each `act()` call in a FIFO queue and pops it in the corresponding `observe()` call. This works correctly with both sequential loops and `bevy-gym`'s parallel ECS flow.
- **Bootstrap value** -- when the rollout buffer fills mid-episode, the critic estimates the value of the final next-observation to bootstrap GAE. If the last transition ended an episode, bootstrap value is 0.
- **Minibatches** -- the rollout is shuffled and split into minibatches each epoch. Incomplete final chunks are dropped.

## Tuning guidance

- **Larger or harder tasks** typically need more `n_steps`, more `n_epochs`, and a higher `entropy_coef` to prevent premature convergence.
- **Parallel envs** (`n_envs > 1`) improve sample diversity and wall-clock throughput. Increase `n_steps` proportionally to keep the total rollout size (`n_steps * n_envs`) sensible.
- **Smaller `clip_epsilon`** (e.g. `0.1`) is more conservative; use when training is unstable.

## Known behaviour

- CartPole-v1: mean reward ~435 at 500k steps with default config and 1 env.
