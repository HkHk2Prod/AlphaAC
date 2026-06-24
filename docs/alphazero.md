# Single-Player AlphaZero

This project treats Andrews-Curtis search as deterministic single-player optimization. MCTS backups do not alternate players and do not negate values. Edge backups use immediate normalized reward plus future value.

The repository ships a runnable CPU policy/value training loop. Self-play uses
`search.puct.PUCTMCTS` — a real single-player PUCT search guided by the model's
priors and value — to collect visit-count policy targets and return-to-go value
targets into a replay buffer. The configured architecture is then optimized by
exact gradient descent before a fixture certificate is solved and independently
verified. PUCT is also exposed directly as the `puct` solve/benchmark agent.

The registered architectures (`linear_policy_value`, `residual_mlp`,
`deepsets`, `gru`, `transformer`) are genuine trainable NumPy models built on a
small reverse-mode autodiff engine; see [architectures](architectures.md). They
are deterministic CPU baselines. Large-scale JAX/Flax training on accelerators
remains the next expansion point, and the manifest machinery records the
metadata needed to audit numerical differences across backends.
