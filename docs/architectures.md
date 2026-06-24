# Policy/Value Architectures

Every architecture implements the `PolicyValueModel` protocol: it maps an
encoded search state to logits over the deterministic action catalog plus a
scalar value in `(-1, 1)`. Callers apply the legal-action mask before sampling.

## Trainable core

The trainable models share a small reverse-mode autodiff engine
(`models/autograd.py`) over 2-D float arrays. `TrainablePolicyValueModel`
(`models/trainable.py`) attaches linear policy and tanh-bounded value heads to an
architecture-specific *trunk*, and trains every parameter — trunk and heads — by
exact gradient descent. The training gradients are checked against finite
differences in the test suite, so the recurrent and attention backward passes are
correct by construction.

Parameters are built lazily on first use, so the action-head width and any
encoding-dependent dimensions (such as the embedding vocabulary) are taken from
real inputs. `to_json` / `load_state` round-trip the parameters exactly for JSON
checkpoints.

## Architectures

- `linear_policy_value`: linear heads over a fixed whole-presentation feature
  vector (bias, normalized horizon, length ratios, token statistics). The
  deterministic CPU baseline.
- `residual_mlp`: the global feature vector through a projection plus one
  residual ReLU block, giving the heads a nonlinear representation.
- `deepsets`: a shared element network embeds each relator independently, the
  embeddings are sum-pooled, and a set network combines them with global
  features. Pooling makes the model invariant to relator order — a property the
  test suite asserts directly.
- `gru`: token embeddings are consumed by a standard GRU cell over the flattened
  relator token sequence; the final hidden state, concatenated with global
  features, feeds the heads. Training is backpropagation through time.
- `transformer`: token plus learned positional embeddings feed one scaled
  dot-product self-attention block with a residual feed-forward network. The
  attended tokens are mean-pooled and concatenated with global features.

## Adding an architecture

Subclass `TrainablePolicyValueModel`, implement `_build_trunk` (allocate trunk
parameters via `self._param` and return the feature dimension) and
`_forward_trunk` (return a `(1, feature_dim)` autodiff `Node`), then register the
class name in `models/registry.py`. The shared base supplies the heads, the
training step, serialization, and the lazy build.
