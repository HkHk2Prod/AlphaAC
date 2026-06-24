# Environment

The environment is deterministic, single-agent, finite-horizon, and Gymnasium-like. The Markov state contains the presentation, initial length, best-so-far length, moves used, moves remaining, catalog version, and optional last action.

The canonical transition reward is the improvement in best-so-far total relator length. Therefore cumulative reward telescopes to the maximum reduction achieved anywhere in the episode.
