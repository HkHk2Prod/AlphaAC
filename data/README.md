# Data

`examples/` contains small, committed presentation fixtures. The standard
series `standard_rank_1.json` through `standard_rank_3.json` is the exact
trivial presentation `<x1,...,xn | x1,...,xn>` for ranks 1, 2, and 3.

Every entry across `examples/`, `generated/`, and `candidates/` carries three
trivialization-label fields:

- `ac_trivial`: `true` if known AC-trivial, `false` if known non-trivial, `null`
  if open/unknown.
- `minimal_known_operations`: fewest strict primitive AC operations of any known
  trivialization, or `null` if none is known.
- `optimal`: whether `minimal_known_operations` is proven minimal (`null` when
  there is no number to qualify).

`generated/` holds the training dataset built by `aczero dataset grow`
(`aczero-dataset-v3`): a graph of guaranteed-solvable presentations expanded
outward from the trivial group. Each instance is `ac_trivial: true` with a known
(best-effort optimal) trivialization, a `difficulty` label (its construction
depth from the trivial group), and a `predecessors` list recording every
co-optimal construction move (multiple back-pointers, for supervised learning).
Groups are deduplicated by content hash, and every run resumes from the
accumulated frontier so the database only ever grows:

```bash
uv run --frozen aczero dataset grow \
  --output data/generated/train_rank2.json \
  --rank 2 --target 1000
```

`candidates/` holds curated literature presentations (`aczero-candidates-v1`):
the Akbulut-Kirby series `AK(n)` and members of the Miller-Schupp series. These
are balanced presentations of the trivial group used as standard potential
Andrews-Curtis counterexamples and hard benchmarks. `AK(2)` is labeled known
AC-trivial; the larger members and the Miller-Schupp instances are open
(`ac_trivial: null`). Regenerate with:

```bash
uv run --frozen aczero dataset candidates --output data/candidates/standard.json
```

Candidates must remain separate from training data to avoid leakage. A failed
search on a candidate is not evidence that it is a genuine counterexample, and
this project does not prove or disprove the Andrews-Curtis conjecture.

## Improving labels

`aczero dataset improve` (see `datasets/update.py`) runs the search agents on each
entry and merges any better trivialization into its labels. Breadth-first search
contributes shortest, sometimes provably optimal, solutions; greedy best-first
contributes heuristic upper bounds. The merge is monotonic:

- a shorter `minimal_known_operations` is never replaced by a longer one;
- `ac_trivial` is never demoted from a known result to unknown;
- `optimal: true` is only set when a search proves it, and never regressed;
- duplicate entries (same content hash) are merged automatically, keeping the
  best label and smallest difficulty;
- proven-optimal entries are skipped, so repeated passes are cheap and the file
  is rewritten atomically.
