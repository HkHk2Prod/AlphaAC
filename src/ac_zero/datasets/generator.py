from __future__ import annotations

import random
from dataclasses import dataclass

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.moves.catalog import ActionCatalog
from ac_zero.moves.primitive import PrimitiveMove, inverse_primitive_sequence


@dataclass(frozen=True, slots=True)
class GeneratedInstance:
    """Synthetic solvable presentation plus a private reverse certificate path.

    `difficulty` is the number of scramble moves that actually changed the
    presentation; it is an upper bound on the strict-AC solution length and a
    useful per-instance training/evaluation label.
    """

    presentation: BalancedPresentation
    reverse_moves: tuple[PrimitiveMove, ...]
    difficulty: int


def generate_solvable(rank: int, depth: int, seed: int) -> GeneratedInstance:
    """Scramble the standard presentation with seeded strict primitive moves.

    The returned presentation is guaranteed to be AC-trivial because it was
    produced from the exact standard presentation. The reverse path is returned
    for validation and fixture generation, but training code should not expose
    it as an observation.
    """

    rng = random.Random(seed)
    catalog = ActionCatalog(rank)
    pres = BalancedPresentation.standard(rank)
    applied: list[PrimitiveMove] = []
    for _ in range(depth):
        move = rng.choice(catalog.moves)
        nxt = move.apply(pres)
        if nxt.content_hash == pres.content_hash:
            continue
        pres = nxt
        applied.append(move)
    reverse_parts: list[PrimitiveMove] = []
    for move in reversed(applied):
        reverse_parts.extend(inverse_primitive_sequence(move))
    reverse = tuple(reverse_parts)
    return GeneratedInstance(
        BalancedPresentation.from_letters(
            rank,
            [r.letters for r in pres.relators],
            presentation_id=f"synthetic-r{rank}-d{depth}-s{seed}",
            provenance={
                "family": "seeded_strict_ac_scramble",
                "seed": seed,
                "depth": depth,
                "difficulty": len(applied),
            },
        ),
        reverse,
        len(applied),
    )
