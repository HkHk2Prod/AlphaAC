from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.labels import UNKNOWN, TrivializationLabel, known_trivial

# Generators are x = 1, y = 2 throughout this module.
_X, _Y = 1, 2
_NAMES = ("x", "y")
_LEAKAGE = (
    "Literature candidate, not training data. A failed search proves nothing about "
    "the Andrews-Curtis conjecture; keep separate from generated training sets."
)


def akbulut_kirby(n: int) -> BalancedPresentation:
    """Akbulut-Kirby presentation AK(n) = <x, y | x^n = y^(n+1), xyx = yxy>.

    These balanced presentations of the trivial group are the canonical potential
    Andrews-Curtis counterexamples. AK(2) is known to be AC-trivializable; larger
    members are progressively harder and were long open.
    """
    if n < 1:
        raise ValueError("akbulut_kirby requires n >= 1")
    relator_power = [_X] * n + [-_Y] * (n + 1)
    relator_braid = [_X, _Y, _X, -_Y, -_X, -_Y]
    return BalancedPresentation.from_letters(
        2,
        [relator_power, relator_braid],
        generator_names=_NAMES,
        presentation_id=f"akbulut-kirby-{n}",
        provenance={
            "family": "akbulut_kirby",
            "n": n,
            "status": "potential_counterexample",
            "reference": "Akbulut and Kirby, A potential smooth counterexample (1985)",
            "leakage_warning": _LEAKAGE,
        },
    )


def miller_schupp(n: int, w: Sequence[int]) -> BalancedPresentation:
    """Miller-Schupp presentation MS(n, w) = <x, y | x^-1 y^n x = y^(n+1), x = w>.

    `w` is a word in the signed generators (x = 1, y = 2) whose x-exponent sum is
    zero, which keeps the abelianization trivial. The series is a standard source
    of hard balanced presentations and potential counterexamples.
    """
    if n < 1:
        raise ValueError("miller_schupp requires n >= 1")
    word = [int(letter) for letter in w]
    if any(letter == 0 or abs(letter) > 2 for letter in word):
        raise ValueError("w must be a word in the two signed generators")
    if sum(1 for letter in word if letter == _X) - sum(1 for letter in word if letter == -_X) != 0:
        raise ValueError("w must have x-exponent sum zero")
    relator_shift = [-_X] + [_Y] * n + [_X] + [-_Y] * (n + 1)
    relator_word = [-_X, *word]
    return BalancedPresentation.from_letters(
        2,
        [relator_shift, relator_word],
        generator_names=_NAMES,
        presentation_id=f"miller-schupp-{n}-{_word_tag(word)}",
        provenance={
            "family": "miller_schupp",
            "n": n,
            "w": word,
            "status": "hard_benchmark",
            "reference": "Miller and Schupp, Some presentations of the trivial group (1999)",
            "leakage_warning": _LEAKAGE,
        },
    )


def candidate_entries() -> list[tuple[BalancedPresentation, TrivializationLabel]]:
    """Return curated candidates paired with their known trivialization labels.

    AK(2) is known to be AC-trivial; the larger Akbulut-Kirby members and the
    Miller-Schupp instances here are open, so they carry the unknown label. No
    minimal operation counts are asserted, since none are independently verified
    in this repository.
    """
    entries: list[tuple[BalancedPresentation, TrivializationLabel]] = [
        (akbulut_kirby(2), known_trivial())
    ]
    entries.extend((akbulut_kirby(n), UNKNOWN) for n in (3, 4, 5))
    entries.extend(
        (miller_schupp(n, w), UNKNOWN)
        for n, w in (
            (1, [_Y]),
            (1, [_Y, _Y]),
            (2, [_Y]),
            (2, [_X, _Y, -_X, -_Y]),
            (3, [_Y, _X, -_Y, -_X]),
        )
    )
    return entries


def standard_candidates() -> list[BalancedPresentation]:
    """Return the curated catalog of standard potential AC counterexamples."""
    return [presentation for presentation, _ in candidate_entries()]


def write_candidates(path: str | Path) -> int:
    """Write the curated candidate catalog as a separate, non-training dataset."""
    entries = candidate_entries()
    data = {
        "schema_version": "aczero-candidates-v1",
        "rank": 2,
        "leakage_warning": _LEAKAGE,
        "instances": [
            {**presentation.to_json(), **label.to_json()} for presentation, label in entries
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return len(entries)


def _word_tag(word: Sequence[int]) -> str:
    symbols = {_X: "X", -_X: "x", _Y: "Y", -_Y: "y"}
    return "".join(symbols[letter] for letter in word) or "e"
