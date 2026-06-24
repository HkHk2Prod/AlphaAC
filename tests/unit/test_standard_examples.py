import json
from pathlib import Path

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.environment.goals import exact_standard_goal

ROOT = Path(__file__).parents[2]


def test_standard_example_series_matches_exact_standard_goal() -> None:
    for rank in range(1, 4):
        path = ROOT / "data" / "examples" / f"standard_rank_{rank}.json"
        data = json.loads(path.read_text())
        pres = BalancedPresentation.from_json(data)

        assert pres.presentation_id == f"standard-rank-{rank}"
        assert pres.content_hash == data["content_hash"]
        assert pres.content_hash == BalancedPresentation.standard(rank).content_hash
        assert exact_standard_goal(pres)
        # the standard presentation is already trivial: zero operations, optimal
        assert data["ac_trivial"] is True
        assert data["minimal_known_operations"] == 0
        assert data["optimal"] is True
