"""Tests for the ball's compact storage: the digest index and the column store."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ac_zero.datasets.annotate import annotation_path
from ac_zero.datasets.ball import BallConfig, grow_ball
from ac_zero.datasets.ball_columns import BallColumns, DigestIndex
from ac_zero.datasets.ball_store import OriginBall

MOVESET = "strict-ac"


def _digest(seed: int) -> bytes:
    """A well-distributed 32-byte digest, as SHA-256 gives the real thing."""
    return hashlib.sha256(str(seed).encode()).digest()


def test_index_round_trips_and_reports_absent_keys() -> None:
    index = DigestIndex()
    a, b = _digest(1), _digest(2)
    index.insert(a, 7)
    assert index.get(a) == 7
    assert index.get(b) is None  # an unseen digest is absent, not a stray slot
    assert len(index) == 1


def test_index_survives_many_rehashes() -> None:
    """Every key stays retrievable across the table doublings that a big ball forces."""
    index = DigestIndex(capacity=1 << 4)  # tiny, so a few thousand inserts rehash repeatedly
    count = 5000
    for i in range(count):
        index.insert(_digest(i), i)
    assert len(index) == count
    assert all(index.get(_digest(i)) == i for i in range(count))
    assert index.get(_digest(count + 1)) is None


def test_index_probes_past_a_home_slot_clash() -> None:
    """Two digests that share a home slot but differ in the key are kept distinct.

    The home slot is the top 8 bytes mod capacity; giving two digests the same top
    8 bytes but different next 8 forces a linear probe, which must not conflate them.
    """
    index = DigestIndex(capacity=1 << 8)
    shared = bytes(8)
    first = shared + (1).to_bytes(8, "big") + bytes(16)
    second = shared + (2).to_bytes(8, "big") + bytes(16)
    index.insert(first, 10)
    index.insert(second, 20)
    assert index.get(first) == 10
    assert index.get(second) == 20


def test_index_matches_a_dict_oracle() -> None:
    reference: dict[bytes, int] = {}
    index = DigestIndex(capacity=1 << 4)
    for i in range(0, 4000, 3):  # a sparse set of indices, inserted once each
        reference[_digest(i)] = i
        index.insert(_digest(i), i)
    assert all(index.get(k) == v for k, v in reference.items())


def test_columns_round_trip_relators_and_totals() -> None:
    columns = BallColumns(rank=2)
    columns.append([[1, 2], [-2, 1, 1]], _digest(1), distance=3, moves_mask=0b0)
    columns.append([[1], [2]], _digest(2), distance=0, moves_mask=0b101)

    assert len(columns) == 2
    assert columns.relators_at(0) == [[1, 2], [-2, 1, 1]]
    assert columns.relators_at(1) == [[1], [2]]
    assert columns.total_length_at(0) == 5
    assert columns.total_length_at(1) == 2
    assert columns.distance_at(0) == 3
    assert columns.digest_at(1) == _digest(2)
    assert columns.max_length() == 5


def test_columns_or_move_accumulates_bits() -> None:
    columns = BallColumns(rank=2)
    columns.append([[1], [2]], _digest(1), distance=1, moves_mask=0b001)
    columns.or_move(0, 0b100)
    columns.or_move(0, 0b001)  # idempotent: an already-set bit changes nothing
    assert columns.moves_at(0) == 0b101


def test_grown_ball_columns_agree_with_presentation_hashes(tmp_path: Path) -> None:
    """The digest stored in a column is exactly the hash of the presentation it rebuilds."""
    groups = tmp_path / "ball.groups.json"
    grow_ball(groups, BallConfig(rank=2, moveset=MOVESET, target=400, workers=1))

    ball = OriginBall.load_or_seed(groups, annotation_path(groups, MOVESET), 2, MOVESET)
    assert len(ball) > 1
    for index in range(len(ball)):
        assert ball.presentation(index).content_hash == ball._columns.digest_at(index).hex()
