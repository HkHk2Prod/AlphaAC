#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

DEFAULT_METRICS = (
    "total_loss",
    "policy_loss",
    "value_loss",
    "replay_size",
    "episodes",
    "mean_return",
    "success_rate",
)

IGNORED_METRICS = {
    "batch_size",
    "iteration",
    "mcts_simulations",
    "optimizer_step",
    "optimizer_updates",
    "rank",
    "requested_model",
    "seed",
    "training_model",
}

COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
)


@dataclass(frozen=True)
class Series:
    name: str
    points: tuple[tuple[float, float], ...]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render training metrics as SVG plots.")
    parser.add_argument("run_directory", type=Path, help="Training run directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to RUN/artifacts/plots.",
    )
    args = parser.parse_args()

    run_directory = args.run_directory
    output_dir = args.output_dir or run_directory / "artifacts" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    series = collect_series(run_directory)
    if not series:
        raise SystemExit(f"No plottable metrics found under {run_directory}")

    dashboard_path = output_dir / "training_metrics.svg"
    dashboard_path.write_text(render_dashboard(series), encoding="utf-8")

    for item in series:
        path = output_dir / f"{slugify(item.name)}.svg"
        path.write_text(render_single_chart(item), encoding="utf-8")

    print(f"Wrote {len(series) + 1} plot files to {output_dir}")
    print(f"Dashboard: {dashboard_path}")


def collect_series(run_directory: Path) -> tuple[Series, ...]:
    events_path = run_directory / "logs" / "training_events.jsonl"
    if events_path.exists():
        rows = _read_jsonl(events_path)
        collected: dict[str, list[tuple[float, float]]] = {}
        for fallback_x, row in enumerate(rows, start=1):
            x = _number(row.get("step")) or float(fallback_x)
            metrics = row.get("metrics")
            if not isinstance(metrics, dict):
                continue
            for name, value in metrics.items():
                numeric = _number(value)
                if numeric is None or name in IGNORED_METRICS:
                    continue
                collected.setdefault(name, []).append((x, numeric))
        return _ordered_series(collected)

    metrics_path = run_directory / "metrics.jsonl"
    if not metrics_path.exists():
        return ()

    rows = _read_jsonl(metrics_path)
    collected = {}
    for fallback_x, row in enumerate(rows, start=1):
        x = _number(row.get("optimizer_step")) or float(fallback_x)
        for name, value in row.items():
            numeric = _number(value)
            if numeric is None or name in IGNORED_METRICS:
                continue
            collected.setdefault(name, []).append((x, numeric))
    return _ordered_series(collected)


def render_dashboard(series: tuple[Series, ...]) -> str:
    panel_width = 520
    panel_height = 270
    columns = 2
    rows = math.ceil(len(series) / columns)
    gap_x = 36
    gap_y = 44
    margin = 32
    title_height = 58
    width = columns * panel_width + (columns - 1) * gap_x + margin * 2
    height = title_height + rows * panel_height + (rows - 1) * gap_y + margin

    parts = [svg_header(width, height)]
    parts.append(f'<rect width="{width}" height="{height}" fill="#f8fafc"/>')
    parts.append(
        '<text x="32" y="38" font-family="Inter, Arial, sans-serif" '
        'font-size="24" font-weight="700" fill="#111827">Training metrics</text>'
    )
    for index, item in enumerate(series):
        col = index % columns
        row = index // columns
        x = margin + col * (panel_width + gap_x)
        y = title_height + row * (panel_height + gap_y)
        parts.append(
            render_panel(
                item,
                x=x,
                y=y,
                width=panel_width,
                height=panel_height,
                color=COLORS[index % len(COLORS)],
            )
        )
    parts.append("</svg>\n")
    return "\n".join(parts)


def render_single_chart(series: Series) -> str:
    width = 920
    height = 520
    parts = [svg_header(width, height)]
    parts.append(f'<rect width="{width}" height="{height}" fill="#f8fafc"/>')
    parts.append(
        render_panel(
            series,
            x=34,
            y=28,
            width=852,
            height=456,
            color=COLORS[0],
            large=True,
        )
    )
    parts.append("</svg>\n")
    return "\n".join(parts)


def render_panel(
    series: Series,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    color: str,
    large: bool = False,
) -> str:
    pad_left = 66
    pad_right = 22
    pad_top = 50 if large else 42
    pad_bottom = 46
    plot_x = x + pad_left
    plot_y = y + pad_top
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom

    xs = [point[0] for point in series.points]
    ys = [point[1] for point in series.points]
    min_x, max_x = _x_bounds(xs)
    min_y, max_y = _bounds(ys)

    def scale_x(value: float) -> float:
        if min_x == max_x:
            return plot_x + plot_width / 2
        return plot_x + ((value - min_x) / (max_x - min_x)) * plot_width

    def scale_y(value: float) -> float:
        if min_y == max_y:
            return plot_y + plot_height / 2
        return plot_y + plot_height - ((value - min_y) / (max_y - min_y)) * plot_height

    points = " ".join(f"{scale_x(px) - x:.2f},{scale_y(py) - y:.2f}" for px, py in series.points)
    title_size = 22 if large else 16
    tick_size = 12 if large else 10
    latest = series.points[-1][1]

    parts = [
        f'<g transform="translate({x},{y})">',
        f'<rect width="{width}" height="{height}" rx="8" fill="#ffffff" stroke="#d8dee9"/>',
        (
            f'<text x="20" y="{30 if large else 26}" font-family="Inter, Arial, sans-serif" '
            f'font-size="{title_size}" font-weight="700" fill="#111827">'
            f"{escape(series.name)}</text>"
        ),
        (
            f'<text x="{width - 20}" y="{30 if large else 26}" text-anchor="end" '
            f'font-family="Inter, Arial, sans-serif" font-size="{tick_size + 1}" '
            f'fill="#475569">latest {format_number(latest)}</text>'
        ),
    ]

    for tick in _ticks(min_y, max_y, 4):
        tick_y = scale_y(tick) - y
        parts.extend(
            (
                f'<line x1="{pad_left}" y1="{tick_y:.2f}" x2="{width - pad_right}" '
                'y2="{:.2f}" stroke="#edf2f7"/>'.format(tick_y),
                f'<text x="{pad_left - 10}" y="{tick_y + 4:.2f}" text-anchor="end" '
                f'font-family="Inter, Arial, sans-serif" font-size="{tick_size}" '
                f'fill="#64748b">{format_number(tick)}</text>',
            )
        )

    for tick in _ticks(min_x, max_x, 4):
        tick_x = scale_x(tick) - x
        parts.extend(
            (
                f'<line x1="{tick_x:.2f}" y1="{pad_top}" x2="{tick_x:.2f}" '
                f'y2="{height - pad_bottom}" stroke="#f1f5f9"/>',
                f'<text x="{tick_x:.2f}" y="{height - 18}" text-anchor="middle" '
                f'font-family="Inter, Arial, sans-serif" font-size="{tick_size}" '
                f'fill="#64748b">{format_number(tick)}</text>',
            )
        )

    parts.extend(
        (
            f'<line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" '
            f'y2="{height - pad_bottom}" stroke="#94a3b8"/>',
            f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" '
            f'y2="{height - pad_bottom}" stroke="#94a3b8"/>',
            f'<text x="{width - pad_right}" y="{height - 4}" text-anchor="end" '
            f'font-family="Inter, Arial, sans-serif" font-size="{tick_size}" '
            'fill="#64748b">event step</text>',
            f'<polyline fill="none" stroke="{color}" stroke-width="{3 if large else 2.5}" '
            f'stroke-linecap="round" stroke-linejoin="round" points="{points}"/>',
        )
    )

    radius = 4 if large else 3
    for px, py in series.points:
        parts.append(
            f'<circle cx="{scale_x(px) - x:.2f}" cy="{scale_y(py) - y:.2f}" '
            f'r="{radius}" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
        )

    parts.append("</g>")
    return "\n".join(parts)


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
    )


def _ordered_series(collected: dict[str, list[tuple[float, float]]]) -> tuple[Series, ...]:
    selected = {
        name: tuple(points)
        for name, points in collected.items()
        if len(points) >= 2 and any(value != points[0][1] for _, value in points[1:])
    }
    ordered_names = [name for name in DEFAULT_METRICS if name in selected]
    ordered_names.extend(sorted(name for name in selected if name not in DEFAULT_METRICS))
    return tuple(Series(name, selected[name]) for name in ordered_names)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float) and math.isfinite(value):
        return float(value)
    return None


def _bounds(values: Iterable[float]) -> tuple[float, float]:
    items = tuple(values)
    low = min(items)
    high = max(items)
    if low == high:
        padding = abs(low) * 0.05 or 1.0
    else:
        padding = (high - low) * 0.08
    return low - padding, high + padding


def _x_bounds(values: Iterable[float]) -> tuple[float, float]:
    items = tuple(values)
    low = min(items)
    high = max(items)
    if low == high:
        padding = abs(low) * 0.05 or 1.0
        return low - padding, high + padding
    return low, high


def _ticks(low: float, high: float, count: int) -> tuple[float, ...]:
    if count <= 1 or low == high:
        return (low,)
    return tuple(low + (high - low) * index / (count - 1) for index in range(count))


def format_number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.3g}"
    return f"{value:.3g}"


def slugify(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name).strip("_").lower()


if __name__ == "__main__":
    main()
