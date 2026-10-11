"""Result-reporting helpers for the pneumonia detection experiments.

These functions operate on the experiment artifacts written by the training
notebooks (``models/results.json`` and ``models/config.json``): they load the
stored metrics, pick the best model for a given metric, and render the
comparison as a plain-text/markdown table for reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_results(path: str | Path) -> dict[str, Any]:
    """Load the experiment results JSON (model name -> metric dict)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"results file {path} does not contain a model mapping")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the training config JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} is not a JSON object")
    return data


def model_names(results: dict[str, Any]) -> list[str]:
    """Return the evaluated model names in the order stored in results."""
    return list(results.keys())


def best_model(results: dict[str, Any], metric: str = "accuracy") -> str:
    """Return the name of the model with the highest non-null *metric* value.

    Raises:
        KeyError: if no model reports the metric.
        ValueError: if every reported value is None.
    """
    scored = {
        name: metrics[metric]
        for name, metrics in results.items()
        if isinstance(metrics, dict) and metrics.get(metric) is not None
    }
    if not scored:
        if not any(isinstance(m, dict) and metric in m for m in results.values()):
            raise KeyError(f"metric {metric!r} not reported by any model")
        raise ValueError(f"metric {metric!r} is null for every model")
    return max(scored, key=scored.get)


def accuracy_gap(results: dict[str, Any], first: str, second: str) -> float:
    """Return ``accuracy[first] - accuracy[second]`` in percentage points."""
    return float(results[first]["accuracy"]) - float(results[second]["accuracy"])


def summary_rows(results: dict[str, Any]) -> list[dict[str, str]]:
    """Build one display row per model: name, accuracy, AUC, CV accuracy.

    Missing values render as ``"-"`` so the table never breaks.
    """
    rows = []
    for name, metrics in results.items():
        metrics = metrics if isinstance(metrics, dict) else {}
        rows.append(
            {
                "model": name,
                "accuracy": f"{metrics['accuracy']:.2f}%" if metrics.get("accuracy") is not None else "-",
                "auc": f"{metrics['auc']:.4f}" if metrics.get("auc") is not None else "-",
                "cv_accuracy": (
                    f"{metrics['cv_accuracy']:.2f}%"
                    if metrics.get("cv_accuracy") is not None
                    else "-"
                ),
            }
        )
    return rows


def render_markdown_table(results: dict[str, Any]) -> str:
    """Render a markdown comparison table of the experiment results."""
    header = "| Model | Accuracy | AUC | CV Accuracy |\n|---|---|---|---|\n"
    lines = [
        f"| {row['model']} | {row['accuracy']} | {row['auc']} | {row['cv_accuracy']} |"
        for row in summary_rows(results)
    ]
    return header + "\n".join(lines) + "\n"
