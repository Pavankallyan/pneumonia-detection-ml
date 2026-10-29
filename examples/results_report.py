"""Print a model-comparison report from the stored experiment results.

Reads the real ``models/results.json`` written by the training notebooks
and renders it with the helpers in ``src/metrics.py``: the best model by
accuracy and by AUC, the accuracy gap between the best and worst models,
and the full markdown comparison table.

Usage (from the repo root):

    python examples/results_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import (  # noqa: E402
    accuracy_gap,
    best_model,
    load_results,
    model_names,
    render_markdown_table,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    results = load_results(root / "models" / "results.json")

    names = model_names(results)
    best_acc = best_model(results, metric="accuracy")
    best_auc = best_model(results, metric="auc")
    worst_acc = min(names, key=lambda n: results[n]["accuracy"])
    gap = accuracy_gap(results, best_acc, worst_acc)

    print(f"Models compared: {', '.join(names)}")
    print(f"Best by accuracy: {best_acc} ({results[best_acc]['accuracy']:.2f}%)")
    print(f"Best by AUC:       {best_auc} ({results[best_auc]['auc']:.4f})")
    print(f"Accuracy gap (best - worst): {gap:.2f} percentage points")
    print()
    print(render_markdown_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
