import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import (
    accuracy_gap,
    best_model,
    load_config,
    load_results,
    model_names,
    render_markdown_table,
    summary_rows,
)


def test_load_results(results_path, config_path):
    data = load_results(results_path)
    assert set(data) == {"decision_tree", "random_forest", "resnet50_cnn"}
    assert data["resnet50_cnn"]["accuracy"] == 91.67
    cfg = load_config(config_path)
    assert cfg["image_size"] == 224


def test_load_results_rejects_empty(empty_results_path):
    with pytest.raises(ValueError):
        load_results(empty_results_path)


def test_model_names(results_path):
    assert model_names(load_results(results_path)) == [
        "decision_tree",
        "random_forest",
        "resnet50_cnn",
    ]


def test_best_model_accuracy(results_path):
    assert best_model(load_results(results_path)) == "resnet50_cnn"


def test_best_model_auc(results_path):
    # decision_tree has null auc; should be ignored, not crash
    assert best_model(load_results(results_path), metric="auc") == "resnet50_cnn"


def test_best_model_missing_metric(results_path):
    with pytest.raises(KeyError):
        best_model(load_results(results_path), metric="f1_score")


def test_best_model_all_null_metric(results_path):
    data = load_results(results_path)
    for m in data.values():
        m["auc"] = None
    with pytest.raises(ValueError):
        best_model(data, metric="auc")


def test_best_model_tie_returns_first():
    data = {"a": {"accuracy": 90.0}, "b": {"accuracy": 90.0}}
    assert best_model(data) == "a"


def test_accuracy_gap(results_path):
    gap = accuracy_gap(load_results(results_path), "resnet50_cnn", "random_forest")
    assert abs(gap - (91.67 - 82.05)) < 1e-9


def test_summary_rows(results_path):
    rows = summary_rows(load_results(results_path))
    assert len(rows) == 3
    dt = rows[0]
    assert dt["model"] == "decision_tree"
    assert dt["accuracy"] == "82.37%"
    assert dt["auc"] == "-"  # null renders as dash
    rf = rows[1]
    assert rf["auc"] == "0.9499"
    assert rf["cv_accuracy"] == "97.62%"
    cnn = rows[2]
    assert cnn["cv_accuracy"] == "-"


def test_render_markdown_table(results_path):
    table = render_markdown_table(load_results(results_path))
    assert table.startswith("| Model | Accuracy | AUC | CV Accuracy |")
    assert "|---|---|---|---|" in table
    assert "| decision_tree | 82.37% | - | - |" in table
    assert "| resnet50_cnn | 91.67% | 0.9627 | - |" in table
