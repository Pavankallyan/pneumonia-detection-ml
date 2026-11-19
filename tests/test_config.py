"""Validation tests for the committed experiment artifacts.

The training notebooks write ``models/config.json`` (training setup) and
``models/results.json`` (per-model metrics). These tests guard the values
the README and the examples/report script quote, so a bad edit to either
file is caught by CI instead of silently shipping wrong numbers.

The loaders under test live in ``src/metrics.py``; the artifact files are
the real ones committed in ``models/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.metrics import (
    accuracy_gap,
    best_model,
    load_config,
    load_results,
    model_names,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "models" / "config.json"
RESULTS_PATH = ROOT / "models" / "results.json"

EXPECTED_MODELS = {"decision_tree", "random_forest", "resnet50_cnn"}


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def results():
    return load_results(RESULTS_PATH)


def test_config_loads_and_is_object(config):
    assert isinstance(config, dict)
    assert config  # non-empty


def test_config_image_and_training_shape(config):
    assert config["img_size"] == 224
    assert config["batch_size"] == 32
    assert config["num_classes"] == 2


def test_config_class_names_and_index_agree(config):
    assert config["class_names"] == ["NORMAL", "PNEUMONIA"]
    assert config["class_to_idx"] == {"NORMAL": 0, "PNEUMONIA": 1}
    # every class name has an index, and indices are 0..n-1
    assert set(config["class_to_idx"]) == set(config["class_names"])
    assert sorted(config["class_to_idx"].values()) == list(
        range(config["num_classes"])
    )


def test_config_dataset_sizes_match_readme(config):
    # README: 5,216 training images, 624 test images
    assert config["train_size"] == 5216
    assert config["test_size"] == 624
    assert config["train_size"] > config["test_size"]


def test_config_data_dirs_share_one_dataset_root(config):
    for key in ("train_dir", "val_dir", "test_dir"):
        assert config[key].startswith("../data/raw/chest_xray/chest_xray/")
        assert config[key].endswith(("train", "val", "test"))
    assert config["train_dir"].endswith("train")
    assert config["val_dir"].endswith("val")
    assert config["test_dir"].endswith("test")


def test_config_device_is_known(config):
    assert config["device"] in {"cpu", "cuda", "mps"}


def test_load_config_rejects_non_object(tmp_path):
    bad = tmp_path / "config.json"
    bad.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        load_config(bad)


def test_results_loads_all_three_models(results):
    assert set(model_names(results)) == EXPECTED_MODELS


def test_results_metric_ranges(results):
    for name, metrics in results.items():
        acc = metrics["accuracy"]
        assert 0 < acc <= 100, f"{name} accuracy out of range: {acc}"
        auc = metrics.get("auc")
        assert auc is None or 0 < auc <= 1, f"{name} AUC out of range: {auc}"


def test_resnet50_is_best_by_accuracy_and_auc(results):
    assert best_model(results, metric="accuracy") == "resnet50_cnn"
    assert best_model(results, metric="auc") == "resnet50_cnn"


def test_accuracy_gap_matches_stored_numbers(results):
    # README quotes 91.67% (ResNet50) vs 82.37% (Decision Tree).
    gap = accuracy_gap(results, "resnet50_cnn", "decision_tree")
    assert gap == pytest.approx(91.67 - 82.37, abs=0.01)
    assert results["resnet50_cnn"]["accuracy"] == pytest.approx(91.67)
    assert results["decision_tree"]["accuracy"] == pytest.approx(82.37)


def test_decision_tree_has_no_auc(results):
    # Classical baseline reports accuracy only; the report renders "-" for AUC.
    assert results["decision_tree"]["auc"] is None
    assert best_model(results, metric="auc") != "decision_tree"


def test_random_forest_reports_cv_and_best_params(results):
    rf = results["random_forest"]
    assert rf["cv_accuracy"] == pytest.approx(97.62)
    params = rf["best_params"]
    assert params["n_estimators"] == 200
    assert params["max_depth"] == 10
    assert params["min_samples_split"] == 2


def test_load_results_rejects_empty_mapping(tmp_path):
    empty = tmp_path / "results.json"
    empty.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain a model mapping"):
        load_results(empty)


def test_config_and_results_agree_on_classes(config, results):
    # The results file has one entry per model, not per class; the config
    # is the single source of truth for the class list the app shows.
    assert len(config["class_names"]) == config["num_classes"] == 2
    assert "resnet50_cnn" in results
