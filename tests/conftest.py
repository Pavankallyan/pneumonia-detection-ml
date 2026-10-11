import json

import pytest

RESULTS = {
    "decision_tree": {"accuracy": 82.37, "auc": None},
    "random_forest": {
        "accuracy": 82.05,
        "auc": 0.9499,
        "best_params": {"max_depth": 10, "min_samples_split": 2, "n_estimators": 200},
        "cv_accuracy": 97.62,
    },
    "resnet50_cnn": {"accuracy": 91.67, "auc": 0.9627},
}


@pytest.fixture()
def results_path(tmp_path):
    p = tmp_path / "results.json"
    p.write_text(json.dumps(RESULTS), encoding="utf-8")
    return p


@pytest.fixture()
def config_path(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"image_size": 224, "batch_size": 32}), encoding="utf-8")
    return p


@pytest.fixture()
def empty_results_path(tmp_path):
    p = tmp_path / "results.json"
    p.write_text(json.dumps({}), encoding="utf-8")
    return p
