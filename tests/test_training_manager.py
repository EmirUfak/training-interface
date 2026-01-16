import json
import os
import numpy as np
from pathlib import Path
from modules.training_manager import TrainingManager
from modules.model_trainer import get_model


def test_training_manager_outputs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 1, 0])

    model = get_model("Logistic Regression")
    models = {"Logistic Regression": model}

    manager = TrainingManager()
    manager.run_training_loop(
        models=models,
        X_train=X,
        X_test=X,
        y_train=y,
        y_test=y,
        optimize=False,
        optimize_strategy="all",
        task_type="classification",
        cv_folds=2,
    )

    # Locate output dir
    results_root = tmp_path / "results"
    result_dirs = [p for p in results_root.iterdir() if p.is_dir() and p.name.startswith("training_results_")]
    assert len(result_dirs) == 1
    save_dir = result_dirs[0]

    # Summary files
    summary_json = save_dir / "training_summary.json"
    summary_md = save_dir / "training_summary.md"
    summary_html = save_dir / "training_summary.html"

    assert summary_json.exists()
    assert summary_md.exists()
    assert summary_html.exists()

    with open(summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["task_type"] == "classification"
    assert data["results"]
