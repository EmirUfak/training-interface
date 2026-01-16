from pathlib import Path
from modules.code_exporter import export_tabular_pipeline


def test_code_exporter(tmp_path: Path):
    out_path = tmp_path / "exported_training.py"
    export_tabular_pipeline(
        csv_path="data.csv",
        feature_cols=["a", "b"],
        target_col="y",
        selected_models=["Random Forest"],
        optimize=False,
        strategy="all",
        is_regression=False,
        output_path=str(out_path),
    )

    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "exported_training.py" not in content
    assert "Random Forest" in content
