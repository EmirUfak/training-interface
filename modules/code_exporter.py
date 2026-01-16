from pathlib import Path
from typing import List


def export_tabular_pipeline(
    csv_path: str,
    feature_cols: List[str],
    target_col: str,
    selected_models: List[str],
    optimize: bool,
    strategy: str,
    is_regression: bool,
    output_path: str = "exported_training.py",
):
    """Basit tabular eğitim kodu export'u üretir."""
    models_list = ", ".join([f"'{m}'" for m in selected_models])
    task_type = "regression" if is_regression else "classification"

    code = f"""import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom modules.data_loader import load_categorical_data\nfrom modules.model_trainer import get_model, get_regressor, train_model\n\nCSV_PATH = r'{csv_path}'\nFEATURE_COLS = {feature_cols}\nTARGET_COL = '{target_col}'\nMODELS = [{models_list}]\nOPTIMIZE = {optimize}\nSTRATEGY = '{strategy}'\nTASK_TYPE = '{task_type}'\n\nX, y, encoder, label_encoder = load_categorical_data(CSV_PATH, FEATURE_COLS, TARGET_COL, is_regression=(TASK_TYPE=='regression'))\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nmodels_to_train = {{}}\nfor name in MODELS:\n    params = {{}}\n    model = get_regressor(name, **params) if TASK_TYPE=='regression' else get_model(name, **params)\n    if model is not None:\n        models_to_train[name] = model\n\nfor name, model in models_to_train.items():\n    res = train_model(model, X_train, y_train, X_test, y_test, optimize=OPTIMIZE, model_name=name, task_type=TASK_TYPE)\n    print(name, res)\n"""

    Path(output_path).write_text(code, encoding="utf-8")
