import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import issparse
from modules.data_loader import load_and_vectorize_text


def test_load_and_vectorize_text_sparse(tmp_path: Path):
    df = pd.DataFrame({
        "text": ["Merhaba dünya", "Makine öğrenmesi"],
        "label": ["a", "b"],
    })
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    X, y, vectorizer = load_and_vectorize_text(
        str(csv_path),
        text_col="text",
        label_col="label",
        max_features=100,
        ngram_range=(1, 1),
        stop_words=None,
        use_sparse=True,
        preprocess_text=True,
    )

    assert X.shape[0] == 2
    assert len(y) == 2
    assert vectorizer is not None
    assert issparse(X)
