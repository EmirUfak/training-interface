import pandas as pd
from modules import data_prep


def test_data_prep_operations():
    df = pd.DataFrame({"a": [1, 1, None], "b": ["X", "X", "Y"], "c": ["Hello!", "Hello!", "World"]})

    dedup = data_prep.remove_duplicates(df, subset_cols=["a", "b"])
    assert len(dedup) == 2

    filled = data_prep.fill_missing(df, strategy="median", cols=["a"])
    assert filled["a"].isna().sum() == 0

    cleaned = data_prep.clean_text(df, "c", lower=True, remove_punct=True)
    assert cleaned["c"].str.contains("!").sum() == 0

    tokenized = data_prep.tokenize_text(df, "c")
    assert isinstance(tokenized.loc[0, "c"], str)


def test_data_prep_drop_cols_rows():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = data_prep.drop_cols(df, ["b"])
    assert "b" not in df2.columns
    df3 = data_prep.drop_rows(df, [0])
    assert 0 not in df3.index
