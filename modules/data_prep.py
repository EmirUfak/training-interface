import re
import pandas as pd
from typing import Iterable, Optional


def remove_duplicates(df: pd.DataFrame, subset_cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    return df.drop_duplicates(subset=list(subset_cols) if subset_cols else None)


def drop_rows(df: pd.DataFrame, row_indices: Iterable[int]) -> pd.DataFrame:
    return df.drop(index=list(row_indices), errors="ignore")


def drop_cols(df: pd.DataFrame, col_names: Iterable[str]) -> pd.DataFrame:
    return df.drop(columns=list(col_names), errors="ignore")


def fill_missing(df: pd.DataFrame, strategy: str = "median", cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    cols = list(cols) if cols else df.columns
    df_copy = df.copy()
    for col in cols:
        if col not in df_copy.columns:
            continue
        if strategy == "median":
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        elif strategy == "mean":
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        elif strategy == "mode":
            if df_copy[col].isna().any():
                mode_val = df_copy[col].mode(dropna=True)
                if not mode_val.empty:
                    df_copy[col] = df_copy[col].fillna(mode_val.iloc[0])
    return df_copy


def clean_text(df: pd.DataFrame, col: str, lower: bool = True, strip: bool = True, remove_punct: bool = True) -> pd.DataFrame:
    if col not in df.columns:
        return df
    df_copy = df.copy()
    series = df_copy[col].astype(str)
    if lower:
        series = series.str.lower()
    if strip:
        series = series.str.strip()
    if remove_punct:
        series = series.str.replace(r"[^\w\s]", "", regex=True)
    df_copy[col] = series
    return df_copy


def tokenize_text(df: pd.DataFrame, col: str, lowercase: bool = True) -> pd.DataFrame:
    if col not in df.columns:
        return df
    df_copy = df.copy()
    series = df_copy[col].astype(str)
    if lowercase:
        series = series.str.lower()
    tokens = series.apply(lambda s: " ".join(re.findall(r"\b\w+\b", s)))
    df_copy[col] = tokens
    return df_copy
