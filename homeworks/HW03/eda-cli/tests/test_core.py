from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_constant_columns_detection():
    """Тест на обнаружение константных колонок"""
    df = pd.DataFrame(
        {
            "const_col": [1, 1, 1, 1],
            "normal_col": [1, 2, 3, 4],
        }
    )
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    assert flags["has_constant_columns"] is True
    assert "const_col" in flags["constant_columns"]
    assert "normal_col" not in flags["constant_columns"]


def test_high_cardinality_categoricals_detection():
    """Тест на обнаружение категориальных признаков с высокой кардинальностью"""
    # Создаём колонку, где почти все значения уникальны
    df = pd.DataFrame(
        {
            "high_card": [f"value_{i}" for i in range(100)],
            "low_card": ["A", "B"] * 50,
            "numeric": range(100),
        }
    )
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    assert flags["has_high_cardinality_categoricals"] is True
    assert "high_card" in flags["high_cardinality_categoricals"]
    assert "low_card" not in flags["high_cardinality_categoricals"]


def test_many_zero_values_detection():
    """Тест на обнаружение колонок с большим количеством нулей"""
    df = pd.DataFrame(
        {
            "mostly_zeros": [0, 0, 0, 0, 0, 0, 1, 1],
            "normal_numbers": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    assert flags["has_many_zero_values"] is True
    assert "mostly_zeros" in flags["zero_heavy_columns"]
    assert "normal_numbers" not in flags["zero_heavy_columns"]


def test_suspicious_id_duplicates_detection():
    """Тест на обнаружение дубликатов в ID-подобных колонках"""
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 2, 3, 4],  # есть дубликат
            "product_id": [10, 20, 30, 40, 50],  # нет дубликатов
            "name": ["A", "B", "C", "D", "E"],
        }
    )
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    assert flags["has_suspicious_id_duplicates"] is True
    assert "user_id" in flags["suspicious_id_duplicates"]
    assert "product_id" not in flags["suspicious_id_duplicates"]
