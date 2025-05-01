import pytest # type: ignore
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from ml_preprocessor import MLPreprocessor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --- Fixtures ---

@pytest.fixture
def raw_data():
    df = pd.DataFrame({
        "age": [25, 30, 29, 999, 28],  # 999 = outlier
        "income": [50000, 60000, None, 80000, 55000],
        "gender": ["male", "female", "female", "other", None],
        "joined": pd.to_datetime(["2020-01-01", "2021-03-15", "2022-06-01", "2023-01-01", "2020-12-31"]),
        "is_member": [True, False, True, True, False],
        "target": [1, 0, 1, 1, 0]
    })
    return df

@pytest.fixture
def preprocessor_auto(raw_data):
    """Preprocessor with auto=True (default), runs clean_data + detect_types + handle_outliers automatically."""
    return MLPreprocessor(raw_data, target="target")

@pytest.fixture
def preprocessor_manual(raw_data):
    """Preprocessor with auto=False for manual method testing."""
    pre = MLPreprocessor(raw_data, target="target", auto=False)
    pre.clean_data()
    pre.detect_types()
    return pre

# --- Tests ---

def test_auto_runs_clean_and_detect(preprocessor_auto):
    # Should already have types detected
    assert "numerical" in preprocessor_auto.column_types
    assert "age" in preprocessor_auto.column_types["numerical"]

def test_manual_mode_requires_calls(preprocessor_manual):
    # Outliers not handled unless called
    assert not np.isnan(preprocessor_manual.data.loc[3, "age"])

def test_handle_outliers_works(preprocessor_manual):
    preprocessor_manual.handle_outliers()
    assert np.isnan(preprocessor_manual.data.loc[3, "age"])

def test_engineer_datetime_creates_columns(preprocessor_manual):
    preprocessor_manual.engineer_datetime()
    for col in ["joined_year", "joined_month", "joined_day", "joined_weekday"]:
        assert col in preprocessor_manual.data.columns

def test_build_pipeline_creates_pipeline(preprocessor_manual):
    preprocessor_manual.engineer_datetime()
    preprocessor_manual.build_pipeline()
    assert isinstance(preprocessor_manual.pipeline, ColumnTransformer)

def test_apply_pipeline_shapes(preprocessor_manual):
    preprocessor_manual.engineer_datetime()
    preprocessor_manual.build_pipeline()
    preprocessor_manual.apply_pipeline()
    assert preprocessor_manual.x.shape[0] == preprocessor_manual.data.shape[0]
    assert preprocessor_manual.y.shape[0] == preprocessor_manual.data.shape[0]

def test_split_data_returns_correct_sizes(preprocessor_manual):
    preprocessor_manual.engineer_datetime()
    preprocessor_manual.build_pipeline()
    preprocessor_manual.apply_pipeline()
    x_train, x_val, x_test, y_train, y_val, y_test = preprocessor_manual.split_data()
    total_rows = preprocessor_manual.x.shape[0]
    assert (x_train.shape[0] + x_val.shape[0] + x_test.shape[0]) == total_rows

def test_save_and_load_pipeline(tmp_path, preprocessor_manual):
    preprocessor_manual.engineer_datetime()
    preprocessor_manual.build_pipeline()
    path = tmp_path / "pipeline.pkl"
    preprocessor_manual.save_pipeline(path)
    assert os.path.exists(path)

    # Load into a fresh instance
    preprocessor_manual.pipeline = None
    preprocessor_manual.load_pipeline(path)
    assert isinstance(preprocessor_manual.pipeline, ColumnTransformer)