'''
Demo of MLPreprocessor.py
'''

import pandas as pd
from ml_preprocessor import MLPreprocessor

def main():
    print("ML Data Preprocessor - Demonstration\n")

    # Load example data
    print("Loading sample dataset...")
    df = pd.DataFrame({
        "age": [25, 30, 29, 999, 28],
        "income": [50000, 60000, None, 80000, 55000],
        "gender": ["male", "female", "female", "other", None],
        "joined": pd.to_datetime(["2020-01-01", "2021-03-15", "2022-06-01", "2023-01-01", "2020-12-31"]),
        "is_member": [True, False, True, True, False],
        "target": [1, 0, 1, 1, 0]
    })

    print(df.head(), "\n")

    # Initialize Preprocessor
    print("Initializing preprocessor (auto=True)...")
    pre = MLPreprocessor(df, target="target")  # auto=True by default

    print("Column types detected:")
    for t, cols in pre.column_types.items():
        print(f"  {t}: {cols}")
    print()

    # Engineer datetime features
    print("Engineering datetime features...")
    pre.engineer_datetime()
    print("Columns after datetime processing:", pre.data.columns.tolist(), "\n")

    # Build and apply pipeline
    print("Building pipeline...")
    pre.build_pipeline()
    
    print("Applying pipeline...")
    pre.apply_pipeline()
    print(f"Transformed X shape: {pre.x.shape}")
    print(f"Target y shape: {pre.y.shape}\n")

    # Split data
    print("Splitting data into train/val/test...")
    x_train, x_val, x_test, y_train, y_val, y_test = pre.split_data()

    print(f"Train: {x_train.shape}, {y_train.shape}")
    print(f"Val:   {x_val.shape}, {y_val.shape}")
    print(f"Test:  {x_test.shape}, {y_test.shape}\n")

    # Save pipeline
    print("Saving pipeline to disk...")
    pre.save_pipeline("demo_pipeline.pkl")

    # Load and use pipeline on new data
    print("Reloading pipeline and transforming new data...")
    pre.load_pipeline("demo_pipeline.pkl")
    sample_new = df.drop(columns="target").iloc[:1]
    transformed = pre.transform_new_data(sample_new)
    print(f"Transformed new row shape: {transformed.shape}")

    print("\nDone.")

if __name__ == "__main__":
    main()
