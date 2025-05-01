'''
ML Data Preprocessor
Author: Owen Rasor

Using pandas and sklearn to prepare data for processing by Machine Learning models.

Features:
    - cleans data
    - data type detection & seperation (numerical, categorical, datetime, boolean)
    - normalization/standardization
    - categorical encoding
    - train/validate/test split
    - datetime feature engineering
    - pipeline compatibility
    - reusability & serialization
    - outlier handling
    - logging
'''


import pandas as pd
import numpy as np
import joblib                                           # for easy loading/saving using serialization
import logging
from sklearn.model_selection import train_test_split    # seperates train/test data, 80/20                        
from sklearn.preprocessing import StandardScaler        # normalize numeric columns
from sklearn.preprocessing import OneHotEncoder         # encode categorical features, unordered
from sklearn.compose import ColumnTransformer           # apply transformers to specific columns
from sklearn.pipeline import Pipeline                   # bundle steps
from sklearn.impute import SimpleImputer                # replace missing values in columns

class MLPreprocessor:
    def __init__(self, data: pd.DataFrame, target: str, log_file: str = "preprocessor.log", auto=True):
        self.data = data.copy()
        self.target = target
        self.pipeline = None
        self.column_types = {}
        self.x = None
        self.y = None
        self.logger = self._setup_logging(log_file)

        if auto:
            self.clean_data()
            self.detect_types()
            self.handle_outliers()

    def _setup_logging(self, log_file):
        '''
        Sets up log - filename, info, and format of log
        '''
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

    def clean_data(self):
        '''
        Removes empty cells
        '''
        self.logger.info("Cleaning data: removing rows with missing target") # logs that data is being cleaned
        self.data.dropna(subset=[self.target], inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def detect_types(self):
        '''
        Classifies data types for easier pipeline manipulation. The target is removed if found in the data to avoid output corruption.
        '''
        self.logger.info("Detecting data types") # logs that data types are being detected and classified
        self.column_types = {
            "numerical": self.data.select_dtypes(include=["int64", "float64"]).columns.tolist(),
            "categorical": self.data.select_dtypes(include=["object", "category"]).columns.tolist(),
            "datetime": self.data.select_dtypes(include=["datetime64[ns]"]).columns.tolist(),
            "boolean": self.data.select_dtypes(include=["bool"]).columns.tolist()
        }
        if self.target in self.column_types["numerical"]:
            self.column_types["numerical"].remove(self.target)

    def handle_outliers(self, method="iqr"):
        '''
        Identifies and handles outliers for more accurate results.
        It replaces outliers with NaN values.
        Default: Interquartile Range method.
        '''
        self.logger.info(f'Handling outliers using: {method}') # logs that outliers will be handled
        for col in self.column_types["numerical"]:
            q1 = self.data[col].quantile(0.25)
            q3 = self.data[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.data[col] = np.where((self.data[col] < lower) | (self.data[col] > upper), np.nan, self.data[col])

    def engineer_datetime(self):
        '''
        Loops through all datetimes and extracts the useful information to be used.
        '''
        self.logger.info("Extracting datetime features") # logs that datetime values are being extracted
        for col in self.column_types["datetime"]:
            self.data[f'{col}_year'] = self.data[col].dt.year
            self.data[f'{col}_month'] = self.data[col].dt.month
            self.data[f'{col}_day'] = self.data[col].dt.day
            self.data[f'{col}_weekday'] = self.data[col].dt.weekday
        self.data.drop(columns=self.column_types["datetime"], inplace=True)

    def build_pipeline(self):
        '''
        Builds modular and reusable pipeline based on data type.
        Output: a unified object to applied to training, validation, and test sets.
        '''
        self.logger.info("Building preprocessor pipeline") # logs that pipeline is being built
        numeric = self.column_types["numerical"]
        categorical = self.column_types["categorical"]

        numeric_pipeline = Pipeline ([
            ("imputer", SimpleImputer(strategy="mean")), # missing values replaced with mean of the column
            ("scaler", StandardScaler()) # values standardized to mean=0 std=1
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")), # missing values replaced with the most frequent value in column
            ("encoder", OneHotEncoder(handle_unknown="ignore")) # transforms categories into encoded binary columns, unknown columns will be ignored
        ])

        self.pipeline = ColumnTransformer([ # applies correct pipeline to column group
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical)
        ])
        # self.pipeline can now be used

    def apply_pipeline(self):
        '''
        Applies the pipeline to the dataset to be used in machine learning.
        Output: Clean dataframe
        '''
        if self.pipeline is None:
            raise ValueError("Pipeline has not been built. Call build_pipeline() first.")
        self.logger.info("Applying pipeline transformation") # logs that the pipeline is being used
        self.x = self.pipeline.fit_transform(self.data.drop(columns=[self.target])) # feature matrix used for training, target is dropped
        self.y = self.data[self.target] # target variable stored

    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        '''
        Splits preprocessed data into three sets: training, validation, and test.
        Inputs: test_size(default=0.2) - percentage of the data that will be used for testing, val_size(default=0.1) - fraction of remaining data used for validation, random_state - sets what random split will happen
        Output: Data split by percentage for their purposes
        '''
        self.logger.info("Splitting data") # logs that the data is being split
        x_temp, x_test, y_temp, y_test = train_test_split(self.x, self.y, test_size=test_size, random_state=random_state) # features and target split
        val_fraction = val_size / (1 - test_size) # formula for fraction of validation data (default 0.1 = 0.125)
        x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=val_fraction, random_state=random_state) # splits features into validation and training
        return x_train, x_val, x_test, y_train, y_val, y_test
    
    def transform_new_data(self, new_data: pd.DataFrame):
        '''
        Transforms new incoming data using the fitted pipeline.
        '''
        self.logger.info("Transforming new data with fitted pipeline.")
        return self.pipeline.transform(new_data)
    
    def save_pipeline(self, path="preprocessor_pipeline.pkl"):
        '''
        Uses joblib to save the pipeline into a .pkl (pickle) file by serializing the data.
        '''
        self.logger.info(f'Saving pipeline to {path}') # logs save is happening
        joblib.dump(self.pipeline, path)

    def load_pipeline(self, path="preprocessor_pipeline.pkl"):
        '''
        Uses joblib to load an old pipeline via a .pkl file by deserializing the data into memory.
        '''
        self.logger.info(f'Loading pipeline from {path}') # logs that pipeline is being loaded
        self.pipeline = joblib.load(path)