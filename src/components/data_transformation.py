import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

# Configuration class to define path for saving the preprocessing object
@dataclass
class DataTransformationConfig:
    preprocessor_ob_file = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Defining numerical and categorical columns for churn dataset")

            # 🔢 Numerical columns
            numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

            # 🏷️ Categorical columns
            categorical_columns = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod'
            ]

            # ⛓️ Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            logging.info("Numerical pipeline created.")

            # ⛓️ Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False))
            ])
            logging.info("Categorical pipeline created.")

            # 🔀 Column Transformer
            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            logging.info("Preprocessing object created.")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test datasets loaded successfully.")

            # Convert TotalCharges to numeric (handling spaces/blanks)
            for df in [train_df, test_df]:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

            preprocessing_obj = self.get_data_transformer_object()

            target_column = "Churn"
            train_df[target_column] = train_df[target_column].map({'Yes': 1, 'No': 0})
            test_df[target_column] = test_df[target_column].map({'Yes': 1, 'No': 0})

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file,
                obj=preprocessing_obj
            )
            logging.info("Preprocessing object saved to file.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file
            )

        except Exception as e:
            raise CustomException(e, sys)
