import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import xgboost as xgb  # Added XGBoost

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "best_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models: dict, params: dict):
        report = {}
        best_models = {}

        for name, model in models.items():
            logging.info(f"Training model: {name}")

            param_dist = params.get(name, {})

            try:
                if param_dist:
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_dist,
                        n_iter=20,
                        cv=3,
                        scoring='f1_weighted',
                        verbose=1,
                        n_jobs=-1,
                        random_state=42
                    )
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    logging.info(f"Best params for {name}: {search.best_params_}")
                else:
                    model.fit(X_train, y_train)
                    best_model = model

                y_pred = best_model.predict(X_test)
                f1 = f1_score(y_test, y_pred, average='weighted')
                report[name] = f1
                best_models[name] = best_model

                logging.info(f"{name} F1 Score: {f1:.4f}")

            except Exception as e:
                logging.warning(f"Model {name} failed with error: {e}")
                continue

        return report, best_models

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting features and target variable")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "DecisionTree": DecisionTreeClassifier(),
                "RandomForest": RandomForestClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "BaggingClassifier": BaggingClassifier(),
                "SVC": SVC(),
                "KNeighbors": KNeighborsClassifier(),
                "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            }

            params = {
                "LogisticRegression": {
                    "C": uniform(0.01, 10),
                    "penalty": ["l2"],
                    "solver": ["lbfgs"]
                },
                "DecisionTree": {
                    "max_depth": randint(3, 50),
                    "min_samples_split": randint(2, 20),
                    "min_samples_leaf": randint(1, 10)
                },
                "RandomForest": {
                    "n_estimators": randint(50, 300),
                    "max_depth": randint(5, 50),
                    "min_samples_split": randint(2, 20),
                    "min_samples_leaf": randint(1, 10),
                    "max_features": ["auto", "sqrt", "log2"]
                },
                "GradientBoosting": {
                    "n_estimators": randint(100, 300),
                    "learning_rate": uniform(0.01, 0.3),
                    "max_depth": randint(3, 10),
                    "subsample": uniform(0.6, 0.4)
                },
                "AdaBoost": {
                    "n_estimators": randint(50, 200),
                    "learning_rate": uniform(0.01, 1)
                },
                "BaggingClassifier": {
                    "n_estimators": randint(10, 100)
                },
                "SVC": {
                    "C": uniform(0.1, 100),
                    "kernel": ["linear", "rbf", "poly"],
                    "gamma": ["scale", "auto"]
                },
                "KNeighbors": {
                    "n_neighbors": randint(3, 15),
                    "weights": ["uniform", "distance"],
                    "p": [1, 2]
                },
                "XGBoost": {
                    "n_estimators": randint(50, 300),
                    "max_depth": randint(3, 15),
                    "learning_rate": uniform(0.01, 0.3),
                    "subsample": uniform(0.6, 1.0),
                    "colsample_bytree": uniform(0.6, 1.0),
                    "gamma": uniform(0, 5)
                }
            }

            model_report, best_models = self.evaluate_models(X_train, y_train, X_test, y_test, models, params)

            if not model_report:
                raise CustomException("No model trained successfully.", sys)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = best_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No good classification model found (F1 < 0.6)", sys)

            logging.info(f"Best Model: {best_model_name} | F1 Score: {best_model_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return {
                "best_model_name": best_model_name,
                "best_model_f1_score": best_model_score,
                "model_path": self.model_trainer_config.trained_model_file_path
            }

        except Exception as e:
            raise CustomException(e, sys)
