from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
if __name__ == "__main__":
    # Step 1: Ingest data
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Step 2: Transform data
    data_transformation_obj = DataTransformation()
    train_array, test_array, _ = data_transformation_obj.initiate_data_transformation(train_data, test_data)

    # Step 3: Train model
    model_trainer = ModelTrainer()
    result = model_trainer.initiate_model_trainer(train_array, test_array)  ## result  like={"best_model_name": RandomForest,"best_model_score": 0.99,"model_path":"\\\\\"}

    # Step 4: Print Results
    print("✅ Model Training Completed")
    print("Best Model:", result["best_model_name"])
    print("f1 Score:", result["best_model_f1_score"])
    print("Model saved at:", result["model_path"])