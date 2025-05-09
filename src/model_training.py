import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.logger import get_logger
from src.custom_exception import CustomException
import mlflow 
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, processed_data_path="artifacts/processed"):
        self.processed_data_path = processed_data_path
        self.model_dir = "artifacts/models"
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info("Model training initialized.")

    def load_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.processed_data_path, "X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.processed_data_path, "X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_data_path, "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_data_path, "y_test.pkl"))
            logger.info("Data loaded successfully for model.")
        except Exception as e:
            logger.error(f"Error while loading data for model: {e}")
            raise CustomException("Failed to load data for model.", e)

    def train_model(self):
        try:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            self.model.fit(self.X_train, self.y_train)
            joblib.dump(self.model, os.path.join(self.model_dir, "model.pkl"))
            logger.info("Model trained and saved successfully.")
        except Exception as e:
            logger.error(f"Error while training model: {e}")
            raise CustomException("Failed to train model.", e)

    def evaluate_model(self):
        try:
            y_pred = self.model.predict(self.X_test)
            y_proba = self.model.predict_proba(self.X_test)[:, 1] if len(np.unique(self.y_test)) == 2 else None

            # Check if target is string and contains "Yes" label
            is_binary_and_string = isinstance(self.y_test.iloc[0], str) and "Yes" in np.unique(self.y_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, pos_label="Yes") if is_binary_and_string else precision_score(self.y_test, y_pred, average="binary")
            recall = recall_score(self.y_test, y_pred, pos_label="Yes") if is_binary_and_string else recall_score(self.y_test, y_pred, average="binary")
            f1 = f1_score(self.y_test, y_pred, pos_label="Yes") if is_binary_and_string else f1_score(self.y_test, y_pred, average="binary")

            logger.info(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")

            mlflow.log_metric("accuracy",accuracy)
            mlflow.log_metric("Precision",precision)
            mlflow.log_metric("Recall_asacore",recall)
            mlflow.log_metric("f1-Score",f1)
            
            if y_proba is not None:
                roc_auc = roc_auc_score(self.y_test, y_proba)
                mlflow.log_metric("ROC-AUC",roc_auc)
                logger.info(f"ROC-AUC Score: {roc_auc:.4f}")

            logger.info("Model evaluation completed successfully.")
        except Exception as e:
            logger.error(f"Error while evaluating model: {e}")
            raise CustomException("Failed to evaluate model.", e)

    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()

if __name__ == "__main__":
    with mlflow.start_run():
        trainer = ModelTraining()
        trainer.run()
