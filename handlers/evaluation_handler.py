import logging
from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from .base_handler import Handler

class EvaluationHandler(Handler):
    """ 
    Handler responsible for evaluating the trained model.
    """

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("EvaluationHandler: Processing request")
        processed_request = self.process(request)
        if processed_request["status"] == "error":
            self.logger.error("EvaluationHandler: Stopping chain due to error")
            return processed_request
        else:
            return super().handle(processed_request)

    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if request["status"] != "model_trained":
            self.logger.warning("EvaluationHandler: Model not trained")
            return request

        config = request["config"]
        data = request["data"]
        model = request["trained_model"]

        try:
            self.logger.info("EvaluationHandler: Starting model evaluation")

            X_val = data["validation"][data["features"]]
            y_val = data["validation"][data["target"]]
            X_test = data["test"][data["features"]]
            y_test = data["test"][data["target"]]

            if hasattr(model, 'predict'):
                val_preds = model.predict(X_val)
                test_preds = model.predict(X_test)
            else:
                self.logger.error("EvaluationHandler: Model does not have a predict method")
                raise AttributeError("Model does not have a predict method")

            if config.problem_type == 'classification':
                val_metrics = self._calculate_classification_metrics(y_val, val_preds)
                test_metrics = self._calculate_classification_metrics(y_test, test_preds)
            elif config.problem_type == 'regression':
                val_metrics = self._calculate_regression_metrics(y_val, val_preds)
                test_metrics = self._calculate_regression_metrics(y_test, test_preds)
            else:
                val_metrics = {}
                test_metrics = {}
            
            request["evaluation"] = {
                "validation": val_metrics,
                "test": test_metrics
            }

            self.logger.info("EvaluationHandler: Model evaluated successfully")
    
        except Exception as e:
            self.logger.error(f"EvaluationHandler: Error evaluating model - {str(e)}")
            request["status"] = "error"
            request["error"] = str(e)
        
        return request
    
    def _calculate_classification_metrics(self, y_true, y_pred):
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_class = np.argmax(y_pred, axis=1)

        else:
            y_pred_class = (y_pred > 0.5).astype(int)
        
        return {
            "accuracy": float(accuracy_score(y_true, y_pred_class)),
            "precision": float(precision_score(y_true, y_pred_class, average="weighted")),
            "recall": float(recall_score(y_true, y_pred_class, average="weighted")),
            "f1": float(f1_score(y_true, y_pred_class, average="weighted"))
        }
    

    def _calculate_regression_metrics(self, y_true, y_pred):
        return {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred))
        }