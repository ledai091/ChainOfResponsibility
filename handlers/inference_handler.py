import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
import pickle
import os
from .base_handler import Handler

class InferenceHandler(Handler):
    """ 
    Handler responsible for model inference and saving mode
    """

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("InferenceHandler: Processing request")
        processed_request = self.process(request)
        return processed_request
    
    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if request["status"] != "model_evaluated":
            self.logger.warning("InferenceHandler: Model not evaluated")
            return request

        config = request["config"]
        model = request["trained_model"]

        try:
            self.logger.info("InferenceHandler: Starting inference process")
            if config.save_model:
                model_path = os.path.join(config.output_dir, config.model_filename)
                os.makedirs(config.output_dir, exist_ok=True)

                self.logger.info(f"InferenceHandler: Saving model to {model_path}")
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                
            if hasattr(config, 'inference_data_path') and config.inference_data_path:
                inference_data = pd.read_csv(config.inference_data_path)
                if hasattr(config, 'process_inference') and config.process_inference:
                    from handlers.dataset_handler import DatasetHandler
                    dataset_handler = DatasetHandler()
                    inference_data = dataset_handler._preprocess_data(inference_data, config)

                if hasattr(model, 'predict'):
                    X_inference = inference_data[config.features]
                    predictions = model.predict(X_inference)

                    if config.save_predictions:
                        pred_path = os.path.join(config.output_dir, config.prediction_file_name)

                        if hasattr(config, 'id_column') and config.id_column in inference_data.columns:
                            pred_df = pd.DataFrame({config.id_column: inference_data[config.id_column], 'prediction': predictions})
                        else:
                            pred_df = pd.DataFrame({'prediction': predictions})
                        
                        self.logger.info(f"InferenceHandler: Saving predictions to {pred_path}")

                        pred_df.to_csv(pred_path, index=False)
                    request["predictions"] = predictions
                else:
                    self.logger.error("InferenceHandler: Model does not have a predict method")
            
            request["status"] = "completed"
            self.logger.info("InferenceHandler: Inference process completed successfully")

        except Exception as e:
            self.logger.error(f"InferenceHandler: Error in inference process - {str(e)}")
            request["status"] = "error"
            request["error"] = str(e)
        
        return request