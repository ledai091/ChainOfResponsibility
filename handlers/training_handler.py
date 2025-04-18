import logging
from typing import Dict, Any
import time
import numpy as np
from .base_handler  import Handler

class TrainingHandler(Handler):
    """ 
    Handler responsible for training the model
    """

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """ 
        Process the request and pass it to the next handler
        """
        self.logger.info("TrainingHandler: Processing request")
        processed_request = self.process(request)

        if processed_request["status"] == "error":
            self.logger.error("TrainingHandler: Stopping chain due to error")
            return processed_request
        
        return super().handle(processed_request)
    
    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """ 
        Train the model using prepared dataset
        """
        if request["status"] != "model_built":
            self.logger.warning("TrainingHandler: Model not built")
            return request
        
        config = request["config"]
        data = request["data"]
        model = request["model"]

        try:
            self.logger.info("TrainingHandler: Starting model training")
            start_time = time.time()

            X_train = data["train"][data["features"]]
            y_train = data["train"][data["target"]]

            X_val = data["validation"][data["features"]]
            y_val = data["validation"][data["target"]]

            if hasattr(model, 'fit'):
                if config.use_validation:
                    if hasattr(model, 'fit') and 'validation_data' in model.fit.__code__.co_varnames:
                        model.fit(X_train, y_train, validation_data=(X_val, y_val), **config.training_params)
                    else:
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train, **config.training_params)
            elif hasattr(model, 'train'):
                model.train(X_train, y_train, X_val, y_val, **config.training_params)
            
            training_time = time.time() - start_time

            request["trained_model"] = model
            request["training_metrics"] =  {"training_time": training_time}
            request["status"] = "model_trained"

            self.logger.info(f"TrainingHandler: Model trained successfully in {training_time: .2f} seconds")

        except Exception as e:
            self.logger.error(f"TrainingHandler: Error training model - {str(e)}")
            request["status"] = "error"
            request["error"] = str(e)
        
        return request