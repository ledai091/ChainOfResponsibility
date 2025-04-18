import logging
from typing import Dict, Any
from .base_handler import Handler
import importlib

class ModelBuilderHandler(Handler):
    """ 
    Handler responsible for building the model architecture
    """
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """ 
        Process the request and pass it to the next handler
        """
        self.logger.info("ModelBuilderHandler: Processing request")
        processed_request = self.process(request)

        if processed_request["status"] == "error":
            self.logger.error("ModelBuilderHandler: Stopping chain due to error")
            return processed_request

        return super().handle(processed_request)
    
    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if request["status"] != "dataset_prepared":
            self.logger.warning("ModelBuilderHandler: Dataset not prepared")
            return request

        config = request["config"]
        data = request["data"]

        try:
            module_path = config.model_module
            class_name = config.model_class

            if module_path == "models.model":
                from models.model import get_model
                model = get_model(
                    model_type=config.model_type,
                    input_shape=len(data["train"][data["features"]].columns),
                    output_shape=3,
                    **config.model_params
                )
            else:
                module =  importlib.import_module(module_path)
                model_class = getattr(module, class_name)
                model = model_class(**config.model_params)

            request["model"] = model
            request["status"] = "model_built"
            self.logger.info(f"ModelBuilderHandler: Model {class_name} built successfully")
        except Exception as e:
            self.logger.error(f"ModelBuilderHandler: Error building model {str(e)}")
            request["status"] = "error"
            request["error"] = str(e)
        return request