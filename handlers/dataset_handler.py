import logging
from typing import Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from .base_handler import Handler

class DatasetHandler(Handler):
    """
    Handler responsible for loading and preparing the dataset
    """
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """ 
        Process the request and pass it to the next handler
        """
        self.logger.info("DatasetHandler: Processing request")
        processed_request = self.process(request)

        if processed_request["status"] == "error":
            self.logger.error("DatasetHandler: Stopping chain due to error")
            return processed_request
        
        return super().handle(processed_request)

    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """ 
        Load and prepare the dataset
        """
        config = request["config"]
        try:
            self.logger.info(f"Loading data from {config.data_path}")
            data = pd.read_csv(config.data_path)

            self.logger.info("Preprocessing data")
            data = self._preprocess_data(data, config)

            self.logger.info("Splitting data into train/validation/test sets")
            train_data, temp_data = train_test_split(
                data,
                test_size=config.val_test_size,
                random_state=config.random_seed
            )

            val_data, test_data = train_test_split(
                temp_data,
                test_size=0.5,
                random_state=config.random_seed
            )

            request["data"] = {
                "train": train_data,
                "validation": val_data,
                "test": test_data,
                "features": config.features,
                "target": config.target
            }

            request["status"] = "dataset_prepared"
            self.logger.info("DatasetHandler: Dataset prepared successfully")
        except Exception as e:
            self.logger.error(f"DatasetHandler: Error preparing dataset {str(e)}")
            request["status"] = "error"
            request["error"] = str(e)

        return request
    
    def _preprocess_data(self, data: pd.DataFrame, config: Any) -> pd.DataFrame:
        if config.handle_missing:
            for col in data.columns:
                if data[col].dtype == 'object':
                    data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else "unknown")
                else:
                    data[col] = data[col].fillna(data[col].median() if not data[col].empty else 0)
        if config.apply_scaling:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
            data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
        if config.one_hot_encode:
            categorical_cols = data.select_dtypes(include=['object']).columns
            data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        
        return data