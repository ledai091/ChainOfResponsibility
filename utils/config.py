import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Config:
    # Non-default arguments must come first
    data_path: str
    output_dir: str
    features: List[str]
    target: str
    problem_type: str
    model_type: str
    
    # Optional arguments with default values
    inference_data_path: Optional[str] = None
    val_test_size: float = 0.3
    random_seed: int = 42
    handle_missing: bool = False
    apply_scaling: bool = False
    one_hot_encode: bool = False
    
    model_module: str = "models.model"
    model_class: str = "get_model"
    model_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
    )
    
    training_params: Dict[str, Any] = field(default_factory=lambda: {})
    use_validation: bool = True
    
    save_model: bool = True
    model_filename: str = "trained_model.pkl"
    save_predictions: bool = True
    predictions_filename: str = "predictions.csv"
    id_column: Optional[str] = None
    preprocess_inference: bool = True

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

        if not self.training_params:
            if self.problem_type == "classification":
                self.training_params = {}
            elif self.problem_type == "regression":
                self.training_params = {}