import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

def get_model(model_type, input_shape, output_shape=1, **kwargs):
    """
    Factory function to create and return a model based on the specified type.
    
    Parameters:
    -----------
    model_type : str
        Type of model to create (e.g., 'linear', 'random_forest', etc.)
    input_shape : int
        Number of input features
    output_shape : int
        Number of output classes (for classification) or dimensions (for regression)
    **kwargs : dict
        Additional parameters to pass to the model constructor
    
    Returns:
    --------
    model : object
        An instance of the specified model
    """

    # Classification models
    if model_type == 'logistic_regression':
        return LogisticRegression(**kwargs)
    elif model_type == 'random_forest_classifier':
        return RandomForestClassifier(**kwargs)
    elif model_type == 'gradient_boosting_classifier':
        return GradientBoostingClassifier(**kwargs)
    elif model_type == 'svm_classifier':
        return SVC(**kwargs)
    elif model_type == 'knn_classifier':
        return KNeighborsClassifier(**kwargs)
    

    # Regression models
    elif model_type == 'linear_regression':
        return LinearRegression(**kwargs)
    elif model_type == 'random_forest_regressor':
        return RandomForestRegressor(**kwargs)
    elif model_type == 'gradient_boosting_regressor':
        return GradientBoostingRegressor(**kwargs)
    elif model_type == 'svm_regressor':
        return SVR(**kwargs)
    elif model_type == 'knn_regressor':
        return KNeighborsRegressor(**kwargs)


    else:
        raise ValueError(f"Unknown model type: {model_type}")