# example_usage.py

"""
Ví dụ sử dụng mẫu Chain of Responsibility cho quy trình machine learning 
với bộ dữ liệu Iris
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Import các module từ project
from utils.config import Config
from utils.logger import setup_logger
from handlers.dataset_handler import DatasetHandler
from handlers.model_builder_handler import ModelBuilderHandler
from handlers.training_handler import TrainingHandler
from handlers.evaluation_handler import EvaluationHandler
from handlers.inference_handler import InferenceHandler

def prepare_iris_dataset():
    """Chuẩn bị bộ dữ liệu Iris"""
    # Tạo thư mục data nếu chưa tồn tại
    os.makedirs('data/raw', exist_ok=True)
    
    # Tải bộ dữ liệu Iris
    iris = load_iris()
    
    # Tạo DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Lưu vào file CSV
    file_path = 'data/raw/iris_dataset.csv'
    df.to_csv(file_path, index=False)
    
    print(f"Bộ dữ liệu Iris đã được lưu vào {file_path}")
    
    return iris.feature_names, file_path

def run_ml_pipeline():
    """Chạy quy trình ML sử dụng Chain of Responsibility"""
    # Thiết lập logger
    logger = setup_logger()
    logger.info("Bắt đầu quy trình ML với bộ dữ liệu Iris")
    
    # Chuẩn bị dữ liệu
    feature_names, data_path = prepare_iris_dataset()
    
    # Tạo cấu hình
    config = Config(
        data_path=data_path,
        output_dir='output',
        features=feature_names,
        target='target',
        problem_type='classification',
        model_type='random_forest_classifier',
        model_params={
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42
        }
    )
    
    # Khởi tạo request
    request = {
        "config": config,
        "data": None,
        "model": None,
        "trained_model": None,
        "evaluation": None,
        "predictions": None,
        "status": "initialized"
    }
    
    logger.info("Thiết lập chuỗi xử lý (Chain of Responsibility)")
    
    # Tạo các handler
    dataset_handler = DatasetHandler()
    model_builder_handler = ModelBuilderHandler()
    training_handler = TrainingHandler()
    evaluation_handler = EvaluationHandler()
    inference_handler = InferenceHandler()
    
    # Liên kết các handler thành chuỗi
    dataset_handler.set_next(
        model_builder_handler).set_next(
        training_handler).set_next(
        evaluation_handler).set_next(
        inference_handler)
    
    # Bắt đầu chuỗi xử lý
    logger.info("Bắt đầu thực hiện chuỗi xử lý")
    
    try:
        result = dataset_handler.handle(request)
        
        # In kết quả đánh giá nếu có
        if result["status"] == "completed" and "evaluation" in result:
            logger.info("=== Kết quả đánh giá ===")
            logger.info("Tập validation:")
            for metric, value in result["evaluation"]["validation"].items():
                logger.info(f"  {metric}: {value:.4f}")
                
            logger.info("Tập test:")
            for metric, value in result["evaluation"]["test"].items():
                logger.info(f"  {metric}: {value:.4f}")
        
        if result["status"] == "error":
            logger.error(f"Quy trình thất bại với lỗi: {result.get('error', 'Lỗi không xác định')}")
            return
        
        logger.info("Quy trình ML hoàn thành thành công")
        
    except Exception as e:
        logger.error(f"Lỗi trong quy trình ML: {str(e)}")

if __name__ == "__main__":
    # Chạy quy trình cơ bản
    run_ml_pipeline()