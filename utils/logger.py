import logging
import os
import sys
from datetime import datetime

def setup_logger(log_level=logging.INFO, log_file=None):
    """
    Set up logging configuration.
    
    Parameters:
    -----------
    log_level : int
        Logging level (e.g., logging.INFO, logging.DEBUG)
    log_file : str, optional
        Path to log file. If None, logs will be printed to console only.
    """
    if log_file:
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exits_ok=True)
    else:
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file  = f"logs/ml_pipeline_{timestamp}.log"
    
    logger = logging.getLogger()
    logger.setLevel(log_level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logger configured. Log file: {log_file}")

    return logger