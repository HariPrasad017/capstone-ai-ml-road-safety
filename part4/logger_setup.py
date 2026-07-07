import logging
import os
from datetime import datetime

def setup_logger():
    os.makedirs('logs', exist_ok=True)
    
    logger = logging.getLogger('road_safety_advisor')
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        # Log to file
        file_handler = logging.FileHandler('logs/app.log')
        file_handler.setLevel(logging.INFO)
        
        # Log to console too
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger