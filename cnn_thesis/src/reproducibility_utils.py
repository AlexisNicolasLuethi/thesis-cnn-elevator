import os
import random
import numpy as np
import tensorflow as tf
import logging

def setup_reproducibility(seed=42):
    """
    Set up all random seeds and configurations for reproducibility
    
    Args:
        seed (int): The random seed to use
    """
    # 1. Set basic random seeds
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # 2. Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. Configure TensorFlow
    tf.keras.utils.set_random_seed(seed)
    
    # 4. Enable deterministic operations
    tf.config.experimental.enable_op_determinism()
    
    # Log the setup
    logging.info(f"Reproducibility setup completed with seed: {seed}")
    logging.info(f"TensorFlow version: {tf.__version__}")
    logging.info(f"NumPy version: {np.__version__}")
    logging.info("Deterministic operations enabled")

def get_reproducible_data_config(seed=42):
    """
    Get configuration for reproducible data loading
    
    Args:
        seed (int): The random seed to use
        
    Returns:
        dict: Configuration settings for data loading
    """
    return {
        'seed': seed,
        'shuffle_seed': seed,
        'deterministic': True,
        'num_workers': 1
    }

def get_reproducible_augmentation_config(seed=42):
    """
    Get configuration for reproducible data augmentation
    
    Args:
        seed (int): The random seed to use
        
    Returns:
        dict: Configuration settings for data augmentation
    """
    return {
        'seed': seed,
        'fill_mode': 'nearest',
        'interpolation': 'nearest',
        'deterministic': True
    }