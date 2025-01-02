import os
import logging
import tensorflow as tf
import subprocess

def setup_gpu_environment():
    """
    Set up GPU environment variables before importing TensorFlow
    Should be called at the very beginning of the program
    """
    # Set environment variables for GPU operations
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
    
    # Set thread and GPU deterministic settings
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # Set thread settings before TF initialization
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

def initialize_gpu():
    """
    Initialize GPU settings and return the distribution strategy
    """
    try:
        # Print diagnostic information
        logging.info("TensorFlow version: %s", tf.__version__)
        logging.info("Is built with CUDA: %s", tf.test.is_built_with_cuda())
        
        # Get available GPUs
        physical_devices = tf.config.list_physical_devices('GPU')
        logging.info("Physical devices: %s", physical_devices)
        
        if physical_devices:
            try:
                # Enable memory growth
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                logging.info("Memory growth enabled for GPU devices")
                
                # Enable mixed precision
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                logging.info("Mixed precision training enabled")
                
                # Create GPU strategy
                strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
                logging.info("Created OneDeviceStrategy for GPU")
                
            except RuntimeError as e:
                logging.error(f"GPU configuration error: {e}")
                strategy = tf.distribute.get_strategy()
                logging.warning("Falling back to default strategy")
                
        else:
            strategy = tf.distribute.get_strategy()
            logging.warning("No GPU devices found. Using default strategy")
            
        # Print NVIDIA-SMI information
        try:
            nvidia_smi = subprocess.check_output(["nvidia-smi"]).decode()
            logging.info("\nNVIDIA-SMI output:\n%s", nvidia_smi)
        except:
            logging.warning("Could not get NVIDIA-SMI information")
            
        return strategy
        
    except Exception as e:
        logging.error(f"Error in GPU initialization: {e}")
        return tf.distribute.get_strategy()