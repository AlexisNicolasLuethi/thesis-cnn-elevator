# Import GPU utils first, before any other imports
from gpu_utils import setup_gpu_environment, initialize_gpu

# Import reproducibility utils
from reproducibility_utils import setup_reproducibility, get_reproducible_data_config, get_reproducible_augmentation_config

# Set GPU environment variables
setup_gpu_environment()

# Now do other imports
import os
import json
import tensorflow as tf
import numpy as np
import logging
import glob
from typing import Dict, Tuple, List
from data_loader import load_data, save_data_split
from model import create_model
from experiment_tracker import ExperimentTracker
from taguchi_optimizer import TaguchiOptimizer
from cache_manager import ImageCacheManager
from sklearn.preprocessing import LabelEncoder

def get_config_files(config_dir: str) -> List[str]:
    """Get all config files from the config directory."""
    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
        
    config_files = []
    for file in os.listdir(config_dir):
        if file.endswith('.json'):
            config_files.append(os.path.join(config_dir, file))
    
    if not config_files:
        raise ValueError(f"No config files found in {config_dir}")
        
    return sorted(config_files)

def setup_logging(log_file: str) -> None:
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict:
    """Load and validate configuration file."""
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Validate required sections
        required_sections = ['experiment', 'data', 'model', 'training']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in config file")
            
        # Add training mode if not present
        if 'training_mode' not in config['experiment']:
            config['experiment']['training_mode'] = 'taguchi'  # Default to taguchi
            
        # Validate training mode
        valid_modes = ['taguchi', 'config_only']
        if config['experiment']['training_mode'] not in valid_modes:
            raise ValueError(f"Invalid training mode. Must be one of: {valid_modes}")
        
        # Add reproducibility settings if not present
        if 'random_seed' not in config['experiment']:
            config['experiment']['random_seed'] = 42
            
        # Add deterministic settings to training
        if 'deterministic' not in config['training']:
            config['training']['deterministic'] = True
            
        return config
    except Exception as e:
        logging.error(f"Error loading config file {config_path}: {str(e)}")
        raise

def prepare_experiment_directory(config: Dict) -> str:
    """Prepare the experiment directory structure."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_experiment_dir = os.path.join(current_dir, config['experiment']['log_dir'])
    version_dir = os.path.join(base_experiment_dir, config['experiment']['version'])
    os.makedirs(version_dir, exist_ok=True)
    logging.info(f"Created version directory: {version_dir}")
    return version_dir

class DatasetManager:
    """Class to manage dataset loading and caching across experiments."""
    
    def __init__(self, cache_manager: ImageCacheManager, strategy: tf.distribute.Strategy):
        self.cache_manager = cache_manager
        self.strategy = strategy
        self.train_paths = None
        self.test_paths = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None
        self.images_per_category_count = None
        self._loaded = False

    def load_initial_data(self, config: Dict) -> None:
        """Load data once and store paths and labels."""
        if not self._loaded:
            logging.info("Loading initial data...")
            
            # Get reproducible data configuration 
            data_config = get_reproducible_data_config(config['experiment']['random_seed'])
            config['data'].update(data_config)
            
            X_train, X_test, y_train, y_test, label_encoder, images_per_category_count, train_paths, test_paths = load_data(
                config['data']['data_dir'],
                img_size=tuple(config['data']['img_size']),
                test_split=config['data']['test_split'],
                images_per_category=config['data']['images_per_category'],
                sampling_config=config['data']['sampling'],
                classification_level=config['data'].get('classification_level', 'subcategory')
            )
            
            self.train_paths = train_paths
            self.test_paths = test_paths
            self.y_train = y_train
            self.y_test = y_test
            self.label_encoder = label_encoder
            self.images_per_category_count = images_per_category_count
            self._loaded = True
            logging.info("Initial data loading completed")

    def create_datasets(self, config: Dict) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Create training and testing datasets using cached images with parallel loading."""
        with self.strategy.scope():
            # Get reproducible augmentation configuration if enabled
            if config['data_augmentation']['enabled']:
                aug_config = get_reproducible_augmentation_config(config['experiment']['random_seed'])
                config['data_augmentation'].update(aug_config)
            
            # Create datasets with parallel loading
            train_dataset = self.cache_manager.create_cached_dataset(
                self.train_paths,
                self.y_train,
                tuple(config['data']['img_size']),
                int(config['training']['batch_size']),
                is_training=True
            )
            
            test_dataset = self.cache_manager.create_cached_dataset(
                self.test_paths,
                self.y_test,
                tuple(config['data']['img_size']),
                int(config['training']['batch_size']),
                is_training=False
            )
            
            # Make datasets deterministic if required
            if config['training'].get('deterministic', True):
                options = tf.data.Options()
                options.experimental_deterministic = True
                train_dataset = train_dataset.with_options(options)
                test_dataset = test_dataset.with_options(options)
            
        return train_dataset, test_dataset

def validate_results(model, dataset, num_runs=3):
    """Validate that the model produces consistent results across multiple runs."""
    results = []
    for i in range(num_runs):
        logging.info(f"Validation run {i+1}/{num_runs}")
        predictions = model.predict(dataset)
        results.append(predictions)
    
    # Compare all results with the first run
    is_consistent = all(np.allclose(results[0], result, rtol=1e-5) for result in results[1:])
    return is_consistent

def run_single_training(config_path: str, strategy: tf.distribute.Strategy, 
                       dataset_manager: DatasetManager, tracker: ExperimentTracker) -> Dict:
    """Run a single training iteration with given configuration."""
    config = load_config(config_path)
    tracker.save_config(config)
    
    # Create datasets using cache
    train_dataset, test_dataset = dataset_manager.create_datasets(config)
    
    # Save data split information
    save_data_split(dataset_manager.train_paths, dataset_manager.test_paths, 
                   os.path.join(tracker.experiment_dir, 'data_split.json'))
    
    # Update config with actual images per category count
    tracker.update_config_with_image_counts(dataset_manager.images_per_category_count)
    
    # Create and compile model within strategy scope
    with strategy.scope():
        model = create_model(
            num_classes=dataset_manager.y_train.shape[1],
            input_shape=tuple(config['model']['input_shape']),
            base_model=config['model']['base'],
            dense_layers=config['model']['dense_layers'],
            dropout_rate=config['model']['dropout_rate'],
            l2_reg=config['model'].get('l2_regularization', 0.01),
            activation_function=config['model'].get('activation_function', 'relu'),
            kernel_initializer=config['model'].get('kernel_initializer', 'glorot_uniform'),
            use_batch_norm=config['model'].get('use_batch_norm', True),
            batch_norm_momentum=config['model'].get('batch_norm_momentum', 0.99)
        )
        
        # Configure optimizer
        optimizer = tf.keras.optimizers.get(config['training']['optimizer'])
        optimizer.learning_rate = config['training']['learning_rate']
        if hasattr(optimizer, 'momentum'):
            optimizer.momentum = config['training']['optimizer_momentum']
        
        model.compile(
            optimizer=optimizer,
            loss=config['training']['loss_function'],
            metrics=['accuracy']
        )
    
    # Get callbacks
    callbacks = tracker.get_callbacks(test_dataset, dataset_manager.label_encoder.classes_, model)
    
    # Train model
    with strategy.scope():
        history = model.fit(
            train_dataset,
            epochs=config['training']['epochs'],
            validation_data=test_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Validate reproducibility
        logging.info("Validating result reproducibility...")
        is_reproducible = validate_results(model, test_dataset)
        if is_reproducible:
            logging.info("Results are reproducible across runs")
        else:
            logging.warning("Results show variation across runs - check configuration")
        
        # Save the model
        model_save_path = os.path.join(tracker.experiment_dir, 'final_model.h5')
        model.save(model_save_path)
    
    return history.history

def run_experiment(config_path: str, strategy: tf.distribute.Strategy, dataset_manager: DatasetManager) -> Dict:
    """Run a single experiment with the given configuration."""
    config = load_config(config_path)
    logging.info(f"Running experiment with config: {config_path}")
    logging.info(f"Training mode: {config['experiment']['training_mode']}")

    # Set random seed for reproducibility
    random_seed = config['experiment'].get('random_seed', 42)
    setup_reproducibility(random_seed)

    try:
        # Prepare directories
        version_dir = prepare_experiment_directory(config)
        
        # Initialize ExperimentTracker
        tracker = ExperimentTracker(base_dir=version_dir)
        tracker.start_experiment(config['experiment']['name'])
        
        # Load initial data before any training starts
        logging.info("Loading initial data for experiment...")
        dataset_manager.load_initial_data(config)
        
        if config['experiment']['training_mode'] == 'taguchi':
            logging.info("Starting Taguchi optimization training")
            optimizer = TaguchiOptimizer(config_path)
            taguchi_config_paths = optimizer.generate_experiment_configs()
            
            best_accuracy = 0
            best_config = None
            
            for i, taguchi_config in enumerate(taguchi_config_paths):
                try:
                    logging.info(f"Running Taguchi experiment {i+1}/{len(taguchi_config_paths)}")
                    history = run_single_training(taguchi_config, strategy, dataset_manager, tracker)
                    val_accuracy = max(history['val_accuracy'])
                    optimizer.record_result(i, val_accuracy)
                    
                    # Track best configuration
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_config = taguchi_config
                        
                except Exception as e:
                    logging.error(f"Error in Taguchi experiment {i+1}: {str(e)}")
                    continue
            
            # Generate and save Taguchi analysis
            taguchi_analysis = optimizer.generate_taguchi_analysis()
            
            # Use the best configuration for final model
            if best_config:
                logging.info(f"Using best Taguchi configuration (accuracy: {best_accuracy})")
                final_history = run_single_training(best_config, strategy, dataset_manager, tracker)
            else:
                raise ValueError("No successful Taguchi configurations found")
                
        else:  # config_only mode
            logging.info("Using configuration file parameters only")
            final_history = run_single_training(config_path, strategy, dataset_manager, tracker)
        
        # End experiment and save results
        tracker.end_experiment()
        tracker.save_experiment_results(final_history)
        
        return final_history

    except Exception as e:
        logging.error(f"An error occurred during the experiment: {str(e)}")
        raise

def main():
    # Setup logging
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(current_dir, 'logs.txt')
    setup_logging(log_file)
    
    # Get config file
    config_dir = os.path.join(os.path.dirname(current_dir), 'configs')
    try:
        config_files = get_config_files(config_dir)
        logging.info(f"Found {len(config_files)} configuration files")
    except FileNotFoundError:
        config_files = [os.path.join(os.path.dirname(current_dir), 'config.json')]
        logging.info("Using single config.json file")
    
    # Initialize GPU and get strategy
    strategy = initialize_gpu()
    
    try:
        # Initialize cache manager
        cache_manager = ImageCacheManager(
            cache_dir=os.path.join(current_dir, 'global_image_cache'),
            memory_limit_gb=8.0
        )
        
        # Initialize dataset manager
        dataset_manager = DatasetManager(cache_manager, strategy)
        
        # Process each config file
        for config_file in config_files:
            try:
                logging.info(f"\n{'='*80}")
                logging.info(f"Processing configuration: {os.path.basename(config_file)}")
                logging.info('='*80)
                
                history = run_experiment(config_file, strategy, dataset_manager)
                logging.info(f"Completed processing {os.path.basename(config_file)}")
                
            except Exception as e:
                logging.error(f"Error processing config file {config_file}: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise
    finally:
        # Clean up
        logging.info("Cleaning up resources...")
        cache_manager.clear_cache(memory=True, disk=False)
        tf.keras.backend.clear_session()

if __name__ == "__main__":
    main()