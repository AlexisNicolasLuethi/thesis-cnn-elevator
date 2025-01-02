import json
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import logging
import psutil
import GPUtil
import time
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import io
import os

class ExperimentTracker:
    def __init__(self, base_dir='experiments'):
        self.base_dir = base_dir
        self.current_experiment = None
        self.start_time = None
        self.end_time = None
        self.config = None

    def start_experiment(self, name):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.current_experiment = f"{name}_{timestamp}"
        self.experiment_dir = os.path.join(self.base_dir, self.current_experiment)
        try:
            os.makedirs(self.experiment_dir, exist_ok=True)
            logging.info(f"Created experiment directory: {self.experiment_dir}")
        except Exception as e:
            logging.error(f"Failed to create experiment directory: {str(e)}")
            raise
        self.start_time = time.time()

    def end_experiment(self):
        self.end_time = time.time()

    def save_config(self, config):
        self.config = config
        config_path = os.path.join(self.experiment_dir, 'config.json')
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logging.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logging.error(f"Failed to save configuration: {str(e)}")
            raise

    def save_data_split(self, train_paths, test_paths):
        split_info = {
            "train_images": train_paths,
            "test_images": test_paths
        }
        split_info_path = os.path.join(self.experiment_dir, 'data_split.json')
        try:
            with open(split_info_path, 'w') as f:
                json.dump(split_info, f, indent=2)
            logging.info(f"Saved data split information to {split_info_path}")
        except Exception as e:
            logging.error(f"Failed to save data split information: {str(e)}")
            raise

    def update_config_with_image_counts(self, images_per_category_count):
        if self.config is None:
            raise ValueError("Config has not been initialized. Call save_config first.")
        
        self.config['data']['actual_images_per_category'] = images_per_category_count
        self.save_config(self.config)

    def get_callbacks(self, test_dataset, class_names, model):
        log_dir = os.path.join(self.experiment_dir, 'logs')
        try:
            os.makedirs(log_dir, exist_ok=True)
            tensorboard_callback = TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch',
                profile_batch=2
            )
            
            # Write the graph
            writer = tf.summary.create_file_writer(log_dir)
            with writer.as_default():
                tf.summary.graph(tf.function(lambda x: model(x)).get_concrete_function(
                    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)).graph)
            
            checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'model_{epoch:02d}-{val_accuracy:.2f}.h5')
            checkpoint_callback = ModelCheckpoint(
                checkpoint_path, 
                save_best_only=True, 
                monitor='val_accuracy', 
                mode='max'
            )
            
            # Initialize callbacks list with basic callbacks
            callbacks = [tensorboard_callback, checkpoint_callback]
            
            # Add learning rate decay callback if enabled
            lr_decay_config = self.config['training'].get('learning_rate_decay', {})
            if lr_decay_config.get('enabled', False):
                lr_decay_callback = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=lr_decay_config.get('monitor', 'val_loss'),
                    factor=lr_decay_config.get('factor', 0.1),
                    patience=lr_decay_config.get('patience', 10),
                    min_lr=lr_decay_config.get('min_lr', 1e-6),
                    verbose=1
                )
                callbacks.append(lr_decay_callback)
                logging.info("Learning rate decay enabled with settings: "
                            f"factor={lr_decay_config['factor']}, "
                            f"patience={lr_decay_config['patience']}, "
                            f"min_lr={lr_decay_config['min_lr']}")
            
            # Custom callback for logging extra data
            class ExtraLoggingCallback(tf.keras.callbacks.Callback):
                def __init__(self, test_dataset, class_names):
                    super().__init__()
                    self.test_dataset = test_dataset
                    self.class_names = class_names

                def on_epoch_end(self, epoch, logs=None):
                    # Log learning rate
                    lr = self.model.optimizer.lr
                    tf.summary.scalar('learning_rate', data=lr, step=epoch)
                    
                    # Log sample images and predictions
                    if epoch % 5 == 0:  # Log every 5 epochs
                        for images, labels in self.test_dataset.take(1):  # Take one batch
                            val_images = images[:5]  # First 5 images in the batch
                            val_predictions = self.model.predict(val_images)
                            for i, (img, pred) in enumerate(zip(val_images, val_predictions)):
                                tf.summary.image(f"Sample_{i}", img[None, ...], step=epoch)
                                tf.summary.text(f"Prediction_{i}", str(pred), step=epoch)

                    # Log confusion matrix
                    if epoch % 10 == 0:  # Log every 10 epochs
                        y_true = []
                        y_pred = []
                        for images, labels in self.test_dataset:
                            predictions = self.model.predict(images)
                            y_true.extend(np.argmax(labels, axis=1))
                            y_pred.extend(np.argmax(predictions, axis=1))
                        
                        cm = confusion_matrix(y_true, y_pred)
                        
                        figure = plt.figure(figsize=(10, 8))
                        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                        plt.title("Confusion Matrix")
                        plt.colorbar()
                        tick_marks = np.arange(len(self.class_names))
                        plt.xticks(tick_marks, self.class_names, rotation=45)
                        plt.yticks(tick_marks, self.class_names)
                        plt.tight_layout()
                        plt.ylabel('True label')
                        plt.xlabel('Predicted label')
                        
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        image = tf.image.decode_png(buf.getvalue(), channels=4)
                        image = tf.expand_dims(image, 0)
                        tf.summary.image("Confusion Matrix", image, step=epoch)
                        plt.close(figure)

            extra_logging = ExtraLoggingCallback(test_dataset, class_names)
            callbacks.append(extra_logging)

            # Add EarlyStopping callback if enabled in config
            if self.config['training'].get('early_stopping', {}).get('enabled', False):
                early_stopping = EarlyStopping(
                    monitor=self.config['training']['early_stopping'].get('monitor', 'val_loss'),
                    patience=self.config['training']['early_stopping'].get('patience', 5),
                    min_delta=self.config['training']['early_stopping'].get('min_delta', 0.001),
                    restore_best_weights=self.config['training']['early_stopping'].get('restore_best_weights', True)
                )
                callbacks.append(early_stopping)
                logging.info("Early stopping enabled")
            
            logging.info(f"Created callbacks. Log dir: {log_dir}, Checkpoint dir: {checkpoint_dir}")
            return callbacks
            
        except Exception as e:
            logging.error(f"Failed to create callbacks: {str(e)}")
            raise

    def get_hardware_info(self):
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "max_frequency": psutil.cpu_freq().max
        }
        
        memory_info = {
            "total": psutil.virtual_memory().total / (1024**3),  # in GB
            "available": psutil.virtual_memory().available / (1024**3)  # in GB
        }
        
        gpu_info = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_total": gpu.memoryTotal,
                    "memory_free": gpu.memoryFree
                })
        except:
            gpu_info = "No GPU detected"
        
        return {
            "cpu": cpu_info,
            "memory": memory_info,
            "gpu": gpu_info
        }

    def save_experiment_results(self, history):
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        results = {
            "hardware_info": self.get_hardware_info(),
            "training_time": self.end_time - self.start_time if self.end_time else None,
            "history": {k: [convert_to_serializable(v) for v in vals] for k, vals in history.items()}
        }
        
        results_path = os.path.join(self.experiment_dir, 'experiment_results.json')
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=convert_to_serializable)
            logging.info(f"Saved experiment results to {results_path}")
        except Exception as e:
            logging.error(f"Failed to save experiment results: {str(e)}")
            raise