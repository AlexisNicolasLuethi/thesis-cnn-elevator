import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Configuration
EXTRACTION_LEVEL = 'subcategory'
EXPERIMENT_TYPE = 'Doordrive'

def setup_logging():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def extract_class_from_path(path):
    """Extract class name based on folder structure."""
    parts = path.split(os.sep)
    try:
        index = parts.index('images') + (1 if EXTRACTION_LEVEL == 'main_category' else 2)
        return parts[index]
    except (ValueError, IndexError):
        raise ValueError(f"Path {path} does not match expected structure.")

def load_test_data(paths, img_size, class_names):
    """Load and preprocess test images."""
    labels = [extract_class_from_path(p) for p in paths]
    valid_paths = [p for p, label in zip(paths, labels) if label in class_names]
    valid_labels = [class_names.index(extract_class_from_path(p)) for p in valid_paths]
    
    def generator():
        for p, lbl in zip(valid_paths, valid_labels):
            img = tf.keras.preprocessing.image.load_img(p, target_size=img_size)
            yield preprocess_input(tf.keras.preprocessing.image.img_to_array(img)), to_categorical(lbl, len(class_names))
    
    return tf.data.Dataset.from_generator(
        generator, 
        output_signature=(
            tf.TensorSpec(shape=(*img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(len(class_names),), dtype=tf.float32)
        )
    ).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

def evaluate_model(model, dataset, class_names):
    """Evaluate model and generate metrics."""
    y_true, y_pred, y_proba = [], [], []
    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        y_proba.extend(preds)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(y_batch, axis=1))
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    return y_true, y_pred, np.array(y_proba)

def plot_enumerated_confusion_matrix(y_true, y_pred, class_names, save_path, mapping_path):
    """Plot confusion matrix with class indices and save the class-name mapping."""
    # Map class names to indices
    class_mapping = {i: name for i, name in enumerate(class_names)}

    # Save mapping to a file
    with open(mapping_path, "w") as f:
        for i, name in class_mapping.items():
            f.write(f"{i}: {name}\n")
    print(f"Class mapping saved to {mapping_path}")

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=False, fmt='d', cmap='Blues',
        xticklabels=list(class_mapping.keys()),
        yticklabels=list(class_mapping.keys()),
        cbar=True
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Enumerated Classes)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def run_experiment(experiment_folder):
    """Run the evaluation pipeline."""
    logger.info(f"Processing: {experiment_folder}")
    
    # Define expected file paths
    model_path = os.path.join(experiment_folder, 'final_model.h5')
    config_path = os.path.join(experiment_folder, 'config.json')
    data_split_path = os.path.join(experiment_folder, 'data_split.json')
    
    # Log paths for debugging
    logger.info(f"Looking for model at: {model_path}")
    logger.info(f"Looking for config at: {config_path}")
    logger.info(f"Looking for data split at: {data_split_path}")
    
    # Check if all files exist
    if not all(os.path.exists(p) for p in [model_path, config_path, data_split_path]):
        logger.error("Missing required files. Skipping.")
        return
    
    # Load model and files
    model = load_model(model_path)
    logger.info("Model loaded successfully.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    with open(data_split_path, 'r') as f:
        data_split = json.load(f)
    
    # Continue with evaluation
    class_names = sorted(next(os.walk(config['data']['data_dir']))[1])
    dataset = load_test_data(data_split['test_images'], tuple(config['data']['img_size']), class_names)
    y_true, y_pred, y_proba = evaluate_model(model, dataset, class_names)
    
    # Save confusion matrix and class mapping
    confusion_matrix_path = os.path.join(experiment_folder, "confusion_matrix_enumerated.png")
    mapping_path = os.path.join(experiment_folder, "class_mapping.txt")
    plot_enumerated_confusion_matrix(y_true, y_pred, class_names, confusion_matrix_path, mapping_path)
    logger.info("Evaluation complete.")

def main():
    experiments_dir = os.path.join('/teamspace/studios/this_studio/cnn_thesis/src/experiments', EXPERIMENT_TYPE)
    for folder in os.listdir(experiments_dir):
        if folder.startswith("cnn_experiment_"):
            run_experiment(os.path.join(experiments_dir, folder))

if __name__ == "__main__":
    main()
