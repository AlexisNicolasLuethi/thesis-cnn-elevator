import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from PIL import Image
from tqdm import tqdm

def parse_image(image_path, label, img_size):
    def _parse_image(image_path, label):
        try:
            img = tf.io.read_file(image_path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, img_size)
            img = tf.cast(img, tf.float32)
            img = preprocess_input(img)
            return img, label, tf.constant(True)
        except tf.errors.InvalidArgumentError as e:
            logging.warning(f"Could not process image: {image_path}. Error: {str(e)}")
            return tf.zeros(img_size + (3,)), label, tf.constant(False)

    [img, label, valid] = tf.py_function(_parse_image, [image_path, label], [tf.float32, tf.float32, tf.bool])
    img.set_shape(img_size + (3,))
    label.set_shape((6,))
    return img, label, valid

def create_dataset(image_data, labels, batch_size, is_training=True):
    """
    Create an optimized dataset for GPU training
    """
    # Convert to tensors and ensure they're on GPU
    dataset = tf.data.Dataset.from_tensor_slices((
        tf.convert_to_tensor(image_data, dtype=tf.float32),
        tf.convert_to_tensor(labels, dtype=tf.float32)
    ))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    # Enable parallel processing and prefetching
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()  # Cache the dataset in memory
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)  # Prefetch next batch while GPU is working
    
    # Configure dataset for optimal performance
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    options.threading.private_threadpool_size = 8  # Use multiple threads for data loading
    options.threading.max_intra_op_parallelism = 1  # Limit intra-op parallelism to reduce CPU contention
    dataset = dataset.with_options(options)
    
    return dataset

def load_and_preprocess_image(path, img_size):
    try:
        img = Image.open(path)
        img = img.resize(img_size)
        img_array = np.array(img)
        if img_array.shape[-1] != 3:
            logging.debug(f"Converting non-RGB image to RGB: {path}")
            img_array = np.stack((img_array,)*3, axis=-1)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        logging.warning(f"Error loading image {path}: {str(e)}")
        return None

def load_data(data_dir, img_size=(224, 224), test_split=0.2, images_per_category=None, 
              sampling_config=None, classification_level='subcategory', outlier_detection=None):
    """
    Load and preprocess image data with support for different classification levels and outlier filtering.
    
    Args:
        data_dir: Root directory containing image data
        img_size: Tuple of (height, width) for resizing images
        test_split: Fraction of data to use for testing
        images_per_category: Maximum number of images per category (optional)
        sampling_config: Configuration for data sampling (optional)
        classification_level: Either 'category' for top-level or 'subcategory' for nested classification
        outlier_detection: Dictionary containing:
            - method: 'none', 'kmeans', or 'isolation_forest'
            - cached_results: Dictionary mapping component paths to sets of outlier image paths
    """
    logging.info("="*80)
    logging.info(f"Starting data loading process at {classification_level} level")
    logging.info(f"Configuration: img_size={img_size}, test_split={test_split}")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist")
    
    image_paths = []
    labels = []
    
    if classification_level == 'category':
        # For top-level classification, use immediate subdirectories
        categories = [d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d))]
        
        for category in categories:
            category_path = os.path.join(data_dir, category)
            # Recursively find all images in category
            for root, _, files in os.walk(category_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(root, file))
                        labels.append(category)
                        
    else:  # subcategory level
        # Original behavior - looking at leaf directories
        for root, _, files in os.walk(data_dir):
            if any(file.lower().endswith(('.jpg', '.jpeg', '.png')) for file in files):
                category = os.path.basename(root)
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(root, file))
                        labels.append(category)

    # Add this after collecting initial image_paths and labels, but before sampling
    if outlier_detection and outlier_detection.get('method') != 'none':
        logging.info(f"Applying {outlier_detection['method']} outlier filtering...")
        initial_count = len(image_paths)
        
        # Get the cached outlier results
        outlier_paths = set()
        cached_results = outlier_detection.get('cached_results', {})
        
        # Convert cached results lists to sets for efficient lookups
        cached_sets = {
            component: set(paths) 
            for component, paths in cached_results.items()
        }
        
        # For each image, check if it's in the outlier set
        filtered_paths = []
        filtered_labels = []
        
        for path, label in zip(image_paths, labels):
            # Extract component path (e.g., "Doorcontrol/Fermator VF5+")
            path_parts = path.split('/images/')[1].split('/')
            component_path = '/'.join(path_parts[:2])
            
            # Check if this path is in outliers for this component
            is_outlier = False
            if component_path in cached_sets:
                if path in cached_sets[component_path]:
                    is_outlier = True
                    outlier_paths.add(path)
            
            if not is_outlier:
                filtered_paths.append(path)
                filtered_labels.append(label)
        
        # Update paths and labels
        image_paths = filtered_paths
        labels = filtered_labels
        
        logging.info(f"Outlier filtering removed {initial_count - len(image_paths)} images")
        logging.info(f"Remaining images after outlier filtering: {len(image_paths)}")
    
    # Sample if required
    if images_per_category:
        unique_labels = list(set(labels))
        sampled_paths = []
        sampled_labels = []
        
        for label in unique_labels:
            label_indices = [i for i, l in enumerate(labels) if l == label]
            sample_size = min(images_per_category, len(label_indices))
            chosen_indices = np.random.choice(label_indices, sample_size, replace=False)
            
            sampled_paths.extend([image_paths[i] for i in chosen_indices])
            sampled_labels.extend([labels[i] for i in chosen_indices])
            
        image_paths = sampled_paths
        labels = sampled_labels
    
    # Rest of the function remains the same
    logging.info(f"\nFound {len(set(labels))} classes at {classification_level} level")
    for label in set(labels):
        count = labels.count(label)
        logging.info(f"{label}: {count} images")
    
    # Continue with existing preprocessing logic...
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)
    
    train_paths, test_paths, y_train, y_test = train_test_split(
        image_paths, labels_categorical, test_size=test_split, random_state=42)
    logging.info(f"Train set: {len(train_paths)} images")
    logging.info(f"Test set: {len(test_paths)} images")
    
    logging.info("\nPreprocessing training images...")
    X_train = []
    y_train_filtered = []
    train_paths_filtered = []
    failed_train_images = 0
    
    for path, label in tqdm(zip(train_paths, y_train), total=len(train_paths), desc="Processing training images"):
        img_array = load_and_preprocess_image(path, img_size)
        if img_array is not None:
            X_train.append(img_array)
            y_train_filtered.append(label)
            train_paths_filtered.append(path)
        else:
            failed_train_images += 1
    
    X_train = np.array(X_train)
    y_train = np.array(y_train_filtered)
    logging.info(f"Failed to process {failed_train_images} training images")
    
    logging.info("\nPreprocessing test images...")
    X_test = []
    y_test_filtered = []
    test_paths_filtered = []
    failed_test_images = 0
    
    for path, label in tqdm(zip(test_paths, y_test), total=len(test_paths), desc="Processing test images"):
        img_array = load_and_preprocess_image(path, img_size)
        if img_array is not None:
            X_test.append(img_array)
            y_test_filtered.append(label)
            test_paths_filtered.append(path)
        else:
            failed_test_images += 1
    
    X_test = np.array(X_test)
    y_test = np.array(y_test_filtered)
    logging.info(f"Failed to process {failed_test_images} test images")
    
    if sampling_config and sampling_config.get('enabled', False):
        logging.info("\n" + "="*80)
        logging.info(f"Applying {sampling_config['method']} sampling strategy...")
        logging.info(f"Original training set shape: {X_train.shape}")
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        y_train_flat = np.argmax(y_train, axis=1)
        
        class_distribution_before = np.bincount(y_train_flat)
        logging.info("Class distribution before sampling:")
        for i, count in enumerate(class_distribution_before):
            logging.info(f"Class {i}: {count} samples ({(count/len(y_train_flat))*100:.1f}%)")
        
        if sampling_config['method'] == 'smote':
            sampler = SMOTE(sampling_strategy=sampling_config['strategy'], random_state=42)
        elif sampling_config['method'] == 'oversample':
            sampler = RandomOverSampler(sampling_strategy=sampling_config['strategy'], random_state=42)
        elif sampling_config['method'] == 'undersample':
            sampler = RandomUnderSampler(sampling_strategy=sampling_config['strategy'], random_state=42)
        else:
            raise ValueError(f"Unsupported sampling method: {sampling_config['method']}")
        
        X_resampled, y_resampled = sampler.fit_resample(X_train_flat, y_train_flat)
        X_train = X_resampled.reshape(-1, img_size[0], img_size[1], 3)
        y_train = to_categorical(y_resampled)
        
        class_distribution_after = np.bincount(y_resampled)
        logging.info("\nClass distribution after sampling:")
        for i, count in enumerate(class_distribution_after):
            logging.info(f"Class {i}: {count} samples ({(count/len(y_resampled))*100:.1f}%)")
        
        logging.info(f"Final training set shape: {X_train.shape}")
        logging.info("="*80)
    
    logging.info("\nFinal dataset summary:")
    logging.info(f"Training samples: {len(X_train)}")
    logging.info(f"Test samples: {len(X_test)}")
    logging.info(f"Image dimensions: {img_size}")
    logging.info("="*80)
    
    # Before the return statement, recompute images_per_category_count from filtered data
    images_per_category_count = {}
    for path in train_paths_filtered + test_paths_filtered:
        category = os.path.basename(os.path.dirname(path))
        images_per_category_count[category] = images_per_category_count.get(category, 0) + 1

    return X_train, X_test, y_train, y_test, le, images_per_category_count, train_paths_filtered, test_paths_filtered

def create_data_generator(train_data, y_train, batch_size, augmentation_config):
    # Ensure batch_size is an integer
    batch_size = int(batch_size)
    
    train_dataset = create_dataset(train_data, y_train, batch_size)
    
    if augmentation_config['enabled']:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(augmentation_config['rotation_range']),
            tf.keras.layers.RandomTranslation(
                augmentation_config['width_shift_range'], 
                augmentation_config['height_shift_range']
            ),
            tf.keras.layers.RandomFlip("horizontal" if augmentation_config['horizontal_flip'] else "vertical"),
            tf.keras.layers.RandomZoom(augmentation_config['zoom_range'])
        ])
        
        # Modify the map function to handle batches correctly
        def apply_augmentation(images, labels):
            return data_augmentation(images, training=True), labels
        
        train_dataset = train_dataset.map(
            apply_augmentation,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    return train_dataset

def save_data_split(train_paths, test_paths, output_file='data_split.json'):
    logging.info(f"\nSaving data split information to {output_file}")
    import json
    split_info = {
        "train_images": train_paths,
        "test_images": test_paths
    }
    with open(output_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    logging.info(f"Successfully saved data split information")