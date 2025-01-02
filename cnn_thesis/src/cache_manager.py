import os
import numpy as np
import h5py
import hashlib
from pathlib import Path
import tensorflow as tf
import logging
from typing import Tuple, Dict, Optional
from tensorflow.keras.applications.vgg16 import preprocess_input

class ImageCacheManager:
    def __init__(self, cache_dir: str = 'image_cache', memory_limit_gb: float = 10.0):
        """
        Initialize the cache manager with both memory and disk caching capabilities.
        
        Args:
            cache_dir: Directory to store the disk cache
            memory_limit_gb: Maximum memory cache size in gigabytes
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_limit_gb = memory_limit_gb
        self.memory_cache: Dict[str, np.ndarray] = {}
        self.memory_cache_size = 0
        
    def _generate_cache_key(self, image_paths: list, img_size: tuple) -> str:
        """Generate a unique key for the dataset based on image paths and size."""
        paths_str = ''.join(sorted(image_paths))
        size_str = f"{img_size[0]}x{img_size[1]}"
        key = f"{paths_str}{size_str}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_disk_cache_path(self, cache_key: str) -> Path:
        """Get the path for the disk cache file."""
        return self.cache_dir / f"cache_{cache_key}.h5"
    
    def _estimate_batch_size_gb(self, img_size: tuple) -> float:
        """Estimate size of a single image in GB."""
        return (img_size[0] * img_size[1] * 3 * 4) / (1024 ** 3)  # 4 bytes per float32
        
    def load_and_cache_images(
        self, 
        image_paths: list, 
        img_size: tuple,
        force_disk: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images with caching support (both memory and disk).
        
        Args:
            image_paths: List of image paths
            img_size: Tuple of (height, width)
            force_disk: Force using disk cache even if memory cache is possible
            
        Returns:
            Tuple of (images array, valid_indices)
        """
        cache_key = self._generate_cache_key(image_paths, img_size)
        disk_cache_path = self._get_disk_cache_path(cache_key)
        
        # Try memory cache first if not forced to disk
        if not force_disk and cache_key in self.memory_cache:
            logging.info("Loading images from memory cache")
            return self.memory_cache[cache_key], np.arange(len(image_paths))
        
        # Try disk cache
        if disk_cache_path.exists():
            logging.info("Loading images from disk cache")
            with h5py.File(disk_cache_path, 'r') as f:
                images = f['images'][:]
                valid_indices = f['valid_indices'][:]
                
            # If it fits in memory and not forced to disk, cache in memory
            if not force_disk:
                cache_size_gb = images.nbytes / (1024 ** 3)
                if cache_size_gb <= self.memory_limit_gb:
                    self._update_memory_cache(cache_key, images)
                    
            return images, valid_indices
        
        # No cache hit, load and process images
        logging.info("Cache miss - loading and processing images")
        images, valid_indices = self._load_and_process_images(image_paths, img_size)
        
        # Save to disk cache
        with h5py.File(disk_cache_path, 'w') as f:
            f.create_dataset('images', data=images)
            f.create_dataset('valid_indices', data=valid_indices)
        
        # If it fits in memory and not forced to disk, cache in memory
        if not force_disk:
            cache_size_gb = images.nbytes / (1024 ** 3)
            if cache_size_gb <= self.memory_limit_gb:
                self._update_memory_cache(cache_key, images)
        
        return images, valid_indices
    
    def _update_memory_cache(self, cache_key: str, images: np.ndarray):
        """Update memory cache with new images, removing old entries if necessary."""
        new_size_gb = images.nbytes / (1024 ** 3)
        
        # Remove old cache entries if necessary
        while (self.memory_cache_size + new_size_gb) > self.memory_limit_gb and self.memory_cache:
            old_key = next(iter(self.memory_cache))
            old_size = self.memory_cache[old_key].nbytes / (1024 ** 3)
            del self.memory_cache[old_key]
            self.memory_cache_size -= old_size
        
        # Add new cache entry
        self.memory_cache[cache_key] = images
        self.memory_cache_size += new_size_gb
    
    def _load_and_process_images(
        self, 
        image_paths: list, 
        img_size: tuple
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess images."""
        valid_images = []
        valid_indices = []
        
        for idx, path in enumerate(image_paths):
            try:
                img = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = preprocess_input(img_array)
                valid_images.append(img_array)
                valid_indices.append(idx)
            except Exception as e:
                logging.warning(f"Error loading image {path}: {str(e)}")
                continue
        
        return np.array(valid_images), np.array(valid_indices)
    
    def clear_cache(self, memory: bool = True, disk: bool = False):
        """Clear the cache."""
        if memory:
            self.memory_cache.clear()
            self.memory_cache_size = 0
            
        if disk:
            for cache_file in self.cache_dir.glob("cache_*.h5"):
                cache_file.unlink()
                
    def create_cached_dataset(
        self,
        image_paths: list,
        labels: np.ndarray,
        img_size: tuple,
        batch_size: int,
        is_training: bool = True
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset using cached images with optimized loading.
        """
        # Load cached images using parallel processing
        images, valid_indices = self.load_and_cache_images(image_paths, img_size)
        valid_labels = labels[valid_indices]
        
        # Create initial dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            tf.convert_to_tensor(images, dtype=tf.float32),
            tf.convert_to_tensor(valid_labels, dtype=tf.float32)
        ))
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=1000)
        
        # Enable parallel processing and prefetching
        dataset = dataset.batch(batch_size)
        dataset = dataset.cache()  # Cache the dataset in memory
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        # Configure dataset for optimal performance
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        options.threading.private_threadpool_size = 8
        options.threading.max_intra_op_parallelism = 1
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.map_parallelization = True
        dataset = dataset.with_options(options)
        
        return dataset