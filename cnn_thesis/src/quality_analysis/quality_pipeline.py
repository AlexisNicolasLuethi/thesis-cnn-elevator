import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path
import logging

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

class QualityAnalyzer:
    def __init__(self, output_dir):
        """Initialize the analyzer with VGG16 for feature extraction"""
        self.feature_extractor = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / 'plots'
        self.outliers_dir = self.output_dir / 'outliers'

        # Create output directories
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.outliers_dir.mkdir(parents=True, exist_ok=True)

        # Track processing timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def extract_features(self, image_paths, batch_size=32):
        """Extract features from images using VGG16"""
        features = []
        valid_paths = []
        
        logging.info("Extracting features...")
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_features = []
            
            for path in batch_paths:
                try:
                    img = image.load_img(path, target_size=(224, 224))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    batch_features.append(x)
                    valid_paths.append(path)
                except Exception as e:
                    logging.error(f"Error processing {path}: {e}")
                    continue
            
            if batch_features:
                batch_images = np.vstack(batch_features)
                batch_features = self.feature_extractor.predict(batch_images)
                features.extend(batch_features.reshape(len(batch_features), -1))
        
        logging.info("Feature extraction complete.")
        return np.array(features), valid_paths
    
    def find_optimal_clusters(self, features, max_clusters=10):
        """Find optimal number of clusters using elbow method"""
        distortions = []
        K = range(1, min(max_clusters + 1, len(features)))
        
        logging.info("Finding optimal number of clusters...")
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features)
            distortions.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('Elbow Method For Optimal k')
        plt.savefig(self.plots_dir / f'elbow_plot_{self.timestamp}.png')
        plt.close()
        
        # Find elbow point (simple method)
        diffs = np.diff(distortions)
        elbow_point = np.argmin(diffs) + 1
        
        logging.info(f"Optimal number of clusters: {elbow_point}")
        return elbow_point
    
    def detect_outliers(self, features, contamination=0.1, method='isolation_forest', config=None):
        """Detect outliers using the specified method."""
        logging.info(f"Detecting outliers using {method}...")
        
        if method == 'isolation_forest':
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outliers = iso_forest.fit_predict(features)
            return outliers == -1  # Outliers are marked as -1
            
        elif method == 'kmeans':
            if config is None:
                raise ValueError("config is required for kmeans method")
                
            # Fit KMeans
            kmeans = KMeans(
                n_clusters=config['parameters'].get('n_clusters', 2),
                random_state=42
            )
            labels = kmeans.fit_predict(features)
            
            # Calculate distances to cluster centers
            distances = np.min(kmeans.transform(features), axis=1)
            
            # Use cluster-specific thresholds
            cluster_outliers = np.zeros(len(features), dtype=bool)
            for cluster_id in range(kmeans.n_clusters):
                cluster_mask = labels == cluster_id
                cluster_distances = distances[cluster_mask]
                
                if len(cluster_distances) > 0:
                    cluster_mean = np.mean(cluster_distances)
                    cluster_std = np.std(cluster_distances)
                    cluster_threshold = cluster_mean + 2 * cluster_std
                    cluster_outliers[cluster_mask] = distances[cluster_mask] > cluster_threshold
            
            return cluster_outliers  # Use cluster-specific thresholds
            
        elif method == 'dbscan':
            if config is None:
                raise ValueError("config is required for dbscan method")
            dbscan = DBSCAN(
                eps=config['parameters']['dbscan_params'].get('eps', 0.5),
                min_samples=config['parameters']['dbscan_params'].get('min_samples', 5)
            )
            labels = dbscan.fit_predict(features)
            return labels == -1  # Outliers are labeled as -1 in DBSCAN
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")

    def visualize_clusters(self, features, clusters, outliers):
        """Visualize clusters and outliers using t-SNE"""
        logging.info("Generating cluster visualizations...")
        
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        # Plot clusters
        plt.figure(figsize=(15, 6))
        
        # Plot 1: Clusters
        plt.subplot(121)
        plt.scatter(features_2d[:, 0], features_2d[:, 1], c=clusters, cmap='viridis')
        plt.title('Clusters')
        
        # Plot 2: Outliers
        plt.subplot(122)
        plt.scatter(features_2d[~outliers, 0], features_2d[~outliers, 1], 
                   c='blue', label='Normal')
        plt.scatter(features_2d[outliers, 0], features_2d[outliers, 1], 
                   c='red', label='Outliers')
        plt.title('Outliers')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'clustering_visualization_{self.timestamp}.png')
        plt.close()
    
    def analyze_dataset(self, image_paths, contamination=0.1, method='isolation_forest', config=None):
        """Run complete analysis pipeline with the specified outlier detection method."""
        # Extract features
        features, valid_paths = self.extract_features(image_paths)

        # Find optimal number of clusters
        n_clusters = self.find_optimal_clusters(features)

        # Perform clustering
        logging.info("Performing clustering...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)

        # Detect outliers
        outliers = self.detect_outliers(features, contamination, method=method, config=config)

        # Visualize results
        self.visualize_clusters(features, clusters, outliers)

        # Save results
        outlier_paths = [path for path, is_outlier in zip(valid_paths, outliers) if is_outlier]
        results = {
            "timestamp": self.timestamp,
            "total_images": int(len(valid_paths)),  # Convert to regular int
            "optimal_clusters": int(n_clusters),     # Convert to regular int
            "outliers_detected": int(len(outlier_paths)),  # Convert to regular int
            "outlier_paths": outlier_paths
        }

        # Save results as JSON
        with open(self.outliers_dir / f'analysis_results_{self.timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        return results
