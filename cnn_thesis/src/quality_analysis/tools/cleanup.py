import os
import json
import shutil
from pathlib import Path
from datetime import datetime

def get_timestamp_from_folder(folder_name):
    """Extract timestamp from folder name and convert to datetime object."""
    try:
        return datetime.strptime(folder_name, "%Y%m%d_%H%M%S")
    except ValueError:
        return None

def get_newest_folder(path):
    """Get the newest timestamp folder from a directory."""
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    timestamp_folders = [(f, get_timestamp_from_folder(f)) for f in folders]
    valid_folders = [(f, ts) for f, ts in timestamp_folders if ts is not None]
    
    if not valid_folders:
        return None
    
    return max(valid_folders, key=lambda x: x[1])[0]

def get_outliers_count(results_dir):
    """Get the number of outliers from the analysis results JSON file."""
    try:
        # Find the JSON file in the results directory
        json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if not json_files:
            return None
            
        json_path = os.path.join(results_dir, json_files[0])
        with open(json_path, 'r') as f:
            data = json.load(f)
            return data.get('outliers_detected')
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None

def compare_outliers_and_cleanup(component_path):
    """Compare outliers between isolation_forest and kmeans, remove kmeans if counts match."""
    iso_forest_path = os.path.join(component_path, 'isolation_forest')
    kmeans_path = os.path.join(component_path, 'kmeans')
    
    # Check if both directories exist
    if not (os.path.exists(iso_forest_path) and os.path.exists(kmeans_path)):
        return
    
    # Get newest folders for both algorithms
    iso_newest = get_newest_folder(iso_forest_path)
    kmeans_newest = get_newest_folder(kmeans_path)
    
    if not (iso_newest and kmeans_newest):
        return
    
    # Get outliers counts from results
    iso_results_dir = os.path.join(iso_forest_path, iso_newest, 'outliers')
    kmeans_results_dir = os.path.join(kmeans_path, kmeans_newest, 'outliers')
    
    iso_count = get_outliers_count(iso_results_dir)
    kmeans_count = get_outliers_count(kmeans_results_dir)
    
    # If counts match and are not None, remove kmeans folder
    if iso_count is not None and kmeans_count is not None and iso_count == kmeans_count:
        print(f"Matching outlier counts ({iso_count}) found in {component_path}")
        print(f"Removing kmeans folder as results match isolation_forest")
        shutil.rmtree(kmeans_path)

def cleanup_component_directory(component_path):
    """Clean up a component directory by keeping only isolation_forest and kmeans folders."""
    print(f"\nCleaning up: {component_path}")
    
    # Keep only these algorithm folders
    allowed_folders = {'isolation_forest', 'kmeans'}
    
    # List all items in the component directory
    try:
        items = os.listdir(component_path)
    except FileNotFoundError:
        print(f"Directory not found: {component_path}")
        return
    except PermissionError:
        print(f"Permission denied: {component_path}")
        return
    
    # Remove unwanted algorithm folders
    for item in items:
        item_path = os.path.join(component_path, item)
        if os.path.isdir(item_path) and item.lower() not in allowed_folders:
            print(f"Removing unwanted algorithm folder: {item}")
            shutil.rmtree(item_path)
    
    # Clean up timestamp folders within allowed algorithm folders
    for algo_folder in allowed_folders:
        algo_path = os.path.join(component_path, algo_folder)
        if not os.path.exists(algo_path):
            continue
            
        newest_folder = get_newest_folder(algo_path)
        if newest_folder is None:
            continue
            
        # Remove all folders except the newest one
        for folder in os.listdir(algo_path):
            folder_path = os.path.join(algo_path, folder)
            if os.path.isdir(folder_path) and folder != newest_folder:
                print(f"Removing old timestamp folder: {folder}")
                shutil.rmtree(folder_path)
    
    # Compare outliers and cleanup kmeans if needed
    compare_outliers_and_cleanup(component_path)

def main():
    # Base output directory
    output_dir = Path("/teamspace/studios/this_studio/cnn_thesis/src/quality_analysis/output")
    
    # Walk through all directories
    for category in os.listdir(output_dir):
        category_path = output_dir / category
        if not os.path.isdir(category_path):
            continue
            
        # Process each component directory
        for component in os.listdir(category_path):
            component_path = category_path / component
            if os.path.isdir(component_path):
                cleanup_component_directory(component_path)

if __name__ == "__main__":
    print("Starting repository cleanup...")
    main()
    print("\nCleanup completed!")