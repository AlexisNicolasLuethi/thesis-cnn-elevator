import os
import logging
import json
from quality_pipeline import QualityAnalyzer
from pathlib import Path
from datetime import datetime

def setup_logging(logs_dir):
    """Set up logging to both file and console"""
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logger.handlers = []
    
    file_handler = logging.FileHandler(str(logs_dir / 'logs.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def check_existing_analysis(output_base, analysis_class, analysis_type, detection_method):
    """Check if analysis already exists for this method"""
    method_path = output_base / analysis_class / analysis_type / detection_method
    
    if not method_path.exists():
        return False
        
    # Check if there are any timestamp folders
    timestamp_folders = [f for f in method_path.iterdir() if f.is_dir()]
    if not timestamp_folders:
        return False
        
    # Check if any of these folders contain results
    for folder in timestamp_folders:
        results_path = folder / 'outliers'
        if results_path.exists() and any(results_path.iterdir()):
            logging.info(f"Analysis already exists for {analysis_class}/{analysis_type} using {detection_method}")
            return True
            
    return False

def create_output_structure(base_dir, analysis_class, analysis_type, detection_method, timestamp):
    """Create structured output directory based on class, type, and detection method"""
    output_path = base_dir / analysis_class / analysis_type / detection_method / timestamp
    
    logs_dir = output_path / 'logs'
    outliers_dir = output_path / 'outliers'
    plots_dir = output_path / 'plots'
    
    for directory in [logs_dir, outliers_dir, plots_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    return output_path, logs_dir, outliers_dir, plots_dir

def get_image_paths(data_dir, max_images=None):
    """Recursively get all image paths from directory, with an option to limit the number of images"""
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
                if max_images and len(image_paths) >= max_images:
                    return image_paths
    return image_paths

def get_subclasses(base_image_dir):
    """Get all subclasses from the image directory structure"""
    subclasses = []
    for class_name in os.listdir(base_image_dir):
        class_path = Path(base_image_dir) / class_name
        if class_path.is_dir():
            for subclass in os.listdir(class_path):
                subclass_path = class_path / subclass
                if subclass_path.is_dir():
                    subclasses.append({
                        'class': class_name,
                        'type': subclass,
                        'path': str(subclass_path)
                    })
    return subclasses

def analyze_subclass(subclass_info, output_base, config):
    """Analyze a single subclass"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detection_method = config['parameters'].get('outlier_method', 'isolation_forest')
    
    # Check if analysis already exists
    if check_existing_analysis(
        output_base, 
        subclass_info['class'], 
        subclass_info['type'],
        detection_method
    ):
        return None
    
    # Create structured output directory
    run_output_dir, logs_dir, outliers_dir, plots_dir = create_output_structure(
        output_base, 
        subclass_info['class'], 
        subclass_info['type'],
        detection_method,
        timestamp
    )
    
    # Set up logging for this run
    setup_logging(logs_dir)
    
    # Update CONFIG with the new paths
    run_config = config.copy()
    run_config['output_dir'] = str(run_output_dir)
    run_config['logs_dir'] = str(logs_dir)
    run_config['outliers_dir'] = str(outliers_dir)
    run_config['plots_dir'] = str(plots_dir)
    
    # Log analysis information
    logging.info(f"\nStarting analysis for {subclass_info['class']} - {subclass_info['type']}")
    logging.info(f"Using outlier detection method: {detection_method}")
    logging.info(f"Scanning directory: {subclass_info['path']}")
    
    # Get image paths
    image_paths = get_image_paths(
        subclass_info['path'], 
        max_images=config['parameters'].get('max_images')
    )
    logging.info(f"Found {len(image_paths)} images")
    
    if not image_paths:
        logging.warning("No images found, skipping analysis")
        return
    
    # Initialize analyzer
    analyzer = QualityAnalyzer(output_dir=run_config['output_dir'])
    
    # Run analyzer
    results = analyzer.analyze_dataset(
        image_paths,
        contamination=config['parameters']['contamination'],
        method=detection_method,
        config=config
    )
    
    # Log results
    logging.info("\nAnalysis Complete!")
    logging.info(f"Results saved in: {run_config['output_dir']}")
    logging.info(f"Optimal number of clusters: {results['optimal_clusters']}")
    logging.info(f"Outliers detected: {results['outliers_detected']} ({results['outliers_detected']/results['total_images']*100:.1f}%)")
    
    return results

def main():
    # Get the path to the quality_analysis directory
    base_dir = Path(__file__).parent
    config_path = base_dir / 'configs' / 'config.json'
    
    # Load the configuration
    with open(config_path) as f:
        CONFIG = json.load(f)
    
    # Set up base output directory
    output_base = base_dir / CONFIG['paths']['output_base']
    
    # Get all subclasses from the image directory
    subclasses = get_subclasses(CONFIG['paths']['data_dir'])
    
    # Process each subclass
    all_results = {}
    for subclass_info in subclasses:
        try:
            results = analyze_subclass(subclass_info, output_base, CONFIG)
            if results:  # Only add results if analysis was performed
                all_results[f"{subclass_info['class']}/{subclass_info['type']}"] = results
        except Exception as e:
            logging.error(f"Error processing {subclass_info['class']}/{subclass_info['type']}: {e}")
            continue
    
    # Save overall summary only if new results were generated
    if all_results:
        summary_path = output_base / f"batch_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logging.info(f"\nBatch analysis complete! Summary saved to {summary_path}")
    else:
        logging.info("\nNo new analyses were needed - all components already processed.")

if __name__ == "__main__":
    main()