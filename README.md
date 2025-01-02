# Elevator Component Classification using CNN Transfer Learning

A deep learning system for automated identification of elevator components using transfer learning with VGG16 architecture. This project was developed as part of a Master's thesis at Lucerne University of Applied Sciences.

## Project Overview

The system achieves 94.25% accuracy in classifying major elevator components across 8 categories by leveraging transfer learning on the VGG16 architecture. Key features:

- Transfer learning implementation using VGG16 pre-trained weights
- Real-time data augmentation pipeline
- Two-tier caching system for efficient data handling
- Taguchi method for hyperparameter optimization
- Comprehensive validation framework

## Repository Structure

```
cnn_thesis/
├── configs/            # Configuration files for experiments
└── src/
    ├── data_loader.py         # Data loading and preprocessing
    ├── model.py              # Model architecture definition
    ├── experiment_tracker.py  # Experiment logging and monitoring
    ├── taguchi_optimizer.py  # Hyperparameter optimization
    └── quality_analysis/     # Data quality assessment pipeline
```

## Key Components

- **Data Pipeline**: Implements robust preprocessing, augmentation, and caching mechanisms
- **Model Architecture**: Adapted VGG16 with custom classification layers
- **Training Framework**: GPU-optimized training with comprehensive monitoring
- **Quality Analysis**: Tools for dataset quality assessment and outlier detection

## Results

- Overall accuracy: 94.25%
- Best performing components:
  - Control units: 97.71% F1-score
  - Speed limiters: 96.10% F1-score
- Cross-validation stability: ±0.76%

## Requirements

See `requirements.txt` for full dependencies.

