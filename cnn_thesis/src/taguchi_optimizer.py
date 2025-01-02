import numpy as np
import pandas as pd
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class TaguchiOptimizer:
    def __init__(self, base_config_path: str):
        """
        Initialize the Taguchi Optimizer with optimized parameter ranges based on experimental results.
        """
        self.base_config = self.load_config(base_config_path)
        self.base_config_path = base_config_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup directories (keeping original structure)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.taguchi_base_dir = os.path.join(current_dir, 'taguchi_experiments')
        self.experiment_dir = os.path.join(self.taguchi_base_dir, f'taguchi_run_{self.timestamp}')
        
        os.makedirs(self.taguchi_base_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "configs"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "results"), exist_ok=True)
        
        # Optimized default values based on experimental results
        self.default_values = {
            'lr_decay_factor': 0.1,  # More aggressive decay
            'lr_decay_patience': 8,   # Adjusted based on early stopping patience
            'lr_decay_min_lr': 1e-7,  # Lower minimum learning rate
            'lr_decay_monitor': 'val_loss',
            'early_stopping_monitor': 'val_loss',
            'early_stopping_restore_best_weights': True
        }
        
        # Optimized parameters with refined ranges based on experimental results
        self.parameters = {
            # Learning parameters (adjusted ranges around successful values)
            'learning_rate': [0.0005, 0.0006, 0.0007],  # Shifted upward based on Exp 2
            'batch_size': [72, 76, 80],  # Narrowed range around successful values
            'optimizer_momentum': [0.4, 0.5, 0.6],  # Centered around 0.5 (best performer)
            'dropout_rate': [0.39, 0.40, 0.41],  # Kept narrow range
            'dense_units': [1024, 1088, 1152],  # Centered around successful 1024
            'epochs': [100, 110, 120],  # Increased based on better performance with higher epochs
            'early_stopping_patience': [4, 5, 6],  # Adjusted based on strong influence
            'min_delta': [0.0008, 0.0009, 0.001],  # Refined range
            'batch_norm_momentum': [0.99, 0.995, 0.999],  # Higher range based on results
            
            # Categorical parameters (keeping best performers)
            'kernel_initializer': ['glorot_uniform', 'glorot_normal', 'he_normal'],
            'activation_function': ['selu', 'elu', 'relu'],  # Prioritized based on results
            
            # Binary parameters (simplified based on results)
            'lr_decay_enabled': [True, False, False],  # Kept as is
            'data_augmentation_enabled': [True, True, False],  # Increased True probability
            'outlier_detection': ['none', 'kmeans', 'isolation_forest']
        }
        
        self.l18_array = np.array([
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,2,2,2,2,2,2,2,2,2,2],
            [1,1,1,1,0,0,0,0,1,1,1,1,2,0],
            [1,1,1,1,1,1,1,1,2,2,2,2,0,1],
            [1,1,1,1,2,2,2,2,0,0,0,0,1,2],
            [2,2,2,2,0,0,0,0,2,2,2,2,1,0],
            [2,2,2,2,1,1,1,1,0,0,0,0,2,1],
            [2,2,2,2,2,2,2,2,1,1,1,1,0,2],
            [0,1,2,2,0,1,2,2,0,1,2,2,0,0],
            [0,1,2,2,1,2,0,0,1,2,0,0,1,1],
            [0,1,2,2,2,0,1,1,2,0,1,1,2,2],
            [1,2,0,2,0,1,2,2,1,2,0,0,2,0],
            [1,2,0,2,1,2,0,0,2,0,1,1,0,1],
            [1,2,0,2,2,0,1,1,0,1,2,2,1,2],
            [2,0,1,2,0,1,2,2,2,0,1,1,1,0],
            [2,0,1,2,1,2,0,0,0,1,2,2,2,1],
            [2,0,1,2,2,0,1,1,1,2,0,0,0,2]
        ])
        
        self.results = []
        self.individual_results = []
        logging.info(f"Initialized enhanced Taguchi Optimizer with experiment directory: {self.experiment_dir}")
    def load_config(self, config_path: str) -> Dict:
        """Load and validate the base configuration file."""
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
            
            logging.info(f"Successfully loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            logging.error(f"Failed to load configuration from {config_path}: {str(e)}")
            raise

    def generate_experiment_configs(self) -> List[str]:
        """Generate configuration files for each Taguchi experiment."""
        configs_dir = os.path.join(self.experiment_dir, "configs")
        config_paths = []

        # Load outlier detection results
        quality_analysis_base = '/teamspace/studios/this_studio/cnn_thesis/src/quality_analysis/output'
        outlier_results = {}

        # Cache outlier results for each class/type combination
        if not Path(quality_analysis_base).exists():
            logging.warning(f"Quality analysis directory not found: {quality_analysis_base}")
            outlier_results = {'kmeans': {}, 'isolation_forest': {}}
        else:
            for analysis_type in ['kmeans', 'isolation_forest']:
                outlier_results[analysis_type] = {}
                loaded_count = 0
                total_outliers = 0
                
                try:
                    for component_dir in Path(quality_analysis_base).rglob('**/analysis_results_*.json'):
                        if analysis_type in str(component_dir):
                            try:
                                with open(component_dir) as f:
                                    result = json.load(f)
                                    # Key by component path pattern
                                    component_path = str(component_dir).split('/output/')[1].split('/'+analysis_type)[0]
                                    # Convert to list instead of set for JSON serialization
                                    outlier_results[analysis_type][component_path] = list(result['outlier_paths'])
                                    loaded_count += 1
                                    total_outliers += len(result['outlier_paths'])
                                    logging.debug(f"Loaded outliers for {component_path}: {len(result['outlier_paths'])} outliers")
                            except Exception as e:
                                logging.error(f"Error loading outlier results from {component_dir}: {str(e)}")
                                continue
                                
                    logging.info(f"Loaded {loaded_count} {analysis_type} analysis results")
                    logging.info(f"Total outliers for {analysis_type}: {total_outliers}")
                    logging.info(f"Components with {analysis_type} results: {list(outlier_results[analysis_type].keys())}")
                    
                except Exception as e:
                    logging.error(f"Error processing {analysis_type} outlier results: {str(e)}")
                    outlier_results[analysis_type] = {}

        try:
            # Generate configs based on L18 array
            for i, experiment in enumerate(self.l18_array):
                config = self.base_config.copy()
                    
                # Update model architecture configuration
                config['model'].update({
                    'dense_layers': [self.parameters['dense_units'][experiment[4]]],
                    'dropout_rate': self.parameters['dropout_rate'][experiment[2]],
                    'kernel_initializer': self.parameters['kernel_initializer'][experiment[6]],
                    'activation_function': self.parameters['activation_function'][experiment[7]],
                    'batch_norm_momentum': self.parameters['batch_norm_momentum'][experiment[8]]
                })
                
                # Update training configuration
                config['training'].update({
                    'learning_rate': self.parameters['learning_rate'][experiment[0]],
                    'batch_size': self.parameters['batch_size'][experiment[1]],
                    'epochs': self.parameters['epochs'][experiment[5]],
                    'optimizer_momentum': self.parameters['optimizer_momentum'][experiment[2]]
                })
                
                # Add learning rate decay configuration with defaults
                config['training']['learning_rate_decay'] = {
                    'enabled': self.parameters['lr_decay_enabled'][experiment[12]],
                    'factor': self.default_values['lr_decay_factor'],
                    'patience': self.default_values['lr_decay_patience'],
                    'min_lr': self.default_values['lr_decay_min_lr'],
                    'monitor': self.default_values['lr_decay_monitor']
                }
                
                # Update early stopping configuration
                config['training']['early_stopping'] = {
                    'enabled': True,
                    'patience': self.parameters['early_stopping_patience'][experiment[8]],
                    'min_delta': self.parameters['min_delta'][experiment[9]],
                    'monitor': self.default_values['early_stopping_monitor'],
                    'restore_best_weights': self.default_values['early_stopping_restore_best_weights']
                }
                
                # Update data augmentation configuration
                config['data_augmentation'] = {
                    'enabled': self.parameters['data_augmentation_enabled'][experiment[11]],
                    'rotation_range': 20,
                    'width_shift_range': 0.2,
                    'height_shift_range': 0.2,
                    'horizontal_flip': True,
                    'zoom_range': 0.15,
                    'deterministic': True
                }

                # Add outlier detection configuration
                outlier_method = self.parameters['outlier_detection'][experiment[len(self.parameters)-1]]  # Use last parameter slot
                config['data']['outlier_detection'] = {
                    'method': outlier_method,
                    'cached_results': outlier_results.get(outlier_method, {})
                }
                
                # Save configuration
                config_path = os.path.join(configs_dir, f'config_experiment_{i+1}.json')
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                config_paths.append(config_path)
                
                logging.info(f"Generated configuration for experiment {i+1}")
            
            return config_paths
            
        except Exception as e:
            logging.error(f"Failed to generate experiment configs: {str(e)}")
            raise

    def record_result(self, experiment_idx: int, val_accuracy: float) -> None:
        """Record the results of an experiment."""
        try:
            # Get parameters used in this experiment
            parameters_used = {}
            for param_name, param_levels in self.parameters.items():
                param_idx = list(self.parameters.keys()).index(param_name)
                level = self.l18_array[experiment_idx][param_idx]
                parameters_used[param_name] = param_levels[level]

            result = {
                'experiment': experiment_idx + 1,
                'parameters_used': parameters_used,
                'validation_accuracy': val_accuracy
            }
            
            self.results.append({'val_accuracy': val_accuracy})
            self.individual_results.append(result)
            
            # Save individual result
            result_path = os.path.join(self.experiment_dir, "results", f"experiment_{experiment_idx+1}_result.json")
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=4)
                
            logging.info(f"Recorded result for experiment {experiment_idx+1}: {val_accuracy}")
            
        except Exception as e:
            logging.error(f"Failed to record result for experiment {experiment_idx}: {str(e)}")
            raise

    def generate_taguchi_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive Taguchi analysis results."""
        if not self.results or not self.individual_results:
            raise ValueError("No results to analyze. Run experiments first.")

        try:
            # Create experiment setup section
            experiment_setup = {
                "experiment_id": self.timestamp,
                "tested_parameters": {
                    param_name: {
                        "values_tested": values,
                        "level_mapping": {
                            f"Level {i}": value for i, value in enumerate(values)
                        }
                    }
                    for param_name, values in self.parameters.items()
                },
                "individual_results": self.individual_results
            }

            # Calculate effects for each parameter level
            effects = {}
            optimal_params = {}
            contributions = {}
            
            # Convert results to numpy array for easier manipulation
            accuracies = np.array([result['val_accuracy'] for result in self.results])
            
            for param_idx, param_name in enumerate(self.parameters.keys()):
                level_effects = []
                for level in range(3):  # 3 levels for each parameter
                    mask = self.l18_array[:, param_idx] == level
                    level_mean = accuracies[mask].mean()
                    level_effects.append(level_mean)
                
                # Calculate parameter contribution
                level_variance = np.var(level_effects)
                total_variance = np.var(accuracies)
                contribution = (level_variance / total_variance) * 100 if total_variance != 0 else 0
                
                # Store effects and optimal parameters
                effects[param_name] = {
                    f"Level {i} ({self.parameters[param_name][i]})": effect
                    for i, effect in enumerate(level_effects)
                }
                
                optimal_level = np.argmax(level_effects)
                optimal_params[param_name] = self.parameters[param_name][optimal_level]
                contributions[param_name] = f"{contribution:.2f}%"

            # Find best accuracy and corresponding parameters
            best_accuracy_idx = np.argmax(accuracies)
            best_accuracy = accuracies[best_accuracy_idx]
            best_experiment = self.individual_results[best_accuracy_idx]['parameters_used']

            # Create analysis results section
            analysis_results = {
                "effects": effects,
                "optimal_parameters": optimal_params,
                "parameter_contributions": contributions,
                "best_accuracy": float(best_accuracy),
                "best_experiment": best_experiment
            }

            # Generate visualizations section
            self._generate_visualizations(effects, contributions)

            # Combine all sections
            taguchi_analysis = {
                "experiment_setup": experiment_setup,
                "analysis_results": analysis_results,
                "visualization_paths": {
                    "parameter_effects": [
                        os.path.join(self.experiment_dir, f'parameter_effects_group_{i+1}.png')
                        for i in range((len(effects) + 5) // 6)
                    ],
                    "parameter_contributions": os.path.join(self.experiment_dir, 'parameter_contributions.png')
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Save analysis to file
            analysis_path = os.path.join(self.experiment_dir, 'taguchi_analysis.json')
            with open(analysis_path, 'w') as f:
                json.dump(taguchi_analysis, f, indent=4)

            logging.info(f"Taguchi analysis saved to {analysis_path}")
            return taguchi_analysis

        except Exception as e:
            logging.error(f"Failed to generate Taguchi analysis: {str(e)}")
            raise

    def _generate_visualizations(self, effects: Dict, contributions: Dict) -> None:
            """Generate visualizations for parameter effects and contributions."""
            try:
                # Generate parameter effects plots
                param_names = list(effects.keys())
                n_params = len(param_names)
                params_per_plot = 6
                n_plots = (n_params + params_per_plot - 1) // params_per_plot

                for plot_idx in range(n_plots):
                    plt.figure(figsize=(15, 10))
                    start_idx = plot_idx * params_per_plot
                    end_idx = min(start_idx + params_per_plot, n_params)
                    
                    for i, param_name in enumerate(param_names[start_idx:end_idx], 1):
                        plt.subplot(3, 2, i)
                        effect_values = list(effects[param_name].values())
                        param_levels = [str(level) for level in self.parameters[param_name]]
                        plt.plot(param_levels, effect_values, 'bo-')
                        plt.title(f'Effect of {param_name}')
                        plt.xlabel('Parameter Value')
                        plt.ylabel('Mean Accuracy')
                        plt.grid(True)
                    
                    plt.tight_layout()
                    effects_viz_path = os.path.join(
                        self.experiment_dir, 
                        f'parameter_effects_group_{plot_idx + 1}.png'
                    )
                    plt.savefig(effects_viz_path)
                    plt.close()

                # Generate contribution plot
                plt.figure(figsize=(12, 6))
                contrib_items = sorted(
                    [(k, float(v.strip('%'))) for k, v in contributions.items()],
                    key=lambda x: x[1],
                    reverse=True
                )
                params, contribs = zip(*contrib_items)
                
                # Create bar plot with parameter contributions
                sns.barplot(x=list(params), y=list(contribs))
                plt.title('Parameter Contributions to Variance')
                plt.xlabel('Parameters')
                plt.ylabel('Contribution (%)')
                plt.xticks(rotation=45, ha='right')
                
                contrib_viz_path = os.path.join(self.experiment_dir, 'parameter_contributions.png')
                plt.savefig(contrib_viz_path, bbox_inches='tight')
                plt.close()

            except Exception as e:
                logging.error(f"Failed to generate visualizations: {str(e)}")
                raise