#!/usr/bin/env python3
"""
Example script demonstrating PEFT adapter usage with the neural network.
This script shows how to:
1. Train a model with PEFT adapters
2. Save the model in PEFT format
3. Load and use the PEFT adapter
"""

import numpy as np
from neural_network import create_network_from_config
from trainer import Trainer
import json
import os

def train_peft_model():
    """Train a model with PEFT adapters"""
    print("Training model with PEFT adapters...")
    
    # Create trainer with PEFT enabled config
    trainer = Trainer("config.json")
    
    # Train the model
    results = trainer.train()
    
    print(f"Training completed with PEFT adapters!")
    print(f"Final test MSE: {results['test_mse']:.6f}")
    print(f"Final test MAE: {results['test_mae']:.6f}")
    
    return results

def load_and_test_peft_model(model_path):
    """Load a PEFT model and test it"""
    print(f"\nLoading PEFT model from {model_path}...")
    
    # Create network from config
    network = create_network_from_config(os.path.join(model_path, "config.json"))
    
    # Load the PEFT model
    network.load_model(model_path, load_format="peft")
    
    # Generate some test data
    X_test = np.random.randn(10, 1) * 2  # 10 test samples
    y_pred = network.predict(X_test)
    
    print(f"PEFT model loaded successfully!")
    print(f"Test predictions shape: {y_pred.shape}")
    print(f"Sample predictions: {y_pred[:5].flatten()}")
    
    return network

def compare_formats():
    """Compare different save formats"""
    print("\n=== Comparing different save formats ===")
    
    # Test numpy format
    config_numpy = {
        "data_source": "huggingface",
        "dataset_name": "synthetic",
        "target_column": "y",
        "feature_columns": ["x"],
        "hidden_sizes": [8, 4],
        "epochs": 1000,
        "learning_rate": 0.01,
        "batch_size": 32,
        "random_seed": 42,
        "test_size": 0.2,
        "model_save_path": "./Model_numpy",
        "data_save_path": "./sample_data",
        "regression_task": true,
        "output_activation": "linear",
        "hidden_activation": "tanh",
        "loss_function": "mse",
        "use_peft": false,
        "save_format": "numpy"
    }
    
    with open("config_numpy.json", "w") as f:
        json.dump(config_numpy, f, indent=2)
    
    # Train numpy format model
    trainer_numpy = Trainer("config_numpy.json")
    results_numpy = trainer_numpy.train()
    
    # Test PEFT format
    config_peft = {
        "data_source": "huggingface",
        "dataset_name": "synthetic",
        "target_column": "y",
        "feature_columns": ["x"],
        "hidden_sizes": [8, 4],
        "epochs": 1000,
        "learning_rate": 0.01,
        "batch_size": 32,
        "random_seed": 42,
        "test_size": 0.2,
        "model_save_path": "./Model_peft",
        "data_save_path": "./sample_data",
        "regression_task": true,
        "output_activation": "linear",
        "hidden_activation": "tanh",
        "loss_function": "mse",
        "use_peft": true,
        "save_format": "peft",
        "peft_config": {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["linear"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "REGRESSION"
        }
    }
    
    with open("config_peft.json", "w") as f:
        json.dump(config_peft, f, indent=2)
    
    # Train PEFT format model
    trainer_peft = Trainer("config_peft.json")
    results_peft = trainer_peft.train()
    
    print(f"\nComparison Results:")
    print(f"NumPy format - Test MSE: {results_numpy['test_mse']:.6f}")
    print(f"PEFT format - Test MSE: {results_peft['test_mse']:.6f}")
    
    # Clean up temporary config files
    os.remove("config_numpy.json")
    os.remove("config_peft.json")

def main():
    """Main function demonstrating PEFT usage"""
    print("=== PEFT Adapter Format Example ===")
    
    # Train a PEFT model
    results = train_peft_model()
    
    # Load and test the PEFT model
    model_path = "./Model"
    if os.path.exists(model_path):
        network = load_and_test_peft_model(model_path)
    else:
        print(f"Model path {model_path} does not exist. Please run training first.")
    
    # Compare different formats (optional)
    # Uncomment to run comparison
    # compare_formats()
    
    print("\n=== PEFT Example Completed ===")

if __name__ == "__main__":
    main()
