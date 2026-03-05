import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split


def generate_synthetic_data(config_path="config.json"):
    """Generate synthetic sin(x) regression data"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    np.random.seed(config['random_seed'])
    
    # Generate x values
    x_min, x_max = config['x_range']
    n_samples = config['n_samples']
    x = np.linspace(x_min, x_max, n_samples)
    
    # Generate y = sin(x) with noise
    noise_std = config['noise_std']
    y = np.sin(x) + np.random.normal(0, noise_std, n_samples)
    
    # Split into train and test
    test_size = config['test_size']
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=config['random_seed']
    )
    
    # Reshape for neural network (n_samples, 1)
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    return X_train, X_test, y_train, y_test


def save_local_data(config_path="config.json"):
    """Save data to local CSV file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    X_train, X_test, y_train, y_test = generate_synthetic_data(config_path)
    
    # Combine all data for CSV
    X_all = np.vstack([X_train, X_test])
    y_all = np.vstack([y_train, y_test])
    
    df = pd.DataFrame({
        'x': X_all.flatten(),
        'y': y_all.flatten()
    })
    
    # Create directory if it doesn't exist
    os.makedirs(config['data_save_path'], exist_ok=True)
    
    # Save to CSV
    csv_path = os.path.join(config['data_save_path'], 'regression_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
    
    return X_train, X_test, y_train, y_test


def load_local_data(config_path="config.json"):
    """Load data from local CSV file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    csv_path = os.path.join(config['data_save_path'], 'regression_data.csv')
    df = pd.read_csv(csv_path)
    
    # Extract features and target
    X = df[config['feature_columns']].values
    y = df[config['target_column']].values
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config['random_seed']
    )
    
    # Reshape for neural network
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    return X_train, X_test, y_train, y_test


def get_data(config_path="config.json"):
    """Main function to get data based on config"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if config['data_source'] == 'huggingface':
        print("Using synthetic data generation")
        return generate_synthetic_data(config_path)
    elif config['data_source'] == 'local':
        print("Using local data from CSV")
        return load_local_data(config_path)
    else:
        raise ValueError(f"Unknown data source: {config['data_source']}")


if __name__ == "__main__":
    # Test data generation
    X_train, X_test, y_train, y_test = get_data()
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")
    print(f"Sample x values: {X_train[:5].flatten()}")
    print(f"Sample y values: {y_train[:5].flatten()}")
