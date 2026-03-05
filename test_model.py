import numpy as np
import json
import os
import pandas as pd
from neural_network import NeuralNetwork
from data_generator import get_data


def load_trained_model(model_path):
    """Load trained model from saved weights"""
    # Load model configuration
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # Create network architecture
    input_size = len(config['feature_columns'])
    hidden_sizes = config['hidden_sizes']
    output_size = 1
    
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    
    network = NeuralNetwork(
        layer_sizes=layer_sizes,
        hidden_activation=config['hidden_activation'],
        output_activation=config['output_activation'],
        random_seed=config['random_seed']
    )
    
    # Load trained weights
    network.load_model(model_path)
    
    return network, config


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    # Calculate R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def print_results_table(X_test, y_true, y_pred, max_rows=10):
    """Print detailed results table"""
    print("\n" + "="*80)
    print("DETAILED PREDICTION RESULTS")
    print("="*80)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'x': X_test.flatten(),
        'predicted_y': y_pred.flatten(),
        'actual_y': y_true.flatten(),
        'absolute_error': np.abs(y_pred.flatten() - y_true.flatten())
    })
    
    # Sort by absolute error to show best and worst predictions
    results_sorted = results_df.sort_values('absolute_error')
    
    print(f"\nTop {max_rows} Best Predictions (Lowest Error):")
    print("-" * 80)
    print(f"{'Index':<6} {'x':<12} {'Predicted y':<12} {'Actual y':<12} {'Abs Error':<12}")
    print("-" * 80)
    
    for i, (idx, row) in enumerate(results_sorted.head(max_rows).iterrows()):
        print(f"{idx:<6} {row['x']:<12.6f} {row['predicted_y']:<12.6f} {row['actual_y']:<12.6f} {row['absolute_error']:<12.6f}")
    
    print(f"\nTop {max_rows} Worst Predictions (Highest Error):")
    print("-" * 80)
    print(f"{'Index':<6} {'x':<12} {'Predicted y':<12} {'Actual y':<12} {'Abs Error':<12}")
    print("-" * 80)
    
    for i, (idx, row) in enumerate(results_sorted.tail(max_rows).iterrows()):
        print(f"{idx:<6} {row['x']:<12.6f} {row['predicted_y']:<12.6f} {row['actual_y']:<12.6f} {row['absolute_error']:<12.6f}")
    
    return results_df


def analyze_predictions(X_test, y_true, y_pred):
    """Analyze prediction quality across different ranges"""
    print("\n" + "="*80)
    print("PREDICTION ANALYSIS BY X RANGE")
    print("="*80)
    
    # Create analysis DataFrame
    analysis_df = pd.DataFrame({
        'x': X_test.flatten(),
        'y_true': y_true.flatten(),
        'y_pred': y_pred.flatten(),
        'error': np.abs(y_pred.flatten() - y_true.flatten())
    })
    
    # Define ranges for analysis
    x_min, x_max = X_test.min(), X_test.max()
    n_ranges = 4
    range_width = (x_max - x_min) / n_ranges
    
    print(f"\nError Analysis by X Range:")
    print("-" * 60)
    
    for i in range(n_ranges):
        range_start = x_min + i * range_width
        range_end = x_min + (i + 1) * range_width
        
        mask = (analysis_df['x'] >= range_start) & (analysis_df['x'] < range_end)
        range_data = analysis_df[mask]
        
        if len(range_data) > 0:
            avg_error = range_data['error'].mean()
            max_error = range_data['error'].max()
            std_error = range_data['error'].std()
            
            print(f"Range [{range_start:.2f}, {range_end:.2f}): "
                  f"Avg Error = {avg_error:.6f}, "
                  f"Max Error = {max_error:.6f}, "
                  f"Std Error = {std_error:.6f}, "
                  f"Samples = {len(range_data)}")


def test_model(model_path="./Model"):
    """Main test function"""
    print("Loading trained model...")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Load model and config
    network, config = load_trained_model(model_path)
    
    # Load test data
    print("Loading test data...")
    X_train, X_test, y_train, y_test = get_data()
    
    # Make predictions
    print("Making predictions...")
    y_pred = network.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Print results
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    print(f"Dataset: {config['data_source']}")
    print(f"Test samples: {len(X_test)}")
    print(f"Network architecture: {config['hidden_sizes']}")
    print(f"Hidden activation: {config['hidden_activation']}")
    print(f"Output activation: {config['output_activation']}")
    print(f"Loss function: {config['loss_function']}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Mean Squared Error (MSE): {metrics['mse']:.8f}")
    print(f"  Mean Absolute Error (MAE): {metrics['mae']:.8f}")
    print(f"  Root Mean Squared Error (RMSE): {metrics['rmse']:.8f}")
    print(f"  R-squared (R²): {metrics['r2']:.8f}")
    
    # Print detailed results table
    results_df = print_results_table(X_test, y_test, y_pred)
    
    # Analyze predictions by range
    analyze_predictions(X_test, y_test, y_pred)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Target function: y = sin(x)")
    print(f"X range: [{config['x_range'][0]:.6f}, {config['x_range'][1]:.6f}]")
    print(f"Noise std: {config['noise_std']}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    print(f"\nPrediction Statistics:")
    print(f"  True y range: [{y_test.min():.6f}, {y_test.max():.6f}]")
    print(f"  Pred y range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
    print(f"  Mean absolute error: {metrics['mae']:.6f}")
    print(f"  Max absolute error: {np.max(np.abs(y_pred - y_test)):.6f}")
    
    print(f"\nModel Quality Assessment:")
    if metrics['r2'] > 0.95:
        print("  ✅ Excellent fit (R² > 0.95)")
    elif metrics['r2'] > 0.90:
        print("  ✅ Good fit (R² > 0.90)")
    elif metrics['r2'] > 0.80:
        print("  ⚠️  Fair fit (R² > 0.80)")
    else:
        print("  ❌ Poor fit (R² ≤ 0.80)")
    
    return metrics, results_df


if __name__ == "__main__":
    test_model()
