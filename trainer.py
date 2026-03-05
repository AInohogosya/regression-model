import numpy as np
import json
import os
import time
from datetime import datetime
from neural_network import create_network_from_config
from data_generator import get_data, save_local_data


class Trainer:
    def __init__(self, config_path="config.json"):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.config_path = config_path
        self.network = create_network_from_config(config_path)
        self.training_log = []
        
    def mse_loss(self, y_true, y_pred):
        """Calculate Mean Squared Error loss"""
        return np.mean((y_true - y_pred) ** 2)
    
    def mae_loss(self, y_true, y_pred):
        """Calculate Mean Absolute Error loss"""
        return np.mean(np.abs(y_true - y_pred))
    
    def create_batches(self, X, y, batch_size):
        """Create mini-batches from data"""
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        
        batches = []
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batches.append((X[batch_indices], y[batch_indices]))
        
        return batches
    
    def train_epoch(self, X_train, y_train, learning_rate, batch_size):
        """Train for one epoch"""
        batches = self.create_batches(X_train, y_train, batch_size)
        epoch_loss = 0
        
        for X_batch, y_batch in batches:
            # Forward pass
            output, activations, z_values = self.network.forward(X_batch)
            
            # Calculate loss
            batch_loss = self.mse_loss(y_batch, output)
            epoch_loss += batch_loss
            
            # Backward pass
            gradients_w, gradients_b = self.network.backward(X_batch, y_batch, activations, z_values)
            
            # Update parameters
            self.network.update_parameters(gradients_w, gradients_b, learning_rate)
        
        return epoch_loss / len(batches)
    
    def evaluate(self, X, y):
        """Evaluate model on given data"""
        predictions = self.network.predict(X)
        mse = self.mse_loss(y, predictions)
        mae = self.mae_loss(y, predictions)
        return mse, mae, predictions
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Configuration: {self.config}")
        
        # Get data
        if self.config['data_source'] == 'local':
            # Save data to CSV if using local pattern
            save_local_data(self.config_path)
        
        X_train, X_test, y_train, y_test = get_data(self.config_path)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Training parameters
        epochs = self.config['epochs']
        learning_rate = self.config['learning_rate']
        batch_size = self.config['batch_size']
        
        print(f"Training for {epochs} epochs with learning rate {learning_rate}")
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train one epoch
            train_loss = self.train_epoch(X_train, y_train, learning_rate, batch_size)
            
            # Evaluate on test set
            test_mse, test_mae, _ = self.evaluate(X_test, y_test)
            
            # Log progress
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'test_mse': float(test_mse),
                'test_mae': float(test_mae),
                'timestamp': datetime.now().isoformat()
            }
            
            self.training_log.append(log_entry)
            
            # Print progress every 1000 epochs
            if (epoch + 1) % 1000 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.6f}, "
                      f"Test MSE = {test_mse:.6f}, Test MAE = {test_mae:.6f}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Final evaluation
        final_train_mse, final_train_mae, train_predictions = self.evaluate(X_train, y_train)
        final_test_mse, final_test_mae, test_predictions = self.evaluate(X_test, y_test)
        
        print(f"\nFinal Results:")
        print(f"Train MSE: {final_train_mse:.6f}, Train MAE: {final_train_mae:.6f}")
        print(f"Test MSE: {final_test_mse:.6f}, Test MAE: {final_test_mae:.6f}")
        
        # Save model and logs
        self.save_results(X_test, y_test, test_predictions)
        
        return {
            'train_mse': final_train_mse,
            'train_mae': final_train_mae,
            'test_mse': final_test_mse,
            'test_mae': final_test_mae,
            'training_time': training_time
        }
    
    def save_results(self, X_test, y_test, test_predictions):
        """Save model, config, and training logs in PEFT adapter format"""
        model_path = self.config['model_save_path']
        os.makedirs(model_path, exist_ok=True)
        
        # Determine save format based on configuration
        save_format = self.config.get('save_format', 'both')  # 'numpy', 'peft', or 'both'
        
        # Save model in specified format
        self.network.save_model(model_path, save_format=save_format)
        
        # Save config
        import shutil
        shutil.copy2(self.config_path, os.path.join(model_path, 'config.json'))
        
        # Save training log
        training_log_data = {
            'config': self.config,
            'training_log': self.training_log,
            'final_results': {
                'test_mse': float(self.mse_loss(y_test, test_predictions)),
                'test_mae': float(self.mae_loss(y_test, test_predictions))
            },
            'model_format': save_format,
            'peft_enabled': getattr(self.network, 'use_peft', False)
        }
        
        with open(os.path.join(model_path, 'training_log.json'), 'w') as f:
            json.dump(training_log_data, f, indent=2)
        
        # Create PEFT-specific metadata if PEFT is enabled
        if getattr(self.network, 'use_peft', False):
            peft_metadata = {
                'adapter_type': 'lora',
                'task_type': 'REGRESSION',
                'base_model_type': 'neural_network',
                'training_data_info': {
                    'n_samples': len(X_test),
                    'n_features': X_test.shape[1],
                    'target_column': self.config.get('target_column', 'y')
                },
                'performance_metrics': {
                    'final_test_mse': float(self.mse_loss(y_test, test_predictions)),
                    'final_test_mae': float(self.mae_loss(y_test, test_predictions))
                }
            }
            
            with open(os.path.join(model_path, 'peft_metadata.json'), 'w') as f:
                json.dump(peft_metadata, f, indent=2)
        
        print(f"Results saved to {model_path} in format: {save_format}")


def main():
    """Main training function"""
    trainer = Trainer()
    results = trainer.train()
    return results


if __name__ == "__main__":
    main()
