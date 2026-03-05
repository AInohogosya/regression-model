import numpy as np
import json
import os
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel


class NeuralNetwork:
    def __init__(self, layer_sizes, hidden_activation='tanh', output_activation='linear', random_seed=42, use_peft=False, peft_config=None):
        """
        Initialize neural network for regression
        
        Args:
            layer_sizes: list of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            hidden_activation: activation function for hidden layers ('tanh')
            output_activation: activation function for output layer ('linear')
            random_seed: random seed for weight initialization
            use_peft: whether to use PEFT (LoRA) for fine-tuning
            peft_config: PEFT configuration for LoRA adapters
        """
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.layer_sizes = layer_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.n_layers = len(layer_sizes)
        self.use_peft = use_peft
        
        # Initialize PyTorch model for PEFT compatibility
        self.pytorch_model = self._create_pytorch_model()
        
        # Apply PEFT if requested
        if self.use_peft:
            if peft_config is None:
                peft_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["0", "2", "4"],
                    lora_dropout=0.1,
                    bias="none",
                    task_type="FEATURE_EXTRACTION"
                )
            self.pytorch_model = get_peft_model(self.pytorch_model, peft_config)
            self.peft_config = peft_config
        
        # Keep numpy version for compatibility with existing code
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers - 1):
            # Xavier initialization for tanh activation
            if hidden_activation == 'tanh':
                limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i + 1]))
                W = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            else:
                W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            
            b = np.zeros((1, layer_sizes[i + 1]))
            
            self.weights.append(W)
            self.biases.append(b)
    
    def tanh(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """Derivative of tanh activation function"""
        return 1 - np.tanh(x) ** 2
    
    def linear(self, x):
        """Linear activation function (identity)"""
        return x
    
    def linear_derivative(self, x):
        """Derivative of linear activation function"""
        return np.ones_like(x)
    
    def forward(self, X):
        """
        Forward propagation
        
        Args:
            X: input data of shape (batch_size, input_size)
        
        Returns:
            output: network output
            activations: list of activations for each layer
            z_values: list of pre-activation values for each layer
        """
        activations = [X]
        z_values = []
        
        # Forward through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            if self.hidden_activation == 'tanh':
                activation = self.tanh(z)
            else:
                raise ValueError(f"Unsupported hidden activation: {self.hidden_activation}")
            
            activations.append(activation)
        
        # Forward through output layer
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        z_values.append(z)
        
        if self.output_activation == 'linear':
            output = self.linear(z)
        else:
            raise ValueError(f"Unsupported output activation: {self.output_activation}")
        
        activations.append(output)
        
        return output, activations, z_values
    
    def backward(self, X, y, activations, z_values):
        """
        Backward propagation
        
        Args:
            X: input data
            y: target values
            activations: activations from forward pass
            z_values: pre-activation values from forward pass
        
        Returns:
            gradients: gradients for weights and biases
        """
        m = X.shape[0]
        gradients_w = []
        gradients_b = []
        
        # Output layer error (MSE derivative)
        output_error = (activations[-1] - y) / m
        
        # Gradient for output layer
        if self.output_activation == 'linear':
            delta = output_error * self.linear_derivative(z_values[-1])
        else:
            raise ValueError(f"Unsupported output activation: {self.output_activation}")
        
        grad_w = np.dot(activations[-2].T, delta)
        grad_b = np.sum(delta, axis=0, keepdims=True)
        
        gradients_w.append(grad_w)
        gradients_b.append(grad_b)
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            if self.hidden_activation == 'tanh':
                delta = np.dot(delta, self.weights[i + 1].T) * self.tanh_derivative(z_values[i])
            else:
                raise ValueError(f"Unsupported hidden activation: {self.hidden_activation}")
            
            grad_w = np.dot(activations[i].T, delta)
            grad_b = np.sum(delta, axis=0, keepdims=True)
            
            gradients_w.append(grad_w)
            gradients_b.append(grad_b)
        
        # Reverse gradients to match forward order
        gradients_w = gradients_w[::-1]
        gradients_b = gradients_b[::-1]
        
        return gradients_w, gradients_b
    
    def update_parameters(self, gradients_w, gradients_b, learning_rate):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients_w[i]
            self.biases[i] -= learning_rate * gradients_b[i]
    
    def predict(self, X):
        """Make predictions"""
        output, _, _ = self.forward(X)
        return output
    
    def _create_pytorch_model(self):
        """Create PyTorch equivalent of the neural network"""
        layers = []
        
        # Input to first hidden layer
        layers.append(nn.Linear(self.layer_sizes[0], self.layer_sizes[1]))
        if self.hidden_activation == 'tanh':
            layers.append(nn.Tanh())
        
        # Hidden layers
        for i in range(1, len(self.layer_sizes) - 2):
            layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            if self.hidden_activation == 'tanh':
                layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1]))
        if self.output_activation == 'linear':
            layers.append(nn.Identity())
        
        return nn.Sequential(*layers)
    
    def sync_numpy_to_pytorch(self):
        """Sync numpy weights to PyTorch model"""
        with torch.no_grad():
            layer_idx = 0
            if self.use_peft:
                # For PEFT models, sync to base layers
                for name, module in self.pytorch_model.base_model.named_modules():
                    if isinstance(module, nn.Linear) and 'base_layer' in name:
                        if layer_idx < len(self.weights):
                            module.weight.copy_(torch.tensor(self.weights[layer_idx].T, dtype=torch.float32))
                            module.bias.copy_(torch.tensor(self.biases[layer_idx].flatten(), dtype=torch.float32))
                            layer_idx += 1
            else:
                # For regular models
                for i, module in enumerate(self.pytorch_model.modules()):
                    if isinstance(module, nn.Linear):
                        if layer_idx < len(self.weights):
                            module.weight.copy_(torch.tensor(self.weights[layer_idx].T, dtype=torch.float32))
                            module.bias.copy_(torch.tensor(self.biases[layer_idx].flatten(), dtype=torch.float32))
                            layer_idx += 1
    
    def sync_pytorch_to_numpy(self):
        """Sync PyTorch weights to numpy arrays"""
        layer_idx = 0
        if self.use_peft:
            # For PEFT models, sync from base layers
            for name, module in self.pytorch_model.base_model.named_modules():
                if isinstance(module, nn.Linear) and 'base_layer' in name:
                    if layer_idx < len(self.weights):
                        self.weights[layer_idx] = module.weight.detach().numpy().T
                        self.biases[layer_idx] = module.bias.detach().numpy().reshape(1, -1)
                        layer_idx += 1
        else:
            # For regular models
            for i, module in enumerate(self.pytorch_model.modules()):
                if isinstance(module, nn.Linear):
                    if layer_idx < len(self.weights):
                        self.weights[layer_idx] = module.weight.detach().numpy().T
                        self.biases[layer_idx] = module.bias.detach().numpy().reshape(1, -1)
                        layer_idx += 1
    
    def save_model(self, path, save_format='both'):
        """Save model weights and biases in multiple formats"""
        os.makedirs(path, exist_ok=True)
        
        if save_format in ['numpy', 'both', 'peft']:
            # Always save numpy format for compatibility
            model_data = {}
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                model_data[f'weights_{i}'] = w
                model_data[f'biases_{i}'] = b
            
            # Save metadata as numpy arrays with proper types
            model_data['layer_sizes'] = np.array(self.layer_sizes, dtype=np.int32)
            model_data['hidden_activation'] = np.array(self.hidden_activation, dtype=np.str_)
            model_data['output_activation'] = np.array(self.output_activation, dtype=np.str_)
            model_data['num_layers'] = np.array(len(self.weights), dtype=np.int32)
            
            np.savez(os.path.join(path, 'model_weights.npz'), **model_data)
        
        if save_format in ['peft', 'both'] and self.use_peft:
            # Save PEFT adapter format
            self.sync_numpy_to_pytorch()
            
            # Save the base model
            torch.save(self.pytorch_model.state_dict(), os.path.join(path, 'pytorch_model.bin'))
            
            # Save PEFT adapter
            if hasattr(self.pytorch_model, 'save_pretrained'):
                self.pytorch_model.save_pretrained(path)
            
            # Save PEFT config
            if hasattr(self, 'peft_config'):
                self.peft_config.save_pretrained(path)
            
            # Save model architecture info
            architecture_info = {
                'layer_sizes': self.layer_sizes,
                'hidden_activation': self.hidden_activation,
                'output_activation': self.output_activation,
                'use_peft': self.use_peft,
                'model_type': 'neural_network_regression'
            }
            
            with open(os.path.join(path, 'adapter_config.json'), 'w') as f:
                json.dump(architecture_info, f, indent=2)
        
        print(f"Model saved to {path} in format: {save_format}")
    
    def load_model(self, path, load_format='auto'):
        """Load model weights and biases from multiple formats"""
        if load_format == 'auto':
            # Auto-detect format
            if os.path.exists(os.path.join(path, 'adapter_config.json')):
                load_format = 'peft'
            elif os.path.exists(os.path.join(path, 'model_weights.npz')):
                load_format = 'numpy'
            else:
                raise FileNotFoundError(f"No compatible model format found in {path}")
        
        if load_format == 'peft':
            # For PEFT models, try to load numpy weights first, then recreate PEFT structure
            numpy_path = os.path.join(path, 'model_weights.npz')
            if os.path.exists(numpy_path):
                # Load numpy format instead for simplicity
                model_data = np.load(numpy_path, allow_pickle=True)
                
                # Load architecture info
                with open(os.path.join(path, 'adapter_config.json'), 'r') as f:
                    architecture_info = json.load(f)
                
                self.layer_sizes = architecture_info['layer_sizes']
                self.hidden_activation = architecture_info['hidden_activation']
                self.output_activation = architecture_info['output_activation']
                self.use_peft = architecture_info.get('use_peft', True)
                
                # Load weights and biases
                num_layers = int(model_data['num_layers'])
                self.weights = []
                self.biases = []
                
                for i in range(num_layers):
                    self.weights.append(model_data[f'weights_{i}'])
                    self.biases.append(model_data[f'biases_{i}'])
                
                # Recreate PyTorch model with PEFT
                self.pytorch_model = self._create_pytorch_model()
                if self.use_peft:
                    from peft import LoraConfig, get_peft_model
                    peft_config = LoraConfig(
                        r=16,
                        lora_alpha=32,
                        target_modules=["0", "2", "4"],
                        lora_dropout=0.1,
                        bias="none",
                        task_type="FEATURE_EXTRACTION"
                    )
                    self.pytorch_model = get_peft_model(self.pytorch_model, peft_config)
                    self.peft_config = peft_config
                
                # Sync weights to PyTorch
                self.sync_numpy_to_pytorch()
            else:
                raise FileNotFoundError(f"No numpy weights found in {path}")
            
        elif load_format == 'numpy':
            # Load numpy format (original)
            model_data = np.load(os.path.join(path, 'model_weights.npz'), allow_pickle=True)
            
            # Load weights and biases
            num_layers = int(model_data['num_layers'])
            self.weights = [model_data[f'weights_{i}'] for i in range(num_layers)]
            self.biases = [model_data[f'biases_{i}'] for i in range(num_layers)]
            
            # Load metadata
            self.layer_sizes = model_data['layer_sizes'].tolist()
            self.hidden_activation = str(model_data['hidden_activation'])
            self.output_activation = str(model_data['output_activation'])
            
            # Recreate PyTorch model if needed
            self.pytorch_model = self._create_pytorch_model()
            self.sync_numpy_to_pytorch()
        
        print(f"Model loaded from {path} using format: {load_format}")


def create_network_from_config(config_path="config.json"):
    """Create neural network from config file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    input_size = len(config['feature_columns'])
    hidden_sizes = config['hidden_sizes']
    output_size = 1  # Regression task has single output
    
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    
    # Check if PEFT is enabled in config
    use_peft = config.get('use_peft', False)
    peft_config_dict = config.get('peft_config', {})
    
    peft_config = None
    if use_peft and peft_config_dict:
        peft_config = LoraConfig(**peft_config_dict)
    
    network = NeuralNetwork(
        layer_sizes=layer_sizes,
        hidden_activation=config['hidden_activation'],
        output_activation=config['output_activation'],
        random_seed=config['random_seed'],
        use_peft=use_peft,
        peft_config=peft_config
    )
    
    return network
