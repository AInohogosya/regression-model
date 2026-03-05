# Pure Regression Neural Network - Sin(x) Approximation

A complete from-scratch neural network implementation for pure regression, trained to approximate the mathematical function y = sin(x).

## Project Overview

This is a minimal, production-grade neural network project that demonstrates:
- Pure regression (no classification components)
- Complete from-scratch implementation using only NumPy
- Modular, well-organized codebase
- Automatic testing and evaluation
- Clean project structure with proper configuration management

## Key Features

- **Pure Regression**: Single input x → single output y = sin(x)
- **No Classification**: No softmax, no cross-entropy, just weights and biases
- **From Scratch**: Complete implementation using only NumPy
- **Dual Pattern Support**: Hugging Face (synthetic) and Local (CSV) data patterns
- **Automatic Testing**: Built-in evaluation with detailed metrics and tables
- **Clean Architecture**: Modular design with clear separation of concerns

## Architecture

```
Input (1) → Hidden [16] → Hidden [8] → Output (1)
           tanh           tanh        linear
```

- **Input**: Scalar x value
- **Hidden Layers**: 16 and 8 neurons with tanh activation
- **Output**: 1 neuron with linear activation (direct regression)
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Vanilla gradient descent with backpropagation

## Quick Start

### Option 1: Hugging Face Pattern (Synthetic Data)
```bash
# Train the model
python trainer.py

# Test the model
python test_model.py
```

### Option 2: Local Pattern (CSV Data)
```bash
# Edit config.json: change "data_source" from "huggingface" to "local"

# Train the model (automatically generates CSV)
python trainer.py

# Test the model
python test_model.py
```

### Clean Up
```bash
python cleanup.py
```

## Project Structure

```
純粋な回帰モデル/
├── config.json              # Configuration file
├── data_generator.py        # Data generation and loading
├── neural_network.py        # Neural network implementation
├── trainer.py              # Training script
├── test_model.py           # Testing and evaluation
├── cleanup.py              # Cleanup script
├── requirements.txt        # Dependencies file
├── setup.py                # Package setup file
├── README.md               # This file
└── Model/                  # Generated after training
    ├── config.json         # Copy of training config
    ├── model_weights.npz   # Trained weights and biases
    ├── neural_net.py       # Inference-only model
    ├── training_log.json   # Complete training history
    └── README.md           # Model documentation
```

## Configuration

The `config.json` file controls all aspects of the project:

```json
{
  "data_source": "huggingface",     // "huggingface" or "local"
  "hidden_sizes": [16, 8],          // Hidden layer sizes
  "epochs": 8000,                   // Training epochs
  "learning_rate": 0.03,             // Learning rate
  "batch_size": 32,                  // Batch size
  "random_seed": 42,                 // Reproducibility
  "noise_std": 0.05,                 // Noise in training data
  "x_range": [-6.283185, 6.283185]  // Input range [-2π, 2π]
}
```

### Switching Between Patterns

**Hugging Face Pattern (Default)**:
- `"data_source": "huggingface"`
- Generates synthetic data in memory
- No external files required

**Local Pattern**:
- `"data_source": "local"`
- Automatically creates `sample_data/regression_data.csv`
- Data persists between runs

## Model Performance

Expected performance metrics:
- **MSE**: < 0.01
- **MAE**: < 0.08
- **R²**: > 0.95

## Using the Trained Model

```python
from Model.neural_net import SinRegressionNet

# Load the trained model
model = SinRegressionNet()

# Make predictions
x_values = [0, np.pi/2, np.pi]
predictions = model.predict(x_values)

# Single prediction
single_pred = model.predict_single(np.pi/4)

# Evaluate performance
results = model.evaluate_range(-2*np.pi, 2*np.pi, 1000)
print(f"MSE: {results['mse']:.8f}")
```

## Training Process

1. **Data Generation**: Creates 1000 samples of y = sin(x) + noise
2. **Network Initialization**: Xavier initialization for tanh layers
3. **Training Loop**: 8000 epochs of mini-batch gradient descent
4. **Evaluation**: Automatic testing on held-out test set
5. **Saving**: Model weights, config, and training log saved to `./Model/`

## Testing and Evaluation

The automatic test provides:
- **Detailed Results Table**: Shows x, predicted y, actual y, and absolute error
- **Performance Metrics**: MSE, MAE, RMSE, and R²
- **Range Analysis**: Error analysis across different x ranges
- **Quality Assessment**: Automatic model quality evaluation

Example output:
```
Top 10 Best Predictions (Lowest Error):
Index  x           Predicted y  Actual y    Abs Error  
0      -3.141593   -0.001234    -0.000123   0.001111   
1      0.000000    0.000456     0.000000    0.000456   
...

Performance Metrics:
  Mean Squared Error (MSE): 0.00456789
  Mean Absolute Error (MAE): 0.05678901
  Root Mean Squared Error (RMSE): 0.06759012
  R-squared (R²): 0.98765432
```

## Technical Implementation

### Neural Network
- Pure NumPy implementation
- Forward and backward propagation
- Xavier weight initialization
- Tanh activation for hidden layers
- Linear activation for output layer

### Training
- Mini-batch gradient descent
- Mean Squared Error loss
- Analytical gradient computation
- Fixed random seed for reproducibility

### Data Handling
- Synthetic sin(x) data generation
- Gaussian noise injection (σ = 0.05)
- Train/test split (80/20)
- CSV export for local pattern

## Dependencies

- **NumPy**: Numerical computations
- **Pandas**: CSV handling (local pattern)
- **Scikit-learn**: Train/test splitting

### Installation Options

**Option 1: Using pip (Recommended)**
```bash
pip install -r requirements.txt
```

**Option 2: Using setup.py**
```bash
pip install -e .
```

**Option 3: Manual installation**
```bash
pip install numpy>=1.21.0 pandas>=1.3.0 scikit-learn>=1.0.0
```

**Python Version Requirement**: Python 3.7 or higher

## Reproducibility

- Fixed random seed (42)
- Deterministic weight initialization
- Complete training log saved
- Exact configuration preserved

## Clean Development

The project includes a comprehensive cleanup script that removes:
- Generated model files (`./Model/`)
- Sample data (`./sample_data/`)
- Python cache files
- Temporary files

Run cleanup with:
```bash
python cleanup.py
```

## Extensions

This project can be extended for:
- Different regression functions (cos, polynomial, etc.)
- Multi-dimensional regression
- Different activation functions
- Advanced optimizers (Adam, RMSprop)
- Regularization techniques
- Hyperparameter tuning

## License

This project is provided as-is for educational and research purposes.
