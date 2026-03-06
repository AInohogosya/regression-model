# Regression Neural Network - Sin(x) Approximation

Neural network with PEFT adapters trained to approximate y = sin(x).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python trainer.py

# Test model
python test_model.py
```

## Architecture

```
Input (1) → Hidden [16] → Hidden [8] → Output (1)
           tanh           tanh        linear
```

## Configuration

Edit `config.json` to adjust:
- `hidden_sizes`: [16, 8] - Layer sizes
- `epochs`: 8000 - Training epochs
- `learning_rate`: 0.03 - Learning rate
- `use_peft`: true - Enable PEFT adapters

## PEFT Support

Enable PEFT adapters for efficient fine-tuning:
```bash
python peft_example_usage.py
```

## Dependencies

- torch>=2.0.0
- peft>=0.6.0
- transformers>=4.30.0
- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=1.0.0

## Project Structure

```
regression-model/
├── config.json              # Configuration
├── neural_network.py        # Neural network with PEFT
├── trainer.py              # Training script
├── test_model.py           # Testing
├── peft_example_usage.py   # PEFT example
├── requirements.txt        # Dependencies
└── Model/                  # Generated after training
```

## Usage

```python
from Model.neural_net import SinRegressionNet

# Load trained model
model = SinRegressionNet()

# Make predictions
predictions = model.predict([0, np.pi/2, np.pi])
```

## Clean Up

```bash
python cleanup.py
```

## License

Educational and research purposes only.
