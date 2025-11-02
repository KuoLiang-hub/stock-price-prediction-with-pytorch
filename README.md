# Stock Price Prediction with PyTorch and yfinance

A machine learning project that predicts stock price movements using deep learning. Built with PyTorch and powered by real-time financial data from Yahoo Finance.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This project demonstrates how to build a binary classification model to predict whether a stock price will go up or down the next trading day. It combines technical analysis indicators with a neural network to make predictions based on historical price data.

### Key Features

- **Real-time data collection** using yfinance API
- **Technical indicator generation** (moving averages, volatility, momentum)
- **Deep learning model** with PyTorch
- **Comprehensive evaluation metrics** and visualizations
- **Easy to customize** for different stocks and time periods

## Demo

The model analyzes historical stock data and technical indicators to predict next-day price movements:

```
Final Train Accuracy: 58.32%
Final Test Accuracy: 55.67%

Classification Report:
              precision    recall  f1-score   support
        Down       0.54      0.62      0.58        89
          Up       0.58      0.50      0.54        95
```

## Installation

### Prerequisites

- Python 3.8 or higher
- JupyterLab
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install torch torchvision yfinance numpy pandas scikit-learn matplotlib jupyter jupyterlab
```

4. Launch JupyterLab:
```bash
jupyter lab
```

## Usage

### Quick Start

1. Open the notebook in JupyterLab
2. Run all cells sequentially
3. The model will automatically:
   - Download stock data
   - Train the neural network
   - Generate performance plots
   - Save the trained model

### Customization

**Change the stock ticker:**
```python
ticker = "AAPL"  # Change to any valid ticker (e.g., "MSFT", "GOOGL", "TSLA")
```

**Adjust the time period:**
```python
start_date = end_date - timedelta(days=365*5)  # 5 years of data
```

**Modify model architecture:**
```python
class StockPredictor(nn.Module):
    def __init__(self, input_size):
        super(StockPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),  # Increase neurons
            nn.ReLU(),
            nn.Dropout(0.3),
            # Add more layers...
        )
```

**Change training parameters:**
```python
epochs = 100  # More training epochs
learning_rate = 0.0001  # Lower learning rate
batch_size = 64  # Different batch size
```

## Project Structure

```
stock-price-prediction/
│
├── stock_prediction.ipynb    # Main Jupyter notebook
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore file
│
├── outputs/                   # Generated files (created on first run)
│   ├── training_history.png  # Training/test curves
│   └── stock_predictor_model.pth  # Saved model weights
│
└── data/                      # Downloaded stock data (optional cache)
```

## How It Works

### 1. Data Collection
- Downloads historical stock data using yfinance
- Default: 3 years of daily OHLCV data

### 2. Feature Engineering
Creates technical indicators:
- **Moving Averages**: 5-day and 20-day MAs
- **Returns**: Daily percentage change
- **Volatility**: 20-day rolling standard deviation
- **Momentum**: 10-day price momentum
- **Volume MA**: 20-day volume moving average

### 3. Model Architecture
Neural network with:
- Input layer (11 features)
- 4 hidden layers (64 → 32 → 16 → 1 neurons)
- ReLU activation functions
- Dropout layers for regularization
- Sigmoid output for binary classification

### 4. Training
- Loss function: Binary Cross-Entropy (BCE)
- Optimizer: Adam with learning rate 0.001
- Batch size: 32
- Epochs: 50

### 5. Evaluation
- Accuracy metrics
- Confusion matrix
- Precision, recall, F1-score
- Training/test loss curves

## Results

The model typically achieves:
- **Test Accuracy**: 52-60%
- **Precision**: 0.54-0.58
- **Recall**: 0.50-0.62

> **Note**: Stock prediction is inherently challenging. Accuracy above 50% indicates the model has learned patterns beyond random guessing. This project is for educational purposes and should not be used for actual trading decisions.

## Requirements

```txt
torch>=2.0.0
yfinance>=0.2.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
jupyterlab>=4.0.0
```

## Contributing

Contributions are welcome! Here are some ways to improve the project:

1. **Add LSTM/GRU layers** for better time series modeling
2. **Include more features** (RSI, MACD, Bollinger Bands)
3. **Implement ensemble methods** for better predictions
4. **Add backtesting framework** to evaluate strategy performance
5. **Create a web dashboard** for real-time predictions

To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Known Limitations

- Model performance is limited by the efficient market hypothesis
- Past performance does not guarantee future results
- Does not account for:
  - News sentiment
  - Macroeconomic factors
  - Market events
  - Trading volume patterns
- Not suitable for production trading

## Future Improvements

- [ ] Add LSTM for sequential modeling
- [ ] Implement attention mechanisms
- [ ] Include sentiment analysis from news
- [ ] Add multi-stock prediction
- [ ] Create interactive dashboard with Streamlit
- [ ] Implement portfolio optimization
- [ ] Add backtesting with transaction costs

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

⚠️ **This project is for educational purposes only.** Do not use this model to make actual investment decisions. Stock market prediction is extremely complex and involves significant risk. Always consult with a qualified financial advisor before making investment decisions.

## Acknowledgments

- **PyTorch** for the deep learning framework
- **yfinance** for easy access to Yahoo Finance data
- **scikit-learn** for preprocessing and evaluation tools
- The financial ML community for inspiration and techniques

## Contact

Kuo Liang - [linkedin](https://www.linkedin.com/in/kuo-l-32968a211/) - liangkuo328@gmail.com

Project Link: [https://github.com/KuoLiang-hub/stock-price-prediction-with-pytorch](https://github.com/KuoLiang-hub/stock-price-prediction-with-pytorch)

---

⭐ If you find this project helpful, please consider giving it a star!
