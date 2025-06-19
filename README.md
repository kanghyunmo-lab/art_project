# Project ART: AI-based Cryptocurrency Trading System

This project is an AI-based cryptocurrency automated trading system as defined in PRD v1.1.

## Project Structure

- `data_pipeline/`: Scripts for collecting and processing market data.
- `features/`: Feature engineering scripts.
- `models/`: Trained machine learning models.
- `backtester/`: Backtesting engine and strategies.
- `risk_management/`: Risk management modules.
- `execution/`: Order execution handler.
- `config/`: Configuration files.

## Setup

1.  Create a virtual environment:
    ```bash
    python -m venv venv
    # On Windows use `venv\Scripts\activate`
    # On MacOS/Linux use `source venv/bin/activate`
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Configure credentials:
    - Copy `config/credentials_example.py` to `config/credentials.py`.
    - Fill in your actual API keys and tokens in `config/credentials.py`.
