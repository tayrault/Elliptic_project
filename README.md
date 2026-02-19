# XGBoost Elliptic Bitcoin Transaction Classifier

A Streamlit web application for detecting illicit Bitcoin transactions using XGBoost machine learning model.

## Features

- **Data Loading**: Automatically loads data from the `../data` folder
- **XGBoost Model**: Trains an XGBoost classifier with class imbalance handling
- **Interactive Threshold**: Adjust prediction threshold via slider
- **Path analysis**: Path lenght, Distance to illicit node

## Project Structure

```
Elliptic/
├── data/
│   ├── elliptic_txs_classes.csv
│   ├── elliptic_txs_edgelist.csv
│   └── elliptic_txs_features.csv
├── streamlit/
│   ├── app.py                  # Main Streamlit application
│   ├── requirements.txt        # Python dependencies
│   ├── README.md               # This file
│   └── edge.csv, feature.csv   # Example test data for predition
└── .venv/                      # Virtual environment
```

## Setup Instructio├s

1. **Navigate to the streamlit folder:**
   ```bash
   cd streamlit
   ```

2. **Activate the virtual environment:**
   - Windows:
     ```bash
     .venv/Scripts/Activate.ps1
     ```
   - Linux/Mac:
     ```bash
     source ../.venv/bin/activate
     ```

## Usage

1. **Run the Streamlit app:**
   ```bash
   python -m streamlit run app.py
   ```

2. **The app will:**
   - Automatically load data from `../data` folder
   - Display data overview with transaction statistics
   - Allow you to train the XGBoost model
   - Show model evaluation metrics (Precision, Recall, F1 Score)
   - Show summary results
   - Show detail results for every transaction

3. **Interact with the app:**
   - Use the slider to adjust prediction threshold (default: 0.7)
   - Click "Train XGBoost Model" to train the classifier
   - Perform predicition by uploading new data

## Model Details

- **Algorithm**: XGBoost Classifier
- **Training Data**: Timesteps 1-36
- **Test Data**: Timesteps 37-49
- **Class Imbalance Handling**: `scale_pos_weight` parameter
- **Default Threshold**: 0.7
- **Hyperparameters**:
  - n_estimators: 100
  - max_depth: 6
  - learning_rate: 0.1

## Requirements

- Python 3.8+
- streamlit>=1.31.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- matplotlib>=3.7.0
- numpy>=1.24.0
- xgboost>=2.0.0

## Data Source

The Elliptic dataset contains Bitcoin transactions with features and labels indicating illicit or licit activity. The dataset includes:
- Transaction features (166 features per transaction)
- Transaction classes (illicit, licit, unknown)
- Transaction graph edges

## Notes

- All functions are contained in a single `app.py` file
- Data is loaded from the `../data` folder relative to the app location
- The virtual environment is located in the Streamlit directory (`.venv`)
- Model training uses caching to improve performance
- The app splits data by timesteps for temporal validation