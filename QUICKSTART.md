# Quick Start Guide

## Running the XGBoost Streamlit App

### Step 1: Activate Virtual Environment

Open a terminal in the Elliptic folder and run:

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### Step 2: Navigate to Streamlit Folder

```bash
cd streamlit
```

### Step 3: Run the App

```bash
streamlit run app.py
```

The app will automatically:
- Open in your default web browser
- Load data from the `../data` folder
- Be ready for model training

### Step 4: Use the App

1. **View Data Overview**: See transaction statistics automatically
2. **Train Model**: Click "Train XGBoost Model" button
3. **Adjust Threshold**: Use the sidebar slider to change prediction threshold (0.5 - 0.95)
4. **View Metrics**: See Precision, Recall, and F1 Score
5. **Analyze Timesteps**: Click "Generate Timestep Analysis" for detailed temporal analysis

## Troubleshooting

### Data Files Not Found
Make sure the data files are in the correct location:
```
Elliptic/
├── data/
│   ├── elliptic_txs_classes.csv
│   ├── elliptic_txs_edgelist.csv
│   └── elliptic_txs_features.csv
└── streamlit/
    └── app.py
```

### Module Not Found
If you get import errors, reinstall dependencies:
```bash
pip install -r requirements.txt
```

### Virtual Environment Not Activated
Make sure you see `(.venv)` at the beginning of your terminal prompt.

## Features at a Glance

- ✅ All functions in one file (app.py)
- ✅ Data loaded from ../data folder
- ✅ Virtual environment created and configured
- ✅ Requirements file with all dependencies
- ✅ XGBoost model with class imbalance handling
- ✅ Interactive threshold adjustment
- ✅ Temporal analysis by timestep
- ✅ Visualization of F1 scores

## Default Settings

- **Prediction Threshold**: 0.7
- **Training Timesteps**: 1-36
- **Testing Timesteps**: 37-49
- **Model**: XGBoost with scale_pos_weight for class imbalance
