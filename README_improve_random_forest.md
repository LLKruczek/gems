# Improved Random Forest Model

Code repository: https://github.com/LLKruczek/gems/tree/improve_random_forest

## Setup

### 1. Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Training the Improved Random Forest Model

```bash
python improve_rf_model.py
```

This will:
- Train k-means preprocessing (if enabled)
- Extract features from training/test images
- Apply SMOTE for class balancing
- Run GridSearchCV for hyperparameter tuning
- Save the model as `rf_main_improved.pkl`

**Note:** Training may take 30 minutes to several hours depending on your CPU and whether hyperparameter tuning is enabled.

## Evaluating the Model

```bash
python evaluate_main_rf_improved_top3.py
```

This evaluates the model and generates:
- Top-1, Top-3, and Top-5 accuracy metrics
- Classification report
- Confusion matrix saved as `confusion_matrix_rf_improved.png`

## Running the Backend Servers

### Main Random Forest Backend (Port 5001)

```bash
cd app
python backend.py
```

Access at: `http://localhost:5001`

### Purple Stones Specialized Backend (Port 5002)

```bash
cd app
python backend_purple.py
```

Access at: `http://localhost:5002`

**Note:** The model file paths in both backend files are already configured correctly. The models are expected to be in the parent directory (`../rf_main_improved.pkl` and `../rf_purple_model.pkl`).

