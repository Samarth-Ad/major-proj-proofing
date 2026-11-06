
import os
import random
import time
import json
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. Environment & Path Setup
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths as per instructions
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\california.csv'
# Saving to the requested directory
MODEL_STORAGE_PATH = './agent_workspace/trained_models' 
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

def preprocess_data(df):
    """
    Implements the robust preprocessing pipeline:
    Imputation -> Scaling -> Target Log Transformation -> Stratified Splitting
    """
    # Schema Validation: 8 numeric features + 1 target = 9 columns
    if df.shape[1] != 9:
        print(f"Warning: Expected 9 columns, found {df.shape[1]}")

    # Separate features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Target Transformation: log(y + 1) to stabilize variance
    y_log = np.log1p(y)

    # Imputation: Median strategy for robustness
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # Scaling: RobustScaler to handle real estate outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Stratified Splitting via Binning (Deciles)
    # We create 10 bins to ensure the price distribution is consistent across sets
    y_binned = pd.qcut(y_log, q=10, labels=False, duplicates='drop')

    # Split: 70% Train, 30% Temp (Valid + Test)
    X_train, X_temp, y_train, y_temp, binned_train, binned_temp = train_test_split(
        X_scaled, y_log, y_binned, test_size=0.3, stratify=y_binned, random_state=SEED
    )

    # Split Temp: 2/3 Valid (20% total), 1/3 Test (10% total)
    # Re-binning temp set for stratification
    y_temp_binned = pd.qcut(y_temp, q=10, labels=False, duplicates='drop')
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.333, stratify=y_temp_binned, random_state=SEED
    )

    # Save preprocessing objects for inference
    preprocessing_pipeline = {'imputer': imputer, 'scaler': scaler}
    joblib.dump(preprocessing_pipeline, os.path.join(MODEL_STORAGE_PATH, 'preprocessing_pipeline.joblib'))

    return (X_train, X_val, X_test), (y_train, y_val, y_test), preprocessing_pipeline

def optimize_and_train(X_train, y_train, X_val, y_val):
    """
    Hyperparameter Optimization using Optuna and Final Training with CatBoost.
    Optimized to prevent timeouts by reducing trial count and search iterations.
    """
    def objective(trial):
        params = {
            'iterations': 200, # Reduced for HPO speed to avoid timeout
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': SEED,
            'verbose': False,
            'early_stopping_rounds': 25
        }
        
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    # Optimization - limited trials to ensure execution within 300s
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5) # Reduced trials to ensure stability
    
    # Final Training with best parameters and full iterations
    best_params = study.best_params
    best_params.update({
        'iterations': 1000,
        'loss_function': 'RMSE',
        'random_seed': SEED,
        'verbose': False,
        'early_stopping_rounds': 50
    })

    final_model = CatBoostRegressor(**best_params)
    final_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    
    # Save final model
    joblib.dump(final_model, os.path.join(MODEL_STORAGE_PATH, 'catboost_housing_model.joblib'))
    
    return final_model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluates model on log-transformed target and profiles inference performance.
    """
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    performance_scores = {
        "Train_RMSE": np.sqrt(mean_squared_error(y_train, train_preds)),
        "Test_RMSE": np.sqrt(mean_squared_error(y_test, test_preds)),
        "MAE": mean_absolute_error(y_test, test_preds),
        "R2": r2_score(y_test, test_preds)
    }

    # Profiling: Latency
    start_time = time.time()
    for i in range(100):
        _ = model.predict(X_test[0:1])
    latency = ((time.time() - start_time) / 100) * 1000 

    # Profiling: Model Size
    model_path = os.path.join(MODEL_STORAGE_PATH, 'catboost_housing_model.joblib')
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    
    complexity_scores = {
        "Inference_Latency_ms": latency,
        "Model_Size_MB": model_size,
        "Generalization_Gap": performance_scores["Test_RMSE"] - performance_scores["Train_RMSE"]
    }

    return performance_scores, complexity_scores

def inference_wrapper(input_data):
    """
    Deployment wrapper: Preprocess -> Predict -> Inverse Log Transform
    """
    pipeline = joblib.load(os.path.join(MODEL_STORAGE_PATH, 'preprocessing_pipeline.joblib'))
    model = joblib.load(os.path.join(MODEL_STORAGE_PATH, 'catboost_housing_model.joblib'))
    
    # Preprocess
    data = np.array(input_data).reshape(1, -1)
    data_imputed = pipeline['imputer'].transform(data)
    data_scaled = pipeline['scaler'].transform(data_imputed)
    
    # Predict (Log scale)
    log_pred = model.predict(data_scaled)[0]
    
    # Inverse transform: exp(x) - 1
    final_pred = np.expm1(log_pred)
    return float(final_pred)

def deploy_model():
    """
    Builds a Gradio web application demo.
    """
    def wrapper_fn(text):
        try:
            features = [float(x.strip()) for x in text.split(',')]
            if len(features) != 8:
                return "Error: Please provide exactly 8 numeric features."
            return inference_wrapper(features)
        except Exception as e:
            return f"Error: {str(e)}"

    interface = gr.Interface(
        fn=wrapper_fn,
        inputs=gr.Textbox(label="Input Features (Comma separated)", placeholder="1.2, 3.4, 5.6, 7.8, 9.0, 1.1, 2.2, 3.3"),
        outputs=gr.Number(label="Predicted House Price"),
        title="Housing Price Predictor (CatBoost)",
        description="Enter the 8 numeric features to predict the house price."
    )
    return interface.launch(share=False, inline=False)

def main():
    # 1. Load Dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    
    # 2. Preprocess and Split
    (X_train, X_val, X_test), (y_train, y_val, y_test), _ = preprocess_data(df)
    
    # 3. Optimize and Train
    model = optimize_and_train(X_train, y_train, X_val, y_val)
    
    # 4. Evaluate
    performance, complexity = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # 5. Deploy
    deploy_model()
    
    return performance, complexity

if __name__ == "__main__":
    try:
        perf, comp = main()
        print("\n--- Model Performance (Log Scale) ---")
        print(json.dumps(perf, indent=4))
        print("\n--- Model Complexity & Profiling ---")
        print(json.dumps(comp, indent=4))
        print(f"\nDeployment Interface launched. Check Gradio logs for URL.")
    except Exception as e:
        print(f"Pipeline failed: {e}")
