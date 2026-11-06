
import os
import random
import time
import json
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Environment Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths as per instructions
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\california.csv'
MODEL_STORAGE_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models'
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

def preprocess_data(df):
    """
    Implements the Second Data Scientist's Proposal:
    Imputation -> Scaling -> Target Log Transformation -> Stratified Splitting
    """
    # 1. Schema Validation
    if df.shape[1] != 9:
        print(f"Warning: Expected 9 columns, found {df.shape[1]}")

    # Separate features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 2. Target Transformation: log(y + 1)
    y_log = np.log1p(y)

    # 3. Imputation
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # 4. Scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 5. Stratified Splitting
    # Binning target into deciles for stratification
    bins = np.linspace(y_log.min(), y_log.max(), 11)
    y_binned = np.digitize(y_log, bins)

    # Split: 70% Train, 30% Temp (Valid + Test)
    X_train, X_temp, y_train, y_temp, binned_train, binned_temp = train_test_split(
        X_scaled, y_log, y_binned, test_size=0.3, stratify=y_binned, random_state=SEED
    )

    # Split Temp: 2/3 Valid (20% total), 1/3 Test (10% total)
    # We need to stratify again based on the temp bins
    # Recalculate bins for temp set to ensure consistency
    temp_bins = np.linspace(y_temp.min(), y_temp.max(), 11)
    y_temp_binned = np.digitize(y_temp, temp_bins)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.333, stratify=y_temp_binned, random_state=SEED
    )

    # Save preprocessing objects
    preprocessing_pipeline = {'imputer': imputer, 'scaler': scaler}
    joblib.dump(preprocessing_pipeline, os.path.join(MODEL_STORAGE_PATH, 'preprocessing_pipeline.joblib'))

    return (X_train, X_val, X_test), (y_train, y_val, y_test), preprocessing_pipeline

def optimize_and_train(X_train, y_train, X_val, y_val):
    """
    Hyperparameter Optimization using Optuna and Training with CatBoost
    """
    def objective(trial):
        params = {
            'iterations': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': SEED,
            'verbose': False,
            'early_stopping_rounds': 50
        }
        
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    # Train final model with best params
    best_params = study.best_params
    best_params['iterations'] = 1000
    best_params['loss_function'] = 'RMSE'
    best_params['random_seed'] = SEED
    best_params['verbose'] = False
    best_params['early_stopping_rounds'] = 50

    final_model = CatBoostRegressor(**best_params)
    final_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    
    # Save model
    joblib.dump(final_model, os.path.join(MODEL_STORAGE_PATH, 'catboost_housing_model.joblib'))
    
    return final_model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluates the model on log-transformed target and calculates complexity.
    """
    # Predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Performance Metrics (on log scale)
    performance_scores = {
        "Train_RMSE": np.sqrt(mean_squared_error(y_train, train_preds)),
        "Test_RMSE": np.sqrt(mean_squared_error(y_test, test_preds)),
        "MAE": mean_absolute_error(y_test, test_preds),
        "R2": r2_score(y_test, test_preds)
    }

    # Complexity Metrics
    start_time = time.time()
    for i in range(100):
        _ = model.predict(X_test[0:1])
    latency = ((time.time() - start_time) / 100) * 1000 # ms per sample

    model_size = os.path.getsize(os.path.join(MODEL_STORAGE_PATH, 'catboost_housing_model.joblib')) / (1024 * 1024)
    
    complexity_scores = {
        "Inference_Latency_ms": latency,
        "Model_Size_MB": model_size,
        "Generalization_Gap": performance_scores["Test_RMSE"] - performance_scores["Train_RMSE"]
    }

    return performance_scores, complexity_scores

def inference_wrapper(input_data):
    """
    Wrapper for deployment: Preprocess -> Predict -> Inverse Log Transform
    """
    pipeline = joblib.load(os.path.join(MODEL_STORAGE_PATH, 'preprocessing_pipeline.joblib'))
    model = joblib.load(os.path.join(MODEL_STORAGE_PATH, 'catboost_housing_model.joblib'))
    
    # Preprocess
    data = np.array(input_data).reshape(1, -1)
    data_imputed = pipeline['imputer'].transform(data)
    data_scaled = pipeline['scaler'].transform(data_imputed)
    
    # Predict
    log_pred = model.predict(data_scaled)[0]
    
    # Inverse transform: exp(x) - 1
    final_pred = np.expm1(log_pred)
    return float(final_pred)

def deploy_model():
    """
    Builds a Gradio web application demo.
    """
    interface = gr.Interface(
        fn=inference_wrapper,
        inputs=gr.Textbox(label="Input Features (Comma separated)", placeholder="1.2, 3.4, ..."),
        outputs=gr.Number(label="Predicted House Price"),
        title="Housing Price Predictor (CatBoost)",
        description="Enter the 8 numeric features to predict the house price."
    )
    
    # To make it work with the Textbox input, we need a small helper
    def wrapper_fn(text):
        features = [float(x) for x in text.split(',')]
        return inference_wrapper(features)

    interface.fn = wrapper_fn
    return interface.launch(share=True)

def main():
    # 1. Load Dataset
    df = pd.read_csv(DATASET_PATH)
    
    # 2. Preprocess and Split
    (X_train, X_val, X_test), (y_train, y_val, y_test), _ = preprocess_data(df)
    
    # 3. Optimize and Train
    model = optimize_and_train(X_train, y_train, X_val, y_val)
    
    # 4. Evaluate
    performance, complexity = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # 5. Deploy
    url_endpoint = deploy_model()
    
    return performance, complexity, url_endpoint

if __name__ == "__main__":
    perf, comp, url = main()
    print("\n--- Model Performance (Log Scale) ---")
    print(json.dumps(perf, indent=4))
    print("\n--- Model Complexity & Profiling ---")
    print(json.dumps(comp, indent=4))
    print(f"\nDeployment URL: {url}")
