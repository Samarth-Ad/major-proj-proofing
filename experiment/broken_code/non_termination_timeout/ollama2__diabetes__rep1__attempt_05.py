
import os
import random
import time
import json
import numpy as np
import pandas as pd
import joblib
import shap
import optuna
from scipy.stats import skew
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.experimental import enable_iterative_imputer  # Required for IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor
import gradio as gr

# --- Environment & Path Specifications ---
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\diabetes.csv'
MODEL_SAVE_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\diabetes_progression_model.joblib'
LOG_FILE = 'pipeline_log.txt'
SEED = 42

# Ensure directories exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)

def log_message(message):
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

def preprocess_data(df):
    """
    Implements the 'Glass Box' Pipeline: 
    Schema Validation -> MICE Imputation -> Robust Scaling -> Target Transformation
    """
    log_message("Starting Data Preprocessing Pipeline...")
    
    # 1. Schema Validation
    if df.shape[1] != 11:
        raise ValueError(f"Expected 11 columns, found {df.shape[1]}")
    
    target_col = df.columns[-1]
    if not (df[target_col] >= 0).all():
        raise ValueError("Target column contains negative values.")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 2. Collinearity Check
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [f"{column}" for column in upper.columns if any(upper[column] > 0.8)]
    if high_corr:
        log_message(f"High collinearity detected in features: {high_corr}")

    # 3. Target Transformation (Log Transform if skewed)
    use_log = False
    target_skew = skew(y)
    if abs(target_skew) > 0.75:
        use_log = True
        y = np.log1p(y)
        log_message(f"Target skew {target_skew:.2f} > 0.75. Applied log1p transformation.")
    else:
        log_message(f"Target skew {target_skew:.2f} is acceptable. No transformation applied.")

    # 4. Build Preprocessing Pipeline (Imputer + Scaler)
    # We return the pipeline object to avoid data leakage during CV
    preprocessor = Pipeline([
        ('imputer', IterativeImputer(random_state=SEED, max_iter=10)),
        ('scaler', RobustScaler())
    ])

    return X, y, preprocessor, use_log

def optimize_hyperparameters(X_train, y_train, preprocessor):
    """
    Optuna optimization with 3-fold Inner CV.
    """
    log_message("Starting Hyperparameter Optimization with Optuna...")
    
    # Pre-transform X_train for the tuning phase to speed up trials
    X_transformed = preprocessor.fit_transform(X_train)
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'depth': trial.suggest_int('depth', 4, 6),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
            'random_seed': SEED,
            'verbose': False,
            'allow_writing_files': False
        }
        
        model = CatBoostRegressor(**params)
        # 3-fold CV for tuning
        score = cross_val_score(model, X_transformed, y_train, cv=3, scoring='neg_root_mean_squared_error')
        return -score.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15) # Efficiency constraint: 15 trials
    
    log_message(f"Best Hyperparameters: {study.best_params}")
    return study.best_params

def train_final_model(X_train, y_train, preprocessor, best_params):
    """
    Trains the final CatBoost model within a Pipeline.
    """
    # Create the final pipeline: Preprocessor -> Regressor
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', CatBoostRegressor(**best_params, random_seed=SEED, verbose=False))
    ])
    
    full_pipeline.fit(X_train, y_train)
    return full_pipeline

def evaluate_model(pipeline, X_train, y_train, X_test, y_test, use_log):
    """
    Evaluates model using RMSE and R2, and calculates operational metrics.
    """
    # Predictions
    y_pred_raw = pipeline.predict(X_test)
    y_train_raw = pipeline.predict(X_train)
    
    # Inverse transform if log was used
    if use_log:
        y_pred = np.expm1(y_pred_raw)
        y_test_inv = np.expm1(y_test)
        y_train_inv = np.expm1(y_train)
    else:
        y_pred = y_pred_raw
        y_test_inv = y_test
        y_train_inv = y_train

    # Performance Metrics
    rmse_test = root_mean_squared_error(y_test_inv, y_pred)
    rmse_train = root_mean_squared_error(y_train_inv, y_train_raw if not use_log else np.expm1(y_train_raw))
    r2 = r2_score(y_test_inv, y_pred)
    mae = mean_absolute_error(y_test_inv, y_pred)

    # Operational Metrics
    start_time = time.time()
    pipeline.predict(X_test[:1])
    latency = (time.time() - start_time) * 1000 # ms
    
    model_size = os.path.getsize(MODEL_SAVE_PATH) / 1024 if os.path.exists(MODEL_SAVE_PATH) else 0
    gen_gap = rmse_test - rmse_train

    performance_scores = {"RMSE": rmse_test, "R2": r2, "MAE": mae}
    complexity_scores = {"Latency_ms": latency, "Model_Size_KB": model_size, "Gen_Gap": gen_gap}
    
    return performance_scores, complexity_scores

def prepare_model_for_deployment(pipeline, use_log):
    """
    Serializes the pipeline and metadata.
    """
    model_data = {
        'pipeline': pipeline,
        'use_log': use_log
    }
    joblib.dump(model_data, MODEL_SAVE_PATH)
    log_message(f"Model serialized to {MODEL_SAVE_PATH}")
    return MODEL_SAVE_PATH

def deploy_model():
    """
    Gradio Web Application Demo.
    """
    model_data = joblib.load(MODEL_SAVE_PATH)
    pipeline = model_data['pipeline']
    use_log = model_data['use_log']
    
    # Get feature names from the training data (assuming 10 features)
    feature_names = [f"Feature {i+1}" for i in range(10)]
    
    def predict_progression(inputs):
        # Convert inputs to DataFrame
        df_input = pd.DataFrame([inputs], columns=feature_names)
        pred = pipeline.predict(df_input)[0]
        if use_log:
            pred = np.expm1(pred)
        return float(pred)

    interface = gr.Interface(
        fn=predict_progression,
        inputs=[gr.Number(label=name) for name in feature_names],
        outputs=gr.Number(label="Predicted Diabetes Progression"),
        title="Diabetes Progression Predictor",
        description="Enter medical metrics to predict the disease progression score."
    )
    return interface.launch(share=True)

def main():
    # 1. Load Dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    
    # 2. Data Split (70% Train, 20% Val, 10% Test)
    # First split: 70% train, 30% temp
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=SEED)
    # Second split: 2/3 of 30% = 20% val, 1/3 of 30% = 10% test
    val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=SEED)
    
    # 3. Preprocessing
    X_train, y_train, preprocessor, use_log = preprocess_data(train_df)
    X_val, y_val = test_df.iloc[:, :-1], test_df.iloc[:, -1] # Using test_df as evaluation set
    X_test, y_test = val_df.iloc[:, :-1], val_df.iloc[:, -1] # Adjusted to match requested split logic
    
    # 4. Hyperparameter Tuning
    best_params = optimize_hyperparameters(X_train, y_train, preprocessor)
    
    # 5. Train Final Model
    model_pipeline = train_final_model(X_train, y_train, preprocessor, best_params)
    
    # 6. Evaluation
    # Using the 10% test split for final validation
    perf, comp = evaluate_model(model_pipeline, X_train, y_train, X_test, y_test, use_log)
    
    # 7. Deployment Preparation
    deployable_model_path = prepare_model_for_deployment(model_pipeline, use_log)
    
    # 8. Deployment
    url_endpoint = deploy_model()
    
    return (df, model_pipeline, deployable_model_path, url_endpoint, perf, comp)

if __name__ == "__main__":
    # Clear log file
    with open(LOG_FILE, 'w') as f:
        f.write("--- Pipeline Execution Log ---\n")
        
    try:
        (
            processed_data,
            model,
            deployable_model,
            url_endpoint,
            model_performance,
            model_complexity,
        ) = main()
        print("\n" + "="*30)
        print("Model Performance on Test Set:", model_performance)
        print("Model Complexity:", model_complexity)
        print("="*30)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        log_message(f"CRITICAL ERROR: {str(e)}")
