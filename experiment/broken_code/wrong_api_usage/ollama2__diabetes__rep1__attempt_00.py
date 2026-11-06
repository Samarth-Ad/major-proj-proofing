
import os
import random
import time
import json
import logging
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import optuna
import shap
from scipy.stats import skew

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.experimental import enable_iterative_imputer  # Required for IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor

# Setup paths and environment
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\diabetes.csv'
MODEL_SAVE_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\diabetes_progression_model.joblib'
LOG_FILE_PATH = "pipeline_log.txt"

# Ensure directories exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def write_log(message):
    with open(LOG_FILE_PATH, "a") as f:
        f.write(f"{time.ctime()}: {message}\n")

def preprocess_data(df):
    """
    Implements the 'Glass Box' Pipeline: Schema Validation, Collinearity Check, 
    and prepares the target transformation.
    """
    write_log("Starting Data Preprocessing Pipeline")
    
    # 1. Schema Validation
    if df.shape[1] != 11:
        raise ValueError(f"Expected 11 columns, found {df.shape[1]}")
    
    target_col = df.columns[-1]
    if not (df[target_col] >= 0).all():
        raise ValueError("Target variable must be positive continuous floats.")
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # 2. Collinearity Check
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [column for column in upper.columns if any(upper[column] > 0.8)]
    if high_corr:
        msg = f"Warning: High collinearity detected in features: {high_corr}"
        logger.warning(msg)
        write_log(msg)

    # 3. Target Transformation (Log1p if skewed)
    target_skew = skew(y)
    write_log(f"Initial target skewness: {target_skew:.4f}")
    
    use_log = False
    if abs(target_skew) > 0.75:
        y = np.log1p(y)
        use_log = True
        write_log("Target skewness high. Applied log1p transformation.")
    
    return X, y, use_log

def train_model(X_train, y_train, X_val, y_val):
    """
    Implements CatBoost with Optuna Hyperparameter Tuning and Nested CV logic.
    """
    # Define the preprocessing pipeline
    # MICE Imputation -> Robust Scaling
    preprocessor = Pipeline([
        ('imputer', IterativeImputer(random_state=SEED)),
        ('scaler', RobustScaler())
    ])
    
    # Fit preprocessor on train and transform
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
            'verbose': False,
            'random_seed': SEED,
            'loss_function': 'RMSE'
        }
        
        # Inner Loop: 3-fold CV for tuning
        kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for train_idx, val_idx in kf.split(X_train_proc):
            xt, xv = X_train_proc[train_idx], X_train_proc[val_idx]
            yt, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = CatBoostRegressor(**params)
            model.fit(xt, yt, eval_set=(xv, yv), early_stopping_rounds=50, verbose=False)
            preds = model.predict(xv)
            scores.append(mean_squared_error(yv, preds, squared=False))
        
        return np.mean(scores)

    # Optuna Optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    best_params = study.best_params
    logger.info(f"Best Hyperparameters: {best_params}")
    
    # Final Model Training
    final_model = CatBoostRegressor(**best_params, verbose=False, random_seed=SEED)
    final_model.fit(X_train_proc, y_train, eval_set=(X_val_proc, y_val), early_stopping_rounds=50, verbose=False)
    
    # Bundle preprocessor and model into a single pipeline for deployment
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', final_model)
    ])
    
    return full_pipeline

def evaluate_model(model, X_test, y_test, use_log):
    """
    Evaluates the model and calculates operational metrics.
    """
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    
    # Inverse transform if log was used
    if use_log:
        y_test = np.expm1(y_test)
        y_pred = np.expm1(y_pred)
    
    # Performance Metrics
    performance_scores = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }
    
    # Complexity Metrics
    latency = (end_time - start_time) * 1000 / len(X_test)
    joblib.dump(model, "temp_eval.joblib")
    model_size = os.path.getsize("temp_eval.joblib") / 1024 # KB
    os.remove("temp_eval.joblib")
    
    complexity_scores = {
        "Inference Latency (ms)": latency,
        "Model Size (KB)": model_size
    }
    
    return performance_scores, complexity_scores

def prepare_model_for_deployment(model):
    """
    Serializes the full pipeline.
    """
    joblib.dump(model, MODEL_SAVE_PATH)
    write_log(f"Model serialized to {MODEL_SAVE_PATH}")
    return model

def deploy_model(model, feature_names, use_log):
    """
    Builds a Gradio interface for clinical prediction.
    """
    def predict(inputs):
        # Convert inputs to DataFrame
        df_input = pd.DataFrame([inputs], columns=feature_names)
        pred = model.predict(df_input)[0]
        if use_log:
            pred = np.expm1(pred)
        return float(round(pred, 4))

    interface = gr.Interface(
        fn=predict,
        inputs=[gr.Number(label=col) for col in feature_names],
        outputs=gr.Textbox(label="Predicted Diabetes Progression"),
        title="Diabetes Progression Predictor",
        description="Clinical tool to predict disease progression based on patient metrics."
    )
    return interface

def main():
    # Step 1: Load Dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    feature_names = df.columns[:-1].tolist()
    
    # Step 2: Data Split (70% Train, 20% Val, 10% Test)
    train_df, test_df = train_test_split(df, test_size=0.10, random_state=SEED)
    train_df, val_df = train_test_split(train_df, test_size=0.2222, random_state=SEED) # 0.2222 of 90% is ~20% of total
    
    # Step 3: Preprocessing
    X_train, y_train, use_log = preprocess_data(train_df)
    X_val, y_val, _ = preprocess_data(val_df)
    X_test, y_test, _ = preprocess_data(test_df)
    
    # Step 4 & 5: Model Training and Optimization
    model_pipeline = train_model(X_train, y_train, X_val, y_val)
    
    # Step 6: Evaluation
    perf, comp = evaluate_model(model_pipeline, X_test, y_test, use_log)
    
    # Step 7: Deployment Preparation
    deployable_model = prepare_model_for_deployment(model_pipeline)
    
    # Step 8: Gradio Deployment
    app = deploy_model(deployable_model, feature_names, use_log)
    
    # XAI: SHAP Analysis (on a sample of test data)
    # We extract the regressor and the processed data for SHAP
    preprocessor = model_pipeline.named_steps['preprocessor']
    regressor = model_pipeline.named_steps['regressor']
    X_test_proc = preprocessor.transform(X_test)
    
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_test_proc)
    write_log("SHAP values calculated for clinical interpretability.")
    
    print("\n--- Final Model Performance ---")
    for k, v in perf.items(): print(f"{k}: {v:.4f}")
    print("\n--- Model Complexity ---")
    for k, v in comp.items(): print(f"{k}: {v:.4f}")
    
    # Launch Gradio
    app.launch(share=True)

if __name__ == "__main__":
    main()
