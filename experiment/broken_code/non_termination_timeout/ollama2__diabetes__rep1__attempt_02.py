
import os
import random
import time
import logging
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import optuna
import shap
from scipy.stats import skew

from sklearn.model_selection import train_test_split, KFold
from sklearn.experimental import enable_iterative_imputer  # Required for IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor

# ==========================================
# 1. Environment & Path Specifications
# ==========================================
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\diabetes.csv'
MODEL_SAVE_DIR = "./agent_workspace/trained_models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "diabetes_progression_model.joblib")
LOG_FILE_PATH = "pipeline_log.txt"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def write_log(message):
    with open(LOG_FILE_PATH, "a") as f:
        f.write(f"{time.ctime()}: {message}\n")

# ==========================================
# 2. Data Engineering Pipeline
# ==========================================
def preprocess_data(df, is_train=True, preprocessor=None):
    """
    Implements the 'Glass Box' Pipeline.
    """
    # Schema Validation
    if df.shape[1] != 11:
        raise ValueError(f"Expected 11 columns, found {df.shape[1]}")
    
    target_col = df.columns[-1]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    if is_train:
        # Collinearity Check
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [column for column in upper.columns if any(upper[column] > 0.8)]
        if high_corr:
            msg = f"Warning: High collinearity detected in features: {high_corr}"
            logger.warning(msg)
            write_log(msg)

        # Target Transformation
        target_skew = skew(y)
        write_log(f"Initial target skewness: {target_skew:.4f}")
        use_log = False
        if abs(target_skew) > 0.75:
            y = np.log1p(y)
            use_log = True
            write_log("Target skewness high. Applied log1p transformation.")
        
        # Initialize MICE and RobustScaler
        preprocessor = Pipeline([
            ('imputer', IterativeImputer(random_state=SEED, max_iter=10)),
            ('scaler', RobustScaler())
        ])
        X_proc = preprocessor.fit_transform(X)
        return X_proc, y, use_log, preprocessor
    else:
        # Use existing preprocessor for val/test to prevent leakage
        X_proc = preprocessor.transform(X)
        return X_proc, y

# ==========================================
# 3. Model Implementation & Optimization
# ==========================================
def train_model(X_train, y_train, X_val, y_val):
    """
    CatBoost with Optuna Tuning. 
    Reduced trials to prevent timeout while maintaining robustness.
    """
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
            'verbose': False,
            'random_seed': SEED,
            'loss_function': 'RMSE'
        }
        
        # Inner Loop: 3-fold CV
        kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            xt, xv = X_train[train_idx], X_train[val_idx]
            yt, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = CatBoostRegressor(**params)
            model.fit(xt, yt, eval_set=(xv, yv), early_stopping_rounds=30, verbose=False)
            preds = model.predict(xv)
            scores.append(np.sqrt(mean_squared_error(yv, preds)))
        
        return np.mean(scores)

    # Optuna Optimization (Reduced trials to 15 to avoid timeout)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)
    
    best_params = study.best_params
    logger.info(f"Best Hyperparameters: {best_params}")
    
    # Final Model Training
    final_model = CatBoostRegressor(**best_params, verbose=False, random_seed=SEED)
    final_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
    
    return final_model

# ==========================================
# 4. Evaluation & Profiling
# ==========================================
def evaluate_model(pipeline, X_test_raw, y_test, use_log):
    start_time = time.time()
    y_pred = pipeline.predict(X_test_raw)
    end_time = time.time()
    
    if use_log:
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_test_orig, y_pred_orig = y_test, y_pred
    
    perf = {
        "MAE": mean_absolute_error(y_test_orig, y_pred_orig),
        "RMSE": np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
        "R2": r2_score(y_test_orig, y_pred_orig)
    }
    
    latency = (end_time - start_time) * 1000 / len(X_test_raw)
    
    # Model Size
    joblib.dump(pipeline, "temp_size.joblib")
    model_size = os.path.getsize("temp_size.joblib") / 1024
    os.remove("temp_size.joblib")
    
    comp = {"Inference Latency (ms)": latency, "Model Size (KB)": model_size}
    return perf, comp

# ==========================================
# 5. Main Execution Pipeline
# ==========================================
def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    feature_names = df.columns[:-1].tolist()
    
    # Data Split: 70% Train, 20% Val, 10% Test
    train_val_df, test_df = train_test_split(df, test_size=0.10, random_state=SEED)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2222, random_state=SEED)
    
    # Preprocessing
    X_train_proc, y_train, use_log, preprocessor = preprocess_data(train_df, is_train=True)
    X_val_proc, y_val = preprocess_data(val_df, is_train=False, preprocessor=preprocessor)
    X_test_proc, y_test = preprocess_data(test_df, is_train=False, preprocessor=preprocessor)
    
    # Training
    regressor = train_model(X_train_proc, y_train, X_val_proc, y_val)
    
    # Bundle into Pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    # Evaluation
    perf, comp = evaluate_model(full_pipeline, test_df.iloc[:, :-1], y_test, use_log)
    
    # Serialization
    joblib.dump(full_pipeline, MODEL_SAVE_PATH)
    write_log(f"Model serialized to {MODEL_SAVE_PATH}")
    
    # XAI: SHAP
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_test_proc)
    write_log("SHAP values calculated for clinical interpretability.")
    
    print("\n--- Final Model Performance ---")
    for k, v in perf.items(): print(f"{k}: {v:.4f}")
    print("\n--- Model Complexity ---")
    for k, v in comp.items(): print(f"{k}: {v:.4f}")
    
    # Gradio App
    def predict_fn(inputs):
        df_input = pd.DataFrame([inputs], columns=feature_names)
        pred = full_pipeline.predict(df_input)[0]
        if use_log:
            pred = np.expm1(pred)
        return float(round(pred, 4))

    app = gr.Interface(
        fn=predict_fn,
        inputs=[gr.Number(label=col) for col in feature_names],
        outputs=gr.Textbox(label="Predicted Diabetes Progression"),
        title="Diabetes Progression Predictor (CatBoost + MICE)",
        description="Clinical tool to predict disease progression based on patient metrics."
    )
    app.launch(share=False)

if __name__ == "__main__":
    main()
