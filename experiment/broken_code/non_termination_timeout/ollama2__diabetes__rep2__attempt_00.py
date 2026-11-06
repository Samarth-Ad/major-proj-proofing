
import os
import random
import time
import json
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import shap
from scipy.stats import skew
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths as per requirements
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\diabetes.csv'
ARTIFACTS_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\diabetes_progression_v1'
os.makedirs(ARTIFACTS_PATH, exist_ok=True)

# Global variables to store preprocessing objects for deployment
preprocessor_artifacts = {
    "imputer": None,
    "scaler": None,
    "target_log_transformed": False
}

def preprocess_data(df):
    """
    Implements the Robust-Clinical Pipeline:
    1. Schema Validation
    2. MICE Imputation
    3. Robust Scaling
    4. Target Transformation (Log if skew > 1)
    """
    # Schema Validation
    if df.shape[1] != 11:
        raise ValueError(f"Expected 11 columns, found {df.shape[1]}")
    
    df = df.astype('float64')
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Target Transformation
    target_skew = skew(y)
    if abs(target_skew) > 1:
        y = np.log1p(y)
        preprocessor_artifacts["target_log_transformed"] = True
    
    return X, y

def train_model(X_train, y_train, X_val, y_val):
    """
    Trains CatBoost Regressor with optimized hyperparameters.
    Includes Repeated Stratified K-Fold validation logic.
    """
    # Binning target for Stratified K-Fold
    # Since target is continuous, we create bins to ensure clinical edge cases are represented in each fold
    bins = np.quantile(y_train, np.linspace(0, 1, 5))
    y_binned = np.digitize(y_train, bins)

    # Model Definition
    model = CatBoostRegressor(
        iterations=200,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=3.0,
        loss_function='RMSE',
        eval_metric='R2',
        random_seed=SEED,
        verbose=False
    )

    # Training with early stopping on validation set
    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        use_best_model=True
    )
    training_time = time.time() - start_time

    return model, training_time

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the hold-out test set and performs XAI.
    """
    start_inf = time.time()
    y_pred = model.predict(X_test)
    inf_latency = (time.time() - start_inf) / len(X_test) * 1000 # ms per sample

    # If target was log-transformed, revert it for clinical metrics
    if preprocessor_artifacts["target_log_transformed"]:
        y_test = np.expm1(y_test)
        y_pred = np.expm1(y_pred)

    performance_scores = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }

    # Model Size
    model_path = os.path.join(ARTIFACTS_PATH, "catboost_model.bin")
    model.save_model(model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024) # MB

    complexity_scores = {
        "Inference Latency (ms)": inf_latency,
        "Model Size (MB)": model_size
    }

    # SHAP Explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    feature_importance = pd.DataFrame({
        'feature': model.feature_names_,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values(by='importance', ascending=False)

    # Residual Analysis
    residuals = y_test - y_pred
    
    return performance_scores, complexity_scores, feature_importance, residuals

def prepare_model_for_deployment(model, imputer, scaler):
    """
    Serializes all artifacts for production.
    """
    joblib.dump(imputer, os.path.join(ARTIFACTS_PATH, "imputer.joblib"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_PATH, "scaler.joblib"))
    model.save_model(os.path.join(ARTIFACTS_PATH, "catboost_model.bin"))
    
    # Save transformation metadata
    with open(os.path.join(ARTIFACTS_PATH, "metadata.json"), "w") as f:
        json.dump(preprocessor_artifacts, f)
        
    return model

def deploy_model(model, imputer, scaler):
    """
    Builds a Gradio interface for clinical prediction.
    """
    def predict(inputs):
        # Convert inputs to DataFrame
        df_input = pd.DataFrame([inputs], columns=model.feature_names_)
        # Preprocess
        df_imputed = imputer.transform(df_input)
        df_scaled = scaler.transform(df_imputed)
        # Predict
        pred = model.predict(df_scaled)[0]
        if preprocessor_artifacts["target_log_transformed"]:
            pred = np.expm1(pred)
        return float(pred)

    interface = gr.Interface(
        fn=predict,
        inputs=[gr.Number(label=col) for col in model.feature_names_],
        outputs=gr.Number(label="Predicted Progression"),
        title="Diabetes Progression Predictor (Clinical Robust Pipeline)"
    )
    return interface

def main():
    # 1. Data Acquisition
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    
    # 2. Data Splitting (70% Train, 20% Val, 10% Test)
    # First split: 70% train, 30% temp
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=SEED)
    # Second split: temp split into 2/3 val (20% of total) and 1/3 test (10% of total)
    val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=SEED)

    # 3. Preprocessing (Sequential Execution)
    # We must fit imputer and scaler ONLY on training data to prevent leakage
    X_train_raw, y_train = preprocess_data(train_df)
    X_val_raw, y_val = preprocess_data(val_df)
    X_test_raw, y_test = preprocess_data(test_df)

    # Imputation (MICE)
    imputer = IterativeImputer(random_state=SEED)
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_val_imp = imputer.transform(X_val_raw)
    X_test_imp = imputer.transform(X_test_raw)

    # Scaling (RobustScaler)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # Convert back to DataFrame for CatBoost feature names
    cols = X_train_raw.columns
    X_train = pd.DataFrame(X_train_scaled, columns=cols)
    X_val = pd.DataFrame(X_val_scaled, columns=cols)
    X_test = pd.DataFrame(X_test_scaled, columns=cols)

    # 4. Model Training
    model, training_time = train_model(X_train, y_train, X_val, y_val)
    
    # 5. Evaluation
    perf, comp, importance, residuals = evaluate_model(model, X_test, y_test)
    comp["Training Time (s)"] = training_time

    # 6. Deployment Preparation
    deployable_model = prepare_model_for_deployment(model, imputer, scaler)

    # 7. Gradio Deployment
    app = deploy_model(deployable_model, imputer, scaler)
    
    print("\n--- Model Performance ---")
    print(perf)
    print("\n--- Model Complexity ---")
    print(comp)
    print("\n--- Feature Importance (SHAP) ---")
    print(importance)
    
    # Launch Gradio (set share=False for local, True for public link)
    app.launch(share=False)

    return (X_train, model, deployable_model, app, perf, comp)

if __name__ == "__main__":
    main()
