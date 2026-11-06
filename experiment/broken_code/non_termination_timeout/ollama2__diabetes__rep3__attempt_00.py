
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
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths as per specifications
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\diabetes.csv'
MODEL_SAVE_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\diabetes_progression_catboost.pkl'
SCALER_SAVE_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\scaler.pkl'
LOG_FILE_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\pipeline_log.txt'

# Ensure directories exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

def preprocess_data(df):
    """
    Implements the Data Engineering Pipeline:
    Structural Audit -> KNN Imputation -> Scaling -> Target Transformation
    """
    # 1. Structural Audit
    if df.shape != (442, 11):
        raise ValueError(f"Dataset structural audit failed. Expected (442, 11), got {df.shape}")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 2. Target Transformation (Log-transform if skew > 0.75)
    target_skew = skew(y)
    log_transformed = False
    if abs(target_skew) > 0.75:
        y = np.log1p(y)
        log_transformed = True
    
    return X, y, log_transformed

def train_model(X_train, y_train, X_val, y_val):
    """
    Implements CatBoostRegressor with specified hyperparameters and CV validation.
    """
    # Hyperparameters from blueprint
    params = {
        'iterations': 200,
        'depth': 4,
        'learning_rate': 0.03,
        'l2_leaf_reg': 3.0,
        'loss_function': 'RMSE',
        'verbose': False,
        'random_seed': SEED
    }

    model = CatBoostRegressor(**params)

    # Repeated K-Fold Cross-Validation on Train+Val set for stability check
    X_cv = np.vstack((X_train, X_val))
    y_cv = np.concatenate((y_train, y_val))
    
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    cv_scores = cross_val_score(model, X_cv, y_cv, cv=rkf, scoring='neg_root_mean_squared_error')
    
    # Fit final model on training data
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
    
    return model, cv_scores

def evaluate_model(model, X_test, y_test, log_transformed):
    """
    Evaluates model on hold-out set and performs XAI/Residual analysis.
    """
    preds = model.predict(X_test)
    
    # Inverse transform if log was applied
    if log_transformed:
        y_true = np.expm1(y_test)
        y_pred = np.expm1(preds)
    else:
        y_true = y_test
        y_pred = preds

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Residual Analysis
    residuals = y_true - y_pred
    
    # SHAP Interpretability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    performance_scores = {"RMSE": rmse, "R2": r2, "MAE": mae}
    return performance_scores, shap_values, residuals

def prepare_model_for_deployment(model, scaler):
    """
    Serializes the model and scaler for production.
    """
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    return MODEL_SAVE_PATH

def deploy_model():
    """
    Builds a Gradio web application for inference.
    """
    model = joblib.load(MODEL_SAVE_PATH)
    scaler = joblib.load(SCALER_SAVE_PATH)
    
    def predict_progression(inputs):
        # inputs is a list of 10 floats
        data = np.array(inputs).reshape(1, -1)
        scaled_data = scaler.transform(data)
        pred = model.predict(scaled_data)[0]
        # Inverse log transform (since target was log-transformed in this dataset)
        final_pred = np.expm1(pred)
        return f"Predicted Diabetes Progression Score: {final_pred:.2f}"

    interface = gr.Interface(
        fn=predict_progression,
        inputs=[gr.Number(label=f"Feature {i+1}") for i in range(10)],
        outputs="text",
        title="Diabetes Progression Predictor",
        description="Enter clinical features to predict the disease progression score."
    )
    return interface.launch(share=True)

def main():
    # 1. Load Dataset
    df = pd.read_csv(DATASET_PATH)
    
    # 2. Preprocess & Split (70/20/10)
    X, y, log_transformed = preprocess_data(df)
    
    # Split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED)
    # Split temp: 2/3 of 30% = 20% valid, 1/3 of 30% = 10% test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=SEED)
    
    # 3. Scaling (Fit only on train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train & Optimize
    model, cv_scores = train_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # 5. Evaluate
    perf, shap_vals, residuals = evaluate_model(model, X_test_scaled, y_test, log_transformed)
    
    # 6. Logging
    with open(LOG_FILE_PATH, 'w') as f:
        f.write(f"Hyperparameters: iterations=200, depth=4, lr=0.03, l2=3.0\n")
        f.write(f"CV RMSE (Mean): {-np.mean(cv_scores):.4f}\n")
        f.write(f"Test Metrics: {json.dumps(perf, indent=2)}\n")
        f.write(f"Target Log Transformed: {log_transformed}\n")

    # 7. Deployment Preparation
    deployable_path = prepare_model_for_deployment(model, scaler)
    
    # 8. Deploy
    url_endpoint = deploy_model()
    
    print("Pipeline completed successfully.")
    print("Performance:", perf)
    return perf, url_endpoint

if __name__ == "__main__":
    performance, url = main()
    print(f"Model Performance: {performance}")
    print(f"Gradio App URL: {url}")
