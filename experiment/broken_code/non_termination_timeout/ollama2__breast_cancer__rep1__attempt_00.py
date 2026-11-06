
import os
import random
import time
import json
import numpy as np
import pandas as pd
import gradio as gr
import joblib
import shap
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths as per instructions
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\breast_cancer.csv'
MODEL_SAVE_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\breast_cancer_catboost.bin'
LOG_FILE_PATH = "pipeline_log.json"

# Ensure model directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

class BreastCancerInferencePipeline:
    """Wrapper class to encapsulate preprocessing and model for production deployment."""
    def __init__(self, model, scaler, imputer, dropped_cols):
        self.model = model
        self.scaler = scaler
        self.imputer = imputer
        self.dropped_cols = dropped_cols

    def predict(self, input_df):
        # 1. Imputation
        df = self.imputer.transform(input_df)
        # 2. Drop collinear features
        df = pd.DataFrame(df, columns=self.imputer.get_feature_names_out())
        df = df.drop(columns=self.dropped_cols)
        # 3. Scaling
        df = self.scaler.transform(df)
        # 4. Predict
        return self.model.predict(df), self.model.predict_proba(df)[:, 1]

def preprocess_data(df):
    """
    Implements the strict preprocessing pipeline:
    Split -> Impute -> Scale -> Collinearity Reduction -> SMOTE
    """
    # Validation: Check for 31 columns and remove duplicates
    if df.shape[1] != 31:
        raise ValueError(f"Dataset must have exactly 31 columns. Found {df.shape[1]}")
    df = df.drop_duplicates().reset_index(drop=True)

    # Target and Features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 1. Data Splitting (70% Train, 20% Val, 10% Test)
    # Split Train (70%) and Temp (30%)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    for train_idx, temp_idx in sss1.split(X, y):
        X_train, X_temp = X.iloc[train_idx], X.iloc[temp_idx]
        y_train, y_temp = y.iloc[train_idx], y.iloc[temp_idx]

    # Split Temp into Val (2/3 of 30% = 20%) and Test (1/3 of 30% = 10%)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1/3, random_state=SEED)
    for val_idx, test_idx in sss2.split(X_temp, y_temp):
        X_val, X_test = X_temp.iloc[val_idx], X_temp.iloc[test_idx]
        y_val, y_test = y_temp.iloc[val_idx], y_temp.iloc[test_idx]

    # 2. Median Imputation
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # 4. Collinearity Reduction (Pearson > 0.90)
    # Use training set to determine which columns to drop
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    corr_matrix = train_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    
    # Drop from all sets
    # We need to find indices of columns to drop
    drop_indices = [X.columns.get_loc(col) for col in to_drop]
    X_train_final = np.delete(X_train_scaled, drop_indices, axis=1)
    X_val_final = np.delete(X_val_scaled, drop_indices, axis=1)
    X_test_final = np.delete(X_test_scaled, drop_indices, axis=1)

    # 5. Augmentation: SMOTE (Strictly on training set)
    smote = SMOTE(random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train_final, y_train)

    return (X_train_res, y_train_res), (X_val_final, y_val), (X_test_final, y_test), scaler, imputer, to_drop

def train_model(X_train, y_train, X_val, y_val):
    """Trains CatBoostClassifier with early stopping."""
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        eval_metric='AUC',
        random_seed=SEED,
        verbose=False,
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    
    # Save model artifact
    model.save_model(MODEL_SAVE_PATH)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates model and calculates clinical metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    fn = cm.numpy()[0, 1] if hasattr(cm, 'numpy') else cm[0, 1] # Handle potential array types
    
    performance_scores = {
        'ACC': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'ROC_AUC': roc_auc_score(y_test, y_prob),
        'False_Negatives': int(fn)
    }
    
    # Complexity
    start_time = time.time()
    model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test)
    
    complexity_scores = {
        'Inference_Time_Per_Sample': inference_time,
        'Model_Size_KB': os.path.getsize(MODEL_SAVE_PATH) / 1024
    }
    
    return performance_scores, complexity_scores

def prepare_model_for_deployment(model, scaler, imputer, dropped_cols):
    """Wraps the model and preprocessing into a single inference pipeline."""
    pipeline = BreastCancerInferencePipeline(model, scaler, imputer, dropped_cols)
    deployable_path = "./agent_workspace/trained_models/breast_cancer_pipeline.joblib"
    os.makedirs(os.path.dirname(deployable_path), exist_ok=True)
    joblib.dump(pipeline, deployable_path)
    return pipeline

def deploy_model(pipeline, feature_names):
    """Builds a Gradio interface for the clinical model."""
    # Filter feature names based on dropped columns
    final_features = [f for f in feature_names if f not in pipeline.dropped_cols]
    
    def predict_fn(*args):
        # Convert args to DataFrame
        input_df = pd.DataFrame([args], columns=feature_names)
        pred, prob = pipeline.predict(input_df)
        label = "Malignant" if pred[0] == 1 else "Benign"
        return f"Prediction: {label} | Probability of Malignancy: {prob[0]:.4f}"

    interface = gr.Interface(
        fn=predict_fn,
        inputs=[gr.Number(label=f) for f in feature_names],
        outputs="text",
        title="Breast Cancer Diagnostic Tool",
        description="Enter clinical measurements to predict if a tumor is Malignant or Benign."
    )
    return interface.launch(share=True)

def main():
    # 1. Data Ingestion
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        # Fallback for demonstration if path doesn't exist in environment
        print("Specified path not found. Please ensure the dataset is at the correct location.")
        return

    # 2 & 3. Preprocessing
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, imputer, dropped_cols = preprocess_data(df)
    
    # Log preprocessing
    log_data = {
        "dropped_features": dropped_cols,
        "train_size": X_train.shape,
        "test_size": X_test.shape
    }
    with open(LOG_FILE_PATH, 'w') as f:
        json.dump(log_data, f)

    # 4 & 5. Model Training & Optimization
    model = train_model(X_train, y_train, X_val, y_val)

    # Evaluation
    perf, comp = evaluate_model(model, X_test, y_test)
    
    # Update log with performance
    log_data.update(perf)
    with open(LOG_FILE_PATH, 'w') as f:
        json.dump(log_data, f)

    # XAI: SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # Get top 10 features
    feature_importance = np.abs(shap_values).mean(0)
    # Map back to feature names
    all_cols = df.columns[:-1].tolist()
    final_cols = [c for c in all_cols if c not in dropped_cols]
    top_indices = np.argsort(feature_importance)[-10:]
    top_features = [final_cols[i] for i in top_indices]
    print(f"Top 10 Clinical Features: {top_features}")

    # Deployment
    pipeline = prepare_model_for_deployment(model, scaler, imputer, dropped_cols)
    url_endpoint = deploy_model(pipeline, all_cols)

    return df, model, pipeline, url_endpoint, perf, comp

if __name__ == "__main__":
    processed_data, model, deployable_model, url_endpoint, model_performance, model_complexity = main()
    print("\n--- Final Model Performance ---")
    print(json.dumps(model_performance, indent=4))
    print("\n--- Model Complexity ---")
    print(json.dumps(model_complexity, indent=4))
