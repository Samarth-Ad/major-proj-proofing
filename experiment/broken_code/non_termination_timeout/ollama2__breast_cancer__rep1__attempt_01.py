
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

# Paths
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\breast_cancer.csv'
# Updated to requested directory
MODEL_SAVE_DIR = "./agent_workspace/trained_models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "breast_cancer_catboost.bin")
PIPELINE_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "breast_cancer_pipeline.joblib")
LOG_FILE_PATH = "pipeline_log.json"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

class BreastCancerInferencePipeline:
    """Wrapper class to encapsulate preprocessing and model for production deployment."""
    def __init__(self, model, scaler, imputer, dropped_cols, feature_names):
        self.model = model
        self.scaler = scaler
        self.imputer = imputer
        self.dropped_cols = dropped_cols
        self.feature_names = feature_names

    def predict(self, input_df):
        # Ensure input is DataFrame
        if not isinstance(input_df, pd.DataFrame):
            input_df = pd.DataFrame(input_df, columns=self.feature_names)
        
        # 1. Imputation
        df_imputed = self.imputer.transform(input_df)
        df_imputed = pd.DataFrame(df_imputed, columns=self.feature_names)
        
        # 2. Drop collinear features
        df_dropped = df_imputed.drop(columns=self.dropped_cols)
        
        # 3. Scaling
        df_scaled = self.scaler.transform(df_dropped)
        
        # 4. Predict
        pred = self.model.predict(df_scaled)
        prob = self.model.predict_proba(df_scaled)[:, 1]
        return pred, prob

def preprocess_data(df):
    """
    Implements the strict preprocessing pipeline:
    Split -> Impute -> Scale -> Collinearity Reduction -> SMOTE
    """
    # Validation
    if df.shape[1] != 31:
        raise ValueError(f"Dataset must have exactly 31 columns. Found {df.shape[1]}")
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    feature_names = X.columns.tolist()

    # 1. Data Splitting (70% Train, 20% Val, 10% Test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    for train_idx, temp_idx in sss1.split(X, y):
        X_train, X_temp = X.iloc[train_idx], X.iloc[temp_idx]
        y_train, y_temp = y.iloc[train_idx], y.iloc[temp_idx]

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
    train_df_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    corr_matrix = train_df_scaled.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    
    # Drop from all sets
    X_train_final = pd.DataFrame(X_train_scaled, columns=feature_names).drop(columns=to_drop).values
    X_val_final = pd.DataFrame(X_val_scaled, columns=feature_names).drop(columns=to_drop).values
    X_test_final = pd.DataFrame(X_test_scaled, columns=feature_names).drop(columns=to_drop).values

    # 5. Augmentation: SMOTE (Strictly on training set)
    smote = SMOTE(random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train_final, y_train)

    return (X_train_res, y_train_res), (X_val_final, y_val), (X_test_final, y_test), scaler, imputer, to_drop, feature_names

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
    
    model.save_model(MODEL_SAVE_PATH)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates model and calculates clinical metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    # False Negatives: Actual Malignant (1) predicted as Benign (0)
    # Assuming 1=Malignant, 0=Benign. If labels are reversed, adjust index.
    fn = cm[1, 0] if cm.shape == (2,2) else 0
    
    performance_scores = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1_Score': f1_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'ROC_AUC': roc_auc_score(y_test, y_prob),
        'False_Negatives': int(fn)
    }
    
    start_time = time.time()
    model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test)
    
    complexity_scores = {
        'Inference_Time_Per_Sample': inference_time,
        'Model_Size_KB': os.path.getsize(MODEL_SAVE_PATH) / 1024
    }
    
    return performance_scores, complexity_scores

def deploy_model(pipeline):
    """Builds a Gradio interface."""
    def predict_fn(*args):
        input_df = pd.DataFrame([args], columns=pipeline.feature_names)
        pred, prob = pipeline.predict(input_df)
        label = "Malignant" if pred[0] == 1 else "Benign"
        return f"Prediction: {label} | Probability of Malignancy: {prob[0]:.4f}"

    interface = gr.Interface(
        fn=predict_fn,
        inputs=[gr.Number(label=f) for f in pipeline.feature_names],
        outputs="text",
        title="Breast Cancer Diagnostic Tool",
        description="Clinical measurements for Malignancy prediction."
    )
    # Use inline=False and share=False to prevent timeout in automated environments
    return interface.launch(share=False, inline=False)

def main():
    # 1. Data Ingestion
    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Preprocessing
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, imputer, dropped_cols, feature_names = preprocess_data(df)
    
    # 3. Training
    model = train_model(X_train, y_train, X_val, y_val)

    # 4. Evaluation
    perf, comp = evaluate_model(model, X_test, y_test)
    
    # Log results
    log_data = {
        "dropped_features": dropped_cols,
        "performance": perf,
        "complexity": comp
    }
    with open(LOG_FILE_PATH, 'w') as f:
        json.dump(log_data, f, indent=4)

    # 5. XAI (SHAP)
    # Use a subset of X_test for SHAP to avoid timeouts
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    final_cols = [c for c in feature_names if c not in dropped_cols]
    feature_importance = np.abs(shap_values).mean(0)
    top_indices = np.argsort(feature_importance)[-10:]
    top_features = [final_cols[i] for i in top_indices]
    print(f"\nTop 10 Clinical Features: {top_features}")

    # 6. Deployment Preparation
    pipeline = BreastCancerInferencePipeline(model, scaler, imputer, dropped_cols, feature_names)
    joblib.dump(pipeline, PIPELINE_SAVE_PATH)
    
    # Launch Gradio (non-blocking)
    deploy_model(pipeline)

    return df, model, pipeline, perf, comp

if __name__ == "__main__":
    res = main()
    if res:
        df, model, pipeline, perf, comp = res
        print("\n--- Final Model Performance ---")
        print(json.dumps(perf, indent=4))
        print("\n--- Model Complexity ---")
        print(json.dumps(comp, indent=4))
