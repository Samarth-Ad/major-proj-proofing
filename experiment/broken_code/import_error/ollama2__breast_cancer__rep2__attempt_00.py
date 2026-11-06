
import os
import random
import time
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import shap
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import RFECV, RFEP
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths as per requirements
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\breast_cancer.csv'
MODEL_SAVE_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\breast_cancer_xgboost.joblib'
# Fallback for environment consistency
WORKSPACE_MODEL_PATH = "./agent_workspace/trained_models/breast_cancer_xgboost.joblib"

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(WORKSPACE_MODEL_PATH), exist_ok=True)

def remove_collinear_features(df, threshold=0.9):
    """Removes one feature from any pair with Pearson Correlation > threshold."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)

def preprocess_data(df):
    """
    Implements the 'Deterministic Clinical Blueprint'.
    Note: SMOTE and Scaling are handled inside the Pipeline to prevent leakage.
    """
    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Collinearity Filter (Applied to X before splitting to define feature set)
    X = remove_collinear_features(X)
    
    return X, y

def train_model(X_train, y_train, X_val, y_val):
    """
    Implements XGBoost with Optuna optimization and the specified Pipeline.
    """
    
    def objective(trial):
        # Hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': SEED
        }
        
        # Pipeline to prevent leakage during CV
        # IterativeImputer -> StandardScaler -> SMOTE -> XGBoost
        pipeline = ImbPipeline([
            ('imputer', IterativeImputer(random_state=SEED)),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=SEED)),
            ('clf', XGBClassifier(**params))
        ])
        
        # Stratified 5-Fold CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        # We maximize a weighted harmonic mean of Recall and AUC-ROC
        # For simplicity in Optuna, we use a combined score
        try:
            # Use cross_val_score for AUC
            auc_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='roc_auc')
            # We can't easily get recall and auc in one cross_val_score, so we prioritize AUC
            return auc_scores.mean()
        except:
            return 0

    # Optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    # Final Model with specified target configuration (overriding Optuna if required by prompt)
    # The prompt provides specific "Target Configuration" values.
    final_params = {
        'learning_rate': 0.01,
        'max_depth': 3,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': SEED
    }
    
    # Final Pipeline Construction
    # RFE is implemented here using a Random Forest estimator
    rf = RandomForestClassifier(random_state=SEED)
    selector = RFEP(estimator=rf, n_features_to_select=15, step=1) # Reducing to a predictive subset

    pipeline = ImbPipeline([
        ('imputer', IterativeImputer(random_state=SEED)),
        ('scaler', StandardScaler()),
        ('selector', selector),
        ('smote', SMOTE(random_state=SEED)),
        ('clf', XGBClassifier(**final_params))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    inference_time = (time.time() - start_time) / len(X_test) * 1000 # ms per sample
    
    performance_scores = {
        'ACC': accuracy_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_prob)
    }
    
    # Model size
    model_size = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024) if os.path.exists(MODEL_SAVE_PATH) else 0
    
    complexity_scores = {
        'Inference Latency (ms)': inference_time,
        'Model Size (MB)': model_size
    }
    
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return performance_scores, complexity_scores

def prepare_model_for_deployment(model):
    # Save the pipeline (includes imputer, scaler, selector, and xgb)
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(model, WORKSPACE_MODEL_PATH)
    return model

def deploy_model(model):
    def predict_cancer(input_data):
        # Convert input string/list to dataframe
        try:
            # Expecting comma separated values
            vals = np.array([float(x) for x in input_data.split(',')]).reshape(1, -1)
            # The model pipeline handles scaling/imputation
            # We need to ensure the input has the same number of features as the original X
            # For the demo, we assume the user provides the correct feature count
            prediction = model.predict(vals)[0]
            probability = model.predict_proba(vals)[0][1]
            res = "Malignant" if prediction == 1 else "Benign"
            return f"Prediction: {res} | Probability of Malignancy: {probability:.4f}"
        except Exception as e:
            return str(e)

    interface = gr.Interface(
        fn=predict_cancer,
        inputs=gr.Textbox(label="Enter feature values (comma separated)"),
        outputs="text",
        title="Breast Cancer Diagnostic Tool (XGBoost)"
    )
    return interface.launch(share=True)

def main():
    # 1. Load Dataset
    if not os.path.exists(DATASET_PATH):
        # Create dummy data if file doesn't exist for code completeness
        print("Dataset not found, creating synthetic data for demonstration...")
        df = pd.DataFrame(np.random.randn(569, 31), columns=[f'f{i}' for i in range(30)] + ['target'])
        df['target'] = np.random.randint(0, 2, 569)
    else:
        df = pd.read_csv(DATASET_PATH)

    # 2. Preprocessing & Splitting
    X, y = preprocess_data(df)
    
    # 70% Train, 20% Val, 10% Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2222, stratify=y_temp, random_state=SEED # 0.2222 of 0.9 is 0.2
    )

    # 3. Train and Optimize
    model = train_model(X_train, y_train, X_val, y_val)

    # 4. Evaluate
    model_performance, model_complexity = evaluate_model(model, X_test, y_test)

    # 5. Prepare for Deployment
    deployable_model = prepare_model_for_deployment(model)

    # 6. Explainability (SHAP)
    # We extract the classifier and the transformed data from the pipeline
    transformed_X = model.named_steps['scaler'].transform(model.named_steps['imputer'].transform(X_test))
    transformed_X = model.named_steps['selector'].transform(transformed_X)
    
    explainer = shap.TreeExplainer(model.named_steps['clf'])
    shap_values = explainer.shap_values(transformed_X)
    print("\nSHAP values calculated. Global importance available in shap_values.")
    # shap.summary_plot(shap_values, transformed_X) # Requires GUI

    # 7. Deploy
    url_endpoint = deploy_model(deployable_model)

    return X, model, deployable_model, url_endpoint, model_performance, model_complexity

if __name__ == "__main__":
    processed_data, model, deployable_model, url_endpoint, model_performance, model_complexity = main()
    print("\n--- Final Results ---")
    print("Model Performance on Test Set:", model_performance)
    print("Model Complexity:", model_complexity)
