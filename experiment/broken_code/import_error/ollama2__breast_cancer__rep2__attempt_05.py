
import os
import random
import time
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import optuna
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths configuration
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\breast_cancer.csv'
MODEL_SAVE_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\breast_cancer_xgboost.joblib'
# Ensure directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

def load_data():
    """Retrieve the dataset from the specified source."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    # Assuming the last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def collinearity_filter(X_train):
    """Identify and return features to drop based on Pearson Correlation > 0.9."""
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    return to_drop

def optimize_hyperparameters(X_train, y_train):
    """Optuna optimization using SimpleImputer to prevent timeouts."""
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': SEED
        }
        
        # Fast pipeline for tuning
        fast_pipe = ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=SEED)),
            ('clf', xgb.XGBClassifier(**params))
        ])
        
        # Maximize Recall as per requirement
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        score = cross_val_score(fast_pipe, X_train, y_train, cv=cv, scoring='recall').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, timeout=120) # Limited trials to prevent timeout
    return study.best_params

def train_final_model(X_train, y_train, best_params):
    """Train the final model using the Robust Clinical Pipeline."""
    # Final Pipeline with IterativeImputer and RFE
    pipeline = ImbPipeline([
        ('imputer', IterativeImputer(random_state=SEED)),
        ('scaler', StandardScaler()),
        ('feature_selection', RFE(estimator=RandomForestClassifier(random_state=SEED), n_features_to_select=15)),
        ('smote', SMOTE(random_state=SEED)),
        ('clf', xgb.XGBClassifier(**best_params, random_state=SEED))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    """Evaluate performance and complexity."""
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    latency = (time.time() - start_time) / len(X_test) * 1000 # ms per sample

    performance_scores = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_prob)
    }
    
    # Model size in MB
    model_size = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024) if os.path.exists(MODEL_SAVE_PATH) else 0
    complexity_scores = {
        'Inference Latency (ms/sample)': latency,
        'Model Size (MB)': model_size
    }
    
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return performance_scores, complexity_scores

def prepare_model_for_deployment(model):
    """Serialize the pipeline for production."""
    joblib.dump(model, MODEL_SAVE_PATH)
    return joblib.load(MODEL_SAVE_PATH)

def deploy_model(model):
    """Build Gradio web application."""
    def predict_cancer(input_data):
        # Convert input list to DataFrame
        df_input = pd.DataFrame([input_data])
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]
        result = "Malignant" if prediction == 1 else "Benign"
        return f"Prediction: {result} | Probability of Malignancy: {probability:.2f}"

    # Dynamically create inputs based on model's expected feature count (after RFE/Collinearity)
    # Note: The model pipeline handles the transformation, so we need raw input features
    # We'll use a generic number of inputs or a file upload for the demo
    interface = gr.Interface(
        fn=predict_cancer,
        inputs=gr.Textbox(lines=2, placeholder="Enter comma-separated feature values"),
        outputs="text",
        title="Breast Cancer Diagnostic Tool",
        description="Enter clinical features to predict malignancy."
    )
    # To make it work with the textbox, we need a wrapper
    def wrapper(text):
        try:
            vals = [float(x) for x in text.split(',')]
            return predict_cancer(vals)
        except Exception as e:
            return f"Error: Please ensure you enter the correct number of features. {str(e)}"

    interface.fn = wrapper
    return interface.launch(share=True)

def main():
    # 1. Load Data
    X, y = load_data()

    # 2. Data Splitting (70% Train, 20% Val, 10% Test)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.222, stratify=y_train_full, random_state=SEED
    ) # 0.222 of 90% is approx 20% of total

    # 3. Collinearity Filter (Pre-Pipeline)
    drop_cols = collinearity_filter(X_train)
    X_train = X_train.drop(columns=drop_cols)
    X_val = X_val.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)

    # 4. Hyperparameter Optimization
    print("Optimizing hyperparameters...")
    best_params = optimize_hyperparameters(X_train, y_train)
    print(f"Best Params: {best_params}")

    # 5. Final Training
    print("Training final robust pipeline...")
    model = train_final_model(X_train, y_train, best_params)

    # 6. Evaluation
    perf, comp = evaluate_model(model, X_test, y_test)

    # 7. Deployment Preparation
    deployable_model = prepare_model_for_deployment(model)

    # 8. SHAP Explainability
    # To get SHAP values, we must transform the data through the pipeline except the classifier
    X_test_transformed = model.named_steps['feature_selection'].transform(
        model.named_steps['scaler'].transform(
            model.named_steps['imputer'].transform(X_test)
        )
    )
    explainer = shap.TreeExplainer(model.named_steps['clf'])
    shap_values = explainer.shap_values(X_test_transformed)
    print("\nSHAP values calculated successfully.")

    # 9. Deploy
    url_endpoint = deploy_model(deployable_model)

    return (X_train, model, deployable_model, url_endpoint, perf, comp)

if __name__ == "__main__":
    processed_data, model, deployable_model, url_endpoint, model_performance, model_complexity = main()
    print("\n--- Final Results ---")
    print("Model Performance on Test Set:", model_performance)
    print("Model Complexity:", model_complexity)
