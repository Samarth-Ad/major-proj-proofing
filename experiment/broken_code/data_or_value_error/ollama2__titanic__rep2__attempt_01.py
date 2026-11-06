
import os
import random
import time
import re
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import shap

# CRITICAL: Enable IterativeImputer before importing it from sklearn.impute
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer, SimpleImputer

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from catboost import CatBoostClassifier

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths as per specifications
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\titanic.csv'
# Updated to the requested directory
MODEL_SAVE_DIR = './agent_workspace/trained_models'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'titanic_catboost_final.joblib')

# Ensure directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

class TitanicFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom Transformer for Title Extraction and Social Status Mapping"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Title Extraction using Regex
        X['Title'] = X['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Group rare titles
        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonk']
        X['Title'] = X['Title'].replace(rare_titles, 'Rare')
        X['Title'] = X['Title'].fillna('Rare')
        
        # Social Status Mapping
        status_map = {
            'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5
        }
        X['SocialStatus'] = X['Title'].map(status_map)
        
        # Drop Name as it's no longer needed for the model
        return X.drop(columns=['Name'])

def get_preprocessing_pipeline():
    """
    Defines the preprocessing logic.
    """
    numeric_features = ['Age', 'Fare']
    categorical_features = ['Sex', 'Embarked', 'Title']
    
    # Preprocessing for numeric data: IterativeImputer -> RobustScaler
    numeric_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(max_iter=10, random_state=SEED)),
        ('scaler', RobustScaler())
    ])
    
    # Preprocessing for categorical data: Constant Imputer -> OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine all preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keeps Pclass and SocialStatus
    )
    
    return preprocessor

def train_model(X_train, y_train, preprocessor):
    """
    Implements the CatBoost model within an Imbalanced-Learn Pipeline.
    """
    catboost_params = {
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': SEED,
        'verbose': False,
        'eval_metric': 'AUC'
    }
    
    # Full Pipeline: Feature Extraction -> Preprocessing -> SMOTE -> CatBoost
    full_pipeline = ImbPipeline(steps=[
        ('extractor', TitanicFeatureExtractor()),
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=SEED)),
        ('classifier', CatBoostClassifier(**catboost_params))
    ])
    
    # Stratified 5-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=skf, scoring='roc_auc')
    print(f"CV ROC-AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    full_pipeline.fit(X_train, y_train)
    return full_pipeline

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the hold-out test set and profiles complexity.
    """
    # Latency profiling
    start_time = time.time()
    # Simulate 1000 predictions
    test_samples = pd.concat([X_test] * (int(np.ceil(1000/len(X_test)))), ignore_index=True)
    _ = model.predict(test_samples.iloc[:1000])
    latency = (time.time() - start_time) / 1000
    
    # Performance Metrics
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    performance_scores = {
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'F1-Score': f1_score(y_test, y_pred),
        'Accuracy': accuracy_score(y_test, y_pred)
    }
    
    # Model Size
    joblib.dump(model, "temp_model.joblib")
    model_size = os.path.getsize("temp_model.joblib") / (1024 * 1024)
    os.remove("temp_model.joblib")
    
    complexity_scores = {
        'Inference Latency (s/sample)': latency,
        'Model Size (MB)': model_size
    }
    
    return performance_scores, complexity_scores

def deploy_model_gradio(model):
    """Builds a Gradio interface for the Titanic survival predictor."""
    def predict_survival(name, age, fare, sex, embarked, pclass):
        input_df = pd.DataFrame([{
            'Name': name, 'Age': float(age), 'Fare': float(fare),
            'Sex': sex, 'Embarked': embarked, 'Pclass': int(pclass)
        }])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        return f"Prediction: {'Survived' if prediction == 1 else 'Did not survive'} (Probability: {probability:.2f})"

    interface = gr.Interface(
        fn=predict_survival,
        inputs=[
            gr.Textbox(label="Name"),
            gr.Number(label="Age"),
            gr.Number(label="Fare"),
            gr.Dropdown(['male', 'female'], label="Sex"),
            gr.Dropdown(['S', 'C', 'Q'], label="Embarked"),
            gr.Dropdown([1, 2, 3], label="Pclass")
        ],
        outputs="text",
        title="Titanic Survival Predictor (CatBoost AutoML)"
    )
    return interface

def main():
    # 1. Load Dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    # 2. Data Split (70% Train, 20% Valid, 10% Test)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df['survived'])
    valid_df, test_df = train_test_split(temp_df, test_size=0.3333, random_state=SEED, stratify=temp_df['survived'])

    # Prepare X and y
    def split_xy(data):
        X = data.drop('survived', axis=1)
        y = data['survived'].astype(int)
        return X, y

    X_train, y_train = split_xy(train_df)
    X_valid, y_valid = split_xy(valid_df)
    X_test, y_test = split_xy(test_df)

    # 3. Preprocessing Setup
    preprocessor = get_preprocessing_pipeline()

    # 4. Model Training
    model = train_model(X_train, y_train, preprocessor)

    # 5. Evaluation
    model_performance, model_complexity = evaluate_model(model, X_test, y_test)
    
    # SHAP Interpretability
    X_test_transformed = model.named_steps['extractor'].transform(X_test)
    X_test_preprocessed = model.named_steps['preprocessor'].transform(X_test_transformed)
    
    # Extract feature names for SHAP
    cat_features = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    all_features = ['Age', 'Fare'] + list(cat_features) + ['Pclass', 'SocialStatus']
    
    explainer = shap.TreeExplainer(model.named_steps['classifier'])
    shap_values = explainer.shap_values(X_test_preprocessed)
    print("\nSHAP Feature Importance calculated successfully.")

    # 6. Serialization
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # 7. Gradio Deployment
    app = deploy_model_gradio(model)
    # app.launch() # Uncomment to launch in local environment

    return model_performance, model_complexity

if __name__ == "__main__":
    perf, comp = main()
    print("\n" + "="*30)
    print("FINAL EVALUATION RESULTS")
    print("="*30)
    print(f"Performance: {perf}")
    print(f"Complexity: {comp}")
