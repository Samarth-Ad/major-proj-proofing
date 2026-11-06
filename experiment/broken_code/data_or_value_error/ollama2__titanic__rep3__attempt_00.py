
import os
import random
import time
import json
import re
import numpy as np
import pandas as pd
import gradio as gr
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # Required for IterativeImputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths as per instructions
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\titanic.csv'
MODEL_SAVE_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\survival_model_catboost.bin'
# Fallback for environment consistency
WORKSPACE_MODEL_DIR = "./agent_workspace/trained_models"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(WORKSPACE_MODEL_DIR, exist_ok=True)

def extract_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return 'Unknown'

def preprocess_data(df):
    """
    Feature Engineering: FamilySize and Title extraction.
    Note: Imputation and Scaling are handled inside the Pipeline to prevent leakage.
    """
    df = df.copy()
    # Feature Engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Title'] = df['Name'].apply(extract_title)
    
    # Drop columns that aren't useful for the model
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df

def build_pipeline():
    """
    Constructs the preprocessing and model pipeline.
    """
    num_features = ['Age', 'Fare', 'FamilySize']
    cat_features = ['Sex', 'Embarked', 'Title', 'Pclass'] # Pclass treated as categorical for CatBoost/OHE

    # Numerical Pipeline: MICE Imputation -> Robust Scaling
    num_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(random_state=SEED)),
        ('scaler', RobustScaler())
    ])

    # Categorical Pipeline: Mode Imputation -> OneHot Encoding
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ]
    )

    # Model with suggested hyperparameters
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        eval_metric='F1',
        random_seed=SEED,
        verbose=False
    )

    # Use ImbPipeline to ensure SMOTE is only applied to training data
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=SEED)),
        ('classifier', model)
    ])
    
    return pipeline

def train_model(pipeline, X_train, y_train, X_val, y_val):
    # CatBoost requires a separate eval_set for early stopping
    # Since SMOTE is in the pipeline, we must preprocess X_val manually for the eval_set
    preprocessor = pipeline.named_steps['preprocessor']
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    
    # Apply SMOTE to training data only
    smote = pipeline.named_steps['smote']
    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)
    
    model = pipeline.named_steps['classifier']
    model.fit(
        X_train_res, y_train_res,
        eval_set=(X_val_proc, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Update the pipeline with the trained model
    pipeline.named_steps['classifier'] = model
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    performance_scores = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC_AUC': roc_auc_score(y_test, y_prob)
    }
    
    # Operational Profiling: Latency
    start_time = time.time()
    for _ in range(1000):
        _ = pipeline.predict(X_test[:1])
    end_time = time.time()
    avg_latency = ((end_time - start_time) / 1000) * 1000 # ms
    
    # Model Size
    pipeline.named_steps['classifier'].save_model(MODEL_SAVE_PATH)
    model_size_kb = os.path.getsize(MODEL_SAVE_PATH) / 1024
    
    complexity_scores = {
        'avg_inference_latency_ms': avg_latency,
        'model_size_kb': model_size_kb
    }
    
    return performance_scores, complexity_scores

def prepare_model_for_deployment(pipeline):
    # For CatBoost, the .bin file is already the optimized production format.
    # We save it to the workspace as well.
    workspace_path = os.path.join(WORKSPACE_MODEL_DIR, "survival_model_catboost.bin")
    pipeline.named_steps['classifier'].save_model(workspace_path)
    return pipeline

def deploy_model(pipeline):
    def predict_survival(pclass, sex, age, fare, sibsp, parch, embarked):
        # Create a dataframe for the input
        input_df = pd.DataFrame([[pclass, sex, age, fare, sibsp, parch, embarked, "Unknown"]], 
                                columns=['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked', 'Title'])
        # Re-calculate FamilySize
        input_df['FamilySize'] = input_df['SibSp'] + input_df['Parch'] + 1
        
        prediction = pipeline.predict(input_df)[0]
        prob = pipeline.predict_proba(input_df)[0][1]
        return f"Survived: {'Yes' if prediction == 1 else 'No'} (Probability: {prob:.2f})"

    interface = gr.Interface(
        fn=predict_survival,
        inputs=[
            gr.Dropdown([1, 2, 3], label="Pclass"),
            gr.Dropdown(["male", "female"], label="Sex"),
            gr.Number(label="Age"),
            gr.Number(label="Fare"),
            gr.Number(label="SibSp"),
            gr.Number(label="Parch"),
            gr.Dropdown(["S", "C", "Q"], label="Embarked"),
        ],
        outputs="text",
        title="Titanic Survival Predictor"
    )
    return interface.launch(share=True)

def main():
    # 1. Load Dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    
    # 2. Preprocessing & Feature Engineering
    df_processed = preprocess_data(df)
    X = df_processed.drop('Survived', axis=1)
    y = df_processed['Survived']
    
    # 3. Data Splitting (70% Train, 20% Val, 10% Test)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2222, random_state=SEED) # 0.2222 of 90% is 20% of total
    
    # 4. Build and Train Pipeline
    pipeline = build_pipeline()
    trained_pipeline = train_model(pipeline, X_train, y_train, X_val, y_val)
    
    # 5. Evaluate
    perf, comp = evaluate_model(trained_pipeline, X_test, y_test)
    
    # Save report
    report = {"performance": perf, "complexity": comp}
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    # 6. Prepare for Deployment
    deployable_model = prepare_model_for_deployment(trained_pipeline)
    
    # 7. Deploy
    url_endpoint = deploy_model(deployable_model)
    
    return df_processed, trained_pipeline, deployable_model, url_endpoint, perf, comp

if __name__ == "__main__":
    processed_data, model, deployable_model, url_endpoint, model_performance, model_complexity = main()
    print("\n--- Final Evaluation Report ---")
    print("Model Performance on Test Set:", json.dumps(model_performance, indent=2))
    print("Model Complexity:", json.dumps(model_complexity, indent=2))
    print(f"Deployment URL: {url_endpoint}")
