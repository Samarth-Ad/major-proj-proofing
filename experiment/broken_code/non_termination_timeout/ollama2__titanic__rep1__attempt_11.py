
import os
import random
import time
import joblib
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Path Specifications
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\titanic.csv'
# Updated to requested directory for the final output
MODEL_DIR = r'./agent_workspace/trained_models'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'survival_model_catboost.bin')
PIPELINE_SAVE_PATH = os.path.join(MODEL_DIR, 'preprocessing_pipeline.joblib')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

class TitanicPreprocessor:
    def __init__(self):
        self.age_medians = {}
        self.embarked_mode = None
        self.scaler = RobustScaler()
        self.cat_cols = ['Sex', 'Embarked', 'Pclass']
        self.num_cols = ['Age', 'Fare']

    def fit(self, df):
        # 1. Conditional Age Imputation logic
        # Group by Pclass and Sex to find medians
        # Ensure Pclass is treated as categorical for grouping
        medians = df.groupby(['Pclass', 'Sex'])['Age'].median().to_dict()
        self.age_medians = medians
        
        # 2. Mode Imputation for Embarked
        self.embarked_mode = df['Embarked'].mode()[0] if not df['Embarked'].mode().empty else 'S'
        
        # 3. Fit Scaler on numeric columns (after filling NaNs for fitting)
        temp_df = df.copy()
        temp_df['Age'] = temp_df['Age'].fillna(temp_df['Age'].median())
        temp_df['Fare'] = temp_df['Fare'].fillna(temp_df['Fare'].median())
        self.scaler.fit(temp_df[self.num_cols])
        
        return self

    def transform(self, df):
        df = df.copy()
        
        # Family Synthesis
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # Conditional Age Imputation
        def impute_age(row):
            key = (row['Pclass'], row['Sex'])
            return self.age_medians.get(key, 28.0) # Default global median if key not found
        
        df['Age'] = df.apply(lambda row: row['Age'] if pd.notnull(row['Age']) else impute_age(row), axis=1)
        
        # Mode Imputation for Embarked
        df['Embarked'] = df['Embarked'].fillna(self.embarked_mode)
        
        # Fare Imputation (Simple median)
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        
        # Scaling
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        
        # CRITICAL FIX: CatBoost requires categorical features to be string or integer.
        # The previous error occurred because Pclass was converted to float (1.0).
        # We force categorical columns to string to avoid "bad object for id" errors.
        for col in self.cat_cols:
            df[col] = df[col].astype(str)
            
        return df

def load_and_validate_data(path):
    df = pd.read_csv(path)
    # Normalization: Convert all columns to lowercase
    df.columns = [col.lower() for col in df.columns]
    
    # Standardization Mapping
    mapping = {
        'pclass': 'Pclass', 'sex': 'Sex', 'age': 'Age', 
        'sibsp': 'SibSp', 'parch': 'Parch', 'fare': 'Fare', 
        'embarked': 'Embarked', 'survived': 'survived'
    }
    
    # Check for missing columns
    missing = [val for key, val in mapping.items() if key not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    
    # Rename and keep only required columns
    df = df.rename(columns=mapping)[list(mapping.values())]
    return df

def train_model(X_train, y_train, X_val, y_val):
    # Define categorical features by name (CatBoost handles this well if passed as list)
    cat_features = ['Sex', 'Embarked', 'Pclass']
    
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3,
        random_strength=1,
        eval_metric='AUC',
        random_seed=SEED,
        verbose=False
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        cat_features=cat_features
    )
    
    return model

def evaluate_model(model, X_test, y_test):
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    end_time = time.time()
    
    latency = ((end_time - start_time) / len(X_test)) * 1000 # ms per sample
    
    performance_scores = {
        'ACC': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC_AUC': roc_auc_score(y_test, y_prob)
    }
    
    complexity_scores = {
        'Inference_Latency_ms_sample': latency,
        'Model_Size_MB': os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024) if os.path.exists(MODEL_SAVE_PATH) else 0
    }
    
    return performance_scores, complexity_scores

def prepare_model_for_deployment(model, preprocessor):
    # Save CatBoost model
    model.save_model(MODEL_SAVE_PATH)
    # Save Preprocessing pipeline
    joblib.dump(preprocessor, PIPELINE_SAVE_PATH)
    return MODEL_SAVE_PATH

def deploy_model():
    # Load assets
    model = CatBoostClassifier().load_model(MODEL_SAVE_PATH)
    preprocessor = joblib.load(PIPELINE_SAVE_PATH)
    
    def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
        # Create input dataframe
        input_df = pd.DataFrame([{
            'Pclass': pclass, 'Sex': sex, 'Age': age, 
            'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Embarked': embarked
        }])
        
        # Process
        processed_df = preprocessor.transform(input_df)
        
        # Predict
        prob = model.predict_proba(processed_df)[0][1]
        prediction = "Survived" if prob > 0.5 else "Not Survived"
        return f"Prediction: {prediction} (Probability: {prob:.2f})"

    interface = gr.Interface(
        fn=predict_survival,
        inputs=[
            gr.Dropdown([1, 2, 3], label="Pclass"),
            gr.Dropdown(["male", "female"], label="Sex"),
            gr.Number(label="Age"),
            gr.Number(label="SibSp"),
            gr.Number(label="Parch"),
            gr.Number(label="Fare"),
            gr.Dropdown(["S", "C", "Q"], label="Embarked")
        ],
        outputs="text",
        title="Titanic Survival Predictor"
    )
    return interface.launch(share=False)

def main():
    # 1. Load and Validate
    df = load_and_validate_data(DATASET_PATH)
    
    # 2. Split (70% Train, 20% Val, 10% Test)
    X = df.drop('survived', axis=1)
    y = df['survived']
    
    # Stratified split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=SEED
    )
    
    # Split remaining 90% into 70% train and 20% val (0.2222 of 0.9 is 0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2222, stratify=y_train_full, random_state=SEED
    )
    
    # 3. Preprocessing (Leakage-Proof)
    preprocessor = TitanicPreprocessor()
    # Fit on training set only (including target for conditional imputation if needed, though not used here)
    preprocessor.fit(X_train) 
    
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    # 4. Train and Optimize
    model = train_model(X_train_proc, y_train, X_val_proc, y_val)
    
    # 5. Prepare for Deployment (Save before evaluation to calculate size)
    prepare_model_for_deployment(model, preprocessor)
    
    # 6. Evaluate
    perf, comp = evaluate_model(model, X_test_proc, y_test)
    
    # 7. Deploy
    url_endpoint = deploy_model()
    
    return X_train_proc, model, MODEL_SAVE_PATH, url_endpoint, perf, comp

if __name__ == "__main__":
    processed_data, model, deployable_model, url_endpoint, model_performance, model_complexity = main()
    print("\n--- Final Validation Results ---")
    print("Model Performance on Test Set:", model_performance)
    print("Model Complexity:", model_complexity)
    print("Deployment Status: Gradio App Running")
