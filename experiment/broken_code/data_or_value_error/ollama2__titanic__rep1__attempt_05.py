
import os
import re
import time
import random
import joblib
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# File Paths
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\titanic.csv'
MODEL_SAVE_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\survival_model_catboost.bin'
PIPELINE_SAVE_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\preprocessing_pipeline.joblib'

# Ensure directories exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

class TitanicPreprocessor:
    def __init__(self):
        self.column_mapping = {
            'name': 'Name', 'age': 'Age', 'fare': 'Fare', 'sex': 'Sex',
            'embarked': 'Embarked', 'pclass': 'Pclass', 'sibsp': 'SibSp', 
            'parch': 'Parch', 'survived': 'survived'
        }
        self.required_cols = ['Name', 'Age', 'Fare', 'Sex', 'Embarked', 'Pclass', 'SibSp', 'Parch']
        self.age_medians = {}
        self.embarked_mode = None
        self.scaler = RobustScaler()
        self.ordinal_encoder = OrdinalEncoder()
        self.cat_features = ['Sex', 'Embarked', 'Pclass', 'Title']

    def standardize_columns(self, df):
        # Normalize column names to lowercase first to handle any case variation
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns=self.column_mapping)
        
        # Validation Step
        missing = [col for col in self.required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in dataset: {missing}")
        return df

    def extract_features(self, df):
        # Title Extraction
        df['Title'] = df['Name'].apply(lambda x: re.search(r', ([A-Za-z]+)\.', str(x)).group(1) if re.search(r', ([A-Za-z]+)\.', str(x)) else 'Unknown')
        # Family Synthesis
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        return df

    def fit(self, train_df):
        # Conditional Age Imputation stats
        self.age_medians = train_df.groupby(['Pclass', 'Sex'])['Age'].median().to_dict()
        # Mode Imputation for Embarked
        self.embarked_mode = train_df['Embarked'].mode()[0]
        # Scaling
        self.scaler.fit(train_df[['Age', 'Fare']])
        # Ordinal Encoding for Pclass
        self.ordinal_encoder.fit(train_df[['Pclass']])
        return self

    def transform(self, df):
        df = df.copy()
        # Apply Age Imputation
        def fill_age(row):
            return self.age_medians.get((row['Pclass'], row['Sex']), 28.0) # Default to global median if key missing
        
        df['Age'] = df['Age'].fillna(df.apply(fill_age, axis=1))
        df['Embarked'] = df['Embarked'].fillna(self.embarked_mode)
        
        # Scaling
        df[['Age', 'Fare']] = self.scaler.transform(df[['Age', 'Fare']])
        # Encoding
        df[['Pclass']] = self.ordinal_encoder.transform(df[['Pclass']])
        
        return df

def preprocess_data(df, is_train=True, preprocessor=None):
    if preprocessor is None:
        preprocessor = TitanicPreprocessor()
    
    df = preprocessor.standardize_columns(df)
    df = preprocessor.extract_features(df)
    
    if is_train:
        preprocessor.fit(df)
    
    df = preprocessor.transform(df)
    return df, preprocessor

def train_model(X_train, y_train, X_val, y_val):
    # Identify categorical indices
    # Features: Age, Fare, SibSp, Parch, FamilySize, Sex, Embarked, Pclass, Title
    # We need to find the indices of the categorical columns
    cat_cols = ['Sex', 'Embarked', 'Pclass', 'Title']
    cat_features_indices = [X_train.columns.get_loc(col) for col in cat_cols]

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3,
        random_strength=1,
        eval_metric='AUC',
        early_stopping_rounds=50,
        verbose=False,
        random_state=SEED
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features_indices,
        use_best_model=True
    )
    return model

def evaluate_model(model, X_test, y_test):
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    end_time = time.time()

    latency = ((end_time - start_time) / len(X_test)) * 1000 # ms per sample

    performance_scores = {
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'F1-Score': f1_score(y_test, y_pred),
        'Accuracy': accuracy_score(y_test, y_pred)
    }
    
    complexity_scores = {
        'Inference Latency (ms/sample)': latency,
        'Model Size (MB)': os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024) if os.path.exists(MODEL_SAVE_PATH) else 0
    }

    return performance_scores, complexity_scores

def prepare_model_for_deployment(model, preprocessor):
    model.save_model(MODEL_SAVE_PATH)
    joblib.dump(preprocessor, PIPELINE_SAVE_PATH)
    return model, preprocessor

def deploy_model(model, preprocessor):
    def predict_survival(name, age, fare, sex, embarked, pclass, sibsp, parch):
        # Create raw input dictionary
        input_data = pd.DataFrame([{
            'Name': name, 'Age': age, 'Fare': fare, 'Sex': sex,
            'Embarked': embarked, 'Pclass': pclass, 'SibSp': sibsp, 'Parch': parch
        }])
        
        # Apply pipeline
        df = preprocessor.standardize_columns(input_data)
        df = preprocessor.extract_features(df)
        df = preprocessor.transform(df)
        
        # Drop target if exists and ensure column order matches training
        # Training features were: Age, Fare, SibSp, Parch, FamilySize, Sex, Embarked, Pclass, Title
        # We must ensure the order is exactly as the model expects
        # The model was trained on X_train.columns
        # Let's assume the order from the transform step
        
        # To be safe, we use the model's feature names
        df = df[model.feature_names_]
        
        prob = model.predict_proba(df)[0][1]
        return f"Survival Probability: {prob:.2%}"

    interface = gr.Interface(
        fn=predict_survival,
        inputs=[
            gr.Textbox(label="Name"),
            gr.Number(label="Age"),
            gr.Number(label="Fare"),
            gr.Dropdown(["male", "female"], label="Sex"),
            gr.Dropdown(["S", "C", "Q"], label="Embarked"),
            gr.Dropdown([1, 2, 3], label="Pclass"),
            gr.Number(label="SibSp"),
            gr.Number(label="Parch"),
        ],
        outputs="text",
        title="Titanic Survival Predictor"
    )
    return interface.launch(share=False)

def main():
    # 1. Load Dataset
    df_raw = pd.read_csv(DATASET_PATH)

    # 2. Split Data (70% Train, 20% Val, 10% Test)
    # First split 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df_raw, test_size=0.3, stratify=df_raw['Survived'], random_state=SEED
    )
    # Split temp into 2/3 val (20% of total) and 1/3 test (10% of total)
    val_df, test_df = train_test_split(
        temp_df, test_size=1/3, stratify=temp_df['Survived'], random_state=SEED
    )

    # 3. Preprocessing
    # We fit on train_df and transform all
    train_processed, preprocessor = preprocess_data(train_df, is_train=True)
    val_processed, _ = preprocess_data(val_df, is_train=False, preprocessor=preprocessor)
    test_processed, _ = preprocess_data(test_df, is_train=False, preprocessor=preprocessor)

    # Separate features and target
    # Note: 'survived' column name depends on the CSV, we standardize it to 'survived' in the mapper
    # But the original CSV usually has 'Survived'. Let's handle both.
    target_col = 'survived' if 'survived' in train_processed.columns else 'Survived'
    
    X_train = train_processed.drop(columns=[target_col])
    y_train = train_processed[target_col]
    X_val = val_processed.drop(columns=[target_col])
    y_val = val_processed[target_col]
    X_test = test_processed.drop(columns=[target_col])
    y_test = test_processed[target_col]

    # 4. Train Model
    model = train_model(X_train, y_train, X_val, y_val)

    # 5. Evaluate
    model_performance, model_complexity = evaluate_model(model, X_test, y_test)

    # 6. Prepare for Deployment
    prepare_model_for_deployment(model, preprocessor)

    # 7. Deploy
    url_endpoint = deploy_model(model, preprocessor)

    return (
        train_processed,
        model,
        MODEL_SAVE_PATH,
        url_endpoint,
        model_performance,
        model_complexity,
    )

if __name__ == "__main__":
    processed_data, model, deploy_path, url_endpoint, model_performance, model_complexity = main()
    print("\n--- Final Evaluation Results ---")
    print("Model Performance on Test Set:", model_performance)
    print("Model Complexity:", model_complexity)
    print(f"Model saved at: {deploy_path}")
