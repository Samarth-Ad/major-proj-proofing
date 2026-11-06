
import os
import re
import time
import random
import joblib
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# File Paths
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\titanic.csv'
# Updated to requested directory for the final output, but keeping original for consistency if needed
MODEL_DIR = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'survival_model_catboost.bin')
PIPELINE_SAVE_PATH = os.path.join(MODEL_DIR, 'preprocessing_pipeline.joblib')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

class TitanicPreprocessor:
    def __init__(self):
        # Strict Column Mapper
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
        """Normalizes column names and validates existence of required columns."""
        df = df.copy()
        # Normalize all current columns to lowercase to match mapping keys
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns=self.column_mapping)
        
        # Validation Step: Ensure all required features exist
        missing = [col for col in self.required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in dataset: {missing}. Available: {list(df.columns)}")
        return df

    def extract_features(self, df):
        """Sequential feature engineering."""
        df = df.copy()
        # Title Extraction using Regex
        df['Title'] = df['Name'].apply(lambda x: re.search(r', ([A-Za-z]+)\.', str(x)).group(1) if re.search(r', ([A-Za-z]+)\.', str(x)) else 'Unknown')
        # Family Synthesis
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        return df

    def fit(self, train_df):
        """Fits preprocessing parameters on training data only to prevent leakage."""
        # Conditional Age Imputation stats: Median Age grouped by Pclass and Sex
        self.age_medians = train_df.groupby(['Pclass', 'Sex'])['Age'].median().to_dict()
        # Mode Imputation for Embarked
        self.embarked_mode = train_df['Embarked'].mode()[0] if not train_df['Embarked'].mode().empty else 'S'
        # Scaling for continuous variables
        self.scaler.fit(train_df[['Age', 'Fare']])
        # Ordinal Encoding for Pclass
        self.ordinal_encoder.fit(train_df[['Pclass']])
        return self

    def transform(self, df):
        """Applies fitted transformations to the dataset."""
        df = df.copy()
        
        # Apply Conditional Age Imputation
        def fill_age(row):
            return self.age_medians.get((row['Pclass'], row['Sex']), 28.0)
        
        df['Age'] = df['Age'].fillna(df.apply(fill_age, axis=1))
        df['Embarked'] = df['Embarked'].fillna(self.embarked_mode)
        
        # Scaling
        df[['Age', 'Fare']] = self.scaler.transform(df[['Age', 'Fare']])
        # Encoding
        df[['Pclass']] = self.ordinal_encoder.transform(df[['Pclass']])
        
        return df

def train_model(X_train, y_train, X_val, y_val):
    # Identify categorical indices for CatBoost
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
        # Create raw input DataFrame
        input_data = pd.DataFrame([{
            'Name': name, 'Age': age, 'Fare': fare, 'Sex': sex,
            'Embarked': embarked, 'Pclass': pclass, 'SibSp': sibsp, 'Parch': parch
        }])
        
        # Apply the exact same pipeline sequence
        df = preprocessor.standardize_columns(input_data)
        df = preprocessor.extract_features(df)
        df = preprocessor.transform(df)
        
        # Ensure column order matches training features
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

    # FIX: Standardize columns BEFORE splitting to avoid KeyError on 'Survived'
    preprocessor = TitanicPreprocessor()
    df_standardized = preprocessor.standardize_columns(df_raw)
    
    # The target column is now 'survived' (lowercase as per mapping)
    target_col = 'survived'
    
    # 2. Split Data (70% Train, 20% Val, 10% Test)
    train_df, temp_df = train_test_split(
        df_standardized, test_size=0.3, stratify=df_standardized[target_col], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=1/3, stratify=temp_df[target_col], random_state=SEED
    )

    # 3. Preprocessing (Sequential and Leakage-Proof)
    # Feature Engineering first
    train_df = preprocessor.extract_features(train_df)
    val_df = preprocessor.extract_features(val_df)
    test_df = preprocessor.extract_features(test_df)

    # Fit preprocessor on train only
    preprocessor.fit(train_df)
    
    # Transform all
    train_processed = preprocessor.transform(train_df)
    val_processed = preprocessor.transform(val_df)
    test_processed = preprocessor.transform(test_df)

    # Separate features and target
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
