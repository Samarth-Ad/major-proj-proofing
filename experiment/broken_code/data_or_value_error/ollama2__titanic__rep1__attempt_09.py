
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

# File Paths - Updated to requested agent_workspace directory
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\titanic.csv'
MODEL_DIR = r'./agent_workspace/trained_models'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'survival_model_catboost.bin')
PIPELINE_SAVE_PATH = os.path.join(MODEL_DIR, 'preprocessing_pipeline.joblib')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

class TitanicPreprocessor:
    def __init__(self):
        # Strict Column Mapper: maps lowercase/variations to Standard Case
        self.column_mapping = {
            'name': 'Name', 'age': 'Age', 'fare': 'Fare', 'sex': 'Sex',
            'embarked': 'Embarked', 'pclass': 'Pclass', 'sibsp': 'SibSp', 
            'parch': 'Parch', 'survived': 'survived'
        }
        # Required columns for feature engineering and training
        self.required_cols = ['Name', 'Age', 'Fare', 'Sex', 'Embarked', 'Pclass', 'SibSp', 'Parch']
        self.age_medians = {}
        self.embarked_mode = None
        self.scaler = RobustScaler()
        self.ordinal_encoder = OrdinalEncoder()

    def standardize_columns(self, df):
        """
        Normalizes column names and validates existence of required columns.
        Fixes the KeyError by ensuring mapping is applied before validation.
        """
        df = df.copy()
        
        # 1. Create a mapping of current columns (lowercase) to their original names
        current_cols_lower = {col.lower(): col for col in df.columns}
        
        # 2. Rename columns based on the strict mapping dictionary
        rename_dict = {}
        for low_col, standard_name in self.column_mapping.items():
            if low_col in current_cols_lower:
                rename_dict[current_cols_lower[low_col]] = standard_name
        
        df = df.rename(columns=rename_dict)
        
        # 3. Validation Step: Ensure all required features exist after renaming
        # If 'Name' is missing, we cannot perform Title extraction.
        missing = [col for col in self.required_cols if col not in df.columns]
        if missing:
            # If 'Name' is missing but we have other columns, we check if it's just a naming issue
            # If it's truly missing from the CSV, we raise the error as requested.
            raise ValueError(f"Missing required columns in dataset: {missing}. Available: {list(df.columns)}")
        
        return df

    def extract_features(self, df):
        """Sequential feature engineering."""
        df = df.copy()
        # Title Extraction using Regex
        def get_title(name):
            if pd.isna(name): return 'Unknown'
            # Matches strings like "Mr.", "Mrs.", "Master.", "Miss."
            match = re.search(r', ([A-Za-z]+)\.', str(name))
            return match.group(1) if match else 'Unknown'
            
        df['Title'] = df['Name'].apply(get_title)
        # Family Synthesis
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        return df

    def fit(self, train_df):
        """Fits preprocessing parameters on training data only to prevent leakage."""
        # Conditional Age Imputation stats: Median Age grouped by Pclass and Sex
        # We use a dictionary to store these for later application
        self.age_medians = train_df.groupby(['Pclass', 'Sex'])['Age'].median().to_dict()
        
        # Mode Imputation for Embarked
        mode_res = train_df['Embarked'].mode()
        self.embarked_mode = mode_res[0] if not mode_res.empty else 'S'
        
        # Scaling for continuous variables (Age, Fare)
        # Fillna temporarily with median to avoid fitting on NaNs
        temp_df = train_df[['Age', 'Fare']].fillna(train_df[['Age', 'Fare']].median())
        self.scaler.fit(temp_df)
        
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
        
        # Fill remaining NaNs in Fare to avoid scaler errors
        df['Fare'] = df['Fare'].fillna(0)
        
        # Scaling
        df[['Age', 'Fare']] = self.scaler.transform(df[['Age', 'Fare']])
        
        # Encoding
        df[['Pclass']] = self.ordinal_encoder.transform(df[['Pclass']])
        
        return df

def train_model(X_train, y_train, X_val, y_val):
    # Categorical features for CatBoost
    cat_cols = ['Sex', 'Embarked', 'Pclass', 'Title']
    # Ensure we get indices based on the current X_train column order
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
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    df_raw = pd.read_csv(DATASET_PATH)

    # Standardize columns BEFORE splitting to ensure consistency
    preprocessor = TitanicPreprocessor()
    df_standardized = preprocessor.standardize_columns(df_raw)
    
    target_col = 'survived'
    
    # 2. Split Data (70% Train, 20% Val, 10% Test)
    # Stratified split to maintain target distribution
    train_df, temp_df = train_test_split(
        df_standardized, test_size=0.3, stratify=df_standardized[target_col], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=1/3, stratify=temp_df[target_col], random_state=SEED
    )

    # 3. Preprocessing (Sequential and Leakage-Proof)
    # Feature Engineering first (doesn't depend on global stats)
    train_df = preprocessor.extract_features(train_df)
    val_df = preprocessor.extract_features(val_df)
    test_df = preprocessor.extract_features(test_df)

    # Fit preprocessor on train only to prevent leakage
    preprocessor.fit(train_df)
    
    # Transform all sets using parameters from train_df
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

    # 5. Prepare for Deployment
    prepare_model_for_deployment(model, preprocessor)

    # 6. Evaluate
    model_performance, model_complexity = evaluate_model(model, X_test, y_test)

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
    try:
        processed_data, model, deploy_path, url_endpoint, model_performance, model_complexity = main()
        print("\n--- Final Evaluation Results ---")
        print("Model Performance on Test Set:", model_performance)
        print("Model Complexity:", model_complexity)
        print(f"Model saved at: {deploy_path}")
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        raise e
