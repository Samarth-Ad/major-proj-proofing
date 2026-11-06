
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
MODEL_DIR = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'survival_model_catboost.bin')
PIPELINE_SAVE_PATH = os.path.join(MODEL_DIR, 'preprocessing_pipeline.joblib')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

class TitanicPreprocessor:
    """
    Robust preprocessing pipeline that prevents data leakage by fitting 
    statistics only on the training set.
    """
    def __init__(self):
        self.age_medians = {}
        self.embarked_mode = None
        self.scaler = RobustScaler()
        self.cat_cols = ['Sex', 'Embarked', 'Pclass']
        self.num_cols = ['Age', 'Fare']
        self.global_age_median = 28.0

    def fit(self, df):
        df_fit = df.copy()
        
        # 1. Conditional Age Imputation: Median Age grouped by Pclass and Sex
        # Store as a dictionary for fast lookup during transform
        medians = df_fit.groupby(['Pclass', 'Sex'])['Age'].median().to_dict()
        self.age_medians = medians
        self.global_age_median = df_fit['Age'].median()
        
        # 2. Mode Imputation for Embarked
        mode_series = df_fit['Embarked'].mode()
        self.embarked_mode = mode_series[0] if not mode_series.empty else 'S'
        
        # 3. Fit Scaler on numeric columns
        # Fill NaNs temporarily to avoid scaler errors during fit
        temp_num = df_fit[self.num_cols].fillna(df_fit[self.num_cols].median())
        self.scaler.fit(temp_num)
        
        return self

    def transform(self, df):
        df = df.copy()
        
        # Feature Engineering: Family Synthesis
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # Conditional Age Imputation
        def impute_age(row):
            key = (row['Pclass'], row['Sex'])
            return self.age_medians.get(key, self.global_age_median)
        
        mask = df['Age'].isna()
        if mask.any():
            df.loc[mask, 'Age'] = df[mask].apply(impute_age, axis=1)
        
        # Mode Imputation for Embarked
        df['Embarked'] = df['Embarked'].fillna(self.embarked_mode)
        
        # Fare Imputation
        df['Fare'] = df['Fare'].fillna(0)
        
        # Scaling
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        
        # Ensure categorical columns are strings for CatBoost
        for col in self.cat_cols:
            df[col] = df[col].astype(str)
            
        return df

def load_and_validate_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
        
    df = pd.read_csv(path)
    # Normalization: Convert all column names to lowercase immediately
    df.columns = [col.lower() for col in df.columns]
    
    # Standardization Mapping
    mapping = {
        'pclass': 'Pclass', 'sex': 'Sex', 'age': 'Age', 
        'sibsp': 'SibSp', 'parch': 'Parch', 'fare': 'Fare', 
        'embarked': 'Embarked', 'survived': 'survived'
    }
    
    # Validation: Check for missing columns
    missing = [val for key, val in mapping.items() if key not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    
    # Rename and keep only required columns
    df = df.rename(columns=mapping)[list(mapping.values())]
    return df

def train_model(X_train, y_train, X_val, y_val):
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
    model.save_model(MODEL_SAVE_PATH)
    joblib.dump(preprocessor, PIPELINE_SAVE_PATH)
    return MODEL_SAVE_PATH

def deploy_model():
    # Load assets
    model = CatBoostClassifier().load_model(MODEL_SAVE_PATH)
    preprocessor = joblib.load(PIPELINE_SAVE_PATH)
    
    def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
        input_df = pd.DataFrame([{
            'Pclass': pclass, 'Sex': sex, 'Age': age, 
            'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Embarked': embarked
        }])
        
        processed_df = preprocessor.transform(input_df)
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
    
    # Stratified split: 10% Test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=SEED
    )
    
    # Split remaining 90% into 70% train and 20% val (0.2222 of 0.9 is approx 0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2222, stratify=y_train_full, random_state=SEED
    )
    
    # 3. Preprocessing (Leakage-Proof)
    preprocessor = TitanicPreprocessor()
    preprocessor.fit(X_train) 
    
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    # 4. Train and Optimize
    model = train_model(X_train_proc, y_train, X_val_proc, y_val)
    
    # 5. Prepare for Deployment
    prepare_model_for_deployment(model, preprocessor)
    
    # 6. Evaluate
    perf, comp = evaluate_model(model, X_test_proc, y_test)
    
    # 7. Deploy
    url_endpoint = deploy_model()
    
    return X_train_proc, model, MODEL_SAVE_PATH, url_endpoint, perf, comp

if __name__ == "__main__":
    try:
        processed_data, model, deployable_model, url_endpoint, model_performance, model_complexity = main()
        print("\n--- Final Validation Results ---")
        print("Model Performance on Test Set:", model_performance)
        print("Model Complexity:", model_complexity)
        print("Deployment Status: Gradio App Running")
    except Exception as e:
        print(f"Pipeline failed: {e}")
