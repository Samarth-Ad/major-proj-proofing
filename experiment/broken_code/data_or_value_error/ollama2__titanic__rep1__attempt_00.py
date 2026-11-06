
import os
import random
import time
import re
import joblib
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier, Pool

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths as per specifications
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\titanic.csv'
MODEL_SAVE_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\survival_model_catboost.bin'
PIPELINE_SAVE_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\preprocessing_pipeline.joblib'

# Ensure directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

class TitanicPreprocessor:
    """
    Custom Preprocessing class to handle conditional imputation and feature engineering
    while preventing data leakage.
    """
    def __init__(self):
        self.scaler = RobustScaler()
        self.ordinal_encoder = OrdinalEncoder()
        self.age_medians = {} # Stores medians for (Pclass, Sex)
        self.embarked_mode = None
        self.cat_features = ['Sex', 'Embarked', 'Pclass', 'Title']

    def _extract_title(self, name):
        title_search = re.search(' (Mr\.|Mrs\.|Master\.|Miss\.) ', name)
        if title_search:
            return title_search.group(1)[ :-1]
        return 'Other'

    def fit(self, df):
        # Feature Engineering (Stateless)
        df_temp = df.copy()
        df_temp['Title'] = df_temp['Name'].apply(self._extract_title)
        df_temp['FamilySize'] = df_temp['SibSp'] + df_temp['Parch'] + 1
        
        # Conditional Imputation for Age
        # Group by Pclass and Sex to find medians
        self.age_medians = df_temp.groupby(['Pclass', 'Sex'])['Age'].median().to_dict()
        
        # Mode Imputation for Embarked
        self.embarked_mode = df_temp['Embarked'].mode()[0]
        
        # Fit Scaler on Age and Fare
        # We fill NaNs temporarily to fit the scaler
        temp_age = df_temp['Age'].fillna(df_temp['Age'].median())
        temp_fare = df_temp['Fare'].fillna(df_temp['Fare'].median())
        self.scaler.fit(df_temp[['Age', 'Fare']].fillna(df_temp[['Age', 'Fare']].median()))
        
        # Fit Ordinal Encoder for Pclass
        self.ordinal_encoder.fit(df_temp[['Pclass']])
        
        return self

    def transform(self, df):
        df_out = df.copy()
        
        # Feature Engineering
        df_out['Title'] = df_out['Name'].apply(self._extract_title)
        df_out['FamilySize'] = df_out['SibSp'] + df_out['Parch'] + 1
        
        # Conditional Imputation for Age
        def fill_age(row):
            key = (row['Pclass'], row['Sex'])
            return self.age_medians.get(key, 28.0) # Default to global median if key not found
        
        df_out['Age'] = df_out.apply(lambda row: row['Age'] if pd.notnull(row['Age']) else fill_age(row), axis=1)
        
        # Mode Imputation for Embarked
        df_out['Embarked'] = df_out['Embarked'].fillna(self.embarked_mode)
        
        # Scaling
        df_out[['Age', 'Fare']] = self.scaler.transform(df_out[['Age', 'Fare']].fillna(0))
        
        # Ordinal Encoding for Pclass
        df_out['Pclass'] = self.ordinal_encoder.transform(df_out[['Pclass']])
        
        # Drop columns not used by CatBoost or handled internally
        cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
        df_out = df_out.drop(columns=[c for c in cols_to_drop if c in df_out.columns])
        
        return df_out

def preprocess_data(df_train, df_val, df_test):
    preprocessor = TitanicPreprocessor()
    preprocessor.fit(df_train)
    
    train_proc = preprocessor.transform(df_train)
    val_proc = preprocessor.transform(df_val)
    test_proc = preprocessor.transform(df_test)
    
    return train_proc, val_proc, test_proc, preprocessor

def train_model(X_train, y_train, X_val, y_val, preprocessor):
    # Identify categorical features indices
    cat_features = [i for i, col in enumerate(X_train.columns) if col in preprocessor.cat_features]
    
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3,
        random_strength=1,
        eval_metric='AUC',
        early_stopping_rounds=50,
        verbose=False,
        random_seed=SEED
    )
    
    # Training with early stopping on validation set
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        use_best_model=True
    )
    
    return model

def evaluate_model(model, X_test, y_test, preprocessor):
    cat_features = [i for i, col in enumerate(X_test.columns) if col in preprocessor.cat_features]
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    end_time = time.time()
    
    latency = ((end_time - start_time) / len(X_test)) * 1000 # ms per sample
    
    performance_scores = {
        'ACC': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    
    complexity_scores = {
        'Inference Latency (ms/sample)': latency,
        'Model Size (MB)': os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024) if os.path.exists(MODEL_SAVE_PATH) else 0
    }
    
    return performance_scores, complexity_scores

def prepare_model_for_deployment(model, preprocessor):
    # Save CatBoost model
    model.save_model(MODEL_SAVE_PATH)
    # Save Preprocessing Pipeline
    joblib.dump(preprocessor, PIPELINE_SAVE_PATH)
    return model

def deploy_model():
    # Load artifacts
    model = CatBoostClassifier().load_model(MODEL_SAVE_PATH)
    preprocessor = joblib.load(PIPELINE_SAVE_PATH)
    
    def predict_survival(pclass, sex, age, fare, embarked, sibsp, parch):
        # Create a raw dataframe mimicking the input CSV row
        input_df = pd.DataFrame([{
            'Pclass': pclass, 'Sex': sex, 'Age': age, 'Fare': fare, 
            'Embarked': embarked, 'SibSp': sibsp, 'Parch': parch, 
            'Name': 'Unknown', 'Ticket': 'Unknown', 'Cabin': 'Unknown'
        }])
        
        processed_df = preprocessor.transform(input_df)
        prob = model.predict_proba(processed_df)[0][1]
        return f"Survival Probability: {prob:.2%}"

    demo = gr.Interface(
        fn=predict_survival,
        inputs=[
            gr.Dropdown([1, 2, 3], label="Pclass"),
            gr.Dropdown(["male", "female"], label="Sex"),
            gr.Number(label="Age"),
            gr.Number(label="Fare"),
            gr.Dropdown(["S", "C", "Q"], label="Embarked"),
            gr.Number(label="SibSp"),
            gr.Number(label="Parch"),
        ],
        outputs="text",
        title="Titanic Survival Predictor (CatBoost)"
    )
    return demo.launch(share=True)

def main():
    # 1. Load Dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    
    # 2. Data Split (70% Train, 20% Val, 10% Test)
    # First split 70% train, 30% temp
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df['survived'])
    # Split temp into 2/3 val (20% total) and 1/3 test (10% total)
    val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=SEED, stratify=temp_df['survived'])
    
    # 3. Preprocessing
    X_train_raw = train_df.drop('survived', axis=1)
    y_train = train_df['survived']
    X_val_raw = val_df.drop('survived', axis=1)
    y_val = val_df['survived']
    X_test_raw = test_df.drop('survived', axis=1)
    y_test = test_df['survived']
    
    X_train, X_val, X_test, preprocessor = preprocess_data(X_train_raw, X_val_raw, X_test_raw)
    
    # 4 & 5. Model Implementation & Training
    model = train_model(X_train, y_train, X_val, y_val, preprocessor)
    
    # Evaluation
    model_performance, model_complexity = evaluate_model(model, X_test, y_test, preprocessor)
    
    # 6. Prepare for Deployment
    deployable_model = prepare_model_for_deployment(model, preprocessor)
    
    # 7. Deploy
    url_endpoint = deploy_model()
    
    return (
        X_train,
        model,
        deployable_model,
        url_endpoint,
        model_performance,
        model_complexity,
    )

if __name__ == "__main__":
    processed_data, model, deployable_model, url_endpoint, model_performance, model_complexity = main()
    print("\n--- Final Evaluation Results ---")
    print("Model Performance on Test Set:", model_performance)
    print("Model Complexity:", model_complexity)
    print("Deployment URL:", url_endpoint)
