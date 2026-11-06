
import os
import random
import time
import re
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

# Paths
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\titanic.csv'
MODEL_DIR = './agent_workspace/trained_models'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'survival_model_catboost.bin')
PIPELINE_SAVE_PATH = os.path.join(MODEL_DIR, 'preprocessing_pipeline.joblib')

# Ensure directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

class TitanicPreprocessor:
    """
    Custom Preprocessing class to handle conditional imputation and feature engineering
    while preventing data leakage.
    """
    def __init__(self):
        self.scaler = RobustScaler()
        self.ordinal_encoder = OrdinalEncoder()
        self.age_medians = {} 
        self.embarked_mode = None
        self.cat_features = ['Sex', 'Embarked', 'Pclass', 'Title']

    def _standardize_columns(self, df):
        """
        Maps actual column names to standard CamelCase names.
        The previous error occurred because the mapping logic was not correctly 
        applying the rename to the dataframe.
        """
        df_copy = df.copy()
        # Create a mapping of lowercase column name -> Standard Name
        standard_names = {
            'name': 'Name', 'age': 'Age', 'fare': 'Fare', 'sex': 'Sex', 
            'embarked': 'Embarked', 'pclass': 'Pclass', 'sibsp': 'SibSp', 
            'parch': 'Parch', 'survived': 'survived'
        }
        
        rename_dict = {}
        for col in df_copy.columns:
            col_lower = col.lower()
            if col_lower in standard_names:
                rename_dict[col] = standard_names[col_lower]
        
        return df_copy.rename(columns=rename_dict)

    def _extract_title(self, name):
        if not isinstance(name, str):
            return 'Other'
        # Regex to extract honorifics
        title_search = re.search(' (Mr\.|Mrs\.|Master\.|Miss\.) ', name)
        if title_search:
            return title_search.group(1)[:-1]
        return 'Other'

    def fit(self, df):
        df = self._standardize_columns(df)
        df_temp = df.copy()
        
        # Feature Engineering
        df_temp['Title'] = df_temp['Name'].apply(self._extract_title)
        df_temp['FamilySize'] = df_temp['SibSp'] + df_temp['Parch'] + 1
        
        # Conditional Imputation for Age: Median of subgroup (Pclass, Sex)
        self.age_medians = df_temp.groupby(['Pclass', 'Sex'])['Age'].median().to_dict()
        
        # Mode Imputation for Embarked
        if 'Embarked' in df_temp.columns:
            mode_series = df_temp['Embarked'].mode()
            self.embarked_mode = mode_series[0] if not mode_series.empty else 'S'
        
        # Fit Scaler on Age and Fare (using medians to avoid NaN during fit)
        age_fill = df_temp['Age'].fillna(df_temp['Age'].median())
        fare_fill = df_temp['Fare'].fillna(df_temp['Fare'].median())
        self.scaler.fit(pd.DataFrame({'Age': age_fill, 'Fare': fare_fill}))
        
        # Fit Ordinal Encoder for Pclass
        self.ordinal_encoder.fit(df_temp[['Pclass']])
        
        return self

    def transform(self, df):
        df = self._standardize_columns(df)
        df_out = df.copy()
        
        # Feature Engineering
        df_out['Title'] = df_out['Name'].apply(self._extract_title)
        df_out['FamilySize'] = df_out['SibSp'] + df_out['Parch'] + 1
        
        # Conditional Imputation for Age
        def fill_age(row):
            key = (row['Pclass'], row['Sex'])
            # Fallback to global median if subgroup median is missing
            return self.age_medians.get(key, 28.0) 
        
        df_out['Age'] = df_out.apply(lambda row: row['Age'] if pd.notnull(row['Age']) else fill_age(row), axis=1)
        
        # Mode Imputation for Embarked
        df_out['Embarked'] = df_out['Embarked'].fillna(self.embarked_mode)
        
        # Scaling
        scale_data = df_out[['Age', 'Fare']].fillna(0)
        df_out[['Age', 'Fare']] = self.scaler.transform(scale_data)
        
        # Ordinal Encoding for Pclass
        df_out['Pclass'] = self.ordinal_encoder.transform(df_out[['Pclass']])
        
        # Drop columns not used by CatBoost
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
    # Identify categorical features indices for CatBoost
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
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        use_best_model=True
    )
    
    return model

def evaluate_model(model, X_test, y_test, preprocessor):
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    end_time = time.time()
    
    latency = ((end_time - start_time) / len(X_test)) * 1000 
    
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
    model.save_model(MODEL_SAVE_PATH)
    joblib.dump(preprocessor, PIPELINE_SAVE_PATH)
    return model

def deploy_model():
    # Load saved artifacts
    model = CatBoostClassifier().load_model(MODEL_SAVE_PATH)
    preprocessor = joblib.load(PIPELINE_SAVE_PATH)
    
    def predict_survival(pclass, sex, age, fare, embarked, sibsp, parch):
        # Create raw dataframe to pass through the same pipeline
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
    return demo.launch(share=False)

def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    
    # Standardize target column name to 'survived'
    df = df.rename(columns={col: 'survived' for col in df.columns if col.lower() == 'survived'})
    
    # Data Split (70% Train, 20% Val, 10% Test)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df['survived'])
    val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=SEED, stratify=temp_df['survived'])
    
    X_train_raw = train_df.drop('survived', axis=1)
    y_train = train_df['survived']
    X_val_raw = val_df.drop('survived', axis=1)
    y_val = val_df['survived']
    X_test_raw = test_df.drop('survived', axis=1)
    y_test = test_df['survived']
    
    # Preprocessing
    X_train, X_val, X_test, preprocessor = preprocess_data(X_train_raw, X_val_raw, X_test_raw)
    
    # Training
    model = train_model(X_train, y_train, X_val, y_val, preprocessor)
    
    # Deployment Preparation
    prepare_model_for_deployment(model, preprocessor)
    
    # Evaluation
    model_performance, model_complexity = evaluate_model(model, X_test, y_test, preprocessor)
    
    # Launch demo
    url_endpoint = deploy_model()
    
    return (
        X_train,
        model,
        model,
        url_endpoint,
        model_performance,
        model_complexity,
    )

if __name__ == "__main__":
    processed_data, model, deployable_model, url_endpoint, model_performance, model_complexity = main()
    print("\n--- Final Evaluation Results ---")
    print("Model Performance on Test Set:", model_performance)
    print("Model Complexity:", model_complexity)
    print("Deployment Status: Demo Launched")
