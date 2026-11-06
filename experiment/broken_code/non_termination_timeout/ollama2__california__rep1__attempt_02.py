
import os
import random
import time
import numpy as np
import pandas as pd
import gradio as gr
import joblib
import optuna
from scipy.stats import skew

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# PyTorch for future-proofing
import torch

# Environment Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\california.csv'
MODEL_SAVE_DIR = "./agent_workspace/trained_models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "final_stacking_model.joblib")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

class HousingPipeline:
    """
    Encapsulates preprocessing and model to prevent training-serving skew.
    """
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = RobustScaler()
        self.model = None
        self.target_log_transformed = False
        self.feature_cols = None

    def fit_preprocess(self, X, y):
        self.feature_cols = X.columns.tolist()
        
        # Target Transformation: Log transform if skewness > 1
        s = skew(y)
        if s > 1:
            self.target_log_transformed = True
            y = np.log1p(y)
        
        # Imputation
        X_imputed = self.imputer.fit_transform(X)
        
        # Feature Engineering
        X_eng = self._engineer_features(pd.DataFrame(X_imputed, columns=self.feature_cols))
        
        # Scaling
        X_scaled = self.scaler.fit_transform(X_eng)
        return X_scaled, y

    def transform_preprocess(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_cols)
            
        X_imputed = self.imputer.transform(X)
        X_eng = self._engineer_features(pd.DataFrame(X_imputed, columns=self.feature_cols))
        X_scaled = self.scaler.transform(X_eng)
        return X_scaled

    def _engineer_features(self, df):
        # 1. Spatial Interaction
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            df['Geo_Interaction'] = df['Latitude'] * df['Longitude']
        
        # 2. Collinearity Reduction: AveRooms / AveBedrms ratio
        if 'AveRooms' in df.columns and 'AveBedrms' in df.columns:
            df['Room_Bed_Ratio'] = df['AveRooms'] / (df['AveBedrms'] + 1e-6)
            df = df.drop(columns=['AveRooms', 'AveBedrms'])
            
        return df

def preprocess_data(df):
    # Load & Validation: Drop rows where target (last column) is NaN
    df = df.dropna(subset=[df.columns[-1]]) 
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    pipeline = HousingPipeline()
    X_processed, y_processed = pipeline.fit_preprocess(X, y)
    return X_processed, y_processed, pipeline

def optimize_and_train(X_train, y_train, X_val, y_val):
    def objective(trial):
        # Optimized search space for speed and performance
        params_xgb = {
            'n_estimators': trial.suggest_int('xgb_n', 100, 500),
            'max_depth': trial.suggest_int('xgb_depth', 3, 7),
            'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.2),
            'random_state': SEED
        }
        params_lgb = {
            'n_estimators': trial.suggest_int('lgb_n', 100, 500),
            'max_depth': trial.suggest_int('lgb_depth', 3, 7),
            'learning_rate': trial.suggest_float('lgb_lr', 0.01, 0.2),
            'boosting_type': 'gbdt', # Changed from 'dart' to avoid timeout
            'random_state': SEED,
            'verbose': -1
        }
        params_cat = {
            'iterations': trial.suggest_int('cat_n', 100, 500),
            'depth': trial.suggest_int('cat_depth', 3, 7),
            'learning_rate': trial.suggest_float('cat_lr', 0.01, 0.2),
            'verbose': False,
            'random_seed': SEED
        }
        
        m1 = XGBRegressor(**params_xgb).fit(X_train, y_train)
        m2 = LGBMRegressor(**params_lgb).fit(X_train, y_train)
        m3 = CatBoostRegressor(**params_cat).fit(X_train, y_train)
        
        preds = (m1.predict(X_val) + m2.predict(X_val) + m3.predict(X_val)) / 3
        return np.sqrt(mean_squared_error(y_val, preds))

    # Reduced trials to 10 to ensure execution within 300s limit
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    
    bp = study.best_params
    best_xgb = XGBRegressor(n_estimators=bp['xgb_n'], max_depth=bp['xgb_depth'], learning_rate=bp['xgb_lr'], random_state=SEED)
    best_lgb = LGBMRegressor(n_estimators=bp['lgb_n'], max_depth=bp['lgb_depth'], learning_rate=bp['lgb_lr'], boosting_type='gbdt', random_state=SEED, verbose=-1)
    best_cat = CatBoostRegressor(iterations=bp['cat_n'], depth=bp['cat_depth'], learning_rate=bp['cat_lr'], verbose=False, random_seed=SEED)
    
    estimators = [('xgb', best_xgb), ('lgb', best_lgb), ('cat', best_cat)]
    stack_model = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())
    stack_model.fit(X_train, y_train)
    
    return stack_model

def evaluate_model(model, pipeline, X_test, y_test):
    y_pred = model.predict(X_test)
    
    if pipeline.target_log_transformed:
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_test_orig, y_pred_orig = y_test, y_pred
    
    performance = {
        "RMSE": np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
        "MAE": mean_absolute_error(y_test_orig, y_pred_orig),
        "R2": r2_score(y_test_orig, y_pred_orig)
    }
    
    # P99 Latency
    latencies = []
    for _ in range(100):
        t0 = time.time()
        model.predict(X_test[0:1])
        latencies.append(time.time() - t0)
    
    complexity = {
        "P99_Latency": np.percentile(latencies, 99),
        "Model_Size_MB": os.path.getsize(MODEL_SAVE_PATH) / (1024*1024) if os.path.exists(MODEL_SAVE_PATH) else 0
    }
    return performance, complexity

def deploy_model():
    pipeline = joblib.load(MODEL_SAVE_PATH)
    
    def predict_house(*args):
        # Input features: Longitude, Latitude, HousingMedianAge, TotalRooms, TotalBedrooms, Population, Households, MedianIncome
        cols = ['Longitude', 'Latitude', 'HousingMedianAge', 'TotalRooms', 'TotalBedrooms', 'Population', 'Households', 'MedianIncome']
        df_input = pd.DataFrame([args], columns=cols)
        X_scaled = pipeline.transform_preprocess(df_input)
        
        # Future-proofing: Convert to torch tensor
        _ = torch.tensor(X_scaled, dtype=torch.float32)
        
        pred = pipeline.model.predict(X_scaled)[0]
        if pipeline.target_log_transformed:
            pred = np.expm1(pred)
        return float(pred)

    inputs = [gr.Number(label=col) for col in ['Longitude', 'Latitude', 'HousingMedianAge', 'TotalRooms', 'TotalBedrooms', 'Population', 'Households', 'MedianIncome']]
    interface = gr.Interface(fn=predict_house, inputs=inputs, outputs="number", title="California Housing Price Predictor")
    return interface.launch(share=False)

def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    
    X_processed, y_processed, pipeline = preprocess_data(df)
    
    # Stratified Split (70/20/10)
    bins = np.quantile(y_processed, np.linspace(0, 1, 11))
    y_binned = np.digitize(y_processed, bins)
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_processed, y_processed, test_size=0.1, stratify=y_binned, random_state=SEED
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.222, random_state=SEED
    )
    
    model = optimize_and_train(X_train, y_train, X_val, y_val)
    
    pipeline.model = model
    joblib.dump(pipeline, MODEL_SAVE_PATH)
    
    performance, complexity = evaluate_model(model, pipeline, X_test, y_test)
    
    print("\n--- Final Results ---")
    print("Performance:", performance)
    print("Complexity:", complexity)
    
    # Launch demo
    deploy_model()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
