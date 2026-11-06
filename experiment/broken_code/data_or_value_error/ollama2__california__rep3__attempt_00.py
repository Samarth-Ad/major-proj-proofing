
import os
import random
import time
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor, Pool

# Ensure directories exist
MODEL_SAVE_DIR = r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "final_catboost_model.bin")
DATASET_PATH = r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\california.csv"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def preprocess_data(df):
    """
    Evidence-Based Preprocessing Pipeline:
    Median Imputation -> Log Target Transform -> IQR Outlier Removal -> Robust Scaling
    """
    # 1. Integrity Check
    df = df.copy()
    # Replace inf with NaN for imputation
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Separate features and target
    # Assuming the last column is the target 'median_house_value'
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 2. Imputation: Median Imputation
    X = X.fillna(X.median())

    # 3. Target Transformation: log(1+x)
    # We store the transformation to inverse it later
    y_transformed = np.log1p(y)

    # 4. Outlier Detection: IQR Method
    # We apply this to the training set usually, but for a small dataset 
    # we filter based on the overall distribution of features to maintain consistency
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    
    # Mask for rows that are NOT outliers in any column
    mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
    X_filtered = X[mask]
    y_filtered = y_transformed[mask]

    # 5. Feature Scaling: RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    
    return X_scaled, y_filtered, scaler

def train_model(X_train, y_train, X_val, y_val):
    """
    CatBoost Regressor implementation with early stopping and CV.
    """
    # Hyperparameters as per blueprint
    params = {
        'iterations': 800,
        'depth': 6,
        'learning_rate': 0.03,
        'l2_leaf_reg': 3,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': SEED,
        'verbose': False
    }

    # 5-Fold Cross-Validation on training set to ensure stability
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        cv_model = CatBoostRegressor(**params)
        cv_model.fit(X_train[train_idx], y_train[train_idx], silent=True)
        preds = cv_model.predict(X_train[val_idx])
        cv_scores.append(np.sqrt(mean_squared_error(y_train[val_idx], preds)))
    
    print(f"CV RMSE (Log Scale): {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Final Training with Early Stopping using the Validation Set
    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Serialization
    model.save_model(MODEL_PATH)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluation on hold-out set with inverse log-transformation.
    """
    # Predict in log scale
    y_pred_log = model.predict(X_test)
    
    # Inverse log-transformation: exp(x) - 1
    y_true_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred_log)

    # Performance Metrics
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2 = r2_score(y_true_orig, y_pred_orig)
    mae = mean_absolute_error(y_true_orig, y_pred_orig)

    # Complexity Metrics
    # Inference Latency for 1,000 single-row predictions
    start_time = time.time()
    for _ in range(1000):
        model.predict(X_test[0].reshape(1, -1))
    latency = (time.time() - start_time) / 1000

    # Model Size
    model_size_kb = os.path.getsize(MODEL_PATH) / 1024

    performance_scores = {
        "Test RMSE (Original Scale)": rmse,
        "R2 Score": r2,
        "MAE": mae
    }
    
    complexity_scores = {
        "Avg Inference Latency (s)": latency,
        "Model Size (KB)": model_size_kb
    }

    return performance_scores, complexity_scores

def prepare_model_for_deployment(model, scaler):
    """
    Wraps the model and scaler for easy inference.
    """
    return {"model": model, "scaler": scaler}

def deploy_model(deployable_model):
    """
    Gradio Web App Demo.
    """
    model = deployable_model["model"]
    scaler = deployable_model["scaler"]
    
    # Get feature names from the scaler if possible, or use generic indices
    feature_count = scaler.n_features_in_
    inputs = [gr.Number(label=f"Feature {i+1}") for i in range(feature_count)]

    def predict( *args):
        # Convert inputs to array and scale
        feat_array = np.array(args).reshape(1, -1)
        scaled_feat = scaler.transform(feat_array)
        # Predict and inverse log
        pred_log = model.predict(scaled_feat)[0]
        pred_orig = np.expm1(pred_log)
        return f"${pred_orig:.2f} (100k)"

    demo = gr.Interface(
        fn=predict,
        inputs=inputs,
        outputs="text",
        title="California House Price Predictor (CatBoost)",
        description="Enter feature values to predict the median house value."
    )
    return demo

def main():
    # Step 1: Load Dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)

    # Step 2: Preprocessing
    X_scaled, y_transformed, scaler = preprocess_data(df)

    # Step 3: Data Split (70% Train, 20% Valid, 10% Test)
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_transformed, test_size=0.30, random_state=SEED
    )
    # Second split: Split temp into 2/3 (20% of total) and 1/3 (10% of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=SEED
    )

    # Step 4: Train and Optimize
    model = train_model(X_train, y_train, X_val, y_val)

    # Step 5: Evaluate
    model_performance, model_complexity = evaluate_model(model, X_test, y_test)

    # Step 6: Prepare for Deployment
    deployable_model = prepare_model_for_deployment(model, scaler)

    # Step 7: Deploy
    app = deploy_model(deployable_model)
    # app.launch() # Uncomment to launch the UI

    return (
        X_scaled,
        model,
        deployable_model,
        app,
        model_performance,
        model_complexity,
    )

if __name__ == "__main__":
    (
        processed_data,
        model,
        deployable_model,
        url_endpoint,
        model_performance,
        model_complexity,
    ) = main()
    
    print("\n" + "="*30)
    print("FINAL EVALUATION REPORT")
    print("="*30)
    for k, v in model_performance.items():
        print(f"{k}: {v:.4f}")
    for k, v in model_complexity.items():
        print(f"{k}: {v:.4f}")
    print("="*30)
