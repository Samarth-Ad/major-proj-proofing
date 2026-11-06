
import os
import random
import time
import joblib
import numpy as np
import pandas as pd
import torch
import gradio as gr
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Absolute Paths as per instructions
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\iris.csv'
SCALER_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\scaler.joblib'
MODEL_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\gmm_iris_model.pkl'

# Ensure directories exist
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def hungarian_mapping(y_true, y_pred):
    """
    Resolves label-switching problem using the Hungarian Algorithm to map 
    predicted cluster IDs to ground-truth labels.
    """
    D = max(y_true.max(), y_pred.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_true.size):
        w[y_true[i], y_pred[i]] += 1
    
    # Maximize overlap by minimizing negative overlap
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in y_pred])

def main():
    # 1. Ingestion
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    
    # 2. Isolation
    X = df.iloc[:, :4].values
    y = df['target'].values

    # 3. Data Split (70% Train, 20% Val, 10% Test)
    # First split: 70% train, 30% temp
    X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED
    )
    # Second split: 2/3 of 30% is 20%, 1/3 of 30% is 10%
    X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=SEED
    )

    # 4. Preprocessing
    # Outlier Removal using Z-score (|Z| > 3) on Training set to define "normal"
    # To prevent leakage, we calculate Z-scores based on training distribution
    mean_train = np.mean(X_train_raw, axis=0)
    std_train = np.std(X_train_raw, axis=0)
    
    def remove_outliers(data, mean, std):
        z_scores = np.abs((data - mean) / std)
        return (z_scores < 3).all(axis=1)

    mask_train = remove_outliers(X_train_raw, mean_train, std_train)
    X_train_filtered = X_train_raw[mask_train]
    y_train_filtered = y_train_raw[mask_train]

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filtered)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Serialization of Scaler
    joblib.dump(scaler, SCALER_PATH)
    
    # 5. Tensor Conversion
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    # 6. Modeling Phase (GMM)
    gmm = GaussianMixture(
        n_components=3,
        covariance_type='full',
        max_iter=100,
        random_state=SEED
    )
    
    # Fit using Training Set (convert tensor back to numpy for sklearn)
    gmm.fit(X_train_tensor.numpy())
    
    # Model Storage
    joblib.dump(gmm, MODEL_PATH)

    # 7. Evaluation & Validation Phase
    # Predict on Test Set
    start_time = time.time()
    labels_pred_raw = gmm.predict(X_test_tensor.numpy())
    end_time = time.time()
    
    # Label Mapping via Hungarian Algorithm
    labels_pred = hungarian_mapping(y_test_raw, labels_pred_raw)
    
    # Metrics
    performance_scores = {
        "ARI": adjusted_rand_score(y_test_raw, labels_pred),
        "NMI": normalized_mutual_info_score(y_test_raw, labels_pred),
        "Silhouette": silhouette_score(X_test_tensor.numpy(), labels_pred_raw)
    }
    
    # Profiling
    latency = ((end_time - start_time) / len(X_test_raw)) * 1000 # ms per sample
    model_size = os.path.getsize(MODEL_PATH) / 1024 # KB
    
    complexity_scores = {
        "Inference Latency (ms/sample)": latency,
        "Model Size (KB)": model_size
    }

    print("\n--- Final Validation Results ---")
    print("Model Performance on Test Set:", performance_scores)
    print("Model Complexity:", complexity_scores)

    # 8. Gradio Web Application Demo
    def predict_cluster(sl, sw, pl, pw):
        features = np.array([[sl, sw, pl, pw]])
        scaled_features = scaler.transform(features)
        cluster = gmm.predict(scaled_features)[0]
        return f"Predicted Cluster ID: {cluster}"

    interface = gr.Interface(
        fn=predict_cluster,
        inputs=[
            gr.Number(label="Sepal Length"),
            gr.Number(label="Sepal Width"),
            gr.Number(label="Petal Length"),
            gr.Number(label="Petal Width")
        ],
        outputs="text",
        title="Iris GMM Clustering Deployment"
    )
    
    # Note: In a production script, we might not call .launch() if it blocks, 
    # but for a demo, it is required.
    interface.launch(share=False, inline=False)

if __name__ == "__main__":
    main()
