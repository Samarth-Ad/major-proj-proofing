
import os
import random
import time
import json
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import torch  # Fixed: Added missing torch import
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset

# Environment Setup
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Using a generic device setting to ensure compatibility across different environments
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\wine.csv'
MODEL_SAVE_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\wine_gmm_v1.joblib'
WORKSPACE_MODEL_PATH = "./agent_workspace/trained_models/wine_gmm_v1.joblib"

# Ensure directories exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(WORKSPACE_MODEL_PATH), exist_ok=True)

class WineDataset(Dataset):
    """PyTorch-compatible Dataset for future migration to Deep Embedding Clustering (DEC)."""
    def __init__(self, X, y=None):
        # Ensure X is a numpy array before converting to tensor
        self.X = torch.tensor(X.astype(np.float32) if isinstance(X, np.ndarray) else X, dtype=torch.float32)
        self.y = torch.tensor(y.astype(np.int64) if y is not None and isinstance(y, np.ndarray) else y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def align_labels(labels_true, labels_pred):
    """
    Hungarian Algorithm to align cluster IDs to ground truth labels.
    Maximizes the overlap between predicted and true labels.
    """
    cm = confusion_matrix(labels_true, labels_pred)
    # We want to maximize the sum of the diagonal, so we minimize negative cm
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Map predicted labels to true labels
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in labels_pred])

def preprocess_data(df):
    """
    Implements the Data Engineering Pipeline: 
    Schema validation, Outlier detection (IQR), and Scaling.
    """
    # 1. Schema Validation
    if df.shape[1] != 14:
        raise ValueError(f"Expected 14 columns, got {df.shape[1]}")
    
    # 2. Target Isolation
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # 3. Outlier Detection (IQR Method)
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Clipping outliers to bounds to maintain sample size (178 rows)
    X_clipped = np.clip(X, lower_bound, upper_bound)
    
    # 4. Scaling Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    return X_clipped, y, pipeline

def train_model(X_train_scaled):
    """
    Trains Gaussian Mixture Model with specified hyperparameters.
    """
    # Model Definition
    gmm = GaussianMixture(
        n_components=3,
        covariance_type='full',
        max_iter=100,
        random_state=SEED
    )
    
    start_time = time.time()
    gmm.fit(X_train_scaled)
    train_time = time.time() - start_time
    
    return gmm, train_time

def evaluate_model(model, pipeline, X_test_raw, y_test):
    """
    Evaluates the GMM model using Internal and External metrics.
    """
    # Inference
    start_inf = time.time()
    X_scaled = pipeline.transform(X_test_raw)
    labels_pred_raw = model.predict(X_scaled)
    inf_latency = (time.time() - start_inf) / len(X_test_raw)
    
    # Label Alignment (Critical Step)
    labels_pred = align_labels(y_test, labels_pred_raw)
    
    # Metrics
    performance_scores = {
        "Silhouette": float(silhouette_score(X_scaled, labels_pred_raw)),
        "ARI": float(adjusted_rand_score(y_test, labels_pred)),
        "NMI": float(normalized_mutual_info_score(y_test, labels_pred))
    }
    
    # Complexity
    import sys
    model_size = sys.getsizeof(model)
    complexity_scores = {
        "Inference_Latency_Per_Sample": float(inf_latency),
        "Model_Size_Bytes": int(model_size)
    }
    
    return performance_scores, complexity_scores

def prepare_model_for_deployment(model, pipeline):
    """
    Saves the model and scaler as joblib files.
    """
    artifacts = {'model': model, 'pipeline': pipeline}
    joblib.dump(artifacts, MODEL_SAVE_PATH)
    joblib.dump(artifacts, WORKSPACE_MODEL_PATH)
    return artifacts

def deploy_model(artifacts):
    """
    Builds a Gradio web app for inference.
    """
    model = artifacts['model']
    pipeline = artifacts['pipeline']

    def predict(input_text):
        try:
            # Convert comma separated string to numpy array
            data = np.array([float(x) for x in input_text.split(',')]).reshape(1, -1)
            scaled_data = pipeline.transform(data)
            cluster = model.predict(scaled_data)[0]
            return f"Predicted Cluster ID: {cluster}"
        except Exception as e:
            return str(e)

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(label="Input Chemical Features (13 comma-separated values)"),
        outputs=gr.Textbox(label="Cluster Assignment"),
        title="Wine Chemical Signature Clustering"
    )
    return demo.launch(share=False) # share=False for local stability

def main():
    # Step 1: Load Dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    
    # Step 2: Data Split (70% Train, 20% Valid, 10% Test)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n = len(df)
    train_end = int(0.7 * n)
    valid_end = int(0.9 * n)
    
    train_df = df.iloc[:train_end]
    valid_df = df.iloc[train_end:valid_end]
    test_df = df.iloc[valid_end:]
    
    # Step 3: Preprocessing
    X_train_raw, y_train, pipeline = preprocess_data(train_df)
    X_valid_raw, y_valid, _ = preprocess_data(valid_df)
    X_test_raw, y_test, _ = preprocess_data(test_df)
    
    # Fit pipeline on training data and transform all sets
    X_train_scaled = pipeline.fit_transform(X_train_raw)
    X_valid_scaled = pipeline.transform(X_valid_raw)
    X_test_scaled = pipeline.transform(X_test_raw)
    
    # PyTorch Integration: Wrap in DataLoader (Requirement for future migration)
    train_loader = DataLoader(WineDataset(X_train_scaled, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(WineDataset(X_test_scaled, y_test), batch_size=32, shuffle=False)

    # Step 4: Dimensionality Reduction (Internal Variance Analysis)
    pca = PCA(n_components=3)
    pca.fit(X_train_scaled)
    explained_variance = pca.explained_variance_ratio_.tolist()

    # Step 5: Model Training & Optimization
    model, train_time = train_model(X_train_scaled)
    
    # Step 6: Evaluation
    model_performance, model_complexity = evaluate_model(model, pipeline, X_test_raw, y_test)
    model_complexity["Training_Time"] = float(train_time)
    model_complexity["Explained_Variance_3D"] = explained_variance
    
    # Step 7: Deployment Preparation
    deployable_model = prepare_model_for_deployment(model, pipeline)
    
    # Step 8: Deploy
    url_endpoint = deploy_model(deployable_model)
    
    return (
        X_train_scaled,
        model,
        deployable_model,
        url_endpoint,
        model_performance,
        model_complexity,
    )

if __name__ == "__main__":
    try:
        processed_data, model, deployable_model, url_endpoint, model_performance, model_complexity = main()
        print("\n--- Final Validation Results ---")
        print("Model Performance on Test Set:", json.dumps(model_performance, indent=4))
        print("Model Complexity & Operational Metrics:", json.dumps(model_complexity, indent=4))
        print(f"Deployment URL: {url_endpoint}")
    except Exception as e:
        print(f"Pipeline failed: {e}")
