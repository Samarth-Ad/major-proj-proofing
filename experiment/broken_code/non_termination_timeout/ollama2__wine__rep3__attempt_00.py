
import os
import random
import time
import json
import joblib
import logging
import numpy as np
import pandas as pd
import gradio as gr
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths as per instructions
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\wine.csv'
MODEL_SAVE_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\wine_gmm_model.pkl'
# Fallback for environment consistency
WORKSPACE_MODEL_DIR = "./agent_workspace/trained_models"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(WORKSPACE_MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TabularDataset(Dataset):
    """PyTorch wrapper for tabular data to ensure PyTorch readiness."""
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def map_clusters(labels_true, labels_pred):
    """
    Implements the Hungarian Algorithm to map GMM cluster IDs to ground truth labels.
    """
    D = max(labels_pred.max(), labels_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in np.nonzero(np.where(labels_pred == np.arange(D)[:, None], 1, 0)):
        # This is a simplified contingency matrix approach
        pass 
    
    # Correct contingency matrix for linear_sum_assignment
    contingency = np.zeros((D, D), dtype=np.int64)
    for i in range(len(labels_true)):
        contingency[labels_pred[i], labels_true[i]] += 1
    
    # We want to maximize the sum of the diagonal (matches), so we minimize negative contingency
    row_ind, col_ind = linear_sum_assignment(-contingency)
    
    # Create mapping dictionary
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[x] for x in labels_pred])

def preprocess_data(df):
    """
    Phase A: Data Engineering
    Implements Median Imputation -> Z-score Scaling -> PCA (95% variance)
    """
    X = df.drop(columns=['target']) if 'target' in df.columns else df
    y = df['target'].values if 'target' in df.columns else None

    # Create the preprocessing pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95))
    ])
    
    X_transformed = pipeline.fit_transform(X)
    return X_transformed, y, pipeline

def train_model(X_train, hyperparameters):
    """
    Phase B: Model Implementation (GMM)
    """
    start_time = time.time()
    
    gmm = GaussianMixture(
        n_components=hyperparameters['n_components'],
        covariance_type=hyperparameters['covariance_type'],
        max_iter=hyperparameters['max_iter'],
        random_state=SEED
    )
    
    gmm.fit(X_train)
    training_time = (time.time() - start_time) * 1000 # ms
    
    return gmm, training_time

def evaluate_model(model, X_test, y_test):
    """
    Phase C: Evaluation & Profiling
    """
    # Inference Latency
    start_inf = time.time()
    labels_pred_raw = model.predict(X_test)
    end_inf = time.time()
    latency = ((end_inf - start_inf) / len(X_test)) * 1000 # ms per sample

    # Label Mapping using Hungarian Algorithm
    labels_pred = map_clusters(y_test, labels_pred_raw)

    # Metrics
    performance_scores = {
        "Silhouette": silhouette_score(X_test, labels_pred_raw),
        "ARI": adjusted_rand_score(y_test, labels_pred),
        "NMI": normalized_mutual_info_score(y_test, labels_pred)
    }

    # Complexity
    import sys
    model_size = sys.getsizeof(model)
    complexity_scores = {
        "Inference Latency (ms/sample)": latency,
        "Model Size (bytes)": model_size
    }

    return performance_scores, complexity_scores

def prepare_model_for_deployment(model, pipeline):
    """
    Serialization of the pipeline and the model.
    """
    deployment_package = {
        'pipeline': pipeline,
        'model': model
    }
    joblib.dump(deployment_package, MODEL_SAVE_PATH)
    # Also save to workspace as requested
    joblib.dump(deployment_package, os.path.join(WORKSPACE_MODEL_DIR, "wine_gmm_model.pkl"))
    return deployment_package

def deploy_model(deployment_package):
    """
    Gradio Web Application Demo
    """
    pipeline = deployment_package['pipeline']
    model = deployment_package['model']

    def predict_cluster(input_csv):
        # Load uploaded CSV
        df = pd.read_csv(input_csv)
        # Preprocess
        X_proc = pipeline.transform(df)
        # Predict
        clusters = model.predict(X_proc)
        return str(clusters)

    interface = gr.Interface(
        fn=predict_cluster,
        inputs=gr.File(label="Upload Wine CSV (Features only)"),
        outputs=gr.Textbox(label="Predicted Cluster IDs"),
        title="Chemistry-based Wine Clustering GMM"
    )
    return interface.launch(share=False)

def main():
    # 1. Load Dataset
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        logger.error(f"Dataset not found at {DATASET_PATH}. Please check path.")
        return None

    # 2. Preprocessing
    X_transformed, y, pipeline = preprocess_data(df)

    # 3. Data Split (70% Train, 20% Valid, 10% Test)
    # Split 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_transformed, y, test_size=0.30, random_state=SEED
    )
    # Split temp into 2/3 valid (20% of total) and 1/3 test (10% of total)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=SEED
    )

    # Convert to PyTorch Loaders for "PyTorch Readiness"
    train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=32, shuffle=True)
    valid_loader = DataLoader(TabularDataset(X_valid, y_valid), batch_size=32)
    test_loader = DataLoader(TabularDataset(X_test, y_test), batch_size=32)

    # 4. Model Training
    hyperparams = {
        'n_components': 3,
        'covariance_type': 'full',
        'max_iter': 100
    }
    model, train_time = train_model(X_train, hyperparams)

    # 5. Evaluation
    perf, comp = evaluate_model(model, X_test, y_test)
    comp['Training Time (ms)'] = train_time

    logger.info(f"Evaluation Results: {perf}")
    logger.info(f"Operational Profiling: {comp}")

    # 6. Deployment Preparation
    deployable_model = prepare_model_for_deployment(model, pipeline)

    # 7. Deployment
    url_endpoint = deploy_model(deployable_model)

    return (
        X_transformed,
        model,
        deployable_model,
        url_endpoint,
        perf,
        comp,
    )

if __name__ == "__main__":
    results = main()
    if results:
        processed_data, model, deployable_model, url_endpoint, model_performance, model_complexity = results
        print("\n" + "="*30)
        print("FINAL RESULTS")
        print("="*30)
        print("Model Performance on Test Set:", model_performance)
        print("Model Complexity:", model_complexity)
