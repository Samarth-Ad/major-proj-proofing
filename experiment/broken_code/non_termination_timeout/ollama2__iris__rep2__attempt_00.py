
import os
import random
import time
import json
import joblib
import numpy as np
import pandas as pd
import torch
import gradio as gr
from torch.utils.data import DataLoader, Dataset
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split

# Define GPU location
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths as per requirements
DATASET_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\iris.csv'
MODEL_SAVE_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\iris_gmm_v1.pkl'
SCALER_SAVE_PATH = r'C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\models\iris_scaler.pkl'
LOG_PATH = "./agent_workspace/clustering_metrics.json"

# Ensure directories exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs("./agent_workspace/trained_models", exist_ok=True)

class IrisDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def align_labels(labels_true, labels_pred):
    """
    Use Hungarian Algorithm to map cluster IDs to ground truth labels.
    """
    # Compute contingency matrix
    cm = confusion_matrix(labels_true, labels_pred)
    # We want to maximize the sum of the diagonal, so we minimize the negative
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Create mapping dictionary
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in labels_pred])

def preprocess_data(df):
    # Feature Isolation
    X = df.drop(columns=['target']).values
    y = df['target'].values

    # Split: 70% Train, 20% Val, 10% Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=SEED, stratify=y_temp)

    # Imputation (Mean)
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # Scaling (Z-score) - Fit only on train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Wrap in PyTorch DataLoaders for consistency
    train_loader = DataLoader(IrisDataset(X_train, y_train), batch_size=32, shuffle=False)
    valid_loader = DataLoader(IrisDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader = DataLoader(IrisDataset(X_test, y_test), batch_size=32, shuffle=False)

    return train_loader, valid_loader, test_loader, scaler

def train_model(train_loader):
    # Extract data from loader to NumPy for GMM
    X_train = np.vstack([batch[0].numpy() for batch in train_loader])
    
    # GMM Implementation
    model = GaussianMixture(
        n_components=3, 
        covariance_type='full', 
        random_state=SEED
    )
    model.fit(X_train)
    
    # BIC Validation (as requested)
    bic = model.bic(X_train)
    print(f"Model trained. BIC: {bic:.2f}")
    
    return model

def evaluate_model(model, test_loader, scaler):
    # Extract data
    X_test = np.vstack([batch[0].numpy() for batch in test_loader])
    y_true = np.concatenate([batch[1].numpy() for batch in test_loader])
    
    # Predict clusters
    labels_pred_raw = model.predict(X_test)
    
    # Label Alignment using Hungarian Algorithm
    labels_pred = align_labels(y_true, labels_pred_raw)
    
    # Metrics
    performance_scores = {
        "ARI": adjusted_rand_score(y_true, labels_pred),
        "NMI": normalized_mutual_info_score(y_true, labels_pred),
        "Silhouette": silhouette_score(X_test, labels_pred_raw)
    }
    
    # Complexity
    start_time = time.time()
    model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test)
    
    complexity_scores = {
        "inference_time_per_sample": inference_time,
        "model_size_kb": os.path.getsize(MODEL_SAVE_PATH) / 1024 if os.path.exists(MODEL_SAVE_PATH) else 0
    }
    
    return performance_scores, complexity_scores

def prepare_model_for_deployment(model, scaler):
    # Save model and scaler
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    
    # Also save to agent_workspace as per general instructions
    joblib.dump(model, "./agent_workspace/trained_models/iris_gmm_v1.pkl")
    
    return model

def deploy_model():
    # Load model and scaler for the app
    model = joblib.load(MODEL_SAVE_PATH)
    scaler = joblib.load(SCALER_SAVE_PATH)

    def predict_species(sepal_l, sepal_w, petal_l, petal_w):
        features = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
        scaled_features = scaler.transform(features)
        cluster = model.predict(scaled_features)[0]
        species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        return f"Predicted Cluster/Species: {species_map.get(cluster, cluster)}"

    interface = gr.Interface(
        fn=predict_species,
        inputs=[
            gr.Number(label="Sepal Length"),
            gr.Number(label="Sepal Width"),
            gr.Number(label="Petal Length"),
            gr.Number(label="Petal Width")
        ],
        outputs="text",
        title="Iris Species Clustering (GMM)"
    )
    return interface.launch(share=False)

def main():
    # 1. Load Dataset
    df = pd.read_csv(DATASET_PATH)
    
    # 2 & 3. Preprocess and Split
    train_loader, valid_loader, test_loader, scaler = preprocess_data(df)
    
    # 4 & 5. Model Definition and Training
    model = train_model(train_loader)
    
    # Evaluation
    model_performance, model_complexity = evaluate_model(model, test_loader, scaler)
    
    # Log metrics to JSON
    with open(LOG_PATH, 'w') as f:
        json.dump({"performance": model_performance, "complexity": model_complexity}, f)
    
    # 6. Prepare for Deployment
    deployable_model = prepare_model_for_deployment(model, scaler)
    
    # 7. Deploy
    url_endpoint = deploy_model()

    return (
        df,
        model,
        deployable_model,
        url_endpoint,
        model_performance,
        model_complexity,
    )

if __name__ == "__main__":
    processed_data, model, deployable_model, url_endpoint, model_performance, model_complexity = main()
    print("\n--- Final Results ---")
    print("Model Performance on Test Set:", model_performance)
    print("Model Complexity:", model_complexity)
