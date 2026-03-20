import logging
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from embeddedexample.preprocess import preprocessing,SmartcareProcess_with_LSTM
import os
import numpy as np
import glob
import re

from collections import OrderedDict

# Configure logging
logging.basicConfig(
    filename="server.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class LSTMModel(nn.Module):

    def __init__(self, input_size=1, hidden_size=50, num_layers=1, forecast_horizon=30):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, forecast_horizon)

    
    def forward(self, x):
        # Initialize hidden state and cell state with correct dimensions
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out
    """
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h_0, c_0))  # out shape: (batch_size, seq_length, hidden_size)

        # print("LSTM Output Shape:", out.shape)  # Debugging shape

        out = out[:, -1, :]  # Taking the last time step output -> (batch_size, hidden_size)

        # print("After selecting last timestep:", out.shape)  # Should be (batch_size, hidden_size)

        out = self.fc(out)  # Final output -> (batch_size, forecast_horizon)

        # print("Final Output Shape:", out.shape)  # Should be (batch_size, forecast_horizon)

        return out
      """
# Model weight management
def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
"""
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=30, num_layers=2, output_dim=30):
        super(LSTMModel, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Use the last time step's output
        return out

# Model weight management
def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
"""


    
def load_data_from_disk_round(patient_count: int, batch_size: int, DATASET_DIR: str, time_step: int, forecast_horizon: int):
    """
    Load and preprocess heart rate time-series data from multiple patients, then create DataLoaders.

    Args:
        patient_count (int): Number of patients to include in the dataset.
        batch_size (int): Batch size for DataLoader.

    Returns:
        trainloader (DataLoader): Training DataLoader.
        testloader (DataLoader): Testing DataLoader.
    """
    all_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "user_*")))  # Get all patient directories
    selected_folders = all_folders[:patient_count]  # Select patient folders up to `patient_count`
    filepaths = [os.path.join(folder, "RR.csv") for folder in selected_folders if os.path.exists(os.path.join(folder, "RR.csv"))]

    if not filepaths:
        raise ValueError("No valid patient files found. Check dataset directory structure.")

    # Initialize lists to store combined patient data
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    # Load and preprocess each patient's dataset
    for filepath in filepaths:
        X_train, X_test, y_train, y_test , scaler= preprocessing(filepath, time_step, forecast_horizon)

        # Append to lists
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)
        y_test_list.append(y_test)

    # Stack all patients' data
    X_train = torch.tensor(np.vstack(X_train_list), dtype=torch.float32)
    y_train = torch.tensor(np.vstack(y_train_list), dtype=torch.float32)
    X_test = torch.tensor(np.vstack(X_test_list), dtype=torch.float32)
    y_test = torch.tensor(np.vstack(y_test_list), dtype=torch.float32)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    # Print training sample size and number of patients
    print(f"Loaded data from {patient_count} patients.")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total testing samples: {len(test_dataset)}")
    return trainloader, testloader , scaler

def natural_sort_key(text):
    """Extract numbers from text for proper numeric sorting (e.g., user_11 after user_2)."""
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', text)]

def load_data_from_disk_one_patient_per_round(patient_count: int, batch_size: int, DATASET_DIR: str, time_step: int, forecast_horizon: int):
    """
    Dynamically load heart rate data from a specific patient folder based on patient_count.

    Args:
        patient_count (int): Number of patients (determines which folder to use).
        batch_size (int): Batch size for DataLoader.
        DATASET_DIR (str): Path to dataset directory.
        time_step (int): Number of timesteps for sequence modeling.
        forecast_horizon (int): Forecast horizon.

    Returns:
        trainloader (DataLoader): Training DataLoader.
        testloader (DataLoader): Testing DataLoader.
        scaler (StandardScaler): Scaler used for normalization.
    """
    # Get sorted list of patient folders
    all_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "user_*")), key=natural_sort_key)

    if patient_count > len(all_folders) or patient_count < 1:
        raise ValueError(f"Invalid patient_count={patient_count}. Only {len(all_folders)} available folders.")

    # Select only the folder corresponding to patient_count
    selected_folder = all_folders[patient_count - 1]  # 1-based indexing (patient_count=1 → first folder)

    filepath = os.path.join(selected_folder, "RR.csv")
    if not os.path.exists(filepath):
        raise ValueError(f"File not found: {filepath}")

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocessing(filepath, time_step, forecast_horizon)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size)

    # Print dataset info
    print(f"Loaded data from {selected_folder}.")
    print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

    return trainloader, testloader, scaler

def load_SmartCare_data_from_disk_one_patient_per_round(patient_count: int, batch_size: int, DATASET_DIR: str, time_step: int, forecast_horizon: int):
    # Get sorted list of CSV files directly in the smartcare dataset directory
    all_csv_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv")), key=natural_sort_key)

    # Check if the patient_count is valid
    if patient_count > len(all_csv_files) or patient_count < 1:
        raise ValueError(f"Invalid patient_count={patient_count}. Only {len(all_csv_files)} CSV files found.")

    # Select the CSV file corresponding to the patient
    selected_file = all_csv_files[patient_count - 1]  # 1-based indexing
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = SmartcareProcess_with_LSTM(selected_file, time_step, forecast_horizon)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size)

    # Print dataset info
    print(f"Loaded data from {selected_file}.")
    print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

    return trainloader, testloader, scaler

def train_step(net, loss_fn, optimizer, device, x, y):
    """ Perform a single training step (forward, loss, backward, optimize). """
    net.train()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    yhat = net(x)
    loss = loss_fn(yhat, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(net, testloader, loss_fn, device, scaler):
    """Evaluate the model on the test set and return average loss with inverse transformation."""
    total_loss = 0.0
    total_samples = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            yhat = net(features)

            # Convert tensors to numpy
            yhat_np = yhat.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Apply inverse scaling
            yhat_original = scaler.inverse_transform(yhat_np)
            labels_original = scaler.inverse_transform(labels_np)

            # Compute loss on original scale (MSE)
            batch_size = labels_np.shape[0]
            loss = loss_fn(
                torch.tensor(yhat_original), torch.tensor(labels_original)
            ).item()
            total_loss += loss * batch_size
            total_samples += batch_size

            # Store for metrics
            all_preds.append(yhat_original)
            all_labels.append(labels_original)

    average_loss = total_loss / total_samples  # Now this is true MSE

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    return average_loss, all_preds, all_labels


def train(net, trainloader, testloader, epochs, learning_rate, device, scaler):
    # Train the model and evaluate it on the test set after training."""
    net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001)
    all_pred, all_labels = [], []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for features, labels in trainloader:
            batch_loss = train_step(net, loss_fn, optimizer, device, features, labels)
            epoch_loss += batch_loss
        epoch_loss /= len(trainloader)
        logging.info(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss:.4f}")

    test_loss, all_pred, all_labels = test(net, testloader, loss_fn, device, scaler)
    logging.info(f"Test Loss after {epochs} epochs: {test_loss:.4f}")

    return {"test_loss": test_loss}


