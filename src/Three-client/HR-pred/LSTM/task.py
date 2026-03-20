import logging
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from embeddedexample.preprocess import loadData
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


def natural_sort_key(text):
    """Extract numbers from text for proper numeric sorting (e.g., user_11 after user_2)."""
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', text)]

def load_data_one_patient_per_round(patient_count: int, batch_size: int, DATASET_DIR: str, time_step: int, forecast_horizon: int):
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
    # Loop over all CSV files
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith(".csv"):
            filepath = os.path.join(DATASET_DIR, filename)


            # Load and preprocess data
            X_train, X_test, y_train, y_test, scaler = loadData(filepath, time_step, forecast_horizon)

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
            #print(f"Loaded data from {selected_folder}.")
            print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

    return trainloader, testloader, scaler

def load_data_one_patient_per_round_rest_merge(patient_count: int, batch_size: int, DATASET_DIR: str,
                                    time_step: int, forecast_horizon: int):

    # List all patient CSV files sorted (round 1 loads first patient, etc.)
    patient_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith(".csv")])
    total_patients = len(patient_files)

    # Select patients to load this round
    selected_files = patient_files[:min(patient_count, total_patients)]

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    scalers = []

    for filename in selected_files:
        filepath = os.path.join(DATASET_DIR, filename)

        # Load and preprocess data
        X_train, X_test, y_train, y_test, scaler = loadData(filepath, time_step, forecast_horizon)

        # Append for merging later
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)
        y_test_list.append(y_test)
        scalers.append(scaler)

    # Merge all selected patients into unified datasets
    X_train = np.vstack(X_train_list)
    y_train = np.vstack(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.vstack(y_test_list)

    # Use the scaler of the first patient (or create combined scaler if needed)
    scaler = scalers[0]

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"✅ Round using {len(selected_files)}/{total_patients} patients")
    print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

    return trainloader, testloader, scaler

def load_data_hybrid_per_round(
    current_patient_count: int,
    batch_size: int,
    DATASET_DIR: str,
    time_step: int,
    forecast_horizon: int,
    cutoff_round: int = 20
):
    """
    Dynamically load patient data based on how many patients have been seen so far.

    Case 1️⃣: total_patients == cutoff_round
        → Load 1 new patient per round (1-to-1 mapping)

    Case 2️⃣: total_patients > cutoff_round
        → Divide patients equally across cutoff rounds

    Case 3️⃣: total_patients < cutoff_round
        → Load 1 new patient until all patients are seen.
          After all are seen, load only the last patient.
    """

    # --- Load all patient CSVs ---
    patient_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith(".csv")])
    total_patients = len(patient_files)
    if total_patients == 0:
        raise ValueError("No patient CSV files found in dataset directory.")

    # --- Determine which patients to load ---
    if total_patients == cutoff_round:
        # Case 1️⃣: one patient per round
        idx = min(current_patient_count - 1, total_patients - 1)
        selected_files = [patient_files[idx]]

    elif total_patients > cutoff_round:
        # Case 2️⃣: more patients than cutoff → divide evenly
        patients_per_round = total_patients // cutoff_round
        start_idx = (current_patient_count - 1) * patients_per_round
        end_idx = current_patient_count * patients_per_round if current_patient_count < cutoff_round else total_patients
        selected_files = patient_files[start_idx:end_idx]

    else:
        # Case 3️⃣: fewer patients than cutoff → one per round until all seen, then last patient only
        if current_patient_count <= total_patients:
            idx = current_patient_count - 1
            selected_files = [patient_files[idx]]
        else:
            selected_files = [patient_files[-1]]

    # --- Load and merge selected patients ---
    X_train_list, y_train_list, X_test_list, y_test_list, scalers = [], [], [], [], []
    for filename in selected_files:
        filepath = os.path.join(DATASET_DIR, filename)
        X_train, X_test, y_train, y_test, scaler = loadData(filepath, time_step, forecast_horizon)
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)
        y_test_list.append(y_test)
        scalers.append(scaler)

    if not X_train_list:
        raise ValueError(f"No patient data loaded for round {current_patient_count}")

    X_train = np.vstack(X_train_list)
    y_train = np.vstack(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.vstack(y_test_list)
    scaler = scalers[0]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    testloader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    print(f"✅ Round {current_patient_count}: Loaded {len(selected_files)} patient(s) → {selected_files}")
    print(f"Training samples: {len(trainloader.dataset)}, Testing samples: {len(testloader.dataset)}")

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


