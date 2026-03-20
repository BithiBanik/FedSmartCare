import os
import numpy as np
import logging
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from embeddedexample.task import (
    LSTMModel,GRUModel,
    get_weights,
    load_data_one_patient_per_round,load_data_one_patient_per_round_rest_merge,load_data_hybrid_per_round,
    set_weights,
    test,
    train,
)

import psutil
import tracemalloc
import time


# Configure logging
logging.basicConfig(
    filename="Flwr.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Create a directory for saved weights if it doesn't exist
weights_dir = "saved_weights"
os.makedirs(weights_dir, exist_ok=True)


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, dataset_path, batch_size, local_epochs, learning_rate, time_step, forecast_horizon,total_rounds):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.time_step = time_step
        self.forecast_horizon = forecast_horizon
        self.current_patient_count = 1  # Start with 1 patient
        self.total_rounds=total_rounds


        # Initialize LSTM model
        #self.net = LSTMModel( input_size=1, hidden_size=50, num_layers=1, forecast_horizon=self.forecast_horizon)
        # Initialize GRU model
        self.net = GRUModel(input_size=1, hidden_size=50, num_layers=1, forecast_horizon=self.forecast_horizon)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        """
        # Load initial dataset (1 patient) MMASH
        self.trainloader, self.testloader, self.scaler = load_data_from_disk_one_patient_per_round(
            self.current_patient_count, self.batch_size, self.dataset_path,  self.time_step, self.forecast_horizon
        )
        """
        # dynamic for all
        self.trainloader, self.testloader, self.scaler = load_data_hybrid_per_round(
            self.current_patient_count, self.batch_size, self.dataset_path, self.time_step, 30, self.total_rounds)

        logging.info("FlowerClient initialized on %s", self.device)

    def load_previous_weights(self, round_num):
        """Load previous model weights to retain knowledge from past patients."""
        weight_path = os.path.join(weights_dir, f"round_{round_num - 1}_weights.pth")
        if round_num > 1 and os.path.exists(weight_path):
            self.net.load_state_dict(torch.load(weight_path))
            logging.info(f"Loaded previous weights from {weight_path}")

    def fit(self, parameters, config):
        """Train model and return weights and extended metrics per round."""
        set_weights(self.net, parameters)

        # Detect dataset name from the dataset path
        dataset_name = os.path.basename(self.dataset_path).lower()

        # Define total patients per dataset
        dataset_patient_counts = {
            "mimicprocessed": 20,
            "smartcare": 10,
            "mit": 4,
            "rritshpro": 147
        }

        # Determine total patients for this dataset (default to 5)
        total_patients = dataset_patient_counts.get(dataset_name, 5)

        current_round = config.get("round", 1)
        self.current_patient_count = min(current_round, total_patients)

        # Start profiling
        start_time = time.time()
        tracemalloc.start()
        process = psutil.Process(os.getpid())
        start_cpu = process.cpu_percent(interval=None)

        self.trainloader, self.testloader, self.scaler = load_data_hybrid_per_round(
                self.current_patient_count, self.batch_size, self.dataset_path, self.time_step, 30, self.total_rounds)


        print(
            f"Training on round {current_round} ({dataset_name}) with "
            f"{self.current_patient_count}/{total_patients} patients, "
            f"trainloader={len(self.trainloader)}, testloader={len(self.testloader)}")

        # Train the model
        train_results = train(
            self.net,
            self.trainloader,
            self.testloader,
            self.local_epochs,
            self.learning_rate,
            self.device,
            self.scaler
        )

        # Stop profiling
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        cpu_usage = process.cpu_percent(interval=None)
        end_time = time.time()

        # Calculate system metrics
        duration = end_time - start_time
        memory_mb = peak / 10 ** 6
        data_rate = len(self.trainloader) / duration if duration > 0 else 0

        # Ensure weight directory exists
        #weights_dir = os.path.join(os.getcwd(), "weights")
        os.makedirs(weights_dir, exist_ok=True)

        # Save model weights
        weight_path = os.path.join(weights_dir, f"round_{current_round}_weights.pth")
        torch.save(self.net.state_dict(), weight_path)
        logging.info(f"Saved model weights for round {current_round} at {weight_path}")

        # Add system metrics to training results
        train_results.update({
            "training_time_sec": duration,
            "peak_memory_mb": memory_mb,
            "cpu_usage_percent": cpu_usage,
            "data_rate_samples_per_sec": data_rate
        })

        # Logging
        logging.info(f"Training time: {duration:.2f}s")
        logging.info(f"Peak memory usage: {memory_mb:.2f} MB")
        logging.info(f"CPU usage: {cpu_usage:.2f}%")
        logging.info(f"Data rate: {data_rate:.2f} samples/sec")

        return get_weights(self.net), len(self.trainloader.dataset), train_results

    def evaluate(self, parameters, config):
        """Evaluate model on the validation set with extended metrics."""
        logging.info("Starting evaluation on client...")

        # Set model parameters
        set_weights(self.net, parameters)

        # Run test and get predictions + labels
        test_loss, all_pred, all_labels = test(
            self.net,
            self.testloader,
            torch.nn.MSELoss(),
            self.device,
            self.scaler
        )

        # Flatten arrays
        forecasts_array = np.array(all_pred).flatten()
        tss_array = np.array(all_labels).flatten()

        # Compute metrics directly from predictions
        rmse = np.sqrt(np.mean((forecasts_array - tss_array) ** 2))
        mae = np.mean(np.abs(forecasts_array - tss_array))

        # MAPE
        mape_mask = tss_array != 0
        if np.any(mape_mask):
            mape = np.mean(np.abs((tss_array[mape_mask] - forecasts_array[mape_mask]) / tss_array[mape_mask])) * 100
        else:
            mape = float("nan")  # or 0.0

        # SMAPE
        denominator = (np.abs(tss_array) + np.abs(forecasts_array)) / 2
        smape_mask = denominator != 0
        if np.any(smape_mask):
            smape = np.mean(np.abs(forecasts_array[smape_mask] - tss_array[smape_mask]) / denominator[smape_mask]) * 100
        else:
            smape = float("nan")  # or 0.0

        # Logging
        logging.info(
            f"Evaluation completed. MSE: {test_loss:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}% | SMAPE: {smape:.2f}%"
        )

        return test_loss, len(self.testloader.dataset), {
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "smape": float(smape)
        }


# Client Function
def client_fn(context: Context):
    """Initialize client with dataset path and training parameters."""
    dataset_path = context.node_config["dataset-path"]
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    time_step = context.run_config["time-step"]
    forecast_horizon = context.run_config["forecast-horizon"]
    total_rounds = context.run_config["num-server-rounds"]

    print(f"Client initialized with batch size: {batch_size}")

    return FlowerClient(dataset_path, batch_size, local_epochs, learning_rate, time_step, forecast_horizon, total_rounds)


# Flower ClientApp
logging.info("Starting the Flower ClientApp...")
app = ClientApp(client_fn)
logging.info("Flower ClientApp initialized.")
