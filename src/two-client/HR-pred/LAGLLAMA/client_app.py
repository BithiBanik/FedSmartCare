import os
import numpy as np
import logging
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from embeddedexample.preprocess import (
process_with_Lag_llama
)
from embeddedexample.task import (
load_data_from_disk_one_patient_per_round_lagLLama,load_SmartCare_data_from_disk_one_patient_per_round_lagLLama)
import sys
from collections import OrderedDict

import flwr as fl
import torch
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
import pandas as pd
import numpy as np
from preprocess import process_with_Lag_llama
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.modules.loss import NegativeLogLikelihood
# Add the required globals to the safe globals list
torch.serialization.add_safe_globals([StudentTOutput, NegativeLogLikelihood])
import os

import psutil
import tracemalloc
import time



from datetime import date
# Configure logging
# Configure logging
logging.basicConfig(
    filename="Flwr.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Create a directory for saved weights if it doesn't exist
weights_dir = "saved_weights"
os.makedirs(weights_dir, exist_ok=True)

class LagLlamaClient(NumPyClient):
    def __init__(self, model_path, data_path, forecast_horizon, context_length, process_function, local_epochs, learning_rate):
        self.num_samples = 20
        self.predictor = None
        self.context_length = context_length
        self.prediction_length = forecast_horizon
        self.model_path = model_path
        self.data_path = data_path
        self.process_function = process_function
        self.device = torch.device("cpu")
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        # Load model and data
        self.model = self.load_model()

        # Initialize patient_count (could be passed dynamically based on round)
        self.patient_count = 1  # For example, starting with the first patient
        self.data = self.load_data()

        # Split into train and test sets
        self.train, self.test = self.split_data(self.data)
        self.evaluator = Evaluator()

    def load_data(self):
        try:
            logging.info(f"Loading data for patient {self.patient_count} from {self.data_path}...")
            # Use the provided function to load one patient's data
            #data = self.process_function(self.patient_count, self.data_path)
            data = self.process_function(self.patient_count, self.data_path).astype(np.float32)

            preprocessed_data = data



            logging.info("Data loaded and preprocessed.")
            return preprocessed_data

        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise



    def load_model(self):
        device = "cpu"
        ckpt = torch.load(self.model_path, map_location=device)
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

        # Initialize the LagLlamaEstimator
        self.estimator = LagLlamaEstimator(
            ckpt_path=self.model_path,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            nonnegative_pred_samples=True,
            aug_prob=0,
            lr=self.learning_rate,
            device=torch.device('cpu'),
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            time_feat=estimator_args["time_feat"],
            rope_scaling={
                "type": "linear",
                "factor": max(1.0, (self.context_length + self.prediction_length) / estimator_args["context_length"]),
            },
            batch_size=64,
            num_parallel_samples=self.num_samples,
            trainer_kwargs={"max_epochs": 50}  # lightning trainer arguments
        )

        # Get the actual model (torch.nn.Module) from the estimator
        self.model = self.estimator.create_lightning_module()

        transformation = self.estimator.create_transformation()
        self.predictor  = self.estimator.create_predictor(transformation, self.model)

        return self.model


    def split_data(self, df, train_ratio=0.8):
        train_end = int(len(df) * train_ratio)
        train = PandasDataset(df.iloc[:train_end], freq="1S", target="hr")
        test = PandasDataset(df.iloc[train_end:], freq="1S", target="hr")


        return train, test

    def get_parameters(self):
        return [param.detach().cpu().numpy() for param in self.model.parameters()]


    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, device=self.device)
    """
    
    def fit(self, parameters, config):
        # 🔹 Load existing model if available
        model_dir = os.path.expanduser("saved_weights")
        round_number = config["round"]  # Assume the round number is passed in the config
        model_path = os.path.join(model_dir, f"round_{round_number}_context_{self.context_length}.pth")

        if os.path.exists(model_path):
            logging.info(f"Loading existing model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            logging.info("No existing model found. Training from scratch.")

        # Set parameters from the server
        self.set_parameters(parameters)

        logging.info("Training model...")
        self.model.train(True)

        # Fine-tune the model using the estimator
        self.estimator.train(
            self.train, self.test, cache_data=True, shuffle_buffer_length=1000
        )

        # Update the model after training
        self.model = self.estimator.create_lightning_module()
        transformation = self.estimator.create_transformation()
        self.predictor = self.estimator.create_predictor(transformation, self.model)

        # 🔹 Save the trained model for the current round
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        torch.save(self.model.state_dict(), model_path)
        logging.info(f"Model saved for round {round_number} at {model_path}")

        logging.info("Training complete.")

        return self.get_parameters(), len(self.train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        trainEnd = len(self.train)
        date_list = pd.date_range(self.data[trainEnd:].index[60], periods=9, freq="60s").tolist()

        forecasts = []
        tss = []

        for d in date_list:
            logging.info(f"Generating forecast for date: {d} with context length: {self.context_length}")

            forecast_it, ts_it = make_evaluation_predictions(
                dataset=PandasDataset(self.data[:d], freq="1S", target="hr"),
                predictor=self.predictor,
                num_samples=self.num_samples
            )

            forecast = list(forecast_it)
            ts = list(ts_it)

            forecasts.extend(forecast)
            tss.extend(ts)

        # Convert to numpy arrays
        forecasts_array = np.concatenate([f.mean for f in forecasts])
        tss_array = np.concatenate([ts.values for ts in tss])

        # Ensure shapes match
        if forecasts_array.shape != tss_array.shape:
            min_len = min(forecasts_array.shape[0], tss_array.shape[0])
            forecasts_array = forecasts_array[:min_len]
            tss_array = tss_array[:min_len]

        # Compute RMSE
        rmse = np.sqrt(np.mean((forecasts_array - tss_array) ** 2))

        # Compute loss using torch (example: MSE loss)
        forecasts_tensor = torch.tensor(forecasts_array, dtype=torch.float32)
        tss_tensor = torch.tensor(tss_array, dtype=torch.float32)

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(forecasts_tensor, tss_tensor).item()

        logging.info(f"RMSE: {rmse:.4f}")
        logging.info(f"Loss: {loss:.4f}")
        logging.info(f"Evaluation complete. Loss: {loss}")
        return float(loss), len(self.test), {"rmse": rmse}

    """
    """

    def fit(self, parameters, config):
        # Get current round from server config
        current_round = config.get("round", 1)

        # Dynamically increase the patient count (capped at 10)
        self.patient_count = min(current_round, 10)
        logging.info(f"Training with data for patient {self.patient_count}")

        # Set parameters from the server
        self.set_parameters(parameters)

        # Load data for the current patient
        self.data = self.load_data()

        # Split into train and test sets
        self.train, self.test = self.split_data(self.data)

        # If it's not the first round, load the weights from the previous round
        if current_round > 1:
            previous_round_model_path = os.path.expanduser(
                f"saved_weights/round_{current_round - 1}_context_{self.context_length}.pth")
            if os.path.exists(previous_round_model_path):
                logging.info(f"Loading model weights from round {current_round - 1}...")
                self.model.load_state_dict(torch.load(previous_round_model_path))
            else:
                logging.warning(
                    f"Model weights for round {current_round - 1} not found, starting with initialized model.")

        logging.info("Training model...")
        self.model.train(True)

        # Fine-tune the model using the estimator
        self.estimator.train(
            self.train, self.test, cache_data=True, shuffle_buffer_length=1000
        )

        # Update the model after training
        self.model = self.estimator.create_lightning_module()

        # Apply transformation and create predictor
        transformation = self.estimator.create_transformation()
        self.predictor = self.estimator.create_predictor(transformation, self.model)

        # Save the trained model for the current round
        model_dir = os.path.expanduser("saved_weights")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, f"round_{current_round}_context_{self.context_length}.pth")
        torch.save(self.model.state_dict(), model_path)
        logging.info(f"Model saved for round {current_round} at {model_path}")

        return self.get_parameters(), len(self.train), {}
        
    """

    def fit(self, parameters, config):
        current_round = config.get("round", 1)
        self.patient_count = min(current_round, 10) # for MIMIC
        #self.patient_count = min(current_round, 5)  # for Smartcare
        logging.info(f"Training with data for patient {self.patient_count}")
        self.set_parameters(parameters)

        self.data = self.load_data()
        self.train, self.test = self.split_data(self.data)

        if current_round > 1:
            previous_model_path = f"saved_weights/round_{current_round - 1}_context_{self.context_length}.pth"
            if os.path.exists(previous_model_path):
                logging.info(f"Loading model weights from round {current_round - 1}...")
                self.model.load_state_dict(torch.load(previous_model_path))

        logging.info("Training model...")
        self.model.train(True)

        # Start profiling
        start_time = time.time()
        tracemalloc.start()
        process = psutil.Process(os.getpid())
        start_cpu = process.cpu_percent(interval=None)

        # Train the model
        self.estimator.train(
            self.train, self.test, cache_data=True, shuffle_buffer_length=1000
        )

        # Stop profiling
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        cpu_usage = process.cpu_percent(interval=None)
        end_time = time.time()
        duration = end_time - start_time
        memory_mb = peak / 10 ** 6  # bytes to MB
        data_rate = len(self.train) / duration if duration > 0 else 0  # samples/sec

        self.model = self.estimator.create_lightning_module()
        transformation = self.estimator.create_transformation()
        self.predictor = self.estimator.create_predictor(transformation, self.model)

        model_dir = os.path.expanduser("saved_weights")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"round_{current_round}_context_{self.context_length}.pth")
        torch.save(self.model.state_dict(), model_path)

        # Log metrics
        logging.info(f"Training time: {duration:.2f}s")
        logging.info(f"Peak memory usage: {memory_mb:.2f} MB")
        logging.info(f"CPU usage: {cpu_usage:.2f}%")
        logging.info(f"Data rate: {data_rate:.2f} samples/sec")

        return self.get_parameters(), len(self.train), {
            "training_time_sec": duration,
            "peak_memory_mb": memory_mb,
            "cpu_usage_percent": cpu_usage,
            "data_rate_samples_per_sec": data_rate
        }

    """
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        # Evaluate using the current patient's data
        self.data = self.load_data()
        self.train, self.test = self.split_data(self.data)

        forecasts = []
        tss = []

        date_list = pd.date_range(self.data.index[60], periods=9, freq="60s").tolist()
        for d in date_list:
            logging.info(f"Generating forecast for date: {d} with context length: {self.context_length}")
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=PandasDataset(self.data[:d], freq="1S", target="hr"),
                predictor=self.predictor,
                num_samples=self.num_samples
            )
            forecast = list(forecast_it)
            ts = list(ts_it)
            forecasts.extend(forecast)
            tss.extend(ts)

        # Convert to numpy arrays
        forecasts_array = np.concatenate([f.mean for f in forecasts])
        tss_array = np.concatenate([ts.values for ts in tss])

        # Ensure shapes match
        if forecasts_array.shape != tss_array.shape:
            min_len = min(forecasts_array.shape[0], tss_array.shape[0])
            forecasts_array = forecasts_array[:min_len]
            tss_array = tss_array[:min_len]

        # Compute RMSE
        rmse = np.sqrt(np.mean((forecasts_array - tss_array) ** 2))

        # Compute loss using torch (example: MSE loss)
        # Ensure forecasts_tensor and tss_tensor are of the same dtype (torch.float32)
        forecasts_tensor = torch.tensor(forecasts_array, dtype=torch.float32)
        tss_tensor = torch.tensor(tss_array, dtype=torch.float32)

        # Compute loss using torch (example: MSE loss)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(forecasts_tensor, tss_tensor).item()

        # Convert loss and rmse to Python-native float types
        rmse = float(rmse)
        loss = float(loss)

        logging.info(f"RMSE: {rmse:.4f}")
        logging.info(f"Loss: {loss:.4f}")

        # Return values as Python-native types
        return loss, len(self.test), {"rmse": rmse}

  """

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        self.data = self.load_data()
        self.train, self.test = self.split_data(self.data)

        forecasts = []
        tss = []

        date_list = pd.date_range(self.data.index[60], periods=9, freq="60s").tolist()
        for d in date_list:
            logging.info(f"Generating forecast for date: {d} with context length: {self.context_length}")
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=PandasDataset(self.data[:d], freq="1S", target="hr"),
                predictor=self.predictor,
                num_samples=self.num_samples
            )
            forecasts.extend(list(forecast_it))
            tss.extend(list(ts_it))

        # Convert to numpy arrays
        forecasts_array = np.concatenate([f.mean for f in forecasts])
        tss_array = np.concatenate([ts.values for ts in tss])

        # Ensure arrays are 1D
        forecasts_array = np.squeeze(forecasts_array)
        tss_array = np.squeeze(tss_array)

        # Ensure shapes match
        if forecasts_array.shape != tss_array.shape:
            min_len = min(forecasts_array.shape[0], tss_array.shape[0])
            forecasts_array = forecasts_array[:min_len]
            tss_array = tss_array[:min_len]

        # Compute metrics
        rmse = np.sqrt(np.mean((forecasts_array - tss_array) ** 2))
        mae = np.mean(np.abs(forecasts_array - tss_array))

        # MAPE
        mape_mask = tss_array != 0
        mape = np.mean(np.abs((tss_array[mape_mask] - forecasts_array[mape_mask]) / tss_array[mape_mask])) * 100

        # SMAPE
        denominator = (np.abs(tss_array) + np.abs(forecasts_array)) / 2
        smape_mask = denominator != 0
        smape = np.mean(np.abs(forecasts_array[smape_mask] - tss_array[smape_mask]) / denominator[smape_mask]) * 100

        # Torch-based loss
        forecasts_tensor = torch.tensor(forecasts_array, dtype=torch.float32)
        tss_tensor = torch.tensor(tss_array, dtype=torch.float32)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(forecasts_tensor, tss_tensor).item()

        # Convert all metrics to float
        rmse = float(rmse)
        mae = float(mae)
        mape = float(mape)
        smape = float(smape)
        loss = float(loss)

        logging.info(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}% | SMAPE: {smape:.2f}% | Loss: {loss:.4f}")

        return loss, len(self.test), {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "smape": smape
        }


# Client Function
def client_fn(context: Context):
    """Initialize client with dataset path and training parameters."""
    dataset_path = context.node_config["dataset-path"]
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    context_length = context.run_config["context"]
    forecast_horizon = context.run_config["forecast-horizon"]
    print(f"Client initialized with batch size: {batch_size}")
    client = LagLlamaClient(
        model_path="fineTunedLLama.ckpt",
        data_path=dataset_path,
        forecast_horizon=forecast_horizon,
        context_length=context_length,
        process_function=load_data_from_disk_one_patient_per_round_lagLLama,
        #process_function=load_SmartCare_data_from_disk_one_patient_per_round_lagLLama,
        local_epochs=local_epochs,
        learning_rate=learning_rate
    )
    return client


# Flower ClientApp
logging.info("Starting the Flower ClientApp...")
app = ClientApp(client_fn)
logging.info("Flower ClientApp initialized.")
