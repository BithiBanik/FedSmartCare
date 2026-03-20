import os
import logging
import numpy as np
from typing import List, Tuple
import flwr as fl
from flwr.common import Context, Metrics, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from embeddedexample.task import LSTMModel, get_weights

# Configure logging
logging.basicConfig(
    filename="server.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Create a directory to store global weights
WEIGHTS_DIR = "global_weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Define metric aggregation function
"""
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    rmses = []
    examples = []
    for num_examples, m in metrics:
        if "rmse" in m:
            rmses.append(num_examples * m["rmse"])
        else:
            logging.warning("RMSE metric missing for a client. Using 0 as default.")
            rmses.append(0)
        examples.append(num_examples)
    
    weighted_rmse = sum(rmses) / sum(examples) if sum(examples) > 0 else 0
    logging.info("Weighted RMSE=%.4f", weighted_rmse)
    return {"rmse": weighted_rmse}
    
"""
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    weighted_sums = {"rmse": 0.0, "mae": 0.0, "mape": 0.0, "smape": 0.0}
    total_examples = 0

    for num_examples, m in metrics:
        total_examples += num_examples
        for key in weighted_sums.keys():
            value = m.get(key, 0.0)
            if key not in m:
                logging.warning(f"{key.upper()} metric missing for a client. Using 0 as default.")
            weighted_sums[key] += num_examples * value

    if total_examples == 0:
        return {k: 0.0 for k in weighted_sums}

    weighted_avg = {k: v / total_examples for k, v in weighted_sums.items()}
    logging.info("Weighted Metrics: %s", weighted_avg)
    return weighted_avg


# Function to save global model weights
def save_global_model_weights(parameters, round_num):
    weights = parameters_to_ndarrays(parameters)  # Convert parameters to numpy arrays
    save_path = os.path.join(WEIGHTS_DIR, f"global_weights_round_{round_num}.npz")
    np.savez(save_path, *weights)
    logging.info(f"Saved global model weights for round {round_num}: {save_path}")

# Define the CustomStrategy class
class CustomStrategy(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"{WEIGHTS_DIR}/round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics

    """
    def aggregate_evaluate(self, server_round, results, failures):
        #Aggregate evaluation rmse using weighted average.
        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh rmse of each client by number of examples used
        rmses = [r.metrics["rmse"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_rmse = sum(rmses) / sum(examples)
        print(f"Round {server_round} rmse aggregated from client results: {aggregated_rmse}")

        # Return aggregated loss and metrics (i.e., aggregated rmse)
        return aggregated_loss, {"rmse": aggregated_rmse}
    """
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation metrics using weighted average."""
        if not results:
            return None, {}

        # Aggregate loss from base class
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)

        # Collect all metrics and number of examples
        metrics = [(r.num_examples, r.metrics) for _, r in results]

        # Use custom weighted average function
        aggregated_metrics = weighted_average(metrics)

        # Print each aggregated metric
        print(f"Round {server_round} aggregated metrics:")
        for key, value in aggregated_metrics.items():
            print(f"  {key}: {value:.4f}")

        return aggregated_loss, aggregated_metrics


# Server function
def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    logging.info("Server initialization started.")
    
    # Read number of rounds from the config
    num_rounds = context.run_config["num-server-rounds"] 
    logging.info("Configured to run for %d rounds.", num_rounds)
    forecast_horizon = context.run_config["forecast-horizon"]
    # Initialize model parameters
    ndarrays = get_weights(LSTMModel(input_size=1, hidden_size=50, num_layers=1, forecast_horizon=forecast_horizon))

    parameters = ndarrays_to_parameters(ndarrays)

    # Define `on_fit_config_fn` to send `round` info to clients
    def fit_config(server_round: int):
        return {"round": server_round, "total_rounds": num_rounds}

    # Define the fit_round_end callback to save global weights
    def fit_round_end(server_round: int, parameters, metrics):
        save_global_model_weights(parameters, server_round)

    # Define the strategy with fit_round_end callback
    strategy = CustomStrategy(
        fraction_fit=1.0,
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=2,
        initial_parameters=parameters,
        on_fit_config_fn=fit_config,
    )

    # Server configuration
    config = ServerConfig(num_rounds=num_rounds)
    logging.info("Server configuration completed.")

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
