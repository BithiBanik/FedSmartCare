import os
import logging
import numpy as np
from typing import List, Tuple
import flwr as fl
from flwr.common import Context, Metrics, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from embeddedexample.task import StressLevelCNN, get_weights, StressLevelMLP, ResNet


# Configure logging
logging.basicConfig(
    filename="server.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Create a directory to store global weights
WEIGHTS_DIR = "global_weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Define metric aggregation function for classification
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    mcc_scores = []
    examples = []

    for num_examples, m in metrics:
        if "accuracy" in m:
            accuracies.append(num_examples * m["accuracy"])
            precisions.append(num_examples * m["precision"])
            recalls.append(num_examples * m["recall"])
            f1_scores.append(num_examples * m["f1_score"])
            mcc_scores.append(num_examples * m["mcc"])
        else:
            logging.warning("Some metrics are missing for a client. Using 0 as default.")
            accuracies.append(0)
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
            mcc_scores.append(0)

        examples.append(num_examples)

    total_examples = sum(examples)

    if total_examples == 0:
        logging.warning("No examples received for aggregation.")
        return {}

    # Compute weighted average
    aggregated_metrics = {
        "accuracy": sum(accuracies) / total_examples,
        "precision": sum(precisions) / total_examples,
        "recall": sum(recalls) / total_examples,
        "f1_score": sum(f1_scores) / total_examples,
        "mcc": sum(mcc_scores) / total_examples,
    }

    logging.info("Aggregated metrics: %s", aggregated_metrics)
    return aggregated_metrics

# Function to save global model weights
def save_global_model_weights(parameters, round_num):
    weights = parameters_to_ndarrays(parameters)
    save_path = os.path.join(WEIGHTS_DIR, f"global_weights_round_{round_num}.npz")
    np.savez(save_path, *weights)
    logging.info(f"Saved global model weights for round {round_num}: {save_path}")

# Define the CustomStrategy class
class CustomStrategy(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Call aggregate_fit from base class (FedAvg)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            np.savez(f"{WEIGHTS_DIR}/round-{server_round}-weights.npz", *aggregated_ndarrays)
            logging.info(f"Saved round {server_round} aggregated weights.")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation metrics using weighted average."""
        if not results:
            logging.warning("No results received for evaluation.")
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg)
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Aggregate classification metrics: accuracy, precision, recall, f1_score, and mcc
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        mcc_scores = []
        examples = []

        for _, r in results:
            if "accuracy" in r.metrics:
                accuracies.append(r.metrics["accuracy"] * r.num_examples)
            else:
                logging.warning(f"Missing accuracy for client in round {server_round}.")

            if "precision" in r.metrics:
                precisions.append(r.metrics["precision"] * r.num_examples)
            else:
                logging.warning(f"Missing precision for client in round {server_round}.")

            if "recall" in r.metrics:
                recalls.append(r.metrics["recall"] * r.num_examples)
            else:
                logging.warning(f"Missing recall for client in round {server_round}.")

            if "f1_score" in r.metrics:
                f1_scores.append(r.metrics["f1_score"] * r.num_examples)
            else:
                logging.warning(f"Missing f1_score for client in round {server_round}.")

            if "mcc" in r.metrics:
                mcc_scores.append(r.metrics["mcc"] * r.num_examples)
            else:
                logging.warning(f"Missing mcc for client in round {server_round}.")

            examples.append(r.num_examples)

        if sum(examples) > 0:
            aggregated_accuracy = sum(accuracies) / sum(examples)
            aggregated_precision = sum(precisions) / sum(examples)
            aggregated_recall = sum(recalls) / sum(examples)
            aggregated_f1 = sum(f1_scores) / sum(examples)
            aggregated_mcc = sum(mcc_scores) / sum(examples)

            logging.info(f"Round {server_round} aggregated metrics from client results:")
            logging.info(f"Accuracy: {aggregated_accuracy}")
            logging.info(f"Precision: {aggregated_precision}")
            logging.info(f"Recall: {aggregated_recall}")
            logging.info(f"F1 Score: {aggregated_f1}")
            logging.info(f"MCC: {aggregated_mcc}")

            # Update aggregated_metrics with new values
            aggregated_metrics["accuracy"] = aggregated_accuracy
            aggregated_metrics["precision"] = aggregated_precision
            aggregated_metrics["recall"] = aggregated_recall
            aggregated_metrics["f1_score"] = aggregated_f1
            aggregated_metrics["mcc"] = aggregated_mcc
        else:
            logging.warning(f"Round {server_round}: No classification metrics to aggregate.")

        return aggregated_loss, aggregated_metrics


# Server function
def server_fn(context: Context):
    logging.info("Server initialization started.")


    # Read number of rounds from the config
    num_rounds = context.run_config["num-server-rounds"]
    logging.info("Configured to run for %d rounds.", num_rounds)

    # Initialize model parameters using CNN
    input_size = 34  # Assuming 34 features for classification
    cnn_model = StressLevelCNN(input_size)
    #resNet = ResNet(input_shape=(input_size, 1), num_classes=3).get_model()
    mlp = StressLevelMLP(input_size)
    #ndarrays = get_weights(cnn_model)
    #ndarrays = get_weights(resNet)
    ndarrays = get_weights(mlp)
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
