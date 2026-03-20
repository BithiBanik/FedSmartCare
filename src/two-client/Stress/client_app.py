import logging
import tensorflow as tf
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from embeddedexample.task import (
    StressLevelCNN, StressLevelMLP, ResNet,
    get_weights,
    load_data_from_disk,
    set_weights,
    test,
    train,
)
import time
import tracemalloc
import psutil
import os


# Configure logging
logging.basicConfig(
    filename='client.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, trainloader,  valloader, testloader, local_epochs, learning_rate, batch_size, input_feature_size):
        logger.info("Initializing FlowerClient")
        #self.model = StressLevelCNN(input_feature_size)
        #self.model = ResNet(input_shape=(input_feature_size, 1), num_classes=3).get_model()

        self.model = StressLevelMLP(input_feature_size)

        self.trainloader = trainloader
        self.testloader = testloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.batch_size = batch_size
        logger.info(f"Learning rate set to {self.lr}")

        # Compile model once during initialization
        self._compile_model()

    def _compile_model(self):
        logger.info("Compiling model")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

    def on_fit_config(self, config):
        logger.info("Configuring model before fit")
        # Update learning rate if it changes between rounds
        self.lr = config.get("lr", 0.001)  # Set a lower default if not provided

        self._compile_model()

    """
    def fit(self, parameters, config):
        logger.info("Starting training")

        set_weights(self.model, parameters)
        results = train(
            self.model,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.batch_size,
        )
        num_samples = sum(1 for _ in self.trainloader.unbatch())

        logger.info(f"Training completed with results: {results}")
        return get_weights(self.model), num_samples, results
    """

    def fit(self, parameters, config):
        logger.info("Starting training")

        # Start profiling
        start_time = time.time()
        tracemalloc.start()
        process = psutil.Process(os.getpid())
        start_cpu = process.cpu_percent(interval=None)

        # Set model weights and train
        set_weights(self.model, parameters)
        results = train(
            self.model,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.batch_size,
        )

        # Stop profiling
        training_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        end_cpu = process.cpu_percent(interval=None)
        tracemalloc.stop()

        # Compute number of samples
        num_samples = sum(1 for _ in self.trainloader.unbatch())

        # Compute data rate
        data_rate = num_samples / training_time if training_time > 0 else 0.0

        # Log and update results
        logger.info(f"Training completed with results: {results}")
        logger.info(
            f"Profiling - Time: {training_time:.4f}s, Memory (current/peak): {current / 1024:.2f}KB/{peak / 1024:.2f}KB, "
            f"CPU Usage: {end_cpu}%, Data Rate: {data_rate:.2f} samples/sec"
        )

        # Attach profiling info to results
        results.update({
            "training_time_sec": float(training_time),
            "peak_memory_kb": float(peak / 1024),
            "cpu_usage_percent": float(end_cpu),
            "data_rate_samples_per_sec": float(data_rate),
        })

        return get_weights(self.model), num_samples, results

    def evaluate(self, parameters, config):
        logger.info("Starting evaluation")
        set_weights(self.model, parameters)
        loss, accuracy, precision, f1, mcc, recall = test(self.model, self.testloader)
        logger.info(
            f"Evaluation results - Loss: {loss}, Accuracy: {accuracy}, Precision: {precision}, F1: {f1}, MCC: {mcc}, Recall: {recall}")

        num_samples = sum(1 for _ in self.testloader.unbatch())  # Corrected dataset

        return loss, num_samples, {
            "accuracy": accuracy,
            "precision": precision,
            "f1_score": f1,
            "mcc": mcc,
            "recall": recall,
        }


def client_fn(context: Context):
    logger.info("Initializing client function")

    # Read config
    dataset_path = context.node_config["dataset-path"]
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Batch size: {batch_size}, Local epochs: {local_epochs}, Learning rate: {learning_rate}")

    trainloader,  valloader, testloader, input_feature_size = load_data_from_disk(dataset_path, batch_size)

    return FlowerClient(trainloader,  valloader, testloader, local_epochs, learning_rate, batch_size, input_feature_size).to_client()

# Flower ClientApp
app = ClientApp(client_fn)
