import torch
import numpy as np
import matplotlib.pyplot as plt
from task import LSTMModel, set_weights, test
import os
from preprocess import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error

EXPERIMENT_PATH= "output/exp-1"
# Path to stored global weights
WEIGHTS_DIR = f"{EXPERIMENT_PATH}/global_weights" # for global weights
#WEIGHTS_DIR = "savedClientWeights"
TEST_HR_DATASET_PATH = "MIMIC/user_1/RR.csv"  # Update this path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_step=30
forecast_horizon=30
# Load test dataset
X_train, X_test, y_train, y_test, scaler = preprocessing(TEST_HR_DATASET_PATH,time_step,forecast_horizon)  # Adjust as needed
#test_data = X_test
#test_tensor = torch.tensor(test_data, dtype=torch.float32)
# Convert to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Define LSTM model
model = LSTMModel( input_size=1, hidden_size=50, num_layers=1, forecast_horizon=30)
model.to(device)
criterion = torch.nn.MSELoss()


# Store RMSE values
rmses = []
global predictionsForPlot
global true_valuesForPlot
# Iterate through all saved rounds
for round_num in range(1, 11):  # Assuming 10 rounds
    weight_path = os.path.join(WEIGHTS_DIR, f"round-{round_num}-weights.npz") ## loading global weights
    #weight_path = os.path.join(WEIGHTS_DIR, f"round_{round_num}_weights.pth")  ## loading local weights

    if not os.path.exists(weight_path):
        print(f"Skipping missing weight file: {weight_path}")
        continue

    # Load global weights
    weights = np.load(weight_path)
    set_weights(model, [weights[key] for key in weights.files])  # Convert to model format

    # load local weights
    #weights = torch.load(weight_path, map_location=device)
    #model.load_state_dict(weights)  # Directly load state_dict into the model

    # Move to device
    model.to(device)
    model.eval()
    # Run predictions
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()
        true_values = y_test_tensor.cpu().numpy()

    # Invert normalization
    predictions = scaler.inverse_transform(predictions)
    true_values = scaler.inverse_transform(true_values)


    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    mae = mean_absolute_error(true_values, predictions)

    # Append RMSE to the list
    rmses.append(rmse)

    print(f"Round {round_num}: RMSE = {rmse:.4f}")
    predictionsForPlot  = predictions
    true_valuesForPlot  = true_values
# **3. Plot RMSE per Round**
plt.plot(range(1, len(rmses) + 1), rmses, marker='o', linestyle='-', label="RMSE per Round")
plt.xlabel("Round")
plt.ylabel("RMSE")
plt.title("RMSE over Federated Rounds")
plt.legend()
plt.grid(True)
plt.savefig("rmse_plot.png")  # Save the RMSE plot
plt.show()

# **4. Show Prediction from Last Round**
last_round = 10



# Plot results
plt.figure(figsize=(12, 6))
plt.plot(true_valuesForPlot[:500, 0], label='True Heart Rate', color='blue')
plt.plot(predictionsForPlot[:500, 0], label='Predicted Heart Rate', color='red')
plt.xlabel('Time')
plt.ylabel('Heart Rate')
plt.title('Actual vs Predicted Heart Rate')
plt.legend()
plt.grid(True)
plt.savefig("last_round_prediction.png", dpi=300, bbox_inches="tight")
plt.show()

