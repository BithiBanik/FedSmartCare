import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, f1_score, matthews_corrcoef, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, Dense, MaxPooling1D, GlobalAveragePooling1D, Flatten




import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Add, MaxPooling1D, GlobalAveragePooling1D, \
    Dense
from tensorflow.keras import Model
import logging
from flwr.client import NumPyClient

# Configure logging
logging.basicConfig(
    filename='client.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



class StressLevelCNN(tf.keras.Sequential):
    def __init__(self, input_size):
        super(StressLevelCNN, self).__init__()
        self.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation="relu", padding="same",
                                        input_shape=(1, input_size)))
        self.add(tf.keras.layers.Dense(16, activation="relu"))
        self.add(tf.keras.layers.MaxPooling1D(padding="same"))
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(3, activation="softmax"))




class StressLevelMLP(tf.keras.Sequential):
    def __init__(self, input_size):
        super(StressLevelMLP, self).__init__()
        # Define layers without using the Input layer
        self.add(tf.keras.layers.Dense(128, activation="relu", input_dim=input_size))  # Specify input_dim directly
        self.add(tf.keras.layers.Dropout(0.3))
        self.add(tf.keras.layers.Dense(64, activation="relu"))
        self.add(tf.keras.layers.Dense(32, activation="relu"))
        self.add(tf.keras.layers.Dense(3, activation="softmax"))  # Output layer for 3-class classification



class ResNet:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def resnet_block(self, input_tensor, filters, kernel_size=3, strides=1):
        x = Conv1D(filters, kernel_size, strides=strides, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv1D(filters, kernel_size, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)

        shortcut = Conv1D(filters, kernel_size=1, strides=strides, padding='same')(input_tensor)
        shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)

        return x

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        x = Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

        x = self.resnet_block(x, 64)
        x = self.resnet_block(x, 64)

        x = self.resnet_block(x, 128)
        x = self.resnet_block(x, 128)

        x = Flatten()(x)
        x = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs, x)
        return model

    def compile_model(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def get_model(self):
        return self.model



def load_data(train_path, test_path):
    # Load train and test datasets
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Concatenate train and test datasets
    swell = pd.concat([train, test], axis=0).reset_index(drop=True)

    # Encode target variable
    encoder = LabelEncoder()
    y = encoder.fit_transform(swell['condition'])

    # Feature selection using ANOVA F-value
    X = swell.drop('condition', axis=1).to_numpy()
    best_features = SelectKBest(score_func=f_classif, k='all')
    X = best_features.fit_transform(X, y)

    # Get the top 34 features
    anova = pd.Series(data=best_features.scores_, index=swell.columns[:-1]).sort_values(ascending=False)
    feature_names = anova.index[:34]

    # Keep only the top 34 features
    X = swell[feature_names].to_numpy()

    # Scale features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Reshape for Conv1D input (samples, time steps, features)
    #X = X.reshape(X.shape[0], 1, X.shape[1])

    print(f"Before Reshape: X.shape = {X.shape}")  # Debugging

    # Ensure X has at least 2 dimensions
    if len(X.shape) == 2:
        X = X.reshape(X.shape[0], 1, X.shape[1])  # Reshape to (samples, time steps, features)
    elif len(X.shape) == 1:
        X = X.reshape(X.shape[0], 1, 1)  # Handle edge case of a 1D array

    print(f"After Reshape: X.shape = {X.shape}")  # Debugging

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

    return X_train, X_test, y_train, y_test, feature_names.size

def load_data_from_disk(path, batch_size):
    # Read the CSV dataset
    swell = pd.read_csv(path)

    # Encode target variable
    encoder = LabelEncoder()
    y = encoder.fit_transform(swell['condition'])

    # Feature selection using ANOVA F-value
    X = swell.drop('condition', axis=1).to_numpy()
    best_features = SelectKBest(score_func=f_classif, k='all')
    X = best_features.fit_transform(X, y)

    # Get the top 34 features
    anova = pd.Series(data=best_features.scores_, index=swell.columns[:-1]).sort_values(ascending=False)
    feature_names = anova.index[:34]

    # Keep only the top 34 features
    X = swell[feature_names].to_numpy()

    # Scale features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Reshape for Conv1D input (samples, time steps, features)
    #X = X.reshape(X.shape[0], 1, X.shape[1])

    # Reshape for ResNet input (samples, time steps, features)
    #X = X.reshape(X.shape[0], X.shape[1], 1)

    # Reshape for MLP input (samples, features)
    X = X.reshape(X.shape[0], X.shape[1])


    # Split data into train (80%), validation (10%), test (10%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=42) # First split
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42) # Second split

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).shuffle(len(X_train))
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    input_feature_size = 34

    return train_dataset, valid_dataset, test_dataset, input_feature_size



def set_weights(model, weights):
    for layer, weight in zip(model.weights, weights):
        layer.assign(weight)


def get_weights(model):
    return [w.numpy() for w in model.weights]


def train(model, train_dataset, valid_dataset, epochs, lr,  batch_size):


    early_stopping = EarlyStopping(patience=10)

    X_train, y_train = zip(*[(x.numpy(), y.numpy()) for x, y in train_dataset])
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    X_valid, y_valid = zip(*[(x.numpy(), y.numpy()) for x, y in valid_dataset])
    X_valid = np.concatenate(X_valid)
    y_valid = np.concatenate(y_valid)

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_valid, y_valid),
        callbacks=[early_stopping]
    )
    end_time = time.time()

    training_time = end_time - start_time
    final_loss = history.history['loss'][-1]

    return {"loss": final_loss, "training_time": training_time}


def test(model, test_dataset):

    X_test, y_test = zip(*[(x.numpy(), y.numpy()) for x, y in test_dataset])
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    loss = model.evaluate(X_test, y_test, verbose=0)[0]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')

    return loss, accuracy, precision, f1, mcc, recall

