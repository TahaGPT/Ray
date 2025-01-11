import os
import time
import ray
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from ray import tune
from ray.train import Trainer
from ray.train.keras import KerasTrainer
from ray.air.config import ScalingConfig, RunConfig, DatasetConfig

start = time.time()
ray.init(ignore_reinit_error=True)

# Preprocess function for the data
def preprocess_data(data):
    if 'milage' in data.columns:
        data['milage'] = data['milage'].apply(lambda x: int(re.sub(r'[^\d]', '', str(x))) if pd.notnull(x) else np.nan)
    if 'clean_title' in data.columns:
        data['clean_title'] = data['clean_title'].fillna('Unknown')
    def extract_hp(engine):
        match = re.search(r'(\d+)\.?\d*\s*HP', engine)
        return int(match.group(1)) if match else np.nan
    if 'engine' in data.columns:
        data['horsepower'] = data['engine'].apply(extract_hp)
    return data

# Load the training CSV file
train_file_path = 'train.csv'
train_data = pd.read_csv(train_file_path)
train_data_cleaned = preprocess_data(train_data)

X_initial = train_data_cleaned.drop(columns='price')
y_initial = train_data_cleaned['price']

# Scale the target variable
target_scaler = StandardScaler()
y_initial_scaled = target_scaler.fit_transform(y_initial.values.reshape(-1, 1)).flatten()

# Define the column transformer for preprocessing
categorical_features = ['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
numerical_features = ['model_year', 'milage', 'horsepower']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=7)),  # Using KNNImputer
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=3, include_bias=False))  # Adding polynomial features
        ]), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# Fit the preprocessor to the training data
X_initial_preprocessed = preprocessor.fit_transform(X_initial)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_initial_preprocessed, y_initial_scaled, test_size=0.2, random_state=25)

# Define a feedforward neural network model with added regularization
def create_ffnn_model(input_shape, learning_rate=0.000001, dropout_rate=0.2, l2_reg=0.0001):
    model = Sequential()
    model.add(Dense(9, activation='relu', input_shape=(input_shape,), kernel_regularizer='l2'))
    model.add(Dense(1, activation='linear'))  # Output layer with 1 unit for regression
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

# Prepare data for Ray
train_dataset = ray.data.from_pandas(pd.DataFrame(X_train))
val_dataset = ray.data.from_pandas(pd.DataFrame(X_val))

# Define the training function
def train_func(config):
    model = create_ffnn_model(X_train.shape[1], learning_rate=config["learning_rate"], 
                              dropout_rate=config["dropout_rate"], l2_reg=config["l2_reg"])
    model.fit(train_dataset.to_tf_dataset(), validation_data=val_dataset.to_tf_dataset(), 
              epochs=200, callbacks=[early_stopping], verbose=1)

# Hyperparameter tuning
param_grid = {
    'learning_rate': tune.grid_search([0.0000001, 0.00000001]),
    'dropout_rate': tune.grid_search([0.1, 0.2]),
    'l2_reg': tune.grid_search([0.0001, 0.00001])
}

# Setup Ray Trainer
trainer = KerasTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=param_grid,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=False),
    run_config=RunConfig(stop={"training_iteration": 1}, checkpoint_config={"checkpoint_at_end": True})
)

# Train the model
results = trainer.fit()
print("Best hyperparameters found were: ", results.get_best_config(metric="val_loss"))

# Load and preprocess the test data
test_file_path = 'test.csv'
test_data = pd.read_csv(test_file_path)
test_data_cleaned = preprocess_data(test_data)

# Separate features
ids = test_data_cleaned['id']
X_new = test_data_cleaned.drop(columns='id')
X_new_preprocessed = preprocessor.transform(X_new)

# Predict the prices for the new data
predicted_prices_scaled = model.predict(X_new_preprocessed)
predicted_prices = target_scaler.inverse_transform(predicted_prices_scaled).flatten()

# Create the output DataFrame
output = pd.DataFrame({'id': ids, 'price': predicted_prices})
# Save the predictions to a new CSV file
output_file_path = 'predicted_prices_lstm_test_03.csv'
output.to_csv(output_file_path, index=False)
output.head(), output_file_path

print("\n\nTotal Time Taken : ", time.time() - start)