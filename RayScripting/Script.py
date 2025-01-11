import os
import time
import ray
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import RandomizedSearchCV

start = time.time()
# ray.init(include_dashboard = True)
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
# def create_ffnn_model(input_shape, learning_rate=0.000001, dropout_rate=0.2, l2_reg=0.0001):
#     model = Sequential()
#     model.add(Dense(9, activation='relu', input_shape=(input_shape,), kernel_regularizer='l2'))
#     # model.add(Dense(5, activation='relu',kernel_regularizer='l2'))
#     # model.add(Dense(1))  # Output layer
#     model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
#     return model
# Define a feedforward neural network model with added regularization

def create_ffnn_model(input_shape, learning_rate=0.000001, dropout_rate=0.2, l2_reg=0.0001):
    model = Sequential()
    model.add(Dense(9, activation='relu', input_shape=(input_shape,), kernel_regularizer='l2'))
    model.add(Dense(1, activation='linear'))  # Output layer with 1 unit for regression
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# Hyperparameter tuning
param_grid = {
    'learning_rate': [0.0000001, 0.00000001],
    'dropout_rate': [0.1, 0.2],
    'l2_reg': [0.0001, 0.00001]
}
# Randomized search for hyperparameter tuning
def create_model(learning_rate, dropout_rate, l2_reg):
    return create_ffnn_model(input_shape=X_train.shape[1], learning_rate=learning_rate, dropout_rate=dropout_rate, l2_reg=l2_reg)

random_search = RandomizedSearchCV(
    estimator=Sequential(),
    param_distributions=param_grid,
    n_iter=15,
    cv=10,
    verbose=1,
    n_jobs=-3,
    random_state=25,
    scoring='neg_mean_squared_error'
)

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

# Train the model with early stopping
model = create_ffnn_model(X_train.shape[1])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, verbose=1, callbacks=[early_stopping])

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