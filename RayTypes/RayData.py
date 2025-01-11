import ray
import time
import numpy as np
from typing import Dict
from sklearn.datasets import load_iris
import pandas as pd

# Initialize Ray
ray.init()
startTime = time.time()
# Load the Iris dataset
data = load_iris(as_frame=True)
df = data.frame

# Create a Ray Dataset from the DataFrame
ds = ray.data.from_pandas(df)

# Define the transformation function
def transform_batch(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    vec_a = batch["petal length (cm)"]
    vec_b = batch["petal width (cm)"]
    batch["petal area (cm^2)"] = vec_a * vec_b
    return batch

# Apply the transformation
transformed_ds = ds.map_batches(transform_batch)

# Materialize and print the transformed dataset
print(transformed_ds.materialize())
print("Time Taken :", time.time() - startTime)
# Shutdown Ray
ray.shutdown()