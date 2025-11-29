from sklearn.datasets import fetch_california_housing
import pandas as pd
import os
# Load dataset
california = fetch_california_housing(as_frame=True)
df = california.frame  # Data + target combined



# Ensure the data folder exists
os.makedirs("../data", exist_ok=True)

# Save dataset to CSV in the correct folder
df.to_csv("../data/raw_data.csv", index=False)
print("Dataset saved to data/raw_data.csv")
