import os
import pickle

import pandas as pd
from surprise import accuracy

print("--- Evaluating Final Model on Test Data ---")

# --- 1. Define Paths & Load Model and Test Data ---
base_path = os.getcwd()
models_path = os.path.join(base_path, "models")
processed_path = os.path.join(base_path, "data", "processed")
model_filepath = os.path.join(models_path, "collaborative_model.pkl")
test_df_path = os.path.join(processed_path, "test_ratings.pkl")

try:
    with open(model_filepath, "rb") as f:
        saved_model = pickle.load(f)
        algo = saved_model["algorithm"]
    print(f"Successfully loaded model from {model_filepath}")
except FileNotFoundError:
    print(f"ERROR: Model not found at {model_filepath}. Please run train.py first.")
    exit()

try:
    test_df = pd.read_pickle(test_df_path)
    print(f"Successfully loaded test data from {test_df_path}")
except FileNotFoundError:
    print(f"ERROR: Test data not found at {test_df_path}")
    exit()

# --- 2. Evaluate the Model ---
testset = list(test_df[["user_id", "blog_id", "ratings"]].itertuples(
    index=False, name=None
))
predictions = algo.test(testset)


# --- 3. Report Final Performance ---
print("\n--- Final Model Performance on Unseen Test Data ---")
accuracy.rmse(predictions)
accuracy.mae(predictions)
print("--- Evaluation Script Finished ---")