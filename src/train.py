import os
import pickle

import pandas as pd
from surprise import SVD, KNNBasic, NMF, Dataset, Reader
from surprise.model_selection import GridSearchCV, cross_validate

print("--- Starting Collaborative Filtering Model Training ---")

# --- 1. Define Paths & Load Training Data ---
base_path = os.getcwd()
processed_path = os.path.join(base_path, "data", "processed")
models_path = os.path.join(base_path, "models")
model_filepath = os.path.join(models_path, "collaborative_model.pkl")

train_df_path = os.path.join(processed_path, "train_ratings.pkl")

try:
    train_df = pd.read_pickle(train_df_path)
    print(f"Successfully loaded training data from {train_df_path}")
    print(f"Training data shape: {train_df.shape}")
except FileNotFoundError:
    print(f"ERROR: Training data not found at {train_df_path}")
    exit()

# Load data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_df[["user_id", "blog_id", "ratings"]], reader)


# --- 2. Compare Base Algorithms on Training Data ---
print("\n--- Step 1: Comparing Base Algorithms (using 3-fold CV) ---")
algorithms = {
    "SVD": SVD(random_state=42),
    "NMF": NMF(random_state=42),
    "Item-Based KNN": KNNBasic(sim_options={"user_based": False}),
}

results = []
for name, algo in algorithms.items():
    cv_results = cross_validate(algo, data, measures=["RMSE"], cv=3, verbose=False)
    mean_rmse = cv_results["test_rmse"].mean()
    results.append({"Model": name, "RMSE": mean_rmse})
    print(f"Model: {name}, Average RMSE on training data: {mean_rmse:.4f}")

# Select the best model based on the lowest RMSE
best_model_info = min(results, key=lambda x: x["RMSE"])
best_model_name = best_model_info["Model"]
print(f"\nBest performing model on CV: {best_model_name}")


# --- 3. Hyperparameter Tuning for the Best Model ---
print(f"\n--- Step 2: Hyperparameter Tuning for {best_model_name} ---")
param_grids = {
    "SVD": {
        "n_epochs": [10, 20],
        "lr_all": [0.005, 0.01],
        "reg_all": [0.02, 0.1],
        "n_factors": [50, 100],
    },
    "NMF": {"n_epochs": [20, 30], "n_factors": [15, 20], "reg_pu": [0.06, 0.1]},
    "Item-Based KNN": {"k": [20, 40], "sim_options": {"name": ["msd", "cosine"]}},
}

selected_param_grid = param_grids[best_model_name]
algo_class = type(algorithms[best_model_name])

gs = GridSearchCV(algo_class, selected_param_grid, measures=["rmse"], cv=3)
gs.fit(data)

print(f"Best RMSE score from tuning: {gs.best_score['rmse']:.4f}")
print("Best parameters found:", gs.best_params["rmse"])


# --- 4. Train and Save the Final, Best Model ---
print("\n--- Step 3: Training Final Model on Full Training Data ---")
final_algo = gs.best_estimator["rmse"]

# Build the training set from the entire training data for final model
full_trainset = data.build_full_trainset()
final_algo.fit(full_trainset)
print("Final model training complete.")

# Save the trained algorithm and the trainset object
model_to_save = {
    "name": best_model_name,
    "algorithm": final_algo,
    "trainset": full_trainset,
}

os.makedirs(models_path, exist_ok=True)
with open(model_filepath, "wb") as f:
    pickle.dump(model_to_save, f)

print(f"\nSuccessfully saved the best model to: {model_filepath}")
print("--- Training Script Finished ---")