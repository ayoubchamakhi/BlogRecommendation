import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from surprise import (
    NMF,
    SVD,
    Dataset,
    KNNBasic,
    Reader,
)
from surprise.model_selection import (
    GridSearchCV,
    cross_validate,
)

print("--- Starting Collaborative Filtering " "Model Training ---")

# 1. Define paths and load training data
base_path = os.getcwd()
processed_path = os.path.join(
    base_path,
    "data",
    "processed",
)
models_path = os.path.join(
    base_path,
    "models",
)
results_path = os.path.join(
    base_path,
    "results",
)
os.makedirs(results_path, exist_ok=True)

model_filepath = os.path.join(
    models_path,
    "collaborative_model.pkl",
)
train_df_path = os.path.join(
    processed_path,
    "train_ratings.pkl",
)

try:
    train_df = pd.read_pickle(train_df_path)
    print("Successfully loaded training data from " f"{train_df_path}")
    print("Training data shape: " f"{train_df.shape}")
except FileNotFoundError:
    print("ERROR: Training data not found at " f"{train_df_path}")
    exit()

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    train_df[["user_id", "blog_id", "ratings"]],
    reader,
)

print("\n--- Step 1: Comparing Base Algorithms " "(using 3-fold CV) ---")
algorithms = {
    "SVD": SVD(random_state=42),
    "NMF": NMF(random_state=42),
    "Item-Based KNN": KNNBasic(sim_options={"user_based": False}),
}

results = []
for name, algo in algorithms.items():
    cv_results = cross_validate(
        algo,
        data,
        measures=["RMSE"],
        cv=3,
        verbose=False,
    )
    mean_rmse = cv_results["test_rmse"].mean()
    results.append({"Model": name, "RMSE": mean_rmse})
    print("Model: " f"{name}, Average RMSE: " f"{mean_rmse:.4f}")

# Plot CV results
df_results = pd.DataFrame(results)
ax = df_results.plot.bar(
    x="Model",
    y="RMSE",
    title="CV RMSE by Algorithm",
)
ax.set_ylabel("RMSE")
fig = ax.get_figure()
fig.savefig(
    os.path.join(
        results_path,
        "cv_rmse_comparison.png",
    )
)
plt.close(fig)

best_model_info = min(
    results,
    key=lambda x: x["RMSE"],
)
best_model_name = best_model_info["Model"]
print("\nBest performing model on CV: " f"{best_model_name}")

print("\n--- Step 2: Hyperparameter Tuning for " f"{best_model_name} ---")
param_grids = {
    "SVD": {
        "n_epochs": [10, 20],
        "lr_all": [0.005, 0.01],
        "reg_all": [0.02, 0.1],
        "n_factors": [50, 100],
    },
    "NMF": {
        "n_epochs": [20, 30],
        "n_factors": [15, 20],
        "reg_pu": [0.06, 0.1],
    },
    "Item-Based KNN": {
        "k": [20, 40],
        "sim_options": {"name": ["msd", "cosine"]},
    },
}

selected_param_grid = param_grids[best_model_name]
algo_class = type(algorithms[best_model_name])

gs = GridSearchCV(
    algo_class,
    selected_param_grid,
    measures=["rmse"],
    cv=3,
)
gs.fit(data)
print("Best RMSE from tuning: " f"{gs.best_score['rmse']:.4f}")
print("Best parameters: " f"{gs.best_params['rmse']}")

# Save grid search results
df_gs = pd.DataFrame(gs.cv_results)
df_gs.to_csv(
    os.path.join(
        results_path,
        "grid_search_results.csv",
    ),
    index=False,
)

print("\n--- Step 3: Training Final Model " "on Full Training Data ---")
final_algo = gs.best_estimator["rmse"]

full_trainset = data.build_full_trainset()
final_algo.fit(full_trainset)
print("Final model training complete.")

model_to_save = {
    "name": best_model_name,
    "algorithm": final_algo,
    "trainset": full_trainset,
}
os.makedirs(models_path, exist_ok=True)
with open(model_filepath, "wb") as f:
    pickle.dump(
        model_to_save,
        f,
    )

print("\nSuccessfully saved the best model to: " f"{model_filepath}")
print("--- Training Script Finished ---")
