import math
import os
import pickle
from collections import defaultdict

import pandas as pd


def precision_recall_at_k(predictions, k=10, threshold=3):
    """
    Return precision and recall at k metrics for each user.
    This version correctly evaluates the quality of the ranked list.
    """
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum(true_r >= threshold for _, true_r in user_ratings)
        n_rel_and_rec_k = sum(true_r >= threshold for _, true_r in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / k
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    mean_precision = sum(prec for prec in precisions.values()) / len(precisions)
    mean_recall = sum(rec for rec in recalls.values()) / len(recalls)
    return mean_precision, mean_recall


print("--- Evaluating Final Model on Test Data ---")

# --- 1. Load Model and Data ---
base_path = os.getcwd()
models_path = os.path.join(base_path, "models")
processed_path = os.path.join(base_path, "data", "processed")
model_filepath = os.path.join(models_path, "collaborative_model.pkl")
test_df_path = os.path.join(processed_path, "test_ratings.pkl")

try:
    with open(model_filepath, "rb") as f:
        saved_model = pickle.load(f)
        algo = saved_model["algorithm"]
    print("Successfully loaded model from " + model_filepath)
except FileNotFoundError:
    print(
        "ERROR: Model not found at " + model_filepath + ". Please run train.py first."
    )
    exit()

try:
    test_df = pd.read_pickle(test_df_path)
    print("Successfully loaded test data from " + test_df_path)
except FileNotFoundError:
    print("ERROR: Test data not found at " + test_df_path)
    exit()

# --- 2. Generate Predictions ---
if "ratings" not in test_df.columns:
    print("ERROR: 'Rating' column not found in test_df. " "Please check your data.")
    exit()

testset = list(
    test_df[["user_id", "blog_id", "ratings"]].itertuples(index=False, name=None)
)
predictions = algo.test(testset)

# --- 2.1 Compute RMSE ---
# predictions: (uid, iid, true_r, est, details)
mse = sum((true_r - est) ** 2 for _, _, true_r, est, _ in predictions) / len(
    predictions
)
rmse = math.sqrt(mse)
print(f"\nRMSE on test set: {rmse:.4f}")

# --- 3. Report Final Performance ---
relevance_threshold = 3
k_value = 10

mean_precision, mean_recall = precision_recall_at_k(
    predictions, k=k_value, threshold=relevance_threshold
)

print("\n--- Final Model Performance on Unseen Test Data ---")
print(
    f"Relevance Threshold: A blog is 'relevant' if its "
    f"true rating is >= {relevance_threshold}"
)
print(f"K value: {k_value}\n")
print(f"RMSE:               {rmse:.4f}")
print(f"Precision@{k_value}: {mean_precision:.4f}")
print(f"Recall@{k_value}:    {mean_recall:.4f}")
print(
    "\nThis means that, on average, if we show a user the top "
    + str(k_value)
    + " blogs, "
    + f"{mean_precision:.1%}"
    + " will be blogs they actually like."
)
print("--- Evaluation Script Finished ---")
