import os
import pickle

import pandas as pd

print("--- Inference Script Initialized ---")

# --- 1. Define Paths ---
base_path = os.path.dirname(os.getcwd())
models_path = os.path.join(base_path, "models")
processed_path = os.path.join(base_path, "data", "processed")

# Paths to required files
model_filepath = os.path.join(models_path, "collaborative_model.pkl")
all_ratings_path = os.path.join(processed_path, "cleaned_blog_ratings.pkl")
metadata_path = os.path.join(processed_path, "cleaned_blog_metadata.pkl")


# --- 2. Load Model and Data ---
try:
    # Load the trained model
    with open(model_filepath, "rb") as f:
        saved_model = pickle.load(f)
        algo = saved_model["algorithm"]
    print("Model loaded successfully.")

    # Load all ratings to know what a user has already seen
    all_ratings_df = pd.read_pickle(all_ratings_path)
    # Load metadata to provide blog titles
    metadata_df = pd.read_pickle(metadata_path)
    print("Ratings and metadata loaded successfully.")

except FileNotFoundError as e:
    print(f"ERROR: A required file was not found: {e.filename}")
    print("Please ensure train.py has been run and all data files are present.")
    exit()


def get_top_n_recommendations(user_id, n=10):
    """
    Generates top N blog recommendations for a given user.

    Args:
        user_id (str): The ID of the user.
        n (int): The number of recommendations to return.

    Returns:
        pd.DataFrame: A DataFrame with 'blog_id', 'title', and 'predicted_rating'.
    """
    if user_id not in all_ratings_df["user_id"].unique():
        print(f"Warning: User '{user_id}' not found in the dataset.")
        return pd.DataFrame()

    # Get a list of all unique blog IDs
    all_blog_ids = all_ratings_df["blog_id"].unique()

    # Get the list of blogs the user has already rated
    rated_blog_ids = all_ratings_df[all_ratings_df["user_id"] == user_id]["blog_id"]

    # Filter out blogs the user has already rated
    blogs_to_predict = [
        blog for blog in all_blog_ids if blog not in rated_blog_ids.values
    ]

    # Make predictions for the unrated blogs
    predictions = [
        (blog_id, algo.predict(uid=user_id, iid=blog_id).est)
        for blog_id in blogs_to_predict
    ]

    # Sort recommendations by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Get top N recommendations
    top_n_preds = predictions[:n]
    top_n_blog_ids = [i[0] for i in top_n_preds]
    top_n_ratings = [i[1] for i in top_n_preds]

    # Create a results dataframe and merge with metadata
    results_df = pd.DataFrame({
        "blog_id": top_n_blog_ids,
        "predicted_rating": top_n_ratings,
    })
    
    # Assuming metadata_df has 'blog_id' and 'title' columns
    results_df = pd.merge(results_df, metadata_df, on="blog_id", how="left")

    return results_df[["blog_id", "blog_title", "predicted_rating", "topic","author_name"]]


if __name__ == "__main__":
    # Find a user with a good number of ratings to test
    user_counts = all_ratings_df["user_id"].value_counts()
    if not user_counts.empty:
        sample_user_id = user_counts.index[0]

        print(f"\n--- Generating recommendations for user: {sample_user_id} ---")
        top_recs_df = get_top_n_recommendations(sample_user_id, n=5)

        if not top_recs_df.empty:
            print(f"Top 5 blog recommendations for user '{sample_user_id}':")
            print(top_recs_df.to_string(index=False))
        else:
            print("Could not generate recommendations.")
    else:
        print("No users found in the dataset to generate recommendations for.")