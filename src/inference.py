import os
import pickle

import pandas as pd

print("--- Inference Script Initialized ---")

# --- 1. Define Paths ---
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

# Paths to required files
model_filepath = os.path.join(
    models_path,
    "collaborative_model.pkl",
)
all_ratings_path = os.path.join(
    processed_path,
    "cleaned_blog_ratings.pkl",
)
metadata_path = os.path.join(
    processed_path,
    "cleaned_blog_metadata.pkl",
)

# --- 2. Load Model and Data ---
try:
    with open(model_filepath, "rb") as f:
        saved_model = pickle.load(f)
        algo = saved_model["algorithm"]
    print("Model loaded successfully.")

    all_ratings_df = pd.read_pickle(all_ratings_path)
    metadata_df = pd.read_pickle(metadata_path)
    print("Ratings and metadata loaded successfully.")

except FileNotFoundError as e:
    print(
        "ERROR: A required file was not found: "
        + e.filename
    )
    print(
        "Please ensure train.py has been run "
        "and all data files are present."
    )
    exit()


def get_top_n_recommendations(user_id, n=10):
    """
    Generate top N blog recommendations for a user.

    Args:
        user_id (str): ID of the user.
        n (int): Number of recommendations.

    Returns:
        pd.DataFrame: 'blog_id', 'blog_title',
        'predicted_rating', 'topic', 'author_name'.
    """
    if user_id not in all_ratings_df["user_id"].unique():
        print(
            f"Warning: User '{user_id}' "
            "not found in the dataset."
        )
        return pd.DataFrame()

    all_blog_ids = all_ratings_df["blog_id"].unique()
    rated_blog_ids = all_ratings_df[
        all_ratings_df["user_id"] == user_id
    ]["blog_id"]

    blogs_to_predict = [
        blog
        for blog in all_blog_ids
        if blog not in rated_blog_ids.values
    ]

    predictions = [
        (
            blog_id,
            algo.predict(uid=user_id, iid=blog_id).est
        )
        for blog_id in blogs_to_predict
    ]

    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top_n_preds = predictions[:n]
    top_n_blog_ids = [i[0] for i in top_n_preds]
    top_n_ratings = [i[1] for i in top_n_preds]

    results_df = pd.DataFrame(
        {
            "blog_id": top_n_blog_ids,
            "predicted_rating": top_n_ratings,
        }
    )

    results_df = pd.merge(
        results_df,
        metadata_df,
        on="blog_id",
        how="left"
    )

    return results_df[
        [
            "blog_id",
            "blog_title",
            "predicted_rating",
            "topic",
            "author_name",
        ]
    ]


if __name__ == "__main__":
    user_counts = all_ratings_df["user_id"].value_counts()
    if not user_counts.empty:
        sample_user_id = user_counts.index[0]

        print(
            f"\n--- Generating recommendations "
            f"for user: {sample_user_id} ---"
        )
        top_recs_df = get_top_n_recommendations(
            sample_user_id,
            n=5
        )

        if not top_recs_df.empty:
            print(
                f"Top 5 recommendations for user "
                f"'{sample_user_id}':"
            )
            print(
                top_recs_df.to_string(
                    index=False
                )
            )
        else:
            print("Could not generate recommendations.")
    else:
        print(
            "No users found to generate "
            "recommendations for."
        )
