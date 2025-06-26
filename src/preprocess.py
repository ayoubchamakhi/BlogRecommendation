"""Preprocessing utilities for the blog recommendation system."""

import os
import random
import re
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud

curr_path = os.getcwd()
base_path = os.path.dirname(os.getcwd())
rawdata_path = os.path.join(base_path, "data", "raw")
processeddata_path = os.path.join(base_path, "data", "processed")
if not os.path.exists(processeddata_path):
    os.makedirs(processeddata_path)
print(f"Current working directory: {curr_path}")
print(f"Base path for data: {base_path}")
print(f"Raw data path: {rawdata_path}")
print("Processed data path:", processeddata_path)


# Load data
def load_data(rawdata_path):
    """
    Load blog, ratings, and author data from the local dataset directory.

    Args:
        base_path (str): Relative or absolute path to the folder with CSVs.

    Returns:
        tuple: blogs, ratings, authors DataFrames.
    """
    blogs_path = os.path.join(rawdata_path, "MediumBlogData.csv")
    ratings_path = os.path.join(rawdata_path, "BlogRatings.csv")
    authors_path = os.path.join(rawdata_path, "AuthorData.csv")

    print("Reading blogs from:", os.path.abspath(blogs_path))
    print("Reading ratings from:", os.path.abspath(ratings_path))
    print("Reading authors from:", os.path.abspath(authors_path))

    assert os.path.exists(blogs_path), f"Missing file: {blogs_path}"
    assert os.path.exists(ratings_path), f"Missing file: {ratings_path}"
    assert os.path.exists(authors_path), f"Missing file: {authors_path}"

    blogs = pd.read_csv(blogs_path)
    ratings = pd.read_csv(ratings_path)
    authors = pd.read_csv(authors_path)

    if "userId" in ratings.columns:
        ratings.rename(columns={"userId": "user_id"}, inplace=True)
    if "blogId" in ratings.columns:
        ratings.rename(columns={"blogId": "blog_id"}, inplace=True)
    if "authorId" in authors.columns:
        authors.rename(columns={"authorId": "author_id"}, inplace=True)
    if "blogId" in blogs.columns:
        blogs.rename(columns={"blogId": "blog_id"}, inplace=True)
    if "authorId" in blogs.columns:
        blogs.rename(columns={"authorId": "author_id"}, inplace=True)

    return blogs, ratings, authors


def preprocess_blogs(blogs: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess blog metadata: convert scrape_time to datetime
    and add scrape_date.

    Args:
        blogs (pd.DataFrame): Blog metadata DataFrame.

    Returns:
        pd.DataFrame: Processed blogs DF with 'scrape_time'
        as datetime and 'scrape_date' added.
    """
    blogs["scrape_time"] = pd.to_datetime(blogs["scrape_time"])
    blogs["scrape_date"] = blogs["scrape_time"].dt.date
    return blogs


def print_summary_info(
    blogs: pd.DataFrame, ratings: pd.DataFrame, authors: pd.DataFrame
) -> None:
    """
    Print summary statistics about blogs, ratings, and authors datasets.

    Args:
        blogs (pd.DataFrame): Blogs DataFrame.
        ratings (pd.DataFrame): Ratings DataFrame.
        authors (pd.DataFrame): Authors DataFrame.
    """
    num_blogs = blogs["blog_id"].nunique()
    num_authors = authors["author_id"].nunique()
    num_users = ratings["user_id"].nunique()
    num_ratings = len(ratings)
    avg_rating_user = num_ratings / num_users
    avg_rating_blog = num_ratings / num_blogs
    min_rating = ratings['ratings'].min()
    max_rating = ratings['ratings'].max()

    print("Dataset Summary:")
    print(f"Number of unique blogs: {num_blogs}")
    print(f"Number of unique authors: {num_authors}")
    print(f"Number of unique users: {num_users}")
    print(f"Number of total ratings: {num_ratings}")
    print(f"Avg. ratings per user: {avg_rating_user:.2f}")
    print(f"Avg. ratings per blog: {avg_rating_blog:.2f}")
    print(f"Rating range: {min_rating} to {max_rating}")


def merge_data(
    blogs: pd.DataFrame, ratings: pd.DataFrame, authors: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge blog and rating data with authors.

    Args:
        blogs (pd.DataFrame): Blogs DataFrame.
        ratings (pd.DataFrame): Ratings DataFrame.
        authors (pd.DataFrame): Authors DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Merged blogs and ratings DataFrames.
    """
    blogs_full = blogs.merge(authors, on="author_id", how="left")
    ratings_full = ratings.merge(blogs_full, on="blog_id", how="left")
    return blogs_full, ratings_full


def plot_topic_distribution(blogs: pd.DataFrame) -> None:
    """
    Plot the frequency distribution of blog topics.

    Args:
        blogs (pd.DataFrame): Blogs DataFrame.
    """
    plt.figure(figsize=(12, 6))
    sns.countplot(
        y="topic",
        data=blogs,
        order=blogs["topic"].value_counts().index,
    )
    plt.title("Distribution of Blog Topics")
    plt.tight_layout()
    plt.show()


def plot_rating_distribution(ratings: pd.DataFrame) -> None:
    """
    Plot a histogram of rating values.

    Args:
        ratings (pd.DataFrame): Ratings DataFrame with a 'ratings' column.
    """
    plt.figure(figsize=(8, 4))
    ratings["ratings"].hist(bins=10)
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_user_activity(ratings: pd.DataFrame) -> None:
    """
    Plot histogram of number of ratings per user.

    Args:
        ratings (pd.DataFrame): Ratings DataFrame with a 'userId' column.
    """
    ratings_per_user = ratings.groupby("user_id").size()
    plt.figure(figsize=(8, 4))
    ratings_per_user.hist(bins=20)
    plt.title("Number of ratings per user")
    plt.xlabel("Ratings Count")
    plt.ylabel("User Frequency")
    plt.tight_layout()
    plt.show()


def plot_scrape_activity(blogs: pd.DataFrame) -> None:
    """
    Visualize blog scraping activity over time.

    Args:
        blogs (pd.DataFrame): DataFrame with a 'scrape_date' col.
    """
    blog_counts = blogs["scrape_date"].value_counts().sort_index()
    blog_counts.plot(kind="line", figsize=(10, 5),
                     title="Blogs scraped over time")
    plt.ylabel("Blogs Published")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()


def plot_top_authors(ratings_full: pd.DataFrame, n: int = 10) -> None:
    """
    Plot top authors by number of ratings.

    Args:
        ratings_full (pd.DataFrame): Ratings DF with 'author_name' col.
        n (int): Number of top authors to display.
    """
    top_authors = ratings_full["author_name"].value_counts().head(n)
    top_authors.plot(kind="barh", title="Top Rated Authors", figsize=(10, 5))
    plt.tight_layout()
    plt.show()


def plot_top_blogs_with_error_bars(
    ratings_full: pd.DataFrame, n: int = 10, min_ratings: int = 3
) -> None:
    """
    Plot bar chart of top-rated blogs with error bars showing std dev.

    Short error bars indicate agreement among users on ratings.
    Long error bars indicate disagreement or polarizing ratings.
    Blogs with high average rating but large std dev might be controversial.
    Blogs with lower average but small std dev might be consistently rated.

    Args:
        ratings_full (pd.DataFrame): DF containing blog_title, ratings cols.
        n (int): Number of top blogs to display.
        min_ratings (int): Minimum number of ratings to consider a blog.
    """
    blog_stats = (
        ratings_full.groupby("blog_title")["ratings"]
        .agg(["mean", "std", "count"])
        .rename(
            columns={"mean": "avg_rating",
                     "std": "std_rating",
                     "count": "num_ratings"}
        )
    )
    blog_stats = blog_stats[blog_stats["num_ratings"] >= min_ratings]
    top_blogs = blog_stats.sort_values("avg_rating", ascending=False).head(n)

    plt.figure(figsize=(10, 6))
    plt.barh(
        top_blogs.index[::-1],
        top_blogs["avg_rating"][::-1],
        xerr=top_blogs["std_rating"][::-1],
        color="skyblue",
        edgecolor="black",
    )
    plt.xlabel("Average Rating")
    plt.title(f"Top {n} Blogs by Average Rating (with Std Dev)")
    plt.xlim(0, 5)
    plt.tight_layout()
    plt.show()


def plot_violin_top_blog_ratings(ratings_full, n=10, min_ratings=3):
    """
    Display a violin plot showing the rating distribution of the top blogs
    with the most ratings.

    Wider sections indicate more ratings concentrated at that value.
    The white dot represents the median, thick bar = interquartile range (IQR).

    Args:
        ratings_full (pd.DataFrame): Merged DF with ratings and blog metadata.
        n (int): Number of top blogs to display.
        min_ratings (int): Minimum number of ratings required per blog.
    """
    rating_counts = ratings_full["blog_title"].value_counts()
    top_blog_titles = rating_counts[rating_counts >= min_ratings].head(n).index

    top_subset = ratings_full[ratings_full["blog_title"].isin(top_blog_titles)]

    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=top_subset,
        y="blog_title",
        x="ratings",
        density_norm="width",
        inner="quartile",
        palette="Pastel1",
        hue="blog_title",
    )
    plt.title(f"Rating Distribution for Top {n} Most Rated Blogs")
    plt.xlabel("Rating")
    plt.ylabel("Blog Title")
    plt.tight_layout()
    plt.show()


def generate_word_cloud_from_content(blogs):
    """
    Generate and display a word cloud from blog content.

    Args:
        blogs (pd.DataFrame): DataFrame containing a 'blog_content' column.
    """
    all_text = " ".join(blogs["blog_content"].dropna())
    stop_words = set(stopwords.words("english"))

    wordcloud = WordCloud(
        stopwords=stop_words,
        background_color="white",
        max_words=200,
        width=1000,
        height=500,
    ).generate(all_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Blog Content")
    plt.tight_layout()
    plt.show()


def heatmap_avg_rating_by_topic(ratings_full, min_ratings=5):
    """Display a heatmap of average ratings per topic.
    min ratings - filters out topics that have too few ratings
    to be statistically reliable.
    """
    topic_stats = (
        ratings_full.groupby("topic")["ratings"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_rating", "count": "num_ratings"})
    )
    topic_stats = topic_stats[topic_stats["num_ratings"] >= min_ratings]
    topic_stats = topic_stats.sort_values("avg_rating", ascending=False)

    avg_rating_matrix = topic_stats["avg_rating"].to_frame()

    plt.figure(figsize=(8, len(avg_rating_matrix) * 0.4 + 1))
    sns.heatmap(
        avg_rating_matrix,
        annot=True,
        cmap="coolwarm",
        # center at overall average
        center=ratings_full["ratings"].mean(),
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Average Rating"},
    )
    plt.title("Average Rating by Blog Topic")
    plt.xlabel("Average Rating")
    plt.ylabel("Topic")
    plt.tight_layout()
    plt.show()


def pre_process_text(
    text: str,
    flg_stemm: bool = False,
    flg_lemm: bool = True,
    lst_stopwords: list[str] = None,
) -> str:
    """
    Clean and normalize blog text using standard NLP techniques.

    Args:
        text (str): Input text to clean.
        flg_stemm (bool): Whether to apply stemming.
        flg_lemm (bool): Whether to apply lemmatization.
        lst_stopwords (list[str], optional): List of stopwords.

    Returns:
        str: Cleaned and preprocessed text.
    """
    text = str(text).lower().strip()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)  # remove extra whitespace

    tokens = text.split()

    if lst_stopwords:
        tokens = [word for word in tokens if word not in lst_stopwords]

    if flg_lemm:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    if flg_stemm and not flg_lemm:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)


def clean_blog_text_column(blogs: pd.DataFrame) -> pd.DataFrame:
    """
    Clean blog_content column and create new 'clean_blog_content' column.

    Args:
        blogs (pd.DataFrame): Original blogs dataframe.

    Returns:
        pd.DataFrame: DataFrame with added clean_blog_content.
    """
    stop_words = stopwords.words("english")
    blogs["clean_blog_content"] = (
        blogs["blog_content"]
        .fillna("")
        .apply(
            lambda text: pre_process_text(
                text, flg_stemm=False, flg_lemm=True, lst_stopwords=stop_words
            )
        )
    )
    return blogs


# def extract_tfidf_features(
#     blogs: pd.DataFrame, max_features: int = 500
# ) -> tuple:
#     """
#     Extract TF-IDF matrix and feature names from clean blog content.

#     Args:
#         blogs (pd.DataFrame): Blogs with clean content.
#         max_features (int): Max number of TF-IDF features.

#     Returns:
#         tuple: TF-IDF matrix, list of feature names.
#     """
#     tfidf_vectorizer = TfidfVectorizer(
#         stop_words="english", max_features=max_features
#     )
#     tfidf_matrix = tfidf_vectorizer.fit_transform(
#         blogs["clean_blog_content"].fillna("")
#     )
#     feature_names = tfidf_vectorizer.get_feature_names_out()
#     return tfidf_matrix, feature_names


def holdout_new_users(
    ratings: pd.DataFrame,
    user_col: str = "user_id",
    min_ratings_per_user: int = 10,
    num_new_users: int = 100,
    seed: int = 42,
):
    """
    Split ratings into train and test sets
    by holding out all ratings from a subset
    of 'new' users with enough ratings.

    This simulates cold-start users in test.

    Args:
        ratings (pd.DataFrame): User-item rating DataFrame.
        user_col (str): Column name for user IDs.
        min_ratings_per_user (int): Minimum ratings for a user
        to be eligible for holdout.
        num_new_users (int): Number of users to hold out as new users.
        seed (int): Random seed for reproducibility.

    Returns:
        train_df (pd.DataFrame): Training data excluding held-out users.
        test_df (pd.DataFrame): Test data including only held-out users.
        new_users (list): List of held-out user IDs.
    """
    # Count ratings per user
    user_counts = ratings[user_col].value_counts()

    # Filter eligible users with enough ratings
    filtered_users = user_counts[user_counts >= min_ratings_per_user]
    eligible_users = filtered_users.index.tolist()

    # Reproducible random selection of new users
    random.seed(seed)
    new_users = random.sample(eligible_users,
                              min(num_new_users, len(eligible_users)))

    # Split datasets
    train_df = (
        ratings[~ratings[user_col].isin(new_users)]
        .reset_index(drop=True)
    )

    test_df = (
        ratings[ratings[user_col].isin(new_users)]
        .reset_index(drop=True)
    )

    return train_df, test_df, new_users


def run_eda_pipeline(base_path: str = "../data/raw"):
    print("Loading raw datasets...")
    blogs, ratings, authors = load_data(base_path)

    print("Printing dataset summary statistics...")
    print_summary_info(blogs, ratings, authors)

    print("Preprocessing blog timestamps...")
    blogs = preprocess_blogs(blogs)

    print("Merging blog, ratings, and author data...")
    blogs_full, ratings_full = merge_data(blogs, ratings, authors)

    print("Plotting blog topic distribution...")
    plot_topic_distribution(blogs)

    print("Plotting rating distribution...")
    plot_rating_distribution(ratings)

    print("Plotting user activity histogram...")
    plot_user_activity(ratings)

    print("Plotting blog scrape activity over time...")
    plot_scrape_activity(blogs)

    print("Plotting top authors by rating count...")
    plot_top_authors(ratings_full)

    print("Plotting top blogs with error bars...")
    plot_top_blogs_with_error_bars(ratings_full)

    print("Plotting violin plot of blog rating distributions...")
    plot_violin_top_blog_ratings(ratings_full)

    print("Generating word cloud from blog content...")
    generate_word_cloud_from_content(blogs)

    print("Creating heatmap of average ratings by topic...")
    heatmap_avg_rating_by_topic(ratings_full)

    print("Cleaning blog text for NLP...")
    blogs = clean_blog_text_column(blogs)

    # print("Extracting TF-IDF features...")
    # tfidf_matrix, feature_names = extract_tfidf_features(blogs)
    # print("Sample TF-IDF features:", feature_names[:20])

    print("Saving cleaned blog and ratings data...")
    # blogs_full.to_csv(os.path.join(base_path, "cleaned_blog_metadata.csv"),
    # index=False)
    # ratings_full.to_csv(os.path.join(base_path, "cleaned_blog_ratings.csv"),
    # index=False)
    blogs_full.to_pickle(os.path.join(processeddata_path,
                                      "cleaned_blog_metadata.pkl"))
    ratings_full.to_pickle(os.path.join(processeddata_path,
                                        "cleaned_blog_ratings.pkl"))
    print(f"Cleaned data exported to {processeddata_path}")

    print("Splitting data into train and test sets with cold-start users...")
    train_df, test_df, new_users = holdout_new_users(
        ratings_full,
        user_col="user_id",
        min_ratings_per_user=10,
        num_new_users=100,
        seed=42,
    )
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"Number of held-out users: {len(new_users)}")
    print(f"Sample held-out users: {new_users[:5]}")

    print("Saving train/test splits...")
    train_path = os.path.join(processeddata_path, "train_ratings.pkl")
    test_path = os.path.join(processeddata_path, "test_ratings.pkl")
    # train_df.to_csv(train_path, index=False)
    # test_df.to_csv(test_path, index=False)
    train_df.to_pickle(os.path.join(processeddata_path, "train_ratings.pkl"))
    test_df.to_pickle(os.path.join(processeddata_path, "test_ratings.pkl"))
    print(f"Train and test splits saved to {train_path} and {test_path}")


def main():
    base_path = "../data/raw"
    run_eda_pipeline(base_path)


if __name__ == "__main__":
    main()
