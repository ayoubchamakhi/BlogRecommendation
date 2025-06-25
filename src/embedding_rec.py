
# Simple recommendation function using the saved embedding model

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recs_to_dataframe(recs, unique_blogs):
    """
    Convert recommendation list into a full DataFrame with metadata.

    Parameters:
    -----------
    recs : list of (blog_id, score) tuples
    unique_blogs : pd.DataFrame
        The DataFrame you created earlier in Cell 5, with one row per blog and
        columns like ['blog_id', 'blog_title', 'blog_content', 'topic', 'author_name', …]

    Returns:
    --------
    pd.DataFrame
        Columns: ['blog_id', 'score', 'blog_title', 'blog_content', 'topic', 'author_name', …]
    """
    # 1. Build a small DataFrame of just IDs and scores
    rec_df = pd.DataFrame(recs, columns=['blog_id', 'score'])
    
    # 2. Merge with the blog metadata
    full_rec_df = rec_df.merge(unique_blogs, on='blog_id', how='left')
    
    # 3. (Optional) Reorder columns to taste
    cols = ['blog_id', 'score']
    # pick up all the other metadata columns
    meta_cols = [c for c in unique_blogs.columns if c != 'blog_id']
    full_rec_df = full_rec_df[cols + meta_cols]
    
    return full_rec_df

def prepare_blog_data(df):
    """
    Extract unique blog information and include average rating per blog.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined dataset with user-blog-rating information
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with unique blog information and avg_rating column
    """
    
    print("PREPARING UNIQUE BLOG DATA (with average ratings)")
    
    
    # 1. Compute average rating per blog
    avg_ratings = (
        df.groupby('blog_id')['ratings']
          .mean()
          .rename('avg_rating')
          .reset_index()
    )
    print(f"✓ Computed average rating for {len(avg_ratings)} blogs")
    
    # 2. Define the blog metadata columns
    blog_columns = ['blog_id', 'blog_title', 'blog_content', 'topic']
    if 'author_name' in df.columns:
        blog_columns.append('author_name')
    for col in ['author_id', 'blog_link', 'blog_img']:
        if col in df.columns:
            blog_columns.append(col)
    
    print(f"Extracting unique blogs with columns: {blog_columns}")
    
    # 3. Extract unique blog rows
    unique_blogs = (
        df[blog_columns]
        .drop_duplicates(subset=['blog_id'])
        .reset_index(drop=True)
    )
    print(f"✓ Extracted {len(unique_blogs)} unique blogs")
    
    # 4. Merge in the avg_rating
    unique_blogs = unique_blogs.merge(avg_ratings, on='blog_id', how='left')
    print("✓ Merged average ratings into unique_blogs")
    print(f"Final shape: {unique_blogs.shape}")
    
    return unique_blogs





def load_embedding_model(model_path='model/embedding_model.pkl'):
    """Load the saved embedding model."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def get_recommendations_for_user(user_id, df, embedding_model, top_n=10):
    """
    Get content-based recommendations for a user using saved embeddings.
    
    Parameters:
    - user_id: User ID
    - df: Original dataframe with ratings
    - embedding_model: Loaded embedding model
    - top_n: Number of recommendations
    """
    embeddings = embedding_model['embeddings']
    blog_ids = embedding_model['blog_ids']
    
    # Get user's liked blogs
    user_ratings = df[df['user_id'] == user_id]
    liked_blogs = user_ratings[user_ratings['ratings'] >= 4]['blog_id'].values
    
    # Find indices of liked blogs in embedding model
    blog_id_to_idx = {bid: idx for idx, bid in enumerate(blog_ids)}
    liked_indices = [blog_id_to_idx[bid] for bid in liked_blogs if bid in blog_id_to_idx]
    
    # Create user profile
    user_profile = np.mean(embeddings[liked_indices], axis=0).reshape(1, -1)
    
    # Calculate similarities
    similarities = cosine_similarity(user_profile, embeddings).flatten()
    
    # Get recommendations
    rated_blogs = set(user_ratings['blog_id'].values)
    recommendations = []
    
    for idx, (blog_id, score) in enumerate(zip(blog_ids, similarities)):
        if blog_id not in rated_blogs:
            recommendations.append((blog_id, score))
    
    # Sort and return top N
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]

# Example usage:
embedding_model = load_embedding_model('model/embedding_model.pkl')
unique_blogs = prepare_blog_data(df)
recs = get_recommendations_for_user(user_id=123, df=df, embedding_model=embedding_model)
# display(recs_to_dataframe(recs, unique_blogs).head())
