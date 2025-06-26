
# Simple recommendation function using the saved embedding model

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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


def generate_personalized_ads(rec_df, user_id, df, unique_blogs, client):
    """
    Generate personalized advertising content for recommendations using OpenAI API.

    Parameters:
    -----------
    rec_df : pd.DataFrame
        DataFrame with recommendations (from recs_to_dataframe)
    user_id : int
        User ID for context about user preferences
    df : pd.DataFrame
        Original dataframe with ratings (for user context)
    unique_blogs : pd.DataFrame
        DataFrame with unique blog information (for user context)
    client : OpenAI
        OpenAI client instance

    Returns:
    --------
    pd.DataFrame
        DataFrame with personalized advertising content
    """

    # Get user's liked blogs for context
    user_ratings = df[df['user_id'] == user_id]
    liked_blogs = user_ratings[user_ratings['ratings'] >= 4]

    # Create context about user's preferences
    if len(liked_blogs) > 0:
        try:
            # Merge to get blog titles
            liked_with_titles = liked_blogs.merge(unique_blogs, on='blog_id', how='left')

            # Check if blog_title column exists
            if 'blog_title' in liked_with_titles.columns:
                liked_titles = liked_with_titles['blog_title'].dropna().tolist()
                user_context = f"User previously liked blogs: {', '.join(liked_titles[:3])}"
            else:
                # Fallback: use blog_ids if titles not available
                liked_ids = liked_blogs['blog_id'].tolist()
                user_context = f"User previously liked blog IDs: {', '.join(map(str, liked_ids[:3]))}"
        except Exception as e:
            user_context = "User has previous ratings but context unavailable"
    else:
        user_context = "New user with no previous ratings"

    # System prompt for generating personalized ads
    system_prompt = """You are a personalized content marketing specialist. 
    Create engaging, personalized blog recommendations that feel natural and compelling.
    Your goal is to create a short, catchy advertisement (2-3 sentences) that connects the user's interests to the recommended blog.
    Use phrases like "Because you enjoyed..." or "Since you liked..." to create personal connections.
    Keep it concise, engaging, and avoid being overly salesy."""

    all_ads = []

    # Generate ads for all recommendations in the DataFrame
    for idx, row in rec_df.iterrows():
        blog_title = row['blog_title']
        blog_content = row['blog_content'][:500] if pd.notna(row['blog_content']) else "No content available"
        author_name = row.get('author_name', 'Unknown Author')
        topic = row.get('topic', 'General')
        score = row['score']

        # Create the prompt for this specific blog
        blog_info = f"""
        Blog Title: {blog_title}
        Author: {author_name}
        Topic: {topic}
        Content Preview: {blog_content}
        Recommendation Score: {score:.3f}
        User Context: {user_context}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Create a personalized recommendation ad for this blog: {blog_info}'}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )

            ad_content = response.choices[0].message.content

            all_ads.append({
                "blog_id": row['blog_id'],
                "blog_title": blog_title,
                "author_name": author_name,
                "topic": topic,
                "recommendation_score": score,
                "personalized_ad": ad_content,
                "content_preview": blog_content[:200] + "..." if len(blog_content) > 200 else blog_content
            })

        except Exception as e:
            all_ads.append({
                "blog_id": row['blog_id'],
                "blog_title": blog_title,
                "author_name": author_name,
                "topic": topic,
                "recommendation_score": score,
                "personalized_ad": f"[Error generating ad] {e}",
                "content_preview": blog_content[:200] + "..." if len(blog_content) > 200 else blog_content
            })

    ads_df = pd.DataFrame(all_ads)
    return ads_df

# Example usage:

# Initialize OpenAI client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# curr_path= os.getcwd()
# base_path = os.path.dirname(os.getcwd())
# rawdata_path = os.path.join(base_path, "data" ,"raw")
# processeddata_path = os.path.join(base_path, "data", "processed") 

# df= pd.read_pickle(os.path.join(processeddata_path, 'cleaned_blog_ratings.pkl'))
# embedding_model = load_embedding_model(os.path.join(base_path,"models","embedding_model.pkl"))
# unique_blogs = prepare_blog_data(df)
# recs = get_recommendations_for_user(user_id=123, df=df, embedding_model=embedding_model)
# #print(recs)
# #display(recs_to_dataframe(recs, unique_blogs).head())
# rec_df = recs_to_dataframe(recs, unique_blogs).head(5)

# # Generate personalized ads using the prepared DataFrame
# personalized_ads = generate_personalized_ads(rec_df, user_id=123, df=df, unique_blogs=unique_blogs, client=client)

# print("PERSONALIZED BLOG RECOMMENDATIONS:")
# print("=" * 50)
# for idx, row in personalized_ads.iterrows():
#     print(f"\n{row['blog_title']}")
#     print(f"  By: {row['author_name']}")
#     print(f"  Topic: {row['topic']}")
#     print(f" Score: {row['recommendation_score']:.3f}")
#     print(f" {row['personalized_ad']}")
#     print("-" * 40)

# print("\nDataFrame Preview:")
# print(personalized_ads[['blog_title', 'author_name', 'recommendation_score', 'personalized_ad']].head())
