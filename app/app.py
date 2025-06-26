import streamlit as st
import pandas as pd
import os
import sys

# --- Single Page Configuration ---
st.set_page_config(layout="wide", page_title="Blog Recommender", page_icon="✍️")

# --- Robust Pathing Logic ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)

    from src.embedding_rec import (
        get_recommendations_for_user,
        prepare_blog_data,
        recs_to_dataframe,
        generate_personalized_ads
    )
    from openai import OpenAI
    import pickle
except ImportError as e:
    st.error(f"Error importing your logic from 'src/embedding_rec.py': {e}. Please ensure your project structure is correct.")
    st.stop()

# --- Custom CSS for Enhanced UI ---
def load_css():
    """Injects custom CSS for a more polished, modern UI."""
    css = """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;700&family=Source+Sans+3:wght@400;600&display=swap');
        :root {
            --medium-beige: #FAF9F6; --medium-black: #1d1d1d; --medium-green: #1a8917;
            --medium-gray: #6c6c6c; --border-color: #e0e0e0; --font-serif: 'Source Serif 4', serif;
            --font-sans: 'Source Sans 3', sans-serif;
            --gradient-start: #2a2a2a; --gradient-end: #1d1d1d;
        }
        body { font-family: var(--font-sans); color: var(--medium-black); }
        
        /* ENHANCEMENT: Subtle gradient on main background */
        [data-testid="stAppViewContainer"] > .main {
             background-image: radial-gradient(circle at 20% 20%, rgba(240, 240, 240, 0.3), var(--medium-beige) 30%);
        }
        
        /* ENHANCEMENT: Wider sidebar */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF; border-right: 1px solid var(--border-color);
            width: 360px !important; min-width: 360px !important;
        }
        h1, h2, h3 { font-family: var(--font-serif); color: var(--medium-black); }
        
        /* ENHANCEMENT: Gradient buttons */
        .stButton>button {
            background-image: linear-gradient(45deg, var(--gradient-start) 0%, var(--gradient-end) 100%);
            color: #FFFFFF; font-family: var(--font-sans);
            border: none; border-radius: 20px; padding: 10px 20px; transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        }
        .stButton>button:focus { box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
        
        /* Card Styling with Hover Effect */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF; border: 1px solid var(--border-color);
            border-radius: 4px; padding: 24px; transition: all 0.2s ease-in-out;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: var(--medium-gray); box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        .personal-pitch {
            border-left: 3px solid var(--medium-green); padding-left: 15px;
            font-style: italic; color: var(--medium-black);
        }
        
        /* ENHANCEMENT: Welcome/Hero Section Styling */
        .hero-section {
            background-color: #FFFFFF;
            padding: 40px; text-align: center;
            border-radius: 10px; border: 1px solid var(--border-color);
            margin-bottom: 30px;
        }
        .hero-section h2 { font-size: 2.5em; margin-bottom: 10px; }
        .hero-section p { font-size: 1.1em; color: var(--medium-gray); max-width: 600px; margin: auto;}
        
        /* ENHANCEMENT: How it Works Section Styling */
        .how-it-works-col { text-align: center; padding: 20px; }
        .how-it-works-col .icon { font-size: 3em; }

        /* ENHANCEMENT: Footer Styling */
        .footer { text-align: center; padding: 20px; color: var(--medium-gray); font-size: 0.9em; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- LOGIN FUNCTION ---
def login_page():
    # ... (Login function remains the same, no changes needed)
    load_css()
    _ , col2, _ = st.columns([1, 1.5, 1])
    with col2:
        st.title("Sign In")
        st.write("A place to read, write, and deepen your understanding.")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Continue")
            if submitted:
                if username == "admin" and password == "password":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect username or password")

# --- AUTHENTICATION CHECK ---
if not st.session_state.get("authenticated", False):
    login_page()
    st.stop()


# --- MAIN APPLICATION ---
load_css()

# --- DATA & MODEL LOADING ---
@st.cache_resource
def get_model():
    model_path = os.path.join(project_root, "models", "embedding_model.pkl")
    with open(model_path, 'rb') as f: return pickle.load(f)

@st.cache_data
def get_data():
    data_path = os.path.join(project_root, "data", "processed", "cleaned_blog_ratings.pkl")
    df = pd.read_pickle(data_path)
    unique_blogs = prepare_blog_data(df)
    return df, unique_blogs

# --- HARDCODED API KEY ---
# OPENAI_API_KEY = "PASTE_YOUR_OPENAI_API_KEY_HERE"
# if OPENAI_API_KEY == "PASTE_YOUR_OPENAI_API_KEY_HERE":
#     st.error("Please set your OpenAI API Key in the `app/main.py` file.", icon="🚨")
#     st.stop()
# client = OpenAI(api_key=OPENAI_API_KEY)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# --- UI FOR THE MAIN APP ---
embedding_model = get_model()
df, unique_blogs = get_data()

with st.sidebar:
    # ENHANCEMENT: Added sidebar title and icon
    st.markdown("## ✍️ Recommender Engine")
    st.markdown("Use the controls below to generate and filter your personalized feed.")
    st.markdown("---")
    
    st.subheader("👤 User Selection")
    user_ids = sorted(df['user_id'].unique())
    selected_user_id = st.selectbox("Select a User Profile", user_ids, index=10, label_visibility="collapsed")
    
    st.subheader("⚙️ Generation Controls")
    top_n = st.slider("Recommendations to generate", 3, 20, 10, help="How many initial recommendations to generate before filtering.")
    
    with st.expander("🔬 Advanced Filters"):
        all_topics = sorted(unique_blogs['topic'].dropna().unique())
        selected_topics = st.multiselect("Filter by Topic", options=all_topics)
        min_rating = st.slider("Minimum Average Rating", min_value=1.0, max_value=5.0, value=1.0, step=0.25)

    generate_btn = st.button("Generate My Recommendations")
    
    st.markdown("---")
    st.write(f"Signed in as **admin**")
    if st.button("Sign out"):
        st.session_state.authenticated = False
        st.rerun()

# --- WELCOME PAGE FUNCTION ---
def display_welcome_page():
    st.markdown("""
        <div class="hero-section">
            <h2>Welcome to Your Personal Discovery Engine</h2>
            <p>Unlock content tailored specifically to your tastes. Our AI analyzes your preferences to bring you articles and stories you're bound to love. Select a user profile and generate your feed to get started.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='how-it-works-col'><span class='icon'>👤</span><h4>1. Select Profile</h4><p>Choose a user profile from the sidebar. Each profile has unique reading tastes.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='how-it-works-col'><span class='icon'>🧠</span><h4>2. Generate Feed</h4><p>Our AI model generates recommendations and a personalized pitch for each article.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='how-it-works-col'><span class='icon'>🔬</span><h4>3. Filter & Explore</h4><p>Use the filters to narrow down the results by topic and rating to find exactly what you want.</p></div>", unsafe_allow_html=True)

# --- Main content area
st.title("For you")

if generate_btn:
    with st.spinner("Generating your personalized feed..."):
        recs = get_recommendations_for_user(
            user_id=selected_user_id, df=df, embedding_model=embedding_model, top_n=top_n
        )
        if recs:
            rec_df = recs_to_dataframe(recs, unique_blogs)
            personalized_ads_df = generate_personalized_ads(
                rec_df=rec_df, user_id=selected_user_id, df=df, unique_blogs=unique_blogs, client=client
            )
            final_df = pd.merge(
                personalized_ads_df,
                rec_df[['blog_id', 'blog_img', 'blog_link', 'avg_rating']],
                on='blog_id', how='left'
            )
            st.session_state.results = final_df
        else:
             st.warning("Could not generate recommendations for the selected user.")
             if 'results' in st.session_state:
                 del st.session_state['results']


if 'results' in st.session_state:
    results_df = st.session_state.results
    filtered_df = results_df.copy()
    
    if selected_topics:
        filtered_df = filtered_df[filtered_df['topic'].isin(selected_topics)]
    filtered_df = filtered_df[filtered_df['avg_rating'] >= min_rating]

    st.markdown(f"#### Displaying **{len(filtered_df)}** of **{len(results_df)}** recommendations")
    st.markdown("---")

    if filtered_df.empty:
        st.warning("No recommendations match your current filter criteria. Try adjusting the filters in the sidebar.")
    else:
        for index, row in filtered_df.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([1, 2.5])
                with col1:
                    final_url = "https://images.unsplash.com/photo-1554629947-334ff61d85dc?q=80&w=400"
                    img_url_from_data = row.get('blog_img')
                    if not pd.isna(img_url_from_data) and isinstance(img_url_from_data, str):
                        cleaned_url = img_url_from_data.strip()
                        if cleaned_url.startswith('http'):
                            final_url = cleaned_url
                    st.image(final_url, use_container_width=True)

                with col2:
                    st.subheader(f"[{row['blog_title']}]({row.get('blog_link', '#')})")
                    rating_text = f"⭐ {row.get('avg_rating', 0):.2f}"
                    st.caption(f"{row.get('author_name', 'N/A')} in {row.get('topic', 'General')} | {rating_text}")
                    st.markdown(f"<div class='personal-pitch'>{row['personalized_ad']}</div>", unsafe_allow_html=True)
            st.write("")
else:
    display_welcome_page()

# --- Footer ---
st.markdown("---")
st.markdown("<div class='footer'>Personalized Recommendation Engine © 2025</div>", unsafe_allow_html=True)