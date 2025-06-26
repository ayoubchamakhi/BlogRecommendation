"""
Streamlit Blog Recommendation App
Author: Abhishek
Description: Hybrid content-based and collaborative filtering blog recommender
"""
import streamlit as st
import pandas as pd
import os
import sys

# --- Single Page Configuration ---
st.set_page_config(layout="wide", page_title="Blog Recommender", page_icon="✍️")

# --- Robust Pathing Logic ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_root)

    from src.embedding_rec import (
        get_recommendations_for_user,
        prepare_blog_data,
        recs_to_dataframe,
        generate_personalized_ads,
    )
    from openai import OpenAI
    import pickle
except ImportError as e:
    st.error(
        f"Error importing your logic from 'src/embedding_rec.py': {e}. Please ensure your project structure is correct."
    )
    st.stop()


# --- Custom CSS for Enhanced UI (Black text, white background, red/green buttons) ---
def load_css():
    css = """
    <style>
        body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
            color: #000 !important;
            background-color: #fff !important;
        }
        [data-testid="stSidebar"] {
            background-color: #fff !important;
            color: #000 !important;
        }
        [data-testid="stAppViewContainer"] > .main {
            background-color: #fff !important;
            color: #000 !important;
        }
        h1, h2, h3, h4, h5, h6, label, .stTextInput, .stSelectbox, .stSlider, .stButton, .stCaption, .stMarkdown, .stForm, .stExpander, .stSubheader, .stRadio, .stCheckbox, .stMultiSelect, .stNumberInput, .stDateInput, .stTimeInput, .stColorPicker, .stFileUploader, .stTextArea, .stDataFrame, .stTable, .stMetric, .stAlert, .stException, .stTooltip, .stHelp, .stInfo, .stWarning, .stError, .stSuccess {
            color: #000 !important;
        }
        .stButton>button {
            background-color: #e0e0e0 !important;
            color: #000 !important;
        }
        .stButton>button:hover {
            opacity: 0.9;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #fff; border: 1px solid #e0e0e0;
            border-radius: 4px; padding: 24px; transition: all 0.2s ease-in-out;
            color: #000 !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: #6c6c6c; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        .personal-pitch {
            border-left: 3px solid #43a047; padding-left: 15px;
            font-style: italic; color: #000;
        }
        .hero-section {
            background-color: #fff;
            padding: 40px; text-align: center;
            border-radius: 10px; border: 1px solid #e0e0e0;
            margin-bottom: 30px;
            color: #000 !important;
        }
        .how-it-works-col { text-align: center; padding: 20px; color: #000 !important; }
        .how-it-works-col .icon { font-size: 3em; }
        .footer { text-align: center; padding: 20px; color: #6c6c6c; font-size: 0.9em; }
        [data-testid="stVerticalBlockBorderWrapper"] h1,
        [data-testid="stVerticalBlockBorderWrapper"] h2,
        [data-testid="stVerticalBlockBorderWrapper"] h3,
        [data-testid="stVerticalBlockBorderWrapper"] h4,
        [data-testid="stVerticalBlockBorderWrapper"] h5,
        [data-testid="stVerticalBlockBorderWrapper"] h6 {
            color: #000 !important;
        }
        .stButton>button {font-weight:bold;}
        /* Force button colors for login/guest on landing page */
        div[data-testid="column"]:nth-of-type(1) button, div[data-testid="column"]:nth-of-type(1) button:active, div[data-testid="column"]:nth-of-type(1) button:focus {
            background-color: #43a047 !important;
            color: #fff !important;
            border: none !important;
        }
        div[data-testid="column"]:nth-of-type(2) button, div[data-testid="column"]:nth-of-type(2) button:active, div[data-testid="column"]:nth-of-type(2) button:focus {
            background-color: #e53935 !important;
            color: #fff !important;
            border: none !important;
        }
        /* Force warning, error, and info messages to have black text */
        .stAlert, .stAlert p, .stAlert div, .stAlert span, .stAlert label {
            color: #000 !important;
        }
        /* Force caption and badge text (topic, avg rating) to black */
        .stCaption, .stCaption span, .stCaption div, .stCaption p {
            color: #000 !important;
        }
        /* Also force any badge-like elements in the card to black */
        .stMarkdown span, .stMarkdown div, .stMarkdown p {
            color: #000 !important;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# --- DATA & MODEL LOADING ---
@st.cache_resource
def get_embedding_model():
    model_path = os.path.join(project_root, "models", "embedding_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def get_collaborative_model():
    model_path = os.path.join(project_root, "models", "collaborative_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def get_data():
    data_path = os.path.join(
        project_root, "data", "processed", "cleaned_blog_ratings.pkl"
    )
    df = pd.read_pickle(data_path)
    unique_blogs = prepare_blog_data(df)
    return df, unique_blogs


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- HYBRID RECOMMENDATION FUNCTION ---
def get_user_rating_count(df, user_id):
    return df[df["user_id"] == user_id].shape[0]


def hybrid_recommend(
    user_id, df, embedding_model, collaborative_model, unique_blogs, top_n
):
    rating_count = get_user_rating_count(df, user_id)
    if rating_count >= 5 and collaborative_model is not None:
        from src.inference import get_top_n_recommendations

        recs = get_top_n_recommendations(user_id=user_id, n=top_n)
        # recs is likely a DataFrame with blog_id, blog_title, predicted_rating
        if hasattr(recs, "to_dict"):
            recs_list = list(zip(recs["blog_id"], recs["predicted_rating"]))
        else:
            recs_list = recs
        rec_df = recs_to_dataframe(recs_list, unique_blogs)
        return rec_df
    else:
        recs = get_recommendations_for_user(
            user_id=user_id, df=df, embedding_model=embedding_model, top_n=top_n
        )
        rec_df = recs_to_dataframe(recs, unique_blogs)
        return rec_df


# --- LANDING PAGE ---
def landing_page():
    load_css()
    st.markdown(
        """
        <div style='max-width: 400px; margin: 60px auto 0 auto; text-align:center;'>
            <h2>Welcome to Your Personal Discovery Engine</h2>
        </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        .stButton>button {font-weight:bold;}
        #login_btn {background-color: #43a047 !important; color: #fff !important; border: none !important;}
        #guest_btn {background-color: #e53935 !important; color: #fff !important; border: none !important;}
        </style>
    """,
        unsafe_allow_html=True,
    )
    st.write("")
    username = st.text_input("Username", key="username_input")
    password = st.text_input("Password", type="password", key="password_input")
    login_btn = st.button("Login", key="login_btn")
    guest_btn = st.button("Continue as Guest", key="guest_btn")
    if login_btn:
        if username == "admin" and password == "password":
            st.session_state.authenticated = True
            st.session_state.mode = "login"
            st.rerun()
        else:
            st.error("Incorrect username or password")
    if guest_btn:
        st.session_state.mode = "guest"
        st.session_state.authenticated = True
        st.rerun()


# --- LOGIN FUNCTION ---
def login_page():
    load_css()
    _, col2, _ = st.columns([1, 1.5, 1])
    with col2:
        st.title("Sign In")
        st.write("A place to read, write, and deepen your understanding.")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Continue", type="primary")
            if submitted:
                if username == "admin" and password == "password":
                    st.session_state.authenticated = True
                    st.session_state.mode = "login"
                    st.rerun()
                else:
                    st.error("Incorrect username or password")


# --- MAIN APP LOGIC ---
embedding_model = get_embedding_model()
collaborative_model = get_collaborative_model()
df, unique_blogs = get_data()

if "mode" not in st.session_state:
    st.session_state.mode = None
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "show_login" not in st.session_state:
    st.session_state.show_login = False

if st.session_state.mode is None:
    landing_page()
    st.stop()

if st.session_state.mode == "login" and not st.session_state.authenticated:
    login_page()
    st.stop()

load_css()

# --- SIDEBAR USER FILTER BASED ON MODE ---
with st.sidebar:
    st.markdown("## ✍️ Recommender Engine")
    st.markdown("Use the controls below to generate and filter your personalized feed.")
    st.markdown("---")
    st.subheader("👤 User Selection")
    if st.session_state.mode == "guest":
        # Only users with <5 ratings
        user_counts = df.groupby("user_id").size()
        guest_users = sorted(user_counts[user_counts < 5].index)
        user_ids = guest_users
    else:
        # Only users with >=5 ratings
        user_counts = df.groupby("user_id").size()
        login_users = sorted(user_counts[user_counts >= 5].index)
        user_ids = login_users
    if user_ids:
        selected_user_id = st.selectbox(
            "Select a User Profile", user_ids, label_visibility="collapsed"
        )
    else:
        st.warning("No users available for this mode.")
        st.stop()
    st.subheader("⚙️ Generation Controls")
    top_n = st.slider(
        "Recommendations to generate",
        3,
        20,
        10,
        help="How many initial recommendations to generate before filtering.",
    )
    with st.expander("🔬 Advanced Filters"):
        all_topics = sorted(unique_blogs["topic"].dropna().unique())
        selected_topics = st.multiselect("Filter by Topic", options=all_topics)
        min_rating = st.slider(
            "Minimum Average Rating", min_value=1.0, max_value=5.0, value=1.0, step=0.25
        )
    generate_btn = st.button(
        "Generate My Recommendations", key="generate_btn", type="primary"
    )
    st.markdown("---")
    if st.session_state.mode == "guest":
        st.write(f"Browsing as **Guest**")
        if st.button("Back to Home", key="back_home_guest", type="secondary"):
            st.session_state.mode = None
            st.rerun()
    else:
        st.write(f"Signed in as **admin**")
        if st.button("Sign out", key="signout_btn", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.mode = None
            st.rerun()


# --- WELCOME PAGE FUNCTION ---
def display_welcome_page():
    st.markdown(
        """
        <div class="hero-section">
            <h2>Welcome to Your Personal Discovery Engine</h2>
            <p>Unlock content tailored specifically to your tastes. Our AI analyzes your preferences to bring you articles and stories you're bound to love. Select a user profile and generate your feed to get started.</p>
        </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("### How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "<div class='how-it-works-col'><span class='icon'>👤</span><h4>1. Select Profile</h4><p>Choose a user profile from the sidebar. Each profile has unique reading tastes.</p></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            "<div class='how-it-works-col'><span class='icon'>🧠</span><h4>2. Generate Feed</h4><p>Our AI model generates recommendations and a personalized pitch for each article.</p></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            "<div class='how-it-works-col'><span class='icon'>🔬</span><h4>3. Filter & Explore</h4><p>Use the filters to narrow down the results by topic and rating to find exactly what you want.</p></div>",
            unsafe_allow_html=True,
        )


# --- Main content area
st.title("For you")

if generate_btn:
    with st.spinner("Generating your personalized feed..."):
        if st.session_state.mode == "guest":
            recs = get_recommendations_for_user(
                user_id=selected_user_id,
                df=df,
                embedding_model=embedding_model,
                top_n=top_n,
            )
            rec_df = recs_to_dataframe(recs, unique_blogs)
        else:
            rec_df = hybrid_recommend(
                user_id=selected_user_id,
                df=df,
                embedding_model=embedding_model,
                collaborative_model=collaborative_model,
                unique_blogs=unique_blogs,
                top_n=top_n,
            )
        if rec_df is not None and not rec_df.empty:
            personalized_ads_df = generate_personalized_ads(
                rec_df=rec_df,
                user_id=selected_user_id,
                df=df,
                unique_blogs=unique_blogs,
                client=client,
            )
            # Merge in all relevant metadata columns
            merge_cols = ["blog_id", "blog_img", "blog_link", "avg_rating"]
            if "topic" in rec_df.columns:
                merge_cols.append("topic")
            final_df = pd.merge(
                personalized_ads_df, rec_df[merge_cols], on="blog_id", how="left"
            )
            # If both topic_x and topic_y exist, coalesce to a single 'topic' column
            if "topic_x" in final_df.columns and "topic_y" in final_df.columns:
                final_df["topic"] = final_df["topic_x"].combine_first(
                    final_df["topic_y"]
                )
                final_df = final_df.drop(columns=["topic_x", "topic_y"])
            elif "topic_x" in final_df.columns:
                final_df = final_df.rename(columns={"topic_x": "topic"})
            elif "topic_y" in final_df.columns:
                final_df = final_df.rename(columns={"topic_y": "topic"})
            # Ensure avg_rating column is present
            if "avg_rating" not in final_df.columns:
                final_df = pd.merge(
                    final_df,
                    unique_blogs[["blog_id", "avg_rating"]],
                    on="blog_id",
                    how="left",
                )
            # Debug: print unique topics and a sample of the DataFrame
            print("Unique topics in final_df:", final_df["topic"].unique())
            print("Sample of final_df:", final_df.head())
            st.session_state.results = final_df
        else:
            st.warning("Could not generate recommendations for the selected user.")

if "results" in st.session_state:
    results_df = st.session_state.results
    filtered_df = results_df.copy()
    if selected_topics:
        filtered_df = filtered_df[filtered_df["topic"].isin(selected_topics)]
    filtered_df = filtered_df[filtered_df["avg_rating"] >= min_rating]
    st.markdown(
        f"#### Displaying **{len(filtered_df)}** of **{len(results_df)}** recommendations"
    )
    st.markdown("---")
    if filtered_df.empty:
        st.warning(
            "No recommendations match your current filter criteria. Try adjusting the filters in the sidebar."
        )
    else:
        for index, row in filtered_df.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([1, 2.5])
                with col1:
                    final_url = "https://images.unsplash.com/photo-1554629947-334ff61d85dc?q=80&w=400"
                    img_url_from_data = row.get("blog_img")
                    if not pd.isna(img_url_from_data) and isinstance(
                        img_url_from_data, str
                    ):
                        cleaned_url = img_url_from_data.strip()
                        if cleaned_url.startswith("http"):
                            final_url = cleaned_url
                    st.image(final_url, use_container_width=True)
                with col2:
                    st.subheader(f"[{row['blog_title']}]({row.get('blog_link', '#')})")
                    rating_text = f"⭐ {row.get('avg_rating', 0):.2f}"
                    st.caption(
                        f"{row.get('author_name', 'N/A')} in {row.get('topic', 'General')} | {rating_text}"
                    )
                    st.markdown(
                        f"<div class='personal-pitch'>{row['personalized_ad']}</div>",
                        unsafe_allow_html=True,
                    )
            st.write("")
else:
    display_welcome_page()

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div class='footer'>Personalized Recommendation Engine © 2025</div>",
    unsafe_allow_html=True,
)
