# Blog Recommendation System

A personalized blog recommendation engine built with Streamlit, featuring content-based filtering using sentence embeddings and OpenAI-powered personalized advertising.

## 🚀 Features

- **Content-Based Filtering**: Uses sentence transformers to create blog embeddings
- **Personalized Recommendations**: AI-generated personalized pitches for each recommendation
- **Interactive UI**: Modern Streamlit interface with filtering capabilities
- **User Authentication**: Simple login system for demo purposes
- **Topic Filtering**: Filter recommendations by blog topics
- **Rating-Based Filtering**: Filter by minimum average ratings

## 📁 Project Structure

```
BlogRecommendation/
├── app/
│   └── app.py                 # Main Streamlit application
├── data/
│   ├── notebooks/             # Jupyter notebooks for development
│   ├── processed/             # Processed data files (.pkl)
│   └── raw/                   # Raw CSV data files
├── models/
│   ├── collaborative_model.pkl
│   └── embedding_model.pkl    # Trained embedding model
├── src/
│   ├── embedding_rec.py       # Core recommendation logic
│   ├── evaluate.py           # Model evaluation
│   ├── inference.py          # Inference utilities
│   ├── preprocess.py         # Data preprocessing
│   └── train.py              # Model training
├── Dockerfile                # Docker configuration
├── docker-compose.yaml       # Docker Compose setup
├── environment.yml           # Conda environment
└── README.md                 # This file
```

## 🛠️ Installation

### Option 1: Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd BlogRecommendation
   ```

2. **Create conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate blog-recommender-env
   ```

3. **Set up OpenAI API Key**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

4. **Run the application**
   ```bash
   streamlit run app/app.py
   ```

### Option 2: Docker Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd BlogRecommendation
   ```

2. **Set up environment variables**
   ```bash
   # On Windows PowerShell
   $env:OPENAI_API_KEY="your-openai-api-key-here"
   
   # On Linux/Mac
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

3. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

4. **Access the application**
   Open your browser and go to `http://localhost:8000`

## 🎯 Usage

### Getting Started

1. **Login**: Use the default credentials:
   - Username: `admin`
   - Password: `password`

2. **Select User Profile**: Choose a user profile from the sidebar to generate recommendations

3. **Generate Recommendations**: Click "Generate My Recommendations" to get personalized blog suggestions

4. **Filter Results**: Use the advanced filters to narrow down recommendations by:
   - Topic
   - Minimum average rating

### Features

- **Personalized Feed**: Each user gets recommendations based on their reading history
- **AI-Generated Pitches**: OpenAI creates personalized advertising content for each recommendation
- **Interactive Filtering**: Filter recommendations by topic and rating
- **Responsive Design**: Modern UI that works on desktop and mobile

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key for generating personalized content

### Model Files

Ensure the following files are present in the `models/` directory:
- `embedding_model.pkl`: Trained sentence embedding model

### Data Files

Ensure the following files are present in the `data/processed/` directory:
- `cleaned_blog_ratings.pkl`: Processed blog ratings data

## 🧪 Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black .
isort .
```

### Linting
```bash
flake8
```

## 📊 Data Sources

The system uses the following data:
- **Blog Metadata**: Title, content, topic, author information
- **User Ratings**: User-blog rating interactions
- **Author Data**: Author information and metadata

## 🤝 Team Members

- **Ekaterina**: EDA + LLM Integration
- **Abhishek**: Content-Based Filtering & Embeddings, Docker
- **Ayoub**: GitHub Structure & Collaborative Filtering
- **Adesh**: Stream lit UI 
- **Jintian**: Presentation and report

## 📝 License

This project is part of the Recommender Systems course at McGill University.

## 🐛 Troubleshooting

### Common Issues

1. **Missing Model Files**: Ensure `models/embedding_model.pkl` exists
2. **Missing Data Files**: Ensure `data/processed/cleaned_blog_ratings.pkl` exists
3. **OpenAI API Issues**: Verify your API key is set correctly
4. **Docker Issues**: Try rebuilding the container with `docker-compose up --build`

### Getting Help

If you encounter any issues:
1. Check the console output for error messages
2. Verify all required files are present
3. Ensure your OpenAI API key is valid and has sufficient credits

## 🔮 Future Enhancements

- [ ] Collaborative filtering integration
- [ ] Real-time recommendation updates
- [ ] User preference learning
- [ ] A/B testing framework
- [ ] Performance optimization
- [ ] Mobile app version

## 🛑 Stopping the App

To stop the running Docker container and free up resources/ports, use:

```bash
docker-compose down
```

This will:
- Stop the running containers
- Remove the containers (but not your data or images)
- Free up the ports (e.g., 8000)

If you want to stop all running containers (not just this app):

```bash
docker stop $(docker ps -q)
```

To remove all stopped containers, networks, and dangling images:

```bash
docker system prune -a
```

## Hybrid Recommendation Approach

- **Guest Users:** Receive content-based recommendations using the embedding model. Only users with fewer than 5 ratings are available in the guest user dropdown.
- **Login Users:** If a user has 5 or more ratings, collaborative filtering is used. If not, the content-based model is used as a fallback. Only users with 5 or more ratings are available in the login user dropdown.
- **LLM Summarization:** Both guest and login users receive LLM-generated personalized summaries for each recommendation.
- **Advanced Filtering:** Users can filter recommendations by topic and minimum average rating.

## UI Overview
- **Landing Page:** Choose between "Continue as Guest" and "Login". Minimal, modern UI.
- **Sidebar:** User selection, recommendation controls, advanced filters, and sign out/back to home.
- **Main Area:** Recommendations are shown with all metadata (title, content, topic, author, etc.) and LLM summary.

## Dependencies
- Python 3.8+
- streamlit
- pandas
- openai
- scikit-learn
- scikit-surprise
- numpy
- python-dotenv
- matplotlib
- flake8, black, pylint (for linting/formatting)

## Running Locally
1. Install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate <env-name>
   # or
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app/app.py
   ```

## Docker
- The Dockerfile and docker-compose.yaml are set up to install all dependencies, including scikit-surprise.
- To build and run:
   ```bash
   docker build -t blog-recommender .
   docker run -p 8501:8501 blog-recommender
   ```

## .streamlit/config.toml
- The app uses a light theme and custom UI. Make sure `.streamlit/config.toml` is present for consistent appearance.

## Notes
- Do not upload sensitive files (e.g., `.env`, `.streamlit/secrets.toml`, data, models) to GitHub.
- See `.gitignore` for details.
