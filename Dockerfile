# Use miniconda as base image
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the environment file into the container first
COPY environment.yml .

# Create the Conda environment from the environment.yml file
RUN conda env create -f environment.yml

# Make Conda's run command the default shell
SHELL ["conda", "run", "-n", "blog-recommender-env", "/bin/bash", "-c"]

# Copy all project files into the container
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p models data/processed

# Expose the port that Streamlit runs on
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Launch the Streamlit app
CMD ["conda", "run", "-n", "blog-recommender-env", "streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]