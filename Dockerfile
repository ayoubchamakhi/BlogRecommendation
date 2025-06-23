# Stage 1: Define the base image
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the environment file into the container first.
COPY environment.yml .

# Create the Conda environment from the unified environment.yml file.
RUN conda env create -f environment.yml

# Make Conda's run command the default shell. This ensures all subsequent
SHELL ["conda", "run", "-n", "recommender_project", "/bin/bash", "-c"]

# Now, copy all of your project code into the container's working directory.
COPY . .

# Expose the port that Streamlit runs on.
EXPOSE 8501

# This will launch the Streamlit app.
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]