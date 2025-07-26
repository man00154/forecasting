# Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for Prophet and other packages.
# 'build-essential', 'gcc', 'python3-dev', 'musl-dev' are crucial for
# compiling 'pystan', a dependency of 'prophet'.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    musl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces the size of the Docker image by not storing build artifacts
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ITSM data file from the host to the container's /app directory
COPY ITSM_data.csv .

# Copy the application code from the host's 'app/' directory to the container's '/app/app/'
COPY app/ ./app/

# Create a directory for models if it doesn't exist.
# This is where trained models will be saved.
RUN mkdir -p app/models

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run the uvicorn server when the container launches.
# 'app.main:app' refers to the 'app' object in 'main.py' inside the 'app' directory.
# '--host 0.0.0.0' makes the server accessible from outside the container.
# '--port 8000' specifies the port the server listens on.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
