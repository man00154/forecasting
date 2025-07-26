# Dockerfile
# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker's caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
# This includes app.py, your trained model (e.g., forecast_model.pkl),
# and ITSM_data.csv if your app needs it at runtime.
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run your FastAPI application using Uvicorn
# The --host 0.0.0.0 is crucial for accessibility from outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
