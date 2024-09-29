# Base image with Python
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for LightGBM (including libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
    
# Copy the requirements.txt file into the container
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Set the Streamlit environment variables
ENV STREAMLIT_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
