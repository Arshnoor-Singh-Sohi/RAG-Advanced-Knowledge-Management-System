# Use the official Python 3.12.7 image as the base
FROM python:3.12.7

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies that are often required
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# (Optional) Uncomment the next line if you need PDF processing utilities
# RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy only the requirements file first to take advantage of Docker cache
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Define the command to run your application
CMD ["python3", "app/main.py"]
