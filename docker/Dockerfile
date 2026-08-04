# Dockerfile for the LLM Web Search app.
#
# Build (from the project root):
#   docker build -t yourusername/llm-web-search-app:1.0 -f docker/Dockerfile .
#
# Run with the full stack (SearXNG + Redis + app):
#   docker compose -f docker/docker-compose.app.yml up -d

# Step 1: Start from a base image with Python 3.11
FROM python:3.11-slim

# Step 2: Tell Python not to write .pyc files (keeps image smaller)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Step 3: Install the system library that crawl4ai/trafilatura need
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Create a folder for the app inside the container
WORKDIR /app

# Step 5: Copy the requirements file first (Docker caches this layer)
COPY requirements.txt .

# Step 6: Install all Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Step 7: Install the headless browser that crawl4ai drives
RUN playwright install chromium --with-deps

# Step 8: Copy all the project files into the container
COPY . .

# Step 9: Tell Docker which port the app uses
EXPOSE 8501

# Step 10: Command to start the app when the container runs.
# "app.main" is the Streamlit entry point inside the app/ package.
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
