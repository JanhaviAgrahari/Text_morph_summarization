# ============================================================
# Optimized Dockerfile for Text Morph: Advanced Summarization using AI
# Faster rebuilds using Docker layer caching
# ============================================================

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# ------------------------------------------------------------
# Step 1: System dependencies (only installed once)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Step 2: Copy only dependency files (cached layer)
# ------------------------------------------------------------
COPY requirements.txt .

# Install dependencies (cached unless requirements.txt changes)
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# Step 3: Copy the rest of the project files
# ------------------------------------------------------------
COPY . .


# ------------------------------------------------------------
# Step 4: Environment variables
# ------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV BACKEND_PORT=5000
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=${PORT}
ENV STREAMLIT_SERVER_ENABLECORS=false

# Cloud Run exposes only one external port (PORT)
EXPOSE ${PORT}

# ------------------------------------------------------------
# Step 5: Start script setup
# ------------------------------------------------------------
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Optional: create non-root user for better security
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

# ------------------------------------------------------------
# Step 6: Default start command
# ------------------------------------------------------------
CMD ["/app/start.sh"]

