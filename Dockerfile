# ============================================================
# Dockerfile for Text Morph: Advanced Summarization using AI
# Python 3.10 | Streamlit + FastAPI | Torch + Transformers
# ============================================================

FROM python:3.10-slim

# Set workdir
WORKDIR /app

# System dependencies (for torch, pdfminer, pypdf2, etc.)
# Note: libgl1-mesa-glx is not present in some newer Debian slim images; use libgl1 instead.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV BACKEND_PORT=5000
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=${PORT}
ENV STREAMLIT_SERVER_ENABLECORS=false

EXPOSE ${PORT} ${BACKEND_PORT}

# Copy and make start script executable
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Create a non-root user (optional but recommended)
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

# Start both FastAPI (backend) and Streamlit (frontend)
CMD ["/start.sh"]
