FROM python:3.10-slim

# Set up environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install system dependencies if any are needed for FAISS or OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to respect HuggingFace Spaces strict requirements
RUN useradd -m -u 1000 user
USER user
WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project code and heavy data assets into the container image
COPY --chown=user . .

# Expose port 7860 for HF Spaces
EXPOSE 7860

# Run the backend using Uvicorn directly
CMD ["python", "-m", "uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "7860"]
