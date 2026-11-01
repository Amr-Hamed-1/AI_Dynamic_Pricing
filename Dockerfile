FROM python:3.12-slim

WORKDIR /app

# System dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --default-timeout=1000 --retries 10 --no-cache-dir -r requirements.txt

# Project source & app
COPY . .

# Generate synthetic data, train model, and run out-of-time backtest at build time
# so the image is completely self-contained.
RUN python src/generate_data.py \
    && python src/train.py \
    && python src/backtest.py

# Expose FastAPI (8000) & Streamlit (8501) ports
EXPOSE 8000 8501

COPY start.sh /start.sh
RUN chmod +x /start.sh
ENTRYPOINT ["/start.sh"]