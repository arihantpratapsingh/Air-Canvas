FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for mediapipe & opencv
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0