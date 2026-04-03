FROM python:3.11-slim

WORKDIR /app

# Install system deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download SAM weights during build to bake them into the image
RUN python -c "from transformers import SamModel, SamProcessor; SamProcessor.from_pretrained('facebook/sam-vit-base'); SamModel.from_pretrained('facebook/sam-vit-base')"
# HF Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
