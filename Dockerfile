FROM python:3.11-slim

WORKDIR /app

# Install system deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Force rembg to download models to a local directory so it persists permissions across user bounds
ENV U2NET_HOME=/app/.u2net
RUN python -c "import os; os.environ['U2NET_HOME'] = '/app/.u2net'; from rembg import new_session; new_session('bria')"
RUN chmod -R 777 /app/.u2net

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
