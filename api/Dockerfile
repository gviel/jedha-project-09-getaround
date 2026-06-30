FROM python:3.12-slim

WORKDIR /app

# Build context = racine du projet (docker build -f api/Dockerfile .)
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/app.py .
COPY models/ models/

EXPOSE 8000

CMD ["sh", "-c", "gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000}"]
