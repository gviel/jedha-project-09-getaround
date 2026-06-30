FROM python:3.12-slim

WORKDIR /app

# Build context = racine du projet (docker build -f dashboard/Dockerfile .)
COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard/app.py .
COPY dashboard/getaround_logo.png .
COPY data/ data/

EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true"]
