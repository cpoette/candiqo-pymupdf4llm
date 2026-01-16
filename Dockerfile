# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    WORKERS=1 \
    THREADS=4 \
    TIMEOUT=180

WORKDIR /app

# (Optionnel mais recommandé) : paquets de base utiles (certs, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Déps Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App
COPY app.py /app/app.py

EXPOSE 5000

# Prod server
CMD ["sh", "-c", "gunicorn -w ${WORKERS} --threads ${THREADS} -t ${TIMEOUT} -b 0.0.0.0:${PORT} app:app"]

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health').read()"
