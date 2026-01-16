FROM python:3.11-alpine

WORKDIR /app

# Install dependencies
RUN apk add --no-cache build-base libffi-dev
RUN pip install --no-cache-dir flask pymupdf4llm gunicorn

# Copy app
COPY app.py /app/app.py

EXPOSE 5000

# Run with gunicorn (production-ready)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "2", "app:app"]