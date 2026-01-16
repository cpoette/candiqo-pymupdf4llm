# PDF Extractor Service

Microservice pour extraction de texte depuis PDF avec pymupdf4llm.

## Endpoints

### GET /health

Health check du service.

**Response:**

```json
{ "status": "ok", "service": "pdf-extractor" }
```

### POST /extract

Extrait le texte d'un PDF en markdown.

**Request:**

- Method: POST
- Content-Type: multipart/form-data
- Body: fichier PDF avec key "pdf"

**Response:**

```json
{
  "success": true,
  "markdown": "contenu en markdown...",
  "extraction_method": "pymupdf4llm",
  "filename": "cv.pdf",
  "char_count": 1234
}
```

## Build local

```bash
docker build -t pdf-extractor .
docker run -p 5000:5000 pdf-extractor
```

## Test

```bash
curl -X POST -F "pdf=@test.pdf" http://localhost:5000/extract
```

```

### 4. .gitignore
```

**pycache**/
_.pyc
_.pyo
.env
.venv/
\*.pdf
/tmp/
