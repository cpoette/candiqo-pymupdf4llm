import base64
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

SERVICE_NAME = os.getenv("SERVICE_NAME", "cv-extractor")  # ex: "docling" / "pdfparse"
MAX_BYTES = int(os.getenv("MAX_BYTES", str(15 * 1024 * 1024)))  # 15MB default

app = Flask(__name__)


# ------------------------------------------------------------
# Helpers: scoring / quality
# ------------------------------------------------------------

def compute_quality(text: str) -> Dict[str, Any]:
    text = text or ""
    char_count = len(text)

    letters = len(re.findall(r"[A-Za-zÀ-ÿ]", text))
    alpha_ratio = (letters / char_count) if char_count else 0.0

    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", text.lower())
    unique_words = len(set(words))

    lines = [ln for ln in text.splitlines() if ln.strip()]
    line_count = len(lines)
    short_lines = sum(1 for ln in lines if len(ln.strip()) < 12)
    short_line_ratio = (short_lines / line_count) if line_count else 0.0

    weird = len(re.findall(r"[�\uFFFD]", text))
    garbage_ratio = (weird / char_count) if char_count else 0.0

    return {
        "char_count": char_count,
        "alpha_ratio": round(alpha_ratio, 4),
        "unique_words": unique_words,
        "line_count": line_count,
        "short_line_ratio": round(short_line_ratio, 4),
        "garbage_ratio": round(garbage_ratio, 4),
    }


def compute_risk(quality: Dict[str, Any]) -> Dict[str, Any]:
    """
    Risk score = proxy 'risque de trahison' / structure instable.
    0 (safe) -> 100 (danger)
    """
    risk = 0
    flags: List[str] = []

    char_count = quality.get("char_count", 0)
    alpha_ratio = quality.get("alpha_ratio", 0.0)
    unique_words = quality.get("unique_words", 0)
    short_line_ratio = quality.get("short_line_ratio", 0.0)
    garbage_ratio = quality.get("garbage_ratio", 0.0)

    if char_count == 0:
        risk = 100
        flags.append("empty_text")
        return {"risk_score": risk, "flags": flags}

    if alpha_ratio < 0.25:
        risk += 40
        flags.append("low_alpha_ratio")

    if unique_words < 80:
        risk += 25
        flags.append("low_unique_words")

    if short_line_ratio > 0.55:
        risk += 30
        flags.append("layout_fragmented")

    if garbage_ratio > 0.002:
        risk += 20
        flags.append("garbage_chars")

    risk = max(0, min(100, risk))
    if risk <= 20:
        flags.append("ok")
    elif risk <= 50:
        flags.append("warn")
    else:
        flags.append("danger")

    return {"risk_score": risk, "flags": flags}


# ------------------------------------------------------------
# File handling
# ------------------------------------------------------------

@dataclass
class IncomingPdf:
    path: str
    filename: str


def _save_uploaded_pdf() -> IncomingPdf:
    """
    Accepts either:
      - multipart/form-data with 'pdf' file
      - application/json with { "filename": "...", "content_base64": "..." }
    Writes to a unique temp file and returns path.
    """
    # multipart
    if "pdf" in request.files:
        pdf_file = request.files["pdf"]
        safe_name = secure_filename(pdf_file.filename or "document.pdf")

        # Create unique temp file
        fd, temp_path = tempfile.mkstemp(prefix="pdf_", suffix="_" + safe_name)
        os.close(fd)

        pdf_file.save(temp_path)
        return IncomingPdf(path=temp_path, filename=safe_name)

    # json base64
    if request.is_json:
        data = request.get_json(silent=True) or {}
        filename = secure_filename(data.get("filename", "document.pdf"))
        content_b64 = data.get("content_base64", "")

        if not content_b64:
            raise ValueError("Missing content_base64 in JSON body")

        raw = base64.b64decode(content_b64)

        if len(raw) > MAX_BYTES:
            raise ValueError(f"File too large: {len(raw)} bytes > {MAX_BYTES}")

        fd, temp_path = tempfile.mkstemp(prefix="pdf_", suffix="_" + filename)
        os.close(fd)

        with open(temp_path, "wb") as f:
            f.write(raw)

        return IncomingPdf(path=temp_path, filename=filename)

    raise ValueError("No PDF provided. Use multipart field 'pdf' or JSON with content_base64.")


# ------------------------------------------------------------
# Extraction implementation placeholder
# ------------------------------------------------------------

def extract_text_impl(pdf_path: str, *, strategy: str = "pymupdf4llm"):
    """
    Strategies:
      - "pymupdf4llm"       : comportement actuel (pas de layout)
      - "pymupdf_layout"    : active PyMuPDF Layout (import order important)
    """
    warnings = []
    meta = {"pages": None}

    # Lazy imports (important for layout import order)
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError(f"PyMuPDF import failed: {e}")

    use_layout = (strategy == "pymupdf_layout")

    # IMPORTANT: layout must be imported BEFORE pymupdf4llm to activate it
    # https://pymupdf.readthedocs.io/.../pymupdf-layout/index.html
    if use_layout:
        try:
            import pymupdf.layout  # noqa: F401
        except Exception as e:
            warnings.append("pymupdf_layout_import_failed")
            raise RuntimeError(f"pymupdf-layout import failed: {e}")

    try:
        import pymupdf4llm
    except Exception as e:
        raise RuntimeError(f"pymupdf4llm import failed: {e}")

    # Open once
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"fitz.open failed: {e}")

    try:
        meta["pages"] = doc.page_count
    except Exception:
        warnings.append("pymupdf_page_count_failed")

    # OCR: OFF by default (keeps container simple & deterministic)
    enable_ocr = os.getenv("ENABLE_OCR", "0").strip() == "1"

    try:
        # PyMuPDF Layout works best by passing a Document object (doc) to pymupdf4llm
        # and requires the import order above.
        text = pymupdf4llm.to_markdown(
            doc,
            margins=0,
            fontsize_limit=0,
            force_text=True,
            # header/footer removal is supported in layout mode (nice for CVs)
            header=False if use_layout else True,
            footer=False if use_layout else True,
            # avoid unexpected OCR dependency unless explicitly enabled
            use_ocr=enable_ocr,
        ) or ""
    except TypeError:
        # In case older pymupdf4llm doesn't accept some params, fallback with minimal args
        warnings.append("pymupdf4llm_param_fallback")
        text = pymupdf4llm.to_markdown(doc) or ""
    except Exception as e:
        warnings.append("pymupdf4llm_failed")
        raise RuntimeError(f"pymupdf4llm.to_markdown failed: {e}")
    finally:
        try:
            doc.close()
        except Exception:
            pass

    if not text.strip():
        warnings.append("pymupdf4llm_empty_output")

    return text, meta, warnings



# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": SERVICE_NAME})


@app.post("/score")
def score():
    if not request.is_json:
        return jsonify({"error": "Expected JSON body {text: ...}"}), 400

    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    quality = compute_quality(text)
    risk = compute_risk(quality)

    # A simple readability score derived from quality/risk
    # (you can tune later; keep deterministic)
    readability_score = max(0, min(100, int(
        (quality["alpha_ratio"] * 60)
        + (min(1.0, quality["unique_words"] / 1200) * 25)
        + (min(1.0, quality["char_count"] / 12000) * 25)
        - (risk["risk_score"] * 0.5)
    )))

    return jsonify({
        "quality": quality,
        "risk": risk,
        "readability_score": readability_score,
    })

def compute_layout_signals(pdf_path: str, max_pages: int = 2) -> Dict[str, Any]:
    """
    Fast structural scan using PyMuPDF layout (NO OCR).
    Purpose: detect multi-column / table-like PDFs.
    """
    import fitz

    doc = fitz.open(pdf_path)

    blocks = []
    page_count = min(len(doc), max_pages)

    for p in range(page_count):
        page = doc[p]
        for b in page.get_text("blocks"):
            # block format: (x0, y0, x1, y1, text, block_no, block_type)
            x0, y0, x1, y1 = b[:4]
            blocks.append({
                "x0": round(x0, 1),
                "y0": round(y0, 1),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
            })

    doc.close()

    if not blocks:
        return {
            "blocks_count": 0,
            "columns_estimate": 1,
            "suspected_multicol": False,
            "table_like_ratio": 0.0,
            "sample_blocks": []
        }

    # --- Heuristics (pure geometry) ---
    xs = [b["x0"] for b in blocks]
    xs_sorted = sorted(xs)

    # crude column clustering by x-gaps
    gaps = [
        xs_sorted[i + 1] - xs_sorted[i]
        for i in range(len(xs_sorted) - 1)
    ]
    large_gaps = [g for g in gaps if g > 40]

    columns_estimate = 2 if len(large_gaps) >= len(xs_sorted) * 0.1 else 1
    suspected_multicol = columns_estimate > 1

    # table-like: many small blocks aligned in rows
    heights = [(b["y1"] - b["y0"]) for b in blocks]
    small_blocks = [h for h in heights if h < 14]
    table_like_ratio = round(len(small_blocks) / len(blocks), 3)

    return {
        "blocks_count": len(blocks),
        "columns_estimate": columns_estimate,
        "suspected_multicol": suspected_multicol,
        "table_like_ratio": table_like_ratio,
        "sample_blocks": blocks[:8],  # debug only
    }


@app.post("/extract")
def extract():
    # Optional: ?strategy=auto|docling|pymupdf4llm|pdfparse
    strategy = request.args.get("strategy", "auto")

    incoming: Optional[IncomingPdf] = None
    try:
        incoming = _save_uploaded_pdf()

        # Optional: ?strategy=auto|pymupdf4llm|pymupdf_layout
        strategy = request.args.get("strategy", "auto").strip().lower()

        # auto: garde ton comportement actuel
        impl_strategy = "pymupdf4llm" if strategy in ("auto", "", None) else strategy

        text, meta, warnings = extract_text_impl(incoming.path, strategy=impl_strategy)

        layout_signals = compute_layout_signals(incoming.path)


        quality = compute_quality(text)
        risk = compute_risk(quality)

        # Keep response stable, even if empty
        return jsonify({
            "strategy_used": SERVICE_NAME if strategy == "auto" else strategy,
            "text": text,
            "meta": {
                "filename": incoming.filename,
                **(meta or {}),
            },
            "quality": quality,
            "risk": risk,
            "layout_signals": layout_signals,
            "warnings": warnings or [],
        })


    except Exception as e:
        return jsonify({
            "strategy_used": SERVICE_NAME,
            "error": str(e),
        }), 400

    finally:
        # Always cleanup temp file
        try:
            if incoming and incoming.path and os.path.exists(incoming.path):
                os.remove(incoming.path)
        except Exception:
            pass


if __name__ == "__main__":
    # Local dev only; in Docker we use gunicorn
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
