import base64
import os
import re
import tempfile
import numpy as np
import cv2
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


def _render_page_to_image(pdf_path: str, page_index: int = 0, zoom: float = 2.0):
    """
    Render a PDF page to an RGB numpy array using PyMuPDF.
    This is NOT OCR: we just rasterize the PDF for layout analysis.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    page = doc[page_index]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    doc.close()

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def opencv_layout_signals(pdf_path: str, page_index: int = 0) -> Dict[str, Any]:
    """
    POC: detect multi-column layout using OpenCV (no OCR).
    Returns signals:
      - suspected_multicol
      - columns_estimate (1/2)
      - x_split (estimated separation x in pixels, if any)
      - separators_count
    """
    img = _render_page_to_image(pdf_path, page_index=page_index, zoom=2.0)

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize (text tends to be dark)
    # Using adaptive threshold to handle different backgrounds.
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,  # blockSize
        15   # C
    )

    # Remove small noise (optional)
    thr = cv2.medianBlur(thr, 3)

    # Compute vertical projection (sum of black pixels per column)
    col_sum = thr.sum(axis=0).astype(np.float32) / 255.0  # count of "ink" pixels per x

    # Smooth projection
    k = max(9, int(w * 0.01) // 2 * 2 + 1)  # odd kernel ~1% width
    col_sum_smooth = cv2.GaussianBlur(col_sum.reshape(1, -1), (k, 1), 0).flatten()

    # Normalize
    maxv = float(col_sum_smooth.max()) if col_sum_smooth.size else 1.0
    norm = col_sum_smooth / (maxv if maxv > 0 else 1.0)

    # Look for a "valley" near center -> possible column gap
    center = w // 2
    span = int(w * 0.25)  # search in middle 50% area
    lo = max(0, center - span)
    hi = min(w - 1, center + span)

    valley_x = int(lo + np.argmin(norm[lo:hi+1])) if hi > lo else center
    valley_val = float(norm[valley_x])

    # Heuristic: if the middle valley is "low enough" compared to average, likely 2 columns.
    # (Tune later with your corpus)
    avg_val = float(norm.mean()) if norm.size else 1.0

    suspected_multicol = (valley_val < 0.25 and avg_val > 0.12)

    # Also count how wide the valley region is (continuous low area)
    low_mask = norm < 0.22
    # find contiguous low segment containing valley_x
    left = valley_x
    while left > 0 and low_mask[left]:
        left -= 1
    right = valley_x
    while right < w - 1 and low_mask[right]:
        right += 1
    valley_width = int(right - left)

    # columns estimate: 2 if suspected, else 1
    columns_estimate = 2 if suspected_multicol else 1

    return {
        "page_index": page_index,
        "img_w": w,
        "img_h": h,
        "suspected_multicol": suspected_multicol,
        "columns_estimate": columns_estimate,
        "x_split": int(valley_x) if suspected_multicol else None,
        "valley_val": round(valley_val, 4),
        "avg_val": round(avg_val, 4),
        "valley_width": valley_width,
    }



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

    layout_signals_opencv = None
    try:
        layout_signals_opencv = opencv_layout_signals(pdf_path, page_index=0)
    except Exception:
        warnings.append("opencv_layout_failed")


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

    return text, {**meta, "layout_signals_opencv": layout_signals_opencv}, warnings


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
    Fast structural scan using PyMuPDF (NO OCR).
    Purpose: detect multi-column / table-ish layouts.
    """
    import fitz

    doc = fitz.open(pdf_path)

    all_blocks = []
    widths = []
    page_w = None

    page_count = min(len(doc), max_pages)
    for p in range(page_count):
        page = doc[p]
        page_w = float(page.rect.width)

        for b in page.get_text("blocks"):
            # (x0, y0, x1, y1, text, block_no, block_type)
            x0, y0, x1, y1 = b[:4]
            w = float(x1 - x0)
            h = float(y1 - y0)

            all_blocks.append({
                "x0": round(float(x0), 1),
                "y0": round(float(y0), 1),
                "x1": round(float(x1), 1),
                "y1": round(float(y1), 1),
                "w": round(w, 1),
                "h": round(h, 1),
            })
            widths.append(w)

    doc.close()

    if not all_blocks or not page_w:
        return {
            "blocks_count": 0,
            "columns_estimate": 1,
            "suspected_multicol": False,
            "table_like_ratio": 0.0,
            "sample_blocks": []
        }

    # ---------- filter: keep "content-like" blocks ----------
    # Drop tiny blocks (bullets, separators, noise)
    useful = []
    for b in all_blocks:
        if b["w"] < 35:      # too narrow (bullet column)
            continue
        if b["h"] < 8:       # too short (lines / separators)
            continue
        useful.append(b)

    # If we filtered too aggressively, fallback to all blocks
    if len(useful) < max(8, int(len(all_blocks) * 0.25)):
        useful = all_blocks

    # ---------- table-like ratio ----------
    # A lot of short-height blocks suggests tables / fragmented layout
    small_blocks = [b for b in useful if b["h"] < 14]
    table_like_ratio = round(len(small_blocks) / len(useful), 3) if useful else 0.0

    # ---------- multi-column detection via histogram peaks ----------
    # Normalize x0 to [0..1]
    xs = [b["x0"] / page_w for b in useful]
    xs = [x for x in xs if 0.0 <= x <= 1.0]

    # Histogram bins
    bins = 20
    hist = [0] * bins
    for x in xs:
        idx = min(bins - 1, max(0, int(x * bins)))
        hist[idx] += 1

    # Find peaks: local maxima above a small threshold
    # threshold adapts to sample size
    thr = max(2, int(len(xs) * 0.08))
    peaks = []
    for i in range(1, bins - 1):
        if hist[i] >= thr and hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]:
            peaks.append(i)

    # Merge adjacent peaks (plateaus)
    merged = []
    for pi in peaks:
        if not merged or pi - merged[-1] > 1:
            merged.append(pi)

    # Decide columns: 2 peaks that are sufficiently separated
    columns_estimate = 1
    suspected_multicol = False
    if len(merged) >= 2:
        # compute separation in normalized x
        # take best-separated two peaks
        best_sep = 0
        for a in merged:
            for b in merged:
                if b <= a:
                    continue
                sep = abs(b - a) / bins
                if sep > best_sep:
                    best_sep = sep
        if best_sep >= 0.18:  # ~18% of page width between peaks
            columns_estimate = 2
            suspected_multicol = True

    return {
        "blocks_count": len(all_blocks),
        "blocks_useful_count": len(useful),
        "columns_estimate": columns_estimate,
        "suspected_multicol": suspected_multicol,
        "table_like_ratio": table_like_ratio,
        "x0_histogram": hist,          # useful for debugging / tuning
        "x0_peaks_bins": merged,       # idem
        "sample_blocks": all_blocks[:8],
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
