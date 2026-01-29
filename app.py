import base64
from curses import raw
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

def compute_layout_signals_from_blocks(blocks, page_width=None, bins=20):
    """
    blocks: list of dicts with x0,x1,y0,y1 (or bbox)
    Returns layout_signals with:
      - x0_histogram
      - x0_peaks_bins
      - columns_estimate
      - suspected_multicol
      - interleave_ratio  (NEW)
    """
    import math

    # Normalize blocks
    b2 = []
    for b in blocks or []:
        if "bbox" in b and len(b["bbox"]) == 4:
            x0, y0, x1, y1 = b["bbox"]
        else:
            x0, y0, x1, y1 = b.get("x0"), b.get("y0"), b.get("x1"), b.get("y1")
        if x0 is None or y0 is None or x1 is None or y1 is None:
            continue
        w = max(0.0, float(x1) - float(x0))
        h = max(0.0, float(y1) - float(y0))
        b2.append({"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1), "w": w, "h": h})

    if not b2:
        return {
            "blocks_count": 0,
            "blocks_useful_count": 0,
            "columns_estimate": 1,
            "suspected_multicol": False,
            "x0_histogram": [0]*bins,
            "x0_peaks_bins": [],
            "interleave_ratio": 0.0,
        }

    # Page width (fallback: max x1)
    if not page_width:
        page_width = max(b["x1"] for b in b2) or 1.0

    # "useful" filter (avoid tiny dots)
    useful = [b for b in b2 if b["w"] >= 8 and b["h"] >= 6]
    if not useful:
        useful = b2[:]  # fallback: keep all

    # Histogram x0
    hist = [0]*bins
    for b in useful:
        x = max(0.0, min(page_width - 1e-6, b["x0"]))
        idx = int((x / page_width) * bins)
        idx = max(0, min(bins-1, idx))
        hist[idx] += 1

    # Find peaks (simple local maxima)
    peaks = []
    for i in range(bins):
        left = hist[i-1] if i-1 >= 0 else -1
        right = hist[i+1] if i+1 < bins else -1
        if hist[i] > 0 and hist[i] >= left and hist[i] >= right:
            peaks.append(i)

    # Keep only "strong" peaks
    maxv = max(hist) if hist else 0
    strong_peaks = [i for i in peaks if hist[i] >= max(2, int(0.35 * maxv))]

    columns_estimate = 2 if len(strong_peaks) >= 2 else 1
    suspected_multicol = columns_estimate >= 2

    # NEW: interleaving ratio
    # Sort by y0 with coarse rounding to keep same "row" together
    sorted_blocks = sorted(useful, key=lambda b: (round(b["y0"] / 10) * 10, b["x0"]))

    # Assign each block to nearest peak bin (or its own bin if no peaks)
    def block_bin(b):
        x = max(0.0, min(page_width - 1e-6, b["x0"]))
        bi = int((x / page_width) * bins)
        bi = max(0, min(bins-1, bi))
        if not strong_peaks:
            return bi
        # nearest peak
        return min(strong_peaks, key=lambda p: abs(p - bi))

    seq = [block_bin(b) for b in sorted_blocks]

    # Count switches (A->B where A != B)
    switches = 0
    for i in range(1, len(seq)):
        if seq[i] != seq[i-1]:
            switches += 1

    interleave_ratio = switches / max(1, (len(seq) - 1))

    return {
        "blocks_count": len(b2),
        "blocks_useful_count": len(useful),
        "columns_estimate": columns_estimate,
        "suspected_multicol": suspected_multicol,
        "x0_histogram": hist,
        "x0_peaks_bins": strong_peaks,
        "interleave_ratio": round(interleave_ratio, 3),
    }


def pymupdf_layout_chaos_signals(pdf_path: str, page_index: int = 0) -> Dict[str, Any]:
    import fitz
    import numpy as np
    import re

    doc = fitz.open(pdf_path)
    page = doc[page_index]
    d = page.get_text("dict")
    doc.close()

    blocks = d.get("blocks", []) or []

    text_blocks = []
    for b in blocks:
        if b.get("type") != 0:  # 0 = text
            continue
        bbox = b.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        # collect text from spans
        txt_parts = []
        for ln in b.get("lines", []) or []:
            for sp in ln.get("spans", []) or []:
                t = sp.get("text", "")
                if t:
                    txt_parts.append(t)
        text = " ".join(txt_parts).strip()
        if not text:
            continue

        x0, y0, x1, y1 = bbox
        words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", text)
        text_blocks.append({
            "x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1),
            "xc": float((x0 + x1) / 2.0),
            "w": float(x1 - x0),
            "h": float(y1 - y0),
            "words": len(words),
        })

    n = len(text_blocks)
    if n == 0:
        return {
            "blocks_text_count": 0,
            "chaos_score": 100,
            "flags": ["no_text_blocks"]
        }

    # --- columns estimate via x0 peaks (simple & fast) ---
    xcs = np.array([b["xc"] for b in text_blocks], dtype=np.float32)
    # histogram bins
    bins = 30
    hist, edges = np.histogram(xcs, bins=bins)
    # find peaks bins (top 2)
    peak_bins = hist.argsort()[-2:][::-1]
    peak_bins = sorted([int(x) for x in peak_bins])
    # heuristic: if two peaks have meaningful mass -> multicol
    total = hist.sum() if hist.sum() else 1
    peak_mass = (hist[peak_bins[0]] + hist[peak_bins[1]]) / total
    suspected_multicol = (peak_mass > 0.45 and abs(peak_bins[1] - peak_bins[0]) >= 5)

    x_split = None
    if suspected_multicol:
        # split between peaks
        left_edge = edges[peak_bins[0] + 1]
        right_edge = edges[peak_bins[1]]
        x_split = float((left_edge + right_edge) / 2.0)

    # --- fragmentation ---
    words_arr = np.array([b["words"] for b in text_blocks], dtype=np.int32)
    median_block_words = float(np.median(words_arr)) if n else 0.0

    # --- choose traversal order to measure "native chaos" ---
    # Option A: native order returned by pymupdf dict (closest to internal order)
    # We'll just keep current text_blocks order (already in that order).
    # If you want: can compute both native + (y,x) and keep worst.
    blocks_native = text_blocks

    # --- y backtrack rate ---
    back = 0
    prev_y = blocks_native[0]["y0"]
    for b in blocks_native[1:]:
        if b["y0"] < prev_y - 3:  # small tolerance
            back += 1
        prev_y = b["y0"]
    y_backtrack_rate = back / max(1, n - 1)

    # --- column switch rate ---
    col_switch_rate = 0.0
    if suspected_multicol and x_split is not None:
        cols = [0 if b["xc"] < x_split else 1 for b in blocks_native]
        switches = sum(1 for i in range(1, len(cols)) if cols[i] != cols[i-1])
        col_switch_rate = switches / max(1, n - 1)

    # --- chaos score (tunable) ---
    chaos = 0
    if n > 120: chaos += 30
    if median_block_words < 4: chaos += 20
    if y_backtrack_rate > 0.12: chaos += 25
    if suspected_multicol and col_switch_rate > 0.20: chaos += 25

    chaos = max(0, min(100, chaos))

    flags = []
    if chaos >= 55: flags.append("layout_chaotic")
    else: flags.append("layout_stable")

    return {
        "blocks_text_count": n,
        "median_block_words": round(median_block_words, 2),
        "suspected_multicol": bool(suspected_multicol),
        "x_split": int(x_split) if x_split is not None else None,
        "y_backtrack_rate": round(float(y_backtrack_rate), 4),
        "col_switch_rate": round(float(col_switch_rate), 4),
        "chaos_score": int(chaos),
        "flags": flags,
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

    layout_signals = None
    page = doc[0]
    raw = page.get_text("dict")
    blocks = raw.get("blocks", [])
    
    try:
        layout_signals = compute_layout_signals_from_blocks(blocks, page_width=page.rect.width, bins=30)
    except Exception:
        warnings.append("pymupdf_layout_signals_failed")

    try:
        chaos = 0
        if layout_signals["suspected_multicol"]:
            chaos += 20
        chaos += int(layout_signals["interleave_ratio"] * 80)   # 0..80
        chaos += min(30, max(0, layout_signals["blocks_useful_count"] - 35))  # spam blocks
        chaos = max(0, min(100, chaos))
    except Exception:
        warnings.append("cannot compute_layout_chaos_score")

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

    return text, {**meta, "layout_signal": layout_signals, "chaos": chaos}, warnings


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
    bins = 30
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
