import base64
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

SERVICE_NAME = os.getenv("SERVICE_NAME", "cv-extractor")  # ex: "docling" / "pymupdf4llm"
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
# Layout signals (NO OCR) - based on PyMuPDF dict blocks
# ------------------------------------------------------------

def compute_layout_signals_from_blocks(blocks, page_width: float, bins: int = 30) -> Dict[str, Any]:
    # Keep only text blocks (lines/spans)
    text_blocks = []
    for b in blocks or []:
        if not b.get("lines"):
            continue
        bbox = b.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = map(float, bbox)
        w = max(0.0, x1 - x0)
        h = max(0.0, y1 - y0)
        text_blocks.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "w": w, "h": h})

    if not text_blocks:
        return {
            "blocks_count": 0,
            "blocks_useful_count": 0,
            "columns_estimate": 1,
            "suspected_multicol": False,
            "x0_histogram": [0] * bins,
            "x0_peaks_bins": [],
            "interleave_ratio": 0.0,
        }

    # Filter parasites (tiny blocks + full-width skinny separators)
    useful = []
    for b in text_blocks:
        if b["w"] < 8 or b["h"] < 6:
            continue
        if b["w"] > 0.92 * page_width and b["h"] < 25:
            continue
        useful.append(b)
    if not useful:
        useful = text_blocks[:]

    # Histogram x0 (ignore extreme left margin artifacts)
    hist = [0] * bins
    margin_left = 0.03 * page_width

    for b in useful:
        x = b["x0"]
        if x < margin_left:
            continue
        x = max(0.0, min(page_width - 1e-6, x))
        idx = int((x / page_width) * bins)
        idx = max(0, min(bins - 1, idx))
        hist[idx] += 1

    # Peaks
    peaks = []
    for i in range(bins):
        left = hist[i - 1] if i - 1 >= 0 else -1
        right = hist[i + 1] if i + 1 < bins else -1
        if hist[i] > 0 and hist[i] >= left and hist[i] >= right:
            peaks.append(i)

    maxv = max(hist) if hist else 0
    strong = [i for i in peaks if hist[i] >= max(2, int(0.35 * maxv))]

    # Never allow peaks in first bins (bin0/1 = margin noise)
    strong = [i for i in strong if i >= 2]

    columns_estimate = 2 if len(strong) >= 2 else 1
    suspected_multicol = columns_estimate >= 2

    # Interleaving: sort by y then x and count switches between peak bins
    sorted_blocks = sorted(useful, key=lambda b: (round(b["y0"] / 10) * 10, b["x0"]))

    def block_bin(b):
        x = b["x0"]
        if x < margin_left:
            return None
        bi = int((max(0.0, min(page_width - 1e-6, x)) / page_width) * bins)
        bi = max(0, min(bins - 1, bi))
        if not strong:
            return bi
        # snap to nearest strong peak bin
        return min(strong, key=lambda p: abs(p - bi))

    seq = [block_bin(b) for b in sorted_blocks]
    seq = [s for s in seq if s is not None]

    switches = 0
    for i in range(1, len(seq)):
        if seq[i] != seq[i - 1]:
            switches += 1
    interleave_ratio = switches / max(1, len(seq) - 1)

    return {
        "blocks_count": len(text_blocks),
        "blocks_useful_count": len(useful),
        "columns_estimate": columns_estimate,
        "suspected_multicol": suspected_multicol,
        "x0_histogram": hist,
        "x0_peaks_bins": strong,
        "interleave_ratio": round(float(interleave_ratio), 3),
    }


def compute_layout_chaos(layout_signals: Dict[str, Any]) -> int:
    """
    IMPORTANT: this is a *layout-only* chaos.
    Downstream (n8n/SSOT) should dampen it with content-gates.
    """
    chaos = 0
    try:
        inter = float(layout_signals.get("interleave_ratio", 0.0) or 0.0)
        blocks_useful = int(layout_signals.get("blocks_useful_count", 0) or 0)
        suspected = bool(layout_signals.get("suspected_multicol", False))

        if suspected:
            chaos += 20
            chaos += int(inter * 80)  # 0..80-ish
        else:
            # if single col, interleave shouldn't drive chaos
            chaos += int(inter * 20)

        # spam blocks penalty (layout fragmentation)
        chaos += min(30, max(0, blocks_useful - 35))

        chaos = max(0, min(100, int(chaos)))
    except Exception:
        chaos = 0
    return chaos


# ------------------------------------------------------------
# Extraction implementation
# ------------------------------------------------------------

def extract_text_impl(pdf_path: str, *, strategy: str = "pymupdf4llm"):
    """
    Strategies:
      - "pymupdf4llm"       : current behavior
      - "pymupdf_layout"    : activates PyMuPDF Layout (import order important)
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

    # layout signals from first page blocks (NO OCR)
    layout_signals = None
    chaos_layout = 0
    try:
        page = doc[0]
        raw = page.get_text("dict")
        blocks = raw.get("blocks", [])
        layout_signals = compute_layout_signals_from_blocks(blocks, page_width=float(page.rect.width), bins=30)
        chaos_layout = compute_layout_chaos(layout_signals)
    except Exception:
        warnings.append("pymupdf_layout_signals_failed")

    # page count
    try:
        meta["pages"] = doc.page_count
    except Exception:
        warnings.append("pymupdf_page_count_failed")

    # OCR OFF by default
    enable_ocr = os.getenv("ENABLE_OCR", "0").strip() == "1"

    try:
        text = pymupdf4llm.to_markdown(
            doc,
            margins=0,
            fontsize_limit=0,
            force_text=True,
            header=False if use_layout else True,
            footer=False if use_layout else True,
            use_ocr=enable_ocr,
        ) or ""
    except TypeError:
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

    # ✅ SSOT: ONE place only
    meta_out = {
        **meta,
        "layout_signals": layout_signals,
        "chaos_layout": chaos_layout,
    }

    # --- Debug meta to prove which branch ran ---
    try:
        import pymupdf4llm
        pymupdf4llm_version = getattr(pymupdf4llm, "__version__", None)
    except Exception:
        pymupdf4llm_version = None

    meta_debug = {
        "impl_strategy": strategy,
        "use_layout": use_layout,
        "enable_ocr": enable_ocr,
        "pymupdf4llm_version": pymupdf4llm_version,
        "used_param_fallback": ("pymupdf4llm_param_fallback" in warnings),
    }

    return text, {**meta, "layout_signal": layout_signals, "chaos": chaos, "debug": meta_debug}, warnings


   # return text, meta_out, warnings


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


@app.post("/extract")
def extract():
    # ?strategy=auto|pymupdf4llm|pymupdf_layout
    strategy_q = (request.args.get("strategy", "auto") or "auto").strip().lower()

    incoming: Optional[IncomingPdf] = None
    try:
        incoming = _save_uploaded_pdf()

        impl_strategy = "pymupdf4llm" if strategy_q in ("auto", "", None) else strategy_q

        text, meta, warnings = extract_text_impl(incoming.path, strategy=impl_strategy)

        quality = compute_quality(text)
        risk = compute_risk(quality)

        # Keep response stable
        return jsonify({
            "strategy_used": impl_strategy,
            "text": text,
            "meta": {
                "filename": incoming.filename,
                **(meta or {}),
            },
            "quality": quality,
            "risk": risk,
            "warnings": warnings or [],
        })

    except Exception as e:
        return jsonify({
            "strategy_used": SERVICE_NAME,
            "error": str(e),
        }), 400

    finally:
        try:
            if incoming and incoming.path and os.path.exists(incoming.path):
                os.remove(incoming.path)
        except Exception:
            pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
