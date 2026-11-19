"""Step 1-2: OCR + structured field extraction from an answer-sheet image.

Backends
--------
* Tesseract (default, via pytesseract) - needs the `tesseract` binary.
* EasyOCR (optional) - used automatically if `pip install easyocr` is done and
  selected by the caller.

The sheet layout is:
    Student ID: MCA001            Question: Q1
    Question:
    <question text>
    Student Answer:
    <answer text>
    End of answer
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image


@dataclass
class OCRResult:
    raw_text: str = ""
    student_id: str = ""
    question_id: str = ""
    question_text: str = ""
    answer_text: str = ""
    engine: str = "tesseract"
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Image pre-processing (Computer Vision part)
# --------------------------------------------------------------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Grayscale -> upscale -> denoise -> adaptive/Otsu threshold.

    Improves OCR accuracy on photographs/scans as well as on clean renders.
    """
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    if w < 1800:  # upscale small images; Tesseract works best at ~300 DPI
        scale = 1800 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------
def _run_tesseract(arr: np.ndarray) -> str:
    import pytesseract

    return pytesseract.image_to_string(arr, config="--oem 3 --psm 6")


_easyocr_reader = None


def _run_easyocr(arr: np.ndarray) -> str:
    global _easyocr_reader
    import easyocr  # optional dependency

    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    # paragraph=False keeps line granularity; sort top->bottom
    results = _easyocr_reader.readtext(arr, detail=1, paragraph=False)
    results.sort(key=lambda r: (round(r[0][0][1] / 20), r[0][0][0]))
    # merge boxes that sit on the same visual line
    lines, current, last_y = [], [], None
    for box, text, _conf in results:
        y = box[0][1]
        if last_y is not None and abs(y - last_y) > 20:
            lines.append("   ".join(current))
            current = []
        current.append(text)
        last_y = y
    if current:
        lines.append("   ".join(current))
    return "\n".join(lines)


def available_engines() -> list[str]:
    engines = []
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        engines.append("tesseract")
    except Exception:
        pass
    try:
        import easyocr  # noqa: F401

        engines.append("easyocr")
    except Exception:
        pass
    return engines


# --------------------------------------------------------------------------
# Field parsing
# --------------------------------------------------------------------------
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8"})


def _normalise_student_id(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]", "", s).upper()
    m = re.match(r"^([A-Z]{2,4}?)([A-Z0-9]{2,5})$", s)
    if s.startswith("MCA") and len(s) > 3:
        return "MCA" + s[3:].translate(_DIGIT_FIX)
    return s


def _normalise_question_id(s: str) -> str:
    m = re.search(r"[Qq]\s*([0-9OoIlSB|]{1,2})", s)
    if not m:
        return ""
    return "Q" + m.group(1).translate(_DIGIT_FIX)


def parse_sheet(text: str) -> OCRResult:
    res = OCRResult(raw_text=text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # ---- student id & question id (header line(s)) ----------------------
    for ln in lines:
        m = re.search(r"student\s*id\s*[:\-]?\s*([A-Za-z0-9|]+)", ln, re.I)
        if m and not res.student_id:
            res.student_id = _normalise_student_id(m.group(1))
        m = re.search(r"question\s*(?:no\.?|number|id)?\s*[:\-]?\s*(Q\s*[0-9OoIlSB|]{1,2})\b", ln, re.I)
        if m and not res.question_id:
            res.question_id = _normalise_question_id(m.group(1))
    if not res.student_id:
        m = re.search(r"\bMCA\s*[0-9OoIl]{3}\b", text, re.I)
        if m:
            res.student_id = _normalise_student_id(m.group(0))

    # ---- locate the label lines ------------------------------------------
    idx_q = idx_a = idx_end = None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if idx_q is None and re.match(r"^question\s*[:\-]?\s*$", low):
            idx_q = i
        elif idx_a is None and re.match(r"^student\s*answer\s*[:\-]?", low):
            idx_a = i
        elif idx_end is None and re.match(r"^end\s*of\s*answer", low):
            idx_end = i

    if idx_q is not None and idx_a is not None and idx_q < idx_a:
        res.question_text = " ".join(lines[idx_q + 1 : idx_a]).strip()
    else:
        # fallback: question text on the same line as the label, e.g. "Question 1: What is NLP?"
        for ln in lines:
            m = re.match(r"^question\s*(?:no\.?)?\s*\d*\s*[:\-]\s*(.{8,})$", ln, re.I)
            if m and not re.match(r"^Q\s*\d+$", m.group(1).strip(), re.I):
                res.question_text = m.group(1).strip()
                break

    if idx_a is not None:
        first = re.sub(r"^student\s*answer\s*[:\-]?\s*", "", lines[idx_a], flags=re.I)
        stop = idx_end if idx_end is not None and idx_end > idx_a else len(lines)
        body = ([first] if first else []) + lines[idx_a + 1 : stop]
        res.answer_text = " ".join(body).strip()

    # ---- warnings --------------------------------------------------------
    if not res.student_id:
        res.warnings.append("Could not read Student ID.")
    if not res.question_id:
        res.warnings.append("Could not read Question ID from the header.")
    if not res.question_text:
        res.warnings.append("Could not isolate the question text.")
    if not res.answer_text:
        res.warnings.append("Could not isolate the student's answer.")
    return res


def extract_from_image(img: Image.Image, engine: str = "tesseract") -> OCRResult:
    arr = preprocess_image(img)
    try:
        text = _run_easyocr(arr) if engine == "easyocr" else _run_tesseract(arr)
    except Exception as exc:  # pragma: no cover - environment dependent
        r = OCRResult(engine=engine)
        r.warnings.append(f"OCR engine '{engine}' failed: {exc}")
        return r
    res = parse_sheet(text)
    res.engine = engine
    return res
