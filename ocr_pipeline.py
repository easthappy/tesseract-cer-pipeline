#!/usr/bin/env python3
"""
ocr_pipeline.py
===============
Six-stage OCR preprocessing pipeline demonstrating how each step
affects Tesseract's Character Error Rate (CER) on a degraded image.

Stages run in order:
  0 - Raw Tesseract (no preprocessing)
  1 - Grayscale only
  2 - Grayscale + deskew
  3 - Grayscale + deskew + denoise
  4 - Full pipeline: grayscale + deskew + denoise + adaptive threshold
  5 - Wrong order (threshold before deskew) - CER regression demo

Each stage prints its CER so the improvement - and the regression - are
visible as numbers, not claims.

The bundled fixture (samples/degraded_invoice.png) is a small, low-DPI,
slightly skewed invoice image - a realistic case where Tesseract returns
high-confidence text that is nonetheless wrong. Every reader running
--demo gets the same image and the same numbers.

Requirements:
    pip install pytesseract opencv-python Pillow numpy
    Tesseract binary: https://github.com/tesseract-ocr/tesseract

    Windows only - after installing Tesseract, uncomment and set
    the tesseract_cmd line below.

Usage:
    python ocr_pipeline.py --demo
    python ocr_pipeline.py --image path/to/image.png --reference "known text"
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pytesseract
from PIL import Image

# Windows: Tesseract path (uncomment + edit).
# Find yours with: where /r c: tesseract.exe
# Common paths:
#   C:\\Program Files\\Tesseract-OCR\\tesseract.exe
#   C:\\Users\\<you>\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe
#   pytesseract.pytesseract.tesseract_cmd = (
#                           r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
#                                           )


# ─────────────────────────────────────────────
# Fixture - the demo image and its known reference text
# ─────────────────────────────────────────────

FIXTURE_REFERENCE = (
    "Invoice INV-2026-0417\n"
    "Date 2026-04-17\n"
    "Bill to ACME Corp\n"
    "\n"
    "Server license x5 $995.00\n"
    "Support annual $120.00\n"
    "Setup fee $75.00\n"
    "\n"
    "Subtotal $1190.00\n"
    "Tax 8.5% $101.15\n"
    "Total $1291.15"
)

_FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "samples", "degraded_invoice.png",
)


# ─────────────────────────────────────────────
# CER - Character Error Rate
# ─────────────────────────────────────────────

def _substitution_cost(a: str, b: str) -> int:
    """Return 0 if characters match, 1 if they differ."""
    return 0 if a == b else 1


def _dp_row(prev: list[int], s_char: str, target: str) -> list[int]:
    """Compute one row of the Levenshtein DP matrix."""
    row = [prev[0] + 1] + [0] * len(target)
    for c, t_char in enumerate(target, start=1):
        row[c] = min(
            prev[c] + 1,
            row[c - 1] + 1,
            prev[c - 1] + _substitution_cost(s_char, t_char),
        )
    return row


def edit_distance(source: str, target: str) -> int:
    """Levenshtein distance between source and target strings."""
    row = list(range(len(target) + 1))
    for s_char in source:
        row = _dp_row(row, s_char, target)
    return row[-1]


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate: edit distance divided by reference length.
    Comparison is case-insensitive - for invoice/receipt OCR, character
    case carries no business meaning and case noise would dominate the
    error signal on near-correct output.
    """
    ref = reference.strip().lower()
    hyp = hypothesis.strip().lower()
    if not ref:
        return 0.0
    return round(edit_distance(ref, hyp) / len(ref), 4)


# ─────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR NumPy array."""
    return cv2.imread(path)


# ─────────────────────────────────────────────
# Preprocessing steps - each does one thing
# ─────────────────────────────────────────────

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to single-channel grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _detect_skew_angle(image: np.ndarray) -> float:
    """
    Detect skew angle via minAreaRect on Otsu-thresholded text pixels.
    Otsu separates text from background automatically; inverting makes
    text the foreground for the rect fit, which is far more robust than
    thresholding raw pixel values (noise contaminates the dark-pixel
    cloud and skews the detected angle). Validation gate: angles beyond
    +/- 15 degrees indicate noise interference rather than real document
    skew - return 0.0 in that case.
    """
    _, bw = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if coords.size == 0:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    angle = angle + 90 if angle < -45 else angle
    if abs(angle) > 15.0:
        return 0.0
    return angle


def deskew(
    image: np.ndarray, interpolation: int = cv2.INTER_CUBIC
) -> np.ndarray:
    """
    Detect skew angle and rotate to correct it.
    Skips rotation when angle is within +/- 0.5 degrees of horizontal
    to avoid over-correcting near-straight images. The interpolation flag
    controls how rotated pixels are resampled: INTER_CUBIC (default)
    produces smooth edges on grayscale but gray halos on binary images;
    INTER_NEAREST picks the closest pixel and cannot introduce halos.
    """
    angle = _detect_skew_angle(image)
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h),
        flags=interpolation,
        borderMode=cv2.BORDER_REPLICATE,
    )


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Binarize using adaptive Gaussian thresholding."""
    return cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10,
    )


def denoise(image: np.ndarray) -> np.ndarray:
    """Remove noise with non-local means denoising."""
    return cv2.fastNlMeansDenoising(image, h=10)


# ─────────────────────────────────────────────
# Tesseract runner
# ─────────────────────────────────────────────

def _mean_confidence(data: dict) -> float:
    """Extract mean confidence score from pytesseract data dict."""
    confidences = [c for c in data["conf"] if c != -1]
    if not confidences:
        return 0.0
    return round(sum(confidences) / len(confidences), 1)


def run_tesseract(image: np.ndarray) -> tuple[str, float]:
    """
    Run Tesseract on a NumPy array.
    Returns (extracted_text, mean_confidence_score).
    PSM 6: assume a uniform block of text.
    OEM 1: LSTM engine only.
    """
    pil_image = Image.fromarray(image)
    config = "--psm 6 --oem 1"
    data = pytesseract.image_to_data(
        pil_image, config=config,
        output_type=pytesseract.Output.DICT,
    )
    text = pytesseract.image_to_string(pil_image, config=config)
    return text, _mean_confidence(data)


# ─────────────────────────────────────────────
# Stage runner and printer
# ─────────────────────────────────────────────

SEPARATOR = "-" * 56


def print_stage(label: str, cer: float, conf: float, text: str) -> None:
    """Print one stage result: label, CER, confidence, extracted text."""
    print(f"\n{SEPARATOR}")
    print(f"  {label}")
    print(SEPARATOR)
    print(f"  CER        : {cer:.2%}")
    print(f"  Confidence : {conf:.1f}%")
    print(f"  Output     : {text.strip()[:120]!r}")


def run_stage(label: str, image: np.ndarray, reference: str) -> float:
    """Run Tesseract on image, compute CER, print result. Returns CER."""
    text, conf = run_tesseract(image)
    cer = compute_cer(reference, text)
    print_stage(label, cer, conf, text)
    return cer


# ─────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────

def build_stages(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Build all seven pipeline variants from the source image.

    Stage 5 moves adaptive threshold from last (Stage 4) to first while
    keeping all four steps, so the only variable vs Stage 4 is the
    threshold/deskew order. Stage 6 repeats Stage 5 with nearest-neighbor
    deskew interpolation instead of cubic -- if the gray-halo hypothesis
    for Stage 5's regression is correct, Stage 6 should recover.
    """
    gray = to_grayscale(image)
    deskewed = deskew(gray)
    denoised = denoise(deskewed)
    thresholded = adaptive_threshold(denoised)
    # Stage 5: threshold first, then deskew, then denoise -- same four
    # steps as Stage 4, only the threshold position changes.
    wrong_order = denoise(deskew(adaptive_threshold(gray)))
    # Stage 6: identical to Stage 5 but deskew uses nearest-neighbor
    # sampling, which does not interpolate and so cannot produce gray
    # halos at binary edges.
    wrong_order_nearest = denoise(
        deskew(adaptive_threshold(gray), interpolation=cv2.INTER_NEAREST)
    )

    return [
        ("Stage 0 - Raw Tesseract (no preprocessing)", image),
        ("Stage 1 - Grayscale only", gray),
        ("Stage 2 - Grayscale + deskew", deskewed),
        ("Stage 3 - Grayscale + deskew + denoise", denoised),
        ("Stage 4 - Full pipeline (correct order)", thresholded),
        ("Stage 5 - Wrong order: threshold before deskew", wrong_order),
        ("Stage 6 - Wrong order, nearest-neighbor deskew",
         wrong_order_nearest),
    ]


# ─────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────

def print_summary(results: list[tuple[str, float]]) -> None:
    """Print a visual CER summary bar chart for all stages."""
    print(f"\n{SEPARATOR}")
    print("  Summary - CER by stage (lower is better)")
    print(SEPARATOR)
    for label, cer in results:
        bar = "#" * max(0, int((1 - cer) * 20))
        short = label.split("-")[1].strip() if "-" in label else label
        print(f"  {cer:.2%}  {bar:<20}  {short}")
    print(f"{SEPARATOR}\n")


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse --image / --reference / --demo arguments."""
    parser = argparse.ArgumentParser(
        description="Seven-stage OCR preprocessing pipeline with CER measurement"
    )
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument(
        "--reference", type=str, help="Known correct text for CER"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run the pipeline on the bundled degraded invoice fixture",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def resolve_inputs(args: argparse.Namespace) -> tuple[str, str]:
    """Return (image_path, reference_text) from parsed args or demo mode."""
    if args.demo:
        return _FIXTURE_PATH, FIXTURE_REFERENCE
    return args.image, args.reference


def _abort(message: str) -> None:
    """Print an error message and exit."""
    print(f"\n  Error: {message}", file=sys.stderr)
    sys.exit(1)


def _validate_args(args: argparse.Namespace) -> None:
    """Exit with a clear message if required arguments are missing."""
    if not args.demo and not args.image:
        _abort("Provide --demo or --image <path> --reference <text>")
    if args.image and not args.reference:
        _abort("--reference is required when using --image")


def _validate_image(image: np.ndarray, path: str) -> None:
    """Exit with a clear message if the image failed to load."""
    if image is None:
        _abort(f"Could not load image: {path}")


def _print_header(image_path: str, reference: str) -> None:
    """Print the pipeline header with image path and reference text."""
    print("\n  OCR Preprocessing Pipeline - seven stages, one image")
    print(f"  Image     : {image_path}")
    suffix = "..." if len(reference) > 60 else ""
    print(f"  Reference : {reference[:60]!r}{suffix}")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    image_path, reference = resolve_inputs(args)
    _print_header(image_path, reference)
    image = load_image(image_path)
    _validate_image(image, image_path)
    stages = build_stages(image)
    results = [
        (label, run_stage(label, processed, reference))
        for label, processed in stages
    ]
    print_summary(results)


if __name__ == "__main__":
    main()
