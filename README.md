# Tesseract CER Preprocessing Pipeline

A seven-stage OCR preprocessing benchmark measuring Character Error Rate (CER) on a degraded 72-DPI invoice fixture using Tesseract 5.4.0 (LSTM / OEM 1, PSM 6) and OpenCV.

## Benchmark Results (`--demo`)

Measured on a 560×300 px 72-DPI invoice (173 characters across 11 lines, 1.8° skew, mild Gaussian blur, and fixed-seed sensor noise):

| Stage | Pipeline | CER (Linux / OpenCV 4.13.0) | CER (Windows / OpenCV 5.0.0) |
| :--- | :--- | :---: | :---: |
| **0** | Raw Tesseract, no preprocessing | 1.73% | 1.73% |
| **1** | Grayscale only | 2.31% | 2.31% |
| **2** | Grayscale + deskew | 2.31% | 2.31% |
| **3** | Grayscale + deskew + denoise | 2.89% | 3.47% |
| **4** | Full pipeline (`grayscale -> deskew -> denoise -> threshold`) | 5.20% | 6.36% |
| **5** | Wrong order: threshold before deskew (`INTER_CUBIC`) | 19.08% | 19.65% |
| **6** | Wrong order: nearest-neighbor deskew (`INTER_NEAREST`) | 19.08% | 19.65% |

## Key Takeaways
- **Modern Tesseract handles mild degradation out of the box:** Raw Stage 0 outperforms every preprocessed variant on recoverable 72-DPI input.
- **Step order is non-commutative:** Applying adaptive thresholding before denoising locks sensor noise into irreversible 1-bit pixels, causing a ~3.7× CER regression.
- **Measure CER, not confidence:** Confidence scores reflect language-model plausibility rather than character-level accuracy.

## Quickstart

### 1. Install Prerequisites
Install the [Tesseract OCR engine](https://github.com) (`v5.4+`), then install the Python dependencies:

```bash
pip install pytesseract opencv-python Pillow numpy
```

*(Windows users: uncomment and verify the `pytesseract.pytesseract.tesseract_cmd` path at the top of `ocr_pipeline.py` if Tesseract is not in your system PATH.)*

### 2. Run the Reproducible Demo
```bash
python ocr_pipeline.py --demo
```

### 3. Run Against Your Own Image
```bash
python ocr_pipeline.py --image path/to/your_image.png --reference "expected ground truth text"
```

## Repository Structure
- `ocr_pipeline.py` — Full 7-stage preprocessing pipeline, skew detector, CER evaluator, and CLI runner (23 functions, Radon Grade A, Ruff clean).
- `samples/degraded_invoice.png` — Bundled 72-DPI test fixture used by `--demo`.
