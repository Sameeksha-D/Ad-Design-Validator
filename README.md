# Ad Design Validator (Pure OpenCV Engine)

A 100% offline, local Computer Vision application that mathematically evaluates marketing creative designs without relying on fragile OCR binaries or cloud APIs.

## Architecture & Visual Heuristics

This application processes image tensors using OpenCV and Flask to evaluate design rules:
1. **Color Quality:** Evaluates standard deviation on HSV channels to distinguish vibrant colors from muddy noise.
2. **Brightness & Contrast:** Evaluates grayscale variance and exposure mapping.
3. **Clutter Level:** Calculates raw pixel density using the Canny Edge Detector natively overriding human subjectivity.
4. **Layout Estimation:** Uses morphological contouring to group visual blocks without reading them linearly, calculating perfect negative space ratios.
5. **CTA Detection:** Geometrically isolates button-shaped standalone objects within specifically targeted lower-third bounding frames.

## Setup Instructions

1. Clone this repository to your local Windows workspace.
2. Ensure you have Python exactly 3.10+ installed.
3. Set up the virtual environment:
```bash
python -m venv venv
venv\Scripts\activate   # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```
4. Run the application standardly:
```bash
python app.py
```
5. Open your local browser precisely to `http://127.0.0.1:5000`

> Note: Completely free of Tesseract OCR / API locks. Runs infinitely scaling natively on-premise without overhead!
