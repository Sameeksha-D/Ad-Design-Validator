import os
import cv2
import numpy as np
import base64
from flask import Flask, request, render_template

app = Flask(__name__)

def analyze_ad(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = gray.shape

    std_s = np.std(hsv[:,:,1])
    std_v = np.std(hsv[:,:,2])
    mean_v = np.mean(hsv[:,:,2])
    contrast = np.std(gray)
    
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / float(h * w)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_regions = 0
    text_area = 0
    cta_score = 0
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch == 0: continue
        area = cw * ch
        if area > (h * w) * 0.005: 
            text_regions += 1
            text_area += area
            if y > (2 * h / 3) and 1.5 < (cw / float(ch)) < 8:
                cta_score = 2
    
    text_density = text_area / float(h * w)
    
    # Generalized Progressive Penalty Model
    total_score = 10.0
    pros = []
    cons = []
    
    # 1. Exposure & Brightness (Mean V)
    if mean_v > 220:
        penalty = (mean_v - 220) * 0.1
        total_score -= penalty
        cons.append(f"Image is overexposed / washed out (Brightness: {mean_v:.1f}).")
    elif mean_v < 60:
        total_score -= 1.5
        cons.append(f"Image is too dark to read easily (Brightness: {mean_v:.1f}).")
    else:
        pros.append(f"Excellent cinematic exposure mapping (Brightness: {mean_v:.1f}).")
        
    # 2. Contrast Depth
    if contrast > 80:
        penalty = (contrast - 80) * 0.15
        total_score -= penalty
        cons.append(f"Contrast is extremely harsh and noisy (Contrast: {contrast:.1f}).")
    elif contrast < 35:
        total_score -= 1.5
        cons.append(f"Image is washed out and lacks depth (Contrast: {contrast:.1f}).")
    else:
        pros.append(f"Smooth, balanced contrast depth (Contrast: {contrast:.1f}).")
        
    # 3. Overall Color Vibrancy
    if std_s < 40:
        total_score -= 1.0
        cons.append(f"Colors lack punch and are generally dull (Saturation Variance: {std_s:.1f}).")
    elif std_s > 140:
        total_score -= 1.0
        cons.append(f"Colors bleed intensely and are over-saturated (Variance: {std_s:.1f}).")
    else:
        pros.append(f"Optimal vibrant branding colors detected (Variance: {std_s:.1f}).")
        
    # 4. Neural Clutter (Edge Density)
    if edge_density > 0.07:
        total_score -= 3.0
        cons.append(f"Extremely chaotic visual UI clutter (Density: {edge_density:.3f}).")
    elif edge_density > 0.05:
        total_score -= 1.5
        cons.append(f"Borderline cluttered background vectors (Density: {edge_density:.3f}).")
    else:
        pros.append(f"Clean design utilizing proper negative space (Density: {edge_density:.3f}).")
        
    # 5. Structure Footprint
    if text_density > 0.85:
        total_score -= 3.0
        cons.append(f"Shapes and text completely crowd the visual frame ({text_density*100:.1f}% footprint).")
    elif text_density > 0.65:
        total_score -= 1.0
        cons.append(f"Content structure is fairly heavy, consider padding ({text_density*100:.1f}% footprint).")
    else:
        pros.append(f"Professional text-to-space layout grouping ({text_density*100:.1f}% footprint).")

    # Hard scale and round
    total_score = round(max(1.0, min(10.0, total_score)), 1)
    
    suggestions = []
    if mean_v > 220: suggestions.append("Your exposure is blown out. Lower the brightness by 10-15% to regain depth.")
    if contrast > 80: suggestions.append("The raw contrast is too harsh between foreground objects. Soften lighting.")
    if edge_density > 0.05: suggestions.append("Clean up background noise/patterns. Embrace empty visual space.")
    if text_density > 0.65: suggestions.append("Consolidate your layout to Rule of Thirds grids. Leave negative padding.")
    
    return {
        "score": total_score,
        "pros": pros,
        "cons": cons,
        "suggestions": suggestions,
        "stats": {
            "v_mean": round(mean_v, 1),
            "contours": text_regions
        }
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            img_bytes = file.read()
            result = analyze_ad(img_bytes)
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            mime = file.mimetype if file.mimetype else 'image/jpeg'
            return render_template('index.html', result=result, image_data=f"data:{mime};base64,{img_b64}")
    return render_template('index.html', result=None)

if __name__ == '__main__':
    print("Starting purely offline CV Ad Validator on Port 5000...")
    app.run(debug=True, port=5000)
