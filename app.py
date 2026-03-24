import os
import cv2
import numpy as np
import time
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def analyze_ad(filepath):
    img = cv2.imread(filepath)
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
    
    # Continuous Fractional Scoring & Tailored Boundaries
    total_score = 0.0
    pros = []
    cons = []
    
    # 1. Color (Vibrancy vs Over-saturation)
    color_pts = max(0, 2.0 - abs(75.0 - std_s)*0.05)
    total_score += color_pts
    if 65 < std_s < 85: pros.append(f"Optimal color saturation variance ({std_s:.1f}).")
    else: cons.append(f"Colors are either dull or aggressively over-saturated ({std_s:.1f}).")
        
    # 2. Brightness & Contrast (Balanced vs Harsh)
    bright_pts = max(0, 2.0 - abs(130.0 - mean_v)*0.03)
    contrast_pts = max(0, 2.0 - abs(72.0 - contrast)*0.05)
    total_score += (bright_pts + contrast_pts) / 2.0
    
    if 115 < mean_v < 145 and contrast < 78:
        pros.append(f"Excellent cinematic exposure and smooth contrast (Brightness: {mean_v:.1f}).")
    else:
        cons.append(f"Lighting is either glaringly overexposed or flat (Brightness: {mean_v:.1f}, Contrast: {contrast:.1f}).")
        
    # 3. Clutter
    clutter_pts = max(0, 2.0 - (edge_density * 40.0))
    total_score += clutter_pts
    if edge_density < 0.0325:
        pros.append(f"Clean design utilizing proper negative space (Edge density: {edge_density:.3f}).")
    else:
        cons.append(f"Design is excessively cluttered with chaotic line formations (Edge density: {edge_density:.3f}).")
        
    # 4. Layout & Contours
    layout_pts = max(0, 2.0 - abs(0.15 - text_density)*10.0)
    total_score += layout_pts
    if 0.05 < text_density < 0.25:
        pros.append(f"Professional text-to-space ratio ({text_density*100:.1f}% physical footprint).")
    else:
        cons.append(f"Content distribution completely crowds the visual hierarchy ({text_density*100:.1f}% footprint).")
        
    # 5. CTA Extraction 
    if cta_score == 2:
        total_score += 2.0
        pros.append("Distinct geometric Call-to-Action (CTA) target isolated in lower third.")
    else:
        cons.append("Failed to locate a primary geometric Call-to-Action button.")
        
    total_score = round(max(1.0, min(10.0, total_score + 1.0)), 1) # Scale floor buffer
    
    suggestions = []
    if std_s > 80: suggestions.append("Lower your core saturation; the colors are bleeding and hurting the eyes.")
    if mean_v > 140: suggestions.append("Your exposure is blown out. Lower the brightness by 15-20% to regain depth.")
    if contrast > 80: suggestions.append("The raw contrast is too harsh between objects. Soften shadows and typography.")
    if edge_density >= 0.0325: suggestions.append("Erase unnecessary background textures to clear up overwhelming noise.")
    if cta_score == 0: suggestions.append("Draft a highly visible rectangular 'Shop Now' block anchored at the bottom.")

    
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
            # Append timestamp to completely bypass browser caching bugs for identical filenames
            unique_filename = str(int(time.time())) + "_" + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            result = analyze_ad(filepath)
            return render_template('index.html', result=result, filename=unique_filename)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    print("Starting purely offline CV Ad Validator on Port 5000...")
    app.run(debug=True, port=5000)
