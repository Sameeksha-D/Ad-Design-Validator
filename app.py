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
    
    # Granular Scoring
    total_score = 0
    pros = []
    cons = []
    
    # 1. Color
    if std_s > 40 and std_v > 40: 
        total_score += 2
        pros.append(f"Vibrant color palette detected (Saturation variance: {std_s:.1f}).")
    else: 
        cons.append(f"Colors lack punch. (Low saturation variance: {std_s:.1f}).")
        
    # 2. Brightness
    if 50 < mean_v < 200 and contrast > 40:
        total_score += 2
        pros.append(f"Excellent exposure and contrast (Contrast ratio: {contrast:.1f}).")
    else:
        cons.append(f"Poor exposure / washed out (Contrast ratio: {contrast:.1f}).")
        
    # 3. Clutter
    if edge_density < 0.05:
        total_score += 2
        pros.append(f"Clean design, low visual noise (Edge density: {edge_density:.3f}).")
    elif edge_density < 0.12:
        total_score += 1
        pros.append(f"Moderate visual noise (Edge density: {edge_density:.3f}).")
    else:
        cons.append(f"Highly cluttered design! (Edge density: {edge_density:.3f}).")
        
    # 4. Layout
    if 0.05 < text_density < 0.30:
        total_score += 2
        pros.append(f"Perfect content-to-space ratio ({text_density*100:.1f}% active area).")
    elif text_density <= 0.05:
        cons.append(f"Too empty! Barely any content detected ({text_density*100:.1f}%).")
    else:
        cons.append(f"Content dominates the frame, needs negative space ({text_density*100:.1f}%).")
        
    # 5. CTA
    if cta_score == 2:
        total_score += 2
        pros.append("Distinct Call-to-Action (CTA) button structurally isolated in lower third.")
    else:
        cons.append("No distinct geometric Call-to-Action (CTA) button detected.")
        
    suggestions = []
    if std_s <= 40: suggestions.append("Boost saturation of primary elements to grab attention.")
    if contrast <= 40: suggestions.append("Increase contrast between background and foreground.")
    if edge_density >= 0.12: suggestions.append("Remove unnecessary background patterns. Embrace negative space.")
    if text_density >= 0.30: suggestions.append("Group your content blocks. Too much space is covered by text/shapes.")
    if cta_score == 0: suggestions.append("Add a highly visible rectangular 'Shop Now' button to the bottom 30% of the image.")
    
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
