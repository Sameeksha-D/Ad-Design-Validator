import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def analyze_ad(filepath):
    img = cv2.imread(filepath)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = gray.shape

    # 1. COLOR ANALYSIS (2 pts)
    # Vibrant colors = high saturation. Cohesive palette = well-distributed hue.
    std_s = np.std(hsv[:,:,1])
    std_v = np.std(hsv[:,:,2])
    color_score = 0
    if std_s > 40 and std_v > 40: 
        color_score = 2
    elif std_s > 20 and std_v > 20: 
        color_score = 1
    
    # 2. BRIGHTNESS & CONTRAST (2 pts)
    mean_v = np.mean(hsv[:,:,2])
    contrast = np.std(gray)
    brightness_score = 0
    if 50 < mean_v < 200 and contrast > 40: 
        brightness_score = 2
    elif 30 < mean_v < 220 and contrast > 20: 
        brightness_score = 1
    
    # 3. CLUTTER DETECTION (2 pts)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (h * w)
    clutter_score = 0
    if edge_density < 0.05: 
        clutter_score = 2
    elif edge_density < 0.12: 
        clutter_score = 1
    
    # 4 & 5 & 6. TEXT ESTIMATION, LAYOUT, AND CTA VIA CONTOURS (4 pts)
    # Use morphology to connect adjacent letters/shapes into solid blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_regions = 0
    text_area = 0
    cta_score = 0
    
    # Analyze blocks
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area > (h * w) * 0.005:  # Ignore tiny noise specks
            text_regions += 1
            text_area += area
            
            # CTA Estimation: Standalone rectangle in the bottom third
            # A CTA is usually wide enough but not a full banner (ratio 1.5 to 8)
            if y > (2 * h / 3) and 1.5 < (cw / float(ch)) < 8:
                cta_score = 2
    
    text_density = text_area / float(h * w)
    
    # Layout Score
    layout_score = 0
    if 0.05 < text_density < 0.30:  # Healthy text-to-space ratio
        layout_score = 2
    elif 0.01 < text_density < 0.40:
        layout_score = 1
        
    total_score = color_score + brightness_score + clutter_score + layout_score + cta_score
    
    # Generate Arrays
    pros = []
    cons = []
    
    if color_score == 2: pros.append("Vibrant and contrasting colors effectively direct attention.")
    else: cons.append("Colors are dull, muddy, or lack sufficient contrast.")
    
    if brightness_score == 2: pros.append("Excellent exposure and visual balance.")
    else: cons.append("Lighting is too dark, overexposed, or washed out flatly.")
    
    if clutter_score == 2: pros.append("Clean design with excellent use of negative space.")
    else: cons.append("Design relies on heavily cluttered and overwhelming visual noise.")
    
    if layout_score == 2: pros.append(f"Balanced content distribution (approx {round(text_density*100)}% active area).")
    else: cons.append("Primary visual elements are either completely missing or dominating the entire frame.")
    
    if cta_score == 2: pros.append("Strong Call-to-Action (CTA) format structurally detected in the lower section.")
    else: cons.append("No distinct isolated Call-to-Action (CTA) button detected.")
    
    suggestions = []
    if color_score < 2: suggestions.append("Increase saturation and use complementary colors for the core product.")
    if brightness_score < 2: suggestions.append("Adjust the image levels/curves to improve standard readability.")
    if clutter_score < 2: suggestions.append("Remove 30-40% of non-essential background patterns to let the product breathe.")
    if layout_score < 2: suggestions.append("Group content blocks into structured grids rather than scattering randomly.")
    if cta_score < 2: suggestions.append("Add a highly visible rectangular 'Shop Now' button explicitly to the bottom 30% of the image.")
    
    return {
        "score": total_score,
        "pros": pros,
        "cons": cons,
        "suggestions": suggestions,
        "text_density": round(text_density * 100, 1),
        "visual_metrics": {
            "contrast": round(contrast),
            "edge_density": round(edge_density, 3)
        }
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            result = analyze_ad(filepath)
            return render_template('index.html', result=result, filename=filename)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    print("Starting completely offline pure CV Ad Validator on Port 5000...")
    app.run(debug=True, port=5000)
