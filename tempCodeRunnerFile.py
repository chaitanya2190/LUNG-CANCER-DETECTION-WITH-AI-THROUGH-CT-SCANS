# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image, ImageDraw

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
model = YOLO('best.pt')
 # Replace with your lung CT model if available

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'ct-scan' not in request.files:
        return redirect(url_for('index'))

    file = request.files['ct-scan']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        results = model(filepath)[0]

        image = Image.open(filepath).convert("RGB")
        draw = ImageDraw.Draw(image)

        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        image.save(result_path)

        return render_template('result.html', result_img=result_path)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)