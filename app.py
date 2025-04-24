# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
import sqlite3
from datetime import datetime
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
model = YOLO('best.pt')  # Replace with your lung CT model if available

# Initialize database
def init_db():
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS patient (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            sex TEXT NOT NULL,
            date_of_entry TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

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

        return render_template('result.html', filename=filename, detected=True, date_today=datetime.now().strftime("%Y-%m-%d"))

    return redirect(url_for('index'))

@app.route('/submit_patient_details', methods=['POST'])
def submit_patient_details():
    name = request.form['name']
    age = request.form['age']
    sex = request.form['sex']
    date_of_entry = request.form.get('date_of_entry', datetime.now().strftime("%Y-%m-%d"))

    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    c.execute('INSERT INTO patient (name, age, sex, date_of_entry) VALUES (?, ?, ?, ?)',
              (name, age, sex, date_of_entry))
    conn.commit()
    conn.close()

    return render_template('thank_you.html')

if __name__ == '__main__':
    app.run(debug=True)
