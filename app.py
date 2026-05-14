from flask import Flask, render_template, request, send_file
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from reportlab.pdfgen import canvas
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load AI models
brain_model = tf.keras.models.load_model('brain_tumor_model.h5')
chest_model = tf.keras.models.load_model('chest_xray_model.h5')

# =========================
# HOME PAGE
# =========================
@app.route('/')
def home():

    return render_template('index.html')

# =========================
# PDF REPORT
# =========================
@app.route('/download_report')
def download_report():

    pdf_path = "report.pdf"

    c = canvas.Canvas(pdf_path)

    c.setFont("Helvetica-Bold", 22)
    c.drawString(170, 800, "MediScan AI Report")

    # Patient Info
    c.setFont("Helvetica-Bold", 15)
    c.drawString(60, 740, "Patient Information")

    c.setFont("Helvetica", 12)

    c.drawString(60, 710, f"Name: {app.patient_name}")
    c.drawString(60, 690, f"Age: {app.patient_age}")
    c.drawString(60, 670, f"Gender: {app.patient_gender}")
    c.drawString(60, 650, f"Medical History: {app.patient_history}")

    # AI Result
    c.setFont("Helvetica-Bold", 15)
    c.drawString(60, 600, "AI Diagnostic Result")

    c.setFont("Helvetica", 12)

    c.drawString(60, 570, f"Prediction: {app.prediction}")
    c.drawString(60, 550, f"Risk Level: {app.risk}")
    c.drawString(60, 530, f"Confidence: {app.confidence}%")

    # Explanation
    c.setFont("Helvetica-Bold", 14)
    c.drawString(60, 480, "Explanation")

    c.setFont("Helvetica", 12)
    c.drawString(60, 455, app.explanation)

    # Symptoms
    c.setFont("Helvetica-Bold", 14)
    c.drawString(60, 410, "Possible Symptoms")

    c.setFont("Helvetica", 12)
    c.drawString(60, 385, app.symptoms)

    # Advice
    c.setFont("Helvetica-Bold", 14)
    c.drawString(60, 340, "Advice")

    c.setFont("Helvetica", 12)
    c.drawString(60, 315, app.advice)

    c.save()

    return send_file(pdf_path, as_attachment=True)

# =========================
# BRAIN MRI ROUTE
# =========================
@app.route('/predict_brain', methods=['POST'])
def predict_brain():

    return process_prediction('brain')

# =========================
# CHEST X-RAY ROUTE
# =========================
@app.route('/predict_chest', methods=['POST'])
def predict_chest():

    return process_prediction('chest')

# =========================
# SHARED FUNCTION
# =========================
def process_prediction(scan_type):

    if 'file' not in request.files:
        return 'No file uploaded'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    # Save patient info
    patient_name = request.form.get('patient_name')
    patient_age = request.form.get('patient_age')
    patient_gender = request.form.get('patient_gender')
    patient_history = request.form.get('patient_history')

    # Save image
    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    file.save(filepath)

    # Open image
    img = Image.open(filepath)

    img_check = np.array(img)

    # Reject blank image
    mean_pixel = img_check.mean()

    if mean_pixel > 240 or mean_pixel < 15:

        return render_template(
            'index.html',
            error='Invalid image. Please upload proper medical scans.'
        )

    # Reject low detail image
    std_pixel = img_check.std()

    if std_pixel < 10:

        return render_template(
            'index.html',
            error='Invalid image. Please upload proper medical scans.'
        )

    # Process image
    img_processed = image.load_img(
        filepath,
        target_size=(128, 128),
        color_mode='grayscale'
    )

    img_array = image.img_to_array(img_processed)

    img_array = np.repeat(img_array, 3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    # =========================
    # BRAIN MRI
    # =========================
    if scan_type == 'brain':

        prediction = brain_model.predict(img_array)[0][0]

        if prediction > 0.3:

            result = 'Brain Tumor Detected'
            risk = 'HIGH RISK'
            confidence = round(prediction * 100, 2)

            explanation = (
                'The AI detected unusual patterns in the brain scan.'
            )

            symptoms = (
                'Headache, dizziness, blurred vision.'
            )

            advice = (
                'Please consult a neurologist immediately.'
            )

            risk_color = 'red'

        else:

            result = 'Normal Brain MRI'
            risk = 'LOW RISK'
            confidence = round((1 - prediction) * 100, 2)

            explanation = (
                'The brain scan appears mostly normal.'
            )

            symptoms = (
                'No major symptoms identified.'
            )

            advice = (
                'Continue regular health checkups.'
            )

            risk_color = 'green'

    # =========================
    # CHEST X-RAY
    # =========================
    else:

        prediction = chest_model.predict(img_array)[0][0]

        if prediction > 0.5:

            result = 'Pneumonia Detected'
            risk = 'MEDIUM / HIGH RISK'
            confidence = round(prediction * 100, 2)

            explanation = (
                'Possible lung infection patterns detected.'
            )

            symptoms = (
                'Cough, fever, chest pain.'
            )

            advice = (
                'Please consult a chest specialist.'
            )

            risk_color = 'orange'

        else:

            result = 'Normal Chest X-ray'
            risk = 'LOW RISK'
            confidence = round((1 - prediction) * 100, 2)

            explanation = (
                'The chest scan appears mostly normal.'
            )

            symptoms = (
                'No major symptoms identified.'
            )

            advice = (
                'Maintain healthy lifestyle.'
            )

            risk_color = 'green'

    risk_width = confidence

    # Store report data
    app.patient_name = patient_name
    app.patient_age = patient_age
    app.patient_gender = patient_gender
    app.patient_history = patient_history

    app.prediction = result
    app.risk = risk
    app.confidence = confidence
    app.explanation = explanation
    app.symptoms = symptoms
    app.advice = advice

    return render_template(
        'index.html',
        prediction=result,
        risk=risk,
        confidence=confidence,
        explanation=explanation,
        symptoms=symptoms,
        advice=advice,
        image_file=filepath,
        risk_width=risk_width,
        risk_color=risk_color
    )

# =========================
# RUN APP
# =========================
if __name__ == '__main__':

    app.run(debug=True)