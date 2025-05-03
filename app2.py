from flask import Flask, render_template, request, redirect, url_for, jsonify
from pymongo import MongoClient
import threading
import time
import numpy as np

from t2 import generate_parameters, predict_risk

app = Flask(__name__)

# MongoDB setup
client = MongoClient('mongodb://localhost:27017')
db = client['RTPM']
patients_collection = db['patients']
initial_data_collection = db['initial_data']
nurses_collection = db['nurses']

# Global state
latest_data = {"params": {}, "risk": 0, "history": []}
current_patient = {}

# Background thread for generating parameters
def stream_data():
    while True:
        params = generate_parameters()
        risk = predict_risk(params)

        clean_params = {
            k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v)
            for k, v in params.items()
        }
        if isinstance(risk, (np.integer,)):
            risk = int(risk)

        latest_data['params'] = clean_params
        latest_data['risk'] = risk
        latest_data['history'].append(risk)
        if len(latest_data['history']) > 50:
            latest_data['history'].pop(0)
        time.sleep(2)

threading.Thread(target=stream_data, daemon=True).start()

# Generate sequential patient ID
def generate_patient_id():
    last_patient = patients_collection.find_one(sort=[("_id", -1)])
    if last_patient and last_patient.get("_id", "").isdigit():
        return str(int(last_patient['_id']) + 1).zfill(6)
    return "000001"

# Routes
@app.route('/')
def home():
    return render_template('home2.html')

@app.route('/register_step1', methods=['GET', 'POST'])
def register_step1():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        
        # Check if patient already exists
        existing_patient = patients_collection.find_one({
            'name': form_data.get('name'),
            'contact': form_data.get('contact')
        })
        
        if existing_patient:
            return redirect(url_for('register_step1', error='duplicate'))
        
        # Verify nurse credentials
        nurse_check = nurses_collection.find_one({
            'nurse_name': form_data.get('nurse_name'),
            'nurse_id': form_data.get('nurse_id')
        })

        if not nurse_check:
            return redirect(url_for('register_step1', error='nurse'))

        patient_id = generate_patient_id()
        form_data['_id'] = patient_id
        global current_patient
        current_patient = form_data
        return render_template('registerr22.html', request=request)

    # GET request handling
    error = request.args.get('error')
    return render_template('registration21.html', error=error)


@app.route('/save_patient', methods=['POST'])
def save_patient():
    global current_patient
    form_data = request.form.to_dict()

    patient_id = current_patient['_id']
    disease = current_patient['disease']

    # Store basic patient data
    if not patients_collection.find_one({'_id': patient_id}):
        patients_collection.insert_one(current_patient)

    # Store medical data from step 2
    medical_data = {'patient_id': patient_id, 'disease': disease}
    for key, value in form_data.items():
        if key not in current_patient:
            medical_data[key] = value

    initial_data_collection.insert_one(medical_data)

    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    global current_patient
    if request.method == 'POST':
        patient_id = request.form.get('_id')
        patient_name = request.form.get('name')
        patient = patients_collection.find_one({"_id": patient_id, "name": patient_name})
        if patient:
            current_patient = patient
            return redirect(url_for('dashboard'))
        else:
            return redirect(url_for('login', error='invalid'))
    return render_template('login2.html')

@app.route('/dashboard')
def dashboard():
    if not current_patient:
        return redirect(url_for('home'))
    return render_template('dashboard2.html', patient=current_patient)

@app.route('/show_graph')
def show_graph():
    if not current_patient:
        return redirect(url_for('home'))
    return render_template('show_graph2.html', patient=current_patient)

@app.route('/api/live_data')
def live_data():
    return jsonify(latest_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
