from flask import Flask, request, jsonify
import pandas as pd
from csv2firebase import upload_csv_to_firebase_and_store_url, process_device_data, get_phosphorus_color, get_Conductivity_color, get_nitrogen_color, get_moisture_color, get_ph_color, get_potassium_color
from firebase2firebase import process_device_data_f2f
import firebase_admin
from firebase_admin import credentials, db, storage, firestore
# Initialize Flask app
app = Flask(__name__)
# service_account = "crop2x.json"
# # Initialize Firebase Admin SDK
# cred = credentials.Certificate(service_account)
# firebase_admin.initialize_app(cred, {
#     'databaseURL': 'https://cropx-6f03a-default-rtdb.firebaseio.com',
#     'storageBucket': 'cropx-6f03a.appspot.com'
# })

# # Initialize Firestore
# firestore_client = firestore.client()
# def get_firestore_client():
#     return firestore_client

# Function to fetch data from CSV

def fetch_data_from_csv(csv_file):
    try:
        data = pd.read_csv(csv_file)
        return data
    except Exception as e:
        return str(e)

# Function to process the device data and create heatmap
def process_and_create_heatmap(csv_file_new):
    # Fetch data from the CSV file
    data = fetch_data_from_csv(csv_file_new)

    if isinstance(data, str):  # If an error message was returned from fetch_data_from_csv
        return {"error": data}

    # Define attributes and color functions
    attributes = ['phosphor', 'conductivity', 'nitrogen', 'moisture', 'pH', 'potassium']
    color_functions = {
        'phosphor': get_phosphorus_color,
        'conductivity': get_Conductivity_color,
        'nitrogen': get_nitrogen_color,
        'moisture': get_moisture_color,
        'pH': get_ph_color,
        'potassium': get_potassium_color
    }

    # Process the device data and create heatmaps
    result = process_device_data(data, attributes, color_functions)

    return {"message": result}

@app.route('/process_csv', methods=['POST'])
def process_csv():
    # Get the file from the request
    csv_file = None
    for file_key in request.files:
        csv_file = request.files[file_key]
        break  # Only pick the first file uploaded

    # Check if a file is uploaded
    if not csv_file:
        return jsonify({"error": "CSV file is required."}), 400

    # Check if the file has a valid name
    if csv_file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    # Get the custom file name from the form (optional)
    custom_filename = request.form.get('custom_filename', csv_file.filename)

    # Save the file locally with the custom filename
    csv_file_path = f"{custom_filename}"
    csv_file.save(csv_file_path)

    # Upload the file to Firebase
    upload_csv_to_firebase_and_store_url(csv_file_path, folder_name="csv_files")

    # Process the file and create heatmap
    result = process_and_create_heatmap(csv_file_path)

    if "error" in result:
        return jsonify(result), 500

    return jsonify(result)

@app.route('/process_data', methods=['POST'])
def process_data():
    try:
        # Extract device ID from the request payload
        payload = request.json
        device_id = payload.get('device_id')
        devices=[device_id]
        print(devices)
        if not device_id:
            return jsonify({'error': 'Device ID is required'}), 400
        attributes = ['phosphor', 'conductivity', 'nitrogen', 'moisture', 'pH','potassium']
        color_functions = {
        'phosphor': get_phosphorus_color,
        'conductivity': get_Conductivity_color,
        'nitrogen': get_nitrogen_color,
        'moisture' : get_moisture_color,
        'pH' : get_ph_color,
        'potassium' : get_potassium_color
        }
        # process_device_data_f2f(devices, attributes, color_functions)   

        # return jsonify({'message': 'Image processed and uploaded successfully'}), 200

        try:
            process_device_data_f2f(devices, attributes, color_functions)
        except Exception as inner_error:
            print(f"Error during process_device_data: {inner_error}")  # Log the error
            return jsonify({'error': f'Failed to process device data: {str(inner_error)}'}), 500

        return jsonify({'message': 'Image processed and uploaded successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)