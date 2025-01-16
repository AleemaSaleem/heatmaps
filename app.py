from flask import Flask, request, jsonify
import os
import pandas as pd
from werkzeug.utils import secure_filename
from final_combine_attributes import fetch_timestamps, fetch_data, get_moisture_color, get_moisture_color, get_nitrogen_color,get_Conductivity_color ,create_heatmap, get_ph_color, get_potassium_color, get_phosphorus_color
# from csv2firebase import upload_csv_to_firebase_and_store_url , create_heatmap, get_phosphorus_color, get_nitrogen_color, get_Conductivity_color, get_ph_color, get_moisture_color, get_potassium_color # Assuming csv2firebase has a `process_csv` function.

app = Flask(__name__)

# Route to process device data
@app.route('/process_device', methods=['POST'])
def process_device():
    try:
        data = request.get_json()
        device_id = data.get('device_id',[])
        if not device_id:
            return jsonify({"error": "Device ID is required"}), 400
        
        timestamps = fetch_timestamps(device_id)
        if not timestamps:
            return jsonify({"message": "No timestamps found for this device"}), 404
        
        for timestamp in timestamps:
            device_data = fetch_data(device_id, timestamp)
            if device_data.empty:
                continue

            attributes = ['phosphor', 'conductivity', 'nitrogen', 'moisture', 'pH', 'potassium']
            color_functions = {
                'phosphor': get_phosphorus_color,
                'conductivity': get_Conductivity_color,
                'nitrogen': get_nitrogen_color,
                'moisture': get_moisture_color,
                'pH': get_ph_color,
                'potassium': get_potassium_color
            }
            devices = []
            create_heatmap(device_data, device_id, timestamp, attributes, color_functions)
            # process_device_data(devices, attributes, color_functions)

        return jsonify({"message": "Device processing completed successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
