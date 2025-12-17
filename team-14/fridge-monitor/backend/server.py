# flask imports
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

# database api functions
from api import (
    connect_mongo, 
    query, 
    add_entry, 
    get_drive_creds, 
    get_drive_file_url_and_set_permission, 
    upload_photo_to_drive,
    load_model,
    CNN,
    analyze_photo_pytorch
)

# load environment variables
load_dotenv()

URI = os.getenv('URI')
DB_NAME = os.getenv('DB_NAME')
COL_NAME = os.getenv('COL_NAME')
DRIVE_FOLDER_ID = os.getenv('DRIVE_FOLDER_ID')

# init clients
mongo_client = connect_mongo(URI)
drive_creds = get_drive_creds()

# init flask app
app = Flask(__name__)
CORS(app) 

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# load model
model = load_model()

@app.route('/api/scan', methods=['POST'])
def scan_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            image_bytes = file.read()
            food_name = analyze_photo_pytorch(model, image_bytes)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            safe_name = food_name.replace(' ', '_').lower()
            drive_filename = f"{safe_name}_{timestamp}.jpg"
            
            drive_file_data = upload_photo_to_drive(
                drive_creds, 
                image_bytes, 
                drive_filename, 
                DRIVE_FOLDER_ID
            )
            
            if not drive_file_data:
                return jsonify({"error": "Failed to upload image to Drive"}), 500
            
            file_id = drive_file_data.get('id')
            public_url = get_drive_file_url_and_set_permission(drive_creds, file_id)
            
            if not public_url:
                return jsonify({"error": "Failed to generate public link"}), 500

            date_placed = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            expiration_date = date_placed

            food_document = {
                "name": food_name,
                "image": {
                    "url": public_url,
                    "file_id": file_id
                },
                "date_placed": date_placed,
                "expiration_date": expiration_date,
                "status": "in_fridge"
            }
            
            add_entry(mongo_client, DB_NAME, COL_NAME, food_document)
            food_document['_id'] = str(food_document.get('_id', ''))
            
            return jsonify({
                "status": "success",
                "message": f"Successfully added {food_name}",
                "item": food_document
            }), 200

        except Exception as e:
            print(f"Error in scan_image: {e}")
            return jsonify({"error": str(e)}), 500

@app.route('/api/manual-add', methods=['POST'])
def manual_add():
    data = request.json
    label = data.get('label')
    
    return jsonify({
        "status": "success", 
        "message": f"Manually added {label}",
        "item": {"name": label}
    }), 200

@app.route('/api/inventory', methods=['GET'])
def list_inventory():
    items = query(mongo_client, DB_NAME, COL_NAME, {})
    return jsonify(items), 200

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)