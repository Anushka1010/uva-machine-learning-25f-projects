from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from api import connect_mongo, query
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# flask app setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# create uploads folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# database setup
URI = os.getenv('URI')
DB_NAME = os.getenv('DB_NAME')
COL_NAME = os.getenv('COL_NAME')

# initialize client
mongo_client = connect_mongo(URI)

# api endpoints
@app.route('/api/scan', methods=['POST'])
def scan_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    scan_type = request.form.get('type', 'IN')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # mock inference result
        dummy_result = {
            "status": "success",
            "filename": file.filename,
            "detected_objects": ["apple", "banana"],
            "scan_type": scan_type
        }
        
        return jsonify(dummy_result), 200

@app.route('/api/manual-add', methods=['POST'])
def manual_add():
    data = request.json
    label = data.get('label')
    
    return jsonify({
        "status": "success", 
        "message": f"Manually added {label}",
        "item": label
    }), 200

@app.route('/api/inventory', methods=['GET'])
def list_inventory():
    items = query(mongo_client, DB_NAME, COL_NAME, {})
    
    return jsonify(items), 200

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)