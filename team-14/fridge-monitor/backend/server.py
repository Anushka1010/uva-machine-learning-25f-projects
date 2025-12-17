from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/api/scan', methods=['POST'])
def scan_image():
    # check if file exists
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    scan_type = request.form.get('type', 'IN')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # save file locally 
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # run inference
        dummy_result = {
            "status": "success",
            "filename": file.filename,
            "detected_objects": ["apple", "banana"],
            "scan_type": scan_type
        }
        
        return jsonify(dummy_result), 200

@app.route('/api/manual-add', methods=['POST'])
def manual_add():
    # handle JSON data
    data = request.json
    label = data.get('label')
    
    return jsonify({
        "status": "success", 
        "message": f"Manually added {label}",
        "item": label
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)