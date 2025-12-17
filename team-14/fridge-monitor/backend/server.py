# flask imports
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from bson import ObjectId

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

# pytorch model setup
CLASSES = [
    "asian pear",
    "cucumber",
    "eggs",
    'hand',
    "leafy green",
    "leftovers",
    "orange",
    "sauce",
    "soda",
    "tomato"
]

MODEL_PATH = '../model/fridge_model_complete.pth' # path to .pth
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# architecture
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, 1, 1)
        self.conv2 = nn.Conv2d(32, 128, 4, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 4, 1, 1)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5) 
        
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = x.view(-1, 256 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

# load model
try:
    model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# define transforms 
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def analyze_photo_pytorch(image_bytes):
    """Predicts class using the loaded PyTorch model."""
    if model is None:
        return "Model Error"
    
    try:
        # convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # apply transforms
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # inference
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            
        class_idx = predicted.item()
        if 0 <= class_idx < len(CLASSES):
            return CLASSES[class_idx]
        else:
            return f"Unknown Class ({class_idx})"
            
    except Exception as e:
        print(f"Inference error: {e}")
        return "Analysis Failed"

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

@app.route('/api/inventory/<item_id>', methods=['PATCH'])
def update_inventory_item(item_id):
    data = request.json
    update_data = {}
    if 'expiration_date' in data:
        update_data['expiration_date'] = data['expiration_date']
    if 'status' in data:
        update_data['status'] = data['status']
    
    if update_data:
        mydb = mongo_client[DB_NAME]
        mycol = mydb[COL_NAME]
        mycol.update_one({'_id': ObjectId(item_id)}, {'$set': update_data})
    
    return jsonify({"status": "success", "message": "Item updated"}), 200

@app.route('/api/inventory', methods=['GET'])
def list_inventory():
    items = query(mongo_client, DB_NAME, COL_NAME, {})
    # Ensure all items have status
    for item in items:
        if 'status' not in item:
            item['status'] = 'in_fridge'
    return jsonify(items), 200

@app.route('/api/events', methods=['GET'])
def list_events():
    return jsonify([]), 200

@app.route('/api/alerts', methods=['GET'])
def list_alerts():
    items = query(mongo_client, DB_NAME, COL_NAME, {})
    now = datetime.now(timezone.utc)
    out = []

    for item in items:
        if item.get('status') != 'in_fridge' or not item.get('expiration_date'):
            continue

        created = datetime.fromisoformat(item['date_placed'].replace('Z', '+00:00'))
        expires = datetime.fromisoformat(item['expiration_date'].replace('Z', '+00:00'))

        total = (expires - created).total_seconds()
        used = (now - created).total_seconds()
        pct = used / total if total > 0 else 0
        days_left = (expires - now).total_seconds() / 86400

        if 0.66 <= pct <= 0.75:
            out.append({
                'type': 'shelf_life',
                'message': f"{item['name']} is ~{round(pct * 100)}% through shelf life",
                'itemId': str(item['_id'])
            })
        if 0 <= days_left <= 1.1:
            out.append({
                'type': 'one_day',
                'message': f"{item['name']} expires in ~1 day",
                'itemId': str(item['_id'])
            })
        if days_left < 0:
            out.append({
                'type': 'expired',
                'message': f"{item['name']} has expired",
                'itemId': str(item['_id'])
            })

    return jsonify(out), 200

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)