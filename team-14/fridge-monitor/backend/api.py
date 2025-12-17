# for database
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# for ai
from google import genai
from google.genai import types

# for google drive
import io
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

SCOPES = ["https://www.googleapis.com/auth/drive"]

# method to connect to client
def connect_mongo(uri: str) -> MongoClient:
    return MongoClient(uri, server_api=ServerApi('1'))

# query from database
def query(client: MongoClient, db_name: str, col_name: str, query: dict) -> list:
    mydb = client[db_name]
    mycol = mydb[col_name]

    mydoc = mycol.find(query)
    results = []

    for item in mydoc:
        item['_id'] = str(item['_id'])    
        results.append(item)

    return results

# add entry to database 
def add_entry(client: MongoClient, db_name: str, col_name: str, entry: dict):
    mydb = client[db_name]
    mycol = mydb[col_name]

    x = mycol.insert_one(entry)

# method to connect to gemini client
def connect_gemini(key: str) -> genai.Client:
    return genai.Client(api_key=key)

# send photo and query to gemini
def analyze_photo(client: genai.Client, image_bytes: bytes, inquiry: str) -> str:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg',
            ),
            inquiry
        ]
    )

    return response.text

# get creds for google drive
def get_drive_creds() -> Credentials:
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return creds

# collect files from google drive
def get_drive_files(creds: Credentials, limit: int) -> list:
    try:
        service = build("drive", "v3", credentials=creds)

        # call drive v3 api
        results = (
            service.files()
            .list(pageSize=limit, fields="nextPageToken, files(id, name)")
            .execute()
        )
        items = results.get("files", [])

        if not items:
            return []
        
        return [item for item in items]
    except HttpError as error:
        print(f"An error occurred: {error}")

    return []

# get public 'webViewLink' and set permissions
def get_drive_file_url_and_set_permission(creds: Credentials, file_id: str) -> str | None:
    try:
        # init service
        service = build("drive", "v3", credentials=creds)

        # get metadata 
        file_metadata = service.files().get(
            fileId=file_id, 
            fields="webViewLink"
        ).execute()
        
        url = file_metadata.get("webViewLink")

        # set perms for anyone to read file
        permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        service.permissions().create(
            fileId=file_id,
            body=permission,
            fields='id',
        ).execute()

        return url
        
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None

# send photo to google drive for database storage
def upload_photo_to_drive(creds: Credentials, image_bytes: bytes, file_name: str, folder_id: str) -> dict | None:
    try:
        # initialize service
        service = build("drive", "v3", credentials=creds)
        
        # prepare file metadata
        file_metadata = {"name": file_name, "parents": [folder_id]}
        
        # create memory stream for raw image bytes
        media = MediaIoBaseUpload(
            io.BytesIO(image_bytes),
            mimetype="image/jpeg", 
            resumable=True
        )

        # upload file to drive
        file = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id, name")
            .execute()
        )

        return file

    except HttpError as error:
        print(f"An error occurred: {error}")
        return None
    
# pytorch code

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
def load_model():
    model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.to(DEVICE)
    model.eval()

    return model

def analyze_photo_pytorch(model, image_bytes: bytes):
    """Predicts class using the loaded PyTorch model."""
    # validation transform
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

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
