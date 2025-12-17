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