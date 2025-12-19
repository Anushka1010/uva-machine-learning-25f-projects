# for video
import cv2

# for database and ai
from api import connect_mongo, connect_gemini, query, add_entry, analyze_photo, get_drive_creds, get_drive_files, get_drive_file_url_and_set_permission, upload_photo_to_drive, CNN, load_model, analyze_photo_pytorch
import time
from datetime import datetime, timezone

# for loading environment variables
import os
from dotenv import load_dotenv
load_dotenv()

# constants
URI = os.getenv('URI')
DB_NAME = os.getenv('DB_NAME')
COL_NAME = os.getenv('COL_NAME')
DRIVE_FOLDER_ID = os.getenv('DRIVE_FOLDER_ID')

if __name__ == "__main__":
    # connect to mongo client 
    mongo_client = connect_mongo(URI)

    # init model
    model = load_model()

    # get drive credentials
    creds = get_drive_creds()

    # camera stuff
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()

        # Display the captured frame
        cv2.imshow('Camera', frame)
        
        # collect key press
        key = cv2.waitKey(1)

        # Press 'a' to snap send photo to gemini for analysis
        if key == ord('a'):
            image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()

            # send photo to gemini
            food_name = analyze_photo_pytorch(model, image_bytes)

            # save photo to Google Drive
            filename = f"{food_name.replace(' ', '_').lower()}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            drive_file_data = upload_photo_to_drive(creds, image_bytes, filename, DRIVE_FOLDER_ID)
            
            if not drive_file_data:
                print("Failed to upload to drive. Discontinuing database storage.")
                continue

            # get public url and set permissions
            file_id = drive_file_data.get('id')
            public_url = get_drive_file_url_and_set_permission(creds, file_id)

            if not public_url:
                print("Failed to get public URL. Discontinuing database storage.")
                continue

            # prepare dates
            date_placed = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            
            # built document
            food_document = {
                "name": food_name,
                "image": {
                    "url": public_url,
                    "file_id": file_id
                },
                "date_placed": date_placed,
                "expiration_date": date_placed
            }

            # add to database
            try:
                add_entry(mongo_client, DB_NAME, COL_NAME, food_document)
            except Exception as e:
                print(f"An error occurred: {e}")

        # Press 'q' to exit the loop
        elif key == ord('q'):
            break

    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()