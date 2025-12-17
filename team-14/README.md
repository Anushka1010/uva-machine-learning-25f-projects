# Fridge Inventory Monitor: A Modern Solution to Expiration Prevention

## Team Information
Team ID: 14
Team Members: Joyce Yang, Mohan Liu, Nurdin Hossain, Rachel Yu

## Overview
The Fridge Inventory Monitor is a smart refrigerator management system designed to help users, particularly college students, reduce food waste, save money, and maintain a balanced diet. Using computer vision and machine learning, our system automatically tracks items stored in your refrigerator, monitors expiration dates, and helps you make the most of your groceries.

We've developed a system that:
- Monitors refrigerator contents using camera-based computer vision
- Tracks food age and expiration dates automatically
- Generates recipe suggestions based on available ingredients, prioritizing items near expiration
- Provides nutritional information to support balanced meal planning

## Technical Architecture
Our system consists of three main components:

1. Object Detection ML Model
- Custom PyTorch CNN architecture trained for food item detection and classification
- Trained on custom-labeled dataset (YOLOv5 format via Roboflow)
- Implemented in Google Colab for model training

2. Backend (MongoDB)
- Stores detected food items with metadata (detection timestamp, expiration estimates)
- Maintains inventory history
- Provides API endpoints for frontend communication

3. Frontend (Next.js)
- User-friendly interface for viewing current inventory
- Visual display of food freshness and expiration warnings
- Recipe suggestions and nutritional tracking

## Usage

1. Model Training
- Open the ipynb notebook in the `model/` directory
- The notebook uses Roboflow API to access our custom-labeled training data 
*Note: read the file `data_usage.md` for more information on how to use the data to run the model*
- Run all cells to train the CNN model
- Export the trained model for integration with the backend

2. Frontend
- Navigate to the `frontend/` directory and run the development server following the `README.md` file

3. Backend
- instructions

## Video Demo
