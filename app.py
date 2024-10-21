import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import pandas as pd
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights

# Load disease and supplement info
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load pre-trained ResNet50 model
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Prediction function that returns the predicted class
def predict():
    try:
        # Get the uploaded image from the request
        image = request.files['image']
        img = Image.open(image)
        img = transform(img)

        # Make a prediction
        with torch.no_grad():
            output = model(img.unsqueeze(0))  # Unsqueeze to add batch dimension
            predicted_class = torch.argmax(output)

        # Return the predicted class index directly (not as JSON)
        return predicted_class.item()

    except Exception as e:
        # Log error for debugging and return None if prediction fails
        print(f"Error during prediction: {str(e)}")
        return None

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    try:
        if request.method == 'POST':
            image = request.files['image']
            filename = image.filename
            file_path = os.path.join('static/uploads', filename)
            image.save(file_path)
            print(f"Saved image to: {file_path}")

            # Call the predict function and get the predicted class
            pred = predict()

            # If prediction fails, return an error response
            if pred is None:
                return jsonify({'error': 'Prediction failed'}), 500

            # Map the predicted class to the range of the disease_info DataFrame
            mapped_pred = pred % len(disease_info)  # Modulo to map prediction to valid index

            # Retrieve disease information based on the mapped prediction
            title = disease_info['disease_name'][mapped_pred]
            description = disease_info['description'][mapped_pred]
            prevent = disease_info['Possible Steps'][mapped_pred]
            image_url = disease_info['image_url'][mapped_pred]
            supplement_name = supplement_info['supplement name'][mapped_pred]
            supplement_image_url = supplement_info['supplement image'][mapped_pred]
            supplement_buy_link = supplement_info['buy link'][mapped_pred]

            # Render the submit template with the disease info
            return render_template('submit.html', title=title, desc=description, prevent=prevent,
                                   image_url=image_url, pred=mapped_pred, sname=supplement_name, 
                                   simage=supplement_image_url, buy_link=supplement_buy_link)
    except Exception as e:
        # Handle errors and return an error response
        return jsonify({'error': str(e)}), 500

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']),
                           buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
