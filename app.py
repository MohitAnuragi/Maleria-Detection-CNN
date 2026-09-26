from flask import Flask, request, jsonify, render_template
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

def is_valid_blood_cell(img_array):
    """
    A mathematical heuristic to detect if an image is likely NOT a microscopic blood cell.
    Prevents natural photos (like faces) from being confidently classified.
    """
    std_dev = np.std(img_array)
    mean_g = np.mean(img_array[:,:,1])
    mean_r = np.mean(img_array[:,:,0])
    
    # Natural photos have very high variance (contrast). Blood smears are mostly background.
    if std_dev > 75: 
        return False
        
    # Microscope slides are backlit and bright; dark images are likely not slides.
    if np.mean(img_array) < 50:
        return False
        
    # If Green is significantly brighter than Red (highly unusual for pink/purple stains)
    if mean_g > mean_r + 20:
        return False

    return True


# Load model globally
MODEL_PATH = 'saved_model/malaria_cnn.keras'
model = None

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
else:
    print(f"Error: Model not found at {MODEL_PATH}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400
        
    try:
        # Save temporarily
        temp_path = os.path.join('static', 'temp.png')
        if not os.path.exists('static'):
            os.makedirs('static')
        file.save(temp_path)
        
        # Preprocess for the CNN (128x128, scaled 0-1)
        img = tf.keras.preprocessing.image.load_img(temp_path, target_size=(128, 128))
        img_array_raw = tf.keras.preprocessing.image.img_to_array(img)
        
        # Out-of-Distribution Check (Anomaly Detection)
        if not is_valid_blood_cell(img_array_raw):
            return jsonify({'error': 'Image rejected: This does not appear to be a valid microscopic blood cell image.'}), 400
            
        img_array = img_array_raw / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)[0][0]
        
        # Class 0: Parasitized, Class 1: Uninfected
        label = 'Uninfected' if prediction > 0.5 else 'Parasitized'
        confidence = float(prediction if prediction > 0.5 else 1 - prediction)
        
        return jsonify({
            'prediction': label,
            'confidence': confidence,
            'image_url': '/static/temp.png'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the web server
    print("Starting server... Access it at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
