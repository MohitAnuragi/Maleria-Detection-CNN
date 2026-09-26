import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def predict_image(model_path, image_path, target_size=(128, 128)):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'. Please train the model first.")
        return
        
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'.")
        return
        
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    
    # Predict
    prediction = model.predict(img_array)[0][0]
    
    # In alphabetical sorting (tf.keras.utils.image_dataset_from_directory default):
    # 'Parasitized' is 0, 'Uninfected' is 1
    label = 'Uninfected' if prediction > 0.5 else 'Parasitized'
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    print(f"Prediction: {label} (Confidence: {confidence:.2%})")
    
    # Plot
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {label}\nConfidence: {confidence:.2%}")
    
    save_dir = 'results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, 'sample_predictions.png')
    plt.savefig(save_path)
    print(f"Prediction image saved to '{save_path}'")
    
    # Try to show it if display is available
    try:
        plt.show()
    except:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Malaria from a Single Image')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model', type=str, default='saved_model/malaria_cnn.keras', help='Path to saved model')
    
    args = parser.parse_args()
    predict_image(args.model, args.image)
