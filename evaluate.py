import os
import argparse
import tensorflow as tf
from utils import load_data, evaluate_model

def main(args):
    model_path = 'saved_model/malaria_cnn.keras'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'. Please run train.py first.")
        return
        
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        return
        
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # In a proper setup, we would evaluate on a strictly separated test set.
    # For this assignment's current loader structure, we will use the validation set
    # as a proxy for the test evaluation (or a separate folder if provided).
    print("Loading test data...")
    _, test_ds = load_data(args.data_dir, img_size=(128, 128), batch_size=32)
    
    evaluate_model(model, test_ds, save_dir='results')
    print("Evaluation complete. Results saved in 'results/' directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Malaria Detection CNN')
    parser.add_argument('--data_dir', type=str, default='data/cell_images', help='Path to dataset directory')
    
    args = parser.parse_args()
    main(args)
