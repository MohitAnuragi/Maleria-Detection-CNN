import os
import argparse
import tensorflow as tf
from model import build_malaria_cnn
from utils import load_data, plot_training_history

def main(args):
    # Setup random seed for reproducibility
    tf.random.set_seed(42)
    
    # Load data
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        print("Please download the dataset and extract it to this directory.")
        return
        
    train_ds, val_ds = load_data(
        args.data_dir, 
        img_size=(128, 128), 
        batch_size=args.batch_size
    )
    
    # Build model
    model = build_malaria_cnn(input_shape=(128, 128, 3))
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    model.summary()
    
    # Callbacks
    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')
        
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath='saved_model/malaria_cnn.keras',
        save_best_only=True,
        monitor='val_loss'
    )
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True,
        monitor='val_loss'
    )
    
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )
    
    # Train
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
    )
    
    # Plot history
    print("Training finished. Plotting history...")
    plot_training_history(history)
    print("Done! Model saved to 'saved_model/malaria_cnn.keras'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Malaria Detection CNN')
    parser.add_argument('--data_dir', type=str, default='data/cell_images', help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    main(args)
