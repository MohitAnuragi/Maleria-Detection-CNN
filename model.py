import tensorflow as tf
from tensorflow.keras import layers, models

def build_malaria_cnn(input_shape=(128, 128, 3)):
    """
    Builds a custom CNN model for malaria detection.
    
    Args:
        input_shape (tuple): The shape of the input images.
        
    Returns:
        tf.keras.Model: The compiled CNN model.
    """
    model = models.Sequential([
        # Data augmentation
        layers.RandomFlip("horizontal_and_vertical", input_shape=input_shape),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        
        # Conv block 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Conv block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Conv block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Conv block 4
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Global Average Pooling and Dense Layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model
