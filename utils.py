import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

def load_data(data_dir, img_size=(128, 128), batch_size=32, val_split=0.2, seed=42):
    """
    Loads and splits dataset into training and validation sets.
    """
    print(f"Loading data from {data_dir}...")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    
    # Normalize pixel values
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Prefetching
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def plot_training_history(history, save_dir='results'):
    """
    Plots training and validation accuracy and loss.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def evaluate_model(model, val_ds, class_names=['Parasitized', 'Uninfected'], save_dir='results'):
    """
    Evaluates the model, generating confusion matrix and metrics.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print("Evaluating model...")
    predictions = []
    labels = []
    
    for x, y in val_ds:
        preds = model.predict(x, verbose=0)
        predictions.extend(preds.flatten())
        labels.extend(y.numpy().flatten())
        
    predictions = np.array(predictions)
    labels = np.array(labels)
    pred_labels = (predictions > 0.5).astype(int)
    
    # Metrics
    report = classification_report(labels, pred_labels, target_names=class_names)
    print("Classification Report:\n", report)
    
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n")
        
    # Confusion Matrix
    cm = confusion_matrix(labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    # ROC-AUC
    try:
        auc = roc_auc_score(labels, predictions)
        fpr, tpr, _ = roc_curve(labels, predictions)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
        plt.close()
        
        print(f"ROC AUC Score: {auc:.4f}")
        with open(os.path.join(save_dir, 'metrics.txt'), 'a') as f:
            f.write(f"\nROC AUC Score: {auc:.4f}\n")
    except ValueError:
        print("ROC-AUC score could not be calculated. This happens when only one class is present in the dataset.")
