import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import models, layers
from keras_visualizer import visualizer
from keras.utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import pandas as pd


# Function to calculate parameter size
def calculate_parameter_size(model):
    return sum([np.product(w.shape) for w in model.trainable_weights])

# Function to calculate weight size
def calculate_weight_size(model):
    return sum([w.nbytes for w in model.trainable_weights])

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, fold):
    plt.plot(fpr, tpr, lw=2, label='ROC curve Fold %d (area = %0.2f)' % (fold, roc_auc))

# Function to plot Precision-Recall curve
def plot_precision_recall_curve(recall, precision, pr_auc, fold):
    plt.plot(recall, precision, lw=2, label='PR curve Fold %d (area = %0.2f)' % (fold, pr_auc))

# Function to plot accuracy-loss curve for training and validation
def plot_convergence(history, fold):
    plt.plot(history.history['accuracy'], label='Training Accuracy - Fold %d' % fold)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy - Fold %d' % fold)
    plt.title('Model Accuracy - Fold %d' % fold)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss - Fold %d' % fold)
    plt.plot(history.history['val_loss'], label='Validation Loss - Fold %d' % fold)
    plt.title('Model Loss - Fold %d' % fold)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Define parameters
train_folder = 'dataset'
test_folder = 'test_dataset'
img_width, img_height = 224, 224
batch_size = 32
epochs = 10
k = 5  # Number of folds
channels=3
# Load data without preprocessing
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,validation_split=0.2)
data_gen = train_datagen.flow_from_directory(
    train_folder, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode='binary', 
    shuffle=True,
    
)
data, labels = next(data_gen)


# Initialize lists to store metrics
roc_auc_scores = []
pr_auc_scores = []
parameter_sizes = []
weight_sizes = []

fprs = []
tprs = []
roc_aucs = []
recalls = []
precisions = []

histories = []
# Define k-fold cross-validation
skf = StratifiedKFold(n_splits=k)

fold = 1
# Use data and labels in skf.split()
for train_index, val_index in skf.split(data, labels):
    print(f'Fold {fold}')
    # Split data into train and validation sets
    train_generator_fold = train_datagen.flow_from_directory(
        train_folder,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',  # Use 'training' subset for training data
        shuffle=True,
        seed=42  # Optional: set seed for reproducibility
    )
    
    validation_generator_fold = train_datagen.flow_from_directory(
        train_folder,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',  # Use 'validation' subset for validation data
        shuffle=True,
        seed=42  # Optional: set seed for reproducibility
    )
        
    # Create and compile model
    base_model = ResNet50(weights='imagenet', include_top=True)
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



    # Train model
    history = model.fit(train_generator_fold, epochs=epochs, validation_data=validation_generator_fold)
    histories.append(history)
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(validation_generator_fold)

    # Calculate ROC AUC and PR AUC
    y_true = validation_generator_fold.labels
    y_pred = model.predict(validation_generator_fold)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    roc_auc_scores.append(roc_auc)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    pr_auc_scores.append(pr_auc)

    # Calculate and store parameter size and weight size
    parameter_sizes.append(calculate_parameter_size(model))
    fprs.append(fpr)
    tprs.append(tpr)
    roc_aucs.append(roc_auc)

    # Store recall, precision for Precision-Recall curve
    recalls.append(recall)
    precisions.append(precision)
    # Plot ROC curve, Precision-Recall curve for testing, and convergence curves
    #plot_roc_curve(fpr, tpr, roc_auc, fold)
    #plot_precision_recall_curve(recall, precision, pr_auc, fold)
    #plot_convergence(history, fold)

    fold += 1

# Print mean metrics
print("Mean ROC AUC:", np.mean(roc_auc_scores))
print("Mean PR AUC:", np.mean(pr_auc_scores))
print("Mean Parameter Size:", np.mean(parameter_sizes))
print("Mean Weight Size:", np.mean(weight_sizes))

plt.figure(1)
for i in range(len(fprs)):
    plt.plot(fprs[i], tprs[i], label='ROC curve Fold %d (area = %0.2f)' % (i+1, roc_aucs[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Plot Precision-Recall curve for each fold
plt.figure(2)
for i in range(len(recalls)):
    plt.plot(recalls[i], precisions[i], label='Precision-Recall curve Fold %d' % (i+1))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower right')
plt.show()

plt.figure(figsize=(12, 6))
for i, history in enumerate(histories):
    plt.plot(history.history['accuracy'], color='blue', label='Train Accuracy Fold %d' % (i+1))
    plt.plot(history.history['val_accuracy'],color='orange', label='Val Accuracy Fold %d' % (i+1))
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()

# Plot training & validation loss values for each fold
plt.figure(figsize=(12, 6))
for i, history in enumerate(histories):
    plt.plot(history.history['loss'],  color='blue', label='Train Loss Fold %d' % (i+1))
    plt.plot(history.history['val_loss'],color='orange', label='Val Loss Fold %d' % (i+1))
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

    
model.summary()   
# Save model
model.save('eczema_detection_modelnopre.keras')