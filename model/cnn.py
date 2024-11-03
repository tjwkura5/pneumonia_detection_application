import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Download ResNet50(pre-trained CNN on ImageNet, great for image classification)
from tensorflow.keras.applications import ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze last 30 layers of ResNet50
for layer in base_model.layers[-30:]:
    layer.trainable = True

from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

# Add classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduces the output to 2048
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)  # Moved after activation
x = Dropout(0.5)(x)  # Added dropout after first dense layer
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)  # Another dropout layer
x = Dense(128, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)  # Final layer for binary classification

from keras.optimizers import Adam
model = Model(inputs=base_model.input, outputs=x)

# Compile with reduced learning rate
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']

)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Updated data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Use class weights if needed
class_weight = {0: 1.0, 1: 1.5}  # Adjust based on class imbalance analysis

# Load the data
train_generator = train_datagen.flow_from_directory(
    '/content/chest_xray/train',
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    '/content/chest_xray/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)
test_generator = test_datagen.flow_from_directory(
    '/content/chest_xray/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

from tensorflow.keras.callbacks import ReduceLROnPlateau

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=20,
    class_weight=class_weight,
    callbacks=callbacks
)

model.save('/home/ubuntu/CNN_deploy/model/best_model.keras')  # Using .keras format instead of .h5
# Load the trained model for inference
from tensorflow.keras.models import load_model

import os
# test the model all images in /content/chest_xray/test and get the predictions of each image
# Get the list of image files in the test directory
test_dir = '/content/chest_xray/test'
for root, dirs, files in os.walk(test_dir):
  for file in files:
    if file.endswith(('.jpg', '.jpeg', '.png')):
      test_image_path = os.path.join(root, file)
      # Load and preprocess the image
      img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(224, 224))
      img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.
      img_array = np.expand_dims(img_array, axis=0)

      # Make the prediction
      prediction = model.predict(img_array)
      confidence = float(prediction[0][0])
      result = "Pneumonia" if confidence > 0.5 else "Normal"

      # Print the results
      print(f"Image: {test_image_path}")
      print(f"Prediction: {result} (confidence: {confidence:.2%})")
      print("---")