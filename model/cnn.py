import math
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Download ResNet50(pre-trained CNN on ImageNet, great for image classification)
from tensorflow.keras.applications import ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation

# Add classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # This reduces the 7x7x2048 output to 2048


x = Dense(256, kernel_regularizer=l2(0.001))(x)  # This is our 1st Dense, connecte4d layer for the model, with 256 neurons. L2 Regularization is affixed to prevent overfitting.
x = BatchNormalization()(x)   #
x = Activation('relu')(x)
x = Dropout(0.3)(x)

x = Dense(256, kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)

x = Dense(1, activation='sigmoid')(x)  # Final layer for binary classification

for layer in base_model.layers[-15:]:  # This will unfreeze the top set # of layers in the ResNet model we imported earlier so we can specifically train it on our xray samples
        layer.trainable = True         # This sets the unfrozen layers to be trained while we train the model

from tensorflow.keras import backend as K
from keras.optimizers import Adam

# Define initial learning rate
initial_learning_rate = 1e-5           # A very small inital learning rate so the model can train gradually on the Xray scans, which tend to be very noisy

# Set up exponential decay schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=10000,                 # This will decay the learning rate every 10,000 training steps, to prevent the model from overfitting to our training dataset
    decay_rate=0.95,                   # This will reduce the learning rate by 5% every 10,000 training steps
    staircase=True                     # This means the learning rate will fall by 5% at every 10,000 steps, instead of gradually during those steps
)
model = Model(inputs=base_model.input, outputs=x)
# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    #ebrightness_range=[0.1, 0.3],
    zoom_range=0.2,
    #shear_range=0.2,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

from pickle import TRUE
import types

# Load the data
train_generator = train_datagen.flow_from_directory(
    '/home/ubuntu/chest_xray/train',
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    '/home/ubuntu/chest_xray/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False,
)
test_generator = test_datagen.flow_from_directory(
    '/home/ubuntu/chest_xray/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False,
)

# # Apply set_length to train and validation generators
# train_generator = set_length(train_generator)
# val_generator = set_length(val_generator)
def convert_to_float32(images, labels):
    return tf.cast(images, tf.float32), labels

train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
                      tf.TensorSpec(shape=(None,), dtype=tf.float32))
).map(convert_to_float32)
val_dataset = tf.data.Dataset.from_generator(
    lambda: val_generator,
    output_signature=(tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
                      tf.TensorSpec(shape=(None,), dtype=tf.float32))
).map(convert_to_float32)

test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator,
    output_signature=(tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
                      tf.TensorSpec(shape=(None,), dtype=tf.float32))
).map(convert_to_float32)

from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True),
    #ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7)
]

from sklearn.utils.class_weight import compute_class_weight


class_weights = {0: 1.1, 1: 1.4}

# Train the model
history = model.fit(
    train_dataset,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,  # Adjust as needed
    validation_data=val_dataset,
    validation_steps = math.ceil(val_generator.samples / val_generator.batch_size),
    class_weight=class_weights,  # Add class_weight here
    callbacks=callbacks
)

model.save('/home/ubuntu/CNN_deploy/model/pneumonia_model.keras')  # Using .keras format instead of .h5
# Load the trained model for inference
from tensorflow.keras.models import load_model

import os
# test the model all images in /content/chest_xray/test and get the predictions of each image
# Get the list of image files in the test directory
test_dir = '/home/ubuntu/chest_xray/test'
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