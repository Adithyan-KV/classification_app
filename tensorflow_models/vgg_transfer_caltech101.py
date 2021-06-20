import tensorflow as tf
from tensorflow.keras import models, layers, datasets
import numpy as np
import matplotlib.pyplot as plt

data_dir = './caltech101'
batch_size = 16
img_height = 224
img_width = 224
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])
num_classes = len(class_names)
print(num_classes)
vgg_16=tf.keras.applications.vgg16.VGG16(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000,
)
vgg_16.trainable = False

prediction_layer = layers.Dense(102)
flatten_layer = layers.Flatten()
fc_layer = layers.Dense(4096)
fc_layer_2 = layers.Dense(512)
dropout_layer = layers.Dropout(0.1)
preprocessing_layer = layers.experimental.preprocessing.Rescaling(1./255)
inputs = tf.keras.Input(shape=(224, 224, 3))
# x = data_augmentation(inputs)
# x = preprocessing_layer(x)
x = preprocessing_layer(inputs)
x = vgg_16(x, training=False)
x = flatten_layer(x)
# x = dropout_layer(x)
# x = fc_layer(x)
x = fc_layer_2(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

initial_epochs = 5

loss0, accuracy0 = model.evaluate(val_ds)

history = model.fit(train_ds,
                    epochs=initial_epochs,
                    validation_data=val_ds)

model.save('./saved_model/vgg_transfer')