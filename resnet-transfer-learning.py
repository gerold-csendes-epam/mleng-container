# %%
import pathlib

import numpy as np

from tensorflow import keras
from tensorflow.keras.datasets import cifar10 
from tensorflow.keras.applications import ResNet152
from tensorflow.keras import layers, models, losses, Model
from tensorflow.keras.callbacks import EarlyStopping

from matplotlib import pyplot
from livelossplot import PlotLossesKeras


# %%
root_dir = pathlib.Path.cwd()
data_dir = root_dir.joinpath("data", "raw")

X_train = np.load(data_dir.joinpath(data_dir, "X_train.npy"))
y_train = np.load(data_dir.joinpath(data_dir, "y_train.npy"))
X_test = np.load(data_dir.joinpath(data_dir, "X_test.npy"))
y_test = np.load(data_dir.joinpath(data_dir, "y_test.npy"))

# %%
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# plot raw pixel data
	pyplot.imshow(X_train[i])
# show the figure
pyplot.show()

print("Labels: ")
print(y_train[:3].flatten())
print(y_train[3:6].flatten())
print(y_train[6:9].flatten())

print(f"Image size: {X_train.shape}")
# %%
# import model
base_resnet = ResNet152(weights='imagenet', include_top=False, input_shape=(32,32,3))
# the resnet layers should be frozen = not trained
for layer in base_resnet.layers:
  layer.trainable = False

# Functional model API
type(base_resnet)

# Add head aka last layer
x = layers.Flatten()(base_resnet.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
predictions = layers.Dense(10, activation = 'softmax')(x)

# put model together
head_model = Model(inputs = base_resnet.input, outputs = predictions)
head_model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
# %%
es = EarlyStopping(monitor='loss', min_delta=0.01, patience=3)

head_model.fit(X_train, y_train,
          epochs=2,
          batch_size=32,
          validation_split=0.2,
          callbacks=[es, PlotLossesKeras()],
          verbose=1)
# %%
head_model.evaluate(X_test, y_test)
# %%
head_model.save(root_dir.joinpath("models", "resnet-transfer"))