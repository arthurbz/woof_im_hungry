import numpy as np
import cv2
import warnings
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.simplefilter('ignore')

image_generator = ImageDataGenerator(
        rescale = 1/255,
        rotation_range = 10,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        zoom_range = 0.1,
        horizontal_flip = True,
        brightness_range = [0.2, 1.2],
        validation_split = 0.2,)

train_dataset = image_generator.flow_from_directory(batch_size = 32,
                                                    directory = 'dataset/train',
                                                    shuffle = True,
                                                    target_size = (224, 224),
                                                    subset = 'training',
                                                    class_mode = 'categorical')

validation_dataset = image_generator.flow_from_directory(batch_size = 32,
                                                         directory = 'dataset/validation',
                                                         shuffle = True,
                                                         target_size = (224, 224),
                                                         subset = 'validation',
                                                         class_mode = 'categorical')

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = [224, 224, 3]),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, (2, 2), activation = 'relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, (2, 2), activation = 'relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation = 'relu'),
    keras.layers.Dense(2, activation = 'softmax')
])

model.compile(optimizer='adam',
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])

callback = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                         patience = 3,
                                         restore_best_weights = True)

model.fit(train_dataset, epochs = 3, validation_data = validation_dataset, callbacks = callback)

loss, accuracy = model.evaluate(validation_dataset)
print('Loss: ', loss)
print('Accuracy: ', accuracy)

model.save('woof-im-hungry-model')
