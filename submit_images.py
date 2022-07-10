import pandas as pd
import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = keras.models.load_model('woof-im-hungry-model')

base_path = 'dataset/submission'

image_generator = ImageDataGenerator(
        rescale = 1/255,
        rotation_range = 10,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        zoom_range = 0.1,
        horizontal_flip = True,
        brightness_range = [0.2, 1.2],
        validation_split = 0.2,)

image_generator_submission = ImageDataGenerator(rescale=1/255)

submission = image_generator_submission.flow_from_directory(
                                                 directory=base_path,
                                                 shuffle=False,
                                                 target_size=(224, 224),
                                                 class_mode=None)
images = []
#Get the images in the order that they were added to the submissions list
for dir in os.listdir(base_path)[::-1]:
    for file in os.listdir(os.path.join(base_path, dir)):
        images.append(file.split('.')[0])

submission_df = pd.DataFrame(images, columns=['images'])

submission_df[['empty', 'full']] = model.predict(submission)

predictions = []
#Create a column with the greater value, full or empty
for index, row in submission_df.iterrows():
    if row['empty'] > row['full']:
        predictions.append('empty')
    else:
        predictions.append('full')

submission_df['prediction'] = predictions

print(submission_df)
submission_df.to_csv('submissions.csv', index = False)

