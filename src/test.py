import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array, image_dataset_from_directory
import pathlib



path_img = os.path.join('validation', 'integras', 'integra2.png')
class_names = ['desmatadas', 'integras'] 
img_width = 320
img_height = 180

model = load_model('modelo')

for i in range(1, 4):
    file =  f'validation/integras/integra{i}.png'   
    img = tf.keras.utils.load_img(
        file, target_size=(img_width, img_height)
    )
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(file)
    print(
        "Essa imagem parece pertencer à uma floresta {} com {:.2f}% de precisão."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

for i in range(1, 5):
    file = f'validation/desmatadas/desmatada{i}.png'    
    img = tf.keras.utils.load_img(
        file, target_size=(img_width, img_height)
    )
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(file)
    print(
        "Essa imagem parece pertencer à uma floresta {} com {:.2f}% de precisão."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

validation_dir = pathlib.Path('validation')
validation_ds = image_dataset_from_directory(
    validation_dir,
    image_size=(img_width, img_height)
) 

predictions = model.predict(validation_ds)
print(predictions)
