import os
import numpy as np
# import pandas as pd
import tensorflow as tf

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

'''
Create tf.data.Dataset from a directory of Images.
Preprocess Images : Resize, Flip, Normalize
'''


def preprocess_image(image, image_size):
    
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.image.random_flip_left_right(image)
    image /= 255.0
    return image


def parse_image_function(image, image_size):
      
    image_string = tf.read_file(image)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = preprocess_image(image, image_size)
    label = image.split('/')[-2]
    return image, label


def get_dataset(dir, params=5, phase='train'):

    dir_paths  =  os.listdir(dir)
    dir_paths  =  [os.path.join(dir, dir_path) for dir_path in dir_paths]

    image_paths = []
    for dir_path in dir_paths:
        for image_path in os.listdir(dir_path):
            image_paths.append(os.path.join(dir_path, image_path))

    AUTOTUNE   =  tf.data.experimental.AUTOTUNE
    dataset    =  tf.data.Dataset.from_tensor_slices(file_paths)
    dataset    =  dataset.map(lambda x: parse_image_function(x, 250)) #change here
    dataset    =  dataset.batch(24).prefetch(AUTOTUNE) #change here
    
    return dataset, len(file_paths)


def old_get_dataset(tfrcd_dir, params, phase='train'):
    
    if phase == 'train':
        df = pd.read_csv(os.path.join(tfrcd_dir,'info.csv'))
        nrof_samples = df['train_num'][0]
        
    elif phase == 'val':
        df = pd.read_csv(os.path.join(tfrcd_dir,'info.csv'))
        nrof_samples = df['val_num'][0]
      
    file_paths =  os.listdir(os.path.join(tfrcd_dir, phase))
    file_paths =  [os.path.join(tfrcd_dir, phase, file_path)for file_path in file_paths]
    AUTOTUNE   =  tf.data.experimental.AUTOTUNE
    dataset    =  tf.data.TFRecordDataset(file_paths)
    dataset    =  dataset.map(lambda x: parse_image_function(x, params.image_size))
    dataset    =  dataset.batch(params.batch_size).prefetch(AUTOTUNE)
    
    return dataset, nrof_samples


if __name__ == "__main__":

    get_dataset('~/face-data')