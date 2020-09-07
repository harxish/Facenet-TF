import numpy as np
import pandas as pd
import tensorflow as tf


def preprocess_image(image):
    
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.image.random_flip_left_right(image)
    image /= 255.0
    return image


def parse_image_function(example_proto):
    
    image_feature_description = {
        'image_shape': tf.io.FixedLenFeature(shape=[3, ], dtype=tf.int64),
        'id': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        'image_raw': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    }
    
    parsed_example = tf.io.parse_single_example(example_proto, image_feature_description)
    image_string = parsed_example['image_raw']
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = preprocess_image(image)
    label = parsed_example['id']
    return image, label


def get_dataset(tfrcd_dir, params, phase='train'):
    
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
    dataset    =  dataset.map(parse_image_function)
    dataset    =  dataset.batch(params.batch_size).prefetch(AUTOTUNE)
    
    return dataset, nrof_samples