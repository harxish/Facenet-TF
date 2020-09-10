import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class face_model(tf.keras.Model):
    
    def __init__(self, params):
        super(face_model, self).__init__()
        img_size = (params.image_size, params.image_size, 3)
        self.base_model = tf.keras.applications.InceptionV3(include_top=False, input_shape=img_size)
        self.base_model.trainable = False
        self.flatten = tf.keras.layers.Flatten()
        self.embedding_layer = tf.keras.layers.Dense(units=params.embedding_size)
        
    def call(self, images):
        x = self.base_model(images)
        x = self.flatten(x)
        x = self.embedding_layer(x)
        return x