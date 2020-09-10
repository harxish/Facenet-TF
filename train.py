import os
import argparse
import tensorflow as tf
from tqdm import tqdm

from src.params import Params
from src.data import get_dataset
from src.model import face_model
from src.loss import batch_all_triplet_loss
from src.loss import batch_hard_triplet_loss


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Trainer():
    
    def __init__(self, json_path, data_dir):
        self.params      = Params(json_path)
        self.model       = face_model(self.params)
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.params.learning_rate,
                                                                          decay_steps=10000, decay_rate=0.96, staircase=True)
        self.optimizer   = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=0.1)
        
        if self.params.triplet_strategy == "batch_all":
            self.loss = batch_all_triplet_loss
            
        elif self.params.triplet_strategy == "batch_hard":
            self.loss = batch_hard_triplet_loss
            
        self.dataset, self.nrof_samples = get_dataset(data_dir, self.params)
        
        
    def train(self, epoch):
#         widgets = ['train :', Percentage(), ' ', Bar('#'), ' ',Timer(), ' ', ETA(), ' ']
#         pbar = ProgressBar(widgets=widgets, max_value=int(self.nrof_samples//self.params.batch_size)+1).start()
        total_loss = 0

        for _, (images, labels) in tqdm(enumerate(self.dataset)):
            loss = self.train_step(images, labels)
            print(loss)
            total_loss += loss
            
#         pbar.finish()
        
        print('\nLoss over epoch {}: {}\n'.format(epoch, total_loss))

        
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            embeddings = self.model(images)
            embeddings = tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10)
            loss = self.loss(labels, embeddings, self.params.margin, self.params.squared)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_dir', default='hyperparameters/batch_all.json',
                        help="Experiment directory containing params.json")
    parser.add_argument('--data_dir', default='/root/shared_folder/Amaan/face/FaceNet-and-FaceLoss-collections-tensorflow2.0/data',
                        help="Directory containing the dataset")
    args = parser.parse_args()
    
    trainer = Trainer(args.params_dir, args.data_dir)
    trainer.train(1)
