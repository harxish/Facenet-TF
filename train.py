import os
import argparse
import datetime
import tensorflow as tf
from progressbar import *

from src.params import Params
from src.model  import face_model
from src.data   import get_dataset
from src.loss   import batch_all_triplet_loss, batch_hard_triplet_loss


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Trainer():
    
    def __init__(self, json_path, data_dir, ckpt_dir, log_dir, restore):
        self.params      = Params(json_path)
        self.model       = face_model(self.params)
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.params.learning_rate,
                                                                          decay_steps=10000, decay_rate=0.96, staircase=True)
        self.optimizer   = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=0.1)
        self.checkpoint  = tf.train.Checkpoint(model=self.model, optimizer=self.model, steps=tf.Variable(0,dtype=tf.int64),
                                               epoch=tf.Variable(0, dtype=tf.int64), loss=tf.Variable(.0, dtype=tf.float32))
        self.ckptmanager = tf.train.CheckpointManager(self.checkpoint, ckpt_dir, 3)
        
        if self.params.triplet_strategy == "batch_all":
            self.loss = batch_all_triplet_loss
            
        elif self.params.triplet_strategy == "batch_hard":
            self.loss = batch_hard_triplet_loss
            
        current_time = datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")
        log_dir += current_time + '/train/'
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
            
        if restore == '1':
            self.checkpoint.restore(self.ckptmanager.latest_checkpoint)
            print(f'Restored from Checkpoint : {self.ckptmanager.latest_checkpoint}\n')
        
        else:
            print('Intializing from scratch\n')
            
        self.dataset, self.nrof_samples = get_dataset(data_dir, self.params)
        
        
    def train(self, epoch):
        widgets = [f'epoch {epoch} :', Percentage(), ' ', Bar('#'), ' ',Timer(), ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, max_value=int(self.nrof_samples//self.params.batch_size)+1).start()
        total_loss = 0

        for _, (images, labels) in pbar(enumerate(self.dataset)):
            loss = self.train_step(images, labels)
            break
            total_loss += loss
            
            with self.train_summary_writer.as_default():
                tf.summary.scalar('step_loss', loss, step=self.checkpoint.steps)
            self.checkpoint.steps.assign_add(1)
        
        with self.train_summary_writer.as_default():
            tf.summary.scalar('batch_loss', total_loss, step=epoch)
        
        self.checkpoint.epoch.assign_add(1)
        if int(self.checkpoint.epoch) % 5 == 0:
            save_path = self.ckptmanager.save()
            print('\nLoss over epoch {}: {}'.format(epoch, total_loss))
            print(f'Saved Checkpoint for step {self.checkpoint.epoch.numpy()} : {save_path}\n')

        
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
    parser.add_argument('--data_dir', default='/root/shared_folder/Harish/Facenet/data/',
                        help="Directory containing the dataset")
    parser.add_argument('--ckpt_dir', default='/root/shared_folder/Harish/Facenet-x/.tf_ckpt/',
                        help="Directory containing the Checkpoints")
    parser.add_argument('--log_dir', default='/root/shared_folder/Harish/Facenet-x/.logs/',
                        help="Directory containing the Logs")
    parser.add_argument('--restore', default='0',
                        help="Restart the model from the previous Checkpoint")
    args = parser.parse_args()
    
    trainer = Trainer(args.params_dir, args.data_dir, args.ckpt_dir, args.log_dir, args.restore)
    
    for i in range(20):
        trainer.train(i)
        break
