import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss

tf.logging.set_verbosity(tf.logging.INFO)

def adjust_image(images, params):
    images = tf.reshape(images,[-1, params.image_size, params.image_size, 3])
    images = tf.image.resize_images(images, (299, 299, 3))
    return images



def inception_v3_model_fn(features, labels, mode, params):
    
    images = features
    assert images.shape[1:] == [params.image_size, params.image_size, 3], "{}".format(images.shape)    
    
    #MODEL: Download Inception v3 module for transfer learning
    module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")
    
    #Adjust images to module input shape [299,299,3]
    input_layer = adjust_image(images, params)
    out = module(input_layer)
    
    #Compute embeddings with the model
    with tf.variable_scope('model'):
        embeddings=tf.layers.dense(inputs=out,units=params.embedding_size)
    
    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)    
    
    #PREDICT
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'embeddings': embeddings}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    labels = tf.cast(labels, tf.int64)
    
    #Define the triplet loss
    if params.triplet_strategy == "batch_all":
        loss, fraction = batch_all_triplet_loss(labels, embeddings, margin=params.margin,
                                                squared=params.squared)
    elif params.triplet_strategy == "batch_hard":
        loss = batch_hard_triplet_loss(labels, embeddings, margin=params.margin,
                                       squared=params.squared)
    else:
        raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))
        
    # EVALUATE    
    # METRICS for evaluation: Use average over whole dataset
    with tf.variable_scope("metrics"):
        eval_metrics_ops = {"embedding_mean_norm": tf.metrics.mean(embedding_mean_norm)}
        
        if params.triplet_strategy == "batch_all":
            eval_metric_ops['fraction_positive_triplets'] = tf.metrics.mean(fraction)
            
            
    if mode == tf.estimator.Modekeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
    #Summaries for training
    tf.summary.scalar('loss', loss)
    if params.triplet_strategy == "batch_all":
        tf.summary.scalar('fraction_positive_triplets', fraction)

    tf.summary.image('train_image', images, max_outputs=1)
        
    
    #TRAINING ROUTINE
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    global_step = tf.train.get_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)