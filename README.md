# Facenet-TF
Facenet implementation using Tensorflow 2.x

### Repo Structure

1. __train.py__      : Creates a trainer object for maintaing checkpoints, logs, train loop and args.
2. __src/data.py__   : Creates tf.data.dataset from tfrecords in the specified directory.
3. __src/loss.py__   : Loss functions and loss utility functions for for triplet loss.
4. __src/model.py__  : Creates model class which gives embeddings from a image.
5. __src/params.py__ : Creates paramters class from a json in the hyperparameters directory.

### References
<a id="arXiv:1503.03832">[1]</a> 
Schroff, Florian, Dmitry Kalenichenko, and James Philbin. 
“FaceNet: A Unified Embedding for Face Recognition and Clustering.” 
2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2015): n. pag. Crossref. Web.
