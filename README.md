# Facenet-TF
Facenet implementation using Tensorflow 2.x

### Repo Structure

1. __train.py__      : Creates a trainer object for maintaing checkpoints, logs, train loop and args.
2. __src/data.py__   : Creates tf.data.dataset from tfrecords in the specified directory.
3. __src/loss.py__   : Loss functions and loss utility functions for for triplet loss.
4. __src/model.py__  : Creates model class which gives embeddings from a image.
5. __src/params.py__ : Creates paramters class from a json in the hyperparameters directory.

### Acknowlegments

 - [Face-Recognition-Triplet-Loss-on-Inception-v3](https://github.com/rishiraj95/Face-Recognition-Triplet-Loss-on-Inception-v3)  - Clean and well documented implementation of Triplet loss implementation.
 - [Face Recognition using Tensorflow](https://github.com/davidsandberg/facenet) - The evaluation script on LFW is used and modified to use TF2. Ideas from the documentation and code are also used.

### References

    @article{Schroff_2015,
       title={FaceNet: A unified embedding for face recognition and clustering},
       ISBN={9781467369640},
       url={http://dx.doi.org/10.1109/CVPR.2015.7298682},
       DOI={10.1109/cvpr.2015.7298682},
       journal={2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
       publisher={IEEE},
       author={Schroff, Florian and Kalenichenko, Dmitry and Philbin, James},
       year={2015},
       month={Jun}
    }
