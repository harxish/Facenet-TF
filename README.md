# Facenet-TF
Facenet implementation using Tensorflow 2.x

Using Inception - V3 to find the n-dimensional embeddings which could represent a face such that it is distinguishable from faces of other people. Triplet loss is used to **minimise the intra-class variance** and **increase the inter-class vairance**. Three variations of triplet losses are used which can be found at `src/triplet_loss.py`.

- `Triplet Loss` : A straight-forward procedure which finds triplets by iterating through all triplets formed.
- `Hard Triplet Loss` : A procedure which chooses triplet pairs such that distance b/w -ve and anchor is less and the distance b/w anchor and +ve is more.
- `Adaptive Triplet Loss` : The main idea to correct the triplet selection bias, so we try to minize the distribution shift between the batch and the tripet set.

## Code Organization

1. `train.py`           : Creates a trainer object for maintaing checkpoints, logs, train loop and args.
2. `src/data.py`        : Creates tf.data.dataset from tfrecords in the specified directory.
3. `src/model.py`       : Creates model class which gives embeddings from a image.
4. `src/params.py`      : Creates paramters class from a json in the hyperparameters directory.
5. `src/triplet_loss.py`: Loss functions and loss utility functions for for triplet loss.

### How to run

Train from scratch
```
python train.py --params_dir ./hyperparameters/batch_all.json --data_dir ./data/ 
                --log_dir ./.logs/ --ckpt_dir ./.ckpt/ --restore 0
```

Restore from previous checkpoint
```
python train.py --params_dir --restore 1
```

### Acknowlegments

 - [Face-Recognition-Triplet-Loss-on-Inception-v3](https://github.com/rishiraj95/Face-Recognition-Triplet-Loss-on-Inception-v3)  - Clean and well documented implementation of Triplet loss implementation.
 - [Face Recognition using Tensorflow](https://github.com/davidsandberg/facenet) - The evaluation script on LFW is used and modified to use TF2. Ideas from the documentation and code are also used.

### References

```
F. Schroff, D. Kalenichenko and J. Philbin, "FaceNet: A unified embedding for face recognition and clustering,
" 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Boston, MA, 2015, 
pp. 815-823, doi: 10.1109/CVPR.2015.7298682.
```

```
Yu B., Liu T., Gong M., Ding C., Tao D. (2018) Correcting the Triplet Selection Bias for Triplet Loss. 
In: Ferrari V., Hebert M., Sminchisescu C., Weiss Y. (eds) Computer Vision – ECCV 2018. ECCV 2018. 
Lecture Notes in Computer Science, vol 11210. Springer, Cham. https://doi.org/10.1007/978-3-030-01231-1_5
```
