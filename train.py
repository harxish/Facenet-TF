import os
import argparse
import tensorflow as tf

from src.data import get_dataset
from src.model import inception_v3_model_fn
from src.params import Params

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--params_dir', default='hyperparameters/batch_hard',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='lfw-dataset',
                    help="Directory containing the dataset")

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = args.params_dir
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.params_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(inception_v3_model_fn, params=params, config=config)

    # Train the model
    tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
    estimator.train(lambda: get_dataset(args.data_dir, params, 'train'))

    # Evaluate the model on the test set
    tf.logging.info("Evaluation on test set.")
    res = estimator.evaluate(lambda: get_dataset(args.data_dir, params, 'test'))
    for key in res:
        print("{}: {}".format(key, res[key]))