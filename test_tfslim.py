 
import math
import numpy as np
import time
import os

import tensorflow as tf
# Main slim library
slim = tf.contrib.slim

from datasets import flowers
from datasets import dataset_utils

from nets import inception
from preprocessing import inception_preprocessing

####################################################################################
################################### Dataset ########################################
####################################################################################
# Download dataset (flowers)
url = "http://download.tensorflow.org/data/flowers.tar.gz"
flowers_data_dir = 'mydata/flowers'

if not tf.gfile.Exists(flowers_data_dir):
    tf.gfile.MakeDirs(flowers_data_dir)
    
#dataset_utils.download_and_uncompress_tarball(url, flowers_data_dir)

# Batch loading method, provides images, and labels 
def load_batch(dataset, batch_size=32, height=299, width=299, is_training=False):
    """Loads a single batch of data.
    
    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.
    
    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label = data_provider.get(['image', 'label'])
    
    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)
    
    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, labels = tf.train.batch(
          [image, image_raw, label],
          batch_size=batch_size,
          num_threads=1,
          capacity=2 * batch_size)
    
    return images, images_raw, labels

####################################################################################
################################## ANN  ############################################
####################################################################################

# Download V1
url = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"
checkpoints_dir = 'mydata/checkpoints'

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

#dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)


# Fine-tune model on flower dataset
image_size = inception.inception_v1.default_image_size


def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes=["InceptionV1/Logits", "InceptionV1/AuxLogits"]
    
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
      os.path.join(checkpoints_dir, 'inception_v1.ckpt'),
      variables_to_restore)


train_dir = 'mydata/inception_finetuned/'

def train():
    with tf.Graph().as_default():
        #tf.logging.set_verbosity(tf.logging.INFO) # not showing INFO logs
        
        dataset = flowers.get_split('train', flowers_data_dir)
        images, _, labels = load_batch(dataset, height=image_size, width=image_size)
        
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, _ = inception.inception_v1(images, num_classes=dataset.num_classes, is_training=True)
            
        # Specify the loss function:
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
       # slim.losses.softmax_cross_entropy(logits, one_hot_labels) # name update (see below)
        tf.losses.softmax_cross_entropy(logits, one_hot_labels)
        #total_loss = slim.losses.get_total_loss()  # name update (see below)
        total_loss = tf.losses.get_total_loss()

        # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/Total Loss', total_loss)
    
        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = slim.learning.create_train_op(total_loss, optimizer)
        
        # Run the training:
        final_loss = slim.learning.train(
            train_op,
            logdir=train_dir,
            init_fn=get_init_fn(),
            number_of_steps=100)

        
    
    print('Finished training. Last batch loss %f' % final_loss)

train()

####################################################################################
################################## Metrics #########################################
####################################################################################

def eval():
    # This might take a few minutes.
    image_size = inception.inception_v1.default_image_size

    with tf.Graph().as_default():
        #tf.logging.set_verbosity(tf.logging.INFO)
        
        dataset = flowers.get_split('train', flowers_data_dir)
        images, images_raw, labels = load_batch(dataset, height=image_size, width=image_size)
        
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, _ = inception.inception_v1(images, num_classes=dataset.num_classes, is_training=True)

        predictions = tf.argmax(logits, 1)     
 
        checkpoint_path = tf.train.latest_checkpoint(train_dir)
        init_fn = slim.assign_from_checkpoint_fn(
        checkpoint_path,
        slim.get_variables_to_restore())
        
        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'eval/Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'eval/Recall@5': slim.metrics.streaming_sparse_recall_at_k(logits, labels, 5)
            
        })

        print('Running evaluation Loop...')
        checkpoint_path = tf.train.latest_checkpoint(train_dir)
        metric_values = slim.evaluation.evaluate_once(
            master='',
            checkpoint_path=checkpoint_path,
            logdir=train_dir,
            eval_op=list(names_to_updates.values()),
            final_op=list(names_to_values.values()))

        names_to_values = dict(zip(names_to_values.keys(), metric_values))
        for name in names_to_values:
            print('%s: %f' % (name, names_to_values[name]))
            
eval()           
