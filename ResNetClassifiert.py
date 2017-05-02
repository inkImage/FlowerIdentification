import math
import numpy as np
import time
import os

import tensorflow as tf
# Main slim library
slim = tf.contrib.slim

from datasets import flowers # TODO only using flowers dataset to make sure model is working

from preprocessing import inception_preprocessing
from nets import resnet_v2
from datasets import dataset_utils




####################################################################################
################################### Dataset ########################################
####################################################################################

# Directory of dataset
flowers_data_dir = 'mydata/flowers' # TODO add Directory of Dataset

# Batch loading method, provides images, and labels 
# Default values: batch size 28 (PlantClef Paper), height and width 224 (default ResNet values)
def load_batch(dataset, batch_size=28, height=224, width=224, is_training=False): 
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
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training) # TODO Own Preprocesssing needed, depended on is_training => by default uses inception_preprocessing
    
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
# TODO Download the ResNet
url = "http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz" # TODO Choose Model
checkpoints_dir = 'mydata/resnet_checkpoints' # TODO add Directory of checkpoints

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

#dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir) 


image_size = resnet_v2.resnet_v2_50.default_image_size


# Init Function for pretrained networks
def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes=["resnet_v2_50/logits"] # TODO Check for scopes!
    
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
      os.path.join(checkpoints_dir, 'resnet_v2_50.ckpt'),
      variables_to_restore)


train_dir = 'mydata/resnet_finetuned/' # TODO add Directory of finetuned network

def train():
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO) # not showing INFO logs
        
        dataset = flowers.get_split('train', flowers_data_dir) # TODO Add directory of dataset, Check for format of dataset!
        images, _, labels = load_batch(dataset, height=image_size, width=image_size)
        
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, _ = resnet_v2.resnet_v2_50(images, num_classes=dataset.num_classes, is_training=True) # TODO Choose Model (50, 101, 152, ...)
            
            
        # Specify the loss function:
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
       
        tf.losses.softmax_cross_entropy(logits, one_hot_labels)
       
        total_loss = tf.losses.get_total_loss()

        # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/Total Loss', total_loss)
        
        #TODO Testing learning rate decay
        #starter_learning_rate = 0.01
        #learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
        #                                   100000, 0.96, staircase=True)
    
        # Specify the optimizer and create the train op:
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum = 0.9) # TODO add lowering of learning_rate, add weight decay (resnet_utils: weight_decay -> 0.0002)
        train_op = slim.learning.create_train_op(total_loss, optimizer)
        
        # Run the training:
        final_loss = slim.learning.train(
            train_op,
            logdir=train_dir,
            init_fn=get_init_fn(),
            number_of_steps=1030)       

    
    print('Finished training. Last batch loss %f' % final_loss)

train()

####################################################################################
################################## Metrics #########################################
####################################################################################

def eval():
    # This might take a few minutes.
    with tf.Graph().as_default():
        #tf.logging.set_verbosity(tf.logging.INFO)
        
        dataset = flowers.get_split('train', flowers_data_dir) # TODO Add direcotry of dataset, Check for format of dataset!
        images, images_raw, labels = load_batch(dataset, height=image_size, width=image_size) # TODO load_batch really necessary? Processing all!
        
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, _ = resnet_v2.resnet_v2_50(images, num_classes=dataset.num_classes, is_training=True) # TODO Choose Model (50, 101, 152, ...)

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