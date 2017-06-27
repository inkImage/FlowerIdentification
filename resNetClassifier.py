# Run Tensorboard using: tensorboard --logdir=mydata/resnet_finetuned_plantclef2015_test_3

import math
import numpy as np
import time
import os

import tensorflow as tf

# Main slim library
slim = tf.contrib.slim

from preprocessing import inception_preprocessing
from nets import resnet_v2
from datasets import dataset_utils

import dataVisualisation
import my_functions
import my_resnet_preprocessing

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


####################################################################################
################################### Dataset ########################################
####################################################################################

# Directory of dataset
flowers_data_dir = 'mydata/PlantClefTraining2015' 

num_classes = 1000

# Batch loading method, provides images, and labels 
# Default values: batch size 28 (PlantClef Paper), height and width 224 (default ResNet values)
def load_batch(dataset, batch_size=28, height=224, width=224, is_training=False): # 35.000 iterations batch size = 2* 14 = 28,
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
       
    # Preprocess the image for display purposes.
    image_raw_disp = tf.expand_dims(image_raw, 0)
    image_raw_disp = tf.image.resize_images(image_raw_disp, [height, width])
    image_raw_disp = tf.squeeze(image_raw_disp)
    

    # Batch it up.
    if is_training:
        image= my_resnet_preprocessing.preprocess_image(image_raw, height, width, is_training=False)
        

        #image = tf.stack([image, image_flipped], 0)
        #label = tf.stack([label, label], 0)

        images, images_raw, labels = tf.train.batch(
          [image, image_raw_disp, label],
          batch_size=batch_size,
          num_threads=1,
          capacity=2 * batch_size)
    
        return images, image_raw_disp, labels
    
    
    
    else:        
        image = my_resnet_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training) 

        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=1,
            capacity= 2*batch_size)
        
        return images, image_raw_disp, labels


    
####################################################################################
################################## ANN  ############################################
####################################################################################
# TODO Download the ResNet
url = "http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz" # TODO Choose Model
checkpoints_dir = 'mydata/resnet_checkpoints' # TODO add Directory of checkpoints

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

#dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir) # TODO Download chosen model


image_size = 224 #resnet_v2.default_image_size 

def my_cnn(images, is_training, dropout_rate =0.5):
    """ Creates a neural network and calculates the logits of a given set of images
    
    Args: 
      images: batch of images of which logits need to be calculated
      is_training: boolean, indicates wether softmax should be calculated or not
      
    Returns:
      logits: logits of images
    """
    
    # Create the model, use the default arg scope to configure the batch norm parameters.
    # TODO resnet_v2: change SpatialSqueeze back to True TODO resnet_v2: "return logits, end_points" instead of "return net, end_points" 
    #with slim.arg_scope(resnet_v2.resnet_arg_scope()): 
    #    logits, _ = resnet_v2.resnet_v2_50(images,num_classes, is_training=True) # TODO Choose Model (50, 101, 152, ...) 
    #TODO add no classes (num_classes=dataset.num_classes,) => features before logit layer gets returned => add own layers afterwards
           
    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=0.0002)): 
        logits, _ = resnet_v2.resnet_v2_50(images, is_training=is_training) #TODO
        
        logits = tf.nn.dropout(logits, 0.5) 
        
        # MAXOUT 
        max_out_unit_1 = slim.fully_connected(logits, 512, activation_fn=None, normalizer_fn=None, scope='my_fc_2')
        max_out_unit_2 = slim.fully_connected(logits, 512, activation_fn=None, normalizer_fn=None, scope='my_fc_3')
        max_out_unit_3 = slim.fully_connected(logits, 512, activation_fn=None, normalizer_fn=None, scope='my_fc_4')
        max_out_unit_4 = slim.fully_connected(logits, 512, activation_fn=None, normalizer_fn=None, scope='my_fc_5')
  
        max_out_joined = tf.concat([max_out_unit_1, max_out_unit_2, max_out_unit_3, max_out_unit_4], 1)
        logits = my_functions.max_out(max_out_joined, num_units = 1, axis = 1)
        
        
        #logits = tf.nn.dropout(logits, 0.5) #TODO needed? "Dropout with a ratio of 50% is applieder after the maxout layer and before the classifier" (paper), "Dropout is performed on x, before multiplication by weights" (presentation)
        
        # Fully Connected. 1000 neurons.
        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        logits = slim.fully_connected(logits, 1000, scope = "my_fc_1", activation_fn=None, normalizer_fn=None)
        
        
        if is_training:
            return logits
        elif dropout_rate == 1.0:
            return logits
        else:
            return slim.softmax(logits)
            


def get_init_fn():
    #Returns a function run by the chief worker to warm-start the training.
    checkpoint_exclude_scopes=["resnet_v2_50/logits", "my_fc_1", "my_fc_2", "my_fc_3", "my_fc_4", "my_fc_5", "logits"] # TODO Check for scopes!!!
    
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
      os.path.join(checkpoints_dir, 'resnet_v2_50.ckpt'), # TODO Choose Model
      variables_to_restore)


#train_dir = 'mydata/resnet_finetuned_plantclef2015_test/' # TODO add Directory of finetuned network

def train(split, train_steps, train_dir):
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO) # showing INFO logs
        
        dataset = dataVisualisation.get_split(split, flowers_data_dir) 
        
        images, _, labels = load_batch(dataset, height=image_size, width=image_size, is_training=True)
        
        one_hot_labels = slim.one_hot_encoding(labels, num_classes)

        # Forward pass with non-flipped images
        logits = my_cnn(images, is_training=True)

        tf.losses.softmax_cross_entropy(one_hot_labels, logits)  
    
        total_loss = tf.losses.get_total_loss()

        # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/Total_Loss', total_loss)
        
        # Learning rate decay
        global_step = variables.get_or_create_global_step()
                
        # Piecewise constant from boundaries and interval values.
        boundaries = [tf.constant(100000, dtype= "int64"), tf.constant(200000, dtype= "int64"), tf.constant(300000, dtype= "int64")]
        values = [0.001, 0.0001, 0.00001, 0.000001] # 20.000 steps lr 0.001 -> 0.0005 # 35000 steps lr 0.0005 -> 0.001 # 40.000 steps lr 0.001 -> 0.0005 # 55.000 lr 0.0005 -> 0.0001
        
        #boundaries = [tf.constant(10000, dtype= "int64"), tf.constant(100000, dtype= "int64"), tf.constant(200000, dtype= "int64"), tf.constant(300000, dtype= "int64")]
        #values = [0.001, 0.0005, 0.0001, 0.00001, 0.000001] # 20.000 steps lr 0.001 -> 0.0005 # 35000 steps lr 0.0005 -> 0.001 # 40.000 steps lr 0.001 -> 0.0005 # 55.000 lr 0.0005 -> 0.0001
        #boundaries = [tf.constant(100000, dtype= "int64"), tf.constant(200000, dtype= "int64"), tf.constant(300000, dtype= "int64"), tf.constant(400000, dtype= "int64")]
        #values = [0.001, 0.0001, 0.00001, 0.000001, 0.000001] # 20.000 steps lr 0.001 -> 0.0005 # 35000 steps lr 0.0005 -> 0.001 # 40.000 steps lr 0.001 -> 0.0005 # 55.000 lr 0.0005 -> 0.0001
        my_learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        # Specify the optimizer and create the train op:
        optimizer = tf.train.MomentumOptimizer(learning_rate=my_learning_rate, momentum = 0.9) 
        train_op = slim.learning.create_train_op(total_loss, optimizer)
        
        #TODO Choose init_fn or get_init_fn()
        #checkpoint_path = tf.train.latest_checkpoint(train_dir)
        #init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_variables_to_restore())
        
        # Run the training:
        final_loss = slim.learning.train(
            train_op,
            logdir=train_dir,
            log_every_n_steps=1,
            init_fn= get_init_fn(),
            number_of_steps=train_steps,
            global_step = global_step)       

    
    print('Finished training. Last batch loss %f' % final_loss)

#train('train_set2', 280, train_dir = 'mydata/test123/')

####################################################################################
################################## Metrics #########################################
####################################################################################


def eval(split, train_dir):
    # This might take a few minutes.
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        
        dataset = dataVisualisation.get_split(split, flowers_data_dir) 
        images, _, labels = load_batch(dataset, batch_size = 100, height=image_size, width=image_size) # TODO load_batch really necessary? Processing all! At least adjust batch_size
        
        logits = my_cnn(images, is_training=False)
        predictions = tf.argmax(logits, 1)  
        
                     
        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'eval/Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'eval/Recall@5': slim.metrics.streaming_sparse_recall_at_k(logits, labels, 5)#,
            #'eval/Recall@1000': slim.metrics.streaming_sparse_recall_at_k(logits, labels, 1000) # should be 1
        })
        
        op = tf.summary.scalar('top5percentError', names_to_values['eval/Recall@5'], collections=[])  
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
        op = tf.summary.scalar('top1percentError', names_to_values['eval/Accuracy'], collections=[])  
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

 #       for name, value in names_to_values.iteritems():
 #           summary_name = 'eval/%s' % name
 #           op = tf.summary.scalar(summary_name, value, collections=[])
 #           op = tf.Print(op, [value], summary_name)
 #           tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        print('Running evaluation Loop...')
        checkpoint_path = tf.train.latest_checkpoint(train_dir)
        
        #print_tensors_in_checkpoint_file(checkpoint_path, tensor_name ='my_fc_1/weights', all_tensors=False)
        #print_tensors_in_checkpoint_file(checkpoint_path, tensor_name ='resnet_v2_50/block1/unit_1/bottleneck_v2/conv3/weights', all_tensors=False)
        
        metric_values = slim.evaluation.evaluate_once(
            master='',
            checkpoint_path=checkpoint_path,
            logdir=train_dir,
            num_evals = 10,  # TODO Adjust number of evaluations
            summary_op=tf.summary.merge_all(),
            eval_op=list(names_to_updates.values()),
            final_op=list(names_to_values.values()))
        
        print("Evaluation 'eval' started.")
        names_to_values = dict(zip(names_to_values.keys(), metric_values))
        for name in names_to_values:
            print('%s: %f' % (name, names_to_values[name]))
            
#eval('train_set2', 'mydata/test123/')
          
# Batch-size = 1, num_evals = 37735
#eval/Recall@5: 0.808242
#eval/Accuracy: 0.615291


# Batch-size = 100, num_evals = 378
#eval/Recall@5: 0.809524
#eval/Accuracy: 0.615265

  
#eval('train_set3', train_dir = 'mydata/resnet_finetuned_plantclef2015_test_3/')   

"""
with tf.Session():
  print(tf.shape(images).eval())
  print(tf.shape(images2).eval())
  print(tf.shape(images3).eval())"""