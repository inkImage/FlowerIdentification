#/home/lolek/.local/lib/python3.5/site-packages/tensorflow/contrib/training/python/training/
import math
import numpy as np
import time
import os
import matplotlib.pyplot as plt

import tensorflow as tf

# Main slim library
slim = tf.contrib.slim


import resNetClassifier
import dataVisualisation
import datasets.dataset_utils as dataset_utils
import my_resnet_preprocessing
import kernel_visualization

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


#log_dir_network1 = 'mydata/resnet_finetuned_plantclef2015_test_3'
log_dir_network1 = 'mydata/resnet_finetuned_plantclef2015_1'
log_dir_network2 = 'mydata/resnet_finetuned_plantclef2015_2'
log_dir_network3 = 'mydata/resnet_finetuned_plantclef2015_3'

#tensorboard --logdir=mydata/resnet_finetuned_plantclef2015_test_3
#tensorboard --logdir=mydata/resnet_finetuned_plantclef2015_1
#tensorboard --logdir=mydata/resnet_finetuned_plantclef2015_2
#tensorboard --logdir=mydata/resnet_finetuned_plantclef2015_3

#Training the first network on folds 1 and 2, evaluated on fold 3 (150.000 iterations, 75k on each set)
def train_network1():
    for x in range(59,60,2):
        resNetClassifier.train('train_set1', 2500*x, log_dir_network1)
        resNetClassifier.train('train_set2', 2500*x+2500, log_dir_network1)
        resNetClassifier.eval('train_set3', log_dir_network1)

#Training the second network on folds 2 and 3, evaluated on fold 1 (150.000 iterations, 75k on each set)
def train_network2():
    for x in range(59,60,2):
        resNetClassifier.train('train_set2', 2500*x, log_dir_network2)
        resNetClassifier.train('train_set3', 2500*x+2500, log_dir_network2)
        resNetClassifier.eval('train_set1', log_dir_network2)    

#Training the third network on folds 1 and 3, evaluated on fold 2 (150.000 iterations, 75k on each set)
def train_network3():    
    for x in range(1,60,2):
        resNetClassifier.train('train_set3', 2500*x, log_dir_network3)
        resNetClassifier.train('train_set1', 2500*x+2500, log_dir_network3)
        resNetClassifier.eval('train_set2', log_dir_network3)        
        
def numpy_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()        

# Voting of the three networks on one image.
# Voting is done by returning the class_id with the overall maximum "likelyhood"
# Saves the data into the given file (Default: 'predictions.txt')
#https://github.com/thomaspark-pkj/resnet-ensemble/blob/master/eval_image_classifier_ensemble.py
def final_evaluation(label_dir, dataset_dir, filename="predictions.txt"):
    number_images =100 #8000
    correct = 0# TODO REMOVE
    
    #[ 2.49181414 -0.33259141 -1.27059281 -1.52844727  2.57813239 -1.20948148
    checkpoint_paths = [ "mydata/resnet_finetuned_plantclef2015_1/model.ckpt-145552", "mydata/resnet_finetuned_plantclef2015_1/model.ckpt-145552", "mydata/resnet_finetuned_plantclef2015_1/model.ckpt-145552"]
    
    output_list = []
    labels_list = []
    
    for index in range(len(checkpoint_paths)):
        with tf.Graph().as_default():
            dataset = dataVisualisation.get_split('train_set3', dataset_dir)
            
            data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, 
                                                                           shuffle=False,
                                                                           common_queue_capacity=8000,
                                                                           common_queue_min=0)
            
            image_raw, label = data_provider.get(['image', 'label'])
            """                 
            imaget = my_resnet_preprocessing.preprocess_image(image_raw, 224, 224, is_training=False)
            image,augmented_image1,augmented_image2, augmented_image3, augmented_image4,augmented_image5, labels = tf.train.batch([imaget,imaget,imaget,imaget,imaget,imaget, label],  batch_size=1,
                                            num_threads=1,
                                            capacity=12 * 1)
            """
            
            image, augmented_image1, augmented_image2, augmented_image3, augmented_image4, augmented_image5 = my_resnet_preprocessing.preprocess_for_final_run(image_raw, 224, 224) 
            
            image,augmented_image1,augmented_image2, augmented_image3, augmented_image4,augmented_image5, labels = tf.train.batch([image, augmented_image1, augmented_image2, augmented_image3, augmented_image4, augmented_image5, label], 
                                            batch_size=1,
                                            num_threads=1,
                                            capacity=2 * 1)
            
            
            
            
            logits1 = resNetClassifier.my_cnn(image, is_training = False, dropout_rate =1)
            logits2 = resNetClassifier.my_cnn(augmented_image1, is_training = False, dropout_rate =1)
            logits3 = resNetClassifier.my_cnn(augmented_image2, is_training = False, dropout_rate =1)
            logits4 = resNetClassifier.my_cnn(augmented_image3, is_training = False, dropout_rate =1)
            logits5 = resNetClassifier.my_cnn(augmented_image4, is_training = False, dropout_rate =1)            
            logits6 = resNetClassifier.my_cnn(augmented_image5, is_training = False, dropout_rate =1)

            total_output = np.empty([number_images * 1, dataset.num_classes])
            total_labels = np.empty([number_images * 1], dtype=np.int32)
            offset = 0
            
            #init_fn = slim.assign_from_checkpoint_fn(checkpoint_paths[index], slim.get_model_variables())

            with tf.Session() as sess:
                coord = tf.train.Coordinator()
                saver = tf.train.Saver()
                
                saver.restore(sess, checkpoint_paths[index])
                #init_fn(sess)
                merged = tf.summary.merge_all()
                test_writer = tf.summary.FileWriter('/home/lolek/FlowerIdentificationM/models/slim/mydata/test/' + '/train')
                visualize_writer = tf.summary.FileWriter('/home/lolek/FlowerIdentificationM/models/slim/mydata/test/' + '/visualize')
                

                
                
                
                #for v in tf.trainable_variables():
                #    print(v.name)
                #print_tensors_in_checkpoint_file(checkpoint_paths[index], tensor_name =None, all_tensors=False)
                #var = [v for v in tf.trainable_variables() if v.name == "my_fc_1/weights:0"]
                #print("my_fc_1/weights \n", sess.run(var))
                #print_tensors_in_checkpoint_file(checkpoint_paths[index], tensor_name ='my_fc_1/weights', all_tensors=False)
                #var2 = [v for v in tf.trainable_variables() if v.name == "resnet_v2_50/block1/unit_1/bottleneck_v2/conv3/weights:0"]
                #print("resnet_v2_50/block1/unit_1/bottleneck_v2/conv3/weights \n", sess.run(var2))
                #print_tensors_in_checkpoint_file(checkpoint_paths[index], tensor_name ='resnet_v2_50/block1/unit_1/bottleneck_v2/conv3/weights', all_tensors=False)
                
                # Visualize Kernels
                tf.get_variable_scope().reuse_variables()
                weights = tf.get_variable("resnet_v2_50/conv1/weights")
                grid = kernel_visualization.put_kernels_on_grid (weights)
                sum1 = tf.summary.image('conv1/kernels', grid, max_outputs=1)
                _, summary1 = sess.run([merged, sum1])
                visualize_writer.add_summary(summary1,2)
                    
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                for i in range(number_images):
                    print('step: %d/%d' % (i+1, number_images))
                    
                    """
                    image_t1, image_t2 = sess.run([image, augmented_image1])
                    plt.figure()
                    plt.imshow(image_t1[0, :, :, :].astype(np.uint8))
                    plt.title('title')
                    plt.axis('off')
                    plt.show()
                    
                    plt.figure()
                    plt.imshow(image_t2[0, :, :, :].astype(np.uint8))
                    plt.title('title')
                    plt.axis('off')
                    plt.show()
                
                    """
                    sum1 = tf.summary.image('final_eval_whole_image1', image)
                    sum2 = tf.summary.image('final_eval_image_center1', augmented_image1)
                    sum3 = tf.summary.image('final_eval_top_left1', augmented_image2)
                    sum4 = tf.summary.image('final_eval_bottom_left1', augmented_image3)
                    sum5 = tf.summary.image('final_eval_top_right1', augmented_image4)
                    sum6 = tf.summary.image('final_eval_bottom_right1', augmented_image5)
                    _, summary1, summary2,summary3, summary4,summary5, summary6 = sess.run([merged, sum1, sum2,sum3, sum4,sum5,sum6])
                    test_writer.add_summary(summary1,1)
                    test_writer.add_summary(summary2,1)
                    test_writer.add_summary(summary3,1)
                    test_writer.add_summary(summary4,1)
                    test_writer.add_summary(summary5,1)
                    test_writer.add_summary(summary6,1)

                    
                    
                    logit1, logit2, logit3, logit4,logit5, logit6, media_id = sess.run([logits1, logits2, logits3, logits4, logits5, logits6, labels])
                    print(media_id, " ", np.argmax(logit1[0])," ",np.argmax(logit2[0])," ",np.argmax(logit3[0]), " ",np.argmax(logit4[0]), " ",np.argmax(logit5[0]), " ",np.argmax(logit6[0])   )
                    #print(np.amax(logit1[0]), " ",np.amax(logit2[0]), " ",np.amax(logit3[0]), " ",np.amax(logit4[0]), " ",np.amax(logit5[0]), " ",np.amax(logit6[0]))
                    #print(len(logit1[0]))
                    
                    media_id = media_id[0]
                    
                    logits = tuple(max(i, j) for i, j in zip(logit1[0], logit2[0]))
                    logits = tuple(max(i, j)  for i, j in zip(logits, logit3[0]))
                    logits = tuple(max(i, j)  for i, j in zip(logits, logit4[0]))
                    logits = tuple(max(i, j)  for i, j in zip(logits, logit5[0]))
                    logits = tuple(max(i, j)  for i, j in zip(logits, logit6[0]))
                    
                    """
                    logits = tuple(i + j for i, j in zip(logit1[0], logit2[0]))
                    logits = tuple(i + j for i, j in zip(logits, logit3[0]))
                    logits = tuple(i + j for i, j in zip(logits, logit4[0]))
                    logits = tuple(i + j for i, j in zip(logits, logit5[0]))
                    logits = tuple(i + j for i, j in zip(logits, logit6[0]))
                    logits = [x / 6 for x in logits] 
                    """
                    
                    
                    logits = numpy_softmax(logits)
                    #print(np.argmax(logits))
                    #print(len(logits))
                    #first_prediction = np.argmax(logits)
                    #print(first_prediction," ", media_id)
                    #print(np.argmax(logits))
                    
                    #logits = logit1
                    
                    #print(np.amax(logits))
                    #pred = [np.argmax(logit1[0]),np.argmax(logit2[0]),np.argmax(logit3[0]), np.argmax(logit4[0]), np.argmax(logit5[0]), np.argmax(logit6[0] )]

                    if media_id == np.argmax(logits):        # TODO REMOVE
                        correct = correct + 1   # TODO REMOVE
                    #print(correct)              # TODO REMOVE   
                    #print(media_id, " ", np.argmax(logit1)," ",np.argmax(logit2)," ",np.argmax(logit3), " ",np.argmax(logit4), " ",np.argmax(logit5), " ",np.argmax(logit6)   ) 
                    
                    total_output[offset:offset + 1] = logits
                    total_labels[offset:offset + 1] = media_id
                    offset += 1
                coord.request_stop()
                coord.join(threads)

            output_list.append(total_output)
            labels_list.append(total_labels)
            
        print(correct)                             # TODO REMOVE
       
    for i in range(len(output_list)):
        logits = tf.cast(tf.constant(output_list[i]), dtype=tf.float32)
        predictions = tf.nn.softmax(logits)
        labels = tf.constant(labels_list[i])
        top1_op = tf.nn.in_top_k(predictions, labels, 1)
        top5_op = tf.nn.in_top_k(predictions, labels, 5)

        with tf.Session() as sess:
            top1, top5 = sess.run([top1_op, top5_op])

        print('Top 1 accuracy: %f' % (np.sum(top1) / float(number_images)))
        print('Top 5 accuracy: %f' % (np.sum(top5) / float(number_images)))   
       
    for i in range(number_images):
        image_id = labels_list[0][i]
        
        prediction1 = np.amax(output_list[0][i])
        
        prediction2 = np.amax(output_list[1][i])
        prediction3 = np.amax(output_list[2][i])
        
        print(prediction1, " ",prediction2, " ", prediction3)
        
        # Find best class with highest prediction (softmax) score
        if prediction1 > prediction2:
            if prediction1 > prediction3:
                prediction = np.argmax(output_list[0][i])
                probability = prediction1
        
            else:
                prediction = np.argmax(output_list[2][i])
                probability = prediction3
        else:
            if prediction2> prediction3:
                prediction = np.argmax(output_list[1][i])
                probability = prediction2
            else:
                prediction = np.argmax(output_list[2][i])
                probability = prediction3
        

        class_id = dataset_utils.read_label_file(label_dir)[prediction]
        
        image_id = dataset_utils.read_label_file(label_dir)[image_id] # TODO REMOVE!!!
        
            
        print('<%s;%s;%f>\n' % (image_id, class_id, probability))
        # Save the predictions
        labels_filename = os.path.join(label_dir, filename)
        with tf.gfile.Open(labels_filename, 'a') as f:
            f.write('<%s;%s;%f>\n' % (image_id, class_id, probability)) # <ImageId;ClassId;Probability>
        

#train_network1()
train_network2()
train_network3()
#resNetClassifier.train('train_set2', 145550, log_dir_network1)


#resNetClassifier.eval('train_set3', "mydata/resnet_finetuned_plantclef2015_test_3")

#resNetClassifier.eval('train_set1', 'mydata/resnet_finetuned_plantclef2015_test_3')
#resNetClassifier.eval('train_set2', 'mydata/resnet_finetuned_plantclef2015_test_3')
#resNetClassifier.eval('train_set3', 'mydata/resnet_finetuned_plantclef2015_test_3')

#resNetClassifier.eval('train_set1', log_dir_network1)
#resNetClassifier.eval('train_set2', log_dir_network1)
#resNetClassifier.eval('train_set3', log_dir_network1)

#resNetClassifier.eval('train_set1', log_dir_network2)
#resNetClassifier.eval('train_set2', log_dir_network2)
#resNetClassifier.eval('train_set3', log_dir_network2)

#final_evaluation(label_dir = 'mydata/PlantClefTraining2015' , dataset_dir='mydata/PlantClefTraining2015' )

#resNetClassifier.train('train_set1', 150500,  'mydata/resnet_finetuned_plantclef2015_test_3')
#resNetClassifier.eval('train_set1', 'mydata/resnet_finetuned_plantclef2015_test_3')