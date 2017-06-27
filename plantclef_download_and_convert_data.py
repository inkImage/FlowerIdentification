from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

import xml.etree.ElementTree as ET

# URLs to download datasets
url_train_2014 =  "http://otmedia.lirmm.fr/LifeCLEF/LifeCLEF2014/TrainPackages/PlantCLEF2014trainAllInOne.tar"
url_test_2014_1 = "http://otmedia.lirmm.fr/LifeCLEF/LifeCLEF2014/TestPackages/PlantCLEF2014test.tar"
url_test_2014_2 = "http://otmedia.lirmm.fr/LifeCLEF/LifeCLEF2014/TestPackages/PlantCLEF2014LastPackage.tar.gz" 
url_train_2016 =  "http://otmedia.lirmm.fr/LifeCLEF/PlantCLEF2015/Packages/TrainingPackage/PlantCLEF2015TrainingData.tar.gz"
url_test_2016_1 = "http://otmedia.lirmm.fr/LifeCLEF/PlantCLEF2016/PlantCLEF2016Test.tar.gz" 
url_train_2016_2 = "http://otmedia.lirmm.fr/LifeCLEF/PlantCLEF2015/Packages/TestPackage/PlantCLEF2015TestDataWithAnnotations.tar.gz"

# The number of images in the validation set.
_NUM_TRAIN_SET1 = 37735
_NUM_TRAIN_SET2 = 75470
_NUM_TRAIN_SET3 = 113205

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 38



class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir, is_train_set=True):
    """Returns a list of filenames and inferred class names.
    Args:
        dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.
    Returns:
        file_names: A list of image file paths
        image_to_id: A dictionary mapping a filename to a class_id
        id_to_class_name: A dictionary mapping a class_id to a string of family, species, and genus
    """  
    if is_train_set:
        flower_root = os.path.join(dataset_dir, 'train')
        image_to_id = {}
        id_to_class_name = {}
        file_names =  []
        class_names = []
        
        for filename in os.listdir(flower_root):
            path = os.path.join(flower_root, filename)
        
            if path.endswith(".xml"):
                class_id, family, species, genus, media_id = _get_class_name_from_xml(path)
                image_to_id[os.path.join(flower_root, ("%s%s" % (media_id, ".jpg")))]= class_id              
                id_to_class_name[class_id] = ("%s %s %s" % (family, species, genus))
            if path.endswith(".jpg"):
                file_names.append(filename)  
        
        for class_id in id_to_class_name:
            class_names.append(class_id)
            
        return file_names, image_to_id, id_to_class_name, sorted(class_names)   
    
    else:
        flower_root = os.path.join(dataset_dir, 'train')
        file_names =  []
        image_to_media_id = {}
        
        for filename in os.listdir(flower_root):
            path = os.path.join(flower_root, filename)

            if path.endswith(".jpg"):
                file_names.append(filename)
                image_to_media_id[filename] = filename.split('.png')[0]
        
        return file_names, image_to_media_id





def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'flowers_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)





#/home/lolek/FlowerIdentificationM/models/slim/mydata/PlantCLEF2014LastPackage/data/testwithgroundtruthxml/0.xml




def _get_class_name_from_xml(file):
  """Read a XML file and returns the information 
  Args:
    file: path to the xml file
      
  Returns:
     class_id: Id of Plant 
     family: Family of plant
      species: Species of plant
      genus: Genus of plant
      media_id: number of picture/file
  """
  tree = ET.parse(file)
  root = tree.getroot()
  class_id = int(tree.find("ClassId").text) # int(root[5].text)
  family = tree.find("Family").text # root[6].text
  species = tree.find("Species").text #root[7].text
  genus =  tree.find("Genus").text#root[8].text
  media_id = tree.find("MediaId").text# root[2].text
    
  return class_id, family, species, genus, media_id



def _convert_dataset(split_name, filenames, class_names_to_ids, image_to_id, dataset_dir, is_train_set = True):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers). 
    dataset_dir: The directory where the converted datasets are stored.
  """
  if is_train_set:
    assert split_name in ['train_set1', 'train_set2', 'train_set3']

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id)

            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
            for i in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i+1, len(filenames), shard_id))
                sys.stdout.flush()

                # Read the filename:
                image_data = tf.gfile.FastGFile(("%s/train/%s" % (dataset_dir, filenames[i])), 'r').read()
                height, width = image_reader.read_image_dims(sess, image_data)

                
                class_of_image = image_to_id[os.path.join(dataset_dir, 'train', filenames[i])]
                class_id = class_names_to_ids[class_of_image] 

                example = dataset_utils.image_to_tfexample(
                    image_data, b'jpg', height, width, class_id)
                tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()
  
  else:#TODO 
    assert split_name in ['test_set']
    num_per_shard = int(math.ceil(len(filenames) / 8.0))
    
    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

        for shard_id in range(8.0):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id)

            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
            for i in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i+1, len(filenames), shard_id))
                sys.stdout.flush()

                # Read the filename:
                image_data = tf.gfile.FastGFile(("%s/train/%s" % (dataset_dir, filenames[i])), 'r').read()
                height, width = image_reader.read_image_dims(sess, image_data)

                
                media_id = image_to_id[filenames[i]]

                example = dataset_utils.image_to_tfexample(
                    image_data, b'jpg', height, width, media_id)
                tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()
      
      

 
def _clean_up_temporary_files(_DATA_URL, dataset_dir):
  """Removes temporary files used to create the dataset.
    Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  #tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, 'train')
  tf.gfile.DeleteRecursively(tmp_dir)


# Checking if a dataset split exists
def _dataset_exists(dataset_dir):
  for split_name in ['train_set1', 'train_set2', 'train_set3']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True



def run(dataset_dir, dataset_type):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
    dataset_type: One of "train", "test", "validation", which set should be downloaded and decompressed
  """
  if(dataset_type == "train2014"):
    _DATA_URL = url_train_2014
  elif(dataset_type == "test2014"):
    _DATA_URL = url_test_2014_1
  elif(dataset_type == "validation2014"):
    _DATA_URL = url_test_2014_2
  elif(dataset_type == "train2016"):
    _DATA_URL = url_train_2016
  elif(dataset_type == "test2016"):
    _DATA_URL = url_test_2016_1
  elif(dataset_type == "train2016_2"):
    _DATA_URL = url_train_2016_2
  else:
    print("There exists no such dataset %s, please choose one of 'train2014', 'test2014', 'validation2014','train2016', 'test2016', 'train2016_2'." % dataset_type)
    
    
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  # TODO Downloading
  #dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
  if dataset_type != "test2016":  
    print("Getting filenames and class_id's ")  
    # Extract and save pictures and class names in dictionary
    photo_filenames, image_to_id, id_to_class_name, class_names = _get_filenames_and_classes(dataset_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    print("All filenames and class_id's found")  
        
    # Divide into train and test: TODO division necessary if packages (urls) contain splits?
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    training1_filenames = photo_filenames[:_NUM_TRAIN_SET1]
    training2_filenames = photo_filenames[_NUM_TRAIN_SET1:_NUM_TRAIN_SET2]
    training3_filenames = photo_filenames[_NUM_TRAIN_SET2:]

    # First, convert the training and validation sets. 
    _convert_dataset('train_set1', training1_filenames, class_names_to_ids, image_to_id, 
                        dataset_dir)
    _convert_dataset('train_set2', training2_filenames, class_names_to_ids, image_to_id, 
                        dataset_dir)
    _convert_dataset('train_set3', training3_filenames, class_names_to_ids, image_to_id,
                        dataset_dir) 

        # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    # writes "labels.txt" with id (1:1000) -> class_id (e.g. 17266)
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)  
    # writes "labels2.txt" with class_id (e.g. 17266) -> class_name (e.g. rose ...)
    dataset_utils.write_label_file(id_to_class_name, dataset_dir, filename="labels2.txt")

    _clean_up_temporary_files(_DATA_URL, dataset_dir)
    print('\nFinished converting the Flowers dataset!')
  else:
      photo_filenames, image_to_media_id = _get_filenames_and_classes(dataset_dir, is_train_set=False)
      
      _convert_dataset('test_set', photo_filenames, None, image_to_media_id, dataset_dir, is_train_set=False) 
     
      print('\nFinished converting the Flowers test dataset!')

run("mydata/PlantClefTraining2015", "train2016")
