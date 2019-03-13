"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import numpy as np

from PIL import Image
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('label_map_path', 'label_map.pbtxt', 'Path to label map proto')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('mask_type', 'png', 'Mask type, numerical or png')
flags.DEFINE_string('output_path', 'data.record', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# # TO-DO replace this with label map
# def class_text_to_int(row_label):
#     if row_label == 'one':
#         return 1
#     elif row_label == 'two':
#         return 2
#     elif row_label == 'three':
#         return 3
#     elif row_label == 'four':
#         return 4
#     elif row_label == 'five':
#         return 5
#     else:
#         None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf(group, path, label_map_dict, mask_type='png'):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    fileNamePNG = group.filename[:group.filename.find(".")] + ".png"
    mask_path = os.path.join(path, fileNamePNG)
    with tf.gfile.GFile(mask_path, 'rb') as fid:
        encoded_mask_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_mask_png)
    mask = Image.open(encoded_png_io)
    if (mask.format != 'PNG') & (mask_type == 'png'):
        raise ValueError('Mask format not PNG')

    mask_np = np.asarray(mask)

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    masks = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        # classes.append(class_text_to_int(row['class']))
        classes.append(label_map_dict[row['class']])
        # print(mask_pat)
        mask_remapped = (mask_np != 0).astype(np.uint8)
        masks.append(mask_remapped)


    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }
    if mask_type == 'numerical':
      mask_stack = np.stack(masks).astype(np.float32)
      masks_flattened = np.reshape(mask_stack, [-1])
      feature_dict['image/object/mask'] = (
          dataset_util.float_list_feature(masks_flattened.tolist()))
    elif mask_type == 'png':
      encoded_mask_png_list = []
      for mask in masks:
        img = Image.fromarray(mask)
        output = io.BytesIO()
        img.save(output, format='PNG')
        encoded_mask_png_list.append(output.getvalue())
      feature_dict['image/object/mask'] = (
          dataset_util.bytes_list_feature(encoded_mask_png_list))

    tf_ = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return tf_


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    grouped = split(examples, 'filename')
    for group in grouped:
        print(group.filename)
        # print(group[class])
        # print(group.filename + label_map_dict[group[class]])
        tf_example = create_tf(group, path, label_map_dict, FLAGS.mask_type)
        writer.write(tf_example.SerializeToString())

    writer.close()
    # output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    # print('Successfully created the TFRecords: {}'.format(output_path))
    print('Successfully created the TFRecords: {}'.format(FLAGS.output_path))


if __name__ == '__main__':
    tf.app.run()
