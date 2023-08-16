import os
import tensorflow as tf
import xml.etree.ElementTree as ET

def xml_to_tf_example(xml_file, image_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract image filename and path
    filename = root.find('filename').text
    image_path = os.path.join(image_dir, filename)
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()

    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    # Extract bounding box and class label for each object
    xmins, xmaxs, ymins, ymaxs, class_names = [], [], [], [], []
    for obj in root.findall('object'):
        xmins.append(float(obj.find('bndbox/xmin').text) / width)
        xmaxs.append(float(obj.find('bndbox/xmax').text) / width)
        ymins.append(float(obj.find('bndbox/ymin').text) / height)
        ymaxs.append(float(obj.find('bndbox/ymax').text) / height)
        class_names.append(obj.find('name').text.encode('utf8'))

    # Create a tf.train.Example
    feature = {
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=class_names)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def main(xml_dir, image_dir, output_path):
    writer = tf.io.TFRecordWriter(output_path)
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            tf_example = xml_to_tf_example(os.path.join(xml_dir, xml_file), image_dir)
            writer.write(tf_example.SerializeToString())
    writer.close()

# Usage
main('path_to_xml_files', 'path_to_images', 'output.tfrecord')
