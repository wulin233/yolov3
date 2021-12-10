import hashlib
import io
import os

import numpy as np
import PIL.Image as pil
from PIL import Image
import tensorflow as tf

import feature_parse
from IoU import iou
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../data/kitti/',
                    help='kitti数据集的位置')
parser.add_argument('--output_path', type=str, default='../data/kitti_tfrecords/',
                    help='TFRecord文件的输出位置')
parser.add_argument('--classes_to_use', default='car,van,truck,pedestrian,cyclist,tram', help='KITTI中需要检测的类别')
parser.add_argument('--validation_set_size', type=int, default=500,
                    help='验证集数据集使用大小')


def prepare_example(image_path, annotations):
    """
    对一个图片的Annotations转换成tf.Example proto.
    :param image_path:
    :param annotations:
    :return:
    """
    # 1、读取图片内容，转换成数组格式
    with open(image_path, 'rb') as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    # print('----', encoded_png_io)
    image = pil.open(encoded_png_io)
    # print('====', image)
    # print('====', type(image))
    image = np.asarray(image)
    # print('=-=-=image', image)

    # 2、构造协议中需要的字典键的值
    # sha256加密结果
    key = hashlib.sha256(encoded_png).hexdigest()
    # print('shape----', image.shape)
    # 进行坐标处理
    width = int(image.shape[1])
    # print("===width", width)
    height = int(image.shape[0])
    # print('===height', height)
    # 存储极坐标归一化格式
    xmin_norm = annotations['2d_bbox_left'] / float(width)
    ymin_norm = annotations['2d_bbox_top'] / float(height)
    xmax_norm = annotations['2d_bbox_right'] / float(width)
    ymax_norm = annotations['2d_bbox_bottom'] / float(height)

    # 其他信息，难度以及字符串类别
    difficult_obj = [0] * len(xmin_norm)
    classes_text = [x.encode('utf8') for x in annotations['type']]

    # 3、构造协议example
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': feature_parse.int64_feature(height),
        'image/width': feature_parse.int64_feature(width),
        'image/filename': feature_parse.bytes_feature(image_path.encode('utf8')),
        'image/source_id': feature_parse.bytes_feature(image_path.encode('utf8')),
        'image/key/sha256': feature_parse.bytes_feature(key.encode('utf8')),
        'image/encoded': feature_parse.bytes_feature(encoded_png),
        'image/format': feature_parse.bytes_feature('png'.encode('utf8')),
        'image/object/bbox/xmin': feature_parse.float_list_feature(xmin_norm),
        'image/object/bbox/xmax': feature_parse.float_list_feature(xmax_norm),
        'image/object/bbox/ymin': feature_parse.float_list_feature(ymin_norm),
        'image/object/bbox/ymax': feature_parse.float_list_feature(ymax_norm),
        'image/object/class/text': feature_parse.bytes_list_feature(classes_text),
        'image/object/difficult': feature_parse.int64_list_feature(difficult_obj),
        'image/object/truncated': feature_parse.float_list_feature(
            annotations['truncated'])
    }))

    return example


def filter_annotations(img_all_annotations, used_classes):
    """
    过滤掉一些没有用的类别和dontcare区域的annotations
    :param img_all_annotations: 图片的所有标注
    :param used_classes: 需要留下记录的类别
    :return:
    """
    img_filtered_annotations = {}

    # 1、过滤这个图片中标注的我们训练指定不需要的类别，把索引记录下来
    # 方便后面在处理对应的一些坐标时候使用
    relevant_annotation_indices = [
        i for i, x in enumerate(img_all_annotations['type']) if x in used_classes
    ]
    # for i, x in enumerate(img_all_annotations['type']):
    #     print(i, x)
    # print('===', relevant_annotation_indices)
    # 2、获取过滤后的下标对应某个标记物体的其它信息
    for key in img_all_annotations.keys():
        img_filtered_annotations[key] = (
            img_all_annotations[key][relevant_annotation_indices])

    # 3、如果dontcare在我们要获取的类别里面，也进行组合获取，然后过滤相关的bboxes不符合要求的
    if 'dontcare' in used_classes:
        dont_care_indices = [i for i,
                             x in enumerate(img_filtered_annotations['type'])
                             if x == 'dontcare']

        # bounding box的格式[y_min, x_min, y_max, x_max]
        all_boxes = np.stack([img_filtered_annotations['2d_bbox_top'],
                              img_filtered_annotations['2d_bbox_left'],
                              img_filtered_annotations['2d_bbox_bottom'],
                              img_filtered_annotations['2d_bbox_right']],
                             axis=1)

        # 计算bboxesIOU，比如这样的
        # Truck 0.00 0 -1.57 599.41 156.40 629.75 189.25 2.85 2.63 12.34 0.47 1.49 69.44 -1.56
        # DontCare -1 -1 -10 503.89 169.71 590.61 190.13 -1 -1 -1 -1000 -1000 -1000 -10
        # DontCare -1 -1 -10 511.35 174.96 527.81 187.45 -1 -1 -1 -1000 -1000 -1000 -10
        # DontCare -1 -1 -10 532.37 176.35 542.68 185.27 -1 -1 -1 -1000 -1000 -1000 -10
        # DontCare -1 -1 -10 559.62 175.83 575.40 183.15 -1 -1 -1 -1000 -1000 -1000 -10
        ious = iou(boxes1=all_boxes,
                   boxes2=all_boxes[dont_care_indices])

        # 删除所有 bounding boxes 与 dontcare region 重叠的区域
        if ious.size > 0:
            # 找出下标
            boxes_to_remove = np.amax(ious, axis=1) > 0.0
            for key in img_all_annotations.keys():
                img_filtered_annotations[key] = (
                    img_filtered_annotations[key][np.logical_not(boxes_to_remove)])

    return img_filtered_annotations


def read_annotation_file(filename):

    with open(filename) as f:
        content = f.readlines()
    # 分割解析内容
    content = [x.strip().split(' ') for x in content]
    # 保存内容到字典结构
    anno = dict()
    anno['type'] = np.array([x[0].lower() for x in content])
    anno['truncated'] = np.array([float(x[1]) for x in content])
    anno['occluded'] = np.array([int(x[2]) for x in content])
    anno['alpha'] = np.array([float(x[3]) for x in content])

    anno['2d_bbox_left'] = np.array([float(x[4]) for x in content])
    anno['2d_bbox_top'] = np.array([float(x[5]) for x in content])
    anno['2d_bbox_right'] = np.array([float(x[6]) for x in content])
    anno['2d_bbox_bottom'] = np.array([float(x[7]) for x in content])
    return anno


def convert_kitti_to_tfrecords(data_dir, output_path, classes_to_use,
                               validation_set_size):
    """
    将KITTI detection 转换成TFRecords.
    :param data_dir: 源数据目录
    :param output_path: 输出文件目录
    :param classes_to_use: 选择需要使用的类别
    :param validation_set_size: 验证集大小
    :return:
    """
    train_count = 0
    val_count = 0

    # 1、创建KITTI训练和验证集的tfrecord位置
    # 标注信息位置
    annotation_dir = os.path.join(data_dir,
                                  'training',
                                  'label_2')

    # 图片位置
    image_dir = os.path.join(data_dir,
                             'data_object_image_2',
                             'training',
                             'image_2')

    train_writer = tf.io.TFRecordWriter(output_path + 'train.tfrecord')
    val_writer = tf.io.TFRecordWriter(output_path + 'val.tfrecord')

    # 2、列出所有的图片，进行每张图片的内容和标注信息的获取，写入到tfrecords文件
    images = sorted(os.listdir(image_dir))
    for img_name in images:

        # （1）获取当前图片的编号数据，并拼接读取相应标注文件
        img_num = int(img_name.split('.')[0])

        # （2）读取标签文件函数
        # 整数需要进行填充成与标签文件相同的6位字符串
        img_anno = read_annotation_file(os.path.join(annotation_dir,
                                                     str(img_num).zfill(6) + '.txt'))
        # print('-=-=', os.path.join(annotation_dir,str(img_num).zfill(6) + '.txt'))

        # （3）过滤标签函数
        # 当前图片的标注中 过滤掉一些没有用的类别和dontcare区域的annotations
        annotation_for_image = filter_annotations(img_anno, classes_to_use)

        # （4）写入训练和验证集合TFRecord文件
        # 读取拼接的图片路径，然后与过滤之后的标注结果进行合并到一个example中
        image_path = os.path.join(image_dir, img_name)
        example = prepare_example(image_path, annotation_for_image)
        # 如果小于验证集数量大小就直接写入验证集，否则写入训练集
        is_validation_img = img_num < validation_set_size
        if is_validation_img:
            val_writer.write(example.SerializeToString())
            val_count += 1
        else:
            train_writer.write(example.SerializeToString())
            train_count += 1

    train_writer.close()
    val_writer.close()

def main(args):

    convert_kitti_to_tfrecords(
        data_dir=args.data_dir,
        output_path=args.output_path,
        classes_to_use=args.classes_to_use.replace(" ", "").split(','),
        validation_set_size=args.validation_set_size)


if __name__ == '__main__':

    args = parser.parse_args(sys.argv[1:])
    main(args)