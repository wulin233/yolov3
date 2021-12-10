import logging
import tensorflow as tf
import numpy as np
import cv2
import sys
import argparse
import os






from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./data/kitti_tfrecords/train.tfrecord',
                    help='训练数据集路径')
parser.add_argument('--val_dataset', type=str, default='./data/kitti_tfrecords/val.tfrecord',
                    help='验证集目录')
parser.add_argument('--tiny', type=bool, default=False, help='加载的模型类型yolov3 or yolov3-tiny')
parser.add_argument('--weights', type=str, default='./data/yolov3.weights',
                    help='模型预训练权重路径')
parser.add_argument('--classes', type=str, default='./data/kitti.names',
                    help='类别文件路径')
parser.add_argument('--mode', type=str, default='fit', choices=['fit', 'eager_tf'],
                    help='fit: model.fit模式, eager_tf: 自定义GradientTape模式')
parser.add_argument('--transfer', type=str, default='none', choices=['none', 'darknet',
                                                                     'no_output', 'frozen',
                                                                     'fine_tune'],
                    help='none: 全部进行训练'
                         '迁移学习并冻结所有, fine_tune: 迁移并只冻结darknet')
parser.add_argument('--size', type=int, default=416,
                    help='图片大小')
parser.add_argument('--epochs', type=int, default=20,
                    help='迭代次数')
parser.add_argument('--batch_size', type=int, default=32,
                    help='每批次大小')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='学习率')
parser.add_argument('--num_classes', type=int, default=6,
                    help='类别数量')


def main(args):

    # 1、判断传入的是需要训练YoloV3Tiny还是YOLOV3正常版本
    if args.tiny:
        model = YoloV3Tiny(args.size, training=True,
                           classes=args.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(args.size, training=True, classes=args.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    # 2、获取传入参数的训练数据
    if args.dataset:
        train_dataset = dataset.load_tfrecord_dataset(
            args.dataset, args.classes)
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.batch(args.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, args.size),
        dataset.transform_targets(y, anchors, anchor_masks, 6)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    # 3、获取传入参数的验证集数据
    if args.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(
            args.val_dataset, args.classes)
    val_dataset = val_dataset.batch(args.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, args.size),
        dataset.transform_targets(y, anchors, anchor_masks, 6)))
    print('===val_dataset:', val_dataset)

    # 4、判断是否进行迁移学习
    if args.transfer != 'none':
        # 加载与训练模型'./data/yolov3.weights'
        model.load_weights(args.weights)
        if args.transfer == 'fine_tune':
            # 冻结darknet
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif args.mode == 'frozen':
            # 冻结所有层
            freeze_all(model)
        else:
            # 重置网络后端结构
            if args.tiny:
                init_model = YoloV3Tiny(
                    args.size, training=True, classes=args.num_classes)
            else:
                init_model = YoloV3(
                    args.size, training=True, classes=args.num_classes)

            # 如果迁移指的是darknet
            if args.transfer == 'darknet':
                # 获取网络的权重
                for l in model.layers:
                    if l.name != 'yolo_darknet' and l.name.startswith('yolo_'):
                        l.set_weights(init_model.get_layer(
                            l.name).get_weights())
                    else:
                        freeze_all(l)
            elif args.transfer == 'no_output':
                for l in model.layers:
                    if l.name.startswith('yolo_output'):
                        l.set_weights(init_model.get_layer(
                            l.name).get_weights())
                    else:
                        freeze_all(l)

    # 5、定义优化器以及损失计算方式
    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=args.num_classes)
            for mask in anchor_masks]

    # 6、训练优化过程，训练指定模式
    # 用eager模式进行训练易于调试
    # keras model模式简单易用
    if args.mode == 'eager_tf':
        # 定义评估方式
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        # 迭代优化
        for epoch in range(1, args.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    # 1、计算模型输出和损失
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        # 根据输出和标签计算出损失
                        pred_loss.append(loss_fn(label, output))

                    # 计算总损失 = 平均损失 + regularization_loss
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                # 计算梯度以及更新梯度
                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                # 打印日志
                logging.info("{}_train_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss)

            # 验证数据集验证输出计算
            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                # 求损失
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                # 输出结果和标签计算损失
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                # 打印总损失
                logging.info("{}_val_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_val_loss.update_state(total_loss)

            logging.info("{}, train: {}, val: {}".format(
                epoch,
                avg_loss.result().numpy(),
                avg_val_loss.result().numpy()))
            # 保存模型位置
            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights(
                'checkpoints/yolov3_train_{}.tf'.format(epoch))
    else:
        model.compile(optimizer=optimizer, loss=loss)

        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint('./me/yolov3_train_{epoch}.tf',
                            verbose=1, save_weights_only=True),
            TensorBoard(log_dir='logs')
        ]

        model.fit(train_dataset,
                            epochs=args.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset,
                            steps_per_epoch=10)


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    main(args)