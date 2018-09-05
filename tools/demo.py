
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
from lib.utils.test import im_detect
from lib.utils.nms import nms


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_40000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def demo(sess, net, image_name):
    # 根据路径，使用opencv读取图片数据
    im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    im = cv2.imread(im_file)

    # 进行目标检查
    timer = Timer()
    timer.tic()
    # 进行预测返回300个box的得分和位置
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # 每个类最高得分上图的阈值
    CONF_THRESH = 0.1
    # 每个类NMS阈值
    NMS_THRESH = 0.1
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # +1需要跳过背景
        # 获取到所有候选框对应这个分类的位置
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        # 获取到所有候选框对应这个分类的得分
        cls_scores = scores[:, cls_ind]
        # 合并所有的位置和得分，(x1,y1,x2,y2,score)
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        # 通过非极大值抑制保留0.1的候选框以及得分
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        # 上图
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def vis_detections(im, class_name, dets, thresh=0.5):
    # 挑选出大于阈值的候选框，没有则返回
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    # 将图片由为BGR转换为RGB，然后开始画图
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,thresh),fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--network', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    # 找到模型的路径
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('../default', DATASETS[dataset][0],'default', NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # 设置tf的配置
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    # 初始化session
    sess = tf.Session(config=tfconfig)

    # 加载网络
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    else:
        raise NotImplementedError
    # 构建网络
    net.create_architecture("TEST", 21, tag='default')
    # 加载训练好的模型
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    print('Loaded network {:s}'.format(tfmodel))

    # 测试数据
    im_names = ['000456.jpg', '000457.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)

    plt.show()



