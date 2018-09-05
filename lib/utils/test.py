import cv2
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os

from lib.utils.timer import Timer
from lib.config import config as cfg
from lib.utils.blob import im_list_to_blob
from lib.utils.bbox_transform import bbox_transform_inv

def _get_image_blob(im):
    # 处理像素值，均值化
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.FLAGS2["pixel_means"]
    # 获取最大和最小边长
    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.FLAGS2["test_scales"]: # 测试scale和训练一样，都是600，还可以多设几个规模
        # 计算缩放比例
        im_scale = float(target_size) / float(im_size_min)
        # 如果缩放过后最大边长超过1000，则按最大边长1000进行缩放
        if np.round(im_scale * im_size_max) > cfg.FLAGS.test_max_size:
            im_scale = float(cfg.FLAGS.test_max_size) / float(im_size_max)
        # 进行缩放
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # 将不同尺寸像素合为一个blob，我们只用了一个尺寸，所以没有意义
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes

def _get_blobs(im):
    # 得到处理后的图片数据以及缩放规模
    blobs = {}
    blobs['data'], im_scale_factors = _get_image_blob(im)

    return blobs, im_scale_factors

def im_detect(sess, net, im):
    # 对图片进行缩放处理
    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    # 封装网络需要的输入参数
    im_blob = blobs['data']
    blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
    # 开始测试,如果采用nms产生rois，则大概有300个rois
    _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])
    # 将预测框复原到原图大小，300*5，第一列全是0
    boxes = rois[:, 1:5] / im_scales[0]
    # 300*21
    scores = np.reshape(scores, [scores.shape[0], -1])
    # 300*84
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])

    if cfg.FLAGS.test_bbox_reg: # ture
        # 原图上的boxes进行根据偏移进行修正，然后修剪超出边界
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # 如果不修正，则简单的将boxes重复对应每一个类
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes