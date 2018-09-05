
import tensorflow as tf
import numpy as np

from lib.config import config as cfg
from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes
from lib.utils.nms import nms

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')

    if cfg_key == "TRAIN":
        pre_nms_topN = cfg.FLAGS.rpn_train_pre_nms_top_n #12000
        post_nms_topN = cfg.FLAGS.rpn_train_post_nms_top_n #2000
        nms_thresh = cfg.FLAGS.rpn_train_nms_thresh #0.7
    else:
        pre_nms_topN = cfg.FLAGS.rpn_test_pre_nms_top_n #6000
        post_nms_topN = cfg.FLAGS.rpn_test_post_nms_top_n #300
        nms_thresh = cfg.FLAGS.rpn_test_nms_thresh #0.7
    # 因为我们的输入是(1,3)维的
    im_info = im_info[0]
    # 1 * H * W * 9 其他维度不变，取18元素后9个为前景得分
    scores = rpn_cls_prob[:, :, :, num_anchors:]  # 9
    # 9WH * 4 个偏移量
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
    # 9WH 个得分
    scores = scores.reshape((-1, 1))
    # 通过偏移量对anchor进行调整，得到proposals
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
    # 修剪proposal，将超出边界的proposal修剪到图片范围内
    proposals = clip_boxes(proposals, im_info[:2])  # im_info[:2] 表示 宽和高

    # 得分从大到小排序，存储的为坐标
    order = scores.ravel().argsort()[::-1]
    # 根据坐标，筛选出前12000个proposals和scores
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # 非极大值抑制
    keep = nms(np.hstack((proposals, scores)), nms_thresh)  # np.hstack((proposals, scores) [x1,y1,x2,y2,score]

    # 因为keep已经排过序了，直接取前2000个
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    # 给proposal叠加一个维度，第一列全是0.0
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

    return blob, scores
