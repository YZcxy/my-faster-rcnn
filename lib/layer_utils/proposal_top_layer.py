
import numpy as np
import numpy.random as npr

from lib.config import config as cfg
from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes

def proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, im_info, _feat_stride, anchors, num_anchors):
    # 只保留300个rois
    rpn_top_n = cfg.FLAGS.rpn_top_n # 300
    # 因为我们的输入是(1,3)维的
    im_info = im_info[0]
    # 1 * H * W * 9 其他维度不变，取18元素后9个为前景得分
    scores = rpn_cls_prob[:, :, :, num_anchors:]
    # 9WH * 4 个偏移量
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
    # 9WH 个得分
    scores = scores.reshape((-1, 1))

    length = scores.shape[0] # 9HW
    if length < rpn_top_n: # 9HW 不足 300
        # 随机挑选，知道凑齐300个
        top_inds = npr.choice(length, size=rpn_top_n, replace=True)
    else:
        # 挑选前300个得分
        top_inds = scores.argsort(0)[::-1]
        top_inds = top_inds[:rpn_top_n]
        top_inds = top_inds.reshape(rpn_top_n, )

    # 拿出前300个anchors，bbox，scores
    anchors = anchors[top_inds, :]
    rpn_bbox_pred = rpn_bbox_pred[top_inds, :]
    scores = scores[top_inds]

    # 用bbox偏移量对anchors进行修正
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
    # 将超出边框的剪切掉
    proposals = clip_boxes(proposals, im_info[:2])

    # 给proposal叠加一个维度，第一列全是0.0
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

    return blob, scores