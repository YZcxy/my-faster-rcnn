import numpy as np
import numpy.random as npr
import tensorflow as tf

from lib.config import config as cfg
from lib.utils.bbox_overlaps import bbox_overlaps
from lib.utils.bbox_transform import bbox_transform

def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
    A = num_anchors  # 9
    total_anchors = all_anchors.shape[0]  # 9HW
    K = total_anchors / num_anchors  # HW
    im_info = im_info[0]

    """根据边界值，将边界值外的anchor抛弃"""
    # 设置最小的边界值，0表示为图片最边上
    _allowed_border = 0
    # rpn_cls_score 为 1 * H * W * 18
    height, width = rpn_cls_score.shape[1:3]
    # 找到所有图片范围内anchors的坐标
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]
    # 只保留了没出边界的anchors
    anchors = all_anchors[inds_inside, :]
    # 只创建范围内数量的lable，填充-1
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    """计算anchor和gt的IOU"""
    # 找出每个anchor对应每个gt的IOU，overlaps (ex, gt)
    overlaps = bbox_overlaps(anchors,gt_boxes)
    # argmax_overlaps 找出每一行的最大值,每个anchor对应最大的gt
    argmax_overlaps = overlaps.argmax(axis=1)
    # 取出每个anchor的最大IOU值
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    # gt_argmax_overlaps 找出每一列的最大值,每一个gt的最大anchors
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    # 取出每个gt的最大IOU值
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    # 返回每个gt对应最大IOU的anchor坐标
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    """根据IOU，为每个anchor打上正负标签"""
    # 根据IOU，为每个anchor打上标签
    if not cfg.FLAGS.rpn_clobber_positives:  # False
        # 每个anchor最大的IOU如果小于0.3,则lable标记为0
        labels[max_overlaps < cfg.FLAGS.rpn_negative_overlap] = 0  # 0.3
    # 将和gt最大IOU的anchor直接标记为1
    labels[gt_argmax_overlaps] = 1
    # 每个anchor最大的IOU如果大于0.7，则lable标记为1
    labels[max_overlaps >= cfg.FLAGS.rpn_positive_overlap] = 1 # 0.7

    # 可以把上面判断负的方法放后面执行，这样gt对应最大IOU小于0.3可能被覆盖
    if cfg.FLAGS.rpn_clobber_positives:
        labels[max_overlaps < cfg.FLAGS.rpn_negative_overlap] = 0

    """随机赛选正负标签，一共256个"""
    # 如果正标签数量超过128个，则将多余的随机置为-1，一般正标签很少，不会超过
    num_fg = int(cfg.FLAGS.rpn_fg_fraction * cfg.FLAGS.rpn_batchsize) # 256 * 0.5
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1
    # 如果负标签超过256-正标签，则将多余的随机置为-1，所以最后正负标签一共256个
    num_bg = cfg.FLAGS.rpn_batchsize - np.sum(labels == 1) # 256 - 正标签数
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    # 计算图片范围内的每个anchor和其对应最大的gt的偏移量
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    """为损失函数设计内外权重"""
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # 内部权重就是将正样本的参数赋值为1.0，其他为0
    bbox_inside_weights[labels == 1, :] = np.array(cfg.FLAGS2["bbox_inside_weights"]) # (1.0, 1.0, 1.0, 1.0)
    # 外部权重设置为统一的 1.0 / 256
    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.FLAGS.rpn_positive_weight < 0: #是否要使用同一的外部权重，这是用统一的
        # 统一的权重
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((cfg.FLAGS.rpn_positive_weight > 0) &
                (cfg.FLAGS.rpn_positive_weight < 1))
        # 也可以分开设置权重
        positive_weights = (cfg.FLAGS.rpn_positive_weight /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.FLAGS.rpn_positive_weight) /
                            np.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # 将所有的集合还原成原来的大小，边界外的填充
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    # labels
    # 1 * 9 * H * W
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    # 1 * 1 * 9H * W
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # bbox_targets
    # 1 * H * W * 36
    bbox_targets = bbox_targets.reshape((1, height, width, A * 4))
    rpn_bbox_targets = bbox_targets

    # bbox_inside_weights
    # 1 * H * W * 36
    bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4))
    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    # 1 * H * W * 36
    bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4))
    rpn_bbox_outside_weights = bbox_outside_weights

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

def _unmap(data, count, inds, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

def _compute_targets(ex_rois, gt_rois):
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

