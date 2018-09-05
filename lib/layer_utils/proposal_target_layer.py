import numpy as np
import numpy.random as npr

from lib.config import config as cfg
from lib.utils.bbox_transform import bbox_transform
from lib.utils.bbox_overlaps import bbox_overlaps

def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):

    # 经过proposal_layer产生了2000个rois以及得分
    all_rois = rpn_rois
    all_scores = rpn_scores

    # 是否需要将gt也放入训练集中，这里不需要
    if cfg.FLAGS.proposal_use_gt: # False
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )
        # not sure if it a wise appending, but anyway i am not using it
        all_scores = np.vstack((all_scores, zeros))

    # 这里我们采用Fast-RCNN论文中的128mini-batch，前景占0.25
    num_images = 1
    rois_per_image = cfg.FLAGS.batch_size / num_images # 128
    fg_rois_per_image = np.round(cfg.FLAGS.proposal_fg_fraction * rois_per_image) # 前景数量128 * 0.25 = 32

    labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
        all_rois, all_scores, gt_boxes, fg_rois_per_image,
        rois_per_image, _num_classes)
    # 128 * 5
    rois = rois.reshape(-1, 5)
    # 128
    roi_scores = roi_scores.reshape(-1)
    # 128 * 1
    labels = labels.reshape(-1, 1)
    # 128 * 84
    bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
    # 128 * 84
    bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
    # 128 * 84 true false
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):

    # 找出每个anchor和每个gt的IOU，overlaps (ex, gt)
    overlaps = bbox_overlaps(all_rois[:, 1:5],gt_boxes[:, :4])
    # argmax_overlaps 找出每一行的最大值,每个anchor对应最大的gt
    gt_assignment = overlaps.argmax(axis=1)
    # 取出每个anchor的最大IOU值
    max_overlaps = overlaps.max(axis=1)
    # 每个anchor对应的lable
    # 这里的lable不是anchor_target_layer里面的0和1，而是真实的分类标签
    labels = gt_boxes[gt_assignment, 4]

    # 这里也是根据Fast-RCNN论文中给的前背景定义，IOU>=0.5为前景，0.5>IOU>=0.1为背景，0.1以下忽略
    fg_inds = np.where(max_overlaps >= cfg.FLAGS.roi_fg_threshold)[0] # 0.5
    bg_inds = np.where((max_overlaps < cfg.FLAGS.roi_bg_threshold_high) & # 0.5
                       (max_overlaps >= cfg.FLAGS.roi_bg_threshold_low))[0] # 0.1

    # 如果前背景都有，先随机挑选前景，最多32个，前景不够用背景补充，如果背景也不够，用replace重复挑选，一共128个
    if fg_inds.size > 0 and bg_inds.size > 0:
        fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
        fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
        bg_rois_per_image = rois_per_image - fg_rois_per_image
        to_replace = bg_inds.size < bg_rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
    # 如果只有前景，则全在前景里随机挑选128个，不够则replace重复挑选
    elif fg_inds.size > 0:
        to_replace = fg_inds.size < rois_per_image
        fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = rois_per_image
    # 如果只有背景，则全在背景里随机挑选128个，不够则replace重复挑选
    elif bg_inds.size > 0:
        to_replace = bg_inds.size < rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = 0
    # 都没有则断点停下来
    else:
        import pdb
        pdb.set_trace()

    # 组合前景和背景lable
    keep_inds = np.append(fg_inds, bg_inds)
    labels = labels[keep_inds]
    # 因为我们前景在数组前面，所以我们可以利用下面的方法，将所有背景的lable变为0
    labels[int(fg_rois_per_image):] = 0
    # 保留从2000个rois筛选出来的128个候选框和标签
    rois = all_rois[keep_inds]
    roi_scores = all_scores[keep_inds]

    # 计算用于128个rois和其对应gt的偏移量，并且将lable拼接到第一列
    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
    # 计算权重
    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, roi_scores, bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    # 对bbox偏移量进行归一化操作，扩大了10倍
    if cfg.FLAGS.bbox_normalize_targets_precomputed: # True
        targets = ((targets - np.array(cfg.FLAGS2["bbox_normalize_means"])) # (0.0, 0.0, 0.0, 0.0)
                   / np.array(cfg.FLAGS2["bbox_normalize_stds"])) # (0.1, 0.1, 0.1, 0.1)
    # 将lable拼接到bbox的第一列
    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    # 取出128个rois对应的lable
    clss = bbox_target_data[:, 0]
    # [128，4 * 21]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    # 前景的位置
    inds = np.where(clss > 0)[0]
    for ind in inds:
        # 对应分类
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        # [128,4 * 21] 21个分类，只有自己分类的才有偏移量数据
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.FLAGS2["bbox_inside_weights"] # (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights
