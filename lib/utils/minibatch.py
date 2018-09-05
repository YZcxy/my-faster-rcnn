import cv2

import numpy as np
import numpy.random as npr

from lib.config import config as cfg
from lib.utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb):
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.FLAGS2["scales"]),
                                    size=num_images)
    assert (cfg.FLAGS.batch_size % num_images == 0), 'num_images ({}) must divide BATCH_SIZE ({})'.format(num_images,
                                                                                                          cfg.FLAGS.batch_size)
    # 得到resize并去均值的像素blob以及缩放比例
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)
    blobs = {'data': im_blob}

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"
    # 找到所有有效GT的下标
    if cfg.FLAGS.use_all_gt: # ture
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        # 对于coco数据，还需要根据IUO来筛选一下
        gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    # [x1,y1,x2,y2,cls]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    # 为gt也乘上缩放比例
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    # 将cls组合进gtbox
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)

    return blobs

def _get_image_blob(roidb, scale_inds):
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        # 读取图片的数据，H*W*3
        im = cv2.imread(roidb[i]['image'])
        # 如果GT翻转了，则图片也翻转
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.FLAGS2["scales"][scale_inds[i]] # 600
        # 得到去均值和缩放后的像素点和比例
        im, im_scale = prep_im_for_blob(im, cfg.FLAGS2["pixel_means"], target_size, cfg.FLAGS.max_size)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # 将所有的图片像素合为一个blob，我们只用了一张图，所以没有意义
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales