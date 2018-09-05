
import time

import numpy as np

from lib.config import config as cfg
from lib.utils.minibatch import get_minibatch

class RoIDataLayer(object):
    """Fast R-CNN data layer used for training."""

    def __init__(self, roidb, num_classes, random=False):
        # 为训练产生随机roidb序列
        self._roidb = roidb
        self._num_classes = num_classes
        self._random = random
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        # 对图片index进行随机排序
        if self._random:
            st0 = np.random.get_state()
            millis = int(round(time.time() * 1000)) % 4294967295
            np.random.seed(millis)

        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        # 存储随机序列
        if self._random:
            np.random.set_state(st0)

        self._cur = 0

    def _get_next_minibatch_inds(self):
        # 获取这次mini-batch需要的图片序号
        # 如果roidb样本以及训练完了，则重新随机排序
        if self._cur + cfg.FLAGS.ims_per_batch >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.FLAGS.ims_per_batch]
        self._cur += cfg.FLAGS.ims_per_batch

        return db_inds

    def _get_next_minibatch(self):
        # 根据序号，返回mini-batch数据
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db)

    def forward(self):
        # minibatch中的blobs包含data,gt_boxes和im_info
        blobs = self._get_next_minibatch()
        return blobs