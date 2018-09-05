import os
import uuid
import pickle
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import scipy.sparse

from lib.config import config as cfg
from lib.datasets.imdb import imdb




class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        # 2007
        self._year = year
        # trainval
        self._image_set = image_set
        # data/VOCdevkit2007
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        # data/VOCdevkit2007/VOC2007
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        # 类型元组
        self._classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        # 类型和index组成的字典集合
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.jpg'
        # 从/VOCdevkit2007/VOC2007/ImageSets/Main/val.txt获取训练集的序号
        self._image_index = self._load_image_set_index()
        # 获取到所有的gt信息，这里没有执行这个方法
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # 配置
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None}

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def _get_default_path(self):
        return os.path.join(cfg.FLAGS2["data_dir"], 'VOCdevkit' + self._year)

    def _load_image_set_index(self):
        # 从这个目录下读取self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        # 查看是否有之前缓存好的gt目录
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # 如果有就直接加载
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb
        # 得到所有的gt信息，并存储在cache里
        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        # 找到文件的Annotations目录下的xml
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']: # false
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        # 找到图片中gt的数量
        num_objs = len(objs)
        # 一系列初始化
        # gt的坐标
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        # gt的类对应的index
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        # gt的对应的类的overlaps为1
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # gt的面积
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # 加载xml中的所有GT坐标
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # 将坐标以0,0为基准
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            # 将GT名转换为index
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])
    def image_path_from_index(self, index):
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path