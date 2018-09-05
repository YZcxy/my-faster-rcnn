import numpy as np
import PIL

def prepare_roidb(imdb):
    roidb = imdb.roidb
    if not (imdb.name.startswith('coco')):
        sizes = [PIL.Image.open(imdb.image_path_at(i)).size
            for i in range(imdb.num_images)]
    for i in range(len(imdb.image_index)):
        # 为roidb添加路径信息
        roidb[i]['image'] = imdb.image_path_at(i)
        # 为roidb添加宽高信息
        if not (imdb.name.startswith('coco')):
            roidb[i]['width'] = sizes[i][0]
            roidb[i]['height'] = sizes[i][1]
        # 为roidb添加最大的gt类已经对应的IOU值
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        max_overlaps = gt_overlaps.max(axis=1)
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # 如果最大IOU为0，则类别也为0
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # 如果最大IOU不为0，则类别也不能为0
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)