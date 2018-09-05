import numpy as np

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    # 求出proposal的面积，并且给得分排序
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 求出i和其他所有框的坐标差
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 就可以求出i和其他所有框的相交面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 求出i和其他所有框的IOU的值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 找到和i的IOU值小于0.7的保留下来
        inds = np.where(ovr <= thresh)[0]
        # 因为ovr不包括i自己，长度少1，坐标差一位
        order = order[inds + 1]

    return keep