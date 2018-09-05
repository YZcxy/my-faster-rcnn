import cv2
import numpy as np

def im_list_to_blob(ims):
    # 找到其中最大的blob形状，将所有的blob合成为一个，我们只用了一张图片，blob为1，这个方法没什么意义
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


def prep_im_for_blob(im, pixel_means, target_size, max_size):
    im = im.astype(np.float32, copy=False)
    # 每个像素都减去像素均值
    im -= pixel_means
    im_shape = im.shape
    # 最长和最短边长
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # 缩放比例
    im_scale = float(target_size) / float(im_size_min)
    # 如果缩放过后最大边长超过1000，则按最大边长1000进行缩放
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    # 进行缩放
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale