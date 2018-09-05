import time
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

import lib.config.config as cfg
from lib.nets.vgg16 import vgg16
from lib.datasets.factory import get_imdb
from lib.datasets import roidb as rdl_roidb
from lib.datasets.imdb import imdb as imdb2
from lib.layer_utils.roi_data_layer import RoIDataLayer
from lib.utils.timer import Timer

try:
  import cPickle as pickle
except ImportError:
  import pickle
import os

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if True:
        print('Appending horizontally-flipped training examples...')
        # 翻转gt
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    rdl_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb

def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        # 初始化了imdb，已经获取了一些基础配置
        imdb = get_imdb(imdb_name)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method("gt")
        print('Set proposal method: {:s}'.format("gt"))
        # 为imdb.roidb添加了一些配置，宽高，最大类别等
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = imdb2(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb



class Train:
    def __init__(self):
        # 初始化网络
        if cfg.FLAGS.network == 'vgg16':
            self.net = vgg16(batch_size=cfg.FLAGS.ims_per_batch) # 1
        else:
            raise NotImplementedError

        self.imdb, self.roidb = combined_roidb("voc_2007_trainval")
        # 输入输出
        self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
        self.output_dir = cfg.get_output_dir(self.imdb, 'default')

    def train(self):
        # 创建session
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)

        with sess.graph.as_default():
            tf.set_random_seed(cfg.FLAGS.rng_seed)  # rng_seed=3
            # 构建网络
            layers = self.net.create_architecture("TRAIN", self.imdb.num_classes, tag='default')
            loss = layers['total_loss']
            lr = tf.Variable(cfg.FLAGS.learning_rate, trainable=False)  # learning_rate=0.001
            momentum = cfg.FLAGS.momentum  # momentum=0.9
            # 创建优化器
            optimizer = tf.train.MomentumOptimizer(lr, momentum)
            # 计算梯度
            gvs = optimizer.compute_gradients(loss)

            # 是否要使用双倍偏差
            if cfg.FLAGS.double_bias:  # True
                final_gvs = []
                with tf.variable_scope('Gradient_Mult'):
                    for grad, var in gvs:
                        scale = 1.
                        if cfg.FLAGS.double_bias and '/biases:' in var.name:
                            scale *= 2.
                        if not np.allclose(scale, 1.0):
                            grad = tf.multiply(grad, scale)
                        final_gvs.append((grad, var))
                train_op = optimizer.apply_gradients(final_gvs)
            else:
                train_op = optimizer.apply_gradients(gvs)

            self.saver = tf.train.Saver(max_to_keep=100000)

            print('Loading initial model weights from {:s}'.format(cfg.FLAGS.pretrained_model))
            # 初始化所有变量
            variables = tf.global_variables()
            sess.run(tf.variables_initializer(variables, name='init'))
            # 从下载目录取加载我们下载的参数
            var_keep_dic = self.get_variables_in_checkpoint_file(cfg.FLAGS.pretrained_model)
            # 除去需要修正的变量，对其他变量进行赋值存储
            variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, cfg.FLAGS.pretrained_model)
            print('Loaded.')
            # 对fc6,fc7,conv1进行修正
            self.net.fix_variables(sess, cfg.FLAGS.pretrained_model)
            print('Fixed.')
            sess.run(tf.assign(lr, cfg.FLAGS.learning_rate)) # learning_rate=0.001
            last_snapshot_iter = 0

            timer = Timer()
            iter = last_snapshot_iter + 1
            last_summary_time = time.time()
            while iter < cfg.FLAGS.max_iters + 1:  # 40000
                # 根据论文描述，将30000步后的学习率调低
                if iter == cfg.FLAGS.step_size + 1:  # 30000
                    sess.run(tf.assign(lr, cfg.FLAGS.learning_rate * cfg.FLAGS.gamma))  # gamma=0.1

                timer.tic()
                # 获取这次mini-batch训练需要用的数据data,gt_boxes和im_info
                blobs = self.data_layer.forward()

                # 将参数传入进行损失的计算
                rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = self.net.train_step(sess, blobs, train_op)
                timer.toc()
                iter += 1

                # 10次迭代打印一次损失
                if iter % (cfg.FLAGS.display) == 0: # 10
                    print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                          '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n ' % \
                          (iter, cfg.FLAGS.max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box))
                    print('speed: {:.3f}s / iter'.format(timer.average_time))
                # 5000次进行一次存储
                if iter % cfg.FLAGS.snapshot_iterations == 0: # 5000
                    self.snapshot(sess, iter)

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")

    def snapshot(self, sess, iter):
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the model snapshot
        filename = 'vgg16_faster_rcnn_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        # Also store some meta information, random state, etc.
        nfilename = 'vgg16_faster_rcnn_iter_{:d}'.format(iter) + '.pkl'
        nfilename = os.path.join(self.output_dir, nfilename)
        # current state of numpy random
        st0 = np.random.get_state()
        # current position in the database
        cur = self.data_layer._cur
        # current shuffled indeces of the database
        perm = self.data_layer._perm

        # Dump the meta info
        with open(nfilename, 'wb') as fid:
            pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename

if __name__ == '__main__':
    train = Train()
    train.train()