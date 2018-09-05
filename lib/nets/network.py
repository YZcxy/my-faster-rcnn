import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope

from lib.config import config as cfg
from lib.layer_utils.generate_anchors_pre import generate_anchors_pre
from lib.layer_utils.proposal_layer import proposal_layer
from lib.layer_utils.anchor_target_layer import anchor_target_layer
from lib.layer_utils.proposal_target_layer import proposal_target_layer
from lib.layer_utils.proposal_top_layer import proposal_top_layer

class Network(object):
    def __init__(self,batch_size=1):
        # 每次mini-batch的图片数
        self._batch_size = batch_size
        # 网络产生的所有的值的字典表
        self._predictions = {}
        # 所有的损失字典
        self._losses = {}
        # 存储训练RPN网络所需要的标签，偏移，权重
        self._anchor_targets = {}
        # 存储训练Fast-RCNN网络所需要的标签，偏移，权重
        self._proposal_targets = {}
        # 网络层级的字典表
        self._layers = {}
        # 存储整个网络特征的数组
        self._act_summaries = []
        # 所有预测值的字典表
        self._score_summaries = {}
        # 所有需要训练的参数集合
        self._train_summaries = []
        # 保存计算的所有损失
        self._event_summaries = {}
        # 需要修复的变量
        self._variables_to_fix = {}



    """
    这里开始是一些辅助方法
    """

    def _reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        # 第一次'rpn_cls_score_reshape'按以下转换
        # 第二次'rpn_cls_prob'反过来即可
        with tf.variable_scope(name):
            # 将数据装换为caffe的格式 1 * 18 * H * W
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # 将数据转换成 1 * 2 * 9H * W
            reshaped = tf.reshape(to_caffe, tf.concat(axis=0, values=[[self._batch_size], [num_dim, -1], [input_shape[2]]]))
            # 将数据装换为tensorflow的格式 1 * 9H * W * 2
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def _softmax_layer(self, bottom, name):
        if name == 'rpn_cls_prob_reshape':
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

    def _anchor_component(self):
        with tf.variable_scope('ANCHOR_' + 'basic'):
            # 根据_im_info存储的图片信息，计算出conv5的特征图大小
            height = tf.to_int32(tf.ceil(self._im_info[0, 0] / np.float32(self._feat_stride[0])))
            width = tf.to_int32(tf.ceil(self._im_info[0, 1] / np.float32(self._feat_stride[0])))
            # 传入特征图大小，计算出HWK个anchors
            anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                                [height, width,
                                                 self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                                [tf.float32, tf.int32], name="generate_anchors")
            anchors.set_shape([None, 4])
            anchor_length.set_shape([])
            # 将anchors的信息保留下来
            self._anchors = anchors
            self._anchor_length = anchor_length

    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name):
            # rois的id，第一列
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # 找到resize图片的大小，并将rois在图片上归一化到0-1范围
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # 不让梯度在这里计算
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            # 14
            pre_pool_size = cfg.FLAGS.roi_pooling_size * 2
            # 在这里通过归一化的rois在特征图上截图并resize成14*14大小的特征图
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')

    """
    proposal层从这里开始
    """
    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name):
            rois, rpn_scores = tf.py_func(proposal_top_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                           self._feat_stride, self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32])
            rois.set_shape([cfg.FLAGS.rpn_top_n, 5])
            rpn_scores.set_shape([cfg.FLAGS.rpn_top_n, 1])

        return rois, rpn_scores

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name):
            rois, rpn_scores = tf.py_func(proposal_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                           self._feat_stride, self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32])
            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

        return rois, rpn_scores

    def _anchor_target_layer(self, rpn_cls_score, name):
        with tf.variable_scope(name):
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer,
                [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
                [tf.float32, tf.float32, tf.float32, tf.float32])
            # rpn_labels 1 * 1 * 9H * W  每个anchor对应的lable
            rpn_labels.set_shape([1, 1, None, None])
            # rpn_bbox_targets 1 * H * W * 36 每个anchor和与其最大IOU的GT的偏移量
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
            # rpn_bbox_inside_weights 1 * H * W * 36 每个anchor的内部权重，正为1.0，其他为0
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
            # rpn_bbox_outside_weights 1 * H * W * 36 每个anchor的外部权重，统一赋值为1.0 / 样本总数
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            self._score_summaries.update(self._anchor_targets)

        return rpn_labels

    def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name):
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer,
                [rois, roi_scores, self._gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

            rois.set_shape([cfg.FLAGS.batch_size, 5])
            roi_scores.set_shape([cfg.FLAGS.batch_size])
            labels.set_shape([cfg.FLAGS.batch_size, 1])
            # 128 * (4*21) 其中每一行只有属于自己类别的才有值
            bbox_targets.set_shape([cfg.FLAGS.batch_size, self._num_classes * 4])
            bbox_inside_weights.set_shape([cfg.FLAGS.batch_size, self._num_classes * 4])
            bbox_outside_weights.set_shape([cfg.FLAGS.batch_size, self._num_classes * 4])

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            self._score_summaries.update(self._proposal_targets)

            return rois, roi_scores

    """
    整个网络的结构可以从这里开始往下看
    """

    def build_head(self, is_training, reuse=None):
        raise NotImplementedError

    def build_rpn(self, net, is_training, initializer):
        # 产生HWK个anchors
        self._anchor_component()
        # RPN网络，3*3卷积之后，分别接两个1*1的卷积
        rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
                          scope="rpn_conv/3x3")
        self._act_summaries.append(rpn)

        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_cls_score')
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "brpn_cls_prob_reshape")
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_bbox_pred')

        return rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape

    def build_proposals(self, is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score):
        # 训练模式
        if is_training:
            # _proposal_layer 使用经过rpn网络层后生成的rpn_bbox_pred转换为proposal并且进行修正
            # 然后按照得分排序，取前12000个proposal,再nms,取前面2000个
            rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            # 根据rpn_cls_score，抛弃边界外的anchor，计算训练RPN网络需要mini-batch的标签，偏移，以及权重
            rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")

            # Try to have a deterministic order for the computing graph, for reproducibility
            with tf.control_dependencies([rpn_labels]):
                # 对2000个rois和GT进行IOU运行，最后挑选出用于Fast-RCNN部分训练的样本
                rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
        # 测试模式
        else:
            if cfg.FLAGS.test_mode == 'nms':
                # 测试模式先排序取6000，然后再nms，取前面300个
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            elif cfg.FLAGS.test_mode == 'top':
                # top模式就没有nms操作，直接排序取前300个
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            else:
                raise NotImplementedError
        return rois

    def build_fc(self, pool5, is_training, reuse=None):
        raise NotImplementedError

    def build_predictions(self, net, rois, is_training, initializer, initializer_bbox):
        # 从特征图中剪切出128个rois，并进行池化操作
        pool5 = self._crop_pool_layer(net, rois, "pool5")
        # 将池化后的特征进行VGG16的全连接层
        fc7 = self.build_fc(pool5, is_training)

        # 全连接后softmax得出128 * 21个分数输出
        cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer, trainable=is_training,
                                         activation_fn=None, scope='cls_score')
        cls_prob = self._softmax_layer(cls_score, "cls_prob")
        # 全连接后84个bbox输出128 * 84个偏移输出
        bbox_prediction = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox,
                                               trainable=is_training, activation_fn=None, scope='bbox_pred')

        return cls_score, cls_prob, bbox_prediction

    def build_network(self, is_training=True):
        with tf.variable_scope(self._scope, self._scope):
            # 截断正态分布初始化，0均值，0.01标准差 （论文中说的使用random标准版）
            if cfg.FLAGS.initializer == "truncated": # 这里使用的是truncated版本
                initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
            else:
                initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

            # 构建卷积层，卷积层属于VGG16
            net = self.build_head(is_training)

            # 构建RPN网络，返回训练RPN网络的分类的值，分数，和reshape数据，以及bbox的值
            rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape = self.build_rpn(net, is_training,initializer)

            # 构建proposals层，其中产生了训练RPN网络需要的参数，以及Fast-RCNN需要的候选框和参数
            rois = self.build_proposals(is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score)

            # 构建Fast-RCNN网络，产生预测值，用于训练Fast-RCNN网络
            cls_score, cls_prob, bbox_pred = self.build_predictions(net, rois, is_training, initializer, initializer_bbox)

            self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
            # RPN网络的预测分数，预测值，bbox预测值
            self._predictions["rpn_cls_score"] = rpn_cls_score
            self._predictions["rpn_cls_prob"] = rpn_cls_prob
            self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
            # Fast-RCNN网络的预测分数，预测值，bbox预测值
            self._predictions["cls_score"] = cls_score
            self._predictions["cls_prob"] = cls_prob
            self._predictions["bbox_pred"] = bbox_pred
            # Fast-RCNN网络用到的候选框
            self._predictions["rois"] = rois

            self._score_summaries.update(self._predictions)

            return rois, cls_prob, bbox_pred

    def create_architecture(self, mode, num_classes, tag=None, anchor_scales=(8, 16, 32),anchor_ratios=(0.5, 1, 2)):
        # 图片具体数据
        self._image = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 3])
        # 图片的高，宽，压缩比例
        self._im_info = tf.placeholder(tf.float32, shape=[self._batch_size, 3])
        # 真实标签的X,Y,W,H 和class
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        # 添加一个标注
        self._tag = tag
        # 分类的数目
        self._num_classes = num_classes
        # 运行的模式，train 或者 test
        self._mode = mode
        training = mode == 'TRAIN'
        testing = mode == 'TEST'
        # anchor的规模和个数
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)
        # anchor的比例和个数
        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)
        # anchor的总个数
        self._num_anchors = self._num_scales * self._num_ratios

        # 创建weight正则化，是否要对bias使用正则化
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.FLAGS.weight_decay)
        if cfg.FLAGS.bias_decay: # 默认配置为False
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # 就算暂时用不到，也可以多列举一些方法
        with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)): # bias初始为0
            rois, cls_prob, bbox_pred = self.build_network(training) # 网络最终的输出为预测框，预测分类

        # 最后需要在图片中画出来的候选框添加到的输出
        layers_to_output = {'rois': rois}
        # 将所有预测产生的值也添加到输出
        layers_to_output.update(self._predictions)

        # 获取所有未标记trainable=False的变量，并保存
        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        if mode == 'TEST':
            # 如果是测试，要把之前在proposal_target_layer归一化的参数还原
            stds = np.tile(np.array(cfg.FLAGS2["bbox_normalize_stds"]), (self._num_classes)) # (0.1, 0.1, 0.1, 0.1)
            means = np.tile(np.array(cfg.FLAGS2["bbox_normalize_means"]), (self._num_classes)) # (0.0, 0.0, 0.0, 0.0)
            self._predictions["bbox_pred"] *= stds
            self._predictions["bbox_pred"] += means
        else:
            self._add_losses()
            # 把所有的损失也添加到输出
            layers_to_output.update(self._losses)

        val_summaries = []
        with tf.device("/cpu:0"):
            # 为图片画上gt
            val_summaries.append(self._add_image_summary(self._image, self._gt_boxes))
            # 所有的损失
            for key, var in self._event_summaries.items():
                val_summaries.append(tf.summary.scalar(key, var))
            # 所有预测值
            for key, var in self._score_summaries.items():
                self._add_score_summary(key, var)
            # 存储整个网络特征的数组
            for var in self._act_summaries:
                self._add_act_summary(var)
            # 所有需要训练的参数集合
            for var in self._train_summaries:
                self._add_train_summary(var)

        self._summary_op = tf.summary.merge_all()
        if not testing:
            self._summary_op_val = tf.summary.merge(val_summaries)

        return layers_to_output

    """所有的损失函数从这里开始"""

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        # 9 或者 1
        # 因为RPN里面的box偏移量没有进行归一化，所以这里采用1/9来控制，使其变成了普通的L1损失
        sigma_2 = sigma ** 2
        # 1 * H * W * 36 预测偏移和实际偏移的偏差
        box_diff = bbox_pred - bbox_targets
        # 内部权重用于控制偏差是否起作用
        in_box_diff = bbox_inside_weights * box_diff
        # 计算|x|，根据|x|的大小，求不同的损失
        abs_in_box_diff = tf.abs(in_box_diff)
        # 训练RPN网络判断|x|是否小于1/9，几乎约等于L2，训练Fast-RCNN判断是否小于1
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        # 如果此处是计算box的内部损失，0.5X^2 + |X|-0.5
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        # 加上一个外部权重，Lambdaλ，用于控制平衡
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box

    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('loss_' + self._tag):
            """RPN class loss"""
            # 9HW * 2
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
            # 9HW个，其中只有256个大于等于0的
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            # 把lable为-1的抛弃
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            # 提取出来256个
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            # 先计算softmax，然后求损失
            rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

            """ RPN, bbox loss"""
            # H * W * 36 预测的偏移量
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            # 1 * H * W * 36 每个ahchor和gt的偏移量
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            # 1 * H * W * 36 正为1.0，其他为0
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            # 1 * H * W * 36 统一赋值为1.0 / 样本总数
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
            # 计算smooth-L1损失
            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

            """RCNN, class loss"""
            # 128 * 21
            cls_score = self._predictions["cls_score"]
            # 128
            label = tf.reshape(self._proposal_targets["labels"], [-1])
            # 先计算softmax，然后求损失
            cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(cls_score, [-1, self._num_classes]), labels=label))

            """RCNN, bbox loss"""
            # 128 * 84 预测的偏移量
            bbox_pred = self._predictions['bbox_pred']
            # 128 * 84 其中每一行只有属于自己类别的才有值
            bbox_targets = self._proposal_targets['bbox_targets']
            # 128 * 84 # (1.0, 1.0, 1.0, 1.0) 其中每一行只有属于自己类别的才有值
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            # 128 * 84 # (1.0, 1.0, 1.0, 1.0) 其中每一行只有属于自己类别的才有值
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
            # 计算smooth-L1损失
            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box
            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box

            loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
            self._losses['total_loss'] = loss

            self._event_summaries.update(self._losses)

        return loss

    def test_image(self, sess, image, im_info):
        feed_dict = {self._image: image,
                     self._im_info: im_info}
        cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                         self._predictions['cls_prob'],
                                                         self._predictions['bbox_pred'],
                                                         self._predictions['rois']],
                                                        feed_dict=feed_dict)
        return cls_score, cls_prob, bbox_pred, rois

    def train_step(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                            self._losses['rpn_loss_box'],
                                                                            self._losses['cross_entropy'],
                                                                            self._losses['loss_box'],
                                                                            self._losses['total_loss'],
                                                                            train_op],
                                                                           feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

    """ 所有的总结整理方法 """

    def _add_image_summary(self, image, boxes):
        # 把均值添加回来
        image += cfg.FLAGS2["pixel_means"]
        # 将bgr转换回rgb
        channels = tf.unstack(image, axis=-1)
        image = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
        # 除去归一化，并从[x1, y1, x2, y2, cls]变化到[y1, x1, y1, x1]
        width = tf.to_float(tf.shape(image)[2])
        height = tf.to_float(tf.shape(image)[1])
        cols = tf.unstack(boxes, axis=1)
        boxes = tf.stack([cols[1] / height,
                          cols[0] / width,
                          cols[3] / height,
                          cols[2] / width], axis=1)
        # 为box添加一个维度
        boxes = tf.expand_dims(boxes, dim=0)
        # 将gt画上图片
        image = tf.image.draw_bounding_boxes(image, boxes)

        return tf.summary.image('ground_truth', image)

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))
    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

