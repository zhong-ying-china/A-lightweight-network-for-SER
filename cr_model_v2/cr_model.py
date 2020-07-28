from collections import defaultdict

import tensorflow as tf
from ruamel.yaml.comments import CommentedSeq
import numpy as np
from utils import var_cnn_util as vcu


def variable_summaries(x):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(x)
        # with tf.name_scope('stddev'):
        mean_summary = tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(x - mean)))
        std_summary = tf.summary.scalar('stddev', stddev)
        max_summary = tf.summary.scalar('max', tf.reduce_max(x))
        min_summary = tf.summary.scalar('min', tf.reduce_min(x))
        his_summary = tf.summary.histogram('histogram', x)
    return [mean_summary, std_summary, max_summary, min_summary, his_summary]


class BaseCRModel(object):

    def __init__(self, hps):
        self.hps = hps

        if hps.float_type == '16':
            float_type = tf.float16
        elif hps.float_type == '64':
            float_type = tf.float64
        else:
            float_type = tf.float32
        self.float_type = float_type

        # ==== placeholder ====
        self.fc_kprob_ph = tf.placeholder(float_type, shape=[], name='fc_kprob_ph')
        self.lr_ph = tf.placeholder(float_type, shape=[], name='lr_ph')
        # # lambda_ph is the balance hyperparameter for auxiliary loss function
        # self.lambda_ph = tf.placeholder(float_type, shape=[], name='lambda_ph')
        self.x_ph = tf.placeholder(float_type, [None, None, hps.freq_size], name='x_ph')
        self.t_ph = tf.placeholder(tf.int32, shape=[None], name='t_ph')  # seq lens
        self.e_ph = tf.placeholder(tf.int32, shape=[None], name='e_ph')  # emo labels
        self.l_ph = tf.placeholder(tf.string, shape=[None], name='l_ph')
        # loss weight of emo classifier for cross entropy
        self.e_w_ph = tf.placeholder(float_type, shape=[None], name='e_w_ph')
        self.beta_B = tf.placeholder(float_type,shape=[], name='beta_B')
        self.beta_C = tf.placeholder(float_type, shape=[], name='beta_C')
        self.is_training_ph = tf.placeholder(tf.bool, shape=[], name='is_training_ph')
        self.dist_loss_lambda_ph = tf.placeholder(float_type, shape=[], name='dist_loss_lambda_ph')
        self.cos_loss_lambda_ph = tf.placeholder(float_type, shape=[], name='cos_loss_lambda_ph')
        self.center_loss_lambda_ph = tf.placeholder(float_type, shape=[],
                                                    name='center_loss_lambda_ph')
        self.center_loss_alpha_ph = tf.placeholder(float_type, shape=[],
                                                   name='center_loss_alpha_ph')
        self.center_loss_beta_ph = tf.placeholder(float_type, shape=[],
                                                  name='center_loss_beta_ph')
        self.center_loss_gamma_ph = tf.placeholder(float_type, shape=[],
                                                   name='center_loss_gamma_ph')
        self.feature_norm_alpha_ph = tf.placeholder(float_type, shape=[],
                                                    name='feature_norm_alpha_ph')

        # todo: debug
        self.debug_dict = dict()

        # build graph
        # self.vars_d = None
        self.output_d = None
        self.metric_d = None
        self.loss_d = None
        self.update_op_d = None
        self.train_op_d = None
        self.grad_d = None
        # merged for training
        self.train_merged = None
        self.build_graph()

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1, dtype=self.float_type)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape, dtype=self.float_type)
        return tf.Variable(initial)

    def get_center_loss_centers_variable(self, shape=None):
        with tf.variable_scope('center_loss_variables') as scope:
            try:
                v = tf.get_variable('center_loss_centers', shape, dtype=self.float_type,
                                    initializer=tf.constant_initializer(0),
                                    trainable=False)
            except ValueError:
                scope.reuse_variables()
                v = tf.get_variable('center_loss_centers', dtype=self.float_type)
        return v

    def get_feature_norm_variable(self, shape=()):
        with tf.variable_scope('norm_variables') as scope:
            try:
                v = tf.get_variable('feature_norm', shape, dtype=self.float_type,
                                    initializer=tf.constant_initializer(1),
                                    trainable=False)
            except ValueError:
                scope.reuse_variables()
                v = tf.get_variable('feature_norm', dtype=self.float_type)
        return v

    def calc_soft_loss(self, label,y_pred, level,weights, num_classes):
        y_true = tf.one_hot(label, num_classes)
        flag_A = tf.cast(tf.equal(level, 'A'), self.float_type)
        flag_B = tf.cast(tf.equal(level, 'B'), self.float_type)
        flag_C = tf.cast(tf.equal(level, 'C'), self.float_type)

        y_pred = tf.nn.softmax(y_pred)
        y_true_shape = tf.shape(y_true)

        mask_B = tf.reshape(flag_B, (y_true_shape[0], 1))
        mask_B = y_true * mask_B
        mask_C = tf.reshape(flag_C, (y_true_shape[0], 1))
        mask_C = y_true * mask_C
        mask_A = tf.reshape(flag_A, (y_true_shape[0], 1))
        mask_A = y_true * mask_A

        loss_B = self.beta_B * mask_B + (1-self.beta_B) * y_pred
        loss_C = self.beta_C * mask_C
        loss_A = mask_A

        loss = loss_A + loss_B + loss_C
        # ce_loss is L-soft
        if self.hps.is_weighted_center_loss:
            soft_loss = -tf.reduce_sum(loss * tf.log(y_pred), axis=-1) * weights
            soft_loss = tf.reduce_mean(soft_loss) / (tf.reduce_mean(weights))
            return soft_loss
        else:
            soft_loss = -tf.reduce_sum(loss * tf.log(y_pred), axis=-1)
            soft_loss = tf.reduce_mean(soft_loss)
            return soft_loss

    def calc_Lq_loss(self, label,y_pred, level, num_classes):
        y_true = tf.one_hot(label, num_classes)
        flag_A = tf.cast(tf.equal(level, 'A'), self.float_type)
        flag_B = tf.cast(tf.equal(level, 'B'), self.float_type)
        flag_C = tf.cast(tf.equal(level, 'C'), self.float_type)

        y_pred = tf.nn.softmax(y_pred)
        y_true_shape = tf.shape(y_true)

        mask_B = tf.reshape(flag_B, (y_true_shape[0], 1))
        mask_B = y_true * mask_B
        mask_C = tf.reshape(flag_C, (y_true_shape[0], 1))
        mask_C = y_true * mask_C
        mask_A = tf.reshape(flag_A, (y_true_shape[0], 1))
        mask_A = y_true * mask_A

        loss_B = self.beta_B * mask_B + (1-self.beta_B) * y_pred
        loss_C = self.beta_C * mask_C + (1-self.beta_C) * y_pred
        loss_A = mask_A

        loss = loss_A + loss_B + loss_C
        # ce_loss is L-soft
        soft_loss = -tf.reduce_sum(loss * tf.log(y_pred), axis=-1)
        soft_loss = tf.reduce_mean(soft_loss)
        return soft_loss

    def calc_FL_loss(self, label, y_pred, level, weights,num_classes):
        y_true = tf.one_hot(label, num_classes)
        flag_A = tf.cast(tf.equal(level, 'A'), self.float_type)
        flag_B = tf.cast(tf.equal(level, 'B'), self.float_type)
        flag_C = tf.cast(tf.equal(level, 'C'), self.float_type)

        y_pred = tf.nn.softmax(y_pred)
        y_true_shape = tf.shape(y_true)

        mask_B = tf.reshape(flag_B, (y_true_shape[0], 1))
        mask_B = y_true * mask_B
        mask_C = tf.reshape(flag_C, (y_true_shape[0], 1))
        mask_C = y_true * mask_C
        mask_A = tf.reshape(flag_A, (y_true_shape[0], 1))
        mask_A = y_true * mask_A

        FL_Aloss = mask_A*tf.log(y_pred)
        FL_Bloss = (tf.multiply(mask_B, (1 - y_pred)**self.beta_B)) * tf.log(y_pred)
        FL_Closs = (tf.multiply(mask_C,(1 - y_pred)**self.beta_B)) * tf.log(y_pred)

        FL_loss = -tf.reduce_sum(FL_Aloss+FL_Bloss+FL_Closs, axis=-1)

        _FL_loss = FL_loss * weights
        _FL_loss = tf.reduce_mean(_FL_loss) / tf.reduce_mean(weights)

        return _FL_loss

    def calc_center_loss(self, features, labels, num_classes):
        len_features = features.get_shape()[1]
        # if self.hps.is_center_loss_f_norm:
        #     features = tf.nn.l2_normalize(features)
        if self.hps.center_loss_f_norm == 'f_norm':
            f_norm = self.get_feature_norm_variable(shape=[])
            features = features / f_norm
            # features = tf.nn.l2_normalize(features)
        elif self.hps.center_loss_f_norm == 'l2':
            features = tf.nn.l2_normalize(features)
        elif self.hps.center_loss_f_norm == 'l2_1':
            features = tf.nn.l2_normalize(features, axis=1)
        centers = self.get_center_loss_centers_variable(shape=[num_classes, len_features])
        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)
        if self.hps.is_weighted_center_loss:
            loss = tf.reduce_sum(tf.reduce_sum(tf.square(features - centers_batch),
                                               axis=-1) )
        else:
            loss = tf.nn.l2_loss(features - centers_batch)

        batch_size = tf.cast(tf.shape(features)[0], dtype=self.float_type)
        loss = loss / batch_size
        return loss

    def calc_center_loss2(self, features, labels, num_classes):
        len_features = features.get_shape()[1]

        if self.hps.center_loss_f_norm == 'f_norm':
            f_norm = self.get_feature_norm_variable(shape=[])
            features = features / f_norm
            # features = tf.nn.l2_normalize(features)
        elif self.hps.center_loss_f_norm == 'l2':
            features = tf.nn.l2_normalize(features)
        elif self.hps.center_loss_f_norm == 'l2_1':
            features = tf.nn.l2_normalize(features, axis=1)

        centers = self.get_center_loss_centers_variable(shape=[num_classes, len_features])
        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)
        if self.hps.is_weighted_center_loss:
            dist_in = tf.reduce_sum(tf.reduce_sum(tf.square(features - centers_batch),
                                                  axis=-1) )
        else:
            dist_in = tf.nn.l2_loss(features - centers_batch)

        batch_size = tf.cast(tf.shape(features)[0], dtype=self.float_type)
        dist_in = dist_in / batch_size

        centers0 = tf.expand_dims(centers, 0)
        centers1 = tf.expand_dims(centers, 1)
        c_diffs = centers0 - centers1

        dist_out = tf.nn.l2_loss(c_diffs) / tf.maximum(1., num_classes * (num_classes - 1.))

        epsilon = 1e-8

        loss = (dist_in + epsilon) / (dist_out + epsilon)

        # loss = tf.maximum(self.hps.dist_margin + dist_in - dist_out, 0)
        return loss

    def calc_center_loss3(self, features, labels, num_classes):
        len_features = features.get_shape()[1]

        if self.hps.center_loss_f_norm == 'f_norm':
            f_norm = self.get_feature_norm_variable(shape=[])
            features = features / f_norm
            # features = tf.nn.l2_normalize(features)
        elif self.hps.center_loss_f_norm == 'l2':
            features = tf.nn.l2_normalize(features)
        elif self.hps.center_loss_f_norm == 'l2_1':
            features = tf.nn.l2_normalize(features, axis=1)
        centers = self.get_center_loss_centers_variable(shape=[num_classes, len_features])
        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)
        # dist_in_2d = tf.losses.mean_squared_error(labels=centers_batch, predictions=features,
        #                                           reduction=tf.losses.Reduction.NONE)

        # [batch_size, 1]
        dist_in_batch = tf.reduce_sum(tf.square(centers_batch - features), axis=-1)

        centers0 = tf.expand_dims(centers, 0)
        centers1 = tf.expand_dims(centers, 1)
        c_diffs = centers0 - centers1
        c_l2s = tf.reduce_sum(tf.square(c_diffs), axis=-1)
        dist_ceiling = 100000
        epsilon = 1e-8
        c_l2s_mask = tf.eye(num_classes, dtype=self.float_type) * dist_ceiling + c_l2s + epsilon
        dist_out_batch = tf.gather(tf.reduce_min(c_l2s_mask, axis=-1), labels)
        dist = dist_in_batch / dist_out_batch
        if self.hps.is_weighted_center_loss:
            loss = tf.reduce_mean(dist)
        else:
            loss = tf.reduce_mean(dist)
        return loss

    # # todo: Test
    # def calc_center_loss3(self, features, labels):
    #     if self.hps.center_loss_f_norm == 'f_norm':
    #         f_norm = self.get_feature_norm_variable(shape=[])
    #         features = features / f_norm
    #         # features = tf.nn.l2_normalize(features)
    #     elif self.hps.center_loss_f_norm == 'l2':
    #         features = tf.nn.l2_normalize(features)
    #     elif self.hps.center_loss_f_norm == 'l2_1':
    #         features = tf.nn.l2_normalize(features, axis=1)
    #     elif self.hps.center_loss_f_norm == 'avg_l2':
    #         f_norm = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(features), axis=1)))
    #         features = features / f_norm
    #     batch_size = tf.cast(tf.shape(features)[0], dtype=self.float_type)
    #     f_norm = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(features), axis=1)))
    #     features = features / f_norm
    #     labels = tf.reshape(labels, [-1])
    #     u_label, u_idx, u_count = tf.unique_with_counts(labels)
    #     idx_matrix = tf.cast(tf.one_hot(u_idx, tf.shape(u_label)[0]), dtype=self.float_type)
    #     # [batch_size, 1, num_classes]
    #     idx_tensor = tf.expand_dims(idx_matrix, 1)
    #     # x_expand [batch_size, dim, 1]
    #     x_expand = tf.expand_dims(features, -1)
    #     # x_expand * idx_tensor [batch_size, dim, num_classes]
    #     # f_mean [dim, num_classes]
    #     f_mean = tf.reduce_sum(x_expand * idx_tensor, axis=0) / tf.cast(u_count, self.float_type)
    #     # centers [num_classes, dim]
    #     centers = tf.transpose(f_mean, [1, 0])
    #     centers_batch = tf.gather(centers, u_idx)
    #     # todo: balance sample rate on batch level
    #     if self.hps.is_weighted_center_loss:
    #         dist_in = tf.reduce_sum(tf.reduce_sum(tf.square(features - centers_batch),
    #                                               axis=-1) * self.e_w_ph) / (
    #                           tf.reduce_mean(self.e_w_ph) * batch_size)
    #     else:
    #         dist_in = tf.nn.l2_loss(features - centers_batch) / batch_size
    #
    #     num_classes = tf.cast(tf.shape(centers)[0], dtype=self.float_type)
    #
    #     centers0 = tf.expand_dims(centers, 0)
    #     centers1 = tf.expand_dims(centers, 1)
    #     c_diffs = centers0 - centers1
    #
    #     dist_out = tf.nn.l2_loss(c_diffs) / tf.maximum(1., num_classes * (num_classes - 1.))
    #     # # todo: debug
    #     # self.debug_dict['dist_in'] = dist_in
    #     # self.debug_dict['dist_out'] = dist_out
    #
    #     epsilon = 1e-8
    #
    #     loss = (dist_in + epsilon) / (dist_out + epsilon)
    #
    #     # loss = tf.maximum(self.hps.dist_margin + dist_in - dist_out, 0)
    #     return loss

    # todo: Test
    def calc_center_loss4(self, features, labels):
        if self.hps.center_loss_f_norm == 'f_norm':
            f_norm = self.get_feature_norm_variable(shape=[])
            features = features / f_norm
            # features = tf.nn.l2_normalize(features)
        elif self.hps.center_loss_f_norm == 'l2':
            features = tf.nn.l2_normalize(features)
        elif self.hps.center_loss_f_norm == 'l2_1':
            features = tf.nn.l2_normalize(features, axis=1)
        elif self.hps.center_loss_f_norm == 'avg_l2':
            f_norm = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(features), axis=1)))
            features = features / f_norm
        epsilon = 1e-8
        batch_size = tf.cast(tf.shape(features)[0], dtype=self.float_type)
        f_norm = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(features), axis=1)))
        features = features / f_norm
        labels = tf.reshape(labels, [-1])
        u_label, u_idx, u_count = tf.unique_with_counts(labels)
        idx_matrix = tf.cast(tf.one_hot(u_idx, tf.shape(u_label)[0]), dtype=self.float_type)
        # [batch_size, 1, num_classes]
        idx_tensor = tf.expand_dims(idx_matrix, 1)
        # x_expand [batch_size, dim, 1]
        x_expand = tf.expand_dims(features, -1)
        # x_expand * idx_tensor [batch_size, dim, num_classes]
        # f_mean [dim, num_classes]
        f_mean = tf.reduce_sum(x_expand * idx_tensor, axis=0) / tf.cast(u_count, self.float_type)
        # centers [num_classes, dim]
        centers = tf.transpose(f_mean, [1, 0])
        centers_batch = tf.gather(centers, u_idx)
        # todo: balance sample rate on batch level
        if self.hps.is_weighted_center_loss:
            dist_in = tf.reduce_sum(tf.reduce_sum(tf.square(features - centers_batch),
                                                  axis=-1) * self.e_w_ph) / (
                              tf.reduce_mean(self.e_w_ph) * batch_size)
        else:
            dist_in = tf.nn.l2_loss(features - centers_batch) / batch_size

        num_classes = tf.cast(tf.shape(centers)[0], dtype=self.float_type)

        centers0 = tf.expand_dims(centers, 0)
        centers1 = tf.expand_dims(centers, 1)
        c_diffs_square = tf.reduce_sum(tf.square(centers0 - centers1), axis=-1)

        # dist_ceiling = 1000
        # c_diffs_mask = tf.eye(tf.shape(c_diffs_square)[0]) * dist_ceiling + c_diffs_square

        gamma = self.center_loss_gamma_ph * tf.reduce_mean(c_diffs_square)
        # gamma = self.center_loss_gamma_ph

        c_diffs_weight = gamma / (c_diffs_square + gamma + epsilon)
        c_diffs_weight_norm = c_diffs_weight / tf.reduce_mean(c_diffs_weight)

        dist_out = tf.reduce_sum(c_diffs_square * c_diffs_weight_norm) / tf.maximum(1.,
                                                                                    num_classes * (num_classes - 1.))

        # loss = (dist_in + epsilon) / (dist_out + epsilon)

        # # todo: debug
        # self.debug_dict['u_label'] = u_label
        # self.debug_dict['c_diffs_square'] = c_diffs_square
        # self.debug_dict['dist_out_square'] = c_diffs_square * c_diffs_weight_norm

        loss = tf.maximum(self.hps.dist_margin + dist_in - dist_out, 0)
        return loss

    # todo: Test
    def calc_center_loss5(self, features, labels):
        if self.hps.center_loss_f_norm == 'f_norm':
            f_norm = self.get_feature_norm_variable(shape=[])
            features = features / f_norm
            # features = tf.nn.l2_normalize(features)
        elif self.hps.center_loss_f_norm == 'l2':
            features = tf.nn.l2_normalize(features)
        elif self.hps.center_loss_f_norm == 'l2_1':
            features = tf.nn.l2_normalize(features, axis=1)
        elif self.hps.center_loss_f_norm == 'avg_l2':
            f_norm = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(features), axis=1)))
            features = features / f_norm
        epsilon = 1e-8
        batch_size = tf.cast(tf.shape(features)[0], dtype=self.float_type)
        f_norm = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(features), axis=1)))
        features = features / f_norm
        labels = tf.reshape(labels, [-1])
        u_label, u_idx, u_count = tf.unique_with_counts(labels)
        idx_matrix = tf.cast(tf.one_hot(u_idx, tf.shape(u_label)[0]), dtype=self.float_type)
        # [batch_size, 1, num_classes]
        idx_tensor = tf.expand_dims(idx_matrix, 1)
        # x_expand [batch_size, dim, 1]
        x_expand = tf.expand_dims(features, -1)
        # x_expand * idx_tensor [batch_size, dim, num_classes]
        # f_mean [dim, num_classes]
        f_mean = tf.reduce_sum(x_expand * idx_tensor, axis=0) / tf.cast(u_count, self.float_type)
        # centers [num_classes, dim]
        centers = tf.transpose(f_mean, [1, 0])
        centers_batch = tf.gather(centers, u_idx)
        # todo: balance sample rate on batch level
        if self.hps.is_weighted_center_loss:
            dist_in = tf.reduce_sum(tf.reduce_sum(tf.square(features - centers_batch),
                                                  axis=-1) * self.e_w_ph) / (
                              tf.reduce_mean(self.e_w_ph) * batch_size)
        else:
            dist_in = tf.nn.l2_loss(features - centers_batch) / batch_size

        # num_classes = tf.cast(tf.shape(centers)[0], dtype=self.float_type)

        centers0 = tf.expand_dims(centers, 0)
        centers1 = tf.expand_dims(centers, 1)
        c_diffs_square = tf.reduce_sum(tf.square(centers0 - centers1), axis=-1)

        dist_ceiling = 1000
        c_diffs_mask = tf.eye(tf.shape(c_diffs_square)[0]) * dist_ceiling + c_diffs_square
        dist_out = tf.reduce_min(c_diffs_mask)

        loss = tf.maximum(self.hps.dist_margin + dist_in - dist_out, 0)

        # # gamma = self.center_loss_gamma_ph * tf.reduce_mean(c_diffs_square)
        # gamma = self.center_loss_gamma_ph
        #
        # c_diffs_weight = gamma / (c_diffs_square + gamma + epsilon)
        # c_diffs_weight_norm = c_diffs_weight / tf.reduce_mean(c_diffs_weight)

        # dist_out = tf.reduce_sum(c_diffs_square * c_diffs_weight_norm) / tf.maximum(1.,
        #                                                                             num_classes * (num_classes - 1.))

        # loss = (dist_in + epsilon) / (dist_out + epsilon)
        #
        # # todo: debug
        # self.debug_dict['u_label'] = u_label
        # self.debug_dict['c_diffs_square'] = c_diffs_square
        # self.debug_dict['dist_out_square'] = c_diffs_square * c_diffs_weight_norm

        # loss = tf.maximum(self.hps.dist_margin + dist_in - dist_out, 0)
        return loss

    def calc_center_loss6(self, features, labels):
        batch_size = tf.cast(tf.shape(features)[0], dtype=self.float_type)
        # f_norm = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(features), axis=1)))
        # features = features / f_norm
        labels = tf.reshape(labels, [-1])
        u_label, u_idx, u_count = tf.unique_with_counts(labels)
        idx_matrix = tf.cast(tf.one_hot(u_idx, tf.shape(u_label)[0]), dtype=self.float_type)
        # [batch_size, 1, num_classes]
        idx_tensor = tf.expand_dims(idx_matrix, 1)
        # x_expand [batch_size, dim, 1]
        x_expand = tf.expand_dims(features, -1)
        # x_expand * idx_tensor [batch_size, dim, num_classes]
        # f_mean [dim, num_classes]
        f_mean = tf.reduce_sum(x_expand * idx_tensor, axis=0) / tf.cast(u_count, self.float_type)
        # centers [num_classes, dim]
        centers = tf.transpose(f_mean, [1, 0])
        centers_batch = tf.gather(centers, u_idx)
        dist_in = tf.nn.l2_loss(features - centers_batch) / batch_size

        num_classes = tf.cast(tf.shape(centers)[0], dtype=self.float_type)

        centers0 = tf.expand_dims(centers, 0)
        centers1 = tf.expand_dims(centers, 1)
        c_diffs = centers0 - centers1

        dist_out = tf.nn.l2_loss(c_diffs) / tf.maximum(1., num_classes * (num_classes - 1.))
        # # todo: debug
        # self.debug_dict['dist_in'] = dist_in
        # self.debug_dict['dist_out'] = dist_out
        epsilon = 1e-8
        loss = (dist_in + epsilon) / (dist_out + epsilon)
        # loss = tf.maximum(self.hps.dist_margin + dist_in - dist_out, 0)

        return loss

    def calc_center_loss7(self, features, labels, num_classes):
        len_features = features.get_shape()[1]
        batch_size = tf.cast(tf.shape(features)[0], dtype=self.float_type)
        num_classes_float = tf.cast(num_classes, self.float_type)
        self.debug_dict['features'] = features

        if self.hps.center_loss_f_norm == 'f_norm':
            f_norm = self.get_feature_norm_variable(shape=[])
            features = features / f_norm
            # features = tf.nn.l2_normalize(features)
        elif self.hps.center_loss_f_norm == 'l2':
            features = tf.nn.l2_normalize(features)
        elif self.hps.center_loss_f_norm == 'l2_1':
            features = tf.nn.l2_normalize(features, axis=1)

        centers = self.get_center_loss_centers_variable(shape=[num_classes, len_features])

        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)
        if self.hps.is_weighted_center_loss:
            dist_in = tf.reduce_sum(tf.reduce_sum(tf.square(features - centers_batch),
                                                  axis=-1) * self.e_w_ph) / tf.reduce_mean(
                self.e_w_ph)
        else:
            dist_in = tf.nn.l2_loss(features - centers_batch)
        # self.debug_dict['dist_in_mul_batch'] = dist_in
        dist_in = dist_in / batch_size

        centers0 = tf.expand_dims(centers, 0)
        centers1 = tf.expand_dims(centers, 1)
        # gamma = tf.sqrt(tf.nn.l2_loss(centers0 - centers1) / num_classes_float * (num_classes_float - 1.))
        # gamma = tf.constant(1., dtype=self.float_type)

        u_label, u_idx, u_count = tf.unique_with_counts(labels)
        idx_matrix = tf.cast(tf.one_hot(u_idx, tf.shape(u_label)[0]), dtype=self.float_type)
        # [batch_size, 1, bnum_classes]
        idx_tensor = tf.expand_dims(idx_matrix, 1)
        # x_expand [batch_size, dim, 1]
        x_expand = tf.expand_dims(features, -1)
        # x_expand * idx_tensor [batch_size, dim, bnum_classes]
        # f_mean [dim, bnum_classes]
        f_mean = tf.reduce_sum(x_expand * idx_tensor, axis=0) / tf.cast(u_count, self.float_type)
        # bcenters [bnum_classes, dim]
        bcenters = tf.transpose(f_mean, [1, 0])

        bnum_classes = tf.cast(tf.shape(bcenters)[0], dtype=self.float_type)

        bcenters0 = tf.expand_dims(bcenters, 0)
        bcenters1 = tf.expand_dims(bcenters, 1)
        bc_diffs = bcenters0 - bcenters1
        bc_l2s = tf.reduce_sum(tf.square(bc_diffs), axis=-1)
        # dist_out = tf.reduce_sum(bc_l2s) / tf.maximum(1., bnum_classes * (bnum_classes - 1.))
        epsilon = 1e-10
        dist_out = tf.square(
            tf.reduce_sum(tf.sqrt(bc_l2s + epsilon)) / tf.maximum(1., bnum_classes * (bnum_classes - 1.)))
        # dist_out = tf.reduce_sum(tf.square(tf.sqrt(bc_l2s + epsilon) + gamma)) / tf.maximum(1., bnum_classes * (bnum_classes - 1.))

        # dist_out = tf.nn.l2_loss(c_diffs) / tf.maximum(1., bnum_classes * (bnum_classes - 1.))

        # dist_out = tf.sqrt(tf.nn.l2_loss(c_diffs) + epsilon) / tf.maximum(1., bnum_classes * (bnum_classes - 1.))
        loss = (dist_in + epsilon) / (dist_out + epsilon)
        # self.debug_dict['gamma'] = gamma
        # self.debug_dict['dist_in'] = dist_in
        # self.debug_dict['dist_out'] = dist_out
        # self.debug_dict['center_loss7'] = loss

        # loss = tf.maximum(self.hps.dist_margin + dist_in - dist_out, 0)
        return loss

    def update_f_norm_op(self, features, alpha):
        f_norm = self.get_feature_norm_variable(shape=[])
        cur_f_n = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(features), axis=1)))
        n_f_n = (1 - alpha) * f_norm + alpha * cur_f_n
        return f_norm.assign(n_f_n)

    # update center only consider intra-distance
    def intra_update_center_op(self, features, labels, alpha, num_classes):
        len_features = features.get_shape()[1]
        # todo: 这里的标准化并不好，自己实现一种标准化。保持一个变量，代表所有的特征的模的平均值。
        if self.hps.center_loss_f_norm == 'f_norm':
            f_norm = self.get_feature_norm_variable(shape=[])
            # cur_f_n = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(features)), axis=1))
            # n_f_n = (1 - self.feature_norm_alpha_ph) * f_norm + self.feature_norm_alpha_ph * cur_f_n
            features = features / f_norm
            # features = tf.nn.l2_normalize(features)
        elif self.hps.center_loss_f_norm == 'l2':
            features = tf.nn.l2_normalize(features)
        elif self.hps.center_loss_f_norm == 'l2_1':
            features = tf.nn.l2_normalize(features, axis=1)
        centers = self.get_center_loss_centers_variable(shape=[num_classes, len_features])
        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)

        diff = centers_batch - features

        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast((1 + appear_times), self.float_type)
        diff = alpha * diff

        intra_update_c_op = tf.scatter_sub(centers, labels, diff)

        return intra_update_c_op

    def inter_update_center_op(self, features, beta, gamma, num_classes):
        dist_ceiling = 1000
        epsilon = 1e-6
        len_features = features.get_shape()[1]
        centers = self.get_center_loss_centers_variable(shape=[num_classes, len_features])
        centers0 = tf.expand_dims(centers, 0)
        centers1 = tf.expand_dims(centers, 1)
        c_diffs = centers0 - centers1
        c_diffs_norm = c_diffs / (
                tf.sqrt(tf.reduce_sum(tf.square(c_diffs), axis=-1, keepdims=True)) + epsilon)
        c_l2s = tf.reduce_sum(tf.square(c_diffs), axis=-1)
        c_l2s_mask = tf.eye(num_classes, dtype=self.float_type) * dist_ceiling + c_l2s
        # c_diff_norm = c_diff / tf.expand_dims(c_dist_mask)
        column_idx = tf.argmin(c_l2s_mask, axis=1, output_type=tf.int32)
        rng = tf.range(0, num_classes, dtype=tf.int32)
        idx = tf.stack([rng, column_idx], axis=1)
        c_diff_norm = tf.gather_nd(c_diffs_norm, idx)
        c_l2 = tf.expand_dims(tf.gather_nd(c_l2s_mask, idx), -1)
        delta = beta * c_diff_norm * gamma / (gamma + c_l2)
        inter_update_c_op = centers.assign(centers - delta)
        return inter_update_c_op

    def update_center_op2(self, features, labels, alpha, beta, gamma, num_classes):

        if self.hps.center_loss_f_norm == 'f_norm':
            f_norm = self.get_feature_norm_variable(shape=[])
            features = features / f_norm
            # features = tf.nn.l2_normalize(features)
        elif self.hps.center_loss_f_norm == 'l2':
            features = tf.nn.l2_normalize(features)
        elif self.hps.center_loss_f_norm == 'l2_1':
            features = tf.nn.l2_normalize(features, axis=1)
        elif self.hps.center_loss_f_norm == 'avg_l2':
            f_norm = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(features), axis=1)))
            features = features / f_norm
        len_features = features.get_shape()[1]
        centers = self.get_center_loss_centers_variable(shape=[num_classes, len_features])
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        # Inter update
        dist_ceiling = 1000
        epsilon = 1e-8
        centers_ = tf.identity(centers)
        centers0 = tf.expand_dims(centers_, 0)
        centers1 = tf.expand_dims(centers_, 1)
        c_diffs = centers0 - centers1
        c_diffs_norm = c_diffs / (
                tf.sqrt(tf.reduce_sum(tf.square(c_diffs), axis=-1, keepdims=True)) + epsilon)
        c_l2s = tf.reduce_sum(tf.square(c_diffs), axis=-1)
        c_l2s_mask = tf.eye(num_classes, dtype=self.float_type) * dist_ceiling + c_l2s
        # c_diff_norm = c_diff / tf.expand_dims(c_dist_mask)
        column_idx = tf.argmin(c_l2s_mask, axis=1, output_type=tf.int32)
        rng = tf.range(0, num_classes, dtype=tf.int32)
        idx = tf.stack([rng, column_idx], axis=1)
        c_diff_norm = tf.gather_nd(c_diffs_norm, idx)

        # todo: 这里取个根号会不会更好
        c_l2 = tf.expand_dims(tf.gather_nd(c_l2s_mask, idx), -1)
        inter_delta = beta * c_diff_norm * gamma / (gamma + c_l2)
        inter_delta = tf.gather(inter_delta, labels) / tf.cast((1 + appear_times), self.float_type)

        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)

        intra_delta = centers_batch - features

        # todo:  这里的 '1 +' 可能不行
        intra_delta = intra_delta / tf.cast((1 + appear_times), self.float_type)
        intra_delta = alpha * intra_delta

        delta = intra_delta + inter_delta

        up_op = tf.scatter_sub(centers, labels, delta)
        return up_op

        # return inter_update_c_op

    def calc_cos_loss(self, features, labels):
        f = tf.nn.l2_normalize(features, axis=1)
        f0 = tf.expand_dims(f, axis=0)
        f1 = tf.expand_dims(f, axis=1)
        d = tf.reduce_sum(f0 * f1, axis=-1)

        label0 = tf.expand_dims(labels, 0)
        label1 = tf.expand_dims(labels, 1)
        eq_mask = tf.cast(tf.equal(label0, label1), dtype=self.float_type)
        ne_mask = 1. - eq_mask
        eq_mask = eq_mask - tf.eye(tf.shape(eq_mask)[0], tf.shape(eq_mask)[1],
                                   dtype=self.float_type)

        eq_num = tf.maximum(tf.reduce_sum(eq_mask), 1)
        ne_num = tf.maximum(tf.reduce_sum(ne_mask), 1)

        l1 = (eq_num - tf.reduce_sum(eq_mask * d)) / (2.0 * eq_num)
        l2 = (ne_num + tf.reduce_sum(ne_mask * d)) / (2.0 * ne_num)
        return (l1 + l2) / 2.0

    def calc_dist_loss(self, features, labels):
        features = tf.nn.l2_normalize(features)
        f0 = tf.expand_dims(features, axis=0)
        f1 = tf.expand_dims(features, axis=1)
        f_diffs = f0 - f1
        f_l2s = tf.reduce_sum(tf.square(f_diffs), axis=-1)

        label0 = tf.expand_dims(labels, 0)
        label1 = tf.expand_dims(labels, 1)
        eq_mask = tf.cast(tf.equal(label0, label1), dtype=self.float_type)
        ne_mask = 1. - eq_mask
        eq_mask = eq_mask - tf.eye(tf.shape(eq_mask)[0], tf.shape(eq_mask)[1],
                                   dtype=self.float_type)

        eq_num = tf.maximum(tf.reduce_sum(eq_mask), 1.)
        ne_num = tf.maximum(tf.reduce_sum(ne_mask), 1.)

        l_intra = tf.reduce_sum(eq_mask * f_l2s) / eq_num
        l_inter = tf.reduce_sum(ne_mask * f_l2s) / ne_num

        dist_loss = tf.maximum(self.hps.dist_margin + l_intra - l_inter, 0)
        return dist_loss

    def model_fn(self, x, t):
        # return output_d
        # output_d['logits']
        # output_d['h_rnn']
        # output_d['hid_fc']
        # output_d['h_cnn']
        raise NotImplementedError("Please Implement this method")

    def get_metric_d(self):
        with tf.name_scope('emo_accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(self.output_d['logits'], axis=1, output_type=tf.int32), self.e_ph)
            correct_prediction = tf.cast(correct_prediction, self.float_type)
            accuracy = tf.reduce_mean(correct_prediction)
        metric_d = defaultdict(lambda: None)
        metric_d['e_acc'] = accuracy
        return metric_d

    def get_loss_d(self):
        if self.hps.is_weighted_cross_entropy_loss:
            weights = self.e_w_ph
        else:
            weights = 1.0

        with tf.name_scope('loss'):
            # ce_loss = tf.losses.sparse_softmax_cross_entropy(
            #     labels=self.e_ph,
            #     logits=self.output_d['logits'],
            #     weights=weights,
            #     reduction=tf.losses.Reduction.MEAN)

            """
            losses of our paper, center loss is not used, it's just previous experiment.
            but if you want to conduct the center loss, you just need to modify parameter of config file
            """
            ce_loss = self.calc_FL_loss(self.e_ph,self.output_d['logits'],
                                          self.l_ph, self.e_w_ph, len(self.hps.emos))
            # ce_loss = self.calc_soft_loss(self.e_ph, self.output_d['logits'], self.l_ph, self.e_w_ph, len(self.hps.emos))
            # ce_loss = self.calc_Lq_loss(self.e_ph, self.output_d['logits'], self.l_ph, len(self.hps.emos))
            l2_reg_loss = 0.
            if self.hps.is_l2_reg:
                l2_reg_loss = tf.add_n([tf.nn.l2_loss(v)
                                        for v in tf.trainable_variables()
                                        if ('kernel' in v.name) or 'w' in v.name])
                l2_reg_loss = self.hps.l2_reg_lambda * l2_reg_loss
            # center_loss = self.calc_center_loss()
            features = self.output_d[self.hps.features_key]
            center_loss = self.calc_center_loss(features=features, labels=self.e_ph,
                                                num_classes=len(self.hps.emos))
            center_loss2 = self.calc_center_loss2(features=features, labels=self.e_ph,
                                                  num_classes=len(self.hps.emos))
            center_loss3 = self.calc_center_loss3(features=features, labels=self.e_ph,
                                                  num_classes=len(self.hps.emos))
            center_loss4 = self.calc_center_loss4(features=features, labels=self.e_ph)
            center_loss5 = self.calc_center_loss5(features=features, labels=self.e_ph)
            center_loss6 = self.calc_center_loss6(features=features, labels=self.e_ph)
            center_loss7 = self.calc_center_loss7(features=features, labels=self.e_ph,
                                                  num_classes=len(self.hps.emos))
            cos_loss = self.calc_cos_loss(features=features, labels=self.e_ph)
            dist_loss = self.calc_dist_loss(features=features, labels=self.e_ph)
            ce_center_loss = ce_loss + self.center_loss_lambda_ph * center_loss
            ce_center_loss2 = ce_loss + self.center_loss_lambda_ph * center_loss2
            ce_center_loss3 = ce_loss + self.center_loss_lambda_ph * center_loss3
            ce_center_loss4 = ce_loss + self.center_loss_lambda_ph * center_loss4
            ce_center_loss5 = ce_loss + self.center_loss_lambda_ph * center_loss5
            ce_center_loss6 = ce_loss + self.center_loss_lambda_ph * center_loss6
            ce_center_loss7 = ce_loss + self.center_loss_lambda_ph * center_loss7
            cos_loss_lambda = self.cos_loss_lambda_ph
            ce_cos_loss = (1 - cos_loss_lambda) * ce_loss + cos_loss_lambda * cos_loss
            dist_loss_lambda = self.dist_loss_lambda_ph
            ce_dist_loss = (1 - dist_loss_lambda) * ce_loss + dist_loss_lambda * dist_loss
        loss_d = defaultdict(lambda: None)
        loss_d['ce_loss'] = ce_loss
        loss_d['center_loss'] = center_loss
        loss_d['center_loss2'] = center_loss2
        loss_d['center_loss3'] = center_loss3
        loss_d['center_loss4'] = center_loss4
        loss_d['center_loss5'] = center_loss5
        loss_d['center_loss6'] = center_loss6
        loss_d['center_loss7'] = center_loss7
        loss_d['cos_loss'] = cos_loss
        loss_d['dist_loss'] = dist_loss
        loss_d['ce_center_loss'] = ce_center_loss
        loss_d['ce_center_loss2'] = ce_center_loss2
        loss_d['ce_center_loss3'] = ce_center_loss3
        loss_d['ce_center_loss4'] = ce_center_loss4
        loss_d['ce_center_loss5'] = ce_center_loss5
        loss_d['ce_center_loss6'] = ce_center_loss6
        loss_d['ce_center_loss7'] = ce_center_loss7
        loss_d['ce_cos_loss'] = ce_cos_loss
        loss_d['ce_dist_loss'] = ce_dist_loss
        loss_d['l2_reg_loss'] = l2_reg_loss
        return loss_d

    def get_update_op_d(self):
        features = self.output_d[self.hps.features_key]
        intra_update_c_op = self.intra_update_center_op(features=features, labels=self.e_ph,
                                                        alpha=self.center_loss_alpha_ph,
                                                        num_classes=len(self.hps.emos))
        inter_update_c_op = self.inter_update_center_op(features=features,
                                                        beta=self.center_loss_beta_ph,
                                                        gamma=self.center_loss_gamma_ph,
                                                        num_classes=len(self.hps.emos))
        update_center_op2 = self.update_center_op2(features=features, labels=self.e_ph,
                                                   alpha=self.center_loss_alpha_ph,
                                                   beta=self.center_loss_beta_ph,
                                                   gamma=self.center_loss_gamma_ph,
                                                   num_classes=len(self.hps.emos))
        f_norm_update_op = self.update_f_norm_op(features=features,
                                                 alpha=self.feature_norm_alpha_ph)
        update_op_d = defaultdict(lambda: None)
        update_op_d['intra_update_c_op'] = intra_update_c_op
        update_op_d['inter_update_c_op'] = inter_update_c_op
        update_op_d['update_c_op2'] = update_center_op2
        update_op_d['f_norm_update_op'] = f_norm_update_op
        return update_op_d

    def get_train_op_d(self):
        train_op_d = defaultdict(tuple)
        optimizer_type = self.hps.optimizer_type
        if optimizer_type.lower() == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lr_ph)
        elif optimizer_type.lower() == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(self.lr_ph)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.lr_ph)
        if self.hps.is_gradient_clip_norm:
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=5.0)
        with tf.name_scope('optimizer'):
            # tp: train op

            ce_tp = optimizer.minimize(self.loss_d['ce_loss'] + self.loss_d['l2_reg_loss'])
            center_tp = optimizer.minimize(self.loss_d['center_loss'] + self.loss_d['l2_reg_loss'])
            center2_tp = optimizer.minimize(
                self.loss_d['center_loss2'] + self.loss_d['l2_reg_loss'])
            center3_tp = optimizer.minimize(
                self.loss_d['center_loss3'] + self.loss_d['l2_reg_loss'])
            center4_tp = optimizer.minimize(
                self.loss_d['center_loss4'] + self.loss_d['l2_reg_loss'])
            center5_tp = optimizer.minimize(
                self.loss_d['center_loss5'] + self.loss_d['l2_reg_loss'])
            center6_tp = optimizer.minimize(
                self.loss_d['center_loss6'] + self.loss_d['l2_reg_loss'])
            center7_tp = optimizer.minimize(
                self.loss_d['center_loss7'] + self.loss_d['l2_reg_loss'])
            cos_tp = optimizer.minimize(self.loss_d['cos_loss'] + self.loss_d['l2_reg_loss'])
            dist_tp = optimizer.minimize(self.loss_d['dist_loss'] + self.loss_d['l2_reg_loss'])
            ce_center_tp = optimizer.minimize(
                self.loss_d['ce_center_loss'] + self.loss_d['l2_reg_loss'])
            ce_center2_tp = optimizer.minimize(
                self.loss_d['ce_center_loss2'] + self.loss_d['l2_reg_loss'])
            ce_center3_tp = optimizer.minimize(
                self.loss_d['ce_center_loss3'] + self.loss_d['l2_reg_loss'])
            ce_center4_tp = optimizer.minimize(
                self.loss_d['ce_center_loss4'] + self.loss_d['l2_reg_loss'])
            ce_center5_tp = optimizer.minimize(
                self.loss_d['ce_center_loss5'] + self.loss_d['l2_reg_loss'])
            ce_center6_tp = optimizer.minimize(
                self.loss_d['ce_center_loss6'] + self.loss_d['l2_reg_loss'])
            ce_center7_tp = optimizer.minimize(
                self.loss_d['ce_center_loss7'] + self.loss_d['l2_reg_loss'])
            ce_cos_tp = optimizer.minimize(self.loss_d['ce_cos_loss'] + self.loss_d['l2_reg_loss'])
            ce_dist_tp = optimizer.minimize(
                self.loss_d['ce_dist_loss'] + self.loss_d['l2_reg_loss'])

        if self.hps.center_loss_f_norm == 'f_norm':
            center_utp = (
                self.update_op_d['f_norm_update_op'], self.update_op_d['inter_update_c_op'],
                self.update_op_d['intra_update_c_op'], center_tp)
            ce_center_utp = (
                self.update_op_d['f_norm_update_op'], self.update_op_d['inter_update_c_op'],
                self.update_op_d['intra_update_c_op'], ce_center_tp)
        else:

            # center_utp = (
            #     self.update_op_d['inter_update_c_op'], self.update_op_d['intra_update_c_op'],
            #     center_tp)
            # ce_center_utp = (
            #     self.update_op_d['inter_update_c_op'], self.update_op_d['intra_update_c_op'],
            #     ce_center_tp)
            center_utp = (
                self.update_op_d['intra_update_c_op'],
                center_tp)
            ce_center_utp = (
                self.update_op_d['intra_update_c_op'], ce_center_tp)

        if self.hps.is_merge_center_loss_centers and self.hps.center_loss_f_norm == 'f_norm':
            # ce_tp2 = ()
            train_op_d['ce_tp'] = (self.update_op_d['f_norm_update_op'], ce_tp)
        else:
            train_op_d['ce_tp'] = ce_tp

        train_op_d['center_tp'] = center_tp
        train_op_d['center_utp'] = center_utp
        train_op_d['center_u2tp'] = (self.update_op_d['update_c_op2'], center_tp)
        train_op_d['center2_tp'] = center2_tp
        train_op_d['center2_utp'] = (self.update_op_d['intra_update_c_op'], center2_tp)
        train_op_d['center3_tp'] = center3_tp
        train_op_d['center3_utp'] = (self.update_op_d['intra_update_c_op'], center3_tp)
        train_op_d['center4_tp'] = center4_tp
        train_op_d['center5_tp'] = center5_tp
        train_op_d['center6_tp'] = center6_tp
        train_op_d['center7_tp'] = center7_tp
        train_op_d['center7_utp'] = (self.update_op_d['intra_update_c_op'], center7_tp)
        train_op_d['cos_tp'] = cos_tp
        train_op_d['dist_tp'] = dist_tp
        train_op_d['ce_center_tp'] = ce_center_tp
        train_op_d['ce_center_utp'] = ce_center_utp
        train_op_d['ce_center_u2tp'] = (self.update_op_d['update_c_op2'], ce_center_tp)
        train_op_d['ce_center2_utp'] = (self.update_op_d['intra_update_c_op'], ce_center2_tp)
        train_op_d['ce_center3_tp'] = ce_center3_tp
        train_op_d['ce_center3_utp'] = (self.update_op_d['intra_update_c_op'], ce_center3_tp)
        train_op_d['ce_center4_tp'] = ce_center4_tp
        train_op_d['ce_center5_tp'] = ce_center5_tp
        train_op_d['ce_center6_tp'] = ce_center6_tp
        train_op_d['ce_center7_utp'] = (self.update_op_d['intra_update_c_op'], ce_center7_tp)
        train_op_d['ce_cos_tp'] = ce_cos_tp
        train_op_d['ce_dist_tp'] = ce_dist_tp
        return train_op_d

    def get_grad_d(self):
        grad_d = defaultdict(lambda: None)
        grad_d['ce2hrnn'] = tf.gradients(self.loss_d['ce_loss'], self.output_d['h_rnn'])[0]
        grad_d['ce2hcnn'] = tf.gradients(self.loss_d['ce_loss'], self.output_d['h_cnn'])[0]
        grad_d['center2hrnn'] = tf.gradients(self.loss_d['center_loss'], self.output_d['h_rnn'])[0]
        grad_d['center2hcnn'] = tf.gradients(self.loss_d['center_loss'], self.output_d['h_cnn'])[0]
        grad_d['center22hrnn'] = \
            tf.gradients(self.loss_d['center_loss2'], self.output_d['h_rnn'])[0]
        grad_d['center22hcnn'] = \
            tf.gradients(self.loss_d['center_loss2'], self.output_d['h_cnn'])[0]
        grad_d['center32hrnn'] = \
            tf.gradients(self.loss_d['center_loss3'], self.output_d['h_rnn'])[0]
        grad_d['center32hcnn'] = \
            tf.gradients(self.loss_d['center_loss3'], self.output_d['h_cnn'])[0]
        grad_d['center42hrnn'] = \
            tf.gradients(self.loss_d['center_loss4'], self.output_d['h_rnn'])[0]
        grad_d['center42hcnn'] = \
            tf.gradients(self.loss_d['center_loss4'], self.output_d['h_cnn'])[0]
        grad_d['center52hrnn'] = \
            tf.gradients(self.loss_d['center_loss5'], self.output_d['h_rnn'])[0]
        grad_d['center52hcnn'] = \
            tf.gradients(self.loss_d['center_loss5'], self.output_d['h_cnn'])[0]
        grad_d['center62hrnn'] = \
            tf.gradients(self.loss_d['center_loss6'], self.output_d['h_rnn'])[0]
        grad_d['center62hcnn'] = \
            tf.gradients(self.loss_d['center_loss6'], self.output_d['h_cnn'])[0]
        grad_d['center72hrnn'] = \
            tf.gradients(self.loss_d['center_loss6'], self.output_d['h_rnn'])[0]
        grad_d['center72hcnn'] = \
            tf.gradients(self.loss_d['center_loss6'], self.output_d['h_cnn'])[0]
        grad_d['cos2hrnn'] = tf.gradients(self.loss_d['cos_loss'], self.output_d['h_rnn'])[0]
        grad_d['cos2hcnn'] = tf.gradients(self.loss_d['cos_loss'], self.output_d['h_cnn'])[0]
        grad_d['dist2hrnn'] = tf.gradients(self.loss_d['dist_loss'], self.output_d['h_rnn'])[0]
        grad_d['dist2hcnn'] = tf.gradients(self.loss_d['dist_loss'], self.output_d['h_cnn'])[0]
        return grad_d

    def get_train_merged(self):
        summary_list = list()
        if isinstance(self.hps.train_output_summ_keys, list) or isinstance(
                self.hps.train_output_summ_keys, CommentedSeq):
            with tf.name_scope('output'):
                for k in self.hps.train_output_summ_keys:
                    with tf.name_scope(k):
                        v_summ_list = variable_summaries(self.output_d[k])
                    summary_list += v_summ_list
        if isinstance(self.hps.train_grad_summ_keys, list) or isinstance(
                self.hps.train_grad_summ_keys, CommentedSeq):
            with tf.name_scope('grad'):
                for k in self.hps.train_grad_summ_keys:
                    with tf.name_scope(k):
                        # for i, ele in zip(range(self.grad_d))
                        v_summ_list = variable_summaries(self.grad_d[k])
                    summary_list += v_summ_list
        if isinstance(self.hps.train_metric_summ_keys, list) or isinstance(
                self.hps.train_metric_summ_keys, CommentedSeq):
            with tf.name_scope('metric'):
                for k in self.hps.train_metric_summ_keys:
                    # with tf.name_scope(k):
                    summ = tf.summary.scalar(k, self.metric_d[k])
                    summary_list.append(summ)
        if isinstance(self.hps.train_loss_summ_keys, list) or isinstance(
                self.hps.train_loss_summ_keys, CommentedSeq):
            with tf.name_scope('loss'):
                for k in self.hps.train_loss_summ_keys:
                    summ = tf.summary.scalar(k, self.loss_d[k])
                    summary_list.append(summ)
        if self.hps.is_merge_center_loss_centers:
            with tf.name_scope('center_loss_dist_squares'):
                features = self.output_d[self.hps.features_key]
                len_features = features.get_shape()[1]
                shape = [len(self.hps.emos), len_features]
                centers = self.get_center_loss_centers_variable(shape=shape)
                centers0 = tf.expand_dims(centers, 0)
                centers1 = tf.expand_dims(centers, 1)
                c_diffs = centers0 - centers1
                c_l2s = tf.reduce_sum(tf.square(c_diffs), axis=-1)
                dist_m = tf.reduce_mean(c_l2s) * 16. / 9.
                c_l2s_mask = tf.eye(tf.shape(c_l2s)[0], dtype=self.float_type) * dist_m + c_l2s
                v_summ_list = variable_summaries(c_l2s_mask)
                summary_list += v_summ_list

        return tf.summary.merge(summary_list)

    def build_graph(self):
        self.output_d = self.model_fn(self.x_ph, self.t_ph)
        self.metric_d = self.get_metric_d()
        self.loss_d = self.get_loss_d()
        self.update_op_d = self.get_update_op_d()
        self.train_op_d = self.get_train_op_d()
        self.grad_d = self.get_grad_d()
        self.train_merged = self.get_train_merged()


class CGRUFCModel(BaseCRModel):
    def cnn(self, input, seq_lens):
        raise NotImplementedError('cnn function not implements yet')

    def rnn(self, inputs, seq_lens):
        rnn_hidden_size = 128
        with tf.name_scope('rnn'):
            rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_hidden_size)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, inputs,
                                                             sequence_length=seq_lens,
                                                             dtype=self.float_type,
                                                             swap_memory=True)
            # rng = tf.range(0, tf.shape(seq_lens)[0])
            # indexes = tf.stack([rng, seq_lens - 1], axis=1, name="indexes")
            # fw_outputs = tf.gather_nd(outputs[0], indexes)
            # bw_outputs = outputs[1][:, 0]
            # outputs_concat = tf.concat([fw_outputs, bw_outputs], axis=1)
            gru, alphas = vcu.attention(inputs=outputs, attention_size=1, return_alphas=True)
        return gru

    def fc(self, inputs):
        out_dim = len(self.hps.emos)
        in_dim = 256
        fc_hidden = 64
        with tf.name_scope('fc1'):
            w_fc1 = self.weight_variable([in_dim, fc_hidden])
            b_fc1 = self.bias_variable([fc_hidden])
            h_fc1 = tf.matmul(inputs, w_fc1) + b_fc1
            h_fc1_drop = tf.nn.dropout(tf.nn.relu(h_fc1), self.fc_kprob_ph)
        with tf.name_scope('fc2'):
            w_fc2 = self.weight_variable([fc_hidden, out_dim])
            b_fc2 = self.bias_variable([out_dim])
            h_fc2 = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
        h_fc = h_fc2
        hid_fc = h_fc1
        return h_fc, hid_fc

    def model_fn(self, x, t):
        # return output_d
        # output_d['logits']
        # output_d['h_rnn']
        # output_d['hid_fc']
        # output_d['h_cnn']
        # raise NotImplementedError("Please Implement this method")
        h_cnn, seq_lens = self.cnn(x, t)
        h_rnn = self.rnn(h_cnn, seq_lens)
        logits, hid_fc = self.fc(h_rnn)
        output_d = defaultdict(lambda: None)
        output_d['h_cnn'] = h_cnn
        output_d['h_rnn'] = h_rnn
        output_d['logits'] = logits
        output_d['hid_fc'] = hid_fc
        return output_d


# CNN: [3, 3, 1, 8], [3, 3, 8, 8], [3, 3, 8, 16], [3, 3, 16, 16]; max_pool 2 * 2
# RNN: BiGRU 128
# FC: 128 -> 64 ->4
class CRModel1(CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('CRModel1')
        h_conv = tf.expand_dims(inputs, 3)
        cnn_kernels = [[3, 3, 1, 8], [3, 3, 8, 8], [3, 3, 8, 16], [3, 3, 16, 16]]
        with tf.name_scope('conv'):
            for cnn_kernel in cnn_kernels:
                w_conv = self.weight_variable(cnn_kernel)
                b_conv = self.bias_variable(cnn_kernel[-1:])
                h_conv, seq_lens = vcu.var_conv2d_v2(h_conv, w=w_conv, bias=b_conv,
                                                     seq_length=seq_lens, strides=[1, 1, 1, 1],
                                                     padding='SAME',
                                                     is_training=self.is_training_ph,
                                                     activation_fn=tf.nn.relu,
                                                     is_bn=self.hps.is_bn,
                                                     is_mask=self.hps.is_var_cnn_mask)
                h_conv, seq_lens = vcu.var_max_pool(h_conv, ksize=[1, 2, 2, 1],
                                                    strides=[1, 2, 2, 1],
                                                    padding='SAME', seq_length=seq_lens)
            h_cnn = tf.reshape(h_conv,
                               [tf.shape(h_conv)[0], -1, h_conv.shape[2] * h_conv.shape[3]])
        return h_cnn, seq_lens

    # def model_fn(self, x, t):
    #     # return output_d
    #     # output_d['logits']
    #     # output_d['h_rnn']
    #     # output_d['hid_fc']
    #     # output_d['h_cnn']
    #     # raise NotImplementedError("Please Implement this method")
    #     h_cnn, seq_lens = self.cnn(x, t)
    #     h_rnn = self.rnn(h_cnn, seq_lens)
    #     logits, hid_fc = self.fc(h_rnn)
    #     output_d = defaultdict(lambda: None)
    #     output_d['h_cnn'] = h_cnn
    #     output_d['h_rnn'] = h_rnn
    #     output_d['logits'] = logits
    #     output_d['hid_fc'] = hid_fc
    #     return output_d


# CNN: [[7, 7, 1, 16], [3, 3, 16, 16], [3, 3, 16, 32], [3, 3, 32, 32]], strides: [1, 2, 2, 1]
class CRModel2(CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('CRModel2')
        h = tf.expand_dims(inputs, 3)
        i = 0
        cnn_kernels = [[7, 7, 1, 16], [3, 3, 16, 16], [3, 3, 16, 32], [3, 3, 32, 32]]
        for cnn_kernel in cnn_kernels:
            i += 1
            with tf.name_scope('conv' + str(i)):
                w = self.weight_variable(cnn_kernel)
                b = self.bias_variable(cnn_kernel[-1:])
                h, seq_lens = vcu.var_conv2d_v2(h, w=w, bias=b, seq_length=seq_lens,
                                                strides=[1, 2, 2, 1], padding='SAME',
                                                is_training=self.is_training_ph,
                                                activation_fn=tf.nn.relu,
                                                is_bn=self.hps.is_bn,
                                                is_mask=self.hps.is_var_cnn_mask)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens


# CNN
# kernels: [[7, 7, 1, 16], [3, 3, 16, 16], [3, 3, 16, 32], [3, 3, 32, 32]]
# strides: [[1, 2, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
# is_poolings: [False, True, True, True]
class CRModel3(CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('CRModel3')
        h = tf.expand_dims(inputs, 3)
        i = 0
        kernels = [[7, 7, 1, 16], [3, 3, 16, 16], [3, 3, 16, 32], [3, 3, 32, 32]]
        strides = [[1, 2, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        is_poolings = [False, True, True, True]
        for kernel, s, is_pool in zip(kernels, strides, is_poolings):
            i += 1
            with tf.name_scope('conv' + str(i)):
                w = self.weight_variable(kernel)
                if self.hps.is_bn:
                    b = None
                else:
                    b = self.bias_variable(kernel[-1:])
                h, seq_lens = vcu.var_conv2d_v2(h, w=w, bias=b, seq_length=seq_lens,
                                                strides=s, padding='SAME',
                                                is_training=self.is_training_ph,
                                                activation_fn=tf.nn.relu,
                                                is_bn=self.hps.is_bn,
                                                is_mask=self.hps.is_var_cnn_mask)
                if is_pool:
                    h, seq_lens = vcu.var_max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME', seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens
