import collections
import operator
import os
import time
from collections import defaultdict
from itertools import accumulate

import numpy as np
import tensorflow as tf
from ruamel.yaml.comments import CommentedSeq
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from cr_model_v2 import cr_model
from cr_model_v2 import data_set
from utils import log_util
from utils import post_process


class VarHps(collections.namedtuple('VarHps',
                                    ('lr', 'cos_loss_lambda', 'center_loss_lambda',
                                     'dist_loss_lambda', 'center_loss_alpha', 'center_loss_beta',
                                     'center_loss_gamma', 'feature_norm_alpha','beta_B', 'beta_C'))):
    pass


class CRModelRun(object):

    def __init__(self, model):
        assert isinstance(model, cr_model.BaseCRModel)
        self.model = model
        self.hps = self.model.hps
        # hps = self.hps
        if self.hps.float_type == '16':
            np_float_type = np.float16
            tf_float_type = tf.float16
        elif self.hps.float_type == '64':
            np_float_type = np.float64
            tf_float_type = tf.float64
        else:
            np_float_type = np.float32
            tf_float_type = tf.float32
        self.np_float_type = np_float_type
        self.tf_float_type = tf_float_type
        self.start_time = time.time()
        self.ckpt_path = os.path.join(self.hps.ckpt_dir, self.hps.id_str)
        self.best_metric = 0.0
        self.best_metric_step = -1
        self.best_metric_ckpt_path = os.path.join(self.hps.bestmetric_ckpt_dir,
                                                  self.hps.id_str)
        self.best_loss = np.float('inf')
        self.best_loss_step = -1
        self.best_loss_ckpt_path = os.path.join(self.hps.bestloss_ckpt_dir,
                                                self.hps.id_str)
        self.ckpt_metric_k = self.hps.ckpt_metric_k
        self.ckpt_loss_k = self.hps.ckpt_loss_k
        self.saver = None
        self.train_writer = None
        self.dev_writer = None
        self.test_writer = None
        self.logger = log_util.MyLogger(self.hps)

        # placeholder
        # used for merge dev set and test set
        self.metric_ph_d = defaultdict(lambda: None)
        self.metric_ph_d['wa'] = tf.placeholder(self.tf_float_type, shape=[], name='wa_ph')
        self.metric_ph_d['ua'] = tf.placeholder(self.tf_float_type, shape=[], name='ua_ph')
        self.metric_ph_d['wua'] = tf.placeholder(self.tf_float_type, shape=[], name='wua_ph')

        self.loss_ph_d = defaultdict(lambda: None)
        self.loss_ph_d['ce_loss'] = tf.placeholder(self.tf_float_type, shape=[],
                                                   name='ce_loss_ph')
        self.loss_ph_d['center_loss'] = tf.placeholder(self.tf_float_type, shape=[],
                                                       name='center_loss_ph')
        self.loss_ph_d['center_loss2'] = tf.placeholder(self.tf_float_type, shape=[],
                                                        name='center_loss2_ph')
        self.loss_ph_d['center_loss3'] = tf.placeholder(self.tf_float_type, shape=[],
                                                        name='center_loss3_ph')
        self.loss_ph_d['center_loss4'] = tf.placeholder(self.tf_float_type, shape=[],
                                                        name='center_loss4_ph')
        self.loss_ph_d['center_loss5'] = tf.placeholder(self.tf_float_type, shape=[],
                                                        name='center_loss5_ph')
        self.loss_ph_d['center_loss6'] = tf.placeholder(self.tf_float_type, shape=[],
                                                        name='center_loss6_ph')
        self.loss_ph_d['center_loss7'] = tf.placeholder(self.tf_float_type, shape=[],
                                                        name='center_loss7_ph')
        self.loss_ph_d['cos_loss'] = tf.placeholder(self.tf_float_type, shape=[],
                                                    name='cos_loss_ph')
        self.loss_ph_d['dist_loss'] = tf.placeholder(self.tf_float_type, shape=[],
                                                     name='dist_loss_ph')
        self.loss_ph_d['ce_center_loss'] = tf.placeholder(self.tf_float_type, shape=[],
                                                          name='ce_center_loss_ph')
        self.loss_ph_d['ce_center_loss2'] = tf.placeholder(self.tf_float_type, shape=[],
                                                           name='ce_center_loss2_ph')
        self.loss_ph_d['ce_center_loss3'] = tf.placeholder(self.tf_float_type, shape=[],
                                                           name='ce_center_loss3_ph')
        self.loss_ph_d['ce_center_loss4'] = tf.placeholder(self.tf_float_type, shape=[],
                                                           name='ce_center_loss4_ph')
        self.loss_ph_d['ce_center_loss5'] = tf.placeholder(self.tf_float_type, shape=[],
                                                           name='ce_center_loss5_ph')
        self.loss_ph_d['ce_center_loss6'] = tf.placeholder(self.tf_float_type, shape=[],
                                                           name='ce_center_loss6_ph')
        self.loss_ph_d['ce_center_loss7'] = tf.placeholder(self.tf_float_type, shape=[],
                                                           name='ce_center_loss7_ph')
        self.loss_ph_d['ce_cos_loss'] = tf.placeholder(self.tf_float_type, shape=[],
                                                       name='ce_cos_loss_ph')
        self.loss_ph_d['ce_dist_loss'] = tf.placeholder(self.tf_float_type, shape=[],
                                                        name='ce_dist_loss_ph')
        self.loss_ph_d['l2_reg_loss'] = tf.placeholder(self.tf_float_type, shape=[],
                                                       name='l2_reg_loss_ph')
        self.eval_merged = self.get_eval_merged(self.hps.eval_metric_ks,
                                                self.hps.eval_loss_ks)

    def get_eval_merged(self, merged_metric_ks, merged_loss_ks):

        summary_list = list()

        if isinstance(merged_metric_ks, list) or isinstance(merged_metric_ks, CommentedSeq):
            with tf.name_scope('metric'):
                for k in merged_metric_ks:
                    summ = tf.summary.scalar(k, self.metric_ph_d[k])
                    summary_list.append(summ)
        if isinstance(merged_loss_ks, list) or isinstance(merged_loss_ks, CommentedSeq):
            with tf.name_scope('loss'):
                for k in merged_loss_ks:
                    summ = tf.summary.scalar(k, self.loss_ph_d[k])
                    summary_list.append(summ)
        return tf.summary.merge(summary_list)

    def get_eval_merged_feed_dict(self, metric_d, loss_d, merged_metric_ks, merged_loss_ks):
        feed_dict = {}
        if isinstance(merged_metric_ks, list) or isinstance(merged_metric_ks, CommentedSeq):
            for k in merged_metric_ks:
                feed_dict[self.metric_ph_d[k]] = metric_d[k]
        if isinstance(merged_metric_ks, list) or isinstance(merged_loss_ks, CommentedSeq):
            for k in merged_loss_ks:
                feed_dict[self.loss_ph_d[k]] = loss_d[k]
        return feed_dict

    def init_saver(self):
        max_to_keep = 5
        if 'saver_max_to_keep' in self.hps:
            max_to_keep = self.hps.saver_max_to_keep
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    def init_summ_writer(self):
        self.train_writer = tf.summary.FileWriter(os.path.join(self.hps.tf_log_dir, 'train'))
        self.dev_writer = tf.summary.FileWriter(os.path.join(self.hps.tf_log_dir, 'dev'))
        self.test_writer = tf.summary.FileWriter(os.path.join(self.hps.tf_log_dir, 'test'))

    def exit(self):
        self.logger.close()
        self.train_writer.close()
        self.dev_writer.close()
        self.test_writer.close()

    @staticmethod
    def get_cur_hp(cur_i, step_list, hp_list):
        acc_steps = accumulate(step_list, operator.add)
        for acc_step, hp in zip(acc_steps, hp_list):
            if cur_i < acc_step:
                return hp
        return hp_list[-1]

    def get_cur_var_hps(self, cur_i):
        lr = self.get_cur_hp(cur_i, self.hps.lr_steps, self.hps.lrs)
        cos_loss_lambda = self.get_cur_hp(cur_i, self.hps.cos_loss_lambda_steps,
                                          self.hps.cos_loss_lambdas)
        center_loss_lambda = self.get_cur_hp(cur_i, self.hps.center_loss_lambda_steps,
                                             self.hps.center_loss_lambdas)
        dist_loss_lambda = self.get_cur_hp(cur_i, self.hps.dist_loss_lambda_steps,
                                           self.hps.dist_loss_lambdas)
        center_loss_alpha = self.get_cur_hp(cur_i, self.hps.center_loss_alpha_steps,
                                            self.hps.center_loss_alhpas)
        center_loss_beta = self.get_cur_hp(cur_i, self.hps.center_loss_beta_steps,
                                           self.hps.center_loss_betas)
        center_loss_gamma = self.get_cur_hp(cur_i, self.hps.center_loss_gamma_steps,
                                            self.hps.center_loss_gammas)
        feature_norm_alpha = self.get_cur_hp(cur_i, self.hps.feature_norm_alpha_steps,
                                             self.hps.feature_norm_alphas)
        beta_B = self.get_cur_hp(cur_i, self.hps.beta_B_steps,
                                 self.hps.beta_B)
        beta_C = self.get_cur_hp(cur_i, self.hps.beta_C_steps,
                                 self.hps.beta_C)
        return VarHps(
            lr=lr,
            cos_loss_lambda=cos_loss_lambda,
            center_loss_lambda=center_loss_lambda,
            dist_loss_lambda=dist_loss_lambda,
            center_loss_alpha=center_loss_alpha,
            center_loss_beta=center_loss_beta,
            center_loss_gamma=center_loss_gamma,
            feature_norm_alpha=feature_norm_alpha,
            beta_B = beta_B,
            beta_C=beta_C
        )

    @staticmethod
    def _dict_list_append(dl, d):
        for k, v in d.items():
            dl[k].append(v)

    @staticmethod
    def _dict_list_weighted_avg(dl, w):
        d = dict()
        for k, v in dl.items():
            value = float(np.dot(v, w) / np.sum(w))
            d[k] = value
        return d

    def eval(self, batched_iter, var_hps, session):
        assert isinstance(batched_iter, data_set.BatchedIter)
        assert isinstance(var_hps, VarHps)
        model = self.model
        prs = list()
        gts = list()

        if isinstance(self.hps.eval_loss_ks, list) or isinstance(self.hps.eval_loss_ks,
                                                                 CommentedSeq):
            model_loss_d = dict()
            for k in self.hps.eval_loss_ks:
                model_loss_d[k] = self.model.loss_d[k]
        else:
            model_loss_d = self.model.loss_d
        weights = list()
        losses_d = defaultdict(list)

        session.run(batched_iter.initializer)
        MAX_LOOP = 9999
        for _ in range(MAX_LOOP):#9
            try:
                batched_input = session.run(batched_iter.BatchedInput)
                batch_len = batched_input.x.shape[0]
                batched_logits, batched_loss_d = session.run(
                    (model.output_d['logits'], model_loss_d), feed_dict={
                        model.fc_kprob_ph: 1.0,
                        model.x_ph: batched_input.x.astype(self.np_float_type),
                        model.t_ph: batched_input.t,
                        model.e_ph: batched_input.e,
                        model.e_w_ph: batched_input.w.astype(self.np_float_type),
                        model.l_ph: batched_input.l,
                        model.is_training_ph: False,
                        model.cos_loss_lambda_ph: var_hps.cos_loss_lambda,
                        model.center_loss_lambda_ph: var_hps.center_loss_lambda,
                        model.center_loss_alpha_ph: var_hps.center_loss_alpha,
                        model.center_loss_beta_ph: var_hps.center_loss_beta,
                        model.center_loss_gamma_ph: var_hps.center_loss_gamma,
                        model.dist_loss_lambda_ph: var_hps.dist_loss_lambda,
                        model.beta_B: var_hps.beta_B,
                        model.beta_C: var_hps.beta_C

                    })

                batched_pr = np.argmax(batched_logits, 1)
                gts += list(batched_input.e)
                prs += list(batched_pr)
                self._dict_list_append(losses_d, batched_loss_d)
                weights.append(batch_len)
            except tf.errors.OutOfRangeError:
                break

        loss_d = self._dict_list_weighted_avg(losses_d, weights)
        wa = accuracy_score(y_true=gts, y_pred=prs)
        ua = recall_score(y_true=gts, y_pred=prs, average='macro')
        f1 = f1_score(y_true=gts, y_pred=prs, average='macro')
        metric_d = dict()
        metric_d['wa'] = wa
        metric_d['ua'] = ua
        metric_d['F1'] = f1
        return metric_d, loss_d

    # calc confusion matrix, save some result
    def process_result(self, test_iter, var_hps, session):
        assert isinstance(test_iter, data_set.BatchedIter)
        assert isinstance(var_hps, VarHps)
        model = self.model

        gts = list()
        prs = list()

        session.run(test_iter.initializer)
        model_logits = model.output_d['logits']

        MAX_LOOP = 999
        for _ in range(MAX_LOOP):
            try:
                batched_input = session.run(test_iter.BatchedInput)
                batched_logits = session.run(
                    model_logits, feed_dict={
                        model.fc_kprob_ph: 1.0,
                        model.x_ph: batched_input.x.astype(self.np_float_type),
                        model.t_ph: batched_input.t,
                        model.e_ph: batched_input.e,
                        model.e_w_ph: batched_input.w.astype(self.np_float_type),
                        model.l_ph: batched_input.l,
                        model.is_training_ph: False,
                        model.cos_loss_lambda_ph: var_hps.cos_loss_lambda,
                        model.center_loss_lambda_ph: var_hps.center_loss_lambda,
                        model.dist_loss_lambda_ph: var_hps.dist_loss_lambda,
                    })
                batched_pr = np.argmax(batched_logits, 1)
                gts += list(batched_input.e)
                prs += list(batched_pr)

            except tf.errors.OutOfRangeError:
                break
        gt_np = np.array(gts)
        pr_np = np.array(prs)

        if self.hps.is_save_emo_result:
            result_npy_dir = self.hps.result_npy_dir
            id_str = self.hps.id_str
            gt_path = os.path.join(result_npy_dir, 'gt_' + id_str + '.npy')
            pr_path = os.path.join(result_npy_dir, 'pr_' + id_str + '.npy')
            np.save(gt_path, gt_np)
            np.save(pr_path, pr_np)
        matrix, _ = post_process.print_csv_confustion_matrix(gt_np, pr_np, self.hps.emos)
        result_npy_path = os.path.join(self.hps.result_matrix_dir,
                                       'matrix_' + self.hps.id_str + '.npy')
        np.save(result_npy_path, matrix)
        result_txt_path = os.path.join(self.hps.log_dir, 'result_' + self.hps.id_str + '.txt')
        with open(result_txt_path, 'w') as outf:
            post_process.print_csv_confustion_matrix(gt_np, pr_np, self.hps.emos, file=outf)
        self.logger.log('id str', self.hps.id_str, level=2)
        self.logger.log('')

    def save_feature(self, batched_iter, var_hps, session, key_str='train'):
        assert isinstance(batched_iter, data_set.BatchedIter)
        assert isinstance(var_hps, VarHps)
        model = self.model

        gts = list()
        features = list()
        model_batched_feature = model.output_d[self.hps.features_key]

        session.run(batched_iter.initializer)

        MAX_LOOP = 999
        for _ in range(MAX_LOOP):
            try:
                batched_input = session.run(batched_iter.BatchedInput)
                batched_feature = session.run(
                    model_batched_feature, feed_dict={
                        model.fc_kprob_ph: 1.0,
                        model.x_ph: batched_input.x.astype(self.np_float_type),
                        model.t_ph: batched_input.t,
                        model.e_ph: batched_input.e,
                        model.e_w_ph: batched_input.w.astype(self.np_float_type),
                        model.l_ph: batched_input.l,
                        model.is_training_ph: False,
                        model.cos_loss_lambda_ph: var_hps.cos_loss_lambda,
                        model.center_loss_lambda_ph: var_hps.center_loss_lambda,
                        model.dist_loss_lambda_ph: var_hps.dist_loss_lambda,
                    })

                gts += list(batched_input.e)
                features += list(batched_feature)
            except tf.errors.OutOfRangeError:
                break
        gt_np = np.array(gts)

        feature_np = np.vstack(features)

        result_npy_dir = self.hps.result_npy_dir
        id_str = self.hps.id_str
        gt_path = os.path.join(result_npy_dir, key_str + '_' + 'gt_' + id_str + '.npy')
        feature_path = os.path.join(result_npy_dir, key_str + '_' + 'feature_' + id_str + '.npy')
        np.save(gt_path, gt_np)
        np.save(feature_path, feature_np)

    def train(self, start_i, session, d_set):
        assert isinstance(d_set, data_set.DataSet)
        train_iter = d_set.get_train_iter()
        dev_iter = d_set.get_dev_iter()
        test_iter = d_set.get_test_iter()
        session.run(train_iter.initializer)
        train_acc = []
        train_loss = []
        valid_acc = []
        valid_loss = []
        model = self.model
        if isinstance(self.hps.eval_loss_ks, list) or isinstance(self.hps.eval_loss_ks,
                                                                 CommentedSeq):
            model_loss_d = dict()
            for k in self.hps.eval_loss_ks:
                model_loss_d[k] = model.loss_d[k]
        else:
            model_loss_d = model.loss_d

        for i in range(start_i, self.hps.max_steps):
            #train_op_k ce_center_utp
            train_op_k = self.get_cur_hp(i, self.hps.train_op_steps, self.hps.train_op_ks)

            train_op = model.train_op_d[train_op_k]
            var_hps = self.get_cur_var_hps(i)
            batch_input = session.run(train_iter.BatchedInput)
            if self.hps.is_log_debug:
                # todo: debug
                # print(session.run(self.model.get_center_loss_centers_variable(shape=[4, 4])))
                debug_dict = session.run(model.debug_dict,
                                         feed_dict={
                                             model.fc_kprob_ph: self.hps.fc_kprob,
                                             model.lr_ph: var_hps.lr,
                                             model.x_ph: batch_input.x.astype(self.np_float_type),
                                             model.t_ph: batch_input.t,
                                             model.e_ph: batch_input.e,
                                             model.e_w_ph: batch_input.w.astype(self.np_float_type),
                                             model.is_training_ph: True,
                                             model.cos_loss_lambda_ph: var_hps.cos_loss_lambda,
                                             model.dist_loss_lambda_ph: var_hps.dist_loss_lambda,
                                             model.center_loss_lambda_ph: var_hps.center_loss_lambda,
                                             model.center_loss_alpha_ph: var_hps.center_loss_alpha,
                                             model.center_loss_beta_ph: var_hps.center_loss_beta,
                                             model.center_loss_gamma_ph: var_hps.center_loss_gamma,
                                             model.feature_norm_alpha_ph: var_hps.feature_norm_alpha,
                                         })
                print('======== debug dict: ==========')
                for k, v in debug_dict.items():
                    print(k)
                    print(v)
                print('======== debug dict: ==========')


            if i % self.hps.train_eval_interval == 0:
                if self.hps.is_tflog:
                    summ, batch_e_acc, batch_loss_d, _ = session.run(
                        (model.train_merged, model.metric_d['e_acc'], model_loss_d, train_op),
                        feed_dict={
                            model.fc_kprob_ph: self.hps.fc_kprob,
                            model.lr_ph: var_hps.lr,
                            model.x_ph: batch_input.x.astype(self.np_float_type),
                            model.t_ph: batch_input.t,
                            model.e_ph: batch_input.e,
                            model.e_w_ph: batch_input.w.astype(self.np_float_type),
                            model.is_training_ph: True,
                            model.cos_loss_lambda_ph: var_hps.cos_loss_lambda,
                            model.dist_loss_lambda_ph: var_hps.dist_loss_lambda,
                            model.center_loss_lambda_ph: var_hps.center_loss_lambda,
                            model.center_loss_alpha_ph: var_hps.center_loss_alpha,
                            model.center_loss_beta_ph: var_hps.center_loss_beta,
                            model.center_loss_gamma_ph: var_hps.center_loss_gamma,
                            model.feature_norm_alpha_ph: var_hps.feature_norm_alpha,
                            model.beta_B: var_hps.beta_B,
                            model.beta_C: var_hps.beta_C
                        })
                else:
                    batch_e_acc, batch_loss_d, _ = session.run(
                        (model.metric_d['e_acc'], model_loss_d, train_op),
                        feed_dict={
                            model.fc_kprob_ph: self.hps.fc_kprob,
                            model.lr_ph: var_hps.lr,
                            model.x_ph: batch_input.x.astype(self.np_float_type),
                            model.t_ph: batch_input.t,
                            model.e_ph: batch_input.e,
                            model.e_w_ph: batch_input.w.astype(self.np_float_type),
                            model.l_ph: batch_input.l,
                            model.is_training_ph: True,
                            model.cos_loss_lambda_ph: var_hps.cos_loss_lambda,
                            model.dist_loss_lambda_ph: var_hps.dist_loss_lambda,
                            model.center_loss_lambda_ph: var_hps.center_loss_lambda,
                            model.center_loss_alpha_ph: var_hps.center_loss_alpha,
                            model.center_loss_beta_ph: var_hps.center_loss_beta,
                            model.center_loss_gamma_ph: var_hps.center_loss_gamma,
                            model.feature_norm_alpha_ph: var_hps.feature_norm_alpha,
                            model.beta_B: var_hps.beta_B,
                            model.beta_C: var_hps.beta_C
                        })

                    train_loss.append(batch_loss_d['ce_loss'])
                    train_acc.append(batch_e_acc)
                self.logger.log('step %d,' % i, 'input shape', batch_input.x.shape, 'e_acc',
                                batch_e_acc, 'batch_loss_d', batch_loss_d, level=2)
            else:
                if self.hps.is_tflog:
                    summ, _ = session.run((model.train_merged, train_op), feed_dict={
                        model.fc_kprob_ph: self.hps.fc_kprob,
                        model.lr_ph: var_hps.lr,
                        model.x_ph: batch_input.x.astype(self.np_float_type),
                        model.t_ph: batch_input.t,
                        model.e_ph: batch_input.e,
                        model.e_w_ph: batch_input.w.astype(self.np_float_type),
                        model.is_training_ph: True,
                        model.cos_loss_lambda_ph: var_hps.cos_loss_lambda,
                        model.dist_loss_lambda_ph: var_hps.dist_loss_lambda,
                        model.center_loss_lambda_ph: var_hps.center_loss_lambda,
                        model.center_loss_alpha_ph: var_hps.center_loss_alpha,
                        model.center_loss_beta_ph: var_hps.center_loss_beta,
                        model.center_loss_gamma_ph: var_hps.center_loss_gamma,
                        model.feature_norm_alpha_ph: var_hps.feature_norm_alpha,
                    })
                    self.train_writer.add_summary(summ, i)
                else:
                    session.run(train_op, feed_dict={
                        model.fc_kprob_ph: self.hps.fc_kprob,
                        model.lr_ph: var_hps.lr,
                        model.x_ph: batch_input.x.astype(self.np_float_type),
                        model.t_ph: batch_input.t,
                        model.e_ph: batch_input.e,
                        model.e_w_ph: batch_input.w.astype(self.np_float_type),
                        model.l_ph: batch_input.l,
                        model.is_training_ph: True,
                        model.cos_loss_lambda_ph: var_hps.cos_loss_lambda,
                        model.dist_loss_lambda_ph: var_hps.dist_loss_lambda,
                        model.center_loss_lambda_ph: var_hps.center_loss_lambda,
                        model.center_loss_alpha_ph: var_hps.center_loss_alpha,
                        model.center_loss_beta_ph: var_hps.center_loss_beta,
                        model.center_loss_gamma_ph: var_hps.center_loss_gamma,
                        model.feature_norm_alpha_ph: var_hps.feature_norm_alpha,
                        model.beta_B: var_hps.beta_B,
                        model.beta_C: var_hps.beta_C

                    })
            if i % self.hps.eval_interval == 0:
                is_eval_test = self.hps.is_eval_test
                l_level = 1
                dev_metric_d, dev_loss_d = self.eval(dev_iter, var_hps, session)
                if self.hps.is_tflog:
                    fd = self.get_eval_merged_feed_dict(dev_metric_d, dev_loss_d,
                                                        self.hps.eval_metric_ks,
                                                        self.hps.eval_loss_ks)
                    dev_summ = session.run(self.eval_merged, feed_dict=fd)
                    self.dev_writer.add_summary(dev_summ, i)
                dev_metric = dev_metric_d[self.ckpt_metric_k]
                dev_loss = dev_loss_d[self.ckpt_loss_k]
                valid_acc.append(dev_metric_d['ua'])
                valid_loss.append(dev_loss_d['ce_center_loss'])
                if i > self.hps.best_params_start_steps and dev_metric > self.best_metric:
                    self.best_metric = dev_metric
                    self.best_metric_step = i
                    self.saver.save(session, self.best_metric_ckpt_path)
                    l_level = 2
                    is_eval_test = True
                if i > self.hps.best_params_start_steps and dev_loss < self.best_loss:
                    self.best_loss = dev_loss
                    self.best_loss_step = i
                    self.saver.save(session, self.best_loss_ckpt_path)
                    is_eval_test = True
                self.logger.log(' dev set: metric_d', dev_metric_d, 'loss_d', dev_loss_d,
                                level=l_level)
                self.logger.log('   best_metric', self.best_metric, 'best_metric_step',
                                self.best_metric_step,
                                level=l_level)
                self.logger.log('   best_loss', self.best_loss, 'best_loss_step',
                                self.best_loss_step,
                                level=l_level)
                if is_eval_test:
                    test_metric_d, test_loss_d = self.eval(test_iter, var_hps, session)
                    if self.hps.is_tflog:
                        fd = self.get_eval_merged_feed_dict(test_metric_d, test_loss_d,
                                                            self.hps.eval_metric_ks,
                                                            self.hps.eval_loss_ks)
                        test_summ = session.run(self.eval_merged, feed_dict=fd)
                        self.test_writer.add_summary(test_summ, i)
                    self.logger.log(' test set: metric_d', test_metric_d, 'loss_d', test_loss_d,
                                    level=1)
                self.logger.log(' Duration %f' % (time.time() - self.start_time), level=1)
            # if i % self.hps.persist_interval == 0 and i > 0:
            #     self.saver.save(session, self.ckpt_path, global_step=i)

        np.save('./train_loss.npy', train_loss)
        np.save('./valid_loss.npy', valid_loss)
        np.save('./train_acc.npy', train_acc)
        np.save('./valid_acc.npy', valid_acc)

    def run(self, d_set):
        tf_config = tf.ConfigProto()
        if 'gpu_allow_growth' in self.hps:
            tf_config.gpu_options.allow_growth = self.hps.gpu_allow_growth
        with tf.Session(config=tf_config) as sess:
            self.init_saver()
            self.init_summ_writer()
            self.train_writer.add_graph(tf.get_default_graph())
            sess.run(tf.global_variables_initializer())
            param_i = self.hps.max_steps - 1
            # eval_ckpt_file = self.hps.restore_file
            if self.hps.is_train:
                start_i = 0
                if self.hps.is_restore:
                    start_i = self.hps.restart_train_steps
                    self.saver.restore(sess, self.hps.restore_file)
                self.train(start_i, sess, d_set)
                if self.hps.best_params_type == 'best_metric':
                    self.saver.restore(sess, self.best_metric_ckpt_path)
                    param_i = self.best_metric_step
                elif self.hps.best_params_type == 'best_loss':
                    self.saver.restore(sess, self.best_loss_ckpt_path)
                    param_i = self.best_loss_step
            else:
                self.saver.restore(sess, self.hps.restore_file)
            test_iter = d_set.get_test_iter()
            train_no_repeat_iter = d_set.get_train_no_repeat_iter()
            dev_iter = d_set.get_dev_iter()
            var_hps = self.get_cur_var_hps(param_i)
            metric_d, loss_d = self.eval(test_iter, var_hps, sess)
            self.logger.log('test set: metric_d', metric_d, 'loss_d', loss_d)
            self.process_result(test_iter, var_hps, sess)
            self.save_feature(train_no_repeat_iter, var_hps, sess, 'train')
            self.save_feature(dev_iter, var_hps, sess, 'dev')
            self.save_feature(test_iter, var_hps, sess, 'test')
        self.exit()
