import os
from utils import cfg_process
import time


class CRHpsPreprocessor(cfg_process.BaseHpsPreprocessor):

    def _extract_from_restore_file(self):
        if 'restore_file' not in self.hparams or self.hparams.restore_file.strip() == '':
            raise ValueError('Empty restore File!')
        eles = self.hparams.restore_file.split('/')
        ele = eles[-1]
        if '.' in ele:
            return "".join(ele.split('.')[:-1])
        else:
            return ele

    def _update_id_str(self):
        suffix = '_e' + str(
            self.hparams.vali_test_ses) + 'v' + self.hparams.vali_type + 't' \
                 + self.hparams.test_type
        id_str = self.hparams.id_prefix + self.hparams.id + suffix
        if 'id_str' in self.hparams:
            self.hparams.id_str = id_str
        else:
            self.hparams.add_hparam('id_str', id_str)

    def _update_id_related(self):
        if self.hparams.id == '':
            if self.hparams.is_train and not self.hparams.is_restore:
                self.hparams.id = time.strftime("%m%d%H%M", time.localtime())
            else:
                self.hparams.id = self._extract_from_restore_file() + \
                                  '_r' + time.strftime('%d%H%M', time.localtime())
        self._update_id_str()
        self.hparams.add_hparam('tf_log_fold',
                                self.hparams.tf_log_fold_prefix + self.hparams.id_str)

    def _check_dir(self):
        if not os.path.exists(self.hparams.out_dir):
            os.makedirs(self.hparams.out_dir)

        self.hparams.add_hparam('tf_log_dir',
                                os.path.join(self.hparams.out_dir, self.hparams.tf_log_fold))
        if not os.path.exists(self.hparams.tf_log_dir):
            os.makedirs(self.hparams.tf_log_dir)

        self.hparams.add_hparam('result_npy_dir',
                                os.path.join(self.hparams.out_dir, self.hparams.result_npy_fold))
        if not os.path.exists(self.hparams.result_npy_dir):
            os.makedirs(self.hparams.result_npy_dir)

        self.hparams.add_hparam('result_matrix_dir',
                                os.path.join(self.hparams.out_dir, self.hparams.result_matrix_fold))
        if not os.path.exists(self.hparams.result_matrix_dir):
            os.makedirs(self.hparams.result_matrix_dir)

        self.hparams.add_hparam('cfg_out_dir',
                                os.path.join(self.hparams.out_dir, self.hparams.cfg_out_fold))
        if not os.path.exists(self.hparams.cfg_out_dir):
            os.makedirs(self.hparams.cfg_out_dir)

        self.hparams.add_hparam('ckpt_dir',
                                os.path.join(self.hparams.out_dir, self.hparams.ckpt_fold))
        if not os.path.exists(self.hparams.ckpt_dir):
            os.makedirs(self.hparams.ckpt_dir)

        self.hparams.add_hparam('bestloss_ckpt_dir',
                                os.path.join(self.hparams.out_dir, self.hparams.bestloss_ckpt_fold))
        if not os.path.exists(self.hparams.bestloss_ckpt_dir):
            os.makedirs(self.hparams.bestloss_ckpt_dir)

        self.hparams.add_hparam('bestmetric_ckpt_dir',
                                os.path.join(self.hparams.out_dir, self.hparams.bestmetric_ckpt_fold))
        if not os.path.exists(self.hparams.bestmetric_ckpt_dir):
            os.makedirs(self.hparams.bestmetric_ckpt_dir)

        self.hparams.add_hparam('log_dir',
                                os.path.join(self.hparams.out_dir, self.hparams.log_fold))
        if not os.path.exists(self.hparams.log_dir):
            os.makedirs(self.hparams.log_dir)
        self.hparams.add_hparam('log_path',
                                os.path.join(self.hparams.log_dir, self.hparams.id_str + '.log'))