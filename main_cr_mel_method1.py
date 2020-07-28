import tensorflow as tf

from cr_model_v2 import cr_cfg_process
from cr_model_v2 import cr_model
from cr_model_v2 import cr_model_impl_mel
from cr_model_v2 import cr_model_run
from cr_model_v2 import data_set
from cr_model_v2 import load_data
from utils import cfg_process
from utils import parser_util
import os
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"

def add_arguments(parser):
    """Build ArgumentParser"""
    parser.add_argument('--config_file', type=str,
                        default='./cr_model_v2/cfgs/mel_rediv_ma_batch64_nodropout_method1.yml',
                        help='config file about hparams')
    parser.add_argument('--config_name', type=str, default='server',
                        help='config name for hparams')
    parser.add_argument('--gpu', type=str, default='',
                        help='config for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='config for center loss beta')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='config for center loss gamma')


def main(unused_argv):
    tf.set_random_seed(4)
    parser = parser_util.MyArgumentParser()
    add_arguments(parser)
    argc, flags_dict = parser.parse_to_dict()
    yparams = cfg_process.YParams(argc.config_file, argc.config_name)
    yparams.center_loss_betas[-1] = argc.beta
    yparams.center_loss_gammas[-1] = argc.gamma
    outdir_prefix = './cr_model_v2/out_mel_rediv_ma_batch64_nodropout_method1/ce_center_m11_origin_lambda03_alpha01_'
    yparams.out_dir = outdir_prefix + 'beta' + str(argc.beta) + '_' + 'gamma' + str(
        argc.gamma)
    yparams = cr_cfg_process.CRHpsPreprocessor(yparams, flags_dict).preprocess()

    print(yparams.center_loss_betas)
    print(yparams.center_loss_gammas)
    print(yparams.out_dir)
    print('id str:', yparams.id_str)
    yparams.save()
    CRM_dict = {
        'MelModel': cr_model_impl_mel.MelModel
    }
    # print('model_key', yparams.model_key)
    CRM = CRM_dict[yparams.model_key]
    model = CRM(yparams)
    if yparams.database == 'IEMOCAP' and yparams.is_rediv_data is True:
        l_data = load_data.load_data_mix(yparams)
    else:
        l_data = load_data.load_data_emodb(yparams)
    d_set = data_set.DataSet(l_data, yparams)
    cr_model_run_v2 = cr_model_run.CRModelRun(model)
    cr_model_run_v2.run(d_set)

    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('the number of parameter:', total_parameters)


if __name__ == '__main__':
    tf.app.run(main=main)
