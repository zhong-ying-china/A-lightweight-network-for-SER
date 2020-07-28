#!/usr/bin/env bash

echo 'run0'

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_impro_batch64.yml --config_name=ce_center_m11_origin_lambda003_nodropout --gpu=0 --vali_type='0' --test_type='1'
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_impro_batch64.yml --config_name=ce_center_m11_origin_lambda003_nodropout --gpu=0 --vali_type='2' --test_type='3'
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_impro_batch64.yml --config_name=ce_center_m11_origin_lambda003_nodropout --gpu=0 --vali_type='4' --test_type='5'
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_impro_batch64.yml --config_name=ce_center_m11_origin_lambda003_nodropout --gpu=0 --vali_type='6' --test_type='7'
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_impro_batch64.yml --config_name=ce_center_m11_origin_lambda003_nodropout --gpu=0 --vali_type='8' --test_type='9'

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_impro_batch64.yml --config_name=ce_center_m11_origin_lambda001_nodropout --gpu=0 --vali_type='0' --test_type='1'
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_impro_batch64.yml --config_name=ce_center_m11_origin_lambda001_nodropout --gpu=0 --vali_type='2' --test_type='3'
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_impro_batch64.yml --config_name=ce_center_m11_origin_lambda001_nodropout --gpu=0 --vali_type='4' --test_type='5'
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_impro_batch64.yml --config_name=ce_center_m11_origin_lambda001_nodropout --gpu=0 --vali_type='6' --test_type='7'
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_impro_batch64.yml --config_name=ce_center_m11_origin_lambda001_nodropout --gpu=0 --vali_type='8' --test_type='9'
