#!/usr/bin/env bash

echo 'beta gamma'

echo 'beta gamma' > run2-server-pzd.log

echo 'ce_center_m11_origin_lambda03_alpha01_beta002_gamma02' >> run2-server-pzd.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma02 --gpu=2 --vali_type='8' --test_type='9'
echo 'v8t9' >> run2-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma02 --gpu=2 --vali_type='6' --test_type='7'
echo 'v6t7' >> run2-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma02 --gpu=2 --vali_type='4' --test_type='5'
echo 'v4t5' >> run2-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma02 --gpu=2 --vali_type='2' --test_type='3'
echo 'v2t3' >> run2-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma02 --gpu=2 --vali_type='0' --test_type='1'
echo 'v0t1' >> run2-server-pzd.log

echo 'ce_center_m11_origin_lambda03_alpha01_beta002_gamma05' >> run2-server-pzd.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma05 --gpu=2 --vali_type='8' --test_type='9'
echo 'v8t9' >> run2-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma05 --gpu=2 --vali_type='6' --test_type='7'
echo 'v6t7' >> run2-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma05 --gpu=2 --vali_type='4' --test_type='5'
echo 'v4t5' >> run2-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma05 --gpu=2 --vali_type='2' --test_type='3'
echo 'v2t3' >> run2-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma05 --gpu=2 --vali_type='0' --test_type='1'
echo 'v0t1' >> run2-server-pzd.log

echo 'ce_center_m11_origin_lambda03_alpha01_beta002_gamma005' >> run2-server-pzd.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma005 --gpu=2 --vali_type='8' --test_type='9'
echo 'v8t9' >> run2-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma005 --gpu=2 --vali_type='6' --test_type='7'
echo 'v6t7' >> run2-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma005 --gpu=2 --vali_type='4' --test_type='5'
echo 'v4t5' >> run2-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma005 --gpu=2 --vali_type='2' --test_type='3'
echo 'v2t3' >> run2-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma005 --gpu=2 --vali_type='0' --test_type='1'
echo 'v0t1' >> run2-server-pzd.log