#!/usr/bin/env bash


echo 'ce_center_m11_origin_lambda0' > server.log

echo 'cross validation 1: ...' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0 --gpu=0  --vali_type='8' --test_type='9'
echo 'v8t9' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='6' --test_type='7'
echo 'v6t7' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='4' --test_type='5'
echo 'v4t5' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='2' --test_type='3'
echo 'v2t3' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='0' --test_type='1'
echo 'v0t1' >> server.log

echo 'cross validation 2: ...' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='8' --test_type='9'
echo 'v8t9' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='6' --test_type='7'
echo 'v6t7' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='4' --test_type='5'
echo 'v4t5' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='2' --test_type='3'
echo 'v2t3' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='0' --test_type='1'
echo 'v0t1' >> server.log

echo 'cross validation 3: ...' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='8' --test_type='9'
echo 'v8t9' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='6' --test_type='7'
echo 'v6t7' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='4' --test_type='5'
echo 'v4t5' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='2' --test_type='3'
echo 'v2t3' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='0' --test_type='1'
echo 'v0t1' >> server.log

echo 'cross validation 4: ...' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='8' --test_type='9'
echo 'v8t9' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='6' --test_type='7'
echo 'v6t7' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='4' --test_type='5'
echo 'v4t5' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='2' --test_type='3'
echo 'v2t3' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='0' --test_type='1'
echo 'v0t1' >> server.log

echo 'cross validation 5: ...' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='8' --test_type='9'
echo 'v8t9' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='6' --test_type='7'
echo 'v6t7' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='4' --test_type='5'
echo 'v4t5' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='2' --test_type='3'
echo 'v2t3' >> server.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout_tencent.yml --config_name=ce_center_m11_origin_lambda0  --vali_type='0' --test_type='1'
echo 'v0t1' >> server.log
