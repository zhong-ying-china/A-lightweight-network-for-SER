#!/usr/bin/env bash

echo 'ce_center_m11_origin_lambda03_alpha05' > run3-server10.log

python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=0 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=0 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=1 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=1 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=2 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=2 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=3 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=3 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=4 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=4 --vali_type=M --test_type=F

echo 'cross validation 1: ...' >> run3-server10.log

python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=0 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=0 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=1 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=1 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=2 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=2 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=3 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=3 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=4 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=4 --vali_type=M --test_type=F

echo 'cross validation 2: ...' >> run3-server10.log

python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=0 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=0 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=1 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=1 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=2 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=2 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=3 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=3 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=4 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=4 --vali_type=M --test_type=F

echo 'cross validation 3: ...' >> run3-server10.log

python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=0 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=0 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=1 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=1 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=2 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=2 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=3 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=3 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=4 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=4 --vali_type=M --test_type=F

echo 'cross validation 4: ...' >> run3-server10.log

python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=0 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=0 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=1 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=1 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=2 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=2 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=3 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=3 --vali_type=M --test_type=F
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=4 --vali_type=F --test_type=M
python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=3 --vali_test_ses=4 --vali_type=M --test_type=F

echo 'cross validation 5: ...' >> run3-server10.log