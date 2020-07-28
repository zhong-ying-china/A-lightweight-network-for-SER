import numpy as np
import os
from shutil import copy2


def filter_wav():
    in_fold = '/home/zy/dataset/emotions/data/iemocap_wavs'
    out_fold = '/home/zy/dataset/emotions/data/iemocap_5emo_wavs'
    if not os.path.exists(out_fold):
        os.mkdir(out_fold)
    file_names = os.listdir(in_fold)
    for file_name in file_names:
        if 'neu' in file_name or 'ang' in file_name or 'hap' in file_name or 'sad' in file_name or 'exc' in file_name:
            print(file_name)
            in_file_path = os.path.join(in_fold, file_name)
            out_file_path = os.path.join(out_fold, file_name)
            copy2(in_file_path, out_file_path)


if __name__ == '__main__':
    filter_wav()
