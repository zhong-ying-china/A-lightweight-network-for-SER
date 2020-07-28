import numpy as np
import random
from collections import defaultdict
from shutil import copy2
import os


def get_key(filename):
    sess_name = filename[0:5]
    if 'impro' in filename:
        type_name = 'impro'
    else:
        type_name = 'script'
    gender = filename[-12]
    emo = filename[-7:-4]
    return '_'.join((sess_name, type_name, gender, emo))


def mix_div_sets(sub_nums=6):
    print('in mix div set')
    file_dict = defaultdict(list)
    in_dir = '/home/zy/dataset/emotions/data/iemocap_5emo_logMelW40fft1024d128'
    out_dir = '/home/zy/dataset/emotions/data/iemocap_rediv'

    if not os.path.exists(out_dir):
        print('make fold')
        os.mkdir(out_dir)
    filenames = os.listdir(in_dir)
    # print(filenames)
    for filename in filenames:
        if '.npy' in filename:
            k = get_key(filename)
            file_dict[k].append(filename)

    for _, v in file_dict.items():
        random.shuffle(v)
        start_idx = random.randint(0, 6)
        for i, filename in zip(range(len(v)), v):
            preffix = 'sub' + str((start_idx + i) % sub_nums) + '_'
            outfilename = preffix + filename
            print(outfilename)
            in_path = os.path.join(in_dir, filename)
            out_path = os.path.join(out_dir, outfilename)
            copy2(in_path, out_path)


if __name__ == '__main__':
    mix_div_sets(sub_nums=6)
    # k = get_key('Ses05M_script03_2_M028_ang.npy')
    # print(k)

