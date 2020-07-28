import os
from shutil import copyfile

data_dir = '/home/dai/Project/emotions/data/IEMOCAP_full_release'
target_wav_dir = '/home/dai/Project/emotions/data/iemocap_wavs'
eval_dir = '/home/dai/Project/emotions/ser_nn_spectrogram/RelativeNet/eval_txts'


def extract_wavs():
    eval_file_names = os.listdir(eval_dir)
    for eval_file_name in eval_file_names:
        session_name = get_session_name(eval_file_name)
        s_fold = os.path.join(data_dir, session_name, 'sentences', 'wav', eval_file_name.strip('.txt'))
        with open(os.path.join(eval_dir, eval_file_name), 'r') as f:
            for line in f:
                if '[' in line and ']' in line and 'Ses' in line:
                    eles = line.split()
                    s_wav_file_name = eles[3] + '.wav'
                    t_wav_file_name = '_'.join((eles[3], eles[4])) + '.wav'
                    # print(s_wav_file_name, t_wav_file_name)
                    s_file_path = os.path.join(s_fold, s_wav_file_name)
                    t_file_path = os.path.join(target_wav_dir, t_wav_file_name)
                    copyfile(s_file_path, t_file_path)


def get_session_name(eval_file_name):
    if 'Ses01' in eval_file_name:
        return 'Session1'
    elif 'Ses02' in eval_file_name:
        return 'Session2'
    elif 'Ses03' in eval_file_name:
        return 'Session3'
    elif 'Ses04' in eval_file_name:
        return 'Session4'
    elif 'Ses05' in eval_file_name:
        return 'Session5'
    return None


if __name__ == '__main__':
    extract_wavs()
