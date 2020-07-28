import os
from shutil import copyfile
import numpy as np
data_dir = '/home/zy/dataset/IEMOCAP/IEMOCAP_full_release'
target_wav_dir = '/home/zy/dataset/emotions/data/ABC_wavs'
txt_path = 'F:\\datasets\\IEMOCAP\\eval_txts'


def extract_wavs():
    if not os.path.exists(target_wav_dir):
        os.makedirs(target_wav_dir)
    eval_file_names = os.listdir(txt_path)
    emo_prob = []
    for eval_file_name in eval_file_names:
        session_name = get_session_name(eval_file_name)
        s_fold = os.path.join(data_dir, session_name, 'sentences', 'wav', eval_file_name.strip('.txt'))
        f = open(os.path.join(txt_path, eval_file_name), 'r').read()
        f = np.array(f.split('\n'))
        # skip null row, reture bool, true means null row
        idx = f == ''
        # idx_n is null row
        idx_n = np.arange(len(f))[idx]
        for i in range(len(idx_n) - 2):
            # g is one wav evaluation
            g = f[idx_n[i] + 1: idx_n[i + 1]]
            head = g[0]
            label = head[head.find('\t[') - 3: head.find('\t[')]
            if label in ('hap', 'ang', 'sad', 'neu', 'exc'):
                filename = head.split('\t')[1]
                s_wav_filename = filename + '.wav'
                emos = []
                j = 1
                while g[j][0] == 'C':
                    head = g[j]
                    start_idx = head.find('\t') + 1
                    # eval_emo = []
                    idx = head.find(';', start_idx)
                    while idx != -1:
                        emos.append(head[start_idx:idx])
                        start_idx = idx + 1
                        idx = head.find(';', start_idx)
                    j += 1
                x = 0
                for m in range(len(emos)):
                    if label in emos[m].lower():
                        x += 1
                prob = float(x/len(emos))
                emo_prob.append(prob)
                level = get_level(prob)
                t_wav_filename = '_'.join((filename, label, level)) + '.wav'
                s_file_path = os.path.join(s_fold, s_wav_filename)
                t_file_path = os.path.join(target_wav_dir, t_wav_filename)
                copyfile(s_file_path, t_file_path)

    np.save('/home/zy/dataset/emotions/prob1.npy', emo_prob)

def get_multilabel():
    eval_file_names = os.listdir(txt_path)
    emo_prob = []
    leve_A = []
    leve_B = []
    leve_C = []
    for eval_file_name in eval_file_names:
        session_name = get_session_name(eval_file_name)
        f = open(os.path.join(txt_path, eval_file_name), 'r').read()
        f = np.array(f.split('\n'))
        # skip null row, reture bool, true means null row
        idx = f == ''
        # idx_n is null row
        idx_n = np.arange(len(f))[idx]
        for i in range(len(idx_n) - 2):
            # g is one wav evaluation
            g = f[idx_n[i] + 1: idx_n[i + 1]]
            head = g[0]
            label = head[head.find('\t[') - 3: head.find('\t[')]
            if label in ('sad'):
                emos = []
                j = 1
                while g[j][0] == 'C':
                    head = g[j]
                    start_idx = head.find('\t') + 1
                    # eval_emo = []
                    idx = head.find(';', start_idx)
                    while idx != -1:
                        emos.append(head[start_idx:idx])
                        start_idx = idx + 1
                        idx = head.find(';', start_idx)
                    j += 1
                x = 0
                for m in range(len(emos)):
                    if label in emos[m].lower():
                        x += 1
                prob = float(x / len(emos))
                emo_prob.append(prob)
                level = get_level(prob)
                if level == 'A':
                    leve_A.append(level)
                elif level == 'B':
                    leve_B.append(level)
                else:
                    leve_C.append(level)
    np.save('./prob.npy', emo_prob)
    print(len(leve_A))
    print(len(leve_B))
    print(len(leve_C))

def get_level(prob):
    if prob==1.0:
        return 'A'
    elif prob >= 0.65:
        return 'B'
    else:
        return 'C'

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
    get_multilabel()
