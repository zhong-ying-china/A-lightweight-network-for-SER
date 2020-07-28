import os
import librosa
import numpy as np
import python_speech_features as ps

def get_log_mel_spectrogram(y, sr, n_fft, win_length, hop_length, power=2, window='hamming',
                            n_mels=128):
    spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                      window=window)) ** power
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=n_mels)
    mel_s = np.dot(mel_basis, spectrogram)
    log_mel_s = librosa.core.power_to_db(mel_s)
    return log_mel_s.transpose()

def get_mean_std():
    n_fft = 1024
    win_time = 0.04
    hop_time = 0.01
    file_path = 'F:\\datasets\\IEMOCAP\\emo_wavs'
    logmel = []
    delta1_s = []
    delta2_s = []
    file_names = os.listdir(file_path)
    for file_name in file_names:
        if file_name.endswith('.wav') and (file_name[3]):
            in_file_path = os.path.join(file_path, file_name)
            y, sr = librosa.load(in_file_path, sr=16000)
            win_length = int(win_time * sr)
            hop_length = int(hop_time * sr)
            log_s = get_log_mel_spectrogram(y=y, sr=sr, n_fft=n_fft, win_length=win_length,
                                            hop_length=hop_length, power=2,
                                            window='hamming',n_mels=128)
            delta1 = ps.delta(log_s, 2)
            delta2 = ps.delta(delta1, 2)

            logmel.append(log_s)
            delta1_s.append(delta1)
            delta2_s.append(delta2)
    logmel = np.vstack(logmel)
    delta1_s = np.vstack(delta1_s)
    delta2_s = np.vstack(delta2_s)

    mean1 = np.mean(logmel, axis=0)
    std1 = np.std(logmel, axis=0)
    mean2 = np.mean(delta1_s, axis=0)
    std2 = np.std(delta1_s, axis=0)
    mean3 = np.mean(delta2_s, axis=0)
    std3 = np.std(delta2_s, axis=0)

    return mean1, std1, mean2, std2, mean3, std3

def norm():
    # mean1, std1, mean2, std2, mean3, std3 = get_mean_std()
    n_fft = 1024
    win_time = 0.04
    hop_time = 0.01
    in_fold = '/home/zy/dataset/emotions/data/ABC_wavs'
    out_fold = '/home/zy/dataset/emotions/data/ABC_mels'
    if not os.path.exists(out_fold):
        os.mkdir(out_fold)
    file_names = os.listdir(in_fold)
    for file_name in file_names:
        if file_name.endswith('.wav'):
            in_file_path = os.path.join(in_fold, file_name)
            out_file_path = os.path.join(out_fold, os.path.splitext(file_name)[0] + ".npy")
            y, sr = librosa.load(in_file_path, sr=16000)
            win_length = int(win_time * sr)
            hop_length = int(hop_time * sr)

            log_s = get_log_mel_spectrogram(y=y, sr=sr, n_fft=n_fft, win_length=win_length,
                                            hop_length=hop_length, power=2,
                                            window='hamming', n_mels=128)
            # delta1 = ps.delta(log_s, 2)
            # delta2 = ps.delta(delta1, 2)
            #
            # log_mel = (log_s - mean1) / std1
            # delta11 = (delta1 - mean2) / std2
            # delta21 = (delta2 - mean3) / std3
            # all_data = np.array([log_mel, delta11, delta21])
            # all_data = np.transpose(all_data, [1, 2, 0])
            # print(all_data.shape)
            np.save(out_file_path, log_s)

if __name__ == '__main__':
    norm()