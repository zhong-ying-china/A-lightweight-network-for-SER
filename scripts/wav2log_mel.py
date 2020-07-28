import os

import librosa
import numpy as np


def get_log_mel_spectrogram(y, sr, n_fft, win_length, hop_length, power=2, window='hamming',
                            n_mels=128):
    spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                      window=window)) ** power
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=n_mels)
    mel_s = np.dot(mel_basis, spectrogram)
    log_mel_s = librosa.core.power_to_db(mel_s)
    return log_mel_s.transpose()


def wav2log_mel():
    n_fft = 1024
    win_time = 0.04
    hop_time = 0.01
    in_fold = '/home/ddy/projects/emotions/data/iemocap_5emo_wavs'
    out_fold = '/home/ddy/projects/emotions/data/iemocap_5emo_logMelW40fft1024d128'
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
                                            window='hamming',
                                            n_mels=128)
            print(file_name, log_s.shape)
            np.save(out_file_path, log_s)


if __name__ == '__main__':
    wav2log_mel()

