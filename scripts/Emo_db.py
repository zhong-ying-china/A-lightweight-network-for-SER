import numpy as np
import librosa
import os

def get_emotion(emotion):
    if emotion == 'W':
        return 'ang'
    elif emotion == 'L':
        return 'bor'
    elif emotion == 'E':
        return 'dis'
    elif emotion == 'A':
        return 'fea'
    elif emotion == 'F':
        return 'hap'
    elif emotion == 'T':
        return 'sad'
    elif emotion == 'N':
        return 'neu'

def get_log_mel_spectrogram(y, sr, n_fft, win_length, hop_length, power=2, window='hamming',
                            n_mels=128):
    spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                      window=window)) ** power
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=n_mels)
    mel_s = np.dot(mel_basis, spectrogram)
    log_mel_s = librosa.core.power_to_db(mel_s)
    return log_mel_s.transpose()

def get_wav_label_file():
    n_fft = 1024
    win_time = 0.04
    hop_time = 0.01
    wav_path = 'F:/datasets/emodb/wav/'
    out_dir = 'F:/datasets/emodb/128mel/'
    num = {'W': 0, 'L': 0, 'E': 0, 'A': 0, 'F': 0, 'T': 0, 'N': 0}
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for filename in os.listdir(wav_path):
        label = filename.split('.')[0][-2]
        num[label]= num[label] + 1
    print(num)

        # in_file = os.path.join(wav_path, filename)
        # out_file = os.path.join(out_dir, filename.split('.')[0] + '.npy')
        # y, sr = librosa.load(in_file, sr=16000)
        # win_length = int(win_time *sr)
        # hop_length = int(hop_time * sr)
        # log_s = get_log_mel_spectrogram(y=y, sr=sr, n_fft=n_fft, win_length=win_length,
        #                                     hop_length=hop_length, power=2,
        #                                     window='hamming',n_mels=128)
        # # print(out_file,log_s.shape)
        # np.save(out_file, log_s)

if __name__ == '__main__':
    get_wav_label_file()