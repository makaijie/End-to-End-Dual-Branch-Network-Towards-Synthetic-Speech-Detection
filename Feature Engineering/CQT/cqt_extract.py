import os
import random
import numpy as np
import torch
import librosa
import soundfile as sf
from scipy import signal
import librosa.display
import matplotlib.pyplot as plt


def load_wav_snf(path):
    wav, sr = sf.read(path, dtype=np.float32)
    return wav

def preemphasis(wav, k=0.97):
    return signal.lfilter([1, -k], [1], wav)

def logpowcqt(wav_path, sr=16000, hop_length=512, n_bins=528, bins_per_octave=48, window="hann", fmin=3.5,pre_emphasis=0.97, ref=1.0, amin=1e-30, top_db=None):
    wav = load_wav_snf(wav_path)
    if pre_emphasis is not None:
        wav = preemphasis(wav, k=pre_emphasis)
    cqtfeats = librosa.cqt(y=wav, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave, window=window, fmin=fmin)
    magcqt = np.abs(cqtfeats)
    powcqt = np.square(magcqt)
    logpowcqt = librosa.power_to_db(powcqt, ref, amin, top_db)
    return logpowcqt


if __name__ == '__main__':
    with open('LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt', 'r') as f:
        audio_info = [info.strip().split() for info in f.readlines()]
    out = 'CQTFeatures/train/'
    for i in range(len(audio_info)):
        speakerid, utterance_id, _, fake_type, label = audio_info[i]
        utterance_path = 'LA/ASVspoof2019_LA_train/flac/' + utterance_id + '.flac'
        cqt = logpowcqt(utterance_path, sr=16000, n_bins=100,hop_length=512, bins_per_octave=12,window='hann', pre_emphasis=0.97, fmin=1)
        out_cqt = out + utterance_id + '.npy'
        np.save(out_cqt,cqt.astype(np.float32))
        print(i)

    with open('LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 'r') as f:
        audio_info = [info.strip().split() for info in f.readlines()]
    out = 'CQTFeatures/dev/'
    for i in range(len(audio_info)):
        speakerid, utterance_id, _, fake_type, label = audio_info[i]
        utterance_path = 'LA/ASVspoof2019_LA_dev/flac/' + utterance_id + '.flac'
        cqt = logpowcqt(utterance_path, sr=16000, n_bins=100,hop_length=512, bins_per_octave=12,window='hann', pre_emphasis=0.97, fmin=1)
        out_cqt = out + utterance_id + '.npy'
        np.save(out_cqt,cqt.astype(np.float32))
        print(i)

    with open('LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt', 'r') as f:
        audio_info = [info.strip().split() for info in f.readlines()]
    out = 'CQTFeatures/eval/'
    for i in range(len(audio_info)):
        speakerid, utterance_id, _, fake_type, label = audio_info[i]
        utterance_path = 'LA/ASVspoof2019_LA_eval/flac/' + utterance_id + '.flac'
        cqt = logpowcqt(utterance_path, sr=16000, n_bins=100,hop_length=512, bins_per_octave=12,window='hann', pre_emphasis=0.97, fmin=1)
        out_cqt = out + utterance_id + '.npy'
        np.save(out_cqt,cqt.astype(np.float32))
        print(i)


    # # visualization
    # filename = 'LA/ASVspoof2019_LA_train/flac/LA_T_1000137.flac'
    #
    # cqt = logpowcqt(filename, sr=16000, n_bins=100, hop_length=512, bins_per_octave=12, window='hann',
    #                 pre_emphasis=0.97, fmin=1)
    # librosa.display.specshow(cqt, y_axis='cqt_note', x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('cqt spectrogram')
    # plt.tight_layout()
    # plt.show()


























