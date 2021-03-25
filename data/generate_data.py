import os
import time
import numpy as np
import librosa
import scipy
from pyvad import trim
from glob import glob
import random
from tqdm import tqdm


def extract_feature(y, sr):
    ### Pre-processing
    Num_Frame = 1500    # max wave length (15 sec)
    Stride = 0.01       # stride (10ms)
    Window_size = 0.025 # filter window size (25ms)
    Num_data = 1
    Num_mels = 40       # Mel filter number
    pre_emphasis = 0.97  # Pre-Emphasis filter coefficient

    # Resampling to 16kHz
    if sr != 16000:
        sr_re = 16000  # sampling rate of resampling
        y = librosa.resample(y, sr, sr_re)
        sr = sr_re

    # Denoising
    y[np.argwhere(y == 0)] = 1e-10
    y_denoise = scipy.signal.wiener(y, mysize=None, noise=None)

    # Pre Emphasis filter → high frequency를 높여주는 부분
    y_Emphasis = np.append(y_denoise[0], y_denoise[1:] - pre_emphasis * y_denoise[:-1])

    # Normalization (Peak)
    y_max = max(y_Emphasis)
    y_Emphasis = y_Emphasis / y_max  # VAD 인식을 위해 normalize

    # Voice Activity Detection (VAD)
    vad_mode = 2  # VAD mode = 0 ~ 3
    y_vad = trim(y_Emphasis, sr, vad_mode=vad_mode, thr=0.01)  ## VAD 사용하여 trim 수행
    if y_vad is None:
        y_vad = y_Emphasis

    # De normalization
    y_vad = y_vad * y_max

    # Obtain the mel spectrogram
    S = librosa.feature.melspectrogram(y=y_vad, sr=sr, hop_length=int(sr * Stride), n_fft=int(sr * Window_size), n_mels=Num_mels, power=2.0)
    r, Frame_length = S.shape
    S = np.log(S + 1e-8)
    # Obtain the normalized mel spectrogram
    S_norm = (S - np.mean(S)) / np.std(S)

    return S_norm


if __name__=='__main__':
    # Download LibriSpeech train-clean-360 dataset from http://www.openslr.org/12
    # Specify data path for training
    base_path = '../../train-clean-360'
    file_list = glob(base_path + '/*/*/*.flac')

    # Generate dictionary based on speaker IDs
    subset = ['tr', 'cv', 'tt']
    subset_dict = dict()
    for sub in subset:
        subset_dict[sub] = []

    spk_dict = dict()
    for i in tqdm(range(len(file_list))):
        spk_id, _, _ = os.path.basename(file_list[i]).split('-')
        if spk_id not in spk_dict.keys():
            spk_dict[spk_id] = []
        else:
            spk_dict[spk_id].append(file_list[i]) 

    # Split train / validation / evaluation data
    spk_list = dict()
    
    for i, spk in enumerate(spk_dict.keys()):
        spk_list[spk] = i
        random.shuffle(spk_dict[spk])
        n_utt = len(spk_dict[spk])
        subset_dict['tr'].extend(spk_dict[spk][:int(n_utt * 0.7)])
        subset_dict['cv'].extend(spk_dict[spk][int(n_utt * 0.7):int(n_utt * 0.9)])
        subset_dict['tt'].extend(spk_dict[spk][int(n_utt * 0.9):])

    # Save features and path
    text_list = ['train.txt', 'val.txt', 'test.txt']
    for j, sub in enumerate(subset):
        file_list = subset_dict[sub]
        os.makedirs('{}/mel_feat'.format(sub), exist_ok=True)
        with open(text_list[j], 'w') as fp:
            for i in tqdm(range(len(file_list))):
    
                x, fs = librosa.load(file_list[i], 16000)
                mel_spec = extract_feature(x, fs)
                save_name = '{}/mel_feat/{}'.format(sub, os.path.basename(file_list[i].replace('.flac', '.npy')))
                np.save(save_name, mel_spec)
                spk_label = spk_list[os.path.basename(save_name).split('-')[0]]
                if mel_spec.shape[-1] < 210:
                    continue
                start_idx = random.randint(10, mel_spec.shape[-1]-200)
                fp.write('{}\t{}\t{}\n'.format(str(spk_label), './data/' + save_name, start_idx))

        
