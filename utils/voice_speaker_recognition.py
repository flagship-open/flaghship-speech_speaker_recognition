import os
import time
import numpy as np
import scipy
import librosa
from pyvad import trim
import glob
from numpy import linalg as LA
import operator
import tensorflow as tf
from keras import backend as K

### Pre-processing
Num_Frame = 1500    # max wave length (15 sec)
Stride = 0.01       # stride (10ms)
Window_size = 0.025 # filter window size (25ms)
Num_data = 1
Num_mels = 40       # Mel filter number
pre_emphasis = 0.97  # Pre-Emphasis filter coefficient
Crop_Size = 200

global graph
graph = tf.get_default_graph()
tf.global_variables_initializer()
# keras.backend.clear_session()

def preprocessing(y, sr):

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

    # zero padding
    Input_Mels = np.zeros((r, Num_Frame), dtype=float)
    if Frame_length < Num_Frame:
        Input_Mels[:, :Frame_length] = S_norm[:, :Frame_length]
    else:
        Input_Mels[:, :Num_Frame] = S_norm[:, :Num_Frame]

    return Input_Mels, Frame_length


def Crop_Mels(Input_Mels_origin,Each_Frame_Num):
    Input_Mels_origin = Input_Mels_origin.T

    # Calculate the number of cropped mel-spectrogram
    if Each_Frame_Num > 1500:
        Number_of_Crop = 14
    else:
        if Each_Frame_Num < 200:
            Number_of_Crop = 1
        else:
            Number_of_Crop = int(round(Each_Frame_Num/100)) - 1

    ## Crop
    Crop_Num_Frame = 200    # Frame size of crop
    Cropped_Mels = np.zeros((Number_of_Crop,Crop_Num_Frame,Input_Mels_origin.shape[1]))
    crop_num = 0  # Crop된 data의 number
    if Each_Frame_Num > 1500:  # If the frame number is higher than 1500, the number of crop is 14
        Each_Crop_Num = 14
        for N_crop in range(0, Each_Crop_Num):
            Cropped_Mels[crop_num, :, :] = Input_Mels_origin[N_crop * 100:N_crop * 100 + 200, :]
            crop_num += 1
    else:
        if Each_Frame_Num < 200:    # If the frame number is lower than 200, the number of crop is 1
            Cropped_Mels[crop_num, :, :] = Input_Mels_origin[:200, :]
            crop_num += 1
        else:
            Each_Crop_Num = int(round(Each_Frame_Num / 100)) - 1    # Calculate the number of crop
            if round(Each_Frame_Num / 100) < Each_Frame_Num / 100:
                for N_crop in range(0, Each_Crop_Num):
                    Cropped_Mels[crop_num, :, :] = Input_Mels_origin[N_crop * 100:N_crop * 100 + 200, :]
                    crop_num += 1
            else:
                for N_crop in range(0, Each_Crop_Num - 1):
                    Cropped_Mels[crop_num, :, :] = Input_Mels_origin[N_crop * 100:N_crop * 100 + 200, :]
                    crop_num += 1
                shift_frame = int((Each_Frame_Num / 100 - round(Each_Frame_Num / 100)) * 100)
                Cropped_Mels[crop_num, :, :] = Input_Mels_origin[(Each_Crop_Num - 1) * 100 + shift_frame:(Each_Crop_Num - 1) * 100 + shift_frame + 200,:]
                crop_num += 1
    return Cropped_Mels, Number_of_Crop


def generate(file_path, model):
   
    y, sr = librosa.load(file_path, sr=16000)
    Input_Mels, Frame_length = preprocessing(y, sr)

    # Crop mel-spectrogram
    Cropped_Mels, Number_of_Crop = Crop_Mels(Input_Mels, Frame_length)
    Cropped_Mels = np.reshape(Cropped_Mels, (Cropped_Mels.shape[0], Cropped_Mels.shape[1], Cropped_Mels.shape[2], 1))

    if model is not None:
        speaker_model = model
    # Generate embedding
    with graph.as_default():
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            embedding_layer = speaker_model.layers[21]
            func= K.function([speaker_model.layers[0].input], [embedding_layer.output])
    
    output_emb = func([Cropped_Mels])[0]

    return output_emb


def voice_identification(file_path, model=None):
    # 등록된 각 화자의 마지막 feature 파일 로드
    spk_list = glob.glob('./voice_gallery/*')
    spk_sim = dict()
    feat_list = []

#    print(spk_list)
    for spk in spk_list:
        feat_path = sorted(glob.glob(spk + os.sep + 'feat*.npy'))[-1]
        feat_list.append(feat_path)

    input_emb = generate(file_path, model)

#    print(input_emb.shape) 
    # Compute cosine similarity for enrolled features
    for spk_feat in feat_list:
        spk_emb = np.load(spk_feat, allow_pickle=True) 
        cos_sim_spk = []
        for input_seg in input_emb:
            for ori_seg in spk_emb:
                cos_sim = input_seg.dot(np.transpose(ori_seg))/(LA.norm(input_seg) * LA.norm(ori_seg) + 1e-8)
                cos_sim_spk.append(cos_sim)
        spk_sim[spk_feat.split(os.sep)[-2]] = np.mean(cos_sim_spk)

#    print("5 maximum candidate speakers")
    max_5_speakers = dict(sorted(spk_sim.items(), key=operator.itemgetter(1), reverse=True)[:5])
    sorted_speakers = sorted(max_5_speakers.items(), key=lambda item: item[1], reverse=True)

    top_value = []
    id_top = []
    for i in range(len(sorted_speakers)):
        top_value.append(sorted_speakers[i][1])
        id_top.append(sorted_speakers[i][0])

    return top_value, id_top


def voice_gallery(path, target):
    gallery_path = glob.glob(os.path.join(path, 'voice_gallery/*'))
    issame = []
    for i, gallery_name in enumerate(gallery_path):
        new_gallery_name = gallery_name.strip().split('/')[-1]
        if str(new_gallery_name) == str(target) and len(glob.glob(gallery_name + '/*.npy')) > 0:
            answer = 1
        else:
            answer = 0
        issame.append(answer)

    if issame.count(1)>0:
        output = True
        print('This person is already registered')
    else:
        output = False
        print("This person isn't registered")
    return output


def voice_registration(path_target, name_class, re_register, model = None):
    aa = time.time()
#    print(path_target)
    output_path = os.path.join('.', ('voice_gallery/{}/'.format(name_class)))

    if re_register == True:
        if not os.path.exists(output_path):
            msg = 'Re_register is True, but speaker is not registered before. So we register speaker. Speaker ID : {}'.format(name_class)
        else:
            msg = 'Re_register is True, so we register speaker again. Speaker ID : {}'.format(name_class)
    else:
        msg = 'We register speaker. Speaker ID : {}'.format(name_class)

    speaker_id = name_class
    print('file upload takes {:.2f}s'.format(time.time() - aa))

    # Voice_gallery에 화자 폴더 생성
    spk_list = glob.glob('./voice_gallery/*')
    spk_folder = './voice_gallery/' + speaker_id
    if spk_folder not in spk_list:
        os.makedirs(spk_folder)
    n_file = len(glob.glob(os.path.join(spk_folder, '*.npy')))

    if re_register == True:
        feat_path = os.path.join(spk_folder, 'feature_{0:03d}.npy'.format(n_file + 1))
    else:
        feat_path = os.path.join(spk_folder, 'feature_001.npy')
    # Extract new speaker feature
    feature_emb = generate(path_target, model)

    np.save(feat_path, feature_emb)

    return msg

if __name__ == '__main__':
    path = "."
    re_register = False

    # name = 'jtj7587@yonsei.ac.kr-1.mp4'
    # name = 'jtj7587@yonsei.ac.kr-2.mp4'
    # name = 'lppom159@naver.com-1.mp4'
    name = 'lppom159@naver.com-2.mp4'

    predict(path, name, re_register)
