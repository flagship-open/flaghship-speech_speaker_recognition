#-*- coding:utf-8 -*-
import io
import os, logging
import glob
import numpy as np
import sys
from flask import Flask, json, Response, request, jsonify
from collections import OrderedDict
from utils.voice_speaker_recognition import voice_registration, voice_identification, voice_gallery
import operator
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
app = Flask(__name__)
from keras.models import load_model
from keras import backend as K
from keras_layer_normalization import LayerNormalization
from best_model.amsoftmax import *

# GPU Setting  : 20%
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config = tf.ConfigProto(device_count = {'GPU': 1})
# config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

# Load model
path = '.'
model_path = './best_model/model_speech_2sec.h5'
speaker_model = load_model(model_path,
                           custom_objects={'LayerNormalization': LayerNormalization,
                                           'amsoftmax_loss': amsoftmax_loss})


@app.route("/Identification_Request", methods=["POST"])
def predict():
    if request.method == "POST":

        result = OrderedDict()
        identify = request.form['identify']
        identify = str_to_bool(identify)
        re_register = request.form['re_register']
        re_register = str_to_bool(re_register)
        path_target = request.form['file_path'] 

        # File is not found 
        if not os.path.exists(path_target):
            result_voice = "Can't find target data. Clarify target path"
            result['100001'] = result_voice
            return json.dumps(result)

        # Perform identification
        if identify == True:
            cos_sim_voice, id_top5_voice = voice_identification(path_target, speaker_model)
            voice_identification_result = id_top5_voice[np.argmax(cos_sim_voice)]
            result_voice = 'Voice Based Speaker Recognition - Speaker ID : {}'\
                            .format(voice_identification_result)

        # Perform registration
        else:
            name_class = os.path.basename(path_target).split('-')[0]
            # Re-register
            if re_register == True:
                result_voice = voice_registration(path_target, name_class, \
                                                  re_register, speaker_model)
                result_voice = 'We register speaker. Speaker ID : {}'.format(name_class)

            else:                                                              
                voice_gallery_exist = voice_gallery('.', name_class)          
                if voice_gallery_exist == False:                               
                    result_voice = voice_registration(path_target, name_class, \
                                                      re_register, speaker_model)
                    result_voice = 'We register speaker. Speaker ID : {}'.format(name_class)
                                                                               
                elif voice_gallery_exist == True:                              
                    cos_sim_voice, id_top5_voice = voice_identification(path_target, \
                                                                        speaker_model)
                    voice_identification_result = id_top5_voice[np.argmax(cos_sim_voice)]
                    result_voice = 'Voice Based Speaker Recognition - Speaker ID : {}'\
                                    .format(voice_identification_result)

        result["100001"] = result_voice

        print(result)
        print('Done!!')

        return json.dumps(result)


def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError



if __name__ == '__main__':
    print("Loading pytorch model and Flask starting server...")
    os.makedirs('voice_gallery', exist_ok=True)
    # load_model()
    print("Network already loading and app running")
    # app.run()
    app.run(host=os.environ.get('165.132.56.182', '0.0.0.0'), port=8888, debug=True)
