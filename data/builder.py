import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import librosa
from CONFIG import DATA_PATH, SOUND_LENGTH, FREQUENCY, MFCC_CHANNELS
import model
'''
read sound file -> tweak sound length -> apply MFCC -> MinMax -> append to data set
'''
def build_data_from(root):
    dataset = []
    dir_sounds = os.listdir(DATA_PATH+root)

    for sound_name in dir_sounds:
        librosa_obj = "data/"+root+"/"+sound_name
        sound, sound_rate = librosa.load(librosa_obj, sr=FREQUENCY)

        sound_tweaked = sound_len_tweak(sound)
        sound_mfcc = librosa.feature.mfcc(y=sound_tweaked, sr=sound_rate, n_mfcc=MFCC_CHANNELS)
        sound_scaled = min_max_transform(sound_mfcc)

        dataset.append(sound_scaled)

    # Due to MFCC we set model input dynamically (so we don't have to check post MFCC audio dimensions)
    model.POST_MFCC_SOUND_LENGTH = sound_mfcc.shape[1]
    return dataset


zeroes_array = [0] * SOUND_LENGTH  # Array to fill length of shorter sounds (to match SOUND_LENGTH)
# Function forces sound length to be equal to constant value
def sound_len_tweak(sound_to_tweak):
    if sound_to_tweak.shape[0] < SOUND_LENGTH:
        required_length = SOUND_LENGTH - sound_to_tweak.shape[0]
        sound_to_tweak = np.append(sound_to_tweak, zeroes_array[:required_length])
    else:
        sound_to_tweak = sound_to_tweak[:SOUND_LENGTH]
    return sound_to_tweak


def min_max_transform(data_to_scale):
    scaler = MinMaxScaler()
    min_maxed = scaler.fit_transform(data_to_scale)
    return min_maxed
