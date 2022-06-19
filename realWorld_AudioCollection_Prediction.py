import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from keras.models import load_model
import librosa
from sklearn.preprocessing import OneHotEncoder
import os
import warnings

warnings.filterwarnings("ignore")

"""
Recording Audio using sounddevice library 
"""


def record_audio():
    fs = 44100
    duration = 5  # seconds
    voiceSample = sd.rec(duration * fs, samplerate=fs, channels=1, dtype='float64')
    print("Start Speaking Now....")
    sd.wait()
    print("Audio recording complete predicting emotion")
    #sd.play(voiceSample, fs)
    #sd.wait()
    print("Playing Audio Complete")
    write('sample.wav', fs, voiceSample)


"""
Performing one hot encoding to sample emotions so we can use the inverse 
of that on the result
"""


def one_hot_encoding():
    arr = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    encoder = OneHotEncoder()
    encoder.fit_transform(np.array(arr).reshape(-1, 1)).toarray()
    return encoder


"""
Feature Extraction of Audio Samples
"""


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen
    # above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation, we are performing the feature extraction to only one file
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    return result


def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result


"""
Recording the audio
Loading the model
Making prediction using the model
"""
while True:
    record_audio()
    feature = get_features('sample.wav')
    feature = np.expand_dims(feature, axis=0)

    loadedModel = load_model('finalModel_Acc_73_61.h5')

    result = loadedModel.predict([feature], batch_size=1)
    encoder = one_hot_encoding()
    prediction = encoder.inverse_transform(result)
    print(prediction)
    os.remove('sample.wav')
