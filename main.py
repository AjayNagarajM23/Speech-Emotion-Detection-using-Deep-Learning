from flask import Flask, request, url_for, redirect, render_template
import numpy as np
from keras.models import load_model
import librosa
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings("ignore")
app = Flask(__name__)


def one_hot_encoding():
    arr = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    encoder = OneHotEncoder()
    encoder.fit_transform(np.array(arr).reshape(-1, 1)).toarray()
    return encoder


def one_hot_encoding_4():
    arr = ['angry', 'calm', 'happy', 'sad']
    encoder = OneHotEncoder()
    encoder.fit_transform(np.array(arr).reshape(-1, 1)).toarray()
    return encoder


def get_features(path):

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


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/detect', methods=["GET", "POST"])
def index():
    val = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            feature = get_features(file)
            feature = np.expand_dims(feature, axis=0)

            loadedModel = load_model('final_model_allSample.h5')

            result = loadedModel.predict([feature], batch_size=1)
            encoder = one_hot_encoding()
            #encoder = one_hot_encoding_4()
            prediction = encoder.inverse_transform(result)
            prediction = np.array(prediction)
            val = prediction[0]
            val = val[0]
            print(val)

    return render_template('up.html', transcript=val)


if __name__ == '__main__':
    app.run(debug=True)
