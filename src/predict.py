import numpy as np
from librosa.core import istft, load, stft, magphase
from librosa.output import write_wav    
from config import *
import keras as keras

if __name__ == '__main__':
    # load test audio and convert to mag/phase
    mix_wav, _ = load("../wav_files/mixture.wav", sr=SAMPLE_RATE)
    mix_wav_mag, mix_wav_phase = magphase(stft(mix_wav, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH))

    vocal_wav, _ = load("../wav_files/vocals.wav", sr=SAMPLE_RATE)
    vocal_wav_mag, vocal_wav_phase = magphase(stft(vocal_wav, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH))

    START = 0
    END = START + 128

    mix_wav_mag=mix_wav_mag[:, START:END]
    mix_wav_phase=mix_wav_phase[:, START:END]

    vocal_wav_mag=vocal_wav_mag[:, START:END]
    vocal_wav_phase=vocal_wav_phase[:, START:END]

    # load saved model
    model = keras.models.load_model('../models/vocal_20_test_model.h5')
    #model = keras.models.load_model('../models/vocal_20.h5')

    # predict and write into file
    X=mix_wav_mag[1:].reshape(1, 512, 128, 1)
    y=model.predict(X, batch_size=32)

    target_pred_mag = np.vstack((np.zeros((128)), y.reshape(512, 128)))
    
    write_wav(f'../wav_files/vocal_20_sample_py.wav', istft(
        target_pred_mag * mix_wav_phase
        , win_length=WINDOW_SIZE, hop_length=HOP_LENGTH), SAMPLE_RATE, norm=True)
    write_wav(f'../wav_files/mix_downsampled.wav', istft(
        mix_wav_mag * mix_wav_phase
        , win_length=WINDOW_SIZE, hop_length=HOP_LENGTH), SAMPLE_RATE, norm=True)
    write_wav(f'../wav_files/vocals_downsampled.wav', istft(
        vocal_wav_mag * vocal_wav_phase
        , win_length=WINDOW_SIZE, hop_length=HOP_LENGTH), SAMPLE_RATE, norm=True)