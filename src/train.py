import numpy as np
from keras import Input
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from config import *
from model import unet
from librosa.util import find_files


def load_npz(target=None, first=None):
    npz_files = find_files('../DSD100_Npz/Dev', ext="npz")[:first]
    # npz_files = find_files('../numpy', ext="npz")[:first]
    for file in npz_files:
        npz = np.load(file)
        assert(npz["mix"].shape == npz[target].shape)
        yield npz['mix'], npz[target]

def sampling(mix_mag, target_mag):
    X, y = [], []
    for mix, target in zip(mix_mag, target_mag):
        starts = np.random.randint(0, mix.shape[1] - PATCH_SIZE, (mix.shape[1] - PATCH_SIZE) // SAMPLE_STRIDE)
        for start in starts:
            end = start + PATCH_SIZE
            X.append(mix[1:, start:end, np.newaxis])
            y.append(target[1:, start:end, np.newaxis])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


if __name__ == '__main__':
    mix_mag, target_mag = zip(*load_npz(target='vocal', first=-1))

    model = unet()
    model.compile(optimizer='adam', loss='mean_absolute_error')

    for e in range(EPOCH):
        X, y = sampling(mix_mag, target_mag)
        model.fit(X, y, batch_size=BATCH, verbose=1, validation_split=0.01)
        model.save('../models/vocal_{:0>2d}.h5'.format(e+1), overwrite=True)


