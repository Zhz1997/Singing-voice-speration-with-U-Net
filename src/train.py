import librosa
from librosa.util import find_files
import numpy as np
from unet import unet
import tensorflow as tf

# Allow import config from parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from config import *

patchSize = 128

def readNpzToLists(filePath):
    fileList = find_files(filePath, ext="npz")
    mixList = []
    instruList = []
    vocalList = []
    for file in fileList:
        data = np.load(file)
        mixList.append(data["mix"])
        instruList.append(data["acc"])
        vocalList.append(data["vocal"])

    return mixList, instruList, vocalList

def getSampleFromList(mixMagList,targetMagList) :
    mixSampleList = []
    targetSampleList = []
    for mix, target in zip(mixMagList,targetMagList) :
        starts = np.random.randint(0, mix.shape[1] - patchSize, (mix.shape[1] - patchSize) // 10)
        for start in starts:
            end = start + patchSize
            mixSampleList.append(mix[1:, start:end, np.newaxis])
            targetSampleList.append(target[1:, start:end, np.newaxis])
    return np.asarray(mixSampleList, dtype=np.float32), np.asarray(targetSampleList, dtype=np.float32)

def getMag(specList):
    result = []
    for s in specList:
        mag, _ = librosa.magphase(s)
        result.append(mag)
    return result


if __name__ == '__main__':
    # load DSD100_Npz
    mixListD, instruListD, vocalListD = readNpzToLists("../DSD100_Npz/Dev")
    # load MedlyDb_Npz
    mixListM, instruListM, vocalListM = readNpzToLists("../MedleyDB_Npz")

    mixList = mixListD + mixListM
    vocalList = vocalListD + vocalListM
    instruList = instruListD + instruListM

    mixMagList = getMag(mixList)
    vocalMagList = getMag(vocalList)
    instruMagList = getMag(instruList)


    mix, target = getSampleFromList(mixMagList, vocalMagList)
    unetModel = tf.compat.v1.estimator.Estimator(model_fn=unet, model_dir="./model")
    inputFn = tf.compat.v1.estimator.inputs.numpy_input_fn(x = {"mix":mix}, y = target, batch_size=16, shuffle=False, num_epochs=40)
    unetModel.train(input_fn = inputFn)
