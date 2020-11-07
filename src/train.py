import librosa
from librosa.util import find_files
import numpy as np

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


if __name__ == '__main__':
    # load DSD100_Npz
    mixList, instruList, vocalList = readNpzToLists("../DSD100_Npz/Dev")
    unet(mixList, vocalList)