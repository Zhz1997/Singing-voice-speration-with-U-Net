import librosa
from librosa.util import find_files
import numpy as np
from unet import unet

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
    mixListD, instruListD, vocalListD = readNpzToLists("../DSD100_Npz/Dev")
    # load MedlyDb_Npz
    mixListM, instruListM, vocalListM = readNpzToLists("../MedleyDB_Npz")
    mixList = mixListD + mixListM
    vocalList = vocalListD + vocalListM
    instruList = instruListD + instruListM
    
    unet(mixList, vocalList)