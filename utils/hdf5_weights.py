import numpy as np
import h5py

def isGroup(obj):
    if isinstance(obj,h5py.Group):
        return True
    return False

def isDataset(obj):
    if isinstance(obj,h5py.Dataset):
        return True
    return False

def getDatasetFromGroup(datasets,obj):
    if isGroup(obj):
        for key in obj:
            x = obj[key]
            getDatasetFromGroup(datasets,x)
    else:
        datasets.append(obj)

def getWeightsFromDataset(obj):
    w = np.array(obj)
    return w 

def getWeightsForLayer(layerName,fileName):

    weights = []
    with h5py.File(fileName, mode='r') as f:
        for key in f:
            if layerName in key:
                obj = f[key]
                datasets = []
                getDatasetFromGroup(datasets,obj)

                for dataset in datasets:
                    w = getWeightsFromDataset(dataset)
                    weights.append(w)

                break

        f.close()

    return weights
