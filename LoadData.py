import sys
sys.path.append("hicplus-master/hicplus")

import utils
import numpy as np
import os

def LoadHIC(file = 'https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic',
            scalerate = 16,
            chromosome = 19,
            name = 'gm12878_in-situ_combined_'):

    if os.path.exists(name+'highres'+'.npy') and os.path.exists(name+'lowres'+'.npy'):
        print("Files found locally, Loading from disk...")
        highres_sub = np.load(name + 'highres' + '.npy')
        lowres_sub = np.load(name + 'lowres' + '.npy')
    else:
        print("Files nod found locally, Loading from web or hic file...")
        highres = utils.matrix_extract(chromosome, 10000, file)
        print('dividing, filtering and downsampling files...')

        highres_sub, index = utils.divide(highres, subImage_size = 40)
        print(highres_sub.shape)

        lowres = utils.genDownsample(highres, 1 / float(scalerate))
        lowres_sub, index = utils.divide(lowres)
        print(lowres_sub.shape)

        np.save(name+"highres",highres_sub)
        np.save(name+"lowres",lowres_sub)

    return highres_sub, lowres_sub

def LoadSmallHIC(file = 'https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic',
            scalerate = 16,
            chromosome = 19,
            name = 'gm12878_in-situ_combined_'):
    if os.path.exists(name+'highres_small'+'.npy') and os.path.exists(name+'lowres_small'+'.npy'):
        print("Small Files found locally, Loading from disk...")
        highres_sub = np.load(name + 'highres_small' + '.npy')
        lowres_sub = np.load(name + 'lowres_small' + '.npy')
    else:
        print("Small Files not found locally")
        highres_sub, lowres_sub = LoadHIC(file, scalerate, chromosome, name)

        highres_sub = highres_sub[1:50,:,:,:]
        lowres_sub = lowres_sub[1:50,:,:,:]

        np.save(name + "highres_small" + '.npy', highres_sub)
        np.save(name + "lowres_small" + '.npy', lowres_sub)

    return highres_sub, lowres_sub

def splitData(X,Y, split=0.10):
    shape = X.shape
    index = np.random.choice(5, int(split*shape[0]), replace=False) # indexes randomly selected

    X_test = X[index, :, :, :]
    Y_test = Y[index, :, :, :]

    not_index = [i for i in range(shape[0]) if i not in index] # lazy conjegate of index
    X_train = X[not_index, :, :, :]
    Y_train = Y[not_index, :, :, :]

    return X_test, Y_test, X_train, Y_train