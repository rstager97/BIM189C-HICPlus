from tqdm import tqdm
from LoadData import LoadHIC, LoadSmallHIC, splitData
import numpy as np
from tensorflow.keras.utils import Sequence

numSeg = 16

#split into 16 segments
def GenSplit(file = 'https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic',
            scalerate = 16,
            chromosome = 19,
            name = 'gm12878_in-situ_combined_'):
    highres_sub, lowres_sub = LoadHIC(file, scalerate, chromosome, name)
    sz = highres_sub.shape
    blockSize = int(sz[0]/numSeg)

    for i in tqdm(range(numSeg)):
        Hig = highres_sub[blockSize*i:blockSize*(i+1),:,:,:]
        low = lowres_sub[blockSize*i:blockSize*(i+1),:,:,:]

        Hig = np.rollaxis(Hig, 1, 4)
        low = np.rollaxis(low, 1, 4)

        np.save("data/"+name+"highres"+ str(i), Hig)
        np.save("data/"+name+"lowres"+ str(i), low)

# def HICGenerator(batch_size = 1):
#     name = 'gm12878_in-situ_combined_'
#     while True:
#         choice = np.random.randint(0, high = numSeg)
#         highres_sub = np.load("data/" + name + "highres" + str(choice) + '.npy')
#         lowres_sub = np.load("data/" + name + "lowres" + str(choice) + '.npy')
#         np.random.shuffle(lowres_sub)
#         np.random.shuffle(highres_sub)
#         X = lowres_sub
#         Y = highres_sub
#         yield (X,Y)

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, name = 'gm12878_in-situ_combined_'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        # self.labels = labels
        # self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.name = name

    def __len__(self):
        'Denotes the number of batches per epoch'
        return numSeg

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        choice = np.random.randint(1, high = numSeg) # valitdation is index 0
        highres_sub = np.load("data/" + self.name + "highres" + str(choice) + '.npy')
        lowres_sub = np.load("data/" + self.name + "lowres" + str(choice) + '.npy')
        np.random.shuffle(lowres_sub)
        np.random.shuffle(highres_sub)
        X = lowres_sub
        Y = highres_sub
        return X,Y

    def Get_Val(self):
        choice = 1
        highres_sub = np.load("data/" + self.name + "highres" + str(choice) + '.npy')
        lowres_sub = np.load("data/" + self.name + "lowres" + str(choice) + '.npy')
        np.random.shuffle(lowres_sub)
        np.random.shuffle(highres_sub)
        X = lowres_sub
        Y = highres_sub
        return X,Y