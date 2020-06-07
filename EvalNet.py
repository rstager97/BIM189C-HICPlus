from scipy.stats import pearsonr
from LoadData import LoadHIC, LoadSmallHIC, splitData
import numpy as np
from Models import *

def testModel(model, testX, testY):
  ypred=model.predict(testX)
  score = pearsonr(ypred.flatten(), testY.flatten())

  return score, ypred



highres, lowres_sub = LoadSmallHIC()
#highres, lowres_sub = LoadHIC()

highres = np.rollaxis(highres, 1, 4)
lowres_sub = np.rollaxis(lowres_sub, 1, 4)


model = HICplus_Default()
# model.compile(loss='mean_squared_error', optimizer='adam')
model.load_weights("HICplus_Default_model.h5")
s, ypredDefault = testModel(model, lowres_sub, highres)
print("Defualt model score is {}".format(s))

model = LapSRN()
# model.compile(loss='mean_squared_error', optimizer='adam')
model.load_weights("HICplus_LapSRN_model_Nightly_Good.h5")
s, ypredLap = testModel(model, lowres_sub, highres)
print("LapSRN model score is {}".format(s))

import matplotlib.pyplot as plt

highres = highres.reshape(49,40,40) #magic numbers yay!
ypredDefault = ypredDefault.reshape(49,40,40)
ypredLap = ypredLap.reshape(49,40,40)
lowres_sub = lowres_sub.reshape(49,40,40) #magic numbers yay!

#================== Comparing ==================
plt.subplot(2,3,1)
plt.imshow(highres[48,:,:], cmap='hot')
plt.colorbar()
plt.title('True')

plt.subplot(2,3,4)
plt.imshow(lowres_sub[48,:,:], cmap='hot')
plt.colorbar()
plt.title('Down Sampled')

plt.subplot(2,3,2)
plt.imshow(ypredDefault[48,:,:], cmap='hot')
plt.colorbar()
plt.title('HiC PlusPredicted')

plt.subplot(2,3,5)
plt.imshow(ypredLap[48,:,:], cmap='hot')
plt.colorbar()
plt.title('LapSRN Predicted')

#================== MSE Comparing ==================
print('MSE compare')
mse = ((highres - ypredDefault)**2).mean(axis=None)
print(mse)
plt.subplot(2,3,3)
plt.imshow(highres[48,:,:]-ypredDefault[48,:,:], cmap='hot')
plt.colorbar()
plt.title('Difference HiC Plus')


mse = ((highres - ypredLap)**2).mean(axis=None)
print(mse)
plt.subplot(2,3,6)
plt.imshow(highres[48,:,:]-ypredLap[48,:,:], cmap='hot')
plt.colorbar()
plt.title('Difference LapSRN')

plt.tight_layout()
plt.show()

