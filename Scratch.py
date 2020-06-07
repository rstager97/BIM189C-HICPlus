from Models import *
from LoadData import LoadHIC, LoadSmallHIC, splitData
from HICGenerator import DataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

# =========================== Create Data Set ===========================
#highres, lowres_sub = LoadSmallHIC()
highres, lowres_sub = LoadHIC()

highres = np.rollaxis(highres, 1, 4)
lowres_sub = np.rollaxis(lowres_sub, 1, 4)
print(highres.shape)

# X_test, Y_test, X_train, Y_train = splitData(highres, lowres_sub)
# print(X_test.shape)
# print(Y_test.shape)
# print(X_train.shape)
# print(Y_train.shape)
# =========================== Train Models ===========================
# model = HICplus_Default()
# model.compile(loss='mean_squared_error', optimizer='adam')
# # define the checkpoint
# filepath = "HICplus_Default_model_scratch.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]
# # fit the model
# model.fit(lowres_sub, highres, epochs=5, batch_size=50, callbacks=callbacks_list, validation_split = 0.1)

# model = LapSRN()
# model.compile(loss='mean_squared_error', optimizer='adam')
# # define the checkpoint
# filepath = "HICplus_LapSRN_model_scratch.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]
# # fit the model
# # generator = DataGenerator()
#
# model.fit(y= highres, x =lowres_sub,
#           batch_size = 512,
#           epochs=5,
#           validation_split = 0.1,
#           callbacks=callbacks_list)

Genmodel = HICplus_Default()
Genmodel.compile(loss='mean_squared_error', optimizer='adam')

desmodel = Discriminator()
desmodel.compile(loss='binary_crossentropy', optimizer='adam')
filepath = "HICplus_GAN_Defualt_model_scratch.h5"
TrainGAN(Genmodel, desmodel, lowres_sub, highres, 10, 1024, Fname=filepath)

Genmodel = LapSRN()
Genmodel.compile(loss='mean_squared_error', optimizer='adam')

desmodel = Discriminator()
desmodel.compile(loss='binary_crossentropy', optimizer='adam')
filepath = "HICplus_GAN_LapSRN_model_scratch.h5"
TrainGAN(Genmodel, desmodel, lowres_sub, highres, 10, 1024, Fname=filepath)
