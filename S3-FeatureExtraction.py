import numpy as np
from FileUtils import load_data
import preprocessing
import featureExtr

# Establishing parameters
window = 250
domain = 'frequency'

print("Loading data...")
names = ['xtrain_rawdata_%ds_50tr-50tst_kfold_lstm.npy'%(window),
    'ytrain_rawdata_%ds_50tr-50tst_kfold_lstm.npy'%(window),
	'xval_rawdata_%ds_50tr-50tst_kfold_lstm.npy'%(window),
	'yval_rawdata_%ds_50tr-50tst_kfold_lstm.npy'%(window),
    'xtest_rawdata_%ds_50tr-50tst_kfold_lstm.npy'%(window),
    'ytest_rawdata_%ds_50tr-50tst_kfold_lstm.npy'%(window)]

xtrain, ytrain = load_data(names[0], names[1])
xval, yval = load_data(names[2], names[3])
xtest, ytest = load_data(names[4], names[5])
print("Loaded complete!")
print("Data train shape: ", xtrain.shape)
print("Data val shape: ", xval.shape)
print("Data test shape: ", xtest.shape)

xtr = np.zeros((xtrain.shape[0],xtrain.shape[1],xtrain.shape[2], xtrain.shape[3],int(xtrain.shape[4]/2)))
xv = np.zeros((xval.shape[0],xval.shape[1],xval.shape[2], xval.shape[3],int(xval.shape[4]/2)))
xtst = np.zeros((xtest.shape[0],xtest.shape[1],xtest.shape[2], xtest.shape[3],int(xtest.shape[4]/2)))

print("Computing features...")
if domain == 'frequency':
    for i in range(xtrain.shape[0]):
        for j in range(xtrain.shape[2]):
            xtr[i,:,j,:,:] = featureExtr.spectrumChn(xtrain[i,:,j,:,:])
    for i in range(xval.shape[0]):
        for j in range(xval.shape[2]):
            xv[i,:,j,:,:] = featureExtr.spectrumChn(xval[i,:,j,:,:])
    for i in range(xtest.shape[0]):
        for j in range(xtest.shape[2]):
            xtst[i,:,j,:,:] = featureExtr.spectrumChn(xtest[i,:,j,:,:])
else:
	xtr = xtrain
	xv = xval
	xtst = xtest

# print(xtr)
xtrain = np.zeros((xtr.shape[0], xtr.shape[1], xtr.shape[2], xtr.shape[3], xtr.shape[3]))
xval = np.zeros((xv.shape[0], xv.shape[1], xv.shape[2], xv.shape[3], xv.shape[3]))
xtest = np.zeros((xtst.shape[0], xtst.shape[1], xtst.shape[2], xtst.shape[3], xtst.shape[3]))

for i in range(len(xtrain)):
	xtrain[i] = featureExtr.chConvLSTM(xtr[i])
for i in range(len(xval)):
	xval[i] = featureExtr.chConvLSTM(xv[i])
for i in range(len(xtest)):
	xtest[i] = featureExtr.chConvLSTM(xtst[i])
print("Done!")
print("Feature train shape: ", xtrain.shape)
print("Feature val shape: ", xval.shape)
print("Feature test shape: ", xtest.shape)

print("Normalizing features...")
for i in range(len(xtrain)):
    for j in range(xtrain.shape[2]):
	    xtrain[i,:,j,:,:], minim, maxim = preprocessing.featureStd(xtrain[i,:,j,:,:], flag = 1)
for i in range(len(xval)):
    for j in range(xval.shape[2]):
	    xval[i,:,j,:,:] = preprocessing.featureStd(xval[i,:,j,:,:], minim, maxim)
for i in range(len(xtest)):
    for j in range(xtest.shape[2]):
	    xtest[i,:,j,:,:] = preprocessing.featureStd(xtest[i,:,j,:,:], minim, maxim)
print("Done!")
print("Normalized xtrain features:")
print(xtrain)
print("Normalized xval features:")
print(xval)
print("Normalized xtest features:")
print(xtest)

print("Saving features...")
names = ['xtrain_%ds_50tr-50tst_cov-%s_kfold_lstm.npy'%(window, domain),
    'ytrain_%ds_50tr-50tst_cov-%s_kfold_lstm.npy'%(window, domain),
	'xval_%ds_50tr-50tst_cov-%s_kfold_lstm.npy'%(window, domain),
	'yval_%ds_50tr-50tst_cov-%s_kfold_lstm.npy'%(window, domain),
    'xtest_%ds_50tr-50tst_cov-%s_kfold_lstm.npy'%(window, domain),
    'ytest_%ds_50tr-50tst_cov-%s_kfold_lstm.npy'%(window, domain)]
np.save(names[0], xtrain)
np.save(names[1], ytrain)
np.save(names[2], xval)
np.save(names[3], yval)
np.save(names[4], xtest)
np.save(names[5], ytest)
print("Saved data!")
