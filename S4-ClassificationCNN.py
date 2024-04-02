from FileUtils import load_data
import tensorflow
from CNN_lstm import CNN_model
from CNN_lstm import CNN_train
import numpy as np
from datetime import datetime
import pandas as pd
from numpy import asarray
import json
from tensorflow.keras.models import load_model

window = 250
domain = 'frequency'
detailes = "LSTM No mean extraction with feature standardization"

names = ['xtrain_%ds_50tr-50tst_cov-%s_kfold_lstm.npy'%(window, domain),
    'ytrain_%ds_50tr-50tst_cov-%s_kfold_lstm.npy'%(window, domain),
	'xval_%ds_50tr-50tst_cov-%s_kfold_lstm.npy'%(window, domain),
	'yval_%ds_50tr-50tst_cov-%s_kfold_lstm.npy'%(window, domain),
    'xtest_%ds_50tr-50tst_cov-%s_kfold_lstm.npy'%(window, domain),
    'ytest_%ds_50tr-50tst_cov-%s_kfold_lstm.npy'%(window, domain)]

xtrain, ytrain = load_data(names[0], names[1])
xvalid, yvalid = load_data(names[2], names[3])
xtest, ytest = load_data(names[4], names[5])

for i in range(xtrain.shape[0]):
    rnd = np.random.permutation(xtrain.shape[1])
    xtrain[i] = xtrain[i, rnd]
    ytrain[i] = ytrain[i, rnd]

dataset = [xtrain, ytrain]
tdataset = [xtest, ytest]
vdataset = [xvalid, yvalid]

dim = xtrain.shape
tdim = xtest.shape
vdim = xvalid.shape

now = datetime.now()
nid = now.strftime("%d%m%H%M%S")

n_epochs = 1000
batch_nr = 32

model_config = {'in_shape': (xtrain.shape[2], xtrain.shape[3], xtrain.shape[4], 1), 'n_classes': 11,
'nr_layers_Conv2D': 3, 'filters': [64, 128, 64], 'dropout_Conv2D': 0, 'activation_fncs': ['relu'], 'recurrent_act': 'sigmoid',
'nr_layers_Dense': 2, 'dropout_Dense': 0, 'nr_neurons': [64, 11], 
'dense_act_fnc': 'tanh', 'll_activation_fnc':'softmax', 'learning_rate': 0.0001,
'loss': 'categorical_crossentropy', 'metrics': 'accuracy', 'id': nid, 'nbatch':batch_nr}

cnn_model = CNN_model(model_config)

def mean_loss_history(history):
    mean = 0
    for i in range(len(history)):
        mean = mean + history[i].history['loss'][len(history[i].history['loss'])-1]
    mean = mean/len(history)
    return mean

cnn_model.summary()
kval = xtrain.shape[0]
history, thistory = CNN_train(model_config, dataset, vdataset, tdataset, n_epochs, batch_nr, nid, kval = kval)
thistory = np.asarray(thistory)
nepochs_fin = np.zeros(kval)

json.dump(history, open(f'history/history_CNN_id{nid}', 'w'))
for i in range(kval):
    nepochs_fin[i] = len(history[i]["history"]["loss"])

np.save('history/' + f'history_CNN_id{nid}.npy', history)
np.save(rf'history/thistory_CNN_id{nid}.npy', thistory)

data_admin = {'model': [f'CNN ({(window/1000):.2f}s) - 50/50 - Cov({domain}) B{band_samples} - {detailes}'],
'id': [str(nid)], 'input_shape': [str(model_config['in_shape'])], 'kfoldval':kval, 'layer_name1': ['Conv2D'],
'nr_layers1': [model_config['nr_layers_Conv2D']], 'filters': [str(model_config['filters'])], 
'activation_function1': [model_config['activation_fncs']], 'Droput_Conv2D': model_config['dropout_Conv2D'],
'layer_name2': ['Dense'], 'nr_layers2': [model_config['nr_layers_Dense']], 
'neurons_nr': [str(model_config['nr_neurons'])], 'dense_act_fnc': [model_config['dense_act_fnc']], 
'Droput_Dense': model_config['dropout_Dense'], 'll_activation_function': [model_config['ll_activation_fnc']],
'optimizer': ['Adam'], 'learning_rate': [model_config['learning_rate']], 'loss': [model_config['loss']], 
'loss_test_value': np.mean(thistory[:,0]), 
'loss_std_test_value': np.std(thistory[:,0]), 
'acc_test_value': np.mean(thistory[:,1]),'acc_std_test_value': np.std(thistory[:,1]),
'nr_epochs': [nepochs_fin], 'nbatch': [batch_nr]}

load_df = pd.read_excel('data02.xlsx', index_col = 0)
df = pd.DataFrame(data=data_admin, columns = ['model', 'id', 'input_shape', 'kfoldval', 'layer_name1',
'nr_layers1', 'filters', 'activation_function1', 'layer_name2', 'nr_layers2', 'neurons_nr',
'dense_act_fnc', 'll_activation_function', 'optimizer', 'learning_rate', 'loss', # 'loss_train_value',
'loss_test_value','loss_std_test_value', 'acc_test_value', 'acc_std_test_value', 'nr_epochs', 'nbatch'])
df = pd.concat([load_df, df], ignore_index=True)
with pd.ExcelWriter("data02.xlsx", mode="a", if_sheet_exists='overlay') as writer:
	df.to_excel(writer)
