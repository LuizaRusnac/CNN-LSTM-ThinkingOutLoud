from inspect import EndOfBlock
import numpy as np
from numpy import zeros
from numpy.random import randint

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback


from LogFiles import wLogFile
from LogFiles import wLogString
from LogFiles import nameLogFile
from LogFiles import configParser

from sklearn.model_selection import KFold

def generate_samples(dataset, n_samples):
	images, labels = dataset
	ix = randint(0, images.shape[0], n_samples)
	X, labels = images[ix], labels[ix]
	y = label2mat(labels)
	return X, y

def label2mat(label, nr_cls=11):
	matl = np.zeros((len(label),nr_cls))
	for i in range(len(label)):
		matl[i,int(label[i])] = 1
	return matl

def convolutional_block(in_layer, filter,  activation_fnc = 'relu', recurrent_act = 'sigmoid', kernel_size = (3,3), strides = (1,1), padding = 'same', dropout = 0):
	model = ConvLSTM2D(filter, kernel_size = kernel_size, strides = strides, padding = padding, activation = activation_fnc, recurrent_activation = recurrent_act, data_format = 'channels_last', return_sequences=True)(in_layer)
	if(dropout != 0):
		model = Dropout(dropout)(model)
	# model = MaxPool2D()(model)
	return model

def dense_block(model, nr_neurons, act_fnc, dropout = 0):
	model = Dense(nr_neurons, activation = act_fnc)(model)
	if(dropout != 0):
		model = Dropout(dropout)(model)
	return model

def config_parser(network_config):
	return network_config['in_shape'], network_config['n_classes'], network_config['nr_layers_Conv2D'], network_config['filters'], network_config['activation_fncs'], network_config['recurrent_act'], network_config['nr_layers_Dense'], network_config['dense_act_fnc'], network_config['nr_neurons'], network_config['ll_activation_fnc'], network_config['learning_rate'], network_config['loss'], network_config['metrics'], network_config['id'], network_config['nbatch'], network_config['dropout_Conv2D'], network_config['dropout_Dense']

def CNN_model(network_config):
	in_shape, n_classes, nr_layers_Conv2D, filters, activation_fncs, recurrent_act, nr_layers_Dense, dense_act_fnc, nr_neurons, ll_activation_fnc, learning_rate, loss, metrics, nid, nbatch, dropout_Conv2D, dropout_Dense = config_parser(network_config)
	if(nr_layers_Conv2D != len(filters)):
		raise ValueError("The number of Conv2D layers are different from the number of filters")
	if(nr_layers_Dense != len(nr_neurons)):
		raise ValueError("The number of Dense layers are different from the number of specified neurons")
	if(len(activation_fncs) != 1):
		if(nr_layers_Conv2D > len(activation_fncs)):
			raise ValueError("Not enaugh activation functions!!")
		else:
			if(nr_layers_Conv2D < len(activation_fncs)):
				raise ValueError("Too much activation functions!!")

	in_image = Input(shape=in_shape)
	model = convolutional_block(in_image, filters[0], activation_fncs[0], recurrent_act, dropout = dropout_Conv2D)

	for i in range(1, nr_layers_Conv2D):
		if(len(activation_fncs) == 1):
			model = convolutional_block(model, filters[i], activation_fncs[0])
		else:
			model = convolutional_block(model, filters[i], activation_fncs[i])

	model = Flatten()(model)
	if(nr_layers_Dense > 1):
		for i in range(nr_layers_Dense - 1):
			model = dense_block(model, nr_neurons[i], dense_act_fnc, dropout_Dense)

	out_layer = Dense(n_classes, activation = ll_activation_fnc)(model)
	model = Model(inputs = in_image, outputs = out_layer, name = f"CNN_id{nid}")
	opt = Adam(lr=learning_rate)
	model.compile(loss=loss, optimizer=opt, metrics=[metrics])
	return model

class endCb(Callback): 
	def on_epoch_end(self, epoch, logs={}, n_epochs = 1000): 
		print('>>%d/%d, Tain loss: %.4f, Train acc: %.4f'%(epoch + 1, n_epochs, logs["loss"], logs["accuracy"]))
		log_string = '>>%d/%d, Tain loss: %.4f, Train acc: %.4f'%(epoch + 1, n_epochs, logs["loss"], logs["accuracy"])
		wLogFile(nameLogFile(r"logfile/logfile"), wLogString(log_string))		
		print('>>%d/%d, Validation loss: %.4f, Validation acc: %.4f'%(epoch + 1, n_epochs, logs["val_loss"], logs["val_accuracy"]))
		log_string = '>>%d/%d, Test loss: %.4f, Test acc: %.4f'%(epoch + 1, n_epochs, logs["val_loss"], logs["val_accuracy"])
		wLogFile(nameLogFile(r"logfile/logfile"), wLogString(log_string))

def CNN_train(model_config, dataset, vdataset, tdataset, n_epochs=5, n_batch=128, nid = 0, kval = 1):
	cnn_model = CNN_model(model_config)
	wLogFile(nameLogFile(r"logfile/logfile"), wLogString("********************NEW RUN********************"))
	wLogFile(nameLogFile(r"logfile/logfile"), wLogString(f"***************CNN ID {nid}***************"))
	wLogFile(nameLogFile(r"logfile/logfile"), wLogString("*************Neural Network Config*************"))
	configParser(cnn_model, r"logfile/logfile")
	wLogFile(nameLogFile(r"logfile/logfile"), wLogString("***********************************************"))
	thistory = list()
	xtrain, ytrain = dataset
	xval, yval = vdataset
	es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
	mc = ModelCheckpoint('CNN/'+ f'CNN_id{nid}.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
	history = []
	for k in range(kval):
		cnn_model = CNN_model(model_config)
		ytr = label2mat(ytrain[k])
		yv = label2mat(yval[k])
		mc = ModelCheckpoint('CNN/'+ f'CNN_id{nid}_k{k}.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
		khistory = cnn_model.fit(xtrain[k], ytr, batch_size=n_batch, verbose = 0, validation_data=(xval[k], yv), epochs=n_epochs, callbacks=[es, mc, endCb()])
		history.append({'kval': k, 'history': khistory.history})

		cnn_model = load_model('CNN/'+ f'CNN_id{nid}_k{k}.h5')
		timages, tlabels = tdataset
		ty = label2mat(tlabels[k])
		kthistory = cnn_model.evaluate(timages[k], ty, verbose = 0)
		thistory.append(kthistory)
		print('>>%d/%d, Test loss: %.4f, Test acc: %.4f'%(k + 1, kval, kthistory[0], kthistory[1]))
		log_string = '>>%d/%d, Test loss: %.4f, Test acc: %.4f'%(k + 1, kval, kthistory[0], kthistory[1])
		wLogFile(nameLogFile(r"logfile/logfile"), wLogString(log_string))
			
	return history, thistory