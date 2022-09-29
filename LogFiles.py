from datetime import datetime

def wLogFile(name, string, path = ""):
	filename = name
	file = open(filename,'a+')
	file.writelines(string)
	file.close()

def wLogString2(ant_string, string, flag = 0):
	now = datetime.now()
	dt_string = now.strftime("[%d-%m-%Y %H:%M:%S]")

	if(flag == 0):
		return ant_string + '\n' + dt_string + " " + string

	if(flag == 1):
		return '\n' + dt_string + " " + ant_string + '\n' + dt_string + " " + string

def wLogString(string):
	now = datetime.now()
	dt_string = now.strftime("[%d-%m-%Y %H:%M:%S]")
	return '\n' + dt_string + " " + string

def nameLogFile(filename):
	now = datetime.now()
	return now.strftime("%d_%m_%Y_logfile.txt")

def configParser(model, logfile = "logfile"):
	config = model.get_config()
	string = "Number of layers: %d"%len(config['layers'])
	wLogFile(nameLogFile(logfile), wLogString(string))
	for i in range(len(config['layers'])):
		string = "Layer %d: %s"%((i+1), config['layers'][i]['class_name'])
		wLogFile(nameLogFile(logfile), wLogString(string))
		string = "\tName: %s"%(config['layers'][i]['config']['name'])
		wLogFile(nameLogFile(logfile), wLogString(string))
		if(config['layers'][i]['class_name']=='InputLayer'):
			string = f"\tInput Shape: {config['layers'][i]['config']['batch_input_shape']}"
			wLogFile(nameLogFile(logfile), wLogString(string))
			string =f"\tActivation Function: None"
			wLogFile(nameLogFile(logfile), wLogString(string))
		if(config['layers'][i]['class_name']=='LSTM'):
			if "batch_input_shape" in config['layers'][i]['config'].keys():
				string =f"\tInput Shape: {config['layers'][i]['config']['batch_input_shape']}"
				wLogFile(nameLogFile(logfile), wLogString(string))
			string =f"\tLayer Units: {config['layers'][i]['config']['units']}"
			wLogFile(nameLogFile(logfile), wLogString(string))
			string =f"\tActivation Function: {config['layers'][i]['config']['activation']}"
			wLogFile(nameLogFile(logfile), wLogString(string))
			string =f"\tRecurrent Activation Function: {config['layers'][i]['config']['recurrent_activation']}"
			wLogFile(nameLogFile(logfile), wLogString(string))
		if(config['layers'][i]['class_name']=='Dropout'):
			string = f"\tRate: {config['layers'][i]['config']['rate']}"
			wLogFile(nameLogFile(logfile), wLogString(string))
		if(config['layers'][i]['class_name']=='Dense'):
			string = f"\tLayer Units: {config['layers'][i]['config']['units']}"
			wLogFile(nameLogFile(logfile), wLogString(string))
			string = f"\tActivation Function: {config['layers'][i]['config']['activation']}"
			wLogFile(nameLogFile(logfile), wLogString(string))
		if(config['layers'][i]['class_name']=='Conv2D'):
			if "batch_input_shape" in config['layers'][i]['config'].keys():
				string =f"\tInput Shape: {config['layers'][i]['config']['batch_input_shape']}"
				wLogFile(nameLogFile(logfile), wLogString(string))
			string = f"\tLayer Filters: {config['layers'][i]['config']['filters']}"
			wLogFile(nameLogFile(logfile), wLogString(string))
			string = f"\tKernel Size: {config['layers'][i]['config']['kernel_size']}"
			wLogFile(nameLogFile(logfile), wLogString(string))
			string = f"\tStrides: {config['layers'][i]['config']['strides']}"
			wLogFile(nameLogFile(logfile), wLogString(string))
			string = f"\tPadding: {config['layers'][i]['config']['padding']}"
			wLogFile(nameLogFile(logfile), wLogString(string))
			string = f"\tDilation Rate: {config['layers'][i]['config']['dilation_rate']}"
			wLogFile(nameLogFile(logfile), wLogString(string))
			string = f"\tActivation Function: {config['layers'][i]['config']['activation']}"
			wLogFile(nameLogFile(logfile), wLogString(string))
		if(config['layers'][i]['class_name']=='LeakyReLU'):
			string = f"\tAlpha: {config['layers'][i]['config']['alpha']}"
			wLogFile(nameLogFile(logfile), wLogString(string))
	if(model.optimizer):
		string = f"Optimizer Name: {model.optimizer._name}"
		wLogFile(nameLogFile(logfile), wLogString(string))
		string = f"Learning rate: {model.optimizer.lr.numpy()}"
		wLogFile(nameLogFile(logfile), wLogString(string))
		string = f"Loss Function: {model.loss}"
		wLogFile(nameLogFile(logfile), wLogString(string))