import os
import torch
import copy
from tqdm import tqdm

from torch import nn
from mnist_model import MLP2

from client_node import ClientNode
from data_loader import Data_Loader


#from torchsummary import summary

#import torch_xla
#import torch_xla.core.xla_model as xm
#import torch_xla.debug.metrics as met
#import torch_xla.distributed.parallel_loader as pl
#import torch_xla.distributed.xla_multiprocessing as xmp
#import torch_xla.utils.utils as xu

#import torch_xla
#import torch_xla.core.xla_model as xm

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#SERIAL_EXEC = xmp.MpSerialExecutor()
#WRAPPED_MODEL = xmp.MpModelWrapper(CNNModel())

#device = xm.xla_device()

torch.manual_seed(1)

class Trainer:
	def __init__(self, n_clients, learning_rate, lr_decay, batch_size, epochs, n_local_epochs, 
		partition_data_file, w_dir, acc_dir, z_dir, 
		dataset, pre_trained_w_file = None):
		
		#self.model = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(device) #cifar10, cifar100
		#summary(self.model, input_size=(3, 32, 32)) #channel, img_size, img_size
		#self.model_name = "sCNN"

		#self.model = CNN2(hidden_dims=[256, 128], output_dim=10).to(device) #cifar100, cifar10
		#summary(self.model, input_size=(3, 32, 32)) #channel, img_size, img_size
		#self.model_name = "CNN2_dr10"

		self.model =  MLP2().to(device) #MNIST().to(device) #CNNModel().to(device) MNIST FEMNIST
		#summary(self.model, (1, 28, 28))
		self.model_name = "MLP2"

		self.learning_rate = learning_rate #* xm.xrt_world_size()#learning_rate
		self.lr_decay = lr_decay
		self.batch_size = batch_size
		self.epochs = epochs
		self.data_loader = Data_Loader(dataset, batch_size)#Data_Loader_EfficientNet_Cifar100(batch_size)
		self.clients = self.generate_clients(n_clients)
		self.n_local_epochs = n_local_epochs

		self.train_data = None
		self.test_data = None


		self.partition_data_file = partition_data_file
		self.w_dir = w_dir
		self.acc_dir = acc_dir
		self.z_dir = z_dir

		self.w_file = "FL_" + self.model_name + "lr" + str(self.learning_rate) + "_dc" + str(self.lr_decay)
		self.w_file = self.w_file + "_B" + str(batch_size) + "_L" + str(n_local_epochs) + "_G" + str(epochs * n_local_epochs) + "_" + partition_data_file+  ".weight.pth"

		self.acc_file = self.w_file + ".acc.csv"

		print ("Configuration: ")
		print ("num_clients: ", n_clients)
		print("L, G: ", n_local_epochs, ", ", epochs)
		print("B, lr, lr_decay: ", batch_size, ", ", learning_rate, ", ", lr_decay)
		print("dataset, model: ", dataset, ", ", self.model_name)
		print("partition_data_file: ", partition_data_file)
		if (pre_trained_w_file != None):
			print("Loading weight from " + pre_trained_w_file)
			self.model.load_state_dict(torch.load(self.w_dir + pre_trained_w_file)) 

	def generate_clients(self, n_clients):
		clients = [] 
		for i in range(n_clients):
			client = ClientNode(self.learning_rate)
			clients.append(client)

		return clients


	def send_model_to_clients(self):
		for client in self.clients:
			client.model["model"] = copy.deepcopy(self.model)

	def average_models(self, models, coefficients):
		averaged_model = copy.deepcopy(self.model)
		p = np.asarray(coefficients)/sum(coefficients)

		with torch.no_grad():
			averaged_values = {}
			for name, param in averaged_model.named_parameters():
				averaged_values[name] = nn.Parameter(torch.zeros_like(param.data))
			i = 0
			for model in models:
				for name, param in model.named_parameters():
					averaged_values[name] += p[i] * param.data
				i += 1

			for name, param in averaged_model.named_parameters():
				param.data = averaged_values[name]

		return averaged_model
	
	def train(self):

		if  not os.path.exists(self.acc_dir):
			os.makedirs(self.acc_dir)

		if  not os.path.exists(self.acc_dir + "FL"):
			os.makedirs(self.acc_dir + "FL")
		acc_path = self.acc_dir + "FL/"

		coefficients = np.asarray([len(cl.data) for cl in self.clients])

		with open(acc_path + self.acc_file, 'w') as the_file:
			the_file.write("global_round,train_loss,train_acc,test_loss,test_acc\n")
        
		print("Start training...")
		lr = self.learning_rate

		
		for epoch in range(self.epochs):
			print(f"Epoch {epoch+1}/{self.epochs}")
			# Send model to all clients
			self.send_model_to_clients()

			# Update local model for several epochs
			print("Local updating on clients")
			for epoch1 in range(self.n_local_epochs):
				for i in tqdm(range(len(self.clients))):
					self.clients[i].train(device, lr)
			lr = lr * self.lr_decay
			
			#lr  = self.learning_rate

			# Get back and average the local models
			client_models = [client.model["model"] for client in self.clients]
			self.model = self.average_models(client_models, coefficients)

			# Validate new model


			train_loss, train_acc = self.train_stats()
			print("Training statistic: ")
			print("Accuracy ","{0:.4}%".format(train_acc) , " Loss: ", f'{train_loss:.3}')

			test_loss, test_acc = self.validate(load_weight=False)
			
			with open(acc_path + self.acc_file, 'a') as the_file:
				the_file.write(str((epoch + 1) * self.n_local_epochs) + "," + str(train_loss) + "," + str(train_acc) + "," + str(test_loss) + "," + str(test_acc)+ "\n")
            
		
		the_file.close()
		self.save_model()

	def train_stats(self):
		train_acc = 0
		train_loss = 0
		s = sum(np.asarray([len(cl.data) for cl in self.clients]))

		p = [len(cl.data)/s for cl in self.clients]
		i = 0
		for cl in self.clients:
			loss, acc = cl.train_stats(device)
			train_acc += p[i] * acc
			train_loss += p[i] * loss
			i += 1

		return train_loss, train_acc 

	def validate(self, load_weight=False):

		print("Validation statistic...")
		if load_weight == True:
			self.model.load_state_dict(torch.load(self.w_dir + self.w_file))

		self.model.eval()
		corrects = 0
		loss = 0

		test_data = self.data_loader.test_data

		with torch.no_grad():
			for batch_idx, (images, labels) in enumerate(test_data):
				images, labels = images.to(device), labels.to(device)
				output = self.model(images)
				pred = output.argmax(dim=1)
				corrects += pred.eq(labels.view_as(pred)).sum().item()

				loss += nn.CrossEntropyLoss()(output, labels).item()


		total_test = len(test_data)*self.batch_size
		accuracy = 100*corrects/total_test
		loss = loss/len(test_data)

		print("Number of corrects: {}/{}".format(corrects, len(test_data)*self.batch_size))
		print("Accuracy, {}%".format(accuracy), " Loss: ", f'{loss:.3}')
		print("-------------------------------------------")

		return loss, accuracy

	def save_model(self):
		print("Saving model...")
		if not os.path.exists(self.w_dir):
			os.makedirs(self.w_dir)
		torch.save(self.model.state_dict(), self.w_dir + self.w_file)
		print("Model saved!")


	
	def run_train(self):
		# load and distribute data to clients
		print("Creating data distribution for clients...")
		if (self.partition_data_file != None):
			print("Partition_data_file: ", self.partition_data_file)
			self.train_data = self.data_loader.prepare_in_batch_given_partition(len(self.clients), self.z_dir +  self.partition_data_file)
		else:
			print("No partition_data_file")
			return

		self.test_data = self.data_loader.test_data

		print("Distributing data...")
		for client_id, client_data in tqdm(self.train_data.items()):
			self.clients[client_id].data = client_data

		print("Computing label distribution across clients: ")
		for i, cl in enumerate(self.clients): 
			list_labels = cl.get_labels(device)
			print("client  ", i, " num_labels: ", len(list_labels), " ",  list_labels)

		

		self.train()
        
