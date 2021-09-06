import random
import numpy as np
import copy
import torch

from torch import nn, optim

#import torch_xla
#import torch_xla.core.xla_model as xm

random.seed(1)
np.random.seed(1)

class ClientNode:
	def __init__(self, learning_rate):
		self.model = {"model": None, "optim": None, "criterion": None, "loss": None}
		self.data = []
		self.set_labels = set({})
		self.learning_rate = learning_rate
		self.model["criterion"] = nn.CrossEntropyLoss() 
	

	def clear_model(self):
		del self.model["model"]
		self.model["model"] = None


	def train(self, device, lr):
		
		self.model["optim"] = optim.SGD(self.model["model"].parameters(), lr=lr) #mnist + MLP2
		#self.model["optim"] = optim.Adam(self.model["model"].parameters()) #cifar10 + CNN2
		#self.model["optim"] = optim.SGD(self.model["model"].parameters(), lr=lr, momentum=0.99) #cifar10 + CNN3
		#lr=0.001, momentum=0.9
		for batch_idx, (images, labels) in enumerate(self.data):
			images, labels = images.to(device), labels.to(device)

			
			self.model["optim"].zero_grad()
			output = self.model["model"].forward(images)

			loss = self.model["criterion"](output, labels)

			loss.backward()
			self.model["optim"].step()
			
	def train_stats(self, device):

		#print("Traning statistic:...")
		self.model["model"].eval()
		corrects = 0
		loss = 0

		batch_size = 0
		for batch_idx, (images, labels) in enumerate(self.data):
			batch_size = len(images)
			break
		#print("batch_size ", batch_size)
		with torch.no_grad():
			for batch_idx, (images, labels) in enumerate(self.data):
				images, labels = images.to(device), labels.to(device)
				output = self.model["model"](images)
				pred = output.argmax(dim=1)

				corrects += pred.eq(labels.view_as(pred)).sum().item()
				loss += self.model["criterion"](output, labels).item()

		#print("len(client.data[0]): ", len(self.data[0]))
		
		total_train = len(self.data)*batch_size
		accuracy = 100*corrects/total_train
		loss = loss/len(self.data)
		self.model["model"].train()
		#print("client: loss, acc: ", loss, " ", accuracy)
		return loss, accuracy 


	def get_labels(self, device):
		
		for batch_idx, (images, labels) in enumerate(self.data):
			#images, labels = images.to(device), labels.to(device)
			self.set_labels.update(labels.numpy())
		return self.set_labels


	def sum_model(self):
		total = 0
		for name, param in self.model["model"].named_parameters():
			total += torch.sum(param.data)
		print(total)
