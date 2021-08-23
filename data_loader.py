from tqdm import tqdm

import torch

import numpy as np
import random
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

from datasets import MNIST_truncated, CIFAR10_truncated, CIFAR100_truncated, SVHN_custom, FashionMNIST_truncated, CustomTensorDataset, CelebA_custom, FEMNIST, Generated, genData


#run in gg colab
data_DIR = "/content/drive/MyDrive/eFL/data/"
#data_DIR = "../../data" # run local

class Data_Loader:
	def __init__(self, dataset, batch_size):

		self.dataset = dataset
		self.batch_size = batch_size

		transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.1307,), (0.3081,))])
		
		self.train_data = self.load_data(train=True, transform = transform)
		print("type of train_data: ", type(self.train_data))
		#print(self.train_data)
		self.test_data = self.load_test(transform)#DataLoader(dataset=self.load_data(train = False, transform = transform), batch_size=self.batch_size, shuffle=True, drop_last=False)
	
	def load_test(self, transform):
		test_data = self.load_data(train=False, transform = transform)
		x_test = test_data.data
		y_test = test_data.target

		num_batch = int(len(x_test) / self.batch_size)
		
		#x_test = self.normalize(x_test)
		batches = []
		#print("xtest.shape: ", x_test.shape)
		#print("ytest.shape: ", y_test.shape)

		
		for i in range(num_batch):
			start = i * self.batch_size
			end = start + self.batch_size

			batch = TensorDataset(x_test[start:end], y_test[start:end])
			batches.append(batch)
		
		#if end < len(x_test):
		#	batches.append(TensorDataset(x_test[end:len(x_test)], y_test[end:len(y_test)]))
		return DataLoader(ConcatDataset(batches), shuffle=True, batch_size=self.batch_size)
		#return DataLoader(TensorDataset(x_test, y_test), shuffle=True, batch_size=self.batch_size, drop_last = False)
	
	def load_data(self, train, transform):
	    
		if self.dataset == "mnist":
		    # from six.moves import urllib
		    # opener = urllib.request.build_opener()
		    # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
		    # urllib.request.install_opener(opener)
		    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
		    # data = datasets.MNIST(data_DIR, train=train, download=True, transform=transform)
		    # return data

		    mnist_ds = MNIST_truncated(data_DIR, train=train, download=True, transform=transform)
		    #print(mnist_ds.target[0])
		    #exit()
		    mnist_ds.data = self.normalize(mnist_ds.data)
		    return mnist_ds
		

		elif self.dataset == "femnist":
			femnist_ds = FEMNIST(data_DIR, train=train, transform=transform, download=False)
			femnist_ds.target = femnist_ds.target.long()
			return femnist_ds

			#print("data value range: ", np.max(femnist_ds.data), ", ", np.min(femnist_ds.data))
			#print("data size: ", len(femnist_ds.data))
			#print(femnist_ds.data[0])
			#exit()
		elif self.dataset == "cifar100":
			cifa100 = CIFAR100_truncated(data_DIR, train=train, transform=transform, download=True)
			return cifa100

		elif self.dataset == "cifar10":
			transform_train = transforms.Compose([
		    transforms.RandomCrop(32, padding=4),
		    transforms.RandomHorizontalFlip(),
		    transforms.ToTensor()

		    #width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
		    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
			if train == True:
				cifa10 = CIFAR10_truncated(data_DIR, train=train, transform=transform_train, download=True)
			else:
				cifa10 = CIFAR10_truncated(data_DIR, train=train, transform=transform, download=True)
			return cifa10


	def normalize(self, x, mean=0.1307, std=0.3081):
		return (x-mean)/std


	def prepare_iid_data(self, no_clients):
		train_data = self.train_data
		
		return self.distribute_in_shards(train_data.data, train_data.target, no_clients)
		

	def prepare_non_iid_data_option1(self, no_clients):
		'''
		Follow original FL paper
		Sort the data by digit label
		Divide it into 200 shards of size 300
		Assign each of n clients 200/n shards
		'''
		sorted_images = []
		sorted_labels = []

		for number in range(10):
			indices = (self.train_data.target == number).int()

			images = self.train_data.data[indices == 1]
			labels = self.train_data.target[indices == 1]

			#images = self.normalize(images).unsqueeze(1)

			sorted_images += images.unsqueeze(0)
			sorted_labels += labels.unsqueeze(0)

		sorted_images = torch.cat(sorted_images)
		sorted_labels = torch.cat(sorted_labels)

		
		return self.distribute_in_shards(sorted_images, sorted_labels, no_clients)

	def prepare_non_iid_data_option_zipf(self, no_clients, zipfz):

		sorted_images = []
		sorted_labels = []
		
		
		for number in range(10):
			indices = (self.train_data.target == number).int()

			images = self.train_data.data[indices == 1]
			labels = self.train_data.target[indices == 1]

			#images = self.normalize(images).unsqueeze(1)

			sorted_images += images.unsqueeze(0)
			sorted_labels += labels.unsqueeze(0)

			
		sorted_images = torch.cat(sorted_images)
		sorted_labels = torch.cat(sorted_labels)

		#exit()
		p = self.draw_zipf_distribution (no_clients, zipfz)
		return self.distribute_in_batches(sorted_images, sorted_labels, no_clients, p)



	def distribute_in_shards(self, images, labels, no_clients):
		shards = []
		for i in range(200):
			start = i*300
			end = start + 300

			shard_images = images[start:end]
			shard_labels = labels[start:end]

			shard = TensorDataset(shard_images, shard_labels)
			shards.append(shard)

		clients_data = {}

		cl_data_size = np.zeros(no_clients)
		set_not_full_client = set(np.arange(no_clients))
		
		shards_per_client = len(shards)/no_clients
		for shard_idx, shard in enumerate(shards):

			cl_IDs = random.sample(set_not_full_client, 1)
			cl_ID = cl_IDs[0]

			if (cl_data_size[cl_ID] == 0):
				clients_data[cl_ID] = [shard]
			elif (cl_data_size[cl_ID] == 1):
				clients_data[cl_ID].append(shard)
			
			cl_data_size[cl_ID] += 1
			if cl_data_size[cl_ID] == 2:
				set_not_full_client = set_not_full_client - set(cl_IDs)


			
		for client_number, client_data in clients_data.items():
			clients_data[client_number] = DataLoader(ConcatDataset(client_data), shuffle=True, batch_size=self.batch_size)

		return clients_data
	
	
	def distribute_in_batches(self, images, labels, no_clients, p):
		
		num_batch = int(len(images) /self.batch_size)
		client_num_batch = []
		batches = []

		for i in range(num_batch):
			start = i * self.batch_size
			end = start + self.batch_size

			batch_images = images[start:end]
			batch_labels = labels[start:end]

			batch = TensorDataset(batch_images, batch_labels)
			batches.append(batch)
		
		clients_data = {}
		
		start = 0
		for i in range(no_clients):
			end = start + int(p[i] * num_batch)

			#print("client, start, end: ", i, ", ", start, ", ", end)
			clients_data[i] = batches[start:end]
			start = end

		if start < num_batch:
			clients_data[0] += batches[start: num_batch]

		for client_number, client_data in clients_data.items():
			clients_data[client_number] = DataLoader(ConcatDataset(client_data), shuffle=True, batch_size=self.batch_size)

		return clients_data

	def draw_zipf_distribution(self, no_clients, z):
		
		p = (1/np.arange(1,no_clients + 1))**(z)
		return p/sum(p)


	def prepare_in_batch_given_partition(self, num_clients, partition_file):

		images = self.train_data.data
		labels = self.train_data.target
		#images = self.normalize(images).unsqueeze(1)
		
		partition_list = []
		clients_data = {}


		with open(partition_file, 'rb') as f:
			for cl in range(num_clients):		
				partition_list.append(np.load(f))
			
		for cl in range(num_clients):
			clients_data[cl] = []

		
		for cl in range(num_clients):
			start = 0
			num_img_taken = 0

			#print("client ", cl, ": ", len(partition_list[cl]))
			#exit()

			while num_img_taken < len(partition_list[cl]):
				if num_img_taken + self.batch_size <= len(partition_list[cl]):
					end = start + self.batch_size
					num_img_taken += self.batch_size
				else: #not enough number of images for a full batch
				#	end = start + len(partition_list[cl]) - num_img_taken
				#	num_img_taken = len(partition_list[cl])
					break


				img_indices = partition_list[cl][start:end]
				batch_img = images[img_indices]
				
				batch_label = labels[img_indices]


				#print("shape of img and labels: ")
				#print(batch_img.shape)
				#print(batch_label.shape, " ", batch_label)
				#print(batch_img)

				batch = TensorDataset(batch_img, batch_label)
				clients_data[cl].append(batch)

				start = end
			#print("cl ", cl, " num_batches: ", len(clients_data[cl]))
		#s = 0
		for client_number, client_data in clients_data.items():
		#	s += len(client_data)
			clients_data[client_number] = DataLoader(ConcatDataset(client_data), shuffle=True, batch_size=self.batch_size)

		#print("num_batches: ", s)
		#exit()
		return clients_data





