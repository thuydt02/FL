import torch
import torch.nn.functional as F

from torch import nn


class CNN1(nn.Module):
	'''
	Follow FedAvg CNN Model
	A CNN with two 5x5 convolution layers (the first with32 channels, the second with 64, each followed with 2x2max pooling), 
	a fully connected layer with 512 units andReLu activation, and a final softmax output layer (1,663,370total parameters)'''

	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
		#self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(3136,512)
		self.fc2 = nn.Linear(512, 10)	

	def forward(self, x):
		#print ("x.shape: ", x.shape)
		x = self.maxpool(F.relu(self.conv1(x)))
		x = self.maxpool(F.relu(self.conv2(x)))
		
		x = x.view(-1, 3136)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.softmax(x, dim=1)

class CNN2(nn.Module):

	def __init__(self):
		super(CNN1, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1)
		self.bn1 = nn.BatchNorm2d(10)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1)
		self.bn2 = nn.BatchNorm2d(20)
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = self.bn1(x)
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		x = self.bn2(x)
		x = torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.softmax(x, dim=1)

class MLP2(nn.Module):

	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(784, 256)
		self.fc2 = nn.Linear(256, 10)

	def forward(self, x):
		x = torch.flatten(x, 1)
		x = torch.sigmoid(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

class CNN3(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
		self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(120,84)
		self.fc2 = nn.Linear(84, 10)

	def forward(self, x):
		x = F.relu(self.maxpool(self.conv1(x)))
		x = F.relu(self.maxpool(self.conv2(x)))
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.softmax(self.fc2(x), dim=1)

		return x



