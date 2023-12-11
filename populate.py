

#Import CIFAR-10 dataset using Pytoch without any transformation

import torch
import torchvision
import torch.utils.data.dataset

#Download only 10% of the whole dataset, only two classes

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

#Create a dataloader for the training and test set

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)
