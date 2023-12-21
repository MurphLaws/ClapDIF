from typing import List
import numpy as np
import torch
from captum.influence import TracInCPFast, TracInCP
import os 
import torch
import torchvision
import sys
from Helpers.nicolas import DatasetLoaders, Resnet20Model
from Helpers import model_train
from poison_frog.dataset import create_perturbed_dataset
from torchsummary import summary
from Helpers.tracin_influence import checkpoints_load_func, compute_train_to_test_influence

path = "poison_frog/model_saves/poisoned_model"

#load the model on the path using torch



model_weights = torch.load(path)
model = Resnet20Model(num_classes=10)
model.load_state_dict(model_weights["net"])

# Print the layer names
for name, layer in model.named_children():
    print(f"Layer Name: {name}")


# Use torchsummary to print the model summary
summary(model, (3, 32, 32))

#Use camptum to copmute the influece of the dataset on the model

attack_images = "poison_frog/datasets/attack_images"

dataset = torch.load(attack_images)

#dataset is already loaded



