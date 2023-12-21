import os 
import torch
import torchvision
import sys
from Helpers.nicolas import DatasetLoaders, Resnet20Model
from Helpers import model_train
from poison_frog.dataset import create_perturbed_dataset



model = Resnet20Model(num_classes=10)

#Freeze all the layers except the last one

for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False




train_data = torch.load("poison_frog/datasets/attack_images")
train_labels = torch.load("poison_frog/datasets/attack_labels")

test_data = torch.load("poison_frog/datasets/sanitized_images_test")
test_labels = torch.load("poison_frog/datasets/sanitized_labels_test")

  #Use dataloaders for the training and test set

train_data = torch.utils.data.TensorDataset(train_data, train_labels)
test_data = torch.utils.data.TensorDataset(test_data, test_labels)



if False:
    def train(
    model,
    epochs,
    learning_rate,
    reg_strength,
    save_dir,
    train_loader,
    test_loader,
    device,
    save_ckpts=True,
    ckpt_number=None,
    ): pass 


#print the shape of my tensor dataset
    
print(train_data[0][0].shape)
#model_train.train(model, 10, 0.001, 0.0001, "poison_frog/model_saves/poisoned_model", train_data, test_data, "cpu", save_ckpts=True, ckpt_number=None)

