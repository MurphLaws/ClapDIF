
import os 
import torch
import torchvision
import sys
sys.path.insert(0, '/users/nicolass80/ClapDIF')
from Helpers.nicolas import DatasetLoaders
from Helpers import model_train
from poison_frog.dataset import create_perturbed_dataset


datasets_path = "poison_frog/datasets"
def populate():

    if not os.path.exists(datasets_path + "/dog"):
        os.makedirs(datasets_path + "/dog")
    if not os.path.exists(datasets_path + "/fish"):
        os.makedirs(datasets_path + "/fish")

    if not os.path.exists(datasets_path + "/dog/train"):
        os.makedirs(datasets_path + "/dog/train")
    if not os.path.exists(datasets_path + "/dog/test"):
        os.makedirs(datasets_path + "/dog/test")


    if not os.path.exists(datasets_path + "/fish/train"):
        os.makedirs(datasets_path + "/fish/train")
    if not os.path.exists(datasets_path + "/fish/test"):
        os.makedirs(datasets_path + "/fish/test")



dataset_loader = DatasetLoaders()
train_data, test_data = dataset_loader.load_cifar(subset_ratio=0.01, seed=42)

#Get all the iamged fron train and test data that belongs to classes dog and bird

def save_images():
    dog_train = []
    dog_test = []

    fish_train = []
    fish_test = []

    for i in range(len(train_data)):
        if train_data[i][1] == 5:
            dog_train.append(train_data[i])
        elif train_data[i][1] == 2:
            fish_train.append(train_data[i])

    for i in range(len(test_data)):
        if test_data[i][1] == 5:
            dog_test.append(test_data[i])
        elif test_data[i][1] == 2:
            fish_test.append(test_data[i])


    #Save the images in the respective folders

    for i in range(len(dog_train)):
        torchvision.utils.save_image(dog_train[i][0], datasets_path + "/dog/train/" + str(i) + ".png")

    for i in range(len(dog_test)):
        torchvision.utils.save_image(dog_test[i][0], datasets_path + "/dog/test/" + str(i) + ".png")

    for i in range(len(fish_train)):
        torchvision.utils.save_image(fish_train[i][0], datasets_path + "/fish/train/" + str(i) + ".png")

    for i in range(len(fish_test)):
        torchvision.utils.save_image(fish_test[i][0], datasets_path + "/fish/test/" + str(i) + ".png")

def clean_folders():

    #Remove all the images from the folders

    for filename in os.listdir(datasets_path + "/dog/train"):
        os.remove(datasets_path + "/dog/train/" + filename)
    for filename in os.listdir(datasets_path + "/dog/test"):
        os.remove(datasets_path + "/dog/test/" + filename)
    for filename in os.listdir(datasets_path + "/fish/train"):
        os.remove(datasets_path + "/fish/train/" + filename)
    for filename in os.listdir(datasets_path + "/fish/test"):
        os.remove(datasets_path + "/fish/test/" + filename)

#save_images()
#clean_folders()


#Import create_perturbed_dataset from poison_frog/dataset.py

