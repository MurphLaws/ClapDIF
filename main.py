
import os 
import torch
import torchvision
import sys
from Helpers.nicolas import DatasetLoaders, Resnet20Model
from Helpers import model_train
from poison_frog.dataset import create_perturbed_dataset

datasets_path = "poison_frog/datasets"
def populate():

    if not os.path.exists(datasets_path + "/train"):
        os.makedirs(datasets_path + "/train")
    if not os.path.exists(datasets_path + "/test"):
        os.makedirs(datasets_path + "/test")

    if not os.path.exists(datasets_path + "/train/class1"):
        os.makedirs(datasets_path + "/train/class1")
    if not os.path.exists(datasets_path + "/train/class0"):
        os.makedirs(datasets_path + "/train/class0")


    if not os.path.exists(datasets_path + "/test/class1"):
        os.makedirs(datasets_path + "/test/class1")
    if not os.path.exists(datasets_path + "/test/class0"):
        os.makedirs(datasets_path + "/test/class0")


dataset_loader = DatasetLoaders()
train_data, test_data = dataset_loader.load_cifar(subset_ratio=0.1, seed=42)

#Get all the iamged fron train and test data that belongs to classes class1 and bird

def save_images(class0=2,class1=5):
    class1_train = []
    class1_test = []

    class0_train = []
    class0_test = []

    for i in range(len(train_data)):
        if train_data[i][1] == class1: #Dog
            class1_train.append(train_data[i])
        elif train_data[i][1] == class0: #Bird
            class0_train.append(train_data[i])

    for i in range(len(test_data)):
        if test_data[i][1] == class1:  # Dog
            class1_test.append(test_data[i])
        elif test_data[i][1] == class0: # Bird
            class0_test.append(test_data[i])


    #Save the images in the respective folders

    for i in range(len(class1_train)):
        torchvision.utils.save_image(class1_train[i][0], datasets_path + "/train/class1/" + str(i) + ".png")

    for i in range(len(class1_test)):
        torchvision.utils.save_image(class1_test[i][0], datasets_path + "/test/class1/" + str(i) + ".png")

    for i in range(len(class0_train)):
        torchvision.utils.save_image(class0_train[i][0], datasets_path + "/train/class0/" + str(i) + ".png")

    for i in range(len(class0_test)):
        torchvision.utils.save_image(class0_test[i][0], datasets_path + "/test/class0/" + str(i) + ".png")

def clean_folders():

    #Remove all the images from the folders

    for filename in os.listdir(datasets_path + "/train/class1"):
        os.remove(datasets_path + "/train/class1/" + filename)

    for filename in os.listdir(datasets_path + "/train/class0"):
        os.remove(datasets_path + "/train/class0/" + filename)

    for filename in os.listdir(datasets_path + "/test/class1"):
        os.remove(datasets_path + "/test/class1/" + filename)

    for filename in os.listdir(datasets_path + "/test/class0"):
        os.remove(datasets_path + "/test/class0/" + filename)


populate()
#clean_folders()
#save_images(class0=7,class1=0)


attack_images = "poison_frog/datasets/attack_images"
attack_images = torch.load(attack_images)
torchvision.utils.save_image(attack_images[-1], "output/attack.png")





model = Resnet20Model(num_classes=10)