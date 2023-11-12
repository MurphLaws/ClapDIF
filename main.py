
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

    if not os.path.exists(datasets_path + "/train"):
        os.makedirs(datasets_path + "/train")
    if not os.path.exists(datasets_path + "/test"):
        os.makedirs(datasets_path + "/test")

    if not os.path.exists(datasets_path + "/train/dog"):
        os.makedirs(datasets_path + "/train/dog")
    if not os.path.exists(datasets_path + "/train/fish"):
        os.makedirs(datasets_path + "/train/fish")


    if not os.path.exists(datasets_path + "/test/dog"):
        os.makedirs(datasets_path + "/test/dog")
    if not os.path.exists(datasets_path + "/test/fish"):
        os.makedirs(datasets_path + "/test/fish")


dataset_loader = DatasetLoaders()
train_data, test_data = dataset_loader.load_cifar(subset_ratio=0.02, seed=42)

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
        torchvision.utils.save_image(dog_train[i][0], datasets_path + "/train/dog/" + str(i) + ".png")

    for i in range(len(dog_test)):
        torchvision.utils.save_image(dog_test[i][0], datasets_path + "/test/dog/" + str(i) + ".png")

    for i in range(len(fish_train)):
        torchvision.utils.save_image(fish_train[i][0], datasets_path + "/train/fish/" + str(i) + ".png")

    for i in range(len(fish_test)):
        torchvision.utils.save_image(fish_test[i][0], datasets_path + "/test/fish/" + str(i) + ".png")

def clean_folders():

    #Remove all the images from the folders

    for filename in os.listdir(datasets_path + "/train/dog"):
        os.remove(datasets_path + "/train/dog/" + filename)

    for filename in os.listdir(datasets_path + "/train/fish"):
        os.remove(datasets_path + "/train/fish/" + filename)

    for filename in os.listdir(datasets_path + "/test/dog"):
        os.remove(datasets_path + "/test/dog/" + filename)

    for filename in os.listdir(datasets_path + "/test/fish"):
        os.remove(datasets_path + "/test/fish/" + filename)


#populate()
#clean_folders()
#save_images()


attack_images = "poison_frog/datasets/attack_images"
attack_images = torch.load(attack_images)
torchvision.utils.save_image(attack_images[-1], "output/attack.png")