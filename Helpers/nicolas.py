import torch
from torch import nn
from typing import List
from .model_train import train
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
import numpy as np


class Resnet20Model(nn.Module):
    def __init__(self, num_classes: int, random_state=None):
        super(Resnet20Model, self).__init__()
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
        )
        self.model = model
        self.num_classes = num_classes
        self.random_state = random_state
        self.__trainable_layers = None
        self.freeze_layers()
        self.set_classification_layer()

    def forward(self, x):
        channels = x.shape[1]
        if channels < 3:
            x = torch.cat([x, x, x], dim=1)
        return self.model(x)

    def freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def set_trainable_layers(self, layers: List[str]):
        for name, param in self.model.named_parameters():
            for layer in layers:
                if layer in name:
                    param.requires_grad = True
        if self.__trainable_layers is None:
            self.__trainable_layers = []
        self.__trainable_layers = layers + self.__trainable_layers

    def set_classification_layer(self):
        num_ftrs = self.model.fc.in_features
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        if self.__trainable_layers is None:
            self.__trainable_layers = []
        self.__trainable_layers.append("fc")

    def count_params(self, trainable=True):
        params = self.model.parameters()
        if trainable:
            params = filter(lambda p: p.requires_grad, params)
        total_params = sum(p.numel() for p in params)
        return total_params

    def trainable_layer_names(self):
        layers = []
        for layer, _ in self.named_modules():
            for tl in self.__trainable_layers:
                if layer.endswith(tl):
                    layers.append(layer)
        return list(set(layers))

class DatasetLoaders:
    
    def __subset_selection(self, data, ratio, seed):
        labels = data.targets
        if type(labels) == torch.Tensor:
            labels = labels.numpy()
        else:
            labels = np.array(labels)
        ratio = int(len(data) * ratio)
        (
            sel_ids,
            _,
        ) = train_test_split(
            np.arange(len(labels)),
            train_size=ratio,
            random_state=seed,
            stratify=labels,
        )
        return Subset(data, sel_ids)

    def load_cifar(self, subset_ratio=None, seed=None):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        trainset = CIFAR10(
            root='data', download=True, train=True, transform=transform
        )
        testset = CIFAR10(
            root='data', download=True, train=False, transform=transform
        )
        if subset_ratio:
            trainset = self.__subset_selection(trainset, subset_ratio, seed)
            testset = self.__subset_selection(testset, subset_ratio, seed)
        return trainset, testset

    def load_mnist(self, subset_ratio=None, seed=None):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        trainset = MNIST(
            root='data', download=True, train=True, transform=transform
        )
        testset = MNIST(
            root='data', download=True, train=False, transform=transform
        )
        if subset_ratio:
            trainset = self.__subset_selection(trainset, subset_ratio, seed)
            testset =  self.__subset_selection(testset, subset_ratio, seed)
        return trainset, testset

    def load_fashion_mnist(self, subset_ratio=None, seed=None):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        trainset = FashionMNIST(
            root='data', download=True, train=True, transform=transform
        )
        testset = FashionMNIST(
            root='data', download=True, train=False, transform=transform
        )
        if subset_ratio:
            trainset = self.__subset_selection(trainset, subset_ratio, seed)
            testset = self.__subset_selection(testset, subset_ratio, seed)
        return trainset, testset




if __name__ == '__main__':

    # Define dataset (start with mnist)
    
    dl = DatasetLoaders()

    train_data, test_data = dl.load_cifar(subset_ratio=0.9, seed=42) # This will keep only 10% of the original samples

    trainloader = DataLoader(
        train_data,
        batch_size=128,
        shuffle=True,
        num_workers=15,
    )

    testloader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        num_workers=15,
    )

    # Define model and train

    num_classes = 10

    #Initialize a device for using gpu

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Resnet20Model(num_classes=num_classes, random_state=42)
    
    savedir = '/users/nicolass80/ClapDIF/Helpers/model_saves'
    


    #model, train_loss, test_loss, train_acc, test_acc = train(
    #    model=model,
    #    epochs=50,
    #    learning_rate = 0.11,
    #    reg_strength = 0,
    #    save_dir = savedir,
    #    train_loader=trainloader,
    #    test_loader=testloader,
    #    device=device,
    #    save_ckpts=True,
    #    ckpt_number=None,
    #)

    sample = test_data[10]
    output = model(sample[0].unsqueeze(0))
    

    #Print the sigmoid of the logits in the output 
    print(sample[1])
    print(torch.sigmoid(output))


    # Select some points from the base class (TRAIN samples)
    # and a point(s) from target class (TEST samples)
    # for the random selection use random seed for reproducability
    # ATTENTION: ensure that the selected test images are correctly classified
    # see how you can use the model to make predictions in model_train.py

    # After that, you will call the appropriate function to create the poisons.
    # you will use the model trained so far and then test the success of the attack
    # at the end you will save the poisoned images using torch.save(poisoned_imgs, savedir)
