import torch
from captum.attr import visualization as viz
from torchvision import models
from captum.influence import TracInCPFast, TracInCP
from Helpers.nicolas import DatasetLoaders, Resnet20Model

# Create a model (example: ResNet50)
model = Resnet20Model(num_classes=10)
model.eval()

# Create a TracInCP attribute method

#Define a random train dataset and dataloader

train_data = torch.rand(100, 3, 224, 224)
train_labels = torch.randint(0, 10, (100,))


train_data = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

ckpt_path = 'poison_frog/model_saves/poisoned_model'
checkpoint = torch.load(ckpt_path)
pretrained = checkpoint["net"]

#load the model on the path using torch

tracincp = TracInCP(model, train_loader, [pretrained])

# Load an input sample (you may need to preprocess your data accordingly)
test_sample = torch.rand(1, 3, 224, 224)
test_label = torch.randint(0, 10, (1,))

#Make a dataloader for the single test sample

test_sample = torch.utils.data.TensorDataset(test_sample, test_label)

#Copmute the influece of input_tensor on the model

#tracin_cp.influence((test_examples_features, test_examples_true_labels), show_progress=True)

tracincp.influence((test_sample, test_label), show_progress=True)

#load the checkpoint



#attributions = tracincp.influence(test_sample)
