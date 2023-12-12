
#Read the a list writen on a python file

import os
import sys
import numpy as np
import torch




def flater(path, idx):
    input = torch.load(path)
    evolution = []
    output = []
    for i in input:
        output.append(torch.cat(i))
    
    for i in output:
        evolution.append(i[idx].tolist())

    return evolution


foo  = flater("outputs1.pt",46)

print(foo)