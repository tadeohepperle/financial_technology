# DNN using torch
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F



def create_model(input_layer_size = 30, hidden_layers = 2, hidden_layer_size = 10):
    layers = [
		("hidden_layer_1", nn.Linear(input_layer_size, hidden_layer_size)),
		("activation_1", nn.ReLU())		
	]
   
    for i in range(2, hidden_layers+1):
        layers.append((f"hidden_layer_{i}",nn.Linear(hidden_layer_size, hidden_layer_size) ))
        layers.append((f"activation_{i}",  nn.ReLU()))

    layers.append(	("output_layer", nn.Linear(hidden_layer_size, 1)) )
    layers.append(  ("softmax", nn.Sigmoid()))
    return nn.Sequential(OrderedDict(layers))
