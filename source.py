# Native Modules
import configparser
import datetime
import logging
import os

# 3rd Party
from torchvision    import datasets, transforms
from torch.autograd import Variable

import torch.nn.functional as F
import torch.optim         as optim
import torch.nn            as nn
import torch


# Load Parameters.....................................
config = configparser.ConfigParser()
config.read('parameters.ini')
# Hypermarapeters
SEED        = int(config["default"]["rand_seed"])
BATCH_TRAIN = int(config["default"]["batch_size"])
BATCH_TEST  = int(config["default"]["batch_size_test"])
EPOCHS      = int(config["default"]["epochs"])
LR          = float(config["default"]["lr"])
MOMENTUM    = float(config["default"]["momentum"])
GPU_ENABLE  = config.getboolean("default","gpu")
LOG_STEPS   = int(config["default"]["log_steps"])
#.....................................................

# Set up logs.....................................................................
if not os.path.exists('logs'):
        os.makedirs('logs')
logging.basicConfig(
        format ='%(levelname)s %(asctime)s %(message)s - ',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        filename='logs/{}.log'.format(
            datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")),
        level=logging.INFO)
logging.info("Started")
#.................................................................................

torch.manual_seed(SEED)
if GPU_ENABLE:
    torch.cuda.manual_seed(SEED)

cuda_args = {
                'num_workers': 1, 
                'pin_memory': True
            } if GPU_ENABLE else {}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = []
        
        # Conv2d(in_channels, out_channels, kernel_size..)
        self.conv1 = nn.Conv2d(1,10,5,stride=1, padding=0)
        self.layers.append(self.conv1)

        # nn.MaxPool2d(kernel_size)
        self.pool1 = nn.MaxPool2d(5)
        self.layers.append(self.pool1)

        self.relu1 = nn.ReLU()
        self.layers.append(self.relu1)
        
        self.conv2 = nn.Conv2d(10,20,7,stride=1, padding=0) 
        self.layers.append(self.conv2)
        
        self.conv2_drop = nn.Dropout2d()
        self.layers.append(self.conv2_drop)
        
        self.pool2 = nn.MaxPool2d(5)
        self.layers.append(self.pool2)

        self.relu2 = nn.ReLU()
        self.layers.append(self.relu2)

        self.reshape = lambda x: x.view(-1, 320)
        self.layers.append(self.reshape)
        
        # nn.Linear(in_size,out_size)
        self.full_conn1 = nn.Linear(320,50)
        self.layers.append(self.full_conn1)

        self.relu3 = nn.ReLU()
        self.layers.append(self.relu3)

        self.fc_dropout = nn.Dropout2d()
        self.layers.append(self.fc_dropout)
        
        self.full_conn2 = nn.Linear(50,10)
        self.layers.append(self.full_conn2)

    def forward(self, x):
        tensor = x
        for a_layer in self.layers:
            print(a_layer)

model = Net()
model.forward(0)


