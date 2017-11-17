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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = []
        
        # Conv2d(in_channels, out_channels, kernel_size..)
        self.conv1 = nn.Conv2d(1,10,5,stride=1, padding=0)
        self.layers.append(self.conv1)

        # nn.MaxPool2d(kernel_size)
        self.pool1 = lambda x: F.max_pool2d(x,2)
        self.layers.append(self.pool1)

        self.relu1 = nn.ReLU()
        self.layers.append(self.relu1)
        
        self.conv2 = nn.Conv2d(10,20,5,stride=1, padding=0) 
        self.layers.append(self.conv2)
        
        self.conv2_drop = nn.Dropout2d()
        self.layers.append(self.conv2_drop)
        
        self.pool2 = lambda x: F.max_pool2d(x,2)
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
            try: tensor = a_layer(tensor)
            except Exception as e: logging.error(a_layer); loggging.error(str(e))
        return F.log_softmax(tensor)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if GPU_ENABLE: data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # Negative log likelihood
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_STEPS == 0:
           logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, 
		    batch_idx * len(data), 
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), 
		    loss.data[0]
		)
	    )

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if GPU_ENABLE: data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, 
            correct, 
            len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
        )
    )


# Run ------------------------------------------------------------------------------------

torch.manual_seed(SEED)
if GPU_ENABLE: torch.cuda.manual_seed(SEED)
cuda_args = {
                'num_workers': 1, 
                'pin_memory': True
            } if GPU_ENABLE else {}

# Download data...........................................................
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('datasets/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_TRAIN, shuffle=True, **cuda_args)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('datasets/mnist', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_TEST, shuffle=True, **cuda_args)
#.........................................................................


model = Net()
if GPU_ENABLE: model.cuda()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

for epoch in range(1, EPOCHS + 1):
    train(epoch)
    test()
