import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from resnet import ResNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

BATCH_SIZE = 64
MEAN = 0.5
STD = 0.5
NUM_EPOCHS = 10
NUM_BLOCKS = 3

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((MEAN, MEAN, MEAN), (STD, STD, STD))])

# loading data
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

print(f'There are {len(training_data)} training samples. ')
print(f'There are {len(test_data)} test samples. ')
print(f'Batch_Size is: {BATCH_SIZE}, therefore there are {len(train_dataloader)} elements in the train_dataloader.')
print(f'Batch_Size is: {BATCH_SIZE}, therefore there are {len(test_dataloader)} elements in the test_dataloader.')


'''
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    img = img.swapaxes(0,1)
    img = img.swapaxes(1,2)
    plt.imshow((img.squeeze() * STD) + MEAN)
plt.show()
'''

net = ResNet(NUM_BLOCKS).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# training
for epoch in range(NUM_EPOCHS):      # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 20 mini-batches
            print(f'[{epoch+1:d}, {i+1:5d}] loss: {running_loss / 20:.3f}')
            running_loss = 0.0

print('Finished Training')

# saving the net: 
PATH = './resnet_net.pt'
torch.save(net.state_dict(), PATH)
# net_state_dict = torch.load(PATH)
# model = ResNet()
# model.load_state_dict(net_state_dict)
# model.to(device)

# testing
correct = 0
total = 0
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {len(test_data)} test images: {100 * correct / total:.2f} %')

