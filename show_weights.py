import torch
from matplotlib import pyplot as plt
from resnet import ResNet

NUM_BLOCKS = 3
MEAN = 0.5
STD = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# load gpu-created-data: add map_location=device
PATH = './resnet_net.pt'
net_state_dict = torch.load(PATH, map_location=device)

model = ResNet(NUM_BLOCKS)
model.load_state_dict(net_state_dict)

#print(model)
# print(len(model.conv1.weight))

# w = model.conv1.weight[0]
# w = w.cpu().detach().numpy()
# print(w)
# plt.imshow(w*0.5 + 0.5)
# plt.show()


def show_layer(W):
    fig = plt.figure(figsize=(4, 4))

    for i in range(len(W)):
        ax = plt.subplot(4, 4, i + 1)
        img = W[i].cpu().detach().numpy()*STD + MEAN
        plt.imshow(img)
        plt.axis('off')
    plt.show()
    
show_layer(model.conv1.weight)