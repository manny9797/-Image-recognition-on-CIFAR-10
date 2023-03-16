from multiprocessing import freeze_support
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn as nn
from PROJECT.datasetCIFAR10 import datasetCFAR10
from PROJECT.neural_network import Net
import torch.optim as optim


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        diction = pickle.load(fo, encoding='bytes')
    return diction


train = []
labels = []
for i in range(1, 5):
    dic = unpickle(f"data_batch_{i}")
    train.extend(list(dic[b'data']))
    labels.extend(list(dic[b'labels']))

dic_test = unpickle("test_batch")
X_test = list(dic_test[b'data'])
labels_test = list(dic_test[b'labels'])
df = pd.DataFrame()
df_test = pd.DataFrame()
df['images'] = train
df['labels'] = labels
df_test['images'] = X_test
df_test['labels'] = labels_test

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar10_dataset = datasetCFAR10(df, transform)
cifar10_data_test = datasetCFAR10(df_test, transform)
trainloader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=3, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(cifar10_data_test, batch_size=2, shuffle=True, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    freeze_support()

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # network

    net = Net()

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs
            inputs, labels = data

            # optimizer
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # Test the network

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # model

    net = Net()
    net.load_state_dict(torch.load(PATH))

    # outputs: energies for each class of animals
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    # test model
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test images: {100 * correct // total} %')


    # use gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    del dataiter
