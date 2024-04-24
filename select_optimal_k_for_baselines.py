

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import F1Score
from sklearn.metrics import balanced_accuracy_score, f1_score
from imblearn import under_sampling, over_sampling
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE, SVMSMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler, KMeansSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from loss import OrthogonalityLoss
from SMOTE import SMOTE
import statistics
from sklearn import preprocessing
import random

class Data(Dataset):
    def __init__(self, X_train, y_train):
        self.X = torch.from_numpy(X_train.astype(np.float32).to_numpy())
        self.y = torch.from_numpy(y_train.to_numpy()).type(torch.LongTensor)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return self.len
    

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_layers)
        self.linear2 = nn.Linear(hidden_layers, output_dim)
    
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x

df = pd.read_csv('data/wisconsin.csv')
X = df.iloc[:, 0:-1]
Y = df.iloc[:, -1]

X=(X-X.min())/(X.max()-X.min())

batch_size = 500
num_classes = len(Y.value_counts())

# number of features (len of X cols)
input_dim = X.shape[1]
# number of hidden layers
hidden_layers = 64
# number of classes (unique of y)
output_dim = num_classes

neighbors = 1

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_layers)
        self.linear2 = nn.Linear(hidden_layers, output_dim)
    
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x


f1_list = []

seed = random.randint(0, 10)

for neighbors in range(6, 0, -1):
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed, stratify = Y)

    #SMOTE
    sm = over_sampling.ADASYN(random_state=seed, n_neighbors = neighbors)

    traindata = Data(X_train, Y_train)

    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=0)

    clf = Network()

    criterion = nn.CrossEntropyLoss()
    orthogonal_loss = OrthogonalityLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=0.05)

    epochs = 100
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # set optimizer to zero grad to remove previous epoch gradients
            optimizer.zero_grad()
            # forward propagation
            inputs, labels = sm.fit_resample(inputs, labels)
            inputs, labels = torch.tensor(inputs), torch.tensor(labels)
            outputs = clf(inputs)
            ce_loss = criterion(outputs, labels)
            loss = ce_loss
            # backward propagation
            loss.backward()
            # optimize
            optimizer.step()
            running_loss += loss.item()
            # display statistics
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
  
    # save the trained model
    PATH = './mymodel.pth'
    torch.save(clf.state_dict(), PATH)


    testdata = Data(X_test, Y_test)
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=0)


    dataiter = iter(testloader)
    inputs, labels = dataiter.next()

    outputs = clf(inputs)
    __, predicted = torch.max(outputs, 1)

    pred = []
    lbl = []
    batches = 0
    # no need to calculate gradients during inference
    with torch.no_grad():
        for data in testloader:
            batches = batches + 1
            inputs, labels = data
            # calculate output by running through the network
            outputs = clf(inputs)
            # get the predictions
            __, predicted = torch.max(outputs.data, 1)
            # update results
            pred.append(predicted)
            lbl.append(labels)

    pred = torch.cat(pred, dim=0) 
    lbl = torch.cat(lbl, dim=0) 
    f1 = 100 * f1_score(lbl, pred, average = 'macro')
    print('F1-Score of the network for seed ', seed, ' on the test data: ', f1)
    f1_list.append(f1)
    
print('F1-Score List: ', f1_list)