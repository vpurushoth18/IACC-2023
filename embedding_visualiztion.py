

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import F1Score
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from imblearn import under_sampling, over_sampling
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE, SVMSMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler, KMeansSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from loss import OrthogonalityLoss
from SMOTE import SMOTE
import statistics
from sklearn import preprocessing

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
import copy




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

df = pd.read_csv('data/glass.csv')
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
f1 = F1Score(num_classes=num_classes)

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_layers)
        self.linear2 = nn.Linear(hidden_layers, output_dim)
    
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x


precision_list = []
recall_list = []
f1_list = []

smote_generated_values = 0
smote_generated_labels = 0

model_generated_values = 0
model_generated_labels = 0

original_values = 0
original_labels = 0

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# NN model

for seed in range(1, 2):
    np.random.seed(seed)
    torch.manual_seed(seed)

    traindata = Data(X_train, Y_train)
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=0)

    clf = Network()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(clf.parameters()), lr=0.05)

    epochs = 100
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # set optimizer to zero grad to remove previous epoch gradients
            optimizer.zero_grad()
            # forward propagation
                           
            inputs, labels = torch.tensor(inputs), torch.tensor(labels)
            labels = labels.type(torch.LongTensor)
            outputs = clf(inputs)
            if(epoch == (epochs-1)):  
                original_values, original_labels = outputs, labels
                
            ce_loss = criterion(outputs, labels)
            loss = ce_loss
            # backward propagation
            loss.backward()
            # optimize
            optimizer.step()
            running_loss += loss.item()
            # display statistics
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
  


# SMOTE-NN model

for seed in range(1, 2):
    np.random.seed(seed)
    torch.manual_seed(seed)

    traindata = Data(X_train, Y_train)
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=0)

    clf = Network()
    sm = over_sampling.SMOTE(random_state=seed, k_neighbors=2)

    criterion = nn.CrossEntropyLoss()
    orthogonal_loss = OrthogonalityLoss()
    optimizer = torch.optim.Adam(list(clf.parameters()), lr=0.05)

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
            labels = labels.type(torch.LongTensor)
            outputs = clf(inputs)
            if(epoch == (epochs-1)):  
                smote_generated_values, smote_generated_labels = outputs, labels
            ce_loss = criterion(outputs, labels)
            loss = ce_loss
            # backward propagation
            loss.backward()
            # optimize
            optimizer.step()
            running_loss += loss.item()
            # display statistics
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
        
        
# OC-SMOTE-NN model

for seed in range(1, 2):
    np.random.seed(seed)
    torch.manual_seed(seed)

    traindata = Data(X_train, Y_train)
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=0)

    clf = Network()
    smote = SMOTE(dims = input_dim, k=2, seed_num = seed)

    criterion = nn.CrossEntropyLoss()
    orthogonal_loss = OrthogonalityLoss()
    optimizer = torch.optim.Adam(list(clf.parameters()) + list(smote.parameters()), lr=0.05)

    epochs = 100
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # set optimizer to zero grad to remove previous epoch gradients
            optimizer.zero_grad()
            # forward propagation
        
            inputs, labels = smote(inputs, labels)
            labels = labels.type(torch.LongTensor)
            outputs = clf(inputs)
            if(epoch == (epochs-1)):  
                model_generated_values, model_generated_labels = outputs, labels
                
            ce_loss = criterion(outputs, labels)
            o_loss = orthogonal_loss(outputs, labels)
            loss = ce_loss +  0.4 * o_loss
            # backward propagation
            loss.backward()
            # optimize
            optimizer.step()
            running_loss += loss.item()
            # display statistics
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
        
    
tsne = TSNE(n_components=2, verbose=1, random_state=123)

z = tsne.fit_transform(original_values.detach().numpy()) 
df = pd.DataFrame()
df["y"] = original_labels.detach().numpy()
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]
plt.figure(figsize=(16,4))
ax1 = plt.subplot(1, 3, 1)
sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
palette=sns.color_palette("hls", 6),
data=df, ax=ax1).set(title="Embedding Generated by NN")

z = tsne.fit_transform(smote_generated_values.detach().numpy()) 
df = pd.DataFrame()
df["y"] = smote_generated_labels.detach().numpy()
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]
ax2 = plt.subplot(1, 3, 2)
sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
palette=sns.color_palette("hls", 6),
data=df, ax=ax2).set(title="Embedding Generated by SMOTE-NN")

z = tsne.fit_transform(model_generated_values.detach().numpy()) 
df = pd.DataFrame()
df["y"] = model_generated_labels.detach().numpy()
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]
ax3 = plt.subplot(1, 3, 3)
sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
palette=sns.color_palette("hls", 6),
data=df, ax=ax3).set(title="Embedding Generated by Our Model")


