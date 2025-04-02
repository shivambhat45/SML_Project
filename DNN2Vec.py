import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader , TensorDataset
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from io import StringIO
import pickle
import json

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
# open the file
with open("config.json", "r") as file:
    config = json.load(file)
def get_data_loader():
    with open ("./dataset.pkl" , "rb") as file:
        data = pickle.load(file)
    data = list(map(lambda x : (torch.tensor(x[0]) , torch.tensor(x[1])) , data))
    # split the data into train and test
    train_size = int(0.9 * len(data))
    test_size = len(data) - train_size
    train_data , test_data = torch.utils.data.random_split(data , [train_size , test_size])
    train_data_loader = DataLoader(train_data , batch_size = 64 , shuffle = True)
    test_data_loader = DataLoader(test_data , batch_size = 1 , shuffle = True)
    return train_data_loader , test_data_loader
# create a model

class Model(nn.Module):
    def __init__(self , input_size ,hidden_size, output_size):
        super(Model , self).__init__()
        self.fc1 = nn.Linear(input_size , hidden_size)
        
        self.fc2 = nn.Linear(hidden_size , hidden_size)
        self.fc3 = nn.Linear(hidden_size , output_size)
    def forward(self , x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

def train(model , train_data_loader , criterion , optimizer):
    epocs = []
    losses = []
    for epoch in range(10000):
        model.train()
        totalloss = 0
        for i in train_data_loader:
            X = torch.tensor(i[0]).float()
            Y = torch.tensor(i[1]).squeeze(0).view(-1 ,20).float()
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output , Y)
            loss.backward()
            optimizer.step()
            totalloss += (loss.item()/ X.shape[0])
        losses = losses + [totalloss / len(train_data_loader)]
        epocs = epocs + [epoch]

    # plot the loss
    plt.plot(epocs , losses)
    plt.show()
    
def test(test_data_loader , model , criterion , optimizer):
    losses = []
    epocs = []
    model.eval()
    count  = 0
    ssm = 0
    tss = 0
    rms = 0
    epoch = 1
    totalloss = 0
    with torch.no_grad():
        for i in test_data_loader:
            X = torch.tensor(i[0]).float()
            Y = torch.tensor(i[1]).squeeze(0).view(-1 ,20).float()
            output = model(X)
            loss = criterion(output , Y)
            # check the absolute difference is less than 0.01
            if torch.all(torch.abs(output - Y) < 10):
                count += 1
                
            # calculate the r^2 score
            ssm += torch.sum((output - Y)**2)
            tss += torch.sum((Y - torch.mean(Y))**2)
            totalloss += (loss.item()/ X.shape[0])
            # calcualte mean absolute percentage error
            
            losses = losses + [totalloss / len(test_data_loader)]
            epocs = epocs + [count / epoch]
            epoch += 1
            
            
        print(count /(len(test_data_loader) )) 
        print("RMSE: " , torch.sqrt(ssm/len(test_data_loader))  )
        print("R^2: " , 1 - ssm/tss)
        plt.plot(losses, epocs)
        plt.xlabel("Loss")
        plt.ylabel("Accuracy")
        plt.show()
    
    

    
def calculate_metrics(model , test_data_loader):
    
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for i in test_data_loader:
            X = torch.tensor(i[0]).float()
            Y = torch.tensor(i[1]).squeeze(0).view(-1 ,20).float()
            output = model(X)
            y_true.extend(Y.numpy())
            y_pred.extend(output.numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        print("R^2: ", r2_score(y_true, y_pred))
        print("MSE: ", mean_squared_error(y_true, y_pred))
        print("MAE: ", mean_absolute_error(y_true, y_pred))
        print("MAPE: ", mean_absolute_percentage_error(y_true, y_pred))

if(__name__ == "__main__"):
    train_data_loader , test_data_loader = get_data_loader()
    model = Model(config["embedding_size"] , config["hidden_size"] ,2* config["output_size"])
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters() )
    train(model , train_data_loader , criterion , optimizer)
    test(test_data_loader , model , criterion , optimizer)
    calculate_metrics(model , test_data_loader)