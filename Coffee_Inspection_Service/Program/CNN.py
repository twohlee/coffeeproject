#######################################################################################################################################
# [ 모듈 가져 오기 ]

# < 학습모델 >
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable 
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer 
from sklearn.model_selection import KFold 
from sklearn.metrics import classification_report 

# < 기본모듈 >
import os 
import io
import pickle
import json
import glob
from datetime import datetime
from time import sleep

# < 이미지 처리 모듈 >
import cv2
import matplotlib.pyplot as plt
from PIL import Image 

# < 데이터 처리 모듈 >
import numpy as np
import pandas as pd
from sklearn import datasets, model_selection
from sklearn.model_selection import train_test_split

# < 데이터베이스 >
import pymongo
import gridfs 

#######################################################################################################################################
# [ 학습데이터 준비 ]

# < 데이터베이스 연결, 데이터 로드 및 준비 >
conn = pymongo.MongoClient('127.0.0.1', 27017)
categories = ['Normal', 'Broken', 'Black']
img_list = list()
label_list = list()
for idx, category in enumerate(categories):
    db = conn.get_database(category)
    fs = gridfs.GridFS(db, collection = category)
    tmp = list()
    for f in fs.find():
        image = np.array(Image.open(io.BytesIO(f.read())).convert("L"))
        tmp.append(image)
    label = [ idx for label in range(len(list(tmp)))]
    label_list = label_list + label        
    img_list = img_list + tmp


# < 데이터베이스 연결, 로그데이터 로드 및 준비 >
# 로그 데이터가 1000개 이상 쌓였을때만 학습 데이터로 사용 된다
categories = ['Normal_pred', 'Broken_pred', 'Black_pred']
img_log_list = list()
label_log_list = list()
for idx, category in enumerate(categories):
    db = conn.get_database(category)
    fs = gridfs.GridFS(db, collection = category)
    tmp = list()
    if len(list(fs.find())) >= 1000:
        for f in fs.find():
            image = np.array(Image.open(io.BytesIO(f.read())).convert("L"))
            tmp.append(image)
        label = [ idx for label in range(len(list(tmp)))]
        label_log_list = label_log_list + label        
        img_log_list = img_log_list + tmp

# < 베이스 데이터 + 로그데이터 >
label_data = np.array(label_log_list + label_list)
img_data = np.array(img_log_list + img_list)

# < 데이터 준비 >
X_data = torch.Tensor(np.array(img_data)).unsqueeze(1)
y_data = torch.from_numpy(label_data).long()


#######################################################################################################################################
# [ CNN 모델 ]

# < CUDA 설정 >
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device : " + device)
print("="*50)

# < 랜덤 시드 고정 >
# GPU 사용 가능일 경우 랜덤 시드 고정
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# < 학습 환경 설정 >
learning_rate = 0.001
training_epochs = 10
batch_size = 300
n_splits = 5

# < CNN 모델 >
class CNN(torch.nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 1, 64, 64)
        #    Conv     -> (?, 32, 64, 64)
        #    Pool     -> (?, 32 ,32, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1 ),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # 두번째층
        # ImgIn shape=(?, 32, 32, 32)
        #    Conv      ->(?, 64, 32, 32)
        #    Pool      ->(?, 64, 16, 16)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # 세번째층
        # ImgIn shape=(?, 64, 16, 16)
        #    Conv      ->(?, 128, 16, 16)
        #    Pool      ->(?, 128, 8, 8)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # 전결합층 8x8x128 inputs -> 3 outputs
        self.fc = torch.nn.Linear(8 * 8 * 128, 3, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out

#######################################################################################################################################
# [ 학습 ]

# < CNN 모델 정의 >
model = CNN().to(device)

# < 비용 함수와 옵티마이저 정의 >
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.train()

# < 교차검증관련 >  
kf = KFold(n_splits=n_splits, random_state=None, shuffle=True)

for epoch in range(training_epochs): 
    for train_index, test_index in kf.split(X_data):
        X_train = X_data[train_index]
        y_train = y_data[train_index]
        X_test = X_data[test_index]
        y_test = y_data[test_index]
        dataset = TensorDataset(X_train, y_train)
        testset = TensorDataset(X_test, y_test)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
        testloader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)  
        total_batch = len(data_loader)
        print('총 배치의 수 : {}'.format(total_batch))
        avg_cost = 0.0
        for X, Y in data_loader:
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            hypothesis = model(X)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()
            avg_cost += cost / total_batch
        
        # < 테스트 >
        model.eval()
        test_loss = 0 
        correct = 0
        for data, target in testloader: 
            data = data.to(device)
            target = target.to(device) 
            output = model(data)
            test_loss += criterion(output, target).data 
            pred = output.data.max(1, keepdim = True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            test_loss /= len(testloader.dataset)/batch_size
        print(classification_report(target.data.view_as(pred).cpu(), pred.cpu()))
        print('\nTest set : Average loss : {: .4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(testloader.dataset), 100. * correct / len(testloader.dataset)))
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

#######################################################################################################################################
# [ 모델 저장 ]
model_path = "./Models/TestModel/crossValidation.pth"
torch.save(model.state_dict(), model_path)







