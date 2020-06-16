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
# [ 모듈 가져 오기 ]

# < 환경 설정값 >
shot = 1                                                 # 라즈베리파이 촬영 번호
thres = 180                                                 # 이미지 전처리시 임계치값
today = datetime.today().strftime('%Y%m%d_%H%M%S')          # 파일이름을 위해 오늘날짜 데이터 저장
base_dir = 'X:\\'.replace('\\', '/')                        # 베이스 디렉토리(라즈베리파이 공유폴더 디렉토리)
model_path = "./Models/TestModel/crossValidation.pth"       # 서비스모델 파일경로 
path_rb = '/static/logdata/raw_data/*.png'                  # 라즈베리파이 촬영이미지 저장소
ip_address = '127.0.0.1'                                    # 데이터베이스 아이피주소

#######################################################################################################################################
# [ 촬영 이미지 획득 ]

# <라즈베리파이 공유폴더에서 데이터 로드>
full_path = base_dir + path_rb
file_path = glob.glob(full_path)[shot - 1]
img = Image.open(file_path)

#######################################################################################################################################
# [ 함수 ]

# < 동시객체인식 함수 >
def multi_objectDetection(img):
    
    # 이미지 그레이 스케일 변환
    img_color = np.array(img)
    img = np.array(img.convert('L'))

    # 촬영이미지로부터 관심영역만 추출
    img_color_ROI = img_color[250:1550,200:1500]
    img_ROI = img[250:1550,200:1500]
    
    # 이미지 이진화
    _, src_bin = cv2.threshold(img_ROI, thres, 255, cv2.THRESH_BINARY)
    src_bin = cv2.bitwise_not(src_bin)

    # 촬영이미지로부터 객체 인식 및 추출
    img_list = list()
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin)
    alpha = 2
    SIZE = 128
    for img_num in range(1,nlabels):
        stat = stats[img_num]
        x = stat[0] - alpha
        y = stat[1] - alpha
        width = stat[2] + 2*alpha
        height = stat[3] + 2*alpha
        n_pixel = stat[4]
        if n_pixel < 3000 : continue
        delta_x = int((SIZE - width) / 2)
        delta_y = int((SIZE - height) / 2)
        tmp1 = img_color_ROI[y : y+height, x : x+width, :].copy()
        tmp2 = np.zeros((SIZE,SIZE,3), dtype = int)
        try:
            for channel in range(3):
                tmp2[delta_y : delta_y + height, delta_x : delta_x + width , channel] = tmp1[0:height, 0:width, channel]
            tmp2 = np.uint8(tmp2)
            img_list.append((img_num, stat, tmp2))
        except:
            print(img_num, '객체인식 실패')
    return img_list, img_color_ROI

# < 이미저 전처리 함수 >
def imgPreprocessing(src, thres, SIZE):

    # 이미지 그레이 스케일 변환
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # 이미지 이진화
    ret, binary = cv2.threshold(gray,thres,255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)

    # 이미지에서 객체 추출
    tmp = np.zeros_like(src)
    for y in range(SIZE):
        for x in range(SIZE):
            if (binary != 0)[y,x]:
                for i in range(3):
                    tmp[y,x,i] = src[y,x,i]
    
    # 이미지 중심과 객체의 무게중심 구하기
    src_processed = tmp   
    height = src_processed.shape[0]
    width = src_processed.shape[1]
    R = list()
    for y in range(height):
        for x in range(width):
            if binary[y,x]:            
                R.append([y, x])
    M = len(R)
    R = np.array(R)
    R_x = R[:,1]
    R_y = R[:,0]
    R_x_sum = R_x.sum()
    R_y_sum = R_y.sum()
    center = np.round(R_x_sum/M) , (np.round(R_y_sum/M))
    height_center = center[0]
    width_center = center[1]

    # 객체 바깥쪽 노이즈 데이터 제거
    _, contours, hierachy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)    
    contours_leng = [len(i) for i in contours]
    contours_leng.sort()
    contours_leng = contours_leng[:-2]
    len_max = np.array(contours_leng).max()
    contour_vector = list()
    for i in contours:
        if len(i) == len_max:
            for point in i:
                contour_vector.append((point[0][1]-height_center, point[0][0]-width_center))
            vector_size = np.sqrt(np.array(contour_vector)[:,0]**2 + np.array(contour_vector)[:,1]**2 )
            max_size = vector_size.max()
    R_size = np.sqrt((R[:,1] - width_center) ** 2 + (R[:,0] - height_center) ** 2)
    R_total = np.hstack([R,R_size.reshape((R_size.shape[0],1))])
    R_filtered = R_total[R_total[:,2] > max_size]
    for i in R_filtered[:,:2]:
        src_processed[int(i[1]),int(i[0])] = 0

    # 이미지 중심과 객체 무게중심 일치시키기    
    src_center = np.array([src_processed.shape[1] / 2, src_processed.shape[0] / 2])
    object_center = np.array([height_center, width_center])
    delta = object_center - src_center
    height, width = src_processed.shape[:2]
    M = np.float32([[1, 0, -delta[0]], [0, 1, -delta[1]]])
    img_translation = cv2.warpAffine(src_processed, M, (width,height))
    
    # 이미지 축소 ( 128 x 128 -> 64 x 64)
    height, width, channel = img_translation.shape
    img_constraction = cv2.pyrDown(img_translation)

    # 최종 반환 값
    return src, binary, src_processed, img_translation, contours_leng, img_constraction

# < 예측값 이미지에 레이블링 >
def labeledImage(img, prediction, img_list):

    # 이미지 컬러타입 변환
    img_color = np.array(img.convert("RGB"))

    # 이미지 그레이 스케일 변환
    img = np.array(img.convert('L'))

    # 촬영이미지로부터 관심영역만 추출
    img_color_ROI = img_color[250:1550,200:1500]
    img_ROI = img[250:1550,200:1500]

    # 이미지 이진화
    _, src_bin = cv2.threshold(img_ROI, thres, 255, cv2.THRESH_BINARY)
    src_bin = cv2.bitwise_not(src_bin)

    # 촬영된 이미지에 레이블링 하기
    alpha = 2
    SIZE = 128
    for idx in range(len(img_processed_list)):
        stat = img_list[idx][1]
        x = stat[0] - alpha
        y = stat[1] - alpha
        width = stat[2] + 2*alpha
        height = stat[3] + 2*alpha
        n_pixel = stat[4]
        if n_pixel < 3000: continue
        cv2.rectangle(img_color_ROI, (x, y), (x+width, y+height), (0,100,255), thickness= 3)
        text = str(prediction[idx,1]) 
        cv2.putText(img_color_ROI, text=text, org=(x+int(width/2), y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=3)
    return img_color_ROI

#######################################################################################################################################
# [ 획득한 이미지 처리 ]

# < 획득한 이미지로부터 객체 인식 후 각각의 이미지로 분리>
img_list = multi_objectDetection(img)[0]

# < 분리된 이미지를 각각 전처리 >
img_processed_list = list()
for num in range(len(img_list)):
    try:
        img_processed = imgPreprocessing(img_list[num][2], thres, 128)[3]
        img_constraction = imgPreprocessing(img_list[num][2], thres, 128)[5]
        img_processed_list.append([img_processed, img_constraction])
        # Image.fromarray(img_constraction).save('./Final_data_rb/Black/black_shot' + str(shot) + '_' + str(num) +'.png' )
    except:
        print(num, '이미지 전처리 중 에러')

#######################################################################################################################################
# [ CNN 이미지 분류 모델 로드 ]

# < CNN 신경망 작성 >
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

    # 전결합층을 위해서 Flatten
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   
        out = self.fc(out)
        return out

# < 모델 정의 >
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location = torch.device(device)))
model.eval()

#######################################################################################################################################
# [ 생두 예측 ]

# < 생두 예측 >
prediction = list()
for i in range(len(img_processed_list)):
    image = torch.Tensor(np.array(Image.fromarray(img_processed_list[i][1]).convert("L")))
    image = torch.unsqueeze(image, 0)
    image = torch.unsqueeze(image, 0)
    image = image.to(device)
    testset = TensorDataset(image)
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size=1,
                                            shuffle=False,
                                            drop_last=True)
    data = testloader.dataset[0][0]
    data = torch.unsqueeze(data, 0)
    output = model(data)
    pred = output.data.max(1, keepdim = True)[1]
    if pred[0][0] == 0:
        prediction.append((pred[0][0], 'Nr'))
    elif pred[0][0] == 1:
        prediction.append((pred[0][0], 'Br'))
    else:
        prediction.append((pred[0][0], 'Bl'))
prediction = np.array(prediction)

# < 획득한 라즈베리파이 이미지에 레이블링 >
img_labeled = Image.fromarray(labeledImage(img, prediction, img_list))

#######################################################################################################################################
# [ 최종 산출물 ]

# < 1. img (라즈베리로 촬영한 이미지) >
path_img = '/static/result/img' + str(shot) + '.png'
img.save(base_dir + path_img)

# < 2. prediction_for_web (prediction에서 생두종류별 예측된 수) >
path_pred = '/static/result/prediction_for_web' + str(shot) + '.pickle'
with open(base_dir + path_pred, 'wb') as f:
    prediction_for_web = json.dumps({
        "Normal_cnt" : np.count_nonzero(prediction[:,1] == 'Nr'),
        "Broken_cnt" : np.count_nonzero(prediction[:,1] == 'Br'),
        "Black_cnt" : np.count_nonzero(prediction[:,1] == 'Bl')
    })
    pickle.dump(prediction_for_web, f)
    
# < 3.img 파일에 레이블링된 이미지 >
path_img_labeled ='/static/result/img_labeled' + str(shot) + '.png'
img_labeled.save(base_dir + path_img_labeled)


#######################################################################################################################################
# [ 데이터베이스에 로그파일 저장 ]

# < 데이터베이스에 데이터 저장 >
conn = pymongo.MongoClient(ip_address, 27017)
categories = ['Nr', 'Br', 'Bl']

for idx, pred in enumerate(prediction):
    if pred[1] == categories[0]:
        db1 = conn.get_database('Normal_pred')
        path1 = base_dir + '/static/logdata/Normal_pred/'
        fn1 = str(today) + '_' + categories[0] + '_' + str(idx) + '.png'
        Image.fromarray(img_processed_list[idx][1]).save(path1 + fn1)
        fp1 = open(path1 + fn1, 'rb')
        data1 = fp1.read()
        image1 = Image.open(io.BytesIO(data1))        
        fs1 = gridfs.GridFS(db1, collection = 'Normal_pred')
        fs1.put(data1, filename = categories[0] )
    elif pred[1] == categories[1]:
        db2 = conn.get_database('Broken_pred')
        path2 = base_dir + '/static/logdata/Broken_pred/'
        fn2 = str(today) +  '_' + categories[1] + '_' + str(idx) + '.png'
        Image.fromarray(img_processed_list[idx][1]).save(path2 + fn2)
        fp2 = open(path2 + fn2, 'rb')
        data2 = fp2.read()
        image2 = Image.open(io.BytesIO(data2))        
        fs2 = gridfs.GridFS(db2, collection = 'Broken_pred')
        fs2.put(data2, filename = categories[1] )
    elif pred[1] == categories[2]:
        db3 = conn.get_database('Black_pred')
        path3 = base_dir + '/static/logdata/Black_pred/'
        fn3 = str(today) + '_' + categories[2] + '_' + str(idx) + '.png'
        Image.fromarray(img_processed_list[idx][1]).save(path3 + fn3)
        fp3 = open(path3 + fn3, 'rb')
        data3 = fp3.read()
        image3 = Image.open(io.BytesIO(data3))        
        fs3 = gridfs.GridFS(db3, collection = 'Black_pred')
        fs3.put(data3, filename = categories[2] )




# ====================================================================================================================================
print('객체인식된 생두 개수: ', len(img_list))
print('전처리완료된 생두 개수: ', len(img_processed_list))
print('전처리 중에 오류난 생두 개수: ',  len(img_list) - len(img_processed_list))
