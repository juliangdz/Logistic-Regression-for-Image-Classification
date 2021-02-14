"""
Author : Julian Gerald Dcruz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from random import shuffle
from tqdm import tqdm
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import argparse
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression

class DataExplorer(object):
    def __init__(self):
        self.input_dir = "D:/Workspace_Codes/logistic_regression/images/"
        self.img_size=128
        self.classes=["messy","clean"]
        self.fdirs=["train","val","test"]
        self.fdirs_path=[]
        self.train1=None
        self.train2=None
        self.test1=None
        self.test2=None
        self.val1=None
        self.val2=None
    def plt_data(self,img_name):
        img_path=os.path.join(self.input_dir,img_name)
        Image.open(img_path)
    def create_img_array(self,fold_path):
        for image in tqdm(os.listdir(fold_path)):
            path = os.path.join(fold_path,image)
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(self.img_size,self.img_size)).flatten()
            np_img=np.asarray(img)
        return np_img
    def dir_path(self):
        for fdirs in range(len(self.fdirs)):
            for label in range(len(self.classes)):
                path = self.input_dir + self.fdirs[fdirs] + "/" + self.classes[label]
                # print("[DEBG]   Path Dirs :",path)
                self.fdirs_path.append(path)
        self.train1=self.fdirs_path[0]
        self.train2=self.fdirs_path[1]
        self.val1=self.fdirs_path[2]
        self.val2=self.fdirs_path[3]
        self.test1=self.fdirs_path[4]
        self.test2=self.fdirs_path[5]
    def create_plt(self,img1,img2):
        plt.figure(figsize=(10,10))
        plt.subplot(1,2,1)
        plt.imshow(img1.reshape(self.img_size,self.img_size))
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(img2.reshape(self.img_size,self.img_size))
        plt.axis('off')
        plt.title("Data in GrayScale")
        plt.show()
    def run(self):
        self.dir_path()
        print("[INFO]   Dataset Directory : ",self.fdirs_path)
        explore_data1=self.create_img_array(self.train1)
        explore_data2=self.create_img_array(self.train2)
        self.create_plt(explore_data1,explore_data2)
        return self.input_dir,self.img_size,self.classes,self.fdirs,self.fdirs_path,self.train1,self.train2,self.test1,self.test2,self.val1,self.val2

class DataAgg(object):
    def __init__(self):
        self.input_dir,self.img_size,self.classes,self.fdirs,self.fdirs_path,self.train1,self.train2,self.test1,self.test2,self.val1,self.val2=DataExplorer().run()
        self.train_data=None
        self.test_data=None
        self.val_data=None
    def create_train_data(self):
        train_data_1 = []
        train_data_2 = []
        for image1 in tqdm(os.listdir(self.train1)):
            path=os.path.join(self.train1,image1)
            img1 = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img1 = cv2.resize(img1,(self.img_size,self.img_size))
            train_data_1.append(img1)
        for image2 in tqdm(os.listdir(self.train2)):
            path=os.path.join(self.train2,image2)
            img2 = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img2 = cv2.resize(img2,(self.img_size,self.img_size))
            train_data_2.append(img2)
        self.train_data=np.concatenate((np.asarray(train_data_1),np.asarray(train_data_2)),axis=0)
    def create_test_data(self):
        test_data_1 = []
        test_data_2 = []
        for image1 in tqdm(os.listdir(self.val1)):
            path=os.path.join(self.val1,image1)
            img1 = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img1 = cv2.resize(img1,(self.img_size,self.img_size))
            test_data_1.append(img1)
        for image2 in tqdm(os.listdir(self.val2)):
            path=os.path.join(self.val1,image2)
            img2 = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img2 = cv2.resize(img2,(self.img_size,self.img_size))
            test_data_2.append(img2)
        self.test_data=np.concatenate((np.asarray(test_data_1),np.asarray(test_data_2)),axis=0)
    def create_y_data(self):
        z1 = np.zeros(len(os.listdir(self.train1)))
        o1 = np.ones(len(os.listdir(self.train2)))
        y_train=np.concatenate((o1,z1),axis=0)
        z=np.zeros(len(os.listdir(self.val1)))
        o=np.ones(len(os.listdir(self.val2)))
        y_test=np.concatenate((o,z),axis=0)
        return y_train,y_test
    def run(self):
        self.create_train_data()
        self.create_test_data()
        
        x_data = np.concatenate((self.train_data,self.test_data),axis=0)
        x_data = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
        
        y_train,y_test=self.create_y_data()
        y_data=np.concatenate((y_train,y_test),axis=0).reshape(x_data.shape[0],1)

        print("[INFO]   X Data Shape : ",x_data.shape)
        print("[INFO]   Y Data Shape : ",y_data.shape)

        x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.15,random_state=42)

        print("[INFO]   Number of training data : ",x_train.shape[0])
        print("[INFO]   Number of testing data : ",x_test.shape[0])

        x_train_flatten=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
        x_test_flatten=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
        print("[DEBG]   X train flatten : ",x_train_flatten.shape)
        print("[DEBG]   X test flatten : ",x_test_flatten.shape)

        x_train=x_train_flatten.T
        x_test=x_test_flatten.T
        y_test=y_test.T
        y_train=y_train.T

        print("[DEBG]   x train : ",x_train.shape)
        print("[DEBG]   x test : ",x_test.shape)
        print("[DEBG]   y train : ",y_train.shape)
        print("[DEBG]   y test : ",y_test.shape)

        return x_train,x_test,y_train,y_test

class CustomLogisticRegression(object):
    def __init__(self):
        self.x_train,self.x_test,self.y_train,self.y_test=DataAgg().run()
        self.init_W=np.full((self.x_train.shape[0],1),0.01)
        self.init_B=0.0
        self.lr_rate=0.01
        self.num_iter=1500
    def sigmoid(self,z):
        y_head=1/(1+np.exp(-z))
        return y_head
    def forward_backward_propogation(self):
        #Forward
        z = np.dot(self.init_W.T,self.x_train) + self.init_B
        y_head = self.sigmoid(z)
        loss= -self.y_train*np.log(y_head)-(1-self.y_train)*np.log(1-y_head)
        cost=(np.sum(loss))/self.x_train.shape[1]
        #Backwad
        deri_W = (np.dot(self.x_train,((y_head-self.y_train).T)))/self.x_train.shape[1]
        deri_B = np.sum(y_head-self.y_train)/self.x_train.shape[1]
        gradients={"derivative_weights":deri_W,"derivative_bias":deri_B}
        return cost,gradients
    def update(self):
        cost_list=[]
        cost_list2=[]
        index=[]
        for i in range(self.num_iter):
            cost,gradients=self.forward_backward_propogation()
            cost_list.append(cost)
            self.init_W=self.init_W - self.lr_rate*gradients["derivative_weights"]
            self.init_B=self.init_B - self.lr_rate*gradients["derivative_bias"]
            if i % 100 == 0:
                cost_list2.append(cost)
                index.append(i)
                print("[INFO]   Training Index and Loss  : %i: %f "%(i,cost))
        params={"weight":self.init_W,"bias":self.init_B}
        plt.plot(index,cost_list2)
        plt.xticks(index,rotation='vertical')
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost")
        plt.show()
        return params,gradients,cost_list
    def predict(self,x_data):
        z = self.sigmoid(np.dot(self.init_W.T,x_data)+self.init_B)
        y_pred=np.zeros((1,x_data.shape[1]))
        for i in range(z.shape[1]):
            if z[0,i]<= 0.5:
                y_pred[0,i]=0
            else:
                y_pred[0,i]=1
        return y_pred
    def hyper_tuning(self):
        grid_search={"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}
        logregres=LogisticRegression(random_state=42)
        log_reg_cv=GridSearchCV(logregres,grid_search,cv=10)
        log_reg_cv.fit(self.x_train.T,self.y_train.T)
        print("[INFO]   Best Hyper Parameters : ",log_reg_cv.best_params_)
        print("[INFO]   Accuracy with Hype Tuning : ",log_reg_cv.best_score_)
        
        #Training with Hyper Tuning Params
        log_reg_ht=LogisticRegression(C=log_reg_cv.best_params_['C'],penalty=log_reg_cv.best_params_['penalty'])
        log_reg_ht.fit(self.x_train.T,self.y_train.T)
        print("[INFO]   Train Accuracy after Hyper Tuning : ",format(log_reg_ht.fit(self.x_train.T,self.y_train.T).score(self.x_train.T,self.y_train.T)))
        print("[INFO]   Test Accuracy after Hyper Tuning : ",format(log_reg_ht.fit(self.x_test.T,self.y_test.T).score(self.x_test.T,self.y_test.T)))
    def run(self):
        params,grads,cost_list=self.update()
        y_pred_test=self.predict(self.x_test)
        y_pred_train=self.predict(self.x_train)
        print("[INFO]   Train Accuracy : {} %".format(round(100-np.mean(np.abs(y_pred_train-self.y_train))*100,2)))
        print("[INFO]   Test Accuracy : {} %".format(round(100-np.mean(np.abs(y_pred_test-self.y_test))*100,2)))
        self.hyper_tuning()

if __name__ == '__main__':
    CustomLogisticRegression().run()