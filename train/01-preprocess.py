# %%
import os
import gc
import cv2
import math
import copy
import time
import random
import shutil, sys    

import glob
# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2


from sklearn.metrics import f1_score,roc_auc_score


import timm
from timm.models.efficientnet import *

# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict


import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
from scipy import ndimage

# %%
def autocropmin(image, threshold=100, kernsel_size = 10):
        
    img = image.copy()
    
    SIZE = img.shape[0]
    imgfilt = ndimage.minimum_filter(img, size=kernsel_size)
    img_b=np.where(imgfilt<threshold,0,255)
    a=img_b[:,:,0].sum(axis=1)
    a=np.concatenate(([0],a,[0]))

    a_=np.where(a==0)[0]
    mina=a_[np.argmax(a_[1:]-a_[:-1])]
    maxa=a_[np.argmax(a_[1:]-a_[:-1])+1]-1

    b=img_b[:,:,0].sum(axis=0)
    b=np.concatenate(([0],b,[0]))

    b_=np.where(b==0)[0]
    minb=b_[np.argmax(b_[1:]-b_[:-1])]
    maxb=b_[np.argmax(b_[1:]-b_[:-1])+1]-1

    if  mina!=maxa and minb!=maxb:
        imageout=img[mina:maxa,minb:maxb,:]
    else:
        imageout=img

    return imageout

# %%
train_df_list_ = pd.read_csv("./chih_4_fold_covid_train_df.csv") #使用已經切割好fold index的影像資訊彙整檔案
valid_df_list_ = pd.read_csv("./chih_4_fold_covid_valid_df.csv") #使用已經切割好fold index的影像資訊彙整檔案

# %%
drop_train_df_list_ = train_df_list_.drop_duplicates(subset='token_key')
drop_valid_df_list_ = valid_df_list_.drop_duplicates(subset='token_key')
# all_train_list=[list(glob.glob(os.path.join("/ssd8/2023COVID19/Train_Valid_dataset/train/positive", "*"))),
#                 list(glob.glob(os.path.join("/ssd8/2023COVID19/Train_Valid_dataset/train/negative", "*"))),
#                 list(glob.glob(os.path.join("/ssd8/2023COVID19/Train_Valid_dataset/valid/positive", "*"))),
#                 list(glob.glob(os.path.join("/ssd8/2023COVID19/Train_Valid_dataset/valid/negative", "*"))),
#                ]

all_train_list =[drop_train_df_list_[drop_train_df_list_['label']==1].path.values.tolist(),
                 drop_train_df_list_[drop_train_df_list_['label']==0].path.values.tolist(),
                 drop_valid_df_list_[drop_valid_df_list_['label']==1].path.values.tolist(),
                 drop_valid_df_list_[drop_valid_df_list_['label']==0].path.values.tolist()]
# all_train_list =[drop_valid_df_list_[drop_valid_df_list_['label']==1].path.values.tolist(),
#                  drop_valid_df_list_[drop_valid_df_list_['label']==0].path.values.tolist()]

# %%
# all_train_list[1].remove("/home/fate/covid19_CT/input/train/non_covid/ct_scan1073")
# all_train_list[1].remove("/home/fate/covid19_CT/input/train/non_covid/ct_scan_781")

# %%
for train_list in all_train_list:
    diff_shape_ct_list=[]
    
    for i in tqdm(range(len(train_list))):
        tmp_list=list(glob.glob(os.path.join(train_list[i], "*")))
        tmp_shape_set=set()
        for j in range(len(tmp_list)):

            str1=tmp_list[j]
            img=cv2.imread(str1)

            try:
                tmp_shape_set.add(img.shape)
            except:
                print(str1)
                print("bug file")
                continue

            img=autocropmin(img)

            
            str1=str1.replace("/valid/","/valid_pure_crop_challenge/")
            str1=str1.replace("/train/","/train_pure_crop_challenge/")


            folder_path="/".join(str1.split("/")[:-1])

            if len(tmp_shape_set)!=1:
                shutil.rmtree(folder_path)
                diff_shape_ct_list.append(folder_path)
                
                break

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            cv2.imwrite(str1,img)
       
    for ct_path in diff_shape_ct_list:
        
        str2=ct_path   
        str1=str2.replace("/valid_pure_crop_challenge/","/valid/")
        str1=str1.replace("/train_pure_crop_challenge/","/train/")

        tmp_list=list(glob.glob(os.path.join(str1, "*")))
        last_file=str(len(tmp_list)-1)+".jpg"
        str1=str1+"/"+last_file
        str2=str2+"/"+last_file
        img=cv2.imread(str1)
        img=autocropmin(img)
        folder_path="/".join(str2.split("/")[:-1])

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        cv2.imwrite(str2,img)
        

# %%



