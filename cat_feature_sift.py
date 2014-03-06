import cv2
import numpy as np
import os
import csv
import sys
import time
import datetime
import random
from sklearn.cluster import KMeans

pathtrain = r'C:\Users\a0271831\Documents\Prog_Projects\Hadoop_Etc\Kaggle_DogCat\train'
pathtest = r'C:\Users\a0271831\Documents\Prog_Projects\Hadoop_Etc\Kaggle_DogCat\test1'
pathfea = r'C:\Users\a0271831\Documents\Prog_Projects\Hadoop_Etc\Kaggle_DogCat\train_feat'

num_train = len(os.listdir(pathtrain))
print "No of train pics->", num_train
num_test = len(os.listdir(pathtest))
print "No of train pics->", num_test

data = {}
i=0
res = []
for dir_entry in os.listdir(pathtrain):
    dir_entry_path = os.path.join(pathtrain,dir_entry)
    dir_fea_path = os.path.join(pathfea,dir_entry)
    if 'cat' in dir_entry:
        res.append(0)
    else: res.append(1)
    if os.path.isfile(dir_entry_path):
        img = cv2.imread(dir_entry_path)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT()
        kp, des = sift.detectAndCompute(gray,None)
        #kp = sift.detect(gray,None)
        #kp,des = sift.compute(gray,kp)
        #img=cv2.drawKeypoints(gray,kp)
        img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        cv2.imwrite(dir_fea_path,img)

    i = i+1
    if i == 1: break

i=0
st1 = time.time()
of2 = open('cat_25Oct1.csv','wb')
writer2 = csv.writer(of2, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
writer2.writerow(["id","label"])
try:
    for dir_entry in os.listdir(pathtest):
    
        label = random.randint(0,1)
        pic_id = int(dir_entry)
        writer2.writerow([int(pic_id),int(label)])
        i = i+1
        if i== 5: break
except:
    print "exception done"



