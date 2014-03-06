import cv2
import numpy as np
import os
import csv
import sys
import time
import datetime
import random
from random import shuffle
from sklearn import svm
#from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

pathtrain = r'C:\Users\a0271831\Documents\Prog_Projects\Hadoop_Etc\Kaggle_DogCat\train'
pathtest = r'C:\Users\a0271831\Documents\Prog_Projects\Hadoop_Etc\Kaggle_DogCat\test1'
pathfea = r'C:\Users\a0271831\Documents\Prog_Projects\Hadoop_Etc\Kaggle_DogCat\train_feat'

num_train = len(os.listdir(pathtrain))
print "No of train pics->", num_train
num_test = len(os.listdir(pathtest))
print "No of test pics->", num_test
STANDARD_SIZE = (300, 167)

feat_dict = {}
i=0
res = []
cat = []
dog = []
data = []
des = 0
all_data = np.empty((100,32))
flist = os.listdir(pathtrain)
shuffle(flist)
for dir_entry in flist:
    dir_entry_path = os.path.join(pathtrain,dir_entry)
    dir_fea_path = os.path.join(pathfea,dir_entry)
    
    
    if os.path.isfile(dir_entry_path):
        img = cv2.imread(dir_entry_path)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #print gray.shape
        #img1 = cv2.resize(gray,STANDARD_SIZE)
        #print "after",img1.shape
        orb = cv2.ORB(nfeatures=10)
        kp, des = orb.detectAndCompute(gray,None)
        #print "Des->",des.shape
        #des1 = des.astype(np.float32)
        #all_data = np.append(all_data,des1,0)
        if len(kp)>0:
            drow = des.shape[0]
            dcol = des.shape[1]
            flatdes = des.reshape(drow*dcol).astype(np.float32)
            a = list(flatdes)
        if len(a)>80:
            data.append(a[:80])
            if 'cat' in dir_entry:
                res.append(0)
                fname = 'cat'
            else:
                res.append(1)
                fname = 'dog'
        #img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv2.imwrite(dir_fea_path,img)
        
    i = i+1
    if i == 5000: break
#data1 = np.array(data,dtype=np.float32)
##dog1 = np.vstack(dog)
##cat1 = np.vstack(cat)
##print "Dog1->",dog1.shape
##print "Cat1->",cat1.shape
##res1 = np.array(res,dtype=np.float32)
##
##criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
##ret,label,center=cv2.kmeans(dog1,2,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
### Now separate the data, Note the flatten()
##A = dog1[label.ravel()==0]
##B = dog1[label.ravel()==1]
## 
### Plot the data
##plt.scatter(A[:,0],A[:,1])
##plt.scatter(B[:,0],B[:,1],c = 'r')
##plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
##plt.xlabel('Height'),plt.ylabel('Weight')
##plt.show()
#data= np.array(data)
#res = np.array(res)
#knn = cv2.KNearest()
#knn.train(data,np.array(res),isRegression = False)
clf = svm.SVC()
clf.fit(data, res)
print "done training"
i=0
st1 = time.time()
of2 = open('cat_25Oct1.csv','wb')
writer2 = csv.writer(of2, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
writer2.writerow(["id","label"])
try:
    for dir_entry in os.listdir(pathtest):
        dir_entry_path = os.path.join(pathtest,dir_entry)
        if os.path.isfile(dir_entry_path):
            img = cv2.imread(dir_entry_path)
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB(nfeatures=10)
            kp, des = orb.detectAndCompute(gray,None)
        if len(kp)>0:
            drow = des.shape[0]
            dcol = des.shape[1]
            flatdes = des.reshape(drow*dcol).astype(np.float32)
            a = list(flatdes)
        if len(a)>80:
            a_50 = a[:80]
            label = clf.predict(a_50)
        else: label=random.randint(0,1)
        #ret,result,neighbours,dist = knn.find_nearest(des1,k=5)
        #label = random.randint(0,1)
        pic_id = int(dir_entry.strip('.jpg'))
        #print "id->",pic_id
        writer2.writerow([int(pic_id),int(label)])
        i = i+1
        #if i== 15: break
except:
    print "exception done",sys.exc_info()[0]

print "done writing"
of2.close()



