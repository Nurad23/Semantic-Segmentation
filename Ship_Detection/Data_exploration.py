
import h5py
import sys,pdb,time
import numpy as np
from sklearn.utils import shuffle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pandas as pd


cwd = os.getcwd()
print(cwd)
dataset_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath('__file__')))))))\
+'/Desktop/OS/USC COURSES/EE 599 - Deep Learning/Project/Project_data_modified/train_v2'

df_train=pd.read_csv('Project_data_modified/train_ship_segmentations_v2.csv')

cwd = os.getcwd()
print(df_train.iloc[0])


mylist = os.listdir('Project_data_modified/train_v2') #List every image in path
#print(mylist)

print(len(mylist))
myarray = np.asarray(mylist)

df_train.set_index(['ImageId'], inplace=True) #set the index to check whether the image files exist in the csv
df_train_new=df_train.loc[myarray,:] #get the instaces with the current data corresponding to current images
df_train_new.head()

print(len(df_train_new))
df_train_new.to_csv('df_train_new.csv') #new csv with deleted dataset
print(df_train_new.columns)

df_train_new.reset_index(inplace=True) #reset the index to be able to count the ships

print(df_train_new.columns)



def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction



ships = df_train_new[~df_train_new.EncodedPixels.isna()].ImageId.unique()
noships = df_train_new[df_train_new.EncodedPixels.isna()].ImageId.unique()

plt.bar(['Ships', 'No Ships'], [len(ships), len(noships)])
plt.ylabel('Number of Images')
plt.show()

id_images_obj = df_train_new.dropna().groupby('ImageId').count()

id_images_obj.rename({'EncodedPixels': 'ObjCount'}, axis='columns', inplace=True)

objects = id_images_obj.ObjCount.sum()

print("Total No. of object: ", objects)

print(id_images_obj.describe())

id_images_obj.ObjCount.hist(bins=15)
plt.xlabel('No. of ships')
plt.ylabel('Images')
plt.show()
