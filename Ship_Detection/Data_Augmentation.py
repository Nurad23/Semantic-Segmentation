from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
import h5py
import sys,pdb,time
import numpy as np
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pandas as pd
#%%
dataset_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath('__file__')))))))\
+'/Desktop/OS/USC COURSES/EE 599 - Deep Learning/Project/Project_data/train_v2'

cwd = os.getcwd()
print(cwd)






def count_ships(file="train"):
    """
    Loads a csv, creates the fields `HasShip` and `TotalShips` dropping `EncodedPixels` and setting `ImageId` as index.
    """
    df = pd.read_csv("out_last.csv")

    df['HasShip'] = df['EncodedPixels'].notnull()
    df = df.groupby("ImageId").agg({'HasShip': ['first', 'sum']}) # counts amount of ships per image, sets ImageId to index
    df.columns = ['HasShip', 'TotalShips']
    return df


filename="0ace8abe2.jpg"
df_train = count_ships("train")


#

deleted=[]
i=0
for  filename in os.listdir(dataset_path): #go through image files 
    
    row=filename
    if row in df_train.index: #check if the image exist in the csv 
        if df_train.loc[[row],['TotalShips']].values[0]==0 and os.path.isfile(dataset_path + '/'+row): # go into this loop if image exists and has 0 ships
                os.remove(dataset_path + '/'+row) #remove that image
                #df_2=df_2[df_2.ImageId != row] 
                deleted.append(row) #append the deleted ships
                i+=1
            #print(i)

    if i==40000: #do this 40000 times to get rid of half of the instances with no ships
        break

print(i)

print('Out of loop')
#print(deleted)

