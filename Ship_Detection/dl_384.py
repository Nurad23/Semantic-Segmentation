
# -*- coding: utf-8 -*-


from keras.utils import plot_model
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.data import imread
from skimage import io
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from itertools import islice
import keras
import tensorflow as tf
import zipfile
from PIL import Image, ImageFile
from keras.layers.normalization import BatchNormalization
from keras import optimizers

ImageFile.LOAD_TRUNCATED_IMAGES = True
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

with tf.Session() as sess:
  devices = sess.list_devices()



if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


marks = pd.read_csv('EE599_Zipped/Project_data_modified/train_ship_segmentations_v2.csv') # Markers for ships
images = os.listdir('EE599_Zipped/Project_data_modified/train_v2') # Images for training
os.chdir('EE599_Zipped/Project_data_modified/train_v2')
new_size=384


### Displaying the masks ######

#ImageId = '03bdfd40f.jpg'
#files=[ImageId,'03c99ebbe.jpg']

def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T 

def mask_part(pic):
    '''
    Function that encodes mask for single ship from .csv entry into numpy matrix
    '''
    mask_img = np.zeros(768*768)
    starts = pic.split()[0::2]
    lens = pic.split()[1::2]
    for i in range(len(lens)):
        mask_img[(int(starts[i])-1):(int(starts[i])-1+int(lens[i]))] = 1
    return np.reshape(mask_img, (768, 768, 1))

def is_empty(key):
    '''
    Function that checks if there is a ship in image
    '''
    df = marks[marks['ImageId'] == key].iloc[:,1]
    if len(df) == 1 and type(df.iloc[0]) != str and np.isnan(df.iloc[0]):
        return True
    else:
        return False
    
def masks_all(key):
    '''
    Merges together all the ship markers corresponding to a single image
    '''
    df = marks[marks['ImageId'] == key].iloc[:,1]
    masks= np.zeros((768,768,1))
    if is_empty(key):
        return masks
    else:
        for i in range(len(df)):
            masks += mask_part(df.iloc[i])
        return np.transpose(masks, (1,0,2))
    
def make_batch(files, batch_size):
    '''
    Creates batches of images and masks in order to feed them to NN
    '''
    X = np.zeros((batch_size, 768, 768, 3))
    Y = np.zeros((batch_size, 768, 768, 1)) # I add 1 here to get 4D batch
    for i in range(batch_size): 
        ship=np.random.choice(files)
        X[i] = (io.imread(ship)) # Original images are in 0-255 range. Normalization
        Y[i]= masks_all(ship)
    return X, Y


def make_batch_resize(files, batch_size):
    '''
    Creates batches of images and masks in order to feed them to NN
    '''
    X = np.zeros((batch_size, new_size, new_size, 3))
    Y = np.zeros((batch_size, new_size, new_size, 1)) # I add 1 here to get 4D batch
    for i in range(batch_size): 
        ship=np.random.choice(files)
        img = Image.open(ship)
        X[i] = img.resize((new_size,new_size))
        X[i]=X[i]/255 # Original images are in 0-255 range, I want it in 0-1
        r= masks_all(ship).reshape(768,768)
        r1 =Image.fromarray(r)
        r2 = r1.resize((new_size,new_size))
        r3 = np.array(r2)
        Y[i]=r3.reshape(new_size,new_size,1)
    return X, Y

# Optional
    
def transform(X, Y):
    '''
    Function for augmenting images. 
    It takes original image and corresponding mask and performs the
    same flipping and rotation transforamtions on both in order to 
    perserve the overlapping of ships and their masks
    '''
    x = np.copy(X)
    y = np.copy(Y)
  
# add noise:
    x[:,:,0] = x[:,:,0] + np.random.normal(loc=0.0, scale=0.01, size=(new_size,new_size))
    x[:,:,1] = x[:,:,1] + np.random.normal(loc=0.0, scale=0.01, size=(new_size,new_size))
    x[:,:,2] = x[:,:,2] + np.random.normal(loc=0.0, scale=0.01, size=(new_size,new_size))
    # Adding Gaussian noise on each rgb channel; this way we will NEVER get two completely same images.
    # Note that this transformation is not performed on Y 
    x[np.where(x<0)] = 0
    x[np.where(x>1)] = 1
# axes swap:
    if np.random.rand()<0.4: # 0.4 chances for this transformation to occur (same for two below)
        x = np.swapaxes(x, 0,1)
        y = np.swapaxes(y, 0,1)
# vertical flip:
    if np.random.rand()<0.4:
        x = np.flip(x, 0)
        y = np.flip(y, 0)
# horizontal flip:
    if np.random.rand()<0.4:
        x = np.flip(x, 1)
        y = np.flip(y, 1)
    return x, y  

def Generator(files, batch_size):
    '''
    Generates batches of images and corresponding masks
    '''
    while True:
        X, Y = make_batch_resize(files, batch_size)
#        datagen = ImageDataGenerator(
#            rescale=1./255,
#                     brightness_range=(0.5, 1.0),
#                     )
#        datagen.fit(X)
        for i in range(batch_size):
            X[i], Y[i] = transform(X[i], Y[i])
        yield X, Y

# Intersection over Union for Objects
def IoU(y_true, y_pred, tresh=1e-10):
    Intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    Union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - Intersection
    return K.mean( (Intersection + tresh) / (Union + tresh), axis=0)
# Intersection over Union for Background
def back_IoU(y_true, y_pred):
    return IoU(1-y_true, 1-y_pred)
# Loss function
def IoU_loss(in_gt, in_pred):
    #return 2 - back_IoU(in_gt, in_pred) - IoU(in_gt, in_pred)
    return 1 - IoU(in_gt, in_pred)

def dice_coef(y_true, y_pred):
    y_true = K.round(K.reshape(y_true, [-1]))
    y_pred = K.round(K.reshape(y_pred, [-1]))
    inter = K.sum(y_true * y_pred)
    return 2 * inter / (K.sum(y_true) + K.sum(y_pred))

model=Sequential()
inputs = Input((new_size, new_size, 3))

c1 = Conv2D(12, (3, 3), activation='relu', padding='same') (inputs)
c1 = Conv2D(12, (3, 3), activation='relu', padding='same') (c1)
c1=BatchNormalization()(c1)

p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(24, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(24, (3, 3), activation='relu', padding='same') (c2)
c2=BatchNormalization()(c2)
p2 = MaxPooling2D((2, 2)) (c2)
    
c3 = Conv2D(48, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(48, (3, 3), activation='relu', padding='same') (c3)
c3=BatchNormalization()(c3)

p3 = MaxPooling2D((2, 2)) (c3)
    
c4 = Conv2D(96, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(96, (3, 3), activation='relu', padding='same') (c4)
c4=BatchNormalization()(c4)

    
u5 = Conv2DTranspose(48, (2, 2), strides=(2, 2), padding='same') (c4)
u5 = concatenate([u5, c3])
c5 = Conv2D(48, (3, 3), activation='relu', padding='same') (u5)
c5 = Conv2D(48, (3, 3), activation='relu', padding='same') (c5)
c5=BatchNormalization()(c5)

    
u6 = Conv2DTranspose(24, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c2])
c6 = Conv2D(24, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(24, (3, 3), activation='relu', padding='same') (c6)
c6=BatchNormalization()(c6)

    
u7 = Conv2DTranspose(12, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c1], axis=3)
c7 = Conv2D(12, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(12, (3, 3), activation='relu', padding='same') (c7)
c7=BatchNormalization()(c7)

    
outputs = Conv2D(1, (1, 1), activation='sigmoid') (c7)
model = keras.models.Model(inputs, outputs)
Sgd = optimizers.SGD(lr=0.01, momentum=0.99)

adam = optimizers.adam(lr=0.0040, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=Sgd, loss= [IoU_loss], metrics=[IoU])  

model.summary()
 
results = model.fit_generator(Generator(images, batch_size = 16),steps_per_epoch=100, epochs = 20)


model.save("UNet_aws_AD_384_low_batch.h5")
#plot_model(model, to_file='UNet_aws_2_model_inc_channel.png')


## Testing
#batch_size=1
#files=['03bdfd40f.jpg']
#model.load_weights('UNet_smaller.h5')
#os.chdir('train_1')
#X, Y = make_batch(files, batch_size)
#Y_pred=model.predict(X)
#plt.imshow(np.reshape(Y[0]*255,(768,768)), cmap="gray")
#plt.show()
#plt.imshow(np.reshape(X[0]*255,(768,768,3)), cmap="gray")
#plt.show()
#plt.imshow(np.reshape(Y_pred[0]*255,(768,768)), cmap="gray")
#plt.show()
