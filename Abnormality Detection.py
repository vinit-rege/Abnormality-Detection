from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D

from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
import theano
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing


img_rows=158
img_cols=238
img_depth=200
listing=os.listdir(r'C:\Users\HP\Documents\neural\input')
ipt_array=[]
for vid in listing:
    vid=r'C:\Users\HP\Documents\neural\input\\'+vid
    frames=[]
    cap=cv2.VideoCapture(vid)
    fps=cap.get(5)
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    
    for k in range(200):
        ret, frame = cap.read()
        #frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        frames.append(gray) 
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)
    print (input.shape)
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print (ipt.shape)
    ipt_array.append(ipt)

listing=os.listdir(r'C:\Users\HP\Documents\neural\input_a')
for vid in listing:
    vid=r'C:\Users\HP\Documents\neural\input_a\\'+vid
    frames=[]
    cap=cv2.VideoCapture(vid)
    fps=cap.get(5)
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    
    for k in range(200):
        ret, frame = cap.read()
        #frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        frames.append(gray) 
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)
    print (input.shape)
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print (ipt.shape)
    ipt_array.append(ipt)

ipt_array1=np.array(ipt_array)
num_samples=len(ipt_array1)
print(num_samples)

#labels

labels=np.ones(num_samples,dtype = int)
labels[0:34]=0
labels[34:68]=1
train_data=[ipt_array1,labels] 
(x_train,y_train)=(train_data[0],train_data[1])
print("X_trainshape=",x_train.shape)
print("Y_trainshape=",y_train.shape)
train_set = np.zeros((num_samples,1, img_rows,img_cols,img_depth))

for h in range(num_samples):
    train_set[h][0][:][:][:]=x_train[h][:][:][:]
#train_set = np.zeros((num_samples,1,img_depth, img_rows,img_cols))    

patch_size=200  #number of frames processed per second
print("train_set shape=",train_set.shape)

#CNN Training Parametres
batch=10
epochs=50
classes=2

#converting class vectors to binary for categorical cross entropy
y_train = np_utils.to_categorical(y_train, classes)
#Y_train=np_utils.to_categorical(y_train,classes)

#number of filters used
filters=[32,64]
#level of pooling
pool=[3,3]
#level of convoluton at each layer
conv=[5,5]

# Pre-processing

train_set = train_set.astype('float32')

train_set -= np.mean(train_set)

train_set /=np.max(train_set)

#train_set /= 255
#Model

model = Sequential()
model.add(Convolution3D(filters[0],kernel_dim1=conv[0], kernel_dim2=conv[0], kernel_dim3=conv[0], input_shape=(1,img_rows, img_cols,patch_size), activation='relu'))

model.add(MaxPooling3D(pool_size=(pool[0], pool[0], pool[0])))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, init='normal', activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(classes,init='normal'))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop',metrics=['accuracy'])

# Split the data

X_train_new, X_val_new, y_train_new,y_val_new =  train_test_split(train_set, y_train, test_size=0.2, random_state=4)


# Train the model
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
#datagen.fit(X_train_new)

    # Fit the model on the batches generated by datagen.flow().
#model.fit_generator(datagen.flow(X_train_new, y_train_new,
 #                       batch_size=batch),
  #                      samples_per_epoch=num_samples,
   #                     nb_epoch=epochs,
    #                    validation_data=(X_val_new, y_val_new))
model.fit(X_train_new, y_train_new, batch_size=batch,nb_epoch = epochs,validation_data=(X_val_new,y_val_new),verbose=2,shuffle=True)


#hist = model.fit(train_set, y_train, batch_size=batch,
#          nb_epoch=epochs,validation_split=0.2,shuffle=True)


 # Evaluate the model
score = model.evaluate(train_set, y_train, batch_size=batch, show_accuracy=True)
print('Test score:', score[0])
print('Test accuracy:', score[1])