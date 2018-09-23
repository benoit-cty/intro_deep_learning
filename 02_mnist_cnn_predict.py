'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback
import numpy as np

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# Filename of the model weights output
DATA_ROOT = './'
model_weights_file = "mnist_cnn_final_weights.hdf5"


from PIL import Image, ImageFilter

def imageprepare(argv):
  """
  This function returns the pixel values.
  The imput is a png file location.
  """
  im = Image.open(argv).convert('L')
  width = float(im.size[0])
  height = float(im.size[1])
  newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels
  
  if width > height: #check which dimension is bigger
    #Width is bigger. Width becomes 20 pixels.
    nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
    if (nheigth == 0): #rare case but minimum is 1 pixel
      nheigth = 1  
    # resize and sharpen
    img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
    newImage.paste(img, (4, wtop)) #paste resized image on white canvas
  else:
    #Height is bigger. Heigth becomes 20 pixels. 
    nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
    if (nwidth == 0): #rare case but minimum is 1 pixel
      nwidth = 1
     # resize and sharpen
    img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
    newImage.paste(img, (wleft, 4)) #paste resized image on white canvas
  
  #newImage.save("sample.png")

  #tv = list(newImage.getdata()) #get pixel values
  #tva = np.asarray( newImage, dtype="uint8" )
  #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
  #tva = [ (255-x)*1.0/255.0 for x in tv] 
  #tva = newImage / 255
  tva = 1 - (np.asarray( newImage, dtype="uint8" ) / 255)
  print(tva)
  #print("Input shape : ", tva.shape())
  return tva
  #print(tva)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.load_weights(DATA_ROOT + 'mnist_cnn_final_weights.hdf5')
filename = DATA_ROOT + '3-2.png'
image = imageprepare(filename)
prediction = model.predict_classes(image.reshape((1, 28, 28, 1)), verbose=2)
print(prediction)
print(filename)
print("The End !")
