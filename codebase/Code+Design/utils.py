import numpy as np


import csv
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb, xyz2lab
from skimage.io import imsave
from keras.utils import to_categorical
import PIL.Image
import numpy as np


def GetBitcoinIndex():
    filename = 'M:/data/bitcoin/bitcoin.csv'
    num_lines = sum(1 for line in open(filename))
    X = np.linspace(-1, 1, num=num_lines, endpoint=True)
    Y = []
    file = open(filename, 'r')
    for i in range(num_lines):
        Y.append(file.readline().split(",")[1])
    Y = np.array(Y, dtype=float) / 20000.
    x = X.reshape(X.shape[0], 1)
    y = Y.reshape(Y.shape[0], 1)
    return x,y

def LoadFaces(n):
    width = 184
    height = 224
    attributefile = open('M:/data/celeba/list_attr_celeba.txt', 'r')
    attributefile.readline()
    attributefile.readline().split()[20]

    # Get images
    images = np.zeros((n, 224, 184, 1), np.float32)
    gender = []
    womenn = 0
    menn = 0

    for i in range(0, n):
        attributes = attributefile.readline().split()
        filename = attributes[0];
        male = float(attributes[21]);
        if male < 0:
            womenn = womenn + 1
        else:
            menn = menn + 1
        
        image = load_img('m:/data/celeba/' + filename)
        image = image.resize((width, height), PIL.Image.ANTIALIAS)
        image = np.reshape(rgb2lab(1./255.*img_to_array(image))[:,:,0], (height, width, 1))
        images[i] = image/128.
        gender.append(male)

    attributefile.close();
    print("loaded " + str(womenn) + " women")
    print("loaded " + str(menn) + " men")
    Y = np.array(gender, np.float32)*(-1)
    Y[Y == -1] = 0
    Y = to_categorical(Y, 2)
    return images, Y
    
def ShowImage(X):
    width = X.shape[1]
    height = X.shape[0]
    cur = np.zeros((height, width, 3))
    cur[:,:,0] = X[:,:,0]*128
    return array_to_img(lab2rgb(cur))

