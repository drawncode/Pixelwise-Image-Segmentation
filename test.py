import numpy as np
import glob
from keras.models import *
from keras.layers import *
import keras
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


dataimages, maskimages = glob.glob("./chota_data/train_data_1/*.tiff"), glob.glob("./chota_data/train_mask_1/*.tiff")
files = np.array([t for t in zip(dataimages, maskimages)], dtype=object)


def read_image(filelist):
#     print(filelist)
    data, mask = list(), list()
    for d,m in filelist:
        print(d)
        data.append(cv2.imread(d))
        mask.append(cv2.imread(m, cv2.IMREAD_GRAYSCALE))
    mask = np.array(mask, dtype=float)/255.0
    return np.array(data, dtype=float)/255.0, mask


def generator(batchsize, flist):
    flist_ = np.copy(flist)
    while True:
        rnd = np.random.permutation(len(flist_))
        flist_ = flist_[rnd]
        batches = int(len(flist_)/batchsize)
        for i in range(batches):
            yield read_image(flist_[i*batchsize:(i+1)*batchsize])

model=load_model('./part23_model.h5')
model.summary()
D_test, M_test = read_image(files)

M_new = model.predict(D_test)
both = np.zeros((600,400))
print(both.shape)
for i in range(2):
    both[:300,:] = M_new[i,:,:]
    both[300:,:] = M_test[i,:,:]
    both*=255.0
#     plt.figure(figsize=(32,32))
#     print(both.shape)
#     plt.imshow(both,cmap="gray")
#     plt.show()
    cv2.imwrite("output/"+str(i)+"_img.jpg",both)