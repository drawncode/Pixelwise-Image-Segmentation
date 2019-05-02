import numpy as np
import glob
from keras.models import *
from keras.layers import *
import keras
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


dataimages, maskimages = glob.glob("./chota_data/train_data_1/*.tiff"), glob.glob("./chota_data/train_mask_1/Mask/*.tiff")
files = np.array([t for t in zip(dataimages, maskimages)], dtype=object)


def read_image(filelist):
    data, mask = list(), list()
    for d,m in filelist:
        data.append(cv2.imread(d))
        mask.append(cv2.imread(m, cv2.IMREAD_GRAYSCALE))
    mask = np.array(mask, dtype=float)/255.0
    # mask = np.stack((1.0-mask, mask), axis=-1)
    return np.array(data, dtype=float)/255.0, mask


def generator(batchsize, flist):
    flist_ = np.copy(flist)
    while True:
        rnd = np.random.permutation(len(flist_))
        flist_ = flist_[rnd]
        batches = int(len(flist_)/batchsize)
        for i in range(batches):
            yield read_image(flist_[i*batchsize:(i+1)*batchsize])


# defining the model

input_shape=(300,400,3)
model = Sequential()
model.add(Conv2D(32, kernel_size=(13,13), strides=(1,1), activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(11,11), strides=(2,2), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(9,9), strides=(1,1), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(7,7), strides=(2,2), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(7,7), strides=(2,2), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2DTranspose(256, kernel_size=(5,5), strides=(2,2), activation='relu',output_padding=(0,1)))
model.add(BatchNormalization())
model.add(Conv2DTranspose(128, kernel_size=(7,7), strides=(2,2), activation='relu', output_padding=(0,0)))
model.add(BatchNormalization())
model.add(Conv2DTranspose(128, kernel_size=(7,7), strides=(2,2), activation='relu', output_padding=(0,0)))
model.add(BatchNormalization())
model.add(Conv2DTranspose(64, kernel_size=(9,9), strides=(1,1), activation='relu', output_padding=(0,0)))
model.add(BatchNormalization())
model.add(Conv2DTranspose(32, kernel_size=(11,11), strides=(2,2), activation='relu', output_padding=(1,1)))
model.add(BatchNormalization())
model.add(Conv2DTranspose(1, kernel_size=(13,13), strides=(1,1), activation='softmax', output_padding=(0,0)))

model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adam(),
             metrics=['accuracy'])

model.summary()

# training
batches = 5
for t in range(5):
 history = model.fit_generator(generator(batches,files),
                              steps_per_epoch=int(len(files)/batches),
                              epochs=2,
                              verbose=1,
                              validation_data=generator(batches, files),
                              validation_steps=1)
 model.save('part_2'+str(t)+'_model.h5')

#predictions
#model.load_weights('part23_model.h5')
#D_test, M_test = read_image(files)
#model.predict(D_test)
#for t in range(10):
#    both[300*i:300*(i+1),:400] = M_test[i,:,:,0]
#    both[300*i:300(i+1),400:] = M_test[i,:,:0]
#plt.figure(figsize=(32,32))
#plt.imshow(both)
#plt.show()