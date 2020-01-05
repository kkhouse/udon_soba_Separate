import keras
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json

num_classes = 2

X_train, X_test, y_train, y_test = np.load("./data/us_data/Xy_data.npy",allow_pickle=True)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.75))
model.add(Dense(2))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
optimizer=opt,
metrics=['accuracy'])

history=model.fit(X_train, y_train,epochs=80, batch_size=40, verbose=1,validation_data=(X_test, y_test))
y_pred = model.predict(X_test)


model_json = model.to_json()
with open("./us_judge/model_in_json.json", "w") as json_file:
    json.dump(model_json, json_file)

model.save_weights("./us_judge/model_weights.h5")







#過学習チェック
plt.figure()
plt.scatter(y_train,model.predict(X_train),label='Train',c='blue')
plt.scatter(y_test,y_pred,c='lightgreen',label='Test',alpha=0.8)
plt.title('Neural Network Predictor')
plt.xlabel('Measured')
plt.ylabel('Predicted') 
plt.show()


loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=len(loss)

plt.plot(range(epochs), loss, marker = '.', label = 'loss')
plt.plot(range(epochs), val_loss, marker = '.', label = 'val_loss')
plt.legend(loc = 'best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()






