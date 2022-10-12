# 초보자 딥러닝 입문하기

# 학습 목표
# 실습을 통해 DNN을 이해해본다.
# 데이터 셋 : cifar-10

# 사전 준비
# 라이브러리 설치 필요
"""
pip install tensorflow
pip install flask==2.1.3
pip install pandas numpy matplotlib 
"""

# 개발 환경
"""
python ver :  3.8.13
tensorflow ver :  2.10.0
keras ver :  2.10.0
matplotlib ver :  3.6.1
numpy ver :  1.23.1
"""



#%% 01 라이브러리 임포트
import time, os
import tensorflow as tf
import keras
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten

from keras.datasets import cifar10

start = time.time()

print("python ver : ", sys.version)
print("tensorflow ver : ", tf.__version__)
print("keras ver : ", keras.__version__)
print("matplotlib ver : ", mpl.__version__)
print("numpy ver : ", np.__version__)

print("location : ", os.getcwd() )

#%% 02 데이터 셋 준비
(X_train, Y_train) , (X_test, Y_test) = cifar10.load_data()
print( X_train.shape, Y_train.shape, X_test.shape, Y_test.shape )

#%% 03 이미지 확인
fig = plt.figure(figsize=(20,5))

for i in range(30):
    ax = fig.add_subplot(3, 10, i+1, xticks=[], yticks=[])
    ax.imshow(X_train[i])

plt.show()

#%% 04 데이터 전처리
# 이미지의 픽셀 값을 0~1로 변경
X_train_n = X_train/255.0
X_test_n = X_test /255.0

print(X_train_n[0][0:1])

# target값을 원핫 인코딩
print(Y_train[0:3])

Y_train_n = to_categorical(Y_train)
Y_test_n = to_categorical(Y_test)

print(Y_train_n[0:3])

#%% 05 모델 구축
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=4, padding='same', strides=1, activation='relu', input_shape=(32,32,3)))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=4, padding='same', strides=1, activation='relu'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=4, padding='same', strides=1, activation='relu'))
model.add(MaxPool2D(pool_size=2))

# FCL(fully connected layer)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10))                # 출력층(10개 노드) 
model.add(Activation('softmax'))

print(model.summary() )

#%% 06 모델 최적화, loss 함수 설정
model.compile(loss='categorical_crossentropy', 
              optimizer='sgd', 
              metrics=['accuracy'])

#%% 07 모델 학습 시키기
hist = model.fit(X_train_n, Y_train_n, 
                 validation_data=(X_test_n, Y_test_n),
                 epochs=10,
                 batch_size=128,
                 verbose=1)

#%% 08 평가 함수를 이용한 모델 확인
loss, acc = model.evaluate(X_test_n, Y_test_n, batch_size=32)

print('')
print('loss : ' + str(loss) )
print('accuray : ' + str(acc) )

#%% 09 학습 결과 확인. 그래프
fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()


#%% 09 모델 저장 및 정리 - HDF5파일로 모델 저장.
path = os.path.join(os.getcwd(), "00_part05_14_flask_ml\cifar10\model")
savefile = os.path.join(path, "my_model.h5"  )
print(path, savefile)
model.save(savefile)

print("시간(s) : {:.3f}s".format( time.time() - start ) )