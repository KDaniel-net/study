from keras.models import Sequential
from keras.layers import Dense
import numpy as np 
import tensorflow as tf

# 1.정제된 데이터
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 2, 3, 5, 4, 6])

# 2. 모델구성 
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(2))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=10,batch_size=7)

# 4.평가,예측
loss=model.evaluate(x,y)
# x와 y값의 평균 
print('loss : ',loss)
results=model.predict([6])
print('예측값 : ',results)

'''
예측값 :  [[7.2423053]]
'''