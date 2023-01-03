from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf

# 1.정제된 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# 2.모델구성
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(1))

# 3.컴파일, 훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=2,batch_size=2)
# batch_size=X는 데이터를 어떠한 형식으로 자를지 결정하는 것이다. 
# size를 전체 데이터 보다 크게 설정한 경우에 한해 1로 인식을 하고 계산을 한다.
# batch_size의 기본값은 32이다.

# 4.평가,예측
results = model.predict([6])
print('예측값 :' , results)