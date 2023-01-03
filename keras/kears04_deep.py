from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf

# 1.정제된 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

# 2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))    
# input layer // 최초의 값을 한개 넣고, 3개로 출력
model.add(Dense(10))               
# hidden layer // 위에서 3개를 입력 받고 10개 출력
model.add(Dense(15))                
# hidden layer
model.add(Dense(1))                 
# output layer 결과 값으로 한개를 출력 

# 3.컴파일 훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=10)

# 4.평가,예측
results = model.predict([6])
print('예측값 : ', results)

'''
예측값 :  [[-0.30432373]]

예측값 :  [[4.059342]]
'''