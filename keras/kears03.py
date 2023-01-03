import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

# 2.모델구성 y=wx+b

model = Sequential()
model.add(Dense(6, input_dim=1))
# 최초의 입력할 값을 넣음! 그 후 출력값의 갯수 // input,output
model.add(Dense(1))
# 처음에 출력된 값을 다시 사용하기 때문에 input값을 따로 넣지 않아도 됨!
# 마지막 모델링은 출력되는 값을 나타내는것

# 3.컴파일, 훈련
model.compile(loss='mae',optimizer='adam')
# mae : 예측값과 결과값의 평균을 나타낸 값
# adam :
model.fit(x,y,epochs=3)
# epochs는 몇번 돌릴지!! 즉, 몇번 계산을 할지에 대하여 정하는 문구이다.
# epochs의 갯수는 많이 할 수 있으나 그만큼 GPU를 잡아 먹는 값이 생김

# 4.평가, 예측
results = model.predict([6])
print('예측값 : ', results)

'''
예측값을 2개로 잡았을때
[[-0.24758148 -1.4171852 ]]

예측값을 1개로 3번 돌렸을때
예측값 :  [[7.1091614]]
'''
