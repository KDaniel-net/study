import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
             [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
            # (3,10)
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
            # (10, )
x = x.T 
# x의 값을 3행 10을 10행 3열로 바꿈
# date.T에서 T는 데이터의 행과 열을 바꾸어줌

print(x.shape)
print(y.shape)

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3))
# input_dim=count date // 처음에 넣는 갯수를 넣음!!
# 3개의 항에서 데이터 값을 받아 오기 때문에 input_dim=3
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(1))
# 우리가 원하는 값은 하나이기 때문에 하나만 출력

# 3. 컴파일, 훈련
model.compile(loss='mae' , optimizer='adam')
model.fit(x,y,epochs=200,batch_size=1)

# 4.예측, 평가
loss = model.evaluate(x,y)
print('loss : ',loss)
results = model.predict([[10, 1.4, 0]])
# 대 괄호 꼭 잘 닫기
print('예측값 : ' ,results)

'''
loss :  0.659609317779541
예측값 :  [[18.697958]]
'''