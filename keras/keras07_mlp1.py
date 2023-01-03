import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]])  
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

print(x.shape)  # (2,10) // 2개 행의 10열
                # shape = 행렬의 구조 
print(y.shape)  # (10, )

x = x.T  
# T = 행과 열을 바꿈 // 데이터를 행으로 만들어야 하기 때문에 변경해줌
print(x.shape)  # (10,2) // 10행 2열

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가 예측
# 완벽한 데이터는 맞으나, 안에 들어있는 훈련된 데이터를 이용하였기 성능이 매우 좋게 나옴 때문에 예제일뿐, 실제로는 이런식으로 사용하지 않음!!
loss = model.evaluate(x, y)
# 실제로 할때는 훈련데이터와 평가데이터를 나누어서 구분해서 사용을 해야함.
print('loss : ', loss)

results = model.predict([[10, 1.4]])
print('[10, 1.4]의 예측값 :', results)


'''
loss :  0.20438604056835175
[10, 1.4]의 예측값 : [[20.426537]]

loss :  0.09501536935567856
[10, 1.4]의 예측값 : [[20.058996]]
'''
