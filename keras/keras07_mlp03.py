from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 1.데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]])

x = x.T
print(x.shape)
y = y.T
print(y.shape)

# 2.모델구성
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))
# Y값이 2개를 구해야 함으로 인해서 마지막 2개를 출력

# 3. 컴파일,훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=200,batch_size=1)

# 4.평가,예측
loss = model.evaluate(x,y)
print('loss 의 값 : ', loss )

results = model.predict([[9, 30, 210]])
print('[9,30,210] 의 예측값 : ', results)

'''
loss 의 값 :  1.472096562385559
[9,30,210]의 예측값 :  [[ 7.7653484 1.766771 ]]

loss 의 값 :  0.2531551420688629
[9,30,210] 의 예측값 :  [[9.68451   1.3590771]]
'''