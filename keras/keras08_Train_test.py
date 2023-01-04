from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 1.데이터
x_train = np.array([1, 2, 3, 4, 5, 6, 7])       # (7, )
x_test = np.array([8, 9, 10])                   # (3, )
y_train = np.array(range(7))                    # (7, )  [0,1,2,3,4,5,6] // range(7) : 0 ~ 7-1
y_test = np.array(range(7, 10))                 # (3, )  [7,8,9]  // range(7,10) : 7 ~ 10-1

# 2.모델
model = Sequential()
model.add(Dense(6,input_dim=1))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(1))

# 3.컴파일,훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss 값 : ' , loss)

result = model.predict([[9]])
print('[[9]]의 예측값 : ' , result)

'''
loss 값 :  0.021538415923714638
[[9]]의 예측값 :  [[7.978463]]
'''