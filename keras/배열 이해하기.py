from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 1.데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
y = np.array([[0,1,2,3,4,5,6,7,8,9]]) 
# (1,10) // 1행 10열
z = np.array([0,1,2,3,4,5,6,7,8,9])  
# (10 , ) // 10행 
t = np.array([[[0,1,2,3,4,5,6,7,8,9]]])
# (1,1,10) // 1층 1행 10열
q = np.array([[24,25],[26,25]])
# (2,2) // 2행 2열
b = np.array([[24,26],[28,30],[1,8]])
# (2,3) // 2행 3열
a = np.array([[[1,4],[2,5],[2,7],[4,5]]])
# (1,4,2) // 1층 4행 2열
tq = np.array([[[[1,4],[2,6]]]])
(1,1,2,2)


print(range(10))
print('y :',y.shape)
print('z :',z.shape)
print('t :',t.shape)
print('q :',q.shape)
print('b :',b.shape)
print('a :',a.shape)
print('tq :',tq.shape)
