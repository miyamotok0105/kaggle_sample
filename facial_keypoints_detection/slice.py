import numpy as np

y = [0,1,2,3,4,5,6,7,8,9]
print(y)

print("1個飛ばし")
print(y[0::2])
print(y[::2])

print("2個飛ばし")
print(y[::3])
print("3個飛ばし")
print(y[::4])

print("2個目から1個飛ばし")
print(y[1::2])
print("2個目から2個飛ばし")
print(y[1::3])


y = np.array(y).reshape(2,5)
print(y.shape)
print(y.flatten().shape)
print(y.flatten().reshape(1,10).shape)
print(y.flatten().reshape(1,10))
# print(y.transpose(1,0,2))