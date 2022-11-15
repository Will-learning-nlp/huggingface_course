import numpy as np

x = np.random.normal(1, 4, (3, 5))
y = np.argmax(x, axis=1
              )  # axis=1指定代表我要查找的最大元素在第1维中的索引值
print(x)
print(x.shape)
print(y)
print(y.shape)
# axis=-1 每一行
# axis=0 每一列
# axis=1 每一行


# [[0.33426754 0.53216384 0.57384213]
#  [0.88361917 0.89788344 0.86475191]]
# 以上 维度为(2,3)的二维数组：它有2个维度，
# 因此，它的轴有2个，分别为轴0（轴的长度为2）、轴1（轴的长度为3），

import numpy as np

a = np.random.random((2, 3))

print(a)


