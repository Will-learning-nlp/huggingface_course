# argmax：一句话概括，返回最大值的索引
import numpy as np
numbers = np.arange(5)
print(numbers)
print(numbers.argmax())
print(numbers.argmax(axis=0))

np.random.shuffle(numbers)
print(numbers)

print(numbers.argmax())
print(numbers.argmax(axis=0))

# 二维数组
numbers_2 = np.arange(6).reshape(2,3)
print(numbers_2)

print(numbers_2.argmax())
# argmax没有 轴参数时，默认将数组展平

# 设置轴为0，这是在列中比较，那么选出的是最大 行 索引

print(numbers_2.argmax(axis=0))

# 设置 轴 为 1， 在行中比较， 选出最大 列 索引

print(numbers_2.argmax(axis=1))

# 设置 轴为-1，与1的结果相同
print(numbers_2.argmax(axis=-1))
# “axis=-1”在Numpy中指“倒数第一个轴”，即最后一个轴；

# 三维数组
numbers_3 = np.arange(24).reshape(4,3,2)
print(numbers_3)
print(numbers_3.shape)

# 第0通道--最后1维的 第1个索引列
print("第0通道")
print(numbers_3[:,:,0])

# 第1通道
print("第一通道")
print(numbers_3[:,:,1])

print(np.argmax(numbers_3))  # 不指定轴，默认展平

print(np.argmax(numbers_3, axis=0))  # 指定0轴
print(np.argmax(numbers_3, axis=1))  # 指定1轴
print(np.argmax(numbers_3, axis=2))  # 指定2轴

# axis=2时，是上下两个图层的对应位置做比较
# 可以理解z轴方向比较，0通道在下，1通道在上。起始索引从0开始，1通道上对应位置都大于0通道，所以都是1.
# 返回的是通道数！！！
# 返回的是通道数！！！
# 返回的是通道数！！！
# 返回的是通道数！！！

print(np.argmax(numbers_3, axis=-1))  # 指定-1轴，和axis=2，相同，因为是最后一位







