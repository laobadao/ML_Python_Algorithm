# print("17//3=", 17//3)
#
# print("17%3=", 17%3)
#
# print("5**2=", 5**2)
#
# print("2**10=", 2**10)
#
# tax = 12.8/100
# price = 200.2
# print(price*tax)
# print(round(price*tax, 2))  # round(_,2) 第二个参数 保留两位小数

# 2017-10-09 numpy-- tile

import numpy as np

# 将[1, 5] 横向 重复 3 次，然后纵向重复 2 次 （纵，横）
newSet = np.tile([1, 5], (2, 3))

# print(newSet)

# [[1 5 1 5 1 5]
#  [1 5 1 5 1 5]]
diffMat = np.array([7, 3]) - np.array([2, 1])
print(diffMat)
print(diffMat ** 2)



