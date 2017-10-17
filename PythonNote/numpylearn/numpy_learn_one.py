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

from numpy import *

# 将[1, 5] 横向 重复 3 次，然后纵向重复 2 次 （纵，横）
newSet = tile([1, 5], (2, 3))

# print(newSet)

# [[1 5 1 5 1 5]
#  [1 5 1 5 1 5]]
diffMat = array([7, 3]) - array([2, 1])
print(diffMat)
print(diffMat ** 2)

"""
learn numpy 

列表 List 用 [] 表示

date : 2017-10-10

"""

listTest = []
listTest.append(1)
print(listTest)
listTest.append(3.33)
listTest.append("string")
print(listTest)
listTest.remove(1)
print(listTest)

list2 = [2, 2.2, "2222"]
print(list2)

"""
字典 ：Dictionary  用 {} 表示
key/value键值对

"""

dicMap = {}
dicMap['key1'] = 'value1'
dicMap[2] = 8
print(dicMap)
dicMap2 = {2: 999, 'key2': 'value2'}
print(dicMap2)

"""
集合 Set   方法 set() {5, 6}

数学概念类似 集合之间 取 交集 并集 补集
"""
a = [1, 1, 1, 3, 3, 3, 3, 2, 2, 5, 6, 7]
set1 = set(a)
print(set1)

set2 = set([2, 3, 4])
print(set2)

set3 = {5, 6, 9}
print(set3)

print("set1 - set3 :", set1 - set3)
print("set1 |  set3 :", set1 | set3)
print("set1 & set3 :", set1 & set3)
print("set3 - set1 :", set3 - set1)

"""
控制语句 if else while for 
"""
x = 2
if x < 3:
    print("python is concise")

if x < 3:
    print("change line python is concise")
    x = x + 1
print(x)
x = 4
if x < 3:
    x += 1
elif x == 3:
    x += 0
else:
    x = 0

print(x)

for itemA in a:
    print("itemA", itemA)

for itemSet in set1:
    print("itemSet", itemSet)

for itemDic in dicMap:
    print("itemDic", itemDic), print("dicMap[itemDic]", dicMap[itemDic])

"""
列表推导式 简洁优雅的方式生成列表
"""

myConciseList = [item * 4 for item in a]
print(myConciseList)

# 复杂的写法创建新列表
myNewList = []

for item in a:
    myNewList.append(item * 3)

print(myNewList)

myVeryConciseList = [item * 2 for item in a if item > 2]

print(myVeryConciseList)

"""
NumPy 数组 矩阵

"""
a1 = array((2, 3, 4))
a2 = array((5, 6, 7))

print("a1:", a1)
print("a2:", a2)

print("a2 - a1:", a2 - a1)
print("a2 + a1:", a2 + a1)
print("a2 * a1:", a2 * a1)
print("每个元素平方：a1 ** 2：", a1 ** 2)

twoDimensionArray = array([[2, 3, 4],
                           [4, 5, 6]])
print("twoDimensionArray[0]:", twoDimensionArray[0])
print("twoDimensionArray[0][1]:", twoDimensionArray[0][1])
print("twoDimensionArray[0, 1]:", twoDimensionArray[0, 1])

"""
矩阵 MAT matrix
"""

newMat = mat([2, 3, 4])
print(newMat)
print(newMat[0, 2])

newMat1 = newMat.T
print("转置 newMat.T: ", newMat1)

print("newMat * newMat1=", newMat * newMat1)
# shape() 查看矩阵维度 （行数，列数）
print("shape(newMat):", shape(newMat))

print("矩阵行数：", newMat.shape[0])
print("矩阵列数：", newMat.shape[1])

# 矩阵中每个元素对应相乘
print("newMat:", newMat)
print("newMat1:", newMat1)
print("multiply(newMat, newMat1);", multiply(newMat, newMat1))

# sort() 排序 占原有存储空间  argsort() 返回的是排序后的 元素的 索引坐标
# sort(a, axis=-1, kind='quicksort', order=None):
disMat = mat([9, 2, 0, 3])
print("disMat:", disMat)
print("sort(disMat):", sort(disMat))
disMat = mat([9, 2, 0, 3])
print("argsort(disMat):", argsort(disMat))

# 求矩阵均值 mean() 相加/个数 （9+2+0+3）/4 =3.5

print("disMat.mean:", disMat.mean())

disMat = mat([[1, 2, 3, 4, 5, 6, 7, 8, 9],
              [9, 8, 7, 6, 5, 4, 3, 2, 1]])
print("disMat.mean:", disMat.mean())

# 多维数组 取出某一行  通过 行号和 ： 来完成
print("取出第二行，索引为 1 ,disMat[1, :] =", disMat[1, :])

# 矩阵 中 某一行 中 指定范围 某一行共 10 个 中间5个
print("取出第二行，索引为 1 到 4 包含1 不含4 ,从 第二个 到第5个 disMat[1, :] =", disMat[1, 1:4])

# [:,1] [: 1] 从矩阵中取值的区别  中间少个逗号

twoDimenMat = mat([[1, 2, 3],
                   [11, 21, 31],
                   [111, 211, 311],
                   [12, 23, 33],
                   [13, 23, 33],
                   [112, 212, 312],
                   [1, 2, 3],
                   [1, 2, 3],
                   [1, 2, 3],
                   [123, 223, 323],
                   ])
#  应该取的是第一列  取的是 twoDimenMat 10 * 3 维矩阵 取出的是 10 *1 且元素都是1
print("twoDimenMat[:, 0]", twoDimenMat[:, 0])
# 相当于取的是第一行数据 twoDimenMat[:1] 相当于是个取值范围 取的是  0行 到1 行的值 1且不包含1
print("twoDimenMat[:1]", twoDimenMat[:1])
# twoDimenMat[1] 取的值具体向量 第二行索引为1 的数据
print("twoDimenMat[1]", twoDimenMat[1])
# 取值 是 0到 5行 不包含5 就是 0 到 4 共5行 第二列的数据
print("twoDimenMat[:5, 1]", twoDimenMat[:5, 1])
# 取值区间 索引为 3 4 5 不包括6 第二列的数据
print("twoDimenMat[:5, 1]", twoDimenMat[3:6, 1])

threeMat = mat([[9, 9, 9], [1, 2, 3], [0.1, 0.1, 0.2]])
print("threeMat.min(0)", threeMat.min(0))
print("threeMat.max(0)", threeMat.max(0))

dataSet1 = [0, 1, 2, 3, 4, 5]

print("dataSet1[:1]", dataSet1[:1])  # : 表示区间 [:1] 对于 list 来说 代表 取 0到1 之间的数据 不包含1
print("dataSet1[1+1:]", dataSet1[1 + 1:])  # [1+1:] 代表 取 索引2 往后的数据 包含2

"""
append() and  extend()  区别

"""

a = [1, 2, 3]
b = [4, 5, 6]
a.append(b)
print("a.append(b):", a)  # [1, 2, 3, [4, 5, 6]]
a = [1, 2, 3]
a.extend(b)
print("a.extend(b):", a)  # [1, 2, 3, 4, 5, 6]

labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
bestFeat = 2  # 索引2 要删除索引为2 的元素
del (labels[bestFeat])
print(labels)  # ['年龄', '有工作', '信贷情况']

key1 = "iii"
dics = {key1: {}}
dics[key1][value] = {"ooo": {}}
print(dics)  # {'iii': {'ooo': {}}}
