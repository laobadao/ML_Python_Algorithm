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

myVeryConciseList = [item*2 for item in a if item > 2]

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
print("每个元素平方：a1 ** 2：", a1 **2)


