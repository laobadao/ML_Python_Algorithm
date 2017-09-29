"""
 题目 ：
 给定一个0、1矩阵，求该矩阵内部的满足以下条件的最大的矩形的左上角所在的行和列，
 以及该矩形的长和宽。该矩形满足边上全为1，内部可以为0，可以为1，不做限定（假设所有输入有且仅有一个矩形满足条件）。

  输入描述：
 首先输入矩阵行和列 m n，接下来输入0,1矩阵 Mat[ m ][ n ]

 输出描述：
 四个数字，依次代表满足条件矩形左上角所在的行数和列数，以及该矩形的高（垂直长度）和宽（水平长度）

 样例输入：
 6 9

0 0 1 1 0 0 0 0 1
0 1 1 1 1 1 1 1 1
0 1 1 0 1 0 1 0 1
1 0 1 0 0 0 0 0 1
0 0 1 1 1 1 1 1 1
1 0 0 0 1 0 1 0 0

样例输出：
左上角 行数：2  列数 ：3 矩形的高（垂直长度）： 4 宽（水平长度） ：7

"""
from numpy import *

M = array([[0, 0, 1, 1, 0, 0, 0, 0, 1],
           [0, 1, 1, 1, 1, 1, 1, 1, 1],
           [0, 1, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 0, 0, 0, 0, 1],
           [0, 0, 1, 1, 1, 1, 1, 1, 1],
           [1, 0, 0, 0, 1, 0, 1, 0, 0], ])

left_up_points = []   # 左上 list
left_down_points = []   # 左下 list
right_up_points = []    # 右上 list
right_down_points = []  # 右下 list

# 矩阵的行
m = M.shape[0]

# 矩阵的列
n = M.shape[1]

# 循环遍历寻找左上角，也就是这个点是 1 且下方和右方的点 也是 1 ,然后将这个点的坐标加到 left_up_points list 中
# 行数从上往下 0 —— m-1 列数从左往右 0 —— n-1
for i in range(m - 1):
    for j in range(n - 1):
        if M[i, j] == M[i, j+1] == M[i+1, j] == 1:
            left_up_points.append((i, j))

# 循环遍历寻找右上角，也就是这个点和这个点的下方以及左边一点都是 1 ，然后加到 right_up_points list 中
# 行数 从 上往下 0 —— m-1 列数 从左往右 1 — n 从第一行第二列 坐标（ 0,1 ）
#  注意： 当 矩阵满足条件是，右上角 的坐标 查询最快的话 也是 （0,1） 也就是 列数 纵坐标肯定大于1
for i in range(m - 1):
    for j in range(1, n):
        if M[i, j] == M[i, j-1] == M[i + 1, j] == 1:
            right_up_points.append((i, j))

# 循环遍历左下角，也就是这个点的右边以及上边 一点是1 ，。。。
# 但这个点不可能是（0，n -1）( m-1 ,n -1)
for i in range(1, m):
    for j in range(n - 1):
        if M[i, j] == M[i-1, j] == M[i, j+1] == 1:
            left_down_points.append((i, j))

# 循环遍历右下角 ，也就是这个点以及点的上方和左边都是 1

for i in range(1, m):
    for j in range(1, n):
        if M[i, j] == M[i-1, j] == M[i, j-1] == 1:
            right_down_points.append((i, j))

# print(M)
# print(left_up_points)
# print(list(range(3)))  # 不包含 最后一个 小于3
# print(list(range(1, 3)))


















