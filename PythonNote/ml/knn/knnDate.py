# -*- coding: UTF-8 -*-
import numpy as np

"""
函数说明:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力

Parameters:
    filename - 文件名
Returns:
    returnMat - 特征矩阵
    classLabelVector - 分类Label向量

Modify:
    2017-10-09
"""


def file2Matrix(fileName):
    # 打开文件
    fr = open(fileName)
    # 读取文件所有内容
    arrayOLines = fr.readlines()
    # print("arrayOLines", arrayOLines)  ['40920\t8.326976\t0.953952\tlargeDoses\n',
    # '14488\t7.153469\t1.673904\tsmallDoses\n', '26052\t1.441871\t0.805124\tdidntLike\n', 得到文件行数
    numberOfLines = len(arrayOLines)
    # 返回的 NumPy 矩阵,解析完成的数据:numberOfLines行,3列  np.zeros  m×n 的double类零矩阵 1.  0.  代表double 类
    returnMat = np.zeros((numberOfLines, 3))
    # print("returnMat", returnMat)
    # 返回的分类标签向量
    classLabelVector = []
    # 行的索引值
    index = 0
    # print("arrayOLines:", arrayOLines)
    for line in arrayOLines:
        # s.strip(rm)，当rm空时,默认删除 字符串 开头和结尾的 空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine = line.split('\t')
        # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index, :] = listFromLine[0:3]
        # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector


"""
函数说明:main函数

Parameters:
    无
Returns:
    无
"""

if __name__ == '__main__':
    # 打开的文件名
    fileName = "datingTestSet.txt"
    datingMat, datingLabels = file2Matrix(fileName)
    print(datingMat)
    print(datingLabels)
