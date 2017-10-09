# -*- coding: UTF-8 -*-
"""
k-近邻算法步骤如下：

计算已知类别数据集中的点与当前点之间的距离；
按照距离递增次序排序；
选取与当前点距离最小的k个点；
确定前k个点所在类别的出现频率；
返回前k个点所出现频率最高的类别作为当前点的预测分类。
"""
import numpy as np
import operator
import time
"""
函数说明：创建数据集

:parameter 无
:returns
    group - 数据集
    labels   -分类标签
:date 
   2017-9-29
"""


def createDataSet():
    # 四组二维特征， [ 打头镜头, 接吻镜头 ]
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = ['爱情片', '爱情片', '武侠片', '武侠片']
    return group, labels


if __name__ == '__main__':
    group, labels = createDataSet()
    print(group)
    print(labels)

"""
函数说明：KNN 算法，分类器

:parameter
    testX - 用于分类的数据 （测试集）
    trainSet - 用于训练的数据 （训练集）
    labels - 用于分类的标签
    k - kNN 算法参数，选择距离最小的 k 个点
    
:return
    sortedClassCount[0][0] 分类结果
    
:date
    2017-09-29
"""

"""

note：
    补充 numpy 函数 
    1. 创建二维数组
    2. 求数组行数
    3. 求数组列数 
"""
# if __name__ == '__main__':
#     testArray = np.array([[1., 2.], [2., 3.], [3., 4.], [4., 5.], [5., 6.]])
#     row = testArray.shape[0]
#     column = testArray.shape[1]
#     print("行：", row, "  列：", column)

def classify0(testX, trainSet, labels, k):
    # numpy函数 shape[0] 返回 trainSet 的行数
    trainSetSize = trainSet.shape[0]
    # 在列向量方向上重复 testX 共1次(横向)，行向量方向上重复 testX 共dataSetSize次(纵向)
    #  np.tile(testX, (trainSetSize, 1)) 是为了 将测试数据  test = [101, 20] 构造成 和 训练集相同的数据形式
    #  这样可以使用 欧式距离公式  多个维度之间的距离公式
    # 在欧几里得空间中，点x = (x1, ..., xn)和 y = (y1, ..., yn)之间的欧氏距离为

    diffMat = np.tile(testX, (trainSetSize, 1)) - trainSet
    print("testX:", testX)
    print("trainSet:", trainSet)
    print("diffMat:", diffMat)
    # 二维特征相减后平方
    sqDiffMat = diffMat ** 2
    print("sqDiffMat:", sqDiffMat)
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加 axis＝0表示按列相加，axis＝1表示按照行的方向相加
    sqDistances = sqDiffMat.sum(axis=1)
    print("sqDistances:", sqDistances)
    # 开方，计算出距离
    distances = sqDistances ** 0.5
    print("distances:", distances)
    # distances: [ 128.68954892  118.22436297   16.55294536   18.43908891]
    # 返回distances中元素从小到大排序后的索引值  因为函数返回的排序后元素在原array中的下标
    # 上面四个数 在array中的下标 索引是 128.68954892 - 0  ,118.22436297- 1 ,16.55294536 - 2 ,18.43908891- 3
    # 排序后是 16.55294536  18.43908891 118.22436297 128.68954892  所以 索引 下标的排序就是 2 3 1 0
    sortedDistIndices = distances.argsort()
    print("sortedDistIndices", sortedDistIndices)
    # sortedDistIndices [2 3 1 0]
    # 定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        print("sortedDistIndices[i]:", sortedDistIndices[i])
        voteIlabel = labels[sortedDistIndices[i]]
        print("voteIlabel:", voteIlabel)
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # python3中用items()替换python2中的iteritems()
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    print("classCount:", classCount)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]


if __name__ == '__main__':
    # 创建数据集
    group, labels = createDataSet()
    # 测试集
    test = [101, 20]
    # kNN分类
    test_class = classify0(test, group, labels, 3)
    # 打印分类结果
    print(test_class)

