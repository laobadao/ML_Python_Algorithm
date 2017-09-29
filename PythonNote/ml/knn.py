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
    inX - 用于分类的数据 （测试集）
    dataSet - 用于训练的数据 （训练集）
    labels - 用于分类的标签
    k - kNN 算法参数，选择距离最小的 k 个点
    
:return
    sortedClassCount[0][0] 分类结果
    
:date
    2017-09-29
"""
# def classify0():
