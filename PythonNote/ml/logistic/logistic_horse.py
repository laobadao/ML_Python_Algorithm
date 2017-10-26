# -*- coding:UTF-8 -*-
from sklearn.linear_model import LogisticRegression
import numpy as np
import random
from scipy.special import expit, logit

"""
函数说明:sigmoid函数

Parameters:
    inX - 数据
Returns:
    sigmoid函数
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-09-05
Note:
    ZJ studied in 2017-10-26
"""


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


"""
函数说明:改进的随机梯度上升算法

Parameters:
    dataMatrix - 数据数组
    classLabels - 数据标签
    numIter - 迭代次数
Returns:
    weights - 求得的回归系数数组(最优参数)
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-09-05
"""


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)  # 参数初始化										#存储每次更新的回归系数
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # 降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选取样本
            h = expit(sum(dataMatrix[randIndex] * weights))  # 选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h  # 计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]  # 更新回归系数
            del (dataIndex[randIndex])  # 删除已经使用的样本
    return weights  # 返回


"""
函数说明:梯度上升算法

Parameters:
    dataMatIn - 数据集
    classLabels - 数据标签
Returns:
    weights.getA() - 求得的权重数组(最优参数)
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-08-28
"""


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  # 转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()  # 转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.01  # 移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500  # 最大迭代次数
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = expit(dataMatrix * weights)  # 梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()  # 将矩阵转换为数组，并返回


"""
函数说明:使用Python写的Logistic分类器做预测

Parameters:
    无
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-09-05
"""


def colicTest():
    frTrain = open('horseColicTraining.txt')  # 打开训练集
    frTest = open('horseColicTest.txt')  # 打开测试集
    # 创建训练集 trainingSet list
    trainingSet = []
    # 创建类别标签 list trainingLabels
    trainingLabels = []
    for line in frTrain.readlines():
        # 去除空白符 使用 \t 切割 返回
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        # currLine[-1] 添加 最后一项 标签分类  到 trainingLabels
        trainingLabels.append(float(currLine[-1]))
    # np.array(trainingSet) 训练集   trainingLabels 分类标签   迭代次数 500
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)  # 使用改进的随即上升梯度训练
    errorCount = 0
    numTestVec = 0.0
    # 计算错误率 利用测试集
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec) * 100  # 错误率计算
    print("测试集错误率为: %.2f%%" % errorRate)


"""
函数说明:使用Python写的Logistic分类器做预测

Parameters:
    无
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-09-05
"""


def colicTest1():
    frTrain = open('horseColicTraining.txt')  # 打开训练集
    frTest = open('horseColicTest.txt')  # 打开测试集
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights = gradAscent(np.array(trainingSet), trainingLabels)  # 使用改进的随即上升梯度训练
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights[:, 0])) != int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec) * 100  # 错误率计算
    print("测试集错误率为: %.2f%%" % errorRate)


"""
函数说明:使用Sklearn构建Logistic回归分类器

Parameters:
    无
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-09-05
"""


def colicSklearn():
    frTrain = open('horseColicTraining.txt')  # 打开训练集
    frTest = open('horseColicTest.txt')  # 打开测试集
    trainingSet = []
    trainingLabels = []
    testSet = []
    testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    # liblinear 适用于小数据集，而 sag 和 saga 适用于大数据集因为速度更快。
    #  solver 算法选择 liblinear max_iter=10 迭代次数
    # classifier = LogisticRegression(solver='liblinear', max_iter=10).fit(trainingSet, trainingLabels)
    # solver='sag', max_iter=5000 适合数据量大
    classifier = LogisticRegression(solver='sag', max_iter=5000).fit(trainingSet, trainingLabels)
    test_accurcy = classifier.score(testSet, testLabels) * 100
    print('正确率:%f%%' % test_accurcy)


"""
函数说明:分类函数

Parameters:
    inX - 特征向量
    weights - 回归系数
Returns:
    分类结果
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-09-05
"""


def classifyVector(inX, weights):
    # θ^T * X  logistic_function() sigmoid() expit
    prob = expit(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def logistic_function(x):
    return .5 * (1 + np.tanh(.5 * x))


if __name__ == '__main__':
    # colicTest()
    # 35.82%
    # colicTest1()
    # 28.36%

    # 结论：
    # 当数据集较小时，我们使用梯度上升算法
    # 当数据集较大时，我们使用改进的随机梯度上升算法

    # sklearn logistic 
    colicSklearn()
    # 正确率:73.134328%

    # 1、Logistic回归的优缺点
    #
    # 优点：
    #
    # 实现简单，易于理解和实现；计算代价不高，速度很快，存储资源低。
    # 缺点：
    #
    # 容易欠拟合，分类精度可能不高。
    # 2、其他
    #
    # Logistic回归的目的是寻找一个非线性函数Sigmoid的最佳拟合参数，求解过程可以由最优化算法完成。
    # 改进的一些最优化算法，比如sag。它可以在新数据到来时就完成参数更新，而不需要重新读取整个数据集来进行批量处理。

    # 机器学习的一个重要问题就是如何处理缺失数据。
    # 这个问题没有标准答案，取决于实际应用中的需求。现有一些解决方案，每种方案都各有优缺点。
    # 我们需要根据数据的情况，这是Sklearn的参数，以期达到更好的分类效果。