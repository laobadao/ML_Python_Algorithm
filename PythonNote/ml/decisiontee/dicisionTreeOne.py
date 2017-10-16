from math import log

"""
函数说明:创建测试数据集

决策树

Parameters:
    无
Returns:
    dataSet - 数据集
    labels - 分类属性
Modify:
    2017-10-16
    
年龄：0代表青年，1代表中年，2代表老年；
有工作：0代表否，1代表是；
有自己的房子：0代表否，1代表是；
信贷情况：0代表一般，1代表好，2代表非常好；
类别(是否给贷款)：no代表否，yes代表是。
    
"""


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集 是 list  不是矩阵
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['不放贷', '放贷']  # 分类属性 共两类

    return dataSet, labels  # 返回数据集和分类属性


"""
函数说明：计算给定数据集的经验熵（香农熵）
Parameters:
    dataSet - 数据集
Returns:
    shannonEnt - 经验熵(香农熵)
Modify:
    2017-10-16
"""


def calcuShannonEnt(dataSet):
    # 返回数据集的行数 len(dataSet) list 的行数用len () 方法 ，矩阵的行数 用shape[0]
    rows = len(dataSet)
    # 保存每个标签(Label)出现次数的字典 {'yes':10,'no':5}
    countLables = {}
    # for 对每组特征向量进行统计
    for itemVector in dataSet:
        # 提取标签(Label)信息 每一行 （每组特征向量 ）用[-1] 取最后一个值
        currentLabel = itemVector[-1]
        # 如果标签(Label)没有放入统计次数的字典,添加进去 （经常这样写，当 key 不存在时，先添加该 key ）
        if currentLabel not in countLables.keys():
            countLables[currentLabel] = 0
        countLables[currentLabel] += 1  # Label计数 累计加一
    # 初始化经验熵(香农熵)
    shannonEnt = 0.0
    # for 计算香农熵 利用公式
    for eachLabel in countLables:
        # 计算选择该标签(Label)的概率
        p = float(countLables[eachLabel]) / rows
        # 利用公式计算
        shannonEnt -= p * log(p, 2)
        # 返回经验熵(香农熵)
    return shannonEnt


"""
函数说明:按照给定特征划分数据集

Parameters:
    dataSet - 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值
Returns:
    retDataSet - 将符合条件的添加到返回的数据集
Modify:
    2017-10-16
"""


# 取出 二维数组dataSet中 ，某一列 axis 列 ，中 值 等于 value 的
def splitDataSet(dataSet, axis, value):
    retDataSet = []  # 创建返回的数据集列表  不包含 该列值等于value的
    for featVec in dataSet:  # 遍历数据集 先取出每一行数据
        if featVec[axis] == value:
            # 去掉 axis 特征 所在列
            reducedFeatVec = featVec[:axis]  # 先存 axis 索引前的数据
            # print("reducedFeatVec", reducedFeatVec)
            reducedFeatVec.extend(featVec[axis + 1:])  # 再存 axis 索引后的数据 将符合条件的添加到返回的数据集
            # print("featVec[axis + 1:]", featVec[axis + 1:])
            # print("reducedFeatVec", reducedFeatVec)
            retDataSet.append(reducedFeatVec)
    return retDataSet


"""
函数说明:选择最优特征

Parameters:
    dataSet - 数据集
Returns:
    bestFeature - 信息增益最大的(最优)特征的索引值

Modify:
    2017-10-16
"""


def chooseBestFeatureToSplit(dataSet):
    # 特征数量 最后一项 yes / or 是 最终结果 不是特征 len(dataSet[0]) 总列数
    numFeatures = len(dataSet[0]) - 1
    # 计算数据集的香农熵 H(D)
    baseEnt = calcuShannonEnt(dataSet)
    # 信息增益 初始化信息增益值
    bestInfoGain = 0.0
    # 最优特征的索引值 也就是 信息增益值 最大的 初始化为 -1
    bestFeature = -1
    # for 遍历所有特征 numFeatures 是 int 值 for  i++ 记得加 range
    for i in range(numFeatures):
        # 获取dataSet的第i个所有特征 也就是将 dataset 的 除去最后一列的 其他每一列 的数据取出来
        featList = [example[i] for example in dataSet]
        # 创建set集合{},元素不可重复 将取出的数据 通过转化为set()集合 去掉重复数据 如 [1,1,1,2] -set()-[1,2]
        uniqueFeatures = set(featList)
        # 经验条件熵 初始化每一个特征的经验条件熵
        newEachFeaEnt = 0.0
        # 计算信息增益 每一个特征的 。得先划分子集
        for value in uniqueFeatures:
            # 第 i 列的数据集中，value 值在该特征中 所占比例，也就是 比如 uniqueFeatures {1,2,3}
            # dataSet 中 对应该列的值 value  所占的比例 [1,1,1,2,2,3] 1 占比3/6  ，2 占比 2/6 ,3 占比 1/6
            # subDataSet划分后的子集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算子集的概率 其实就是先默认选一个特征为root 特征
            prob = len(subDataSet) / float(len(dataSet))
            # 根据公式计算经验条件熵 [5/15 H(D1) + 5/15 H(D2)+5/15 H(D3) ]
            newEachFeaEnt += prob * calcuShannonEnt(subDataSet)
            # 信息增益 g(D,A) = H(D) - H(D|A)
        infoGain = baseEnt - newEachFeaEnt
        print("第%d个特征的增益为%.3f" % (i, infoGain))  # 打印每个特征的信息增益
        if infoGain > bestInfoGain:  # 两两对比找最大值 存储 索引
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    # shannonEnt = calcuShannonEnt(dataSet)
    # print(dataSet)
    # print(shannonEnt)  # 0.9709505944546686

    # subData = splitDataSet(dataSet, 2, 0)
    # print("subData:", subData)
    print("最优特征索引值:" + str(chooseBestFeatureToSplit(dataSet)))
