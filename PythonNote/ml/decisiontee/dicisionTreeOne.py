from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
import pickle

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
    # labels = ['不放贷', '放贷']  # 分类属性 共两类
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 特征标签 将前4个值 给出特征具体含义
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


# 划分数据集的大原则是：将无序的数据变得更加有序。我们可以使用多种方法划分数据集，
# 但是每种方法都有各自的优缺点。组织杂乱无章数据的一种方法就是使用信息论度量信息，信息论
# 是量化处理信息的分支科学。我们可以在划分数据之前或之后使用信息论量化度量信息的内容。
# 在划分数据集之前之后信息发生的变化称为信息增益，知道如何计算信息增益，我们就可以
# 计算每个特征值划分数据集获得的信息增益，获得信息增益最高的特征就是最好的选择。

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
        # print("第%d个特征的增益为%.3f" % (i, infoGain))  # 打印每个特征的信息增益
        if infoGain > bestInfoGain:  # 两两对比找最大值 存储 索引
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


"""
函数说明:统计 classList 类标签 list中出现此处最多的元素(类标签)

Parameters:
    classList - 类标签列表
Returns:
    sortedClassCount[0][0] - 出现此处最多的元素(类标签) 降序排序后
Modify:
      2017-10-17 
"""


def majorityCnt(classList):
    # 遍历完所有特征时返回出现次数最多的类标签 当只有一列时 在这个例子中是没有用到这个方法的
    classCount = {}
    for vote in classList:  # 统计 classList 中每个元素出现的次数 创建字典 key value 存储
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 根据 value 值  key=operator.itemgetter(1) 降序排序 reverse=True
    # print(classCount.items()) # ([('no', 6), ('yes', 9)])
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 根据字典的值降序排序
    # print("sortedClassCount", sortedClassCount)
    return sortedClassCount[0][0]  # 返回classList中出现次数最多的元素


def findMaxLabels(classLabels):
    # 创建字典存数 标签 key  和 对应个数
    classDics = {}
    for itemKey in classLabels:
        if itemKey not in classDics.keys():
            classDics[itemKey] = 0
        classDics[itemKey] += 1
    sortedClass = sorted(classDics.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClass[0][0]


"""
函数说明:创建决策树

Parameters:
    dataSet - 训练数据集
    labels - 分类属性标签
    featLabels - 存储选择的最优特征标签  featLabels = [] 创建的空list  也可以不作为参数传进来
Returns:
    myTree - 决策树
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-25  
    
ZJ leaning in 2017- 10-17
"""


def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]  # 取分类标签(是否放贷:yes or no)
    # print(classList)
    # count() 计算 某个值在 classList 中的个数 取第一个值，
    # 若count 这个值在classList中的个数 与 len(classList)本身长度相同 则所有数据类别全部相同
    if classList.count(classList[0]) == len(classList):  # 如果类别完全相同则停止继续划分
        # print("classList[0]", classList[0])  # 用来返回 yes  或 no 也就是最终的分类
        return classList[0]
    # dataSet[0] 第一行数据 的 列数 如果只有一个特征
    # print("len(dataSet[0])", len(dataSet[0]))
    if len(dataSet[0]) == 1:  # 遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最优特征
    # ['年龄', '有工作', '有自己的房子', '信贷情况']  根据最优特征的索引 找到对应的 文字说明
    bestFeatLabel = labels[bestFeat]  # 最优特征的标签
    featLabels.append(bestFeatLabel)
    # print("featLabels.append", featLabels)
    myTree = {bestFeatLabel: {}}  # 根据最优特征的标签生成树 先加入根 root
    # print("myTree", myTree)
    del (labels[bestFeat])  # 删除已经使用特征标签
    featValues = [example[bestFeat] for example in dataSet]  # 得到训练集中所有最优特征的属性值
    # print("featValues", featValues)
    uniqueVals = set(featValues)  # 去掉重复的属性值
    # print("uniqueVals", uniqueVals)
    for featKey in uniqueVals:  # 遍历特征，创建决策树。递归遍历
        # splitDataSet(dataSet, bestFeat, value) 去掉 某一列的子数据集 子数据集中再进行决策树的创建
        # myTree[bestFeatLabel][featKey]  代表 bestFeatLabel 是个 key
        # myTree[bestFeatLabel] 是个 value  同时这个value 又是个字典 然后
        # myTree[bestFeatLabel][featKey]  value中的字典中的 featKey 给它进行赋值
        myTree[bestFeatLabel][featKey] = createTree(splitDataSet(dataSet, bestFeat, featKey), labels, featLabels)
        # print("bestFeatLabel ", bestFeatLabel)
        # print("myTree[bestFeatLabel][value]  ", myTree[bestFeatLabel])
    return myTree


"""
函数说明:获取决策树叶子结点的数目

Parameters:
    myTree - 决策树
Returns:
    numLeafs - 决策树的叶子结点的数目
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""


def getNumLeafs(myTree):
    # 初始化叶子数目 也就是最终分类的数目
    numLeafs = 0
    # python3中myTree.keys()返回的是dict_keys,不再是list,所以不能使用myTree.keys()[0]的方法获取结点属性，
    # 可以使用list(myTree.keys())[0]
    firstStr = next(iter(myTree))
    # print("firstStr", firstStr)
    secondDict = myTree[firstStr]  # 获取下一组字典
    for key in secondDict.keys():
        # print("type(secondDict[key]).__name__ ", type(secondDict[key]).__name__)
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])  # 递归 每一个字典检索一次叶子节点
        else:
            numLeafs += 1
    return numLeafs


"""
函数说明:获取决策树的层数

Parameters:
    myTree - 决策树
Returns:
    maxDepth - 决策树的层数
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""


def getTreeDepth(myTree):
    # 初始化决策树深度
    maxDepth = 0
    # python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，
    # 可以使用list(myTree.keys())[0]
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]  # 获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth  # 更新层数
    return maxDepth


"""
函数说明:绘制结点

Parameters:
    nodeTxt - 结点名
    centerPt - 文本位置
    parentPt - 标注的箭头位置
    nodeType - 结点格式
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")  # 定义箭头格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',  # 绘制结点
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)


"""
函数说明:标注有向边属性值

Parameters:
    cntrPt、parentPt - 用于计算标注位置
    txtString - 标注的内容
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 计算标注位置 两点之间中点的坐标值计算
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    # print("xMid", xMid)
    # print("yMid", yMid)
    createPlot.ax1.text(xMid - 0.01, yMid - 0.01, txtString, va="center", ha="center", rotation=0)


"""
函数说明:绘制决策树

Parameters:
    myTree - 决策树(字典)
    parentPt - 标注的内容
    nodeTxt - 结点名
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""


def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")  # 设置叶结点格式
    numLeafs = getNumLeafs(myTree)  # 获取决策树叶结点数目，决定了树的宽度
    # print("numLeafs:", numLeafs)
    depth = getTreeDepth(myTree)  # 获取决策树层数
    firstStr = next(iter(myTree))  # 下个字典
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)  # 中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)  # 标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘制结点
    secondDict = myTree[firstStr]  # 下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key], cntrPt, str(key))  # 不是叶结点，递归调用继续绘制
        else:  # 如果是叶结点，绘制叶结点，并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


"""
函数说明:创建绘制面板

Parameters:
    inTree - 决策树(字典)
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')  # 创建fig
    fig.clf()  # 清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))  # 获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))  # 获取决策树层数
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0  # x偏移
    plotTree(inTree, (0.5, 1.0), '')  # 绘制决策树
    plt.show()


"""
函数说明:使用决策树分类

Parameters:
    inputTree - 已经生成的决策树
    featLabels - 存储选择的最优特征标签 也就是 能把全部数据进行分类的  特征标签的list
    testVec - 测试数据列表，顺序对应最优特征标签
Returns:
    classLabel - 分类结果
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-25
Note: 
    ZJ learning in 2017-10-18
"""


def classify(inputTree, featLabels, testVec):
    #  对 inputTree 决策树 字典形式的 进行迭代 取值 iter() 迭代器 next() 返回迭代的下一个元素的值
    # 取出字典中第一个 根root 也就是第一个 最优特征值
    print('inputTree', inputTree)
    # {'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}
    firstFeatKey = next(iter(inputTree))
    # 有自己的房子
    print('firstFeatKey', firstFeatKey)
    # 获取 第一个特征值 key 对应的 字典
    secondDic = inputTree[firstFeatKey]
    # {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}
    print('secondDic', secondDic)
    # 在存储选择的最优特征标签 featLabels 中找到第一个 最优特征的索引
    featIndex = featLabels.index(firstFeatKey)
    print('featIndex', featIndex)
    # featIndex 0
    # 循环遍历 secondDic 下一个字典  中的 key
    for key in secondDic.keys():
        # 先判断测试数据列表 testVec 在中索引为 featIndex 也就是最优特征 的元素是否等于 key
        # key  有 0 和 1 两种
        print('testVec[featIndex]', testVec[featIndex])
        print('key', key)
        # 这哥判断相当于找到所属分支 要么直接找到叶子节点 或者 找到字典
        if testVec[featIndex] == key:
            # 判断 key 对应的value 的类型是否还是字典 判断方法就是对比 value的 .__name__ == 'dict'
            # 进行迭代 也就是上一层中 没有找到测试数据的分类结果 也就是没有找到叶子节点
            if type(secondDic[key]).__name__ == 'dict':
                classLabel = classify(secondDic[key], featLabels, testVec)
            # 如果 secondDic[key] 对应的 value 不是字典，那么就是 yes  或 no 的分类结果 叶子节点
            else:
                # secondDic[key] 对应的value 不是字典 而是 叶子节点 具体的分类标签 所以取值 直接赋值给classLabel
                # 取值结果肯定是 yes  或 no
                classLabel = secondDic[key]
    return classLabel


"""
函数说明:存储决策树

Parameters:
    inputTree - 已经生成的决策树
    filename - 决策树的存储文件名
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-25
"""


def storeTree(inputTree, filename):
    # 打开文件 open () 操作后 需要关闭 close()
    # Python引入了with语句来自动帮我们调用close()方法：
    # 'r' 是读文件 read 传入标识符'w'或者'wb'表示写文本文件或写二进制文件： wb write binary
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


"""
函数说明:读取决策树

Parameters:
    filename - 决策树的存储文件名
Returns:
    pickle.load(fr) - 决策树字典
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-25
Note: 
    ZJ learning in 2017-10-18
"""


def grabTree(filename):
    # 'rb' read binary 读取二进制文件
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    # shannonEnt = calcuShannonEnt(dataSet)
    # print(dataSet)
    # print(shannonEnt)  # 0.9709505944546686

    # subData = splitDataSet(dataSet, 2, 0)
    # print("subData:", subData)
    # print("最优特征索引值:" + str(chooseBestFeatureToSplit(dataSet)))
    # featLabels = []
    # myTree = createTree(dataSet, labels, featLabels)
    # print(myTree)
    # print(myTree)
    # createPlot(myTree)
    # classList = ['no', 'no', 'yes', 'yes', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
    # a = majorityCnt(classList)
    # print(a)
    # 使用决策树 进行分类
    # featLabels 在外层初始化 初始化 是空list 在进行迭代的过程中featLabels 是有新赋值的
    # featLabels  ['有自己的房子']  ['有自己的房子', '有工作'] ，
    # 也就是 最后 存储这两个 特征值使用这两个特征值就可以将数据全部分类完毕
    # featLabels = []
    # myTree = createTree(dataSet, labels, featLabels)
    # # featLabels ['有自己的房子', '有工作']
    # print("featLabels", featLabels)
    # # 根据 featLabels 只有两个特征值，所有测试数据给出两个值 就可以使用决策树得到最后分类结果
    # testVec = [1, 1]  # 没房 有工作
    # # featLabels ['有自己的房子', '有工作']
    # result = classify(myTree, featLabels, testVec)
    # if result == 'yes':
    #     print('放贷')
    # if result == 'no':
    #     print('不放贷')

    # 五、决策树的存储
    # featLabels = []
    # myTree = createTree(dataSet, labels, featLabels)
    # print(myTree)
    # storeTree(myTree, 'classifierStorage.txt')

    # 读取决策树
    tree = grabTree('classifierStorage.txt')
    print("tree", tree)
