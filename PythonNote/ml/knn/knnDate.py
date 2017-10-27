# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import operator

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


def classify0(testX, trainSet, labels, k):
    # numpy函数 shape[0] 返回 trainSet 的行数
    trainSetSize = trainSet.shape[0]
    # 在列向量方向上重复 testX 共1次(横向)，行向量方向上重复 testX 共dataSetSize次(纵向)
    #  np.tile(testX, (trainSetSize, 1)) 是为了 将测试数据  test = [101, 20] 构造成 和 训练集相同的数据形式
    #  这样可以使用 欧式距离公式  多个维度之间的距离公式
    # 在欧几里得空间中，点x = (x1, ..., xn)和 y = (y1, ..., yn)之间的欧氏距离为
    diffMat = np.tile(testX, (trainSetSize, 1)) - trainSet
    # 二维特征相减后平方
    sqDiffMat = diffMat ** 2
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加 axis＝0表示按列相加，axis＝1表示按照行的方向相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方，计算出距离
    distances = sqDistances ** 0.5
    # distances: [ 128.68954892  118.22436297   16.55294536   18.43908891]
    # 返回distances中元素从小到大排序后的索引值  因为函数返回的排序后元素在原array中的下标
    # 上面四个数 在array中的下标 索引是 128.68954892 - 0  ,118.22436297- 1 ,16.55294536 - 2 ,18.43908891- 3
    # 排序后是 16.55294536  18.43908891 118.22436297 128.68954892  所以 索引 下标的排序就是 2 3 1 0
    sortedDistIndices = distances.argsort()
    # print("sortedDistIndices", sortedDistIndices)
    # sortedDistIndices [2 3 1 0]
    # 定一个记录类别次数的字典  字典 类似于 map 存储 键值对
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数  比如 第一次 加入 武侠片时 先取之前存过的 武侠片的值 没有则默认 0 classCount.get(voteIlabel, 0)
        # 也就是 2 对应的 武侠片 先存到 classCount 字典中 然后 累加次数
        # 爱情片 同理 第一次 字典中没有 因为是空的 或者 只有武侠 或只有爱情 然后默认0 再加1
        # 循环到 i =1 取出 索引 3 的值 对应的 labels 是武侠 再取字典之前的1 累加 2
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # python3中用items()替换python2中的iteritems()
    # key=operator.itemgetter(1)根据字典的值进行排序  在这个字典中，爱情片是键，1 是值
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典  从大到小排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别  前三个值里面 武侠片出现两次 分类为 武侠片
    return sortedClassCount[0][0]


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
    # returnMat 先构造成 numberOfLines * 3 矩阵
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
        # print("line:", line)
        # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine = line.split('\t')
        # print("listFromLine:", listFromLine)
        # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        # listFromLine[0:3]  index= 0 从 第0 行开始取 取所有行，每一行 取前三列 数据
        #  赋值给  returnMat[index, :] 第 index 行 所有列
        returnMat[index, :] = listFromLine[0:3]
        # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        # 判断 最后一列 字符串 存 labels 标签  list[-1] -1 代表最后一个元素
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector


"""
函数说明:对数据进行归一化

Parameters:
    dataSet - 特征矩阵
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值

Modify:
    2017-10-12
    
    公式 ： newValues = (oldValues - minValues)/(max - min)
"""


def normData(dataSet):
    # 获取数据最小值 和最大值
    minVals = dataSet.min(0)
    # [ 0.        0.        0.001156]
    maxVals = dataSet.max(0)
    # [  9.12730000e+04   2.09193490e+01   1.69551700e+00]
    # print("maxVals", maxVals)
    # 计算最大最小值的区间
    ranges = maxVals - minVals
    # 创建和 dataSet 一样行列数的0 矩阵
    normDataSet = np.zeros(np.shape(dataSet))
    # 返回 dataSet 的行数
    m = dataSet.shape[0]
    # 原始矩阵 减去 最小矩阵 公式上半部分
    # print("np.tile(minVals, (m, 1))", np.tile(minVals, (m, 1)))
    # 需要注意的是，特征值矩阵有1000×3 个值，
    # 而minVals和range的值都为1×3。为了解决这个问题，我们使用NumPy库中tile()
    # 函数将变量内容复制成输入矩阵同样大小的矩阵，
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # print("normDataSet:", normDataSet)
    # 除以公式下半部分 最大值和最小值的差 得到归一化数据
    # print("np.tile(ranges, (m, 1)", np.tile(ranges, (m, 1)))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    # print("normDataSet:", normDataSet)
    return normDataSet, ranges, minVals


"""
函数说明:分类器测试函数

Parameters:
    无
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值

Modify:
    2017-10-12
"""


def datingClassTest():
    # 打开的文件名
    fileName1 = "datingTestSet.txt"
    # 将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    datingDataMat, datingLabels = file2Matrix(fileName1)
    # 取所有数据的百分之十
    hoRatio = 0.10
    # 数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minVals = normData(datingDataMat)
    # 获得normMat的行数
    m = normMat.shape[0]
    # 百分之十的测试数据的个数 前 m 行数据
    numTestVecs = int(m * hoRatio)
    # 分类错误计数
    errorCount = 0.0

    # 取 10% 的数据然后判断 每一行是一个数据 ，将每一行 作为测试集 放到 所有剩余 90% 的训练集中测试
    for i in range(numTestVecs):
        # 前 numTestVecs 100个数据作为测试集,后 m - numTestVecs 900 个数据作为训练集
        #  numTestVecs:m 是为了取区间用的 所有后面要加上：m --- datingLabels 是list
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)
        print("分类器分类结果:%d\t训练集中真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
            print("第 i %d 个数据分类错了" % i)
    print("错误率:%f%%" % (errorCount / float(numTestVecs) * 100))


"""
函数说明 ：可视化数据

Parameters:
    datingDataMat - 特征矩阵
    datingLabels - 分类标签
    
Date:2017-10-12

"""


def showData(datingDataMat, datingLabels):
    # 设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 将fig画布分隔成共 2行2列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))
    # 标签的个数
    # numberOfLabels = len(datingLabels)
    # 循环遍历 根据 标签中元素数据 1  2  3 判断 给定不同的显示颜色 创建list  用来存储颜色数据
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=LabelsColors, s=15, alpha=.5)
    # 设置标题、x 轴，y 轴 的文字 字体 颜色 等显示
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示可视化图
    plt.show()


"""
函数说明:通过输入一个人的三维特征,进行分类输出

Parameters:
    无
Returns:
    无
Modify:
    2017-10-12
"""


def classifyPerson():
    # 输出结果
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    # 三维特征用户输入 input() 根据 用户数据输入 得到实际数据然后执行后续运算
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    # 打开的文件名
    filename = "datingTestSet.txt"
    # 打开并处理数据
    datingDataMat, datingLabels = file2Matrix(filename)
    # 训练集归一化
    normMat, ranges, minVals = normData(datingDataMat)
    # 生成NumPy数组,测试集
    inArr = np.array([precentTats, ffMiles, iceCream])
    # 测试集归一化
    norminArr = (inArr - minVals) / ranges
    # 进行分类器算法分类 返回测试结果
    classifyResult = classify0(norminArr, normMat, datingLabels, 3)
    print("你可能 %s 这个人" % resultList[classifyResult-1])


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
    normDataSet, ranges, minVals = normData(datingMat)
    # print("normDataSet", normDataSet)
    # print("ranges", ranges)
    # print("minVals", minVals)
    # 测试分类器算法 KNN
    # datingClassTest()
    # 最后：构建可用的约会系统
    # classifyPerson()



# print(datingMat)
# print(datingLabels)
# showData(datingMat, datingLabels)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# # 取 矩阵中所有行中 索引为 1和2  也就是 第二和第三列的 数据 将其可视化，单没有对点 进行颜色的区分
# # ax.scatter(datingMat[:, 1], datingMat[:, 2])
# # 将点 根据  datingLabels 中 1 2 3 不同点 对应的标签不同 进行颜色区分
# # 利用变量datingLabels存储的类标签属性，在散点图上绘制了色彩不等、尺寸不同的点。
# ax.scatter(datingMat[:, 1], datingMat[:, 2], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
# plt.show()
