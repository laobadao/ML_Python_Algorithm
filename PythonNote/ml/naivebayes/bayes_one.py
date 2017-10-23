import numpy as np
from functools import reduce

"""
函数说明:创建实验样本
侮辱类和非侮辱类，使用1和0分别表示。
Parameters:
    无
Returns:
    postingList - 实验样本切分的词条
    classVec - 类别标签向量
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-11
Note:
    ZJ studied in 2017-10-19.
"""


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇 语句，0代表不是
    return postingList, classVec


"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

Parameters:
    dataSet - 整理的样本数据集
Returns:
    vocabSet - 返回不重复的词条列表，也就是词汇表
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-11
"""


def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 集合之间 取并集
        # print("vocabSet: \n ", vocabSet)
    return list(vocabSet)


# def createVocabSet(postingList):
#     vocabSet = set([])
#     for sentence in postingList:
#         vocabSet = vocabSet | set(sentence)
#     return vocabSet


"""
函数说明:根据vocabList词汇表，将 inputSet 向量化，向量的每个元素为1或0
        inputSet 是 之前 创建的 postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        中的每一行，每一个向量 如  ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'] 
        将这一行中的 每一个单词 元素  向量化，向量的每个元素为1或0

Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量,词集模型
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-11
Note:
    ZJ studied in 2017-10-19.
"""


def setOfWords2Vec(vocabList, inputSet):
    # 先根据 vocabList 词汇表 的长度，创建一个 同样长度 的  向量 且元素都为 0
    # 比如  vocabList 的长度是10 ，然后创建长度是 10 的向量 [ 0,0,0,0,0,0,0,0,0]
    #  {'help', 'take', 'stupid', 'so', 'him', 'not', 'cute', 'dalmation', 'stop', 'please',}
    # inputSet  的长度是 4 {'stop'，'so'，'stupid'，'please'}
    #  那么 返回的 向量就是 [0,0,1,1,0,0,0,0,1,1 ] 其中 1 就代表上面4个 单词 包含在 词汇表中，
    # 并且上面那句话没有词汇表中标志为 0 的 那些单词
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:  # 遍历每个词条 单词
        if word in vocabList:
            # 如果词条存在于词汇表中，
            # 找到 vocabList.index(word) 此单词 在词汇表中的 索引位置 给returnVec 该索引位置 置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec  # 返回文档向量


"""
函数说明:朴素贝叶斯分类器训练函数

Parameters:
    trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
    p0Vect - 侮辱类的条件概率数组
    p1Vect - 非侮辱类的条件概率数组
    pAbusive - 文档属于侮辱类的概率
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-12
"""


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 计算训练的文档数目 行
    numWords = len(trainMatrix[0])  # 计算每篇文档的词条数 列
    #  trainCategory list 中 只有 0 或1  sum 求和 则算出 包含 1 侮辱性词汇语句的 个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 文档属于侮辱类的概率
    p0Num = np.zeros(numWords)  # [0,0,...,0,0]
    p1Num = np.zeros(numWords)  # 创建numpy.zeros数组,词条出现数初始化为0 [0,0,...,0,0]
    p0Denom = 0.0
    p1Denom = 0.0  # 分母初始化为0
    for i in range(numTrainDocs):
        # trainCategory = [0, 1, 0, 1, 0, 1]
        # trainMatrix[i]
        #  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        if trainCategory[i] == 1:  # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            # 若 trainMatrix[i] 这个 向量 数据的 标签 label 是 1  则将其加入进 p1Num 一维数组 累加
            p1Num += trainMatrix[i]
            # sum(trainMatrix[i]) 向量求和 累加 所有是 1 的总和
            p1Denom += sum(trainMatrix[i])
        else:  # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Num:  [ 1.  1.  0.  1.  0.  1.  0.  0.  0.  1.  0.  1.  0.  3.  2.  0.  0.  0.
    # 1.  0.  0.  1.  0.  1.  0.  1.  1.  2.  0.  1.  0.  0.]
    # print("p1Num:", p1Num)
    # print("p1Denom:", p1Denom)  # 19.0
    # print("p0Num: ", p0Num)
    # print("p0Denom:", p0Denom)  # 24.0
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive  # 返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率


"""
函数说明:朴素贝叶斯分类器分类函数

Parameters:
    vec2Classify - 待分类的词条数组
    p0Vec - 侮辱类的条件概率数组
    p1Vec -非侮辱类的条件概率数组
    pClass1 - 文档属于侮辱类的概率
Returns:
    0 - 属于非侮辱类
    1 - 属于侮辱类
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-12
"""


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    print("vec2Classify", vec2Classify)
    print("p1Vec", p1Vec)
    print("pClass1", pClass1)
    # vec2Classify [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0]
    # p1Vec [ 0.05263158  0.          0.          0.          0.          0.          0.
    # 0.          0.05263158  0.          0.          0.05263158  0.          0.
    # 0.05263158  0.          0.05263158  0.05263158  0.05263158  0.
    # 0.05263158  0.          0.          0.10526316  0.05263158  0.10526316
    # 0.05263158  0.          0.05263158  0.05263158  0.15789474  0.        ]
    # 两个一维数组 相乘 里面每一二个元素对应相乘
    p1 = reduce(lambda x, y: x * y, vec2Classify * p1Vec) * pClass1  # 对应元素相乘
    p0 = reduce(lambda x, y: x * y, vec2Classify * p1Vec) * (1.0 - pClass1)
    print('p0:', p0)
    print('p1:', p1)
    if p1 > p0:
        return 1
    else:
        return 0


"""
函数说明:朴素贝叶斯分类器分类函数

Parameters:
    vec2Classify - 待分类的词条数组
    p0Vec - 侮辱类的条件概率数组
    p1Vec -非侮辱类的条件概率数组
    pClass1 - 文档属于侮辱类的概率
Returns:
    0 - 属于非侮辱类
    1 - 属于侮辱类
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-12
"""


def classifyNB1(vec2Classify, p0Vec, p1Vec, pClass1):
    # 在 trainNB1 () 中 ，
    # p1Vect = np.log(p1Num / p1Denom)  # 取对数，防止下溢出
    # p0Vect = np.log(p0Num / p0Denom)
    # 对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    # 因为在 trainNB1 () 中都是取的自然对数，所以 相乘 在这里 对应相加
    #  p1 = reduce(lambda x, y: x * y, vec2Classify * p1Vec) * pClass1
    # p1 = np.log( vec2Classify * p1Vec * pClass1 ) = np.log( vec2Classify * p1Vec) + np.log(pClass1)
    #  = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


"""
函数说明:测试朴素贝叶斯分类器

注：该方法中，由于 0 的影响，导致 概率整体为 0，以及下溢出 的问题，影响最后结果，导致结局不准确。

Parameters:
    无
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-12
"""


def testingNB():
    listOPosts, listClasses = loadDataSet()  # 创建实验样本
    myVocabList = createVocabList(listOPosts)  # 创建词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  # 将实验样本向量化
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))  # 训练朴素贝叶斯分类器
    testEntry = ['love', 'my', 'dalmation']  # 测试样本1
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB1(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果
    testEntry = ['stupid', 'garbage']  # 测试样本2

    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB1(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果


"""
函数说明:朴素贝叶斯分类器训练函数

注：为了解决是数据中0 的问题，以及下溢出，导致的结果输出错误，对数据进行修改，采用拉普拉斯平滑， 也叫加1 平滑，
 分子初始化为 1，分母初始化为 2.

Parameters:
    trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
    p0Vect - 侮辱类的条件概率数组
    p1Vect - 非侮辱类的条件概率数组
    pAbusive - 文档属于侮辱类的概率
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-12
Note:
    ZJ studied in 2017-10-23.
"""


# 从上图可以看出，在计算的时候已经出现了概率为0的情况。如果新实例文本，包含这种概率为0的分词，
# 那么最终的文本属于某个类别的概率也就是0了。显然，这样是不合理的，为了降低这种影响，可以将所有词的出现数初始化为1，
# 并将分母初始化为2。这种做法就叫做拉普拉斯平滑(Laplace Smoothing)又被称为加1平滑，是比较常用的平滑方法，
# 它就是为了解决0概率问题。
#
# 除此之外，另外一个遇到的问题就是下溢出，这是由于太多很小的数相乘造成的。
# 学过数学的人都知道，两个小数相乘，越乘越小，这样就造成了下溢出。在程序中，
# 在相应小数位置进行四舍五入，计算结果可能就变成0了。为了解决这个问题，对乘积结果取自然对数。
# 通过求对数可以避免下溢出或者浮点数舍入导致的错误。同时，采用自然对数进行处理不会有任何损失。
# 下图给出函数f(x)和ln(f(x))的曲线。


def trainNB1(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 计算训练的文档数目 行数
    numWords = len(trainMatrix[0])  # 计算每篇文档的词条数 列数
    # trainCategory 是 对所有的文档 进行标记，1 侮辱 0  非侮辱 ，
    # sum(trainCategory) 是 trainCategory  中 所有元素相加
    #  相当于 所有的 1 相加，也就是 1 所占 的总个数 再除以 float(numTrainDocs)  总数，就是 侮辱类的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 文档属于侮辱类的概率
    # 下面4 行 与 trainNB0（）方法中相比，则是 采用了 拉普拉斯平滑 ， 分子 初始化 为 1 ，分母初始化 为 2
    #  上面 trainNB0（） 中创建的 np.zeros(numWords) 0 矩阵 ，这里创建的是 1 矩阵
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)  # 创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0
    p1Denom = 2.0  # 分母初始化为2,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:  # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)  # 取对数，防止下溢出
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive  # 返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率


"""
reduce () lanmada

"""


def test_reduce():
    a = np.array([1, 2, 3, 4])
    b = np.array([0.1, 0.1, 0.1, 0.1])
    # 0.1 *0.2 *0.3 *0.4 数组 对应元素相乘后，然后得到的数 再依次相乘
    c = reduce(lambda x, y: x * y, a * b)
    print(c)


if __name__ == '__main__':
    test_reduce()
    # postingList, classVec = loadDataSet()
    #
    # for each in postingList:
    #     print(each)
    # print(classVec)

    # postingList, classVec = loadDataSet()
    # myVocabList = createVocabList(postingList)
    # print('myVocabList:\n', myVocabList)
    # trainMat = []
    # for postinDoc in postingList:
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # # print('trainMat:\n', trainMat)
    # p0V, p1V, pAb = trainNB1(trainMat, classVec)
    # # 先制作 myVocabList  词汇表，然后
    # print('p0V:\n', p0V)
    # print('p1V:\n', p1V)
    # print('classVec:\n', classVec)
    # print('pAb:\n', pAb)

    # 测试分类，将输入 语句 进行分类，是否为 侮辱性 语句，在 修改 trainNB1 和 classifyNB1 () 后，结果正确。
    testingNB()
    # ['love', 'my', 'dalmation'] 属于非侮辱类
    # ['stupid', 'garbage'] 属于侮辱类

"""
朴素贝叶斯推断的一些优点：

生成式模型，通过计算概率来进行分类，可以用来处理多分类问题。
对小规模的数据表现很好，适合多分类任务，适合增量式训练，算法也比较简单。

朴素贝叶斯推断的一些缺点：

对输入数据的表达形式很敏感。
由于朴素贝叶斯的“朴素”特点，所以会带来一些准确率上的损失。
需要计算先验概率，分类决策存在错误率。

"""
