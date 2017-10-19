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
    print("p1Num:", p1Num)
    print("p1Denom:", p1Denom)  # 19.0
    print("p0Num: ", p0Num)
    print("p0Denom:", p0Denom)  # 24.0
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive  # 返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率


if __name__ == '__main__':
    # postingList, classVec = loadDataSet()
    #
    # for each in postingList:
    #     print(each)
    # print(classVec)

    postingList, classVec = loadDataSet()
    myVocabList = createVocabList(postingList)
    print('myVocabList:\n', myVocabList)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # print('trainMat:\n', trainMat)
    p0V, p1V, pAb = trainNB0(trainMat, classVec)
    # 先制作 myVocabList  词汇表，然后
    print('p0V:\n', p0V)
    print('p1V:\n', p1V)
    print('classVec:\n', classVec)
    print('pAb:\n', pAb)
