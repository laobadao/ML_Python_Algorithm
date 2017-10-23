import re
import numpy as np
import random

"""
函数说明:创建实验样本

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
"""


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList, classVec  # 返回实验样本切分的词条和类别标签向量


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
        vocabSet = vocabSet | set(document)  # 取并集
    return list(vocabSet)


"""
函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0

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
"""


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:  # 遍历每个词条
        if word in vocabList:  # 如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec  # 返回文档向量


"""
函数说明:根据vocabList词汇表，构建词袋模型

Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量,词袋模型
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-14
"""


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:  # 遍历每个词条
        if word in vocabList:  # 如果词条存在于词汇表中，则计数加一
            returnVec[vocabList.index(word)] += 1
    return returnVec  # 返回词袋模型


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
    numTrainDocs = len(trainMatrix)  # 计算训练的文档数目
    numWords = len(trainMatrix[0])  # 计算每篇文档的词条数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 文档属于侮辱类的概率
    p0Num = np.ones(numWords);
    p1Num = np.ones(numWords)  # 创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0;
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
    # 对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


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
    numTrainDocs = len(trainMatrix)  # 计算训练的文档数目
    numWords = len(trainMatrix[0])  # 计算每篇文档的词条数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 文档属于侮辱类的概率
    p0Num = np.ones(numWords);
    p1Num = np.ones(numWords)  # 创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0;
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


# 正则表达式本身是一种小型的、高度专业化的编程语言，而在python中，通过内嵌集成re模块，re - regular expression 正则表达式
"""
函数说明:接收一个大字符串并将其解析为字符串列表

Parameters:
    无
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-14
Note:
    ZJ studied in 2017-10-23
"""


def textParse(bigString):
    # re - regular expression 正则表达式 ，导入 import re  库 模块
    # 将字符串转换为字符列表 split（） 切割 最后返回 list ,
    # 正则表达式为 \W* ，将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    listOfTokens = re.split(r'\W*', bigString)
    # 除了单个字母，例如大写的I，其它单词变成小写
    # len(token) > 2 除了 单个字母 其他 全变小写  token.lower()
    return [token.lower() for token in listOfTokens if len(token) > 2]


# 我们将数据集分为训练集和测试集，使用交叉验证的方式测试朴素贝叶斯分类器的准确性
"""
函数说明:测试朴素贝叶斯分类器

Parameters:
    无
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-14
"""


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):  # 遍历25个txt文件
        wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())  # 读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)  # 标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())  # 读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)  # 标记非垃圾邮件，0 表示非垃圾文件
    vocabList = createVocabList(docList)  # 创建词汇表，不重复
    # 创建大小为 50 的list range(50) 0到 50 不包含50 然后存储到 list
    # [0,1,2...,49]
    trainingSet = list(range(50))
    testSet = []  # 创建存储训练集的索引值的列表和测试集的索引值的列表
    # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
    # 先随机选择10 个作为 测试集，然后再从数据中去除 这个10个 剩下的40 个作为 训练集
    for i in range(10):
        # 0 到 50  内，随机选取 索引值 random.uniform(x, y) x -- 随机数的最小值，包含该值。y -- 随机数的最大值，不包含该值。
        randIndex = int(random.uniform(0, len(trainingSet)))  # 随机选取索索引值
        # 前面先随机选取索引，后面再将该索引值 添加到  testSet 添加测试集的索引值
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])  # 在训练集列表中删除添加到测试集的索引值
    # 创建训练集矩阵和训练集类别标签系向量  trainMat 存储向量数据
    trainMat = []
    trainClasses = []
    # trainingSet 中现在只有40 个数据 且存储的是 索引值
    for docIndex in trainingSet:  # 遍历训练集
        # setOfWords2Vec(vocabList, docList[docIndex])  vocabList 前面得到的所有词的 汇总的不重复的词汇表
        # docList[docIndex] 要向量化的 语句 根据 docIndex 索引查找对应语句，然后向量化
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))  # 将生成的词集模型添加到训练矩阵中
        # 将类别添加到训练集类别标签系向量中  classList 中有50个 根据docIndex 索引，循环 选取与之对应的40个存储
        trainClasses.append(classList[docIndex])
    # 将上面 得到的 40个 训练向量化的数据，以及40个对应的标签转化成array 作为参数传入到   trainNB0（）中
    # p0Vect - 侮辱类的条件概率数组
    # p1Vect - 非侮辱类的条件概率数组
    # pAbusive - 文档属于侮辱类的概率
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))  # 训练朴素贝叶斯模型
    errorCount = 0  # 错误分类计数
    for docIndex in testSet:  # 遍历10 个测试集
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])  # 测试集的词集模型
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:  # 如果分类错误
            errorCount += 1  # 错误计数加1
            print("分类错误的测试集：", docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))


if __name__ == '__main__':
    spamTest()
# docList = []
# classList = []
# for i in range(1, 26):
#     # 遍历25个txt文件 email/spam/ 下面的 文件名 命名是 1.text 2.text
#     # 所以 open('email/spam/%d.txt' % i, 'r').read() 这样遍历 每一个文件 spam 中 每一个 都是 垃圾邮件
#     # ham 中每一个文件都是 非垃圾邮件
#     wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())  # 读取每个垃圾邮件，并字符串转换成字符串列表
#     docList.append(wordList)
#     classList.append(1)  # 标记垃圾邮件，1表示垃圾文件
#     wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())  # 读取每个非垃圾邮件，并字符串转换成字符串列表
#     docList.append(wordList)
#     classList.append(0)  # 标记非垃圾邮件，1表示垃圾文件
# vocabList = createVocabList(docList)  # 创建词汇表，不重复
# print(vocabList)
