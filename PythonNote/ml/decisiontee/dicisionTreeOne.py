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


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    shannonEnt = calcuShannonEnt(dataSet)
    print(dataSet)
    print(shannonEnt)  # 0.9709505944546686
