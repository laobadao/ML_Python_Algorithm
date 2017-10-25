import matplotlib.pyplot as plt
import numpy as np

"""
函数说明:梯度上升算法测试函数

求函数f(x) = -x^2 + 4x的极大值

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
    2017-08-28
Note :
    ZJ studied in 2017-10-24
"""


def gradient_ascent_test():
    # f(x) = -x^2 + 4x ; f(x) 的导数 f'(x)= -2x+4
    def f_prime(x_old):
        return -2 * x_old + 4

    # 初始值，给一个小于x_new的值
    x_old = -1
    # 梯度上升算法初始值，即从(0,0)开始
    x_new = 0
    # 步长，也就是学习速率，控制更新的幅度
    alpha = 0.01
    # 精度，也就是更新阈值
    presision = 0.00000001
    while abs(x_new - x_old) > presision:
        x_old = x_new
        # Xi := Xi + alpha * (导数f'(x))
        x_new = x_old + alpha * f_prime(x_old)
    print(x_new)


"""
函数说明:加载数据

Parameters:
    无
Returns:
    dataMat - 数据列表
    labelMat - 标签列表
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-08-28
Note:
	ZJ studied in 2017-10-25
"""


def loadDataSet():
    # 创建数据列表
    dataMat = []
    # 创建标签列表
    labelMat = []
    # 打开文件
    fr = open('testset.txt')
    # 逐行读取
    for line in fr.readlines():
        # 去除首尾空白符 切割 放入列表 创建 lineArr list ,line 处理完后 添加到list 中
        lineArr = line.strip().split()
        # 添加数据
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 添加标签
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, labelMat

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
    2017-08-28
Note:
    ZJ studied in 2017-10-25
"""


def sigmoid(inX):
    # exp() 方法返回x的指数,e^x  以e 为底的 x 次方
    return 1.0 / (1 + np.exp(-inX))


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
Note:
    ZJ studied in 2017-10-25
"""


def gradAscent(dataMatIn, classLabels):
    # dataMatIn 数据集转换成 numpy 的 mat 矩阵
    dataMatrix = np.mat(dataMatIn)
    # 数据标签转化为 numpy 的 mat 矩阵，并进行转置 transpose()
    labelMat = np.mat(classLabels).transpose()
    # 计算 dataMatrix 矩阵的大小 ，返回 行数 m ，列数 n
    m, n = np.shape(dataMatrix)
    # 每次移动的步长，也就是学习速率，控制更新的幅度
    alpha = 0.001
    # 最大迭代次数
    maxCycles = 500
    # 创建 n 行1 列的矩阵 数据全是1 ，n 是前面的列数 也就是 3 行 1 列
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        # dataMatrix 相当于X矩阵 weights 相当于 w^T
        h = sigmoid(dataMatrix * weights)
        # sigmoid 函数计算出来 返回的数跟实际的类别做对比
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
        # 将矩阵转换为数组 getA() ，返回权重数组
    return weights.getA()


"""
函数说明:绘制数据集

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
	2017-08-30
Note:
	ZJ studied in 2017-10-25
"""


def plotDataSet():
    # 加载数据集
    dataMat, labelMat = loadDataSet()
    # 将 dataMat list 转换成 numpy 的 array 数组
    dataArr = np.array(dataMat)
    # 数据个数
    n = np.shape(dataMat)[0]
    # 正样本
    xcord1 = []
    ycord1 = []
    # 负样本
    xcord2 = []
    ycord2 = []
    # 根据数据集标签进行分类
    for i in range(n):
        # 1为正样本
        if int(labelMat[i] == 1):
            # [1.0, 0.5465456, 0.95465456] 比如 第一（i）行 索引为 1 的 0.5465456 添加到 xcord1
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        # 0为负样本
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    # 添加subplot
    fig = plt.figure()
    # 绘制正样本
    ax = fig.add_subplot(111)
    # 绘制负样本
    ax.scatter(xcord1, ycord1, s=20, c='red', alpha=.5)
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)
    # 绘制titl
    plt.title('DataSet')
    # 绘制label
    plt.xlabel('x')
    plt.ylabel('y')
    # 显示
    plt.show()


"""
函数说明:绘制数据集

Parameters:
    weights - 权重参数数组
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-08-30
"""


def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()  # 加载数据集
    dataArr = np.array(dataMat)  # 转换成numpy的array数组
    n = np.shape(dataMat)[0]  # 数据个数
    xcord1 = []
    ycord1 = []  # 正样本
    xcord2 = []
    ycord2 = []  # 负样本
    for i in range(n):  # 根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])  # 1为正样本
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])  # 0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.scatter(xcord1, ycord1, s=20, c='red', alpha=.5)  # 绘制正样本
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)  # 绘制负样本
    # weights 是前面求出来的回归系数 arange(-3.0, 3.0, 0.1) 区间 从-3 到3 间隔 0.1
    x = np.arange(-3.0, 3.0, 0.1)

    # 处设置了 sigmoid 函数为0。-《机器学习实战》
    # 回忆5.2节，0是两个分类（类别1和类别0）的分界处。
    # 因此，我们设定 0 = w0x0 + w1x1 + w2x2，
    # 然后解出X2和X1的关系式（即分隔线的方程，注意X0＝1）。
    # 相当于 X2是 Y ，然后就是 y 对 x 的关系式
    # 对于为什么要把 x2 当y,看 画图那 就是这样的,横轴是 x1 ，也是数据矩阵里索引为[1] 的数据
    # 纵轴 是 数据矩阵里 索引为 [2] 的数据
    # 0 = w0x0 + w1x1 + w2x2，  x0=1
    # 0 = w0 + w1x1 + w2x2
    # x2 =(w0 + w1x1 )/w2
    y = (-weights[0] - weights[1] * x) / weights[2]

    ax.plot(x, y)

    plt.title('BestFit')  # 绘制title
    plt.xlabel('X1')
    plt.ylabel('X2')  # 绘制label
    plt.show()


if __name__ == '__main__':
    # gradient_ascent_test()
    # 1.999999515279857 也就是 x 为 1.999999515279857 四舍五入2 时 可以取得函数最大值
    # dataMat, labelMat = loadDataSet()
    # print(dataMat)
    # plotDataSet()

    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    plotBestFit(weights)
# 这是求解出来的回归系数 [w0,w1,w2]
# [[ 4.12414349]
# [ 0.48007329]
# [-0.6168482 ]]
