from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import random

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

注：若数据集成千上亿，那计算复杂度太大 

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
函数说明:梯度上升算法

Parameters:
    dataMatIn - 数据集
    classLabels - 数据标签
Returns:
    weights.getA() - 求得的权重数组(最优参数)
    weights_array - 每次更新的回归系数
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-08-28
"""


def gradAscent1(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  # 转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()  # 转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.01  # 移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500  # 最大迭代次数
    weights = np.ones((n, 1))
    weights_array = np.array([])
    # print("weights_array",weights_array)
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
        weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(maxCycles, n)
    # print("weights_array", weights_array)
    return weights.getA(), weights_array  # 将矩阵转换为数组，并返回


"""
函数说明:改进的随机梯度上升算法
目的：为了减少降低计算复杂度

Parameters:
	dataMatrix - 数据数组
	classLabels - 数据标签
	numIter - 迭代次数
Returns:
	weights - 求得的回归系数数组(最优参数)
	weights_array - 每次更新的回归系数
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Zhihu:
	https://www.zhihu.com/people/Jack--Cui/
Modify:
	2017-08-31
Note:
    ZJ studied in 2017-10-26
"""


def stocGradAscent2(dataMatrix, classLabels, numIter=150):
    # 返回数据集 dataMatrix 的大小，m 行数, n 列数
    m, n = np.shape(dataMatrix)
    # print("m", m)
    # 初始化参数，也就是系数 创建元素都是 1 的 n 行 1 列数组
    weights = np.ones(n)
    # 创建空二维数组，用来存储每次更新的回归系数
    weights_array = np.array([])
    # 循环 numIter 次，进行迭代
    for j in range(numIter):
        # a = list(range(5)) [0, 1, 2, 3, 4] 创建 list  范围是 0到5
        dataIndex = list(range(m))
        for i in range(m):
            # 降低 alpha 的大小，原来的 0.01 ，每次减小 1/(j+i)
            # 第一个改进之处在于，alpha在每次迭代的时候都会调整，并且，虽然alpha会随着迭代次数不断减小，
            # 但永远不会减小到0，因为这里还存在一个常数项 4 是经验值，不要纠结 ，从 4.01 开始逐渐减小到 0.02
            alpha = 4 / (j + i + 1.0) + 0.01
            # print("alpha", alpha)
            # 随机选取样本 dataIndex 是之前创建的 m 行 个的数据的 索引的 list 存储的是索引
            randIndex = int(random.uniform(0, len(dataIndex)))
            # 只将从 dataMatrix 随机选取的 dataIndex 索引上的数据 进行 sigmoid 函数运算
            # 选择随机选取的一个样本，计算 h,h 是计算出来的 分类情况 ，二分类
            # 回想 sigmoid 函数图 ，输入 Z , 根据公式 1/ 1+ e^(-Z)
            # Z是矩阵，是 w 系数矩阵 和 X 矩阵的乘积
            h = sigmoid(sum(dataMatrix[randIndex] * weights))  # 计算误差，计算出的 h 与实际分类之间的 误差
            error = classLabels[randIndex] - h
            # W = W + alpha * (y - h(x))* X 梯度上升算法的迭代公式,更新回归系数
            # 其中 W 就是 weights , y - h(x) = error ,X 就是 dataMatrix[randIndex]
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 添加回归系数到数组中 axis=0 行 上累加 axis=1 是列方向
            weights_array = np.append(weights_array, weights, axis=0)
            # 从 dataMatrix 数据集中，删除已经使用过的样本
            # 第二个改进的地方在于更新回归系数(最优参数)时，只使用一个样本点，
            # 并且选择的样本点是随机的，每次迭代不使用已经用过的样本点。
            # 这样的方法，就有效地减少了计算量，并保证了回归效果
            del (dataIndex[randIndex])
            # print(weights)
            # 改变 weights_array 的维度 numIter * m  150* m 行，n 列 ？？？
    weights_array = weights_array.reshape(numIter * m, n)
    # , weights_array
    return weights, weights_array


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)  # 参数初始化
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # 降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex] * weights))  # 选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h  # 计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]  # 更新回归系数
            del (dataIndex[randIndex])  # 删除已经使用的样本
    return weights


"""
函数说明:改进的随机梯度上升算法

Parameters:
    dataMatrix - 数据数组
    classLabels - 数据标签
    numIter - 迭代次数
Returns:
    weights - 求得的回归系数数组(最优参数)
    weights_array - 每次更新的回归系数
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-08-31
"""


def stocGradAscent3(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)  # 参数初始化
    weights_array = np.array([])  # 存储每次更新的回归系数
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # 降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex] * weights))  # 选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h  # 计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]  # 更新回归系数
            weights_array = np.append(weights_array, weights, axis=0)  # 添加回归系数到数组中
            del (dataIndex[randIndex])  # 删除已经使用的样本

    # aa =  np.array([]); a =[1,2,3]; b= [2,3,4] ;aa = append(aa, a, axis = 0);aa = append(aa, b, axis = 0)
    # aa = array([ 1.,  2.,  3.,  2.,  3.,  4.]) ;  aa = aa.reshape(2,3) ; 将原来 1 行6 列的数组 变成 2 行3列的
    # array([[ 1.,  2.,  3.],
    #      [ 2.,  3.,  4.]])
    weights_array = weights_array.reshape(numIter * m, n)  # 改变维度
    return weights, weights_array


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


"""
函数说明:绘制回归系数与迭代次数的关系
注：
从这个分类结果中，我们不好看出迭代次数和回归系数的关系，
也就不能直观的看到每个回归方法的收敛情况。
因此，我们编写程序，绘制出回归系数和迭代次数的关系曲线

Parameters:
    weights_array1 - 回归系数数组1
    weights_array2 - 回归系数数组2
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
    ZJ studied in 2017-10-26
"""


def plotWeights(weights_array1, weights_array2):
    # 设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 将 fig 画布分隔成 1 行 1 列,不共享 x 轴和 y 轴, fig 画布的大小为 (20, 10)
    # 当 nrow=3, nclos=2时,代表 fig 画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20, 10))
    x1 = np.arange(0, len(weights_array1), 1)
    # weights_array1 [[w0,w1,w2],[w0,w1,w2],[w0,w1,w2],[w0,w1,w2],[w0,w1,w2]]
    # 绘制 w0 与迭代次数的关系 weights_array1[:, 0] 第 0 列 所有行 也就是 所有的  w0
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    # x 轴上画的是 从 0 到  len(weights_array2)  间隔 1
    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()
# 由于改进的随机梯度上升算法，随机选取样本点，所以每次的运行结果是不同的。
# 但是大体趋势是一样的。我们改进的随机梯度上升算法收敛效果更好。
# 为什么这么说呢？让我们分析一下。我们一共有100个样本点，改进的随机梯度上升算法迭代次数为150。
# 而上图显示15000次迭代次数的原因是，使用一次样本就更新一下回归系数。
# 因此，迭代150次，相当于更新回归系数150*100=15000次。简而言之，迭代150次，更新1.5万次回归参数。
# 从上图左侧的改进随机梯度上升算法回归效果中可以看出，其实在更新2000次回归系数的时候，已经收敛了。
# 相当于遍历整个数据集20次的时候，回归系数已收敛。训练已完成。


if __name__ == '__main__':
    # gradient_ascent_test()
    # 1.999999515279857 也就是 x 为 1.999999515279857 四舍五入2 时 可以取得函数最大值
    # dataMat, labelMat = loadDataSet()
    # print(dataMat)
    # plotDataSet()

    # 梯度上升算法
    # dataMat, labelMat = loadDataSet()
    # weights = gradAscent(dataMat, labelMat)
    # plotBestFit(weights)
    # 这是求解出来的回归系数 [w0,w1,w2]
    # [[ 4.12414349]
    # [ 0.48007329]
    # [-0.6168482 ]]

    # 随机梯度上升算法
    # dataMat, labelMat = loadDataSet()
    # weights = stocGradAscent2(np.array(dataMat), labelMat)
    # plotBestFit(weights)

    # 绘制 迭代次数 和 回归系数之间的关系

    dataMat, labelMat = loadDataSet()
    weights1, weights_array1 = stocGradAscent3(np.array(dataMat), labelMat)
    m1, n1 = np.shape(weights_array1)
    print("m1=", m1, "n1=", n1)
    # m1= 15000 n1= 3

    weights2, weights_array2 = gradAscent1(dataMat, labelMat)
    m2, n2 = np.shape(weights_array2)
    print("m2=", m2, "n2=", n2)
    plotWeights(weights_array1, weights_array2)
    # m2= 500 n2= 3
