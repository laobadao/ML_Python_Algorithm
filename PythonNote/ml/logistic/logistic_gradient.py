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


if __name__ == '__main__':
    # gradient_ascent_test()
    # 1.999999515279857 也就是 x 为 1.999999515279857 四舍五入2 时 可以取得函数最大值
    # dataMat, labelMat = loadDataSet()
    # print(dataMat)
    plotDataSet()
