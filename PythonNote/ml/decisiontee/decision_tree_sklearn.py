from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import StringIO
from sklearn import tree
import pandas as pd
import numpy as np
import pydotplus

if __name__ == '__main__':
    # 加载文件
    with open('lenses.txt', 'r') as fr:
        # 处理文件  fr.readline()  是读取一行  readlines 读取所有行
        lenses = [line.strip().split('\t') for line in fr.readlines()]
    # 提取每组数据的类别，保存在列表里
    lenses_target = []
    # 把数据集中 最后一列的 标签 存储起来
    for each in lenses:
        lenses_target.append(each[-1])
    print(lenses_target)
    # 特征标签
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 保存lenses数据的临时列表
    lenses_list = []
    # 保存lenses数据的字典，用于生成pandas
    lenses_dict = {}
    # 提取信息，生成字典 for 循环 lensesLabels 所有标签里面的每一项
    for each_label in lensesLabels:
        # 再for 循环 取出 数据集中每一行数据
        for each in lenses:
            # lensesLabels.index(each_label) 该标签所在特征标签 list 中的索引
            # each[lensesLabels.index(each_label)] 取出 每一行数据中 索引为 lensesLabels.index(each_label) 的元素
            # 只提取每一行 中某个元素 添加到 lenses_list
            lenses_list.append(each[lensesLabels.index(each_label)])
        # 字典中 以 某个特征为 key ,对应存储的 value 为一个是该特征数据的 list
        lenses_dict[each_label] = lenses_list
        # 再清空 重置 lenses_list ，再存储下一组特征的数据
        lenses_list = []
    # print("lenses_dict: ", lenses_dict)
    # pandas DataFrame
    lenses_pd = pd.DataFrame(lenses_dict)
    # print(lenses_pd)

    # 创建LabelEncoder()对象，用于序列化  将string类型数字化
    # 简单来说 LabelEncoder 是对不连续的数字或者文本进行编号
    le = LabelEncoder()
    # 为每一列序列化
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    # print(lenses_pd)

    # 可视化数据
    # 创建DecisionTreeClassifier()类  最大深度限制为4
    # clf = tree.DecisionTreeClassifier(max_depth=4)
    # 使用数据，构建决策树 取出 lenses_pd 中的 values 转化为 list
    # 传入 训练数据和类别 list
    # clf = clf.fit(lenses_pd.values.tolist(), lenses_target)
    # dot_data = StringIO()
    # tree.export_graphviz(clf, out_file=dot_data,
    #                      feature_names=lenses_pd.keys(),
    #                      class_names=clf.classes_,
    #                      filled=True, rounded=True,
    #                      special_characters=True)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("tree.pdf")

    clf = tree.DecisionTreeClassifier(max_depth=4)  # 创建DecisionTreeClassifier()类
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)  # 使用数据，构建决策树
    # dot_data = StringIO()
    # tree.export_graphviz(clf, out_file=dot_data,  # 绘制决策树
    #                      feature_names=lenses_pd.keys(),
    #                      class_names=clf.classes_,
    #                      filled=True, rounded=True,
    #                      special_characters=True)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("tree.pdf")
    # 预测： ['hard'] 
    print('预测：', clf.predict([[1, 1, 1, 0]]))


"""七、总结

决策树的一些优点：

易于理解和解释。决策树可以可视化。
几乎不需要数据预处理。其他方法经常需要数据标准化，创建虚拟变量和删除缺失值。决策树还不支持缺失值。
使用树的花费（例如预测数据）是训练数据点(data points)数量的对数。
可以同时处理数值变量和分类变量。其他方法大都适用于分析一种变量的集合。
可以处理多值输出变量问题。
使用白盒模型。如果一个情况被观察到，使用逻辑判断容易表示这种规则。
相反，如果是黑盒模型（例如人工神经网络），结果会非常难解释。
即使对真实模型来说，假设无效的情况下，也可以较好的适用。

决策树的一些缺点：

决策树学习可能创建一个过于复杂的树，并不能很好的预测数据。也就是过拟合。修剪机制（现在不支持），
设置一个叶子节点需要的最小样本数量，或者数的最大深度，可以避免过拟合。
决策树可能是不稳定的，因为即使非常小的变异，可能会产生一颗完全不同的树。
这个问题通过decision trees with an ensemble来缓解。
决策树可能是不稳定的，因为即使非常小的变异，可能会产生一颗完全不同的树。
这个问题通过decision trees with an ensemble来缓解。
概念难以学习，因为决策树没有很好的解释他们，例如，XOR, parity or multiplexer problems。
如果某些分类占优势，决策树将会创建一棵有偏差的树。因此，建议在训练之前，先抽样使样本均衡。
"""