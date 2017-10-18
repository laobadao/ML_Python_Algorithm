import pandas as pd

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
    print("lenses_dict: ", lenses_dict)
    # pandas DataFrame
    lenses_pd = pd.DataFrame(lenses_dict)
    print(lenses_pd)


    # from sklearn import tree
    #
    # if __name__ == '__main__':
    #     fr = open('lenses.txt')
    #     # for 循环遍历 行 每行 去掉首尾空白符 根据 \t 进行切割
    #     lenses = [inst.strip().split('\t') for inst in fr.readline()]
    #     print(lenses)
    #     lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    #     # 使用 sklearn 中决策树的  DecisionTreeClassifier （）方法
    #     clf = tree.DecisionTreeClassifier()
    #     lenses = clf.fit(lenses, lensesLabels)
