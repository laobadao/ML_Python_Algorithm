from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import os
import random
import jieba

"""
函数说明:中文文本处理

Parameters:
    folder_path - 文本存放的路径
    test_size - 测试集占比，默认占所有数据集的百分之20
Returns:
    all_words_list - 按词频降序排序的训练集列表
    train_data_list - 训练集列表
    test_data_list - 测试集列表
    train_class_list - 训练集标签列表
    test_class_list - 测试集标签列表
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-22
Note:
    ZJ studied in 2017-10-23
"""


def TextProcessing(folder_path, test_size=0.2):
    # 查看folder_path下的文件夹  遍历所有文件夹名称加入到 folder_list
    folder_list = os.listdir(folder_path)
    # ['C000008', 'C000010', 'C000013', 'C000014', 'C000016', 'C000020', 'C000022', 'C000023', 'C000024']
    # print("folder_list", folder_list)
    # 数据集数据
    data_list = []
    # 数据集类别
    class_list = []

    # 遍历每个子文件夹
    for folder in folder_list:
        # 根据子文件夹，生成新的路径   folder_path = './SogouC/Sample'
        #  将 folder 加入到前面的路径后 组合成新的路径  如./SogouC/Sample\C000010   ./SogouC/Sample\C000013
        new_folder_path = os.path.join(folder_path, folder)
        # print("new_folder_path", new_folder_path)
        # 存放子文件夹下的txt文件的列表
        files = os.listdir(new_folder_path)
        # print(files)
        # ['10.txt', '11.txt', '12.txt', '13.txt', '14.txt', '15.txt', '16.txt', '17.txt', '18.txt', '19.txt']
        # ['10.txt', '11.txt', '12.txt', '13.txt', '14.txt', '15.txt', '16.txt', '17.txt', '18.txt', '19.txt']
        # 存放子文件夹下的txt文件的列表j = 1
        # 遍历每个txt文件
        j = 1
        for file in files:
            if j > 100:  # 每类txt样本数最多100个
                break
            # ./SogouC/Sample\C000010/10.txt  组合文件完整路径
            with open(os.path.join(new_folder_path, file), 'r', encoding='utf-8') as f:  # 打开txt文件
                raw = f.read()
            # cut_all=True 全模式 cut_all=False 精简模式 默认为精简模式
            word_cut = jieba.cut(raw, cut_all=False)  # 精简模式，返回一个可迭代的generator
            # print("word_cut", word_cut) 
            # <generator object Tokenizer.cut at 0x000001B284657F10>
            word_list = list(word_cut)  # generator转换为list
            # print("word_list", word_list) 
            data_list.append(word_list)
            # folder= C000010  ...每個 folder 代表一种类型
            class_list.append(folder)
            j += 1
            # print(data_list)
            # print(class_list)

    # x = [1, 2, 3]
    # y = [4, 5, 6]
    # z = [7, 8, 9]
    # xyz = zip(x, y, z)
    # print(xyz)  [(1, 4, 7), (2, 5, 8), (3, 6, 9)] 三个list  列方向上 每一个 索引 对应进行组合
    data_class_list = list(zip(data_list, class_list))  # zip压缩合并，将数据与标签对应压缩
    # print('data_class_list', data_class_list)
    random.shuffle(data_class_list)  # 将data_class_list乱序
    index = int(len(data_class_list) * test_size) + 1  # 训练集和测试集切分的索引值
    # [index:] 从该索引位置开始 到最后 区间的数据
    train_list = data_class_list[index:]  # 训练集
    # [:index] 从0 开始 到该索引，不包括 该 index 的数据
    test_list = data_class_list[:index]  # 测试集
    # zip(* ) * 代表解压缩 将之前打乱的 训练数据 和 相应类别的合计，再还原 成 两个独立的list
    train_data_list, train_class_list = zip(*train_list)  # 训练集解压缩
    test_data_list, test_class_list = zip(*test_list)  # 测试集解压缩

    all_words_dict = {}  # 统计训练集词频 字典 key 词  对应 value 出现的次数
    # 双层 for 循环 ，给一个 空 字典 添加 key  并复制 value 的基本思路
    # 不存在 则加入 该KEY 且 初始化 赋值1 ，存在则累计 加1
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    # for word_list in train_data_list:
    #     for word in word_list:
    #         if word in all_words_dict.keys():
    #             all_words_dict[word] += 1
    #         else:
    #             all_words_dict[word] = 1

    # 根据键的值倒序排序  也就是 单词出现贫度 越高的 往前排
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)  # 解压缩
    all_words_list = list(all_words_list)  # 转换成列表
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


"""
函数说明:读取文件里的内容，并去重

Parameters:
    words_file - 文件路径
Returns:
    words_set - 读取的内容的set集合
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-22
Note:
    ZJ studied in 2017-10-23
"""


def MakeWordsSet(word_file):
    # 创建 set 集合 
    word_set = set()
    # 打开文件 
    with open(word_file, 'r', encoding='utf-8') as f:
        # 一行一行读取
        for line in f.readlines():
            word = line.strip()  # 去除首尾空白符
            if len(line) > 0:  # 若 文件 不为空 len() > 0
                word_set.add(word)  # 则添加到 set 中 ，set 中是去重的
    return word_set


"""
函数说明:根据feature_words将文本向量化

Parameters:
    train_data_list - 训练集
    test_data_list - 测试集
    feature_words - 特征集
Returns:
    train_feature_list - 训练集向量化列表
    test_feature_list - 测试集向量化列表
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-22
"""


def TextFeatures(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):  # 出现在特征集中，则置1
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list  # 返回结果


# def TextFeatures1(train_data_list, test_data_list, feature_words):
#     def text_features(text, feature_words):
#         text_words = set(text)
#         features = [1 if word in text_words else 0 for word in feature_words]
#         return features
#
#     train_feature_list = [text_features(text, feature_words) for text in train_data_list]
#     test_feature_list = [text_features(text, feature_words) for text in test_data_list]
#     return train_feature_list, test_feature_list


"""
函数说明:文本特征选取 特征词选取

也就是滤除那些没有用的词组，如 了  在 吗 ？ ， 等返回的 feature_words 就是我们最终选出的用于新闻分类的特征

Parameters:
    all_words_list - 训练集所有文本列表
    deleteN - 删除词频最高的 deleteN 个词 int 型 给出要删除的个数， 因为all_words_list 是按降序排序的 list  q,
    前面 最靠前的 比如 400 个 都是些无用词 如 ， 。 了 好 等，先把这些词去除掉，在从剩下的词中选取 特征词
    stopwords_set - 指定的结束语 是 MakeWordsSet（）方法中的返回结果 读取的内容的 set 集合 
Returns:
    feature_words - 特征集
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-22
"""

# all_words_list 已经是 按降序排的 list 特征词 出现频度 最高的 在前面
def words_dict(all_words_list, deleteN, stopwords_set=set()):
    feature_words = []  # 特征列表
    n = 1
    # range(1,5,2) #代表从1到5，间隔2(不包含5) 
    # 从 deleteN 到 len(all_words_list) 不包含 len(all_words_list) 间隔 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:  # feature_words的维度为1000
            break
            # 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
        # not isdigit()  不是数字 
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(
                all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words


"""
函数说明:新闻分类器

Parameters:
    train_feature_list - 训练集向量化的特征文本
    test_feature_list - 测试集向量化的特征文本
    train_class_list - 训练集分类标签
    test_class_list - 测试集分类标签
Returns:
    test_accuracy - 分类器精度
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-22
Note:
    ZJ studied in 2017-10-24
"""


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy


if __name__ == '__main__':
    # 文本预处理
    folder_path = './SogouC/Sample'  # 训练集存放地址
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path,
                                                                                                        test_size=0.2)
    # print(all_words_list)

    # 生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    # feature_words = words_dict(all_words_list, 100, stopwords_set)
    # print(feature_words)

    # test_accuracy_list = []
    # # 从 0 到 1000 每20 个间隔 不包含1000 选取不同的 deleteN 与分类器 精度（准确率）之间的关系
    # deleteNs = range(0, 1000, 20)  # 0 20 40 60 ... 980
    # for deleteN in deleteNs:
    #     feature_words = words_dict(all_words_list, deleteN)
    #     train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list,feature_words)
    #     test_accuracy = TextClassifier(train_feature_list,test_feature_list,train_class_list,test_class_list)
    #     test_accuracy_list.append(test_accuracy)

    # plt.figure()
    # plt.plot(deleteNs,test_accuracy_list)
    # plt.title('Relationship of deleteNs and test_accuracy')
    # plt.xlabel('deleteNs')
    # plt.ylabel('test_accuracy')
    # plt.show()

    #  上面这段代码 为了找出 合适的 deleteN 的值 运行了 4 次 根据图的 比较 选取 400 这个值 
    # deleteN 选取 400 时 ，精准度的值 都较高

    test_accuracy_list = []
    feature_words = words_dict(all_words_list, 400, stopwords_set)
    train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
    test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
    test_accuracy_list.append(test_accuracy)
    ave = lambda c: sum(c) / len(c)
    print(ave(test_accuracy_list))

    # 0.789473684211
    # 也就是 给出的所有数据中 ，取出 20% 作为 测试数据 80% 为训练数据， 使用 朴素贝叶斯算法 训练 训练数据，
    # 然后 再使用测试数据 去使用 该分类算法，并且 跟 实际上 测试数据所属分类 做比较  0.789473684211 是准确率 
    # 也就是 这个分类算法的 准确率 有 80% 这么高
# 总结：
# 在训练朴素贝叶斯分类器之前，要处理好训练集，文本的清洗还是有很多需要学习的东西。
# 根据提取的分类特征将文本向量化，然后训练朴素贝叶斯分类器。
# 去高频词汇数量的不同，对结果也是有影响的的。
# 拉普拉斯平滑对于改善朴素贝叶斯分类器的分类效果有着积极的作用。