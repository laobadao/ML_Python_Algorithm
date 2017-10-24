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
函数说明:文本特征选取 特征词选取

也就是滤除那些没有用的词组，如 了  在 吗 ？ ， 等返回的 feature_words 就是我们最终选出的用于新闻分类的特征

Parameters:
    all_words_list - 训练集所有文本列表
    deleteN - 删除词频最高的 deleteN 个词 int 型 给出要删除的个数
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


if __name__ == '__main__':
    # 文本预处理
    folder_path = './SogouC/Sample'  # 训练集存放地址
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path,
                                                                                                        test_size=0.2)
    # print(all_words_list)

    #生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    feature_words = words_dict(all_words_list, 100, stopwords_set)
    print(feature_words)
