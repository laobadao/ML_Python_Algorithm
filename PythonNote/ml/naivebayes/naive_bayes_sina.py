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
    train_data_list, train_class_list = zip(*train_list)  # 训练集解压缩
    test_data_list, test_class_list = zip(*test_list)  # 测试集解压缩

    all_words_dict = {}  # 统计训练集词频
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    # 根据键的值倒序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)  # 解压缩
    all_words_list = list(all_words_list)  # 转换成列表
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


if __name__ == '__main__':
    # 文本预处理
    folder_path = './SogouC/Sample'  # 训练集存放地址
    TextProcessing(folder_path)
