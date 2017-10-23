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


def TextProcessing(folder_path):
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
        # ['10.txt', '11.txt', '12.txt', '13.txt', '14.txt', '15.txt', '16.txt', '17.txt', '18.txt', '19.txt']
        # ['10.txt', '11.txt', '12.txt', '13.txt', '14.txt', '15.txt', '16.txt', '17.txt', '18.txt', '19.txt']
        # ['10.txt', '11.txt', '12.txt', '13.txt', '14.txt', '15.txt', '16.txt', '17.txt', '18.txt', '19.txt']
        # ['10.txt', '11.txt', '12.txt', '13.txt', '14.txt', '15.txt', '16.txt', '17.txt', '18.txt', '19.txt']
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


if __name__ == '__main__':
    # 文本预处理
    folder_path = './SogouC/Sample'  # 训练集存放地址
    TextProcessing(folder_path)
