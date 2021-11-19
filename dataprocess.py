# -*- coding: utf-8 -*-
import os
import torch
from torch import optim
from torch.nn import RNN, LSTM, LSTMCell
import numpy as np
import re
import torch.nn as nn
import torch.nn.functional as F
import random


def load_data(path, flag='train'):
    labels = ['pos', 'neg']
    data = []
    for label in labels:
        files = os.listdir(os.path.join(path, flag, label))
        # 去除标点符号
        r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
        for file in files:
            with open(os.path.join(path, flag, label, file), 'r', encoding='utf8') as rf:
                temp = rf.read().replace('\n', '')
                temp = temp.replace('<br /><br />', ' ')
                temp = re.sub(r, '', temp)
                temp = temp.split(' ')
                temp = [temp[i].lower() for i in range(len(temp)) if temp[i] != '']
                if label == 'pos':
                    data.append([temp, 1])
                elif label == 'neg':
                    data.append([temp, 0])
    return data


# 对每一个句子进行处理，最大长度为250
def process_sentence():
    sentence_code = []
    vocabulary_vectors = np.load('vocabulary_vectors_1.npy', allow_pickle=True)
    word_list = np.load('word_list_1.npy', allow_pickle=True)
    word_list = word_list.tolist()
    train_data = load_data('aclImdb','train')
    for i in range(len(train_data)):
        print(i)
        vec = train_data[i][0]
        temp = []
        index = 0
        for j in range(len(vec)):
            try:
                index = word_list.index(vec[j])
            except ValueError:  # 没找到
                index = 399999
            finally:
                temp.append(index)  # temp表示一个单词在词典中的序号
        if len(temp) < 250:
            for k in range(len(temp), 250):  # 不足补0
                temp.append(0)
        else:
            temp = temp[0:250]  # 只保留250个
        sentence_code.append(temp)

    # print(sentence_code)

    sentence_code = np.array(sentence_code)
    np.save('sentence_code_1', sentence_code)  # 存下来

    sentence_code = []
    test_data = load_data('aclImdb', 'test')
    for i in range(len(test_data)):
        print(i)
        vec = test_data[i][0]
        temp = []
        index = 0
        for j in range(len(vec)):
            try:
                index = word_list.index(vec[j])
            except ValueError:  # 没找到
                index = 399999
            finally:
                temp.append(index)  # temp表示一个单词在词典中的序号
        if len(temp) < 250:
            for k in range(len(temp), 250):  # 不足补0
                temp.append(0)
        else:
            temp = temp[0:250]  # 只保留250个
        sentence_code.append(temp)

    # print(sentence_code)

    sentence_code = np.array(sentence_code)
    np.save('sentence_code_2', sentence_code)  # 存下来



# 定义词向量表
def load_cab_vector():
    word_list = []
    vocabulary_vectors = []
    data = open('E:\IMDB\glove.6B\glove.6B.50d.txt', encoding='utf-8')
    for line in data.readlines():
        temp = line.strip('\n').split(' ')  # 一个列表
        name = temp[0]
        word_list.append(name.lower())
        vector = [temp[i] for i in range(1, len(temp))]  # 向量
        vector = list(map(float, vector))  # 变成浮点数
        vocabulary_vectors.append(vector)
    # 保存
    vocabulary_vectors = np.array(vocabulary_vectors)
    word_list = np.array(word_list)
    np.save('vocabulary_vectors_1', vocabulary_vectors)
    np.save('word_list_1', word_list)
    return vocabulary_vectors, word_list

def main():
    #创建词向量表
    load_cab_vector()
    #文本转序列
    process_sentence()

if __name__ == '__main__':
    main()
