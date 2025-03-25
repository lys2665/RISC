import heapq
from numpy import random as rd
import numpy as np
import scipy.stats as stats
import collections
import time
import random
import copy
import  math
'''def purity(cluster, label):
    cluster = np.array(cluster)
    label = np. array(label)
    indedata1 = {}
    for p in np.unique(label):
        indedata1[p] = np.argwhere(label == p)
    indedata2 = {}
    for q in np.unique(cluster):
        indedata2[q] = np.argwhere(cluster == q)

    count_all = []
    for i in indedata1.values():
        count = []
        for j in indedata2.values():
            a = np.intersect1d(i, j).shape[0]
            count.append(a)
        count_all.append(count)
    return sum(np.max(count_all, axis=0))/len(cluster)
def true_label():
    file_path1='C:/dasixia/dataset/activity.txt'
    true_labels=[]
    file=open(file_path1)
    for line in file:
        data=line.split(' ')
        true_labels.append((data[0]))
    return true_labels'''
qq=50
w=30
numP=200
minNumP=50
k=2
file_path='dataset/webkb.txt'
data=[]

with open(file_path) as file_object:
    for line in file_object:
        line=line.replace('\n','')
        data.append(line.split(' '))
flag=0
maxl=8
len_data=len(data)

#all_label=[]


ini_clu = [i for i in range(len_data)]


def if_or_not(str1, str2):
    flag = 1
    remain_str = copy.deepcopy(str2)
    for i in range(len(str1)):
        if str1[i] in remain_str:
            index = remain_str.index(str1[i])
            remain_str = remain_str[index + 1:]
                # print(remain_str)
        else:
            flag = 0
    return flag


time_start = time.time()
while flag == 0:
    save_pat = []
    thirty_pattern = []
    for i in range(numP):
        x = random.randint(0, len_data - 1)
        if len(data[x]) >= maxl:
            start_pos = random.randint(0, len(data[x]) - maxl)
            random_sel_pattern = data[x][start_pos:start_pos + maxl]
            thirty_pattern.append(random_sel_pattern)
    # print(thirty_pattern)
    portion = 0
    for i in range(len(thirty_pattern)):
        sum_num = 0
        for j in range(len_data):
            if if_or_not(thirty_pattern[i], data[j]):
                sum_num = sum_num + 1
        # print("当前pattern的support：",thirty_pattern[i],sum_num)
        if sum_num >= 0.2 * len_data and sum_num <= 0.8 * len_data:
            portion = portion + 1
            save_pat.append(thirty_pattern[i])
    # print("满足的pattern数量：",portion)
    if maxl == 1:
        break
    if portion >= minNumP:
        flag = 1
    else:
        maxl = maxl - 1
print("符合要求的pattern集合", len(save_pat), save_pat)
print("最终选取pattern长度：", maxl)
