# 遗传算法找最佳适应度，没有设置特征个数
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" ###指定此处为-1即可
# 个人电脑cudnn设置错误，开启gpu比不开启更慢，故关闭。

import pandas as pd
import matplotlib.pyplot as plt
import random

import datetime


from tensorflow.keras.utils import to_categorical
import extract_feats.opensmile as of
import models
from utils import parse_opt
import numpy as np


# 自带数据
# data = pd.read_csv('./dataset/sonar.all-data',header=None,sep=',')
# print(data)
# X = data.iloc[:,:-1]
# y = data.iloc[:,-1:].values.flatten()
# print(X)
# print(y)

# 自己数据
data = pd.read_csv('./dataset/train.csv',header=None)
X = data.iloc[:,:-1]
y = data.iloc[:,-1:].values.flatten()

iterations = 3 # 迭代次数
pop_size = 3   # 种群大小，多少个染色体
pc = 0.8   # 交叉概率
pm = 0.01   # 变异概率
num = 200 # 初始化选中的特征个数约数
kp = 1 # 是否为继续上次训练 1是，0不是


chrom_length = len(data.columns)-1    # 染色体长度
pop = []    # 种群
fitness_list = []   # 适应度
ratio_list = []     # 累计概率

# shap排名
shapRank = [198,193,78,34,21,224,17,200,186,344,40,191,366,38,199,35,15,39,14,77,7,5,54,45,178,58,179,64,10,264,114,99,248,111,65,60,238,41,30,382,33,256,102,205,171,44,96,214,49,12,310,137,89,370,113,338,334,20,121,95,368,369,109,87,173,0,52,6,208,82,220,207,31,312,112,364,100,271,66,270,339,203,255,59,363,210,146,194,367,70,223,42,37,28,240,204,68,257,267,51,62,23,260,250,346,69,97,347,302,46,263,304,71,26,48,151,107,296,98,90,277,354,24,29,282,56,36,377,211,3,53,195,166,57,221,336,79,232,284,50,138,311,315,235,32,127,8,11,27,175,81,144,291,192,101,247,187,342,174,16,103,228,380,308,133,55,154,80,91,150,134,196,330,328,373,120,130,88,234,206,157,218,316,356,149,136,262,348,63,227,106,123,197,125,292,215,83,319,272,116,162,222,251,329,225,142,371,236,323,47,212,372,294,176,84,169,280,108,140,298,301,295,25,241,275,164,226,85,67,105,243,131,233,115,253,305,231,141,246,383,324,258,287,229,110,135,13,293,314,242,94,365,375,160,341,75,239,307,379,332,276,18,219,93,155,244,331,117,43,252,327,335,147,104,306,148,274,374,86,118,61,167,349,269,74,72,201,337,283,145,124,320,217,378,73,266,139,76,158,190,128,216,22,245,318,152,265,119,359,286,268,299,322,132,92,129,345,259,180,340,249,290,313,289,170,122,278,165,362,300,143,325,281,360,352,19,355,351,185,4,254,279,326,357,183,358,288,163,353,273,376,333,317,361,161,209,261,350,230,168,343,202,159,188,153,189,177,1,303,285,172,297,321,213,309,126,156,9,2,181,182,184,237,381]


# 初始化种群
# def geneEncoding():
#     i = 0
#     while i < pop_size:
#         temp = []
#         has_1 = False   # 这条染色体是否有1
#         for j in range(chrom_length):
#             rand = random.randint(0,1)
#             if rand == 1:
#                 has_1 = True
#             temp.append(rand)
#         if has_1:   # 染色体不能全0
#             i += 1
#             pop.append(temp)
# 初始化种群与可解释性相结合
def geneEncoding():
    i = 0
    while i < pop_size:
        temp = []
        has_1 = False   # 这条染色体是否有1
        for j in range(chrom_length):
            jIndex = shapRank.index(j) #返回在shap中的排序0-383
            jState = (chrom_length//num + 1)-(jIndex//num + 1)+1 # 对应权重
            # 计算出现概率
            randSum = 0 #权重总和
            for k in range((chrom_length//num + 1)):
                randSum += (k+1)
            rand = random.randint(1, randSum) # 包含1和36
            if(rand <= jState):
                has_1 = True
                temp.append(1)
            else:
                temp.append(0)
        if has_1:   # 染色体不能全0
            i += 1
            pop.append(temp)

# 读取上次数据 继续训练
def geneEncodingKeepLast():
    f = open("./sample_output/GAPop.txt")  # 返回一个文件对象
    line = f.readline()  # 调用文件的 readline()方法,把第一行代表第几代的数据丢掉
    pop = []
    while True:
        line = f.readline()
        if not line:  # 读取的最后一行EOF丢掉
            break
        line = line.strip('\n')
        popI = eval(line)
        pop.append(popI)
    f.close()
    return pop


# 计算适应度
# def calFitness():
#     fitness_list.clear()
#     for i in range(pop_size):   # 计算种群中每条染色体的适应度
#         X_test = X
#
#         has_1 = False
#         for j in range(chrom_length):
#             if pop[i][j] == 0:
#                 X_test =X_test.drop(columns = j)
#             else:
#                 has_1 = True
#         X_test = X_test.values
#
#         if has_1:
#             clf = tree.DecisionTreeClassifier() # 决策树作为分类器
#             fitness = cross_val_score(clf, X_test, y, cv=5).mean()  # 5次交叉验证
#             fitness_list.append(fitness)
#         else:
#             fitness = 0     # 全0的适应度为0
#             fitness_list.append(fitness)
# 计算适应度与自己模型相结合
def calFitness():
    fitness_list.clear()
    for i in range(pop_size):   # 计算种群中每条染色体的适应度
        # has_1 = False
        # X_test = X
        # for j in range(chrom_length):
        #     if pop[i][j] == 0:
        #         X_test =X_test.drop(columns = j)
        #     else:
        #         has_1 = True
        # X_test = X_test.values


        # 将数据划分为自己模型数据    *可能需要原本自己数据划分*
        has_1 = False
        config = parse_opt()
        x_train, x_test, y_train, y_test = of.load_feature(config, train=True)
        sum = 0 # 用于填补删除的列的位移
        for j in range(chrom_length):
            if pop[i][j] == 0:
                x_train = np.delete(x_train, j-sum, 1)
                x_test = np.delete(x_test, j-sum, 1)
                sum += 1
            else:
                has_1 = True

        if has_1:
            # 原来训练过程
            # clf = tree.DecisionTreeClassifier()  # 决策树作为分类器
            # fitness = cross_val_score(lstm, X_test, y, cv=5).mean()  # 5次交叉验证
            # 与自己模型结合过程
            model = models.make(config=config, n_feats=x_train.shape[1])
            print('----- start training', config.model, '-----')
            if config.model in ['lstm', 'cnn1d', 'cnn2d']:
                y_train, y_val = to_categorical(y_train), to_categorical(y_test)  # 独热编码
                model.train(
                    x_train, y_train,
                    x_test, y_val,
                    batch_size=config.batch_size,
                    n_epochs=config.epochs
                )
            else:
                model.train(x_train, y_train)
            print('----- end training ', config.model, ' -----')
            fitness = model.evaluate(x_test, y_test)
            fitness_list.append(fitness)
        else:
            fitness = 0     # 全0的适应度为0
            fitness_list.append(fitness)

# 计算适应度的总和
def sumFitness():
    total = 0
    for i in range(pop_size):
        total += fitness_list[i]
    return total

# 计算每条染色体的累计概率
def getRatio():
    ratio_list.clear()
    ratio_list.append(fitness_list[0])
    for i in range(1, pop_size):
        ratio_list.append(ratio_list[i-1] + fitness_list[i])
    ratio_list[-1] = 1

# 选择
def selection():
    global pop
    total_fitness = sumFitness()
    for i in range(pop_size):
        fitness_list[i] = fitness_list[i] / total_fitness
    getRatio()

    rand_ratio = [] # 随机概率
    for i in range(pop_size):
        rand_ratio.append(random.random())
    rand_ratio.sort()

    new_pop = []    # 新种群
    i = 0  # 已经处理的随机概率数
    j = 0  # 超出范围的染色体数

    while i < pop_size:
        if rand_ratio[i] < ratio_list[j]:   # 随机数在第j个染色体的概率范围内
            new_pop.append(pop[j])
            i += 1
        else:
            j += 1

    pop = new_pop

# 交叉
def crossover():
    for i in range(pop_size-1): # 若交叉，则染色体i与染色体i+1交叉
        if random.random() < pc:# 发生交叉
            cpoint = random.randint(0, chrom_length-1)    # 随机选择交叉点
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][:cpoint])
            temp1.extend(pop[i+1][cpoint:])
            temp2.extend(pop[i+1][:cpoint])
            temp2.extend(pop[i][cpoint:])
            pop[i] = temp1
            pop[i+1] = temp2

# 变异
def mutation():
    for i in range(pop_size):
        if random.random() < pm: # 发生变异
            mpoint = random.randint(0, chrom_length-1)  # 随机选择变异点
            if pop[i][mpoint] == 1:
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1

# 最优解
def getBest():
    best_chrom = pop[0]
    best_fitness = fitness_list[0]
    for i in range(1,pop_size):
        if fitness_list[i] > best_fitness:
            best_fitness = fitness_list[i]  # 最佳适应值
            best_chrom = pop[i] # 最佳染色体

    return best_chrom, best_fitness

# 自写所需方法
# 计算平均每条染色体选中特征数
def evalAverageFeature():
    sum = 0
    for i in range(len(pop)):
        count =0
        for j in range(len(pop[i])):
            if(pop[i][j] == 1):
                count +=1
        sum+=count;
    sum = sum/pop_size
    print(sum)
    print("\n")

# 返回某条染色体上选中的特征数
def evalChromFeature(chrom):
    count = 0
    for j in range(len(chrom)):
        if (chrom[j] == 1):
            count += 1
    return count


if __name__=='__main__':

    plt.xlabel('iterations')
    plt.ylabel('best fitness')
    plt.xlim((0,iterations-1))    # x坐标范围
    plt.ylim((0,1)) # y坐标范围
    px = []
    py = []
    plt.ion()

    results = []

    bestAllChrom = []  # 记录下所有过程中最好适应度的个体，适应度、选中特征数目和出现轮次
    bestAllFitness = 0
    bestAllIterations = 0
    bestAllChromFeature = 0
    bestAll = []  # 保存排名考前的最优解

    if(kp == 0):
        geneEncoding()
    else:
        pop = geneEncodingKeepLast()

    # 记录输出结果
    file_handle = open('./sample_output/GAResult.txt', mode='a+')
    file_handle.seek(0)
    file_handle.truncate()



    startTime = datetime.datetime.now()
    print("程序训练开始时间：")
    print(startTime)

    for i in range(iterations):


        print("当前轮次：")
        print(i+1)
        print("平均染色体个数：")
        evalAverageFeature()
        print("种群代数：",i+1)
        # 写入每代种群
        file_handle1 = open('./sample_output/GAPop.txt', mode='a+')
        file_handle1.seek(0)
        file_handle1.truncate()
        file_handle1.writelines(str(i+1)+"\n")
        for j in range(pop_size):
            file_handle1.writelines(str(pop[j])+"\n")
            file_handle1.flush()
        # 计算种群中每条染色体适应度
        calFitness()
        # 获得每代最优个体
        best_chrom, best_fitness = getBest()
        results.append([i+1, best_chrom, best_fitness])

        # 记录下所有过程中曾出现过最好适应度的个体，适应度、选中特征数目和出现轮次
        if (best_fitness > bestAllFitness or best_fitness == bestAllFitness):
            bestAllChrom = best_chrom
            bestAllFitness = best_fitness
            bestAllChromFeature = evalChromFeature(best_chrom)
            bestAllIterations = i+1
            outputBest = str([bestAllIterations, bestAllFitness, bestAllChromFeature, bestAllChrom])
            bestAll.append(outputBest)
        output = "Present Time: " + str([i + 1, best_fitness, evalChromFeature(best_chrom), best_chrom]) + "\n"
        output1 = "All Time: " + str([bestAllIterations, bestAllFitness, bestAllChromFeature, bestAllChrom]) + "\n"
        print(output)
        print(output1)
        file_handle.writelines(output)

        selection() # 选择
        crossover() # 交叉
        mutation()  # 变异

        px.append(i)    # 画图
        py.append(best_fitness)
        plt.plot(px,py)
        plt.savefig("sample_output/GAResult.png")
        plt.show()
        plt.pause(0.001)

    # 训练完足够次数再写入所有过程中最好适应度的个体，适应度、选中特征数目和出现轮次，要不太乱
    file_handle.writelines(output1)
    # 训练完足够次数再写入所有过程中突破过记录的个体，适应度、选中特征数目和出现轮次，要不太乱
    bestAll.reverse()  # 列表反转，大的考前
    print("the rank of bestAll")
    print(len(bestAll))
    file_handle.writelines("the rank of bestAll"+"\n")
    file_handle.writelines(str(len(bestAll)) + "\n")
    for i in range(len(bestAll)):
        print(bestAll[i])
        file_handle.writelines(bestAll[i]+"\n")
    file_handle.close()
    file_handle1.close()

    # 算一下花费时间
    print("程序训练开始时间：")
    print(startTime)
    endTime = datetime.datetime.now()
    print("程序训练结束时间：")
    print(endTime)
    print("程序花费时间：")
    print(endTime - startTime)






