import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import tree
from sklearn.model_selection import cross_val_score, train_test_split
from collections.abc import Iterable
from collections import defaultdict
import warnings
from itertools import chain, combinations
from math import ceil, floor
import numbers
from abc import ABCMeta, abstractmethod
from inspect import signature
import numpy as np
from scipy.special import comb
import datetime


from tensorflow.keras.utils import to_categorical
import extract_feats.opensmile as of
import extract_feats.librosa as lf
import models
from utils import parse_opt
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import torch
import torch.nn as nn
# 数据为excel中数据标准化，一一对应，特征也一一对应，删除对应特征也已验证
config = parse_opt()
x_train, x_test, y_train, y_test = of.load_feature(config, train=True)
chrom = [0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
sum = 0
for i in range(len(chrom)):
    if chrom[i] == 0:
        x_train = np.delete(x_train, i - sum, 1)
        x_test = np.delete(x_test, i - sum, 1)
        sum += 1
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
print(fitness)
