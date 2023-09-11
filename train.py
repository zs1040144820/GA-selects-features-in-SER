from tensorflow.keras.utils import to_categorical
import extract_feats.opensmile as of
import extract_feats.librosa as lf
import models
from utils import parse_opt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf

def train(config) -> None:
    """
    训练模型

    Args:
        config: 配置项

    Returns:
        model: 训练好的模型
    """

    # 加载被 preprocess.py 预处理好的特征
    if config.feature_method == 'o':
        x_train, x_test, y_train, y_test = of.load_feature(config, train=True)

    elif config.feature_method == 'l':
        x_train, x_test, y_train, y_test = lf.load_feature(config, train=True)

    # x_train, x_test (n_samples, n_feats)
    # y_train, y_test (n_samples)

    # 搭建模型
    model = models.make(config=config, n_feats=x_train.shape[1])

    # 训练模型
    print('----- start training', config.model, '-----')
    if config.model in ['lstm', 'cnn1d', 'cnn2d']:
        y_train, y_val = to_categorical(y_train), to_categorical(y_test)  # 独热编码
        model.train(
            x_train, y_train,
            x_test, y_val,
            batch_size = config.batch_size,
            n_epochs = config.epochs
        )
    else:
        model.train(x_train, y_train)
    print('----- end training ', config.model, ' -----')

    # 验证模型
    accuracy = model.evaluate(x_test, y_test)
    # 保存训练好的模型
    model.save(config.checkpoint_path, config.checkpoint_name)

    return  accuracy

if __name__ == '__main__':
    time = 3    # 训练次数
    accs = []   # 存放全部正确率
    sum = 0     # 所有正确总和，用于求平均
    for i in range(time):
        config = parse_opt()    #指定模型在这里面
        acci = train(config)
        sum = sum + acci
        accs.append(acci)
    print(accs)
    print(sum/time)

