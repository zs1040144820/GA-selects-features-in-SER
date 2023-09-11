import os
import numpy as np
import extract_feats.opensmile as of
import extract_feats.opensmile as of
import extract_feats.librosa as lf
import csv
import models
import utils




def predict(config, model) -> None:
    """
    预测音频情感

    Args:
        config: 配置项
        audio_path (str): 要预测的音频路径
        model: 加载的模型
    """

    # utils.play_audio(audio_path)

    test_feature = of.load_feature(config, train=False)
    result = model.predict(test_feature)
    result_prob = model.predict_proba(test_feature)
    print('Recogntion: ', config.class_labels[int(result)])
    print('Probability: ', result_prob)
    # utils.radar(result_prob, config.class_labels)

if __name__ == '__main__':
    config = utils.parse_opt()
    model = models.load(config)
    predict(config, model)
