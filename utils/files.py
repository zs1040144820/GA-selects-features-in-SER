"""数据集整理"""

import os
import shutil
import traceback

def remove(file_path: str) -> None:
    """批量删除指定路径下所有非 `.wav` 文件"""
    for root, dirs, files in os.walk(file_path):
        for item in files:
            if not item.endswith('.wav'):
                try:
                    print("Delete file: ", os.path.join(root, item))
                    os.remove(os.path.join(root, item))
                except:
                    continue


def rename(file_path: str) -> None:
    """批量按指定格式改名（不然把相同情感的音频整理到同一个文件夹时会重名）"""
    for root, dirs, files in os.walk(file_path):
        for item in files:
            if item.endswith('.wav'):
                people_name = root.split('/')[-2]
                emotion_name = root.split('/')[-1]
                item_name = item[:-4]  # 音频原名（去掉.wav）
                old_path = os.path.join(root, item)
                new_path = os.path.join(root, item_name + '-' + emotion_name + '-' + people_name + '.wav')  # 新音频路径
                try:
                    os.rename(old_path, new_path)
                    print('converting ', old_path, ' to ', new_path)
                except:
                    continue


def move(file_path: str) -> None:
    """把音频按情感分类，放在不同文件夹下"""
    for root, dirs, files in os.walk(file_path):
        for item in files:
            if item.endswith('.wav'):
                emotion_name = root.split('/')[-1]
                old_path = os.path.join(root, item)
                new_path = os.path.join(file_path, emotion_name, item)
                try:
                    shutil.move(old_path, new_path)
                    print("Move ", old_path, " to ", new_path)
                except:
                    continue


def mkdirs(folder_path: str) -> None:
    """检查文件夹是否存在，如果不存在就创建一个"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


# CASIA,自定义函数处理文件语音. 名称必须按照，原名-情感-人名处理，应该是情感必须放到第二个。

# def rename1(path, emotion, name):
#     for root, dirs, files in os.walk(path):
#         for item in files:
#             try:
#                 item_name = item[:-4]  # 去掉.wav
#                 old_path = os.path.join(root, item)
#                 new_path = os.path.join(root, item_name + '-' + emotion + '-' + name + '.wav')  # 新音频名称
#                 os.rename(old_path, new_path)
#                 print('converting ', old_path, ' to ', new_path)
#             except:
#                 continue
# # 语音者姓名
# name = ("liuchanhg", "wangzhe", "zhaoquanyin", "ZhaoZuoxiang")
#
# # 情感名称
# emotion = ("angry", "fear", "happy", "neutral", "sad", "surprise")
#
# for i in range(0, 4):
#     for j in range(0, 6):
#         path = r"D:\SoftwareData\PyCharmData\Z\Speech-Emotion-Recognition-master\configs\datasets\CASIA" + '\\' + name[i] + '\\' + emotion[j]
#         remove(path)
#         rename1(path, emotion[j], name[i])





# RAVDESS,自定义函数处理文件语音. 名称必须按照，人名_情绪代码-情感-编号（编号防止重复）处理，应该是情感必须放到第二个。
# 情感名称
# emotion = ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised")
#
# # 将file文件(带后缀名)从src_path移动到dst_path
# def move_file(src_path, dst_path, file):
#     print ('from : ',src_path)
#     print ('to : ',dst_path)
#     try:
#         # cmd = 'chmod -R +x ' + src_path
#         # os.popen(cmd)
#         f_src = os.path.join(src_path)
#         if not os.path.exists(dst_path):
#             os.mkdir(dst_path)
#         f_dst = os.path.join(dst_path, file)
#         print(f_src+"-----"+f_dst)
#         shutil.move(f_src, f_dst)
#     except Exception as e:
#         print ('move_file ERROR: ',e)
#         traceback.print_exc()
#
# def move_Allfile(path):
#     for root, dirs, files in os.walk(path):
#         for item in files:
#             try:
#                 emotion_num = int(item[9:11])  # 查找情绪代码
#                 old_path = os.path.join(root, item)
#                 new_path = "D:\SoftwareData\PyCharmData\Z\Speech-Emotion-Recognition-master\configs\datasets\RAVDESS\\" + emotion[int(emotion_num - 1)]  # 新音频路径
#                 move_file(old_path, new_path, item)
#             except:
#                 continue
#
#
# def rename2(path, name):
#     for root, dirs, files in os.walk(path):
#         i = 1
#         for item in files:
#             try:
#                 emotion_num = int(item[6:8])  # 查找情绪代码
#                 old_path = os.path.join(root, item)
#                 new_path = os.path.join(root, name + "_" + item[6:8] + '-' + emotion[emotion_num - 1] + '-' + str(i) + '.wav')  # 新音频名称
#                 os.rename(old_path, new_path)
#                 print('converting ', old_path, ' to ', new_path)
#                 i = i + 1
#             except:
#                 continue
# for i in range(1, 25):
#     name = "Actor_" + str(str(i).zfill(2))
#     path = r"D:\SoftwareData\PyCharmData\Z\Speech-Emotion-Recognition-master\configs\datasets\RAVDESS" + '\\' + name
#     print(path)
#     rename2(path, name)
#     move_Allfile(path)



