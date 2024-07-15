# `.\Chat-Haruhi-Suzumiya\yuki_builder\recognize.py`

```py
# 指定编码为UTF-8，以支持中文等特殊字符
# 导入命令行参数解析模块
import argparse
# 导入操作系统相关模块和正则表达式模块
import os, re
# 导入pickle模块，用于对象序列化和反序列化
import pickle
# 导入K最近邻分类器模型
from sklearn.neighbors import KNeighborsClassifier
# 导入NumPy数学计算库，并使用别名np
import numpy as np
# 导入自定义的音频特征提取工具函数和保存列表到文本文件函数
from audio_feature_ext.tool import get_subdir, get_filelist, save_lis2txt
# 导入音频特征提取类
from audio_feature_ext.audio_fea_ext import AudioFeatureExtraction
# 导入视频分割模块中的视频分割函数
from crop import video_Segmentation
# 导入CSV文件读写操作相关模块
import csv
# 导入文件和目录操作相关模块
import shutil

# K最近邻分类器基于特征矩阵和标签列表进行初始化
class KNN_Classifier_lis:
    def __init__(self, feature, labels, n_neighbors=3):
        self.feature = feature
        self.labels = labels
        # 创建K最近邻分类器对象，使用余弦相似度作为距离度量
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
        # 使用特征矩阵和标签列表训练K最近邻分类器
        self.classifier.fit(self.feature, self.labels)

    def predict(self, x):
        # 预测输入样本的类标签
        predicted_label = self.classifier.predict(x.reshape(1, -1))

        # 获取最近邻的距离和索引
        dists, indices = self.classifier.kneighbors(x.reshape(1, -1))

        # 获取最近邻的类标签列表
        nearest_labels = [self.labels[i] for i in indices[0]]

        # 返回预测的类标签、最近邻的类标签及其对应的距离
        return predicted_label[0], list(zip(nearest_labels, dists[0]))


# K最近邻分类器基于特征矩阵和标签列表进行初始化
class KNN_Classifier:
    def __init__(self, feature, labels, n_neighbors=3):
        self.feature = feature
        self.labels = labels
        # 创建K最近邻分类器对象，使用余弦相似度作为距离度量
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
        # 使用特征矩阵和标签列表训练K最近邻分类器
        self.classifier.fit(self.feature, self.labels)

    def predict(self, x):
        # 预测输入样本的类标签
        predicted_label = self.classifier.predict(x.reshape(1, -1))

        # 获取到最近邻的距离
        dist, _ = self.classifier.kneighbors(x.reshape(1, -1))

        # 返回最常见类别的标签及其最小距离
        return predicted_label[0], dist[0].min()


# 音频分类器管理类
class AudioClassification:
    def __init__(self):
        pass

    # 创建指定类型的分类器实例
    def create_classifier(self, class_name, features, labels, n_neighbors=None):
        # 根据类名动态获取对应的分类器类，并初始化
        classifier_class = globals()[class_name](features, labels, n_neighbors)
        return classifier_class

    # 获取音频特征和对应的标签列表
    def get_feature(self, audio_feature_dir):
        # 初始化特征列表和标签列表
        features = []
        labels = []
        dim = 0
        # 获取音频特征目录下的子目录列表
        role_dirs = get_subdir(audio_feature_dir + '/feature')
        for role_dir in role_dirs:
            # 提取角色名作为标签
            role = os.path.basename(os.path.normpath(role_dir))
            # 获取当前角色目录下的文件列表
            file_list = get_filelist(role_dir)
            for feature_fname in file_list:
                with open(feature_fname, 'rb') as f:
                    # 从文件中加载音频特征
                    feature = pickle.load(f)
                # 将加载的特征添加到特征矩阵中
                if dim == 0:
                    features = feature
                    dim = feature.shape[0]
                else:
                    features = np.vstack((features, feature))

                # 将角色标签添加到标签列表中
                labels.append(role)

        # 返回特征矩阵和标签列表
        return features, labels
    def save_to_csv(self,filename,data):
        # 打开文件，写入数据到CSV文件中，每行数据作为一个列表写入
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
        # 打印保存成功的消息，包含保存的文件名
        print(f'识别结果保存到csv, {filename}')

    def correct_timestamp_format(self, s):
        # 使用正则表达式替换时间戳字符串的第3个点为冒号
        corrected_s = re.sub(r'(\d{2}).(\d{2}).(\d{2}).(\d{3})', r'\1:\2:\3.\4', s)
        return corrected_s

    def save_lis2txt(self,filename, lines):
        # 打开文件，写入每行文本到TXT文件中，每行后添加换行符
        with open(filename, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(str(line) + '\n')
        # 打印保存成功的消息，包含保存的文件名
        print(f'识别结果保存到txt, {filename}')

    def get_pridict(self,role_audios,input_video,output_folder,temp_folder,n_neighbors=3):
        # 从本地加载音频特征和标签
        self.feat_sel, self.label_sel = self.get_feature(role_audios)
        # 使用KNN分类器进行训练，使用给定的邻居数
        self.my_classifier = KNN_Classifier(self.feat_sel, self.label_sel,n_neighbors)

        # 设置阈值，用于确定分类的确信度和疑问度
        threshold_certain = 0.4
        threshold_doubt = 0.6  # 遍历视频切割的目录

        # 获取输入视频文件的基本名称和格式
        file = os.path.basename(input_video)
        filename, format = os.path.splitext(file)  # haruhi_01 .mkv

        # 构建临时目录路径，以视频文件名命名
        sub_dir = f'{temp_folder}/{filename}'

        # 设置CSV和TXT文件的保存路径和名称
        csv_save_name = os.path.join(output_folder, f'{filename}_output.csv')
        txt_save_name = os.path.join(output_folder, f'{filename}_output.txt')
        # 设置特征文件夹路径，用于遍历特征文件
        feature_folder = os.path.join(sub_dir,"feature")  # 遍历特征文件

        # 获取特征文件列表
        file_list = get_filelist(feature_folder)
        # 初始化结果列表，包含表头信息
        res_lis = [['人物','人物台词','开始时间','结束时间']]
        txt_lis = []

        # 遍历特征文件列表
        for file in file_list[:]:
            try:
                # 从文件名中解析出ID和时间信息
                file = os.path.basename(file)
                id_str = file[:-8]
                index,start_time, end_time , text= id_str.split('_')
                full_file_name = os.path.join(feature_folder, file)

                # 打开特征文件，加载特征数据
                with open(full_file_name, 'rb') as f:
                    feature = pickle.load(f)

                # 使用分类器预测角色标签和距离
                predicted_label, distance = self.my_classifier.predict(feature)
                role_name = ''

                # 根据距离设定角色名称，如果距离小于某阈值则加入(可能)标记
                if distance < threshold_certain:
                    role_name = predicted_label
                elif distance < threshold_doubt:
                    role_name = '(可能)' + predicted_label

                # 校正开始和结束时间的时间戳格式
                start_time = self.correct_timestamp_format(start_time)
                end_time = self.correct_timestamp_format(end_time)
                
                # 将结果添加到结果列表和文本列表中
                res_lis.append([role_name, text, start_time, end_time])
                text_content = role_name + ':「' + text + '」'
                txt_lis.append(text_content)
            except:
                continue

        # 将结果保存到CSV和TXT文件中
        self.save_to_csv(csv_save_name,res_lis)
        self.save_lis2txt(txt_save_name,txt_lis)
# 识别函数，接收一个参数 args
def recognize(args):

    # 如果 args.verbose 为真，打印 'runing recognize'
    if args.verbose:
        print('runing recognize')

    # 检查 args.input_video 是否是一个文件
    # 如果不是，打印 'input_video is not exist'，然后返回
    if not os.path.isfile(args.input_video):
        print('input_video is not exist')
        return
    
    # 检查 args.input_srt 是否是一个文件
    # 如果不是，打印 'input_srt is not exist'，然后返回
    if not os.path.isfile(args.input_srt):
        print('input_srt is not exist')
        return
    
    # 检查 args.role_audios 是否是一个文件夹
    # 如果不是，打印 'role_audios is not exist'，然后返回
    if not os.path.isdir(args.role_audios):
        print('role_audios is not exist')
        return

    # 检查 args.output_folder 是否是一个文件夹
    # 如果不是，打印 'warning output_folder is not exist'
    # 并且创建 args.output_folder 文件夹，然后打印 'create folder' + args.output_folder
    if not os.path.isdir(args.output_folder):
        print('warning output_folder is not exist')
        os.mkdir(args.output_folder)
        print('create folder', args.output_folder)

    # 在 args.role_audios 目录下创建临时文件夹 temp_folder
    temp_dir = os.path.dirname(args.role_audios)
    temp_folder = f'{temp_dir}/temp_folder'
    os.makedirs(temp_folder, exist_ok=True)

    # 根据字幕文件时间戳剪辑音频片段，输出为 *.wav 文件
    # 未经角色标记的字幕文件
    video_pth_segmentor = video_Segmentation()
    video_pth_segmentor.clip_video_bysrt(args.input_video,args.input_srt,temp_folder)

    # 提取音频特征，从 wav 文件生成 pkl 文件
    audio_feature_extractor = AudioFeatureExtraction()
    video_pth_segmentor.extract_new_pkl_feat(audio_feature_extractor, args.input_video,temp_folder)

    # 角色分类
    audio_classification = AudioClassification()
    audio_classification.get_pridict(args.role_audios,args.input_video,args.output_folder,temp_folder)

    # 删除临时文件夹
    # shutil.rmtree(temp_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract audio by subtitle time stamp',
        epilog='author:fengyunzaidushi(https://github.com/fengyunzaidushi)'
    )
    # 添加命令行参数 verbose, input_video, input_srt, role_audios, output_folder
    parser.add_argument("verbose", type=bool, action="store")
    parser.add_argument('--input_video', default='input_file', type=str, required=True, help="video path")
    parser.add_argument('--input_srt', default='input_srt', type=str, required=True,help="path of input .srt/.ass file")
    parser.add_argument('--role_audios', default='./input_folder/role_audios', type=str, required=True, help= "audio directories and feature folder categorized by role") # Better to change the default folder
    parser.add_argument('--output_folder', default='./output_folder', type=str, required=False,
                        help="the output_folder role recoginize csv file")

    args = parser.parse_args()
    parser.print_help()
    recognize(args)

"""
cd yuki_builder/
python verbose=True 
        --input_video Haruhi_16.mkv
        --input_srt Haruhi_16.srt
        --role_audios ./input_folder/role_audios  # Better change it to your own path
        --output_folder ./data_crop  # You can change it to your own path
"""
```