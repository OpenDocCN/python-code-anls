# `.\Chat-Haruhi-Suzumiya\yuki_builder\crop.py`

```py
# coding: utf-8

import argparse  # 导入用于解析命令行参数的模块
import os  # 导入操作系统相关功能的模块
import re  # 导入正则表达式模块
import subprocess  # 导入子进程管理模块
from collections import Counter  # 导入用于计数的Counter类
import chardet  # 导入用于字符编码检测的模块
import pysrt  # 导入处理SRT字幕文件的模块
import pysubs2  # 导入处理ASS字幕文件的模块
import pickle  # 导入用于序列化数据的模块
from audio_feature_ext.tool import get_filename, get_subdir  # 导入自定义的文件操作函数
import pandas as pd  # 导入数据处理库Pandas
from audio_feature_ext.audio_fea_ext import AudioFeatureExtraction  # 导入音频特征提取类
from tqdm import tqdm  # 导入用于显示进度条的模块


def detect_encoding(file_name):
    # 使用chardet检测文件的编码格式
    with open(file_name, 'rb') as file:
        result = chardet.detect(file.read())
    return result['encoding']


def most_common_element(lst, num=1):
    # 统计列表中出现次数最多的元素
    counter = Counter(lst)
    most = counter.most_common(num)
    return most


def make_filename_safe(filename):
    # 将文件名中的非法字符替换为下划线，并处理多余的空格
    filename = re.sub(r'[\\/:*?"<>|_]', '', filename)
    filename = re.sub(r'\s+', ' ', filename)
    filename = filename.strip()  # 去除开头和结尾的空格
    return filename


class video_Segmentation:
    def __init__(self):
        pass

    def ffmpeg_extract_audio(self, video_input, audio_output, start_time, end_time):
        # 使用ffmpeg从视频中提取音频
        command = ['ffmpeg', '-ss', str(start_time), '-to', str(end_time), '-i', f'{video_input}', "-vn",
                   '-c:a', 'pcm_s16le', '-y', audio_output, '-loglevel', 'quiet']
        subprocess.run(command)

    def extract_pkl_feat(self, audio_extractor, role_audios):
        # 提取指定目录下音频文件的特征，并将特征保存为.pkl文件
        sub_dirs = get_subdir(f'{role_audios}/voice')  # 获取子目录列表

        for dir in sub_dirs[:]:
            voice_files = get_filename(dir)  # 获取目录下的文件列表
            name = os.path.basename(os.path.normpath(dir))  # 获取目录名
            for file, pth in tqdm(voice_files, f'extract {name} audio features ,convert .wav to .pkl'):
                new_dir = os.path.join(role_audios, 'feature', name)  # 创建特征保存的新目录
                os.makedirs(new_dir, exist_ok=True)  # 如果目录不存在则创建
                try:
                    feature = audio_extractor.infer(pth)[0]  # 提取音频特征
                    with open(f"{new_dir}/{file}.pkl", "wb") as f:
                        pickle.dump(feature, f)  # 将特征保存为.pkl文件
                except:
                    continue
        print('音频特征提取完成')

    def extract_new_pkl_feat(self, audio_extractor, input_video, temp_folder):
        # 提取新视频的音频特征并保存为.pkl文件
        file = os.path.basename(input_video)
        filename, format = os.path.splitext(file)  # 分离文件名和扩展名

        sub_dir = f'{temp_folder}/{filename}'  # 设置临时目录路径

        voice_files = get_filename(f'{sub_dir}/voice')  # 获取音频文件列表
        for file, pth in tqdm(voice_files, f'extract {filename} audio features ,convert .wav to .pkl'):
            new_dir = os.path.join(sub_dir, 'feature')  # 设置特征保存的新目录
            os.makedirs(new_dir, exist_ok=True)  # 如果目录不存在则创建
            try:
                feature = audio_extractor.infer(pth)[0]  # 提取音频特征
                with open(f"{new_dir}/{file}.pkl", "wb") as f:
                    pickle.dump(feature, f)  # 将特征保存为.pkl文件
            except:
                continue
        print('音频特征提取完成')
    # 根据给定的 CSV 文件和视频路径，从中提取角色对应的音频片段
    def clip_audio_bycsv(self,annotate_csv,video_pth,role_audios):
        # 设置注释CSV文件路径、视频路径和角色音频路径
        self.annotate_csv = annotate_csv
        self.video_pth = video_pth
        self.role_audios = role_audios
        # 读取CSV文件中的前四列数据
        srt_data = pd.read_csv(self.annotate_csv).iloc[:,:4]
        # 删除空值
        srt_data = srt_data.dropna()
        # 将数据转换成列表
        srt_list = srt_data.values.tolist()
        # 遍历数据列表
        for index, (person,subtitle,start_time,end_time) in enumerate(tqdm(srt_list[:], 'video clip by csv file start')):
            # 设置音频输出路径
            audio_output = f'{self.role_audios}/voice/{person}'
            # 创建目录（如已存在则不创建）
            os.makedirs(audio_output, exist_ok=True)
            # 格式化序号
            index = str(index).zfill(4)
            # 使文件名安全有效
            text = make_filename_safe(subtitle)

            # 设置开始时间（转换为毫秒，补零后转换为12位字符串）
            ss = start_time.zfill(11).ljust(12, '0')[:12]
            # 设置结束时间（转换为毫秒，补零后转换为12位字符串）
            ee = end_time.zfill(11).ljust(12, '0')[:12]

            # 生成音频文件名（带序号、起止时间和文本内容）
            name = f'{index}_{ss}_{ee}_{text}'.replace(':', '.')

            # 设置音频输出路径
            audio_output = f'{audio_output}/{name}.wav'
            # 提取视频中的音频片段
            self.ffmpeg_extract_audio(self.video_pth,audio_output,start_time,end_time)

    # 将给定的秒数转换为SRT格式的时间戳（HH:MM:SS,sss）
    def srt_format_timestamp(self, seconds):
        # 确保秒数为非负数
        assert seconds >= 0, "non-negative timestamp expected"
        # 将秒数转换为毫秒
        milliseconds = round(seconds * 1000.0)

        # 计算小时部分
        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        # 计算分钟部分
        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        # 计算秒部分
        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        # 返回格式化后的时间戳字符串
        return (f"{hours:02d}:") + f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
# 定义一个名为 crop 的函数，接受一个参数 args，该参数包含命令行传入的参数信息
def crop(args):

    # 如果参数中 verbose 属性为真，则打印 'runing crop'
    if args.verbose:
        print('runing crop')

    # 检查 annotate_map 是否是一个文件
    if not os.path.isfile(args.annotate_map):
        # 如果不是文件，则打印出错信息并返回
        print(f'annotate_map {args.annotate_map} is not exist')
        return

    # 检查 role_audios 是否是一个文件夹
    if not os.path.isdir(args.role_audios):
        # 如果不是文件夹，则打印提示信息
        print(f'role_audios {args.role_audios} is not exist')
        # 创建 role_audios 文件夹
        os.mkdir(args.role_audios)

    # 从 annotate_map 文件读取数据，存入 data 变量中
    data = pd.read_csv(args.annotate_map)
    # 创建 video_pth_segmentor 对象用于视频分段
    video_pth_segmentor = video_Segmentation()
    # 创建 audio_feature_extractor 对象用于音频特征提取
    audio_feature_extractor = AudioFeatureExtraction()

    # 遍历 data 变量中的每一行数据，每行数据包含 annotate_csv 和 video_pth 两列
    for index, (annotate_csv, video_pth) in data.iterrows():
        # 根据 subtitle 文件的时间戳剪辑音频段，输出为 *.wav 文件
        # subtitle 文件已经按角色标记，这里是一个以 .csv 格式保存的文件
        video_pth_segmentor.clip_audio_bycsv(annotate_csv, video_pth, args.role_audios)

        # 提取音频特征，将音频文件从 wav 格式转换为 pkl 格式
        video_pth_segmentor.extract_pkl_feat(audio_feature_extractor, args.role_audios)


if __name__ == '__main__':
    # 创建 argparse.ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser(
        description='Extract audio by subtitle time stamp',
        epilog='author:fengyunzaidushi(https://github.com/fengyunzaidushi)'
    )
    # 添加命令行参数 verbose，类型为布尔型
    parser.add_argument("verbose", type=bool, action="store")
    # 添加命令行参数 annotate_map，默认为 './input_folder/haruhi_EP3_annotate_map.csv'，类型为字符串，必须提供
    parser.add_argument('--annotate_map', default='./input_folder/haruhi_EP3_annotate_map.csv', type=str, required=True, help="list of video_pth and subtitle paths")
    # 添加命令行参数 role_audios，默认为 './input_folder/role_audios'，类型为字符串，必须提供，用于存储音频文件和特征文件的目录
    parser.add_argument('--role_audios', default='./input_folder/role_audios', type=str, required=True, help= "audio directories and feature directories categorized by role") # Better change it to your own path
    # parser.add_argument('--model_directory', default='./audio_feature_ext/models', type=str, required=False, help= "huggine face model weight download pth")

    # 解析命令行参数，并将结果存储在 args 变量中
    args = parser.parse_args()
    # 打印命令行参数的帮助信息
    parser.print_help()
    # 调用 crop 函数，传入解析后的命令行参数 args
    crop(args)

"""
cd yuki_builder/
python verbose=True 
        --annotate_map ./input_folder/haruhi_EP3_annotate_map.csv'
        --role_audios ./input_folder/role_audios          # Better change it to your own path
"""
```