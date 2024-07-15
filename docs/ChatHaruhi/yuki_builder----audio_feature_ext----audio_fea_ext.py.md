# `.\Chat-Haruhi-Suzumiya\yuki_builder\audio_feature_ext\audio_fea_ext.py`

```py
import os  # 导入操作系统模块
import numpy as np  # 导入NumPy库，用于数值计算
import torch  # 导入PyTorch深度学习框架
import pickle  # 导入pickle模块，用于序列化和反序列化对象

import requests  # 导入requests模块，用于HTTP请求
from .modules.ecapa_tdnn import EcapaTdnn, SpeakerIdetification  # 从本地模块导入EcapaTdnn和SpeakerIdetification类
from .data_utils.reader import load_audio, CustomDataset  # 从本地模块导入load_audio和CustomDataset函数

class AudioFeatureExtraction:
    def __init__(self, model_director='./audio_feature_ext/models', audio_duration=3, feature_method='melspectrogram'):
        self.use_model = ''  # 初始化空字符串，用于记录使用的模型
        self.audio_duration = audio_duration  # 设置音频片段的持续时间
        self.model_director = model_director  # 指定模型保存和加载的目录路径
        self.feature_method = feature_method  # 设置特征提取方法
        self.model = None  # 初始化模型为空
        self.device = None  # 初始化设备为空
        self.load_model()  # 加载模型

    def init_models(self, path):
        # 预定义模型文件的URL列表
        model_urls = ['https://huggingface.co/scixing/voicemodel/resolve/main/model.pth',
                      'https://huggingface.co/scixing/voicemodel/resolve/main/model.state',
                      'https://huggingface.co/scixing/voicemodel/resolve/main/optimizer.pth']
        listdir = os.listdir(path)  # 获取目录下所有文件列表
        for url in model_urls:
            filename = url.split('/')[-1]  # 提取URL中的文件名部分
            if filename in listdir:  # 如果文件名已存在于目录中，则跳过下载
                continue
            r = requests.get(url, allow_redirects=True)  # 发起HTTP GET请求获取模型文件
            print(f'downloading model pth {filename}')  # 打印下载进度信息
            open(f'{path}/{filename}', 'wb').write(r.content)  # 将下载的内容写入文件
            print(f'{filename} success download')  # 打印下载成功信息

    def load_model(self):
        # 创建自定义数据集对象，使用特定的特征提取方法
        dataset = CustomDataset(data_list_path=None, feature_method=self.feature_method)
        # 创建EcapaTdnn模型对象，指定输入尺寸为数据集的输入尺寸
        ecapa_tdnn = EcapaTdnn(input_size=dataset.input_size)
        # 创建SpeakerIdetification模型对象，使用EcapaTdnn作为骨干网络
        self.model = SpeakerIdetification(backbone=ecapa_tdnn)
        self.device = torch.device("cuda")  # 指定使用CUDA加速计算
        self.model.to(self.device)  # 将模型移动到指定设备（GPU）

        if not os.path.exists(self.model_director):  # 如果模型保存目录不存在，则创建该目录
            os.makedirs(self.model_director)
        model_files = ['model.pth', 'model.state', 'optimizer.pth']  # 预定义模型文件名列表
        for file in model_files:
            if not os.path.exists(f'{self.model_director}/{file}'):  # 如果模型文件不存在，则初始化模型
                self.init_models(self.model_director)

        # 加载模型参数
        model_path = os.path.join(self.model_director, 'model.pth')  # 模型文件路径
        model_dict = self.model.state_dict()  # 获取模型的参数字典
        param_state_dict = torch.load(model_path)  # 加载模型参数的状态字典
        for name, weight in model_dict.items():
            if name in param_state_dict.keys():
                if list(weight.shape) != list(param_state_dict[name].shape):
                    param_state_dict.pop(name, None)  # 移除不匹配的参数项
        self.model.load_state_dict(param_state_dict, strict=False)  # 加载模型参数字典到模型中
        print(f"成功加载模型参数和优化方法参数：{model_path}")  # 打印加载成功信息
        self.model.eval()  # 设置模型为评估（推断）模式

    def infer(self, audio_path):
        # 加载音频数据，使用指定的推断模式和特征提取方法
        data = load_audio(audio_path, mode='infer', feature_method=self.feature_method,
                          chunk_duration=self.audio_duration)
        data = data[np.newaxis, :]  # 添加一个维度，用于模型推断
        data = torch.tensor(data, dtype=torch.float32, device=self.device)  # 转换为PyTorch张量，并移动到设备上
        feature = self.model.backbone(data)  # 提取音频特征
        return feature.data.cpu().numpy()  # 将特征数据移回CPU并转换为NumPy数组
    # 提取音频特征的方法，接收一个根目录作为参数
    def extract_features(self, root_dir):
        # 获取根目录下所有子目录的列表
        sub_dirs = get_subdir(root_dir)

        # 遍历每一个子目录
        for dir in sub_dirs[:]:
            # 获取当前子目录下所有音频文件的文件名和路径
            voice_files = get_filename(os.path.join(dir, 'voice'))

            # 遍历当前子目录下的每个音频文件
            for file, pth in voice_files:
                # 创建存放特征文件的新目录，如果已存在则忽略
                new_dir = os.path.join(dir, 'feature')
                os.makedirs(new_dir, exist_ok=True)

                try:
                    # 使用预测模型提取音频文件的特征
                    feature = self.infer(pth)[0]
                    
                    # 将提取得到的特征序列化并保存为.pkl文件
                    with open(f"{new_dir}/{file}.pkl", "wb") as f:
                        pickle.dump(feature, f)
                except:
                    # 如果提取特征出现异常，则跳过当前文件继续处理下一个
                    continue

        # 所有子目录的音频特征提取完成后打印提示信息
        print('音频特征提取完成')

    # 提取.pkl文件特征的方法，接收一个根目录作为参数
    def extract_pkl_feat(self, root_dir):
        # 获取根目录下所有子目录的列表
        sub_dirs = get_subdir(root_dir)

        # 遍历每一个子目录
        for dir in sub_dirs[:]:
            # 获取当前子目录下所有音频文件的文件名和路径
            voice_files = get_filename(os.path.join(dir, 'voice'))

            # 遍历当前子目录下的每个音频文件
            for file, pth in voice_files:
                # 创建存放特征文件的新目录，如果已存在则忽略
                new_dir = os.path.join(dir, 'feature')
                os.makedirs(new_dir, exist_ok=True)

                try:
                    # 使用预测模型提取音频文件的特征
                    feature = self.infer(pth)[0]
                    
                    # 将提取得到的特征序列化并保存为.pkl文件
                    with open(f"{new_dir}/{file}.pkl", "wb") as f:
                        pickle.dump(feature, f)
                except:
                    # 如果提取特征出现异常，则跳过当前文件继续处理下一个
                    continue

        # 所有子目录的.pkl文件特征提取完成后打印提示信息
        print('音频特征提取完成')
# 如果作为主程序运行（而不是作为模块导入），则执行以下代码块
if __name__ == '__main__':
    # 在此处放置主程序的逻辑，例如函数调用、数据处理等
    pass
```