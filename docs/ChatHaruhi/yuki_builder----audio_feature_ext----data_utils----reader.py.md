# `.\Chat-Haruhi-Suzumiya\yuki_builder\audio_feature_ext\data_utils\reader.py`

```py
# 导入随机数、系统、日期时间操作相关的库
import random
import sys
from datetime import datetime

# 导入PyTorch深度学习库、音频处理库Librosa、NumPy库
import torch
import librosa
import numpy as np

# 导入PyTorch的数据加载模块
from torch.utils import data


def load_audio(audio_path,
               feature_method='melspectrogram',
               mode='train',
               sr=16000,
               chunk_duration=3,
               min_duration=0.5,
               augmentors=None):
    """
    加载并预处理音频
    :param audio_path: 音频路径
    :param feature_method: 预处理方法，可以是'melspectrogram'或'spectrogram'
    :param mode: 数据处理模式，如'train'（训练）、'eval'（评估）、'infer'（推断）
    :param sr: 音频的采样率
    :param chunk_duration: 每个音频片段的时长（单位：秒）
    :param min_duration: 最小允许的音频长度（单位：秒）
    :param augmentors: 数据增强方法的字典，例如{'specaug': function}
    :return: 预处理后的音频特征
    """
    # 读取音频数据并获取实际的采样率
    wav, sr_ret = librosa.load(audio_path, sr=sr)
    num_wav_samples = wav.shape[0]

    # 如果处理模式为训练，检查音频长度是否满足最小需求
    if mode == 'train':
        if num_wav_samples < int(min_duration * sr):
            raise Exception(f'音频长度小于{min_duration}s，实际长度为：{(num_wav_samples/sr):.2f}s')

    # 如果音频长度小于等于训练或评估所需长度，则通过填充复制来扩展音频
    num_chunk_samples = int(chunk_duration * sr)
    if num_wav_samples <= num_chunk_samples:
        shortage = num_chunk_samples - num_wav_samples
        wav = np.pad(wav, (0, shortage), 'wrap')

    # 根据处理模式进行音频裁剪和数据增强
    if mode == 'train':
        num_wav_samples = wav.shape[0]
        num_chunk_samples = int(chunk_duration * sr)
        if num_wav_samples > num_chunk_samples + 1:
            # 随机选择音频起始位置并裁剪出指定长度的音频片段
            start = random.randint(0, num_wav_samples - num_chunk_samples - 1)
            stop = start + num_chunk_samples
            wav = wav[start:stop]
            
            # 随机对每个音频片段进行部分置零和再次裁剪
            if random.random() > 0.5:
                wav[:random.randint(1, sr // 4)] = 0
                wav = wav[:-random.randint(1, sr // 4)]

        # 如果指定了数据增强方法，则对音频进行增强处理
        if augmentors is not None:
            for key, augmentor in augmentors.items():
                if key == 'specaug':
                    continue  # 跳过specaug数据增强
                wav = augmentor(wav)

    elif mode == 'eval':
        # 在评估模式下，限制音频长度以防止显存溢出
        num_wav_samples = wav.shape[0]
        num_chunk_samples = int(chunk_duration * sr)
        if num_wav_samples > num_chunk_samples + 1:
            wav = wav[:num_chunk_samples]

    # 根据预处理方法获取音频特征
    if feature_method == 'melspectrogram':
        # 计算梅尔频谱特征
        features = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=400, n_mels=80, hop_length=160, win_length=400)
    elif feature_method == 'spectrogram':
        # 计算声谱图特征
        linear = librosa.stft(wav, n_fft=400, win_length=400, hop_length=160)
        features, _ = librosa.magphase(linear)
    else:
        raise Exception(f'预处理方法 {feature_method} 不存在！')

    # 将特征转换为分贝形式，并进行数据增强（如果指定了）
    features = librosa.power_to_db(features, ref=1.0, amin=1e-10, top_db=None)
    if mode == 'train' and augmentors is not None:
        for key, augmentor in augmentors.items():
            if key == 'specaug':
                features = augmentor(features)

    # 对特征进行归一化处理
    mean = np.mean(features, 0, keepdims=True)
    std = np.std(features, 0, keepdims=True)
    features = (features - mean) / (std + 1e-5)

    return features

# 数据加载器的定义暂时未提供
class CustomDataset(data.Dataset):
    """
    加载并预处理音频
    :param data_list_path: 数据列表的文件路径
    :param feature_method: 预处理方法，默认为'melspectrogram'
    :param mode: 数据处理模式，如'train'、'eval'、'infer'
    :param sr: 采样率，默认为16000
    :param chunk_duration: 每个音频片段的长度，默认为3秒
    :param min_duration: 最小音频长度，默认为0.5秒
    :param augmentors: 数据增强方法的列表
    :return: None
    """
    def __init__(self, data_list_path,
                 feature_method='melspectrogram',
                 mode='train',
                 sr=16000,
                 chunk_duration=3,
                 min_duration=0.5,
                 augmentors=None):
        super(CustomDataset, self).__init__()
        # 如果提供了数据列表路径，则读取数据列表文件的每一行作为数据集的行
        if data_list_path is not None:
            with open(data_list_path, 'r') as f:
                self.lines = f.readlines()
        self.feature_method = feature_method
        self.mode = mode
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.min_duration = min_duration
        self.augmentors = augmentors

    def __getitem__(self, idx):
        try:
            # 从数据列表的每一行中提取音频路径和标签
            audio_path, label = self.lines[idx].replace('\n', '').split('\t')
            # 调用load_audio函数加载和预处理音频数据
            features = load_audio(audio_path, feature_method=self.feature_method, mode=self.mode, sr=self.sr,
                                  chunk_duration=self.chunk_duration, min_duration=self.min_duration,
                                  augmentors=self.augmentors)
            return features, np.array(int(label), dtype=np.int64)
        except Exception as ex:
            # 如果出现异常，记录异常信息并返回随机索引的样本
            print(f"[{datetime.now()}] 数据: {self.lines[idx]} 出错，错误信息: {ex}", file=sys.stderr)
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        # 返回数据集的长度，即数据列表的行数
        return len(self.lines)

    @property
    def input_size(self):
        # 返回输入特征的大小，根据预处理方法不同而不同
        if self.feature_method == 'melspectrogram':
            return 80
        elif self.feature_method == 'spectrogram':
            return 201
        else:
            raise Exception(f'预处理方法 {self.feature_method} 不存在！')


# 对一个batch的数据处理
def collate_fn(batch):
    # 根据音频片段的长度降序排序batch中的样本
    batch = sorted(batch, key=lambda sample: sample[0].shape[1], reverse=True)
    # 获取最长音频的频率特征大小和时间步长
    freq_size = batch[0][0].shape[0]
    max_audio_length = batch[0][0].shape[1]
    batch_size = len(batch)
    # 创建全零张量，用于存放批次中所有样本的频率特征数据
    inputs = np.zeros((batch_size, freq_size, max_audio_length), dtype='float32')
    input_lens = []
    labels = []
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        labels.append(sample[1])
        seq_length = tensor.shape[1]
        # 将每个样本的频率特征数据填充到全零张量中，实现padding
        inputs[x, :, :seq_length] = tensor[:, :]
        input_lens.append(seq_length / max_audio_length)
    input_lens = np.array(input_lens, dtype='float32')
    labels = np.array(labels, dtype='int64')
    # 返回填充后的输入张量、标签张量和输入长度列表
    return torch.tensor(inputs), torch.tensor(labels), torch.tensor(input_lens)
```