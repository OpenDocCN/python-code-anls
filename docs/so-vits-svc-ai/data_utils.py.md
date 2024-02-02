# `so-vits-svc\data_utils.py`

```py
# 导入必要的库
import os
import random
import numpy as np
import torch
import torch.utils.data
import utils
from modules.mel_processing import spectrogram_torch
from utils import load_filepaths_and_text, load_wav_to_torch

# 定义一个多说话者版本的数据加载器类
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) 加载音频、说话者ID、文本对
        2) 标准化文本并将其转换为整数序列
        3) 从音频文件计算频谱图
    """

    # 初始化函数，接受音频路径、超参数和是否全部加载到内存中的标志
    def __init__(self, audiopaths, hparams, all_in_mem: bool = False, vol_aug: bool = True):
        # 加载音频路径和对应的文本
        self.audiopaths = load_filepaths_and_text(audiopaths)
        # 设置超参数
        self.hparams = hparams
        # 设置最大音频值
        self.max_wav_value = hparams.data.max_wav_value
        # 设置采样率
        self.sampling_rate = hparams.data.sampling_rate
        # 设置滤波器长度
        self.filter_length = hparams.data.filter_length
        # 设置跳跃长度
        self.hop_length = hparams.data.hop_length
        # 设置窗口长度
        self.win_length = hparams.data.win_length
        # 设置单位插值模式
        self.unit_interpolate_mode = hparams.data.unit_interpolate_mode
        # 再次设置采样率（此处可能有误，需要根据实际情况调整）
        self.sampling_rate = hparams.data.sampling_rate
        # 设置是否使用采样率
        self.use_sr = hparams.train.use_sr
        # 设置频谱图长度
        self.spec_len = hparams.train.max_speclen
        # 设置说话者映射
        self.spk_map = hparams.spk
        # 设置音量嵌入
        self.vol_emb = hparams.model.vol_embedding
        # 设置音量增强
        self.vol_aug = hparams.train.vol_aug and vol_aug
        # 设置随机种子
        random.seed(1234)
        # 随机打乱音频路径
        random.shuffle(self.audiopaths)
        
        # 设置是否全部加载到内存中的标志
        self.all_in_mem = all_in_mem
        # 如果全部加载到内存中
        if self.all_in_mem:
            # 缓存所有音频数据
            self.cache = [self.get_audio(p[0]) for p in self.audiopaths]
    # 从音频数据中随机截取片段，并进行音量增强
    def random_slice(self, c, f0, spec, audio_norm, spk, uv, volume):
        # 如果频谱的时间维度小于30，则跳过
        # if spec.shape[1] < 30:
        #     print("skip too short audio:", filename)
        #     return None

        # 如果随机选择音量增强，并且允许音量增强，并且音量不为空
        if random.choice([True, False]) and self.vol_aug and volume is not None:
            # 计算音频的最大振幅
            max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
            # 计算最大振幅对应的音量变化范围
            max_shift = min(1, np.log10(1/max_amp))
            # 在对数音量变化范围内随机选择一个值
            log10_vol_shift = random.uniform(-1, max_shift)
            # 对音频数据进行音量变化
            audio_norm = audio_norm * (10 ** log10_vol_shift)
            # 对音量数据进行相应的变化
            volume = volume * (10 ** log10_vol_shift)
            # 重新计算音频的频谱
            spec = spectrogram_torch(audio_norm,
            self.hparams.data.filter_length,
            self.hparams.data.sampling_rate,
            self.hparams.data.hop_length,
            self.hparams.data.win_length,
            center=False)[0]

        # 如果频谱的时间维度大于800
        if spec.shape[1] > 800:
            # 随机选择起始位置
            start = random.randint(0, spec.shape[1]-800)
            # 计算结束位置
            end = start + 790
            # 对频谱、c、f0、uv进行截取
            spec, c, f0, uv = spec[:, start:end], c[:, start:end], f0[start:end], uv[start:end]
            # 对音频数据进行截取
            audio_norm = audio_norm[:, start * self.hop_length : end * self.hop_length]
            # 如果音量不为空，对音量数据进行截取
            if volume is not None:
                volume = volume[start:end]
        # 返回截取后的数据
        return c, f0, spec, audio_norm, spk, uv,volume

    # 获取数据集中指定索引的数据
    def __getitem__(self, index):
        # 如果所有数据都在内存中
        if self.all_in_mem:
            # 返回随机截取的数据
            return self.random_slice(*self.cache[index])
        else:
            # 返回随机截取的数据
            return self.random_slice(*self.get_audio(self.audiopaths[index][0]))

    # 获取数据集的长度
    def __len__(self):
        return len(self.audiopaths)
class TextAudioCollate:

    def __call__(self, batch):
        # 过滤掉空的数据
        batch = [b for b in batch if b is not None]

        # 计算输入长度，并按照长度降序排序
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[1] for x in batch]),
            dim=0, descending=True)

        # 计算最大的字符长度
        max_c_len = max([x[0].size(1) for x in batch])
        # 计算最大的音频长度
        max_wav_len = max([x[3].size(1) for x in batch])

        # 创建长度张量
        lengths = torch.LongTensor(len(batch))

        # 创建字符张量，并用零填充
        c_padded = torch.FloatTensor(len(batch), batch[0][0].shape[0], max_c_len)
        c_padded.zero_()
        
        # 创建基频张量，并用零填充
        f0_padded = torch.FloatTensor(len(batch), max_c_len)
        f0_padded.zero_()
        
        # 创建频谱张量，并用零填充
        spec_padded = torch.FloatTensor(len(batch), batch[0][2].shape[0], max_c_len)
        spec_padded.zero_()
        
        # 创建音频张量，并用零填充
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        wav_padded.zero_()
        
        # 创建说话人 ID 张量
        spkids = torch.LongTensor(len(batch), 1)
        
        # 创建无声区域张量，并用零填充
        uv_padded = torch.FloatTensor(len(batch), max_c_len)
        uv_padded.zero_()
        
        # 创建音量张量，并用零填充
        volume_padded = torch.FloatTensor(len(batch), max_c_len)
        volume_padded.zero_()

        # 遍历排序后的索引
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            # 处理字符数据
            c = row[0]
            c_padded[i, :, :c.size(1)] = c
            lengths[i] = c.size(1)

            # 处理基频数据
            f0 = row[1]
            f0_padded[i, :f0.size(0)] = f0

            # 处理频谱数据
            spec = row[2]
            spec_padded[i, :, :spec.size(1)] = spec

            # 处理音频数据
            wav = row[3]
            wav_padded[i, :, :wav.size(1)] = wav

            # 处理说话人 ID 数据
            spkids[i, 0] = row[4]

            # 处理无声区域数据
            uv = row[5]
            uv_padded[i, :uv.size(0)] = uv
            
            # 处理音量数据
            volume = row[6]
            if volume is not None:
                volume_padded[i, :volume.size(0)] = volume
            else :
                volume_padded = None
        # 返回处理后的数据
        return c_padded, f0_padded, spec_padded, wav_padded, spkids, lengths, uv_padded, volume_padded
```