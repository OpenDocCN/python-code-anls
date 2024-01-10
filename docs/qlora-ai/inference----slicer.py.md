# `so-vits-svc\inference\slicer.py`

```
# 导入 librosa 库，用于音频处理
import librosa
# 导入 torch 库，用于深度学习
import torch
# 导入 torchaudio 库，用于音频处理
import torchaudio

# 定义一个音频切片器类
class Slicer:
    # 初始化方法，接受音频采样率、阈值、最小长度、最小间隔、跳跃大小和最大保留静音长度参数
    def __init__(self,
                 sr: int,
                 threshold: float = -40.,
                 min_length: int = 5000,
                 min_interval: int = 300,
                 hop_size: int = 20,
                 max_sil_kept: int = 5000):
        # 检查参数是否满足条件，不满足则抛出 ValueError 异常
        if not min_length >= min_interval >= hop_size:
            raise ValueError('The following condition must be satisfied: min_length >= min_interval >= hop_size')
        if not max_sil_kept >= hop_size:
            raise ValueError('The following condition must be satisfied: max_sil_kept >= hop_size')
        # 将时间间隔转换为采样点数
        min_interval = sr * min_interval / 1000
        # 计算阈值
        self.threshold = 10 ** (threshold / 20.)
        # 计算跳跃大小
        self.hop_size = round(sr * hop_size / 1000)
        # 计算窗口大小
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        # 计算最小长度
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        # 计算最小间隔
        self.min_interval = round(min_interval / self.hop_size)
        # 计算最大保留静音长度
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    # 定义一个私有方法，用于应用切片
    def _apply_slice(self, waveform, begin, end):
        # 如果音频是多通道的，则返回指定时间段内的所有通道数据
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)]
        # 如果音频是单通道的，则返回指定时间段内的数据
        else:
            return waveform[begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)]

# 定义一个切割音频的函数，接受音频路径、分贝阈值和最小长度参数
def cut(audio_path, db_thresh=-30, min_len=5000):
    # 加载音频文件，并获取采样率
    audio, sr = librosa.load(audio_path, sr=None)
    # 创建 Slicer 对象
    slicer = Slicer(
        sr=sr,
        threshold=db_thresh,
        min_length=min_len
    )
    # 对音频进行切片
    chunks = slicer.slice(audio)
    # 返回切片后的结果
    return chunks

# 定义一个将切片转换为音频的函数，接受音频路径和切片参数
def chunks2audio(audio_path, chunks):
    # 将切片转换为字典形式
    chunks = dict(chunks)
    # 加载音频文件，并获取采样率
    audio, sr = torchaudio.load(audio_path)
    # 如果音频是双通道的，则取平均值
    if len(audio.shape) == 2 and audio.shape[1] >= 2:
        audio = torch.mean(audio, dim=0).unsqueeze(0)
    # 将音频转换为 numpy 数组
    audio = audio.cpu().numpy()[0]
    # 创建一个空列表
    result = []
    # 遍历字典 chunks 中的键值对
    for k, v in chunks.items():
        # 将 v 中的 split_time 字段按逗号分割，并赋值给 tag
        tag = v["split_time"].split(",")
        # 如果 tag 中的第一个元素不等于第二个元素
        if tag[0] != tag[1]:
            # 将 v 中的 slice 字段和根据 tag 切片后的音频数据组成元组，添加到结果列表中
            result.append((v["slice"], audio[int(tag[0]):int(tag[1])]))
    # 返回结果列表和音频采样率
    return result, sr
```