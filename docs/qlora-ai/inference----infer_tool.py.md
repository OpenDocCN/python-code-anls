# `so-vits-svc\inference\infer_tool.py`

```
# 导入所需的库
import gc  # 垃圾回收模块
import hashlib  # 哈希算法模块
import io  # IO操作模块
import json  # JSON数据处理模块
import logging  # 日志记录模块
import os  # 系统操作模块
import pickle  # 数据序列化模块
import time  # 时间模块
from pathlib import Path  # 路径操作模块

import librosa  # 音频处理库
import numpy as np  # 数组操作库

# import onnxruntime
import soundfile  # 音频文件读写库
import torch  # PyTorch深度学习库
import torchaudio  # PyTorch音频处理库

import cluster  # 聚类模块
import utils  # 工具模块
from diffusion.unit2mel import load_model_vocoder  # 导入音频模型加载函数
from inference import slicer  # 导入推断模块的切片函数
from models import SynthesizerTrn  # 导入合成器模型

# 设置matplotlib日志级别
logging.getLogger('matplotlib').setLevel(logging.WARNING)


# 读取临时文件内容
def read_temp(file_name):
    # 如果文件不存在，则创建一个空的临时文件
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write(json.dumps({"info": "temp_dict"}))
        return {}
    else:
        try:
            with open(file_name, "r") as f:
                data = f.read()
            data_dict = json.loads(data)
            # 如果文件大小超过50MB，则清理文件内容
            if os.path.getsize(file_name) > 50 * 1024 * 1024:
                f_name = file_name.replace("\\", "/").split("/")[-1]
                print(f"clean {f_name}")
                # 删除超过14天的数据
                for wav_hash in list(data_dict.keys()):
                    if int(time.time()) - int(data_dict[wav_hash]["time"]) > 14 * 24 * 3600:
                        del data_dict[wav_hash]
        except Exception as e:
            print(e)
            print(f"{file_name} error,auto rebuild file")
            data_dict = {"info": "temp_dict"}
        return data_dict


# 写入临时文件内容
def write_temp(file_name, data):
    with open(file_name, "w") as f:
        f.write(json.dumps(data))


# 计时装饰器
def timeit(func):
    def run(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print('executing \'%s\' costed %.3fs' % (func.__name__, time.time() - t))
        return res

    return run


# 格式化音频文件为.wav格式
def format_wav(audio_path):
    if Path(audio_path).suffix == '.wav':
        return
    raw_audio, raw_sample_rate = librosa.load(audio_path, mono=True, sr=None)
    soundfile.write(Path(audio_path).with_suffix(".wav"), raw_audio, raw_sample_rate)


# 获取指定后缀的文件列表
def get_end_file(dir_path, end):
    file_lists = []
    # 遍历指定目录下的所有文件和子目录
    for root, dirs, files in os.walk(dir_path):
        # 过滤掉文件名以"."开头的文件
        files = [f for f in files if f[0] != '.']
        # 过滤掉子目录名以"."开头的子目录
        dirs[:] = [d for d in dirs if d[0] != '.']
        # 遍历筛选后的文件列表
        for f_file in files:
            # 判断文件是否以指定的后缀结尾
            if f_file.endswith(end):
                # 将符合条件的文件路径添加到文件列表中，同时将路径中的反斜杠替换为斜杠
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    # 返回符合条件的文件路径列表
    return file_lists
# 计算给定内容的 MD5 值并返回
def get_md5(content):
    return hashlib.new("md5", content).hexdigest()

# 将列表 a 填充到和列表 b 一样的长度
def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])

# 创建目录
def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

# 将数组填充到指定长度
def pad_array(arr, target_length):
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr
    else:
        pad_width = target_length - current_length
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_arr = np.pad(arr, (pad_left, pad_right), 'constant', constant_values=(0, 0))
        return padded_arr

# 将列表按照指定长度 n 进行分割
def split_list_by_n(list_collection, n, pre=0):
    for i in range(0, len(list_collection), n):
        yield list_collection[i-pre if i-pre>=0 else i: i + n]

# 自定义异常类
class F0FilterException(Exception):
    pass

# Svc 类
class Svc(object):
    # 加载模型
    def load_model(self, spk_mix_enable=False):
        # 获取模型配置
        self.net_g_ms = SynthesizerTrn(
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            **self.hps_ms.model)
        _ = utils.load_checkpoint(self.net_g_path, self.net_g_ms, None)
        self.dtype = list(self.net_g_ms.parameters())[0].dtype
        if "half" in self.net_g_path and torch.cuda.is_available():
            _ = self.net_g_ms.half().eval().to(self.dev)
        else:
            _ = self.net_g_ms.eval().to(self.dev)
        if spk_mix_enable:
            self.net_g_ms.EnableCharacterMix(len(self.spk2id), self.dev)

    # 清空空闲的显存
    def clear_empty(self):
        torch.cuda.empty_cache()
    # 卸载模型
    # 将 self.net_g_ms 移到 CPU 上
    self.net_g_ms = self.net_g_ms.to("cpu")
    # 删除 self.net_g_ms
    del self.net_g_ms
    # 如果存在属性 "enhancer"
    if hasattr(self,"enhancer"): 
        # 将 self.enhancer.enhancer 移到 CPU 上
        self.enhancer.enhancer = self.enhancer.enhancer.to("cpu")
        # 删除 self.enhancer.enhancer
        del self.enhancer.enhancer
        # 删除 self.enhancer
        del self.enhancer
    # 执行垃圾回收
    gc.collect()
# 定义一个名为 RealTimeVC 的类
class RealTimeVC:
    # 初始化方法，设置初始属性值
    def __init__(self):
        # 上一个音频块的数据
        self.last_chunk = None
        # 上一个输出数据
        self.last_o = None
        # 音频块的长度
        self.chunk_len = 16000  # chunk length
        # 交叉淡入淡出的长度，是 640 的倍数
        self.pre_len = 3840  # cross fade length, multiples of 640
    # 输入和输出都是一维的 numpy 波形数组
    # 定义一个处理函数，接受多个参数
    def process(self, svc_model, speaker_id, f_pitch_change, input_wav_path,
                cluster_infer_ratio=0,
                auto_predict_f0=False,
                noice_scale=0.4,
                f0_filter=False):

        # 导入maad模块
        import maad
        # 使用torchaudio加载输入的wav文件，并获取采样率
        audio, sr = torchaudio.load(input_wav_path)
        # 将音频数据转换为numpy数组
        audio = audio.cpu().numpy()[0]
        # 创建一个字节流对象
        temp_wav = io.BytesIO()
        # 如果上一个音频块为空
        if self.last_chunk is None:
            # 重置输入wav文件的指针位置
            input_wav_path.seek(0)
            # 使用svc_model进行推断，获取音频数据
            audio, sr = svc_model.infer(speaker_id, f_pitch_change, input_wav_path,
                                        cluster_infer_ratio=cluster_infer_ratio,
                                        auto_predict_f0=auto_predict_f0,
                                        noice_scale=noice_scale,
                                        f0_filter=f0_filter)
            # 将音频数据转换为numpy数组
            audio = audio.cpu().numpy()
            # 保存最后一个音频块的数据
            self.last_chunk = audio[-self.pre_len:]
            self.last_o = audio
            # 返回最后一个音频块的数据
            return audio[-self.chunk_len:]
        else:
            # 将上一个音频块和当前音频数据连接起来
            audio = np.concatenate([self.last_chunk, audio])
            # 将音频数据写入临时wav文件
            soundfile.write(temp_wav, audio, sr, format="wav")
            temp_wav.seek(0)
            # 使用svc_model进行推断，获取音频数据
            audio, sr = svc_model.infer(speaker_id, f_pitch_change, temp_wav,
                                        cluster_infer_ratio=cluster_infer_ratio,
                                        auto_predict_f0=auto_predict_f0,
                                        noice_scale=noice_scale,
                                        f0_filter=f0_filter)
            # 将音频数据转换为numpy数组
            audio = audio.cpu().numpy()
            # 对音频数据进行交叠淡入淡出处理
            ret = maad.util.crossfade(self.last_o, audio, self.pre_len)
            # 保存最后一个音频块的数据
            self.last_chunk = audio[-self.pre_len:]
            self.last_o = audio
            # 返回处理后的音频数据
            return ret[self.chunk_len:2 * self.chunk_len]
```