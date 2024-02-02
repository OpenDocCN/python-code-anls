# `so-vits-svc\inference\infer_tool_grad.py`

```py
# 导入所需的库
import io
import logging
import os

import librosa
import numpy as np
import parselmouth
import soundfile
import torch
import torchaudio

import utils
from inference import slicer
from models import SynthesizerTrn

# 设置日志级别
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# 定义函数，将输入的F0序列调整为目标长度
def resize2d_f0(x, target_len):
    # 将输入的F0序列转换为numpy数组
    source = np.array(x)
    # 将小于0.001的值替换为NaN
    source[source < 0.001] = np.nan
    # 使用线性插值将F0序列调整为目标长度
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)),
                       source)
    # 将NaN替换为0
    res = np.nan_to_num(target)
    return res

# 定义函数，获取音频的F0序列
def get_f0(x, p_len,f0_up_key=0):
    # 设置时间步长
    time_step = 160 / 16000 * 1000
    # 设置F0的最小和最大值
    f0_min = 50
    f0_max = 1100
    # 计算F0的Mel值的最小和最大值
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    # 使用Praat库计算音频的F0序列
    f0 = parselmouth.Sound(x, 16000).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']

    # 计算需要填充的大小
    pad_size=(p_len - len(f0) + 1) // 2
    # 如果需要填充的大小大于0，则进行填充
    if(pad_size>0 or p_len - len(f0) - pad_size>0):
        f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')

    # 根据音高变换参数对F0序列进行调整
    f0 *= pow(2, f0_up_key / 12)
    # 计算F0的Mel值
    f0_mel = 1127 * np.log(1 + f0 / 700)
    # 对F0的Mel值进行归一化处理
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    # 将F0的Mel值四舍五入并转换为整数
    f0_coarse = np.rint(f0_mel).astype(np.int)
    return f0_coarse, f0

# 定义函数，清洗F0序列中的异常值
def clean_pitch(input_pitch):
    # 统计F0序列中值为1的数量
    num_nan = np.sum(input_pitch == 1)
    # 如果值为1的数量占比超过90%，则将非1的值替换为1
    if num_nan / len(input_pitch) > 0.9:
        input_pitch[input_pitch != 1] = 1
    return input_pitch

# 定义函数，将F0序列中的值为1的部分转换为NaN
def plt_pitch(input_pitch):
    input_pitch = input_pitch.astype(float)
    input_pitch[input_pitch == 1] = np.nan
    return input_pitch

# 定义函数，将F0序列转换为音高值
def f0_to_pitch(ff):
    f0_pitch = 69 + 12 * np.log2(ff / 440)
    return f0_pitch

# 定义函数，将序列a填充至长度为b
def fill_a_to_b(a, b):
    # 如果列表 a 的长度小于列表 b 的长度
    if len(a) < len(b):
        # 循环，使列表 a 的长度与列表 b 的长度相等
        for _ in range(0, len(b) - len(a)):
            # 将列表 a 的第一个元素添加到列表 a 的末尾
            a.append(a[0])
# 定义一个函数，用于创建多个目录
def mkdir(paths: list):
    # 遍历传入的路径列表
    for path in paths:
        # 如果路径不存在，则创建该路径
        if not os.path.exists(path):
            os.mkdir(path)

# 定义一个类
class VitsSvc(object):
    # 初始化方法
    def __init__(self):
        # 根据 GPU 是否可用选择设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.SVCVITS = None
        self.hps = None
        self.speakers = None
        # 获取 Hubert 模型
        self.hubert_soft = utils.get_hubert_model()

    # 设置设备方法
    def set_device(self, device):
        # 设置设备
        self.device = torch.device(device)
        self.hubert_soft.to(self.device)
        if self.SVCVITS is not None:
            self.SVCVITS.to(self.device)

    # 加载检查点方法
    def loadCheckpoint(self, path):
        # 从文件中获取超参数
        self.hps = utils.get_hparams_from_file(f"checkpoints/{path}/config.json")
        # 创建 SynthesizerTrn 对象
        self.SVCVITS = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model)
        # 加载检查点
        _ = utils.load_checkpoint(f"checkpoints/{path}/model.pth", self.SVCVITS, None)
        _ = self.SVCVITS.eval().to(self.device)
        self.speakers = self.hps.spk

    # 获取单位方法
    def get_units(self, source, sr):
        source = source.unsqueeze(0).to(self.device)
        # 进入推断模式
        with torch.inference_mode():
            # 获取单位
            units = self.hubert_soft.units(source)
            return units

    # 获取单位音高方法
    def get_unit_pitch(self, in_path, tran):
        # 加载音频文件
        source, sr = torchaudio.load(in_path)
        # 重新采样音频
        source = torchaudio.functional.resample(source, sr, 16000)
        # 如果音频是双声道，则取平均值
        if len(source.shape) == 2 and source.shape[1] >= 2:
            source = torch.mean(source, dim=0).unsqueeze(0)
        # 获取单位
        soft = self.get_units(source, sr).squeeze(0).cpu().numpy()
        # 获取音高
        f0_coarse, f0 = get_f0(source.cpu().numpy()[0], soft.shape[0]*2, tran)
        return soft, f0
    # 推断函数，根据说话者 ID、文本、原始路径生成音频
    def infer(self, speaker_id, tran, raw_path):
        # 将说话者 ID 转换为对应的索引
        speaker_id = self.speakers[speaker_id]
        # 将索引转换为 LongTensor 类型，并移动到指定设备上，增加一个维度
        sid = torch.LongTensor([int(speaker_id)]).to(self.device).unsqueeze(0)
        # 获取单位音高和音调
        soft, pitch = self.get_unit_pitch(raw_path, tran)
        # 将音调数据清洗并转换为 FloatTensor 类型，并移动到指定设备上，增加一个维度
        f0 = torch.FloatTensor(clean_pitch(pitch)).unsqueeze(0).to(self.device)
        # 将单位音高数据转换为 FloatTensor 类型，并移动到指定设备上
        stn_tst = torch.FloatTensor(soft)
        # 关闭梯度计算
        with torch.no_grad():
            # 将音频数据转换为 FloatTensor 类型，并移动到指定设备上，增加一个维度
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            # 将音频数据重复两次，并交换维度
            x_tst = torch.repeat_interleave(x_tst, repeats=2, dim=1).transpose(1, 2)
            # 使用 SVCVITS 模型进行推断，获取音频数据和音频长度
            audio,_ = self.SVCVITS.infer(x_tst, f0=f0, g=sid)[0,0].data.float()
        # 返回音频数据和音频长度
        return audio, audio.shape[-1]

    # 推断函数，根据原始音频、文本、切片数据库生成音频
    def inference(self, srcaudio, chara, tran, slice_db):
        # 获取采样率和音频数据
        sampling_rate, audio = srcaudio
        # 将音频数据归一化并转换为浮点数类型
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        # 如果音频数据维度大于 1，则转换为单声道
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        # 如果采样率不是 16000，则进行重采样
        if sampling_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        # 将音频数据写入临时 WAV 文件
        soundfile.write("tmpwav.wav", audio, 16000, format="wav")
        # 使用切片器对临时 WAV 文件进行切片
        chunks = slicer.cut("tmpwav.wav", db_thresh=slice_db)
        # 将切片后的音频数据合并为完整音频
        audio_data, audio_sr = slicer.chunks2audio("tmpwav.wav", chunks)
        audio = []
        # 遍历切片后的音频数据
        for (slice_tag, data) in audio_data:
            # 计算音频数据长度
            length = int(np.ceil(len(data) / audio_sr * self.hps.data.sampling_rate))
            # 创建一个字节流对象
            raw_path = io.BytesIO()
            # 将音频数据写入字节流对象
            soundfile.write(raw_path, data, audio_sr, format="wav")
            # 将字节流对象指针移动到开头
            raw_path.seek(0)
            # 如果是切片标记为真，则生成空音频数据；否则进行推断生成音频数据
            if slice_tag:
                _audio = np.zeros(length)
            else:
                out_audio, out_sr = self.infer(chara, tran, raw_path)
                _audio = out_audio.cpu().numpy()
            # 将音频数据添加到列表中
            audio.extend(list(_audio))
        # 将音频数据转换为整型并返回
        audio = (np.array(audio) * 32768.0).astype('int16')
        return (self.hps.data.sampling_rate, audio)
```