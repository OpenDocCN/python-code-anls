# `.\SenseVoiceSmall-src\utils\frontend.py`

```
# 指定文件编码为 UTF-8
# -*- encoding: utf-8 -*-
# 从 pathlib 模块导入 Path 类，用于文件路径操作
from pathlib import Path
# 从 typing 模块导入类型提示工具
from typing import Any, Dict, Iterable, List, NamedTuple, Set, Tuple, Union
# 导入 copy 模块，用于对象复制
import copy

# 导入 numpy 库并简化为 np
import numpy as np
# 导入 kaldi_native_fbank 模块并简化为 knf
import kaldi_native_fbank as knf

# 获取当前文件所在目录的绝对路径
root_dir = Path(__file__).resolve().parent

# 初始化一个空的字典，用于记录日志状态
logger_initialized = {}

# 定义 WavFrontend 类，表示 ASR 的常规前端结构
class WavFrontend:
    """Conventional frontend structure for ASR."""

    # 初始化方法，设置前端参数
    def __init__(
        self,
        cmvn_file: str = None,  # CMVN 文件路径，默认为 None
        fs: int = 16000,  # 采样频率，默认为 16000
        window: str = "hamming",  # 窗函数类型，默认为汉明窗
        n_mels: int = 80,  # 梅尔频带数量，默认为 80
        frame_length: int = 25,  # 帧长度，单位为毫秒，默认为 25
        frame_shift: int = 10,  # 帧移，单位为毫秒，默认为 10
        lfr_m: int = 1,  # LFR 的 m 值，默认为 1
        lfr_n: int = 1,  # LFR 的 n 值，默认为 1
        dither: float = 1.0,  # 加性噪声的强度，默认为 1.0
        **kwargs,  # 其他可选参数
    ) -> None:

        # 创建 FbankOptions 实例以配置参数
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = fs  # 设置采样频率
        opts.frame_opts.dither = dither  # 设置加性噪声
        opts.frame_opts.window_type = window  # 设置窗函数类型
        opts.frame_opts.frame_shift_ms = float(frame_shift)  # 设置帧移
        opts.frame_opts.frame_length_ms = float(frame_length)  # 设置帧长度
        opts.mel_opts.num_bins = n_mels  # 设置梅尔频带数量
        opts.energy_floor = 0  # 设置能量下限
        opts.frame_opts.snip_edges = True  # 启用边缘剪切
        opts.mel_opts.debug_mel = False  # 关闭梅尔调试
        self.opts = opts  # 保存配置参数

        # 保存 LFR 的 m 和 n 值
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        # 保存 CMVN 文件路径
        self.cmvn_file = cmvn_file

        # 如果指定了 CMVN 文件，则加载其内容
        if self.cmvn_file:
            self.cmvn = self.load_cmvn()
        # 初始化其他属性
        self.fbank_fn = None  # 存储在线 Fbank 对象
        self.fbank_beg_idx = 0  # 开始索引初始化
        self.reset_status()  # 重置状态

    # 计算梅尔频率倒谱系数 (FBANK)
    def fbank(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 将波形数据缩放
        waveform = waveform * (1 << 15)
        # 创建在线 Fbank 实例
        self.fbank_fn = knf.OnlineFbank(self.opts)
        # 接受波形数据
        self.fbank_fn.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())
        # 获取准备好的帧数
        frames = self.fbank_fn.num_frames_ready
        # 创建空矩阵以存储 FBANK 特征
        mat = np.empty([frames, self.opts.mel_opts.num_bins])
        # 遍历所有帧并获取特征
        for i in range(frames):
            mat[i, :] = self.fbank_fn.get_frame(i)
        # 转换特征矩阵数据类型
        feat = mat.astype(np.float32)
        # 获取特征长度
        feat_len = np.array(mat.shape[0]).astype(np.int32)
        # 返回特征和特征长度
        return feat, feat_len

    # 计算在线 FBANK 特征
    def fbank_online(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 将波形数据缩放
        waveform = waveform * (1 << 15)
        # 接受波形数据
        self.fbank_fn.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())
        # 获取准备好的帧数
        frames = self.fbank_fn.num_frames_ready
        # 创建空矩阵以存储 FBANK 特征
        mat = np.empty([frames, self.opts.mel_opts.num_bins])
        # 遍历从开始索引到准备好的帧
        for i in range(self.fbank_beg_idx, frames):
            mat[i, :] = self.fbank_fn.get_frame(i)
        # 特征矩阵转换数据类型
        feat = mat.astype(np.float32)
        # 获取特征长度
        feat_len = np.array(mat.shape[0]).astype(np.int32)
        # 返回特征和特征长度
        return feat, feat_len

    # 重置 FBANK 状态
    def reset_status(self):
        # 创建新的在线 Fbank 实例
        self.fbank_fn = knf.OnlineFbank(self.opts)
        # 重置开始索引
        self.fbank_beg_idx = 0
    # 定义一个进行LFR（长时频率平滑）和CMVN（均值方差归一化）的函数
    def lfr_cmvn(self, feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 如果lfr_m或lfr_n不为1，应用LFR处理
        if self.lfr_m != 1 or self.lfr_n != 1:
            feat = self.apply_lfr(feat, self.lfr_m, self.lfr_n)
    
        # 如果存在cmvn_file，应用CMVN处理
        if self.cmvn_file:
            feat = self.apply_cmvn(feat)
    
        # 获取处理后特征的长度，并转为32位整数
        feat_len = np.array(feat.shape[0]).astype(np.int32)
        # 返回处理后的特征和特征长度
        return feat, feat_len
    
    # 定义一个静态方法用于应用LFR处理
    @staticmethod
    def apply_lfr(inputs: np.ndarray, lfr_m: int, lfr_n: int) -> np.ndarray:
        # 初始化存储LFR处理后结果的列表
        LFR_inputs = []
    
        # 获取输入的时间步长
        T = inputs.shape[0]
        # 计算LFR后的时间步长
        T_lfr = int(np.ceil(T / lfr_n))
        # 对输入进行左侧填充
        left_padding = np.tile(inputs[0], ((lfr_m - 1) // 2, 1))
        # 将填充后的输入与原输入合并
        inputs = np.vstack((left_padding, inputs))
        # 更新时间步长以包括填充
        T = T + (lfr_m - 1) // 2
        # 遍历每个LFR帧
        for i in range(T_lfr):
            # 如果当前LFR帧能完全填充
            if lfr_m <= T - i * lfr_n:
                # 提取当前帧并调整形状
                LFR_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).reshape(1, -1))
            else:
                # 处理最后一个LFR帧
                num_padding = lfr_m - (T - i * lfr_n)
                frame = inputs[i * lfr_n :].reshape(-1)
                # 填充剩余帧
                for _ in range(num_padding):
                    frame = np.hstack((frame, inputs[-1]))
    
                # 添加处理后的帧到结果列表
                LFR_inputs.append(frame)
        # 将所有LFR帧合并为一个数组并转为浮点型
        LFR_outputs = np.vstack(LFR_inputs).astype(np.float32)
        # 返回LFR处理后的结果
        return LFR_outputs
    
    # 定义应用CMVN处理的方法
    def apply_cmvn(self, inputs: np.ndarray) -> np.ndarray:
        """
        使用均值方差归一化处理输入数据
        """
        # 获取输入的帧数和维度
        frame, dim = inputs.shape
        # 创建均值和方差的填充数组
        means = np.tile(self.cmvn[0:1, :dim], (frame, 1))
        vars = np.tile(self.cmvn[1:2, :dim], (frame, 1))
        # 应用CMVN处理
        inputs = (inputs + means) * vars
        # 返回处理后的输入
        return inputs
    
    # 定义加载CMVN数据的方法
    def load_cmvn(
        self,
    ) -> np.ndarray:
        # 打开指定的CMVN文件，读取内容
        with open(self.cmvn_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    
        # 初始化均值和方差列表
        means_list = []
        vars_list = []
        # 遍历文件中的每一行
        for i in range(len(lines)):
            line_item = lines[i].split()
            # 检查是否为<AddShift>行
            if line_item[0] == "<AddShift>":
                line_item = lines[i + 1].split()
                # 检查是否为<LearnRateCoef>行
                if line_item[0] == "<LearnRateCoef>":
                    # 提取均值数据
                    add_shift_line = line_item[3 : (len(line_item) - 1)]
                    means_list = list(add_shift_line)
                    continue
            # 检查是否为<Rescale>行
            elif line_item[0] == "<Rescale>":
                line_item = lines[i + 1].split()
                # 检查是否为<LearnRateCoef>行
                if line_item[0] == "<LearnRateCoef>":
                    # 提取方差数据
                    rescale_line = line_item[3 : (len(line_item) - 1)]
                    vars_list = list(rescale_line)
                    continue
    
        # 将均值和方差转换为浮点数组
        means = np.array(means_list).astype(np.float64)
        vars = np.array(vars_list).astype(np.float64)
        # 创建CMVN数组并返回
        cmvn = np.array([means, vars])
        return cmvn
# 定义 WavFrontendOnline 类，继承自 WavFrontend
class WavFrontendOnline(WavFrontend):
    # 初始化方法，接受关键字参数
    def __init__(self, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 初始化频谱特征提取器（注释掉的部分）
        # self.fbank_fn = knf.OnlineFbank(self.opts)
        # 计算每帧样本长度
        self.frame_sample_length = int(
            self.opts.frame_opts.frame_length_ms * self.opts.frame_opts.samp_freq / 1000
        )
        # 计算帧移样本长度
        self.frame_shift_sample_length = int(
            self.opts.frame_opts.frame_shift_ms * self.opts.frame_opts.samp_freq / 1000
        )
        # 初始化波形数据
        self.waveform = None
        # 预留波形列表
        self.reserve_waveforms = None
        # 输入缓存
        self.input_cache = None
        # LFR拼接缓存列表
        self.lfr_splice_cache = []

    @staticmethod
    # 静态方法，应用 LFR 处理
    def apply_lfr(
        inputs: np.ndarray, lfr_m: int, lfr_n: int, is_final: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        应用 LFR 处理数据
        """

        # 初始化 LFR 输入列表
        LFR_inputs = []
        # 获取输入的时间步数
        T = inputs.shape[0]  # 包含右上下文
        # 计算 LFR 处理后时间步数
        T_lfr = int(
            np.ceil((T - (lfr_m - 1) // 2) / lfr_n)
        )  # 减去右上下文: (lfr_m - 1) // 2
        # 初始化拼接索引
        splice_idx = T_lfr
        # 遍历 LFR 输出时间步
        for i in range(T_lfr):
            # 如果 LFR 窗口在输入范围内
            if lfr_m <= T - i * lfr_n:
                # 将当前 LFR 窗口重塑并加入列表
                LFR_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).reshape(1, -1))
            else:  # 处理最后一个 LFR 窗口
                if is_final:
                    # 计算需要填充的数量
                    num_padding = lfr_m - (T - i * lfr_n)
                    # 处理最后帧
                    frame = (inputs[i * lfr_n :]).reshape(-1)
                    # 填充多余的帧
                    for _ in range(num_padding):
                        frame = np.hstack((frame, inputs[-1]))
                    # 添加处理后的帧到 LFR 输入列表
                    LFR_inputs.append(frame)
                else:
                    # 更新拼接索引并退出循环
                    splice_idx = i
                    break
        # 确保拼接索引不超出范围
        splice_idx = min(T - 1, splice_idx * lfr_n)
        # 获取 LFR 拼接缓存
        lfr_splice_cache = inputs[splice_idx:, :]
        # 堆叠 LFR 输入形成输出
        LFR_outputs = np.vstack(LFR_inputs)
        # 返回转换后的 LFR 输出、拼接缓存和拼接索引
        return LFR_outputs.astype(np.float32), lfr_splice_cache, splice_idx

    @staticmethod
    # 静态方法，计算帧数
    def compute_frame_num(
        sample_length: int, frame_sample_length: int, frame_shift_sample_length: int
    ) -> int:
        # 计算帧数
        frame_num = int((sample_length - frame_sample_length) / frame_shift_sample_length + 1)
        # 返回有效帧数
        return frame_num if frame_num >= 1 and sample_length >= frame_sample_length else 0

    # 频谱特征提取方法，接受输入和输入长度
    def fbank(
        self, input: np.ndarray, input_lengths: np.ndarray
    # 返回一个包含三个 numpy 数组的元组
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 创建一个在线滤波器bank实例，使用类的选项
        self.fbank_fn = knf.OnlineFbank(self.opts)
        # 获取输入数据的批次大小
        batch_size = input.shape[0]
        # 如果输入缓存为空，初始化缓存为指定形状的空数组
        if self.input_cache is None:
            self.input_cache = np.empty((batch_size, 0), dtype=np.float32)
        # 将输入缓存和新输入在列的方向上拼接
        input = np.concatenate((self.input_cache, input), axis=1)
        # 计算输入的帧数
        frame_num = self.compute_frame_num(
            input.shape[-1], self.frame_sample_length, self.frame_shift_sample_length
        )
        # 更新输入缓存，保留最近的有效输入
        self.input_cache = input[
            :, -(input.shape[-1] - frame_num * self.frame_shift_sample_length) :
        ]
        # 初始化空的波形和特征数组
        waveforms = np.empty(0, dtype=np.float32)
        feats_pad = np.empty(0, dtype=np.float32)
        feats_lens = np.empty(0, dtype=np.int32)
        # 如果有帧数，则处理输入数据
        if frame_num:
            waveforms = []
            feats = []
            feats_lens = []
            # 遍历批次中的每个输入样本
            for i in range(batch_size):
                waveform = input[i]
                # 截取波形的有效部分
                waveforms.append(
                    waveform[
                        : (
                            (frame_num - 1) * self.frame_shift_sample_length
                            + self.frame_sample_length
                        )
                    ]
                )
                # 将波形数据放大
                waveform = waveform * (1 << 15)
    
                # 接受波形到滤波器，转换为列表格式
                self.fbank_fn.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())
                # 获取准备好的帧数
                frames = self.fbank_fn.num_frames_ready
                # 初始化用于存放特征的矩阵
                mat = np.empty([frames, self.opts.mel_opts.num_bins])
                # 填充特征矩阵
                for i in range(frames):
                    mat[i, :] = self.fbank_fn.get_frame(i)
                # 将矩阵转换为浮点数类型
                feat = mat.astype(np.float32)
                # 记录特征的长度
                feat_len = np.array(mat.shape[0]).astype(np.int32)
                feats.append(feat)
                feats_lens.append(feat_len)
    
            # 将所有波形和特征堆叠成数组
            waveforms = np.stack(waveforms)
            feats_lens = np.array(feats_lens)
            feats_pad = np.array(feats)
        # 将特征和长度存储到类属性中
        self.fbanks = feats_pad
        self.fbanks_lens = copy.deepcopy(feats_lens)
        # 返回波形、特征和特征长度
        return waveforms, feats_pad, feats_lens
    
    # 定义获取滤波器特征的函数
    def get_fbank(self) -> Tuple[np.ndarray, np.ndarray]:
        # 返回特征和特征长度
        return self.fbanks, self.fbanks_lens
    
    # 定义线性频谱均值归一化函数
    def lfr_cmvn(
        self, input: np.ndarray, input_lengths: np.ndarray, is_final: bool = False
    # 定义函数返回值类型，包括两个 numpy 数组和一个整数列表
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        # 获取输入数据的批量大小
        batch_size = input.shape[0]
        # 初始化特征列表
        feats = []
        # 初始化特征长度列表
        feats_lens = []
        # 初始化 LFR 拼接帧索引列表
        lfr_splice_frame_idxs = []
        # 遍历每个输入样本
        for i in range(batch_size):
            # 获取当前样本的有效数据矩阵
            mat = input[i, : input_lengths[i], :]
            # 初始化 LFR 拼接帧索引
            lfr_splice_frame_idx = -1
            # 检查是否需要进行 LFR 拼接处理
            if self.lfr_m != 1 or self.lfr_n != 1:
                # 在 apply_lfr 函数中更新 self.lfr_splice_cache
                mat, self.lfr_splice_cache[i], lfr_splice_frame_idx = self.apply_lfr(
                    mat, self.lfr_m, self.lfr_n, is_final
                )
            # 如果指定了 CMVN 文件，则应用 CMVN 处理
            if self.cmvn_file is not None:
                mat = self.apply_cmvn(mat)
            # 获取当前特征矩阵的长度
            feat_length = mat.shape[0]
            # 将特征矩阵添加到特征列表中
            feats.append(mat)
            # 将特征长度添加到特征长度列表中
            feats_lens.append(feat_length)
            # 将 LFR 拼接帧索引添加到索引列表中
            lfr_splice_frame_idxs.append(lfr_splice_frame_idx)
    
        # 将特征长度列表转换为 numpy 数组
        feats_lens = np.array(feats_lens)
        # 将特征列表转换为 numpy 数组
        feats_pad = np.array(feats)
        # 返回填充后的特征矩阵、特征长度数组和 LFR 拼接帧索引列表
        return feats_pad, feats_lens, lfr_splice_frame_idxs
    
    # 定义提取 FBank 特征的函数
    def extract_fbank(
        self, input: np.ndarray, input_lengths: np.ndarray, is_final: bool = False
    # 获取波形数据
    def get_waveforms(self):
        # 返回存储的波形数据
        return self.waveforms
    
    # 重置缓存
    def cache_reset(self):
        # 初始化在线 FBank 特征函数
        self.fbank_fn = knf.OnlineFbank(self.opts)
        # 清空保留的波形数据
        self.reserve_waveforms = None
        # 清空输入缓存
        self.input_cache = None
        # 清空 LFR 拼接缓存
        self.lfr_splice_cache = []
# 定义加载字节数据的函数
def load_bytes(input):
    # 从输入缓冲区中读取数据，转换为 int16 类型的 NumPy 数组
    middle_data = np.frombuffer(input, dtype=np.int16)
    # 将数据转换为 NumPy 数组
    middle_data = np.asarray(middle_data)
    # 检查数组的数据类型是否为整数类型
    if middle_data.dtype.kind not in "iu":
        raise TypeError("'middle_data' must be an array of integers")  # 如果不是整数类型，抛出异常
    # 定义浮点数类型的 NumPy 数据类型
    dtype = np.dtype("float32")
    # 检查定义的数据类型是否为浮点数类型
    if dtype.kind != "f":
        raise TypeError("'dtype' must be a floating point type")  # 如果不是浮点数类型，抛出异常

    # 获取 middle_data 的整数信息
    i = np.iinfo(middle_data.dtype)
    # 计算最大绝对值
    abs_max = 2 ** (i.bits - 1)
    # 计算偏移量
    offset = i.min + abs_max
    # 将 middle_data 转换为浮点数并归一化
    array = np.frombuffer((middle_data.astype(dtype) - offset) / abs_max, dtype=np.float32)
    # 返回处理后的数组
    return array


# 定义正弦位置编码器类
class SinusoidalPositionEncoderOnline:
    """Streaming Positional encoding."""

    # 定义编码方法
    def encode(self, positions: np.ndarray = None, depth: int = None, dtype: np.dtype = np.float32):
        # 获取批次大小
        batch_size = positions.shape[0]
        # 将位置数组转换为指定的数据类型
        positions = positions.astype(dtype)
        # 计算对数时间尺度增量
        log_timescale_increment = np.log(np.array([10000], dtype=dtype)) / (depth / 2 - 1)
        # 计算逆时间尺度
        inv_timescales = np.exp(np.arange(depth / 2).astype(dtype) * (-log_timescale_increment))
        # 调整逆时间尺度的形状
        inv_timescales = np.reshape(inv_timescales, [batch_size, -1])
        # 缩放时间并调整形状
        scaled_time = np.reshape(positions, [1, -1, 1]) * np.reshape(inv_timescales, [1, 1, -1])
        # 计算正弦和余弦编码
        encoding = np.concatenate((np.sin(scaled_time), np.cos(scaled_time)), axis=2)
        # 返回编码结果，转换为指定的数据类型
        return encoding.astype(dtype)

    # 定义前向传播方法
    def forward(self, x, start_idx=0):
        # 获取输入的批次大小、时间步和输入维度
        batch_size, timesteps, input_dim = x.shape
        # 创建位置数组
        positions = np.arange(1, timesteps + 1 + start_idx)[None, :]
        # 获取位置编码
        position_encoding = self.encode(positions, input_dim, x.dtype)

        # 返回输入与位置编码的和
        return x + position_encoding[:, start_idx : start_idx + timesteps]


# 定义测试函数
def test():
    # 定义音频文件路径
    path = "/nfs/zhifu.gzf/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav"
    import librosa  # 导入 librosa 库用于音频处理

    # 定义 CMVN 文件和配置文件路径
    cmvn_file = "/nfs/zhifu.gzf/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/am.mvn"
    config_file = "/nfs/zhifu.gzf/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/config.yaml"
    from funasr.runtime.python.onnxruntime.rapid_paraformer.utils.utils import read_yaml  # 导入 YAML 读取函数

    # 读取配置文件
    config = read_yaml(config_file)
    # 加载音频文件
    waveform, _ = librosa.load(path, sr=None)
    # 创建前端处理对象
    frontend = WavFrontend(
        cmvn_file=cmvn_file,
        **config["frontend_conf"],
    )
    # 在线计算声谱特征
    speech, _ = frontend.fbank_online(waveform)  # 1d, (sample,), numpy
    # 进行 LFR-CMVN 处理，得到特征和特征长度
    feat, feat_len = frontend.lfr_cmvn(
        speech
    )  # 2d, (frame, 450), np.float32 -> torch, torch.from_numpy(), dtype, (1, frame, 450)

    # 重置前端处理状态，清除缓存
    frontend.reset_status()  # clear cache
    # 返回特征和特征长度
    return feat, feat_len


# 如果此文件是主程序，运行测试函数
if __name__ == "__main__":
    test()
```