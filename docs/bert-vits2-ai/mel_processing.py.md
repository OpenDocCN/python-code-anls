# `Bert-VITS2\mel_processing.py`

```
# 导入 torch 库
import torch
# 导入 torch.utils.data 模块
import torch.utils.data
# 从 librosa.filters 模块中导入 mel 函数并重命名为 librosa_mel_fn
from librosa.filters import mel as librosa_mel_fn
# 导入 warnings 模块

import warnings

# 忽略 FutureWarning 类别的警告
# warnings.simplefilter(action='ignore', category=FutureWarning)
# 忽略所有警告
warnings.filterwarnings(action="ignore")
# 设置最大的 WAV 值
MAX_WAV_VALUE = 32768.0


# 定义动态范围压缩函数 dynamic_range_compression_torch
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: 压缩因子
    """
    # 对输入张量进行动态范围压缩
    return torch.log(torch.clamp(x, min=clip_val) * C)


# 定义动态范围解压缩函数 dynamic_range_decompression_torch
def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: 用于压缩的压缩因子
    """
    # 对输入张量进行动态范围解压缩
    return torch.exp(x) / C


# 定义频谱归一化函数 spectral_normalize_torch
def spectral_normalize_torch(magnitudes):
    # 调用动态范围压缩函数对输入张量进行处理
    output = dynamic_range_compression_torch(magnitudes)
    return output


# 定义频谱反归一化函数 spectral_de_normalize_torch
def spectral_de_normalize_torch(magnitudes):
    # 调用动态范围解压缩函数对输入张量进行处理
    output = dynamic_range_decompression_torch(magnitudes)
    return output


# 初始化 mel_basis 和 hann_window 字典
mel_basis = {}
hann_window = {}


# 定义频谱图函数 spectrogram_torch
def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    # 如果输入张量的最小值小于 -1.0，则打印最小值
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    # 如果输入张量的最大值大于 1.0，则打印最大值
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    # 获取全局变量 hann_window
    global hann_window
    # 构建 dtype_device 字符串
    dtype_device = str(y.dtype) + "_" + str(y.device)
    # 构建 wnsize_dtype_device 字符串
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    # 如果 wnsize_dtype_device 不在 hann_window 字典中，则进行以下操作
    if wnsize_dtype_device not in hann_window:
        # 将 hann_window[wnsize_dtype_device] 设置为 win_size 大小的汉宁窗口
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    # 在输入张量的第一维度上添加反射填充
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    # 去除添加的维度
    y = y.squeeze(1)

    # 对输入张量进行短时傅里叶变换
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    # 计算频谱的幅度
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec
# 将频谱转换为梅尔频谱，使用 PyTorch 实现
def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    # 声明 mel_basis 为全局变量
    global mel_basis
    # 获取输入 spec 的数据类型和设备信息
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    # 构建 fmax 对应的数据类型和设备信息字符串
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    # 如果 mel_basis 中不存在对应的 fmax_dtype_device，则进行计算并存储
    if fmax_dtype_device not in mel_basis:
        # 使用 librosa_mel_fn 函数计算 mel
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        # 将 mel 转换为 PyTorch 张量，并存储到 mel_basis 中
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )
    # 使用 mel_basis[fmax_dtype_device] 对 spec 进行矩阵乘法操作
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    # 对 spec 进行频谱归一化处理
    spec = spectral_normalize_torch(spec)
    # 返回处理后的 spec
    return spec


# 使用 PyTorch 实现梅尔频谱图
def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    # 如果输入 y 中存在小于 -1.0 的值，则打印提示信息
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    # 如果输入 y 中存在大于 1.0 的值，则打印提示信息
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    # 声明 mel_basis 和 hann_window 为全局变量
    global mel_basis, hann_window
    # 获取输入 y 的数据类型和设备信息
    dtype_device = str(y.dtype) + "_" + str(y.device)
    # 构建 fmax 对应的数据类型和设备信息字符串
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    # 构建 win_size 对应的数据类型和设备信息字符串
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    # 如果 mel_basis 中不存在对应的 fmax_dtype_device，则进行计算并存储
    if fmax_dtype_device not in mel_basis:
        # 使用 librosa_mel_fn 函数计算 mel
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        # 将 mel 转换为 PyTorch 张量，并存储到 mel_basis 中
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=y.dtype, device=y.device
        )
    # 如果 hann_window 中不存在对应的 wnsize_dtype_device，则进行计算并存储
    if wnsize_dtype_device not in hann_window:
        # 使用 torch.hann_window 函数计算 hann_window
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    # 在 y 的第一维度上进行填充操作
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    # 去除填充后的 y 的第一维度
    y = y.squeeze(1)

    # 使用 torch.stft 函数计算短时傅里叶变换
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    # 对 spec 进行平方和开方运算，并加上 1e-6，得到最终的频谱图
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    # 使用 torch.matmul 函数计算 mel_basis[fmax_dtype_device] 和 spec 的矩阵乘法
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    # 对 spec 进行光谱归一化处理
    spec = spectral_normalize_torch(spec)
    # 返回处理后的 spec
    return spec
```