# `d:/src/tocomm/Bert-VITS2\mel_processing.py`

```
import torch  # 导入 PyTorch 库
import torch.utils.data  # 导入 PyTorch 数据处理模块
from librosa.filters import mel as librosa_mel_fn  # 从 librosa 库中导入 mel 滤波器函数
import warnings  # 导入警告模块

# 忽略 FutureWarning 类别的警告
warnings.filterwarnings(action="ignore")

MAX_WAV_VALUE = 32768.0  # 设置最大 WAV 值为 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: 压缩因子
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)  # 对输入进行动态范围压缩并返回结果


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: 用于压缩的压缩因子
    """
    return torch.exp(x) / C  # 返回输入张量 x 的指数函数值除以压缩因子 C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)  # 对输入的幅度进行动态范围压缩
    return output  # 返回压缩后的幅度

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)  # 对输入的幅度进行动态范围解压缩
    return output  # 返回解压缩后的幅度

mel_basis = {}  # 存储梅尔滤波器组
hann_window = {}  # 存储汉宁窗函数
        # 检查输入信号的最小值是否小于-1.0，如果是则打印最小值
        if torch.min(y) < -1.0:
            print("min value is ", torch.min(y))
        # 检查输入信号的最大值是否大于1.0，如果是则打印最大值
        if torch.max(y) > 1.0:
            print("max value is ", torch.max(y))

        # 定义全局变量 hann_window
        global hann_window
        # 创建一个字符串，包含输入信号的数据类型和设备信息
        dtype_device = str(y.dtype) + "_" + str(y.device)
        # 创建一个字符串，包含窗口大小、数据类型和设备信息
        wnsize_dtype_device = str(win_size) + "_" + dtype_device
        # 如果 wnsize_dtype_device 不在 hann_window 中，则将其添加到 hann_window 中
        if wnsize_dtype_device not in hann_window:
            hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
                dtype=y.dtype, device=y.device
            )

        # 对输入信号进行零填充，使其长度等于 n_fft
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
            mode="reflect",
    )  # 关闭括号，可能是缺少了对应的代码段，需要检查是否有遗漏的代码或者删除多余的括号

    y = y.squeeze(1)  # 压缩张量y的第二维度，去除维度为1的维度

    spec = torch.stft(
        y,  # 输入的时域信号
        n_fft,  # FFT窗口大小
        hop_length=hop_size,  # 帧移大小
        win_length=win_size,  # 窗口长度
        window=hann_window[wnsize_dtype_device],  # 使用的窗口函数
        center=center,  # 是否在信号的中心进行STFT
        pad_mode="reflect",  # 信号边界填充模式
        normalized=False,  # 是否进行归一化
        onesided=True,  # 是否只返回单边频谱
        return_complex=False,  # 是否返回复数形式的STFT结果
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)  # 计算STFT结果的幅度谱
    return spec  # 返回幅度谱
def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    # 声明全局变量 mel_basis
    global mel_basis
    # 获取输入 spec 的数据类型和设备信息
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    # 获取 fmax、数据类型和设备信息的组合字符串
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    # 如果 mel_basis 中不存在对应的键值
    if fmax_dtype_device not in mel_basis:
        # 使用 librosa_mel_fn 函数生成 mel 频谱
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        # 将 mel 转换为 torch 张量，并存储到 mel_basis 中
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )
    # 将输入 spec 转换为 mel 频谱
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    # 对转换后的频谱进行归一化处理
    spec = spectral_normalize_torch(spec)
    # 返回处理后的频谱
    return spec


def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    # 如果输入 y 中存在小于 -1.0 的数值，则打印提示信息
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    # 如果输入 y 中存在大于 1.0 的数值，则打印提示信息
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))  # 打印张量 y 中的最大值

    global mel_basis, hann_window  # 声明 mel_basis 和 hann_window 为全局变量
    dtype_device = str(y.dtype) + "_" + str(y.device)  # 将张量 y 的数据类型和设备转换为字符串
    fmax_dtype_device = str(fmax) + "_" + dtype_device  # 将 fmax、数据类型和设备组合成字符串
    wnsize_dtype_device = str(win_size) + "_" + dtype_device  # 将 win_size、数据类型和设备组合成字符串
    if fmax_dtype_device not in mel_basis:  # 如果 fmax_dtype_device 不在 mel_basis 中
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)  # 调用 librosa_mel_fn 函数生成 mel 滤波器
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(  # 将 mel 转换为张量并存储在 mel_basis 中
            dtype=y.dtype, device=y.device
        )
    if wnsize_dtype_device not in hann_window:  # 如果 wnsize_dtype_device 不在 hann_window 中
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(  # 生成大小为 win_size 的汉宁窗口并存储在 hann_window 中
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(  # 使用反射模式对张量 y 进行填充
        y.unsqueeze(1),  # 在 y 的第一维度上添加一个维度
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),  # 填充的宽度
        mode="reflect",  # 填充模式为反射
    )  # 结束括号，可能是某个函数或方法的参数列表的结束
    y = y.squeeze(1)  # 压缩张量y的第二个维度，去除大小为1的维度

    spec = torch.stft(
        y,  # 输入的时域信号
        n_fft,  # 离散傅里叶变换的窗口大小
        hop_length=hop_size,  # 帧移大小
        win_length=win_size,  # 窗口长度
        window=hann_window[wnsize_dtype_device],  # 汉宁窗口函数
        center=center,  # 是否在信号的中心进行STFT
        pad_mode="reflect",  # 信号边界填充模式
        normalized=False,  # 是否进行归一化
        onesided=True,  # 是否只返回正频率部分
        return_complex=False,  # 是否返回复数形式的STFT结果
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)  # 计算STFT结果的幅度谱

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)  # 将幅度谱与梅尔滤波器矩阵相乘
    spec = spectral_normalize_torch(spec)  # 对频谱进行归一化处理
# 返回变量 spec 的值，即函数的结果。
```