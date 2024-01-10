# `so-vits-svc\modules\mel_processing.py`

```
# 导入 torch 库
import torch
# 导入 torch.utils.data 模块
import torch.utils.data
# 从 librosa.filters 模块中导入 mel 函数，并重命名为 librosa_mel_fn
from librosa.filters import mel as librosa_mel_fn

# 定义最大的 WAV 值
MAX_WAV_VALUE = 32768.0

# 定义动态范围压缩函数，使用 torch 库
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: 压缩因子
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

# 定义动态范围解压缩函数，使用 torch 库
def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: 用于压缩的压缩因子
    """
    return torch.exp(x) / C

# 定义光谱归一化函数，使用 torch 库
def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

# 定义光谱反归一化函数，使用 torch 库
def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

# 初始化 mel_basis 和 hann_window 字典
mel_basis = {}
hann_window = {}

# 定义 torch 光谱图函数
def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    # 如果 y 中的最小值小于 -1，则打印最小值
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    # 如果 y 中的最大值大于 1，则打印最大值
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    # 全局变量 hann_window
    global hann_window
    # 定义 dtype_device 变量
    dtype_device = str(y.dtype) + '_' + str(y.device)
    # 定义 wnsize_dtype_device 变量
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    # 如果 wnsize_dtype_device 不在 hann_window 中，则将其添加到 hann_window 中
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    # 使用反射模式对 y 进行填充
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    
    # 定义 y_dtype 变量
    y_dtype = y.dtype
    # 如果 y 的数据类型是 torch.bfloat16，则将其转换为 torch.float32
    if y.dtype == torch.bfloat16:
        y = y.to(torch.float32)

    # 使用短时傅里叶变换对 y 进行处理
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec).to(y_dtype)

    # 计算光谱的平方和，并加上 1e-6
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

# 定义将光谱图转换为 mel 频谱图的函数
def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    # 全局变量 mel_basis
    global mel_basis
    # 定义 dtype_device 变量
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    # 将 fmax 和 dtype_device 拼接成字符串，作为 mel_basis 字典的键
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    # 如果 fmax_dtype_device 不在 mel_basis 字典中
    if fmax_dtype_device not in mel_basis:
        # 使用 librosa_mel_fn 函数生成 Mel 频谱
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        # 将生成的 Mel 频谱转换成 PyTorch 张量，并存储在 mel_basis 字典中
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    # 将 mel_basis[fmax_dtype_device] 与 spec 进行矩阵相乘
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    # 对 spec 进行光谱归一化处理
    spec = spectral_normalize_torch(spec)
    # 返回处理后的 spec
    return spec
# 使用 PyTorch 计算音频信号的梅尔频谱图
def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # 计算音频信号的幅度谱
    spec = spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center)
    # 将幅度谱转换为梅尔频谱图
    spec = spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)
    
    # 返回梅尔频谱图
    return spec
```