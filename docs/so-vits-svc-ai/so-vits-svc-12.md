# SO-VITS-SVC源码解析 12

# `vdecoder/hifiganwithsnake/nvSTFT.py`

This is a function that loads an audio file from a given path and returns it as a NumPy array or a PyTorch tensor. It uses the librosa library to handle the audio file and the Pandas library to handle NumPy arrays.

It first checks if the input audio data is of type float or if it is NaN or inf. If it is NaN or inf, it returns an empty array or the input value, respectively.

If the input audio data is of type float, it is normalized to have a range of 0 to 1, except if the audio file is 16-bit INT. In this case, the audio data will be converted to the range [-1 to 1] of float.

The audio data is then converted to a PyTorch tensor and normalized to have a range of 0 to 1.

If the target sampling rate is different from the original sampling rate, it is adjusted to match the target sampling rate.

The function then returns the audio data as a NumPy array or a PyTorch tensor.


```py
import os

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

os.environ["LRU_CACHE_CAPACITY"] = "3"

def load_wav_to_torch(full_path, target_sr=None, return_empty_on_exception=False):
    sampling_rate = None
    try:
        data, sampling_rate = sf.read(full_path, always_2d=True)# than soundfile.
    except Exception as ex:
        print(f"'{full_path}' failed to load.\nException:")
        print(ex)
        if return_empty_on_exception:
            return [], sampling_rate or target_sr or 32000
        else:
            raise Exception(ex)
    
    if len(data.shape) > 1:
        data = data[:, 0]
        assert len(data) > 2# check duration of audio file is > 2 samples (because otherwise the slice operation was on the wrong dimension)
    
    if np.issubdtype(data.dtype, np.integer): # if audio data is type int
        max_mag = -np.iinfo(data.dtype).min # maximum magnitude = min possible value of intXX
    else: # if audio data is type fp32
        max_mag = max(np.amax(data), -np.amin(data))
        max_mag = (2**31)+1 if max_mag > (2**15) else ((2**15)+1 if max_mag > 1.01 else 1.0) # data should be either 16-bit INT, 32-bit INT or [-1 to 1] float32
    
    data = torch.FloatTensor(data.astype(np.float32))/max_mag
    
    if (torch.isinf(data) | torch.isnan(data)).any() and return_empty_on_exception:# resample will crash with inf/NaN inputs. return_empty_on_exception will return empty arr instead of except
        return [], sampling_rate or target_sr or 32000
    if target_sr is not None and sampling_rate != target_sr:
        data = torch.from_numpy(librosa.core.resample(data.numpy(), orig_sr=sampling_rate, target_sr=target_sr))
        sampling_rate = target_sr
    
    return data, sampling_rate

```

This code appears to be a implementation of a simple audio processing pipeline, where the input audio is first converted to a numpy array and then passed through a function that applies some pre-processing to the audio, such as dynamically normalizing the audio data. After that, the audio is passed through a function that generates a Mel-Frequency Cepstral Coefficients (MFCC) representation of the audio data, which is then stored in a dictionary that maps the mel frequency to the HFLWIN window and a dictionary that maps the original audio to the original audio window. The audio is then added to the mel-frequency dictionary and the dictionary is returned.


```py
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

class STFT():
    def __init__(self, sr=22050, n_mels=80, n_fft=1024, win_size=1024, hop_length=256, fmin=20, fmax=11025, clip_val=1e-5):
        self.target_sr = sr
        
        self.n_mels     = n_mels
        self.n_fft      = n_fft
        self.win_size   = win_size
        self.hop_length = hop_length
        self.fmin     = fmin
        self.fmax     = fmax
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = {}
    
    def get_mel(self, y, center=False):
        sampling_rate = self.target_sr
        n_mels     = self.n_mels
        n_fft      = self.n_fft
        win_size   = self.win_size
        hop_length = self.hop_length
        fmin       = self.fmin
        fmax       = self.fmax
        clip_val   = self.clip_val
        
        if torch.min(y) < -1.:
            print('min value is ', torch.min(y))
        if torch.max(y) > 1.:
            print('max value is ', torch.max(y))
        
        if fmax not in self.mel_basis:
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
            self.mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
            self.hann_window[str(y.device)] = torch.hann_window(self.win_size).to(y.device)
        
        y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_length)/2), int((n_fft-hop_length)/2)), mode='reflect')
        y = y.squeeze(1)
        
        spec = torch.stft(y, n_fft, hop_length=hop_length, win_length=win_size, window=self.hann_window[str(y.device)],
                          center=center, pad_mode='reflect', normalized=False, onesided=True)
        # print(111,spec)
        spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
        # print(222,spec)
        spec = torch.matmul(self.mel_basis[str(fmax)+'_'+str(y.device)], spec)
        # print(333,spec)
        spec = dynamic_range_compression_torch(spec, clip_val=clip_val)
        # print(444,spec)
        return spec
    
    def __call__(self, audiopath):
        audio, sr = load_wav_to_torch(audiopath, target_sr=self.target_sr)
        spect = self.get_mel(audio.unsqueeze(0)).squeeze(0)
        return spect

```

这段代码是使用 FastSTFT(FastSpeechTransformer)模型的 API 库中训练一个名为 "stft" 的 STFT(Speech Transformer)，用于语音识别任务。

具体来说，该代码会执行以下操作：

1. 加载训练数据：使用 GST(Google Cloud Storage)等库读取训练数据，并将其加载到内存中。

2. 准备数据：将训练数据中的每个语音信号转换为浮点数向量，并指定每个样本的声学特征(如语音的音量、频率等)。

3. 创建 STFT 模型：使用 FastSTFT(FastSpeechTransformer)模型的 API 库中训练一个名为 "stft" 的 STFT。

4. 构建数据流程：使用 FastSTFT(FastSpeechTransformer)模型的 API 库中训练一个名为 "stft" 的 STFT，使用训练数据中的每个语音信号作为输入，并输出一个二进制数据流，其中每个二进制数据样本包含一个声学特征。

5. 训练模型：使用 FastSTFT(FastSpeechTransformer)模型的 API 库中训练一个名为 "stft" 的 STFT，使用训练数据中的每个语音信号作为输入，并输出一个二进制数据流，其中每个二进制数据样本包含一个声学特征。

6. 使用模型：使用训练好的模型 "stft"，对新的语音数据进行 STFT 分析，并提取其中的声学特征。


```py
stft = STFT()

```

# `vdecoder/hifiganwithsnake/utils.py`

这段代码的主要作用是定义一个名为`plot_spectrogram`的函数，用于绘制一个给定光谱图的低级散射强度分布。

具体来说，它使用Python的`glob`库和`os`库来获取要分析的光谱图文件，并导入`matplotlib`库中的`pyplot`函数和`torch`库中的`weight_norm`函数，以便在函数中使用它们。

函数的主要部分包括以下几个步骤：

1. 导入了`glob`库和`os`库，用于获取要分析的光谱图文件。

2. 导入了`matplotlib`库中的`pyplot`函数和`torch`库中的`weight_norm`函数，以便在函数中使用它们。

3. 定义了一个名为`plot_spectrogram`的函数，它接受一个名为`spectrogram`的参数，这个参数可能是一个由`torch`库中的`torch.Tensor`类组成的张量。

4. 在函数中使用`plot_spectrogram`函数绘制了给定光谱图的低级散射强度分布。这个绘图使用了`imshow`函数来显示张量中的值，并使用`colorbar`函数来显示颜色维度。

5. 使用`draw`函数来请求图形窗口中的画布，并使用`close`函数关闭图形窗口。

函数可能会在未来的应用中发挥作用，具体取决于它的具体实现和使用方式。


```py
import glob
import os

# matplotlib.use("Agg")
import matplotlib.pylab as plt
import torch
from torch.nn.utils import weight_norm


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


```



这段代码定义了四个函数，用于对一个神经网络中的权重进行归一化和处理。

第一个函数 `init_weights` 接受一个参数 `m`，表示神经网络的输入张量大小。函数内部根据输入张量的大小，将权重初始化为均值为 `mean`，标准差为 `std` 的正态分布中。

第二个函数 `apply_weight_norm` 与 `init_weights` 类似，但仅仅对输入张量中的前 `kernel_size` 行进行归一化处理。

第三个函数 `get_padding` 接受一个参数 `kernel_size`，表示神经网络卷积层中的一个核大小，对输入张量进行归一化处理。该函数使用公式 `kernel_size = kernel_size / dilation` 计算归一化后的尺寸，其中 `dilation` 参数表示卷积层的 dilation 值，即对输入张量进行的尺寸变化比例。

第四个函数 `mean_function` 和 `std_function` 分别对传入的 `mean` 和 `std` 函数进行调用，并在函数内部使用了它们。这两个函数的作用是执行平均化和标准化操作，分别对神经网络中的输入张量进行处理。


```py
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


```

这段代码定义了三种函数，分别是 `load_checkpoint`、`save_checkpoint` 和 `del_old_checkpoints`。它们的具体实现如下：

1. `load_checkpoint`：

这个函数的作用是加载一个指定文件夹下的模型checkpoint，并将其存储为在指定设备（device）上搜寻的模型checkpoint。

它首先检查指定的文件是否可用，如果是，就打印 "Loading '{}'". 接着，它加载 checkpoint 字典，并将其存储在 `device` 设备上。最后，它打印 "Complete."，表示加载完成。

2. `save_checkpoint`：

这个函数的作用是将指定的模型保存为指定文件夹下的 checkpoint。

它首先打印 "Saving checkpoint to ..."，然后使用 `torch.save` 函数将指定的模型保存到指定的文件。最后，它打印 "Complete."，表示保存完成。

3. `del_old_checkpoints`：

这个函数的作用是删除指定文件夹中旧的 checkpoint，仅在指定的模型数量内保留最近的 n_models 个 checkpoint。

它首先打印 "旧 checkpoint files will be deleted"，然后遍历指定文件夹下的所有 checkpoint 文件，将指定的文件从列表中删除。接着，它创建一个以 "old" 为后缀，以指定模型数量为前缀的新文件夹，并将旧文件移动到新文件夹中。


```py
def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def del_old_checkpoints(cp_dir, prefix, n_models=2):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern) # get checkpoint paths
    cp_list = sorted(cp_list)# sort by iter
    if len(cp_list) > n_models: # if more than n_models models are found
        for cp in cp_list[:-n_models]:# delete the oldest models other than lastest n_models
            open(cp, 'w').close()# empty file contents
            os.unlink(cp)# delete file (move to trash when using Colab)


```

该函数的作用是扫描指定目录下的所有 Checkpoint 文件，返回最后一个被扫描到的文件。

具体实现过程如下：

1. 首先，定义一个名为 scan_checkpoint 的函数，它接受两个参数：cp_dir 和 prefix。cp_dir 是指定要扫描的 Checkpoint 目录，prefix 是一个字符串，用于指定文件名中的前缀。

2. 在函数内部，使用 os.path.join 函数将 cp_dir 和 prefix 拼接成一个文件路径，然后使用 glob.glob 函数遍历指定目录下的所有文件。这里使用的是 join 函数将目录和文件名拼接在一起，这样就可以获取到文件名中的前缀。

3. 使用 if 语句检查是否找到了被遍历的文件。如果找到了文件，使用 sorted 函数对文件列表进行排序，并返回排好序的最后一个文件。如果找不到被遍历的文件，返回 None，表示没有找到匹配的文件。

4. 最后，函数返回排好序的最后一个文件，如果找到了则返回该文件路径，否则返回 None。


```py
def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


```

# `vdecoder/hifiganwithsnake/alias/act.py`

这段代码定义了一个名为 `Activation1d` 的类，继承自 PyTorch 中的 `nn.Module` 类。这个类的实现了一个可以对输入数据进行加权下采样和加权上采样操作的函数。

该类中包含了一个参数 `activation`，它是一个可以定义为 `'relu'`、`'sigmoid'` 或 `'tanh'` 的字符串，用于指定激活函数的类型。

该类中还包含两个参数 `up_ratio` 和 `down_ratio`，它们分别定义了输入数据的下采样和上采样倍率。在上采样时，`up_ratio` 指定了输入数据中 Up Sampling 的比例，`down_ratio` 指定了输入数据中 Down Sampling 的比例。在 Down采样时，`up_ratio` 指定了输入数据中 Down Sampling 的比例，`down_ratio` 指定了输入数据中 Up Sampling 的比例。

另外，该类中还包含一个名为 `upsample` 的函数和一个名为 `downsample` 的函数。这两个函数分别实现了输入数据的下采样和上采样操作。

最后，该类的 `__init__` 方法是在 `__init__` 函数中执行的，它首先调用父类的 `__init__` 方法，然后设置本类的参数。


```py
# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import pow, sin
from torch.nn import Parameter

from .resample import DownSample1d, UpSample1d


class Activation1d(nn.Module):
    def __init__(self,
                 activation,
                 up_ratio: int = 2,
                 down_ratio: int = 2,
                 up_kernel_size: int = 12,
                 down_kernel_size: int = 12):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x


```

This is a implementation of the SnakeBeta class in the TorchScript library for neural network building. The SnakeBeta class is a version of a neural network input layer that uses log-space信贷动态规划(log-space credit dynamic programming, LCCDP) to learn.

The `__init__` method initializes the neural network's parameters, including the input, the logscale of thealpha, and the magnitude of the beta, and also sets the initial values of the parameters.

The `forward` method applies the neural network to the input element-wise. It computes the output of the network as `x + 1/b * sin^2 (xa)`, where `x` is the input, `b` is the bandwidth, `a` is the首项， and `sin(x * alpha)` is the input `x` processed through the trained parameters.

In this implementation, the parameter `alpha_logscale` is set to 1, meaning that the logscale of the `alpha` parameter is used, and the parameter `beta_logscale` is also set to 1, meaning that the logscale of the `beta` parameter is used. If you want to use a linearly proportional relationship between the input and the output, you can set the `alpha_logscale` and `beta_logscale` parameters to 0.

Also, if you want to optimize the parameters, you can set the `requires_grad` parameter of the alpha and beta parameters to `True`, which allows the optimization algorithms to move the parameters during backpropagation.


```py
class SnakeBeta(nn.Module):
    '''
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        '''
        super(SnakeBeta, self).__init__()
        self.in_features = in_features
        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta = x + 1/b * sin^2 (xa)
        '''
        alpha = self.alpha.unsqueeze(
            0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)
        return x


```

这段代码定义了一个名为Mish的类，它继承自PyTorch中的nn.Module类。Mish激活函数被引入自“Mish: A Self-Regularized Non-Monotonic Neural Activation Function”论文，该论文的URL为https://arxiv.org/abs/1908.08681。

Mish类中包含了一个__init__方法，该方法在创建Mish实例时执行。在__init__方法中，创建了一个与父类相同的参数列表，然后调用父类的构造函数。

Mish类中包含了一个forward方法，该方法在传入一个输入张量x时执行。在forward方法中，首先将输入张量x传递给Mish激活函数，然后根据设置的up_ratio和down_ratio对结果进行归一化，并使用自定义的up_kernel_size和down_kernel_size对结果进行上采样和下采样。

接着，定义了一个名为SnakeAlias的类，该类也继承自nn.Module类。SnakeAlias类中包含了一个__init__方法，该方法在创建实例时执行。在__init__方法中，定义了与Mish类中参数列表相同的参数，并创建了一个Mish激活函数。

最后，定义了一个激活函数SnakeBeta，并将其传递给Mish类中的forward方法。


```py
class Mish(nn.Module):
    """
    Mish activation function is proposed in "Mish: A Self 
    Regularized Non-Monotonic Neural Activation Function" 
    paper, https://arxiv.org/abs/1908.08681.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SnakeAlias(nn.Module):
    def __init__(self,
                 channels,
                 up_ratio: int = 2,
                 down_ratio: int = 2,
                 up_kernel_size: int = 12,
                 down_kernel_size: int = 12,
                 C = None):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = SnakeBeta(channels, alpha_logscale=True)
        self.upsample = UpSample1d(up_ratio, up_kernel_size, C)
        self.downsample = DownSample1d(down_ratio, down_kernel_size, C)

    # x: [B,C,T]
    def forward(self, x, C=None):
        x = self.upsample(x, C)
        x = self.act(x)
        x = self.downsample(x)

        return x
```

# `vdecoder/hifiganwithsnake/alias/filter.py`

这段代码是一个PyTorch实现，它将一个输入张量（通常是一个ndarray或一个复杂的张量）中的值归一化到[-1, 1]范围内。

具体来说，这段代码首先使用PyTorch中的math模块中的torch.sinc函数，如果系统中已经安装了julius库，则直接使用该函数。否则，它将执行以下计算：
```py
x = torch.tensor([-1, 1] * 1000, device=x.device, dtype=x.dtype)
sinc_map = sinc(x)
```
这段代码将输入张量中的每个值映射到[-1, 1]范围内，从而使得整个张量的值都能够在[-1, 1]范围内。


```py
# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

if 'sinc' in dir(torch):
    sinc = torch.sinc
else:
    # This code is adopted from adefossez's julius.core.sinc under the MIT License
    # https://adefossez.github.io/julius/julius/core.html
    #   LICENSE is in incl_licenses directory.
    def sinc(x: torch.Tensor):
        """
        Implementation of sinc, i.e. sin(pi * x) / (pi * x)
        __Warning__: Different to julius.sinc, the input is multiplied by `pi`!
        """
        return torch.where(x == 0,
                           torch.tensor(1., device=x.device, dtype=x.dtype),
                           torch.sin(math.pi * x) / math.pi / x)


```

这段代码定义了一个名为 "kaiser_sinc_filter1d" 的函数，它接受三个参数：cutoff（阈值），half_width（ half-width），kernel_size（内核大小）。函数返回一个低通滤波器，对传入的阈值、半带宽和内核大小进行处理，然后将其返回。

函数内部使用了两种方法来实现 Kuser-Sinc 低通滤波器。第一种方法是基于 Kuser-Sinc 算法的实现，需要指定半带宽（half-width）。第二种方法是基于已知的实现，需要指定阈值（cutoff），同时对半带宽进行归一化处理。

函数使用了两个局部变量 A 和 beta，其中 A 是一个根据 half-width 计算得到的指数加权函数，用于实现 Kuser-Sinc 算法。beta 是一个根据 A 值计算得到的参数，用于对 A 的值进行平滑处理。函数还使用了另一个局部变量 delta_f，用于在 half-width 上计算一个与 A 相关的权重系数，用于平滑 A 的值。

函数的实现主要分为三部分：1）实现 Kuser-Sinc 算法，用于计算半带宽 A；2）实现对半带宽 A 的归一化处理；3）根据传入的阈值和半带宽参数，返回一个低通滤波器。


```py
# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
#   LICENSE is in incl_licenses directory.
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size): # return filter [1,1,kernel_size]
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2

    #For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
    else:
        beta = 0.
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = (torch.arange(-half_size, half_size) + 0.5)
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        # Normalize filter to have sum = 1, otherwise we will have a small leakage
        # of the constant component in the input signal.
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)

    return filter


```

这段代码是用于实现一个 1D 卷积神经网络 (CNN) 的。这个 CNN 支持残差连接 (residual connection)，也就是说，在输出每个卷积层之后，还会输出一个残差块，用于对输入数据进行进一步的传递。

在这个实现中，这个 CNN 有以下几个参数：

- `filter`: 一个 1D 可训练的 KIRSTEIN 滤波器，用于对输入数据进行曲线拟合。这个滤波器是通过对 `x` 进行直线拟合得到的，然后通过指数加权来对数据进行加权。
- `half_width`: 这个参数控制了卷积核的宽度，也就是对于每个输入数据点，卷积核只对数据的第一半部分进行卷积运算。
- `kernel_size`: 这个参数控制了卷积核的大小，也就是对于每个输入数据点，卷积核只对数据的一个局部区域进行卷积运算。
- `even`: 这个参数用于判断是否对输入数据进行平分操作。如果 `even` 为 1，则表示输入数据是奇数长度，否则是偶数长度。
- `pad_left`: 这个参数用于对输入数据进行平分操作，当 `even` 为 1 时，这个参数用于对奇数长度的输入数据进行平分，当 `even` 为 0 时，这个参数用于对偶数长度的输入数据进行平分。
- `pad_right`: 这个参数用于对输入数据进行平分操作，当 `even` 为 1 时，这个参数用于对奇数长度的输入数据进行平分，当 `even` 为 0 时，这个参数用于对偶数长度的输入数据进行平分。
- `stride`: 这个参数控制了卷积核对输入数据进行步幅操作时，每个输入数据点的步幅。
- `padding`: 这个参数用于对输入数据进行残差连接，当 `even` 为 1 时，这个参数用于在输入数据后面添加差分块，当 `even` 为 0 时，这个参数用于在输入数据前面添加差分块。
- `padding_mode`: 这个参数用于指定在创建张量时，如何对输入数据进行填充。可以指定为 `'constant'`、`'inf'` 或 `'symmetric'` 中的任意一个。
- `filter`: 这个参数是一个 1D 可训练的 KIRSTEIN 滤波器，用于对输入数据进行曲线拟合。这个滤波器是通过对 `x` 进行直线拟合得到的，然后通过指数加权来对数据进行加权。

这个 CNN 的实现中，对输入数据进行了平分操作，并且在输出数据时，还对数据进行了残差连接。


```py
class LowPassFilter1d(nn.Module):
    def __init__(self,
                 cutoff=0.5,
                 half_width=0.6,
                 stride: int = 1,
                 padding: bool = True,
                 padding_mode: str = 'replicate',
                 kernel_size: int = 12,
                 C=None):
        # kernel_size should be even number for stylegan3 setup,
        # in this implementation, odd number is also possible.
        super().__init__()
        if cutoff < -0.:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = (kernel_size % 2 == 0)
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)
        self.conv1d_block = None
        if C is not None:
            self.conv1d_block = [nn.Conv1d(C,C,kernel_size,stride=self.stride, groups=C, bias=False),]
            self.conv1d_block[0].weight = nn.Parameter(self.filter.expand(C, -1, -1))
            self.conv1d_block[0].requires_grad_(False)

    #input [B, C, T]
    def forward(self, x):
        if self.conv1d_block[0].weight.device != x.device:
            self.conv1d_block[0] = self.conv1d_block[0].to(x.device)
        if self.conv1d_block is None:
            _, C, _ = x.shape

            if self.padding:
                x = F.pad(x, (self.pad_left, self.pad_right),
                            mode=self.padding_mode)
            out = F.conv1d(x, self.filter.expand(C, -1, -1),
                            stride=self.stride, groups=C)
        else:
            if self.padding:
                x = F.pad(x, (self.pad_left, self.pad_right),
                            mode=self.padding_mode)
            out = self.conv1d_block[0](x)

        return out
```

# `vdecoder/hifiganwithsnake/alias/resample.py`

This appears to be a module within a neural network that is responsible for processing a sequence of data. The module has a single input, `x`, and an optional parameter, `C`, which is the number of channels in the input data.

The module first checks if the `C` parameter is set and if the input `x` has the correct shape. If `C` is set, the input is reshaped to have a shape of `(batch_size, max_seq_length, input_channel_number)` and then padded to have the same shape as the input data.

If `C` is not set, the input data is passed through the first convolutional neural network block. This block has a single convolutional layer with a randomly initialized filter weight, a linear layer with a learned linear activation function, and a group attribute set to `C` to allow the convolutional and linear layers to communicate with each other. The output of the convolutional layer is passed through a transpose layer with a learned linear activation function and then added to the input data.

The output of the convolutional neural network block is then passed through a transpose layer with a learned linear activation function and a group attribute set to `C`. This allows the output of the convolutional neural network block to be added to the input data.

The module then returns the input data.


```py
# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import torch.nn as nn
from torch.nn import functional as F

from .filter import LowPassFilter1d, kaiser_sinc_filter1d


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, C=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio,
                                      half_width=0.6 / ratio,
                                      kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)
        self.conv_transpose1d_block = None
        if C is not None:
            self.conv_transpose1d_block = [nn.ConvTranspose1d(C,
                                                            C,
                                                            kernel_size=self.kernel_size,
                                                            stride=self.stride, 
                                                            groups=C, 
                                                            bias=False
                                                            ),]
            self.conv_transpose1d_block[0].weight = nn.Parameter(self.filter.expand(C, -1, -1).clone())
            self.conv_transpose1d_block[0].requires_grad_(False)
            
            

    # x: [B, C, T]
    def forward(self, x, C=None):
        if self.conv_transpose1d_block[0].weight.device != x.device:
            self.conv_transpose1d_block[0] = self.conv_transpose1d_block[0].to(x.device)
        if self.conv_transpose1d_block is None:
            if C is None:
                _, C, _ = x.shape
            # print("snake.conv_t.in:",x.shape)
            x = F.pad(x, (self.pad, self.pad), mode='replicate')
            x = self.ratio * F.conv_transpose1d(
                x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
            # print("snake.conv_t.out:",x.shape)
            x = x[..., self.pad_left:-self.pad_right]
        else:
            x = F.pad(x, (self.pad, self.pad), mode='replicate')
            x = self.ratio * self.conv_transpose1d_block[0](x)
            x = x[..., self.pad_left:-self.pad_right]
        return x


```

这段代码定义了一个名为 "DownSample1d" 的类，继承自 PyTorch 中的 nn.Module 类。这个类的实现主要为了对输入数据进行 down-sample（下采样）操作，将输入数据中的每个元素按照一定比例进行缩放，使得输入数据在通道上有一定的规律性。

在类的初始化方法 "__init__" 中，定义了三个参数：

1. ratio： down-sample 的比例，默认值为 2。
2. kernel_size： 卷积核的大小，如果为 None，则表示使用默认的尺寸，即 6*ratio//2 个 channels。
3. C：一个 None 类型的参数，表示卷积核的注意力系数，可以理解为一个缩放因子，用于控制 down-sample 对输入通道的贡献程度。

在 "__init__" 方法中，还调用了父类的初始化方法，以便正确地初始化 down-sample。

在 forward 方法中，主要实现了一个低通滤波器，对输入数据 x 进行 down-sample 操作。具体实现包括以下几个步骤：

1. 如果指定了 kernel_size，则创建一个 2x2 的卷积核，并将输入数据 x 按比例衰减至该尺寸。
2. 创建一个 1x2 的卷积核，对输入数据 x 进行直接前向传递，并将输出结果按比例衰减至原来的一半。
3. 创建一个 2x2 的卷积核，使用 down-sample 操作对输入数据 x 进行平滑处理，将输出结果按比例衰减至原来的一半。
4. 保存输出结果，并返回 down-sample 后的结果。

总的来说，这段代码主要实现了一个 down-sample 的功能，可以对输入数据进行一定的缩放处理，使得输入数据在通道上有一定的规律性。


```py
class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, C=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio,
                                       half_width=0.6 / ratio,
                                       stride=ratio,
                                       kernel_size=self.kernel_size,
                                       C=C)


    def forward(self, x):
        xx = self.lowpass(x)

        return xx
```

# `vdecoder/hifiganwithsnake/alias/__init__.py`

这段代码是从[https://github.com/junjun3518/alias-free-torch](https://github.com/junjun3518/alias-free-torch)项目中提取出来的，用于在PyTorch中实现自定义卷积操作的类。它包括从.act、.filter和.resample导入的三个函数：Activator、Conv2d和BatchNorm2d。这些函数可以用于实现各种卷积操作，例如对输入数据进行归一化、加权聚合等。


```py
# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

from .act import *  # noqa: F403
from .filter import *  # noqa: F403
from .resample import *  # noqa: F403

```

# `vdecoder/nsf_hifigan/env.py`

这段代码定义了一个名为 AttrDict 的类，该类采用Python中的 dict类作为其父类，并添加了一个 `__init__` 方法。

在 AttrDict 的 `__init__` 方法中，通过 `*args` 和 `**kwargs` 参数分别获取输入参数列表和字典类型的 keyword arguments，并将它们添加到 parent class（即 dict类）中。

接着，定义了一个名为 build_env 的函数，该函数接受一个 config（配置参数）、config_name（配置文件名）和一个 path（输出路径）参数。函数首先创建一个名为 config_name 且与传入路径相同的文件夹，如果该文件夹已存在，则进行复制并覆盖原有文件。然后，将 config 对象赋值给 config_name 文件夹中的文件。


```py
import os
import shutil


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))

```

# `vdecoder/nsf_hifigan/models.py`

这段代码包括以下几个部分：

1. 导入需要使用的Python库，包括json、os、numpy、torch和相关的nn、nn.functional、torch.nn、torch.nn.functional.torchmap、torch.nn.utils.体重初始化、pop等库。

2. 从nn.nn模块中定义了几个类，如AvgPool1d、Conv1d、Conv2d和ConvTranspose1d，这些类在网络层中起到重要作用。

3. 从nn.nn模块的extras.流产中定义了一个AttrDict，这个AttrDict用于存储一个自定义字典，里面包括一些权重初始化、训练范围等策略。

4. 从get_padding函数中定义了一个长度为16的二维数组，用于实现对不同输入大小下的padding策略。

5. 加载预训练的权重，并执行初始化操作。


```py
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from .env import AttrDict
from .utils import get_padding, init_weights

LRELU_SLOPE = 0.1


```

这段代码定义了一个名为 `load_model` 的函数，用于加载一个预训练的模型，并指定使用哪种设备来运行模型。

具体来说，代码首先定义了一个名为 `load_config` 的函数，用于从预训练模型中加载配置信息。这个配置信息包含模型的架构、损失函数、优化器等参数。函数首先从传入的模型路径中加载配置文件，然后使用 `os.path.join` 函数将路径中的目录和配置文件名拼接起来，得到配置文件的完整路径。接着，函数使用 `with` 语句打开配置文件，并逐行读取文件内容。函数将文件内容转换为 JSON 格式，并从文件中提取出模型的配置信息。最后，函数使用 `json.loads` 函数将 JSON 格式的数据转换为 Python 字典类型，并返回该字典类型的 `h` 变量。

接下来，定义了一个名为 `load_model` 的函数，用于加载预训练模型并指定使用哪种设备来运行模型。函数中使用 `load_config` 函数从预训练模型中加载配置信息，并使用 `Generator` 类从配置信息中加载出 `Generator` 实例，该实例使用指定的设备(在函数中通过 `device` 参数指定)加载和初始化模型。函数还通过调用 `Generator` 实例的 `load_state_dict` 方法，从配置文件中加载出模型的参数。最后，函数将加载出的模型和配置信息缓存起来，以便在后续使用。


```py
def load_model(model_path, device='cuda'):
    h = load_config(model_path)

    generator = Generator(h).to(device)

    cp_dict = torch.load(model_path, map_location=device)
    generator.load_state_dict(cp_dict['generator'])
    generator.eval()
    generator.remove_weight_norm()
    del cp_dict
    return generator, h

def load_config(model_path):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    return h


```

This is a module that creates a convolutional neural network (CNN) and applies a pre-trained weights to the network.

The network consists of two convolutional layers with differentkernel\_size, one with dilation\_mean and one with dilation\_std, two pooling layers with the samekernel\_size and pool\_type, and one output layer.

The weight\_norm function is used to apply the pre-trained weights from the convolutional layers.

The forward method applies the pre-trained weights to the input and applies the forwardPass function to the convolutional and pooling layers.


```py
class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


```



这段代码定义了一个名为ResBlock2的类，继承自PyTorch中的nn.Module类。

在ResBlock2的构造函数中，首先调用父类的构造函数，然后定义了h参数，以及 channels参数和kernel_size参数。之后，定义了一个convs列表，其中包含两个Conv1d类，每个Conv1d层包含一个卷积层和一个权重激活函数。通过apply方法将初始化权重应用于convs列表中的每个层。

在forward方法中，循环遍历convs列表中的每个层，对每个层中的输入x执行以下操作：通过F.leaky_relu函数执行一个带LRELU_SLope偏置的ReLU激活函数，然后将结果与输入x相加，得到一个新的output x。最后，将最后一个输出值返回。

另外，定义了一个remove_weight_norm方法，该方法用于移除每个conv层中的权重 normalization。


```py
class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


```

这是一个PyTorch实现的函数，名为```py
```
`).forward(in_chunk, out_chunk, half=False)
```py

`in_chunk`参数是一个张量，表示输入数据，输出是一个张量，表示经过预处理后的输出数据。

`out_chunk`参数是一个张量，表示经过预处理后的输出数据，这个张量的大小是输入张量的大小减半。

`half`参数是一个布尔值，表示是否是 half 浮点数。

函数的作用是先对输入数据 `in_chunk` 进行预处理，然后计算输出数据 `out_chunk`。

具体来说，函数对输入数据 `in_chunk` 中的数据进行以下操作：

1. 对数据进行 half 浮点数转换，使得输入数据可以被理解为一个浮点数序列。
2. 对数据进行归一化，使得数据都落在 [0, 1] 范围内。
3. 对数据进行插值，使用高斯分布插值，使得数据更加平滑。
4. 对数据进行降噪，使用自定义的噪响模型。
5. 对输入数据 `in_chunk` 进行索引化，对每个样本计算输出数据 `out_chunk`。

函数返回经过预处理后的输出数据 `out_chunk`。


```
class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    @torch.no_grad()
    def forward(self, f0, upp):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0 = f0.unsqueeze(-1)
        fn = torch.multiply(f0, torch.arange(1, self.dim + 1, device=f0.device).reshape((1, 1, -1)))
        rad_values = (fn / self.sampling_rate) % 1  ###%1意味着n_har的乘积无法后处理优化
        rand_ini = torch.rand(fn.shape[0], fn.shape[2], device=fn.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        is_half = rad_values.dtype is not torch.float32
        tmp_over_one = torch.cumsum(rad_values.double(), 1)  # % 1  #####%1意味着后面的cumsum无法再优化
        if is_half:
            tmp_over_one = tmp_over_one.half()
        else:
            tmp_over_one = tmp_over_one.float()
        tmp_over_one *= upp
        tmp_over_one = F.interpolate(
            tmp_over_one.transpose(2, 1), scale_factor=upp,
            mode='linear', align_corners=True
        ).transpose(2, 1)
        rad_values = F.interpolate(rad_values.transpose(2, 1), scale_factor=upp, mode='nearest').transpose(2, 1)
        tmp_over_one %= 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
        rad_values = rad_values.double()
        cumsum_shift = cumsum_shift.double()
        sine_waves = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)
        if is_half:
            sine_waves = sine_waves.half()
        else:
            sine_waves = sine_waves.float()
        sine_waves = sine_waves * self.sine_amp
        uv = self._f02uv(f0)
        uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp, mode='nearest').transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


```py

This is a class definition for a `SourceModuleHnNSF` model that generates a cepstral basis representation (a type of Romanescano frequency table) for a given input signal `x`, where `x` is a 1-dimensional tensor of high-frequency samples.

The `SourceModuleHnNSF` model takes as input the high-frequency samples `x`, and outputs a 1-dimensional tensor `uv` of the cepstral basis representation.

The cepstral basis representation is computed as follows:

1. The input `x` is passed through a feedback山谷消噪的预处理过程， which removing some high-frequency noise and interpolating others.
2. Then, the input `x` is passed through a升频过程， which increases the sample rate to the specified sampling rate.
3. Next, the output of the升频过程 is passed through a混频过程， which combines the input signal with a stable noise signal that follows an additive Gaussian noise distribution with a given noise standard deviation `add_noise_std`. The resulting output is then passed through a夏皮逊调谐器的F0阈值过程， which sets the threshold from the output of the混频过程 to a given frequency `F0`.
4. Finally, the output of the夏皮逊调谐器 is passed through the左线性变换和左归一化变换， which result in the cepstral basis representation `uv`.

Note that the `SourceModuleHnNSF` model assumes that the input `x` has already been processed to produce a 1-dimensional tensor of high-frequency samples. If not, the input tensor should be obtained by applying some additional pre-processing techniques, such as取整、插值等。


```
class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge


```py

This is a PyTorch implementation of a Upsample layer. It takes a source tensor `x` and an initial focus value `f0`, and applies a series of upsamplings to the input tensor.

The layer has a fixed number of upsamplings, which is determined by the `self.num_upsamples` parameter (default: 4). During the training phase, the layer applies the `F.leaky_relu` activation function to the input tensor, and also applies the `self.upsample_rates` parameter (e.g., a list of integers) to the convolutional weights.

The `self.resblocks` list is a list of residual block modules, which are applied to the input tensor after each upsampling. Each residual block has three parts:

* The first part is a sub-module (residual connection) that applies the given convolutional neural network (CNN) block to the input tensor.
* The second part is a sub-module (dilation tensor) that applies the given dilation factor to the residual tensor.
* The third part is the final residual tensor that is the result of the upsampling operation.

The `self.noise_convs` list is a list of noise concentration blocks, which are used to prevent overshooting in the upsampled tensor. Each block has two parts:

* The first part is the input tensor that is passed through the noise concentration layer.
* The second part is the residual tensor that is passed through the noise concentration layer.

This implementation is based on the PyTorch paper "Upsampling Layer with Leaky ReLU Activation for Robustness Analysis".


```
class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=h.sampling_rate,
            harmonic_num=8
        )
        self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            c_cur = h.upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i), h.upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))
            if i + 1 < len(h.upsample_rates):  #
                stride_f0 = int(np.prod(h.upsample_rates[i + 1:]))
                self.noise_convs.append(Conv1d(
                    1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2))
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        ch = h.upsample_initial_channel
        for i in range(len(self.ups)):
            ch //= 2
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = int(np.prod(h.upsample_rates))

    def forward(self, x, f0):
        har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


```py

This is a PyTorch implementation of a convolutional neural network (CNN) model. It consists of a pre-trained ResNet18 model and a few custom layers to improve the last few layers of the network.

The custom layers include a leaky ReLU activation function, which is a variation of the ReLU activation function that applies a small, leaky value to the output at the time of a incoming dynamic reference. This allows for better spatial understanding of the output, as the ReLU will continue to output 0 for negative values. This ReLU is used in the `fmap` array, which is a list of the output from each of the convolutional layers.

The `norm_f` function is a custom normalization function that applies squashing and optionally, normalization to the output of each layer. This is done to improve the numerical stability of the network and prevent large/ NaN values.

The `Conv2d` and `Conv2d` classes are the custom convolutional layers that add depth and an extra dimension to the input feature. The `norm_f` function is applied to the output of each convolutional layer.

The `forward` method takes an input tensor `x` and returns the output tensor.

Note that this model is pre-trained and can be used for various tasks like image classification, object detection, etc. The last few layers of the network might require more根据自己的 specific task-specific architecture adjustments to achieve the desired performance.


```
class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


```py

这段代码定义了一个名为MultiPeriodDiscriminator的类，继承自PyTorch中的nn.Module类。

MultiPeriodDiscriminator类包含一个构造函数(__init__)，该函数接收一个参数perions，并在__init__函数内部初始化。如果perions参数为None，则默认值为一个包含6个周期(2, 3, 5, 7, 11)的列表。

MultiPeriodDiscriminator类包含一个名为discriminators的列表，该列表包含一个类DiscriminatorP的实例。

MultiPeriodDiscriminator类包含一个名为forward的函数，该函数接收两个参数y和y_hat，并返回一个包含y_d_rs,y_d_gs和fmap_rs,fmap_gs的元组。

在forward函数中，首先定义了一个空列表y_d_rs,y_d_gs和fmap_rs,fmap_gs。

然后，使用for循环遍历discriminators列表中的每个实例，并对其进行forward计算。

对于每个discriminator实例，首先使用其传入的函数d(y)计算y_d_r,fmap_r，然后将这些值添加到对应的列表中。

最后，对于每个discriminator实例，将其生成的y_d_rs,y_d_gs和fmap_rs,fmap_gs分别返回。

总结起来，MultiPeriodDiscriminator类通过定义一个DiscriminatorP的实例来定义一个具有多个周期的分类器，可以接受任意数量的输入，并输出每个周期生成的预测结果。


```
class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods=None):
        super(MultiPeriodDiscriminator, self).__init__()
        self.periods = periods if periods is not None else [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList()
        for period in self.periods:
            self.discriminators.append(DiscriminatorP(period))

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


```py

这段代码定义了一个名为 `DiscriminatorS` 的类，继承自 PyTorch 的 `nn.Module` 类。这个类的实现与图像分类任务相关，它将输入数据 x 传递给一个名为 `forward` 的方法，输出一个包含预测结果 fmap 的元组。

在类的初始化函数 `__init__` 中，首先调用父类的初始化函数，然后设置 `use_spectral_norm` 的参数。如果 `use_spectral_norm` 为 False，则按照默认情况使用 dropout 激活函数对输入数据进行归一化，使得输入数据具有相同的缩放因子。

`__init__` 之后，定义了一个 `norm_f` 函数，用于计算每个卷积层的归一化参数。然后，定义了一个 `Conv1d` 层，用于提取输入数据中的特征图。接着，定义了一个 `norm_f` 函数，对每个卷积层进行归一化处理。最后，定义了一个 `Conv1d` 层，用于对最后一个卷积层的输出进行归一化处理。

在 `forward` 方法中，对输入数据 x 进行处理，首先执行每个卷积层的 forward 方法，然后执行一个 F.leaky_relu 激活函数，对每个卷积层的输出进行非线性变换。接着，将所有卷积层的输出进行拼接，得到一个归一化后的特征图。最后，将特征图输入到 `nn.functional.linear` 函数中，得到一个包含预测结果 fmap 的元组。


```
class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


```py

这段代码定义了一个名为 MultiScaleDiscriminator 的类，继承自 PyTorch 的 nn.Module 类。

MultiScaleDiscriminator 用于实现图像分类任务，它由多个 discriminator（用于对输入数据进行分类）和多个平均池化层（用于提取特征图）组成。

具体来说，MultiScaleDiscriminator 在 __init__ 方法中创建了一个包含三个 discriminator 和三个平均池化层的列表，然后在 forward 方法中按顺序使用这些列表进行前向传播，提取输入数据对应的特征图。

这里的多尺度（MultiScale）体现在多个含义上：第一个，MultiScaleDiscriminator 中的三个 discriminator 在不同位置使用不同的 meanpool，以捕捉输入数据中的不同特征；第二个，MultiScaleDiscriminator 的输出特征图具有多个尺度，意味着在提取特征图时，不同的池化层具有不同的尺寸；第三个，MultiScaleDiscriminator 的输入和输出数据分别属于同一张特征图，即具有相同的尺寸，这样不同的 discriminator 可以对接收到的数据进行不同的处理。


```
class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


```py

这段代码定义了两个函数，一个是 `feature_loss()`，另一个是 `discriminator_loss()`。这两个函数都接受两个全连接层的输出，分别是 `fmap_r` 和 `fmap_g`。

`feature_loss()` 的作用是计算输入特征的损失。它通过遍历输入的特征图 `fmap_r` 和 `fmap_g`，并且每个输入的特征都通过一个二元组 `(dr, dg)` 进行遍历。对于每个二元组，它计算输入特征的绝对值与输入特征的差值的平均值，然后将这个平均值乘以 2，最后将得到的结果相加，得到损失值。

`discriminator_loss()` 的作用是计算判别器的损失。它接受两个全连接层的输出 `disc_real_outputs` 和 `disc_generated_outputs`，然后分别计算输入真实数据与生成模型的差值的平均值，并将这两个平均值相加，得到 loss。此外，它还计算了每个真实数据的平均值，并将它们加入到 `r_losses` 和 `g_losses` 列表中，最后返回损失、平均真实数据和平均生成数据。


```
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


```py

这段代码定义了一个名为 `generator_loss` 的函数，它接受一个名为 `disc_outputs` 的输入参数。函数内部定义了一个变量 `loss`，一个变量 `gen_losses`，以及一个循环变量 `for`。

在循环变量 `for` 的作用下，函数遍历一个叫做 `disc_outputs` 的列表。对于列表中的每个元素 `dg`，函数计算一个损失值 `l`，这里损失值是目标函数，根据输入的采样数据预测模型输出与真实输出之差的平方。然后，函数将 `l` 添加到 `gen_losses` 列表中，并将 `l` 的和添加到 `loss` 中。

最后，函数返回 `loss` 和 `gen_losses`。这段代码的主要目的是计算一个图像的损失函数，以评估模型对图像的生成结果。


```
def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

```py

# `vdecoder/nsf_hifigan/nvSTFT.py`

It seems like this function is a general-purpose audio loader that takes in an audio file path and returns a tuple of the audio data (assuming it's in a format that can be played on the targeted device) and the sampling rate of the audio.

It first checks if the audio file is in the correct format (e.g., if it's a 2D array, it converts it to a 1D array and then checks if it's all NaNs or if it's a mix of NaNs and infs). If it's all NaNs, it returns an empty tuple. If it's a mix of NaNs and infs, it returns an empty tuple. If the audio file is in the format of a float32 array, it normalizes the amplitude to have a maximum value of 2**31+1 (e.g., the maximum volume level).

If the audio data is in the format of a floating-point audio data (e.g., an audio file that's been recorded on a computer), it first converts it to a librosa audio object. It then resamples the audio data to the specified target sampling rate (e.g., 48000) if it's not already at the target sampling rate. It also supports the return of an empty array if the audio file can't be loaded or if the target sampling rate is None.

It also contains a try-except block that catches any exceptions that occur when loading the audio file, and returns an empty array if the audio file can't be loaded.


```
import os

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

os.environ["LRU_CACHE_CAPACITY"] = "3"

def load_wav_to_torch(full_path, target_sr=None, return_empty_on_exception=False):
    sampling_rate = None
    try:
        data, sampling_rate = sf.read(full_path, always_2d=True)# than soundfile.
    except Exception as ex:
        print(f"'{full_path}' failed to load.\nException:")
        print(ex)
        if return_empty_on_exception:
            return [], sampling_rate or target_sr or 48000
        else:
            raise Exception(ex)
    
    if len(data.shape) > 1:
        data = data[:, 0]
        assert len(data) > 2# check duration of audio file is > 2 samples (because otherwise the slice operation was on the wrong dimension)
    
    if np.issubdtype(data.dtype, np.integer): # if audio data is type int
        max_mag = -np.iinfo(data.dtype).min # maximum magnitude = min possible value of intXX
    else: # if audio data is type fp32
        max_mag = max(np.amax(data), -np.amin(data))
        max_mag = (2**31)+1 if max_mag > (2**15) else ((2**15)+1 if max_mag > 1.01 else 1.0) # data should be either 16-bit INT, 32-bit INT or [-1 to 1] float32
    
    data = torch.FloatTensor(data.astype(np.float32))/max_mag
    
    if (torch.isinf(data) | torch.isnan(data)).any() and return_empty_on_exception:# resample will crash with inf/NaN inputs. return_empty_on_exception will return empty arr instead of except
        return [], sampling_rate or target_sr or 48000
    if target_sr is not None and sampling_rate != target_sr:
        data = torch.from_numpy(librosa.core.resample(data.numpy(), orig_sr=sampling_rate, target_sr=target_sr))
        sampling_rate = target_sr
    
    return data, sampling_rate

```py

This is a PyTorch implementation of a tool called `stft_to_mel_spec` which takes a audio signal and a specified window size and hop length, and returns a Mel-Frequency Image (MFIR).

The `stft_to_mel_spec` function takes two arguments: `audio` is a PyTorch tensor containing the input audio, and `window_size` is an integer representing the number of samples to keep after each short-time Fourier Transform (STFT) output.

The function returns a PyTorch tensor `spect` containing the Mel-Frequency Image (MFIR). The MFIR is computed by multiplying the input audio with the mel basis and then passing through a short-time Fourier Transform (STFT).

The `mel_basis` array is an integer array that maps the mel scale frequency to the corresponding frequency for a given number of shifts. This array is provided by the `stft_to_mel_spec` function and can be used to compute the Mel-Frequency Image.

The `get_mel` function maps the input audio to the Mel scale and returns a tensor of the same size as `input_audio`.

The `dynamic_range_compression_torch` function is used to normalize the dynamic range of the Mel-Frequency Image (MFIR) by subtracting the range from 0 and dividing by the range.

Note that the `stft_to_mel_spec` function assumes that the input audio is a mono-channel audio. If the audio is a multichannel audio, you can use the `stft_to_mel_spec` function with the `.squeeze()` method to extract the audio from each channel and then use the `multiply()` method to multiply the audio with the mel basis.


```
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

class STFT():
    def __init__(self, sr=22050, n_mels=80, n_fft=1024, win_size=1024, hop_length=256, fmin=20, fmax=11025, clip_val=1e-5):
        self.target_sr = sr
        
        self.n_mels     = n_mels
        self.n_fft      = n_fft
        self.win_size   = win_size
        self.hop_length = hop_length
        self.fmin     = fmin
        self.fmax     = fmax
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = {}
    
    def get_mel(self, y, keyshift=0, speed=1, center=False):
        sampling_rate = self.target_sr
        n_mels     = self.n_mels
        n_fft      = self.n_fft
        win_size   = self.win_size
        hop_length = self.hop_length
        fmin       = self.fmin
        fmax       = self.fmax
        clip_val   = self.clip_val
        
        factor = 2 ** (keyshift / 12)       
        n_fft_new = int(np.round(n_fft * factor))
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))
        
        if torch.min(y) < -1.:
            print('min value is ', torch.min(y))
        if torch.max(y) > 1.:
            print('max value is ', torch.max(y))
        
        mel_basis_key = str(fmax)+'_'+str(y.device)
        if mel_basis_key not in self.mel_basis:
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
            self.mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)
        
        keyshift_key = str(keyshift)+'_'+str(y.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_size_new).to(y.device)
        
        pad_left = (win_size_new - hop_length_new) //2
        pad_right = max((win_size_new- hop_length_new + 1) //2, win_size_new - y.size(-1) - pad_left)
        if pad_right < y.size(-1):
            mode = 'reflect'
        else:
            mode = 'constant'
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode = mode)
        y = y.squeeze(1)
        
        spec = torch.stft(y, n_fft_new, hop_length=hop_length_new, win_length=win_size_new, window=self.hann_window[keyshift_key],
                          center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        # print(111,spec)
        spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
        if keyshift != 0:
            size = n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size-resize))
            spec = spec[:, :size, :] * win_size / win_size_new
            
        # print(222,spec)
        spec = torch.matmul(self.mel_basis[mel_basis_key], spec)
        # print(333,spec)
        spec = dynamic_range_compression_torch(spec, clip_val=clip_val)
        # print(444,spec)
        return spec
    
    def __call__(self, audiopath):
        audio, sr = load_wav_to_torch(audiopath, target_sr=self.target_sr)
        spect = self.get_mel(audio.unsqueeze(0)).squeeze(0)
        return spect

```py

这是一个使用 STFT() 函数来创建一个 Time-series-to-time (TST) 数据结构的代码。

STFT() 函数是 Time-series 分析工具包 (Python) 中用于将 Time-series 数据转换为 Time-series-to-time 数据结构的一组函数之一。它接收一个 Time-series 对象，返回一个 Time-series-to-time 数据结构，该数据结构包含 Time-series 中每一对时刻之间的间隔。

因此，运行上述代码将返回一个 Time-series-to-time 数据结构，其中每一对时刻之间的间隔被保存为一个 Time-series 的间隔。


```
stft = STFT()

```