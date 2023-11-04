# SO-VITS-SVC源码解析 8

# `modules/losses.py`



这段代码定义了两个函数，一个是 `feature_loss`，另一个是 `discriminator_loss`。这两个函数是训练中的两个损失函数，用于评估图像的特征在不同数据上的效果。

`feature_loss` 函数的目的是计算图像特征(例如特征图)的损失。它接受两个特征图 `fmap_r` 和 `fmap_g`，并计算它们之间的差异。为了计算差异，它使用了一个差分公式(`dr` 和 `dg` 的差)，然后对每个差值应用一个均方误差(MSE)计算损失。最后，它将所有损失相加并乘以 2，得到一个总的损失。

`discriminator_loss` 函数的目的是计算生成器模型(即 `disc_generated_outputs`)在训练数据上的损失，同时计算真实数据上的损失。它接受两个数据集 `disc_real_outputs` 和 `disc_generated_outputs`，并计算生成器模型的输出与真实数据之间的差异。为了计算差异，它使用了一个平方误差(MSE)计算损失。对于每个生成器输出 `dg`，它计算一个损失，其中包括一个平方误差项，这个误差项根据 `dg` 的平方来定义。对于每个真实样本 `dr`，它计算一个损失，其中包括一个平方误差项和一个 `1-dr` 的平方项。最后，它将所有损失相加并返回。


```py
import torch


def feature_loss(fmap_r, fmap_g):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2 


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    dr = dr.float()
    dg = dg.float()
    r_loss = torch.mean((1-dr)**2)
    g_loss = torch.mean(dg**2)
    loss += (r_loss + g_loss)
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


```



这段代码定义了两个函数：`generator_loss` 和 `kl_loss`。它们的作用分别如下：

1. `generator_loss` 函数：

这个函数的输入是 `disc_outputs`，也就是Disc区的人才输出。函数内部维护一个名为 `gen_losses` 的列表，用于存储每个样本到生成器模型的损失。

函数的主要步骤如下：

1. 初始化 `loss` 为 0。

2. 遍历 `disc_outputs` 中的每个元素，将其转换为 float 类型并赋值给 `dg`。

3. 计算每个样本的损失 `l`，使用均方误差（MSE）计算，即 `l = torch.mean((1 - dg) ** 2)`。

4. 将每个样本的损失 `l` 添加到 `gen_losses` 列表中，并将总损失 `loss` 累加到全局变量中。

5. 返回全局变量 `loss` 和 `gen_losses`。

1. `kl_loss` 函数：

这个函数的输入包括前一个函数 `generator_loss` 中计算得到的 `gen_losses` 列表，以及一个名为 `z_mask` 的掩码。函数的主要步骤如下：

1. 将 `z_mask` 和 `gen_losses` 中的每个元素取反，得到一个差分图。

2. 使用 KL 散度公式计算 `z_p` 和 `z_q`，其中 `z_p` 和 `z_q` 是 `disc_outputs` 中的样本。这里我们假设 `disc_outputs` 中的每个元素都是从同一个分布中采样得到的，因此 `z_p` 和 `z_q` 具有相同的分布。

3. 将步骤 2 中计算得到的 `z_p` 和 `z_q` 带入到步骤 1 中计算得到的 `l` 中，得到每个样本的损失 `l`。

4. 将每个样本的损失 `l` 添加到 `kl_losses` 列表中，并将总损失 `loss` 累加到全局变量中。

5. 返回全局变量 `kl_loss`。


```py
def generator_loss(disc_outputs):
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = torch.mean((1-dg)**2)
    gen_losses.append(l)
    loss += l

  return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()
  #print(logs_p)
  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l

```

# `modules/mel_processing.py`

这段代码使用了PyTorch库来实现动态范围压缩。其作用是将输入信号x转换为浮点数，并执行以下操作：将x乘以压缩因子C，同时限制x的值在clip_val范围内。这个函数的输入参数包括x，压缩因子C和clip_val，它们都是浮点数。

这个函数的作用是将输入信号x的值缩放到clip_val范围内，并且使用压缩因子C来控制缩放的程度。这样可以使得输入信号x的值更加紧凑，从而方便进行后续的处理和分析。


```py
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


```



这段代码定义了两个函数 `dynamic_range_decompression_torch` 和 `spectral_normalize_torch`，以及一个名为 `spectral_de_normalize_torch` 的函数。它们都是基于 PyTorch 的 `torch` 数据类型。

`dynamic_range_decompression_torch` 的作用是将输入的 `x` 经过压缩后得到一个压缩后的结果，压缩因子由参数 `C` 确定。压缩后的结果被除以 `C` 得到一个 normalized 的结果。

`spectral_normalize_torch` 的作用是将输入的 `magnitudes` 经过压缩后得到一个 normalized 的结果。它调用了 `dynamic_range_compression_torch` 函数。

`spectral_de_normalize_torch` 的作用是将输入的 `magnitudes` 恢复成原始值。它调用了 `dynamic_range_decompression_torch` 函数，并对结果进行取反。

这里需要注意的是，`dynamic_range_compression_torch` 和 `spectral_normalize_torch` 函数中的 `C` 参数都是固定值，即都是 1。而 `dynamic_range_decompression_torch` 和 `spectral_de_normalize_torch` 函数中的 `C` 参数是可变的。


```py
def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


```

这段代码的主要作用是实现了一个名为“mel_basis”和“hann_window”的PyTorch字典，用于存储频谱数据。

首先， mel_basis 字典存储了一个空字典，用于存储 Mel 频谱数据。hann_window 字典也是一个空字典，用于存储汉明窗数据。

然后， 函数 spectrogram_torch() 实现了一个将 Mel 频谱数据转换为汉明窗数据的方法。该函数接收一个音频信号（y），其采样率为采样率，步长为步长大小，窗口大小为窗口大小，是否以中心为参考值也是一个布尔选项。函数首先检查输入信号的值是否在 -1 到正无穷之间，如果是，则输出一个错误信息。然后，将输入信号的 n-fft 波段进行逆变换，将其转换为 Mel 频谱数据，并将其存储在 hann_window 字典中。

最后， mel_basis 和 hann_window 字典用于存储 mel 和汉明窗数据，以便在后续的处理中使用。


```py
mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    
    y_dtype = y.dtype
    if y.dtype == torch.bfloat16:
        y = y.to(torch.float32)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec).to(y_dtype)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


```

这段代码定义了两个函数，分别将音频信号`spec`转换为Mel级数表示，并返回其 Mel 级数表示。

第一个函数 `spec_to_mel_torch` 接收一个音频信号 `spec`，以及采样率 `sampling_rate`、每秒采样数 `n_fft`、Mel 数量 `num_mels` 和 Fmin/Fmax 值。函数首先检查 Mel 基函数是否包含 `fmax` 类型的值，如果没有，则调用 librosa 的 `mel_fn` 函数将秒数转换为 Mel，然后将结果转换为浮点数类型并存储到 `mel_basis` 字典中。接着，使用 `mel_basis` 字典中的值将 `spec` 中的每个采样数与 Mel 基函数相乘，得到一个新的 `spec` 对象，然后使用 `spectral_normalize_torch` 函数对其进行谱聚对，最后返回新的 `spec` 对象。

第二个函数 `mel_spectrogram_torch` 接收一个音频信号 `y`、采样率 `sampling_rate`、每秒采样数 `n_fft`、Mel 数量 `num_mels`、步长 `hop_size`、窗口大小 `win_size` 和 Fmin/Fmax 值。函数首先将 `y` 的每个采样数与 Mel 基函数相乘，得到一个新的 `spec` 对象，然后使用 `spec_to_mel_torch` 函数将 `spec` 对象转换为 Mel 级数表示，最后使用 Mel 基函数将每个采样数与 Mel 基函数相乘，得到一个新的 `spec` 对象。函数可以使用中心化 `center` 参数来对输入信号进行中心化处理。


```py
def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    spec = spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center)
    spec = spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)
    
    return spec

```

# `modules/modules.py`

这段代码定义了一个自定义的PyTorch模型，它的主要目的是实现一个具有残差连接的密集块（Depthwise Separable Conv-1D）。这个模型是由一些自定义的PyTorch函数和自定义的Attention模块组成的。具体来说：

1. 首先，引入了两个PyTorch中的nn.Module类，分别是Depthwise Separable Conv-1D类和Attention模块。

2. 在Attention模块中，引入了两个从modules.attentions继承下来的函数：get_parameters和forward。其中，get_parameters函数用于获取Attention模块中的参数，forward函数用于计算Attention模块的前向传播计算。

3. 在Attention模块中，还通过import modules.attentions as attentions，引入了其它的相关函数和类。

4. 在 Depthwise Separable Conv-1D类中，通过import modules.commons as commons，引入了其它的相关函数和类。具体来说，这里使用了name进行参数标识，这样就可以在依赖链中追踪到其它的类中。同时，通过使用DepthwiseSeparableConv1D类，实现了残差连接。

5. 通过在 Depthwise Separable Conv-1D类的构造函数中，使用了weight_norm_modules和remove_weight_norm_modules，实现了权重的引入和裁剪。

6. 最后，通过get_padding函数和init_weights函数，设置了一些额外的参数，比如输入通道的数量和批归一化的范数等。

综上所述，这段代码实现了一个具有残差连接的密集块（Depthwise Separable Conv-1D）。这个模型可以用于实现图像分类等任务中的图像分割。


```py
import torch
from torch import nn
from torch.nn import functional as F

import modules.attentions as attentions
import modules.commons as commons
from modules.commons import get_padding, init_weights
from modules.DSConv import (
    Depthwise_Separable_Conv1D,
    remove_weight_norm_modules,
    weight_norm_modules,
)

LRELU_SLOPE = 0.1

```

这段代码定义了一个名为 "LayerNorm" 的类，继承自 PyTorch 中的 nn.Module 类。这个类的目的是在创建一个具有特定通道数的卷积层时，提供一种比直接创建卷积层时更方便的方式来进行参数设置。

具体来说，这段代码定义了一个名为 "set\_Conv1dModel" 的函数，用于设置一个卷积层的模型（即，继承自 Depthwise\_Separable\_Conv1D 的卷积层）。如果设置了 use\_depthwise\_conv 参数为 True，则将直接创建一个大小为 channels 的卷积层；否则，将创建一个大小为 channels 的卷积层，其中一个 depthwise 卷积层和一个 separable 卷积层。

接着，定义了一个名为 "LayerNorm" 的类，该类包含一个 forward 方法。该方法的主要作用是在创建一个具有特定通道数的卷积层时，提供一种比直接创建卷积层时更方便的方式来进行参数设置。具体来说，该方法接收一个通道数为 channels 的输入 x，对其进行一些转换操作，然后返回经过这些操作后的 x。这些操作包括：将输入 x 的通道数设置为 channels，对输入 x 应用一个 gamma 参数，该参数大小为 channels，对输入 x 应用一个 beta 参数，该参数大小为 channels，对输入 x 应用 layer\_norm 函数，该函数使用 gamma 和 beta 参数对输入 x 进行归一化操作，同时设置 eps 参数为 1e-5，以确保输入 x 不会出现数值约简。


```py
Conv1dModel = nn.Conv1d

def set_Conv1dModel(use_depthwise_conv):
    global Conv1dModel
    Conv1dModel = Depthwise_Separable_Conv1D if use_depthwise_conv else nn.Conv1d


class LayerNorm(nn.Module):
  def __init__(self, channels, eps=1e-5):
    super().__init__()
    self.channels = channels
    self.eps = eps

    self.gamma = nn.Parameter(torch.ones(channels))
    self.beta = nn.Parameter(torch.zeros(channels))

  def forward(self, x):
    x = x.transpose(1, -1)
    x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
    return x.transpose(1, -1)

 
```

以下是输出源代码的 class 定义：

```pypython
class ConvReluNorm(nn.Module):
   def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
       super().__init__()
       self.in_channels = in_channels
       self.hidden_channels = hidden_channels
       self.out_channels = out_channels
       self.kernel_size = kernel_size
       self.n_layers = n_layers
       self.p_dropout = p_dropout

       self.conv_layers = nn.ModuleList()
       self.norm_layers = nn.ModuleList()
       self.conv_layers.append(Conv1dModel(in_channels, hidden_channels, kernel_size, padding=kernel_size//2))
       self.norm_layers.append(LayerNorm(hidden_channels))
       self.relu_drop = nn.Sequential(
           nn.ReLU(),
           nn.Dropout(p_dropout))
       for _ in range(n_layers-1):
           self.conv_layers.append(Conv1dModel(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2))
           self.norm_layers.append(LayerNorm(hidden_channels))
       self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
       self.proj.weight.data.zero_()
       self.proj.bias.data.zero_())

   def forward(self, x, x_mask):
       x_org = x
       for i in range(self.n_layers):
           x = self.conv_layers[i](x * x_mask)
           x = self.norm_layers[i](x)
           x = self.relu_drop(x)
       x = x_org + self.proj(x)
       return x * x_mask
```

这个类是实现了一个名为“ConvReluNorm”的 `nn.Module` 类，包含了前向传递的卷积层、池化层、归一化和激活函数，用于实现图像分类任务。在初始化函数中，指定了输入通道数、隐藏通道数、输出通道数、卷积核大小、层数和抽样概率。在 forward 函数中，按顺序执行了卷积层、池化层、归一化和前向传递，最终将结果返回。


```py
class ConvReluNorm(nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
    super().__init__()
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.p_dropout = p_dropout
    assert n_layers > 1, "Number of layers should be larger than 0."

    self.conv_layers = nn.ModuleList()
    self.norm_layers = nn.ModuleList()
    self.conv_layers.append(Conv1dModel(in_channels, hidden_channels, kernel_size, padding=kernel_size//2))
    self.norm_layers.append(LayerNorm(hidden_channels))
    self.relu_drop = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(p_dropout))
    for _ in range(n_layers-1):
      self.conv_layers.append(Conv1dModel(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2))
      self.norm_layers.append(LayerNorm(hidden_channels))
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
    self.proj.weight.data.zero_()
    self.proj.bias.data.zero_()

  def forward(self, x, x_mask):
    x_org = x
    for i in range(self.n_layers):
      x = self.conv_layers[i](x * x_mask)
      x = self.norm_layers[i](x)
      x = self.relu_drop(x)
    x = x_org + self.proj(x)
    return x * x_mask


```

This is a PyTorch implementation of a multi-layer self-attention model. The model has a classification layer with a hyperbolic tangent (tanh) activation function, followed by a residual connection (ResNet) that skips the input attention. The hyperparameters are `hidden_channels`, `res_skip_channels`, and `res_skip_layer_strategy`.

The model has a forward pass that takes an input tensor `x`, along with an attention mask `x_mask`, and optionally a `weight_norm_modules` function for each residual connection block. The `forward pass` calculates the output of each residual connection block and adds them up.

The model also has a `remove_weight_norm` method that removes the weight normalization for each residual connection block.

I hope this helps! Let me know if you have any questions.


```py
class WN(torch.nn.Module):
  def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
    super(WN, self).__init__()
    assert(kernel_size % 2 == 1)
    self.hidden_channels =hidden_channels
    self.kernel_size = kernel_size,
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.p_dropout = p_dropout

    self.in_layers = torch.nn.ModuleList()
    self.res_skip_layers = torch.nn.ModuleList()
    self.drop = nn.Dropout(p_dropout)

    if gin_channels != 0:
      cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
      self.cond_layer = weight_norm_modules(cond_layer, name='weight')

    for i in range(n_layers):
      dilation = dilation_rate ** i
      padding = int((kernel_size * dilation - dilation) / 2)
      in_layer = Conv1dModel(hidden_channels, 2*hidden_channels, kernel_size,
                                 dilation=dilation, padding=padding)
      in_layer = weight_norm_modules(in_layer, name='weight')
      self.in_layers.append(in_layer)

      # last one is not necessary
      if i < n_layers - 1:
        res_skip_channels = 2 * hidden_channels
      else:
        res_skip_channels = hidden_channels

      res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
      res_skip_layer = weight_norm_modules(res_skip_layer, name='weight')
      self.res_skip_layers.append(res_skip_layer)

  def forward(self, x, x_mask, g=None, **kwargs):
    output = torch.zeros_like(x)
    n_channels_tensor = torch.IntTensor([self.hidden_channels])

    if g is not None:
      g = self.cond_layer(g)

    for i in range(self.n_layers):
      x_in = self.in_layers[i](x)
      if g is not None:
        cond_offset = i * 2 * self.hidden_channels
        g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
      else:
        g_l = torch.zeros_like(x_in)

      acts = commons.fused_add_tanh_sigmoid_multiply(
          x_in,
          g_l,
          n_channels_tensor)
      acts = self.drop(acts)

      res_skip_acts = self.res_skip_layers[i](acts)
      if i < self.n_layers - 1:
        res_acts = res_skip_acts[:,:self.hidden_channels,:]
        x = (x + res_acts) * x_mask
        output = output + res_skip_acts[:,self.hidden_channels:,:]
      else:
        output = output + res_skip_acts
    return output * x_mask

  def remove_weight_norm(self):
    if self.gin_channels != 0:
      remove_weight_norm_modules(self.cond_layer)
    for l in self.in_layers:
      remove_weight_norm_modules(l)
    for l in self.res_skip_layers:
      remove_weight_norm_modules(l)


```

In summary, this is a module that inherits from the `WeightNormModule` class and appears to be used for the convolutional neural networks (CNNs) in a given model. The `CNN` class inherits from the `Module` class and contains several `weight_norm_modules` that apply the normalization operations for the convolutional layers.

The `CNN` class has two convolutional layers with different filter sizes and dilation values. The `forward` method applies the normalization operations to the input features and returns the output. The `remove_weight_norm` method removes the normalization for the convolutional layers.


```py
class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm_modules(l)
        for l in self.convs2:
            remove_weight_norm_modules(l)


```



这段代码定义了一个名为ResBlock2的类，继承自PyTorch中的nn.Module类。这个类在网络中被用于实现残差块(Residual Block)的功能。

在__init__方法中， channels参数表示输入通道的数量，kernel_size参数表示卷积核的大小，dilation参数表示 dilation 的参数。其中，dilation参数是指在卷积过程中，对输入通道进行dilation操作的大小，可以通过设置dilation=(1, 3)来扩大卷积的输出尺寸，以适应残差块的需求。

在forward方法中，首先定义了一个Conv1dModel类，用于在网络中产生x的残差块。在Conv1dModel类中，使用了weight_norm_modules方法对产生的x进行归一化处理，并应用到self.convs列表中的每个卷积层上，以实现残差块的功能。

最后，通过应用卷积层、dilation操作和归一化操作，以及输出x的结果，实现了ResBlock2的作用。


```py
class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm_modules(l)


```

这是一个基于PyTorch的机器学习模型的类，名为“Log”和“Flip”。这两个类都是继承自nn.Module类。

Log类包含一个前向传播函数forward，以及一个反向传播函数backward。在前向传播函数中，如果参数reverse为False，则计算输入x的梯度并反向传播；如果reverse为True，则计算输入x的指数并反向传播。计算出的梯度和反向传播的梯度将返回给输入x。

Flip类包含一个前向传播函数forward，以及一个反向传播函数backward。在前向传播函数中，如果参数reverse为False，则对输入x进行翻转操作；如果reverse为True，则不做任何操作。计算出的梯度和反向传播的梯度将返回给输入x。

总的来说，这两个类都是用于对输入数据进行前向传播和反向传播的处理，以便在训练神经网络时计算梯度和反向传播的梯度。


```py
class Log(nn.Module):
  def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
      logdet = torch.sum(-y, [1, 2])
      return y, logdet
    else:
      x = torch.exp(x) * x_mask
      return x
    

class Flip(nn.Module):
  def forward(self, x, *args, reverse=False, **kwargs):
    x = torch.flip(x, [1])
    if not reverse:
      logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
      return x, logdet
    else:
      return x


```

这段代码定义了一个名为 ElementwiseAffine 的类，继承自 PyTorch 中的 nn.Module 类。这个类在图像处理、计算机视觉等领域中广泛使用，主要用于对输入数据进行非线性变换和处理。

具体来说，这段代码的作用是定义了一个类 ElementwiseAffine，它包含一个构造函数 __init__，一个前向函数 forward，以及两个参数 channels 和 logs。

在构造函数 __init__ 中，首先调用父类的构造函数，然后创建一个 channels 维的参数张量，将其赋值为零。接着，创建两个 channels 维的参数张量，并将它们都赋值为零。最后，创建两个 channels 维的参数张量，并将它们都赋值为零。

在 forward 函数中，根据输入 x 和 x_mask 是否包含反转信息（即是否设置了 reverse=True 参数），来决定输出 y 和 logdet 的计算方式。如果没有设置反转信息，则先将 x 和 x_mask 中的值取反，然后加上相应的参数值，最后对结果进行 log 计算。如果设置了反转信息，则先将 x 和 x_mask 中的值取反，然后对结果进行 exp 计算。

在使用时，只需创建 ElementwiseAffine 类的实例，并传入需要处理的输入数据即可。例如：
```py
element = ElementwiseAffine(2)
output = element(x, x_mask)
```
这段代码的作用就是对输入数据 x 和 x_mask 进行非线性变换，并返回结果。


```py
class ElementwiseAffine(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.channels = channels
    self.m = nn.Parameter(torch.zeros(channels,1))
    self.logs = nn.Parameter(torch.zeros(channels,1))

  def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = self.m + torch.exp(self.logs) * x
      y = y * x_mask
      logdet = torch.sum(self.logs * x_mask, [1,2])
      return y, logdet
    else:
      x = (x - self.m) * torch.exp(-self.logs) * x_mask
      return x


```

This is a class definition for a neural network model. It inherits from the `torch.nn.Module` class and contains several parameters:

* `channels`: The number of channels in each layer.
* `hidden_channels`: The number of hidden channels in each layer.
* `kernel_size`: The size of the kernel in each layer.
* `dilation_rate`: The dilation rate used in each layer.
* `n_layers`: The number of layers in the network.
* `half_channels`: The number of channels in each half of a layer.
* `mean_only`: A boolean indicating whether to compute the mean of the hidden channels.
* `pre`: A pre-layer with a given number of channels.
* `enc`: A `ModelWrapper` layer with a given number of hidden channels and a pre-layer.
* `post`: A post-layer with a given number of channels.
* `mean_only`: A boolean indicating whether to compute the mean of the hidden channels.
* ` statistics`: A `Group` object containing statistics of the input.
* `params`: A list of parameters.

The `forward` method defines the forward pass of the network. It takes an input tensor `x` and an optional tensor `g` and applies the given layers to it. The method returns the output tensor.

Note that the `mean_only` parameter determines whether the mean of the hidden channels should be computed. If `mean_only` is `True`, the function will compute the mean of the hidden channels.


```py
class ResidualCouplingLayer(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      p_dropout=0,
      gin_channels=0,
      mean_only=False,
      wn_sharing_parameter=None
      ):
    assert channels % 2 == 0, "channels should be divisible by 2"
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.half_channels = channels // 2
    self.mean_only = mean_only

    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels) if wn_sharing_parameter is None else wn_sharing_parameter
    self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
    self.post.weight.data.zero_()
    self.post.bias.data.zero_()

  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = torch.split(x, [self.half_channels]*2, 1)
    h = self.pre(x0) * x_mask
    h = self.enc(h, x_mask, g=g)
    stats = self.post(h) * x_mask
    if not self.mean_only:
      m, logs = torch.split(stats, [self.half_channels]*2, 1)
    else:
      m = stats
      logs = torch.zeros_like(m)

    if not reverse:
      x1 = m + x1 * torch.exp(logs) * x_mask
      x = torch.cat([x0, x1], 1)
      logdet = torch.sum(logs, [1,2])
      return x, logdet
    else:
      x1 = (x1 - m) * torch.exp(-logs) * x_mask
      x = torch.cat([x0, x1], 1)
      return x

```

This is a class definition for an neural network model for image classification tasks. The model has encoder and decoder blocks, and it uses a multi-layer perceptron (MPP) architecture. The encoder uses a combination of fully connected layers (FFT-based), convolutional neural networks (CNNs), and attention mechanisms. The decoder uses only fully connected layers and is based on the mean-only attention mechanism. The model takes an input image and its associated mask, and it returns the classification logits and the corresponding image. The model has a parameter `self.channels`, `self.hidden_channels`, `self.kernel_size`, `self.n_layers`, and `self.half_channels`, which are used to configure the parameters of the CNNs, the FFT component, and the attention mechanisms, respectively.


```py
class TransformerCouplingLayer(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      n_layers,
      n_heads,
      p_dropout=0,
      filter_channels=0,
      mean_only=False,
      wn_sharing_parameter=None,
      gin_channels = 0
      ):
    assert channels % 2 == 0, "channels should be divisible by 2"
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.half_channels = channels // 2
    self.mean_only = mean_only

    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    self.enc = attentions.FFT(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, isflow = True, gin_channels = gin_channels) if wn_sharing_parameter is None else wn_sharing_parameter
    self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
    self.post.weight.data.zero_()
    self.post.bias.data.zero_()

  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = torch.split(x, [self.half_channels]*2, 1)
    h = self.pre(x0) * x_mask
    h = self.enc(h, x_mask, g=g)
    stats = self.post(h) * x_mask
    if not self.mean_only:
      m, logs = torch.split(stats, [self.half_channels]*2, 1)
    else:
      m = stats
      logs = torch.zeros_like(m)

    if not reverse:
      x1 = m + x1 * torch.exp(logs) * x_mask
      x = torch.cat([x0, x1], 1)
      logdet = torch.sum(logs, [1,2])
      return x, logdet
    else:
      x1 = (x1 - m) * torch.exp(-logs) * x_mask
      x = torch.cat([x0, x1], 1)
      return x

```

# `modules/__init__.py`

很抱歉，我没有看到您提供的代码。请提供代码，以便我为您提供详细的解释。


```py

```

# `modules/F0Predictor/crepe.py`

这段代码定义了一个名为“repeat_expand”的函数，它接受一个输入参数“content”，它可能是张量（如numpy数组）或标量（如整数或浮点数）。函数还有一个名为“target_len”的输入参数，它指定了要重复的序列的长度。函数还有一个名为“mode”的输入参数，它决定了在插值时如何对齐数据。默认情况下，模式为“nearest”，它表示对齐后在序列中的最接近的样本进行插值。

函数实现的核心部分是使用PyTorch中的nn.functional.interpolate函数来执行插值。这个函数接受两个参数：要插值的序列和目标序列的长度。它返回一个新的序列，其中插值点按照指定的模式对齐，然后返回该新序列。如果输入是numpy数组，函数将数组转换为张量，然后对齐并返回张量。如果输入是标量，函数直接返回输入。

如果函数在尝试从标准库中导入Literal类型时出错，它将使用Literal类型来代替。这种类型的实例只包含输入中提供的数据类型和数量，而不会创建新的实例或执行任何操作。


```py
from typing import Optional, Union

try:
    from typing import Literal
except Exception:
    from typing_extensions import Literal
import numpy as np
import torch
import torchcrepe
from torch import nn
from torch.nn import functional as F

#from:https://github.com/fishaudio/fish-diffusion

def repeat_expand(
    content: Union[torch.Tensor, np.ndarray], target_len: int, mode: str = "nearest"
):
    """Repeat content to target length.
    This is a wrapper of torch.nn.functional.interpolate.

    Args:
        content (torch.Tensor): tensor
        target_len (int): target length
        mode (str, optional): interpolation mode. Defaults to "nearest".

    Returns:
        torch.Tensor: tensor
    """

    ndim = content.ndim

    if content.ndim == 1:
        content = content[None, None]
    elif content.ndim == 2:
        content = content[None]

    assert content.ndim == 3

    is_np = isinstance(content, np.ndarray)
    if is_np:
        content = torch.from_numpy(content)

    results = torch.nn.functional.interpolate(content, size=target_len, mode=mode)

    if is_np:
        results = results.numpy()

    if ndim == 1:
        return results[0, 0]
    elif ndim == 2:
        return results[0]


```

This is a function that takes a one-dimensional tensor `f0` and an optional tensor `pad_to`, and returns the padded tensor and the corresponding vector `vuv_vector`.

The `f0` tensor is assumed to be a one-dimensional tensor of float numbers, and should be passed through a neural network model that has a output dimension of 1D.

The `pad_to` tensor is an optional tensor that should be padded to the same length as `f0`.

The function uses a combination of NumPy and PyTorch to handle the computation of the padding vector and the interpolation of the `vuv_vector`.

The `F.interpolate` method from PyTorch is used to perform the interpolation of the `vuv_vector`. The method takes two arguments: the `vuv_vector`, which is a tensor of shape `(batch_size, sequence_length, n_channels)`, and the desired interpolation parameters, which are specified as a tuple `(hint_size, axis, order)`.

The `hint_size` is the desired length of the output插值点， `axis` is the dimension of the output插值点与 `vuv_vector` 的维度，`order` 是插值算法的阶数。

The function assumes that the input tensor `f0` has a non-zero value only in the frequency domain, and that the `pad_to` tensor should have the same non-zero values as `f0`.


```py
class BasePitchExtractor:
    def __init__(
        self,
        hop_length: int = 512,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        keep_zeros: bool = True,
    ):
        """Base pitch extractor.

        Args:
            hop_length (int, optional): Hop length. Defaults to 512.
            f0_min (float, optional): Minimum f0. Defaults to 50.0.
            f0_max (float, optional): Maximum f0. Defaults to 1100.0.
            keep_zeros (bool, optional): Whether keep zeros in pitch. Defaults to True.
        """

        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.keep_zeros = keep_zeros

    def __call__(self, x, sampling_rate=44100, pad_to=None):
        raise NotImplementedError("BasePitchExtractor is not callable.")

    def post_process(self, x, sampling_rate, f0, pad_to):
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float().to(x.device)

        if pad_to is None:
            return f0

        f0 = repeat_expand(f0, pad_to)

        if self.keep_zeros:
            return f0
        
        vuv_vector = torch.zeros_like(f0)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0
        
        # 去掉0频率, 并线性插值
        nzindex = torch.nonzero(f0).squeeze()
        f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()
        time_org = self.hop_length / sampling_rate * nzindex.cpu().numpy()
        time_frame = np.arange(pad_to) * self.hop_length / sampling_rate
        
        vuv_vector = F.interpolate(vuv_vector[None,None,:],size=pad_to)[0][0]

        if f0.shape[0] <= 0:
            return torch.zeros(pad_to, dtype=torch.float, device=x.device),vuv_vector.cpu().numpy()
        if f0.shape[0] == 1:
            return torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[0],vuv_vector.cpu().numpy()
    
        # 大概可以用 torch 重写?
        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
        #vuv_vector = np.ceil(scipy.ndimage.zoom(vuv_vector,pad_to/len(vuv_vector),order = 0))
        
        return f0,vuv_vector.cpu().numpy()


```

This is a PyTorch implementation of a 2D convolutional neural network (CNN) with a custom `FastRCNN` class. The class has three instance variables: `kernel_size`, `stride`, and `padding`.

The `kernel_size` is the size of the kernel, or the number of columns in a 2D filter. The `stride` is the step size of the filter, or the number of channels in each axis. The `padding` is the amount of padding added to the input tensor, which can be used for spatial上奖励（spatial correlation）。

In the `forward` method, if `mask` is `None`, the function will create a random binary mask of the same shape as the input tensor. If `mask` is not `None`, it will convert the input tensor to a binary format, and then apply the mask to the input tensor using element-wise indexing.

The `avg_pooling` is the average pooling operation. It is the sum of the input tensor divided by the number of valid elements in each pooling window. If the number of valid elements is less than one, the average pooling will be a gather value.

Overall, this implementation is useful for implementing a custom 2D CNN with various options, such as the ability to add spatial上奖励。


```py
class MaskedAvgPool1d(nn.Module):
    def __init__(
        self, kernel_size: int, stride: Optional[int] = None, padding: Optional[int] = 0
    ):
        """An implementation of mean pooling that supports masked values.

        Args:
            kernel_size (int): The size of the median pooling window.
            stride (int, optional): The stride of the median pooling window. Defaults to None.
            padding (int, optional): The padding of the median pooling window. Defaults to 0.
        """

        super(MaskedAvgPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x, mask=None):
        ndim = x.dim()
        if ndim == 2:
            x = x.unsqueeze(1)

        assert (
            x.dim() == 3
        ), "Input tensor must have 2 or 3 dimensions (batch_size, channels, width)"

        # Apply the mask by setting masked elements to zero, or make NaNs zero
        if mask is None:
            mask = ~torch.isnan(x)

        # Ensure mask has the same shape as the input tensor
        assert x.shape == mask.shape, "Input tensor and mask must have the same shape"

        masked_x = torch.where(mask, x, torch.zeros_like(x))
        # Create a ones kernel with the same number of channels as the input tensor
        ones_kernel = torch.ones(x.size(1), 1, self.kernel_size, device=x.device)

        # Perform sum pooling
        sum_pooled = nn.functional.conv1d(
            masked_x,
            ones_kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.size(1),
        )

        # Count the non-masked (valid) elements in each pooling window
        valid_count = nn.functional.conv1d(
            mask.float(),
            ones_kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.size(1),
        )
        valid_count = valid_count.clamp(min=1)  # Avoid division by zero

        # Perform masked average pooling
        avg_pooled = sum_pooled / valid_count

        # Fill zero values with NaNs
        avg_pooled[avg_pooled == 0] = float("nan")

        if ndim == 2:
            return avg_pooled.squeeze(1)

        return avg_pooled


```

This code appears to be a implementation of a function that performs a masked inference on a 2D tensor `x`. It is written using the PyTorch library and runs on a PyTorch tensor `x` that has a shape of (batch\_size, batch\_size, embedding\_dim). The function has an input mask `mask` and an output tensor `x_masked` with a shape of (batch\_size, batch\_size, embedding\_dim).

The function performs several operations to apply the mask to the input tensor `x`. First, it adds a value of float("inf") to the input tensor `x` to indicate that the input should be considered valid. Then, it uses the `torch.where()` method to apply the mask to the input tensor `x`. This creates a tensor `x_masked` with a shape of (batch\_size, batch\_size, embedding\_dim) that indicates which elements of the input tensor `x` are non-zero (i.e., are masked).

The function then performs a few additional operations to manipulate the masked tensor `x_masked`. First, it uses the `torch.contiguous()` method to convert the masked tensor `x_masked` to a contiguous tensor that has a shape of (batch\_size \* embedding\_dim, embedding\_dim). Then, it uses the `view()` method to add a new dimension of size 1 to the input tensor `x_masked`, which computes the median value of the masked elements along the last dimension.

Finally, the function returns the masked tensor `x_masked`.


```py
class MaskedMedianPool1d(nn.Module):
    def __init__(
        self, kernel_size: int, stride: Optional[int] = None, padding: Optional[int] = 0
    ):
        """An implementation of median pooling that supports masked values.

        This implementation is inspired by the median pooling implementation in
        https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598

        Args:
            kernel_size (int): The size of the median pooling window.
            stride (int, optional): The stride of the median pooling window. Defaults to None.
            padding (int, optional): The padding of the median pooling window. Defaults to 0.
        """

        super(MaskedMedianPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x, mask=None):
        ndim = x.dim()
        if ndim == 2:
            x = x.unsqueeze(1)

        assert (
            x.dim() == 3
        ), "Input tensor must have 2 or 3 dimensions (batch_size, channels, width)"

        if mask is None:
            mask = ~torch.isnan(x)

        assert x.shape == mask.shape, "Input tensor and mask must have the same shape"

        masked_x = torch.where(mask, x, torch.zeros_like(x))

        x = F.pad(masked_x, (self.padding, self.padding), mode="reflect")
        mask = F.pad(
            mask.float(), (self.padding, self.padding), mode="constant", value=0
        )

        x = x.unfold(2, self.kernel_size, self.stride)
        mask = mask.unfold(2, self.kernel_size, self.stride)

        x = x.contiguous().view(x.size()[:3] + (-1,))
        mask = mask.contiguous().view(mask.size()[:3] + (-1,)).to(x.device)

        # Combine the mask with the input tensor
        #x_masked = torch.where(mask.bool(), x, torch.fill_(torch.zeros_like(x),float("inf")))
        x_masked = torch.where(mask.bool(), x, torch.FloatTensor([float("inf")]).to(x.device))

        # Sort the masked tensor along the last dimension
        x_sorted, _ = torch.sort(x_masked, dim=-1)

        # Compute the count of non-masked (valid) values
        valid_count = mask.sum(dim=-1)

        # Calculate the index of the median value for each pooling window
        median_idx = (torch.div((valid_count - 1), 2, rounding_mode='trunc')).clamp(min=0)

        # Gather the median values using the calculated indices
        median_pooled = x_sorted.gather(-1, median_idx.unsqueeze(-1).long()).squeeze(-1)

        # Fill infinite values with NaNs
        median_pooled[torch.isinf(median_pooled)] = float("nan")
        
        if ndim == 2:
            return median_pooled.squeeze(1)

        return median_pooled


```

This is a function that takes a tensor `x` with a shape of (batch\_size, number of samples, hop\_length) and returns the result of a segmentation task. The function has the following arguments:

-   `padding_mode`: Specifies whether to pad the tensor to the maximum length of the hop\_length. Defaults to None.
-   `use_fast_filters`: Whether to use fast filters for computing the mean and median values. Defaults to False.
-   `hop_length`: The number of samples to consider for the calculation of the median filter.
-   `f0_min`: The minimum value of the f0-threshold.
-   `f0_max`: The maximum value of the f0-threshold.
-   `model`: The model to use for computing the f0-threshold values.
-   `batch_size`: The batch size used for the computations.
-   `device`: The device on which the computation is performed.
-   `decoder`: The decoder being used for the computation (not used here, but it should be defined in the warehouse readme).

The function uses the `torchcrepe` package to perform the segmentation task. It first converts the input tensor to a 2D tensor and extracts the first channel. Then it performs various thresholding and filtering operations, including the median filter, to compute the f0-threshold values. If `use_fast_filters` is `True`, the function uses the `mean_filter` and `silence_filter` functions from `torchcrepe` instead of the `mean` and `silence` functions.

The function returns the result of the segmentation task, which is a tensor with the same shape as the input tensor with the f0-threshold values added. If the f0-threshold values have NaN values, the function returns the same tensor.


```py
class CrepePitchExtractor(BasePitchExtractor):
    def __init__(
        self,
        hop_length: int = 512,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        threshold: float = 0.05,
        keep_zeros: bool = False,
        device = None,
        model: Literal["full", "tiny"] = "full",
        use_fast_filters: bool = True,
        decoder="viterbi"
    ):
        super().__init__(hop_length, f0_min, f0_max, keep_zeros)
        if decoder == "viterbi":
            self.decoder = torchcrepe.decode.viterbi
        elif decoder == "argmax":
            self.decoder = torchcrepe.decode.argmax
        elif decoder == "weighted_argmax":
            self.decoder = torchcrepe.decode.weighted_argmax
        else:
            raise "Unknown decoder"
        self.threshold = threshold
        self.model = model
        self.use_fast_filters = use_fast_filters
        self.hop_length = hop_length
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        if self.use_fast_filters:
            self.median_filter = MaskedMedianPool1d(3, 1, 1).to(device)
            self.mean_filter = MaskedAvgPool1d(3, 1, 1).to(device)

    def __call__(self, x, sampling_rate=44100, pad_to=None):
        """Extract pitch using crepe.


        Args:
            x (torch.Tensor): Audio signal, shape (1, T).
            sampling_rate (int, optional): Sampling rate. Defaults to 44100.
            pad_to (int, optional): Pad to length. Defaults to None.

        Returns:
            torch.Tensor: Pitch, shape (T // hop_length,).
        """

        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D tensor."
        assert x.shape[0] == 1, f"Expected 1 channel, got {x.shape[0]} channels."

        x = x.to(self.dev)
        f0, pd = torchcrepe.predict(
            x,
            sampling_rate,
            self.hop_length,
            self.f0_min,
            self.f0_max,
            pad=True,
            model=self.model,
            batch_size=1024,
            device=x.device,
            return_periodicity=True,
            decoder=self.decoder
        )

        # Filter, remove silence, set uv threshold, refer to the original warehouse readme
        if self.use_fast_filters:
            pd = self.median_filter(pd)
        else:
            pd = torchcrepe.filter.median(pd, 3)

        pd = torchcrepe.threshold.Silence(-60.0)(pd, x, sampling_rate, self.hop_length)
        f0 = torchcrepe.threshold.At(self.threshold)(f0, pd)
        
        if self.use_fast_filters:
            f0 = self.mean_filter(f0)
        else:
            f0 = torchcrepe.filter.mean(f0, 3)

        f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)[0]

        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if pad_to is None else np.zeros(pad_to)
            return rtn,rtn
        
        return self.post_process(x, sampling_rate, f0, pad_to)

```

# `modules/F0Predictor/CrepeF0Predictor.py`

This is a class called `CrepeF0Predictor` that inherits from the `F0Predictor` class. It is designed to predict the top 512 high-frequency notes in a given audio signal using the `CrepePitchExtractor` class.

The `__init__` method initializes the audio processor with a specified hop length, minimum and maximum F0 frequency, device to use (such as a GPU if specified), sampling rate, and threshold value, as well as defining the name of the model.

The `compute_f0` method takes a pre-processed audio waveform and a pad length as input and computes the F0 frequency and its associated University of Western O听证到的时间信息。

The `compute_f0_uv` method is a wrapper around the `compute_f0` method, it takes the same input as the previous one and returns the F0 frequency and University of Western O听证到的时间信息。


```py
import torch

from modules.F0Predictor.crepe import CrepePitchExtractor
from modules.F0Predictor.F0Predictor import F0Predictor


class CrepeF0Predictor(F0Predictor):
    def __init__(self,hop_length=512,f0_min=50,f0_max=1100,device=None,sampling_rate=44100,threshold=0.05,model="full"):
        self.F0Creper = CrepePitchExtractor(hop_length=hop_length,f0_min=f0_min,f0_max=f0_max,device=device,threshold=threshold,model=model)
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.device = device
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.name = "crepe"

    def compute_f0(self,wav,p_len=None):
        x = torch.FloatTensor(wav).to(self.device)
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        f0,uv = self.F0Creper(x[None,:].float(),self.sampling_rate,pad_to=p_len)
        return f0
    
    def compute_f0_uv(self,wav,p_len=None):
        x = torch.FloatTensor(wav).to(self.device)
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        f0,uv = self.F0Creper(x[None,:].float(),self.sampling_rate,pad_to=p_len)
        return f0,uv
```

# `modules/F0Predictor/DioF0Predictor.py`



This code appears to be a function that computes the F0-瑕值， which is a measure of the quality of a speech signal. It has been implemented using the PyAudio library to perform audio processing tasks.

The function takes in two arguments: a waveform (striated) and a processing parameters array. The waveform is expected to be a one-dimensional array of audio samples, and should have a sampling rate of at least 22050 Hz. The processing parameters array is a two-dimensional array that can have any shape (e.g., a 2x2 array).

The function returns the computed F0-瑕值 as a numpy array.

The function uses a variety of algorithms to compute the F0-瑕值， including a simple low-pass filter, a 13-point菲涅尔 filter, and a combination of these filters. The F0-瑕值的计算过程 is based on the定義 F0_瑕value = 20*log2(1+sqrt(F0^2-4*sqrt(F0)*2))$, 

The low pass filter is implemented using the '扎哈罗夫filter'，菲涅尔滤波器 Implementation 'Bessel菲涅尔滤波器'，

而13点菲涅尔滤波器则对应于 librosa 库中的 filer 函数。


```py
import numpy as np
import pyworld

from modules.F0Predictor.F0Predictor import F0Predictor


class DioF0Predictor(F0Predictor):
    def __init__(self,hop_length=512,f0_min=50,f0_max=1100,sampling_rate=44100):
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sampling_rate = sampling_rate
        self.name = "dio"

    def interpolate_f0(self,f0):
        '''
        对F0进行插值处理
        '''
        vuv_vector = np.zeros_like(f0, dtype=np.float32)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0
    
        nzindex = np.nonzero(f0)[0]
        data = f0[nzindex]
        nzindex = nzindex.astype(np.float32)
        time_org = self.hop_length / self.sampling_rate * nzindex
        time_frame = np.arange(f0.shape[0]) * self.hop_length / self.sampling_rate

        if data.shape[0] <= 0:
            return np.zeros(f0.shape[0], dtype=np.float32),vuv_vector

        if data.shape[0] == 1:
            return np.ones(f0.shape[0], dtype=np.float32) * f0[0],vuv_vector

        f0 = np.interp(time_frame, time_org, data, left=data[0], right=data[-1])
        
        return f0,vuv_vector

    def resize_f0(self,x, target_len):
        source = np.array(x)
        source[source<0.001] = np.nan
        target = np.interp(np.arange(0, len(source)*target_len, len(source))/ target_len, np.arange(0, len(source)), source)
        res = np.nan_to_num(target)
        return res
        
    def compute_f0(self,wav,p_len=None):
        if p_len is None:
            p_len = wav.shape[0]//self.hop_length
        f0, t = pyworld.dio(
            wav.astype(np.double),
            fs=self.sampling_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        return self.interpolate_f0(self.resize_f0(f0, p_len))[0]

    def compute_f0_uv(self,wav,p_len=None):
        if p_len is None:
            p_len = wav.shape[0]//self.hop_length
        f0, t = pyworld.dio(
            wav.astype(np.double),
            fs=self.sampling_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        return self.interpolate_f0(self.resize_f0(f0, p_len))

```

# `modules/F0Predictor/F0Predictor.py`



这段代码定义了一个名为 `F0Predictor` 的类，用于计算频步预测值(f0)和频谱能量(uv)。

具体来说，`compute_f0` 方法接受一个信号声级(wav)和一个脉冲长度(p_len)，并输出一个复数类型的频步预测值(f0)。它是通过将 wav 信号中的所有能量值乘以一个衰减因子，然后取反得到 f0 信号。这个衰减因子可以根据需要进行调整以获取更准确的预测结果。

`compute_f0_uv` 方法与 `compute_f0` 类似，但同时也计算了频谱能量(uv)。它是通过将 wav 信号中的所有能量值乘以一个衰减因子，然后取反得到 f0 和 uv 信号。这个衰减因子也可以根据需要进行调整以获取更准确的预测结果。

频步预测值(f0)表示每个时间步中预测的频率分量的幅度，而频谱能量(uv)则表示每个时间步中预测的频率分量的能量大小。这两个信号都可以用于分析语音信号的频率成分和语音特征，例如用于训练和评估语音识别模型。


```py
class F0Predictor(object):
    def compute_f0(self,wav,p_len):
        '''
        input: wav:[signal_length]
               p_len:int
        output: f0:[signal_length//hop_length]
        '''
        pass

    def compute_f0_uv(self,wav,p_len):
        '''
        input: wav:[signal_length]
               p_len:int
        output: f0:[signal_length//hop_length],uv:[signal_length//hop_length]
        '''
        pass
```

# `modules/F0Predictor/FCPEF0Predictor.py`

This is a function that computes the f0 (first harmonic) component of a waveform. It takes a waveform in the form of a numpy array and an optional parameter `p_len` which is the number of samples to keep after each zero-crossing window (the uv data is included in this case as well).

The function first converts the waveform to a numpy array and then passes it through a function `self.fcpe()` which applies a finite convolution operation ( a 2D convolution with a filter that is the same size as the input data and a step function). The f0 is then extracted from the first output of the function.

After that, it applies a post-processing step, which in this case is a thresholding operation.

Note that in the code snippet you provided, the function `compute_f0()` is defined first, and then the second function `compute_f0_uv()` is defined. It is possible that `compute_f0_uv()` is meant to be a class method, but I am not sure since it is not defined in the class and it has the same name as the first function defined in the class.


```py
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from modules.F0Predictor.F0Predictor import F0Predictor

from .fcpe.model import FCPEInfer


class FCPEF0Predictor(F0Predictor):
    def __init__(self, hop_length=512, f0_min=50, f0_max=1100, dtype=torch.float32, device=None, sampling_rate=44100,
                 threshold=0.05):
        self.fcpe = FCPEInfer(model_path="pretrain/fcpe.pt", device=device, dtype=dtype)
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self.name = "fcpe"

    def repeat_expand(
            self, content: Union[torch.Tensor, np.ndarray], target_len: int, mode: str = "nearest"
    ):
        ndim = content.ndim

        if content.ndim == 1:
            content = content[None, None]
        elif content.ndim == 2:
            content = content[None]

        assert content.ndim == 3

        is_np = isinstance(content, np.ndarray)
        if is_np:
            content = torch.from_numpy(content)

        results = torch.nn.functional.interpolate(content, size=target_len, mode=mode)

        if is_np:
            results = results.numpy()

        if ndim == 1:
            return results[0, 0]
        elif ndim == 2:
            return results[0]

    def post_process(self, x, sampling_rate, f0, pad_to):
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float().to(x.device)

        if pad_to is None:
            return f0

        f0 = self.repeat_expand(f0, pad_to)

        vuv_vector = torch.zeros_like(f0)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0

        # 去掉0频率, 并线性插值
        nzindex = torch.nonzero(f0).squeeze()
        f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()
        time_org = self.hop_length / sampling_rate * nzindex.cpu().numpy()
        time_frame = np.arange(pad_to) * self.hop_length / sampling_rate

        vuv_vector = F.interpolate(vuv_vector[None, None, :], size=pad_to)[0][0]

        if f0.shape[0] <= 0:
            return torch.zeros(pad_to, dtype=torch.float, device=x.device).cpu().numpy(), vuv_vector.cpu().numpy()
        if f0.shape[0] == 1:
            return (torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[
                0]).cpu().numpy(), vuv_vector.cpu().numpy()

        # 大概可以用 torch 重写?
        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
        # vuv_vector = np.ceil(scipy.ndimage.zoom(vuv_vector,pad_to/len(vuv_vector),order = 0))

        return f0, vuv_vector.cpu().numpy()

    def compute_f0(self, wav, p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        if p_len is None:
            p_len = x.shape[0] // self.hop_length
        else:
            assert abs(p_len - x.shape[0] // self.hop_length) < 4, "pad length error"
        f0 = self.fcpe(x, sr=self.sampling_rate, threshold=self.threshold)[0,:,0]
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn, rtn
        return self.post_process(x, self.sampling_rate, f0, p_len)[0]

    def compute_f0_uv(self, wav, p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        if p_len is None:
            p_len = x.shape[0] // self.hop_length
        else:
            assert abs(p_len - x.shape[0] // self.hop_length) < 4, "pad length error"
        f0 = self.fcpe(x, sr=self.sampling_rate, threshold=self.threshold)[0,:,0]
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn, rtn
        return self.post_process(x, self.sampling_rate, f0, p_len)
```

# `modules/F0Predictor/HarvestF0Predictor.py`

This is a function that uses PyAudio to compute the F0 score of a given audio signal. The F0 score is a measure of the quality of the audio signal, with higher scores indicating better quality.

The function takes two arguments: a audio signal (stored in the `wav` parameter) and an optional parameter for the length of the audio signal (stored in the `p_len` parameter). If neither of these arguments are provided, the audio signal is assumed to be a fixed-length sequence.

The function uses the PyAudio library to perform the following operations:

1. Extract a small, continuous-time audio signal from the input audio signal.
2. Convert the extracted audio signal to a double-precision floating-point number array.
3. Compute the F0 score of the audio signal using a helper function `compute_f0(self, x, target_len)` (which is defined in the `pyworld.net.f0` module), passing in the `x` audio signal and the target length of the audio signal.
4. If a `p_len` parameter was provided, compute the F0 score of the audio signal using the `compute_f0(self, wav, p_len)` function, instead of the `compute_f0(self, x, target_len)` function.
5. Return the F0 score as a double-precision floating-point number.


```py
import numpy as np
import pyworld

from modules.F0Predictor.F0Predictor import F0Predictor


class HarvestF0Predictor(F0Predictor):
    def __init__(self,hop_length=512,f0_min=50,f0_max=1100,sampling_rate=44100):
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sampling_rate = sampling_rate
        self.name = "harvest"

    def interpolate_f0(self,f0):
        '''
        对F0进行插值处理
        '''
        vuv_vector = np.zeros_like(f0, dtype=np.float32)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0
    
        nzindex = np.nonzero(f0)[0]
        data = f0[nzindex]
        nzindex = nzindex.astype(np.float32)
        time_org = self.hop_length / self.sampling_rate * nzindex
        time_frame = np.arange(f0.shape[0]) * self.hop_length / self.sampling_rate

        if data.shape[0] <= 0:
            return np.zeros(f0.shape[0], dtype=np.float32),vuv_vector

        if data.shape[0] == 1:
            return np.ones(f0.shape[0], dtype=np.float32) * f0[0],vuv_vector

        f0 = np.interp(time_frame, time_org, data, left=data[0], right=data[-1])
        
        return f0,vuv_vector
    def resize_f0(self,x, target_len):
        source = np.array(x)
        source[source<0.001] = np.nan
        target = np.interp(np.arange(0, len(source)*target_len, len(source))/ target_len, np.arange(0, len(source)), source)
        res = np.nan_to_num(target)
        return res
        
    def compute_f0(self,wav,p_len=None):
        if p_len is None:
            p_len = wav.shape[0]//self.hop_length
        f0, t = pyworld.harvest(
                wav.astype(np.double),
                fs=self.hop_length,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop_length / self.sampling_rate,
            )
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.fs)
        return self.interpolate_f0(self.resize_f0(f0, p_len))[0]

    def compute_f0_uv(self,wav,p_len=None):
        if p_len is None:
            p_len = wav.shape[0]//self.hop_length
        f0, t = pyworld.harvest(
            wav.astype(np.double),
            fs=self.sampling_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        return self.interpolate_f0(self.resize_f0(f0, p_len))

```