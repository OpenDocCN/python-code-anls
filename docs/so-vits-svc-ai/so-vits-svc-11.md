# SO-VITS-SVC源码解析 11

# `vdecoder/hifigan/models.py`

这段代码定义了一个名为“model”的类，它从两个不同的PyTorch模块中继承了AttrDict和get_padding的方法。然后，它导入了一些必要的库，包括json、os、numpy、torch和torch.nn。接下来，它定义了一系列的卷积层、池化层和归一化层，这些层都是从torch.nn中继承的。最后，它还定义了一个名为“env”的属性，以及一个名为“init_weights”的静态方法来初始化权重。


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

这段代码定义了一个名为 `load_model` 的函数，用于加载一个预训练的模型，并指定使用哪种设备(cuda或cpu)。

函数的核心部分如下：

```py 
1 $$ os.path.join(os.path.split(model_path)[0], 'config.json') 指向模型文件的配置文件，是一个json格式的文件。
2 $$ f.read() 方法从文件中读取文件内容，并返回一个字符串。
3 $$ json_config = json.loads(data) 加载并解析了配置文件的内容，返回一个Python对象。
4 $$ h = AttrDict(json_config) 创建了一个AttrDict对象，存储了模型配置文件中的所有属性。
5 $$ Generator(h).to(device) 创建了一个Generator模型，并将其加载到指定的设备(cuda或cpu)上。
6 $$ cp_dict = torch.load(model_path) 加载了预训练模型的权重，存储在`cp_dict`字典中。
7 $$ Generator.load_state_dict(cp_dict['generator']) 将预训练模型的权重加载到`Generator`对象中。
8 $$ Generator.eval() 将模型转换为评估模式，以便在评估时进行计算。
9 $$ Generator.remove_weight_norm() 移除了模型的权重在训练时执行的约束。
10 $$ del cp_dict 删除了`cp_dict`字典，这个字典包含了模型预训练时的权重。
11 $$ return generator, h 返回了预加载的模型和配置文件，其中`generator`是模型对象，`h`是配置文件对象。
```

这段代码的用途是加载一个预训练的模型，并指定使用哪种设备。这个函数的作用是将一个文件中的配置信息读取到Python对象中，并使用这些配置信息加载一个预训练的模型，并将其加载到指定的设备上，以便在评估时进行计算。


```py
def load_model(model_path, device='cuda'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = Generator(h).to(device)

    cp_dict = torch.load(model_path)
    generator.load_state_dict(cp_dict['generator'])
    generator.eval()
    generator.remove_weight_norm()
    del cp_dict
    return generator, h


```

This is a Python module that implements a U-Net architecture for image classification
It consists of a U-Net decorator and a forward function.
The U-Net decorator takes in an `stem_channels` and an `stem_padding` parameters.
It applies a series of convolutional layers, followed by a
Read more


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

在初始化函数__init__中，定义了模型的输入特征h、卷积层数 channels以及卷积核大小 kernel_size，同时定义了卷积层的参数dilation，用于控制卷积层的步幅和大小。

在__forward__函数中，按照顺序遍历了卷积层列表，对于每个卷积层，首先进行自定义的初始化，其中第一个参数是卷积层的输入，第二个参数是卷积层的参数，第三个参数是卷积层的步幅大小，第四个参数是卷积层的 dilation参数中的第一个值。然后，对于每个输入，进行F.leaky_relu激活函数的处理，再将结果与第一个卷积层的输出进行加权求和，得到第二个卷积层的输出，最后返回加权求和的结果。

在__remove_weight_norm__函数中，移除了第一个卷积层中所有卷积层的weight_norm函数，用于去除第一个卷积层的参数对外部计算引入的权重norm。


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



This is a function definition that takes in a single argument, `f0`, which is the fundamental frequency (or the frequency of a signal). The function returns three values:

1.  Sine waveforms, `sine_waves`: A tensor of the same shape as `f0` with the same values in each element.
2.  UV values: A tensor of the same shape as `f0` with the same values in each element.
3.  Noise values: A tensor of the same shape as `sine_waves` with the same values in each element.

The sine waveforms are generated by first applying the `self._f02sine` function to the input `f0`, which generates a sine wave with the same fundamental frequency as `f0`. The sine wave is then multiplied by the `self.sine_amp` factor to adjust the amplitude of the sine wave.

The UV values are generated by applying the `self._f02uv` function to the input `f0`. This function takes the UV values of `f0` and returns the UV values corresponding to the current frequency.

The noise values are generated by applying the `noise_std` parameter to the input `sine_waves`. This noise is added to the sine wave signals to introduce additional randomness in the output.

Note that the `self.sine_amp` and `self.noise_std` parameters are arbitrary and can be adjusted as needed by the user.


```py
def padDiff(x):
    return F.pad(F.pad(x, (0,0,-1,1), 'constant', 0) - x, (0,0,0,-1), 'constant', 0)

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
                 voiced_threshold=0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.onnx = False

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], \
                              device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # for normal case

            # To prevent torch.cumsum numerical overflow,
            # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
            # Buffer tmp_over_one_idx indicates the time step to add -1.
            # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1)
                              * 2 * np.pi)
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0, upp=None):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        if self.onnx:
            with torch.no_grad():
                f0 = f0[:, None].transpose(1, 2)
                f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
                # fundamental component
                f0_buf[:, :, 0] = f0[:, :, 0]
                for idx in np.arange(self.harmonic_num):
                    f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (
                        idx + 2
                    )  # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
                rad_values = (f0_buf / self.sampling_rate) % 1  ###%1意味着n_har的乘积无法后处理优化
                rand_ini = torch.rand(
                    f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device
                )
                rand_ini[:, 0] = 0
                rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
                tmp_over_one = torch.cumsum(rad_values, 1)  # % 1  #####%1意味着后面的cumsum无法再优化
                tmp_over_one *= upp
                tmp_over_one = F.interpolate(
                    tmp_over_one.transpose(2, 1),
                    scale_factor=upp,
                    mode="linear",
                    align_corners=True,
                ).transpose(2, 1)
                rad_values = F.interpolate(
                    rad_values.transpose(2, 1), scale_factor=upp, mode="nearest"
                ).transpose(
                    2, 1
                )  #######
                tmp_over_one %= 1
                tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
                cumsum_shift = torch.zeros_like(rad_values)
                cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
                sine_waves = torch.sin(
                    torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi
                )
                sine_waves = sine_waves * self.sine_amp
                uv = self._f02uv(f0)
                uv = F.interpolate(
                    uv.transpose(2, 1), scale_factor=upp, mode="nearest"
                ).transpose(2, 1)
                noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
                noise = noise_amp * torch.randn_like(sine_waves)
                sine_waves = sine_waves * uv + noise
            return sine_waves, uv, noise
        else:
            with torch.no_grad():
                # fundamental component
                fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

                # generate sine waveforms
                sine_waves = self._f02sine(fn) * self.sine_amp

                # generate uv signal
                # uv = torch.ones(f0.shape)
                # uv = uv * (f0 > self.voiced_threshold)
                uv = self._f02uv(f0)

                # noise: for unvoiced should be similar to sine_amp
                #        std = self.sine_amp/3 -> max value ~ self.sine_amp
                # .       for voiced regions is self.noise_std
                noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
                noise = noise_amp * torch.randn_like(sine_waves)

                # first: set the unvoiced part to 0 by uv
                # then: additive noise
                sine_waves = sine_waves * uv + noise
            return sine_waves, uv, noise


```

This is a PyTorch implementation of a module called `SourceModuleHnNSF` that processes sound samples. It takes in an input sample `x`, and can be used for both sine and noise sources. The sine source can be produced by a custom `SineGen` class that takes in the sampling rate, number of harmonics, and amplitude. The noise source is added with a low noise amplitude and added to the input sample. The output is the merged sine and noise sources, as well as the original input sample.


```py
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

    def forward(self, x, upp=None):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs.to(self.l_linear.weight.dtype)))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


```

In the `Masked` class, the `remove_weight_norm` method can be used to remove the weight normalization from the `LeakyReLU` and `UpsampleResBlock` classes. This can be useful when training models that use a masked input, as it removes the learned normalization that may cause issues during training.

The `remove_weight_norm` method takes an iterable of weights and calls the `remove_weight_norm` method on each weight in the iterable. This can be done in two ways:

1. Using the `remove_weight_norm` method recursively on the `LeakyReLU` and `UpsampleResBlock` classes:
```pypython
def remove_weight_norm(cls):
   for name, member in cls.__dict__.items():
       if isinstance(member, (torch.nn.functional.LeakyReLU, torch.nn.functional.UpsampleResBlock)):
           member.remove_weight_norm()
```
2. Using a proxy class to remove the normalization from the `LeakyReLU` and `UpsampleResBlock` classes:
```pypython
class NoNormalize:
   def __init__(self, model):
       self.model = model

   def forward(self, x):
       return self.model(x)

class MaskedLeakyReLU(torch.nn.functional.LeakyReLU):
   def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)
       self.no_normalize = NoNormalize(self)

   def forward(self, x):
       return self.no_normalize(x)

class MaskedUpsampleResBlock(torch.nn.functional.UpsampleResBlock):
   def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)
       self.no_normalize = NoNormalize(self)

   def forward(self, x):
       return self.no_normalize(x)
```
In this example, the `NoNormalize` class is used to create a proxy class for the `LeakyReLU` and `UpsampleResBlock` classes that do not perform normalization. The `forward` method of the `NoNormalize` class returns an instance of the `MaskedLeakyReLU` and `MaskedUpsampleResBlock` classes.


```py
class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h

        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(h["upsample_rates"]))
        self.m_source = SourceModuleHnNSF(
            sampling_rate=h["sampling_rate"],
            harmonic_num=8)
        self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(Conv1d(h["inter_channels"], h["upsample_initial_channel"], 7, 1, padding=3))
        resblock = ResBlock1 if h["resblock"] == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"])):
            c_cur = h["upsample_initial_channel"] // (2 ** (i + 1))
            self.ups.append(weight_norm(
                ConvTranspose1d(h["upsample_initial_channel"] // (2 ** i), h["upsample_initial_channel"] // (2 ** (i + 1)),
                                k, u, padding=(k - u +1 ) // 2)))
            if i + 1 < len(h["upsample_rates"]):  #
                stride_f0 = np.prod(h["upsample_rates"][i + 1:])
                self.noise_convs.append(Conv1d(
                    1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0+1) // 2))
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h["upsample_initial_channel"] // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"])):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.cond = nn.Conv1d(h['gin_channels'], h['upsample_initial_channel'], 1)
        self.upp = np.prod(h["upsample_rates"])
        self.onnx = False

    def OnnxExport(self):
        self.onnx = True
        self.m_source.l_sin_gen.onnx = True

    def forward(self, x, f0, g=None):
        # print(1,x.shape,f0.shape,f0[:, None].shape)
        if not self.onnx:
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        # print(2,f0.shape)
        har_source, noi_source, uv = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)
        x = x + self.cond(g)
        # print(124,x.shape,har_source.shape)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            # print(3,x.shape)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            # print(4,x_source.shape,har_source.shape,x.shape)
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


```

This is a PyTorch implementation of a Conv2d model for image classification. The model consists of a series of convolutional layers with a space-efficient way of computing the norms of the activations. The `norm_f` function computes the softmax of a function, and is used throughout the model.

The model takes in an input of shape `(batch_size, input_shape, input_channels)`. The `input_shape` should be (3, 16, 16) or (3, 16, 16, 16) depending on the input shape of the input tensor.

The model consists of the following components:

1. A 16x16 dense block with 32 channels and a kernel size of 3x3.
2. A 16x16 dense block with 32 channels and a kernel size of 3x3.
3. A 16x512 dense block with 32 channels and a kernel size of 3x3.
4. A 16x512 dense block with 32 channels and a kernel size of 3x3.
5. A 16x1024 dense block with 32 channels and a kernel size of 3x3.
6. A 16x1024 dense block with 32 channels and a kernel size of 3x3.
7. A 16x1024 dense block with 32 channels and a kernel size of 3x3.
8. A 16x1024 dense block with 32 channels and a kernel size of 3x3.
9. A 16x1024 dense block with 32 channels and a kernel size of 3x3.
10. A 16x1024 dense block with 32 channels and a kernel size of 3x3.
11. A 16x1024 dense block with 32 channels and a kernel size of 3x3.
12. A 16x1024 dense block with 32 channels and a kernel size of 3x3.
13. A 16x512 dense block with 32 channels and a kernel size of 3x3.
14. A 16x512 dense block with 32 channels and a kernel size of 3x3.
15. A 16x512 dense block with 32 channels and a kernel size of 3x3.
16. A 16x512 dense block with 32 channels and a kernel size of 3x3.
17. A 16x512 dense block with 32 channels and a kernel size of 3x3.
18. A 16x512 dense block with 32 channels and a kernel size of 3x3.
19. A 16x512 dense block with 32 channels and a kernel size of 3x3.
20. A 16x512 dense block with 32 channels and a kernel size of 3x3.
21. A 16x512 dense block with 32 channels and a kernel size of 3x3.
22. A 16x512 dense block with 32 channels and a kernel size of 3x3.
23. A 16x512 dense block with 32 channels and a kernel size of 3x3.
24. A 16x512 dense block with 32 channels and a kernel size of 3x3.
25. A 16x512 dense block with 32 channels and a kernel size of 3x3.


```py
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


```

这段代码定义了一个名为MultiPeriodDiscriminator的类，继承自PyTorch中的nn.Module类。

在MultiPeriodDiscriminator的初始化函数中，传递了一个periods参数，用于指定需要计算 discriminator 的周期数。如果该参数传递了空字符串，则默认为[2, 3, 5, 7, 11]。

在forward函数中，首先定义了两个列表y_d_rs和y_d_gs，用于存储计算不同周期时得到的预测结果fmap_rs和fmap_gs。接着，使用for循环遍历self.discriminators列表中的每个实例，并分别对其传入y和y_hat参数，得到对应的预测结果fmap_r和fmap_g。最后，将计算得到的预测结果存储到y_d_rs和y_d_gs列表中，并返回。

总之，这段代码的作用是定义了一个名为MultiPeriodDiscriminator的类，用于实现多个周期的Discriminator模型，以区分正负样本。


```py
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


```

这段代码定义了一个名为 DiscriminatorS 的类，继承自 PyTorch 中的 torch.nn.Module 类。该类在模型的初始化函数 `__init__` 中进行了自定义初始化，并定义了一系列的卷积层。这些卷积层在网络中起到了重要的作用，用于学习输入数据中的特征，以减少模型的复杂度，避免过拟合。

在 `__init__` 函数中，首先调用父类的 `__init__` 函数，以确保所有基本的初始化操作都已完成。然后，定义了 `norm_f` 函数，如果 `use_spectral_norm` 为 False，则使用普通的 L2 正则化，否则使用预定义的 spectral_norm。接下来，定义了一系列的卷积层，包括在 `__init__` 函数中的初始化语句以及 `forward` 函数中的前向传播语句。

在 `forward` 函数中，第一个输入是输入数据 `x`，第二个输入是一个列表，存储着所有卷积层的输出。通过循环遍历所有的卷积层，对每个卷积层中的输入数据进行处理，并使用 F.leaky_relu 激活函数对输入数据进行非线性变换。然后，将这些卷积层的输出存储到一个列表中，并使用 self.conv_post 对输入数据进行前向传播，得到一个更全面的特征表示。最后，对最后一个卷积层的输出进行扁平化操作，并将其输入到 `forward` 函数的返回中。

这段代码定义了一个用于图像分类任务的卷积神经网络模型，通过对输入数据进行特征提取和前向传播，来学习输入数据中的特征，从而实现对图像分类的任务。


```py
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


```

这段代码定义了一个名为 MultiScaleDiscriminator 的类，继承自 PyTorch 的nn.Module类。

MultiScaleDiscriminator的主要作用是训练一个判别器网络，该网络需要对输入数据进行多次降采样，并在每次降采样后对其进行不同大小的加权平均操作，从而提高模型的鲁棒性。

具体来说，在MultiScaleDiscriminator的__init__方法中，首先调用父类的构造函数，然后创建一个包含三个DiscriminatorS的实例，每个实例都采用 use\_spectral\_norm=True 的参数来对网络进行操作，这样可以帮助我们了解每个实例是如何工作的。

接着，定义了两个meanpools模块，每个模块包含两个AvgPool1d层，用于对输入数据进行平均化操作，并设置的步长为4，对输入数据进行上采样操作，并设置的步长也为4。

最后，定义了两个变量 fmap\_rs 和 fmap\_gs，用于存储对输入数据进行多次降采样的结果，并分别存储了每个降采样的大小为2的窗口的输出值。

在 forward 方法中，首先获取输入数据 y 和 y\_hat，然后遍历 self.discriminators 中的每个实例，对每个实例的 input 和 output 分别进行操作。其中，i 从0开始，表示从第一个实例开始，而不是循环 through all the instances。

对于每个 instance，首先将输入数据 y 和 y\_hat 传入 instance 的 forward 方法中，得到输出值 fmap\_r 和 fmap\_g，然后分别存储到 instance 的 meanpools 列表中。

在循环结束后，将 instance 的 meanpools 列表中的所有值提取出来，并按顺序返回，得到输出数据 y\_d\_rs，y\_d\_gs 和 fmap\_rs，fmap\_gs。


```py
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


```

这段代码定义了两个函数：feature_loss 和 discriminator_loss。它们在训练过程中用于计算 Discriminator（区分器）的损失。

feature_loss 的作用是计算输入特征（fmap_r 和 fmap_g）的梯度，然后将这些梯度平方并求和。最后，它将梯度平方的和乘以 2，得到一个损失值。这个损失值用于计算 Discriminator 的输入损失，从而训练输入数据。

discriminator_loss 稍微复杂一些。它计算输入数据（disc_real_outputs 和 disc_generated_outputs）的平方，然后将这些平方加起来。接着，它计算一个稀疏分布（用平均值表示）（r_loss 和 g_loss），将其加到总和的平方里。最后，它将这个和乘以 2，得到一个损失值。同样，这个损失值用于计算 Discriminator 的输入损失，从而训练输入数据。


```py
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


```

这段代码定义了一个名为 `generator_loss` 的函数，它接受一个名为 `disc_outputs` 的输入参数。函数的主要目的是计算生成器中的损失。

具体来说，函数首先初始化一个名为 `loss` 的变量为0，以及一个名为 `gen_losses` 的列表，用于存储生成器中的损失值。

接下来，函数遍历一个名为 `dg` 的输出张量，并计算每个输出元素通过一个立方差来实现的均方误差（MSE）。这里，MSE是一种常用的损失函数，可以鼓励生成器产生更接近真实数据的样本。

在循环结束后，函数将每个 `dg` 元素通过一个数学函数（`(1 - dg) ** 2`）计算得到，然后将其添加到 `gen_losses` 列表中。最后，函数将 `loss` 和 `gen_losses` 中的所有元素相加，得到一个匿实的损失值，作为生成器的总损失。


```py
def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

```

# `vdecoder/hifigan/nvSTFT.py`

This function appears to be for resampling audio files to a higher sampling rate. It takes as input an audio signal represented as a NumPy array, and an optional target sampling rate. The function returns the resampled audio signal and the sampling rate.

The function first checks if the input audio signal is a type of integer or a float32 data type. If it is a type of integer, the function finds the maximum magnitude of the data and sets the max_mag to that value. If the data type is a float32, the function finds the maximum magnitude of the data and sets the max_mag to 2**31 plus 1.

The function then converts the input audio signal to a PyTorch tensor and normalizes the data by dividing it by the maximum magnitude. This normalization step is done to normalize the data to a range between 0 and 1, which is what the speed of audio processing is typically讲究.

If the target sampling rate is specified, the function resamples the audio signal to that target sampling rate. If the target sampling rate is not specified, the function defaults to 32000.

The function also checks if the input audio signal contains NaN or Inf values. If either of these values are present and return_empty_on_exception is set to `True`, the function returns an empty array instead of raising an exception.

Overall, this function is a useful tool for resampling audio files to a higher sampling rate, and it can handle both integer and float32 data types.


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

This code appears to be a implementation of a Mel-Frequency Cepstral Coefficients (MFCC) model for speech processing. The Mel-Frequency Cepstral Coefficients model is a type of frequency representation for speech data that is superior to the Hz-Frequency Cepstral Coefficients model.

The code defines a class `MelFrequencyCe pilots`. This class inherits from the `torch.nn.Module` class and contains a method `forward`, which applies the mel-frequency cepstral coefficients model to an input audio waveform represented as a 2D tensor.

The `MelFrequencyCe` class includes several utility methods, including `get_ mel_features` which converts a given audio signal into a mel-frequency representation, and `generate_mel_spec` which generates a mel-frequency representation from an audio signal.

The `generate_mel_spec` method uses the `stft` function from the `librosa` library to convert the input audio waveform into a mel-frequency signal. It then uses this mel-frequency signal to compute the Mel-Frequency Cepstral Coefficients, and stores the result in the `spec` attribute.

The `forward` method then calls the `generate_mel_spec` method and passes it the input audio waveform. It uses the mel-frequency basis dictionary (`self.mel_basis`) to convert the input audio waveform into a mel-frequency representation and stores the result in the `spec` attribute.

It also passes the input audio waveform through a pre-processing step (right-clicking on the audio waveform) and a normalization step (max value clipping).

In summary, this code implements a Mel-Frequency Cepstral Coefficients model for speech processing that is superior to the Hz-Frequency Cepstral Coefficients model.


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

这是一个使用STFT()函数来创建一个STFT(静音模式)的陶渊明(tentative)数据类型的二世应用程序根对象(可能)。STFT()函数的作用是将一个音频信号(模拟或数字)应用STFT算法，将其转换为时间序列数据，其中静音模式(默认模式)将信号的较低部分保留，而将信号的高部分替换为零，从而实现提取信号的低频成分。


```py
stft = STFT()

```

# `vdecoder/hifigan/utils.py`

这段代码的主要作用是定义一个名为`plot_spectrogram`的函数，用于将给定的`spectrogram`图像进行可视化。

具体来说，它使用PyTorch从当前目录中查找所有.csv文件，并读取其中的数据。然后，它将读取的图像转换为numpy数组，并将其传递给`plot_spectrogram`函数。

函数的核心部分是使用Matplotlib库将图像进行可视化，并返回画布和图例。函数还使用`torch.nn.utils.weight_norm`函数对图像中的权重进行归一化，以便在图中有更好的可视化效果。


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



这段代码定义了三个函数，可能是在一个神经网络模型的初始化过程中使用的。

第一个函数 `init_weights` 接受一个参数 `m`，表示一个特殊的神经网络模型类，它需要一个权重矩阵。这个函数的作用是将权重向量初始化为具有给定参数的平均值和标准差的一个示例。如果模型的类名包含 "Conv"，那么函数会使用预定义的 "functional-network-init.py" 中的 `weight_norm` 函数来对权重进行归一化处理。

第二个函数 `apply_weight_norm` 是第二个用于对权重进行归一化处理的功能函数。它接收一个神经网络模型对象 `m`，然后将其传递给 `weight_norm` 函数，这个函数将根据预定义的参数对权重进行归一化处理。

第三个函数 `get_padding` 接收两个参数，一个是神经网络模型的卷积核大小，另一个是卷积核对输入图像的填充模式(dilation)，它会计算出卷积核对输入图像的填充缺口的尺寸，然后将该尺寸除以2，得到一个合适的填充厚度。

总结起来，这段代码定义了三个函数，它们是在神经网络模型初始化过程中使用的，分别用于设置权重、对权重进行归一化处理和对输入图像进行填充，从而使得神经网络模型能够更好地对数据进行处理和学习。


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

这段代码定义了三个函数：load_checkpoint、save_checkpoint和del_old_checkpoints。

load_checkpoint函数用于从指定文件夹中加载模型checkpoint，并使用指定设备来加载。它首先检查文件是否存在，然后加载模型checkpoint，将加载的模型存储为字典类型。最后，打印成功加载的提示信息，并返回加载的模型字典。

save_checkpoint函数用于将指定的模型保存到文件中。它接收一个文件路径和一个对象（通常是模型），然后将模型保存到文件中。最后，打印保存成功的提示信息。

del_old_checkpoints函数用于删除不需要的旧checkpoint。它接收一个checkpoint目录和一个前缀（通常是数字），然后遍历目录中的所有文件。对于每个旧的checkpoint，它首先尝试打开文件，如果文件已存在，则打印文件已被删除的提示信息，并将其从文件系统中删除。如果文件不存在，它将创建一个新的checkpoint文件。在遍历过程中，它创建的新的checkpoint数量不能超过指定的n_models数量。


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

该函数的作用是目录中查找一个名为 "cp_dir" 和 "prefix" 的目录，并在其中查找一个名为 "???" 的文件。如果目录中不存在这样的文件，则返回 None。如果存在，则返回该文件。

具体来说，函数调用了 os.path.join(cp_dir, prefix + '??') 获取目录中的 "cp_dir" 和 "prefix" 目录，并在这个目录下查找一个名为 "???" 的文件。函数使用 glob.glob(pattern) 函数遍历目录下的所有文件，并使用 sorted(cp_list)[-1] 方法获取排好序的文件列表中的最后一个文件，即找到的文件的名称。如果目录中不存在这样的文件，则返回 None。


```py
def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


```

# `vdecoder/hifiganwithsnake/env.py`

这段代码定义了一个名为 AttrDict 的类，该类使用 Python 的 built-in `**` 星号参数来分发参数。AttrDict 的构造函数接收任意数量的参数，并通过 `*args` 和 `**kwargs` 来获取它们。

该类有一个特殊的方法 `__init__`，用于初始化 AttrDict 对象，该方法使用 `super()` 来调用父类（如果没有的话）的 `__init__` 方法，并使用 `**kwargs` 来自动获取和处理传递给 `__init__` 的关键字参数。

另外，该类还定义了一个名为 `build_env` 的函数，该函数接受一个配置对象（使用 `config` 参数指定）、一个配置文件名（使用 `config_name` 参数指定）和一个输出目录（使用 `path` 参数指定）。该函数首先创建一个名为 `path/config_name` 的目录，然后将配置文件复制到该目录中，最后在目录中创建一个名为 `.config` 的自定义文件。

总结起来，这段代码定义了一个用于构建环境配置对象的 AttrDict 类，以及一个名为 `build_env` 的函数来将配置文件从源目录复制到目标目录中。


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

# `vdecoder/hifiganwithsnake/models.py`

这段代码是一个基于PyTorch实现的神经网络模型，其作用是实现了一个神经网络模型的实例，该模型使用了Snake Alias。这个模型是由一个编码器和一个解码器组成的。

具体来说，这个模型使用了PyTorch中的json和os模块，来读取和创建JSON文件和目录。它也使用了NumPy和PyTorch中的nn模块，来实现神经网络模型的设计和训练。

在具体实现中，这个模型通过继承自nn.Module，来实现模型的结构和训练方法。它包含了一个编码器和一个解码器，其中编码器主要负责将输入序列转化为模型的输出，解码器则主要负责将模型的输出转化为图像的还原。

此外，这个模型还包含了一些函数，如get_padding和init_weights，用于设置模型的参数和初始化权重在训练之前。


```py
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from vdecoder.hifiganwithsnake.alias.act import SnakeAlias

from .env import AttrDict
from .utils import get_padding, init_weights

```

这段代码定义了一个名为`load_model`的函数，它接受一个模型路径参数，并返回一个用于加载模型的`Generator`对象。

具体来说，这段代码的作用如下：

1. 读取指定模型的配置文件（配置文件中包含模型的结构、参数等）。
2. 从配置文件中解析出模型的h参数。
3. 加载预训练的生成器模型，并将其加载到定义的设备上（目前只支持cuda设备）。
4. 从加载的模型中加载已经训练好的状态，并将其设置为当前状态，以便后续的训练。
5. 执行预处理操作，包括将权重norm归零，以及移除已经保存的cpog。
6. 返回生成的`Generator`对象和模型配置结构（即h参数）。


```py
LRELU_SLOPE = 0.1


def load_model(model_path, device='cuda'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = Generator(h).to(device)

    cp_dict = torch.load(model_path)
    generator.load_state_dict(cp_dict['generator'])
    generator.eval()
    generator.remove_weight_norm()
    del cp_dict
    return generator, h


```

This is a PyTorch implementation of a two-stage convolutional neural network (CNN) model. The first stage performs a series of 1D convolutional neural networks (CNNs), while the second stage performs a series of 2D convolutional neural networks (2D CNNs). The model has two sets of weights, one for the 1D CNNs and one for the 2D CNNs.

The 1D CNNs are initialized with some default values, such as a small padding value for the input dimension and the value of the initial weights for the weights. The 2D CNNs are initialized with the same default values.

The model also has a snake fusing layer, which is used to concatenate the output from the two sets of weights.

In the forward pass, the input is passed through the first set of weights, which performs a series of 1D convolutional neural networks. Then, the output from the first set of weights is passed through the second set of weights, which performs a series of 2D convolutional neural networks. Finally, the output from the second set of weights is passed through the snake fusing layer.


```py
class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5), C=None):
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

        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = nn.ModuleList([
            SnakeAlias(channels, C=C) for _ in range(self.num_layers)
        ])

    def forward(self, x, DIM=None):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x, DIM)
            xt = c1(xt)
            xt = a2(xt, DIM)
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

在__init__方法中，参数h表示输入图像的高度， channels表示输入通道的数量，kernel_size表示卷积核的大小，dilation表示卷积层的 dilation 参数，C表示卷积层中使用的参数。

在init_weights方法中，对卷积层的权重进行初始化。

在forward方法中，首先定义了一个从输入图像到隐藏图的映射，然后对每个卷积层进行处理，最后返回处理后的结果。

ResBlock2类实现了以下方法：

- __init__(self, h, channels, kernel_size=3, dilation=(1, 3), C=None)：与父类中的__init__方法类似，初始化参数并调用父类的init_weights方法。
- forward(self, x, DIM=None)：从输入图像 x 中进行一些处理，首先通过每个卷积层的第一个权重参数得到一个xt，然后通过第二个卷积层的两个权重参数得到一个xt+x，最后返回处理后的结果。
- remove_weight_norm(self)：移除每个卷积层的权重，从每个卷积层的第一个权重参数开始。


```py
class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3), C=None):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)
        
        self.num_layers = len(self.convs)
        self.activations = nn.ModuleList([
            SnakeAlias(channels, C=C) for _ in range(self.num_layers)
        ])

    def forward(self, x, DIM=None):
        for c,a in zip(self.convs, self.activations):
            xt = a(x, DIM)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


```

It seems like there is a missing input or a problem in the code snippet you provided. Can you please provide more context or have a specific question? I\'ll be happy to help if you can provide more information.


```py
def padDiff(x):
    return F.pad(F.pad(x, (0,0,-1,1), 'constant', 0) - x, (0,0,0,-1), 'constant', 0)

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
                 voiced_threshold=0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.onnx = False

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], \
                              device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # for normal case

            # To prevent torch.cumsum numerical overflow,
            # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
            # Buffer tmp_over_one_idx indicates the time step to add -1.
            # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1)
                              * 2 * np.pi)
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0, upp=None):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        
        if self.onnx:
            with torch.no_grad():
                f0 = f0[:, None].transpose(1, 2)
                f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
                # fundamental component
                f0_buf[:, :, 0] = f0[:, :, 0]
                for idx in np.arange(self.harmonic_num):
                    f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (
                        idx + 2
                    )  # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
                rad_values = (f0_buf / self.sampling_rate) % 1  ###%1意味着n_har的乘积无法后处理优化
                rand_ini = torch.rand(
                    f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device
                )
                rand_ini[:, 0] = 0
                rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
                tmp_over_one = torch.cumsum(rad_values, 1)  # % 1  #####%1意味着后面的cumsum无法再优化
                tmp_over_one *= upp
                tmp_over_one = F.interpolate(
                    tmp_over_one.transpose(2, 1),
                    scale_factor=upp,
                    mode="linear",
                    align_corners=True,
                ).transpose(2, 1)
                rad_values = F.interpolate(
                    rad_values.transpose(2, 1), scale_factor=upp, mode="nearest"
                ).transpose(
                    2, 1
                )  #######
                tmp_over_one %= 1
                tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
                cumsum_shift = torch.zeros_like(rad_values)
                cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
                sine_waves = torch.sin(
                    torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi
                )
                sine_waves = sine_waves * self.sine_amp
                uv = self._f02uv(f0)
                uv = F.interpolate(
                    uv.transpose(2, 1), scale_factor=upp, mode="nearest"
                ).transpose(2, 1)
                noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
                noise = noise_amp * torch.randn_like(sine_waves)
                sine_waves = sine_waves * uv + noise
            return sine_waves, uv, noise
        else:
            with torch.no_grad():
                # fundamental component
                fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

                # generate sine waveforms
                sine_waves = self._f02sine(fn) * self.sine_amp

                # generate uv signal
                # uv = torch.ones(f0.shape)
                # uv = uv * (f0 > self.voiced_threshold)
                uv = self._f02uv(f0)

                # noise: for unvoiced should be similar to sine_amp
                #        std = self.sine_amp/3 -> max value ~ self.sine_amp
                # .       for voiced regions is self.noise_std
                noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
                noise = noise_amp * torch.randn_like(sine_waves)

                # first: set the unvoiced part to 0 by uv
                # then: additive noise
                sine_waves = sine_waves * uv + noise
            return sine_waves, uv, noise


```

This is a PyTorch implementation of a `SourceModuleHnNSF` class that generates a source for a sine wave with a specified sampling rate and a specified add noise standard. The class has an `__init__` method that sets the basic parameters of the sine wave and a `__call__` method that generates the source. The `__call__` method takes in an input signal `x` and returns the output of the `l_sin_gen` and other methods.


```py
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

    def forward(self, x, upp=None):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs.to(self.l_linear.weight.dtype)))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


```

This is a Python implementation of a neural network model that uses snake models to capture temporal dependencies in input data. The snake models are initialized in the constructor and have a `pre`, `back`, and `post` method to process the input data accordingly. The model also has a `remove_weight_norm` method to remove the weight normalization from the input data.


```py
class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h

        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(h["upsample_rates"]))
        self.m_source = SourceModuleHnNSF(
            sampling_rate=h["sampling_rate"],
            harmonic_num=8)
        self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(Conv1d(h["inter_channels"], h["upsample_initial_channel"], 7, 1, padding=3))
        resblock = ResBlock1 if h["resblock"] == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"])):
            c_cur = h["upsample_initial_channel"] // (2 ** (i + 1))
            self.ups.append(weight_norm(
                ConvTranspose1d(h["upsample_initial_channel"] // (2 ** i), h["upsample_initial_channel"] // (2 ** (i + 1)),
                                k, u, padding=(k - u + 1) // 2)))
            if i + 1 < len(h["upsample_rates"]):  #
                stride_f0 = np.prod(h["upsample_rates"][i + 1:])
                self.noise_convs.append(Conv1d(
                    1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0+ 1) // 2))
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        self.snakes = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h["upsample_initial_channel"] // (2 ** (i + 1))
            self.snakes.append(SnakeAlias(h["upsample_initial_channel"] // (2 ** (i)), C = h["upsample_initial_channel"] >> i))
            for j, (k, d) in enumerate(zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"])):
                self.resblocks.append(resblock(h, ch, k, d, C = h["upsample_initial_channel"] >> (i + 1)))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.snake_post = SnakeAlias(ch, C = h["upsample_initial_channel"] >> len(self.ups))
        self.cond = nn.Conv1d(h['gin_channels'], h['upsample_initial_channel'], 1)
        self.upp = np.prod(h["upsample_rates"])
        self.onnx = False

    def OnnxExport(self):
        self.onnx = True
        self.m_source.l_sin_gen.onnx = True
        
    def forward(self, x, f0, g=None):
        # print(1,x.shape,f0.shape,f0[:, None].shape)
        if not self.onnx:
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        # print(2,f0.shape)
        har_source, noi_source, uv = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)
        x = x + self.cond(g)
        # print(124,x.shape,har_source.shape)
        for i in range(self.num_upsamples):
            # print(f"self.snakes.{i}.pre:", x.shape)
            x = self.snakes[i](x)
            # print(f"self.snakes.{i}.after:", x.shape)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            # print(4,x_source.shape,har_source.shape,x.shape)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            # print(f"self.resblocks.{i}.after:", xs.shape)
            x = xs / self.num_kernels
        x = self.snake_post(x)
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


```

This is a PyTorch implementation of a Conv2d neural network model. It consists of a pre-trained conv2d model and a post-trained normalization function. The `Conv2d` model takes a 2D input of shape `(batch_size, input_shape, input_shape)` and applies a series of convolutional, activation, and padding operations. The `norm_f` function applies a forward-propagated normalization function to the output of the `Conv2d` model.

The `forward` method takes a 2D input of shape `(batch_size, input_shape, input_shape)` and returns the output of the `Conv2d` model. It works by first converting the input to a 1D tensor, then applying padding to the 1D tensor to make it a 2D tensor that has the same shape as the input shape. It then applies the `Conv2d` model to the 2D tensor. Finally, it applies the `norm_f` function to each output of the `Conv2d` model.

The `norm_f` function is defined in the `torch.nn.functional` module and applies a normalization function to the output of the `Conv2d` model. It takes as input the output of the `Conv2d` model and applies the function to each feature map. The function returns the normalized output.


```py
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


```

这段代码定义了一个名为MultiPeriodDiscriminator的类，继承自PyTorch中的nn.Module类。

在MultiPeriodDiscriminator的构造函数中，我们创建了一个`periods`变量，如果没有任何传递的参数，`periods`变量将包含预设的值，即`[2, 3, 5, 7, 11]`。我们创建了一个`discriminators`变量，它是一个`nn.ModuleList`，用于存储多个Discriminator组件。

在`forward`函数中，我们首先定义了两个变量`y_d_rs`和`y_d_gs`，它们分别存储了输入`y`和预测`y_hat`对应的discriminator输出。我们还定义了四个变量`fmap_rs`、`fmap_gs`和`fmap_d_rs`、`fmap_d_gs`，它们分别存储了输入`y`和预测`y_hat`对应的discriminator的输出，以及相应的discriminator的输出。

对于每个discriminator组件，我们首先将输入`y`和预测`y_hat`传入，然后获取输出`y_d_r`、`fmap_r`、`y_d_g`和`fmap_g`。我们还将`y_d_rs`、`fmap_rs`、`y_d_gs`和`fmap_gs`分别添加到了`y_d_rs`、`fmap_rs`、`y_d_gs`和`fmap_gs`中。

最后，我们返回了`y_d_rs`、`y_d_gs`、`fmap_rs`和`fmap_gs`，作为 forward 函数的输出。


```py
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


```

这段代码定义了一个名为 DiscriminatorS 的类，继承自 PyTorch 中的nn.Module类。该类在__init__函数中进行了初始化，并定义了一些用于从输入数据中提取特征的卷积层。这些卷积层使用 weight_norm 参数来使用空间归一化，如果没有使用该参数，则将使用 spectral_norm 函数对卷积层的权重进行归一化。

在forward函数中，该类将输入数据 x 通过第一个卷积层，然后对结果进行 F.leaky_relu 激活函数，并将其存储在 fmap 列表中。接着，通过第二个卷积层，对 fmap 进行再次应用 F.leaky_relu 激活函数，并将其存储在 fmap 列表中。最后，将 fmap 列表中的一维数据扁平化，使其具有形状（batch\_size，view\_size）。

该类实例的 forward 函数在每次前向传播过程中都会计算 fmap，并在每次前向传播后将其存储在 fmap 列表中。最终，该类将返回一个由 fmap 列表中的数据组成的输出，作为前馈神经网络中的前向隐藏层的输出。


```py
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


```

这段代码定义了一个名为 MultiScaleDiscriminator 的类，继承自 PyTorch 中的 nn.Module 类。

MultiScaleDiscriminator 用于实现图像分类任务，它包含一个 DiscriminatorS 组件，用于生成与输入数据相似的模拟数据。这个 DiscriminatorS 组件包含两个相同的层，每个层使用不同规格的卷积来提取特征。

在 forward 方法中，MultiScaleDiscriminator 对输入数据 y 和 y_hat 进行处理，首先将它们输入到第一个 DiscriminatorS 层中，然后提取出输出。对于每个输出样本，MultiScaleDiscriminator 会提取出输入数据中的特征，并输入到第二个 DiscriminatorS 层中。这样，MultiScaleDiscriminator 就可以输出一个包含两个数据相似度的结果，以及每个数据对应的输出特征。


```py
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


```

这段代码定义了两个函数，一个是 `feature_loss()`，另一个是 `discriminator_loss()`。它们的作用是计算在训练过程中来自数据增强的输入（fmap_r 和 fmap_g）的损失。

`feature_loss()` 的作用是计算数据增强（fmap_r 和 fmap_g）的梯度，然后乘以 2。具体地，它遍历 fmap_r 和 fmap_g 中的每一对点（dr, dg），并计算它们之间的差的平方。然后，对于每一对点，它计算其与逆元（也就是通过 1/根号2 得到的相似度）之间的差的绝对值。最后，它将这两个绝对值相加，得到一个与数据增强相关的损失值。

`discriminator_loss()` 的作用是计算生成数据（disc_generated_outputs）与真实数据（disc_real_outputs）之间的损失。具体地，它遍历数据对中的每一对点（dr, dg），并计算它们之间的差的平方。然后，对于每一对点，它计算其与逆元（也就是通过 1/根号2 得到的相似度）之间的差的绝对值。最后，它将这两个绝对值相加，得到一个与生成数据相关的损失值。


```py
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


```

这段代码定义了一个名为 `generator_loss` 的函数，它接受一个名为 `disc_outputs` 的输入参数。函数内部有一些变量，包括 `loss` 和 `gen_losses`，它们都被初始化为0。

函数内部使用一个 for 循环来遍历一个名为 `disc_outputs` 的列表。在循环中，使用一个变量 `dg` 来表示当前遍历的样本。通过 `dg` 计算一个样本的损失值，并将其添加到 `gen_losses` 列表中。最后，将 `gen_losses` 列表中的所有值相加，并返回总损失值。

这段代码的具体作用是计算一个数据集中所有样本的平均损失值，并返回它。通过遍历数据集中的每个样本，并对其进行计算，函数可以适应不同的数据分布和采样策略。


```py
def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

```