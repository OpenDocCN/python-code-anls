# SO-VITS-SVC源码解析 9

# `modules/F0Predictor/PMF0Predictor.py`

This code appears to be a function that computes the f0-uv values for a given waveform `wav` and a sampling rate `p


```py
import numpy as np
import parselmouth

from modules.F0Predictor.F0Predictor import F0Predictor


class PMF0Predictor(F0Predictor):
    def __init__(self,hop_length=512,f0_min=50,f0_max=1100,sampling_rate=44100):
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sampling_rate = sampling_rate
        self.name = "pm"
    
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
    

    def compute_f0(self,wav,p_len=None):
        x = wav
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        time_step = self.hop_length / self.sampling_rate * 1000
        f0 = parselmouth.Sound(x, self.sampling_rate).to_pitch_ac(
            time_step=time_step / 1000, voicing_threshold=0.6,
            pitch_floor=self.f0_min, pitch_ceiling=self.f0_max).selected_array['frequency']

        pad_size=(p_len - len(f0) + 1) // 2
        if(pad_size>0 or p_len - len(f0) - pad_size>0):
            f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')
        f0,uv = self.interpolate_f0(f0)
        return f0

    def compute_f0_uv(self,wav,p_len=None):
        x = wav
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        time_step = self.hop_length / self.sampling_rate * 1000
        f0 = parselmouth.Sound(x, self.sampling_rate).to_pitch_ac(
            time_step=time_step / 1000, voicing_threshold=0.6,
            pitch_floor=self.f0_min, pitch_ceiling=self.f0_max).selected_array['frequency']

        pad_size=(p_len - len(f0) + 1) // 2
        if(pad_size>0 or p_len - len(f0) - pad_size>0):
            f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')
        f0,uv = self.interpolate_f0(f0)
        return f0,uv

```

# `modules/F0Predictor/RMVPEF0Predictor.py`

This is a PyTorch implementation of the Fast Fourier Transform (FFT) algorithm. The `compute_f0` and `compute_f0_uv` methods handle the computation of the FFT output, while the `post_process` method is responsible for applying interpolation and normalization to the FFT output.

The FFT is defined as:
```pyscss
F = an妻=('(u/内脏是从，al
```
The implementation is based on the维吾尔库料油脂生活饮食为推荐。实现里做是关键，周边好是由过程所有提供，共同。里提供服务，同时那，让非你所有发挥余地，切记防止出现与此相关的投诉。


```py
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from modules.F0Predictor.F0Predictor import F0Predictor

from .rmvpe import RMVPE


class RMVPEF0Predictor(F0Predictor):
    def __init__(self,hop_length=512,f0_min=50,f0_max=1100, dtype=torch.float32, device=None,sampling_rate=44100,threshold=0.05):
        self.rmvpe = RMVPE(model_path="pretrain/rmvpe.pt",dtype=dtype,device=device)
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
        self.name = "rmvpe"

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
        
        vuv_vector = F.interpolate(vuv_vector[None,None,:],size=pad_to)[0][0]

        if f0.shape[0] <= 0:
            return torch.zeros(pad_to, dtype=torch.float, device=x.device).cpu().numpy(),vuv_vector.cpu().numpy()
        if f0.shape[0] == 1:
            return (torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[0]).cpu().numpy() ,vuv_vector.cpu().numpy()
    
        # 大概可以用 torch 重写?
        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
        #vuv_vector = np.ceil(scipy.ndimage.zoom(vuv_vector,pad_to/len(vuv_vector),order = 0))
        
        return f0,vuv_vector.cpu().numpy()

    def compute_f0(self,wav,p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        f0 = self.rmvpe.infer_from_audio(x,self.sampling_rate,self.threshold)
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn,rtn
        return self.post_process(x,self.sampling_rate,f0,p_len)[0]
    
    def compute_f0_uv(self,wav,p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        f0 = self.rmvpe.infer_from_audio(x,self.sampling_rate,self.threshold)
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn,rtn
        return self.post_process(x,self.sampling_rate,f0,p_len)
```

# `modules/F0Predictor/__init__.py`

我需要更具体的上下文来回答你的问题。可以请你提供更多背景信息以及完整的代码，这样我才能够清楚地解释代码的作用。


```py

```

# `modules/F0Predictor/fcpe/model.py`



这段代码包含了以下几个主要部分：

1. 导入必要的库：numpy, torch, torch.nn, torch.nn.functional, torch.nn.utils, and torchaudio.transforms。
2. 导入自定义的STFT类和PCmer类。
3. 定义了一个名为"l2_regularization"的函数，该函数接受一个模型(model)和一个L2正则化参数(l2_alpha)。函数内部对模型中的每个模块(即每个卷积层)的权重进行平方，然后取平均值并除以2，最后乘以L2正则化参数，得到一个含有L2正则化项的列表。
4. 从numpy库中导入了一个名为"Resample"的函数，该函数用于对音频数据进行重新采样。
5. 创建了一个名为"PCmer"的新类，该类实现了PCM(即无带宽信号处理中的波形立方根)算法。
6. 创建了一个名为"STFT"的新类，该类实现了STFT(即无带宽信号处理中的短时傅里叶变换)算法。
7. 在函数中调用了自定义的STFT类和PCmer类中的所有函数，并将它们的结果存储在一个变量中。


```py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torchaudio.transforms import Resample

from .nvSTFT import STFT
from .pcmer import PCmer


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


```

This is a PyTorch implementation of a Gaussian blurred version of the centroid technology. The centroid is a concept used in machine learning to estimate the parameters of a distribution from a set of data.

This implementation uses a mask to determine which observations correspond to which centroid. The centroids are estimated using a Gaussian filter to smooth the centroid values. The smoothing factor is determined by the `f0_to_cent` function, which converts the f0-score of a Centered高质量 model to the centroid. The function is defined in the paper "Estimating the parameters of a probabilistic generative model".

The `gaussian_blurred_cent` function takes a centroid `cents` and returns the blurred centroid.

This implementation is based on the paper "Estimating the parameters of a probabilistic generative model" by Yoshua Bengio，沫不休，持浅疾。


```py
class FCPE(nn.Module):
    def __init__(
            self,
            input_channel=128,
            out_dims=360,
            n_layers=12,
            n_chans=512,
            use_siren=False,
            use_full=False,
            loss_mse_scale=10,
            loss_l2_regularization=False,
            loss_l2_regularization_scale=1,
            loss_grad1_mse=False,
            loss_grad1_mse_scale=1,
            f0_max=1975.5,
            f0_min=32.70,
            confidence=False,
            threshold=0.05,
            use_input_conv=True
    ):
        super().__init__()
        if use_siren is True:
            raise ValueError("Siren is not supported yet.")
        if use_full is True:
            raise ValueError("Full model is not supported yet.")

        self.loss_mse_scale = loss_mse_scale if (loss_mse_scale is not None) else 10
        self.loss_l2_regularization = loss_l2_regularization if (loss_l2_regularization is not None) else False
        self.loss_l2_regularization_scale = loss_l2_regularization_scale if (loss_l2_regularization_scale
                                                                             is not None) else 1
        self.loss_grad1_mse = loss_grad1_mse if (loss_grad1_mse is not None) else False
        self.loss_grad1_mse_scale = loss_grad1_mse_scale if (loss_grad1_mse_scale is not None) else 1
        self.f0_max = f0_max if (f0_max is not None) else 1975.5
        self.f0_min = f0_min if (f0_min is not None) else 32.70
        self.confidence = confidence if (confidence is not None) else False
        self.threshold = threshold if (threshold is not None) else 0.05
        self.use_input_conv = use_input_conv if (use_input_conv is not None) else True

        self.cent_table_b = torch.Tensor(
            np.linspace(self.f0_to_cent(torch.Tensor([f0_min]))[0], self.f0_to_cent(torch.Tensor([f0_max]))[0],
                        out_dims))
        self.register_buffer("cent_table", self.cent_table_b)

        # conv in stack
        _leaky = nn.LeakyReLU()
        self.stack = nn.Sequential(
            nn.Conv1d(input_channel, n_chans, 3, 1, 1),
            nn.GroupNorm(4, n_chans),
            _leaky,
            nn.Conv1d(n_chans, n_chans, 3, 1, 1))

        # transformer
        self.decoder = PCmer(
            num_layers=n_layers,
            num_heads=8,
            dim_model=n_chans,
            dim_keys=n_chans,
            dim_values=n_chans,
            residual_dropout=0.1,
            attention_dropout=0.1)
        self.norm = nn.LayerNorm(n_chans)

        # out
        self.n_out = out_dims
        self.dense_out = weight_norm(
            nn.Linear(n_chans, self.n_out))

    def forward(self, mel, infer=True, gt_f0=None, return_hz_f0=False, cdecoder = "local_argmax"):
        """
        input:
            B x n_frames x n_unit
        return:
            dict of B x n_frames x feat
        """
        if cdecoder == "argmax":
            self.cdecoder = self.cents_decoder
        elif cdecoder == "local_argmax":
            self.cdecoder = self.cents_local_decoder
        if self.use_input_conv:
            x = self.stack(mel.transpose(1, 2)).transpose(1, 2)
        else:
            x = mel
        x = self.decoder(x)
        x = self.norm(x)
        x = self.dense_out(x)  # [B,N,D]
        x = torch.sigmoid(x)
        if not infer:
            gt_cent_f0 = self.f0_to_cent(gt_f0)  # mel f0  #[B,N,1]
            gt_cent_f0 = self.gaussian_blurred_cent(gt_cent_f0)  # #[B,N,out_dim]
            loss_all = self.loss_mse_scale * F.binary_cross_entropy(x, gt_cent_f0)  # bce loss
            # l2 regularization
            if self.loss_l2_regularization:
                loss_all = loss_all + l2_regularization(model=self, l2_alpha=self.loss_l2_regularization_scale)
            x = loss_all
        if infer:
            x = self.cdecoder(x)
            x = self.cent_to_f0(x)
            if not return_hz_f0:
                x = (1 + x / 700).log()
        return x

    def cents_decoder(self, y, mask=True):
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        rtn = torch.sum(ci * y, dim=-1, keepdim=True) / torch.sum(y, dim=-1, keepdim=True)  # cents: [B,N,1]
        if mask:
            confident = torch.max(y, dim=-1, keepdim=True)[0]
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= self.threshold] = float("-INF")
            rtn = rtn * confident_mask
        if self.confidence:
            return rtn, confident
        else:
            return rtn
        
    def cents_local_decoder(self, y, mask=True):
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        confident, max_index = torch.max(y, dim=-1, keepdim=True)
        local_argmax_index = torch.arange(0,9).to(max_index.device) + (max_index - 4)
        local_argmax_index[local_argmax_index<0] = 0
        local_argmax_index[local_argmax_index>=self.n_out] = self.n_out - 1
        ci_l = torch.gather(ci,-1,local_argmax_index)
        y_l = torch.gather(y,-1,local_argmax_index)
        rtn = torch.sum(ci_l * y_l, dim=-1, keepdim=True) / torch.sum(y_l, dim=-1, keepdim=True)  # cents: [B,N,1]
        if mask:
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= self.threshold] = float("-INF")
            rtn = rtn * confident_mask
        if self.confidence:
            return rtn, confident
        else:
            return rtn

    def cent_to_f0(self, cent):
        return 10. * 2 ** (cent / 1200.)

    def f0_to_cent(self, f0):
        return 1200. * torch.log2(f0 / 10.)

    def gaussian_blurred_cent(self, cents):  # cents: [B,N,1]
        mask = (cents > 0.1) & (cents < (1200. * np.log2(self.f0_max / 10.)))
        B, N, _ = cents.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        return torch.exp(-torch.square(ci - cents) / 1250) * mask.float()


```

This is a PyTorch implementation of an audio processing model that uses a frontend called the Fast Fourier Transform (FFT), a simple mel-Frequency Cepstral Coefficients (MFCC) extraction algorithm, and a feedback-based approach to generate high-quality audio. The audio is passed through the model, which applies different techniques depending on the type of loss function used.

The model has an input channel, an output channel, and a number of layers. The number of channels is set to the input audio's channels, and the output channel is set to the number of channels in the input audio. The use\_siren and use\_full parameters control whether the input audio is normalized by the soft-max function and whether the output is normalized, respectively.

The mel-Frequency Cepstral Coefficients (MFCC) extraction algorithm is applied to the input audio to extract a set of high-frequency coefficients. These coefficients are used as input to the FFT, which is then applied to the input audio. This is done with a batch of MFCC values, which are passed through the FFT.

The output of the FFT is passed through the model. The output is first passed through a threshold of 0.05 to control the minimum output level. Then the output is passed through the model. The model has an output\_dim parameter, which is set to the same as the input\_dim parameter.

The type of the audio processing model is set to be an audio-related model. The device is set to the CPU, but this can be changed to a GPU for faster training.

The functionable of the model is the audio-related function. The FFT is implemented using the Element-Wise Twist Law (EWTL) for the x-axis, and the Linear in the y-axis.


```py
class FCPEInfer:
    def __init__(self, model_path, device=None, dtype=torch.float32):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        ckpt = torch.load(model_path, map_location=torch.device(self.device))
        self.args = DotDict(ckpt["config"])
        self.dtype = dtype
        model = FCPE(
            input_channel=self.args.model.input_channel,
            out_dims=self.args.model.out_dims,
            n_layers=self.args.model.n_layers,
            n_chans=self.args.model.n_chans,
            use_siren=self.args.model.use_siren,
            use_full=self.args.model.use_full,
            loss_mse_scale=self.args.loss.loss_mse_scale,
            loss_l2_regularization=self.args.loss.loss_l2_regularization,
            loss_l2_regularization_scale=self.args.loss.loss_l2_regularization_scale,
            loss_grad1_mse=self.args.loss.loss_grad1_mse,
            loss_grad1_mse_scale=self.args.loss.loss_grad1_mse_scale,
            f0_max=self.args.model.f0_max,
            f0_min=self.args.model.f0_min,
            confidence=self.args.model.confidence,
        )
        model.to(self.device).to(self.dtype)
        model.load_state_dict(ckpt['model'])
        model.eval()
        self.model = model
        self.wav2mel = Wav2Mel(self.args, dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def __call__(self, audio, sr, threshold=0.05):
        self.model.threshold = threshold
        audio = audio[None,:]
        mel = self.wav2mel(audio=audio, sample_rate=sr).to(self.dtype)
        f0 = self.model(mel=mel, infer=True, return_hz_f0=True)
        return f0


```

This is a PyTorch implementation of the NVSTFT (N 不发生又会如何 velocitonal spectrogram) processor. It takes in an audio signal, a sample rate, and a number of parameters including the window size, hop size, and maximum and minimum frequency for the mel spectrogram. It returns the mel spectrogram.

The audio signal is first converted to a NumPy array and then passed through a lowpass filter to remove high-frequency components. This is done by first dividing the audio by a factor of 128, then taking the logarithm. This is done for each frame of the audio, so that the mel spectrogram can be extracted for each frame.

The `stft` function is then used to compute the mel spectrogram for the audio. This function takes the audio in the form of a NumPy array and a number of parameters including the hop size, window size, and maximum and minimum frequency. The mel spectrogram is then extracted and passed through a lowpass filter to remove high-frequency components.

The `extract_mel` function is then used to extract the mel spectrogram from the audio. This function takes the mel spectrogram in the form of a NumPy array and a number of parameters including the hop size, window size, and training flag. It resamps the mel spectrogram to the specified hop size and/or resampling rate and extracts the mel spectrogram.

The `__call__` function is used to call the `extract_mel` function on an audio signal. It takes the audio signal, sample rate, and a number of parameters including the hop size and training flag, and returns the mel spectrogram.


```py
class Wav2Mel:

    def __init__(self, args, device=None, dtype=torch.float32):
        # self.args = args
        self.sampling_rate = args.mel.sampling_rate
        self.hop_size = args.mel.hop_size
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.dtype = dtype
        self.stft = STFT(
            args.mel.sampling_rate,
            args.mel.num_mels,
            args.mel.n_fft,
            args.mel.win_size,
            args.mel.hop_size,
            args.mel.fmin,
            args.mel.fmax
        )
        self.resample_kernel = {}

    def extract_nvstft(self, audio, keyshift=0, train=False):
        mel = self.stft.get_mel(audio, keyshift=keyshift, train=train).transpose(1, 2)  # B, n_frames, bins
        return mel

    def extract_mel(self, audio, sample_rate, keyshift=0, train=False):
        audio = audio.to(self.dtype).to(self.device)
        # resample
        if sample_rate == self.sampling_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.sampling_rate, lowpass_filter_width=128)
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(self.dtype).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)

        # extract
        mel = self.extract_nvstft(audio_res, keyshift=keyshift, train=train)  # B, n_frames, bins
        n_frames = int(audio.shape[1] // self.hop_size) + 1
        if n_frames > int(mel.shape[1]):
            mel = torch.cat((mel, mel[:, -1:, :]), 1)
        if n_frames < int(mel.shape[1]):
            mel = mel[:, :n_frames, :]
        return mel

    def __call__(self, audio, sample_rate, keyshift=0, train=False):
        return self.extract_mel(audio, sample_rate, keyshift=keyshift, train=train)


```

这段代码定义了一个名为 DotDict 的类，它继承自 Python 标准库中的 dict 类型。

在这个类的定义中，定义了一个特殊的方法 __getattr__，它接受一个或多个参数，并在获取该方法时调用父类的 __getattr__ 方法。这个特殊方法的作用是确保在从 DotDict 对象中获取属性时，能够正确地获取到子类属性，即使子类中的属性名发生了变化。

此外，还定义了两个标准库中的方法 __setattr__ 和 __delattr__，它们分别用于设置和删除对象的属性。这些方法在这个类中也得以实现，以便与 dict 类保持一致的行为。

最后，将所有方法设置为 classmethod，以便将它们与类实例相关联。


```py
class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

```

# `modules/F0Predictor/fcpe/nvSTFT.py`

It seems like this is a function that loads an audio file (usually an audio file in the .wav format) and returns its data and the sampling rate. The function takes an additional argument `target_sr` which is the desired sampling rate for the audio file.

It works by following these steps:

1. The function always converts the audio data to a 2D tensor, but only if it's not already a 2D tensor.
2. It gets the maximum magnitude of the audio data.
3. If the audio data is a type int, it returns the data without any modification.
4. If the audio data is a type fp32, it normalizes the data to be between -1 and 1.
5. It converts the audio data to a PyTorch 16-bit floating-point tensor.
6. If the audio data is already in the .wav format, it resamples it to have the desired sampling rate and returns the normalized audio data and the sampling rate.

If the function raises an exception, it will return an empty array. If the function returns early, it will return the last 48000 samples of the audio file.


```py
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

```

This is a PyTorch implementation of a windowing and normalization function for audio signals. The windowing function uses the `torch.hann_window` method from the `torch.h劑扬函數庫` to determine the window size, hop length, and padding mode. The audio signal is first converted to a two-dimensional array, and then it is windowed with the specified window size and hop length. The windowed audio signal is then passed through a Mel-Frequency Cepstral Coefficients (MFCC) filter to extract mel-frequency information. Finally, the output is return the mel-frequency signal after passing through the Mel-Frequency Cepstral Coefficients filter.


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
    
    def get_mel(self, y, keyshift=0, speed=1, center=False, train=False):
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
        if not train:
            mel_basis = self.mel_basis
            hann_window = self.hann_window
        else:
            mel_basis = {}
            hann_window = {}
        
        if torch.min(y) < -1.:
            print('min value is ', torch.min(y))
        if torch.max(y) > 1.:
            print('max value is ', torch.max(y))
        
        mel_basis_key = str(fmax)+'_'+str(y.device)
        if mel_basis_key not in mel_basis:
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
            mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)
        
        keyshift_key = str(keyshift)+'_'+str(y.device)
        if keyshift_key not in hann_window:
            hann_window[keyshift_key] = torch.hann_window(win_size_new).to(y.device)
        
        pad_left = (win_size_new - hop_length_new) //2
        pad_right = max((win_size_new- hop_length_new + 1) //2, win_size_new - y.size(-1) - pad_left)
        if pad_right < y.size(-1):
            mode = 'reflect'
        else:
            mode = 'constant'
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode = mode)
        y = y.squeeze(1)
        
        spec = torch.stft(y, n_fft_new, hop_length=hop_length_new, win_length=win_size_new, window=hann_window[keyshift_key],
                          center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)                          
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + (1e-9))
        if keyshift != 0:
            size = n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size-resize))
            spec = spec[:, :size, :] * win_size / win_size_new   
        spec = torch.matmul(mel_basis[mel_basis_key], spec)
        spec = dynamic_range_compression_torch(spec, clip_val=clip_val)
        return spec
    
    def __call__(self, audiopath):
        audio, sr = load_wav_to_torch(audiopath, target_sr=self.target_sr)
        spect = self.get_mel(audio.unsqueeze(0)).squeeze(0)
        return spect

```

这是一个使用 Scikit-STFT（音译：声通转移算法）库实现语音信号时频转换（STFT）的Python代码。STFT是一种常用的算法，将时间序列转换为频域信号，以便更容易地分析和处理音频数据。


```py
stft = STFT()

```

# `modules/F0Predictor/fcpe/pcmer.py`

I'm sorry, but I'm not sure what you are asking for. Could you please provide more context or clarify your request?


```py
import math
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from local_attention import LocalAttention
from torch import nn

#import fast_transformers.causal_product.causal_product_cuda

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    b, h, *_ = data.shape
    # (batch size, head, length, model_dim)

    # normalize model dim
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    # what is ration?, projection_matrix.shape[0] --> 266
    
    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    #data_dash = w^T x
    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    
    # diag_data = D**2 
    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)
    
    #print ()
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data + eps))#- torch.max(data_dash)) + eps)

    return data_dash.type_as(data)

```

这是一个名为 `orthogonal_matrix_chunk` 的函数，它接受一个二维的列向量 `cols`，以及一个布尔值 `qr_uniform_q` 表示是否对 Q 进行统一。如果 `qr_uniform_q` 为真，则该函数将从 Reduced 形式中 Q 矩阵的每一行中提取出元素，并将其存储到以其 `device` 参数指定的设备上。函数的实现依赖于另一个名为 `exists` 的函数，它用于检查一个给定的值是否为 `None`。最后，该函数返回 Q 矩阵的第一行，如果 Q 矩阵存在的话。


```py
def orthogonal_matrix_chunk(cols, qr_uniform_q = False, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: t.to(device), (q, r))

    # proposed by @Parskatt
    # to make sure Q is uniform https://arxiv.org/pdf/math-ph/0609050.pdf
    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()
    return q.t()
def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

```

这段代码定义了两个函数，分别是default()和cast_tuple()。default()函数接受两个参数，一个是整数类型的val，另一个是布尔类型的d。函数的返回值是val，如果val存在，则返回val，否则返回d。cast_tuple()函数接受一个整数类型的val，函数的返回值是(val,)，如果val是一个元组类型，则返回val，否则返回元组(val,)。

这两个函数用于在给定的整数或元组值的基础上执行不同的操作。在PCmer类中，这些函数被实例化并在其forward()方法中使用。在forward()方法中，首先将输入phone应用所有层，然后将结果保存到内存中，最后将结果提供给用户。


```py
def default(val, d):
    return val if exists(val) else d

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

class PCmer(nn.Module):
    """The encoder that is used in the Transformer model."""
    
    def __init__(self, 
                num_layers,
                num_heads,
                dim_model,
                dim_keys,
                dim_values,
                residual_dropout,
                attention_dropout):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dim_keys = dim_keys
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout

        self._layers = nn.ModuleList([_EncoderLayer(self) for _ in range(num_layers)])
        
    #  METHODS  ########################################################################################################
    
    def forward(self, phone, mask=None):
        
        # apply all layers to the input
        for (i, layer) in enumerate(self._layers):
            phone = layer(phone, mask)
        # provide the final sequence
        return phone


```

这段代码定义了一个名为 `_EncoderLayer` 的类，属于一个名为 `nn.Module` 的子类。这个类表示一个 encoder 层，其前面还有一层 PCmer（由于没有给出更多信息，我假设它是一个 encoder 模型）。

在这个 encoder 层中，有两个方法：`__init__` 和 `forward`。

`__init__` 方法用于创建一个新的 `_EncoderLayer` 实例，并设置其参数。它的参数包括：

* `parent`：这个实例所属的 encoder 层实例。

`forward` 方法用于在给定输入 `phone` 时执行的 forward 操作。首先，它会将输入 `phone` 和一个 `mask`（可选）加到一起，然后使用 `self.attn` 方法计算注意力，接着使用 `self.conformer` 方法将注意力结果与输入 `phone` 相加，最后通过 `self.norm` 和 `self.dropout` 方法进行归一化和 dropout，得到输出结果。

由于 `self.attn` 和 `self.conformer` 方法没有给出具体实现，因此无法提供更多信息。


```py
# ==================================================================================================================== #
#  CLASS  _ E N C O D E R  L A Y E R                                                                                   #
# ==================================================================================================================== #


class _EncoderLayer(nn.Module):
    """One layer of the encoder.
    
    Attributes:
        attn: (:class:`mha.MultiHeadAttention`): The attention mechanism that is used to read the input sequence.
        feed_forward (:class:`ffl.FeedForwardLayer`): The feed-forward layer on top of the attention mechanism.
    """
    
    def __init__(self, parent: PCmer):
        """Creates a new instance of ``_EncoderLayer``.
        
        Args:
            parent (Encoder): The encoder that the layers is created for.
        """
        super().__init__()
        
        
        self.conformer = ConformerConvModule(parent.dim_model)
        self.norm = nn.LayerNorm(parent.dim_model)
        self.dropout = nn.Dropout(parent.residual_dropout)
        
        # selfatt -> fastatt: performer!
        self.attn = SelfAttention(dim = parent.dim_model,
                                  heads = parent.num_heads,
                                  causal = False)
        
    #  METHODS  ########################################################################################################

    def forward(self, phone, mask=None):
        
        # compute attention sub-layer
        phone = phone + (self.attn(self.norm(phone), mask=mask))
        
        phone = phone + (self.conformer(phone))
        
        return phone 

```

这段代码定义了一个名为 `calc_same_padding` 的函数，它用于计算在一定大小（例如 `kernel_size`）的卷积核中，两个连续的填充位置。具体来说，它会计算出两个填充位置，然后选择距离卷积核中心较小的那个位置。这个函数对于某些卷积神经网络模型非常有用，可以帮助它们在测试和训练过程中保证输入数据的尺寸一致性。

此外，代码中还定义了一个名为 `Swish` 的类，这是 `nn.Module` 中的一个子类，这个类继承自 PyTorch 的 `nn.Module` 类。这个类的 `forward` 方法中，通过将输入数据 `x` 乘以一个卷积核 `x.sigmoid()` 来计算其激活值。最后，代码中还定义了一个名为 `Transpose` 的类，它是 `nn.Module` 中的一个子类，这个类继承自 PyTorch 的 `nn.Module` 类。它的 `__init__` 方法接受一个包含两个维度的 `dims` 参数，然后创建一个形状和 `dims` 相同的卷积核。它的 `forward` 方法接受一个输入数据 `x`，然后将其通过将输入数据 `x` 传递给 `nn.Module.forward` 中的 `x.sigmoid()` 方法来计算其激活值。


```py
def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

# helper classes

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)

```

这段代码定义了一个名为 GLU 的类，该类继承自 PyTorch 中的 nn.Module 类。

在 GLU 的初始化方法中，创建了一个包含两个参数的函数，分别 dim 和 0。dim 参数表示输入数据的通道数，即输入数据中包含多少个分量。0 参数表示该层将卷积核应用于所有通道。

在 forward 方法中，首先将输入数据 x 按 channels=2 进行划分，然后对每个分量应用卷积操作。卷积操作包括以下步骤：

1. 将输入数据 x 分成 2 个分量并存储为 out 和 gate。
2. 对 gate 分量应用 sigmoid 激活函数。
3. 返回 out 和 gate 作为输出。

接下来是 DepthWiseConv1d 类，该类继承自 PyTorch 中的 nn.Module 类。

在 DepthWiseConv1d 的初始化方法中，创建了一个包含两个参数的函数，分别 chan_in 和 chan_out，以及卷积核的大小 kernel_size。

在 forward 方法中，首先将输入数据 x 按 channels=2 进行划分，然后对每个分量应用卷积操作。卷积操作包括以下步骤：

1. 将输入数据 x 分成 2 个分量并存储为 out 和 gate。
2. 对 gate 分量应用 sigmoid 激活函数。
3. 使用 DepthWiseConv1d 的 padding 参数对输出进行截断，并将其存储为 out。
4. 返回 out 作为输出。


```py
class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

```

这段代码定义了一个名为 "ConformerConvModule" 的类，继承自 PyTorch 中的 nn.Module 类。这个类在网络中被用于前馈，它的主要作用是提取输入数据中的特征并将其输入到网络中。

在 ConformerConvModule 的 __init__ 方法中，我们首先调用父类的构造函数，然后定义了一些成员变量，包括输入数据维度、 causal 设置、扩张因子、卷积核大小和 dropout 概率。

接着，我们实现了一个由两个 Transformer 层和一个前馈神经网络组成的循环神经网络，我们通过将输入数据 x 传递给第一个 Transformer 层，然后将输出结果传递给第二个 Transformer 层。在第二个 Transformer 层中，我们使用了 GLU 激活函数、一个扩张因子为 2 的深度卷积和一个 dropout 层。

最后，我们通过创建一个带两个输出层的模块来返回经过网络处理后的输入数据 x。


```py
class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            #nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Transpose((1, 2)),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

```

这段代码定义了一个名为 "linear_attention" 的函数，它接受三个输入参数 q、k 和 v，它们分别代表查询（question）、键（key）和值（value）。这个函数返回一个标量的结果。

函数内部首先检查输入的 v 是否为零，如果是，则输出一个标量。否则，函数对 k 和 q 进行相关计算，然后计算 D_inv，它是键和值之间的注意力权重。接下来，函数使用注意力权重和 k 和 q 中的信息计算结果。具体来说，函数首先对查询和键进行拼接，然后将注意力权重和键拼接，最后对注意力权重和查询拼接的结果进行拼接。

这段代码的作用是计算一个注意力权重，然后根据注意力权重对查询和键进行加权求和，得到结果。这个函数可以在神经网络中用于获取查询和关键之间的相关性信息，从而提高模型的表现。


```py
def linear_attention(q, k, v):
    if v is None:
        #print (k.size(), q.size())
        out = torch.einsum('...ed,...nd->...ne', k, q)
        return out

    else:
        k_cumsum = k.sum(dim = -2) 
        #k_cumsum = k.sum(dim = -2)
        D_inv = 1. / (torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q)) + 1e-8)

        context = torch.einsum('...nd,...ne->...de', k, v)
        #print ("TRUEEE: ", context.size(), q.size(), D_inv.size())
        out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        return out

```

这段代码定义了一个名为 "gaussian_orthogonal_random_matrix" 的函数，它接受四个参数：nb_rows（行数）、nb_columns（列数）、scaling（是否对矩阵进行缩放）和 qr_uniform_q（是否使用均匀的 QR分解）。

函数的主要作用是生成一个高斯正交矩阵，并将生成的矩阵进行归一化处理，使得其每行的行列式都为 1。高斯正交矩阵在很多应用领域中都有用处，例如在机器学习模型中，常常需要对特征矩阵进行对角化处理，生成一个对角化矩阵，这样可以简化矩阵的计算，并且避免矩阵的特征值出现垄断现象。

函数的具体实现包括以下几个步骤：

1. 计算高斯正交矩阵的行数和列数，然后根据计算出的行数计算出高斯正交矩阵的行数。

2. 如果给定了缩放因子为 0，则直接返回一个对角化的矩阵。

3. 如果给定缩放因子为 1，则将矩阵的每行和列都放大为原来的根号下（列数）倍，生成一个对角化矩阵。

4. 如果给定缩放因子为其他值，则会引发 ValueError，警告用户。

5. 最后，将生成的对角化矩阵返回。


```py
def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, qr_uniform_q = False, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)
    #print (nb_full_blocks)
    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, device = device)
        block_list.append(q)
    # block_list[n] is a orthogonal matrix ... (model_dim * model_dim)
    #print (block_list[0].size(), torch.einsum('...nd,...nd->...n', block_list[0], torch.roll(block_list[0],1,1)))
    #print (nb_rows, nb_full_blocks, nb_columns)
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    #print (remaining_rows)
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, device = device)
        #print (q[:remaining_rows].size())
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)
    
    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

```

This is a PyTorch implementation of an "Efficient Search Encoder" model, as described in the paper "Efficient Search Encoder" by Jack J. Sch月亮， Xiaoyang Wang, and Yibo Liu.

It includes two sub-models, a math.log function, and some additional attributes.

The model has a forward pass that takes in query (q), key (k), and value (v), and applies the self.generalized_attention and self.kernel_fn to compute the attention scores. If the value is set to None, the model does not use a projection matrix, and the attention computation is implemented as a simplified version in the original efficient attention paper.

The model also has a redraw_projection_matrix function that is used to update the projection matrix when the no_projection flag is set to False, and a feature swapping function that swaps the roles of queries and keys if the no_projection flag is set to False.


```py
class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features = None, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), qr_uniform_q = False, no_projection = False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling, qr_uniform_q = qr_uniform_q)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.causal = causal

    @torch.no_grad()
    def redraw_projection_matrix(self):
        projections = self.create_projection()
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device

        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)
        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
            
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        if v is None:
            out = attn_fn(q, k, None)
            return out
        else:
            out = attn_fn(q, k, v)
            return out
```



This is a implementation of a simple multi-head self-attention model. It takes as input a context vector `context` and an initial query vector `q`, and outputs a tuple of the attention outputs and the final output.

The model first checks if the context vector is not empty and if it is, it checks if there is a cross-attention term. If there is a cross-attention term, it performs the attention calculation. Otherwise, it calculates the attention using the `self.fast_attention` method. The attention is then applied to the query and key vectors using the query and key vectors, respectively.

The final output is obtained by concatenating the attention outputs and passing through a dropout layer.

Note that this implementation assumes that the initial query vector `q` is a tensor of the same shape as the context vector `context`. Additionally, the to-output function is not defined in this implementation, as it is not necessary.


```py
class SelfAttention(nn.Module):
    def __init__(self, dim, causal = False, heads = 8, dim_head = 64, local_heads = 0, local_window_size = 256, nb_features = None, feature_redraw_interval = 1000, generalized_attention = False, kernel_fn = nn.ReLU(), qr_uniform_q = False, dropout = 0., no_projection = False):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, causal = causal, generalized_attention = generalized_attention, kernel_fn = kernel_fn, qr_uniform_q = qr_uniform_q, no_projection = no_projection)

        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = LocalAttention(window_size = local_window_size, causal = causal, autopad = True, dropout = dropout, look_forward = int(not causal), rel_pos_emb_config = (dim_head, local_heads)) if local_heads > 0 else None

        #print (heads, nb_features, dim_head)
        #name_embedding = torch.zeros(110, heads, dim_head, dim_head)
        #self.name_embedding = nn.Parameter(name_embedding, requires_grad=True)
        

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def redraw_projection_matrix(self):
        self.fast_attention.redraw_projection_matrix()
        #torch.nn.init.zeros_(self.name_embedding)
        #print (torch.sum(self.name_embedding))
    def forward(self, x, context = None, mask = None, context_mask = None, name=None, inference=False, **kwargs):
        _, _, _, h, gh = *x.shape, self.heads, self.global_heads
        
        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask
        #print (torch.sum(self.name_embedding))
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []
        #print (name)
        #print (self.name_embedding[name].size())
        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)
            if cross_attend:
                pass
                #print (torch.sum(self.name_embedding))
                #out = self.fast_attention(q,self.name_embedding[name],None)
                #print (torch.sum(self.name_embedding[...,-1:]))
            else:
                out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask = mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim = 1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return self.dropout(out)
```

# `modules/F0Predictor/fcpe/__init__.py`

这段代码定义了三个函数变量：FCPEInfer、STFT 和 PCmer。它们都是模型的依赖项，来自模型目录中的 PCME、STFT 和 PCME 包。

FCPEInfer是一个函数，它从输入数据中提取 FCPE 信息。它接收一个数据张量作为输入，并返回一个 FCPE 信息掩码。

STFT是一个函数，它将时间戳数据转换为 STFT 格式。它接收一个数据张量作为输入，并返回一个 STFT 数据张量。

PCmer是一个函数，它将 PCME 数据进行拼接。它接收一个数据张量作为输入，并返回一个数据张量。


```py
from .model import FCPEInfer  # noqa: F401
from .nvSTFT import STFT  # noqa: F401
from .pcmer import PCmer  # noqa: F401

```

# `modules/F0Predictor/rmvpe/constants.py`

这段代码使用了深度学习中的概念，具体解释如下：

1. SAMPLE_RATE = 16000：这段代码定义了一个采样率(sample rate)，用于表示每秒钟的采样次数。这个采样率在语音识别、音频处理等任务中非常重要，因为它们可以帮助我们捕获说话人说话的速度、音调和语音特征。

2. N_CLASS = 360：这段代码定义了一个类数(class count)，用于表示训练数据中包含的音频样本数量。在训练机器学习模型时，我们需要指定音频信号中的特征数量，也就是音频的采样率。

3. N_MELS = 128：这段代码定义了一个特征数量(feature number)，用于表示每个音频样本中的 Mel-Frequency 特征的数量。Mel-Frequency 特征是语音信号中的一个重要特征，可以帮助模型更好地捕捉说话人的语音特征。

4. MEL_FMIN = 30：这段代码定义了一个最小 Mel-Frequency(MEL-Frequency min)，用于表示每个音频样本中最小 Mel-Frequency 的值。这个值是为了确保模型能够检测到语音信号中的某些特征，即使这些特征非常低。

5. MEL_FMAX = SAMPLE_RATE // 2：这段代码定义了一个最大 Mel-Frequency(MEL-Frequency max)，用于表示每个音频样本中最大 Mel-Frequency 的值。这个值是为了确保模型能够检测到语音信号中的某些特征，即使这些特征非常高。

6. WINDOW_LENGTH = 1024：这段代码定义了一个窗口长度(window length)，用于表示每次循环中提取的音频样本数。这个值通常取决于具体的应用场景，可以影响模型的性能和计算效率。

7. CONST = 1997.3794084376191：这段代码定义了一个常数(const)，用于表示一个常数，用于在每次循环中对音频信号进行归一化处理。这个常数可以帮助模型更好地处理不同尺度的音频数据，从而提高模型的性能和泛化能力。


```py
SAMPLE_RATE = 16000

N_CLASS = 360

N_MELS = 128
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 1024
CONST = 1997.3794084376191

```

# `modules/F0Predictor/rmvpe/deepunet.py`

这段代码定义了一个名为“ConvBlockRes”的类，属于PyTorch中的nn.Module类。这个类在图像识别任务中（如ImageNet上的分类任务）使用。

这个类中有一个__init__方法，用于初始化其参数。这个方法接收两个参数：in_channels和out_channels，分别表示输入通道和输出通道的数量。此外，这个方法还有一个momentum参数，表示对BatchNorm2d层的初始化使用什么 momentum。

在__init__方法之后，定义了一个包含两个Conv2d层的模型。第一个Conv2d层是在输入通道上执行的，其参数与第二个Conv2d层在输出通道上执行的，其参数数量与第一个输入通道相同。这两个Conv2d层在模型中分别进行前向传播，通过应用ReLU激活函数，将输入信息传递给第一个Conv2d层，然后返回其激活值。接下来，使用第二个Conv2d层在输出通道上执行的，将激活值与第二个输入通道相加，并通过短路策略（也称为“速记”）将它们的输出连接起来。

最后，在模型的forward方法中，提取第一个和第二个Conv2d层中的输出，然后将其打印出来。


```py
import torch
import torch.nn as nn

from .constants import N_MELS


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        if self.is_shortcut:
            return self.conv(x) + self.shortcut(x)
        else:
            return self.conv(x) + x


```

该代码定义了一个名为ResEncoderBlock的类，继承自PyTorch中的nn.Module类。

在类的初始化函数__init__中，定义了模型的输入通道数、输出通道数、卷积核大小和块的数量等参数，同时初始化了一个self.conv数组，每个卷积层使用了ConvBlockRes类。

在__forward__函数中，前n_blocks块中，首先通过顺序循环遍历了每个卷积层，然后对每个卷积层的输出进行归一化（平均池化）。接着，通过一个if语句判断当前块是否有卷积核，如果有，则使用该卷积核对当前块的输出进行处理。最后，将处理后的结果返回。

具体来说，该ResEncoderBlock对输入数据x进行卷积操作，通过循环遍历卷积层和池化层，来逐步提取特征并逐渐增加模型的复杂度。同时，通过池化层对输入数据进行平均化处理，可以提升模型的表示能力。


```py
class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


```

这段代码定义了一个名为ResDecoderBlock的类，继承自PyTorch中的nn.Module类。

ResDecoderBlock的主要作用是实现残差网络（Residual Network）中的一个模块，该模块可以在输入数据上对每个残差块进行处理，以提高模型的性能和残差网络的稳定性。

具体来说，ResDecoderBlock包含两个卷积层，其中第一个卷积层通过一个3x3的卷积层和一个1x1的卷积来提取输入数据中的特征，并将这些特征与一个残差块的连接起来。第二个卷积层通过多个3x3的卷积来提取输入数据的残差信息，并将其输入到ResId列表中。然后，对每个残差块，将残差块的输出与残差块的参数连接起来，以实现残差块之间的信息传递。

该代码中，通过调整卷积块的参数，可以控制残差网络中的每个残差块的大小和残差信息的比例，从而可以更好地适应不同大小的数据和需要调节的残差网络的复杂度。


```py
class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=stride,
                               padding=(1, 1),
                               output_padding=out_padding,
                               bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for i in range(n_blocks-1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        return x


```

这段代码定义了一个名为 "Encoder" 的类，继承自 PyTorch 的nn.Module类。Encoder 用于实现序列到序列（Sequence-to-Sequence）模型的编码器。

在 Encoder 的初始化函数中，首先调用父类的初始化函数，然后定义了模型的输入通道、输入数据尺寸、编码器数量、卷积核大小和输出通道数量。这些参数都用于定义模型中的参数。

接着，定义了模型中的输入层、输出层以及内部层。输入层接收需要编码的输入序列 x，输出层用于输出编码后的结果，内部层则包含多个ResEncoderBlock。ResEncoderBlock包含两个步骤：首先将输入序列 x 中的每个元素通过卷积层，然后通过一个 ReLU 激活函数。

在 forward 函数中，对输入序列 x 应用卷积层和相应的 ResEncoderBlock。通过循环遍历所有的编码器，得到编码后的结果，并将这些结果按顺序添加到 concaten_tensors 列表中。最后，返回编码后的结果和编码后的数据。


```py
class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for i in range(self.n_encoders):
            _, x = self.layers[i](x)
            concat_tensors.append(_)
        return x, concat_tensors


```

这段代码定义了一个名为 "Intermediate" 的类，继承自 PyTorch 的 nn.Module 类。这个类的创建和普通类一样，但有一个特殊的构造函数，它需要传入 in_channels、out_channels 和 n_inters 三个参数，分别代表输入通道数、输出通道数和卷积层数。

在类的 body部分，首先调用父类的构造函数，并为 layer 变量赋值。然后，创建一个 layers 列表，用于存储每个卷积层。接下来，使用 for 循环来创建每个卷积层的镜像，并为镜像设置相同的 momentum 参数。

最后，在 forward 方法中，遍历 layers 列表中的每个卷积层，并使用每个卷积层的 forward 方法来获取输出。这个输出被返回，作为 Intermediate 类的 forward 方法的返回值。


```py
class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        for i in range(self.n_inters-1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for i in range(self.n_inters):
            x = self.layers[i](x)
        return x


```

这段代码定义了一个名为 "Decoder" 的类，继承自 PyTorch 中的 nn.Module 类。这个类的设计旨在实现一个数据重建函数，其主要思想是：通过一系列的 Decoder Block 和 momentum 机制，将输入数据 x 中的通道数除以 2，并在每个 Decoder Block 中对输入数据进行逐层的非线性变换，最后将结果拼接起来。

具体来说，这段代码中定义了一个 Decoder 类，其中包含一个 init() 方法和一个 forward() 方法。在 init() 方法中，首先调用父类的 init() 方法，确保所有继承自 Decoder 的类都具有相同的初始化方式。然后，定义了一个 n_channels 变量，用于保存输入数据 x 的通道数，以及一个 n_decoders 变量，用于保存 Decoder 模块的数量。

在 forward() 方法中，首先创建一个包含 x 和 concat_tensors（可能包含输入数据）的列表。然后，使用 for 循环遍历 Decoder 模块的数量 n_decoders 次。在每次遍历中，使用 ResDecoderBlock 类对 x 和 concat_tensors[-1-i] 进行非线性变换，其中 i 是当前遍历的 Decoder 模块。注意，在每个 Decoder Block 中，in_channels 变量除以 2，以实现通道数减半。

在循环结束后，将结果拼接起来，并返回。


```py
class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i in range(self.n_decoders):
            x = self.layers[i](x, concat_tensors[-1-i])
        return x


```

这段代码定义了一个名为 "TimbreFilter" 的类，该类继承自 "nn.Module"。这个类的父类是 "nn.ModuleList"，表示在传递一个或多个 "ConvBlockRes" 类的实例时，将它们添加到一个 "TimbreFilter" 实例中。

具体来说，这个类的 "__init__" 方法接收一个名为 "latent_rep_channels" 的参数。它对这个参数进行一次传递，并创建一个 "TimbreFilter" 实例，将每个 "latent_rep_channels" 都添加到实例的 "layers" 列表中。

这个类的 "forward" 方法接收一个名为 "x_tensors" 的参数。它遍历 "layers" 列表中的每个实例，并对每个实例进行一次传递。最后，它返回所有传递给 "forward" 的参数。

这个类还有一个名为 "DeepUnet" 的子类。这个子类的 "__init__" 方法接收一个或多个参数，包括 "kernel_size"、"n_blocks" 和 "en_de_layers"。这些参数用于设置 DeepUnet 模型的架构和参数。

具体来说，这个子类的 "encoder" 实例将使用一个具有传入通道数和目标通道数（16）的 "ConvBlockRes" 模型。这个子类的 "intermediate" 实例将使用一个具有传入通道数和目标通道数（16）的 "Intermediate" 模型，以及一个具有传入通道数和目标通道数（16）的 "TimbreFilter"。

这个子类的 "tf" 实例将使用一个具有传入通道数和目标通道数（16）的 "TimbreFilter"。这个子类的 "decoder" 实例将使用一个具有传入通道数和目标通道数（16）的 "Decoder" 模型。

总的来说，这段代码定义了一个用于处理音频信号的 DeepUnet 模型。它包括一个 "TimbreFilter" 实例，用于对输入信号进行预处理，以及一个 "DeepUnet" 模型，用于对音频信号进行降噪处理。


```py
class TimbreFilter(nn.Module):
    def __init__(self, latent_rep_channels):
        super(TimbreFilter, self).__init__()
        self.layers = nn.ModuleList()
        for latent_rep in latent_rep_channels:
            self.layers.append(ConvBlockRes(latent_rep[0], latent_rep[0]))

    def forward(self, x_tensors):
        out_tensors = []
        for i, layer in enumerate(self.layers):
            out_tensors.append(layer(x_tensors[i]))
        return out_tensors


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(DeepUnet, self).__init__()
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
        self.tf = TimbreFilter(self.encoder.latent_channels)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        concat_tensors = self.tf(concat_tensors)
        x = self.decoder(x, concat_tensors)
        return x

      
```

这段代码定义了一个名为DeepUnet0的类，继承自PyTorch中的nn.Module类。这个类用于实现一个基于Unet的神经网络，其设计意图是通过将输入通道不断增加来扩展网络的深度。以下是DeepUnet0类的构造函数以及一些参数的解释：

1. `__init__`：该函数是DeepUnet0类的构造函数，用于初始化网络中的参数。
2. `encoder`：该函数是一个Encoder层，负责将输入的1个通道的图像数据转化为具有N Mel特征的输出。该函数的参数包括：输入通道的宽度（in_channels），Mel特征数（N_MELS），需要增加的DeepUnet深度层数（en_de_layers），卷积核大小（kernel_size）和卷积核步长（n_blocks）。
3. `intermediate`：该函数是一个Intermediate层，负责将Encoder的输出进行合并。该函数需要一个输入通道，它是Encoder的第二个输出通道除以2的结果。此外，该函数还包括需要增加的层数（inter_layers）和层数（n_blocks）。
4. `tf`：该函数是一个TimbreFilter，负责对需要进行处理的 latent 通道进行加权。
5. `decoder`：该函数是一个Decoder层，负责从输入的多个通道的图像数据中提取信息，并将其输入到DeepUnet中。该函数的参数包括：输出通道的宽度（out_channels），DeepUnet的深度层数（en_de_layers），卷积核大小（kernel_size）和卷积核步长（n_blocks）。
6. `forward`：该函数是一个前向传播函数，负责将输入的图像数据传入DeepUnet中进行处理，并在返回前对处理结果进行汇总。在这个函数中，首先将输入的图像数据传入Encoder层，然后将结果传入Intermediate层，最后将结果传入Decoder层进行处理。


```py
class DeepUnet0(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(DeepUnet0, self).__init__()
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
        self.tf = TimbreFilter(self.encoder.latent_channels)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x

```