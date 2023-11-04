# SO-VITS-SVC源码解析 7

# `inference/infer_tool_grad.py`

这段代码的作用是实现了一个基于 librosa 的语音合成模型，可以进行实时语音合成。具体来说，代码中定义了一系列的函数和类，可以进行以下操作：

1. 导入需要用到的库，包括 io、logging、os、librosa、numpy、torch、torchaudio、soundfile 和 utils。

2. 从 librosa 库中导入了一个名为aind的函数，可以进行实时语音合成。

3. 从 librosa 库中导入了一个名为NoiseGenerator的函数，可以生成各种类型的噪声。

4. 从 parselmouth 库中导入了一个名为合成器的类，可以进行实时语音合成。

5. 在程序中定义了一个名为Slicer的类，可以对音频数据进行切片。

6. 在程序中定义了一个名为SynthesizerTrn的类，可以进行实时语音合成。

7. 在程序中定义了一个名为tasller_load_checkpoint的函数，可以在加载音频合成的实例时进行验证。

8. 在程序中定义了一个名为tasller_save_checkpoint的函数，可以将音频合成的实例保存到文件中。

9. 在程序中定义了一个名为run_server的函数，可以运行服务器，使得用户可以通过网络连接听到实时语音合成。

10. 在程序中定义了一个名为main的函数，是程序的入口点。

11. 在 main 函数中，加载了音频合成实例，并运行了服务器，使得用户可以通过网络连接听到实时语音合成。


```py
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

```

以下是 Python 函数实现：

```pypython

def get_f0(x, p_len,f0_up_key=0):
   f0_coarse, f0 = get_f0_range(x, p_len, f0_up_key)
   f0 *= np.power(2, f0_up_key / 12)
   f0_mel = 1127 * np.log(1 + f0 / 700)
   f0_mel = f0_mel[f0_mel > 0]
   f0_coarse = np.int(f0_mel)
   return f0_coarse, f0
```

该函数 `get_f0` 接受一个时间序列 `x`，长度为 `p_len` 的信号，并返回一个二进制分数 `f0` 和一个浮点数 `f0_coarse`，其中 `f0` 是用 P rule 计算得到的语音信号的 F0 值，`f0_coarse` 是浮点数形式的 `f0` 值。`f0_up_key` 是用于控制 `f0_coarse` 值输出的键，它的值在 0 到 1127 之间。如果 `f0_up_key` 为 0，则 `f0` 和 `f0_coarse` 将返回相同的结果。

函数首先使用 `get_f0_range` 函数计算 `f0_up_key` 对应的 F0 范围，该函数接受一个时间序列 `x`，长度为 `p_len` 的信号，并返回一个浮点数 `f0_up_key` 和一个浮点数 `f0_min`。`get_f0_range` 函数使用单个能量函数(energy function)和高斯滤波器(gaussian filter)对信号进行预处理，以提高计算效率。


```py
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def resize2d_f0(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)),
                       source)
    res = np.nan_to_num(target)
    return res

def get_f0(x, p_len,f0_up_key=0):

    time_step = 160 / 16000 * 1000
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    f0 = parselmouth.Sound(x, 16000).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']

    pad_size=(p_len - len(f0) + 1) // 2
    if(pad_size>0 or p_len - len(f0) - pad_size>0):
        f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')

    f0 *= pow(2, f0_up_key / 12)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int)
    return f0_coarse, f0

```



这段代码定义了三个函数，分别是 `clean_pitch()`、`plt_pitch()` 和 `f0_to_pitch()`。它们的作用如下：

1. `clean_pitch()` 函数的输入参数 `input_pitch` 是一个NumPy数组，代表音频信号的节奏。该函数的主要目的是对输入的 `input_pitch` 数进行预处理，主要方法是检测 `input_pitch` 中是否包含实数 1，并将其替换为相应的值。替换的条件是，`input_pitch` 中包含的实数 1 的数量与 `input_pitch` 数的长度之比大于 90%。这是因为，如果 `input_pitch` 中包含的实数 1 的数量大于 90%，那么实数 1 的频率将会比其他频率高得多，从而影响输出结果的准确性。

2. `plt_pitch()` 函数的输入参数与 `clean_pitch()` 函数相同，同样是 `input_pitch` 数。该函数的主要目的是将输入的 `input_pitch` 数转换为浮点数，并对输入的 `input_pitch` 数进行类似 `clean_pitch()` 函数的预处理。具体来说，它使用 `np.nan` 函数从 `input_pitch` 中创建缺失值，然后使用 `np.log2` 函数将其转换为浮点数。最后，它使用这些浮点数计算出一个更陡峭的曲线，从而将实数 1 的频率替换为相应的值。

3. `f0_to_pitch()` 函数的输入参数是一个浮点数，表示一个音频信号的F0值(即频率)。该函数的主要目的是将这些F0值转换为对应的音高，即Pitch。它使用一个简单的公式，将F0值乘以12再加上69，得到一个介于69和100之间的浮点数，表示为完整的音高。这个函数的输出结果是一个浮点数，表示将F0值转换为对应的音高后得到的值。


```py
def clean_pitch(input_pitch):
    num_nan = np.sum(input_pitch == 1)
    if num_nan / len(input_pitch) > 0.9:
        input_pitch[input_pitch != 1] = 1
    return input_pitch


def plt_pitch(input_pitch):
    input_pitch = input_pitch.astype(float)
    input_pitch[input_pitch == 1] = np.nan
    return input_pitch


def f0_to_pitch(ff):
    f0_pitch = 69 + 12 * np.log2(ff / 440)
    return f0_pitch


```

2) This code appears to be a Python implementation of the Lirien Surround (LRS) audio codec. LRS is a surround audio codec that uses a different analysis technique than B-tree based codecs, such as the analysis of the sound waves rather than the frequency or time-domain representation of the audio.

3) In this code, the `LRS` class inherits from the `soundpy` library and overrides several methods including `infer` and `chunks2audio`. The `infer` method takes in an audio signal, a sampling rate, and a number of slices (a combination of left and right channels), and returns the processed audio signal. The `chunks2audio` method takes in a binary audio file and a number of slices, and returns a binary audio file with the processed audio samples.

4) The `write_wav` function from the `io` module is used to write the audio signal to a WAV file. The `chunks` variable is created by calling the `slicer` module, which is a library for audio slicing. This library is used to slice the audio file based on certain parameters such as the sampling rate, the number of channels, and the number of slices.

5) The `extend` method is used to add audio data to the `audio` array. This method is called on every slice in the `LRS` class and multiplies the audio data by 32768 to convert it to an integer scale.


```py
def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])


def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


class VitsSvc(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.SVCVITS = None
        self.hps = None
        self.speakers = None
        self.hubert_soft = utils.get_hubert_model()

    def set_device(self, device):
        self.device = torch.device(device)
        self.hubert_soft.to(self.device)
        if self.SVCVITS is not None:
            self.SVCVITS.to(self.device)

    def loadCheckpoint(self, path):
        self.hps = utils.get_hparams_from_file(f"checkpoints/{path}/config.json")
        self.SVCVITS = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model)
        _ = utils.load_checkpoint(f"checkpoints/{path}/model.pth", self.SVCVITS, None)
        _ = self.SVCVITS.eval().to(self.device)
        self.speakers = self.hps.spk

    def get_units(self, source, sr):
        source = source.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            units = self.hubert_soft.units(source)
            return units


    def get_unit_pitch(self, in_path, tran):
        source, sr = torchaudio.load(in_path)
        source = torchaudio.functional.resample(source, sr, 16000)
        if len(source.shape) == 2 and source.shape[1] >= 2:
            source = torch.mean(source, dim=0).unsqueeze(0)
        soft = self.get_units(source, sr).squeeze(0).cpu().numpy()
        f0_coarse, f0 = get_f0(source.cpu().numpy()[0], soft.shape[0]*2, tran)
        return soft, f0

    def infer(self, speaker_id, tran, raw_path):
        speaker_id = self.speakers[speaker_id]
        sid = torch.LongTensor([int(speaker_id)]).to(self.device).unsqueeze(0)
        soft, pitch = self.get_unit_pitch(raw_path, tran)
        f0 = torch.FloatTensor(clean_pitch(pitch)).unsqueeze(0).to(self.device)
        stn_tst = torch.FloatTensor(soft)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            x_tst = torch.repeat_interleave(x_tst, repeats=2, dim=1).transpose(1, 2)
            audio,_ = self.SVCVITS.infer(x_tst, f0=f0, g=sid)[0,0].data.float()
        return audio, audio.shape[-1]

    def inference(self,srcaudio,chara,tran,slice_db):
        sampling_rate, audio = srcaudio
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        soundfile.write("tmpwav.wav", audio, 16000, format="wav")
        chunks = slicer.cut("tmpwav.wav", db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio("tmpwav.wav", chunks)
        audio = []
        for (slice_tag, data) in audio_data:
            length = int(np.ceil(len(data) / audio_sr * self.hps.data.sampling_rate))
            raw_path = io.BytesIO()
            soundfile.write(raw_path, data, audio_sr, format="wav")
            raw_path.seek(0)
            if slice_tag:
                _audio = np.zeros(length)
            else:
                out_audio, out_sr = self.infer(chara, tran, raw_path)
                _audio = out_audio.cpu().numpy()
            audio.extend(list(_audio))
        audio = (np.array(audio) * 32768.0).astype('int16')
        return (self.hps.data.sampling_rate,audio)

```

# `inference/slicer.py`

这段代码的作用是计算基于 RMS 和silence_start 的音频片段，并将它们按照时间分割成不同的片段。其中，RMS 指的均方误差，silence_start 和 silence_end 分别指静音开始时间和静音结束时间，hop_size 指时间分割步长，total_frames 指总的音频帧数。

具体实现过程如下：

1. 首先，定义了一些变量和函数，包括 total_frames、silence_start、silence_end、hop_size、max_sil_kept。

2. 如果静音开始时间等于总的音频帧数，则说明有一段音频没有任何标签，直接返回一个片段，这个片段的长度为 0。

3. 定义一个 sil_tags 列表，用于记录每个静音片段的位置和时长。

4. 遍历 sil_tags 列表中的每个元素，首先找到该元素的起始位置和结束位置，然后计算出该元素在 total_frames 中的偏移量，并将它加入 sil_tags 列表中。

5. 对于每个静音片段，根据起始时间和结束时间计算出该片段的时长，并将其添加到 sil_tags 列表中。

6. 如果 RMS 列表中没有任何元素，则直接返回一个片段，这个片段的长度为 0。

7. 最后，如果 RMS 列表中的最后一个元素有音频帧数超过总的音频帧数，则该片段的结束时间将会超过总的音频帧数，需要进行调整。

8. 最后，定义了一个 chunk_dict 字典，用于记录每个 RMS 对象，该字典的键为 RMS 对象的索引，值为一个字典，其中包含该 RMS 对象的所有信息，包括起始位置、结束位置、时长等。

9. 返回 chunk_dict。


```py
import librosa
import torch
import torchaudio


class Slicer:
    def __init__(self,
                 sr: int,
                 threshold: float = -40.,
                 min_length: int = 5000,
                 min_interval: int = 300,
                 hop_size: int = 20,
                 max_sil_kept: int = 5000):
        if not min_length >= min_interval >= hop_size:
            raise ValueError('The following condition must be satisfied: min_length >= min_interval >= hop_size')
        if not max_sil_kept >= hop_size:
            raise ValueError('The following condition must be satisfied: max_sil_kept >= hop_size')
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)]

    # @timeit
    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = librosa.to_mono(waveform)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return {"0": {"slice": False, "split_time": f"0,{len(waveform)}"}}
        rms_list = librosa.feature.rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.threshold:
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue
            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue
            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # Need slicing. Record the range of silent frames to be removed.
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start: i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept: silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        # Apply and return slices.
        if len(sil_tags) == 0:
            return {"0": {"slice": False, "split_time": f"0,{len(waveform)}"}}
        else:
            chunks = []
            # 第一段静音并非从头开始，补上有声片段
            if sil_tags[0][0]:
                chunks.append(
                    {"slice": False, "split_time": f"0,{min(waveform.shape[0], sil_tags[0][0] * self.hop_size)}"})
            for i in range(0, len(sil_tags)):
                # 标识有声片段（跳过第一段）
                if i:
                    chunks.append({"slice": False,
                                   "split_time": f"{sil_tags[i - 1][1] * self.hop_size},{min(waveform.shape[0], sil_tags[i][0] * self.hop_size)}"})
                # 标识所有静音片段
                chunks.append({"slice": True,
                               "split_time": f"{sil_tags[i][0] * self.hop_size},{min(waveform.shape[0], sil_tags[i][1] * self.hop_size)}"})
            # 最后一段静音并非结尾，补上结尾片段
            if sil_tags[-1][1] * self.hop_size < len(waveform):
                chunks.append({"slice": False, "split_time": f"{sil_tags[-1][1] * self.hop_size},{len(waveform)}"})
            chunk_dict = {}
            for i in range(len(chunks)):
                chunk_dict[str(i)] = chunks[i]
            return chunk_dict


```

这两段代码的主要作用是实现一个音频剪辑的功能，其中第一段代码定义了一个名为 `cut` 的函数，它接收一个音频文件路径以及一个数据库阈值（db_thresh）和一个最小长度（min_len）。这个函数使用 `librosa` 和 `torchaudio` 库来加载并处理音频文件，并使用一个自定义的切片算法对音频文件进行切片，然后返回切片的列表。第二段代码定义了一个名为 `chunks2audio` 的函数，它接收一个音频文件路径以及一个切片的字典，然后使用 `torchaudio` 库加载音频文件，并对音频进行平均值处理，使其符合第二段代码中设定的音频特征，然后将切片结果返回。


```py
def cut(audio_path, db_thresh=-30, min_len=5000):
    audio, sr = librosa.load(audio_path, sr=None)
    slicer = Slicer(
        sr=sr,
        threshold=db_thresh,
        min_length=min_len
    )
    chunks = slicer.slice(audio)
    return chunks


def chunks2audio(audio_path, chunks):
    chunks = dict(chunks)
    audio, sr = torchaudio.load(audio_path)
    if len(audio.shape) == 2 and audio.shape[1] >= 2:
        audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio = audio.cpu().numpy()[0]
    result = []
    for k, v in chunks.items():
        tag = v["split_time"].split(",")
        if tag[0] != tag[1]:
            result.append((v["slice"], audio[int(tag[0]):int(tag[1])]))
    return result, sr

```

# `inference/__init__.py`

很抱歉，我没有看到您提供的代码。如果您能提供代码或更多上下文信息，我将非常乐意帮助您解释代码的作用。


```py

```

# `modules/attentions.py`

This is a PyTorch implementation of a Multi-Head Attention Network (MHAN) decoder, which processes image data. MHAN is a type of neural network that performs multi-head self-attention operations to extract features from input data, especially image data.

The MHAN decoder consists of several fully connected (FFN) layers followed by some layers of LayerNorm, which help to store and retrieve information from the input data. The normalization layers are helpful for faster convergence and improved numerical stability.

The input to the MHAN decoder is a decoder input x and some additional information for encoding purpose, such as an encoder output g. The decoder input x is first transformed through self-attention layers, which calculate the attention weights for each image block in the input x. The attention weights are then used to compute a weighted sum of the image blocks, along with the encoded data.

After the self-attention layers, there are multiple normalization layers to help the information from the input x be stored and retrieved. Finally, the output is passed through the fully connected (FFN) layers, which perform interpolation and normalization.

Note that this implementation assumes that the input data is a tensor of image data, and it has the shape of (batch\_size, height, width, channels). You should adapt this code to your specific use case by replacing the dtype and size of the input tensor.


```py
import math

import torch
from torch import nn
from torch.nn import functional as F

import modules.commons as commons
from modules.DSConv import weight_norm_modules
from modules.modules import LayerNorm


class FFT(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers=1, kernel_size=1, p_dropout=0.,
               proximal_bias=False, proximal_init=True, isflow = False, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init
    if isflow:
      cond_layer = torch.nn.Conv1d(kwargs["gin_channels"], 2*hidden_channels*n_layers, 1)
      self.cond_pre = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, 1)
      self.cond_layer = weight_norm_modules(cond_layer, name='weight')
      self.gin_channels = kwargs["gin_channels"]
    self.drop = nn.Dropout(p_dropout)
    self.self_attn_layers = nn.ModuleList()
    self.norm_layers_0 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    for i in range(self.n_layers):
      self.self_attn_layers.append(
        MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias,
                           proximal_init=proximal_init))
      self.norm_layers_0.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(
        FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
      self.norm_layers_1.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask, g = None):
    """
    x: decoder input
    h: encoder output
    """
    if g is not None:
      g = self.cond_layer(g)

    self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
    x = x * x_mask
    for i in range(self.n_layers):
      if g is not None:
        x = self.cond_pre(x)
        cond_offset = i * 2 * self.hidden_channels
        g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
        x = commons.fused_add_tanh_sigmoid_multiply(
          x,
          g_l,
          torch.IntTensor([self.hidden_channels]))
      y = self.self_attn_layers[i](x, x, self_attn_mask)
      y = self.drop(y)
      x = self.norm_layers_0[i](x + y)

      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y)
    x = x * x_mask
    return x


```

This is a Python implementation of a multi-head self-attention model. Multi-head self-attention is a type of neural network that is designed to handle input sequences with multiple attention points. This model consists of multiple layers of self-attention modules, each of which performs multi-head attention on the input input and applies dropout to prevent over-parameterization. The model takes a variable number of input layers, and each layer has a different number of channels. This implementation includes two layer normalization layers, which are used to perform layer-norm and are necessary for some operations.

Multi-Head Attention
---------------

Multi-Head Attention performs attention by taking multiple queries (or contexts) and calculating a weighted sum of them. The `MultiHeadAttention` class takes in a query tensor and an attention mask, and returns the output attention tensor.

Here is the implementation for the Multi-Head Attention:
```pypython
class MultiHeadAttention:
   def __init__(self, hidden_channels, filter_channels, n_heads, p_dropout, window_size):
       self.hidden_channels = hidden_channels
       self.filter_channels = filter_channels
       self.n_heads = n_heads
       self.p_dropout = p_dropout
       self.window_size = window_size
       self.query_embedding = nn.Embedding(hidden_channels, 1, p_dropout)
       self.attention = nn.MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout, self.window_size)
       self.fc = nn.Linear(hidden_channels * n_heads, hidden_channels)

   def forward(self, query, mask):
       attn_mask = mask.unsqueeze(2).div(self.window_size)
       output = self.query_embedding(query, attn_mask)
       output = self.attention(output, attn_mask)
       output = output.squeeze()
       output = self.fc(output)
       return output

Function for the Dropout
-----------------------

This implementation provides an implementation for the `dropout` function that is used in the model. The `dropout` function is a simple way to prevent over-parameterization by randomly setting a fraction of the input units to zero.
```python
import random

class Dropout:
   def __init__(self, p_dropout):
       self.p_dropout = p_dropout

   def forward(self, x):
       return random.dropout(x, self.p_dropout)
```py
This implementation uses the `random.dropout` function from the `random` module. This function is used to randomly select a percentage of the input units (in this case, a tensor) to set to zero. This allows the input to retain most of its information, while still preventing over-parameterization.


```
class Encoder(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=4, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.window_size = window_size

    self.drop = nn.Dropout(p_dropout)
    self.attn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_2 = nn.ModuleList()
    for i in range(self.n_layers):
      self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size))
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
      self.norm_layers_2.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask):
    attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    x = x * x_mask
    for i in range(self.n_layers):
      y = self.attn_layers[i](x, x, attn_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y)

      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_2[i](x + y)
    x = x * x_mask
    return x


```py

This is a PyTorch implementation of a neural network model for a natural language processing task. The model has two main components: the encoder and the decoder.

The encoder takes an input of type `x` and a mask `x_mask` of length `(batch_size, max_seq_length)`. It applies multiple layers of attention mechanisms and applies a final layer of normalization.

The decoder takes the output of the encoder and a mask `h_mask` of length `(batch_size, max_seq_length)`. It applies the same attention mechanisms and applies a final layer of normalization.

The attention mechanisms are implemented using the `Commons` library, which provides efficient computation of matrix operations. The `MultiHeadAttention` layer is implemented as described in the paper `Making Text Models Better`, by Yao Sun et al.

The `FFN` layer is an implementation of the feedforward neural network architecture.

The `LayerNorm` layer is an implementation of the Layer Normalization architecture, which normalizes the activations of each layer in the network.
```
class LayerNorm(nn.Module):
   def __init__(self, input_size, hidden_size, num_layers, dropout=0.1,bn=False):
       super(LayerNorm, self).__init__()
       self.norm1 = nn.BatchNorm1L(hidden_size)
       self.norm2 = nn.BatchNorm2L(hidden_size)
       self.dropout = nn.Dropout(dropout)
       self.norm3 = nn.LayerNorm(hidden_size)
       self.norm4 = nn.LayerNorm(hidden_size)

   def forward(self, x, out):
       out = self.norm1(x)
       out = self.norm2(out)
       out = self.dropout(out)
       out = self.norm3(x)
       out = self.norm4(out)
       return out
```py
The model also has a few hyperparameters:

* `hidden_channels`: The number of hidden units in the encoder and decoder.
* `num_layers`: The number of layers in the encoder and decoder.
* `p_dropout`: A dropout probability for each layer.
* `proximal_bias`: A bias term for each layer.
* `proximal_init`: An initialization function for each layer.

The model can be forwarded through as follows:
```
x = input(input_size, max_seq_length, h=None)
out = model(x, h, h_mask)
```py
This will apply the layers of the model to the input `x` and return the output `out`, which is the decoded output.


```
class Decoder(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., proximal_bias=False, proximal_init=True, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init

    self.drop = nn.Dropout(p_dropout)
    self.self_attn_layers = nn.ModuleList()
    self.norm_layers_0 = nn.ModuleList()
    self.encdec_attn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_2 = nn.ModuleList()
    for i in range(self.n_layers):
      self.self_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias, proximal_init=proximal_init))
      self.norm_layers_0.append(LayerNorm(hidden_channels))
      self.encdec_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
      self.norm_layers_2.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask, h, h_mask):
    """
    x: decoder input
    h: encoder output
    """
    self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
    encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    x = x * x_mask
    for i in range(self.n_layers):
      y = self.self_attn_layers[i](x, x, self_attn_mask)
      y = self.drop(y)
      x = self.norm_layers_0[i](x + y)

      y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y)
      
      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_2[i](x + y)
    x = x * x_mask
    return x


```py

This is a class that wraps a neural network model for sequence prediction, called "AttentionBiasPredictive".

It inherits from the original class "PredictiveModel" and adds some new methods:

* `__call__`: a method to get the attention scores for the input sequence.
* `__init__`: an is色素 passed to the constructor.
* `train_ begin_训练`：a method to specify the训练 mode.
* `save_pretrained_save`：一个用于保存预训练的模型。


```
class MultiHeadAttention(nn.Module):
  def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
    super().__init__()
    assert channels % n_heads == 0

    self.channels = channels
    self.out_channels = out_channels
    self.n_heads = n_heads
    self.p_dropout = p_dropout
    self.window_size = window_size
    self.heads_share = heads_share
    self.block_length = block_length
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init
    self.attn = None

    self.k_channels = channels // n_heads
    self.conv_q = nn.Conv1d(channels, channels, 1)
    self.conv_k = nn.Conv1d(channels, channels, 1)
    self.conv_v = nn.Conv1d(channels, channels, 1)
    self.conv_o = nn.Conv1d(channels, out_channels, 1)
    self.drop = nn.Dropout(p_dropout)

    if window_size is not None:
      n_heads_rel = 1 if heads_share else n_heads
      rel_stddev = self.k_channels**-0.5
      self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
      self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

    nn.init.xavier_uniform_(self.conv_q.weight)
    nn.init.xavier_uniform_(self.conv_k.weight)
    nn.init.xavier_uniform_(self.conv_v.weight)
    if proximal_init:
      with torch.no_grad():
        self.conv_k.weight.copy_(self.conv_q.weight)
        self.conv_k.bias.copy_(self.conv_q.bias)
      
  def forward(self, x, c, attn_mask=None):
    q = self.conv_q(x)
    k = self.conv_k(c)
    v = self.conv_v(c)
    
    x, self.attn = self.attention(q, k, v, mask=attn_mask)

    x = self.conv_o(x)
    return x

  def attention(self, query, key, value, mask=None):
    # reshape [b, d, t] -> [b, n_h, t, d_k]
    b, d, t_s, t_t = (*key.size(), query.size(2))
    query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
    key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

    scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
    if self.window_size is not None:
      assert t_s == t_t, "Relative attention is only available for self-attention."
      key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
      rel_logits = self._matmul_with_relative_keys(query /math.sqrt(self.k_channels), key_relative_embeddings)
      scores_local = self._relative_position_to_absolute_position(rel_logits)
      scores = scores + scores_local
    if self.proximal_bias:
      assert t_s == t_t, "Proximal bias is only available for self-attention."
      scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e4)
      if self.block_length is not None:
        assert t_s == t_t, "Local attention is only available for self-attention."
        block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
        scores = scores.masked_fill(block_mask == 0, -1e4)
    p_attn = F.softmax(scores, dim=-1) # [b, n_h, t_t, t_s]
    p_attn = self.drop(p_attn)
    output = torch.matmul(p_attn, value)
    if self.window_size is not None:
      relative_weights = self._absolute_position_to_relative_position(p_attn)
      value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
      output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
    output = output.transpose(2, 3).contiguous().view(b, d, t_t) # [b, n_h, t_t, d_k] -> [b, d, t_t]
    return output, p_attn

  def _matmul_with_relative_values(self, x, y):
    """
    x: [b, h, l, m]
    y: [h or 1, m, d]
    ret: [b, h, l, d]
    """
    ret = torch.matmul(x, y.unsqueeze(0))
    return ret

  def _matmul_with_relative_keys(self, x, y):
    """
    x: [b, h, l, d]
    y: [h or 1, m, d]
    ret: [b, h, l, m]
    """
    ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
    return ret

  def _get_relative_embeddings(self, relative_embeddings, length):
    2 * self.window_size + 1
    # Pad first before slice to avoid using cond ops.
    pad_length = max(length - (self.window_size + 1), 0)
    slice_start_position = max((self.window_size + 1) - length, 0)
    slice_end_position = slice_start_position + 2 * length - 1
    if pad_length > 0:
      padded_relative_embeddings = F.pad(
          relative_embeddings,
          commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
    else:
      padded_relative_embeddings = relative_embeddings
    used_relative_embeddings = padded_relative_embeddings[:,slice_start_position:slice_end_position]
    return used_relative_embeddings

  def _relative_position_to_absolute_position(self, x):
    """
    x: [b, h, l, 2*l-1]
    ret: [b, h, l, l]
    """
    batch, heads, length, _ = x.size()
    # Concat columns of pad to shift from relative to absolute indexing.
    x = F.pad(x, commons.convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]))

    # Concat extra elements so to add up to shape (len+1, 2*len-1).
    x_flat = x.view([batch, heads, length * 2 * length])
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0,0],[0,0],[0,length-1]]))

    # Reshape and slice out the padded elements.
    x_final = x_flat.view([batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
    return x_final

  def _absolute_position_to_relative_position(self, x):
    """
    x: [b, h, l, l]
    ret: [b, h, l, 2*l-1]
    """
    batch, heads, length, _ = x.size()
    # padd along column
    x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length-1]]))
    x_flat = x.view([batch, heads, length**2 + length*(length -1)])
    # add 0's in the beginning that will skew the elements after reshape
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
    x_final = x_flat.view([batch, heads, length, 2*length])[:,:,:,1:]
    return x_final

  def _attention_bias_proximal(self, length):
    """Bias for self-attention to encourage attention to close positions.
    Args:
      length: an integer scalar.
    Returns:
      a Tensor with shape [1, 1, length, length]
    """
    r = torch.arange(length, dtype=torch.float32)
    diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
    return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


```py

This is a class that implements a convolutional neural network (CNN) model with a skip connection. The model takes an input of shape `(batch_size, input_shape, input_shape)`.

The model has two convolutional layers with a filter size of `kernel_size` and a dropout rate of `p_dropout`. The convolutional layers have a padding of `padded_conv_pad_size` to make sure that the output of the convolutional layers matches the expected input shape.

The output of the convolutional layers is passed through a `Dropout` layer with a dropout rate of `p_dropout`.

The output of the `Dropout` layer is then passed through a `Conv1d` layer with a filter size of `filter_channels` and a dropout rate of `p_dropout`. This layer skips the first `filter_channels` channels of the output, and replaces the values in these channels with the mean and standard deviation of the output channels.

The output of the `Conv1d` layer is the input to the `Activation` layer, which is a fully-connected (linear) layer with a ReLU activation function.

The model also has a `batch_size` parameter, which is used to divide the input of the model among multiple smaller input-output pairs. This is done by dividing the input of the model by `batch_size` to obtain a set of one-dimensional input-output pairs.

The model can be trained using the `torch.optim.SGD` algorithm.


```
class FFN(nn.Module):
  def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., activation=None, causal=False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.activation = activation
    self.causal = causal

    if causal:
      self.padding = self._causal_padding
    else:
      self.padding = self._same_padding

    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
    self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
    self.drop = nn.Dropout(p_dropout)

  def forward(self, x, x_mask):
    x = self.conv_1(self.padding(x * x_mask))
    if self.activation == "gelu":
      x = x * torch.sigmoid(1.702 * x)
    else:
      x = torch.relu(x)
    x = self.drop(x)
    x = self.conv_2(self.padding(x * x_mask))
    return x * x_mask
  
  def _causal_padding(self, x):
    if self.kernel_size == 1:
      return x
    pad_l = self.kernel_size - 1
    pad_r = 0
    padding = [[0, 0], [0, 0], [pad_l, pad_r]]
    x = F.pad(x, commons.convert_pad_shape(padding))
    return x

  def _same_padding(self, x):
    if self.kernel_size == 1:
      return x
    pad_l = (self.kernel_size - 1) // 2
    pad_r = self.kernel_size // 2
    padding = [[0, 0], [0, 0], [pad_l, pad_r]]
    x = F.pad(x, commons.convert_pad_shape(padding))
    return x

```py

# `modules/commons.py`

这段代码的主要目的是实现一个随机抽取语音信号的片段的功能。它接受一个输入信号 `x`，一个用于标识信号中每个片段的唯一标识符 `ids_str`，以及一个片段大小 `segment_size`。函数 `slice_pitch_segments` 将在 `x` 中抽取一个指定长度的片段，并将它们返回。函数 `rand_slice_segments_with_pitch` 将在 `x` 中抽取一个指定长度的片段，并返回一个二元组，其中第一个元素是抽出的片段，第二个元素是抽出的片段的音频特征，即 `rand_slice_segments` 函数的返回值。


```
import math

import torch
from torch.nn import functional as F


def slice_pitch_segments(x, ids_str, segment_size=4):
  ret = torch.zeros_like(x[:, :segment_size])
  for i in range(x.size(0)):
    idx_str = ids_str[i]
    idx_end = idx_str + segment_size
    ret[i] = x[i, idx_str:idx_end]
  return ret

def rand_slice_segments_with_pitch(x, pitch, x_lengths=None, segment_size=4):
  b, d, t = x.size()
  if x_lengths is None:
    x_lengths = t
  ids_str_max = x_lengths - segment_size + 1
  ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
  ret = slice_segments(x, ids_str, segment_size)
  ret_pitch = slice_pitch_segments(pitch, ids_str, segment_size)
  return ret, ret_pitch, ids_str

```py

这段代码定义了两个函数，分别用于初始化神经网络的权重和计算神经网络的padding大小。

第一个函数 `init_weights` 接收一个参数 `m`，表示神经网络的层数。函数内部根据传入的参数 `mean` 和 `std` 来设置神经网络层权重的初始值。如果传递的层类中包含 `Depthwise_Separable` 类，那么会分别对输入层和输出层的权重进行归一化处理。

第二个函数 `get_padding` 接收一个参数 `kernel_size`，表示神经网络的卷积层和池化层的核大小，以及一个参数 `dilation`，表示卷积层和池化层的 dilation 步长。函数返回一个大小为 `(kernel_size*dilation - dilation) / 2` 的填充步长。

第三个函数 `convert_pad_shape` 接收一个参数 `pad_shape`，表示神经网络输入层的padding形状。函数返回一个大小为 `[item for sublist in pad_shape for item in sublist]` 的去掉填充后的形状。

总结起来，这两个函数是神经网络构建中非常重要的函数，用于初始化神经网络层的权重和计算神经网络的填充大小。


```
def init_weights(m, mean=0.0, std=0.01):
  classname = m.__class__.__name__
  if "Depthwise_Separable" in classname:
    m.depth_conv.weight.data.normal_(mean, std)
    m.point_conv.weight.data.normal_(mean, std) 
  elif classname.find("Conv") != -1:
    m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
  return int((kernel_size*dilation - dilation)/2)


def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape


```py



这段代码定义了三个函数，其中第一个函数 `interperse()` 接收一个列表 `lst` 和一个元素 `item`，并返回一个间隔排列的整数数组。第二个函数 `kl_divergence()` 计算两个张量之间的 KL 散度，其中 `m_p` 和 `m_q` 是两个张量的参数。第三个函数 `rand_gumbel()` 从一个指定形状的概率分布中随机采样一个样本，并防止过冲现象。

具体来说，第一个函数 `interperse()` 的实现方式是通过创建一个长度为 `len(lst)` 倍，每个元素为 `item` 的列表，并将这个列表复制两次，加上一个元素为 `item` 的元素，从而形成一个长度为 `len(lst)` 倍的间隔数组，其中间隔数组的元素值都是非负的整数。

第二个函数 `kl_divergence()` 的实现方式是计算两个张量之间的 KL 散度。具体来说，它首先将两个张量的对应位置的元素相减，然后计算一个对数，这个对数是两个张量对应位置元素差的负对数之和，再乘以一个对数，最后加上一个平均值。这个平均值需要通过计算一个对数来估计，这里使用了贝尔分布对平均值的估计。

第三个函数 `rand_gumbel()` 的实现方式是从一个指定形状的概率分布中随机采样一个样本，并防止过冲现象。它具体实现了从高斯分布中随机采样一个样本，并使用了一些技巧来防止过冲现象，比如对样本进行立方修正，以及使用了一些近似计算技术，比如对数、指数和平方根等。


```
def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
  """KL(P||Q)"""
  kl = (logs_q - logs_p) - 0.5
  kl += 0.5 * (torch.exp(2. * logs_p) + ((m_p - m_q)**2)) * torch.exp(-2. * logs_q)
  return kl


def rand_gumbel(shape):
  """Sample from the Gumbel distribution, protect from overflows."""
  uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
  return -torch.log(-torch.log(uniform_samples))


```py

这段代码定义了两个函数，一个是`rand_slice_segments`，另一个是`rand_gumbel_like`。它们都接受一个二维的张量`x`作为输入参数。

`rand_slice_segments`函数的输入参数中包含一个字符串`ids_str`，它指定了要切分的段的起始和结束位置。函数返回一个与输入张量`x`具有相同大小和形状的新的张量，其中每个元素都对应输入张量中对应起始和结束位置的元素。

`rand_gumbel_like`函数的输入参数中包含一个张量`x`，它需要自己提供一个`rand_gumbel`函数来生成随机数。函数返回一个与输入张量`x`具有相同大小和形状的新的张量，其中每个元素都对应输入张量中的随机数。

`rand_gumbel_like`函数的实现可能有些复杂，因为它使用了PyTorch中的`torch.rand`函数来生成随机数。在这里，我们生成了一个具有形状`(x.size(0), max_size_str)`的张量，其中`x.size(0)`是输入张量`x`的大小，`max_size_str`是随机数中字符串的长度。由于`max_size_str`是固定的，因此我们可以在运行时计算出它的值，并将结果存储在函数中。

最后，这两个函数的实现都是在PyTorch中进行的。


```
def rand_gumbel_like(x):
  g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
  return g


def slice_segments(x, ids_str, segment_size=4):
  ret = torch.zeros_like(x[:, :, :segment_size])
  for i in range(x.size(0)):
    idx_str = ids_str[i]
    idx_end = idx_str + segment_size
    ret[i] = x[i, :, idx_str:idx_end]
  return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
  b, d, t = x.size()
  if x_lengths is None:
    x_lengths = t
  ids_str_max = x_lengths - segment_size + 1
  ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
  ret = slice_segments(x, ids_str, segment_size)
  return ret, ids_str


```py



这段代码定义了两个函数，分别是 `rand_spec_segments` 和 `get_timing_signal_1d`。

`rand_spec_segments` 函数接受一个二维的张量 `x`，一个长度参数 `x_lengths`，以及一个片段大小参数 `segment_size`。它通过计算得到 `x` 的 IDS，然后对 IDS 进行随机化，最后返回一个包含 IDS 和片段大小的列表。

`get_timing_signal_1d` 函数接受一个长度参数 `length`，以及一个通道数参数 `channels`。它返回一个经过归一化的 timing signal，即对 `position` 向量进行归一化后，再对每个时间步进行归一化。

具体地，`rand_spec_segments` 函数通过以下步骤来生成片段：

1. 计算 `x` 的 IDS:`x_lengths = x.size()[1] - segment_size`

2. 对 IDS 进行随机化：`torch.rand([b]).to(device=x.device) * x_lengths.to(device=x.device)`

3. 对 IDS 进行归一化：`ids_str = id * x_lengths.to(device=x.device) / math.max(x_lengths)`

4. 返回片段 IDS、片段大小和片段开始时间：`ret, ids_str`

`get_timing_signal_1d` 函数通过以下步骤来获取信号：

1. 计算信号的 IDS:`channels //= 2`

2. 对 IDS 进行归一化：`num_timescales = channels // 2`

3. 计算时间步的归一化步长：`log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1))`

4. 对位置向量进行插值：`scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)`

5. 对信号进行上采样：`signal = signal.view(1, channels, length)`

6. 对信号进行裁剪：`signal = signal.view(1, channels, length)`

7. 返回信号：`signal`


```
def rand_spec_segments(x, x_lengths=None, segment_size=4):
  b, d, t = x.size()
  if x_lengths is None:
    x_lengths = t
  ids_str_max = x_lengths - segment_size
  ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
  ret = slice_segments(x, ids_str, segment_size)
  return ret, ids_str


def get_timing_signal_1d(
    length, channels, min_timescale=1.0, max_timescale=1.0e4):
  position = torch.arange(length, dtype=torch.float)
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (num_timescales - 1))
  inv_timescales = min_timescale * torch.exp(
      torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment)
  scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
  signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
  signal = F.pad(signal, [0, 0, 0, channels % 2])
  signal = signal.view(1, channels, length)
  return signal


```py



这段代码定义了三个函数，分别用于对1D信号的时间戳进行修改。

第一个函数 `add_timing_signal_1d` 接收一个1D张量 `x`，并使用 `min_timescale` 和 `max_timescale` 参数来指定时间戳的采样频率。它使用 `get_timing_signal_1d` 函数获取预定义采样频率下的时间戳信号，然后将信号与 `x` 进行加法操作，最后将结果转换为与 `x` 相同的数据类型并返回。

第二个函数 `cat_timing_signal_1d` 接收一个1D张量 `x`，并使用相同的 `min_timescale` 和 `max_timescale` 参数来指定时间戳的采样频率。它使用 `get_timing_signal_1d` 函数获取预定义采样频率下的时间戳信号，然后使用 `torch.cat` 函数将 `x` 和时间戳信号进行拼接，最后将结果指定为与 `x` 相同的数据类型并返回。

第三个函数 `subsequent_mask` 接收一个定长的1D张量 `length`，用于返回一个与 `length` 长度相同的时间戳掩码。它使用 `torch.tril` 函数创建一个大小为 `length` 行 `length` 列的掩码，然后将掩码的起始位置和维度设置为 `(0, 0)`，以表示不包括的时间戳位置。


```
def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
  b, channels, length = x.size()
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
  return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
  b, channels, length = x.size()
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
  return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length):
  mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
  return mask


```py

这段代码定义了一个名为 "fused_add_tanh_sigmoid_multiply" 的函数，它接受两个输入参数 "input_a" 和 "input_b"，以及一个表示输入通道数量的参数 "n_channels"。这个函数执行以下操作：

1. 将输入 "input_a" 和 "input_b" 相加，得到一个形状为 [batch_size, input_channel_size, n_channels_per_channel] 的张量。
2. 对上述张量中的每个通道应用 tanh 函数。
3. 对上述张量中的每个通道应用 sigmoid 函数。
4. 对上述张量中的每个通道应用 input_a 和 input_b 相乘的结果，得到一个形状为 [batch_size, input_channel_size, n_channels_per_channel] 的张量。
5. 将上述张量的结果返回。

另外，定义了一个名为 "shift_1d" 的函数，它接受一个输入参数 "x"，执行以下操作：

1. 对输入 "x" 应用 F.pad 函数，得到一个形状为 [batch_size, max_channel_size] 的张量。
2. 对上述张量中的每个通道应用将张量左移一个通道长度并去除扩展零的操作。
3. 将上述张量中的结果返回。


```
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
  n_channels_int = n_channels[0]
  in_act = input_a + input_b
  t_act = torch.tanh(in_act[:, :n_channels_int, :])
  s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
  acts = t_act * s_act
  return acts


def shift_1d(x):
  x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
  return x


```py

这段代码定义了两个函数，分别是`sequence_mask`和`generate_path`。它们的作用如下：

1. `sequence_mask`函数接收两个参数：`length` 和 `max_length`。如果`max_length` 是 None，函数将 `max_length` 的值设置为 `length`。函数返回一个张量，其中元素按照从大到小的顺序排列，并且元素之间用小于号分隔。

2. `generate_path`函数接收两个参数：`duration` 和 `mask`。函数将一个张量 `duration` 中的元素从头到尾求和，然后将其扁平化。接着，函数使用 `sequence_mask` 函数对结果进行处理，使其满足 `mask` 的形状。最后，函数返回一个张量，其中元素按照从大到小的顺序排列，并且元素之间用小于号分隔。

这两个函数的主要目的是为了在给定一个时间序列 `duration` 和一个掩码 `mask` 时，生成一个指定长度的音频路径。具体来说，`generate_path` 函数会将一个长度为 `max_length` 的前缀部分添加到路径中，然后将其与一个长度为 `length` 的后缀部分相加，最后对其进行 `sequence_mask` 函数的处理，生成一个满足 `mask` 条件的音频路径。


```
def sequence_mask(length, max_length=None):
  if max_length is None:
    max_length = length.max()
  x = torch.arange(max_length, dtype=length.dtype, device=length.device)
  return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
  """
  duration: [b, 1, t_x]
  mask: [b, 1, t_y, t_x]
  """
  
  b, _, t_y, t_x = mask.shape
  cum_duration = torch.cumsum(duration, -1)
  
  cum_duration_flat = cum_duration.view(b * t_x)
  path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
  path = path.view(b, t_x, t_y)
  path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
  path = path.unsqueeze(1).transpose(2,3) * mask
  return path


```py

这段代码定义了一个名为 `clip_grad_value_` 的函数，它接受三个参数：`parameters`、`clip_value` 和 `norm_type`。

函数首先检查 `parameters` 是否为 PyTorch 的 `torch.Tensor` 类型，如果是，则将其转换为张量类型。然后，函数使用列表过滤器检查每个参数是否具有梯度信息，如果是，则将其添加到结果列表中。

接下来，函数检查 `clip_value` 是否为可选项。如果是，则将其转换为浮点数类型。否则，函数将 `clip_value` 转换为浮点数类型。

在循环中，函数计算每个参数的梯度平方的 `norm_type` 次方。如果 `clip_value` 存在，则函数将每个参数的梯度数限制在 `-clip_value` 和 `clip_value` 之间。

最后，函数将所有参数的梯度平方的 `norm_type` 次方相加，然后将其除以 `norm_type` 得到总的梯度值。


```
def clip_grad_value_(parameters, clip_value, norm_type=2):
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]
  parameters = list(filter(lambda p: p.grad is not None, parameters))
  norm_type = float(norm_type)
  if clip_value is not None:
    clip_value = float(clip_value)

  total_norm = 0
  for p in parameters:
    param_norm = p.grad.data.norm(norm_type)
    total_norm += param_norm.item() ** norm_type
    if clip_value is not None:
      p.grad.data.clamp_(min=-clip_value, max=clip_value)
  total_norm = total_norm ** (1. / norm_type)
  return total_norm

```py

# `modules/DSConv.py`

这段代码定义了一个名为 "Depthwise_Separable_Conv1D" 的类，继承自 PyTorch 中的 nn.Module 类。这个类在图像识别任务中用于实现深度可分离卷积。

该类的构造函数包含以下参数：

- in_channels：输入通道数。
- out_channels：输出通道数。
- kernel_size：卷积核的大小。
- stride：卷积核的步长。
- padding：卷积核的 padding。
- dilation：卷积核的 dilation。
- bias：是否使用生物启发的解决方案。
- padding_mode：用于指定填充的 mode。
- device：卷积操作的设备。
- dtype：输入和输出数据类型的数据类型。

该类包含两个方法：

- forward：通过将输入数据传递给 depth-wise separable convolution，得到输出数据。
- weight_norm：设置 input 和 output 数据中使用的数据规范化。


```
import torch.nn as nn
from torch.nn.utils import remove_weight_norm, weight_norm


class Depthwise_Separable_Conv1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        bias = True,
        padding_mode = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ):
      super().__init__()
      self.depth_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels,stride = stride,padding=padding,dilation=dilation,bias=bias,padding_mode=padding_mode,device=device,dtype=dtype)
      self.point_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias, device=device,dtype=dtype)
    
    def forward(self, input):
      return self.point_conv(self.depth_conv(input))

    def weight_norm(self):
      self.depth_conv = weight_norm(self.depth_conv, name = 'weight')
      self.point_conv = weight_norm(self.point_conv, name = 'weight')

    def remove_weight_norm(self):
      self.depth_conv = remove_weight_norm(self.depth_conv, name = 'weight')
      self.point_conv = remove_weight_norm(self.point_conv, name = 'weight')

```py

这段代码定义了一个名为 "Depthwise_Separable_TransposeConv1D" 的类，继承自 PyTorch 中的 nn.Module 类。这个类在图像处理、神经网络等领域中可能会有用到。

这个类的初始化函数包含了一些参数，如输入通道、输出通道、卷积核大小、步幅、填充方式等。通过这些参数，可以对输入数据进行处理，以实现深度可分离卷积操作。

从 forward 函数来看，这个类提供了一个将输入数据经过深度卷积和点卷积操作，再将结果减去输出步长的函数。其中，深度卷积和点卷积操作分别对输入数据进行处理，并将结果存储在 depth_conv 和 point_conv 变量中。最后，通过 remove_weight_norm 函数来移除输出数据中的权重规范化。


```
class Depthwise_Separable_TransposeConv1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0, 
        output_padding = 0,
        bias = True,
        dilation = 1,
        padding_mode = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ):
      super().__init__()
      self.depth_conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels,stride = stride,output_padding=output_padding,padding=padding,dilation=dilation,bias=bias,padding_mode=padding_mode,device=device,dtype=dtype)
      self.point_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias, device=device,dtype=dtype)
    
    def forward(self, input):
      return self.point_conv(self.depth_conv(input))

    def weight_norm(self):
      self.depth_conv = weight_norm(self.depth_conv, name = 'weight')
      self.point_conv = weight_norm(self.point_conv, name = 'weight')

    def remove_weight_norm(self):
      remove_weight_norm(self.depth_conv, name = 'weight')
      remove_weight_norm(self.point_conv, name = 'weight')


```py



这段代码定义了两个函数 `weight_norm_modules` 和 `remove_weight_norm_modules`，用于在 Keras 中对张量(Module)进行 weight_norm 操作。weight_norm 是一个 Keras 中的类，通过 normalize(proxied=True) 方法对张量进行归一化操作，使得张量的值在 range(0, input_shape[-1]) 范围内。

函数 `weight_norm_modules` 接收一个模块(Module)和一个可选的名称(name)，用于在创建模块时应用 weight_norm 操作。如果输入的模块是 Depthwise Separable Conv1D 或 Depthwise Separable TransposeConv1D，则直接应用 weight_norm 操作并返回。否则，调用 weight_norm 函数对模块进行初始化，并返回加上了 weight_norm 的模块。

函数 `remove_weight_norm_modules` 同样接收一个模块(Module)，并应用 remove_weight_norm 函数对模块进行去 weight_norm 操作。如果输入的模块是 Depthwise Separable Conv1D 或 Depthwise Separable TransposeConv1D，则直接应用 remove_weight_norm 函数。否则，调用 remove_weight_norm 函数对模块进行去 weight_norm 操作。


```
def weight_norm_modules(module, name = 'weight', dim = 0):
    if isinstance(module,Depthwise_Separable_Conv1D) or isinstance(module,Depthwise_Separable_TransposeConv1D):
      module.weight_norm()
      return module
    else:
      return weight_norm(module,name,dim)

def remove_weight_norm_modules(module, name = 'weight'):
    if isinstance(module,Depthwise_Separable_Conv1D) or isinstance(module,Depthwise_Separable_TransposeConv1D):
      module.remove_weight_norm()
    else:
      remove_weight_norm(module,name)
```py

# `modules/enhancer.py`

This code appears to be a script for applying effects to audio signals, such as reverb, compression, and high-pass filtering. It is written in Python and runs on the Torch library, which suggests that it is meant for use on the Linux platform with the CUDA toolkit.

The script takes in an input audio signal and applies two main effects: a low-pass filter (to remove the high frequencies from the audio) and an enhancer (to increase the volume of the audio). It uses the Resample and Enhancement classes for the respective effects, which seem to be implemented in the `torch.nn` module. The script also includes a variable for the start time of the audio, which can be used to pad the silence frames at the beginning.

The script creates a dictionary with the output results of the effects for each audio sample, and returns the enhanced audio for the last frame. It also includes a key to enable adaptive sampling rate tuning, which can be used to automatically adjust the sampling rate for the best results.


```
import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from vdecoder.nsf_hifigan.models import load_model
from vdecoder.nsf_hifigan.nvSTFT import STFT


class Enhancer:
    def __init__(self, enhancer_type, enhancer_ckpt, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        if enhancer_type == 'nsf-hifigan':
            self.enhancer = NsfHifiGAN(enhancer_ckpt, device=self.device)
        else:
            raise ValueError(f" [x] Unknown enhancer: {enhancer_type}")
        
        self.resample_kernel = {}
        self.enhancer_sample_rate = self.enhancer.sample_rate()
        self.enhancer_hop_size = self.enhancer.hop_size()
        
    def enhance(self,
                audio, # 1, T
                sample_rate,
                f0, # 1, n_frames, 1
                hop_size,
                adaptive_key = 0,
                silence_front = 0
                ):
        # enhancer start time 
        start_frame = int(silence_front * sample_rate / hop_size)
        real_silence_front = start_frame * hop_size / sample_rate
        audio = audio[:, int(np.round(real_silence_front * sample_rate)) : ]
        f0 = f0[: , start_frame :, :]
        
        # adaptive parameters
        adaptive_factor = 2 ** ( -adaptive_key / 12)
        adaptive_sample_rate = 100 * int(np.round(self.enhancer_sample_rate / adaptive_factor / 100))
        real_factor = self.enhancer_sample_rate / adaptive_sample_rate
        
        # resample the ddsp output
        if sample_rate == adaptive_sample_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate) + str(adaptive_sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, adaptive_sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
        
        n_frames = int(audio_res.size(-1) // self.enhancer_hop_size + 1)
        
        # resample f0
        f0_np = f0.squeeze(0).squeeze(-1).cpu().numpy()
        f0_np *= real_factor
        time_org = (hop_size / sample_rate) * np.arange(len(f0_np)) / real_factor
        time_frame = (self.enhancer_hop_size / self.enhancer_sample_rate) * np.arange(n_frames)
        f0_res = np.interp(time_frame, time_org, f0_np, left=f0_np[0], right=f0_np[-1])
        f0_res = torch.from_numpy(f0_res).unsqueeze(0).float().to(self.device) # 1, n_frames

        # enhance
        enhanced_audio, enhancer_sample_rate = self.enhancer(audio_res, f0_res)
        
        # resample the enhanced output
        if adaptive_factor != 0:
            key_str = str(adaptive_sample_rate) + str(enhancer_sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(adaptive_sample_rate, enhancer_sample_rate, lowpass_filter_width = 128).to(self.device)
            enhanced_audio =  self.resample_kernel[key_str](enhanced_audio)
        
        # pad the silence frames
        if start_frame > 0:
            enhanced_audio = F.pad(enhanced_audio, (int(np.round(enhancer_sample_rate * real_silence_front)), 0))
            
        return enhanced_audio, enhancer_sample_rate
        
        
```py



这段代码定义了一个名为 NsfHifiGAN 的类，继承自 PyTorch 的 `torch.nn.Module` 类。这个类在模型的初始化函数 `__init__` 中，根据传入的 `device` 参数来确定使用哪种设备来加载预训练的 HifiGAN 模型。如果 `device` 为 `None`，则使用 CPU 设备加载模型。

`sample_rate` 方法返回的是 HifiGAN 模型的采样率，也就是 mel 层输出的音频的采样率。

`hop_size` 方法返回的是 HifiGAN 模型中使用的 hopping size，也就是 mel 层之间的小波束的数量。

`forward` 方法接受两个输入，一个是 audio 信号，另一个是 f0 参数，它是用来训练时使用的低音。这个方法首先将音频信号进行 STFT 处理，然后使用 mel 计算 HoFeGAN 模型的输出。接着使用模型函数 `forward` 计算输出，并返回。


```
class NsfHifiGAN(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        print('| Load HifiGAN: ', model_path)
        self.model, self.h = load_model(model_path, device=self.device)
    
    def sample_rate(self):
        return self.h.sampling_rate
        
    def hop_size(self):
        return self.h.hop_size
        
    def forward(self, audio, f0):
        stft = STFT(
                self.h.sampling_rate, 
                self.h.num_mels, 
                self.h.n_fft, 
                self.h.win_size, 
                self.h.hop_size, 
                self.h.fmin, 
                self.h.fmax)
        with torch.no_grad():
            mel = stft.get_mel(audio)
            enhanced_audio = self.model(mel, f0[:,:mel.size(-1)]).view(-1)
            return enhanced_audio, self.h.sampling_rate
```