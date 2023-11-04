# SO-VITS-SVC源码解析 0

# `compress_model.py`

这段代码是一个PyTorch代码片段，它的作用是将从collections.OrderedDict导入的OrderedDict对象复制到models.SynthesizerTrn类中。

具体来说，这段代码首先定义了一个名为copyStateDict的函数，它接收一个OrderedDict对象作为参数。函数内部创建了一个新的OrderedDict对象，并遍历输入的OrderedDict对象的键值对。如果输入的键值对包含'module'这个前缀，则从模块中导入该模块并将其存储到新字典中，否则直接将键值对存储到新字典中。最后，函数返回新字典。

接着，代码定义了一个名为copyStateDictionary的函数，它与copyStateCode的函数类似，只是返回类型为PyTorch版本的OrderedDict对象。


```py
from collections import OrderedDict

import torch

import utils
from models import SynthesizerTrn


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith('module'):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ','.join(k.split('.')[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


```

这段代码是一个名为 `removeOptimizer` 的函数，其作用是移除一个特定模型的优化器配置，并将其保存到指定的输出模型文件中。优化器配置包括网络结构、学习率设置、优化器类型、半径设置等。

函数首先通过调用 `utils.get_hparams_from_file` 函数获取输入模型的参数配置。接着，根据参数配置初始化一个 `SynthesizerTrn` 对象，这个对象用于构建网络结构。然后，创建一个用于优化网络参数的实例，设置优化器的超参数，并使用优化器对网络参数进行优化。接下来，加载输入模型的参数配置，并将其复制到一个新字典中，这个新字典只包含保存的模型参数。

最后，将新的字典保存到指定的输出模型文件中，同时保存一个包含模型参数的字典，其中保存的参数是只用 half 精度计算的。


```py
def removeOptimizer(config: str, input_model: str, ishalf: bool, output_model: str):
    hps = utils.get_hparams_from_file(config)

    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1,
                           hps.train.segment_size // hps.data.hop_length,
                           **hps.model)

    optim_g = torch.optim.AdamW(net_g.parameters(),
                                hps.train.learning_rate,
                                betas=hps.train.betas,
                                eps=hps.train.eps)

    state_dict_g = torch.load(input_model, map_location="cpu")
    new_dict_g = copyStateDict(state_dict_g)
    keys = []
    for k, v in new_dict_g['model'].items():
        if "enc_q" in k: continue  # noqa: E701
        keys.append(k)
    
    new_dict_g = {k: new_dict_g['model'][k].half() for k in keys} if ishalf else {k: new_dict_g['model'][k] for k in keys}

    torch.save(
        {
            'model': new_dict_g,
            'iteration': 0,
            'optimizer': optim_g.state_dict(),
            'learning_rate': 0.0001
        }, output_model)


```

这段代码是一个条件判断语句，如果当前脚本被命名为 `__main__`，那么就会执行以下操作：

1. 导入 `argparse` 模块以使用 `argparse.ArgumentParser`；
2. 定义了 `parser` 对象，用于解析用户输入的参数；
3. 向 `parser` 添加了四个参数：`-c`、`-i`、`-o` 和 `-hf`；
4. 通过 `parser.parse_args()` 方法对用户输入的参数进行解析，并返回一个名为 `args` 的对象；
5. 如果 `args.half` 为 `True`，那么将 `_half` 添加到 `args.output` 的后缀中；
6. 如果 `args.output` 是空字符串，则执行 2 和 3 步，否则执行 3 和 4 步；
7. `removeOptimizer()` 函数用于移除特定文件中的 `_optimize` 注释。


```py
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--config",
                        type=str,
                        default='configs/config.json')
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument('-hf', '--half', action='store_true', default=False, help='Save as FP16')
    
    args = parser.parse_args()

    output = args.output

    if output is None:
        import os.path
        filename, ext = os.path.splitext(args.input)
        half = "_half" if args.half else ""
        output = filename + "_release" + half + ext

    removeOptimizer(args.config, args.input, args.half, output)
```

# `data_utils.py`

这段代码的作用是实现一个基于多个说话人语音数据的学习绕音识别系统。其中，主要分为以下几个部分：

1. 导入必要的库，包括os、random、numpy、torch、torch.utils.data、utils、 mel_processing、spectrogram_torch等库。

2. 导入h5py库，以便于读取h5格式的数据。

3. 定义了一个基于 mel 的数据处理模块，包括将wav格式数据转换为torch张量、对数据进行归一化、对数据进行绘制、绘制图形等。

4. 加载多个说话人的数据，并保存在一个变量中，以便于后续的处理。

5. 加载一个数据集文件夹，并返回其中的所有wav文件和对应的保护 Range。

6. 定义一个基于读取文件的函数，其中包括读取文件、返回对应的torch张量等操作。

7. 在数据读取完成后，对数据进行处理，并更新对应的图绘制函数，以便于后续的处理。

8. 在每次处理完数据后，对系统进行优化，包括对数据进行加权平均等操作。

9. 最终，定义一个主函数，用来启动程序，读取多个说话人的数据，并绕音识别。


```py
import os
import random

import numpy as np
import torch
import torch.utils.data

import utils
from modules.mel_processing import spectrogram_torch
from utils import load_filepaths_and_text, load_wav_to_torch

# import h5py


"""Multi speaker version"""


```

This appears to be a PyTorch implementation of a sample-to-speech (STS) model. It appears to have some limitations, such as a maximum audio length of 3 seconds and a maximum number of frequencies that can be processed at once. It also appears to use a simplified version of the STS model that only performs audio normalization and does not perform any other processing.

The `SampleToSpeech` class appears to be responsible for loading an audio file and returning a spectrogram (a visualization of the frequency and amplitude of the audio). The `spec_to_spec` method maps an audio spectrogram to a spectrogram that represents the frequency and amplitude of the original audio. The `f0_to_spec` method maps a frequency to a spectrogram that represents the frequency and amplitude of the original audio.

The `slice_to_speech` method maps an audio slice to a spectrogram that represents the frequency and amplitude of the original audio. This method takes in an audio slice (a tuple of audio samples) and a maximum audio length. It returns a tuple of the spectrogram and the maximum frequency shift.

The `__getitem__` method is used to return an audio slice from the STS model. It takes in an audio index and returns the audio slice that corresponds to that index. If the STS model is all-in-memory, it will return the entire audio slice. Otherwise, it will return the audio slice for a specific audio path.

The `__len__` method is used to return the length of the STS model.

Note that this implementation is just one possible implementation of a STS model, and it may not be the most effective or efficient way to perform STS.


```py
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths, hparams, all_in_mem: bool = False, vol_aug: bool = True):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.hparams = hparams
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.unit_interpolate_mode = hparams.data.unit_interpolate_mode
        self.sampling_rate = hparams.data.sampling_rate
        self.use_sr = hparams.train.use_sr
        self.spec_len = hparams.train.max_speclen
        self.spk_map = hparams.spk
        self.vol_emb = hparams.model.vol_embedding
        self.vol_aug = hparams.train.vol_aug and vol_aug
        random.seed(1234)
        random.shuffle(self.audiopaths)
        
        self.all_in_mem = all_in_mem
        if self.all_in_mem:
            self.cache = [self.get_audio(p[0]) for p in self.audiopaths]

    def get_audio(self, filename):
        filename = filename.replace("\\", "/")
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "Sample Rate not match. Expect {} but got {} from {}".format(
                    self.sampling_rate, sampling_rate, filename))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")

        # Ideally, all data generated after Mar 25 should have .spec.pt
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                                     self.sampling_rate, self.hop_length, self.win_length,
                                     center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)

        spk = filename.split("/")[-2]
        spk = torch.LongTensor([self.spk_map[spk]])

        f0, uv = np.load(filename + ".f0.npy",allow_pickle=True)
        
        f0 = torch.FloatTensor(np.array(f0,dtype=float))
        uv = torch.FloatTensor(np.array(uv,dtype=float))

        c = torch.load(filename+ ".soft.pt")
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[0], mode=self.unit_interpolate_mode)
        if self.vol_emb:
            volume_path = filename + ".vol.npy"
            volume = np.load(volume_path)
            volume = torch.from_numpy(volume).float()
        else:
            volume = None

        lmin = min(c.size(-1), spec.size(-1))
        assert abs(c.size(-1) - spec.size(-1)) < 3, (c.size(-1), spec.size(-1), f0.shape, filename)
        assert abs(audio_norm.shape[1]-lmin * self.hop_length) < 3 * self.hop_length
        spec, c, f0, uv = spec[:, :lmin], c[:, :lmin], f0[:lmin], uv[:lmin]
        audio_norm = audio_norm[:, :lmin * self.hop_length]
        if volume is not None:
            volume = volume[:lmin]
        return c, f0, spec, audio_norm, spk, uv, volume

    def random_slice(self, c, f0, spec, audio_norm, spk, uv, volume):
        # if spec.shape[1] < 30:
        #     print("skip too short audio:", filename)
        #     return None

        if random.choice([True, False]) and self.vol_aug and volume is not None:
            max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
            max_shift = min(1, np.log10(1/max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)
            audio_norm = audio_norm * (10 ** log10_vol_shift)
            volume = volume * (10 ** log10_vol_shift)
            spec = spectrogram_torch(audio_norm,
            self.hparams.data.filter_length,
            self.hparams.data.sampling_rate,
            self.hparams.data.hop_length,
            self.hparams.data.win_length,
            center=False)[0]

        if spec.shape[1] > 800:
            start = random.randint(0, spec.shape[1]-800)
            end = start + 790
            spec, c, f0, uv = spec[:, start:end], c[:, start:end], f0[start:end], uv[start:end]
            audio_norm = audio_norm[:, start * self.hop_length : end * self.hop_length]
            if volume is not None:
                volume = volume[start:end]
        return c, f0, spec, audio_norm, spk, uv,volume

    def __getitem__(self, index):
        if self.all_in_mem:
            return self.random_slice(*self.cache[index])
        else:
            return self.random_slice(*self.get_audio(self.audiopaths[index][0]))

    def __len__(self):
        return len(self.audiopaths)


```

This is a function that takes in a batch of data, including the audio features (spec, f0, wav), and the corresponding metadata (c, length, spkid, uv). It paddens the data to a fixed size, wraps the data in a tensor to make it easier to index, and returns the padded data. The audio features are first converted to a tensor, with each spec, f0, wav, and spec having a shape of (batch\_size, n\_spec, n\_c, n\_h, n\_w). The metadata are also converted to a tensor, with each length, spkid, uv, and c having a shape of (batch\_size, n\_meta). The input batch is then passed through the function, and the output is a tensor with the same shape as the input.

Please note that this function is


```py
class TextAudioCollate:

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[1] for x in batch]),
            dim=0, descending=True)

        max_c_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])

        lengths = torch.LongTensor(len(batch))

        c_padded = torch.FloatTensor(len(batch), batch[0][0].shape[0], max_c_len)
        f0_padded = torch.FloatTensor(len(batch), max_c_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][2].shape[0], max_c_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        spkids = torch.LongTensor(len(batch), 1)
        uv_padded = torch.FloatTensor(len(batch), max_c_len)
        volume_padded = torch.FloatTensor(len(batch), max_c_len)

        c_padded.zero_()
        spec_padded.zero_()
        f0_padded.zero_()
        wav_padded.zero_()
        uv_padded.zero_()
        volume_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            c = row[0]
            c_padded[i, :, :c.size(1)] = c
            lengths[i] = c.size(1)

            f0 = row[1]
            f0_padded[i, :f0.size(0)] = f0

            spec = row[2]
            spec_padded[i, :, :spec.size(1)] = spec

            wav = row[3]
            wav_padded[i, :, :wav.size(1)] = wav

            spkids[i, 0] = row[4]

            uv = row[5]
            uv_padded[i, :uv.size(0)] = uv
            volume = row[6]
            if volume is not None:
                volume_padded[i, :volume.size(0)] = volume
            else :
                volume_padded = None
        return c_padded, f0_padded, spec_padded, wav_padded, spkids, lengths, uv_padded, volume_padded

```

# `export_index_for_onnx.py`

这段代码的作用是：

1. 导入操作系统（os）和Pickle库。
2. 导入 Faiss 库，以便用于搜索快速（但可能更慢）的点云。
3. 设置路径，指定要搜索的目录。
4. 指定索引文件保存的路径，该路径包含要索引的点云数据。
5. 创建一个名为 "checkpoints" 的目录，用于保存已经索引的点云数据。
6. 使用 "with" 语句打开 "feature_and_index.pkl" 文件，并将其中的内容读取到 Python 变量中。
7. 遍历保存的索引。
8. 将索引保存为 Faiss 库中的索引文件，并指定索引文件保存的路径。
9. 由于使用了 "with" 语句，这段代码将一直运行，直到程序被手动中断。


```py
import os
import pickle

import faiss

path = "crs"
indexs_file_path = f"checkpoints/{path}/feature_and_index.pkl"
indexs_out_dir = f"checkpoints/{path}/"

with open("feature_and_index.pkl",mode="rb") as f:
    indexs = pickle.load(f)

for k in indexs:
    print(f"Save {k} index")
    faiss.write_index(
        indexs[k],
        os.path.join(indexs_out_dir,f"Index-{k}.index")
    )

```

这段代码是在 Python 中打印字符串 "Saved all index"。它并没有做任何实际的计算或操作，而只是一个简单的字符串输出。这个字符串 "Saved all index" 会被打印到屏幕上，而不是被解释或执行任何有意义的操作。它的作用是告知屏幕它要输出什么内容。


```py
print("Saved all index")
```

# `flask_api.py`

这段代码使用了多个Python库，包括io、logging、soundfile、torch、torchaudio、Flask和CORS。它们的作用如下：

1. io库：提供了一个跨平台的I/O操作库，用于文件和字节流的操作。
2. logging库：提供了一个用于记录日志信息的库，可以用于输出调试信息以便调试程序。
3. soundfile库：一个Python库，用于处理音频文件。
4. torch库：一个PyTorch库，用于支持深度学习的计算任务。
5. torchaudio库：一个PyTorch库，用于录制和播放音频数据。
6. Flask库：一个Python web框架，提供了一个用于创建Web应用程序的工具。
7. Flask-CORS库：一个Python库，用于处理跨域请求。
8. inference.infer_tool库：一个用于进行实时语音识别（Inference）的库。

具体来说，这段代码实现了一个Web应用程序，可以接收实时音频流，并将其传递给inference.infer_tool库进行实时语音识别。应用程序通过Flask-CORS库与浏览器之间建立了一个跨域通信，使得浏览器可以在音频流上发送请求并接收实时流回。


```py
import io
import logging

import soundfile
import torch
import torchaudio
from flask import Flask, request, send_file
from flask_cors import CORS

from inference.infer_tool import RealTimeVC, Svc

app = Flask(__name__)

CORS(app)

```

在样本转调音频功能中，我们通过http获取wav文件并将其转换为该功能所需的模型推理。具体来说，我们首先从http请求中获取样本音频文件，然后通过http客户端下载wav文件。接着，我们为采样率设定值，并从http请求中获取自定义模型推理。我们根据设定的采样率，将模型推理的结果存储到内存中，并将其以wav格式存储。最后，我们通过函数计算出我们需要生成的音频文件，然后使用函数生成wav文件，并将文件保存到内存中。因此，函数实现的是：通过http请求下载样本音频，并生成模型推理生成的wav文件。


```py
logging.getLogger('numba').setLevel(logging.WARNING)


@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    request_form = request.form
    wave_file = request.files.get("sample", None)
    # 变调信息
    f_pitch_change = float(request_form.get("fPitchChange", 0))
    # DAW所需的采样率
    daw_sample = int(float(request_form.get("sampleRate", 0)))
    speaker_id = int(float(request_form.get("sSpeakId", 0)))
    # http获得wav文件并转换
    input_wav_path = io.BytesIO(wave_file.read())

    # 模型推理
    if raw_infer:
        # out_audio, out_sr = svc_model.infer(speaker_id, f_pitch_change, input_wav_path)
        out_audio, out_sr = svc_model.infer(speaker_id, f_pitch_change, input_wav_path, cluster_infer_ratio=0,
                                            auto_predict_f0=False, noice_scale=0.4, f0_filter=False)
        tar_audio = torchaudio.functional.resample(out_audio, svc_model.target_sample, daw_sample)
    else:
        out_audio = svc.process(svc_model, speaker_id, f_pitch_change, input_wav_path, cluster_infer_ratio=0,
                                auto_predict_f0=False, noice_scale=0.4, f0_filter=False)
        tar_audio = torchaudio.functional.resample(torch.from_numpy(out_audio), svc_model.target_sample, daw_sample)
    # 返回音频
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, tar_audio.cpu().numpy(), daw_sample, format="wav")
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


```

这段代码的作用是开启一个名为"__main__"的命名空间，如果当前进程是脚本进程，则执行一些操作来优化VST插件的性能。

首先，它设置了一个名为"raw_infer"的布尔值，表示启用直接切片合成而不是交叉淡化方式。

接下来，它指定了每个模型和配置文件的路径，并将一个名为"cluster_model_path"的路径指定为集群模型文件的路径。

然后，它创建了一个名为"svc"的服务对象，并指定了其模型和配置文件的路径。它还创建了一个名为"svc_model"的服务对象，并将其设置为使用前面创建的服务对象。

最后，它创建了一个名为"app"的Linux应用程序实例，并将其绑定到本地IP地址(0.0.0.0)的6842端口上，以便在后台运行。

综合来看，这段代码的主要目的是启用VST插件的性能优化，通过设置一些参数和创建一些服务对象来实现。


```py
if __name__ == '__main__':
    # 启用则为直接切片合成，False为交叉淡化方式
    # vst插件调整0.3-0.5s切片时间可以降低延迟，直接切片方法会有连接处爆音、交叉淡化会有轻微重叠声音
    # 自行选择能接受的方法，或将vst最大切片时间调整为1s，此处设为Ture，延迟大音质稳定一些
    raw_infer = True
    # 每个模型和config是唯一对应的
    model_name = "logs/32k/G_174000-Copy1.pth"
    config_name = "configs/config.json"
    cluster_model_path = "logs/44k/kmeans_10000.pt"
    svc_model = Svc(model_name, config_name, cluster_model_path=cluster_model_path)
    svc = RealTimeVC()
    # 此处与vst插件对应，不建议更改
    app.run(port=6842, host="0.0.0.0", debug=False, threaded=False)

```

# `flask_api_full_song.py`

This is a Python function that performs audio processing and segmentation on a given audio file. It can detect and remove empty segments andSave the output as a WAV file.

The function takes an audio file path and a number of options for audio processing and segmentation. The options include the audio format to use for segmentation, the threshold for the minimum detectable sound energy (in dB), and a maximum number of empty segments to allow before removing them.

The function first reads in the audio file and loops through each segment. If the segment is empty, it does not add any data to the output and continues to the next segment. Otherwise, it performs some processing on the audio data using the specified options and adds the processed audio data to the output.

Finally, the function saves the output as a WAV file using the `soundfile` library


```py
import io

import numpy as np
import soundfile
from flask import Flask, request, send_file

from inference import infer_tool, slicer

app = Flask(__name__)


@app.route("/wav2wav", methods=["POST"])
def wav2wav():
    request_form = request.form
    audio_path = request_form.get("audio_path", None)  # wav文件地址
    tran = int(float(request_form.get("tran", 0)))  # 音调
    spk = request_form.get("spk", 0)  # 说话人(id或者name都可以,具体看你的config)
    wav_format = request_form.get("wav_format", 'wav')  # 范围文件格式
    infer_tool.format_wav(audio_path)
    chunks = slicer.cut(audio_path, db_thresh=-40)
    audio_data, audio_sr = slicer.chunks2audio(audio_path, chunks)

    audio = []
    for (slice_tag, data) in audio_data:
        print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')

        length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
        if slice_tag:
            print('jump empty segment')
            _audio = np.zeros(length)
        else:
            # padd
            pad_len = int(audio_sr * 0.5)
            data = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])])
            raw_path = io.BytesIO()
            soundfile.write(raw_path, data, audio_sr, format="wav")
            raw_path.seek(0)
            out_audio, out_sr = svc_model.infer(spk, tran, raw_path)
            svc_model.clear_empty()
            _audio = out_audio.cpu().numpy()
            pad_len = int(svc_model.target_sample * 0.5)
            _audio = _audio[pad_len:-pad_len]

        audio.extend(list(infer_tool.pad_array(_audio, length)))
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, audio, svc_model.target_sample, format=wav_format)
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name=f"temp.{wav_format}", as_attachment=True)


```

这段代码的作用是创建一个名为"logs/44k/G_60000.pth"的模型和名为"configs/config.json"的配置文件，然后使用"infer_tool.Svc"类创建一个名为"svc_model"的服务器模型，并将模型和配置文件指定为"model_name"和"config_name"，然后将服务器模型命名为"svc_model"，最后在应用程序的主进程中运行该应用程序，将服务器监听在端口1145上，并将主机设置为"0.0.0.0"，应用程序将使用"debug"参数输出调试信息，并使用"threaded"参数使应用程序具有多线程特性。


```py
if __name__ == '__main__':
    model_name = "logs/44k/G_60000.pth"  # 模型地址
    config_name = "configs/config.json"  # config地址
    svc_model = infer_tool.Svc(model_name, config_name)
    app.run(port=1145, host="0.0.0.0", debug=False, threaded=False)

```

# `inference_main.py`

This appears to be a Python script that uses a deep learning model for speech separation and merge. It takes in a list of speaker names, a raw audio path, a text parameter for specifying the speech segmentation and a text parameter for specifying the cluster number.

It then loops through each speaker and uses the specified model to predict the cluster number and audio quality. It then saves the results, including the audio, to a file.

The script also appears to use a number of other parameters, such as the number of seconds to allow for processing, the maximum number of clusters to predict, and the type of diffusion to use. It also seems to use a custom key for specifying the cluster number, with underscores for the first word and numbers for any subsequent words.


```py
import logging

import soundfile

from inference import infer_tool
from inference.infer_tool import Svc
from spkmix import spk_mix_map

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")



def main():
    import argparse

    parser = argparse.ArgumentParser(description='sovits4 inference')

    # 一定要设置的部分
    parser.add_argument('-m', '--model_path', type=str, default="logs/44k/G_37600.pth", help='模型路径')
    parser.add_argument('-c', '--config_path', type=str, default="logs/44k/config.json", help='配置文件路径')
    parser.add_argument('-cl', '--clip', type=float, default=0, help='音频强制切片，默认0为自动切片，单位为秒/s')
    parser.add_argument('-n', '--clean_names', type=str, nargs='+', default=["君の知らない物語-src.wav"], help='wav文件名列表，放在raw文件夹下')
    parser.add_argument('-t', '--trans', type=int, nargs='+', default=[0], help='音高调整，支持正负（半音）')
    parser.add_argument('-s', '--spk_list', type=str, nargs='+', default=['buyizi'], help='合成目标说话人名称')
    
    # 可选项部分
    parser.add_argument('-a', '--auto_predict_f0', action='store_true', default=False, help='语音转换自动预测音高，转换歌声时不要打开这个会严重跑调')
    parser.add_argument('-cm', '--cluster_model_path', type=str, default="", help='聚类模型或特征检索索引路径，留空则自动设为各方案模型的默认路径，如果没有训练聚类或特征检索则随便填')
    parser.add_argument('-cr', '--cluster_infer_ratio', type=float, default=0, help='聚类方案或特征检索占比，范围0-1，若没有训练聚类模型或特征检索则默认0即可')
    parser.add_argument('-lg', '--linear_gradient', type=float, default=0, help='两段音频切片的交叉淡入长度，如果强制切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，单位为秒')
    parser.add_argument('-f0p', '--f0_predictor', type=str, default="pm", help='选择F0预测器,可选择crepe,pm,dio,harvest,rmvpe,fcpe默认为pm(注意：crepe为原F0使用均值滤波器)')
    parser.add_argument('-eh', '--enhance', action='store_true', default=False, help='是否使用NSF_HIFIGAN增强器,该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭')
    parser.add_argument('-shd', '--shallow_diffusion', action='store_true', default=False, help='是否使用浅层扩散，使用后可解决一部分电音问题，默认关闭，该选项打开时，NSF_HIFIGAN增强器将会被禁止')
    parser.add_argument('-usm', '--use_spk_mix', action='store_true', default=False, help='是否使用角色融合')
    parser.add_argument('-lea', '--loudness_envelope_adjustment', type=float, default=1, help='输入源响度包络替换输出响度包络融合比例，越靠近1越使用输出响度包络')
    parser.add_argument('-fr', '--feature_retrieval', action='store_true', default=False, help='是否使用特征检索，如果使用聚类模型将被禁用，且cm与cr参数将会变成特征检索的索引路径与混合比例')

    # 浅扩散设置
    parser.add_argument('-dm', '--diffusion_model_path', type=str, default="logs/44k/diffusion/model_0.pt", help='扩散模型路径')
    parser.add_argument('-dc', '--diffusion_config_path', type=str, default="logs/44k/diffusion/config.yaml", help='扩散模型配置文件路径')
    parser.add_argument('-ks', '--k_step', type=int, default=100, help='扩散步数，越大越接近扩散模型的结果，默认100')
    parser.add_argument('-se', '--second_encoding', action='store_true', default=False, help='二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，有时候效果好，有时候效果差')
    parser.add_argument('-od', '--only_diffusion', action='store_true', default=False, help='纯扩散模式，该模式不会加载sovits模型，以扩散模型推理')
    

    # 不用动的部分
    parser.add_argument('-sd', '--slice_db', type=int, default=-40, help='默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50')
    parser.add_argument('-d', '--device', type=str, default=None, help='推理设备，None则为自动选择cpu和gpu')
    parser.add_argument('-ns', '--noice_scale', type=float, default=0.4, help='噪音级别，会影响咬字和音质，较为玄学')
    parser.add_argument('-p', '--pad_seconds', type=float, default=0.5, help='推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现')
    parser.add_argument('-wf', '--wav_format', type=str, default='flac', help='音频输出格式')
    parser.add_argument('-lgr', '--linear_gradient_retain', type=float, default=0.75, help='自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭')
    parser.add_argument('-eak', '--enhancer_adaptive_key', type=int, default=0, help='使增强器适应更高的音域(单位为半音数)|默认为0')
    parser.add_argument('-ft', '--f0_filter_threshold', type=float, default=0.05,help='F0过滤阈值，只有使用crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音')


    args = parser.parse_args()

    clean_names = args.clean_names
    trans = args.trans
    spk_list = args.spk_list
    slice_db = args.slice_db
    wav_format = args.wav_format
    auto_predict_f0 = args.auto_predict_f0
    cluster_infer_ratio = args.cluster_infer_ratio
    noice_scale = args.noice_scale
    pad_seconds = args.pad_seconds
    clip = args.clip
    lg = args.linear_gradient
    lgr = args.linear_gradient_retain
    f0p = args.f0_predictor
    enhance = args.enhance
    enhancer_adaptive_key = args.enhancer_adaptive_key
    cr_threshold = args.f0_filter_threshold
    diffusion_model_path = args.diffusion_model_path
    diffusion_config_path = args.diffusion_config_path
    k_step = args.k_step
    only_diffusion = args.only_diffusion
    shallow_diffusion = args.shallow_diffusion
    use_spk_mix = args.use_spk_mix
    second_encoding = args.second_encoding
    loudness_envelope_adjustment = args.loudness_envelope_adjustment

    if cluster_infer_ratio != 0:
        if args.cluster_model_path == "":
            if args.feature_retrieval:  # 若指定了占比但没有指定模型路径，则按是否使用特征检索分配默认的模型路径
                args.cluster_model_path = "logs/44k/feature_and_index.pkl"
            else:
                args.cluster_model_path = "logs/44k/kmeans_10000.pt"
    else:  # 若未指定占比，则无论是否指定模型路径，都将其置空以避免之后的模型加载
        args.cluster_model_path = ""

    svc_model = Svc(args.model_path,
                    args.config_path,
                    args.device,
                    args.cluster_model_path,
                    enhance,
                    diffusion_model_path,
                    diffusion_config_path,
                    shallow_diffusion,
                    only_diffusion,
                    use_spk_mix,
                    args.feature_retrieval)
    
    infer_tool.mkdir(["raw", "results"])
    
    if len(spk_mix_map)<=1:
        use_spk_mix = False
    if use_spk_mix:
        spk_list = [spk_mix_map]
    
    infer_tool.fill_a_to_b(trans, clean_names)
    for clean_name, tran in zip(clean_names, trans):
        raw_audio_path = f"raw/{clean_name}"
        if "." not in raw_audio_path:
            raw_audio_path += ".wav"
        infer_tool.format_wav(raw_audio_path)
        for spk in spk_list:
            kwarg = {
                "raw_audio_path" : raw_audio_path,
                "spk" : spk,
                "tran" : tran,
                "slice_db" : slice_db,
                "cluster_infer_ratio" : cluster_infer_ratio,
                "auto_predict_f0" : auto_predict_f0,
                "noice_scale" : noice_scale,
                "pad_seconds" : pad_seconds,
                "clip_seconds" : clip,
                "lg_num": lg,
                "lgr_num" : lgr,
                "f0_predictor" : f0p,
                "enhancer_adaptive_key" : enhancer_adaptive_key,
                "cr_threshold" : cr_threshold,
                "k_step":k_step,
                "use_spk_mix":use_spk_mix,
                "second_encoding":second_encoding,
                "loudness_envelope_adjustment":loudness_envelope_adjustment
            }
            audio = svc_model.slice_inference(**kwarg)
            key = "auto" if auto_predict_f0 else f"{tran}key"
            cluster_name = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"
            isdiffusion = "sovits"
            if shallow_diffusion :
                isdiffusion = "sovdiff"
            if only_diffusion :
                isdiffusion = "diff"
            if use_spk_mix:
                spk = "spk_mix"
            res_path = f'results/{clean_name}_{key}_{spk}{cluster_name}_{isdiffusion}_{f0p}.{wav_format}'
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
            svc_model.clear_empty()
            
```

这段代码是一个if语句，它会判断当前脚本是否作为主程序运行。如果当前脚本作为主程序运行，那么程序会执行if语句中的代码块。

在这个if语句中，另外还有一条语句：

```py
if __name__ == '__main__':
   main()
```

这条语句是判断当前脚本是否作为主程序运行。如果当前脚本作为主程序运行，那么程序会执行if语句中的代码块。如果当前脚本不是主程序运行，那么这条语句不会被执行。

因此，这段代码的作用是判断当前脚本是否作为主程序运行，如果是，则执行if语句中的代码块，否则不执行。


```py
if __name__ == '__main__':
    main()

```

# `models.py`

This is a class that defines a `ResidualCouplingBlock` module for a neural network.

It has the following attributes:

* `channels`: The number of input channels.
* `hidden_channels`: The number of output channels.
* `kernel_size`: The kernel size of the convolutional neural network.
* `dilation_rate`: The dilation rate used in the convolutional neural network.
* `n_layers`: The number of convolutional layers in the network.
* `n_flows`: The number of attention flows in the network.
* `gin_channels`: The number of groups in the attention mechanism.
* `share_parameter`: A boolean indicating whether to share parameters between the attention mechanism and the network.

It also has the following methods:

* `__init__`: Initializes the module and sets the parameters.
* `forward`: Returns the forward pass of the module. This includes the computation of the output feature map and the attention computation.

The forward method的具体 implementation is not provided in the class definition, but it contains the computation of the residual connection and the attention computation.


```py
import torch
from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

import modules.attentions as attentions
import modules.commons as commons
import modules.modules as modules
import utils
from modules.commons import get_padding
from utils import f0_to_coarse


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0,
                 share_parameter=False
                 ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=gin_channels) if share_parameter else None

        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                                              gin_channels=gin_channels, mean_only=True, wn_sharing_parameter=self.wn))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

```

这段代码定义了一个名为“TransformerCouplingBlock”的类，它继承自PyTorch中的nn.Module类。这个类在__init__方法中定义了一些参数，包括输入通道、隐藏通道、过滤通道、头数、层数、内核大小、Pdropout概率、流数和管道参数。

在TransformerCouplingBlock类中，还定义了一个复杂的 forward 方法。在这个方法中，首先调用了一个自定义的FFT类，这个类自定义了计算注意力。然后，对于每个流，调用了一个TransformerCouplingLayer类和一个Flip类。TransformerCouplingLayer类实现了对输入序列的注意力，而Flip类则实现了对输入通道的翻转。

总的来说，这段代码定义了一个用于实现Transformer模型中的coupling block的类，通过不同的组合方式可以实现多种数据流之间的相关性，从而提高模型的表现。


```py
class TransformerCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 n_flows=4,
                 gin_channels=0,
                 share_parameter=False
                 ):
            
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = attentions.FFT(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, isflow = True, gin_channels = self.gin_channels) if share_parameter else None

        for i in range(n_flows):
            self.flows.append(
                modules.TransformerCouplingLayer(channels, hidden_channels, kernel_size, n_layers, n_heads, p_dropout, filter_channels, mean_only=True, wn_sharing_parameter=self.wn, gin_channels = self.gin_channels))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


```

这段代码定义了一个名为 "Encoder" 的类，继承自 PyTorch 的nn.Module类。这个类用于实现一个具有隐层神经网络的编码器，可以对输入数据进行处理并输出编码结果。

在 Encoder 的构造函数中，我们定义了模型的输入和输出通道、隐藏层神经网络的参数，包括通道数、内核大小、 dilation 率、层数以及 gin 通道（使用 conditionally Promise 中的 `gin_channels` 参数）。

在 forward 方法中，我们首先对输入数据 x 应用卷积操作，然后将结果与输入中的位置掩码（使用序列掩码 `commons.sequence_mask`）进行乘积操作，将结果中的位置信息用于对输入数据进行遮盖。

接着，我们对输入数据 x 应用一系列的隐层神经网络模块，并将输出结果与每个模块中的注意力权重（使用 gin 参数）和维度信息（使用 layer 参数）一起拼接起来，再将拼接后的结果输出。

最后，我们对输出的编码结果进行处理，包括对结果应用条件 fc 层以及计算统计信息等。


```py
class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        # print(x.shape,x_lengths.shape)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


```

这段代码定义了一个名为 TextEncoder 的类，它继承自 PyTorch 中的 nn.Module 类。这个类负责将输入序列（通常是文本数据）编码成具有一定语义性的表示，以便用于文本分类任务。

TextEncoder 类包含了一个 __init__ 方法，用于设置编码器的基本参数，包括输出通道、隐藏通道、卷积核大小、层数、自注意力（S注意力）模块中的 key_channels、value_channels，以及 dropout 概率。这些参数会在实例化 TextEncoder 时设置，并对其行为产生影响。

TextEncoder 类还包含一个 forward 方法。这个方法接收两个输入：当前的输入序列 x 和一个掩码（用于指示哪些位置是可用于编码的）。forward 方法首先将 x 和掩码转换为输入序列中的第一组数据，然后将其输入到 self.enc_ 中进行编码。在编码的过程中，通过 self.f0_emb 嵌入一个随机的注意力权重 f0，以便在编码过程中捕获输入序列中的关键信息。

self.f0_emb 是一个包含 f0 大小的 Embedding，其大小为 256。通过这个 Embedding，可以方便地计算 f0 注意力。

接下来，将生成的编码结果统计信息（如注意力权重、可用的掩码等）返回给输入序列，并返回编码结果。


```py
class TextEncoder(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 n_layers,
                 gin_channels=0,
                 filter_channels=None,
                 n_heads=None,
                 p_dropout=None):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.f0_emb = nn.Embedding(256, hidden_channels)

        self.enc_ = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)

    def forward(self, x, x_mask, f0=None, noice_scale=1):
        x = x + self.f0_emb(f0).transpose(1, 2)
        x = self.enc_(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs) * noice_scale) * x_mask

        return z, m, logs, x_mask


```

This is a custom implementation of a U-Net architecture for image classification task. U-Net is a type of architecture that uses a self-attention mechanism to integrate different feature levels of an image.

The `Ueconv` class defines the forward pass of the U-Net. It takes a 2D feature tensor `x`, a list of 2D feature maps `fmap`, and a kernel size `kernel_size`, a padding parameter `stride`, and a scaling parameter `scale`. It applies a convolution operation to each feature map, and applies a leaky-relu activation function to each output of the convolution. It also applies a scaling operation to the input, which is based on the value of the `scale` parameter.

The `norm_f` function is used to normalize the output of each feature map.

The `forward` method is the main method that builds and applies the U-Net to a given input. It takes an input tensor `x` and returns the output tensor. It initializes a list `fmap_list` to a vector of 2D feature maps. Then it loops over each feature map and applies the convolution and normalization operations.

It then applies the `norm_f` function to each feature map to normalize it.

After the loop is finished, it returns the output tensor.


```py
class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
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
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


```

这段代码定义了一个名为 DiscriminatorS 的类，继承自 PyTorch 中的 torch.nn.Module 类。该类在__init__ 方法中进行初始化，并定义了一个数据流图（input graph）和输出数据流图（output graph）。

在 DiscriminatorS 的 forward 方法中，首先定义了一个变量 fmap，用于记录输入数据在经过一系列卷积操作后产生的特征映射。然后，遍历输入数据流图中的每个卷积层，对每个卷积层执行以下操作：

1. 对输入数据进行 F.leaky_relu 激活函数，应用一个 slope 参数。
2. 将激活后的输入数据输入到卷积层中，执行一次卷积操作。
3. 对卷积后的输入数据进行传播，得到一个张量。
4. 将卷积层输出的张量（即 fmap）与最后一个卷积层的输出连接起来，得到一个完整的输入数据流图。
5. 将输入数据流图、输出数据流图以及 fmap 一并打印出来。

这段代码的作用是定义了一个用于训练卷积神经网络（CNN）的数据流图，该数据流图包含了输入数据、卷积层、池化层和全连接层的输出。通过使用这种数据流图，可以使得训练任务更加模块化，且能够更好地捕捉数据在网络中的传播过程。


```py
class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


```

这段代码定义了一个名为 MultiPeriodDiscriminator 的类，继承自 PyTorch 的nn.Module类。这个类在__init__方法中创建了一个包含多个Discriminator的列表，每个Discriminator都包含一个使用Spectral Normalization的激活函数。

在forward方法中，该类实现了两个函数，用于前向传播输入数据 y 和 y_hat 到输出数据。对于每个输入样本，该类返回一个包含两个输出数据的列表，以及一个包含四个输出数据的列表。这些输出数据分别是Discriminator的输出数据，用于计算输出标签。

具体来说，对于每个Discriminator，该类将其传入的输入数据与相应的激活函数一起传递给输出。对于每个Discriminator，该类返回一个包含了两个输出数据：一个是输入数据y和另一个是预测数据y_hat的标签，另一个是四个输出数据，分别是Discriminator的输出数据。这些输出数据被存储在两个PyTorch的ModuleList中，分别名为 y_d_rs 和 fmap_rs。


```py
class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


```

This is a PyTorch implementation of an LSTM-based text encoder model for speech recognition. The model takes a mel spectrogram as input and outputs a synthesized embed.

The model has a LSTM layer with a hidden size of `model_hidden_size` and a num of layers of `model_num_layers`. The LSTM layer is initialized with a mel spectrogram that has `mel_n_channels` channels and zero initial values.

The model also has a linear layer with `model_embedding_size` channels, which is used to get the mel spectrogram embedding in the input space. This linear layer is initialized with zero and applies the ReLU activation function.

The `forward` method applies the mel spectrogram to the LSTM layer, and it returns the output embedding.

The `compute_partial_slices` method computes the partial slices of the mel spectrogram that correspond to the input range and the partial hop size. It returns a list of mel slices.

The `embed_utterance` method takes a mel spectrogram and a partial hop size as input and returns the synthesized embed. It uses the `compute_partial_slices` method to compute the partial slices of the mel spectrogram, and it applies the ReLU activation function to the last mel slice of the input embed.

Note that in the code, the `batch_first=True` parameter is set to `True`, which means that the input embes are generated for each input sequence in the batch, rather than for each sample. This is done to increase the computational efficiency of the model.


```py
class SpeakerEncoder(torch.nn.Module):
    def __init__(self, mel_n_channels=80, model_num_layers=3, model_hidden_size=256, model_embedding_size=256):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        mel_slices = []
        for i in range(0, total_frames - partial_frames, partial_hop):
            mel_range = torch.arange(i, i + partial_frames)
            mel_slices.append(mel_range)

        return mel_slices

    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:, -partial_frames:]

        if mel_len > partial_frames:
            mel_slices = self.compute_partial_slices(mel_len, partial_frames, partial_hop)
            mels = list(mel[:, s] for s in mel_slices)
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)

            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
            # embed = embed / torch.linalg.norm(embed, 2)
        else:
            with torch.no_grad():
                embed = self(last_mel)

        return embed

```

这段代码定义了一个名为F0Decoder的类，该类继承自PyTorch中的nn.Module类。F0Decoder类包含了一个数据预处理和一个编码器部分。

在初始化函数__init__中，F0Decoder类指定了一系列参数，包括输出通道、隐藏通道、滤波器通道、头数、层数、卷积核大小、点概率dropout和语音通道数量等。这些参数对于训练和推理过程都很重要。

F0Decoder类中包含一个前馈网络（Prenet）和一个编码器（Decoder），这两个网络都是根据输入的隐藏状态来学习的。在前馈网络中，使用了一个大小为hidden\_channels的3x卷积核，并应用了点概率dropout。在编码器中，使用了输入通道和隐藏状态，以及一个大小为hidden\_channels的1x卷积核来编码输入数据。同时，编码器还使用了一个大小为hidden\_channels的FFT卷积核，来处理音频数据。

F0Decoder类中还包含一个数据依赖函数forward，该函数将输入数据和相应的位置掩码（如果有的话）作为参数，返回编码器的输出。通过调用forward函数，可以在训练和推理过程中使用F0Decoder类来处理音频和视频数据。


```py
class F0Decoder(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 spk_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.spk_channels = spk_channels

        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = attentions.FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def forward(self, x, norm_f0, x_mask, spk_emb=None):
        x = torch.detach(x)
        if (spk_emb is not None):
            x = x + self.cond(spk_emb)
        x += self.f0_prenet(norm_f0)
        x = self.prenet(x) * x_mask
        x = self.decoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask
        return x


```

This is a PyTorch implementation of a neural network model for text classification tasks. The model takes a sequence of words or sentences as input, represented as a tensor of integers, and outputs a probability distribution over the labels.

The model has an embedding layer with a size of 768, a position encoder, and a linear layer with 10 output classes. The linear layer has two sub-layers, one for the input embeddings and the other for the softmax function.

The pre-processing step is an auto-fitting of the model parameters. The flowchart of the model shows the input through the different layers of the model and the output.

It should be noted that this model is based on a simple, unidirectional neural network and it may not be able to handle some complex problems that other models like. It is also intended for use in small-scale text classification tasks and the architecture may not be efficient for larger datasets.


```py
class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 gin_channels,
                 ssl_dim,
                 n_speakers,
                 sampling_rate=44100,
                 vol_embedding=False,
                 vocoder_name = "nsf-hifigan",
                 use_depthwise_conv = False,
                 use_automatic_f0_prediction = True,
                 flow_share_parameter = False,
                 n_flow_layer = 4,
                 n_layers_trans_flow = 3,
                 use_transformer_flow = False,
                 **kwargs):

        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.ssl_dim = ssl_dim
        self.vol_embedding = vol_embedding
        self.emb_g = nn.Embedding(n_speakers, gin_channels)
        self.use_depthwise_conv = use_depthwise_conv
        self.use_automatic_f0_prediction = use_automatic_f0_prediction
        self.n_layers_trans_flow = n_layers_trans_flow
        if vol_embedding:
           self.emb_vol = nn.Linear(1, hidden_channels)

        self.pre = nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout
        )
        hps = {
            "sampling_rate": sampling_rate,
            "inter_channels": inter_channels,
            "resblock": resblock,
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "resblock_dilation_sizes": resblock_dilation_sizes,
            "upsample_rates": upsample_rates,
            "upsample_initial_channel": upsample_initial_channel,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "gin_channels": gin_channels,
            "use_depthwise_conv":use_depthwise_conv
        }
        
        modules.set_Conv1dModel(self.use_depthwise_conv)

        if vocoder_name == "nsf-hifigan":
            from vdecoder.hifigan.models import Generator
            self.dec = Generator(h=hps)
        elif vocoder_name == "nsf-snake-hifigan":
            from vdecoder.hifiganwithsnake.models import Generator
            self.dec = Generator(h=hps)
        else:
            print("[?] Unkown vocoder: use default(nsf-hifigan)")
            from vdecoder.hifigan.models import Generator
            self.dec = Generator(h=hps)

        self.enc_q = Encoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        if use_transformer_flow:
            self.flow = TransformerCouplingBlock(inter_channels, hidden_channels, filter_channels, n_heads, n_layers_trans_flow, 5, p_dropout, n_flow_layer,  gin_channels=gin_channels, share_parameter= flow_share_parameter)
        else:
            self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, n_flow_layer, gin_channels=gin_channels, share_parameter= flow_share_parameter)
        if self.use_automatic_f0_prediction:
            self.f0_decoder = F0Decoder(
                1,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                spk_channels=gin_channels
            )
        self.emb_uv = nn.Embedding(2, hidden_channels)
        self.character_mix = False

    def EnableCharacterMix(self, n_speakers_map, device):
        self.speaker_map = torch.zeros((n_speakers_map, 1, 1, self.gin_channels)).to(device)
        for i in range(n_speakers_map):
            self.speaker_map[i] = self.emb_g(torch.LongTensor([[i]]).to(device))
        self.speaker_map = self.speaker_map.unsqueeze(0).to(device)
        self.character_mix = True

    def forward(self, c, f0, uv, spec, g=None, c_lengths=None, spec_lengths=None, vol = None):
        g = self.emb_g(g).transpose(1,2)

        # vol proj
        vol = self.emb_vol(vol[:,:,None]).transpose(1,2) if vol is not None and self.vol_embedding else 0

        # ssl prenet
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1,2) + vol
        
        # f0 predict
        if self.use_automatic_f0_prediction:
            lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
            norm_lf0 = utils.normalize_f0(lf0, x_mask, uv)
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
        else:
            lf0 = 0
            norm_lf0 = 0
            pred_lf0 = 0
        # encoder
        z_ptemp, m_p, logs_p, _ = self.enc_p(x, x_mask, f0=f0_to_coarse(f0))
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)

        # flow
        z_p = self.flow(z, spec_mask, g=g)
        z_slice, pitch_slice, ids_slice = commons.rand_slice_segments_with_pitch(z, f0, spec_lengths, self.segment_size)

        # nsf decoder
        o = self.dec(z_slice, g=g, f0=pitch_slice)

        return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0

    @torch.no_grad()
    def infer(self, c, f0, uv, g=None, noice_scale=0.35, seed=52468, predict_f0=False, vol = None):

        if c.device == torch.device("cuda"):
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)

        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

        if self.character_mix and len(g) > 1:   # [N, S]  *  [S, B, 1, H]
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
            g = g * self.speaker_map  # [N, S, B, 1, H]
            g = torch.sum(g, dim=1) # [N, 1, B, 1, H]
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0) # [B, H, N]
        else:
            if g.dim() == 1:
                g = g.unsqueeze(0)
            g = self.emb_g(g).transpose(1, 2)
        
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        # vol proj
        
        vol = self.emb_vol(vol[:,:,None]).transpose(1,2) if vol is not None and self.vol_embedding else 0

        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol

        
        if self.use_automatic_f0_prediction and predict_f0:
            lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
            norm_lf0 = utils.normalize_f0(lf0, x_mask, uv, random_scale=False)
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
            f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)
        
        z_p, m_p, logs_p, c_mask = self.enc_p(x, x_mask, f0=f0_to_coarse(f0), noice_scale=noice_scale)
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g, f0=f0)
        return o,f0


```

# `onnx_export.py`

It looks like this is a Python script that uses the ONNX、ONNX-ACCESS 和 ONNX-PUSH function names to create an ONNX model from a set of input TensorFlow images and export it to a file named "model.onnx".

The script has a number of functions that are used to convert the input TensorFlow images to the ONNX format. These functions include `to_device()`, `batch_size()`, `dim_names()`, `export_mix()`, `features_names()`, and `hidden_size_from_channel()`.

It appears that the script is using a pre-trained ONNX model ("SoVities") that has been trained on a set of image and audio data. It is using the knowledge engineer's template to specify the input images and audio that the model should be used on, and then exporting the model to a file.

It is also using the `torch.onnx.export()` function to export the model to a file named "model.onnx". This function takes as input the ONNX model and the device that the model should be run on, and returns a file path where the model can be saved.

It looks like the script is using a number of different devices, including a device with the device type "cuda" and a device with the device type "cpu". It is using the `device()` function to specify which device to use for each function that needs to be performed, such as `test_inputs()` and `test_sid()`.

Overall, it appears that this script is a tool for converting a set of input TensorFlow images to the ONNX format, and then exporting a pre-trained ONNX model that has been trained on those images to a file named "model.onnx".


```py
import argparse
import json

import torch

import utils
from onnxexport.model_onnx_speaker_mix import SynthesizerTrn

parser = argparse.ArgumentParser(description='SoVitsSvc OnnxExport')

def OnnxExport(path=None):
    device = torch.device("cpu")
    hps = utils.get_hparams_from_file(f"checkpoints/{path}/config.json")
    SVCVITS = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    _ = utils.load_checkpoint(f"checkpoints/{path}/model.pth", SVCVITS, None)
    _ = SVCVITS.eval().to(device)
    for i in SVCVITS.parameters():
        i.requires_grad = False
    
    num_frames = 200

    test_hidden_unit = torch.rand(1, num_frames, SVCVITS.gin_channels)
    test_pitch = torch.rand(1, num_frames)
    test_vol = torch.rand(1, num_frames)
    test_mel2ph = torch.LongTensor(torch.arange(0, num_frames)).unsqueeze(0)
    test_uv = torch.ones(1, num_frames, dtype=torch.float32)
    test_noise = torch.randn(1, 192, num_frames)
    test_sid = torch.LongTensor([0])
    export_mix = True
    if len(hps.spk) < 2:
        export_mix = False
    
    if export_mix:
        spk_mix = []
        n_spk = len(hps.spk)
        for i in range(n_spk):
            spk_mix.append(1.0/float(n_spk))
        test_sid = torch.tensor(spk_mix)
        SVCVITS.export_chara_mix(hps.spk)
        test_sid = test_sid.unsqueeze(0)
        test_sid = test_sid.repeat(num_frames, 1)
    
    SVCVITS.eval()

    if export_mix:
        daxes = {
            "c": [0, 1],
            "f0": [1],
            "mel2ph": [1],
            "uv": [1],
            "noise": [2],
            "sid":[0]
        }
    else:
        daxes = {
            "c": [0, 1],
            "f0": [1],
            "mel2ph": [1],
            "uv": [1],
            "noise": [2]
        }
    
    input_names = ["c", "f0", "mel2ph", "uv", "noise", "sid"]
    output_names = ["audio", ]

    if SVCVITS.vol_embedding:
        input_names.append("vol")
        vol_dadict = {"vol" : [1]}
        daxes.update(vol_dadict)
        test_inputs = (
            test_hidden_unit.to(device),
            test_pitch.to(device),
            test_mel2ph.to(device),
            test_uv.to(device),
            test_noise.to(device),
            test_sid.to(device),
            test_vol.to(device)
        )
    else:
        test_inputs = (
            test_hidden_unit.to(device),
            test_pitch.to(device),
            test_mel2ph.to(device),
            test_uv.to(device),
            test_noise.to(device),
            test_sid.to(device)
        )

    # SVCVITS = torch.jit.script(SVCVITS)
    SVCVITS(test_hidden_unit.to(device),
            test_pitch.to(device),
            test_mel2ph.to(device),
            test_uv.to(device),
            test_noise.to(device),
            test_sid.to(device),
            test_vol.to(device))

    SVCVITS.dec.OnnxExport()

    torch.onnx.export(
        SVCVITS,
        test_inputs,
        f"checkpoints/{path}/{path}_SoVits.onnx",
        dynamic_axes=daxes,
        do_constant_folding=False,
        opset_version=16,
        verbose=False,
        input_names=input_names,
        output_names=output_names
    )

    vec_lay = "layer-12" if SVCVITS.gin_channels == 768 else "layer-9"
    spklist = []
    for key in hps.spk.keys():
        spklist.append(key)

    MoeVSConf = {
        "Folder" : f"{path}",
        "Name" : f"{path}",
        "Type" : "SoVits",
        "Rate" : hps.data.sampling_rate,
        "Hop" : hps.data.hop_length,
        "Hubert": f"vec-{SVCVITS.gin_channels}-{vec_lay}",
        "SoVits4": True,
        "SoVits3": False,
        "CharaMix": export_mix,
        "Volume": SVCVITS.vol_embedding,
        "HiddenSize": SVCVITS.gin_channels,
        "Characters": spklist,
        "Cluster": ""
    }

    with open(f"checkpoints/{path}.json", 'w') as MoeVsConfFile:
        json.dump(MoeVSConf, MoeVsConfFile, indent = 4)


```

这段代码是一个Python脚本，它定义了一个if条件，只有在运行脚本时才会执行if条件内的代码。

if __name__ == '__main__':：
   parser = argparse.ArgumentParser(description='Argument parser for training and exporting a model in ONNX format')
   parser.add_argument('-n', '--model_name', type=str, default="TransformerFlow", help='模型文件夹名（根目录下新建ckeckpoints文件夹，在此文件夹下建立一个新的文件夹，放置模型，该文件夹名即为此项）')
   args = parser.parse_args()
   path = args.model_name
   OnnxExport(path)
```py

这段代码的作用是创建一个ArgumentParser对象，用于解析用户提供的命令行参数。它添加了一个名为“-n”的参数（--model_name），它的类型是字符串，默认为“TransformerFlow”。该参数是一个必需的参数，如果没有提供，程序将会崩溃。

接下来，它使用args.parse_args()方法解析用户提供的命令行参数，并把args对象的值赋给变量path。然后，它调用了一个名为"OnnxExport"的函数，将path变量作为参数传给该函数，该函数可能将模型文件夹 export 成 ONNX 格式。


```
if __name__ == '__main__':
    parser.add_argument('-n', '--model_name', type=str, default="TransformerFlow", help='模型文件夹名（根目录下新建ckeckpoints文件夹，在此文件夹下建立一个新的文件夹，放置模型，该文件夹名即为此项）')
    args = parser.parse_args()
    path = args.model_name
    OnnxExport(path)

```