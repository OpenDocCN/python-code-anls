# BertVITS2源码解析 2

# `preprocess_text.py`

这段代码使用了多个库函数，其中包括 `import json` 用于导入 JSON 数据格式，`from collections import defaultdict` 用于创建字典列表，`from random import shuffle` 用于生成随机整数，`from typing import Optional` 用于定义可选项，`from tqdm import tqdm` 用于使用 ASCII 模式制控制符中的 `tqdm` 函数，`import click` 用于导入 `click` 库，`from text.cleaner import clean_text` 用于从文本文件中提取干净的文本内容。

这段代码的作用是实现了一个命令行工具，用于将指定的文本文件中的一行文本每行进行转录，并对转录结果进行去噪处理。具体实现过程如下：

1. 读取指定路径的文本文件，并导入 JSON 数据格式文件中的数据。
2. 导入 `text.cleaner` 库中的 `clean_text` 函数，对文本进行去噪处理。
3. 使用 `tqdm` 函数控制转录过程，并对转录结果进行汇总输出。
4. 运行程序时，使用 `--transcription-path` 选项指定要转录的文本文件路径，如果没有指定，则默认为 `filelists/genshin.list`。


```py
import json
from collections import defaultdict
from random import shuffle
from typing import Optional

from tqdm import tqdm
import click
from text.cleaner import clean_text


@click.command()
@click.option(
    "--transcription-path",
    default="filelists/genshin.list",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
```

This looks like a Python script that reads in a raw text file containing a speech recognition transcription and a word map. It then trains and validates a machine learning model using the given text data and the word map. Finally, it saves the trained model to a file and the trained and validation data to disk.

The script takes several arguments:

- `config_path`: The path to a JSON configuration file containing the machine learning model's architecture and parameters.
- `data_path`: The path to a directory containing the raw text data.
- `transcription_path`: The path to a directory containing the transcription of the raw text data into audio files.
- `output_path`: The path to a directory to which the trained model will be saved.
- `val_per_spk`: The number of samples to use from each speaker in the validation set.
- `max_val_total`: The maximum number of samples to use from the validation set.

The script reads in the raw text data from the `data_path` directory and stores it in the `spk2id` dictionary. It then creates a shuffled list of the speaker IDs for the validation set and the speaker data for training set.

It creates a new file in the `output_path` directory called `model.model.h5` (with half-width64GB). This file contains the machine learning model's parameters.

The script then loops over the training data and uses it to train the machine learning model. It saves the trained model to the `output_path` directory.

It also creates two new files in the `output_path` directory called `model.validation.txt` and `model.validation.txt`, respectively. These files contain the validation data and the validation labels.

Finally, the script saves the machine learning model to the `output_path` directory and the configuration file to the `config_path` directory.


```py
@click.option("--cleaned-path", default=None)
@click.option("--train-path", default="filelists/train.list")
@click.option("--val-path", default="filelists/val.list")
@click.option(
    "--config-path",
    default="configs/config.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--val-per-spk", default=4)
@click.option("--max-val-total", default=8)
@click.option("--clean/--no-clean", default=True)
def main(
    transcription_path: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    config_path: str,
    val_per_spk: int,
    max_val_total: int,
    clean: bool,
):
    if cleaned_path is None:
        cleaned_path = transcription_path + ".cleaned"

    if clean:
        out_file = open(cleaned_path, "w", encoding="utf-8")
        for line in tqdm(open(transcription_path, encoding="utf-8").readlines()):
            try:
                utt, spk, language, text = line.strip().split("|")
                norm_text, phones, tones, word2ph = clean_text(text, language)
                out_file.write(
                    "{}|{}|{}|{}|{}|{}|{}\n".format(
                        utt,
                        spk,
                        language,
                        norm_text,
                        " ".join(phones),
                        " ".join([str(i) for i in tones]),
                        " ".join([str(i) for i in word2ph]),
                    )
                )
            except Exception as error:
                print("err!", line, error)

        out_file.close()

        transcription_path = cleaned_path

    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    with open(transcription_path, encoding="utf-8") as f:
        for line in f.readlines():
            utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")
            spk_utt_map[spk].append(line)

            if spk not in spk_id_map.keys():
                spk_id_map[spk] = current_sid
                current_sid += 1

    train_list = []
    val_list = []

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_spk]
        train_list += utts[val_per_spk:]

    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    config = json.load(open(config_path, encoding="utf-8"))
    config["data"]["spk2id"] = spk_id_map
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


```

这段代码是一个if语句，判断当前程序是否作为主程序运行。如果当前程序是作为主程序运行，那么程序会执行if语句中的内容。

在if语句中，包含了一个名为__main__的常量，如果这个常量的值为"__main__"，那么if语句中的内容会被视为代码块，也就是说，如果当前程序是作为主程序运行，那么这个if语句中的代码块就会被执行。

所以，这段代码的作用是检查当前程序是否作为主程序运行，如果是，就执行if语句中的代码块。


```py
if __name__ == "__main__":
    main()

```

# Bert-VITS2

VITS2 Backbone with bert
## 成熟的旅行者/开拓者/舰长/博士/sensei/猎魔人/喵喵露/V应当参阅代码自己学习如何训练。
### 严禁将此项目用于一切违反《中华人民共和国宪法》，《中华人民共和国刑法》，《中华人民共和国治安管理处罚法》和《中华人民共和国民法典》之用途。
### 严禁用于任何政治相关用途。
#### Video:https://www.bilibili.com/video/BV1hp4y1K78E
#### Demo:https://www.bilibili.com/video/BV1TF411k78w
## References
+ [anyvoiceai/MassTTS](https://github.com/anyvoiceai/MassTTS)
+ [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
+ [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch)
+ [svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
+ [PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)
## 感谢所有贡献者作出的努力
<a href="https://github.com/fishaudio/Bert-VITS2/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=fishaudio/Bert-VITS2"/>
</a>


# `resample.py`

这段代码使用了以下一些常用的 Python 库：

- os：用于处理文件和目录操作
- argparse：用于解析命令行参数
- librosa：用于处理音频数据，包括录制、加载和转换等
- multiprocessing：用于并行处理计算密集型任务
- soundfile：用于播放和记录声音文件
- tqdm：用于在命令行界面上显示进度的工具，主要用于命令行应用程序

具体来说，这段代码的作用是：

- `import librosa`：用于从librosa库中导入与录制和转换音频数据相关的函数和类。
- `from multiprocessing import Pool, cpu_count`：用于从multiprocessing库中导入multiprocessing.Pool和cpu_count函数，用于在多个计算节点上并行执行计算任务。
- `import soundfile`：用于从soundfile库中导入函数用于将录制好的音频文件转换为可播放的WAV格式。
- `from tqdm import tqdm`：用于从tqdm库中导入函数用于在命令行界面上显示进度。
- `process(item)`：定义了一个函数process，接收一个包含要处理的音频数据(item)的参数。
- `spkdir`：用于从指定目录中提取出所有说话者的数据，并保存到out_dir目录中。
- `wav_name`：用于为每个录音文件指定一个名称，同时将录制时的说话者的姓名添加到文件名中。
- `args.in_dir`：用于指定说话者的录音文件目录，并指定为os.path.join(args.out_dir, speaker)的形式。
- `args.out_dir`：用于指定保存录音文件目录，并指定为os.path.join(args.out_dir, speaker)的形式。
- `if os.path.exists(wav_path) and ".wav" in wav_path:`：用于检查wav文件是否存在，并且wav文件是否以".wav"扩展名保存。
- `os.makedirs(os.path.join(args.out_dir, speaker), exist_ok=True)`：如果wav文件存在，则创建目录并允许其存在。
- `wav.read()`：从wav文件中读取音频数据，并将其存储为numpy数组。
- `soundfile.write()`：将numpy数组中的音频数据写入到wav文件中。
- `return`：返回一个布尔值，表示函数是否成功执行。


```py
import os
import argparse
import librosa
from multiprocessing import Pool, cpu_count

import soundfile
from tqdm import tqdm


def process(item):
    spkdir, wav_name, args = item
    speaker = spkdir.replace("\\", "/").split("/")[-1]
    wav_path = os.path.join(args.in_dir, speaker, wav_name)
    if os.path.exists(wav_path) and ".wav" in wav_path:
        os.makedirs(os.path.join(args.out_dir, speaker), exist_ok=True)
        wav, sr = librosa.load(wav_path, sr=args.sr)
        soundfile.write(os.path.join(args.out_dir, speaker, wav_name), wav, sr)


```

这段代码是一个命令行工具，主要用于搜索并提取地铁车站声音数据。它接受用户输入的几个参数：采样率（以秒为单位）、输入目录（包含地铁车站声音数据的目录）和目标目录（存储已提取数据的目录）。

首先，解析用户输入的参数，设置一个用于处理文件的进程数（4到8个，具体根据CPU核心数确定），并创建一个处理文件的小线程池。

接着，遍历输入目录中的所有车站。对于每个车站，首先检查它是否包含声音数据（根据文件后缀）。如果是，提取并处理这些声音数据，然后以相似的方式合并它们。这个过程会一直持续到所有车站都被处理。


```py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=44100, help="sampling rate")
    parser.add_argument(
        "--in_dir", type=str, default="./raw", help="path to source dir"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./dataset", help="path to target dir"
    )
    args = parser.parse_args()
    # processes = 8
    processes = cpu_count() - 2 if cpu_count() > 4 else 1
    pool = Pool(processes=processes)

    for speaker in os.listdir(args.in_dir):
        spk_dir = os.path.join(args.in_dir, speaker)
        if os.path.isdir(spk_dir):
            print(spk_dir)
            for _ in tqdm(
                pool.imap_unordered(
                    process,
                    [
                        (spk_dir, i, args)
                        for i in os.listdir(spk_dir)
                        if i.endswith("wav")
                    ],
                )
            ):
                pass

```

# `server.py`

这段代码使用了PyTorch和Flask框架来实现一个文本到语音的合成系统。它的主要作用是实现一个将文本数据通过Flask服务器发送给用户，并将用户的音频数据通过Flask服务器接收并合成为语音的API。下面是具体的实现步骤：

1. Flask框架的导入：from flask import Flask, request, Response
2. 定义一个Flask应用程序类：app = Flask(__name__)
3. 从io模块中导入BytesIO对象：from io import BytesIO
4. 从av库中导入open函数：from av import open as avopen
5. 定义一个SynthesizerTrn类，用于声音合成任务的训练和预测：class SynthesizerTrn
6. 定义一个fn_pros_cols类，用于读取符号和数字的序列，以及获取词嵌入：from text.symbols import symbols
from text.cleaner import clean_text
from models import SynthesizerTrn
7. 加载预训练的BERT模型：from bert.tokenization import bert_预训练
get_bert = bert_预训练
8. 定义一个 clean_text函数，用于清洗和清理文本：from text.cleaner import clean_text
9. 加载预训练的DEGREGAN模型：from transformers import auto
model = get_bert.run_from_pretrained("bert-base")
10. 通过Flask服务器接收和发送音频数据：from flask import request, Response
11. 创建一个 BytesIO 对象并初始化：BytesIO()
12. 创建一个空的 Av 对象：av = avopen("test.wav")
13. 循环读取音频数据，并将其转换为 wav 格式：while True:
           try:
               data = wavfile.read(av)
               # 将数据转换为 audio_asset 类型并将其保存到本地： BytesIO(data).audio_asset
               # 在循环中执行必要的预处理，如降噪和解码： clean_text("2.兮形象.mp3")
               # 在循环中执行必要的后处理，如增加时长和生成合成声音： SynthesizerTrn.design_model_from_dict({
                   "model": model,
                   "batch_size": options.batch_size,
                   "output_dir": "output",
                   "num_epochs": 10,
                   "steps_per_epoch": 1000,
                   "fp16": True,
                   "隐藏_layer_sizes": [0],
                   "num_attention_states": 0,
                   "的学习率": 0.001,
                   "生长的步长": 0.001,
                   "平衡步骤": True,
                   "梯度裁剪": True,
                   "预测前庭": True,
                   "伏特兄弟神秘实验": True,
                   "实验最终目标": "尽力保证合成的音频与原始音频的准确度": "绝对准确"
               })
               # 将数据保存到 disk：保存_m2m.py
               # 创建一个 BytesIO 对象并保存到 disk：save_m2m.py
               # 在循环中执行必要的清理和卸载操作： None
               # 将数据重置为零：None
               # 循环结束，开始下一个循环：av.close()


```py
from flask import Flask, request, Response
from io import BytesIO
import torch
from av import open as avopen

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text
from scipy.io import wavfile

# Flask Init
app = Flask(__name__)
```

这段代码是一个Python代码片段，它是一个名为`app.config["JSON_AS_ASCII"] = False`的键值对。具体来说，它将一个名为`JSON_AS_ASCII`的键设置为`False`，意味着它将阻止这个键在应用程序中产生JSON格式的数据。

如果更多的上下文信息，这个配置可能是一个JSON配置文件中的一个键，但它在这里被Python代码直接访问。




```py
app.config["JSON_AS_ASCII"] = False


def get_text(text, language_str, hps):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str)
    del word2ph
    assert bert.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert
        ja_bert = torch.zeros(768, len(phone))
    elif language_str == "JA":
        ja_bert = bert
        bert = torch.zeros(1024, len(phone))
    else:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))
    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language


```

这段代码定义了一个名为 `infer` 的函数，它接受一个文本 `text`，以及一个采样率 `sdp_ratio`，一个噪声 scale `noise_scale` 和一个噪声 scale `noise_scale_w`，一个文本长度 `length_scale` 和一个语言类别 `language`。它还接受一个 BERT 模型 `bert` 和一个语言建模模型 `ja_bert`。

函数首先使用 `get_text` 函数获取文本中的单词，并将这些单词转换为语言模型的输入。然后，它将输入文本和语言模型的输入以及相应的长度信息输入到函数中的模型，然后对输入的每个文本时间步进行前馈并获取模型的输出。

接下来，函数将从每个时间步获得的输出中提取出一个称为 `x_tst` 的序列，并为这个序列设置一个长度为 `length_scale` 的卷积并传递给模型。此外，它还设置一个称为 `tones` 的序列，这个序列包含一个预定义的噪声，然后将这个噪声与 `x_tst` 进行加权求和，再将它们输入到另一个称为 `net_g` 的模型中。

最后，函数使用 `infer` 函数对文本进行预处理，并返回预处理后的音频数据。


```py
def infer(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, language):
    bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps)
    with torch.no_grad():
        x_tst = phones.to(dev).unsqueeze(0)
        tones = tones.to(dev).unsqueeze(0)
        lang_ids = lang_ids.to(dev).unsqueeze(0)
        bert = bert.to(dev).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(dev)
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(dev)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        return audio


```

这两段代码的作用是定义了一个名为 replace_punctuation 的函数和一个名为 wav2 的函数。

replace_punctuation 函数的作用是将给定的文本中的标点符号（。、。、？、！）替换为指定长度的字符。该函数的参数 i 表示要替换的字符数量，从 2 开始。函数内部遍历给定的标点符号，并将它们替换为指定长度的字符。最后，函数返回修改后的文本。

wave2 函数的作用是将一个音频文件（以 wav 格式）中的音频数据录写到另一个音频文件（以 wav 格式）中。该函数的参数 i 是目标文件索引，从 0 开始。函数内部打开一个输入文件（以 wav 格式）和一个输出文件（以 wav 格式），如果输入文件的格式是 "ogg"，则函数将只从输入文件中读取音频数据。函数内部循环读取输入文件中的每一帧数据，并将其编码为 wav 格式。然后，函数将编码后的数据写入输出文件中。函数内部循环直到输入文件读取完所有数据。最后，函数关闭所有文件并返回。


```py
def replace_punctuation(text, i=2):
    punctuation = "，。？！"
    for char in punctuation:
        text = text.replace(char, char * i)
    return text


def wav2(i, o, format):
    inp = avopen(i, "rb")
    out = avopen(o, "wb", format=format)
    if format == "ogg":
        format = "libvorbis"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


```

这段代码的主要作用是加载一个名为 "configs/config.json" 的配置文件，并从中获取出用于训练的一些参数。

然后，它创建了一个名为 "net_g" 的 Synthesizer2Net 模型，这个模型采用通用的 "utils.get_hparams_from_file" 方法训练得到的。

接着，它设置模型的开发者为 "cuda"，也就是使用 CUDA 而不是传统的前向神经网络来进行训练。

然后，它使用 "utils.load_checkpoint" 方法加载了一个名为 "logs/G_649000.pth" 的训练好的模型，这个模型之前的优化器也没有被加载。

最后，它使用这个加载好的模型进行训练和测试，具体训练的参数可能由 "hps" 参数给出。


```py
# Load Generator
hps = utils.get_hparams_from_file("./configs/config.json")

dev = "cuda"
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model,
).to(dev)
_ = net_g.eval()

_ = utils.load_checkpoint("logs/G_649000.pth", net_g, None, skip_optimizer=True)


```

It looks like this is a Python Flask web application that is using an external library called "pyAudioPhone" to handle audio input and output. The application is using the parameters that are passed in the URL, and it is trying to process the audio in real-time. The application is also returning the audio in the WAV format.


```py
@app.route("/")
def main():
    try:
        speaker = request.args.get("speaker")
        text = request.args.get("text").replace("/n", "")
        sdp_ratio = float(request.args.get("sdp_ratio", 0.2))
        noise = float(request.args.get("noise", 0.5))
        noisew = float(request.args.get("noisew", 0.6))
        length = float(request.args.get("length", 1.2))
        language = request.args.get("language")
        if length >= 2:
            return "Too big length"
        if len(text) >= 250:
            return "Too long text"
        fmt = request.args.get("format", "wav")
        if None in (speaker, text):
            return "Missing Parameter"
        if fmt not in ("mp3", "wav", "ogg"):
            return "Invalid Format"
        if language not in ("JA", "ZH"):
            return "Invalid language"
    except:
        return "Invalid Parameter"

    with torch.no_grad():
        audio = infer(
            text,
            sdp_ratio=sdp_ratio,
            noise_scale=noise,
            noise_scale_w=noisew,
            length_scale=length,
            sid=speaker,
            language=language,
        )

    with BytesIO() as wav:
        wavfile.write(wav, hps.data.sampling_rate, audio)
        torch.cuda.empty_cache()
        if fmt == "wav":
            return Response(wav.getvalue(), mimetype="audio/wav")
        wav.seek(0, 0)
        with BytesIO() as ofp:
            wav2(wav, ofp, fmt)
            return Response(
                ofp.getvalue(), mimetype="audio/mpeg" if fmt == "mp3" else "audio/ogg"
            )

```

# `train_ms.py`

这段代码是一个PyTorch程序，它使用了NumPy（PyTorch的扩展库），SummaryWriter和Tqdm库来实现训练和记录数据。它的主要目的是训练一个神经网络，并记录训练过程中的信息。

具体来说，它实现了以下功能：

1. 定义了一些函数，包括数据加载器、数据集中成、网络架构、优化器、损失函数、清空日志、设置跟踪（train/test）日志、设置死标签等。
2. 加载了指定的数据集，并将其存储为DataLoader对象，以便于训练数据的使用。
3. 创建了一个SummaryWriter对象，用于记录训练过程中的信息。
4. 创建了一个DistributedDataParallel类，实现了NumPy中的`distributed`参数。
5. 创建了一个GradScaler，用于动态调整学习率。
6. 创建了一个高性能的`autocast`库，以便于记录样本来源。
7. 创建了一个简单的神经网络，并继承自DistributedDataParallel类，用于在分布式环境中训练模型。
8. 在训练过程中使用了`tqdm`库来跟踪进度。
9. 在`__init__`函数中设置了日志记录的日志级别。

它旨在成为一个高效的神经网络训练程序，支持分布式训练，并能够记录训练过程中的信息。


```py
# flake8: noqa: E402

import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
import commons
```

这段代码的主要作用是定义了一个数据预处理和训练模型的框架，实现了对语音数据的预处理、加载和处理，以及模型的训练和损失计算。

具体来说，这段代码包括以下功能：

1. 导入`utils`模块，没有具体的作用。
2. 从`data_utils`模块中导入了一系列的函数和类，如`TextAudioSpeakerLoader`、`TextAudioSpeakerCollate`、`DistributedBucketSampler`等，实现了对语音数据的预处理和加载。
3. 从`models`模块中定义了`SynthesizerTrn`、`MultiPeriodDiscriminator`、`DurationDiscriminator`等模型，实现了对语音数据的主干分析和处理。
4. 从`losses`模块中定义了`generator_loss`、`discriminator_loss`、`feature_loss`、`kl_loss`等损失函数，实现了对模型的损失计算。
5. 从`mel_processing`模块中导入了`mel_spectrogram_torch`和`spec_to_mel_torch`，实现了对语音数据的Mel频率表示法的处理。
6. 从`text.symbols`模块中导入了`symbols`，实现了对文本数据中符号的编码。
7. 在`utils.py`中定义了一些常量和函数，用于实现上述的功能。


```py
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    DurationDiscriminator,
)
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

```

这段代码是针对PyTorch中的CUDA（CUDA Universal Performance醛）API实现在交友，并允许使用CUDA的计算引擎（例如：梯度、卷积、广播等）对输入数据进行稀疏矩阵MATLAB数乘，从而实现高效的图形和深度学习模型的部署。

具体来说，这段代码的作用如下：

1. 允许使用CUDA的计算引擎执行MATLAB数乘操作。
2. 如果正在训练，则禁止使用CUDA的计算引擎，以避免在训练过程中使用CUDA造成性能提升。
3. 设置输入数据为稀疏数据类型（float32）。
4. 设置CUDA的基准测试。
5. 如果使用的是PyTorch版本小于2.0，则禁止使用CUDA的计算引擎，以获得更好的性能。
6. 如果使用的是PyTorch版本2.0，则允许使用CUDA的计算引擎。
7. 设置输入数据为稀疏数据类型。
8. 设置CUDA的数学功能。
9. 在一个爱国者网站上创建一个全局变量，用于跟踪当前的训练步骤。



```py
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = (
    True  # If encontered training problem,please try to disable TF32.
)
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.sdp_kernel("flash")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(
    True
)  # Not available if torch version is lower than 2.0
torch.backends.cuda.enable_math_sdp(True)
global_step = 0


```

This is a PyTorch implementation of a custom learning rate scheduler. The custom scheduler is applied to the training and validation loads, while the default scheduler is applied to the development load. The custom scheduler has a schedule every two epochs, and the learning rate of the development load is resumed after every two epochs.


```py
def run():
    dist.init_process_group(
        backend="gloo",
        init_method="env://",  # Due to some training problem,we proposed to use gloo instead of nccl.
    )  # Use torchrun instead of mp.spawn
    rank = dist.get_rank()
    n_gpus = dist.get_world_size()
    hps = utils.get_hparams()
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=16,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=4,
    )  # DataLoader config could be adjusted.
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    if (
        "use_noise_scaled_mas" in hps.model.keys()
        and hps.model.use_noise_scaled_mas is True
    ):
        print("Using noise scaled MAS for VITS2")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        print("Using normal MAS for VITS1")
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0
    if (
        "use_duration_discriminator" in hps.model.keys()
        and hps.model.use_duration_discriminator is True
    ):
        print("Using duration discriminator for VITS2")
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).cuda(rank)
    if (
        "use_spk_conditioned_encoder" in hps.model.keys()
        and hps.model.use_spk_conditioned_encoder is True
    ):
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
            )
    else:
        print("Using normal encoder for VITS1")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model,
    ).cuda(rank)

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_dur_disc = None
    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    if net_dur_disc is not None:
        net_dur_disc = DDP(net_dur_disc, device_ids=[rank], find_unused_parameters=True)
    try:
        if net_dur_disc is not None:
            _, _, dur_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
                net_dur_disc,
                optim_dur_disc,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            _, optim_g, g_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
                net_g,
                optim_g,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            _, optim_d, d_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
                net_d,
                optim_d,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            if not optim_g.param_groups[0].get("initial_lr"):
                optim_g.param_groups[0]["initial_lr"] = g_resume_lr
            if not optim_d.param_groups[0].get("initial_lr"):
                optim_d.param_groups[0]["initial_lr"] = d_resume_lr
            if not optim_dur_disc.param_groups[0].get("initial_lr"):
                optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr

        epoch_str = max(epoch_str, 1)
        global_step = (epoch_str - 1) * len(train_loader)
    except Exception as e:
        print(e)
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    if net_dur_disc is not None:
        if not optim_dur_disc.param_groups[0].get("initial_lr"):
            optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
            optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
    else:
        scheduler_dur_disc = None
    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d, net_dur_disc],
                [optim_g, optim_d, optim_dur_disc],
                [scheduler_g, scheduler_d, scheduler_dur_disc],
                scaler,
                [train_loader, eval_loader],
                logger,
                [writer, writer_eval],
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d, net_dur_disc],
                [optim_g, optim_d, optim_dur_disc],
                [scheduler_g, scheduler_d, scheduler_dur_disc],
                scaler,
                [train_loader, None],
                None,
                None,
            )
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()


```

This is a Python code that defines an object that wraps a neural network's training and evaluation during the training phase. The neural network is initialized with a graph and the attributes needed for training and evaluation.

The code defines several functions:

- `utils.plot_alignment_to_numpy`: This function plots the alignment of an attention mechanism to a numpy array.
- `utils.summarize`: This function summarizes the output of the neural network given a specified writer.
- `evaluate`: This function evaluates the performance of the neural network on the given dataset during the training phase.
- `utils.save_checkpoint`: This function saves the neural network's parameters to disk in specified formats.
- `utils.clean_checkpoints`: This function removes any saved checkpoints and their associated metadata, allowing the network to be reloaded without dependency on the previous checkpoint.

The code also initializes a few variables:

- `net_g`: This variable defines the graph of the neural network.
- `net_d`: This variable defines the Dur discriminator network.
- `global_step`: This variable stores the current iteration of the training loop.
- `epoch`: This variable stores the current epoch of the training loop.
- `hps`: This variable stores the hyperparameters of the training loop such as the learning rate and the number of checkpoints to save.
- `image_dict`: This variable stores the image data for the given input image.
- `scalar_dict`: This variable stores the scalar data for the given input image.
- `net_dur_disc`: This variable is defined but not used in the training or evaluation functions.


```py
def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers
):
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
    for batch_idx, (
        x,
        x_lengths,
        spec,
        spec_lengths,
        y,
        y_lengths,
        speakers,
        tone,
        language,
        bert,
        ja_bert,
    ) in tqdm(enumerate(train_loader)):
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(
            rank, non_blocking=True
        )
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(
            rank, non_blocking=True
        )
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
            rank, non_blocking=True
        )
        speakers = speakers.cuda(rank, non_blocking=True)
        tone = tone.cuda(rank, non_blocking=True)
        language = language.cuda(rank, non_blocking=True)
        bert = bert.cuda(rank, non_blocking=True)
        ja_bert = ja_bert.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
            ) = net_g(
                x,
                x_lengths,
                spec,
                spec_lengths,
                speakers,
                tone,
                language,
                bert,
                ja_bert,
            )
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(), logw.detach(), logw_.detach()
                )
                with autocast(enabled=False):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                commons.clip_grad_value_(net_dur_disc.parameters(), None)
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw, logw_)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/dur": loss_dur,
                        "loss/g/kl": loss_kl,
                    }
                )
                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/attn": utils.plot_alignment_to_numpy(
                        attn[0, 0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
                if net_dur_disc is not None:
                    utils.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
                    )
                keep_ckpts = getattr(hps.train, "keep_ckpts", 5)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(
                        path_to_models=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )

        global_step += 1

    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))


```

It looks like this is a Python script that uses the PyTorch library to train a neural network for speech recognition. The script takes as input a spectrogram, which is a visualization of the data that the model will use for training. It also takes as input the topology of the network, as well as some hyperparameters such as the minimum and maximum frequencies for the mel spectrogram.

The script then uses the mel spectrogram to generate predicted audio, which is compared to the true audio in the script. It then updates the image and audio dictionaries with the predicted values, and uses these dictionaries to summary the training information. Finally, it trains the neural network using the generator and audio generator gens, and updates the global step counter.


```py
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    print("Evaluating ...")
    with torch.no_grad():
        for batch_idx, (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            speakers,
            tone,
            language,
            bert,
            ja_bert,
        ) in enumerate(eval_loader):
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            speakers = speakers.cuda()
            bert = bert.cuda()
            ja_bert = ja_bert.cuda()
            tone = tone.cuda()
            language = language.cuda()
            for use_sdp in [True, False]:
                y_hat, attn, mask, *_ = generator.module.infer(
                    x,
                    x_lengths,
                    speakers,
                    tone,
                    language,
                    bert,
                    ja_bert,
                    y=spec,
                    max_len=1000,
                    sdp_ratio=0.0 if not use_sdp else 1.0,
                )
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                image_dict.update(
                    {
                        f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].cpu().numpy()
                        )
                    }
                )
                audio_dict.update(
                    {
                        f"gen/audio_{batch_idx}_{use_sdp}": y_hat[
                            0, :, : y_hat_lengths[0]
                        ]
                    }
                )
                image_dict.update(
                    {
                        f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            mel[0].cpu().numpy()
                        )
                    }
                )
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()


```

这段代码是一个if语句，它的判断条件是(__name__ == "__main__")。如果这个条件为真，那么执行if语句块内的语句，否则跳过if语句块。在这个例子中，if(__name__ == "__main__")后面没有其他的语句，因此这个if语句块不会被执行。

__name__是一个特殊变量，它存储当前程序的名称。在这个代码中，它被用来检查当前程序是否已经被传入了命令行参数，如果是，那么程序就可以正常运行，否则就会产生一个错误。

运行()函数未定义，因此它的具体实现可能会因程序而异。通常情况下，这个函数会被用来执行程序的主干部分，也就是程序的主要代码。


```py
if __name__ == "__main__":
    run()

```

# `transforms.py`

这段代码定义了一个名为 `piecewise_rational_quadratic_transform` 的函数，它接受一个由 `inputs`，`unnormalized_widths`，`unnormalized_heights` 和 `unnormalized_derivatives` 四个参数组成的输入。

这个函数的作用是将输入数据通过提出为 `inverse` 控制的高低频部分，然后对其进行有理化变换，最后输出经过有理化变换后的结果以及对应的拉普拉斯加法梯度。

具体来说，这个函数实现的过程如下：

1. 如果 `inverse` 为 `True`，则函数采用有理化方法，直接对输入数据进行高斯平滑处理。
2. 如果 `inverse` 为 `False`，则函数采用分段线性方法，将输入数据划分为多个子区间，分别对每个子区间采用不同的有理化方法。

具体实现中，函数还接收一个名为 `tails` 的参数，用于指定是否有尾数。当 `tails` 为 `None` 时，表示采用最小二分法确定是否有尾数，如果没有尾数，则函数将无法实现有理化。当 `tails` 给出时，函数将用该尾数约束有理化分段线性方法。此外，函数还接收一个名为 `min_bin_width` 和 `min_bin_height` 的参数，用于指定高频率部分的最小二分宽度和最小高度。


```py
import torch
from torch.nn import functional as F

import numpy as np


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def piecewise_rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails=None,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline
        spline_kwargs = {"tails": tails, "tail_bound": tail_bound}

    outputs, logabsdet = spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs
    )
    return outputs, logabsdet


```

This is a function that uses rational quadratic splines to estimate outputs from inputs based on tails (i.e., the values outside of an interval). The function takes two arguments: `inputs` and `min_bin_height` (or `tail_bound` if `min_bin_height` is not specified).

The function first checks if the `inputs` are within an interval (i.e., `-tail_bound` <= `inputs` <= `tail_bound`). If they are, the function returns the corresponding output value (assigned by the `rational_quadratic_spline` function if `min_bin_height` is specified). If the `inputs` are outside the interval, the function uses `min_derivative` to compute the normalized derivatives of the `inputs`. It then uses these normalized derivatives, along with the `inverse` parameter, to define the interval domain for the rational quadratic spline interpolation. The `outputs` and `logabsdet` values are computed based on the spline interpolation.

If `min_bin_height` is specified, the function uses the `min_bin_width` parameter to control the width of the interval bin (i.e., the number of bins used for the spline interpolation). If `min_derivative` is specified, the function uses this parameter to compute the normalized derivatives for the rational quadratic spline interpolation.

If `tails` is set to "linear", the function computes the normalized derivatives and uses them to define the interval domain for the rational quadratic spline interpolation. If `tails` is set to anything else, the function raises a `RuntimeError`.


```py
def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    (
        outputs[inside_interval_mask],
        logabsdet[inside_interval_mask],
    ) = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    return outputs, logabsdet


```

This is a function that appears to calculate the output of a neural network given the input. The function takes in two arguments, `root` and `theta_one_minus_theta`, and returns the output as well as the negative logarithm of the denominator.

The function first calculates the value of `theta_one_minus_theta` by subtracting `root` from the ratio of `(2 * c) / ((-b) + `root`) where `c` is the number of input neurons, `b` is the coefficient of the weight to the input layer, and `theta` is the value of the input neuron.

Next, the function calculates the derivative of `theta_one_minus_theta` using the formula `(1 - theta) * (1 - theta)` and `theta_one_minus_theta * (1 - theta)`.

Finally, the function calculates the output of the neural network by summing the input values and dividing by the denominator that is calculated as the negative of the sum of the squares of the input values. The function also takes into account the numerical value of `theta_one_minus_theta`.

The function returns the output as well as the negative logarithm of the denominator.


```py
def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input to a transform is not within its domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet

```

# `utils.py`

这段代码的作用是定义一个名为“script.py”的 Python 脚本，该脚本会在命令行上运行一些命令，并对运行结果进行 logging。

具体来说，该脚本会执行以下操作：

1. 导入一些必要的库，包括 os、glob、argparse、logging 和 json 等。
2. 通过 argparse 库的 parse_args() 函数，将命令行参数转换为一个名为“--batch-size”的选项，以及一个名为“--num-gpus”的选项。
3. 通过 json 库的 dumps() 函数，将脚本内部的配置信息输出到控制台。
4. 通过 logging 库的 getLogger() 和 log() 函数，在运行结果输出时记录日志信息。
5. 通过 subprocess 库的 call() 函数，在 Linux 系统中运行一系列的工具命令，包括并行化 sh matrix multiplication 和计算等。
6. 通过 numpy 和 scipy 等库，从多个来源读取一些数据，包括 wav 文件中的音频数据。
7. 将计算结果保存为 WAV 文件。

该脚本的具体实现可能会根据具体需求和环境而有所不同，但是其核心功能是为了在机器学习和深度学习等领域中，方便管理和处理数据。


```py
import os
import glob
import argparse
import logging
import json
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch

MATPLOTLIB_FLAG = False

logger = logging.getLogger(__name__)


```

It looks like you have written code for training a pre-trained InceptionV3 model on a target dataset. The `InferAndResumeCheckpoint` class seems to be used to disable gradient checking during training, which can improve training speed.

The `disable_line` argument is used to disable gradient checking for the Infer layer if `Infer` and `ResumeCheckpoint` are checked. Then, the line is enabled before training starts.

The code also creates an optimization dictionary (keyword arguments of the `optimizer` object) and sets it as the initial dictionary for the optimizer. If a `checkpoint_dict` is provided, it is loaded before training starts.

Then, the model and the optimizer are loaded from the checkpoint, and the learning rate is set. Finally, the code logs information about the loaded checkpoint, the iteration number, and the model and optimizer.


```py
def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None
        and not skip_optimizer
        and checkpoint_dict["optimizer"] is not None
    ):
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    elif optimizer is None and not skip_optimizer:
        # else:      Disable this line if Infer and resume checkpoint,then enable the line upper
        new_opt_dict = optimizer.state_dict()
        new_opt_dict_params = new_opt_dict["param_groups"][0]["params"]
        new_opt_dict["param_groups"] = checkpoint_dict["optimizer"]["param_groups"]
        new_opt_dict["param_groups"][0]["params"] = new_opt_dict_params
        optimizer.load_state_dict(new_opt_dict)

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # assert "emb_g" not in k
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (
                saved_state_dict[k].shape,
                v.shape,
            )
        except:
            # For upgrading from the old version
            if "ja_bert_proj" in k:
                v = torch.zeros_like(v)
                logger.warn(
                    f"Seems you are using the old version of the model, the {k} is automatically set to zero for backward compatibility"
                )
            else:
                logger.error(f"{k} is not in the checkpoint")

            new_state_dict[k] = v

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)

    logger.info(
        "Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration)
    )

    return model, optimizer, learning_rate, iteration


```

这段代码定义了一个名为 `save_checkpoint` 的函数，它接受五个参数：模型(model)、优化器(optimizer)、学习率(learning_rate)、迭代次数(iteration)和检查点路径(checkpoint_path)。

函数的作用是保存模型、优化器和learn_rate的状态，并将其保存在指定的checkpoint_path。

具体来说，函数首先定义了一个logger，用来输出信息，输出内容是当前迭代次数和checkpoint_path。

接下来，函数检查模型是否包含`module`属性，如果是，就从model中创建该对象的`state_dict`；否则，就从模型中创建一个简单的`state_dict`。然后，使用`torch.save()`函数将模型的state_dict保存到一个文件中，并包含当前迭代次数、优化器state_dict和learning_rate。注意，如果当前迭代次数为0，则优化器的state_dict将不会被保存。

最终，函数将已经保存好的state_dict文件保存到指定的checkpoint_path。


```py
def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


```

这段代码的作用是定义了一个名为 `summarize` 的函数，它接受六个参数：

1. 一个 `writer` 对象，用于记录每个参数的值。
2. 一个 `global_step` 变量，用于跟踪全局参数的步数。
3. 一个字典 `scalars`，用于存储各个参数的值。
4. 一个字典 `histograms`，用于存储各个参数的值。
5. 一个字典 `images`，用于存储各个参数的图像值。
6. 一个字典 `audios`，用于存储各个参数的音频值。
7. 一个浮点数 `audio_sampling_rate`，用于控制音频的采样率。

函数中首先遍历 `scalars` 中的参数，将它们的值记录到 `writer` 对象中。接着遍历 `histograms` 中的参数，将它们的值记录到 `writer` 对象中。然后遍历 `images` 中的参数，将它们的值记录到 `writer` 对象中，注意这个值是以 `"H"` 开头的，表示水平通道（height, width, channels）。接着遍历 `audios` 中的参数，将它们的值记录到 `writer` 对象中，注意这个值是以 `"w"` 开头的，表示音频采样率。


```py
def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


```

这两函数的主要作用是获取指定目录下所有匹配特定正则表达式的文件，并对它们进行排序，然后返回排好序的文件中最后一个文件的位置。这里有一个简单的dir_path和regex作为输入参数，可以根据实际需求修改。

plot\_spectrogram\_to\_numpy函数主要用于将给定的光度图转换为numpy数组。其功能是将一个捕获的Matplotlib光度图中的数据读取并将其保存为numpy数组。通过使用Matplotlib库中的一些辅助函数，例如use()函数来禁用Matplotlib的输出渲染，然后通过调用plots()函数创建一个新的Matplotlib图表，再将光度图中的数据转换为numpy数组，最后将图表关闭并返回其numpy表示。


```py
def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


```

该函数的目的是使用Matplotlib库绘制图像，并将给定的对齐信息与图像一起显示。

具体来说，函数将在函数内部创建一个新的Matplotlib图表，并使用给定的alignment参数进行绘制。如果Matplotlib库的安装目录中不存在`MATPLOTLIB_FLAG`设置，则函数将默认创建一个新的图表，并将设置为`Agg`的Matplotlib库版本。

然后，函数将创建一个新的6x4英寸的图像，并将其设置为imshow函数的输入参数。函数还将在图像的背景上设置一个颜色bar，并设置其交互式颜色以响应alpha参数的值。

接下来，函数将设置x轴标签和额外的信息，如果info参数为有效值，则其中的info将添加到图像上。然后，函数使用tight_layout函数来设置图像的布局，以便在打印时不会留下空白行或列。

最后，函数使用canvas.draw函数从图表中获取图像数据，并将其转换为8个字节双精度浮点数数组。然后，函数将此数据重塑为六行四列的格式，以便在图像中正确显示。

返回数据以供plot_alignment_to_numpy函数中的plot函数使用。


```py
def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


```



该代码定义了两个函数，以及一个参数`init`。

函数`load_wav_to_torch`接收一个完整的音频文件路径，并返回一个PyTorch浮点数张量和一个采样率。函数`load_filepaths_and_text`接收一个文件名和一个分隔符，并返回一个包含所有文件路径和文本的列表。函数`get_hparams`是一个带有两个参数的函数，用于从用户获取需要使用的参数。

函数`load_wav_to_torch`的作用是将读取的音频文件转换为PyTorch浮点数张量，并返回它。这个函数需要使用`read`函数读取音频文件，并将其转换为浮点数张量。然后，它将返回这个张量和采样率。

函数`get_hparams`的作用是从用户那里获取需要使用的参数。它需要使用`argparse`模块中的`ArgumentParser`函数来读取用户输入。它还使用`os.path.join`函数将模型目录与参数文件组合在一起。如果用户没有提供参数文件，它将提供一个默认的模型目录和参数文件。如果提供了参数文件，它将读取并保存到参数文件中。

函数`argparse.ArgumentParser`是一个Python的`ArgumentParser`类，用于解析命令行参数。它接受一个或多个参数，并将它们存储在`parser.add_argument`函数中。

函数`HParams`是一个类，它代表了程序中的参数。它接受一个模型名称和一个配置文件路径，并从这两个参数中读取配置信息。它将这些信息存储在一个`config`属性中，并将这个属性存储在一个`model_dir`属性中。

函数`HParams`的`model_dir`属性是一个字符串，用于指定存储模型参数的目录。


```py
def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/base.json",
        help="JSON file for configuration",
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")

    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r", encoding="utf-8") as f:
            data = f.read()
        with open(config_save_path, "w", encoding="utf-8") as f:
            f.write(data)
    else:
        with open(config_save_path, "r", vencoding="utf-8") as f:
            data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


```

This is a Python function that deletes some saved checkpoints from a specified model directory. The function takes several arguments:

- `path_to_models`: The path to the directory containing the model files.
- `n_ckpts_to_keep`: The number of checkpoints to keep, excluding the files with the extension `_0.pth`.
- `sort_by_time`: A boolean value indicating whether the checkpoints should be sorted by time or lexicographically.

The function starts by looking for checkpoints files and deleting those that match the given criteria. If `sort_by_time` is `True`, the checkpoints are sorted by time and then deleted. If `sort_by_time` is `False`, the checkpoints are deleted lexicographically, which means the files are deleted based on the order in which they were saved.

The function also provides a `del_info` function to print a message for each deleted checkpoint file, and a `del_routine` function to remove the file and its associated message from the `to_del` list.


```py
def clean_checkpoints(path_to_models="logs/44k/", n_ckpts_to_keep=2, sort_by_time=True):
    """Freeing up space by deleting saved ckpts

    Arguments:
    path_to_models    --  Path to the model directory
    n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
    sort_by_time      --  True -> chronologically delete ckpts
                          False -> lexicographically delete ckpts
    """
    import re

    ckpts_files = [
        f
        for f in os.listdir(path_to_models)
        if os.path.isfile(os.path.join(path_to_models, f))
    ]

    def name_key(_f):
        return int(re.compile("._(\\d+)\\.pth").match(_f).group(1))

    def time_key(_f):
        return os.path.getmtime(os.path.join(path_to_models, _f))

    sort_key = time_key if sort_by_time else name_key

    def x_sorted(_x):
        return sorted(
            [f for f in ckpts_files if f.startswith(_x) and not f.endswith("_0.pth")],
            key=sort_key,
        )

    to_del = [
        os.path.join(path_to_models, fn)
        for fn in (x_sorted("G")[:-n_ckpts_to_keep] + x_sorted("D")[:-n_ckpts_to_keep])
    ]

    def del_info(fn):
        return logger.info(f".. Free up space by deleting ckpt {fn}")

    def del_routine(x):
        return [os.remove(x), del_info(x)]

    [del_routine(fn) for fn in to_del]


```

这两个函数的作用是读取一个指定目录下的参数配置文件，并返回其中的参数对象。通过读取文件中的 JSON 格式的数据，将文件读取的参数解析为 Python 内置的 `HParams` 类中可以使用的参数，然后将参数对象的模型目录设置为传入的模型目录，最后返回解析后的参数对象。

具体来说，`get_hparams_from_dir` 函数的作用是：

1. 读取指定目录下的配置文件（由 `model_dir` 参数指定）。
2. 将文件内容读取为 JSON 格式。
3. 使用 `json.loads` 函数将 JSON 格式的数据解析为 Python 内置的 `HParams` 类中可以使用的参数。
4. 将解析后的参数对象的模型目录设置为传入的模型目录。
5. 返回解析后的参数对象。

`get_hparams_from_file` 函数的作用是：

1. 读取指定文件路径的配置文件（由 `config_path` 参数指定）。
2. 将文件内容读取为 JSON 格式。
3. 使用 `json.loads` 函数将 JSON 格式的数据解析为 Python 内置的 `HParams` 类中可以使用的参数。
4. 返回解析后的参数对象。


```py
def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


```

这段代码的作用是检查一个名为 "model_dir" 的目录是否是一个 git 仓库，如果不是，则输出一条警告信息，并返回。如果是一个 git 仓库，则计算并存储当前目录的 git 哈希值。

具体来说，代码首先定义了一个名为 "check_git_hash" 的函数，它接受一个名为 "model_dir" 的参数。函数内部先定义了一个名为 "source_dir" 的变量，使用 os.path.dirname() 和 os.path.realpath() 函数获取出 "model_dir" 目录的父目录和文件的路径，然后使用 os.path.exists() 函数检查 "source_dir" 目录是否存在，如果不存在，则输出一条警告信息，并返回。

接着，函数内部定义了一个名为 "cur_hash" 的变量，使用 subprocess.getoutput() 函数获取出当前目录的 git 分支的哈希值。然后，定义了一个名为 "path" 的变量，使用 os.path.join() 函数将当前目录 "model_dir" 和 "githash" 文件夹的路径连接起来，得到 "path" 变量。如果 "path" 文件已经存在，则使用 open() 函数读取并写入 "path" 文件中的内容，否则使用 write() 函数将 "cur_hash" 写入 "path" 文件中。

最后，在函数内部使用 if 语句判断 "source_dir" 是否为 git 仓库，如果是，则输出一条警告信息，否则跳过该部分代码。


```py
def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn(
            "{} is not a git repository, therefore hash value comparison will be ignored.".format(
                source_dir
            )
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn(
                "git hash values are different. {}(saved) != {}(current)".format(
                    saved_hash[:8], cur_hash[:8]
                )
            )
    else:
        open(path, "w").write(cur_hash)


```

这段代码定义了一个名为 `get_logger` 的函数，它接受一个参数 `model_dir`，和一个可选的参数 `filename`。函数内部创建了一个名为 `logger` 的全局变量，并设置其日志级别为 `DEBUG`。

接下来，函数创建了一个日志格式器 `formatter`，其中包含当前时间戳、日志名称、日志级别和消息。如果 `model_dir` 目录不存在，函数会创建它。

然后，函数创建了一个名为 `h` 的日志文件handler，并设置其日志级别为 `DEBUG`。接着，函数将创建的日志文件handler添加到 `logger` 对象中，以便将日志输出到文件中。

最后，函数返回 `logger` 变量，以便在需要时进行调用。


```py
def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


```

这段代码定义了一个名为 HParams 的类，用于在 Python 中处理参数和属性。该类的方法可以用来获取、设置和判断一个参数或属性是否存在。具体来说，HParams 类的 __init__ 方法接受一个字典 kwargs，并将它们的所有键值对分别赋值给类的属性。这样，当你创建一个 HParams 对象时，你可以通过 `HParams(**kwargs)` 创建一个新对象，它将拥有一个与 kwargs 参数完全相同的数据。

HParams类的其他方法包括：

* keys(): 返回参数或属性字典的内容，它们通常是命名参数。
* items(): 返回参数或属性对象的键和值列表，它们通常是迭代返回的。
* values(): 返回参数或属性对象的内容，通常是对象或属性值。
* len(): 返回参数或属性对象的数量。
* getter(index): 通过索引获取指定参数或属性对象的值。
* setter(index, value): 通过索引设置指定参数或属性对象的值。
* contains(key): 判断参数或属性对象中是否存在指定键。
* __repr__(): 返回一个字符串表示该对象。


```py
class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

```