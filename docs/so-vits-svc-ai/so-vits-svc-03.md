# SO-VITS-SVC源码解析 3

# `wav_upload.py`

这段代码的作用是上传不同类型的文件。传来的文件将被存储为Google Colab中的文件。

具体来说，代码实现以下步骤：

1. 定义一个文件类型参数 `--type`, 从命令行输入中读取此参数。
2. 如果 `--type` 是 "zip" 或 "audio", 则定义上传文件的目标目录。
3. 如果 `--type` 是 "zip", 则将上传的文件移动到指定的目录中。
4. 如果 `--type" audio"`, 则将上传的文件移动到指定的目录中。
5. 在 `__main__` 函数中， 创建一个文件上传器并将 `--type` 参数添加到它。
6. 读取命令行输入并设置 `args.type`。
7. 如果 `args.type` 是 "zip", 则将文件上传到 `os.path.join(basepath, args.type)`。
8. 如果 `args.type` 是 "audio", 则将文件上传到 `os.path.join(basepath, args.type)`。
9. 调用 `files.upload()` 并解析结果， 将结果存储为 `uploaded`。
10. 根据 `args.type` 的值， 将文件移动到正确的目录中。

此代码使用 `argparse` 和 `os` 模块实现文件上传。上传的文件将存储在Google Colab中的指定目录中。


```py
import argparse
import os
import shutil

from google.colab import files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, help="type of file to upload")
    args = parser.parse_args()
    file_type = args.type

    basepath = os.getcwd()
    uploaded = files.upload() # 上传文件
    assert(file_type in ['zip', 'audio'])
    if file_type == "zip":
        upload_path = "./upload/"
        for filename in uploaded.keys():
            #将上传的文件移动到指定的位置上
            shutil.move(os.path.join(basepath, filename), os.path.join(upload_path, "userzip.zip"))
    elif file_type == "audio":
        upload_path = "./raw/"
        for filename in uploaded.keys():
            #将上传的文件移动到指定的位置上
            shutil.move(os.path.join(basepath, filename), os.path.join(upload_path, filename))
```

# `webUI.py`

这段代码是一个用于获取语料库中所有文本文件的Python脚本。它首先导入了所需的模块，包括glob、json、logging、os、re、subprocess、sys、time和traceback。然后，它使用os模块中的system函数获取一个URL，该URL包含一个用于下载预训练权的GPU设备。接下来，它使用gradio库从用户那里获取输入数据。然后，它使用librosa模块下载音频文件并将其保存到指定的目录中。最后，它使用o进展望册和时间模块来处理文件和目录。


```py
import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from itertools import chain
from pathlib import Path

# os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
import gradio as gr
import librosa
```

这段代码的作用是实现一个基于压缩感知和转录技术的自然语言处理模型，用于生成文本音频。主要步骤如下：

1. 导入所需的库：numpy, soundfile, torch, 压缩感知模型，用于去除优化器，支持的语音语言列表，用于进行语音识别的模型，用于混合训练和测试的模型。

2. 创建一个带有去除优化器的压缩感知模型，并加载预训练权重。

3. 加载用于进行语音识别的模型，并加载其可用的语音语言列表。

4. 加载用于混合训练和测试的模型。

5. 编译模型，并将输入数据（文本和音频）转换为模型可以处理的格式。

6. 训练模型，并将产生的模型保存到文件中。


```py
import numpy as np
import soundfile
import torch

from compress_model import removeOptimizer
from edgetts.tts_voices import SUPPORTED_LANGUAGES
from inference.infer_tool import Svc
from utils import mix_model

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

```

这段代码定义了几个变量，以及一个名为 `model` 的引用，一个名为 `spk` 的引用，一个名为 `debug` 的布尔值，一个名为 `local_model_root` 的字符串，一个名为 `cuda` 的字典，以及一个名为 `upload_mix_append_file` 的函数。

`model` 变量是一个引用，指向一个类，这个类的名称没有被定义。`spk` 变量也是一个引用，指向一个类，这个类的名称没有被定义。`debug` 是一个布尔值，表示在某些情况下，输出堆栈跟踪信息。`local_model_root` 是一个字符串，表示一个目录，用于存储已经训练好的模型。

`cuda` 是一个字典，用于存储 CUDA 设备的映射。这些设备在 CUDA 安装目录中存在。`upload_mix_append_file` 是一个函数，它接受两个参数：一个列表（可能是音频文件）和一个列表（可能是混音文件）。这个函数尝试将混音文件和音频文件上传到服务器，并返回上传的文件列表和混音模型的输出。它使用一个字典 `cuda` 来存储上传文件的设备信息。


```py
model = None
spk = None
debug = False

local_model_root = './trained'

cuda = {}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"

def upload_mix_append_file(files,sfiles):
    try:
        if(sfiles is None):
            file_paths = [file.name for file in files]
        else:
            file_paths = [file.name for file in chain(files,sfiles)]
        p = {file:100 for file in file_paths}
        return file_paths,mix_model_output1.update(value=json.dumps(p,indent=2))
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

```

这段代码是一个 Python 函数，名为 `mix_submit_click`，它接受两个参数 `js` 和 `mode`。这个函数的作用是判断给定的 JSON 数据是否合法，如果合法，则执行一系列操作，并将结果返回。

具体来说，这个函数首先尝试从给定的 JSON 数据中取出 `model_path` 和 `mix_rate`，然后将这些数据传给一个名为 `mix_model` 的函数。如果 `mix_model` 函数能够正确地加载数据，则会将结果保存到指定的路径。如果加载失败或者输入的数据格式不正确，则会抛出异常并返回一个错误信息。


```py
def mix_submit_click(js,mode):
    try:
        assert js.lstrip()!=""
        modes = {"凸组合":0, "线性组合":1}
        mode = modes[mode]
        data = json.loads(js)
        data = list(data.items())
        model_path,mix_rate = zip(*data)
        path = mix_model(model_path,mix_rate,mode)
        return f"成功，文件被保存在了{path}"
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

```

This is a function that loads a trained speech recognition (SR) model from a file using the specified path. The path must be a string that includes the name of the configuration file and the name of the cluster file, which define the parameters for the SR model.

The function takes as input the path to the model configuration file and the path to the cluster file. It returns the status code, a message string, and a list of available model sounds.

If there is a problem loading the model, the function will raise an exception and return a non-zero status code. If the model cannot be loaded, the function will print an error message and return a non-zero status code. If the model is successfully loaded, the function will print a message and return a non-zero status code.


```py
def updata_mix_info(files):
    try:
        if files is None :
            return mix_model_output1.update(value="")
        p = {file.name:100 for file in files}
        return mix_model_output1.update(value=json.dumps(p,indent=2))
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def modelAnalysis(model_path,config_path,cluster_model_path,device,enhance,diff_model_path,diff_config_path,only_diffusion,use_spk_mix,local_model_enabled,local_model_selection):
    global model
    try:
        device = cuda[device] if "CUDA" in device else device
        cluster_filepath = os.path.split(cluster_model_path.name) if cluster_model_path is not None else "no_cluster"
        # get model and config path
        if (local_model_enabled):
            # local path
            model_path = glob.glob(os.path.join(local_model_selection, '*.pth'))[0]
            config_path = glob.glob(os.path.join(local_model_selection, '*.json'))[0]
        else:
            # upload from webpage
            model_path = model_path.name
            config_path = config_path.name
        fr = ".pkl" in cluster_filepath[1]
        model = Svc(model_path,
                config_path,
                device=device if device != "Auto" else None,
                cluster_model_path = cluster_model_path.name if cluster_model_path is not None else "",
                nsf_hifigan_enhance=enhance,
                diffusion_model_path = diff_model_path.name if diff_model_path is not None else "",
                diffusion_config_path = diff_config_path.name if diff_config_path is not None else "",
                shallow_diffusion = True if diff_model_path is not None else False,
                only_diffusion = only_diffusion,
                spk_mix_enable = use_spk_mix,
                feature_retrieval = fr
                )
        spks = list(model.spk2id.keys())
        device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
        msg = f"成功加载模型到设备{device_name}上\n"
        if cluster_model_path is None:
            msg += "未加载聚类模型或特征检索模型\n"
        elif fr:
            msg += f"特征检索模型{cluster_filepath[1]}加载成功\n"
        else:
            msg += f"聚类模型{cluster_filepath[1]}加载成功\n"
        if diff_model_path is None:
            msg += "未加载扩散模型\n"
        else:
            msg += f"扩散模型{diff_model_path.name}加载成功\n"
        msg += "当前模型的可用音色：\n"
        for i in spks:
            msg += i + " "
        return sid.update(choices = spks,value=spks[0]), msg
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

    
```

This is a Python function that creates an output file for an audio clip based on various parameters. It takes in different parameters such as the audio path, the output format, the sound quality, and several other parameters and then writes the audio to a file.

Here's a brief description of the parameters:

- `audio_path`: The path to the audio file.
- `output_format`: The format of the output file.
- `quality`: The quality of the audio.
- `output_file_name`: The name of the output file.
- `_audio`: The audio signal.
- `model.target_sample`: The target sample rate of the audio.
- `model.transform`: The audio transform used in the model.
- `model.slice_db`: The audio database size used in the model.
- `model.cluster_ratio`: The ratio of the clusters in the DNN.
- `model.auto_f0`: Whether to use the auto-optimized F0 values in the model or not.
- `model.noise_scale`: The noise scale used in the model.
- `model.pad_seconds`: The number of seconds of padding added to the end of the audio signal.
- `model.cl_num`: The number of clusters in the DNN.
- `model.lg_num`: The number of loudness classes in the output file.
- `model.lgr_num`: The number of cluster ratio lines in the output file.
- `model.f0_predictor`: Whether to use the predicted F0 values in the model or not.
- `model.enhancer_adaptive_key`: Whether to use the adaptive key in the enhancer or not.
- `model.cr_threshold`: The threshold for the critical region display.
- `model.k_step`: The number of learning steps in the auto-enhancer.
- `model.use_spk_mix`: Whether to use theSpeak-to-Mix approach or not.
- `model.second_encoding`: Whether to enable the second encoding or not.
- `model.loudness_envelope_adjustment`: Whether to adjust the loudness envelope or not.

It is important to note that this function is based on the `soundfile` library


```py
def modelUnload():
    global model
    if model is None:
        return sid.update(choices = [],value=""),"没有模型需要卸载!"
    else:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
        return sid.update(choices = [],value=""),"模型卸载完毕!"
    
def vc_infer(output_format, sid, audio_path, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    global model
    _audio = model.slice_inference(
        audio_path,
        sid,
        vc_transform,
        slice_db,
        cluster_ratio,
        auto_f0,
        noise_scale,
        pad_seconds,
        cl_num,
        lg_num,
        lgr_num,
        f0_predictor,
        enhancer_adaptive_key,
        cr_threshold,
        k_step,
        use_spk_mix,
        second_encoding,
        loudness_envelope_adjustment
    )  
    model.clear_empty()
    #构建保存文件的路径，并保存到results文件夹内
    str(int(time.time()))
    if not os.path.exists("results"):
        os.makedirs("results")
    key = "auto" if auto_f0 else f"{int(vc_transform)}key"
    cluster = "_" if cluster_ratio == 0 else f"_{cluster_ratio}_"
    isdiffusion = "sovits"
    if model.shallow_diffusion:
        isdiffusion = "sovdiff"

    if model.only_diffusion:
        isdiffusion = "diff"
    
    output_file_name = 'result_'+truncated_basename+f'_{sid}_{key}{cluster}{isdiffusion}.{output_format}'
    output_file = os.path.join("results", output_file_name)
    soundfile.write(output_file, _audio, model.target_sample, format=output_format)
    return output_file

```

This is a Python function that takes in an audio file path, and a model and initializes the model. It then reads the audio file, normalizes it, and writes the processed audio to a file.

The function uses the `soundfile` library


```py
def vc_fn(sid, input_audio, output_format, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment):
    global model
    try:
        if input_audio is None:
            return "You need to upload an audio", None
        if model is None:
            return "You need to upload an model", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
        #print(input_audio)    
        audio, sampling_rate = soundfile.read(input_audio)
        #print(audio.shape,sampling_rate)
        if np.issubdtype(audio.dtype, np.integer):
            audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        #print(audio.dtype)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        # 未知原因Gradio上传的filepath会有一个奇怪的固定后缀，这里去掉
        truncated_basename = Path(input_audio).stem[:-6]
        processed_audio = os.path.join("raw", f"{truncated_basename}.wav")
        soundfile.write(processed_audio, audio, sampling_rate, format="wav")
        output_file = vc_infer(output_format, sid, processed_audio, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)

        return "Success", output_file
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

```

This is a Python function that retrieves a text-to-speech model to convert text to an audio file. The function takes in a text, language, and rate (in percentage) as inputs, and outputs the path to the audio file.

The function uses the TTS-G Client Library to retrieve the audio file. If the text-to-speech model is not found, or the input file is not accessible, an exception is raised and the function prints an error message.

The function can also perform cluster ratio calculations on the input audio file. If you want to calculate the cluster ratio, you need to first convert the audio file to a librosa feature, such as a Mel-Frequency Cepstral Coefficient (MFCC).

The function then uses the librosa features to train a retrieval model, and assigns a cluster ratio to the output file. The cluster ratio is a measure of how similar the cluster of audio files are to each other. The function uses an AutoF0 predictor to improve the accuracy of the retrieval model.

Overall, the function is designed to convert text to audio files in a text-to-speech model, and can be useful for generating subtitles for videos or audio descriptions for text-based inputs.


```py
def text_clear(text):
    return re.sub(r"[\n\,\(\) ]", "", text)

def vc_fn2(_text, _lang, _gender, _rate, _volume, sid, output_format, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold, k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment):
    global model
    try:
        if model is None:
            return "You need to upload an model", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
        _rate = f"+{int(_rate*100)}%" if _rate >= 0 else f"{int(_rate*100)}%"
        _volume = f"+{int(_volume*100)}%" if _volume >= 0 else f"{int(_volume*100)}%"
        if _lang == "Auto":
            _gender = "Male" if _gender == "男" else "Female"
            subprocess.run([sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume, _gender])
        else:
            subprocess.run([sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume])
        target_sr = 44100
        y, sr = librosa.load("tts.wav")
        resampled_y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        soundfile.write("tts.wav", resampled_y, target_sr, subtype = "PCM_16")
        input_audio = "tts.wav"
        #audio, _ = soundfile.read(input_audio)
        output_file_path = vc_infer(output_format, sid, input_audio, "tts", vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)
        os.remove("tts.wav")
        return "Success", output_file_path
    except Exception as e:
        if debug: traceback.print_exc()  # noqa: E701
        raise gr.Error(e)

```

这段代码定义了两个函数：`model_compression` 和 `scan_local_models`。它们的作用如下：

1. `model_compression` 函数接收一个参数 `_model`，它是一个字符串，表示要压缩的模型。如果模型路径不存在，函数返回错误消息。否则，函数将创建一个新的文件名，并使用 `os.path.splitext` 函数获取文件名和文件的后缀，创建一个新的输出文件名，并删除优化器，最后返回成功消息。
2. `scan_local_models` 函数扫描本地模型根目录中的所有模型文件。它使用 `glob` 函数遍历模型文件，将文件路径和文件名存储在两个列表中。然后，它遍历候选模型并检查它们是否只包含一个 JSON 文件和一个或多个 PTH 文件。如果是这样，函数将返回模型路径，否则将返回错误消息。

这两个函数的主要目的是让用户可以选择要压缩的模型，并扫描本地模型根目录中的所有模型文件，以便在需要时进行压缩。


```py
def model_compression(_model):
    if _model == "":
        return "请先选择要压缩的模型"
    else:
        model_path = os.path.split(_model.name)
        filename, extension = os.path.splitext(model_path[1])
        output_model_name = f"{filename}_compressed{extension}"
        output_path = os.path.join(os.getcwd(), output_model_name)
        removeOptimizer(_model.name, output_path)
        return f"模型已成功被保存在了{output_path}"

def scan_local_models():
    res = []
    candidates = glob.glob(os.path.join(local_model_root, '**', '*.json'), recursive=True)
    candidates = set([os.path.dirname(c) for c in candidates])
    for candidate in candidates:
        jsons = glob.glob(os.path.join(candidate, '*.json'))
        pths = glob.glob(os.path.join(candidate, '*.pth'))
        if (len(jsons) == 1 and len(pths) == 1):
            # must contain exactly one json and one pth file
            res.append(candidate)
    return res

```

This is a script written in PyQt5, GUI implemented in Qt5, using the PyTorch library for deep learning. It seems to be a model training and evaluation tool for speech synthesis.

It starts by opening a window with a reference to a local model, and allows the user to enable or disable the local model. It then enters an infinite loop to train the local model using the pre-trained speech synthesis model and enduring noise.

The training loop uses the `train_loop` function from PyTorch to handle the training process, which takes as input the training data, the loss function, and the learning rate. The function also initializes the model, the optimizer, and the counter for updating the counter.

The function also uses the `evaluate` function to evaluate the model on the test data, which is passed to the model. This function returns the loss and the accuracy of the model.

The script also uses a function `create_connection` to create a connection between the host machine and the remote server, which allows the user to submit their training data.

The script also uses a function `vc_submit2` to submit the training data to the remote server, and another function `vc_submit` to submit the test data to the remote server.

It also uses a function `output_format` to specify the output format for the test data, which is passed to the `output_model_format` function.

There is also a function `text2tts`, which is a function that converts text to text-to-speech, and a function `tts_lang`, which is a function that converts text-to-speech to the desired language.

There is also a function `tts_gender` which is a function that converts text-to-speech to male or female, and a function `tts_rate` which is a function that converts text-to-speech to the desired rate, it can be set from 100 to 3000.

There is also a function `tts_volume` which is a function that converts text-to-speech to the desired volume, it can be set from 500 to 10000.

There is also a function `sid` which is a function that generates a unique identifier for the audio, and `output_format` which is a function that formats the output, it can be set from %煅φ米飄splice audio, %煅°与国家领导人习近平的小道谢， and %煅询号 question。

There is also a function ` VC_LAN , which is a function that sets the local audio sampling rate, and a function ` VC_SPEECH, which is a function that sets the remote audio sampling rate, and a function ` VC_CHECK, which is a function that checks if the audio is good, and a function ` VC_ADD_CINFO`, which is a function that adds the connection information to the audio file.


```py
def local_model_refresh_fn():
    choices = scan_local_models()
    return gr.Dropdown.update(choices=choices)

def debug_change():
    global debug
    debug = debug_button.value

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue = gr.themes.colors.green,
        font=["Source Sans Pro", "Arial", "sans-serif"],
        font_mono=['JetBrains mono', "Consolas", 'Courier New']
    ),
) as app:
    with gr.Tabs():
        with gr.TabItem("推理"):
            gr.Markdown(value="""
                So-vits-svc 4.0 推理 webui
                """)
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=2> 模型设置</font>
                        """)
                    with gr.Tabs():
                        # invisible checkbox that tracks tab status
                        local_model_enabled = gr.Checkbox(value=False, visible=False)
                        with gr.TabItem('上传') as local_model_tab_upload:
                            with gr.Row():
                                model_path = gr.File(label="选择模型文件")
                                config_path = gr.File(label="选择配置文件")
                        with gr.TabItem('本地') as local_model_tab_local:
                            gr.Markdown(f'模型应当放置于{local_model_root}文件夹下')
                            local_model_refresh_btn = gr.Button('刷新本地模型列表')
                            local_model_selection = gr.Dropdown(label='选择模型文件夹', choices=[], interactive=True)
                    with gr.Row():
                        diff_model_path = gr.File(label="选择扩散模型文件")
                        diff_config_path = gr.File(label="选择扩散模型配置文件")
                    cluster_model_path = gr.File(label="选择聚类模型或特征检索文件（没有可以不选）")
                    device = gr.Dropdown(label="推理设备，默认为自动选择CPU和GPU", choices=["Auto",*cuda.keys(),"cpu"], value="Auto")
                    enhance = gr.Checkbox(label="是否使用NSF_HIFIGAN增强,该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭", value=False)
                    only_diffusion = gr.Checkbox(label="是否使用全扩散推理，开启后将不使用So-VITS模型，仅使用扩散模型进行完整扩散推理，默认关闭", value=False)
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=3>左侧文件全部选择完毕后(全部文件模块显示download)，点击“加载模型”进行解析：</font>
                        """)
                    model_load_button = gr.Button(value="加载模型", variant="primary")
                    model_unload_button = gr.Button(value="卸载模型", variant="primary")
                    sid = gr.Dropdown(label="音色（说话人）")
                    sid_output = gr.Textbox(label="Output Message")


            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=2> 推理设置</font>
                        """)
                    auto_f0 = gr.Checkbox(label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声勾选此项会究极跑调）", value=False)
                    f0_predictor = gr.Dropdown(label="选择F0预测器,可选择crepe,pm,dio,harvest,rmvpe,默认为pm(注意：crepe为原F0使用均值滤波器)", choices=["pm","dio","harvest","crepe","rmvpe"], value="pm")
                    vc_transform = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
                    cluster_ratio = gr.Number(label="聚类模型/特征检索混合比例，0-1之间，0即不启用聚类/特征检索。使用聚类/特征检索能提升音色相似度，但会导致咬字下降（如果使用建议0.5左右）", value=0)
                    slice_db = gr.Number(label="切片阈值", value=-40)
                    output_format = gr.Radio(label="音频输出格式", choices=["wav", "flac", "mp3"], value = "wav")
                    noise_scale = gr.Number(label="noise_scale 建议不要动，会影响音质，玄学参数", value=0.4)
                    k_step = gr.Slider(label="浅扩散步数，只有使用了扩散模型才有效，步数越大越接近扩散模型的结果", value=100, minimum = 1, maximum = 1000)
                with gr.Column():
                    pad_seconds = gr.Number(label="推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现", value=0.5)
                    cl_num = gr.Number(label="音频自动切片，0为不切片，单位为秒(s)", value=0)
                    lg_num = gr.Number(label="两端音频切片的交叉淡入长度，如果自动切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，注意，该设置会影响推理速度，单位为秒/s", value=0)
                    lgr_num = gr.Number(label="自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭", value=0.75)
                    enhancer_adaptive_key = gr.Number(label="使增强器适应更高的音域(单位为半音数)|默认为0", value=0)
                    cr_threshold = gr.Number(label="F0过滤阈值，只有启动crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音", value=0.05)
                    loudness_envelope_adjustment = gr.Number(label="输入源响度包络替换输出响度包络融合比例，越靠近1越使用输出响度包络", value = 0)
                    second_encoding = gr.Checkbox(label = "二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，效果时好时差，默认关闭", value=False)
                    use_spk_mix = gr.Checkbox(label = "动态声线融合", value = False, interactive = False)
            with gr.Tabs():
                with gr.TabItem("音频转音频"):
                    vc_input3 = gr.Audio(label="选择音频", type="filepath")
                    vc_submit = gr.Button("音频转换", variant="primary")
                with gr.TabItem("文字转音频"):
                    text2tts=gr.Textbox(label="在此输入要转译的文字。注意，使用该功能建议打开F0预测，不然会很怪")
                    with gr.Row():
                        tts_gender = gr.Radio(label = "说话人性别", choices = ["男","女"], value = "男")
                        tts_lang = gr.Dropdown(label = "选择语言，Auto为根据输入文字自动识别", choices=SUPPORTED_LANGUAGES, value = "Auto")
                        tts_rate = gr.Slider(label = "TTS语音变速（倍速相对值）", minimum = -1, maximum = 3, value = 0, step = 0.1)
                        tts_volume = gr.Slider(label = "TTS语音音量（相对值）", minimum = -1, maximum = 1.5, value = 0, step = 0.1)
                    vc_submit2 = gr.Button("文字转换", variant="primary")
            with gr.Row():
                with gr.Column():
                    vc_output1 = gr.Textbox(label="Output Message")
                with gr.Column():
                    vc_output2 = gr.Audio(label="Output Audio", interactive=False)

        with gr.TabItem("小工具/实验室特性"):
            gr.Markdown(value="""
                        <font size=2> So-vits-svc 4.0 小工具/实验室特性</font>
                        """)
            with gr.Tabs():
                with gr.TabItem("静态声线融合"):
                    gr.Markdown(value="""
                        <font size=2> 介绍:该功能可以将多个声音模型合成为一个声音模型(多个模型参数的凸组合或线性组合)，从而制造出现实中不存在的声线 
                                          注意：
                                          1.该功能仅支持单说话人的模型
                                          2.如果强行使用多说话人模型，需要保证多个模型的说话人数量相同，这样可以混合同一个SpaekerID下的声音
                                          3.保证所有待混合模型的config.json中的model字段是相同的
                                          4.输出的混合模型可以使用待合成模型的任意一个config.json，但聚类模型将不能使用
                                          5.批量上传模型的时候最好把模型放到一个文件夹选中后一起上传
                                          6.混合比例调整建议大小在0-100之间，也可以调为其他数字，但在线性组合模式下会出现未知的效果
                                          7.混合完毕后，文件将会保存在项目根目录中，文件名为output.pth
                                          8.凸组合模式会将混合比例执行Softmax使混合比例相加为1，而线性组合模式不会
                        </font>
                        """)
                    mix_model_path = gr.Files(label="选择需要混合模型文件")
                    mix_model_upload_button = gr.UploadButton("选择/追加需要混合模型文件", file_count="multiple")
                    mix_model_output1 = gr.Textbox(
                                            label="混合比例调整，单位/%",
                                            interactive = True
                                         )
                    mix_mode = gr.Radio(choices=["凸组合", "线性组合"], label="融合模式",value="凸组合",interactive = True)
                    mix_submit = gr.Button("声线融合启动", variant="primary")
                    mix_model_output2 = gr.Textbox(
                                            label="Output Message"
                                         )
                    mix_model_path.change(updata_mix_info,[mix_model_path],[mix_model_output1])
                    mix_model_upload_button.upload(upload_mix_append_file, [mix_model_upload_button,mix_model_path], [mix_model_path,mix_model_output1])
                    mix_submit.click(mix_submit_click, [mix_model_output1,mix_mode], [mix_model_output2])
                
                with gr.TabItem("模型压缩工具"):
                    gr.Markdown(value="""
                        该工具可以实现对模型的体积压缩，在**不影响模型推理功能**的情况下，将原本约600M的So-VITS模型压缩至约200M, 大大减少了硬盘的压力。
                        **注意：压缩后的模型将无法继续训练，请在确认封炉后再压缩。**
                    """)
                    model_to_compress = gr.File(label="模型上传")
                    compress_model_btn = gr.Button("压缩模型", variant="primary")
                    compress_model_output = gr.Textbox(label="输出信息", value="")

                    compress_model_btn.click(model_compression, [model_to_compress], [compress_model_output])
                    
                    
    with gr.Tabs():
        with gr.Row(variant="panel"):
            with gr.Column():
                gr.Markdown(value="""
                    <font size=2> WebUI设置</font>
                    """)
                debug_button = gr.Checkbox(label="Debug模式，如果向社区反馈BUG需要打开，打开后控制台可以显示具体错误提示", value=debug)
        # refresh local model list
        local_model_refresh_btn.click(local_model_refresh_fn, outputs=local_model_selection)
        # set local enabled/disabled on tab switch
        local_model_tab_upload.select(lambda: False, outputs=local_model_enabled)
        local_model_tab_local.select(lambda: True, outputs=local_model_enabled)
        
        vc_submit.click(vc_fn, [sid, vc_input3, output_format, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment], [vc_output1, vc_output2])
        vc_submit2.click(vc_fn2, [text2tts, tts_lang, tts_gender, tts_rate, tts_volume, sid, output_format, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment], [vc_output1, vc_output2])

        debug_button.change(debug_change,[],[])
        model_load_button.click(modelAnalysis,[model_path,config_path,cluster_model_path,device,enhance,diff_model_path,diff_config_path,only_diffusion,use_spk_mix,local_model_enabled,local_model_selection],[sid,sid_output])
        model_unload_button.click(modelUnload,[],[sid,sid_output])
    os.system("start http://127.0.0.1:7860")
    app.launch()


 

```

---
name: Default issue
about: 如果模板中没有你想发起的issue类型，可以选择此项，但这个issue也许会获得一个较低的处理优先级 / If there is no issue type you want to raise, you can start with this one. But this issue maybe will get a lower priority to deal with.
title: ''
labels: 'not urgent'
assignees: ''
---


# `cluster/kmeans.py`

`kmeans++` is a popular clustering algorithm that aims to improve the k-means method by reducing the time required for the convergence of the algorithm. It is particularly useful for unstructured or high-dimensional data, where the k-means method may converge slowly or even fail to converge altogether.

The `kmeans++` method has several parameters that control its functionality:

* `data`: The input data is expected to be a rank 1 or 2 array. In the case of 1-dimensional data, it is expected to hold a single observation.
* `k`: The number of samples to generate.
* `sample_size`: The number of samples to avoid memory overflow during calculation.

The function returns an initial array of `k` centroids.

The function relies on the following references:

* [1] D. Arthur and S. Vassilvitskii, "k-means++: the advantages of careful seeding", Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms, 2007.
* [2] scipy/cluster/vq.py: `_kpp` (k-means++).


```py
from time import time

import numpy as np
import pynvml
import torch
from torch.nn.functional import normalize


# device=torch.device("cuda:0")
def _kpp(data: torch.Tensor, k: int, sample_size: int = -1):
    """ Picks k points in the data based on the kmeans++ method.

    Parameters
    ----------
    data : torch.Tensor
        Expect a rank 1 or 2 array. Rank 1 is assumed to describe 1-D
        data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    sample_size : int
        sample data to avoid memory overflow during calculation

    Returns
    -------
    init : ndarray
        A 'k' by 'N' containing the initial centroids.

    References
    ----------
    .. [1] D. Arthur and S. Vassilvitskii, "k-means++: the advantages of
       careful seeding", Proceedings of the Eighteenth Annual ACM-SIAM Symposium
       on Discrete Algorithms, 2007.
    .. [2] scipy/cluster/vq.py: _kpp
    """
    batch_size=data.shape[0]
    if batch_size>sample_size:
        data = data[torch.randint(0, batch_size,[sample_size], device=data.device)]
    dims = data.shape[1] if len(data.shape) > 1 else 1
    init = torch.zeros((k, dims)).to(data.device)
    r = torch.distributions.uniform.Uniform(0, 1)
    for i in range(k):
        if i == 0:
            init[i, :] = data[torch.randint(data.shape[0], [1])]
        else:
            D2 = torch.cdist(init[:i, :][None, :], data[None, :], p=2)[0].amin(dim=0)
            probs = D2 / torch.sum(D2)
            cumprobs = torch.cumsum(probs, dim=0)
            init[i, :] = data[torch.searchsorted(cumprobs, r.sample([1]).to(data.device))]
    return init
```



This is a function that performs cluster k-means on data points `X` and returns the cluster with the closest similarity to the data point. The similarity is measured using the Euclidean distance between the data point and each other cluster in the dataset.

The function takes in data point `X` as input and performs k-means clustering on it, using the k-means algorithm to group the data points into k clusters based on their similarity. The clusters are identified using the `k-means` function from the `torchvision.models.clOTExtract` library, which is trained to perform cluster k-means on data points.

The function returns a tuple of the cluster with the closest similarity to the input data point and a list of counts of the data points that belong to that cluster.


```py
class KMeansGPU:
  '''
  Kmeans clustering algorithm implemented with PyTorch

  Parameters:
    n_clusters: int, 
      Number of clusters

    max_iter: int, default: 100
      Maximum number of iterations

    tol: float, default: 0.0001
      Tolerance
    
    verbose: int, default: 0
      Verbosity

    mode: {'euclidean', 'cosine'}, default: 'euclidean'
      Type of distance measure
      
    init_method: {'random', 'point', '++'}
      Type of initialization

    minibatch: {None, int}, default: None
      Batch size of MinibatchKmeans algorithm
      if None perform full KMeans algorithm
      
  Attributes:
    centroids: torch.Tensor, shape: [n_clusters, n_features]
      cluster centroids
  '''
  def __init__(self, n_clusters, max_iter=200, tol=1e-4, verbose=0, mode="euclidean",device=torch.device("cuda:0")):
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.tol = tol
    self.verbose = verbose
    self.mode = mode
    self.device=device
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
    info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    self.minibatch=int(33e6/self.n_clusters*info.free/ 1024 / 1024 / 1024)
    print("free_mem/GB:",info.free/ 1024 / 1024 / 1024,"minibatch:",self.minibatch)
    
  @staticmethod
  def cos_sim(a, b):
    """
      Compute cosine similarity of 2 sets of vectors

      Parameters:
      a: torch.Tensor, shape: [m, n_features]

      b: torch.Tensor, shape: [n, n_features]
    """
    return normalize(a, dim=-1) @ normalize(b, dim=-1).transpose(-2, -1)

  @staticmethod
  def euc_sim(a, b):
    """
      Compute euclidean similarity of 2 sets of vectors
      Parameters:
      a: torch.Tensor, shape: [m, n_features]
      b: torch.Tensor, shape: [n, n_features]
    """
    return 2 * a @ b.transpose(-2, -1) -(a**2).sum(dim=1)[..., :, None] - (b**2).sum(dim=1)[..., None, :]

  def max_sim(self, a, b):
    """
      Compute maximum similarity (or minimum distance) of each vector
      in a with all of the vectors in b
      Parameters:
      a: torch.Tensor, shape: [m, n_features]
      b: torch.Tensor, shape: [n, n_features]
    """
    if self.mode == 'cosine':
      sim_func = self.cos_sim
    elif self.mode == 'euclidean':
      sim_func = self.euc_sim
    sim = sim_func(a, b)
    max_sim_v, max_sim_i = sim.max(dim=-1)
    return max_sim_v, max_sim_i

  def fit_predict(self, X):
    """
      Combination of fit() and predict() methods.
      This is faster than calling fit() and predict() seperately.
      Parameters:
      X: torch.Tensor, shape: [n_samples, n_features]
      centroids: {torch.Tensor, None}, default: None
        if given, centroids will be initialized with given tensor
        if None, centroids will be randomly chosen from X
      Return:
      labels: torch.Tensor, shape: [n_samples]

            mini_=33kk/k*remain
            mini=min(mini_,fea_shape)
            offset=log2(k/1000)*1.5
            kpp_all=min(mini_*10/offset,fea_shape)
            kpp_sample=min(mini_/12/offset,fea_shape)
    """
    assert isinstance(X, torch.Tensor), "input must be torch.Tensor"
    assert X.dtype in [torch.half, torch.float, torch.double], "input must be floating point"
    assert X.ndim == 2, "input must be a 2d tensor with shape: [n_samples, n_features] "
    # print("verbose:%s"%self.verbose)

    offset = np.power(1.5,np.log(self.n_clusters / 1000))/np.log(2)
    with torch.no_grad():
      batch_size= X.shape[0]
      # print(self.minibatch, int(self.minibatch * 10 / offset), batch_size)
      start_time = time()
      if (self.minibatch*10//offset< batch_size):
        x = X[torch.randint(0, batch_size,[int(self.minibatch*10/offset)])].to(self.device)
      else:
        x = X.to(self.device)
      # print(x.device)
      self.centroids = _kpp(x, self.n_clusters, min(int(self.minibatch/12/offset),batch_size))
      del x
      torch.cuda.empty_cache()
      # self.centroids = self.centroids.to(self.device)
      num_points_in_clusters = torch.ones(self.n_clusters, device=self.device, dtype=X.dtype)#全1
      closest = None#[3098036]#int64
      if(self.minibatch>=batch_size//2 and self.minibatch<batch_size):
        X = X[torch.randint(0, batch_size,[self.minibatch])].to(self.device)
      elif(self.minibatch>=batch_size):
        X=X.to(self.device)
      for i in range(self.max_iter):
        iter_time = time()
        if self.minibatch<batch_size//2:#可用minibatch数太小，每次都得从内存倒腾到显存
          x = X[torch.randint(0, batch_size, [self.minibatch])].to(self.device)
        else:#否则直接全部缓存
          x = X

        closest = self.max_sim(a=x, b=self.centroids)[1].to(torch.int16)#[3098036]#int64#0~999
        matched_clusters, counts = closest.unique(return_counts=True)#int64#1k
        expanded_closest = closest[None].expand(self.n_clusters, -1)#[1000, 3098036]#int16#0~999
        mask = (expanded_closest==torch.arange(self.n_clusters, device=self.device)[:, None]).to(X.dtype)#==后者是int64*1000
        c_grad = mask @ x / mask.sum(-1)[..., :, None]
        c_grad[c_grad!=c_grad] = 0 # remove NaNs
        error = (c_grad - self.centroids).pow(2).sum()
        if self.minibatch is not None:
          lr = 1/num_points_in_clusters[:,None] * 0.9 + 0.1
        else:
          lr = 1
        matched_clusters=matched_clusters.long()
        num_points_in_clusters[matched_clusters] += counts#IndexError: tensors used as indices must be long, byte or bool tensors
        self.centroids = self.centroids * (1-lr) + c_grad * lr
        if self.verbose >= 2:
          print('iter:', i, 'error:', error.item(), 'time spent:', round(time()-iter_time, 4))
        if error <= self.tol:
          break

      if self.verbose >= 1:
        print(f'used {i+1} iterations ({round(time()-start_time, 4)}s) to cluster {batch_size} items into {self.n_clusters} clusters')
    return closest

```

# `cluster/train_cluster.py`

这段代码使用了多个第三方库来实现不同的任务。具体解释如下：

1. `argparse` 库用于解析命令行参数。在这个例子中，我们使用 `argparse.ArgumentParser` 来定义命令行参数。

2. `logging` 库用于输出日志信息。我们使用 `logging.getLogger` 函数来获取当前日志名称，然后使用 `logging.INFO` 级别来设置日志格式。

3. `os` 库用于操作系统相关的操作。我们使用 `Path` 类来获取文件或目录路径。

4. `time` 库用于时间相关的操作。我们使用 `time.time_str` 函数来获取当前时间，并使用 `time.sleep` 函数来暂停执行一段时间。

5. `torch` 库是一个用于机器学习的开源库。我们使用 `torch` 库来执行机器学习任务。

6. `tqdm` 库是一个用于可视化数据爬取进度的工具。我们使用 `tqdm` 库来实时跟踪数据进度。

7. `sklearn` 库是一个用于机器学习的开源库。我们使用 `sklearn.cluster` 库来实现聚类任务，使用 `MiniBatchKMeans` 类来实现快速聚类。

8. `numpy` 库是一个用于科学计算的开源库。我们使用 `numpy` 库来执行数值计算任务。

9. `pathlib` 库是一个用于Python中路径处理的库。我们使用 `Pathlib` 类来获取文件或目录路径。

这段代码的具体作用是执行一个聚类任务，使用机器学习算法来对数据进行聚类。在任务执行过程中，会使用 `tqdm` 库来实时跟踪数据进度，使用 `logging` 库来输出日志信息。


```py
import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
from kmeans import KMeansGPU
from sklearn.cluster import KMeans, MiniBatchKMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

```

This is a Python script that uses the PyTorch library to perform clustering on feature data. Here's a brief description of how it works:

1. The script takes in a directory path (`in_dir`) and a feature file name (`name`).
2. The script reads in the feature file and loads it into memory.
3. The script creates a list of the features, ensuring that each feature is squared and converted to a NumPy array.
4. The features are concatenated and converted to a NumPy array.
5. The script uses PyTorch's `torch.load()` function to load the feature files as binary data. The `map_location` argument maps the data to memory on the CPU, which is faster than reading from disk.
6. The `squeeze()` method is used to remove any singleton dimensions from the feature arrays.
7. The script uses PyTorch's `MiniBatchKMeans` class for k-means clustering. This is done by passing in the number of clusters (`n_clusters`), the batch size (`batch_size`), and the maximum number of iterations (`max_iter`).
8. If the `use_gpu` argument is `False`, the script uses the `KMeans` class. This is done by passing in the number of clusters and the verbosity level (`verbose`).
9. The script uses the `to()` method to convert the features to the device specified by the `device` parameter (either `'cpu'` or `'cuda'`).
10. The script uses the `astype()` method to convert the features to a NumPy array.
11. The script uses the `time.time()` function to measure the time taken for each step of the clustering process.
12. The script uses the `MiniBatchKMeans` class again if `use_gpu` is `False`, for better performance.

Note: This script assumes that the data files are binary format, and that the `use_minibatch` and `use_gpu` arguments are not passed.


```py
def train_cluster(in_dir, n_clusters, use_minibatch=True, verbose=False,use_gpu=False):#gpu_minibatch真拉，虽然库支持但是也不考虑
    if str(in_dir).endswith(".ipynb_checkpoints"):
        logger.info(f"Ignore {in_dir}")

    logger.info(f"Loading features from {in_dir}")
    features = []
    nums = 0
    for path in tqdm.tqdm(in_dir.glob("*.soft.pt")):
    # for name in os.listdir(in_dir):
    #     path="%s/%s"%(in_dir,name)
        features.append(torch.load(path,map_location="cpu").squeeze(0).numpy().T)
        # print(features[-1].shape)
    features = np.concatenate(features, axis=0)
    print(nums, features.nbytes/ 1024**2, "MB , shape:",features.shape, features.dtype)
    features = features.astype(np.float32)
    logger.info(f"Clustering features of shape: {features.shape}")
    t = time.time()
    if(use_gpu is False):
        if use_minibatch:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters,verbose=verbose, batch_size=4096, max_iter=80).fit(features)
        else:
            kmeans = KMeans(n_clusters=n_clusters,verbose=verbose).fit(features)
    else:
            kmeans = KMeansGPU(n_clusters=n_clusters, mode='euclidean', verbose=2 if verbose else 0,max_iter=500,tol=1e-2)#
            features=torch.from_numpy(features)#.to(device)
            kmeans.fit_predict(features)#

    print(time.time()-t, "s")

    x = {
            "n_features_in_": kmeans.n_features_in_ if use_gpu is False else features.shape[1],
            "_n_threads": kmeans._n_threads if use_gpu is False else 4,
            "cluster_centers_": kmeans.cluster_centers_ if use_gpu is False else kmeans.centroids.cpu().numpy(),
    }
    print("end")

    return x

```

这段代码是一个Python程序，用于创建一个聚类模型，用于图像分割任务。程序中定义了一些参数，用于控制聚类模型的配置和训练过程。

具体来说，这段代码实现了一个基于K-means聚类算法的图像分割模型。使用了PyTorch的`argparse`模块来解析用户输入的参数，使用`torch.save`函数来保存聚类结果。程序中定义了几个参数，包括：

- `--dataset`：聚类算法的训练数据目录，这里使用了根目录下的一个名为“44k”的文件夹作为训练数据，训练数据被划分成了44个聚类；
- `--output`：聚类算法的模型输出目录，这里将聚类结果保存为log文件；
- `--gpu`：是否使用GPU进行计算，default为False，表示不使用；
- `n_clusters`：聚类算法的聚类数量，default值为10000。

程序的主要部分是一个if语句，只要用户输入了`--gpu`参数，则会使用GPU进行计算。在主程序运行时，首先会定义一些变量，包括：

- `args`：`argparse`模块解析的用户输入参数；
- `dataset`：训练数据的目录，这里与`--dataset`参数相同；
- `use_gpu`：指示是否使用GPU进行计算，default为False；
- `n_clusters`：聚类算法的聚类数量，default值为10000。

然后，定义了一个`checkpoint_dir`变量，用于保存聚类算法的模型，这里保存了每个聚类的`kmeans`结果，并使用`torch.save`函数将它们保存到了指定目录中。最后，如果用户输入了`--gpu`参数，则会创建一个名为“kmeans_{n_clusters}-{dataset}”的文件夹，并将聚类结果保存到该文件夹中。


```py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default="./dataset/44k",
                        help='path of training data directory')
    parser.add_argument('--output', type=Path, default="logs/44k",
                        help='path of model output directory')
    parser.add_argument('--gpu',action='store_true', default=False ,
                        help='to use GPU')


    args = parser.parse_args()

    checkpoint_dir = args.output
    dataset = args.dataset
    use_gpu = args.gpu
    n_clusters = 10000
    
    ckpt = {}
    for spk in os.listdir(dataset):
        if os.path.isdir(dataset/spk):
            print(f"train kmeans for {spk}...")
            in_dir = dataset/spk
            x = train_cluster(in_dir, n_clusters,use_minibatch=False,verbose=False,use_gpu=use_gpu)
            ckpt[spk] = x

    checkpoint_path = checkpoint_dir / f"kmeans_{n_clusters}.pt"
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(
        ckpt,
        checkpoint_path,
    )
    

```

# `cluster/__init__.py`

这段代码的作用是定义了一个名为 `get_cluster_model` 的函数，它接受一个保存模型 checkpoint 的文件 path，然后将其加载到内存中。这个函数使用了一个自定义的 `KMeans` 类，可能是一个机器学习中的聚类算法。

`get_cluster_model` 函数首先使用 `torch.load` 函数加载 checkpoint 文件，这个函数将下载的模型保存到本地，并返回一个 PyTorch 模型对象。然后，它遍历 checkpoint 中的所有模型，每个模型是一个 `KMeans` 类，并从每个模型中提取一些特征，将这些特征存回到一个字典中，并将模型本身存储为字典的键。最终，它将所有模型和对应的特征存回一个字典中，并返回这个字典。

由于 `KMeans` 类的参数 `n_features_in_` 在每个模型中是固定的，因此 `get_cluster_model` 函数返回的格式是固定的，它只是一个将 `KMeans` 模型加载到内存中的函数，并不对模型的参数进行更改。


```py
import torch
from sklearn.cluster import KMeans


def get_cluster_model(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    kmeans_dict = {}
    for spk, ckpt in checkpoint.items():
        km = KMeans(ckpt["n_features_in_"])
        km.__dict__["n_features_in_"] = ckpt["n_features_in_"]
        km.__dict__["_n_threads"] = ckpt["_n_threads"]
        km.__dict__["cluster_centers_"] = ckpt["cluster_centers_"]
        kmeans_dict[spk] = km
    return kmeans_dict

```

这段代码定义了三个函数，用于不同的 Cluster Model。第一个函数 get_cluster_result() 接受一个二维数组 x 和一个特定说话者的整数列表，它预测模型对给定说话者的 x 的值。第二个函数 get_cluster_center_result() 也接受 x 和说话者的整数列表，它返回模型中对应说话者的聚类中心。第三个函数 get_center() 接受一个特定说话者的整数列表，它返回对应说话者的聚类中心。这三个函数都使用 model 和 x 作为输入，并返回模型中对应说话者的聚类中心或结果。


```py
def get_cluster_result(model, x, speaker):
    """
        x: np.array [t, 256]
        return cluster class result
    """
    return model[speaker].predict(x)

def get_cluster_center_result(model, x,speaker):
    """x: np.array [t, 256]"""
    predict = model[speaker].predict(x)
    return model[speaker].cluster_centers_[predict]

def get_center(model, x,speaker):
    return model[speaker].cluster_centers_[x]

```

# `diffusion/data_loaders.py`

这段代码是一个PyTorch实现的库，用于对音频文件进行处理和分析。它包括了以下主要功能：

1. 导入必要的库，包括os、random、librosa、numpy、torch以及torch.utils.data和tqdm。
2. 从librosa库导入了一组音频处理函数，包括rep Air、speed、sleep、cut_噪音等。
3. 从numpy库导入了一组数学函数，包括sum、mean、max、min等。
4. 从torch库导入了一组数据处理函数，包括Tensor、Slice、Gather等。
5. 定义了一个名为traverse_dir的函数，它接收一个根目录、文件类型扩展名、每个文件最大数量、包含的文件扩展名、是否按照文件名排序、是否按文件大小排序等参数。
6. 在函数中使用os.walk遍历目录中的所有文件，并将每个文件的路径存储在file_list中。
7. 如果定义了文件数量的数量，那么函数将只取指定数量的文件。
8. 对于每个文件，函数会根据所定义的文件扩展名来切分文件路径，并将file_list、文件数量和当前文件路径保存到全局变量中。
9. 在函数中使用了tqdm库来对文件数量进行实时监控。


```py
import os
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import repeat_expand_2d


def traverse_dir(
        root_dir,
        extensions,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


```

It looks like you're defining two data loaders, one for training data and one for validation data. Both loaders use the `torch.utils.data.DataLoader` class to load the data in batches and return them to the `DataLoader` in PyTorch.

The main difference between the two loaders is in the batch size and the number of workers. The loader for training data has a batch size of 16 and a number of workers set to 0. This means that the data will be loaded from disk and the PyTorch audio will be loaded in a single batch at a time. This is done to improve the audio quality by avoiding parallel loading of data.

The loader for validation data has a batch size of 1 and a number of workers set to 1. This means that the data will be loaded from disk and the PyTorch audio will be loaded in a batch of 1 at a time. This is done to load the validation data in parallel with the training data in order to speed up the training process.

Note that the extension argument for the audio file is specified in the `args.data.extensions` parameter. This allows you to specify additional extensions to be included in the loaded audio data.


```py
def get_data_loaders(args, whole_audio=False):
    data_train = AudioDataset(
        filelists = args.data.training_files,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=whole_audio,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        spk=args.spk,
        device=args.train.cache_device,
        fp16=args.train.cache_fp16,
        unit_interpolate_mode = args.data.unit_interpolate_mode,
        use_aug=True)
    loader_train = torch.utils.data.DataLoader(
        data_train ,
        batch_size=args.train.batch_size if not whole_audio else 1,
        shuffle=True,
        num_workers=args.train.num_workers if args.train.cache_device=='cpu' else 0,
        persistent_workers=(args.train.num_workers > 0) if args.train.cache_device=='cpu' else False,
        pin_memory=True if args.train.cache_device=='cpu' else False
    )
    data_valid = AudioDataset(
        filelists = args.data.validation_files,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        spk=args.spk,
        extensions=args.data.extensions,
        unit_interpolate_mode = args.data.unit_interpolate_mode,
        n_spk=args.model.n_spk)
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return loader_train, loader_valid 


```

This class seems to be a Python implementation of a file that contains multiple NIIINewton displacement maps, along with metadata such as the name and extension of each file.

The `Mel笛声_类` 似乎负责从数据缓冲区中读取f0值，并从路径中读取f0数据的逆采样率。它还从缓冲区中读取每张地图的逆时针数组，并将其转换为浮点数。

然后，它通过从缓冲区中获取 aug_vol 数据来填充每个地图的逆时针数组。

此外，它似乎使用 mel 和 f0 作为 NIIINewton 数据的核心，并使用数据缓冲区中的元数据（如文件名和扩展名）对数据进行命名。


```py
class AudioDataset(Dataset):
    def __init__(
        self,
        filelists,
        waveform_sec,
        hop_size,
        sample_rate,
        spk,
        load_all_data=True,
        whole_audio=False,
        extensions=['wav'],
        n_spk=1,
        device='cpu',
        fp16=False,
        use_aug=False,
        unit_interpolate_mode = 'left'
    ):
        super().__init__()
        
        self.waveform_sec = waveform_sec
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.filelists = filelists
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.data_buffer={}
        self.pitch_aug_dict = {}
        self.unit_interpolate_mode = unit_interpolate_mode
        # np.load(os.path.join(self.path_root, 'pitch_aug_dict.npy'), allow_pickle=True).item()
        if load_all_data:
            print('Load all the data filelists:', filelists)
        else:
            print('Load the f0, volume data filelists:', filelists)
        with open(filelists,"r") as f:
            self.paths = f.read().splitlines()
        for name_ext in tqdm(self.paths, total=len(self.paths)):
            path_audio = name_ext
            duration = librosa.get_duration(filename = path_audio, sr = self.sample_rate)
            
            path_f0 = name_ext + ".f0.npy"
            f0,_ = np.load(path_f0,allow_pickle=True)
            f0 = torch.from_numpy(np.array(f0,dtype=float)).float().unsqueeze(-1).to(device)
                
            path_volume = name_ext + ".vol.npy"
            volume = np.load(path_volume)
            volume = torch.from_numpy(volume).float().unsqueeze(-1).to(device)
            
            path_augvol = name_ext + ".aug_vol.npy"
            aug_vol = np.load(path_augvol)
            aug_vol = torch.from_numpy(aug_vol).float().unsqueeze(-1).to(device)
                        
            if n_spk is not None and n_spk > 1:
                spk_name = name_ext.split("/")[-2]
                spk_id = spk[spk_name] if spk_name in spk else 0
                if spk_id < 0 or spk_id >= n_spk:
                    raise ValueError(' [x] Muiti-speaker traing error : spk_id must be a positive integer from 0 to n_spk-1 ')
            else:
                spk_id = 0
            spk_id = torch.LongTensor(np.array([spk_id])).to(device)

            if load_all_data:
                '''
                audio, sr = librosa.load(path_audio, sr=self.sample_rate)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio)
                audio = torch.from_numpy(audio).to(device)
                '''
                path_mel = name_ext + ".mel.npy"
                mel = np.load(path_mel)
                mel = torch.from_numpy(mel).to(device)
                
                path_augmel = name_ext + ".aug_mel.npy"
                aug_mel,keyshift = np.load(path_augmel, allow_pickle=True)
                aug_mel = np.array(aug_mel,dtype=float)
                aug_mel = torch.from_numpy(aug_mel).to(device)
                self.pitch_aug_dict[name_ext] = keyshift

                path_units = name_ext + ".soft.pt"
                units = torch.load(path_units).to(device)
                units = units[0]  
                units = repeat_expand_2d(units,f0.size(0),unit_interpolate_mode).transpose(0,1)
                
                if fp16:
                    mel = mel.half()
                    aug_mel = aug_mel.half()
                    units = units.half()
                    
                self.data_buffer[name_ext] = {
                        'duration': duration,
                        'mel': mel,
                        'aug_mel': aug_mel,
                        'units': units,
                        'f0': f0,
                        'volume': volume,
                        'aug_vol': aug_vol,
                        'spk_id': spk_id
                        }
            else:
                path_augmel = name_ext + ".aug_mel.npy"               
                aug_mel,keyshift = np.load(path_augmel, allow_pickle=True)
                self.pitch_aug_dict[name_ext] = keyshift
                self.data_buffer[name_ext] = {
                        'duration': duration,
                        'f0': f0,
                        'volume': volume,
                        'aug_vol': aug_vol,
                        'spk_id': spk_id
                        }
           

    def __getitem__(self, file_idx):
        name_ext = self.paths[file_idx]
        data_buffer = self.data_buffer[name_ext]
        # check duration. if too short, then skip
        if data_buffer['duration'] < (self.waveform_sec + 0.1):
            return self.__getitem__( (file_idx + 1) % len(self.paths))
            
        # get item
        return self.get_data(name_ext, data_buffer)

    def get_data(self, name_ext, data_buffer):
        name = os.path.splitext(name_ext)[0]
        frame_resolution = self.hop_size / self.sample_rate
        duration = data_buffer['duration']
        waveform_sec = duration if self.whole_audio else self.waveform_sec
        
        # load audio
        idx_from = 0 if self.whole_audio else random.uniform(0, duration - waveform_sec - 0.1)
        start_frame = int(idx_from / frame_resolution)
        units_frame_len = int(waveform_sec / frame_resolution)
        aug_flag = random.choice([True, False]) and self.use_aug
        '''
        audio = data_buffer.get('audio')
        if audio is None:
            path_audio = os.path.join(self.path_root, 'audio', name) + '.wav'
            audio, sr = librosa.load(
                    path_audio, 
                    sr = self.sample_rate, 
                    offset = start_frame * frame_resolution,
                    duration = waveform_sec)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            # clip audio into N seconds
            audio = audio[ : audio.shape[-1] // self.hop_size * self.hop_size]       
            audio = torch.from_numpy(audio).float()
        else:
            audio = audio[start_frame * self.hop_size : (start_frame + units_frame_len) * self.hop_size]
        '''
        # load mel
        mel_key = 'aug_mel' if aug_flag else 'mel'
        mel = data_buffer.get(mel_key)
        if mel is None:
            mel = name_ext + ".mel.npy"
            mel = np.load(mel)
            mel = mel[start_frame : start_frame + units_frame_len]
            mel = torch.from_numpy(mel).float() 
        else:
            mel = mel[start_frame : start_frame + units_frame_len]
            
        # load f0
        f0 = data_buffer.get('f0')
        aug_shift = 0
        if aug_flag:
            aug_shift = self.pitch_aug_dict[name_ext]
        f0_frames = 2 ** (aug_shift / 12) * f0[start_frame : start_frame + units_frame_len]
        
        # load units
        units = data_buffer.get('units')
        if units is None:
            path_units = name_ext + ".soft.pt"
            units = torch.load(path_units)
            units = units[0]  
            units = repeat_expand_2d(units,f0.size(0),self.unit_interpolate_mode).transpose(0,1)
            
        units = units[start_frame : start_frame + units_frame_len]

        # load volume
        vol_key = 'aug_vol' if aug_flag else 'volume'
        volume = data_buffer.get(vol_key)
        volume_frames = volume[start_frame : start_frame + units_frame_len]
        
        # load spk_id
        spk_id = data_buffer.get('spk_id')
        
        # load shift
        aug_shift = torch.from_numpy(np.array([[aug_shift]])).float()
        
        return dict(mel=mel, f0=f0_frames, volume=volume_frames, units=units, spk_id=spk_id, aug_shift=aug_shift, name=name, name_ext=name_ext)

    def __len__(self):
        return len(self.paths)
```

# `diffusion/diffusion.py`

这段代码的主要作用是创建一个名为 "model" 的函数对象，该函数对象通过 "add(x)" 方法向一个名为 "deque" 的 "collections" 集合中添加一个元素 "x"。

具体来说，这段代码首先定义了一个名为 "exists" 的函数，它接收一个参数 "x"，并返回 "x" 是否为真。接下来，定义了一个名为 "isfunction" 的函数，它接收一个参数 "x"，并返回 "x" 是否为函数类型。

接着，从 "collections" 导入了一个名为 "deque" 的函数，该函数可以创建一个 "deque" 对象。最后，从 "torch" 和 "torch.nn.functional" 导入了一些常用的函数和类，包括 "is None" 和 "nn.functional.linear"，以及 "torch.nn" 导入的 "nn.Module" 和 "nn.functional" 包中的 "function"。

总体来说，这段代码的主要目的是创建一个可以添加元素到 "deque" 集合中的函数对象，用于在训练过程中添加数据或梯度。


```py
from collections import deque
from functools import partial
from inspect import isfunction

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


def exists(x):
    return x is not None


```



这段代码定义了三个函数，分别是：

1. `default(val, d)`：该函数接收两个参数 `val` 和 `d`，如果 `val` 存在，则返回 `val`，否则返回 `d`。这里使用了 `isfunction` 函数来检查 `d` 是否为函数。

2. `extract(a, t, x_shape)`：该函数接收三个参数 `a`、`t` 和 `x_shape`。该函数的作用是提取 `a` 中与 `t` 中的元素匹配的行，并将它们与 `x_shape` 中的元素拼接成一个新的张量，然后将结果返回。

3. `noise_like(shape, device, repeat=False)`：该函数接收两个参数 `shape` 和 `device`，以及一个可选参数 `repeat`。该函数的作用是在 `device` 上循环噪声，并在每个噪声样本上进行随机变换。如果 `repeat` 为 `True`，则函数将循环 `shape` 中的所有噪声样本。

这三个函数的具体实现可以看作是在机器学习领域中的一些常见操作，例如数据预处理、数据扩充和数据随机化等。


```py
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    def noise():
        return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


```

这两段代码都是定义了函数 `linear_beta_schedule`，用于计算一个线性逐步增加的因子，它的值在 `timesteps` 参数的范围内从 1e-4 逐步增加到 `max_beta` 并返回该范围内的所有值。

对于第二个函数 `cosine_beta_schedule`，它定义了一个逐步增加的因子，使用的并不是线性增长，而是通过 `cosine` 函数计算出来的。具体来说，它计算了这样一个分数：`x / steps` 加上一个固定的常数 `s`，然后对结果进行 `cosine` 函数，再对结果进行逐步增加操作。这个操作一直持续到 `timesteps` 步，最后返回一个范围在 0 和 1 之间的值。


```py
def linear_beta_schedule(timesteps, max_beta=0.02):
    """
    linear schedule
    """
    betas = np.linspace(1e-4, max_beta, timesteps)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


```

This is a PyTorch implementation of a neural network model for image classification. The model has two hyperparameters, "x_start" and "v", which can be used to specify the initial position and velocity of the neural network. The model also has a scorer function, which is used to compute the score of each input image.

The model consists of two sub-models: a UniPC model and a sample-by-multistep model. The UniPC model is used for feature extraction, while the sample-by-multistep model is used for sampling the input images. The UniPC model has two overrides: "model\_fn" and "noise\_schedule". The "model\_fn" overrides are used to specify the function to be executed by the UniPC model, while the "noise\_schedule" overrides are used to specify the schedule for the noise samples.

The model also has a sample function, which is used to sample the input images. The "sample" function takes the input image, the steps parameter, and the condition parameter. The condition parameter is used to specify whether the function should perform a certain operation. If the condition is "x\_start", the function will start sampling from the specified position.

The model also has a "score" function, which is used to compute the score of each input image. This function is passed the output of the "sample" function and returns a tensor that represents the score. This tensor is then used to compute the final score of each input image.

Note that the implementation assumes that the input images have a shape of (batch\_size, batch\_size, input\_shape), where "batch\_size" is the size of the batch. The model also assumes that the input images are dtype long and are stored in memory in the device specified by the device\_kind parameter. If this is not the case, the model may require modifications to work correctly.


```py
beta_schedule = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
}


class GaussianDiffusion(nn.Module):
    def __init__(self, 
                denoise_fn, 
                out_dims=128,
                timesteps=1000, 
                k_step=1000,
                max_beta=0.02,
                spec_min=-12, 
                spec_max=2):
        
        super().__init__()
        self.denoise_fn = denoise_fn
        self.out_dims = out_dims
        betas = beta_schedule['linear'](timesteps, max_beta=max_beta)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.k_step = k_step if k_step>0 and k_step<timesteps else timesteps

        self.noise_list = deque(maxlen=4)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        self.register_buffer('spec_min', torch.FloatTensor([spec_min])[None, None, :out_dims])
        self.register_buffer('spec_max', torch.FloatTensor([spec_max])[None, None, :out_dims])

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond):
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_ddim(self, x, t, interval, cond):
        """
        Use the DDIM method from
        """
        a_t = extract(self.alphas_cumprod, t, x.shape)
        a_prev = extract(self.alphas_cumprod, torch.max(t - interval, torch.zeros_like(t)), x.shape)

        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_prev = a_prev.sqrt() * (x / a_t.sqrt() + (((1 - a_prev) / a_prev).sqrt()-((1 - a_t) / a_t).sqrt()) * noise_pred)
        return x_prev

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_plms(self, x, t, interval, cond, clip_denoised=True, repeat_noise=False):
        """
        Use the PLMS method from
        [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778).
        """

        def get_x_pred(x, noise_t, t):
            a_t = extract(self.alphas_cumprod, t, x.shape)
            a_prev = extract(self.alphas_cumprod, torch.max(t - interval, torch.zeros_like(t)), x.shape)
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

            x_delta = (a_prev - a_t) * ((1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x - 1 / (
                    a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
            x_pred = x + x_delta

            return x_pred

        noise_list = self.noise_list
        noise_pred = self.denoise_fn(x, t, cond=cond)

        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(x_pred, max(t - interval, 0), cond=cond)
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2
        elif len(noise_list) == 1:
            noise_pred_prime = (3 * noise_pred - noise_list[-1]) / 2
        elif len(noise_list) == 2:
            noise_pred_prime = (23 * noise_pred - 16 * noise_list[-1] + 5 * noise_list[-2]) / 12
        else:
            noise_pred_prime = (55 * noise_pred - 59 * noise_list[-1] + 37 * noise_list[-2] - 9 * noise_list[-3]) / 24

        x_prev = get_x_pred(x, noise_pred_prime, t)
        noise_list.append(noise_pred)

        return x_prev

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond, noise=None, loss_type='l2'):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond)

        if loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, 
                condition, 
                gt_spec=None, 
                infer=True, 
                infer_speedup=10, 
                method='dpm-solver',
                k_step=300,
                use_tqdm=True):
        """
            conditioning diffusion, use fastspeech2 encoder output as the condition
        """
        cond = condition.transpose(1, 2)
        b, device = condition.shape[0], condition.device

        if not infer:
            spec = self.norm_spec(gt_spec)
            t = torch.randint(0, self.k_step, (b,), device=device).long()
            norm_spec = spec.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            return self.p_losses(norm_spec, t, cond=cond)
        else:
            shape = (cond.shape[0], 1, self.out_dims, cond.shape[2])
            
            if gt_spec is None:
                t = self.k_step
                x = torch.randn(shape, device=device)
            else:
                t = k_step
                norm_spec = self.norm_spec(gt_spec)
                norm_spec = norm_spec.transpose(1, 2)[:, None, :, :]
                x = self.q_sample(x_start=norm_spec, t=torch.tensor([t - 1], device=device).long())
                        
            if method is not None and infer_speedup > 1:
                if method == 'dpm-solver' or method == 'dpm-solver++':
                    from .dpm_solver_pytorch import (
                        DPM_Solver,
                        NoiseScheduleVP,
                        model_wrapper,
                    )
                    # 1. Define the noise schedule.
                    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas[:t])

                    # 2. Convert your discrete-time `model` to the continuous-time
                    # noise prediction model. Here is an example for a diffusion model
                    # `model` with the noise prediction type ("noise") .
                    def my_wrapper(fn):
                        def wrapped(x, t, **kwargs):
                            ret = fn(x, t, **kwargs)
                            if use_tqdm:
                                self.bar.update(1)
                            return ret

                        return wrapped

                    model_fn = model_wrapper(
                        my_wrapper(self.denoise_fn),
                        noise_schedule,
                        model_type="noise",  # or "x_start" or "v" or "score"
                        model_kwargs={"cond": cond}
                    )

                    # 3. Define dpm-solver and sample by singlestep DPM-Solver.
                    # (We recommend singlestep DPM-Solver for unconditional sampling)
                    # You can adjust the `steps` to balance the computation
                    # costs and the sample quality.
                    if method == 'dpm-solver':
                        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")
                    elif method == 'dpm-solver++':
                        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                        
                    steps = t // infer_speedup
                    if use_tqdm:
                        self.bar = tqdm(desc="sample time step", total=steps)
                    x = dpm_solver.sample(
                        x,
                        steps=steps,
                        order=2,
                        skip_type="time_uniform",
                        method="multistep",
                    )
                    if use_tqdm:
                        self.bar.close()
                elif method == 'pndm':
                    self.noise_list = deque(maxlen=4)
                    if use_tqdm:
                        for i in tqdm(
                                reversed(range(0, t, infer_speedup)), desc='sample time step',
                                total=t // infer_speedup,
                        ):
                            x = self.p_sample_plms(
                                x, torch.full((b,), i, device=device, dtype=torch.long),
                                infer_speedup, cond=cond
                            )
                    else:
                        for i in reversed(range(0, t, infer_speedup)):
                            x = self.p_sample_plms(
                                x, torch.full((b,), i, device=device, dtype=torch.long),
                                infer_speedup, cond=cond
                            )
                elif method == 'ddim':
                    if use_tqdm:
                        for i in tqdm(
                                reversed(range(0, t, infer_speedup)), desc='sample time step',
                                total=t // infer_speedup,
                        ):
                            x = self.p_sample_ddim(
                                x, torch.full((b,), i, device=device, dtype=torch.long),
                                infer_speedup, cond=cond
                            )
                    else:
                        for i in reversed(range(0, t, infer_speedup)):
                            x = self.p_sample_ddim(
                                x, torch.full((b,), i, device=device, dtype=torch.long),
                                infer_speedup, cond=cond
                            )
                elif method == 'unipc':
                    from .uni_pc import NoiseScheduleVP, UniPC, model_wrapper
                    # 1. Define the noise schedule.
                    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas[:t])

                    # 2. Convert your discrete-time `model` to the continuous-time
                    # noise prediction model. Here is an example for a diffusion model
                    # `model` with the noise prediction type ("noise") .
                    def my_wrapper(fn):
                        def wrapped(x, t, **kwargs):
                            ret = fn(x, t, **kwargs)
                            if use_tqdm:
                                self.bar.update(1)
                            return ret

                        return wrapped

                    model_fn = model_wrapper(
                        my_wrapper(self.denoise_fn),
                        noise_schedule,
                        model_type="noise",  # or "x_start" or "v" or "score"
                        model_kwargs={"cond": cond}
                    )

                    # 3. Define uni_pc and sample by multistep UniPC.
                    # You can adjust the `steps` to balance the computation
                    # costs and the sample quality.
                    uni_pc = UniPC(model_fn, noise_schedule, variant='bh2')

                    steps = t // infer_speedup
                    if use_tqdm:
                        self.bar = tqdm(desc="sample time step", total=steps)
                    x = uni_pc.sample(
                        x,
                        steps=steps,
                        order=2,
                        skip_type="time_uniform",
                        method="multistep",
                    )
                    if use_tqdm:
                        self.bar.close()
                else:
                    raise NotImplementedError(method)
            else:
                if use_tqdm:
                    for i in tqdm(reversed(range(0, t)), desc='sample time step', total=t):
                        x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
                else:
                    for i in reversed(range(0, t)):
                        x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
            x = x.squeeze(1).transpose(1, 2)  # [B, T, M]
            return self.denorm_spec(x)

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

```