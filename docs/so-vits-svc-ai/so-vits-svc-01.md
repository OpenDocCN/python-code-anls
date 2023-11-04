# SO-VITS-SVC源码解析 1

# `onnx_export_old.py`

This is a code snippet that uses the ONNX version of the SVCVITS model training script. It exports the model trained on the Onnx platform, and can be used to deploy the model for deployment.

The model is trained to predict audio data, with the input data being 2D arrays of audio features (e.g. mel-Frequency Cepstral Coefficients), followed by a 3D array of noise. The model outputs a 1D array of audio data, which can be used to play the audio.

The model uses the SVCVITS pre-trained model as the backbone, and adds some additional layers to perform the predictions. The model has a batch size of 1, and runs on a GPU for training.

The model is exported as an ONNX model, and can be loaded and run using the ONNX dynamic load/run method.


```py
import torch

import utils
from onnxexport.model_onnx import SynthesizerTrn


def main(NetExport):
    path = "SoVits4.0"
    if NetExport:
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
        
        n_frame = 10
        test_hidden_unit = torch.rand(1, n_frame, 256)
        test_pitch = torch.rand(1, n_frame)
        test_mel2ph = torch.arange(0, n_frame, dtype=torch.int64)[None] # torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).unsqueeze(0)
        test_uv = torch.ones(1, n_frame, dtype=torch.float32)
        test_noise = torch.randn(1, 192, n_frame)
        test_sid = torch.LongTensor([0])
        input_names = ["c", "f0", "mel2ph", "uv", "noise", "sid"]
        output_names = ["audio", ]
        
        torch.onnx.export(SVCVITS,
                          (
                              test_hidden_unit.to(device),
                              test_pitch.to(device),
                              test_mel2ph.to(device),
                              test_uv.to(device),
                              test_noise.to(device),
                              test_sid.to(device)
                          ),
                          f"checkpoints/{path}/model.onnx",
                          dynamic_axes={
                              "c": [0, 1],
                              "f0": [1],
                              "mel2ph": [1],
                              "uv": [1],
                              "noise": [2],
                          },
                          do_constant_folding=False,
                          opset_version=16,
                          verbose=False,
                          input_names=input_names,
                          output_names=output_names)


```

这段代码是一个Python程序中的一个if语句，其作用是在程序运行时执行if语句中的语句，如果程序是作为程序的主程序(即'__main__')运行，则会先执行if语句中的语句，如果条件为真，则会执行if语句中的语句块，否则跳过if语句。

在这个例子中，if语句中的语句为main(True)，传递给程序的参数为True。main函数是一个通常用于在Python应用程序中进行更多的操作的函数，但在这种情况下，它被用作一个测试函数。

因此，这段代码的作用是测试main函数是否成功接受True参数，如果main函数接受True参数并且程序是作为程序的主程序运行，那么if语句块将会执行main(True)函数，否则程序将跳过if语句并继续运行。


```py
if __name__ == '__main__':
    main(True)

```

# `preprocess_flist_config.py`

这段代码的作用是读取一个指定路径的WAV文件，并将其时长（秒）显示出来。

具体来说，它使用了`argparse`模块来解析命令行参数，`json`和`os`模块用于读取和操作文件，`re`模块用于正则表达式匹配，`wave`模块用于读取WAV文件，`random`模块用于从列表中随机选择元素，`diffusion.logger.utils`模块中的`du`函数用于记录日志信息。

接下来，它定义了一个`get_wav_duration`函数，该函数接受一个文件路径参数，返回WAV文件的时长。函数首先使用`wave.open`函数打开WAV文件，然后使用`getnframes`函数获取文件中的帧数，接着使用`getframerate`函数获取采样率，最后使用`n_frames / float(framerate)`计算出时长。

如果函数在执行过程中遇到错误，则会记录日志信息并抛出异常。


```py
import argparse
import json
import os
import re
import wave
from random import shuffle

from loguru import logger
from tqdm import tqdm

import diffusion.logger.utils as du

pattern = re.compile(r'^[\.a-zA-Z0-9_\/]+$')

def get_wav_duration(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            # 获取音频帧数
            n_frames = wav_file.getnframes()
            # 获取采样率
            framerate = wav_file.getframerate()
            # 计算时长（秒）
            return n_frames / float(framerate)
    except Exception as e:
        logger.error(f"Reading {file_path}")
        raise e

```

This is a Python script that generates a configuration file for a speech recognition model based on the parameters passed in the command-line. The script has the following functions:

* `build_config_template`: This function takes a dictionary of parameters and uses them to configure the model and write the configuration template to a file. The file is written in JSON format and includes the model architecture, filter channels, and other relevant information.
* `build_d_config_template`: This function takes the same parameters as `build_config_template` and writes the data dictionary to a file.
* `build_model_config`: This function takes the same parameters as `build_config_template` and writes the model configuration to a file.
* `setup_wavlmb`: This function sets the model architecture based on the speech encoder used. It checks whether the encoder is a) wavlmb, b) vec256l9, c) hubertsoft, d) whisper-ppg, or e) whisper-ppg-large.
* `setup_small_model`: This function sets a small model architecture for comparison with larger models.
* `setup_volume_aug`: This function sets the augmentation flag for the volume data.
* `build_configs`: This function calls the `build_config_template` function multiple times to build the main configuration file and a dependency file.
* `main`: This function starts the script and calls the `setup_volume_aug` function to set the augmentation flag. It then calls the `build_configs` function to build the main configuration file. Finally, it logs information about the script and writes the configs to the files specified by the arguments.


```py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="./filelists/train.txt", help="path to train list")
    parser.add_argument("--val_list", type=str, default="./filelists/val.txt", help="path to val list")
    parser.add_argument("--source_dir", type=str, default="./dataset/44k", help="path to source dir")
    parser.add_argument("--speech_encoder", type=str, default="vec768l12", help="choice a speech encoder|'vec768l12','vec256l9','hubertsoft','whisper-ppg','cnhubertlarge','dphubert','whisper-ppg-large','wavlmbase+'")
    parser.add_argument("--vol_aug", action="store_true", help="Whether to use volume embedding and volume augmentation")
    parser.add_argument("--tiny", action="store_true", help="Whether to train sovits tiny")
    args = parser.parse_args()
    
    config_template =  json.load(open("configs_template/config_tiny_template.json")) if args.tiny else json.load(open("configs_template/config_template.json"))
    train = []
    val = []
    idx = 0
    spk_dict = {}
    spk_id = 0

    for speaker in tqdm(os.listdir(args.source_dir)):
        spk_dict[speaker] = spk_id
        spk_id += 1
        wavs = []

        for file_name in os.listdir(os.path.join(args.source_dir, speaker)):
            if not file_name.endswith("wav"):
                continue
            if file_name.startswith("."):
                continue

            file_path = "/".join([args.source_dir, speaker, file_name])

            if not pattern.match(file_name):
                logger.warning("Detected non-ASCII file name: " + file_path)

            if get_wav_duration(file_path) < 0.3:
                logger.info("Skip too short audio: " + file_path)
                continue

            wavs.append(file_path)

        shuffle(wavs)
        train += wavs[2:]
        val += wavs[:2]

    shuffle(train)
    shuffle(val)

    logger.info("Writing " + args.train_list)
    with open(args.train_list, "w") as f:
        for fname in tqdm(train):
            wavpath = fname
            f.write(wavpath + "\n")

    logger.info("Writing " + args.val_list)
    with open(args.val_list, "w") as f:
        for fname in tqdm(val):
            wavpath = fname
            f.write(wavpath + "\n")


    d_config_template = du.load_config("configs_template/diffusion_template.yaml")
    d_config_template["model"]["n_spk"] = spk_id
    d_config_template["data"]["encoder"] = args.speech_encoder
    d_config_template["spk"] = spk_dict
    
    config_template["spk"] = spk_dict
    config_template["model"]["n_speakers"] = spk_id
    config_template["model"]["speech_encoder"] = args.speech_encoder
    
    if args.speech_encoder == "vec768l12" or args.speech_encoder == "dphubert" or args.speech_encoder == "wavlmbase+":
        config_template["model"]["ssl_dim"] = config_template["model"]["filter_channels"] = config_template["model"]["gin_channels"] = 768
        d_config_template["data"]["encoder_out_channels"] = 768
    elif args.speech_encoder == "vec256l9" or args.speech_encoder == 'hubertsoft':
        config_template["model"]["ssl_dim"] = config_template["model"]["gin_channels"] = 256
        d_config_template["data"]["encoder_out_channels"] = 256
    elif args.speech_encoder == "whisper-ppg" or args.speech_encoder == 'cnhubertlarge':
        config_template["model"]["ssl_dim"] = config_template["model"]["filter_channels"] = config_template["model"]["gin_channels"] = 1024
        d_config_template["data"]["encoder_out_channels"] = 1024
    elif args.speech_encoder == "whisper-ppg-large":
        config_template["model"]["ssl_dim"] = config_template["model"]["filter_channels"] = config_template["model"]["gin_channels"] = 1280
        d_config_template["data"]["encoder_out_channels"] = 1280
        
    if args.vol_aug:
        config_template["train"]["vol_aug"] = config_template["model"]["vol_embedding"] = True

    if args.tiny:
        config_template["model"]["filter_channels"] = 512

    logger.info("Writing to configs/config.json")
    with open("configs/config.json", "w") as f:
        json.dump(config_template, f, indent=2)
    logger.info("Writing to configs/diffusion.yaml")
    du.save_config("configs/diffusion.yaml",d_config_template)

```

# `preprocess_hubert_f0.py`

这段代码使用了多个第三方库，包括 PyTorch，librosa，和 loguru。它们的作用是：

* 导入argparse库以便在命令行中使用参数parser
* 导入logging库以便在应用程序中记录输出
* 导入os库以便在命令行中与操作系统交互
* 导入random库以便在应用程序中生成随机数
* 从glob库中导入一组音频文件，这些文件将用于训练神经网络
* 从librosa库中导入shuffle函数以便对音频数据进行随机化
* 通过将音频数据分成训练集和测试集来对神经网络进行训练
* 使用concurrent.futures库中的ProcessPoolExecutor函数来并行处理大量的数据
* 使用tqdm库中的tqdm函数来显示进度的百分比

此外，运行这段代码的应用程序还具有以下功能：

* 在训练期间，它将在每个音频文件上应用shuffle函数，以便在训练神经网络时获得更好的数据分布
* 使用librosa库中的static函数来加载预录制好的音频数据，而不是从网络中下载它们
* 在命令行中运行应用程序时，它将使用默认设置，包括在应用程序中指定要训练的神经网络架构和要使用的音频数据。


```py
import argparse
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from random import shuffle

import librosa
import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

```

这段代码的作用是：

1. 导入扩散模型（Diffusion Model）中的日志工具类（Logger）和用户配置类（User Config）中的`configs`文件夹。
2. 导入日志工具类中的`get_hparams_from_file`函数，从指定的配置文件中读取用户参数。
3. 导入`vocoder`类，从`diffusion.vocoder`包中继承并重新实现了`Vocoder`类。
4. 导入` mel_processing`包中的`spectrogram_torch`函数，从`modules.mel_processing`包中继承并实现了`SpeechEncoder`接口。
5. 设置日志输出等级为`WARNING`，开启`stdout`输出模式。
6. 从`configs/diffusion.yaml`文件中读取指定的训练参数（hps）。
7. 从`data`参数中获取采样率（sampling_rate）和子词长度（hop_length）。
8. 从`model`字典中获取预定义的语音编码器（speech_encoder）。
9. 对输入数据进行预处理，并将其转换为浮点数。
10. 在`SpeechEncoder`中实现从浮点数数据到文本的映射。


```py
import diffusion.logger.utils as du
import utils
from diffusion.vocoder import Vocoder
from modules.mel_processing import spectrogram_torch

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

hps = utils.get_hparams_from_file("configs/config.json")
dconfig = du.load_config("configs/diffusion.yaml")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length
speech_encoder = hps["model"]["speech_encoder"]


```

This code appears to process an audio file (a.mp3) and extract certain volumes from it. It does this by first extracting the volume information from the audio file using a custom Volume_Extractor, which is then converted to a numpy array. Next, it extracts the mel-Frequency Cepstral Coefficients (MFCCs) from the volume information, which are then saved to a file called "mel.npy". It then does this process for the volume, and if the difference between the original volume and the augmented volume is greater than zero, it extracts the augmented volume using the same Volume_Extractor. It then saves the augmented volume to a file called "aug_vol.npy". It then continues this process for the augmented volume, and saves the augmented volume to a file called "aug_vol.npy".


```py
def process_one(filename, hmodel, f0p, device, diff=False, mel_extractor=None):
    wav, sr = librosa.load(filename, sr=sampling_rate)
    audio_norm = torch.FloatTensor(wav)
    audio_norm = audio_norm.unsqueeze(0)
    soft_path = filename + ".soft.pt"
    if not os.path.exists(soft_path):
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        c = hmodel.encoder(wav16k)
        torch.save(c.cpu(), soft_path)

    f0_path = filename + ".f0.npy"
    if not os.path.exists(f0_path):
        f0_predictor = utils.get_f0_predictor(f0p,sampling_rate=sampling_rate, hop_length=hop_length,device=None,threshold=0.05)
        f0,uv = f0_predictor.compute_f0_uv(
            wav
        )
        np.save(f0_path, np.asanyarray((f0,uv),dtype=object))


    spec_path = filename.replace(".wav", ".spec.pt")
    if not os.path.exists(spec_path):
        # Process spectrogram
        # The following code can't be replaced by torch.FloatTensor(wav)
        # because load_wav_to_torch return a tensor that need to be normalized

        if sr != hps.data.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sr, hps.data.sampling_rate
                )
            )

        #audio_norm = audio / hps.data.max_wav_value

        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        torch.save(spec, spec_path)

    if diff or hps.model.vol_embedding:
        volume_path = filename + ".vol.npy"
        volume_extractor = utils.Volume_Extractor(hop_length)
        if not os.path.exists(volume_path):
            volume = volume_extractor.extract(audio_norm)
            np.save(volume_path, volume.to('cpu').numpy())

    if diff:
        mel_path = filename + ".mel.npy"
        if not os.path.exists(mel_path) and mel_extractor is not None:
            mel_t = mel_extractor.extract(audio_norm.to(device), sampling_rate)
            mel = mel_t.squeeze().to('cpu').numpy()
            np.save(mel_path, mel)
        aug_mel_path = filename + ".aug_mel.npy"
        aug_vol_path = filename + ".aug_vol.npy"
        max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
        max_shift = min(1, np.log10(1/max_amp))
        log10_vol_shift = random.uniform(-1, max_shift)
        keyshift = random.uniform(-5, 5)
        if mel_extractor is not None:
            aug_mel_t = mel_extractor.extract(audio_norm * (10 ** log10_vol_shift), sampling_rate, keyshift = keyshift)
        aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
        aug_vol = volume_extractor.extract(audio_norm * (10 ** log10_vol_shift))
        if not os.path.exists(aug_mel_path):
            np.save(aug_mel_path,np.asanyarray((aug_mel,keyshift),dtype=object))
        if not os.path.exists(aug_vol_path):
            np.save(aug_vol_path,aug_vol.to('cpu').numpy())


```

这段代码定义了一个名为 `process_batch` 的函数，它接受一个文件句柄列表 `file_chunk`，一个声学编码器 `speech_encoder`，一个 `diff` 参数，以及一个 Mel 提取器。它的作用是处理每个文件句柄，并将每个文件句柄的内容传递给声学编码器，然后可以进一步处理这些结果。

该函数使用 Python 的 `mp`（multiprocessing）库来实现并行处理。具体来说，该函数创建了一个名为 `parallel_process` 的函数，它使用 `ProcessPoolExecutor` 对 `file_chunk` 中的所有文件句柄进行并行处理。这个并行处理是通过 `executor.submit` 函数实现的，它将 `file_chunk` 中的所有文件句柄提交给 `parallel_process` 函数进行处理。

该函数的输入参数包括：一个文件句柄列表 `file_chunk`，一个声学编码器 `speech_encoder`，一个 `diff` 参数，一个 Mel 提取器，以及一个设备，可以是 CPU 或者 GPU。函数需要使用的函数包括：`utils.get_speech_encoder`，这个函数用于从列表中选择正确的语音编码器。


```py
def process_batch(file_chunk, f0p, diff=False, mel_extractor=None, device="cpu"):
    logger.info("Loading speech encoder for content...")
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"Rank {rank} uses device {device}")
    hmodel = utils.get_speech_encoder(speech_encoder, device=device)
    logger.info(f"Loaded speech encoder for rank {rank}")
    for filename in tqdm(file_chunk, position = rank):
        process_one(filename, hmodel, f0p, device, diff, mel_extractor)

def parallel_process(filenames, num_processes, f0p, diff, mel_extractor, device):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(filenames) / num_processes)
            end = int((i + 1) * len(filenames) / num_processes)
            file_chunk = filenames[start:end]
            tasks.append(executor.submit(process_batch, file_chunk, f0p, diff, mel_extractor, device=device))
        for task in tqdm(tasks, position = 0):
            task.result()

```

It looks like this is a Python script that uses the Fairline speech encoder and a diffusion model. The script takes command-line arguments to control the input files, the device to use for the diffusion model (GPU for CPU), and an option to use the diffusion model for F0 extraction.

The script first imports the necessary libraries, including speech-encoder, fairline-data, and scikit-learn-audio. It then sets the options for the diffusion model and F0 extraction and runs the script.

The script uses the parallel-process function from the fairline-data library to parallelize the processing of the input files. This allows the script to run faster on multi-core systems.


```py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default=None)
    parser.add_argument(
        "--in_dir", type=str, default="dataset/44k", help="path to input dir"
    )
    parser.add_argument(
        '--use_diff',action='store_true', help='Whether to use the diffusion model'
    )
    parser.add_argument(
        '--f0_predictor', type=str, default="rmvpe", help='Select F0 predictor, can select crepe,pm,dio,harvest,rmvpe,fcpe|default: pm(note: crepe is original F0 using mean filter)'
    )
    parser.add_argument(
        '--num_processes', type=int, default=1, help='You are advised to set the number of processes to the same as the number of CPU cores'
    )
    args = parser.parse_args()
    f0p = args.f0_predictor
    device = args.device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(speech_encoder)
    logger.info("Using device: " + str(device))
    logger.info("Using SpeechEncoder: " + speech_encoder)
    logger.info("Using extractor: " + f0p)
    logger.info("Using diff Mode: " + str(args.use_diff))

    if args.use_diff:
        print("use_diff")
        print("Loading Mel Extractor...")
        mel_extractor = Vocoder(dconfig.vocoder.type, dconfig.vocoder.ckpt, device=device)
        print("Loaded Mel Extractor.")
    else:
        mel_extractor = None
    filenames = glob(f"{args.in_dir}/*/*.wav", recursive=True)  # [:10]
    shuffle(filenames)
    mp.set_start_method("spawn", force=True)

    num_processes = args.num_processes
    if num_processes == 0:
        num_processes = os.cpu_count()

    parallel_process(filenames, num_processes, f0p, args.use_diff, mel_extractor, device)

```

<div align="center">
<img alt="LOGO" src="https://avatars.githubusercontent.com/u/127122328?s=400&u=5395a98a4f945a3a50cb0cc96c2747505d190dbc&v=4" width="300" height="300" />
  
# SoftVC VITS Singing Voice Conversion

[**English**](./README.md) | [**中文简体**](./README_zh_CN.md)

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/svc-develop-team/so-vits-svc/blob/4.1-Stable/sovits4_for_colab.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-AGPL3.0-green.svg?style=for-the-badge)](https://github.com/svc-develop-team/so-vits-svc/blob/4.1-Stable/LICENSE)

This round of limited time update is coming to an end, the warehouse will enter the Archieve state, please know

</div>

> ✨ A studio that contains visible f0 editor, speaker mix timeline editor and other features (Where the Onnx models are used) : [MoeVoiceStudio](https://github.com/NaruseMioShirakana/MoeVoiceStudio)

> ✨ A fork with a greatly improved user interface: [34j/so-vits-svc-fork](https://github.com/34j/so-vits-svc-fork)

> ✨ A client supports real-time conversion: [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

**This project differs fundamentally from VITS, as it focuses on Singing Voice Conversion (SVC) rather than Text-to-Speech (TTS). In this project, TTS functionality is not supported, and VITS is incapable of performing SVC tasks. It's important to note that the models used in these two projects are not interchangeable or universally applicable.**

## Announcement

The purpose of this project was to enable developers to have their beloved anime characters perform singing tasks. The developers' intention was to focus solely on fictional characters and avoid any involvement of real individuals, anything related to real individuals deviates from the developer's original intention.

## Disclaimer

This project is an open-source, offline endeavor, and all members of SvcDevelopTeam, as well as other developers and maintainers involved (hereinafter referred to as contributors), have no control over the project. The contributors have never provided any form of assistance to any organization or individual, including but not limited to dataset extraction, dataset processing, computing support, training support, inference, and so on. The contributors do not and cannot be aware of the purposes for which users utilize the project. Therefore, any AI models and synthesized audio produced through the training of this project are unrelated to the contributors. Any issues or consequences arising from their use are the sole responsibility of the user.

This project is run completely offline and does not collect any user information or gather user input data. Therefore, contributors to this project are not aware of all user input and models and therefore are not responsible for any user input.

This project serves as a framework only and does not possess speech synthesis functionality by itself. All functionalities require users to train the models independently. Furthermore, this project does not come bundled with any models, and any secondary distributed projects are independent of the contributors of this project.

## 📏 Terms of Use

# Warning: Please ensure that you address any authorization issues related to the dataset on your own. You bear full responsibility for any problems arising from the usage of non-authorized datasets for training, as well as any resulting consequences. The repository and its maintainer, svc develop team, disclaim any association with or liability for the consequences. 

1. This project is exclusively established for academic purposes, aiming to facilitate communication and learning. It is not intended for deployment in production environments.
2. Any sovits-based video posted to a video platform must clearly specify in the introduction the input source vocals and audio used for the voice changer conversion, e.g., if you use someone else's video/audio and convert it by separating the vocals as the input source, you must give a clear link to the original video or music; if you use your own vocals or a voice synthesized by another voice synthesis engine as the input source, you must also state this in your introduction.
3. You are solely responsible for any infringement issues caused by the input source and all consequences. When using other commercial vocal synthesis software as an input source, please ensure that you comply with the regulations of that software, noting that the regulations of many vocal synthesis engines explicitly state that they cannot be used to convert input sources!
4. Engaging in illegal activities, as well as religious and political activities, is strictly prohibited when using this project. The project developers vehemently oppose the aforementioned activities. If you disagree with this provision, the usage of the project is prohibited.
5. If you continue to use the program, you will be deemed to have agreed to the terms and conditions set forth in README and README has discouraged you and is not responsible for any subsequent problems.
6. If you intend to employ this project for any other purposes, kindly contact and inform the maintainers of this repository in advance.

## 📝 Model Introduction

The singing voice conversion model uses SoftVC content encoder to extract speech features from the source audio. These feature vectors are directly fed into VITS without the need for conversion to a text-based intermediate representation. As a result, the pitch and intonations of the original audio are preserved. Meanwhile, the vocoder was replaced with [NSF HiFiGAN](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan) to solve the problem of sound interruption.

### 🆕 4.1-Stable Version Update Content

- Feature input is changed to the 12th Layer of [Content Vec](https://github.com/auspicious3000/contentvec) Transformer output, And compatible with 4.0 branches.
- Update the shallow diffusion, you can use the shallow diffusion model to improve the sound quality.
- Added Whisper-PPG encoder support
- Added static/dynamic sound fusion
- Added loudness embedding
- Added Functionality of feature retrieval from [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
  
### 🆕 Questions about compatibility with the 4.0 model

- To support the 4.0 model and incorporate the speech encoder, you can make modifications to the `config.json` file. Add the `speech_encoder` field to the "model" section as shown below:

```py
  "model": {
    .........
    "ssl_dim": 256,
    "n_speakers": 200,
    "speech_encoder":"vec256l9"
  }
```

### 🆕 Shallow diffusion
![Diagram](shadowdiffusion.png)

## 💬 Python Version

Based on our testing, we have determined that the project runs stable on `Python 3.8.9`.

## 📥 Pre-trained Model Files

#### **Required**

**You need to select one encoder from the list below**

##### **1. If using contentvec as speech encoder(recommended)**

`vec768l12` and `vec256l9` require the encoder

- ContentVec: [checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
  - Place it under the `pretrain` directory

Or download the following ContentVec, which is only 199MB in size but has the same effect:
- ContentVec: [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt)
  - Change the file name to `checkpoint_best_legacy_500.pt` and place it in the `pretrain` directory

```pyshell
# contentvec
wget -P pretrain/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -O checkpoint_best_legacy_500.pt
# Alternatively, you can manually download and place it in the hubert directory
```

##### **2. If hubertsoft is used as the speech encoder**
- soft vc hubert: [hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)
  - Place it under the `pretrain` directory

##### **3. If whisper-ppg as the encoder**
- download model at [medium.pt](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), the model fits `whisper-ppg`
- or download model at [large-v2.pt](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt), the model fits `whisper-ppg-large`
  - Place it under the `pretrain` directory
  
##### **4. If cnhubertlarge as the encoder**
- download model at [chinese-hubert-large-fairseq-ckpt.pt](https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/chinese-hubert-large-fairseq-ckpt.pt)
  - Place it under the `pretrain` directory

##### **5. If dphubert as the encoder**
- download model at [DPHuBERT-sp0.75.pth](https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPHuBERT-sp0.75.pth)
  - Place it under the `pretrain` directory

##### **6. If WavLM is used as the encoder**
- download model at  [WavLM-Base+.pt](https://valle.blob.core.windows.net/share/wavlm/WavLM-Base+.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D), the model fits `wavlmbase+`
  - Place it under the `pretrain` directory

##### **7. If OnnxHubert/ContentVec as the encoder**
- download model at [MoeSS-SUBModel](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main)
  - Place it under the `pretrain` directory

#### **List of Encoders**
- "vec768l12"
- "vec256l9"
- "vec256l9-onnx"
- "vec256l12-onnx"
- "vec768l9-onnx"
- "vec768l12-onnx"
- "hubertsoft-onnx"
- "hubertsoft"
- "whisper-ppg"
- "cnhubertlarge"
- "dphubert"
- "whisper-ppg-large"
- "wavlmbase+"

#### **Optional(Strongly recommend)**

- Pre-trained model files: `G_0.pth` `D_0.pth`
  - Place them under the `logs/44k` directory

- Diffusion model pretraining base model file: `model_0.pt`
  - Put it in the `logs/44k/diffusion` directory

Get Sovits Pre-trained model from svc-develop-team(TBD) or anywhere else.

Diffusion model references [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) diffusion model. The pre-trained diffusion model is universal with the DDSP-SVC's. You can go to [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC)'s repo to get the pre-trained diffusion model.

While the pretrained model typically does not pose copyright concerns, it is essential to remain vigilant. It is advisable to consult with the author beforehand or carefully review the description to ascertain the permissible usage of the model. This helps ensure compliance with any specified guidelines or restrictions regarding its utilization.

#### **Optional(Select as Required)**

##### NSF-HIFIGAN

If you are using the `NSF-HIFIGAN enhancer` or `shallow diffusion`, you will need to download the pre-trained NSF-HIFIGAN model.

- Pre-trained NSF-HIFIGAN Vocoder: [nsf_hifigan_20221211.zip](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip)
  - Unzip and place the four files under the `pretrain/nsf_hifigan` directory

```pyshell
# nsf_hifigan
wget -P pretrain/ https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip
unzip -od pretrain/nsf_hifigan pretrain/nsf_hifigan_20221211.zip
# Alternatively, you can manually download and place it in the pretrain/nsf_hifigan directory
# URL: https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1
```

##### RMVPE

If you are using the `rmvpe` F0 Predictor, you will need to download the pre-trained RMVPE model.

+ download model at [rmvpe.zip](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip), this weight is recommended.
  + unzip `rmvpe.zip`，and rename the `model.pt` file to `rmvpe.pt` and place it under the `pretrain` directory.

- ~~download model at [rmvpe.pt](https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/rmvpe.pt)~~
  - ~~Place it under the `pretrain` directory~~

##### FCPE(Preview version)

[FCPE(Fast Context-base Pitch Estimator)](https://github.com/CNChTu/MelPE) is a dedicated F0 predictor designed for real-time voice conversion and will become the preferred F0 predictor for sovits real-time voice conversion in the future.(The paper is being written)

If you are using the `fcpe` F0 Predictor, you will need to download the pre-trained FCPE model.

- download model at [fcpe.pt](https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/fcpe.pt)
  - Place it under the `pretrain` directory

## 📊 Dataset Preparation

Simply place the dataset in the `dataset_raw` directory with the following file structure:

```py
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```
There are no specific restrictions on the format of the name for each audio file (naming conventions such as `000001.wav` to `999999.wav` are also valid), but the file type must be `WAV``.

You can customize the speaker's name as showed below:

```py
dataset_raw
└───suijiSUI
    ├───1.wav
    ├───...
    └───25788785-20221210-200143-856_01_(Vocals)_0_0.wav
```

## 🛠️ Preprocessing

### 0. Slice audio

To avoid video memory overflow during training or pre-processing, it is recommended to limit the length of audio clips. Cutting the audio to a length of "5s - 15s" is more recommended. Slightly longer times are acceptable, however, excessively long clips may cause problems such as `torch.cuda.OutOfMemoryError`.

To facilitate the slicing process, you can use [audio-slicer-GUI](https://github.com/flutydeer/audio-slicer) or [audio-slicer-CLI](https://github.com/openvpi/audio-slicer)

In general, only the `Minimum Interval` needs to be adjusted. For spoken audio, the default value usually suffices, while for singing audio, it can be adjusted to around `100` or even `50`, depending on the specific requirements.

After slicing, it is recommended to remove any audio clips that are excessively long or too short.

**If you are using whisper-ppg encoder for training, the audio clips must shorter than 30s.**

### 1. Resample to 44100Hz and mono

```pyshell
python resample.py
```

#### Cautions

Although this project has resample.py scripts for resampling, mono and loudness matching, the default loudness matching is to match to 0db. This can cause damage to the sound quality. While python's loudness matching package pyloudnorm does not limit the level, this can lead to sonic boom. Therefore, it is recommended to consider using professional sound processing software, such as `adobe audition` for loudness matching. If you are already using other software for loudness matching, add the parameter `-skip_loudnorm` to the run command:

```pyshell
python resample.py --skip_loudnorm
```

### 2. Automatically split the dataset into training and validation sets, and generate configuration files.

```pyshell
python preprocess_flist_config.py --speech_encoder vec768l12
```

speech_encoder has the following options

```py
vec768l12
vec256l9
hubertsoft
whisper-ppg
cnhubertlarge
dphubert
whisper-ppg-large
wavlmbase+
```

If the speech_encoder argument is omitted, the default value is `vec768l12`

**Use loudness embedding**

Add `--vol_aug` if you want to enable loudness embedding:

```pyshell
python preprocess_flist_config.py --speech_encoder vec768l12 --vol_aug
```

After enabling loudness embedding, the trained model will match the loudness of the input source; otherwise, it will match the loudness of the training set.

#### You can modify some parameters in the generated config.json and diffusion.yaml

* `keep_ckpts`: Keep the the the number of previous models during training. Set to `0` to keep them all. Default is `3`.

* `all_in_mem`: Load all dataset to RAM. It can be enabled when the disk IO of some platforms is too low and the system memory is **much larger** than your dataset.
  
* `batch_size`: The amount of data loaded to the GPU for a single training session can be adjusted to a size lower than the GPU memory capacity.

* `vocoder_name`: Select a vocoder. The default is `nsf-hifigan`.

##### diffusion.yaml

* `cache_all_data`: Load all dataset to RAM. It can be enabled when the disk IO of some platforms is too low and the system memory is **much larger** than your dataset.

* `duration`: The duration of the audio slicing during training, can be adjusted according to the size of the video memory, **Note: this value must be less than the minimum time of the audio in the training set!**

* `batch_size`: The amount of data loaded to the GPU for a single training session can be adjusted to a size lower than the video memory capacity.

* `timesteps`: The total number of steps in the diffusion model, which defaults to 1000.

* `k_step_max`: Training can only train `k_step_max` step diffusion to save training time, note that the value must be less than `timesteps`, 0 is to train the entire diffusion model, **Note: if you do not train the entire diffusion model will not be able to use only_diffusion!**

##### **List of Vocoders**

```py
nsf-hifigan
nsf-snake-hifigan
```

### 3. Generate hubert and f0

```pyshell
python preprocess_hubert_f0.py --f0_predictor dio
```

f0_predictor has the following options

```py
crepe
dio
pm
harvest
rmvpe
fcpe
```

If the training set is too noisy,it is recommended to use `crepe` to handle f0

If the f0_predictor parameter is omitted, the default value is `rmvpe`

If you want shallow diffusion (optional), you need to add the `--use_diff` parameter, for example:

```pyshell
python preprocess_hubert_f0.py --f0_predictor dio --use_diff
```

**Speed Up preprocess**

If your dataset is pretty large,you can increase the param `--num_processes` like that:

```pyshell
python preprocess_hubert_f0.py --f0_predictor dio --num_processes 8
```
All the worker will be assigned to different GPU if you have more than one GPUs.

After completing the above steps, the dataset directory will contain the preprocessed data, and the dataset_raw folder can be deleted.

## 🏋️‍ Training

### Sovits Model

```pyshell
python train.py -c configs/config.json -m 44k
```

### Diffusion Model (optional)

If the shallow diffusion function is needed, the diffusion model needs to be trained. The diffusion model training method is as follows:

```pyshell
python train_diff.py -c configs/diffusion.yaml
```

During training, the model files will be saved to `logs/44k`, and the diffusion model will be saved to `logs/44k/diffusion`

## 🤖 Inference

Use [inference_main.py](https://github.com/svc-develop-team/so-vits-svc/blob/4.0/inference_main.py)

```pyshell
# Example
python inference_main.py -m "logs/44k/G_30400.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "nen"
```

Required parameters:
- `-m` | `--model_path`: path to the model.
- `-c` | `--config_path`: path to the configuration file.
- `-n` | `--clean_names`: a list of wav file names located in the `raw` folder.
- `-t` | `--trans`: pitch shift, supports positive and negative (semitone) values.
- `-s` | `--spk_list`: Select the speaker ID to use for conversion.
- `-cl` | `--clip`: Forced audio clipping, set to 0 to disable(default), setting it to a non-zero value (duration in seconds) to enable.

Optional parameters: see the next section
- `-lg` | `--linear_gradient`: The cross fade length of two audio slices in seconds. If there is a discontinuous voice after forced slicing, you can adjust this value. Otherwise, it is recommended to use the default value of 0.
- `-f0p` | `--f0_predictor`: Select a F0 predictor, options are `crepe`, `pm`, `dio`, `harvest`, `rmvpe`,`fcpe`, default value is `pm`(note: f0 mean pooling will be enable when using `crepe`)
- `-a` | `--auto_predict_f0`: automatic pitch prediction, do not enable this when converting singing voices as it can cause serious pitch issues.
- `-cm` | `--cluster_model_path`: Cluster model or feature retrieval index path, if left blank, it will be automatically set as the default path of these models. If there is no training cluster or feature retrieval, fill in at will.
- `-cr` | `--cluster_infer_ratio`: The proportion of clustering scheme or feature retrieval ranges from 0 to 1. If there is no training clustering model or feature retrieval, the default is 0.
- `-eh` | `--enhance`: Whether to use NSF_HIFIGAN enhancer, this option has certain effect on sound quality enhancement for some models with few training sets, but has negative effect on well-trained models, so it is disabled by default.
- `-shd` | `--shallow_diffusion`: Whether to use shallow diffusion, which can solve some electrical sound problems after use. This option is disabled by default. When this option is enabled, NSF_HIFIGAN enhancer will be disabled
- `-usm` | `--use_spk_mix`: whether to use dynamic voice fusion
- `-lea` | `--loudness_envelope_adjustment`：The adjustment of the input source's loudness envelope in relation to the fusion ratio of the output loudness envelope. The closer to 1, the more the output loudness envelope is used
- `-fr` | `--feature_retrieval`：Whether to use feature retrieval If clustering model is used, it will be disabled, and `cm` and `cr` parameters will become the index path and mixing ratio of feature retrieval
  
Shallow diffusion settings:
- `-dm` | `--diffusion_model_path`: Diffusion model path
- `-dc` | `--diffusion_config_path`: Diffusion config file path
- `-ks` | `--k_step`: The larger the number of k_steps, the closer it is to the result of the diffusion model. The default is 100
- `-od` | `--only_diffusion`: Whether to use Only diffusion mode, which does not load the sovits model to only use diffusion model inference
- `-se` | `--second_encoding`：which involves applying an additional encoding to the original audio before shallow diffusion. This option can yield varying results - sometimes positive and sometimes negative.

### Cautions

If inferencing using `whisper-ppg` speech encoder, you need to set `--clip` to 25 and `-lg` to 1. Otherwise it will fail to infer properly.

## 🤔 Optional Settings

If you are satisfied with the previous results, or if you do not feel you understand what follows, you can skip it and it will have no effect on the use of the model. The impact of these optional settings mentioned is relatively small, and while they may have some impact on specific datasets, in most cases the difference may not be significant.

### Automatic f0 prediction

During the training of the 4.0 model, an f0 predictor is also trained, which enables automatic pitch prediction during voice conversion. However, if the results are not satisfactory, manual pitch prediction can be used instead. Please note that when converting singing voices, it is advised not to enable this feature as it may cause significant pitch shifting.

- Set `auto_predict_f0` to `true` in `inference_main.py`.

### Cluster-based timbre leakage control

Introduction: The clustering scheme implemented in this model aims to reduce timbre leakage and enhance the similarity of the trained model to the target's timbre, although the effect may not be very pronounced. However, relying solely on clustering can reduce the model's clarity and make it sound less distinct. Therefore, a fusion method is adopted in this model to control the balance between the clustering and non-clustering approaches. This allows manual adjustment of the trade-off between "sounding like the target's timbre" and "have clear enunciation" to find an optimal balance.

No changes are required in the existing steps. Simply train an additional clustering model, which incurs relatively low training costs.

- Training process:
  - Train on a machine with good CPU performance. According to extant experience, it takes about 4 minutes to train each speaker on a Tencent Cloud machine with 6-core CPU.
  - Execute `python cluster/train_cluster.py`. The output model will be saved in `logs/44k/kmeans_10000.pt`.
  - The clustering model can currently be trained using the gpu by executing `python cluster/train_cluster.py --gpu`
- Inference process:
  - Specify `cluster_model_path` in `inference_main.py`. If not specified, the default is `logs/44k/kmeans_10000.pt`.
  - Specify `cluster_infer_ratio` in `inference_main.py`, where `0` means not using clustering at all, `1` means only using clustering, and usually `0.5` is sufficient.

### Feature retrieval

Introduction: As with the clustering scheme, the timbre leakage can be reduced, the enunciation is slightly better than clustering, but it will reduce the inference speed. By employing the fusion method, it becomes possible to linearly control the balance between feature retrieval and non-feature retrieval, allowing for fine-tuning of the desired proportion.

- Training process: 
  First, it needs to be executed after generating hubert and f0: 

```pyshell
python train_index.py -c configs/config.json
```

The output of the model will be in `logs/44k/feature_and_index.pkl`

- Inference process: 
  - The `--feature_retrieval` needs to be formulated first, and the clustering mode automatically switches to the feature retrieval mode.
  - Specify `cluster_model_path` in `inference_main.py`. If not specified, the default is `logs/44k/feature_and_index.pkl`.
  - Specify `cluster_infer_ratio` in `inference_main.py`, where `0` means not using feature retrieval at all, `1` means only using feature retrieval, and usually `0.5` is sufficient.

## 🗜️ Model compression

The generated model contains data that is needed for further training. If you confirm that the model is final and not be used in further training, it is safe to remove these data to get smaller file size (about 1/3).

```pyshell
# Example
python compress_model.py -c="configs/config.json" -i="logs/44k/G_30400.pth" -o="logs/44k/release.pth"
```

## 👨‍🔧 Timbre mixing

### Static Tone Mixing

**Refer to `webUI.py` file for stable Timbre mixing of the gadget/lab feature.**

Introduction: This function can combine multiple models into one model (convex combination or linear combination of multiple model parameters) to create mixed voice that do not exist in reality

**Note:**
1. This feature is only supported for single-speaker models
2. If you force a multi-speaker model, it is critical to make sure there are the same number of speakers in each model. This will ensure that sounds with the same SpeakerID can be mixed correctly.
3. Ensure that the `model` fields in config.json of all models to be mixed are the same
4. The mixed model can use any config.json file from the models being synthesized. However, the clustering model will not be functional after mixed.
5. When batch uploading models, it is best to put the models into a folder and upload them together after selecting them
6. It is suggested to adjust the mixing ratio between 0 and 100, or to other numbers, but unknown effects will occur in the linear combination mode
7. After mixing, the file named output.pth will be saved in the root directory of the project
8. Convex combination mode will perform Softmax to add the mix ratio to 1, while linear combination mode will not

### Dynamic timbre mixing

**Refer to the `spkmix.py` file for an introduction to dynamic timbre mixing**

Character mix track writing rules:

Role ID: \[\[Start time 1, end time 1, start value 1, start value 1], [Start time 2, end time 2, start value 2]]

The start time must be the same as the end time of the previous one. The first start time must be 0, and the last end time must be 1 (time ranges from 0 to 1).

All roles must be filled in. For unused roles, fill \[\[0., 1., 0., 0.]]

The fusion value can be filled in arbitrarily, and the linear change from the start value to the end value within the specified period of time. The 

internal linear combination will be automatically guaranteed to be 1 (convex combination condition), so it can be used safely

Use the `--use_spk_mix` parameter when reasoning to enable dynamic timbre mixing

## 📤 Exporting to Onnx

Use [onnx_export.py](https://github.com/svc-develop-team/so-vits-svc/blob/4.0/onnx_export.py)

- Create a folder named `checkpoints` and open it
- Create a folder in the `checkpoints` folder as your project folder, naming it after your project, for example `aziplayer`
- Rename your model as `model.pth`, the configuration file as `config.json`, and place them in the `aziplayer` folder you just created
- Modify `"NyaruTaffy"` in `path = "NyaruTaffy"` in [onnx_export.py](https://github.com/svc-develop-team/so-vits-svc/blob/4.0/onnx_export.py) to your project name, `path = "aziplayer"`（onnx_export_speaker_mix makes you can mix speaker's voice）
- Run [onnx_export.py](https://github.com/svc-develop-team/so-vits-svc/blob/4.0/onnx_export.py)
- Wait for it to finish running. A `model.onnx` will be generated in your project folder, which is the exported model.

Note: For Hubert Onnx models, please use the models provided by MoeSS. Currently, they cannot be exported on their own (Hubert in fairseq has many unsupported operators and things involving constants that can cause errors or result in problems with the input/output shape and results when exported.)


## 📎 Reference

| URL | Designation | Title | Implementation Source |
| --- | ----------- | ----- | --------------------- |
|[2106.06103](https://arxiv.org/abs/2106.06103) | VITS (Synthesizer)| Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech | [jaywalnut310/vits](https://github.com/jaywalnut310/vits) |
|[2111.02392](https://arxiv.org/abs/2111.02392) | SoftVC (Speech Encoder)| A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion | [bshall/hubert](https://github.com/bshall/hubert) |
|[2204.09224](https://arxiv.org/abs/2204.09224) | ContentVec (Speech Encoder)| ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers | [auspicious3000/contentvec](https://github.com/auspicious3000/contentvec) |
|[2212.04356](https://arxiv.org/abs/2212.04356) | Whisper (Speech Encoder) | Robust Speech Recognition via Large-Scale Weak Supervision | [openai/whisper](https://github.com/openai/whisper) |
|[2110.13900](https://arxiv.org/abs/2110.13900) | WavLM (Speech Encoder) | WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing | [microsoft/unilm/wavlm](https://github.com/microsoft/unilm/tree/master/wavlm) |
|[2305.17651](https://arxiv.org/abs/2305.17651) | DPHubert (Speech Encoder) | DPHuBERT: Joint Distillation and Pruning of Self-Supervised Speech Models | [pyf98/DPHuBERT](https://github.com/pyf98/DPHuBERT) |
|[DOI:10.21437/Interspeech.2017-68](http://dx.doi.org/10.21437/Interspeech.2017-68) | Harvest (F0 Predictor) | Harvest: A high-performance fundamental frequency estimator from speech signals | [mmorise/World/harvest](https://github.com/mmorise/World/blob/master/src/harvest.cpp) |
|[aes35-000039](https://www.aes.org/e-lib/online/browse.cfm?elib=15165) | Dio (F0 Predictor) | Fast and reliable F0 estimation method based on the period extraction of vocal fold vibration of singing voice and speech | [mmorise/World/dio](https://github.com/mmorise/World/blob/master/src/dio.cpp) |
|[8461329](https://ieeexplore.ieee.org/document/8461329) | Crepe (F0 Predictor) | Crepe: A Convolutional Representation for Pitch Estimation | [maxrmorrison/torchcrepe](https://github.com/maxrmorrison/torchcrepe) |
|[DOI:10.1016/j.wocn.2018.07.001](https://doi.org/10.1016/j.wocn.2018.07.001) | Parselmouth (F0 Predictor) | Introducing Parselmouth: A Python interface to Praat | [YannickJadoul/Parselmouth](https://github.com/YannickJadoul/Parselmouth) |
|[2306.15412v2](https://arxiv.org/abs/2306.15412v2) | RMVPE (F0 Predictor) | RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music | [Dream-High/RMVPE](https://github.com/Dream-High/RMVPE) |
|[2010.05646](https://arxiv.org/abs/2010.05646) | HIFIGAN (Vocoder) | HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis | [jik876/hifi-gan](https://github.com/jik876/hifi-gan) |
|[1810.11946](https://arxiv.org/abs/1810.11946.pdf) | NSF (Vocoder) | Neural source-filter-based waveform model for statistical parametric speech synthesis | [openvpi/DiffSinger/modules/nsf_hifigan](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan)
|[2006.08195](https://arxiv.org/abs/2006.08195) | Snake (Vocoder) | Neural Networks Fail to Learn Periodic Functions and How to Fix It | [EdwardDixon/snake](https://github.com/EdwardDixon/snake)
|[2105.02446v3](https://arxiv.org/abs/2105.02446v3) | Shallow Diffusion (PostProcessing)| DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism | [CNChTu/Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) |
|[K-means](https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=01D65490BADCC216F350D06F84D721AD?doi=10.1.1.308.8619&rep=rep1&type=pdf) | Feature K-means Clustering (PreProcessing)| Some methods for classification and analysis of multivariate observations | This repo |
| | Feature TopK Retrieval (PreProcessing)| Retrieval based Voice Conversion | [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) |
| | whisper ppg| whisper ppg | [PlayVoice/whisper_ppg](https://github.com/PlayVoice/whisper_ppg) |
| | bigvgan| bigvgan | [PlayVoice/so-vits-svc-5.0](https://github.com/PlayVoice/so-vits-svc-5.0/tree/bigvgan-mix-v2/vits_decoder/alias) |


## ☀️ Previous contributors

For some reason the author deleted the original repository. Because of the negligence of the organization members, the contributor list was cleared because all files were directly reuploaded to this repository at the beginning of the reconstruction of this repository. Now add a previous contributor list to README.md.

*Some members have not listed according to their personal wishes.*

<table>
  <tr>
    <td align="center"><a href="https://github.com/MistEO"><img src="https://avatars.githubusercontent.com/u/18511905?v=4" width="100px;" alt=""/><br /><sub><b>MistEO</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/XiaoMiku01"><img src="https://avatars.githubusercontent.com/u/54094119?v=4" width="100px;" alt=""/><br /><sub><b>XiaoMiku01</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/ForsakenRei"><img src="https://avatars.githubusercontent.com/u/23041178?v=4" width="100px;" alt=""/><br /><sub><b>しぐれ</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/TomoGaSukunai"><img src="https://avatars.githubusercontent.com/u/25863522?v=4" width="100px;" alt=""/><br /><sub><b>TomoGaSukunai</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/Plachtaa"><img src="https://avatars.githubusercontent.com/u/112609742?v=4" width="100px;" alt=""/><br /><sub><b>Plachtaa</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/zdxiaoda"><img src="https://avatars.githubusercontent.com/u/45501959?v=4" width="100px;" alt=""/><br /><sub><b>zd小达</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/Archivoice"><img src="https://avatars.githubusercontent.com/u/107520869?v=4" width="100px;" alt=""/><br /><sub><b>凍聲響世</b></sub></a><br /></td>
  </tr>
</table>

## 📚 Some legal provisions for reference

#### Any country, region, organization, or individual using this project must comply with the following laws.

#### 《民法典》

##### 第一千零一十九条 

任何组织或者个人不得以丑化、污损，或者利用信息技术手段伪造等方式侵害他人的肖像权。未经肖像权人同意，不得制作、使用、公开肖像权人的肖像，但是法律另有规定的除外。未经肖像权人同意，肖像作品权利人不得以发表、复制、发行、出租、展览等方式使用或者公开肖像权人的肖像。对自然人声音的保护，参照适用肖像权保护的有关规定。

#####  第一千零二十四条 

【名誉权】民事主体享有名誉权。任何组织或者个人不得以侮辱、诽谤等方式侵害他人的名誉权。  

#####  第一千零二十七条

【作品侵害名誉权】行为人发表的文学、艺术作品以真人真事或者特定人为描述对象，含有侮辱、诽谤内容，侵害他人名誉权的，受害人有权依法请求该行为人承担民事责任。行为人发表的文学、艺术作品不以特定人为描述对象，仅其中的情节与该特定人的情况相似的，不承担民事责任。  

#### 《[中华人民共和国宪法](http://www.gov.cn/guoqing/2018-03/22/content_5276318.htm)》

#### 《[中华人民共和国刑法](http://gongbao.court.gov.cn/Details/f8e30d0689b23f57bfc782d21035c3.html?sw=%E4%B8%AD%E5%8D%8E%E4%BA%BA%E6%B0%91%E5%85%B1%E5%92%8C%E5%9B%BD%E5%88%91%E6%B3%95)》

#### 《[中华人民共和国民法典](http://gongbao.court.gov.cn/Details/51eb6750b8361f79be8f90d09bc202.html)》

#### 《[中华人民共和国合同法](http://www.npc.gov.cn/zgrdw/npc/lfzt/rlyw/2016-07/01/content_1992739.htm)》

## 💪 Thanks to all contributors for their efforts
<a href="https://github.com/svc-develop-team/so-vits-svc/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=svc-develop-team/so-vits-svc" />
</a>


<div align="center">
<img alt="LOGO" src="https://avatars.githubusercontent.com/u/127122328?s=400&u=5395a98a4f945a3a50cb0cc96c2747505d190dbc&v=4" width="300" height="300" />

# SoftVC VITS Singing Voice Conversion

[**English**](./README.md) | [**中文简体**](./README_zh_CN.md)

[![在Google Cloab中打开](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/svc-develop-team/so-vits-svc/blob/4.1-Stable/sovits4_for_colab.ipynb)
[![LICENSE](https://img.shields.io/badge/LICENSE-AGPL3.0-green.svg?style=for-the-badge)](https://github.com/svc-develop-team/so-vits-svc/blob/4.1-Stable/LICENSE)

本轮限时更新即将结束，仓库将进入Archieve状态，望周知

</div>


#### ✨ 带有 F0 曲线编辑器，角色混合时间轴编辑器的推理端 (Onnx 模型的用途）: [MoeVoiceStudio](https://github.com/NaruseMioShirakana/MoeVoiceStudio)

#### ✨ 改善了交互的一个分支推荐: [34j/so-vits-svc-fork](https://github.com/34j/so-vits-svc-fork)

#### ✨ 支持实时转换的一个客户端: [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

**本项目与 Vits 有着根本上的不同。Vits 是 TTS，本项目是 SVC。本项目无法实现 TTS，Vits 也无法实现 SVC，这两个项目的模型是完全不通用的。**

## 重要通知

这个项目是为了让开发者最喜欢的动画角色唱歌而开发的，任何涉及真人的东西都与开发者的意图背道而驰。

## 声明

本项目为开源、离线的项目，SvcDevelopTeam 的所有成员与本项目的所有开发者以及维护者（以下简称贡献者）对本项目没有控制力。本项目的贡献者从未向任何组织或个人提供包括但不限于数据集提取、数据集加工、算力支持、训练支持、推理等一切形式的帮助；本项目的贡献者不知晓也无法知晓使用者使用该项目的用途。故一切基于本项目训练的 AI 模型和合成的音频都与本项目贡献者无关。一切由此造成的问题由使用者自行承担。

此项目完全离线运行，不能收集任何用户信息或获取用户输入数据。因此，这个项目的贡献者不知道所有的用户输入和模型，因此不负责任何用户输入。

本项目只是一个框架项目，本身并没有语音合成的功能，所有的功能都需要用户自己训练模型。同时，这个项目没有任何模型，任何二次分发的项目都与这个项目的贡献者无关。

## 📏 使用规约

# Warning：请自行解决数据集授权问题，禁止使用非授权数据集进行训练！任何由于使用非授权数据集进行训练造成的问题，需自行承担全部责任和后果！与仓库、仓库维护者、svc develop team 无关！

1. 本项目是基于学术交流目的建立，仅供交流与学习使用，并非为生产环境准备。
2. 任何发布到视频平台的基于 sovits 制作的视频，都必须要在简介明确指明用于变声器转换的输入源歌声、音频，例如：使用他人发布的视频 / 音频，通过分离的人声作为输入源进行转换的，必须要给出明确的原视频、音乐链接；若使用是自己的人声，或是使用其他歌声合成引擎合成的声音作为输入源进行转换的，也必须在简介加以说明。
3. 由输入源造成的侵权问题需自行承担全部责任和一切后果。使用其他商用歌声合成软件作为输入源时，请确保遵守该软件的使用条例，注意，许多歌声合成引擎使用条例中明确指明不可用于输入源进行转换！
4. 禁止使用该项目从事违法行为与宗教、政治等活动，该项目维护者坚决抵制上述行为，不同意此条则禁止使用该项目。
5. 继续使用视为已同意本仓库 README 所述相关条例，本仓库 README 已进行劝导义务，不对后续可能存在问题负责。
6. 如果将此项目用于任何其他企划，请提前联系并告知本仓库作者，十分感谢。

## 📝 模型简介

歌声音色转换模型，通过 SoftVC 内容编码器提取源音频语音特征，与 F0 同时输入 VITS 替换原本的文本输入达到歌声转换的效果。同时，更换声码器为 [NSF HiFiGAN](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan) 解决断音问题。

### 🆕 4.1-Stable 版本更新内容

+ 特征输入更换为 [Content Vec](https://github.com/auspicious3000/contentvec) 的第 12 层 Transformer 输出，并兼容 4.0 分支
+ 更新浅层扩散，可以使用浅层扩散模型提升音质
+ 增加 whisper 语音编码器的支持
+ 增加静态/动态声线融合
+ 增加响度嵌入
+ 增加特征检索，来自于 [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

### 🆕 关于兼容 4.0 模型的问题

+ 可通过修改 4.0 模型的 config.json 对 4.0 的模型进行支持，需要在 config.json 的 model 字段中添加 speech_encoder 字段，具体见下

```py
  "model": {
    .........
    "ssl_dim": 256,
    "n_speakers": 200,
    "speech_encoder":"vec256l9"
  }
```

### 🆕 关于浅扩散
![Diagram](shadowdiffusion.png)

## 💬 关于 Python 版本问题

在进行测试后，我们认为`Python 3.8.9`能够稳定地运行该项目

## 📥 预先下载的模型文件

#### **必须项**

**以下编码器需要选择一个使用**

##### **1. 若使用 contentvec 作为声音编码器（推荐）**

`vec768l12`与`vec256l9` 需要该编码器

+ contentvec ：[checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
  + 放在`pretrain`目录下

或者下载下面的 ContentVec，大小只有 199MB，但效果相同：
+ contentvec ：[hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt)
  + 将文件名改为`checkpoint_best_legacy_500.pt`后，放在`pretrain`目录下

```pyshell
# contentvec
wget -P pretrain/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -O checkpoint_best_legacy_500.pt
# 也可手动下载放在 pretrain 目录
```

##### **2. 若使用 hubertsoft 作为声音编码器**
+ soft vc hubert：[hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)
  + 放在`pretrain`目录下

##### **3. 若使用 Whisper-ppg 作为声音编码器**
+ 下载模型 [medium.pt](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), 该模型适配`whisper-ppg`
+ 下载模型 [large-v2.pt](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt), 该模型适配`whisper-ppg-large`
  + 放在`pretrain`目录下

##### **4. 若使用 cnhubertlarge 作为声音编码器**
+ 下载模型 [chinese-hubert-large-fairseq-ckpt.pt](https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/chinese-hubert-large-fairseq-ckpt.pt)
  + 放在`pretrain`目录下

##### **5. 若使用 dphubert 作为声音编码器**
+ 下载模型 [DPHuBERT-sp0.75.pth](https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPHuBERT-sp0.75.pth)
  + 放在`pretrain`目录下

##### **6. 若使用 WavLM 作为声音编码器**
+ 下载模型 [WavLM-Base+.pt](https://valle.blob.core.windows.net/share/wavlm/WavLM-Base+.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D), 该模型适配`wavlmbase+`
  + 放在`pretrain`目录下

##### **7. 若使用 OnnxHubert/ContentVec 作为声音编码器**
+ 下载模型 [MoeSS-SUBModel](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main)
  + 放在`pretrain`目录下

#### **编码器列表**
- "vec768l12"
- "vec256l9"
- "vec256l9-onnx"
- "vec256l12-onnx"
- "vec768l9-onnx"
- "vec768l12-onnx"
- "hubertsoft-onnx"
- "hubertsoft"
- "whisper-ppg"
- "cnhubertlarge"
- "dphubert"
- "whisper-ppg-large"
- "wavlmbase+"

#### **可选项（强烈建议使用）**

+ 预训练底模文件： `G_0.pth` `D_0.pth`
  + 放在`logs/44k`目录下

+ 扩散模型预训练底模文件： `model_0.pt`
  + 放在`logs/44k/diffusion`目录下

从 svc-develop-team（待定）或任何其他地方获取 Sovits 底模

扩散模型引用了 [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) 的 Diffusion Model，底模与 [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) 的扩散模型底模通用，可以去 [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) 获取扩散模型的底模

虽然底模一般不会引起什么版权问题，但还是请注意一下，比如事先询问作者，又或者作者在模型描述中明确写明了可行的用途

#### **可选项（根据情况选择）**

##### NSF-HIFIGAN

如果使用`NSF-HIFIGAN 增强器`或`浅层扩散`的话，需要下载预训练的 NSF-HIFIGAN 模型，如果不需要可以不下载

+ 预训练的 NSF-HIFIGAN 声码器 ：[nsf_hifigan_20221211.zip](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip)
  + 解压后，将四个文件放在`pretrain/nsf_hifigan`目录下

```pyshell
# nsf_hifigan
wget -P pretrain/ https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip
unzip -od pretrain/nsf_hifigan pretrain/nsf_hifigan_20221211.zip
# 也可手动下载放在 pretrain/nsf_hifigan 目录
# 地址：https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1
```

##### RMVPE

如果使用`rmvpe`F0预测器的话，需要下载预训练的 RMVPE 模型

+ 下载模型[rmvpe.zip](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip)，目前首推该权重。
  + 解压缩`rmvpe.zip`，并将其中的`model.pt`文件改名为`rmvpe.pt`并放在`pretrain`目录下

+ ~~下载模型 [rmvpe.pt](https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/rmvpe.pt)~~
  + ~~放在`pretrain`目录下~~

##### FCPE(预览版)

> 你说的对,但是[FCPE](https://github.com/CNChTu/MelPE)是由svc-develop-team自主研发的一款全新的F0预测器，后面忘了

[FCPE(Fast Context-base Pitch Estimator)](https://github.com/CNChTu/MelPE)是一个为实时语音转换所设计的专用F0预测器，他将在未来成为Sovits实时语音转换的首选F0预测器.（论文未来会有的）

如果使用 `fcpe` F0预测器的话，需要下载预训练的 FCPE 模型

+ 下载模型 [fcpe.pt](https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/fcpe.pt)
  + 放在`pretrain`目录下


## 📊 数据集准备

仅需要以以下文件结构将数据集放入 dataset_raw 目录即可。

```py
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```
对于每一个音频文件的名称并没有格式的限制(`000001.wav`~`999999.wav`之类的命名方式也是合法的)，不过文件类型必须是`wav`。

可以自定义说话人名称

```py
dataset_raw
└───suijiSUI
    ├───1.wav
    ├───...
    └───25788785-20221210-200143-856_01_(Vocals)_0_0.wav
```

## 🛠️ 数据预处理

### 0. 音频切片

将音频切片至`5s - 15s`, 稍微长点也无伤大雅，实在太长可能会导致训练中途甚至预处理就爆显存

可以使用 [audio-slicer-GUI](https://github.com/flutydeer/audio-slicer)、[audio-slicer-CLI](https://github.com/openvpi/audio-slicer)

一般情况下只需调整其中的`Minimum Interval`，普通陈述素材通常保持默认即可，歌唱素材可以调整至`100`甚至`50`

切完之后手动删除过长过短的音频

**如果你使用 Whisper-ppg 声音编码器进行训练，所有的切片长度必须小于 30s**

### 1. 重采样至 44100Hz 单声道

```pyshell
python resample.py
```

#### 注意

虽然本项目拥有重采样、转换单声道与响度匹配的脚本 resample.py，但是默认的响度匹配是匹配到 0db。这可能会造成音质的受损。而 python 的响度匹配包 pyloudnorm 无法对电平进行压限，这会导致爆音。所以建议可以考虑使用专业声音处理软件如`adobe audition`等软件做响度匹配处理。若已经使用其他软件做响度匹配，可以在运行上述命令时添加`--skip_loudnorm`跳过响度匹配步骤。如：

```pyshell
python resample.py --skip_loudnorm
```

### 2. 自动划分训练集、验证集，以及自动生成配置文件

```pyshell
python preprocess_flist_config.py --speech_encoder vec768l12
```

speech_encoder 拥有以下选择

```py
vec768l12
vec256l9
hubertsoft
whisper-ppg
whisper-ppg-large
cnhubertlarge
dphubert
wavlmbase+
```

如果省略 speech_encoder 参数，默认值为 vec768l12

**使用响度嵌入**

若使用响度嵌入，需要增加`--vol_aug`参数，比如：

```pyshell
python preprocess_flist_config.py --speech_encoder vec768l12 --vol_aug
```
使用后训练出的模型将匹配到输入源响度，否则为训练集响度。

#### 此时可以在生成的 config.json 与 diffusion.yaml 修改部分参数

##### config.json

* `keep_ckpts`：训练时保留最后几个模型，`0`为保留所有，默认只保留最后`3`个

* `all_in_mem`：加载所有数据集到内存中，某些平台的硬盘 IO 过于低下、同时内存容量 **远大于** 数据集体积时可以启用

* `batch_size`：单次训练加载到 GPU 的数据量，调整到低于显存容量的大小即可

* `vocoder_name` : 选择一种声码器，默认为`nsf-hifigan`.

##### diffusion.yaml

* `cache_all_data`：加载所有数据集到内存中，某些平台的硬盘 IO 过于低下、同时内存容量 **远大于** 数据集体积时可以启用

* `duration`：训练时音频切片时长，可根据显存大小调整，**注意，该值必须小于训练集内音频的最短时间！**

* `batch_size`：单次训练加载到 GPU 的数据量，调整到低于显存容量的大小即可

* `timesteps` : 扩散模型总步数，默认为 1000.

* `k_step_max` : 训练时可仅训练`k_step_max`步扩散以节约训练时间，注意，该值必须小于`timesteps`，0 为训练整个扩散模型，**注意，如果不训练整个扩散模型将无法使用仅扩散模型推理！**

##### **声码器列表**

```py
nsf-hifigan
nsf-snake-hifigan
```

### 3. 生成 hubert 与 f0

```pyshell
python preprocess_hubert_f0.py --f0_predictor dio
```

f0_predictor 拥有以下选择

```py
crepe
dio
pm
harvest
rmvpe
fcpe
```

如果训练集过于嘈杂，请使用 crepe 处理 f0

如果省略 f0_predictor 参数，默认值为 rmvpe

尚若需要浅扩散功能（可选），需要增加--use_diff 参数，比如

```pyshell
python preprocess_hubert_f0.py --f0_predictor dio --use_diff
```

**加速预处理**
如若您的数据集比较大，可以尝试添加`--num_processes`参数：
```pyshell
python preprocess_hubert_f0.py --f0_predictor dio --use_diff --num_processes 8
```
所有的Workers会被自动分配到多个线程上

执行完以上步骤后 dataset 目录便是预处理完成的数据，可以删除 dataset_raw 文件夹了

## 🏋️‍ 训练

### 主模型训练

```pyshell
python train.py -c configs/config.json -m 44k
```

### 扩散模型（可选）

尚若需要浅扩散功能，需要训练扩散模型，扩散模型训练方法为：

```pyshell
python train_diff.py -c configs/diffusion.yaml
```

模型训练结束后，模型文件保存在`logs/44k`目录下，扩散模型在`logs/44k/diffusion`下

## 🤖 推理

使用 [inference_main.py](inference_main.py)

```pyshell
# 例
python inference_main.py -m "logs/44k/G_30400.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "nen"
```

必填项部分：
+ `-m` | `--model_path`：模型路径
+ `-c` | `--config_path`：配置文件路径
+ `-n` | `--clean_names`：wav 文件名列表，放在 raw 文件夹下
+ `-t` | `--trans`：音高调整，支持正负（半音）
+ `-s` | `--spk_list`：合成目标说话人名称
+ `-cl` | `--clip`：音频强制切片，默认 0 为自动切片，单位为秒/s

可选项部分：部分具体见下一节
+ `-lg` | `--linear_gradient`：两段音频切片的交叉淡入长度，如果强制切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值 0，单位为秒
+ `-f0p` | `--f0_predictor`：选择 F0 预测器，可选择 crepe,pm,dio,harvest,rmvpe,fcpe, 默认为 pm（注意：crepe 为原 F0 使用均值滤波器）
+ `-a` | `--auto_predict_f0`：语音转换自动预测音高，转换歌声时不要打开这个会严重跑调
+ `-cm` | `--cluster_model_path`：聚类模型或特征检索索引路径，留空则自动设为各方案模型的默认路径，如果没有训练聚类或特征检索则随便填
+ `-cr` | `--cluster_infer_ratio`：聚类方案或特征检索占比，范围 0-1，若没有训练聚类模型或特征检索则默认 0 即可
+ `-eh` | `--enhance`：是否使用 NSF_HIFIGAN 增强器，该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭
+ `-shd` | `--shallow_diffusion`：是否使用浅层扩散，使用后可解决一部分电音问题，默认关闭，该选项打开时，NSF_HIFIGAN 增强器将会被禁止
+ `-usm` | `--use_spk_mix`：是否使用角色融合/动态声线融合
+ `-lea` | `--loudness_envelope_adjustment`：输入源响度包络替换输出响度包络融合比例，越靠近 1 越使用输出响度包络
+ `-fr` | `--feature_retrieval`：是否使用特征检索，如果使用聚类模型将被禁用，且 cm 与 cr 参数将会变成特征检索的索引路径与混合比例

浅扩散设置：
+ `-dm` | `--diffusion_model_path`：扩散模型路径
+ `-dc` | `--diffusion_config_path`：扩散模型配置文件路径
+ `-ks` | `--k_step`：扩散步数，越大越接近扩散模型的结果，默认 100
+ `-od` | `--only_diffusion`：纯扩散模式，该模式不会加载 sovits 模型，以扩散模型推理
+ `-se` | `--second_encoding`：二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，有时候效果好，有时候效果差

### 注意！

如果使用`whisper-ppg` 声音编码器进行推理，需要将`--clip`设置为 25，`-lg`设置为 1。否则将无法正常推理。

## 🤔 可选项

如果前面的效果已经满意，或者没看明白下面在讲啥，那后面的内容都可以忽略，不影响模型使用（这些可选项影响比较小，可能在某些特定数据上有点效果，但大部分情况似乎都感知不太明显）

### 自动 f0 预测

4.0 模型训练过程会训练一个 f0 预测器，对于语音转换可以开启自动音高预测，如果效果不好也可以使用手动的，但转换歌声时请不要启用此功能！！！会严重跑调！！
+ 在 inference_main 中设置 auto_predict_f0 为 true 即可

### 聚类音色泄漏控制

介绍：聚类方案可以减小音色泄漏，使得模型训练出来更像目标的音色（但其实不是特别明显），但是单纯的聚类方案会降低模型的咬字（会口齿不清）（这个很明显），本模型采用了融合的方式，可以线性控制聚类方案与非聚类方案的占比，也就是可以手动在"像目标音色" 和 "咬字清晰" 之间调整比例，找到合适的折中点

使用聚类前面的已有步骤不用进行任何的变动，只需要额外训练一个聚类模型，虽然效果比较有限，但训练成本也比较低

+ 训练过程：
  + 使用 cpu 性能较好的机器训练，据我的经验在腾讯云 6 核 cpu 训练每个 speaker 需要约 4 分钟即可完成训练
  + 执行`python cluster/train_cluster.py`，模型的输出会在`logs/44k/kmeans_10000.pt`
  + 聚类模型目前可以使用 gpu 进行训练，执行`python cluster/train_cluster.py --gpu`
+ 推理过程：
  + `inference_main.py`中指定`cluster_model_path` 为模型输出文件，留空则默认为`logs/44k/kmeans_10000.pt`
  + `inference_main.py`中指定`cluster_infer_ratio`，`0`为完全不使用聚类，`1`为只使用聚类，通常设置`0.5`即可

### 特征检索

介绍：跟聚类方案一样可以减小音色泄漏，咬字比聚类稍好，但会降低推理速度，采用了融合的方式，可以线性控制特征检索与非特征检索的占比，

+ 训练过程：
  首先需要在生成 hubert 与 f0 后执行：

```pyshell
python train_index.py -c configs/config.json
```

模型的输出会在`logs/44k/feature_and_index.pkl`

+ 推理过程：
  + 需要首先指定`--feature_retrieval`，此时聚类方案会自动切换到特征检索方案
  + `inference_main.py`中指定`cluster_model_path` 为模型输出文件，留空则默认为`logs/44k/feature_and_index.pkl`
  + `inference_main.py`中指定`cluster_infer_ratio`，`0`为完全不使用特征检索，`1`为只使用特征检索，通常设置`0.5`即可


## 🗜️ 模型压缩

生成的模型含有继续训练所需的信息。如果确认不再训练，可以移除模型中此部分信息，得到约 1/3 大小的最终模型。

使用 [compress_model.py](compress_model.py)

```pyshell
# 例
python compress_model.py -c="configs/config.json" -i="logs/44k/G_30400.pth" -o="logs/44k/release.pth"
```

## 👨‍🔧 声线混合

### 静态声线混合

**参考`webUI.py`文件中，小工具/实验室特性的静态声线融合。**

介绍：该功能可以将多个声音模型合成为一个声音模型（多个模型参数的凸组合或线性组合），从而制造出现实中不存在的声线
**注意：**

1. 该功能仅支持单说话人的模型
2. 如果强行使用多说话人模型，需要保证多个模型的说话人数量相同，这样可以混合同一个 SpaekerID 下的声音
3. 保证所有待混合模型的 config.json 中的 model 字段是相同的
4. 输出的混合模型可以使用待合成模型的任意一个 config.json，但聚类模型将不能使用
5. 批量上传模型的时候最好把模型放到一个文件夹选中后一起上传
6. 混合比例调整建议大小在 0-100 之间，也可以调为其他数字，但在线性组合模式下会出现未知的效果
7. 混合完毕后，文件将会保存在项目根目录中，文件名为 output.pth
8. 凸组合模式会将混合比例执行 Softmax 使混合比例相加为 1，而线性组合模式不会

### 动态声线混合

**参考`spkmix.py`文件中关于动态声线混合的介绍**

角色混合轨道 编写规则：

角色 ID : \[\[起始时间 1, 终止时间 1, 起始数值 1, 起始数值 1], [起始时间 2, 终止时间 2, 起始数值 2, 起始数值 2]]

起始时间和前一个的终止时间必须相同，第一个起始时间必须为 0，最后一个终止时间必须为 1 （时间的范围为 0-1）

全部角色必须填写，不使用的角色填、[\[0., 1., 0., 0.]] 即可

融合数值可以随便填，在指定的时间段内从起始数值线性变化为终止数值，内部会自动确保线性组合为 1（凸组合条件），可以放心使用

推理的时候使用`--use_spk_mix`参数即可启用动态声线混合

## 📤 Onnx 导出

使用 [onnx_export.py](onnx_export.py)

+ 新建文件夹：`checkpoints` 并打开
+ 在`checkpoints`文件夹中新建一个文件夹作为项目文件夹，文件夹名为你的项目名称，比如`aziplayer`
+ 将你的模型更名为`model.pth`，配置文件更名为`config.json`，并放置到刚才创建的`aziplayer`文件夹下
+ 将 [onnx_export.py](onnx_export.py) 中`path = "NyaruTaffy"` 的 `"NyaruTaffy"` 修改为你的项目名称，`path = "aziplayer" (onnx_export_speaker_mix，为支持角色混合的 onnx 导出）`
+ 运行 [onnx_export.py](onnx_export.py)
+ 等待执行完毕，在你的项目文件夹下会生成一个`model.onnx`，即为导出的模型

注意：Hubert Onnx 模型请使用 MoeSS 提供的模型，目前无法自行导出（fairseq 中 Hubert 有不少 onnx 不支持的算子和涉及到常量的东西，在导出时会报错或者导出的模型输入输出 shape 和结果都有问题）

## 📎 引用及论文

| URL | 名称 | 标题 | 源码 |
| --- | ----------- | ----- | --------------------- |
|[2106.06103](https://arxiv.org/abs/2106.06103) | VITS (Synthesizer)| Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech | [jaywalnut310/vits](https://github.com/jaywalnut310/vits) |
|[2111.02392](https://arxiv.org/abs/2111.02392) | SoftVC (Speech Encoder)| A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion | [bshall/hubert](https://github.com/bshall/hubert) |
|[2204.09224](https://arxiv.org/abs/2204.09224) | ContentVec (Speech Encoder)| ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers | [auspicious3000/contentvec](https://github.com/auspicious3000/contentvec) |
|[2212.04356](https://arxiv.org/abs/2212.04356) | Whisper (Speech Encoder) | Robust Speech Recognition via Large-Scale Weak Supervision | [openai/whisper](https://github.com/openai/whisper) |
|[2110.13900](https://arxiv.org/abs/2110.13900) | WavLM (Speech Encoder) | WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing | [microsoft/unilm/wavlm](https://github.com/microsoft/unilm/tree/master/wavlm) |
|[2305.17651](https://arxiv.org/abs/2305.17651) | DPHubert (Speech Encoder) | DPHuBERT: Joint Distillation and Pruning of Self-Supervised Speech Models | [pyf98/DPHuBERT](https://github.com/pyf98/DPHuBERT) |
|[DOI:10.21437/Interspeech.2017-68](http://dx.doi.org/10.21437/Interspeech.2017-68) | Harvest (F0 Predictor) | Harvest: A high-performance fundamental frequency estimator from speech signals | [mmorise/World/harvest](https://github.com/mmorise/World/blob/master/src/harvest.cpp) |
|[aes35-000039](https://www.aes.org/e-lib/online/browse.cfm?elib=15165) | Dio (F0 Predictor) | Fast and reliable F0 estimation method based on the period extraction of vocal fold vibration of singing voice and speech | [mmorise/World/dio](https://github.com/mmorise/World/blob/master/src/dio.cpp) |
|[8461329](https://ieeexplore.ieee.org/document/8461329) | Crepe (F0 Predictor) | Crepe: A Convolutional Representation for Pitch Estimation | [maxrmorrison/torchcrepe](https://github.com/maxrmorrison/torchcrepe) |
|[DOI:10.1016/j.wocn.2018.07.001](https://doi.org/10.1016/j.wocn.2018.07.001) | Parselmouth (F0 Predictor) | Introducing Parselmouth: A Python interface to Praat | [YannickJadoul/Parselmouth](https://github.com/YannickJadoul/Parselmouth) |
|[2306.15412v2](https://arxiv.org/abs/2306.15412v2) | RMVPE (F0 Predictor) | RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music | [Dream-High/RMVPE](https://github.com/Dream-High/RMVPE) |
|[2010.05646](https://arxiv.org/abs/2010.05646) | HIFIGAN (Vocoder) | HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis | [jik876/hifi-gan](https://github.com/jik876/hifi-gan) |
|[1810.11946](https://arxiv.org/abs/1810.11946.pdf) | NSF (Vocoder) | Neural source-filter-based waveform model for statistical parametric speech synthesis | [openvpi/DiffSinger/modules/nsf_hifigan](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan)
|[2006.08195](https://arxiv.org/abs/2006.08195) | Snake (Vocoder) | Neural Networks Fail to Learn Periodic Functions and How to Fix It | [EdwardDixon/snake](https://github.com/EdwardDixon/snake)
|[2105.02446v3](https://arxiv.org/abs/2105.02446v3) | Shallow Diffusion (PostProcessing)| DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism | [CNChTu/Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) |
|[K-means](https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=01D65490BADCC216F350D06F84D721AD?doi=10.1.1.308.8619&rep=rep1&type=pdf) | Feature K-means Clustering (PreProcessing)| Some methods for classification and analysis of multivariate observations | 本代码库 |
| | Feature TopK Retrieval (PreProcessing)| Retrieval based Voice Conversion | [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) |

## ☀️ 旧贡献者

因为某些原因原作者进行了删库处理，本仓库重建之初由于组织成员疏忽直接重新上传了所有文件导致以前的 contributors 全部木大，现在在 README 里重新添加一个旧贡献者列表

*某些成员已根据其个人意愿不将其列出*

<table>
  <tr>
    <td align="center"><a href="https://github.com/MistEO"><img src="https://avatars.githubusercontent.com/u/18511905?v=4" width="100px;" alt=""/><br /><sub><b>MistEO</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/XiaoMiku01"><img src="https://avatars.githubusercontent.com/u/54094119?v=4" width="100px;" alt=""/><br /><sub><b>XiaoMiku01</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/ForsakenRei"><img src="https://avatars.githubusercontent.com/u/23041178?v=4" width="100px;" alt=""/><br /><sub><b>しぐれ</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/TomoGaSukunai"><img src="https://avatars.githubusercontent.com/u/25863522?v=4" width="100px;" alt=""/><br /><sub><b>TomoGaSukunai</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/Plachtaa"><img src="https://avatars.githubusercontent.com/u/112609742?v=4" width="100px;" alt=""/><br /><sub><b>Plachtaa</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/zdxiaoda"><img src="https://avatars.githubusercontent.com/u/45501959?v=4" width="100px;" alt=""/><br /><sub><b>zd 小达</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/Archivoice"><img src="https://avatars.githubusercontent.com/u/107520869?v=4" width="100px;" alt=""/><br /><sub><b>凍聲響世</b></sub></a><br /></td>
  </tr>
</table>

## 📚 一些法律条例参考

#### 任何国家，地区，组织和个人使用此项目必须遵守以下法律

#### 《民法典》

##### 第一千零一十九条

任何组织或者个人不得以丑化、污损，或者利用信息技术手段伪造等方式侵害他人的肖像权。未经肖像权人同意，不得制作、使用、公开肖像权人的肖像，但是法律另有规定的除外。未经肖像权人同意，肖像作品权利人不得以发表、复制、发行、出租、展览等方式使用或者公开肖像权人的肖像。对自然人声音的保护，参照适用肖像权保护的有关规定。

##### 第一千零二十四条

【名誉权】民事主体享有名誉权。任何组织或者个人不得以侮辱、诽谤等方式侵害他人的名誉权。

##### 第一千零二十七条

【作品侵害名誉权】行为人发表的文学、艺术作品以真人真事或者特定人为描述对象，含有侮辱、诽谤内容，侵害他人名誉权的，受害人有权依法请求该行为人承担民事责任。行为人发表的文学、艺术作品不以特定人为描述对象，仅其中的情节与该特定人的情况相似的，不承担民事责任。

#### 《[中华人民共和国宪法](http://www.gov.cn/guoqing/2018-03/22/content_5276318.htm)》

#### 《[中华人民共和国刑法](http://gongbao.court.gov.cn/Details/f8e30d0689b23f57bfc782d21035c3.html?sw=中华人民共和国刑法)》

#### 《[中华人民共和国民法典](http://gongbao.court.gov.cn/Details/51eb6750b8361f79be8f90d09bc202.html)》

#### 《[中华人民共和国合同法](http://www.npc.gov.cn/zgrdw/npc/lfzt/rlyw/2016-07/01/content_1992739.htm)》

## 💪 感谢所有的贡献者
<a href="https://github.com/svc-develop-team/so-vits-svc/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=svc-develop-team/so-vits-svc" />
</a>
