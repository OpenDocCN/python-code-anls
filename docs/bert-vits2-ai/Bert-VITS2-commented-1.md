# Bert-VITS2 源码解析 1

# `D:\src\Bert-VITS2\preprocess_text.py`

```python
import json  # 导入json模块
from collections import defaultdict  # 从collections模块中导入defaultdict类
from random import shuffle  # 从random模块中导入shuffle函数
from typing import Optional  # 从typing模块中导入Optional类型
import os  # 导入os模块
from tqdm import tqdm  # 从tqdm模块中导入tqdm函数
import click  # 导入click模块
from text.cleaner import clean_text  # 从text.cleaner模块中导入clean_text函数
from config import config  # 从config模块中导入config对象
from infer import latest_version  # 从infer模块中导入latest_version函数

preprocess_text_config = config.preprocess_text_config  # 设置preprocess_text_config为config.preprocess_text_config

@click.command()  # 装饰器，定义命令行接口
@click.option(
    "--transcription-path",
    default=preprocess_text_config.transcription_path,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)  # 添加命令行选项
@click.option("--cleaned-path", default=preprocess_text_config.cleaned_path)  # 添加命令行选项
@click.option("--train-path", default=preprocess_text_config.train_path)  # 添加命令行选项
@click.option("--val-path", default=preprocess_text_config.val_path)  # 添加命令行选项
@click.option(
    "--config-path",
    default=preprocess_text_config.config_path,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)  # 添加命令行选项
@click.option("--val-per-lang", default=preprocess_text_config.val_per_lang)  # 添加命令行选项
@click.option("--max-val-total", default=preprocess_text_config.max_val_total)  # 添加命令行选项
@click.option("--clean/--no-clean", default=preprocess_text_config.clean)  # 添加命令行选项
@click.option("-y", "--yml_config")  # 添加命令行选项
def preprocess(
    transcription_path: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    config_path: str,
    val_per_lang: int,
    max_val_total: int,
    clean: bool,
    yml_config: str,  # 这个不要删
):  # 定义preprocess函数及其参数

    # ...（以下省略）
```

# `D:\src\Bert-VITS2\resample.py`

```python
import os  # 导入os模块
import argparse  # 导入argparse模块
import librosa  # 导入librosa模块
from multiprocessing import Pool, cpu_count  # 从multiprocessing模块中导入Pool和cpu_count
import soundfile  # 导入soundfile模块
from tqdm import tqdm  # 从tqdm模块中导入tqdm
from config import config  # 从config模块中导入config

def process(item):  # 定义process函数，参数为item
    spkdir, wav_name, args = item  # 将item解包为spkdir, wav_name, args
    wav_path = os.path.join(args.in_dir, spkdir, wav_name)  # 拼接路径
    if os.path.exists(wav_path) and wav_path.lower().endswith(".wav"):  # 判断路径是否存在且以.wav结尾
        wav, sr = librosa.load(wav_path, sr=args.sr)  # 调用librosa.load加载音频文件
        soundfile.write(os.path.join(args.out_dir, spkdir, wav_name), wav, sr)  # 将音频文件写入指定路径

if __name__ == "__main__":  # 判断是否为主程序
    parser = argparse.ArgumentParser()  # 创建ArgumentParser对象
    parser.add_argument(  # 添加参数
        "--sr",
        type=int,
        default=config.resample_config.sampling_rate,
        help="sampling rate",
    )
    parser.add_argument(  # 添加参数
        "--in_dir",
        type=str,
        default=config.resample_config.in_dir,
        help="path to source dir",
    )
    parser.add_argument(  # 添加参数
        "--out_dir",
        type=str,
        default=config.resample_config.out_dir,
        help="path to target dir",
    )
    parser.add_argument(  # 添加参数
        "--processes",
        type=int,
        default=0,
        help="cpu_processes",
    )
    args, _ = parser.parse_known_args()  # 解析参数

    if args.processes == 0:  # 判断参数值
        processes = cpu_count() - 2 if cpu_count() > 4 else 1  # 计算processes的值
    else:
        processes = args.processes  # 获取参数值
    pool = Pool(processes=processes)  # 创建进程池

    tasks = []  # 创建空列表

    for dirpath, _, filenames in os.walk(args.in_dir):  # 遍历目录
        spk_dir = os.path.relpath(dirpath, args.in_dir)  # 获取相对路径
        spk_dir_out = os.path.join(args.out_dir, spk_dir)  # 拼接路径
        if not os.path.isdir(spk_dir_out):  # 判断路径是否为目录
            os.makedirs(spk_dir_out, exist_ok=True)  # 创建目录
        for filename in filenames:  # 遍历文件
            if filename.lower().endswith(".wav"):  # 判断文件名是否以.wav结尾
                twople = (spk_dir, filename, args)  # 创建元组
                tasks.append(twople)  # 将元组添加到列表

    for _ in tqdm(  # 使用tqdm显示进度
        pool.imap_unordered(process, tasks),  # 并行处理任务
    ):
        pass

    pool.close()  # 关闭进程池
    pool.join()  # 阻塞主进程，直到所有子进程执行完毕

    print("音频重采样完毕!")  # 打印信息
```

# `D:\src\Bert-VITS2\resample_legacy.py`

```python
import os  # 导入os模块
import argparse  # 导入argparse模块
import librosa  # 导入librosa模块
from multiprocessing import Pool, cpu_count  # 从multiprocessing模块中导入Pool和cpu_count
import soundfile  # 导入soundfile模块
from tqdm import tqdm  # 从tqdm模块中导入tqdm
from config import config  # 从config模块中导入config

def process(item):  # 定义process函数，参数为item
    wav_name, args = item  # 将item解包为wav_name和args
    wav_path = os.path.join(args.in_dir, wav_name)  # 拼接路径
    if os.path.exists(wav_path) and wav_path.lower().endswith(".wav"):  # 判断路径是否存在且以.wav结尾
        wav, sr = librosa.load(wav_path, sr=args.sr)  # 调用librosa.load加载音频文件
        soundfile.write(os.path.join(args.out_dir, wav_name), wav, sr)  # 将音频文件写入指定路径

if __name__ == "__main__":  # 判断是否为主程序
    parser = argparse.ArgumentParser()  # 创建ArgumentParser对象
    parser.add_argument(  # 添加参数
        "--sr",
        type=int,
        default=config.resample_config.sampling_rate,
        help="sampling rate",
    )
    parser.add_argument(  # 添加参数
        "--in_dir",
        type=str,
        default=config.resample_config.in_dir,
        help="path to source dir",
    )
    parser.add_argument(  # 添加参数
        "--out_dir",
        type=str,
        default=config.resample_config.out_dir,
        help="path to target dir",
    )
    parser.add_argument(  # 添加参数
        "--processes",
        type=int,
        default=0,
        help="cpu_processes",
    )
    args, _ = parser.parse_known_args()  # 解析参数

    if args.processes == 0:  # 判断参数值
        processes = cpu_count() - 2 if cpu_count() > 4 else 1  # 计算processes的值
    else:
        processes = args.processes  # 获取参数值
    pool = Pool(processes=processes)  # 创建进程池

    tasks = []  # 创建空列表

    for dirpath, _, filenames in os.walk(args.in_dir):  # 遍历目录
        if not os.path.isdir(args.out_dir):  # 判断目录是否存在
            os.makedirs(args.out_dir, exist_ok=True)  # 创建目录
        for filename in filenames:  # 遍历文件
            if filename.lower().endswith(".wav"):  # 判断文件名是否以.wav结尾
                tasks.append((filename, args))  # 将文件名和参数添加到tasks列表中

    for _ in tqdm(  # 使用tqdm显示进度
        pool.imap_unordered(process, tasks),  # 使用进程池处理任务
    ):
        pass  # 空语句

    pool.close()  # 关闭进程池
    pool.join()  # 等待所有子进程结束

    print("音频重采样完毕!")  # 打印信息
```

# `D:\src\Bert-VITS2\re_matching.py`

```python
import re

# 使用正则表达式匹配<语言>标签和其后的文本
def extract_language_and_text_updated(speaker, dialogue):
    pattern_language_text = r"<(\S+?)>([^<]+)"
    matches = re.findall(pattern_language_text, dialogue, re.DOTALL)
    speaker = speaker[1:-1]
    # 清理文本：去除两边的空白字符
    matches_cleaned = [(lang.upper(), text.strip()) for lang, text in matches]
    matches_cleaned.append(speaker)
    return matches_cleaned

# 验证说话人的正则表达式
def validate_text(input_text):
    pattern_speaker = r"(\[\S+?\])((?:\s*<\S+?>[^<\[\]]+?)+)"
    # 使用re.DOTALL标志使.匹配包括换行符在内的所有字符
    matches = re.findall(pattern_speaker, input_text, re.DOTALL)
    # 对每个匹配到的说话人内容进行进一步验证
    for _, dialogue in matches:
        language_text_matches = extract_language_and_text_updated(_, dialogue)
        if not language_text_matches:
            return (
                False,
                "Error: Invalid format detected in dialogue content. Please check your input.",
            )
    # 如果输入的文本中没有找到任何匹配项
    if not matches:
        return (
            False,
            "Error: No valid speaker format detected. Please check your input.",
        )
    return True, "Input is valid."

# 匹配文本
def text_matching(text: str) -> list:
    speaker_pattern = r"(\[\S+?\])(.+?)(?=\[\S+?\]|$)"
    matches = re.findall(speaker_pattern, text, re.DOTALL)
    result = []
    for speaker, dialogue in matches:
        result.append(extract_language_and_text_updated(speaker, dialogue))
    return result

# 按段分
def cut_para(text):
    splitted_para = re.split("[\n]", text)
    splitted_para = [
        sentence.strip() for sentence in splitted_para if sentence.strip()
    ]
    return splitted_para

# 断句
def cut_sent(para):
    para = re.sub("([。！;？\?])([^”’])", r"\1\n\2", para)  # 单字符断句符
    para = re.sub("(\.{6})([^”’])", r"\1\n\2", para)  # 英文省略号
    para = re.sub("(\…{2})([^”’])", r"\1\n\2", para)  # 中文省略号
    para = re.sub("([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    return para.split("\n")

if __name__ == "__main__":
    text = """
    [说话人1]
    [说话人2]<zh>你好吗？<jp>元気ですか？<jp>こんにちは，世界。<zh>你好吗？
    [说话人3]<zh>谢谢。<jp>どういたしまして。
    """
    text_matching(text)
    # 测试函数
    test_text = """
    [说话人1]<zh>你好，こんにちは！<jp>こんにちは，世界。
    [说话人2]<zh>你好吗？
    """
    text_matching(test_text)
    res = validate_text(test_text)
    print(res)
```

# `D:\src\Bert-VITS2\server_fastapi.py`

```python
# api服务 多版本多模型 fastapi实现
```
```python
# 导入logging模块
import logging
```
```python
# 导入gc模块
import gc
```
```python
# 导入random模块
import random
```
```python
# 导入librosa模块
import librosa
```
```python
# 导入gradio模块
import gradio
```
```python
# 导入numpy模块
import numpy as np
```
```python
# 导入utils模块
import utils
```
```python
# 从fastapi模块中导入FastAPI, Query, Request, File, UploadFile, Form, Response, FileResponse, StaticFiles
from fastapi import FastAPI, Query, Request, File, UploadFile, Form, Response, FileResponse, StaticFiles
```
```python
# 从io模块中导入BytesIO
from io import BytesIO
```
```python
# 从scipy.io模块中导入wavfile
from scipy.io import wavfile
```
```python
# 导入torch模块
import torch
```
```python
# 导入webbrowser模块
import webbrowser
```
```python
# 导入psutil模块
import psutil
```
```python
# 导入GPUtil模块
import GPUtil
```
```python
# 从typing模块中导入Dict, Optional, List, Set, Union
from typing import Dict, Optional, List, Set, Union
```
```python
# 导入os模块
import os
```
```python
# 从tools.log模块中导入logger
from tools.log import logger
```
```python
# 从urllib.parse模块中导入unquote
from urllib.parse import unquote
```
```python
# 从infer模块中导入infer, get_net_g, latest_version
from infer import infer, get_net_g, latest_version
```
```python
# 从config模块中导入config
from config import config
```
```python
# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```
```python
# 定义模型封装类Model
class Model:
    """模型封装类"""

    def __init__(self, config_path: str, model_path: str, device: str, language: str):
        self.config_path: str = os.path.normpath(config_path)
        self.model_path: str = os.path.normpath(model_path)
        self.device: str = device
        self.language: str = language
        self.hps = utils.get_hparams_from_file(config_path)
        self.spk2id: Dict[str, int] = self.hps.data.spk2id  # spk - id 映射字典
        self.id2spk: Dict[int, str] = dict()  # id - spk 映射字典
        for speaker, speaker_id in self.hps.data.spk2id.items():
            self.id2spk[speaker_id] = speaker
        self.version: str = (
            self.hps.version if hasattr(self.hps, "version") else latest_version
        )
        self.net_g = get_net_g(
            model_path=model_path,
            version=self.version,
            device=device,
            hps=self.hps,
        )

    def to_dict(self) -> Dict[str, any]:
        return {
            "config_path": self.config_path,
            "model_path": self.model_path,
            "device": self.device,
            "language": self.language,
            "spk2id": self.spk2id,
            "id2spk": self.id2spk,
            "version": self.version,
        }
```
```python
# 定义Models类
class Models:
    def __init__(self):
        self.models: Dict[int, Model] = dict()
        self.num = 0
        # spkInfo[角色名][模型id] = 角色id
        self.spk_info: Dict[str, Dict[int, int]] = dict()
        self.path2ids: Dict[str, Set[int]] = dict()  # 路径指向的model的id

    def init_model(
        self, config_path: str, model_path: str, device: str, language: str
    ) -> int:
        """
        初始化并添加一个模型

        :param config_path: 模型config.json路径
        :param model_path: 模型路径
        :param device: 模型推理使用设备
        :param language: 模型推理默认语言
        """
        # 若文件不存在则不进行加载
        if not os.path.isfile(model_path):
            if model_path != "":
                logger.warning(f"模型文件{model_path} 不存在，不进行初始化")
            return self.num
        if not os.path.isfile(config_path):
            if config_path != "":
                logger.warning(f"配置文件{config_path} 不存在，不进行初始化")
            return self.num

        # 若路径中的模型已存在，则不添加模型，若不存在，则进行初始化。
        model_path = os.path.realpath(model_path)
        if model_path not in self.path2ids.keys():
            self.path2ids[model_path] = {self.num}
            self.models[self.num] = Model(
                config_path=config_path,
                model_path=model_path,
                device=device,
                language=language,
            )
            logger.success(f"添加模型{model_path}，使用配置文件{os.path.realpath(config_path)}")
        else:
            # 获取一个指向id
            m_id = next(iter(self.path2ids[model_path]))
            self.models[self.num] = self.models[m_id]
            self.path2ids[model_path].add(self.num)
            logger.success("模型已存在，添加模型引用。")
        # 添加角色信息
        for speaker, speaker_id in self.models[self.num].spk2id.items():
            if speaker not in self.spk_info.keys():
                self.spk_info[speaker] = {self.num: speaker_id}
            else:
                self.spk_info[speaker][self.num] = speaker_id
        # 修改计数
        self.num += 1
        return self.num - 1

    def del_model(self, index: int) -> Optional[int]:
        """删除对应序号的模型，若不存在则返回None"""
        if index not in self.models.keys():
            return None
        # 删除角色信息
        for speaker, speaker_id in self.models[index].spk2id.items():
            self.spk_info[speaker].pop(index)
            if len(self.spk_info[speaker]) == 0:
                # 若对应角色的所有模型都被删除，则清除该角色信息
                self.spk_info.pop(speaker)
        # 删除路径信息
        model_path = os.path.realpath(self.models[index].model_path)
        self.path2ids[model_path].remove(index)
        if len(self.path2ids[model_path]) == 0:
            self.path2ids.pop(model_path)
            logger.success(f"删除模型{model_path}, id = {index}")
        else:
            logger.success(f"删除模型引用{model_path}, id = {index}")
        # 删除模型
        self.models.pop(index)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return index

    def get_models(self):
        """获取所有模型"""
        return self.models
```
```python
# 如果是主程序入口
if __name__ == "__main__":
    app = FastAPI()
    app.logger = logger
    # 挂载静态文件
    logger.info("开始挂载网页页面")
    StaticDir: str = "./Web"
    if not os.path.isdir(StaticDir):
        logger.warning(
            "缺少网页资源，无法开启网页页面，如有需要请在 https://github.com/jiangyuxiaoxiao/Bert-VITS2-UI 或者Bert-VITS对应版本的release页面下载"
        )
    else:
        dirs = [fir.name for fir in os.scandir(StaticDir) if fir.is_dir()]
        files = [fir.name for fir in os.scandir(StaticDir) if fir.is_dir()]
        for dirName in dirs:
            app.mount(
                f"/{dirName}",
                StaticFiles(directory=f"./{StaticDir}/{dirName}"),
                name=dirName,
            )
    loaded_models = Models()
    # 加载模型
    logger.info("开始加载模型")
    models_info = config.server_config.models
    for model_info in models_info:
        loaded_models.init_model(
            config_path=model_info["config"],
            model_path=model_info["model"],
            device=model_info["device"],
            language=model_info["language"],
        )
    # ...（省略部分代码）
```

# `D:\src\Bert-VITS2\spec_gen.py`

```python
import torch  # 导入torch库
from tqdm import tqdm  # 从tqdm库中导入tqdm函数
from multiprocessing import Pool  # 从multiprocessing库中导入Pool类
from mel_processing import spectrogram_torch, mel_spectrogram_torch  # 从mel_processing库中导入spectrogram_torch和mel_spectrogram_torch函数
from utils import load_wav_to_torch  # 从utils库中导入load_wav_to_torch函数

class AudioProcessor:  # 定义一个名为AudioProcessor的类
    def __init__(  # 定义初始化方法，接收一系列参数
        self,
        max_wav_value,
        use_mel_spec_posterior,
        filter_length,
        n_mel_channels,
        sampling_rate,
        hop_length,
        win_length,
        mel_fmin,
        mel_fmax,
    ):
        self.max_wav_value = max_wav_value  # 将max_wav_value参数赋值给self.max_wav_value
        self.use_mel_spec_posterior = use_mel_spec_posterior  # 将use_mel_spec_posterior参数赋值给self.use_mel_spec_posterior
        self.filter_length = filter_length  # 将filter_length参数赋值给self.filter_length
        self.n_mel_channels = n_mel_channels  # 将n_mel_channels参数赋值给self.n_mel_channels
        self.sampling_rate = sampling_rate  # 将sampling_rate参数赋值给self.sampling_rate
        self.hop_length = hop_length  # 将hop_length参数赋值给self.hop_length
        self.win_length = win_length  # 将win_length参数赋值给self.win_length
        self.mel_fmin = mel_fmin  # 将mel_fmin参数赋值给self.mel_fmin
        self.mel_fmax = mel_fmax  # 将mel_fmax参数赋值给self.mel_fmax

    def process_audio(self, filename):  # 定义名为process_audio的方法，接收filename参数
        audio, sampling_rate = load_wav_to_torch(filename)  # 调用load_wav_to_torch函数，将返回值分别赋值给audio和sampling_rate
        audio_norm = audio / self.max_wav_value  # 将audio除以self.max_wav_value得到audio_norm
        audio_norm = audio_norm.unsqueeze(0)  # 调用unsqueeze方法，将audio_norm在0维度上扩展
        spec_filename = filename.replace(".wav", ".spec.pt")  # 将filename中的".wav"替换为".spec.pt"得到spec_filename
        if self.use_mel_spec_posterior:  # 如果self.use_mel_spec_posterior为True
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")  # 将spec_filename中的".spec.pt"替换为".mel.pt"
        try:  # 尝试执行以下代码
            spec = torch.load(spec_filename)  # 调用torch.load函数，将返回值赋值给spec
        except:  # 如果出现异常
            if self.use_mel_spec_posterior:  # 如果self.use_mel_spec_posterior为True
                spec = mel_spectrogram_torch(  # 调用mel_spectrogram_torch函数
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.mel_fmin,
                    self.mel_fmax,
                    center=False,
                )
            else:  # 如果self.use_mel_spec_posterior为False
                spec = spectrogram_torch(  # 调用spectrogram_torch函数
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            spec = torch.squeeze(spec, 0)  # 调用torch.squeeze方法，将spec在0维度上压缩
            torch.save(spec, spec_filename)  # 调用torch.save函数，将spec保存到spec_filename
        return spec, audio_norm  # 返回spec和audio_norm

processor = AudioProcessor(  # 创建AudioProcessor对象processor
    max_wav_value=32768.0,
    use_mel_spec_posterior=False,
    filter_length=2048,
    n_mel_channels=128,
    sampling_rate=44100,
    hop_length=512,
    win_length=2048,
    mel_fmin=0.0,
    mel_fmax="null",
)

with open("filelists/train.list", "r") as f:  # 打开文件"filelists/train.list"，赋值给f
    filepaths = [line.split("|")[0] for line in f]  # 遍历文件f的每一行，取每一行的第一部分作为audiopath，将结果存储在filepaths列表中

with Pool(processes=32) as pool:  # 创建进程池pool，指定进程数为32
    with tqdm(total=len(filepaths)) as pbar:  # 创建tqdm对象pbar，总数为filepaths的长度
        for i, _ in enumerate(pool.imap_unordered(processor.process_audio, filepaths)):  # 遍历pool.imap_unordered(processor.process_audio, filepaths)的结果
            pbar.update()  # 更新pbar
```

# `D:\src\Bert-VITS2\train_ms.py`

```python
# flake8: noqa: E402
import platform  # 导入platform模块
import os  # 导入os模块
import torch  # 导入torch模块
from torch.nn import functional as F  # 从torch.nn模块中导入functional并重命名为F
from torch.utils.data import DataLoader  # 从torch.utils.data模块中导入DataLoader
from torch.utils.tensorboard import SummaryWriter  # 从torch.utils.tensorboard模块中导入SummaryWriter
import torch.distributed as dist  # 导入torch.distributed模块并重命名为dist
from torch.nn.parallel import DistributedDataParallel as DDP  # 从torch.nn.parallel模块中导入DistributedDataParallel并重命名为DDP
from torch.cuda.amp import autocast, GradScaler  # 从torch.cuda.amp模块中导入autocast和GradScaler
from tqdm import tqdm  # 导入tqdm模块
import logging  # 导入logging模块
from config import config  # 从config模块中导入config
import argparse  # 导入argparse模块
import datetime  # 导入datetime模块
import gc  # 导入gc模块
import commons  # 导入commons模块
import utils  # 导入utils模块
from data_utils import (  # 从data_utils模块中导入TextAudioSpeakerLoader, TextAudioSpeakerCollate, DistributedBucketSampler
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
)
from models import (  # 从models模块中导入SynthesizerTrn, MultiPeriodDiscriminator, DurationDiscriminator, WavLMDiscriminator
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    DurationDiscriminator,
    WavLMDiscriminator,
)
from losses import (  # 从losses模块中导入generator_loss, discriminator_loss, feature_loss, kl_loss, WavLMLoss
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss,
    WavLMLoss,
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch  # 从mel_processing模块中导入mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols  # 从text.symbols模块中导入symbols

# 设置一些torch的后端参数
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True  # If encontered training problem,please try to disable TF32.
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.sdp_kernel("flash")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)  # Not available if torch version is lower than 2.0
global_step = 0  # 初始化global_step为0

# 环境变量解析
envs = config.train_ms_config.env
for env_name, env_value in envs.items():
    if env_name not in os.environ.keys():
        print("加载config中的配置{}".format(str(env_value)))
        os.environ[env_name] = str(env_value)
print(
    "加载环境变量 \nMASTER_ADDR: {},\nMASTER_PORT: {},\nWORLD_SIZE: {},\nRANK: {},\nLOCAL_RANK: {}".format(
        os.environ["MASTER_ADDR"],
        os.environ["MASTER_PORT"],
        os.environ["WORLD_SIZE"],
        os.environ["RANK"],
        os.environ["LOCAL_RANK"],
    )
)

backend = "nccl"
if platform.system() == "Windows":
    backend = "gloo"  # If Windows,switch to gloo backend.
dist.init_process_group(
    backend=backend,
    init_method="env://",
    timeout=datetime.timedelta(seconds=300),
)  # Use torchrun instead of mp.spawn
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
n_gpus = dist.get_world_size()

# 命令行/config.yml配置解析
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default=config.train_ms_config.config_path,
    help="JSON file for configuration",
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="数据集文件夹路径，请注意，数据不再默认放在/logs文件夹下。如果需要用命令行配置，请声明相对于根目录的路径",
    default=config.dataset_path,
)
args = parser.parse_args()
model_dir = os.path.join(args.model, config.train_ms_config.model)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
hps = utils.get_hparams_from_file(args.config)
hps.model_dir = model_dir
if os.path.realpath(args.config) != os.path.realpath(
    config.train_ms_config.config_path
):
    with open(args.config, "r", encoding="utf-8") as f:
        data = f.read()
    with open(config.train_ms_config.config_path, "w", encoding="utf-8") as f:
        f.write(data)

torch.manual_seed(hps.train.seed)
torch.cuda.set_device(local_rank)

# 其他代码...
```

# `D:\src\Bert-VITS2\transforms.py`

```python
import torch  # 导入torch库
from torch.nn import functional as F  # 从torch.nn库中导入functional模块并重命名为F
import numpy as np  # 导入numpy库

DEFAULT_MIN_BIN_WIDTH = 1e-3  # 设置默认最小箱宽度
DEFAULT_MIN_BIN_HEIGHT = 1e-3  # 设置默认最小箱高度
DEFAULT_MIN_DERIVATIVE = 1e-3  # 设置默认最小导数

# 定义piecewise_rational_quadratic_transform函数
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
    # ...
    return outputs, logabsdet  # 返回outputs和logabsdet

# 定义searchsorted函数
def searchsorted(bin_locations, inputs, eps=1e-6):
    # ...
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1  # 返回torch.sum的结果减1

# 定义unconstrained_rational_quadratic_spline函数
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
    # ...
    return outputs, logabsdet  # 返回outputs和logabsdet

# 定义rational_quadratic_spline函数
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
    # ...
    return outputs, logabsdet  # 返回outputs和logabsdet
```

# `D:\src\Bert-VITS2\update_status.py`

```python
import os  # 导入os模块
import gradio as gr  # 导入gradio模块

lang_dict = {"EN(英文)": "_en", "ZH(中文)": "_zh", "JP(日语)": "_jp"}  # 定义一个字典

def raw_dir_convert_to_path(target_dir: str, lang):
    res = target_dir.rstrip("/").rstrip("\\")  # 去除目标目录末尾的斜杠
    if (not target_dir.startswith("raw")) and (not target_dir.startswith("./raw")):  # 如果目标目录不以"raw"或"./raw"开头
        res = os.path.join("./raw", res)  # 将目标目录添加到"./raw"路径下
    if (
        (not res.endswith("_zh"))
        and (not res.endswith("_jp"))
        and (not res.endswith("_en"))
    ):  # 如果目标目录不以"_zh"、"_jp"、"_en"结尾
        res += lang_dict[lang]  # 在目标目录末尾添加对应语言的后缀
    return res  # 返回处理后的目标目录路径

def update_g_files():
    g_files = []  # 定义一个空列表
    cnt = 0  # 计数器初始化为0
    for root, dirs, files in os.walk(os.path.abspath("./logs")):  # 遍历"./logs"目录下的文件和子目录
        for file in files:  # 遍历文件
            if file.startswith("G_") and file.endswith(".pth"):  # 如果文件以"G_"开头且以".pth"结尾
                g_files.append(os.path.join(root, file))  # 将文件路径添加到g_files列表中
                cnt += 1  # 计数器加1
    print(g_files)  # 打印g_files列表
    return f"更新模型列表完成, 共找到{cnt}个模型", gr.Dropdown.update(choices=g_files)  # 返回更新模型列表完成的信息和更新下拉框选项

# 其余函数的注释与解释与上述相似，不再赘述
```

# `D:\src\Bert-VITS2\utils.py`

```python
import os  # Import the os module
import glob  # Import the glob module
import argparse  # Import the argparse module
import logging  # Import the logging module
import json  # Import the json module
import shutil  # Import the shutil module
import subprocess  # Import the subprocess module
import numpy as np  # Import the numpy module and alias it as np
from huggingface_hub import hf_hub_download  # Import the hf_hub_download function from the huggingface_hub module
from scipy.io.wavfile import read  # Import the read function from the scipy.io.wavfile module
import torch  # Import the torch module
import re  # Import the re module

MATPLOTLIB_FLAG = False  # Set the MATPLOTLIB_FLAG variable to False

logger = logging.getLogger(__name__)  # Get the logger for the current module
```

# `D:\src\Bert-VITS2\webui.py`

```python
# flake8: noqa: E402
import os  # 导入os模块
import logging  # 导入logging模块
import re_matching  # 导入re_matching模块
from tools.sentence import split_by_language  # 从tools.sentence模块中导入split_by_language函数

logging.getLogger("numba").setLevel(logging.WARNING)  # 设置numba的日志级别为WARNING
logging.getLogger("markdown_it").setLevel(logging.WARNING)  # 设置markdown_it的日志级别为WARNING
logging.getLogger("urllib3").setLevel(logging.WARNING)  # 设置urllib3的日志级别为WARNING
logging.getLogger("matplotlib").setLevel(logging.WARNING)  # 设置matplotlib的日志级别为WARNING

logging.basicConfig(  # 配置logging的基本信息
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)  # 获取当前模块的logger

import torch  # 导入torch模块
import utils  # 导入utils模块
from infer import infer, latest_version, get_net_g, infer_multilang  # 从infer模块中导入infer、latest_version、get_net_g、infer_multilang函数
import gradio as gr  # 导入gradio模块并重命名为gr
import webbrowser  # 导入webbrowser模块
import numpy as np  # 导入numpy模块并重命名为np
from config import config  # 从config模块中导入config类
from tools.translate import translate  # 从tools.translate模块中导入translate函数
import librosa  # 导入librosa模块

net_g = None  # 初始化net_g为None

device = config.webui_config.device  # 从config中获取设备信息
if device == "mps":  # 如果设备为"mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 设置环境变量PYTORCH_ENABLE_MPS_FALLBACK为"1"


def generate_audio(  # 定义generate_audio函数
    slices,  # 参数slices
    sdp_ratio,  # 参数sdp_ratio
    noise_scale,  # 参数noise_scale
    noise_scale_w,  # 参数noise_scale_w
    length_scale,  # 参数length_scale
    speaker,  # 参数speaker
    language,  # 参数language
    reference_audio,  # 参数reference_audio
    emotion,  # 参数emotion
    style_text,  # 参数style_text
    style_weight,  # 参数style_weight
    skip_start=False,  # 参数skip_start，默认值为False
    skip_end=False,  # 参数skip_end，默认值为False
):  # 函数定义结束
    audio_list = []  # 初始化audio_list为空列表
    with torch.no_grad():  # 使用torch的no_grad上下文管理器
        for idx, piece in enumerate(slices):  # 遍历slices中的元素
            skip_start = idx != 0  # 设置skip_start为idx是否不等于0
            skip_end = idx != len(slices) - 1  # 设置skip_end为idx是否不等于slices的长度减1
            audio = infer(  # 调用infer函数
                piece,  # 参数piece
                reference_audio=reference_audio,  # 参数reference_audio
                emotion=emotion,  # 参数emotion
                sdp_ratio=sdp_ratio,  # 参数sdp_ratio
                noise_scale=noise_scale,  # 参数noise_scale
                noise_scale_w=noise_scale_w,  # 参数noise_scale_w
                length_scale=length_scale,  # 参数length_scale
                sid=speaker,  # 参数sid
                language=language,  # 参数language
                hps=hps,  # 参数hps
                net_g=net_g,  # 参数net_g
                device=device,  # 参数device
                skip_start=skip_start,  # 参数skip_start
                skip_end=skip_end,  # 参数skip_end
                style_text=style_text,  # 参数style_text
                style_weight=style_weight,  # 参数style_weight
            )  # infer函数调用结束
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)  # 将audio转换为16位wav格式
            audio_list.append(audio16bit)  # 将audio16bit添加到audio_list中
    return audio_list  # 返回audio_list


def generate_audio_multilang(  # 定义generate_audio_multilang函数
    slices,  # 参数slices
    sdp_ratio,  # 参数sdp_ratio
    noise_scale,  # 参数noise_scale
    noise_scale_w,  # 参数noise_scale_w
    length_scale,  # 参数length_scale
    speaker,  # 参数speaker
    language,  # 参数language
    reference_audio,  # 参数reference_audio
    emotion,  # 参数emotion
    skip_start=False,  # 参数skip_start，默认值为False
    skip_end=False,  # 参数skip_end，默认值为False
):  # 函数定义结束
    audio_list = []  # 初始化audio_list为空列表
    with torch.no_grad():  # 使用torch的no_grad上下文管理器
        for idx, piece in enumerate(slices):  # 遍历slices中的元素
            skip_start = idx != 0  # 设置skip_start为idx是否不等于0
            skip_end = idx != len(slices) - 1  # 设置skip_end为idx是否不等于slices的长度减1
            audio = infer_multilang(  # 调用infer_multilang函数
                piece,  # 参数piece
                reference_audio=reference_audio,  # 参数reference_audio
                emotion=emotion,  # 参数emotion
                sdp_ratio=sdp_ratio,  # 参数sdp_ratio
                noise_scale=noise_scale,  # 参数noise_scale
                noise_scale_w=noise_scale_w,  # 参数noise_scale_w
                length_scale=length_scale,  # 参数length_scale
                sid=speaker,  # 参数sid
                language=language[idx],  # 参数language
                hps=hps,  # 参数hps
                net_g=net_g,  # 参数net_g
                device=device,  # 参数device
                skip_start=skip_start,  # 参数skip_start
                skip_end=skip_end,  # 参数skip_end
            )  # infer_multilang函数调用结束
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)  # 将audio转换为16位wav格式
            audio_list.append(audio16bit)  # 将audio16bit添加到audio_list中
    return audio_list  # 返回audio_list
```

# `D:\src\Bert-VITS2\webui_preprocess.py`

```python
import gradio as gr  # 导入 gradio 库
import webbrowser  # 导入 webbrowser 库
import os  # 导入 os 库
import json  # 导入 json 库
import subprocess  # 导入 subprocess 库
import shutil  # 导入 shutil 库

# 获取数据集路径
def get_path(data_dir):
    start_path = os.path.join("./data", data_dir)  # 获取数据集的起始路径
    lbl_path = os.path.join(start_path, "esd.list")  # 获取标签文件路径
    train_path = os.path.join(start_path, "train.list")  # 获取训练文件路径
    val_path = os.path.join(start_path, "val.list")  # 获取验证文件路径
    config_path = os.path.join(start_path, "configs", "config.json")  # 获取配置文件路径
    return start_path, lbl_path, train_path, val_path, config_path  # 返回路径信息

# 生成配置文件
def generate_config(data_dir, batch_size):
    assert data_dir != "", "数据集名称不能为空"  # 断言数据集名称不为空
    start_path, _, train_path, val_path, config_path = get_path(data_dir)  # 获取路径信息
    if os.path.isfile(config_path):  # 如果配置文件存在
        config = json.load(open(config_path, "r", encoding="utf-8"))  # 读取配置文件
    else:
        config = json.load(open("configs/config.json", "r", encoding="utf-8"))  # 否则读取默认配置文件
    config["data"]["training_files"] = train_path  # 设置训练文件路径
    config["data"]["validation_files"] = val_path  # 设置验证文件路径
    config["train"]["batch_size"] = batch_size  # 设置批大小
    out_path = os.path.join(start_path, "configs")  # 输出路径
    if not os.path.isdir(out_path):  # 如果输出路径不存在
        os.mkdir(out_path)  # 创建输出路径
    model_path = os.path.join(start_path, "models")  # 模型路径
    if not os.path.isdir(model_path):  # 如果模型路径不存在
        os.mkdir(model_path)  # 创建模型路径
    with open(config_path, "w", encoding="utf-8") as f:  # 打开配置文件
        json.dump(config, f, indent=4)  # 写入配置信息
    if not os.path.exists("config.yml"):  # 如果配置文件不存在
        shutil.copy(src="default_config.yml", dst="config.yml")  # 复制默认配置文件
    return "配置文件生成完成"  # 返回信息

# 音频文件预处理
def resample(data_dir):
    assert data_dir != "", "数据集名称不能为空"  # 断言数据集名称不为空
    start_path, _, _, _, config_path = get_path(data_dir)  # 获取路径信息
    in_dir = os.path.join(start_path, "raw")  # 输入文件夹路径
    out_dir = os.path.join(start_path, "wavs")  # 输出文件夹路径
    subprocess.run(  # 运行子进程
        f"python resample_legacy.py "  # 执行预处理脚本
        f"--sr 44100 "  # 设置采样率
        f"--in_dir {in_dir} "  # 设置输入文件夹路径
        f"--out_dir {out_dir} ",  # 设置输出文件夹路径
        shell=True,  # 在 shell 中执行
    )
    return "音频文件预处理完成"  # 返回信息

# 预处理标签文件
def preprocess_text(data_dir):
    assert data_dir != "", "数据集名称不能为空"  # 断言数据集名称不为空
    start_path, lbl_path, train_path, val_path, config_path = get_path(data_dir)  # 获取路径信息
    lines = open(lbl_path, "r", encoding="utf-8").readlines()  # 读取标签文件内容
    with open(lbl_path, "w", encoding="utf-8") as f:  # 打开标签文件
        for line in lines:  # 遍历每行内容
            path, spk, language, text = line.strip().split("|")  # 拆分每行内容
            path = os.path.join(start_path, "wavs", os.path.basename(path)).replace(  # 设置路径
                "\\", "/"  # 替换路径分隔符
            )
            f.writelines(f"{path}|{spk}|{language}|{text}\n")  # 写入处理后的内容
    subprocess.run(  # 运行子进程
        f"python preprocess_text.py "  # 执行预处理脚本
        f"--transcription-path {lbl_path} "  # 设置标签文件路径
        f"--train-path {train_path} "  # 设置训练文件路径
        f"--val-path {val_path} "  # 设置验证文件路径
        f"--config-path {config_path}",  # 设置配置文件路径
        shell=True,  # 在 shell 中执行
    )
    return "标签文件预处理完成"  # 返回信息

# 生成 BERT 特征文件
def bert_gen(data_dir):
    assert data_dir != "", "数据集名称不能为空"  # 断言数据集名称不为空
    _, _, _, _, config_path = get_path(data_dir)  # 获取路径信息
    subprocess.run(  # 运行子进程
        f"python bert_gen.py " f"--config {config_path}",  # 执行生成 BERT 特征文件脚本
        shell=True,  # 在 shell 中执行
    )
    return "BERT 特征文件生成完成"  # 返回信息

# 创建 Gradio 应用
if __name__ == "__main__":
    with gr.Blocks() as app:  # 创建 Gradio 应用
        # ...（略）
    webbrowser.open("http://127.0.0.1:7860")  # 打开浏览器
    app.launch(share=False, server_port=7860)  # 启动应用
```

# `D:\src\Bert-VITS2\for_deploy\infer.py`

```python
"""
版本管理、兼容推理及模型加载实现。
版本说明：
    1. 版本号与github的release版本号对应，使用哪个release版本训练的模型即对应其版本号
    2. 请在模型的config.json中显示声明版本号，添加一个字段"version" : "你的版本号"
特殊版本说明：
    1.1.1-fix： 1.1.1版本训练的模型，但是在推理时使用dev的日语修复
    2.2：当前版本
"""
import torch  # 导入torch库
import commons  # 导入commons库
from text import cleaned_text_to_sequence  # 从text库中导入cleaned_text_to_sequence函数
from text.cleaner import clean_text  # 从text.cleaner库中导入clean_text函数
import utils  # 导入utils库
import numpy as np  # 导入numpy库

from models import SynthesizerTrn  # 从models库中导入SynthesizerTrn类
from text.symbols import symbols  # 从text.symbols库中导入symbols变量

from oldVersion.V210.models import SynthesizerTrn as V210SynthesizerTrn  # 从oldVersion.V210.models库中导入SynthesizerTrn类并重命名为V210SynthesizerTrn
from oldVersion.V210.text import symbols as V210symbols  # 从oldVersion.V210.text库中导入symbols变量并重命名为V210symbols
from oldVersion.V200.models import SynthesizerTrn as V200SynthesizerTrn  # 从oldVersion.V200.models库中导入SynthesizerTrn类并重命名为V200SynthesizerTrn
from oldVersion.V200.text import symbols as V200symbols  # 从oldVersion.V200.text库中导入symbols变量并重命名为V200symbols
from oldVersion.V111.models import SynthesizerTrn as V111SynthesizerTrn  # 从oldVersion.V111.models库中导入SynthesizerTrn类并重命名为V111SynthesizerTrn
from oldVersion.V111.text import symbols as V111symbols  # 从oldVersion.V111.text库中导入symbols变量并重命名为V111symbols
from oldVersion.V110.models import SynthesizerTrn as V110SynthesizerTrn  # 从oldVersion.V110.models库中导入SynthesizerTrn类并重命名为V110SynthesizerTrn
from oldVersion.V110.text import symbols as V110symbols  # 从oldVersion.V110.text库中导入symbols变量并重命名为V110symbols
from oldVersion.V101.models import SynthesizerTrn as V101SynthesizerTrn  # 从oldVersion.V101.models库中导入SynthesizerTrn类并重命名为V101SynthesizerTrn
from oldVersion.V101.text import symbols as V101symbols  # 从oldVersion.V101.text库中导入symbols变量并重命名为V101symbols

from oldVersion import V111, V110, V101, V200, V210  # 从oldVersion库中导入V111, V110, V101, V200, V210模块

# 当前版本信息
latest_version = "2.2"  # 定义变量latest_version为"2.2"

# 版本兼容
SynthesizerTrnMap = {  # 定义字典SynthesizerTrnMap
    "2.1": V210SynthesizerTrn,  # 键为"2.1"，值为V210SynthesizerTrn
    "2.0.2-fix": V200SynthesizerTrn,  # 键为"2.0.2-fix"，值为V200SynthesizerTrn
    "2.0.1": V200SynthesizerTrn,  # 键为"2.0.1"，值为V200SynthesizerTrn
    "2.0": V200SynthesizerTrn,  # 键为"2.0"，值为V200SynthesizerTrn
    "1.1.1-fix": V111SynthesizerTrn,  # 键为"1.1.1-fix"，值为V111SynthesizerTrn
    "1.1.1": V111SynthesizerTrn,  # 键为"1.1.1"，值为V111SynthesizerTrn
    "1.1": V110SynthesizerTrn,  # 键为"1.1"，值为V110SynthesizerTrn
    "1.1.0": V110SynthesizerTrn,  # 键为"1.1.0"，值为V110SynthesizerTrn
    "1.0.1": V101SynthesizerTrn,  # 键为"1.0.1"，值为V101SynthesizerTrn
    "1.0": V101SynthesizerTrn,  # 键为"1.0"，值为V101SynthesizerTrn
    "1.0.0": V101SynthesizerTrn,  # 键为"1.0.0"，值为V101SynthesizerTrn
}

symbolsMap = {  # 定义字典symbolsMap
    "2.1": V210symbols,  # 键为"2.1"，值为V210symbols
    "2.0.2-fix": V200symbols,  # 键为"2.0.2-fix"，值为V200symbols
    "2.0.1": V200symbols,  # 键为"2.0.1"，值为V200symbols
    "2.0": V200symbols,  # 键为"2.0"，值为V200symbols
    "1.1.1-fix": V111symbols,  # 键为"1.1.1-fix"，值为V111symbols
    "1.1.1": V111symbols,  # 键为"1.1.1"，值为V111symbols
    "1.1": V110symbols,  # 键为"1.1"，值为V110symbols
    "1.1.0": V110symbols,  # 键为"1.1.0"，值为V110symbols
    "1.0.1": V101symbols,  # 键为"1.0.1"，值为V101symbols
    "1.0": V101symbols,  # 键为"1.0"，值为V101symbols
    "1.0.0": V101symbols,  # 键为"1.0.0"，值为V101symbols
}


def get_net_g(model_path: str, version: str, device: str, hps):  # 定义函数get_net_g，参数为model_path、version、device、hps
    if version != latest_version:  # 如果version不等于latest_version
        net_g = SynthesizerTrnMap[version](  # net_g等于SynthesizerTrnMap[version]
            len(symbolsMap[version]),  # 调用len函数，参数为symbolsMap[version]
            hps.data.filter_length // 2 + 1,  # hps.data.filter_length整除2加1
            hps.train.segment_size // hps.data.hop_length,  # hps.train.segment_size整除hps.data.hop_length
            n_speakers=hps.data.n_speakers,  # n_speakers等于hps.data.n_speakers
            **hps.model,  # hps.model的所有参数
        ).to(device)  # 调用to方法，参数为device
    else:  # 否则
        # 当前版本模型 net_g
        net_g = SynthesizerTrn(  # net_g等于SynthesizerTrn
            len(symbols),  # 调用len函数，参数为symbols
            hps.data.filter_length // 2 + 1,  # hps.data.filter_length整除2加1
            hps.train.segment_size // hps.data.hop_length,  # hps.train.segment_size整除hps.data.hop_length
            n_speakers=hps.data.n_speakers,  # n_speakers等于hps.data.n_speakers
            **hps.model,  # hps.model的所有参数
        ).to(device)  # 调用to方法，参数为device
    _ = net_g.eval()  # 调用eval方法
    _ = utils.load_checkpoint(model_path, net_g, None, skip_optimizer=True)  # 调用utils.load_checkpoint函数
    return net_g  # 返回net_g


def get_text(text, language_str, bert, hps, device):  # 定义函数get_text，参数为text、language_str、bert、hps、device
    # 在此处实现当前版本的get_text
    norm_text, phone, tone, word2ph = clean_text(text, language_str)  # 调用clean_text函数
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)  # 调用cleaned_text_to_sequence函数

    if hps.data.add_blank:  # 如果hps.data.add_blank为真
        phone = commons.intersperse(phone, 0)  # 调用commons.intersperse函数
        tone = commons.intersperse(tone, 0)  # 调用commons.intersperse函数
        language = commons.intersperse(language, 0)  # 调用commons.intersperse函数
        for i in range(len(word2ph)):  # 遍历range(len(word2ph))
            word2ph[i] = word2ph[i] * 2  # word2ph[i]乘以2
        word2ph[0] += 1  # word2ph[0]加1
    # bert_ori = get_bert(norm_text, word2ph, language_str, device)
    bert_ori = bert[language_str].get_bert_feature(norm_text, word2ph, device)  # bert_ori等于bert[language_str].get_bert_feature(norm_text, word2ph, device)
    del word2ph  # 删除word2ph
    assert bert_ori.shape[-1] == len(phone), phone  # 断言bert_ori.shape[-1]等于len(phone)，并输出phone

    if language_str == "ZH":  # 如果language_str等于"ZH"
        bert = bert_ori  # bert等于bert_ori
        ja_bert = torch.randn(1024, len(phone))  # ja_bert等于torch.randn(1024, len(phone))
        en_bert = torch.randn(1024, len(phone))  # en_bert等于torch.randn(1024, len(phone))
    elif language_str == "JP":  # 如果language_str等于"JP"
        bert = torch.randn(1024, len(phone))  # bert等于torch.randn(1024, len(phone))
        ja_bert = bert_ori  # ja_bert等于bert_ori
        en_bert = torch.randn(1024, len(phone))  # en_bert等于torch.randn(1024, len(phone))
    elif language_str == "EN":  # 如果language_str等于"EN"
        bert = torch.randn(1024, len(phone))  # bert等于torch.randn(1024, len(phone))
        ja_bert = torch.randn(1024, len(phone))  # ja_bert等于torch.randn(1024, len(phone))
        en_bert = bert_ori  # en_bert等于bert_ori
    else:  # 否则
        raise ValueError("language_str should be ZH, JP or EN")  # 抛出ValueError异常，输出"language_str should be ZH, JP or EN"

    assert bert.shape[-1] == len(  # 断言bert.shape[-1]等于len
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"  # 输出f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)  # phone等于torch.LongTensor(phone)
    tone = torch.LongTensor(tone)  # tone等于torch.LongTensor(tone)
    language = torch.LongTensor(language)  # language等于torch.LongTensor(language)
    return bert, ja_bert, en_bert, phone, tone, language  # 返回bert, ja_bert, en_bert, phone, tone, language


def infer(  # 定义函数infer
    text,  # 参数text
    emotion,  # 参数emotion
    sdp_ratio,  # 参数sdp_ratio
    noise_scale,  # 参数noise_scale
    noise_scale_w,  # 参数noise_scale_w
    length_scale,  # 参数length_scale
    sid,  # 参数sid
    language,  # 参数language
    hps,  # 参数hps
    net_g,  # 参数net_g
    device,  # 参数device
    bert=None,  # 参数bert默认值为None
    clap=None,  # 参数clap默认值为None
    reference_audio=None,  # 参数reference_audio默认值为None
    skip_start=False,  # 参数skip_start默认值为False
    skip_end=False,  # 参数skip_end默认值为False
):  # 以下为函数体
    # 2.2版本参数位置变了
    # 2.1 参数新增 emotion reference_audio skip_start skip_end
    inferMap_V3 = {  # 定义字典inferMap_V3
        "2.1": V210.infer,  # 键为"2.1"，值为V210.infer
    }
    # 支持中日英三语版本
    inferMap_V2 = {  # 定义字典inferMap_V2
        "2.0.2-fix": V200.infer,  # 键为"2.0.2-fix"，值为V200.infer
        "2.0.1": V200.infer,  # 键为"2.0.1"，值为V200.infer
        "2.0": V200.infer,  # 键为"2.0"，值为V200.infer
        "1.1.1-fix": V111.infer_fix,  # 键为"1.1.1-fix"，值为V111.infer_fix
        "1.1.1": V111.infer,  # 键为"1.1.1"，值为V111.infer
        "1.1": V110.infer,  # 键为"1.1"，值为V110.infer
        "1.1.0": V110.infer,  # 键为"1.1.0"，值为V110.infer
    }
    # 仅支持中文版本
    # 在测试中，并未发现两个版本的模型不能互相通用
    inferMap_V1 = {  # 定义字典inferMap_V1
        "1.0.1": V101.infer,  # 键为"1.0.1"，值为V101.infer
        "1.0": V101.infer,  # 键为"1.0"，值为V101.infer
        "1.0.0": V101.infer,  # 键为"1.0.0"，值为V101.infer
    }
    version = hps.version if hasattr(hps, "version") else latest_version  # version等于hps.version，如果hps有"version"属性，否则等于latest_version
    # 非当前版本，根据版本号选择合适的infer
    if version != latest_version:  # 如果version不等于latest_version
        if version in inferMap_V3.keys():  # 如果version在inferMap_V3的键中
            return inferMap_V3[version](  # 返回inferMap_V3[version]
                text,  # 参数text
                sdp_ratio,  # 参数sdp_ratio
                noise_scale,  # 参数noise_scale
                noise_scale_w,  # 参数noise_scale_w
                length_scale,  # 参数length_scale
                sid,  # 参数sid
                language,  # 参数language
                hps,  # 参数hps
                net_g,  # 参数net_g
                device,  # 参数device
                reference_audio,  # 参数reference_audio
                emotion,  # 参数emotion
                skip_start,  # 参数skip_start
                skip_end,  # 参数skip_end
            )
        if version in inferMap_V2.keys():  # 如果version在inferMap_V2的键中
            return inferMap_V2[version](  # 返回inferMap_V2[version]
                text,  # 参数text
                sdp_ratio,  # 参数sdp_ratio
                noise_scale,  # 参数noise_scale
                noise_scale_w,  # 参数noise_scale_w
                length_scale,  # 参数length_scale
                sid,  # 参数sid
                language,  # 参数language
                hps,  # 参数hps
                net_g,  # 参数net_g
                device,  # 参数device
            )
        if version in inferMap_V1.keys():  # 如果version在inferMap_V1的键中
            return inferMap_V1[version](  # 返回inferMap_V1[version]
                text,  # 参数text
                sdp_ratio,  # 参数sdp_ratio
                noise_scale,  # 参数noise_scale
                noise_scale_w,  # 参数noise_scale_w
                length_scale,  # 参数length_scale
                sid,  #
```

# `D:\src\Bert-VITS2\for_deploy\infer_utils.py`

```python
import sys  # 导入sys模块

import torch  # 导入torch模块
from transformers import (  # 从transformers模块中导入以下类
    AutoModelForMaskedLM,  # 导入AutoModelForMaskedLM类
    AutoTokenizer,  # 导入AutoTokenizer类
    DebertaV2Model,  # 导入DebertaV2Model类
    DebertaV2Tokenizer,  # 导入DebertaV2Tokenizer类
    ClapModel,  # 导入ClapModel类
    ClapProcessor,  # 导入ClapProcessor类
)

from config import config  # 从config模块中导入config类
from text.japanese import text2sep_kata  # 从text.japanese模块中导入text2sep_kata函数


class BertFeature:  # 定义BertFeature类
    def __init__(self, model_path, language="ZH"):  # 定义初始化方法
        self.model_path = model_path  # 设置model_path属性
        self.language = language  # 设置language属性
        self.tokenizer = None  # 初始化tokenizer属性为None
        self.model = None  # 初始化model属性为None
        self.device = None  # 初始化device属性为None

        self._prepare()  # 调用_prepare方法

    def _get_device(self, device=config.bert_gen_config.device):  # 定义_get_device方法
        if (  # 判断条件
            sys.platform == "darwin"  # 判断条件
            and torch.backends.mps.is_available()  # 判断条件
            and device == "cpu"  # 判断条件
        ):  # 条件成立
            device = "mps"  # 设置device为"mps"
        if not device:  # 判断条件
            device = "cuda"  # 设置device为"cuda"
        return device  # 返回device

    def _prepare(self):  # 定义_prepare方法
        self.device = self._get_device()  # 调用_get_device方法并设置结果为device属性

        if self.language == "EN":  # 判断条件
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_path)  # 设置tokenizer属性为DebertaV2Tokenizer类的实例
            self.model = DebertaV2Model.from_pretrained(self.model_path).to(self.device)  # 设置model属性为DebertaV2Model类的实例
        else:  # 条件不成立
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)  # 设置tokenizer属性为AutoTokenizer类的实例
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_path).to(  # 设置model属性为AutoModelForMaskedLM类的实例
                self.device  # 设置device属性
            )
        self.model.eval()  # 调用model的eval方法

    def get_bert_feature(self, text, word2ph):  # 定义get_bert_feature方法
        if self.language == "JP":  # 判断条件
            text = "".join(text2sep_kata(text)[0])  # 设置text为text2sep_kata函数的返回值
        with torch.no_grad():  # 使用torch.no_grad上下文管理器
            inputs = self.tokenizer(text, return_tensors="pt")  # 设置inputs为tokenizer处理后的结果
            for i in inputs:  # 遍历inputs
                inputs[i] = inputs[i].to(self.device)  # 设置inputs[i]为在device上的结果
            res = self.model(**inputs, output_hidden_states=True)  # 设置res为model处理后的结果
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 设置res为torch.cat处理后的结果

        word2phone = word2ph  # 设置word2phone为word2ph
        phone_level_feature = []  # 初始化phone_level_feature为列表
        for i in range(len(word2phone)):  # 遍历word2phone
            repeat_feature = res[i].repeat(word2phone[i], 1)  # 设置repeat_feature为res[i]的重复结果
            phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature中

        phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 设置phone_level_feature为torch.cat处理后的结果

        return phone_level_feature.T  # 返回phone_level_feature的转置


class ClapFeature:  # 定义ClapFeature类
    def __init__(self, model_path):  # 定义初始化方法
        self.model_path = model_path  # 设置model_path属性
        self.processor = None  # 初始化processor属性为None
        self.model = None  # 初始化model属性为None
        self.device = None  # 初始化device属性为None

        self._prepare()  # 调用_prepare方法

    def _get_device(self, device=config.bert_gen_config.device):  # 定义_get_device方法
        if (  # 判断条件
            sys.platform == "darwin"  # 判断条件
            and torch.backends.mps.is_available()  # 判断条件
            and device == "cpu"  # 判断条件
        ):  # 条件成立
            device = "mps"  # 设置device为"mps"
        if not device:  # 判断条件
            device = "cuda"  # 设置device为"cuda"
        return device  # 返回device

    def _prepare(self):  # 定义_prepare方法
        self.device = self._get_device()  # 调用_get_device方法并设置结果为device属性

        self.processor = ClapProcessor.from_pretrained(self.model_path)  # 设置processor属性为ClapProcessor类的实例
        self.model = ClapModel.from_pretrained(self.model_path).to(self.device)  # 设置model属性为ClapModel类的实例
        self.model.eval()  # 调用model的eval方法

    def get_clap_audio_feature(self, audio_data):  # 定义get_clap_audio_feature方法
        with torch.no_grad():  # 使用torch.no_grad上下文管理器
            inputs = self.processor(  # 设置inputs为processor处理后的结果
                audios=audio_data, return_tensors="pt", sampling_rate=48000  # 设置audios、return_tensors和sampling_rate
            ).to(self.device)  # 设置inputs为在device上的结果
            emb = self.model.get_audio_features(**inputs)  # 设置emb为model.get_audio_features处理后的结果
        return emb.T  # 返回emb的转置

    def get_clap_text_feature(self, text):  # 定义get_clap_text_feature方法
        with torch.no_grad():  # 使用torch.no_grad上下文管理器
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)  # 设置inputs为processor处理后的结果
            emb = self.model.get_text_features(**inputs)  # 设置emb为model.get_text_features处理后的结果
        return emb.T  # 返回emb的转置
```

# `D:\src\Bert-VITS2\for_deploy\webui.py`

```python
# flake8: noqa: E402
import os  # 导入os模块
import logging  # 导入logging模块
import re_matching  # 导入re_matching模块
from tools.sentence import split_by_language  # 从tools.sentence模块导入split_by_language函数

logging.getLogger("numba").setLevel(logging.WARNING)  # 设置numba的日志级别为WARNING
logging.getLogger("markdown_it").setLevel(logging.WARNING)  # 设置markdown_it的日志级别为WARNING
logging.getLogger("urllib3").setLevel(logging.WARNING)  # 设置urllib3的日志级别为WARNING
logging.getLogger("matplotlib").setLevel(logging.WARNING)  # 设置matplotlib的日志级别为WARNING

logging.basicConfig(  # 配置logging的基本信息
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)  # 获取当前模块的logger

import torch  # 导入torch模块
import utils  # 导入utils模块
from infer import infer, latest_version, get_net_g, infer_multilang  # 从infer模块导入infer, latest_version, get_net_g, infer_multilang函数
import gradio as gr  # 导入gradio模块并重命名为gr
import webbrowser  # 导入webbrowser模块
import numpy as np  # 导入numpy模块并重命名为np
from config import config  # 从config模块导入config类
from tools.translate import translate  # 从tools.translate模块导入translate函数
import librosa  # 导入librosa模块
from infer_utils import BertFeature, ClapFeature  # 从infer_utils模块导入BertFeature, ClapFeature类

net_g = None  # 初始化net_g为None

device = config.webui_config.device  # 从config模块的webui_config属性获取device
if device == "mps":  # 如果device为"mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 设置环境变量PYTORCH_ENABLE_MPS_FALLBACK为"1"

os.environ["OMP_NUM_THREADS"] = "1"  # 设置环境变量OMP_NUM_THREADS为"1"
os.environ["MKL_NUM_THREADS"] = "1"  # 设置环境变量MKL_NUM_THREADS为"1"

bert_feature_map = {  # 定义bert_feature_map字典
    "ZH": BertFeature(  # 键为"ZH"，值为BertFeature对象
        "./bert/chinese-roberta-wwm-ext-large",  # 参数1
        language="ZH",  # 参数2
    ),
    "JP": BertFeature(  # 键为"JP"，值为BertFeature对象
        "./bert/deberta-v2-large-japanese-char-wwm",  # 参数1
        language="JP",  # 参数2
    ),
    "EN": BertFeature(  # 键为"EN"，值为BertFeature对象
        "./bert/deberta-v3-large",  # 参数1
        language="EN",  # 参数2
    ),
}

clap_feature = ClapFeature("./emotional/clap-htsat-fused")  # 初始化clap_feature为ClapFeature对象

# 定义generate_audio函数
def generate_audio(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    reference_audio,
    emotion,
    skip_start=False,
    skip_end=False,
):
    audio_list = []  # 初始化audio_list为空列表
    with torch.no_grad():  # 使用torch的no_grad上下文管理器
        for idx, piece in enumerate(slices):  # 遍历slices
            skip_start = (idx != 0) and skip_start  # 更新skip_start
            skip_end = (idx != len(slices) - 1) and skip_end  # 更新skip_end
            audio = infer(  # 调用infer函数
                piece,  # 参数1
                reference_audio=reference_audio,  # 参数2
                emotion=emotion,  # 参数3
                sdp_ratio=sdp_ratio,  # 参数4
                noise_scale=noise_scale,  # 参数5
                noise_scale_w=noise_scale_w,  # 参数6
                length_scale=length_scale,  # 参数7
                sid=speaker,  # 参数8
                language=language,  # 参数9
                hps=hps,  # 参数10
                net_g=net_g,  # 参数11
                device=device,  # 参数12
                skip_start=skip_start,  # 参数13
                skip_end=skip_end,  # 参数14
                bert=bert_feature_map,  # 参数15
                clap=clap_feature,  # 参数16
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)  # 将音频转换为16位wav格式
            audio_list.append(audio16bit)  # 将音频添加到audio_list中
    return audio_list  # 返回audio_list

# 定义generate_audio_multilang函数
def generate_audio_multilang(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    reference_audio,
    emotion,
    skip_start=False,
    skip_end=False,
):
    audio_list = []  # 初始化audio_list为空列表
    with torch.no_grad():  # 使用torch的no_grad上下文管理器
        for idx, piece in enumerate(slices):  # 遍历slices
            skip_start = (idx != 0) and skip_start  # 更新skip_start
            skip_end = (idx != len(slices) - 1) and skip_end  # 更新skip_end
            audio = infer_multilang(  # 调用infer_multilang函数
                piece,  # 参数1
                reference_audio=reference_audio,  # 参数2
                emotion=emotion,  # 参数3
                sdp_ratio=sdp_ratio,  # 参数4
                noise_scale=noise_scale,  # 参数5
                noise_scale_w=noise_scale_w,  # 参数6
                length_scale=length_scale,  # 参数7
                sid=speaker,  # 参数8
                language=language[idx],  # 参数9
                hps=hps,  # 参数10
                net_g=net_g,  # 参数11
                device=device,  # 参数12
                skip_start=skip_start,  # 参数13
                skip_end=skip_end,  # 参数14
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)  # 将音频转换为16位wav格式
            audio_list.append(audio16bit)  # 将音频添加到audio_list中
    return audio_list  # 返回audio_list

# 定义tts_split函数
def tts_split(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    cut_by_sent,
    interval_between_para,
    interval_between_sent,
    reference_audio,
    emotion,
):
    # 函数体略（未添加注释）

# 定义tts_fn函数
def tts_fn(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    reference_audio,
    emotion,
    prompt_mode,
):
    # 函数体略（未添加注释）

# 定义load_audio函数
def load_audio(path):
    audio, sr = librosa.load(path, 48000)  # 调用librosa.load函数
    return sr, audio  # 返回sr和audio

# 定义gr_util函数
def gr_util(item):
    # 函数体略（未添加注释）

if __name__ == "__main__":  # 如果当前模块是主模块
    if config.webui_config.debug:  # 如果config模块的webui_config属性的debug为True
        logger.info("Enable DEBUG-LEVEL log")  # 记录INFO级别的日志
        logging.basicConfig(level=logging.DEBUG)  # 配置logging的基本信息为DEBUG级别
    hps = utils.get_hparams_from_file(config.webui_config.config_path)  # 从config模块的webui_config属性获取config_path，并调用utils.get_hparams_from_file函数
    version = hps.version if hasattr(hps, "version") else latest_version  # 获取hps的version属性或者latest_version
    net_g = get_net_g(  # 调用get_net_g函数
        model_path=config.webui_config.model,  # 参数1
        version=version,  # 参数2
        device=device,  # 参数3
        hps=hps,  # 参数4
    )
    speaker_ids = hps.data.spk2id  # 从hps的data属性获取spk2id
    speakers = list(speaker_ids.keys())  # 获取speaker_ids的键并转换为列表
    languages = ["ZH", "JP", "EN", "mix", "auto"]  # 定义languages列表
    with gr.Blocks() as app:  # 使用gr.Blocks上下文管理器并重命名为app
        # 函数体略（未添加注释）
    print("推理页面已开启!")  # 打印提示信息
    webbrowser.open(f"http://127.0.0.1:{config.webui_config.port}")  # 使用webbrowser打开指定网址
    app.launch(share=config.webui_config.share, server_port=config.webui_config.port)  # 调用app的launch方法
```