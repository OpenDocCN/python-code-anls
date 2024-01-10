# `so-vits-svc\preprocess_hubert_f0.py`

```
# 导入必要的库
import argparse  # 用于解析命令行参数
import logging  # 用于记录日志
import os  # 用于处理文件路径
import random  # 用于生成随机数
from concurrent.futures import ProcessPoolExecutor  # 用于并行处理任务
from glob import glob  # 用于匹配文件路径
from random import shuffle  # 用于打乱列表顺序

import librosa  # 用于音频处理
import numpy as np  # 用于数值计算
import torch  # 用于构建神经网络
import torch.multiprocessing as mp  # 用于多进程处理
from loguru import logger  # 用于记录日志
from tqdm import tqdm  # 用于显示进度条

import diffusion.logger.utils as du  # 导入自定义的日志工具
import utils  # 导入自定义的工具函数
from diffusion.vocoder import Vocoder  # 导入自定义的声码器模块
from modules.mel_processing import spectrogram_torch  # 导入自定义的梅尔频谱处理模块

# 设置 numba 和 matplotlib 的日志级别
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# 从配置文件中获取超参数
hps = utils.get_hparams_from_file("configs/config.json")
# 加载扩散模型的配置
dconfig = du.load_config("configs/diffusion.yaml")
# 设置采样率和帧移
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length
# 获取语音编码器模型
speech_encoder = hps["model"]["speech_encoder"]

# 定义处理单个文件的函数
def process_one(filename, hmodel, f0p, device, diff=False, mel_extractor=None):
    # 读取音频文件
    wav, sr = librosa.load(filename, sr=sampling_rate)
    # 将音频数据转换为张量
    audio_norm = torch.FloatTensor(wav)
    audio_norm = audio_norm.unsqueeze(0)
    # 设置软路径
    soft_path = filename + ".soft.pt"
    # 如果软路径不存在
    if not os.path.exists(soft_path):
        # 重采样音频数据
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        # 使用编码器模型对音频数据进行编码，并保存结果
        c = hmodel.encoder(wav16k)
        torch.save(c.cpu(), soft_path)

    # 设置 F0 路径
    f0_path = filename + ".f0.npy"
    # 如果 F0 路径不存在
    if not os.path.exists(f0_path):
        # 获取 F0 预测器，并计算音频的 F0 和声门特征
        f0_predictor = utils.get_f0_predictor(f0p, sampling_rate=sampling_rate, hop_length=hop_length, device=None, threshold=0.05)
        f0, uv = f0_predictor.compute_f0_uv(wav)
        # 保存 F0 和声门特征
        np.save(f0_path, np.asanyarray((f0, uv), dtype=object))

    # 设置梅尔频谱路径
    spec_path = filename.replace(".wav", ".spec.pt")
    # 如果指定路径不存在
    if not os.path.exists(spec_path):
        # 处理音频的频谱图
        # 以下代码不能用 torch.FloatTensor(wav) 替换
        # 因为 load_wav_to_torch 返回一个需要进行归一化的张量

        # 如果音频采样率不等于指定的采样率
        if sr != hps.data.sampling_rate:
            # 抛出数值错误，指出采样率不匹配
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sr, hps.data.sampling_rate
                )
            )

        # 对音频进行归一化处理
        # audio_norm = audio / hps.data.max_wav_value

        # 使用 torch 库生成音频的频谱图
        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        # 去除频谱图的单维度
        spec = torch.squeeze(spec, 0)
        # 将频谱图保存到指定路径
        torch.save(spec, spec_path)

    # 如果 diff 为真或者 hps.model.vol_embedding 为真
    if diff or hps.model.vol_embedding:
        # 生成音频的音量路径
        volume_path = filename + ".vol.npy"
        # 创建音频音量提取器对象
        volume_extractor = utils.Volume_Extractor(hop_length)
        # 如果音量路径不存在
        if not os.path.exists(volume_path):
            # 提取音频的音量
            volume = volume_extractor.extract(audio_norm)
            # 将音量保存为 numpy 数组
            np.save(volume_path, volume.to('cpu').numpy())
    # 如果存在音频数据的差异
    if diff:
        # 生成 MEL 数据的文件路径
        mel_path = filename + ".mel.npy"
        # 如果 MEL 数据文件不存在，并且 MEL 提取器不为空
        if not os.path.exists(mel_path) and mel_extractor is not None:
            # 提取音频的 MEL 数据
            mel_t = mel_extractor.extract(audio_norm.to(device), sampling_rate)
            # 将 MEL 数据转换为 CPU 上的 numpy 数组并保存到文件
            mel = mel_t.squeeze().to('cpu').numpy()
            np.save(mel_path, mel)
        # 生成增强后的 MEL 数据文件路径
        aug_mel_path = filename + ".aug_mel.npy"
        # 生成增强后的音量数据文件路径
        aug_vol_path = filename + ".aug_vol.npy"
        # 计算音频的最大振幅
        max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
        # 计算最大振幅对应的最大频率偏移
        max_shift = min(1, np.log10(1/max_amp))
        # 生成一个随机的对数音量偏移
        log10_vol_shift = random.uniform(-1, max_shift)
        # 生成一个随机的音调偏移
        keyshift = random.uniform(-5, 5)
        # 如果 MEL 提取器不为空
        if mel_extractor is not None:
            # 提取增强后的 MEL 数据
            aug_mel_t = mel_extractor.extract(audio_norm * (10 ** log10_vol_shift), sampling_rate, keyshift = keyshift)
        # 将增强后的 MEL 数据转换为 CPU 上的 numpy 数组
        aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
        # 提取增强后的音量数据
        aug_vol = volume_extractor.extract(audio_norm * (10 ** log10_vol_shift))
        # 如果增强后的 MEL 数据文件不存在
        if not os.path.exists(aug_mel_path):
            # 将增强后的 MEL 数据和音调偏移保存到文件
            np.save(aug_mel_path,np.asanyarray((aug_mel,keyshift),dtype=object))
        # 如果增强后的音量数据文件不存在
        if not os.path.exists(aug_vol_path):
            # 将增强后的音量数据保存到文件
            np.save(aug_vol_path,aug_vol.to('cpu').numpy())
# 处理批量文件的函数，接受文件块、f0p、是否使用差分、mel_extractor和设备参数
def process_batch(file_chunk, f0p, diff=False, mel_extractor=None, device="cpu"):
    # 记录日志，加载内容的语音编码器
    logger.info("Loading speech encoder for content...")
    # 获取当前进程的排名
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    # 如果有可用的 CUDA 设备，根据排名选择设备
    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
    # 记录日志，显示当前进程使用的设备
    logger.info(f"Rank {rank} uses device {device}")
    # 获取语音编码器
    hmodel = utils.get_speech_encoder(speech_encoder, device=device)
    # 记录日志，显示加载了语音编码器
    logger.info(f"Loaded speech encoder for rank {rank}")
    # 遍历文件块中的文件，处理每个文件
    for filename in tqdm(file_chunk, position = rank):
        process_one(filename, hmodel, f0p, device, diff, mel_extractor)

# 并行处理函数，接受文件名列表、进程数、f0p、是否使用差分、mel_extractor和设备参数
def parallel_process(filenames, num_processes, f0p, diff, mel_extractor, device):
    # 使用进程池执行器创建执行器
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        # 根据进程数拆分文件名列表，创建任务列表
        for i in range(num_processes):
            start = int(i * len(filenames) / num_processes)
            end = int((i + 1) * len(filenames) / num_processes)
            file_chunk = filenames[start:end]
            tasks.append(executor.submit(process_batch, file_chunk, f0p, diff, mel_extractor, device=device))
        # 遍历任务列表，等待任务完成
        for task in tqdm(tasks, position = 0):
            task.result()

# 主函数入口
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加设备参数
    parser.add_argument('-d', '--device', type=str, default=None)
    # 添加输入目录参数
    parser.add_argument(
        "--in_dir", type=str, default="dataset/44k", help="path to input dir"
    )
    # 添加是否使用差分参数
    parser.add_argument(
        '--use_diff',action='store_true', help='Whether to use the diffusion model'
    )
    # 添加f0_predictor参数
    parser.add_argument(
        '--f0_predictor', type=str, default="rmvpe", help='Select F0 predictor, can select crepe,pm,dio,harvest,rmvpe,fcpe|default: pm(note: crepe is original F0 using mean filter)'
    )
    # 添加进程数参数
    parser.add_argument(
        '--num_processes', type=int, default=1, help='You are advised to set the number of processes to the same as the number of CPU cores'
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 从参数中获取 f0_predictor
    f0p = args.f0_predictor
    # 从参数中获取 device
    device = args.device
    # 如果 device 为空，则根据是否有 GPU 来选择设备
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 打印 speech_encoder
    print(speech_encoder)
    # 记录所使用的设备信息
    logger.info("Using device: " + str(device))
    # 记录所使用的 SpeechEncoder 信息
    logger.info("Using SpeechEncoder: " + speech_encoder)
    # 记录所使用的 extractor 信息
    logger.info("Using extractor: " + f0p)
    # 记录是否使用 diff 模式
    logger.info("Using diff Mode: " + str(args.use_diff))

    # 如果使用了 diff 模式
    if args.use_diff:
        # 打印提示信息
        print("use_diff")
        # 打印加载 Mel Extractor 的提示信息
        print("Loading Mel Extractor...")
        # 根据配置加载 Mel Extractor
        mel_extractor = Vocoder(dconfig.vocoder.type, dconfig.vocoder.ckpt, device=device)
        # 打印加载完成的提示信息
        print("Loaded Mel Extractor.")
    # 如果没有使用 diff 模式
    else:
        # 将 mel_extractor 设为 None
        mel_extractor = None
    # 获取指定目录下的所有 wav 文件名
    filenames = glob(f"{args.in_dir}/*/*.wav", recursive=True)  # [:10]
    # 打乱文件名顺序
    shuffle(filenames)
    # 设置多进程的启动方式
    mp.set_start_method("spawn", force=True)

    # 获取指定的进程数，如果为 0 则使用 CPU 核心数
    num_processes = args.num_processes
    if num_processes == 0:
        num_processes = os.cpu_count()

    # 并行处理文件名列表中的文件
    parallel_process(filenames, num_processes, f0p, args.use_diff, mel_extractor, device)
```