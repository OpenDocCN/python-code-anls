# `so-vits-svc\resample.py`

```
# 导入必要的库
import argparse  # 用于解析命令行参数
import concurrent.futures  # 用于并发执行任务
import os  # 用于处理文件路径等操作
from concurrent.futures import ProcessPoolExecutor  # 用于创建进程池执行器
from multiprocessing import cpu_count  # 用于获取 CPU 核心数量

import librosa  # 用于音频处理
import numpy as np  # 用于数值计算
from rich.progress import track  # 用于显示进度条
from scipy.io import wavfile  # 用于读写 WAV 文件


# 加载 WAV 文件
def load_wav(wav_path):
    return librosa.load(wav_path, sr=None)


# 对 WAV 文件进行修剪
def trim_wav(wav, top_db=40):
    return librosa.effects.trim(wav, top_db=top_db)


# 对 WAV 文件进行峰值归一化
def normalize_peak(wav, threshold=1.0):
    peak = np.abs(wav).max()
    if peak > threshold:
        wav = 0.98 * wav / peak
    return wav


# 对 WAV 文件进行重采样
def resample_wav(wav, sr, target_sr):
    return librosa.resample(wav, orig_sr=sr, target_sr=target_sr)


# 将 WAV 文件保存到指定路径
def save_wav_to_path(wav, save_path, sr):
    wavfile.write(
        save_path,
        sr,
        (wav * np.iinfo(np.int16).max).astype(np.int16)
    )


# 处理单个音频文件
def process(item):
    spkdir, wav_name, args = item
    speaker = spkdir.replace("\\", "/").split("/")[-1]

    # 构建 WAV 文件路径
    wav_path = os.path.join(args.in_dir, speaker, wav_name)
    # 如果 WAV 文件存在且是 WAV 格式
    if os.path.exists(wav_path) and '.wav' in wav_path:
        # 创建输出目录
        os.makedirs(os.path.join(args.out_dir2, speaker), exist_ok=True)

        # 加载 WAV 文件
        wav, sr = load_wav(wav_path)
        # 修剪 WAV 文件
        wav, _ = trim_wav(wav)
        # 对 WAV 文件进行峰值归一化
        wav = normalize_peak(wav)
        # 对 WAV 文件进行重采样
        resampled_wav = resample_wav(wav, sr, args.sr2)

        # 如果不跳过响度归一化
        if not args.skip_loudnorm:
            resampled_wav /= np.max(np.abs(resampled_wav))

        # 构建保存路径
        save_path2 = os.path.join(args.out_dir2, speaker, wav_name)
        # 将重采样后的 WAV 文件保存到指定路径
        save_wav_to_path(resampled_wav, save_path2, args.sr2)


# 处理所有说话者的音频文件
"""
def process_all_speakers():
    process_count = 30 if os.cpu_count() > 60 else (os.cpu_count() - 2 if os.cpu_count() > 4 else 1)
"""  # 处理所有说话者的音频文件，根据 CPU 核心数量确定并发处理的数量
    # 使用线程池执行器创建具有指定最大工作线程数的执行器
    with ThreadPoolExecutor(max_workers=process_count) as executor:
        # 遍历说话者列表
        for speaker in speakers:
            # 拼接输入目录和说话者目录，得到说话者目录的完整路径
            spk_dir = os.path.join(args.in_dir, speaker)
            # 如果说话者目录存在
            if os.path.isdir(spk_dir):
                # 打印说话者目录路径
                print(spk_dir)
                # 使用执行器提交任务，处理说话者目录下以.wav结尾的文件
                futures = [executor.submit(process, (spk_dir, i, args)) for i in os.listdir(spk_dir) if i.endswith("wav")]
                # 遍历已完成的任务，并显示进度条
                for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    pass
# 多进程处理

# 定义处理所有发言者的函数
def process_all_speakers():
    # 根据 CPU 核心数量确定进程数
    process_count = 30 if os.cpu_count() > 60 else (os.cpu_count() - 2 if os.cpu_count() > 4 else 1)
    # 使用进程池执行器创建执行器，最大工作进程数为 process_count
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        # 遍历所有发言者
        for speaker in speakers:
            # 获取发言者目录
            spk_dir = os.path.join(args.in_dir, speaker)
            # 如果目录存在
            if os.path.isdir(spk_dir):
                # 打印目录
                print(spk_dir)
                # 提交任务到执行器，处理目录下所有以.wav结尾的文件
                futures = [executor.submit(process, (spk_dir, i, args)) for i in os.listdir(spk_dir) if i.endswith("wav")]
                # 迭代已完成的任务，显示进度条
                for _ in track(concurrent.futures.as_completed(futures), total=len(futures), description="resampling:"):
                    pass

# 如果当前脚本为主程序
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument("--sr2", type=int, default=44100, help="sampling rate")
    parser.add_argument("--in_dir", type=str, default="./dataset_raw", help="path to source dir")
    parser.add_argument("--out_dir2", type=str, default="./dataset/44k", help="path to target dir")
    parser.add_argument("--skip_loudnorm", action="store_true", help="Skip loudness matching if you have done it")
    # 解析命令行参数
    args = parser.parse_args()

    # 打印 CPU 核心数量
    print(f"CPU count: {cpu_count()}")
    # 获取所有发言者目录
    speakers = os.listdir(args.in_dir)
    # 处理所有发言者
    process_all_speakers()
```