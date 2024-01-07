# `Bert-VITS2\resample_legacy.py`

```

# 导入所需的模块
import os
import argparse
import librosa
from multiprocessing import Pool, cpu_count

import soundfile
from tqdm import tqdm

from config import config

# 定义处理函数，用于处理音频文件
def process(item):
    # 获取音频文件名和参数
    wav_name, args = item
    # 构建音频文件路径
    wav_path = os.path.join(args.in_dir, wav_name)
    # 如果音频文件存在且是.wav格式
    if os.path.exists(wav_path) and wav_path.lower().endswith(".wav"):
        # 使用librosa库加载音频文件
        wav, sr = librosa.load(wav_path, sr=args.sr)
        # 将处理后的音频文件写入目标目录
        soundfile.write(os.path.join(args.out_dir, wav_name), wav, sr)

# 主程序入口
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument(
        "--sr",
        type=int,
        default=config.resample_config.sampling_rate,
        help="sampling rate",
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        default=config.resample_config.in_dir,
        help="path to source dir",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=config.resample_config.out_dir,
        help="path to target dir",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=0,
        help="cpu_processes",
    )
    # 解析命令行参数
    args, _ = parser.parse_known_args()
    # 根据参数设置处理进程数
    if args.processes == 0:
        processes = cpu_count() - 2 if cpu_count() > 4 else 1
    else:
        processes = args.processes
    # 创建进程池
    pool = Pool(processes=processes)

    tasks = []

    # 遍历源目录下的所有.wav文件
    for dirpath, _, filenames in os.walk(args.in_dir):
        # 如果目标目录不存在，则创建目标目录
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir, exist_ok=True)
        for filename in filenames:
            # 将.wav文件添加到任务列表中
            if filename.lower().endswith(".wav"):
                tasks.append((filename, args))

    # 使用进度条展示处理进度
    for _ in tqdm(
        pool.imap_unordered(process, tasks),
    ):
        pass

    # 关闭进程池
    pool.close()
    # 等待所有进程结束
    pool.join()

    # 打印处理完成的提示信息
    print("音频重采样完毕!")

```