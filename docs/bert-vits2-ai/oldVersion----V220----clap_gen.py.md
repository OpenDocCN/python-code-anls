# `Bert-VITS2\oldVersion\V220\clap_gen.py`

```

# 导入必要的库
import argparse  # 用于解析命令行参数
from multiprocessing import Pool, cpu_count  # 用于多进程处理
import torch  # PyTorch 深度学习库
import torch.multiprocessing as mp  # 多进程支持
from tqdm import tqdm  # 进度条显示

import utils  # 自定义的工具函数
from config import config  # 导入配置文件
from .clap_wrapper import get_clap_audio_feature  # 导入自定义的音频处理函数
import librosa  # 用于音频处理
import os  # 用于操作系统相关功能

# 设置环境变量，限制线程数
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 定义处理每行数据的函数
def process_line(line):
    # 获取设备信息
    device = config.emo_gen_config.device
    # 如果使用多设备
    if config.emo_gen_config.use_multi_device:
        # 获取当前进程的标识
        rank = mp.current_process()._identity
        rank = rank[0] if len(rank) > 0 else 0
        # 如果有可用的 GPU
        if torch.cuda.is_available():
            gpu_id = rank % torch.cuda.device_count()
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
    # 解析每行数据
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")

    # 生成 CLAP 特征文件路径
    clap_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".emo.npy")
    # 如果 CLAP 特征文件已存在，则直接返回
    if os.path.isfile(clap_path):
        return

    # 加载音频文件
    audio = librosa.load(wav_path, 48000)[0]
    # 获取 CLAP 特征
    clap = get_clap_audio_feature(audio, device)
    # 保存 CLAP 特征文件
    torch.save(clap, clap_path)

# 主函数入口
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=config.emo_gen_config.config_path
    )
    parser.add_argument(
        "--num_processes", type=int, default=config.emo_gen_config.num_processes
    )
    args, _ = parser.parse_known_args()
    config_path = args.config
    hps = utils.get_hparams_from_file(config_path)
    lines = []
    # 读取训练数据文件
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    # 读取验证数据文件
    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    # 如果有数据
    if len(lines) != 0:
        # 获取可用的处理器核心数
        num_processes = min(args.num_processes, cpu_count())
        # 使用进程池并行处理数据
        with Pool(processes=num_processes) as pool:
            for _ in tqdm(pool.imap_unordered(process_line, lines), total=len(lines)):
                pass

    # 打印处理完成信息
    print(f"clap生成完毕!, 共有{len(lines)}个emo.pt生成!")

```