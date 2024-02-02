# `Bert-VITS2\oldVersion\V220\clap_gen.py`

```py
# 导入必要的库
import argparse
from multiprocessing import Pool, cpu_count
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import utils
from config import config
from .clap_wrapper import get_clap_audio_feature
import librosa
import os

# 设置环境变量，限制线程数
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 定义处理每行数据的函数
def process_line(line):
    # 获取设备信息
    device = config.emo_gen_config.device
    # 如果使用多设备，则根据当前进程的排名选择设备
    if config.emo_gen_config.use_multi_device:
        rank = mp.current_process()._identity
        rank = rank[0] if len(rank) > 0 else 0
        if torch.cuda.is_available():
            gpu_id = rank % torch.cuda.device_count()
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
    # 解析输入行的信息
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")
    # 构建对应的 clap 文件路径
    clap_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".emo.npy")
    # 如果 clap 文件已存在，则直接返回
    if os.path.isfile(clap_path):
        return
    # 加载音频文件并提取特征
    audio = librosa.load(wav_path, 48000)[0]
    # audio = librosa.resample(audio, 44100, 48000)
    clap = get_clap_audio_feature(audio, device)
    # 保存提取的特征到文件
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
    # 读取训练文件和验证文件中的内容
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    # 如果有数据，则使用多进程处理
    if len(lines) != 0:
        num_processes = min(args.num_processes, cpu_count())
        with Pool(processes=num_processes) as pool:
            # 使用进程池并行处理数据
            for _ in tqdm(pool.imap_unordered(process_line, lines), total=len(lines)):
                pass
    # 打印生成完毕的提示信息，包括生成的文件数量
    print(f"clap生成完毕!, 共有{len(lines)}个emo.pt生成!")
```