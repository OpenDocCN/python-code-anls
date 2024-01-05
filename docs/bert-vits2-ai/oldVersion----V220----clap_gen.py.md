# `d:/src/tocomm/Bert-VITS2\oldVersion\V220\clap_gen.py`

```
import argparse  # 导入命令行参数解析模块
from multiprocessing import Pool, cpu_count  # 导入多进程模块和CPU核心数统计模块

import torch  # 导入PyTorch深度学习框架
import torch.multiprocessing as mp  # 导入PyTorch的多进程模块
from tqdm import tqdm  # 导入进度条模块

import utils  # 导入自定义的工具模块
from config import config  # 从配置文件中导入配置
from .clap_wrapper import get_clap_audio_feature  # 从clap_wrapper模块中导入get_clap_audio_feature函数
import librosa  # 导入音频处理库
import os  # 导入操作系统模块

os.environ["OMP_NUM_THREADS"] = "1"  # 设置环境变量，限制OpenMP线程数为1
os.environ["MKL_NUM_THREADS"] = "1"  # 设置环境变量，限制MKL线程数为1

def process_line(line):
    device = config.emo_gen_config.device  # 从配置中获取设备信息
    if config.emo_gen_config.use_multi_device:  # 如果配置中指定使用多设备
        rank = mp.current_process()._identity  # 获取当前进程的标识
        rank = rank[0] if len(rank) > 0 else 0  # 如果标识长度大于0，则取第一个值，否则设为0
        if torch.cuda.is_available():  # 检查是否有可用的 CUDA 设备
            gpu_id = rank % torch.cuda.device_count()  # 计算当前进程在 CUDA 设备上的 ID
            device = torch.device(f"cuda:{gpu_id}")  # 设置使用的 CUDA 设备
        else:
            device = torch.device("cpu")  # 如果没有可用的 CUDA 设备，则使用 CPU
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")  # 从输入行中解析出音频路径、语言、文本等信息

    clap_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".emo.npy")  # 根据音频路径生成对应的情感特征文件路径
    if os.path.isfile(clap_path):  # 如果情感特征文件已经存在，则直接返回，不再处理该音频
        return

    audio = librosa.load(wav_path, 48000)[0]  # 使用 librosa 加载音频文件
    # audio = librosa.resample(audio, 44100, 48000)  # 对音频进行重采样（注释掉的代码，可能是可选的处理步骤）

    clap = get_clap_audio_feature(audio, device)  # 调用自定义函数获取音频的情感特征
    torch.save(clap, clap_path)  # 将情感特征保存到文件中
# 如果当前文件被直接执行而不是被导入，则执行以下代码
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加一个配置参数，类型为字符串，默认值为config.emo_gen_config.config_path
    parser.add_argument(
        "-c", "--config", type=str, default=config.emo_gen_config.config_path
    )
    # 添加一个进程数量参数，类型为整数，默认值为config.emo_gen_config.num_processes
    parser.add_argument(
        "--num_processes", type=int, default=config.emo_gen_config.num_processes
    )
    # 解析命令行参数，将结果存储在args中
    args, _ = parser.parse_known_args()
    # 从args中获取配置文件路径
    config_path = args.config
    # 从配置文件中获取超参数
    hps = utils.get_hparams_from_file(config_path)
    # 创建一个空列表用于存储文件内容
    lines = []
    # 打开训练文件，读取内容并添加到lines列表中
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    # 打开验证文件，读取内容并添加到lines列表中
    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    # 如果lines列表不为空
    if len(lines) != 0:
        # 计算可用的进程数量
        num_processes = min(args.num_processes, cpu_count())
        # 创建一个进程池对象，使用最大进程数量为num_processes
        with Pool(processes=num_processes) as pool:
# 使用进度条展示并发处理的进度，调用process_line函数处理lines中的每一项
for _ in tqdm(pool.imap_unordered(process_line, lines), total=len(lines)):
    pass

# 打印生成完毕的提示信息，包括生成的emo.pt文件数量
print(f"clap生成完毕!, 共有{len(lines)}个emo.pt生成!")
```