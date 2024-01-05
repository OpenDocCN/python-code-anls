# `d:/src/tocomm/Bert-VITS2\resample_legacy.py`

```
import os  # 导入os模块，用于操作文件和目录
import argparse  # 导入argparse模块，用于解析命令行参数
import librosa  # 导入librosa模块，用于音频处理
from multiprocessing import Pool, cpu_count  # 导入Pool和cpu_count类，用于多进程处理

import soundfile  # 导入soundfile模块，用于音频文件的读写
from tqdm import tqdm  # 导入tqdm模块，用于显示进度条

from config import config  # 导入config模块，用于读取配置信息


def process(item):
    # 定义process函数，用于处理音频文件
    wav_name, args = item  # 解包item，获取音频文件名和命令行参数
    wav_path = os.path.join(args.in_dir, wav_name)  # 拼接音频文件的完整路径
    if os.path.exists(wav_path) and wav_path.lower().endswith(".wav"):  # 判断音频文件是否存在且为.wav格式
        wav, sr = librosa.load(wav_path, sr=args.sr)  # 使用librosa加载音频文件
        soundfile.write(os.path.join(args.out_dir, wav_name), wav, sr)  # 将处理后的音频文件写入指定目录


if __name__ == "__main__":
    # 程序入口
    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    parser.add_argument(
        "--sr",
        type=int,
        default=config.resample_config.sampling_rate,
        help="sampling rate",
    )
    # 添加一个名为"sr"的参数，类型为整数，如果没有提供该参数，则使用config.resample_config.sampling_rate的默认值，该参数用于设置采样率

    parser.add_argument(
        "--in_dir",
        type=str,
        default=config.resample_config.in_dir,
        help="path to source dir",
    )
    # 添加一个名为"in_dir"的参数，类型为字符串，如果没有提供该参数，则使用config.resample_config.in_dir的默认值，该参数用于设置源目录的路径

    parser.add_argument(
        "--out_dir",
        type=str,
        default=config.resample_config.out_dir,
        help="path to target dir",
    )
    # 添加一个名为"out_dir"的参数，类型为字符串，如果没有提供该参数，则使用config.resample_config.out_dir的默认值，该参数用于设置目标目录的路径

    parser.add_argument(
```

这段代码是使用argparse库创建一个参数解析器对象，并添加了四个参数。这些参数用于设置采样率、源目录路径和目标目录路径。如果没有提供这些参数，则会使用默认值。
        "--processes",  # 定义一个命令行参数，用于指定进程数
        type=int,  # 参数类型为整数
        default=0,  # 默认值为0
        help="cpu_processes",  # 帮助信息，用于显示在命令行中
    )
    args, _ = parser.parse_known_args()  # 解析命令行参数，并将结果赋值给args变量
    # autodl 无卡模式会识别出46个cpu
    if args.processes == 0:  # 如果命令行参数的值为0
        processes = cpu_count() - 2 if cpu_count() > 4 else 1  # 进程数为CPU核心数减2，如果CPU核心数大于4，否则为1
    else:
        processes = args.processes  # 进程数为命令行参数的值
    pool = Pool(processes=processes)  # 创建进程池，进程数为processes

    tasks = []  # 创建一个空列表，用于存储任务

    for dirpath, _, filenames in os.walk(args.in_dir):  # 遍历指定目录下的所有文件和文件夹
        if not os.path.isdir(args.out_dir):  # 如果输出目录不存在
            os.makedirs(args.out_dir, exist_ok=True)  # 创建输出目录
        for filename in filenames:  # 遍历文件名列表
            if filename.lower().endswith(".wav"):  # 如果文件名以.wav结尾
# 创建一个空列表用于存储任务
tasks = []

# 将文件名和参数组成的元组添加到任务列表中
tasks.append((filename, args))

# 使用进度条显示任务的处理进度
for _ in tqdm(
    # 并行处理任务
    pool.imap_unordered(process, tasks),
):
    pass

# 关闭进程池
pool.close()
# 等待所有任务完成
pool.join()

# 打印提示信息
print("音频重采样完毕!")
```

这段代码的作用是将任务添加到任务列表中，使用进度条显示任务的处理进度，然后关闭进程池并等待所有任务完成，最后打印提示信息。
```