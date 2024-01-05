# `d:/src/tocomm/Bert-VITS2\resample.py`

```
import os  # 导入os模块，用于操作文件和目录
import argparse  # 导入argparse模块，用于解析命令行参数
import librosa  # 导入librosa模块，用于音频处理
from multiprocessing import Pool, cpu_count  # 导入Pool和cpu_count类，用于多进程处理

import soundfile  # 导入soundfile模块，用于音频文件的读写
from tqdm import tqdm  # 导入tqdm模块，用于显示进度条

from config import config  # 导入config模块，用于读取配置信息


def process(item):
    spkdir, wav_name, args = item  # 解析item元组
    wav_path = os.path.join(args.in_dir, spkdir, wav_name)  # 拼接音频文件路径
    if os.path.exists(wav_path) and wav_path.lower().endswith(".wav"):  # 判断音频文件是否存在且为.wav格式
        wav, sr = librosa.load(wav_path, sr=args.sr)  # 使用librosa加载音频文件
        soundfile.write(os.path.join(args.out_dir, spkdir, wav_name), wav, sr)  # 将处理后的音频文件写入指定路径


if __name__ == "__main__":
    # 主程序入口
    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    parser.add_argument(
        "--sr",
        type=int,
        default=config.resample_config.sampling_rate,
        help="sampling rate",
    )
    # 添加一个名为"sr"的参数，类型为整数，如果没有提供该参数，则使用config.resample_config.sampling_rate的默认值，帮助信息为"sampling rate"

    parser.add_argument(
        "--in_dir",
        type=str,
        default=config.resample_config.in_dir,
        help="path to source dir",
    )
    # 添加一个名为"in_dir"的参数，类型为字符串，如果没有提供该参数，则使用config.resample_config.in_dir的默认值，帮助信息为"path to source dir"

    parser.add_argument(
        "--out_dir",
        type=str,
        default=config.resample_config.out_dir,
        help="path to target dir",
    )
    # 添加一个名为"out_dir"的参数，类型为字符串，如果没有提供该参数，则使用config.resample_config.out_dir的默认值，帮助信息为"path to target dir"

    parser.add_argument(
```

这段代码使用了argparse模块来解析命令行参数。首先创建了一个参数解析器对象parser，然后使用add_argument方法添加了四个参数。每个参数都有不同的名称、类型、默认值和帮助信息。这些参数可以在命令行中使用，并且可以通过parser.parse_args()方法来获取用户提供的参数值。
        "--processes",
        type=int,
        default=0,
        help="cpu_processes",
    )
```
- `--processes`：命令行参数，用于指定进程数量。
- `type=int`：指定参数的类型为整数。
- `default=0`：如果没有指定参数，则默认值为0。
- `help="cpu_processes"`：在命令行中显示的帮助信息。

```
    args, _ = parser.parse_known_args()
```
- `args`：解析命令行参数后的命名空间对象。
- `_`：未使用的变量，用于接收解析命令行参数时返回的其他参数。

```
    if args.processes == 0:
        processes = cpu_count() - 2 if cpu_count() > 4 else 1
    else:
        processes = args.processes
```
- 如果命令行参数 `--processes` 的值为0，则将 `processes` 设置为 CPU 核心数量减去2（如果核心数量大于4），否则设置为1。
- 如果命令行参数 `--processes` 的值不为0，则将 `processes` 设置为命令行参数的值。

```
    pool = Pool(processes=processes)
```
- 创建一个进程池对象，用于并行执行任务。
- `processes`：指定进程池中的进程数量。

```
    tasks = []
```
- 创建一个空列表，用于存储任务。

```
    for dirpath, _, filenames in os.walk(args.in_dir):
```
- 遍历指定目录及其子目录下的所有文件和文件夹。
- `dirpath`：当前遍历到的文件夹路径。
- `_`：未使用的变量，用于接收当前文件夹下的所有子文件夹的名称。
- `filenames`：当前文件夹下的所有文件的名称。

```
        spk_dir = os.path.relpath(dirpath, args.in_dir)
        spk_dir_out = os.path.join(args.out_dir, spk_dir)
        if not os.path.isdir(spk_dir_out):
```
- `spk_dir`：将当前文件夹路径相对于输入目录的路径。
- `spk_dir_out`：将输出目录与 `spk_dir` 拼接得到的路径。
- 如果 `spk_dir_out` 不是一个目录，则执行以下操作。
# 创建输出目录，如果目录已存在则不做任何操作
os.makedirs(spk_dir_out, exist_ok=True)

# 遍历文件名列表
for filename in filenames:
    # 判断文件名是否以.wav结尾
    if filename.lower().endswith(".wav"):
        # 将说话人目录、文件名和其他参数封装成元组，并添加到任务列表中
        twople = (spk_dir, filename, args)
        tasks.append(twople)

# 使用多线程处理任务列表中的任务，并显示进度条
for _ in tqdm(
    pool.imap_unordered(process, tasks),
):
    pass

# 关闭线程池
pool.close()
# 等待所有线程结束
pool.join()

# 打印提示信息
print("音频重采样完毕!")
```