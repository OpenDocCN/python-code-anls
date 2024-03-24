# `.\lucidrains\big-sleep\big_sleep\cli.py`

```
# 导入 fire 模块，用于命令行接口
import fire
# 导入 random 模块并重命名为 rnd
import random as rnd
# 从 big_sleep 模块中导入 Imagine 类和 version 变量
from big_sleep import Imagine, version
# 从 pathlib 模块中导入 Path 类
from pathlib import Path
# 从当前目录下的 version 模块中导入 __version__ 变量
from .version import __version__;

# 定义 train 函数，接受多个参数
def train(
    text=None,
    img=None,
    text_min="",
    lr = .07,
    image_size = 512,
    gradient_accumulate_every = 1,
    epochs = 20,
    iterations = 1050,
    save_every = 50,
    overwrite = False,
    save_progress = False,
    save_date_time = False,
    bilinear = False,
    open_folder = True,
    seed = 0,
    append_seed = False,
    random = False,
    torch_deterministic = False,
    max_classes = None,
    class_temperature = 2.,
    save_best = False,
    experimental_resample = False,
    ema_decay = 0.5,
    num_cutouts = 128,
    center_bias = False,
    larger_model = False
):
    # 打印版本信息
    print(f'Starting up... v{__version__}')

    # 如果 random 为 True，则生成一个随机种子
    if random:
        seed = rnd.randint(0, 1e6)

    # 创建 Imagine 对象，传入各种参数
    imagine = Imagine(
        text=text,
        img=img,
        text_min=text_min,
        lr = lr,
        image_size = image_size,
        gradient_accumulate_every = gradient_accumulate_every,
        epochs = epochs,
        iterations = iterations,
        save_every = save_every,
        save_progress = save_progress,
        bilinear = bilinear,
        seed = seed,
        append_seed = append_seed,
        torch_deterministic = torch_deterministic,
        open_folder = open_folder,
        max_classes = max_classes,
        class_temperature = class_temperature,
        save_date_time = save_date_time,
        save_best = save_best,
        experimental_resample = experimental_resample,
        ema_decay = ema_decay,
        num_cutouts = num_cutouts,
        center_bias = center_bias,
        larger_clip = larger_model
    )

    # 如果不覆盖且文件已存在，则询问是否覆盖
    if not overwrite and imagine.filename.exists():
        answer = input('Imagined image already exists, do you want to overwrite? (y/n) ').lower()
        if answer not in ('yes', 'y'):
            exit()

    # 调用 Imagine 对象的方法开始训练
    imagine()

# 定义主函数
def main():
    # 使用 fire 模块创建命令行接口，传入 train 函数
    fire.Fire(train)
```