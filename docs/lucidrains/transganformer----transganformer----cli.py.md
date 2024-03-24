# `.\lucidrains\transganformer\transganformer\cli.py`

```
# 导入所需的库
import os
import fire
import random
from retry.api import retry_call
from tqdm import tqdm
from datetime import datetime
from functools import wraps
from transganformer import Trainer, NanException

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np

# 检查值是否存在
def exists(val):
    return val is not None

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 将元素转换为列表
def cast_list(el):
    return el if isinstance(el, list) else [el]

# 生成带时间戳的文件名
def timestamped_filename(prefix = 'generated-'):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# 运行训练过程
def run_training(rank, world_size, model_args, data, load_from, new, num_train_steps, name, seed):
    is_main = rank == 0
    is_ddp = world_size > 1

    if is_ddp:
        set_seed(seed)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        print(f"{rank + 1}/{world_size} process initialized.")

    model_args.update(
        is_ddp = is_ddp,
        rank = rank,
        world_size = world_size
    )

    model = Trainer(**model_args)

    if not new:
        model.load(load_from)
    else:
        model.clear()

    model.set_data_src(data)

    for _ in tqdm(range(num_train_steps - model.steps), initial = model.steps, total = num_train_steps, mininterval=10., desc=f'{name}<{data}>'):
        retry_call(model.train, tries=3, exceptions=NanException)
        if is_main and _ % 50 == 0:
            model.print_log()

    model.save(model.checkpoint_num)

    if is_ddp:
        dist.destroy_process_group()

# 从文件夹中训练模型
def train_from_folder(
    data = './data',
    results_dir = './results',
    models_dir = './models',
    name = 'default',
    new = False,
    load_from = -1,
    image_size = 32,
    fmap_max = 512,
    transparent = False,
    greyscale = False,
    batch_size = 10,
    gradient_accumulate_every = 4,
    num_train_steps = 150000,
    learning_rate = 2e-4,
    save_every = 1000,
    evaluate_every = 1000,
    generate = False,
    generate_types = ['default', 'ema'],
    generate_interpolation = False,
    aug_test = False,
    aug_prob=None,
    aug_types=['cutout', 'translation'],
    dataset_aug_prob=0.,
    interpolation_num_steps = 100,
    save_frames = False,
    num_image_tiles = None,
    num_workers = None,
    multi_gpus = False,
    calculate_fid_every = None,
    calculate_fid_num_images = 12800,
    clear_fid_cache = False,
    seed = 42,
    amp = False,
    show_progress = False,
):
    num_image_tiles = default(num_image_tiles, 4 if image_size > 512 else 8)

    model_args = dict(
        name = name,
        results_dir = results_dir,
        models_dir = models_dir,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        image_size = image_size,
        num_image_tiles = num_image_tiles,
        num_workers = num_workers,
        fmap_max = fmap_max,
        transparent = transparent,
        greyscale = greyscale,
        lr = learning_rate,
        save_every = save_every,
        evaluate_every = evaluate_every,
        aug_prob = aug_prob,
        aug_types = cast_list(aug_types),
        dataset_aug_prob = dataset_aug_prob,
        calculate_fid_every = calculate_fid_every,
        calculate_fid_num_images = calculate_fid_num_images,
        clear_fid_cache = clear_fid_cache,
        amp = amp
    )
    # 如果需要生成样本图片
    if generate:
        # 创建训练器对象
        model = Trainer(**model_args)
        # 从指定路径加载模型
        model.load(load_from)
        # 生成带时间戳的文件名
        samples_name = timestamped_filename()
        # 获取模型的检查点编号
        checkpoint = model.checkpoint_num
        # 生成样本图片并返回结果目录
        dir_result = model.generate(samples_name, num_image_tiles, checkpoint, generate_types)
        # 打印生成的样本图片目录
        print(f'sample images generated at {dir_result}')
        return

    # 如果需要生成插值图片
    if generate_interpolation:
        # 创建训练器对象
        model = Trainer(**model_args)
        # 从指定路径加载模型
        model.load(load_from)
        # 生成带时间戳的文件名
        samples_name = timestamped_filename()
        # 生成插值图片
        model.generate_interpolation(samples_name, num_image_tiles, num_steps = interpolation_num_steps, save_frames = save_frames)
        # 打印生成的插值图片目录
        print(f'interpolation generated at {results_dir}/{name}/{samples_name}')
        return

    # 如果需要展示训练进度
    if show_progress:
        # 创建训练器对象
        model = Trainer(**model_args)
        # 展示训练进度
        model.show_progress(num_images=num_image_tiles, types=generate_types)
        return

    # 获取当前可用的 GPU 数量
    world_size = torch.cuda.device_count()

    # 如果只有一个 GPU 或者不使用多 GPU 训练
    if world_size == 1 or not multi_gpus:
        # 单 GPU 训练
        run_training(0, 1, model_args, data, load_from, new, num_train_steps, name, seed)
        return

    # 使用多 GPU 训练
    mp.spawn(run_training,
        args=(world_size, model_args, data, load_from, new, num_train_steps, name, seed),
        nprocs=world_size,
        join=True)
# 定义主函数
def main():
    # 使用 Fire 库将 train_from_folder 函数转换为命令行接口
    fire.Fire(train_from_folder)
```