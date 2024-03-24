# `.\lucidrains\stylegan2-pytorch\stylegan2_pytorch\cli.py`

```py
# 导入所需的库
import os
import fire
import random
from retry.api import retry_call
from tqdm import tqdm
from datetime import datetime
from functools import wraps
from stylegan2_pytorch import Trainer, NanException

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np

# 定义一个函数，将输入转换为列表
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

    progress_bar = tqdm(initial = model.steps, total = num_train_steps, mininterval=10., desc=f'{name}<{data}>')
    while model.steps < num_train_steps:
        retry_call(model.train, tries=3, exceptions=NanException)
        progress_bar.n = model.steps
        progress_bar.refresh()
        if is_main and model.steps % 50 == 0:
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
    image_size = 128,
    network_capacity = 16,
    fmap_max = 512,
    transparent = False,
    batch_size = 5,
    gradient_accumulate_every = 6,
    num_train_steps = 150000,
    learning_rate = 2e-4,
    lr_mlp = 0.1,
    ttur_mult = 1.5,
    rel_disc_loss = False,
    num_workers =  None,
    save_every = 1000,
    evaluate_every = 1000,
    generate = False,
    num_generate = 1,
    generate_interpolation = False,
    interpolation_num_steps = 100,
    save_frames = False,
    num_image_tiles = 8,
    trunc_psi = 0.75,
    mixed_prob = 0.9,
    fp16 = False,
    no_pl_reg = False,
    cl_reg = False,
    fq_layers = [],
    fq_dict_size = 256,
    attn_layers = [],
    no_const = False,
    aug_prob = 0.,
    aug_types = ['translation', 'cutout'],
    top_k_training = False,
    generator_top_k_gamma = 0.99,
    generator_top_k_frac = 0.5,
    dual_contrast_loss = False,
    dataset_aug_prob = 0.,
    multi_gpus = False,
    calculate_fid_every = None,
    calculate_fid_num_images = 12800,
    clear_fid_cache = False,
    seed = 42,
    log = False
):
    model_args = dict(
        name = name,  # 模型名称
        results_dir = results_dir,  # 结果保存目录
        models_dir = models_dir,  # 模型保存目录
        batch_size = batch_size,  # 批量大小
        gradient_accumulate_every = gradient_accumulate_every,  # 梯度积累频率
        image_size = image_size,  # 图像尺寸
        network_capacity = network_capacity,  # 网络容量
        fmap_max = fmap_max,  # 最大特征图数
        transparent = transparent,  # 是否透明
        lr = learning_rate,  # 学习率
        lr_mlp = lr_mlp,  # MLP学习率
        ttur_mult = ttur_mult,  # TTUR倍数
        rel_disc_loss = rel_disc_loss,  # 相对鉴别器损失
        num_workers = num_workers,  # 工作进程数
        save_every = save_every,  # 保存频率
        evaluate_every = evaluate_every,  # 评估频率
        num_image_tiles = num_image_tiles,  # 图像瓦片数
        trunc_psi = trunc_psi,  # 截断参数
        fp16 = fp16,  # 是否使用FP16
        no_pl_reg = no_pl_reg,  # 是否无PL正则化
        cl_reg = cl_reg,  # CL正则化
        fq_layers = fq_layers,  # FQ层
        fq_dict_size = fq_dict_size,  # FQ字典大小
        attn_layers = attn_layers,  # 注意力层
        no_const = no_const,  # 是否无常数
        aug_prob = aug_prob,  # 数据增强概率
        aug_types = cast_list(aug_types),  # 数据增强类型
        top_k_training = top_k_training,  # Top-K训练
        generator_top_k_gamma = generator_top_k_gamma,  # 生成器Top-K Gamma
        generator_top_k_frac = generator_top_k_frac,  # 生成器Top-K分数
        dual_contrast_loss = dual_contrast_loss,  # 双对比损失
        dataset_aug_prob = dataset_aug_prob,  # 数据集增强概率
        calculate_fid_every = calculate_fid_every,  # 计算FID频率
        calculate_fid_num_images = calculate_fid_num_images,  # 计算FID图像数
        clear_fid_cache = clear_fid_cache,  # 清除FID缓存
        mixed_prob = mixed_prob,  # 混合概率
        log = log  # 日志
    )

    if generate:
        model = Trainer(**model_args)  # 创建Trainer模型
        model.load(load_from)  # 加载模型
        samples_name = timestamped_filename()  # 生成时间戳文件名
        for num in tqdm(range(num_generate)):  # 迭代生成指定数量的样本
            model.evaluate(f'{samples_name}-{num}', num_image_tiles)  # 评估模型生成样本
        print(f'sample images generated at {results_dir}/{name}/{samples_name}')  # 打印生成的样本图像保存路径
        return

    if generate_interpolation:
        model = Trainer(**model_args)  # 创建Trainer模型
        model.load(load_from)  # 加载模型
        samples_name = timestamped_filename()  # 生成时间戳文件名
        model.generate_interpolation(samples_name, num_image_tiles, num_steps = interpolation_num_steps, save_frames = save_frames)  # 生成插值图像
        print(f'interpolation generated at {results_dir}/{name}/{samples_name}')  # 打印插值图像保存路径
        return

    world_size = torch.cuda.device_count()  # 获取GPU数量

    if world_size == 1 or not multi_gpus:
        run_training(0, 1, model_args, data, load_from, new, num_train_steps, name, seed)  # 单GPU训练
        return

    mp.spawn(run_training,
        args=(world_size, model_args, data, load_from, new, num_train_steps, name, seed),
        nprocs=world_size,
        join=True)  # 多GPU训练
# 定义主函数
def main():
    # 使用 Fire 库将 train_from_folder 函数转换为命令行接口
    fire.Fire(train_from_folder)
```