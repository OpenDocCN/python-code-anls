# `so-vits-svc\train.py`

```
# 导入日志模块
import logging
# 导入多进程模块
import multiprocessing
# 导入操作系统模块
import os
# 导入时间模块
import time

# 导入 PyTorch 模块
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 导入自定义模块
import modules.commons as commons
import utils
from data_utils import TextAudioCollate, TextAudioSpeakerLoader
from models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

# 设置 matplotlib 和 numba 模块的日志级别
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

# 设置 PyTorch 的一些后端优化
torch.backends.cudnn.benchmark = True
# 初始化全局步数和开始时间
global_step = 0
start_time = time.time()

# 设置环境变量
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'


def main():
    """Assume Single Node Multi GPUs Training Only"""
    # 检查是否有可用的 GPU，如果没有则抛出异常
    assert torch.cuda.is_available(), "CPU training is not allowed."
    # 获取超参数
    hps = utils.get_hparams()

    # 获取可用的 GPU 数量
    n_gpus = torch.cuda.device_count()
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = hps.train.port

    # 使用多进程进行训练
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    global global_step
    # 如果是主进程
    if rank == 0:
        # 获取日志记录器
        logger = utils.get_logger(hps.model_dir)
        # 记录超参数信息
        logger.info(hps)
        # 检查 Git 版本
        utils.check_git_hash(hps.model_dir)
        # 创建 TensorBoard 的 SummaryWriter
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    
    # 对于 Windows 上的 PyTorch，使用 gloo 后端
    # 对于其他系统，使用 nccl 后端
    dist.init_process_group(backend=  'gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus, rank=rank)
    # 设置随机种子
    torch.manual_seed(hps.train.seed)
    # 设置当前设备为指定 GPU
    torch.cuda.set_device(rank)
    # 创建数据加载器的拼接函数
    collate_fn = TextAudioCollate()
    # 如果内存足够，可以将数据全部加载到内存中，避免磁盘IO，加快训练速度
    all_in_mem = hps.train.all_in_mem   # If you have enough memory, turn on this option to avoid disk IO and speed up training.
    # 根据训练文件和参数创建文本音频加载器
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps, all_in_mem=all_in_mem)
    # 确定并行加载数据的工作进程数
    num_workers = 5 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()
    if all_in_mem:
        num_workers = 0
    # 创建数据加载器，用于训练模型
    train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=False, pin_memory=True,
                              batch_size=hps.train.batch_size, collate_fn=collate_fn)
    # 如果是主进程，创建用于评估的数据加载器
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps, all_in_mem=all_in_mem,vol_aug = False)
        eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
                                 batch_size=1, pin_memory=False,
                                 drop_last=False, collate_fn=collate_fn)

    # 创建音频合成模型
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda(rank)
    # 创建多周期鉴别器模型
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    # 创建音频合成模型的优化器
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    # 创建多周期鉴别器模型的优化器
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    # 使用分布式数据并行将音频合成模型放到指定的GPU上
    net_g = DDP(net_g, device_ids=[rank])  # , find_unused_parameters=True)
    # 使用分布式数据并行将多周期鉴别器模型放到指定的GPU上
    net_d = DDP(net_d, device_ids=[rank])

    # 是否跳过优化器的更新
    skip_optimizer = False
    # 尝试加载最新的生成器模型的检查点文件，并获取其中的信息
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
                                               optim_g, skip_optimizer)
    # 尝试加载最新的判别器模型的检查点文件，并获取其中的信息
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
                                               optim_d, skip_optimizer)
    # 将 epoch_str 设置为最大值，或者 1
    epoch_str = max(epoch_str, 1)
    # 获取最新的判别器模型的检查点文件路径
    name=utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
    # 从文件路径中提取全局步数
    global_step=int(name[name.rfind("_")+1:name.rfind(".")])+1
    # 如果出现异常，打印错误信息，并将 epoch_str 和 global_step 设置为默认值
    except Exception:
        print("load old checkpoint failed...")
        epoch_str = 1
        global_step = 0
    # 如果 skip_optimizer 为真，则将 epoch_str 和 global_step 设置为默认值
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    # 设置预热周期
    warmup_epoch = hps.train.warmup_epochs
    # 创建生成器的学习率调度器
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    # 创建判别器的学习率调度器
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    # 创建梯度缩放器
    scaler = GradScaler(enabled=hps.train.fp16_run)
    # 遍历训练周期范围
    for epoch in range(epoch_str, hps.train.epochs + 1):
        # 设置温暖期学习率
        if epoch <= warmup_epoch:
            # 针对生成器参数组设置学习率
            for param_group in optim_g.param_groups:
                param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
            # 针对判别器参数组设置学习率
            for param_group in optim_d.param_groups:
                param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
        # 训练
        if rank == 0:
            # 如果是主进程，进行训练和评估
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                               [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            # 如果不是主进程，进行训练
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                               [train_loader, None], None, None)
        # 更新学习率
        scheduler_g.step()
        scheduler_d.step()
# 训练和评估函数，接受训练进程编号、当前轮次、超参数、网络模型、优化器、学习率调度器、混合精度缩放器、数据加载器、日志记录器和写入器作为参数
def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    # 解包网络模型和优化器
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    # 如果有写入器，则解包写入器
    if writers is not None:
        writer, writer_eval = writers
    
    # 根据训练配置选择半精度类型
    half_type = torch.bfloat16 if hps.train.half_type=="bf16" else torch.float16

    # 设置训练数据加载器的当前轮次
    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    # 将生成器和判别器设置为训练模式
    net_g.train()
    net_d.train()
    # 如果进程编号为0，则记录当前时间并输出日志
    if rank == 0:
        global start_time
        now = time.time()
        durtaion = format(now - start_time, '.2f')
        logger.info(f'====> Epoch: {epoch}, cost {durtaion} s')
        start_time = now


# 评估函数，接受超参数、生成器、评估数据加载器和评估写入器作为参数
def evaluate(hps, generator, eval_loader, writer_eval):
    # 将生成器设置为评估模式
    generator.eval()
    # 初始化图像和音频字典
    image_dict = {}
    audio_dict = {}
    # 使用 torch.no_grad() 上下文管理器，关闭梯度计算
    with torch.no_grad():
        # 遍历评估数据加载器中的批次索引和数据项
        for batch_idx, items in enumerate(eval_loader):
            # 从数据项中获取 c, f0, spec, y, spk, _, uv, volume
            c, f0, spec, y, spk, _, uv, volume = items
            # 将 spk 的第一个元素移动到 GPU 上
            g = spk[:1].cuda(0)
            # 将 spec, y 的第一个元素移动到 GPU 上
            spec, y = spec[:1].cuda(0), y[:1].cuda(0)
            # 将 c 的第一个元素移动到 GPU 上
            c = c[:1].cuda(0)
            # 将 f0 的第一个元素移动到 GPU 上
            f0 = f0[:1].cuda(0)
            # 将 uv 的第一个元素移动到 GPU 上
            uv = uv[:1].cuda(0)
            # 如果 volume 不为空，则将其第一个元素移动到 GPU 上
            if volume is not None:
                volume = volume[:1].cuda(0)
            # 使用 spec_to_mel_torch 函数将 spec 转换为 mel 频谱
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)
            # 使用 generator 模型进行推理，生成音频 y_hat
            y_hat, _ = generator.module.infer(c, f0, uv, g=g, vol=volume)
    
            # 使用 mel_spectrogram_torch 函数将 y_hat 转换为 mel 频谱
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
    
            # 更新音频字典，包括生成的音频和原始音频
            audio_dict.update({
                f"gen/audio_{batch_idx}": y_hat[0],
                f"gt/audio_{batch_idx}": y[0]
            })
        # 更新图像字典，包括生成的 mel 频谱和原始的 mel 频谱
        image_dict.update({
            "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
            "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
        })
    # 使用 utils.summarize 函数将结果写入评估的 writer 中
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    # 将 generator 恢复为训练模式
    generator.train()
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```