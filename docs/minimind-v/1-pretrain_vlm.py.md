# `.\minimind-v\1-pretrain_vlm.py`

```py
# 导入操作系统相关模块
import os
# 导入平台相关模块，用于获取系统平台信息
import platform
# 导入命令行参数解析模块
import argparse
# 导入时间相关功能模块
import time
# 导入数学计算模块
import math
# 导入警告模块，控制警告信息
import warnings
# 导入 JSON 编解码模块
import json

# 导入 pandas 库，常用于数据处理和分析
import pandas as pd
# 导入 PyTorch 框架相关模块
import torch
# 导入 PyTorch 的功能模块，如激活函数等
import torch.nn.functional as F
# 导入 PyTorch 分布式训练相关模块
import torch.distributed as dist
# 从上下文管理器库导入 nullcontext，用于创建空上下文管理器
from contextlib import nullcontext

# 导入 PyTorch 优化器模块
from torch import optim
# 导入分布式数据并行模块，用于在多个 GPU 上并行训练
from torch.nn.parallel import DistributedDataParallel
# 导入数据加载器模块，用于加载数据集
from torch.utils.data import DataLoader, DistributedSampler
# 从 Hugging Face 的 transformers 库导入自动加载的 Tokenizer 和 Model
from transformers import AutoTokenizer, AutoModel
# 导入自定义的 Transformer 模型
from model.model import Transformer
# 导入自定义的配置模块，用于 LM 配置
from model.LMConfig import LMConfig
# 导入自定义的数据集模块，用于预训练数据集
from model.dataset import PretrainDataset
# 导入自定义的视觉相关工具函数，如获取视觉模型和图像嵌入
from model.vision_utils import get_vision_model, get_img_embedding

# 忽略所有警告信息
warnings.filterwarnings('ignore')


# 定义一个函数，用于计算模型中所有需要梯度更新的参数总数
def count_parameters(model):
    # 使用 list comprehension 遍历模型的每个参数，计算所有可训练的参数个数
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 定义一个日志输出函数，仅在分布式训练中 rank 为 0 的进程输出日志
def Logger(content):
    # 如果不是分布式训练或者当前进程是 rank 0 的进程，则打印内容
    if not ddp or dist.get_rank() == 0:
        print(content)


# 定义一个学习率调整函数，根据迭代次数动态调整学习率
def get_lr(it, all):
    # 获取 warmup 的迭代次数
    warmup_iters = args.warmup_iters
    # 获取总的迭代次数
    lr_decay_iters = all
    # 设置最小学习率为初始学习率的十分之一
    min_lr = args.learning_rate / 10

    # 如果当前迭代次数小于 warmup 迭代次数，则按比例增加学习率
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    # 如果当前迭代次数大于学习率衰减的迭代次数，则使用最小学习率
    if it > lr_decay_iters:
        return min_lr
    # 计算当前迭代在衰减区间内的比例
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    # 确保衰减比例在合理范围内
    assert 0 <= decay_ratio <= 1
    # 根据余弦衰减公式计算学习率的调整系数
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    # 返回衰减后的学习率
    return min_lr + coeff * (args.learning_rate - min_lr)


# 定义训练一个 epoch 的函数
def train_epoch(epoch, wandb):
    # 记录当前时间，用于计算本 epoch 的训练时间
    start_time = time.time()
    # 遍历训练数据加载器中的每个步骤，获取输入数据 X、标签数据 Y、损失掩码 loss_mask 和图像处理数据 image_process
    for step, (X, Y, loss_mask, image_process) in enumerate(train_loader):
        # 将输入数据 X 移动到指定设备上
        X = X.to(args.device)
        # 将标签数据 Y 移动到指定设备上
        Y = Y.to(args.device)
        # 将损失掩码 loss_mask 移动到指定设备上
        loss_mask = loss_mask.to(args.device)
        # 将图像处理数据 image_process 移动到指定设备上
        image_process = image_process.to(args.device)
        # 使用图像处理数据获取图像编码器
        image_encoders = get_img_embedding(image_process, vision_model)
        # 根据当前 epoch 和步骤计算学习率 lr
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        # 更新优化器中每个参数组的学习率为 lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 进入自动混合精度训练环境
        with ctx:
            # 获取模型输出 logits
            logits = model(X, Y, image_encoders=image_encoders).logits
            # 计算交叉熵损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0, reduction='none')
            # 将损失掩码展平
            loss_mask = loss_mask.view(-1)
            # 计算加权损失
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        # 反向传播并缩放损失
        scaler.scale(loss).backward()

        # 每累积一定步数的梯度更新一次模型参数
        if (step + 1) % args.accumulation_steps == 0:
            # 反缩放优化器的梯度
            scaler.unscale_(optimizer)
            # 对模型参数进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 更新模型参数
            scaler.step(optimizer)
            scaler.update()

            # 清空优化器的梯度
            optimizer.zero_grad(set_to_none=True)

        # 每隔一定步数记录训练日志
        if step % args.log_interval == 0:
            # 计算已经花费的时间
            spend_time = time.time() - start_time
            # 记录日志信息
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # 如果使用了 wandb 并且不是分布式训练或者当前进程是主进程
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                # 记录训练指标到 wandb
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 每隔一定步数保存模型参数
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            # 将模型设置为评估模式
            model.eval()
            # 根据是否使用 moe 构建模型保存路径
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{lm_config.dim}{moe_path}_vlm_pretrain.pth'

            # 如果模型是分布式数据并行模型，则获取模型的状态字典
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 保存模型参数到指定路径
            torch.save(state_dict, ckp)
            # 将模型设置为训练模式
            model.train()
# 初始化模型，包括加载 tokenizer 和模型权重
def init_model(lm_config):
    # 从指定路径加载预训练的 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    # 根据配置决定是否使用 MoE，构建模型文件路径
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/{lm_config.dim}{moe_path}_llm.pth'

    # 使用配置初始化 Transformer 模型
    model = Transformer(lm_config)
    # 加载模型权重
    state_dict = torch.load(ckp, map_location=args.device)

    # 处理模型权重中不需要的前缀（移除 '_orig_mod.' 前缀）
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            # 删除不需要的前缀并更新字典
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # 将权重加载到模型中，strict=False 表示忽略不匹配的权重
    model.load_state_dict(state_dict, strict=False)
    # 将模型移动到指定设备（如 GPU）
    model = model.to(args.device)

    # 输出模型的可学习参数数量，单位是百万和十亿
    print(f'模型可学习参数: {count_parameters(model) / 1e6} 百万 = {count_parameters(model) / 1e9} B (Billion)')

    # 获取视觉模型和预处理函数
    (vision_model, preprocess) = get_vision_model(args.visual_encoder)
    # 将视觉模型也移动到指定设备
    vision_model = vision_model.to(args.device)
    # 返回模型、tokenizer 和视觉模型、预处理函数
    return model, tokenizer, (vision_model, preprocess)


# 初始化分布式训练环境
def init_distributed_mode():
    # 如果没有启用分布式模式，直接返回
    if not ddp: return
    global ddp_local_rank, DEVICE

    # 初始化分布式进程组，使用 NCCL 后端
    dist.init_process_group(backend="nccl")
    # 获取当前进程的 rank 和 local rank
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    # 获取分布式训练的总进程数
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    # 设置当前设备为对应的 GPU
    DEVICE = f"cuda:{ddp_local_rank}"
    # 设置 CUDA 设备
    torch.cuda.set_device(DEVICE)


# 主函数
if __name__ == "__main__":
    # 创建参数解析器，定义训练过程中的各种超参数
    parser = argparse.ArgumentParser(description="MiniMind-V Pretrain")
    # 定义输出目录
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
    # 定义训练周期数
    parser.add_argument("--epochs", type=int, default=19, help="Number of epochs")
    # 定义批量大小
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    # 定义学习率
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="Learning rate")
    # 定义设备（默认 GPU 或 CPU）
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    # 定义数据类型（如 bfloat16）
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    # 是否使用 Weights & Biases 进行实验记录
    parser.add_argument("--use_wandb", default=False, action="store_true", help="Use Weights & Biases")
    # Weights & Biases 项目名称
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V", help="Weights & Biases project name")
    # 定义数据加载的工作线程数
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    # 定义训练数据的路径
    parser.add_argument("--data_path", type=str, default="./dataset/LLaVA-Pretrain/chat-translated.json",
                        help="Path to training data")
    # 是否启用分布式训练（DistributedDataParallel）
    parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel")
    # 定义梯度累积步数
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    # 定义梯度裁剪阈值
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    # 定义预热迭代次数
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup iterations")
    # 定义日志记录的间隔
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    # 定义模型保存的间隔
    parser.add_argument("--save_interval", type=int, default=100, help="Model saving interval")
    # 添加命令行参数'--local_rank'，用于分布式训练中指定本地排名
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training')
    # 添加命令行参数'--visual_encoder'，用于指定视觉编码器的类型
    parser.add_argument('--visual_encoder', type=str, default="clip", help='type of visual endcoder')

    # 解析命令行参数
    args = parser.parse_args()

    # 根据视觉编码器类型选择LMConfig配置
    if args.visual_encoder == "clip":
        lm_config = LMConfig()
    else:
        lm_config = LMConfig(image_special_token='<' * 98 + '>' * 98, image_ids=[30] * 98 + [32] * 98)

    # 获取最大序列长度
    max_seq_len = lm_config.max_seq_len
    # 设置保存目录
    args.save_dir = os.path.join(args.out_dir)
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    # 计算每次迭代的标记数
    tokens_per_iter = args.batch_size * max_seq_len
    # 设置随机种子
    torch.manual_seed(1337)
    # 确定设备类型
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 设置WandB运行名称
    args.wandb_run_name = f"MiniMind-V Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 根据设备类型选择上下文管理器
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    # 判断是否为分布式训练
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    # 如果是分布式训练，初始化分布式模式
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # 如果使用WandB并且不是分布式训练或者是分布式训练的主进程
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        # 初始化WandB
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型、分词器、视觉模型和预处理器
    model, tokenizer, (vision_model, preprocess) = init_model(lm_config)

    # 设置使用版本号
    use_version = 0
    # 创建预训练数据集
    train_ds = PretrainDataset(args.data_path, tokenizer, vision_model=(vision_model, preprocess),
                               image_special_token=lm_config.image_special_token,
                               max_length=max_seq_len)
    # 如果是分布式训练，使用分布式采样器
    train_sampler = DistributedSampler(train_ds) if ddp else None
    # 创建数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 创建梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 如果条件满足，编译模型
    if False and not lm_config.use_moe and platform.system() != 'Windows' and float(
            torch.__version__.split('.')[0]) >= 2:
        Logger("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)

    # 如果是分布式训练，设置忽略的参数和缓冲区
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 计算每个epoch的迭代次数
    iter_per_epoch = len(train_loader)
    # 遍历每个epoch进行训练
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
```