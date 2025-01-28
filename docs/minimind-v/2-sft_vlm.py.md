# `.\minimind-v\2-sft_vlm.py`

```
# 导入操作系统相关模块，用于与操作系统交互
import os
# 导入平台相关模块，判断当前操作系统类型
import platform
# 导入命令行参数解析模块，用于处理命令行输入的参数
import argparse
# 导入时间模块，用于处理时间相关操作
import time
# 导入数学模块，提供数学运算支持
import math
# 导入警告模块，用于管理警告信息
import warnings
# 导入 JSON 模块，用于处理 JSON 格式数据
import json

# 导入 pandas 库，用于数据处理和分析
import pandas as pd
# 导入 PyTorch 库，提供深度学习支持
import torch
# 导入 PyTorch 神经网络功能模块，提供常用的神经网络操作
import torch.nn.functional as F
# 导入 PyTorch 分布式训练模块，用于分布式计算
import torch.distributed as dist
# 导入上下文管理器模块，用于管理程序上下文
from contextlib import nullcontext

# 导入优化器模块，提供常见的优化算法
from torch import optim
# 导入分布式数据并行模块，支持模型并行训练
from torch.nn.parallel import DistributedDataParallel
# 导入数据加载模块，支持分布式数据加载
from torch.utils.data import DataLoader, DistributedSampler
# 导入 transformers 库，提供与预训练模型相关的工具
from transformers import AutoTokenizer, AutoModel
# 导入自定义的 Transformer 模型
from model.model import Transformer
# 导入自定义的配置文件，用于定义语言模型的配置
from model.LMConfig import LMConfig
# 导入自定义的数据集类，用于处理数据加载和预处理
from model.dataset import SFTDataset, SFTDataset_multi
# 导入计算机视觉相关的工具函数
from model.vision_utils import get_vision_model, get_img_embedding

# 关闭所有警告信息的显示
warnings.filterwarnings('ignore')


# 定义一个函数，计算模型中可训练的参数数量
def count_parameters(model):
    # 遍历模型的参数，计算所有需要梯度更新的参数数量
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 定义日志记录函数，控制是否打印日志信息
def Logger(content):
    # 如果没有分布式训练或者当前进程是主进程，则打印日志
    if not ddp or dist.get_rank() == 0:
        print(content)


# 定义学习率调度函数，根据当前训练步数调整学习率
def get_lr(it, all):
    # 获取预热步数
    warmup_iters = args.warmup_iters
    # 获取总的学习率衰减步数
    lr_decay_iters = all
    # 设置最小学习率为初始学习率的 1/10
    min_lr = args.learning_rate / 10

    # 如果当前训练步数小于预热步数，按比例递增学习率
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    # 如果当前训练步数大于学习率衰减步数，返回最小学习率
    if it > lr_decay_iters:
        return min_lr
    # 计算学习率衰减的比例
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    # 确保衰减比例在合理范围内
    assert 0 <= decay_ratio <= 1
    # 使用余弦退火函数计算衰减后的学习率
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    # 返回最终的学习率
    return min_lr + coeff * (args.learning_rate - min_lr)


# 定义训练一个 epoch 的函数
def train_epoch(epoch, wandb):
    # 记录当前时间，用于计算一个 epoch 训练所需的时间
    start_time = time.time()
    # 遍历训练数据集中的每个批次，获取输入数据、标签、损失掩码和图像处理数据
    for step, (X, Y, loss_mask, image_process) in enumerate(train_loader):
        # 将输入数据、标签、损失掩码和图像处理数据转移到指定的设备（如GPU）
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        image_process = image_process.to(args.device)
        # 获取图像编码器的输出，通常是图像嵌入特征
        image_encoders = get_img_embedding(image_process, vision_model)
        # 计算当前步长的学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        # 更新优化器中每个参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
        # 使用上下文管理器 ctx 进行梯度缩放等操作
        with ctx:
            # 将模型输出的 logits 计算损失
            logits = model(X, Y, image_encoders=image_encoders).logits
            # 使用交叉熵损失计算方法，忽略标签为0的部分
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0, reduction='none')
            # 将损失掩码展平
            loss_mask = loss_mask.view(-1)
            # 计算加权平均的损失，使用损失掩码进行过滤
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()
    
        # 对损失进行缩放，准备反向传播
        scaler.scale(loss).backward()
    
        # 每隔指定步数进行一次梯度更新
        if (step + 1) % args.accumulation_steps == 0:
            # 解除优化器的缩放，准备进行梯度裁剪
            scaler.unscale_(optimizer)
            # 对模型的梯度进行裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    
            # 执行优化器一步更新
            scaler.step(optimizer)
            # 更新缩放器
            scaler.update()
    
            # 清空梯度缓存
            optimizer.zero_grad(set_to_none=True)
    
        # 每隔一定的步骤记录日志信息
        if step % args.log_interval == 0:
            # 计算当前步骤的训练时间
            spend_time = time.time() - start_time
            # 使用 Logger 输出当前的训练状态
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
    
            # 如果启用了 Wandb 并且不是分布式训练，则记录指标到 Wandb
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})
    
        # 每隔一定步数保存模型
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            # 将模型切换到评估模式
            model.eval()
            # 根据配置判断是否使用 MOE（混合专家模型）
            moe_path = '_moe' if lm_config.use_moe else ''
            # 根据是否使用多图训练，决定模型的保存路径和文件名
            if args.multi:  # 多图训练权重保存
                ckp = f'{args.save_dir}/{lm_config.dim}{moe_path}_vlm_sft_multi.pth'
            else:  # 单图训练权重保存
                ckp = f'{args.save_dir}/{lm_config.dim}{moe_path}_vlm_sft.pth'
    
            # 获取模型的 state_dict，如果是分布式训练，则需要获取 model.module 的 state_dict
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
    
            # 保存模型的 state_dict 到指定路径
            torch.save(state_dict, ckp)
            # 将模型切换回训练模式
            model.train()
    # 如果设置了保存最后一个模型，并且不是分布式训练或者当前进程是主进程，并且是多GPU训练模式
    if args.save_last and (not ddp or dist.get_rank() == 0) and args.multi:
        # 将模型设置为评估模式
        model.eval()
        # 如果配置中使用了Mixture of Experts（MOE），则在模型路径中添加'_moe'后缀
        moe_path = '_moe' if lm_config.use_moe else ''
        # 拼接模型保存路径
        ckp = f'{args.save_dir}/{lm_config.dim}{moe_path}_vlm_sft_multi.pth'

        # 如果模型是torch.nn.parallel.DistributedDataParallel类型，则获取模型的module的状态字典
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            # 否则获取模型的状态字典
            state_dict = model.state_dict()

        # 保存模型状态字典到指定路径
        torch.save(state_dict, ckp)
        # 将模型设置为训练模式
        model.train()
# 初始化模型的配置，加载模型和分词器
def init_model(lm_config):
    # 加载预训练的分词器
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    # 根据配置选择是否使用 MOE（Mixture of Experts）
    moe_path = '_moe' if lm_config.use_moe else ''

    # 多图推理建议在单图推理之后
    if args.multi:
        # 如果是多图推理，加载对应的 checkpoint 文件
        ckp = f'./out/{lm_config.dim}{moe_path}_vlm_sft.pth'
    else:
        # 如果是单图推理，加载预训练的 checkpoint 文件
        ckp = f'./out/{lm_config.dim}{moe_path}_vlm_pretrain.pth'

    # 初始化 Transformer 模型
    model = Transformer(lm_config)
    # 加载保存的模型状态字典
    state_dict = torch.load(ckp, map_location=args.device)

    # 处理模型中的不需要的前缀（通常是为了兼容不同版本的模型）
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        # 如果模型参数名包含不需要的前缀，去掉前缀
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # 加载模型参数到 Transformer 模型中
    model.load_state_dict(state_dict, strict=False)
    # 将模型移动到指定设备（如 GPU）
    model = model.to(args.device)

    # 打印模型的可学习参数数量，以百万和十亿为单位
    print(f'模型可学习参数: {count_parameters(model) / 1e6} 百万 = {count_parameters(model) / 1e9} B (Billion)')

    # 获取视觉模型和数据预处理函数
    (vision_model, preprocess) = get_vision_model(args.visual_encoder)
    # 将视觉模型移至指定设备（如 GPU）
    vision_model = vision_model.to(args.device)
    # 返回模型、分词器和视觉模型
    return model, tokenizer, (vision_model, preprocess)


# 初始化分布式训练模式
def init_distributed_mode():
    # 如果没有开启分布式训练，则跳过
    if not ddp: return
    # 声明分布式训练相关的全局变量
    global ddp_local_rank, DEVICE

    # 初始化分布式进程组，使用 NCCL 后端
    dist.init_process_group(backend="nccl")
    # 获取当前进程的全局排名
    ddp_rank = int(os.environ["RANK"])
    # 获取当前进程在本地机器中的排名
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    # 获取分布式训练中的总进程数
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    # 设置当前设备为对应的 GPU
    DEVICE = f"cuda:{ddp_local_rank}"
    # 设置 PyTorch 当前 GPU 设备
    torch.cuda.set_device(DEVICE)


# 主程序入口
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="MiniMind-V-SFT")
    # 定义输出目录参数
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
    # 定义训练轮数参数
    parser.add_argument("--epochs", type=int, default=19, help="Number of epochs")
    # 定义批次大小参数
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    # 定义学习率参数
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    # 定义使用的设备（默认为 CUDA 或 CPU）
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    # 定义数据类型（默认为 bfloat16）
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    # 定义是否使用 Weights & Biases 进行监控
    parser.add_argument("--use_wandb", default=False, action="store_true", help="Use Weights & Biases")
    # 定义 Weights & Biases 项目名称
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V", help="Weights & Biases project name")
    # 定义数据加载时使用的工作线程数
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    # 定义训练数据的路径
    parser.add_argument("--data_path", type=str, default="./dataset/LLaVA-Instruct/llava_instruct_230k.json",
                        help="Path to training data")
    # 定义多图数据的路径
    parser.add_argument("--data_path_multi", type=str, default="./dataset/sft_multi_images/output.json",
                        help="Path to multi images training data")
    # 定义是否启用分布式数据并行训练
    parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel")
    # 定义梯度累积步数
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    # 定义梯度裁剪阈值
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    # 添加参数：热身迭代次数
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup iterations")
    # 添加参数：日志记录间隔
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    # 添加参数：模型保存间隔
    parser.add_argument("--save_interval", type=int, default=100, help="Model saving interval")
    # 添加参数：分布式训练的本地排名
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training')
    # 添加参数：多图像训练
    parser.add_argument('--multi', type=bool, default=False, help='multi-images training')
    # 添加参数：是否保存最后一步模型
    parser.add_argument('--save_last', type=bool, default=True, help='save last step model')
    # 添加参数：视觉编码器类型
    parser.add_argument('--visual_encoder', type=str, default="clip", help='type of visual endcoder')

    # 解析命令行参数
    args = parser.parse_args()

    # 根据视觉编码器类型选择语言模型配置
    if args.visual_encoder == "clip":
        lm_config = LMConfig()
    else:
        lm_config = LMConfig(image_special_token='<' * 98 + '>' * 98, image_ids=[30] * 98 + [32] * 98)

    # 获取最大序列长度
    max_seq_len = lm_config.max_seq_len
    # 设置保存目录
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    # 计算每次迭代的标记数
    tokens_per_iter = args.batch_size * max_seq_len
    # 设置随机种子
    torch.manual_seed(1337)
    # 确定设备类型
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 设置WandB运行名称
    args.wandb_run_name = f"MiniMind-V-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 根据设备类型选择上下文管理器
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    # 判断是否为分布式训练
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # 初始化WandB
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型、分词器和视觉模型
    model, tokenizer, (vision_model, preprocess) = init_model(lm_config)

    # 设置使用版本
    use_version = 0
    # 根据是否多图像训练选择数据集
    if args.multi:
        print("进行多图训练，建议在指令微调后进行...")
        train_ds = SFTDataset_multi(args.data_path_multi, tokenizer, vision_model=(vision_model, preprocess),
                                    image_special_token=lm_config.image_special_token,
                                    max_length=max_seq_len)
    else:
        train_ds = SFTDataset(args.data_path, tokenizer, vision_model=(vision_model, preprocess),
                              image_special_token=lm_config.image_special_token,
                              max_length=max_seq_len)
    # 如果是分布式训练，设置分布式采样器
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
    # 如果条件满足，则编译模型以优化性能
        if False and not lm_config.use_moe and platform.system() != 'Windows' and float(
                torch.__version__.split('.')[0]) >= 2:
            # 输出日志，提示正在编译模型
            Logger("compiling the model... (takes a ~minute)")
            # 将原始模型保存为未优化模型
            unoptimized_model = model
            # 对模型进行编译，以提高性能
            model = torch.compile(model)
    
    # 如果启用了分布式数据并行（DDP），则进行分布式训练设置
        if ddp:
            # 设置需要忽略的参数和缓冲区，这些在分布式训练时不参与计算
            model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
            # 使用分布式数据并行包装模型，并指定本地的设备 ID
            model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
    
    # 获取每个训练周期的迭代次数
        iter_per_epoch = len(train_loader)
        # 遍历训练周期，进行每个周期的训练
        for epoch in range(args.epochs):
            # 在每个 epoch 开始时调用 train_epoch 函数进行训练
            train_epoch(epoch, wandb)
```