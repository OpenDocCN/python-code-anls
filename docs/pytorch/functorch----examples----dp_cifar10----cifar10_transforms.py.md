# `.\pytorch\functorch\examples\dp_cifar10\cifar10_transforms.py`

```py
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Runs CIFAR10 training with differential privacy.
"""

# 引入必要的库和模块
import argparse  # 导入命令行参数解析模块
import logging  # 导入日志记录模块
import shutil  # 导入文件操作模块
import sys  # 导入系统相关模块
from datetime import datetime, timedelta  # 导入日期时间处理模块

import numpy as np  # 导入数值计算库 numpy
import torchvision.transforms as transforms  # 导入图像处理模块
from torchvision import models  # 导入 torchvision 中的模型
from torchvision.datasets import CIFAR10  # 导入 CIFAR10 数据集
from tqdm import tqdm  # 导入进度条模块

import torch  # 导入 PyTorch 深度学习框架
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import torch.utils.data  # 导入数据处理模块

from torch.func import functional_call, grad_and_value, vmap  # 导入自定义函数

# 配置日志记录格式
logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("ddp")
logger.setLevel(level=logging.INFO)


def save_checkpoint(state, is_best, filename="checkpoint.tar"):
    torch.save(state, filename)  # 保存训练状态到文件
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")  # 复制最佳模型文件


def accuracy(preds, labels):
    return (preds == labels).mean()  # 计算预测准确率


def compute_norms(sample_grads):
    batch_size = sample_grads[0].shape[0]  # 获取批次大小
    norms = [
        sample_grad.view(batch_size, -1).norm(2, dim=-1) for sample_grad in sample_grads
    ]  # 计算每个样本梯度的范数
    norms = torch.stack(norms, dim=0).norm(2, dim=0)  # 计算所有样本梯度范数的总和
    return norms, batch_size  # 返回范数和批次大小


def clip_and_accumulate_and_add_noise(
    model, max_per_sample_grad_norm=1.0, noise_multiplier=1.0
):
    sample_grads = tuple(param.grad_sample for param in model.parameters())  # 获取参数梯度样本

    # step 0: compute the norms
    sample_norms, batch_size = compute_norms(sample_grads)  # 计算梯度范数和批次大小

    # step 1: compute clipping factors
    clip_factor = max_per_sample_grad_norm / (sample_norms + 1e-6)  # 计算剪切因子
    clip_factor = clip_factor.clamp(max=1.0)  # 对剪切因子进行限制

    # step 2: clip
    grads = tuple(
        torch.einsum("i,i...", clip_factor, sample_grad) for sample_grad in sample_grads
    )  # 剪切梯度

    # step 3: add gaussian noise
    stddev = max_per_sample_grad_norm * noise_multiplier  # 计算高斯噪声的标准差
    noises = tuple(
        torch.normal(0, stddev, grad_param.shape, device=grad_param.device)
        for grad_param in grads
    )  # 生成高斯噪声
    grads = tuple(noise + grad_param for noise, grad_param in zip(noises, grads))  # 添加噪声到梯度

    # step 4: assign the new grads, delete the sample grads
    for param, param_grad in zip(model.parameters(), grads):
        param.grad = param_grad / batch_size  # 分配新梯度并进行批次平均
        del param.grad_sample  # 删除梯度样本


def train(args, model, train_loader, optimizer, epoch, device):
    start_time = datetime.now()  # 记录训练开始时间

    criterion = nn.CrossEntropyLoss()  # 定义损失函数

    losses = []  # 初始化损失列表
    top1_acc = []  # 初始化Top-1准确率列表

    train_duration = datetime.now() - start_time  # 计算训练时长
    return train_duration  # 返回训练时长


def test(args, model, test_loader, device):
    model.eval()  # 切换模型为评估模式
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    losses = []  # 初始化损失列表
    top1_acc = []  # 初始化Top-1准确率列表
    # 使用 torch.no_grad() 上下文管理器，确保在评估模型时不进行梯度计算
    with torch.no_grad():
        # 遍历测试数据加载器中的图像和目标标签
        for images, target in tqdm(test_loader):
            # 将图像和目标标签移动到指定的设备（通常是 GPU）
            images = images.to(device)
            target = target.to(device)

            # 使用模型进行前向推断，生成输出
            output = model(images)
            # 计算模型输出与目标标签之间的损失
            loss = criterion(output, target)
            # 将模型输出转换为 numpy 数组，计算每个样本的预测类别
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            # 将目标标签转换为 numpy 数组
            labels = target.detach().cpu().numpy()
            # 计算预测准确率
            acc1 = accuracy(preds, labels)

            # 将损失值加入损失列表
            losses.append(loss.item())
            # 将准确率加入准确率列表
            top1_acc.append(acc1)

    # 计算所有样本的平均 top-1 准确率
    top1_avg = np.mean(top1_acc)

    # 打印测试集的平均损失和平均 top-1 准确率
    print(f"\tTest set:" f"Loss: {np.mean(losses):.6f} " f"Acc@1: {top1_avg :.6f} ")
    # 返回所有样本的平均 top-1 准确率
    return np.mean(top1_acc)
# flake8: noqa: C901  # 禁用 flake8 对函数长度过长的警告

# 主函数入口
def main():
    # 解析命令行参数
    args = parse_args()

    # 如果设置了调试级别为1或更高，则设置日志级别为调试模式
    if args.debug >= 1:
        logger.setLevel(level=logging.DEBUG)

    # 设备选择
    device = args.device

    # 如果启用安全随机数生成器选项
    if args.secure_rng:
        try:
            import torchcsprng as prng
        except ImportError as e:
            # 抛出导入错误并提示安装 torchcsprng 包的指导信息
            msg = (
                "To use secure RNG, you must install the torchcsprng package! "
                "Check out the instructions here: https://github.com/pytorch/csprng#installation"
            )
            raise ImportError(msg) from e

        # 使用 /dev/urandom 创建安全随机数生成器
        generator = prng.create_random_device_generator("/dev/urandom")

    else:
        generator = None

    # 图像增强处理列表
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    # 图像标准化处理列表
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    # 训练数据集转换器
    train_transform = transforms.Compose(normalize)

    # 测试数据集转换器
    test_transform = transforms.Compose(normalize)

    # 加载 CIFAR-10 训练数据集
    train_dataset = CIFAR10(
        root=args.data_root, train=True, download=True, transform=train_transform
    )

    # 创建 CIFAR-10 训练数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(args.sample_rate * len(train_dataset)),
        generator=generator,
        num_workers=args.workers,
        pin_memory=True,
    )

    # 加载 CIFAR-10 测试数据集
    test_dataset = CIFAR10(
        root=args.data_root, train=False, download=True, transform=test_transform
    )

    # 创建 CIFAR-10 测试数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=args.workers,
    )

    # 记录最佳准确率初始化
    best_acc1 = 0

    # 根据指定的架构选择模型，并初始化
    model = models.__dict__[args.architecture](
        pretrained=False, norm_layer=(lambda c: nn.GroupNorm(args.gn_groups, c))
    )

    # 将模型移动到指定的设备上
    model = model.to(device)

    # 根据指定的优化器类型初始化优化器
    if args.optim == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        # 抛出错误，未识别的优化器类型
        raise NotImplementedError("Optimizer not recognized. Please check spelling")

    # 存储每个 epoch 的准确率日志
    accuracy_per_epoch = []

    # 存储每个 epoch 的训练时间日志
    time_per_epoch = []
    # 对每个 epoch 进行循环，从 args.start_epoch 到 args.epochs + 1
    for epoch in range(args.start_epoch, args.epochs + 1):
        # 如果学习率调度方式为余弦退火
        if args.lr_schedule == "cos":
            # 计算当前 epoch 对应的学习率
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / (args.epochs + 1)))
            # 更新 optimizer 中每个参数组的学习率
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # 执行训练，并记录训练时长
        train_duration = train(args, model, train_loader, optimizer, epoch, device)
        # 执行测试，并获取 top1 精度
        top1_acc = test(args, model, test_loader, device)

        # 更新最佳的 top1 精度和保存 checkpoint
        is_best = top1_acc > best_acc1
        best_acc1 = max(top1_acc, best_acc1)

        # 记录每个 epoch 的训练时长和精度
        time_per_epoch.append(train_duration)
        accuracy_per_epoch.append(float(top1_acc))

        # 保存当前 epoch 的模型 checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": "Convnet",
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            filename=args.checkpoint_file + ".tar",
        )

    # 将每个 epoch 的训练时长转换为秒数，并计算平均每个 epoch 的时间
    time_per_epoch_seconds = [t.total_seconds() for t in time_per_epoch]
    avg_time_per_epoch = sum(time_per_epoch_seconds) / len(time_per_epoch_seconds)

    # 构建指标 metrics，包括最佳精度、每个 epoch 的精度列表、平均每个 epoch 的时间字符串和每个 epoch 的时间列表
    metrics = {
        "accuracy": best_acc1,
        "accuracy_per_epoch": accuracy_per_epoch,
        "avg_time_per_epoch_str": str(timedelta(seconds=int(avg_time_per_epoch))),
        "time_per_epoch": time_per_epoch_seconds,
    }

    # 记录日志信息，说明 'total_time' 包括数据加载时间、训练时间和测试时间，'time_per_epoch' 衡量仅训练时间
    logger.info(
        "\nNote:\n- 'total_time' includes the data loading time, training time and testing time.\n- 'time_per_epoch' measures the training time only.\n"
    )
    # 输出 metrics 中的信息到日志中
    logger.info(metrics)
# 定义函数用于解析命令行参数
def parse_args():
    # 创建参数解析器，并设置描述信息
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")
    
    # 添加命令行参数：数据加载工作线程数，默认为2
    parser.add_argument(
        "-j",
        "--workers",
        default=2,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 2)",
    )
    
    # 添加命令行参数：总共运行的轮次数，默认为90
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    
    # 添加命令行参数：手动指定起始轮次，默认为1，在重新启动时有用
    parser.add_argument(
        "--start-epoch",
        default=1,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    
    # 添加命令行参数：测试数据集的小批量大小，默认为256
    parser.add_argument(
        "-b",
        "--batch-size-test",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size for test dataset (default: 256)",
    )
    
    # 添加命令行参数：用于批量构建的采样率，默认为0.005
    parser.add_argument(
        "--sample-rate",
        default=0.005,
        type=float,
        metavar="SR",
        help="sample rate used for batch construction (default: 0.005)",
    )
    
    # 添加命令行参数：初始学习率，默认为0.1
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    
    # 添加命令行参数：SGD 动量，默认为0.9
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="SGD momentum"
    )
    
    # 添加命令行参数：SGD 权重衰减，默认为0
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0,
        type=float,
        metavar="W",
        help="SGD weight decay",
        dest="weight_decay",
    )
    
    # 添加命令行参数：打印频率，默认为10
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    
    # 添加命令行参数：恢复训练时使用的最新检查点路径，默认为""
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    
    # 添加命令行参数：评估模型在验证集上的表现
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    
    # 添加命令行参数：初始化训练时的随机种子
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    
    # 添加命令行参数：噪声乘数，默认为1.5
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.5,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    
    # 添加命令行参数：每个样本的梯度规范裁剪阈值，默认为10.0
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=10.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    
    # 添加命令行参数：启用安全随机数生成器以获得可信的隐私保证
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees."
        "Comes at a performance cost. Opacus will emit a warning if secure rng is off,"
        "indicating that for production use it's recommender to turn it on.",
    )
    # 添加一个名为 `--delta` 的命令行参数，用于指定目标的 delta 值，默认为 1e-5
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )

    # 添加一个名为 `--checkpoint-file` 的命令行参数，用于指定保存检查点文件的路径，默认为 "checkpoint"
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="checkpoint",
        help="path to save check points",
    )

    # 添加一个名为 `--data-root` 的命令行参数，用于指定 CIFAR10 数据集的存储路径，默认为 "../cifar10"
    parser.add_argument(
        "--data-root",
        type=str,
        default="../cifar10",
        help="Where CIFAR10 is/will be stored",
    )

    # 添加一个名为 `--log-dir` 的命令行参数，用于指定 Tensorboard 日志的存储路径，默认为 "/tmp/stat/tensorboard"
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/tmp/stat/tensorboard",
        help="Where Tensorboard log will be stored",
    )

    # 添加一个名为 `--optim` 的命令行参数，用于指定优化器的选择，默认为 "SGD"，可选值为 "Adam", "RMSprop", "SGD"
    parser.add_argument(
        "--optim",
        type=str,
        default="SGD",
        help="Optimizer to use (Adam, RMSprop, SGD)",
    )

    # 添加一个名为 `--lr-schedule` 的命令行参数，用于指定学习率调度的方法，默认为 "cos"，可选值为 "constant", "cos"
    parser.add_argument(
        "--lr-schedule", type=str, choices=["constant", "cos"], default="cos"
    )

    # 添加一个名为 `--device` 的命令行参数，用于指定代码运行的设备，默认为 "cpu"
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device on which to run the code."
    )

    # 添加一个名为 `--architecture` 的命令行参数，用于指定使用的 torchvision 模型，默认为 "resnet18"
    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet18",
        help="model from torchvision to run",
    )

    # 添加一个名为 `--gn-groups` 的命令行参数，用于指定 GroupNorm 中的组数，默认为 8
    parser.add_argument(
        "--gn-groups",
        type=int,
        default=8,
        help="Number of groups in GroupNorm",
    )

    # 添加一个名为 `--clip-per-layer` 的命令行参数，如果设置，则使用每层静态裁剪，否则使用统一裁剪阈值。在分布式数据并行（DDP）中必须设置为 `True`。
    parser.add_argument(
        "--clip-per-layer",
        "--clip_per_layer",
        action="store_true",
        default=False,
        help="Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.",
    )

    # 添加一个名为 `--debug` 的命令行参数，用于设置调试级别，默认为 0
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="debug level (default: 0)",
    )

    # 解析并返回所有添加的命令行参数
    return parser.parse_args()
# 如果当前脚本作为主程序执行，则调用主函数 main()
if __name__ == "__main__":
    main()
```