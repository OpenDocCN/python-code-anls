# `.\pytorch\functorch\examples\dp_cifar10\cifar10_opacus.py`

```
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Runs CIFAR10 training with differential privacy.
"""

import argparse               # 导入解析命令行参数的模块
import logging                # 导入日志记录模块
import shutil                 # 导入文件操作模块
import sys                    # 导入系统相关模块
from datetime import datetime, timedelta  # 导入日期时间处理模块

import numpy as np           # 导入数值计算模块numpy
import torchvision.transforms as transforms  # 导入图像变换模块
from opacus import PrivacyEngine  # 导入隐私工具包中的PrivacyEngine
from torchvision import models     # 导入torchvision中的模型
from torchvision.datasets import CIFAR10  # 导入CIFAR10数据集
from tqdm import tqdm         # 导入进度条模块tqdm

import torch                  # 导入PyTorch深度学习库
import torch.nn as nn         # 导入神经网络模块
import torch.optim as optim   # 导入优化器模块
import torch.utils.data       # 导入数据加载模块


logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("ddp")  # 创建名为"ddp"的日志记录器对象
logger.setLevel(level=logging.INFO)  # 设置日志记录器的日志级别为INFO


def save_checkpoint(state, is_best, filename="checkpoint.tar"):
    """
    保存训练模型的检查点。

    Args:
        state (dict): 包含模型状态的字典。
        is_best (bool): 是否为最佳模型。
        filename (str): 保存检查点的文件名，默认为"checkpoint.tar"。
    """
    torch.save(state, filename)  # 使用PyTorch保存模型状态到文件
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")  # 如果是最佳模型，复制到"model_best.pth.tar"


def accuracy(preds, labels):
    """
    计算预测精度。

    Args:
        preds (numpy.ndarray): 模型预测的标签。
        labels (numpy.ndarray): 实际标签。

    Returns:
        float: 精度值，即预测正确的比例。
    """
    return (preds == labels).mean()  # 计算预测正确的比例作为精度


def train(args, model, train_loader, optimizer, privacy_engine, epoch, device):
    """
    训练模型。

    Args:
        args (argparse.Namespace): 命令行参数。
        model (torch.nn.Module): 待训练的模型。
        train_loader (torch.utils.data.DataLoader): 训练数据加载器。
        optimizer (torch.optim.Optimizer): 优化器。
        privacy_engine (PrivacyEngine): 隐私引擎。
        epoch (int): 当前训练的轮次。
        device (torch.device): 训练设备。

    Returns:
        datetime.timedelta: 训练持续时间。
    """
    start_time = datetime.now()  # 记录训练开始时间

    model.train()  # 设置模型为训练模式
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数

    losses = []  # 存储每个batch的损失
    top1_acc = []  # 存储每个batch的Top-1精度

    for i, (images, target) in enumerate(tqdm(train_loader)):
        images = images.to(device)  # 将图像数据移动到指定设备
        target = target.to(device)  # 将标签数据移动到指定设备

        # 计算模型输出
        output = model(images)
        loss = criterion(output, target)  # 计算损失
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)  # 预测的标签
        labels = target.detach().cpu().numpy()  # 实际的标签

        # 计算精度并记录损失
        acc1 = accuracy(preds, labels)

        losses.append(loss.item())  # 记录当前batch的损失值
        top1_acc.append(acc1)  # 记录当前batch的Top-1精度

        # 计算梯度并执行SGD优化步骤
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 在处理最后一个小批次后，确保我们在下一个epoch开始时有一个干净的状态
        if i % args.print_freq == 0:
            if not args.disable_dp:
                epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(
                    delta=args.delta,
                    alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                )
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc):.6f} "
                    f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
                )
            else:
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc):.6f} "
                )

    train_duration = datetime.now() - start_time  # 计算训练持续时间
    return train_duration  # 返回训练持续时间


def test(args, model, test_loader, device):
    """
    在测试集上评估模型。

    Args:
        args (argparse.Namespace): 命令行参数。
        model (torch.nn.Module): 待评估的模型。
        test_loader (torch.utils.data.DataLoader): 测试数据加载器。
        device (torch.device): 评估设备。
    """
    model.eval()  # 设置模型为评估模式
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    losses = []  # 存储每个batch的损失
    top1_acc = []  # 存储每个batch的Top-1精度
    # 使用 torch.no_grad() 上下文管理器，确保在此期间不进行梯度计算，适用于推断阶段
    with torch.no_grad():
        # 遍历测试数据加载器中的图像和目标
        for images, target in tqdm(test_loader):
            # 将图像和目标移到指定的计算设备上（如 GPU）
            images = images.to(device)
            target = target.to(device)

            # 使用模型进行推断，计算输出
            output = model(images)
            # 计算损失，使用预定义的损失函数 criterion
            loss = criterion(output, target)
            # 将模型输出转换为 numpy 数组，并计算每个样本的预测类别
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            # 将目标标签转换为 numpy 数组
            labels = target.detach().cpu().numpy()
            # 计算预测的准确率
            acc1 = accuracy(preds, labels)

            # 将当前批次的损失值添加到损失列表中
            losses.append(loss.item())
            # 将当前批次的 top-1 准确率添加到准确率列表中
            top1_acc.append(acc1)

    # 计算所有批次的平均 top-1 准确率
    top1_avg = np.mean(top1_acc)

    # 打印测试集上的平均损失和平均 top-1 准确率
    print(f"\tTest set:" f"Loss: {np.mean(losses):.6f} " f"Acc@1: {top1_avg :.6f} ")
    # 返回所有批次的平均 top-1 准确率作为函数结果
    return np.mean(top1_acc)
# 定义程序的主函数
def main():
    # 解析命令行参数
    args = parse_args()

    # 如果启用调试模式，则设置日志级别为DEBUG
    if args.debug >= 1:
        logger.setLevel(level=logging.DEBUG)

    # 获取设备信息
    device = args.device

    # 如果启用安全随机数生成器选项
    if args.secure_rng:
        try:
            import torchcsprng as prng
        except ImportError as e:
            # 抛出导入错误并提供安装torchcsprng包的详细说明链接
            msg = (
                "To use secure RNG, you must install the torchcsprng package! "
                "Check out the instructions here: https://github.com/pytorch/csprng#installation"
            )
            raise ImportError(msg) from e

        # 创建基于/dev/urandom的安全随机数生成器
        generator = prng.create_random_device_generator("/dev/urandom")

    else:
        generator = None

    # 图像增强操作列表
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    # 图像归一化操作列表
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    # 根据命令行参数设置训练数据集的变换操作
    train_transform = transforms.Compose(
        augmentations + normalize if args.disable_dp else normalize
    )

    # 根据命令行参数设置测试数据集的变换操作
    test_transform = transforms.Compose(normalize)

    # 创建训练数据集对象CIFAR10
    train_dataset = CIFAR10(
        root=args.data_root, train=True, download=True, transform=train_transform
    )

    # 创建训练数据集的数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(args.sample_rate * len(train_dataset)),
        generator=generator,
        num_workers=args.workers,
        pin_memory=True,
    )

    # 创建测试数据集对象CIFAR10
    test_dataset = CIFAR10(
        root=args.data_root, train=False, download=True, transform=test_transform
    )

    # 创建测试数据集的数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=args.workers,
    )

    # 初始化最佳准确率为0
    best_acc1 = 0

    # 根据命令行参数创建指定架构的模型
    model = models.__dict__[args.architecture](
        pretrained=False, norm_layer=(lambda c: nn.GroupNorm(args.gn_groups, c))
    )
    # 将模型移动到指定的设备上
    model = model.to(device)

    # 根据命令行参数选择优化器
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
        # 如果未识别的优化器，则抛出NotImplementedError异常
        raise NotImplementedError("Optimizer not recognized. Please check spelling")

    # 隐私引擎初始化为None
    privacy_engine = None
    # 如果没有禁用差分隐私
    if not args.disable_dp:
        # 如果设置了按层剪裁
        if args.clip_per_layer:
            # 计算模型中需要计算梯度的层数
            n_layers = len(
                [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            )
            # 计算每层的最大梯度范数，确保总的梯度范数不超过每个样本的最大梯度范数
            max_grad_norm = [
                args.max_per_sample_grad_norm / np.sqrt(n_layers)
            ] * n_layers
        else:
            # 如果按样本剪裁，则所有参数共享一个最大梯度范数
            max_grad_norm = args.max_per_sample_grad_norm

        # 创建差分隐私引擎实例
        privacy_engine = PrivacyEngine(
            secure_mode=args.secure_rng,
        )
        # 根据剪裁方式设置剪裁策略
        clipping = "per_layer" if args.clip_per_layer else "flat"
        # 将模型和优化器设置为私有模式
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.sigma,
            max_grad_norm=max_grad_norm,
            clipping=clipping,
        )

    # 存储一些日志信息
    accuracy_per_epoch = []
    time_per_epoch = []

    # 循环执行每个 epoch
    for epoch in range(args.start_epoch, args.epochs + 1):
        # 如果使用余弦学习率调度
        if args.lr_schedule == "cos":
            # 计算当前 epoch 的学习率
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / (args.epochs + 1)))
            # 更新优化器中的学习率
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # 执行训练，并记录训练时长
        train_duration = train(
            args, model, train_loader, optimizer, privacy_engine, epoch, device
        )
        # 执行测试，并获取测试集上的 top1 准确率
        top1_acc = test(args, model, test_loader, device)

        # 记录是否达到了最佳的 top1 准确率
        is_best = top1_acc > best_acc1
        # 更新记录的最佳 top1 准确率
        best_acc1 = max(top1_acc, best_acc1)

        # 记录本 epoch 的训练时长和 top1 准确率
        time_per_epoch.append(train_duration)
        accuracy_per_epoch.append(float(top1_acc))

        # 保存 checkpoint
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

    # 计算每个 epoch 的训练时长（秒）
    time_per_epoch_seconds = [t.total_seconds() for t in time_per_epoch]
    # 计算平均每个 epoch 的训练时长
    avg_time_per_epoch = sum(time_per_epoch_seconds) / len(time_per_epoch_seconds)
    # 构建记录的指标信息
    metrics = {
        "accuracy": best_acc1,
        "accuracy_per_epoch": accuracy_per_epoch,
        "avg_time_per_epoch_str": str(timedelta(seconds=int(avg_time_per_epoch))),
        "time_per_epoch": time_per_epoch_seconds,
    }

    # 输出一些信息到日志中
    logger.info(
        "\nNote:\n- 'total_time' includes the data loading time, training time and testing time.\n- 'time_per_epoch' measures the training time only.\n"
    )
    # 记录指标信息到日志中
    logger.info(metrics)
# 定义一个函数用于解析命令行参数
def parse_args():
    # 创建一个参数解析器对象，并设置描述信息
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")
    
    # 添加一个命令行参数，用于指定数据加载的工作进程数，默认为2
    parser.add_argument(
        "-j",
        "--workers",
        default=2,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 2)",
    )
    
    # 添加一个命令行参数，用于指定总共运行的 epochs 数，默认为90
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    
    # 添加一个命令行参数，用于指定开始的 epoch 数，默认为1，对重启训练时特别有用
    parser.add_argument(
        "--start-epoch",
        default=1,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    
    # 添加一个命令行参数，用于指定测试数据集的 mini-batch 大小，默认为256
    parser.add_argument(
        "-b",
        "--batch-size-test",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size for test dataset (default: 256)",
    )
    
    # 添加一个命令行参数，用于指定构建批次时的样本采样率，默认为0.005
    parser.add_argument(
        "--sample-rate",
        default=0.005,
        type=float,
        metavar="SR",
        help="sample rate used for batch construction (default: 0.005)",
    )
    
    # 添加一个命令行参数，用于指定初始学习率，默认为0.1
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    
    # 添加一个命令行参数，用于指定 SGD 的动量，默认为0.9
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="SGD momentum"
    )
    
    # 添加一个命令行参数，用于指定 SGD 的权重衰减，默认为0
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0,
        type=float,
        metavar="W",
        help="SGD weight decay",
        dest="weight_decay",
    )
    
    # 添加一个命令行参数，用于指定打印频率，默认为10
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    
    # 添加一个命令行参数，用于指定恢复训练的最新检查点的路径，默认为空
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    
    # 添加一个命令行参数，用于在验证集上评估模型性能
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    
    # 添加一个命令行参数，用于初始化训练的随机种子，默认为 None
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    
    # 添加一个命令行参数，用于指定噪声乘子，默认为 1.5
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.5,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    
    # 添加一个命令行参数，用于指定每个样本梯度的最大范数，默认为 10.0
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=10.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    
    # 添加一个命令行参数，用于禁用差分隐私训练，仅使用普通的 SGD 训练
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    
    # 添加`
    # 添加命令行参数 `--secure-rng`，用于启用安全随机数生成器，提供可信的隐私保证，但会降低性能
    # 如果未启用安全随机数生成器，Opacus将发出警告，建议在生产环境中启用
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost. Opacus will emit a warning if secure rng is off, indicating that for production use it's recommender to turn it on.",
    )
    
    # 添加命令行参数 `--delta`，设置目标 delta 值，默认为 1e-5
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    
    # 添加命令行参数 `--checkpoint-file`，指定保存检查点文件的路径，默认为 "checkpoint"
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="checkpoint",
        help="path to save check points",
    )
    
    # 添加命令行参数 `--data-root`，指定 CIFAR10 数据集的存储路径，默认为 "../cifar10"
    parser.add_argument(
        "--data-root",
        type=str,
        default="../cifar10",
        help="Where CIFAR10 is/will be stored",
    )
    
    # 添加命令行参数 `--log-dir`，指定 Tensorboard 日志存储路径，默认为 "/tmp/stat/tensorboard"
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/tmp/stat/tensorboard",
        help="Where Tensorboard log will be stored",
    )
    
    # 添加命令行参数 `--optim`，选择要使用的优化器，默认为 "SGD"
    parser.add_argument(
        "--optim",
        type=str,
        default="SGD",
        help="Optimizer to use (Adam, RMSprop, SGD)",
    )
    
    # 添加命令行参数 `--lr-schedule`，选择学习率调度方式，默认为 "cos"
    parser.add_argument(
        "--lr-schedule",
        type=str,
        choices=["constant", "cos"],
        default="cos",
    )
    
    # 添加命令行参数 `--device`，指定代码运行的设备，默认为 "cuda"
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device on which to run the code.",
    )
    
    # 添加命令行参数 `--architecture`，选择要使用的 torchvision 模型，默认为 "resnet18"
    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet18",
        help="model from torchvision to run",
    )
    
    # 添加命令行参数 `--gn-groups`，设置 GroupNorm 中的组数，默认为 8
    parser.add_argument(
        "--gn-groups",
        type=int,
        default=8,
        help="Number of groups in GroupNorm",
    )
    
    # 添加命令行参数 `--clip-per-layer`，如果设置，则使用每层静态剪裁，每层使用相同的剪裁阈值，对于 DDP 必需。默认为 `False`，使用平坦剪裁。
    parser.add_argument(
        "--clip-per-layer",
        "--clip_per_layer",
        action="store_true",
        default=False,
        help="Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.",
    )
    
    # 添加命令行参数 `--debug`，设置调试级别，默认为 0
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="debug level (default: 0)",
    )
    
    # 返回解析后的命令行参数对象
    return parser.parse_args()
if __name__ == "__main__":
    # 检查当前模块是否作为主程序运行
    main()
```