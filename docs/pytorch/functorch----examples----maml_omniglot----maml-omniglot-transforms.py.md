# `.\pytorch\functorch\examples\maml_omniglot\maml-omniglot-transforms.py`

```
# 导入必要的模块和库
import argparse  # 用于解析命令行参数的库
import functools  # 用于部分应用（partial application）函数的库
import time  # 用于处理时间的库

import matplotlib as mpl  # matplotlib 库的主要接口
import matplotlib.pyplot as plt  # matplotlib 的绘图模块

import numpy as np  # 用于数值计算的库

import pandas as pd  # 提供数据分析功能的库
from support.omniglot_loaders import OmniglotNShot  # 导入自定义的 Omniglot 数据加载器

import torch  # PyTorch 深度学习框架
import torch.nn.functional as F  # PyTorch 中的函数操作模块
import torch.optim as optim  # PyTorch 中的优化器模块
from torch import nn  # PyTorch 中的神经网络模块
from torch.func import functional_call, grad, vmap  # 导入不明确的 PyTorch 函数

mpl.use("Agg")  # 设置 matplotlib 使用 Agg 后端来保存图像而非显示
plt.style.use("bmh")  # 设置 matplotlib 绘图风格为 bmh 风格


def main():
    # 解析命令行参数
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n-way", "--n_way", type=int, help="n way", default=5)
    argparser.add_argument(
        "--k-spt", "--k_spt", type=int, help="k shot for support set", default=5
    )
    argparser.add_argument(
        "--k-qry", "--k_qry", type=int, help="k shot for query set", default=15
    )
    argparser.add_argument("--device", type=str, help="device", default="cuda")
    argparser.add_argument(
        "--task-num",
        "--task_num",
        type=int,
        help="meta batch size, namely task num",
        default=32,
    )
    argparser.add_argument("--seed", type=int, help="random seed", default=1)
    args = argparser.parse_args()

    # 设置随机种子以便于可复现性
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # 设置设备
    device = args.device

    # 初始化 Omniglot 数据加载器
    db = OmniglotNShot(
        "/tmp/omniglot-data",
        batchsz=args.task_num,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry,
        imgsz=28,
        device=device,
    )

    # 创建一个普通的 PyTorch 神经网络
    inplace_relu = True
    # 创建一个神经网络模型，使用序列容器 `nn.Sequential` 来定义网络的结构
    net = nn.Sequential(
        # 第一个卷积层，输入通道数为1，输出通道数为64，卷积核大小为3x3
        nn.Conv2d(1, 64, 3),
        # 批量归一化层，对输入进行归一化处理，affine=True 表示使用可学习的参数，track_running_stats=False 表示不追踪统计信息
        nn.BatchNorm2d(64, affine=True, track_running_stats=False),
        # ReLU 激活函数，inplace=inplace_relu 表示在原地进行操作以节省内存
        nn.ReLU(inplace=inplace_relu),
        # 最大池化层，窗口大小为2x2，步长为2
        nn.MaxPool2d(2, 2),
        # 第二个卷积层，输入通道数为64，输出通道数为64，卷积核大小为3x3
        nn.Conv2d(64, 64, 3),
        # 批量归一化层，对输入进行归一化处理
        nn.BatchNorm2d(64, affine=True, track_running_stats=False),
        # ReLU 激活函数
        nn.ReLU(inplace=inplace_relu),
        # 最大池化层
        nn.MaxPool2d(2, 2),
        # 第三个卷积层，输入通道数为64，输出通道数为64，卷积核大小为3x3
        nn.Conv2d(64, 64, 3),
        # 批量归一化层，对输入进行归一化处理
        nn.BatchNorm2d(64, affine=True, track_running_stats=False),
        # ReLU 激活函数
        nn.ReLU(inplace=inplace_relu),
        # 最大池化层
        nn.MaxPool2d(2, 2),
        # 将输入展平为一维向量，为全连接层做准备
        nn.Flatten(),
        # 全连接层，输入大小为64，输出大小为 args.n_way，用于分类
        nn.Linear(64, args.n_way),
    ).to(device)
    
    # 设置模型为训练模式
    net.train()
    
    # 使用 Adam 优化器来(meta-)优化模型的初始参数
    meta_opt = optim.Adam(net.parameters(), lr=1e-3)
    
    # 日志列表，用于记录训练过程中的指标变化
    log = []
    
    # 循环训练模型，总共训练100个 epoch
    for epoch in range(100):
        # 在训练集上训练模型，更新参数
        train(db, net, device, meta_opt, epoch, log)
        # 在测试集上测试模型，记录指标
        test(db, net, device, epoch, log)
        # 绘制训练过程中的指标变化图
        plot(log)
# 训练一个模型，使用支持集来进行 n_inner_iter 次迭代，并返回在查询集上的损失
def loss_for_task(net, n_inner_iter, x_spt, y_spt, x_qry, y_qry):
    # 获取网络参数的字典
    params = dict(net.named_parameters())
    # 获取网络缓冲区的字典
    buffers = dict(net.named_buffers())
    # 查询集大小
    querysz = x_qry.size(0)

    # 定义计算损失的函数，接受新的参数和缓冲区
    def compute_loss(new_params, buffers, x, y):
        # 使用新的参数和缓冲区计算 logits
        logits = functional_call(net, (new_params, buffers), x)
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, y)
        return loss

    # 初始化新的参数为当前网络参数
    new_params = params
    # 进行 n_inner_iter 次内部迭代
    for _ in range(n_inner_iter):
        # 计算当前参数下的梯度
        grads = grad(compute_loss)(new_params, buffers, x_spt, y_spt)
        # 使用梯度更新参数
        new_params = {k: new_params[k] - g * 1e-1 for k, g in grads.items()}

    # 使用调整后的参数计算查询集上的 logits
    qry_logits = functional_call(net, (new_params, buffers), x_qry)
    # 计算查询集上的交叉熵损失
    qry_loss = F.cross_entropy(qry_logits, y_qry)
    # 计算查询集上的准确率
    qry_acc = (qry_logits.argmax(dim=1) == y_qry).sum() / querysz

    # 返回查询集上的损失和准确率
    return qry_loss, qry_acc


# 训练函数，用于元学习的训练过程
def train(db, net, device, meta_opt, epoch, log):
    # 获取网络参数的字典
    params = dict(net.named_parameters())
    # 获取网络缓冲区的字典
    buffers = dict(net.named_buffers())
    # 计算每个 epoch 中的训练迭代次数
    n_train_iter = db.x_train.shape[0] // db.batchsz

    # 遍历每个训练批次
    for batch_idx in range(n_train_iter):
        start_time = time.time()
        # 从数据库中获取支持集和查询集的图像和标签
        x_spt, y_spt, x_qry, y_qry = db.next()

        # 获取任务数量、支持集大小和图像维度信息
        task_num, setsz, c_, h, w = x_spt.size()

        # 设置内部迭代次数为 5
        n_inner_iter = 5
        # 清除元优化器的梯度
        meta_opt.zero_grad()

        # 使用偏函数创建计算任务损失的函数
        compute_loss_for_task = functools.partial(loss_for_task, net, n_inner_iter)
        # 使用 vmap 并行计算每个任务的损失和准确率
        qry_losses, qry_accs = vmap(compute_loss_for_task)(x_spt, y_spt, x_qry, y_qry)

        # 计算 MAML 的总损失，通过对返回的损失求和得到
        qry_losses.sum().backward()

        # 执行元优化步骤
        meta_opt.step()
        # 分离损失，并计算平均任务的损失
        qry_losses = qry_losses.detach().sum() / task_num
        # 计算平均任务的准确率（以百分比形式）
        qry_accs = 100.0 * qry_accs.sum() / task_num
        # 计算当前 epoch 中的迭代次数
        i = epoch + float(batch_idx) / n_train_iter
        # 计算当前迭代的时间
        iter_time = time.time() - start_time
        # 每 4 次迭代输出一次训练损失、准确率和时间信息
        if batch_idx % 4 == 0:
            print(
                f"[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}"
            )

        # 将当前训练结果记录到日志中
        log.append(
            {
                "epoch": i,
                "loss": qry_losses,
                "acc": qry_accs,
                "mode": "train",
                "time": time.time(),
            }
        )


# 测试函数，用于评估模型在测试集上的性能
def test(db, net, device, epoch, log):
    # 获取网络参数的字典
    params = dict(net.named_parameters())
    # 获取网络缓冲区的字典
    buffers = dict(net.named_buffers())

    # 在测试过程中，不对模型进行微调以保持简单性
    # 大多数使用 MAML 的研究论文在测试阶段会进行额外的微调，
    # 如果需要进行研究，应考虑添加这一步骤。
    # 计算测试集迭代次数
    n_test_iter = db.x_test.shape[0] // db.batchsz

    # 初始化存储查询损失和准确率的列表
    qry_losses = []
    qry_accs = []

    # 遍历测试集上的每个批次
    for batch_idx in range(n_test_iter):
        # 从数据集对象中获取测试集的支持集和查询集
        x_spt, y_spt, x_qry, y_qry = db.next("test")
        # 获取任务数量、支持集大小、通道数、高度和宽度
        task_num, setsz, c_, h, w = x_spt.size()

        # 定义内部迭代次数
        n_inner_iter = 5

        # 遍历当前批次中的每个任务
        for i in range(task_num):
            # 将参数初始化为当前模型参数
            new_params = params
            # 在当前任务上执行指定次数的内部迭代
            for _ in range(n_inner_iter):
                # 计算支持集数据的预测输出
                spt_logits = functional_call(net, (new_params, buffers), x_spt[i])
                # 计算支持集数据的交叉熵损失
                spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                # 计算参数的梯度
                grads = torch.autograd.grad(spt_loss, new_params.values())
                # 更新参数
                new_params = {
                    k: new_params[k] - g * 1e-1 for k, g in zip(new_params, grads)
                }

            # 使用更新后的参数计算查询集数据的预测输出
            qry_logits = functional_call(net, (new_params, buffers), x_qry[i]).detach()
            # 计算查询集数据的交叉熵损失
            qry_loss = F.cross_entropy(qry_logits, y_qry[i], reduction="none")
            # 将查询损失添加到损失列表中
            qry_losses.append(qry_loss.detach())
            # 计算查询集数据的准确率并添加到准确率列表中
            qry_accs.append((qry_logits.argmax(dim=1) == y_qry[i]).detach())

    # 计算所有查询损失的平均值
    qry_losses = torch.cat(qry_losses).mean().item()
    # 计算所有查询准确率的平均值并转换为百分比
    qry_accs = 100.0 * torch.cat(qry_accs).float().mean().item()
    # 打印当前测试的损失和准确率
    print(f"[Epoch {epoch+1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}")
    # 将当前测试结果记录到日志列表中
    log.append(
        {
            "epoch": epoch + 1,
            "loss": qry_losses,
            "acc": qry_accs,
            "mode": "test",
            "time": time.time(),
        }
    )
# 定义一个绘图函数，用于绘制训练和测试准确率随 epoch 变化的曲线图
def plot(log):
    # 通常情况下，应该将绘图代码从训练脚本中分离出来，但这里为了简洁起见，我们在这里进行绘制。
    
    # 使用日志数据创建一个 Pandas 数据框
    df = pd.DataFrame(log)

    # 创建一个新的图形和坐标轴
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 从数据框中选择训练模式下的数据
    train_df = df[df["mode"] == "train"]
    # 从数据框中选择测试模式下的数据
    test_df = df[df["mode"] == "test"]
    
    # 绘制训练准确率随 epoch 的变化曲线
    ax.plot(train_df["epoch"], train_df["acc"], label="Train")
    # 绘制测试准确率随 epoch 的变化曲线
    ax.plot(test_df["epoch"], test_df["acc"], label="Test")
    
    # 设置 x 轴标签为 "Epoch"
    ax.set_xlabel("Epoch")
    # 设置 y 轴标签为 "Accuracy"
    ax.set_ylabel("Accuracy")
    # 设置 y 轴的数值范围在 70 到 100 之间
    ax.set_ylim(70, 100)
    
    # 添加图例，分为两列，位置在右下角
    fig.legend(ncol=2, loc="lower right")
    # 调整图形布局，使得子图适应图形区域
    fig.tight_layout()
    
    # 定义保存文件名为 "maml-accs.png"
    fname = "maml-accs.png"
    # 打印保存信息，指示正在将准确率图保存到文件
    print(f"--- Plotting accuracy to {fname}")
    # 将图形保存为 PNG 格式的文件
    fig.savefig(fname)
    # 关闭图形对象，释放资源
    plt.close(fig)

# 如果运行时当前脚本是主程序，则执行 main 函数
if __name__ == "__main__":
    main()
```