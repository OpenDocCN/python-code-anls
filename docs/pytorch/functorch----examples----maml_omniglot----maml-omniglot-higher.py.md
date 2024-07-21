# `.\pytorch\functorch\examples\maml_omniglot\maml-omniglot-higher.py`

```
    # 导入必要的库和模块
    import argparse  # 用于解析命令行参数
    import time  # 用于时间相关操作

    import higher  # 用于元学习（MAML）的扩展支持
    import matplotlib as mpl  # 用于绘图
    import matplotlib.pyplot as plt  # 用于绘图
    import numpy as np  # 用于数值计算

    import pandas as pd  # 用于数据处理
    from support.omniglot_loaders import OmniglotNShot  # 导入自定义的Omniglot数据加载器

    import torch  # 导入PyTorch深度学习框架
    import torch.nn.functional as F  # 导入PyTorch中的函数接口
    import torch.optim as optim  # 导入PyTorch中的优化算法
    from torch import nn  # 导入PyTorch中的神经网络模块

    mpl.use("Agg")  # 设置matplotlib使用的后端为Agg，用于无显示环境下的绘图
    plt.style.use("bmh")  # 设置绘图风格为bmh

    def main():
        # 创建参数解析器
        argparser = argparse.ArgumentParser()
        
        # 添加命令行参数选项
        argparser.add_argument("--n-way", "--n_way", type=int, help="n way", default=5)  # n way分类任务中的类别数
        argparser.add_argument("--k-spt", "--k_spt", type=int, help="k shot for support set", default=5)  # 支持集中每类的样本数
        argparser.add_argument("--k-qry", "--k_qry", type=int, help="k shot for query set", default=15)  # 查询集中每类的样本数
        argparser.add_argument("--device", type=str, help="device", default="cuda")  # 指定设备（GPU或CPU）
        argparser.add_argument("--task-num", "--task_num", type=int, help="meta batch size, namely task num", default=32)  # 元学习的任务数（元批大小）
        argparser.add_argument("--seed", type=int, help="random seed", default=1)  # 随机种子
        args = argparser.parse_args()

        # 设置随机种子以保证实验的可重复性
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

        # 设置设备
        device = args.device
        
        # 初始化Omniglot数据加载器
        db = OmniglotNShot(
            "/tmp/omniglot-data",
            batchsz=args.task_num,
            n_way=args.n_way,
            k_shot=args.k_spt,
            k_query=args.k_qry,
            imgsz=28,
            device=device,
        )

        # 创建一个基本的PyTorch神经网络模型，稍后将被higher自动修补
        # 在使用higher之前，模型不能像这样简单地创建，需要手动更新和复制参数
    # 定义一个神经网络模型，使用 nn.Sequential 构建网络层序列
    net = nn.Sequential(
        # 第一个卷积层，输入通道数为1，输出通道数为64，卷积核大小为3x3
        nn.Conv2d(1, 64, 3),
        # 对第一卷积层的输出进行批量归一化，momentum设置为1，affine为True表示使用可学习的仿射变换
        nn.BatchNorm2d(64, momentum=1, affine=True),
        # ReLU 激活函数，inplace=True表示直接覆盖原始输入，节省内存
        nn.ReLU(inplace=True),
        # 最大池化层，池化窗口大小为2x2，步幅为2
        nn.MaxPool2d(2, 2),
        # 第二个卷积层，输入通道数为64，输出通道数为64，卷积核大小为3x3
        nn.Conv2d(64, 64, 3),
        # 对第二卷积层的输出进行批量归一化，momentum设置为1，affine为True表示使用可学习的仿射变换
        nn.BatchNorm2d(64, momentum=1, affine=True),
        # ReLU 激活函数，inplace=True表示直接覆盖原始输入，节省内存
        nn.ReLU(inplace=True),
        # 最大池化层，池化窗口大小为2x2，步幅为2
        nn.MaxPool2d(2, 2),
        # 第三个卷积层，输入通道数为64，输出通道数为64，卷积核大小为3x3
        nn.Conv2d(64, 64, 3),
        # 对第三卷积层的输出进行批量归一化，momentum设置为1，affine为True表示使用可学习的仿射变换
        nn.BatchNorm2d(64, momentum=1, affine=True),
        # ReLU 激活函数，inplace=True表示直接覆盖原始输入，节省内存
        nn.ReLU(inplace=True),
        # 最大池化层，池化窗口大小为2x2，步幅为2
        nn.MaxPool2d(2, 2),
        # 将最终的二维特征图展平为一维向量
        Flatten(),
        # 全连接层，输入特征维度为64，输出特征维度为args.n_way，用于分类任务
        nn.Linear(64, args.n_way),
    ).to(device)
    
    # 使用 Adam 优化器来(meta-)优化神经网络模型的参数，学习率设置为1e-3
    meta_opt = optim.Adam(net.parameters(), lr=1e-3)
    
    # 初始化一个空列表，用于记录每个epoch的训练和测试结果
    log = []
    
    # 循环训练模型，总共训练100个epoch
    for epoch in range(100):
        # 调用 train 函数进行模型训练，传入数据集 db、神经网络模型 net、设备 device、优化器 meta_opt、当前 epoch 数、日志列表 log
        train(db, net, device, meta_opt, epoch, log)
        # 调用 test 函数进行模型测试，传入数据集 db、神经网络模型 net、设备 device、当前 epoch 数、日志列表 log
        test(db, net, device, epoch, log)
        # 调用 plot 函数绘制训练过程中的日志变化图，传入日志列表 log
        plot(log)
def train(db, net, device, meta_opt, epoch, log):
    # 将神经网络设置为训练模式，启用 dropout 等训练相关操作
    net.train()
    # 计算每个 epoch 中训练数据集可分成的 batch 数量
    n_train_iter = db.x_train.shape[0] // db.batchsz

    # 循环处理每个 batch
    for batch_idx in range(n_train_iter):
        start_time = time.time()
        # 从数据集中获取一个 batch 的支持集和查询集的图像和标签数据
        x_spt, y_spt, x_qry, y_qry = db.next()

        # 获取支持集的任务数量，支持集大小，通道数，高度和宽度
        task_num, setsz, c_, h, w = x_spt.size()
        # 查询集大小
        querysz = x_qry.size(1)

        # 初始化内部优化器，用于在支持集上调整参数
        n_inner_iter = 5
        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        qry_losses = []  # 存储查询集每个任务的损失
        qry_accs = []    # 存储查询集每个任务的准确率
        meta_opt.zero_grad()  # 清空元优化器的梯度缓存
        # 遍历每个任务
        for i in range(task_num):
            # 使用 higher 库创建内部循环上下文，实现在支持集上优化模型参数
            with higher.innerloop_ctx(net, inner_opt, copy_initial_weights=False) as (
                fnet,
                diffopt,
            ):
                # 在支持集上进行多次梯度更新，优化支持集的对数损失
                for _ in range(n_inner_iter):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                # 使用优化后的模型计算查询集的对数损失和准确率
                qry_logits = fnet(x_qry[i])
                qry_loss = F.cross_entropy(qry_logits, y_qry[i])
                qry_losses.append(qry_loss.detach())  # 将损失值添加到列表中
                qry_acc = (qry_logits.argmax(dim=1) == y_qry[i]).sum().item() / querysz
                qry_accs.append(qry_acc)  # 将准确率添加到列表中

                # 打印缓冲区的形状，用于调试目的
                # print([b.shape for b in fnet[1].buffers()])

                # 反向传播查询集损失，更新元参数
                qry_loss.backward()

        meta_opt.step()  # 更新元参数
        qry_losses = sum(qry_losses) / task_num  # 计算平均查询集损失
        qry_accs = 100.0 * sum(qry_accs) / task_num  # 计算平均查询集准确率
        i = epoch + float(batch_idx) / n_train_iter  # 计算当前 epoch 和 batch 的组合数
        iter_time = time.time() - start_time  # 计算当前 batch 的运行时间
        if batch_idx % 4 == 0:
            # 打印当前 epoch 中的训练损失、准确率和时间
            print(
                f"[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}"
            )

        log.append(
            {
                "epoch": i,
                "loss": qry_losses,
                "acc": qry_accs,
                "mode": "train",
                "time": time.time(),  # 记录当前时间戳
            }
        )
    # 在我们的测试过程中，重要的是不对模型进行微调，以保持简单。
    # 大多数使用MAML进行此任务的研究论文在这里进行额外的微调阶段，
    # 如果您将此代码用于研究，应该添加这一步骤。
    net.train()  # 设置模型为训练模式，即启用dropout等训练相关操作

    # 计算测试集上的迭代次数
    n_test_iter = db.x_test.shape[0] // db.batchsz

    qry_losses = []  # 存储每个查询样本的损失
    qry_accs = []    # 存储每个查询样本的准确率

    # 对于每个测试集上的任务
    for _ in range(n_test_iter):
        # 获取测试集上的支持集和查询集
        x_spt, y_spt, x_qry, y_qry = db.next("test")

        # 获取任务数量、支持集大小、通道数、高度、宽度等信息
        task_num, setsz, c_, h, w = x_spt.size()

        # 设置内部迭代次数和内部优化器
        n_inner_iter = 5
        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        # 对于每个任务
        for i in range(task_num):
            # 使用higher库创建内部环境，跟踪高阶梯度
            with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (
                fnet,
                diffopt,
            ):
                # 在模型参数上进行梯度步骤，优化支持集的似然度
                # 这会使模型的元参数适应当前任务
                for _ in range(n_inner_iter):
                    spt_logits = fnet(x_spt[i])  # 获取支持集的预测结果
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i])  # 计算支持集的交叉熵损失
                    diffopt.step(spt_loss)  # 在内部优化器上执行优化步骤

                # 计算使用当前参数得到的查询集的损失和准确率
                qry_logits = fnet(x_qry[i]).detach()  # 获取查询集的预测结果（分离）
                qry_loss = F.cross_entropy(qry_logits, y_qry[i], reduction="none")  # 计算查询集的交叉熵损失（不进行汇总）
                qry_losses.append(qry_loss.detach())  # 存储查询集损失（分离）
                qry_accs.append((qry_logits.argmax(dim=1) == y_qry[i]).detach())  # 存储查询集准确率（分离）

    # 计算所有查询样本的平均损失和准确率
    qry_losses = torch.cat(qry_losses).mean().item()
    qry_accs = 100.0 * torch.cat(qry_accs).float().mean().item()

    # 打印测试结果的损失和准确率
    print(f"[Epoch {epoch+1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}")

    # 记录测试结果到日志
    log.append(
        {
            "epoch": epoch + 1,
            "loss": qry_losses,
            "acc": qry_accs,
            "mode": "test",
            "time": time.time(),
        }
    )
# 绘制训练和测试准确率曲线的函数
def plot(log):
    # 将日志数据转换为 pandas DataFrame
    df = pd.DataFrame(log)

    # 创建一个新的图形和轴对象，设置图形大小为 6x4
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 从日志中提取训练数据和测试数据
    train_df = df[df["mode"] == "train"]
    test_df = df[df["mode"] == "test"]
    
    # 绘制训练准确率曲线，以 epoch 为 x 轴，acc 为 y 轴
    ax.plot(train_df["epoch"], train_df["acc"], label="Train")
    
    # 绘制测试准确率曲线，以 epoch 为 x 轴，acc 为 y 轴
    ax.plot(test_df["epoch"], test_df["acc"], label="Test")
    
    # 设置 x 轴标签为 Epoch
    ax.set_xlabel("Epoch")
    
    # 设置 y 轴标签为 Accuracy
    ax.set_ylabel("Accuracy")
    
    # 设置 y 轴范围为 70 到 100
    ax.set_ylim(70, 100)
    
    # 添加图例，分成两列，位于右下角
    fig.legend(ncol=2, loc="lower right")
    
    # 调整图形布局
    fig.tight_layout()
    
    # 定义保存图片的文件名
    fname = "maml-accs.png"
    
    # 打印保存图片的信息
    print(f"--- Plotting accuracy to {fname}")
    
    # 保存图形为 PNG 文件
    fig.savefig(fname)
    
    # 关闭图形对象，释放资源
    plt.close(fig)
```