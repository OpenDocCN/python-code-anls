# `.\pytorch\functorch\examples\maml_omniglot\maml-omniglot-ptonly.py`

```py
    argparser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    argparser.add_argument("--n-way", "--n_way", type=int, help="n way", default=5)
    # 添加一个命令行参数，用于设置 n-way 分类任务中的类别数，默认为 5

    argparser.add_argument(
        "--k-spt", "--k_spt", type=int, help="k shot for support set", default=5
    )
    # 添加一个命令行参数，用于设置支持集中的 k-shot 数量，默认为 5

    argparser.add_argument(
        "--k-qry", "--k_qry", type=int, help="k shot for query set", default=15
    )
    # 添加一个命令行参数，用于设置查询集中的 k-shot 数量，默认为 15

    argparser.add_argument("--device", type=str, help="device", default="cuda")
    # 添加一个命令行参数，用于设置设备（如 cuda），默认为 "cuda"

    argparser.add_argument(
        "--task-num",
        "--task_num",
        type=int,
        help="meta batch size, namely task num",
        default=32,
    )
    # 添加一个命令行参数，用于设置元任务批次大小，默认为 32

    argparser.add_argument("--seed", type=int, help="random seed", default=1)
    # 添加一个命令行参数，用于设置随机种子，默认为 1

    torch.manual_seed(args.seed)
    # 设置 PyTorch 的随机种子为命令行参数指定的值

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # 如果 CUDA 可用，设置所有 CUDA 设备的随机种子为命令行参数指定的值

    np.random.seed(args.seed)
    # 设置 NumPy 的随机种子为命令行参数指定的值

    # Set up the Omniglot loader.
    device = args.device
    # 将设备设置为命令行参数指定的设备

    db = OmniglotNShot(
        "/tmp/omniglot-data",
        batchsz=args.task_num,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry,
        imgsz=28,
        device=device,
    )
    # 创建 OmniglotNShot 对象，用于加载 Omniglot 数据集
    # 初始化参数包括数据存储路径、批量大小、分类任务类别数、支持集中的 k-shot 数量、查询集中的 k-shot 数量、图像尺寸和设备

    # Create a vanilla PyTorch neural network that will be
    # automatically monkey-patched by higher later.
    # Before higher, models could *not* be created like this
    # and the parameters needed to be manually updated and copied
    # for the updates.
    # 定义一个神经网络模型，包括多层卷积、批量归一化、ReLU激活函数和最大池化层
    net = nn.Sequential(
        nn.Conv2d(1, 64, 3),                    # 输入通道数1，输出通道数64，卷积核大小3x3
        nn.BatchNorm2d(64, momentum=1, affine=True),  # 批量归一化层，64个通道，使用全局统计量，affine参数为True
        nn.ReLU(inplace=True),                  # 使用inplace方式进行ReLU激活
        nn.MaxPool2d(2, 2),                     # 最大池化层，池化核大小2x2，步长2
        nn.Conv2d(64, 64, 3),                   # 输入通道数64，输出通道数64，卷积核大小3x3
        nn.BatchNorm2d(64, momentum=1, affine=True),  # 批量归一化层，64个通道，使用全局统计量，affine参数为True
        nn.ReLU(inplace=True),                  # 使用inplace方式进行ReLU激活
        nn.MaxPool2d(2, 2),                     # 最大池化层，池化核大小2x2，步长2
        nn.Conv2d(64, 64, 3),                   # 输入通道数64，输出通道数64，卷积核大小3x3
        nn.BatchNorm2d(64, momentum=1, affine=True),  # 批量归一化层，64个通道，使用全局统计量，affine参数为True
        nn.ReLU(inplace=True),                  # 使用inplace方式进行ReLU激活
        nn.MaxPool2d(2, 2),                     # 最大池化层，池化核大小2x2，步长2
        Flatten(),                              # 将多维输入展平为一维
        nn.Linear(64, args.n_way),              # 全连接层，输入大小为64，输出大小为args.n_way
    ).to(device)                                # 将网络模型移动到指定设备（如GPU）
    
    net.train()  # 将网络设置为训练模式
    
    # 使用make_functional_with_buffers函数将网络转换为函数式表示，并获取其参数和缓冲区
    fnet, params, buffers = make_functional_with_buffers(net)
    
    # 使用Adam优化器(meta_opt)来(meta-)优化初始参数，学习率为1e-3
    meta_opt = optim.Adam(params, lr=1e-3)
    
    log = []  # 初始化一个空列表用于记录训练和测试日志
    
    # 循环训练100个epoch
    for epoch in range(100):
        train(db, [params, buffers, fnet], device, meta_opt, epoch, log)  # 执行训练函数，更新参数
        test(db, [params, buffers, fnet], device, epoch, log)  # 执行测试函数，评估模型性能
        plot(log)  # 绘制训练和测试过程中的日志信息图表
# 定义训练函数，用于元学习中的一个训练步骤
def train(db, net, device, meta_opt, epoch, log):
    # 解包网络的参数、缓冲区和模型函数
    params, buffers, fnet = net
    # 计算每个 epoch 中训练迭代的次数
    n_train_iter = db.x_train.shape[0] // db.batchsz

    # 循环执行每个训练迭代
    for batch_idx in range(n_train_iter):
        # 记录当前时间
        start_time = time.time()
        
        # 从数据集中获取一批支持集和查询集的图像及其标签
        x_spt, y_spt, x_qry, y_qry = db.next()

        # 获取支持集的任务数量、集合大小、通道数、高度和宽度
        task_num, setsz, c_, h, w = x_spt.size()
        # 获取查询集的大小
        querysz = x_qry.size(1)

        # TODO: 可能将这部分提取到一个单独的模块中，以避免在 `train` 和 `test` 之间重复？

        # 初始化用于调整参数到支持集的内部优化器
        n_inner_iter = 5
        # inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        # 初始化查询集的损失列表和准确率列表
        qry_losses = []
        qry_accs = []
        # 清空梯度，准备进行元优化器的步骤
        meta_opt.zero_grad()

        # 遍历每个任务
        for i in range(task_num):
            # 通过对模型参数进行梯度步骤来优化支持集的似然概率
            # 这会调整模型的元参数以适应当前任务
            new_params = params
            for _ in range(n_inner_iter):
                # 计算支持集的预测结果和损失
                spt_logits = fnet(new_params, buffers, x_spt[i])
                spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                # 计算梯度
                grads = torch.autograd.grad(spt_loss, new_params, create_graph=True)
                # 更新参数
                new_params = [p - g * 1e-1 for p, g, in zip(new_params, grads)]

            # 使用最终调整后的参数在查询集上计算损失和准确率
            qry_logits = fnet(new_params, buffers, x_qry[i])
            qry_loss = F.cross_entropy(qry_logits, y_qry[i])
            qry_losses.append(qry_loss.detach())
            qry_acc = (qry_logits.argmax(dim=1) == y_qry[i]).sum().item() / querysz
            qry_accs.append(qry_acc)

            # 对查询集的损失进行反向传播，用于更新模型的元参数
            qry_loss.backward()

        # 执行元优化器的步骤
        meta_opt.step()

        # 计算平均查询集的损失和准确率
        qry_losses = sum(qry_losses) / task_num
        qry_accs = 100.0 * sum(qry_accs) / task_num

        # 计算当前 epoch 和训练迭代的总步数
        i = epoch + float(batch_idx) / n_train_iter
        # 计算当前训练迭代的时间
        iter_time = time.time() - start_time

        # 每 4 个训练迭代打印当前的训练损失、准确率和时间
        if batch_idx % 4 == 0:
            print(
                f"[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}"
            )

        # 将当前训练迭代的信息添加到日志中
        log.append(
            {
                "epoch": i,
                "loss": qry_losses,
                "acc": qry_accs,
                "mode": "train",
                "time": time.time(),
            }
        )
    # 解构 net 变量，获取参数、缓冲区和神经网络函数
    [params, buffers, fnet] = net
    # 计算测试集迭代次数
    n_test_iter = db.x_test.shape[0] // db.batchsz

    # 初始化用于存储查询损失和准确率的列表
    qry_losses = []
    qry_accs = []

    # 遍历测试集的每个批次
    for batch_idx in range(n_test_iter):
        # 获取当前测试批次的支持集和查询集数据及标签
        x_spt, y_spt, x_qry, y_qry = db.next("test")
        # 获取支持集数据的任务数量、集合大小、通道数、高度和宽度
        task_num, setsz, c_, h, w = x_spt.size()

        # 每个任务的内部循环迭代次数
        n_inner_iter = 5

        # 对每个任务执行内部循环迭代
        for i in range(task_num):
            # 初始化新的参数副本
            new_params = params
            # 执行内部迭代更新参数
            for _ in range(n_inner_iter):
                # 使用当前参数计算支持集的预测结果
                spt_logits = fnet(new_params, buffers, x_spt[i])
                # 计算支持集的交叉熵损失
                spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                # 计算损失关于参数的梯度
                grads = torch.autograd.grad(spt_loss, new_params)
                # 更新参数
                new_params = [p - g * 1e-1 for p, g in zip(new_params, grads)]

            # 使用更新后的参数计算查询集的预测结果
            qry_logits = fnet(new_params, buffers, x_qry[i]).detach()
            # 计算查询集的交叉熵损失
            qry_loss = F.cross_entropy(qry_logits, y_qry[i], reduction="none")
            # 将查询损失添加到列表中
            qry_losses.append(qry_loss.detach())
            # 计算查询准确率并添加到列表中
            qry_accs.append((qry_logits.argmax(dim=1) == y_qry[i]).detach())

    # 计算所有任务的平均查询损失
    qry_losses = torch.cat(qry_losses).mean().item()
    # 计算所有任务的平均查询准确率百分比
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
# 绘图函数，用于绘制训练和测试的精度曲线
def plot(log):
    # 将日志数据转换为 pandas DataFrame
    df = pd.DataFrame(log)

    # 创建绘图对象和轴
    fig, ax = plt.subplots(figsize=(6, 4))

    # 从 DataFrame 中提取训练数据和测试数据
    train_df = df[df["mode"] == "train"]
    test_df = df[df["mode"] == "test"]

    # 绘制训练精度曲线
    ax.plot(train_df["epoch"], train_df["acc"], label="Train")

    # 绘制测试精度曲线
    ax.plot(test_df["epoch"], test_df["acc"], label="Test")

    # 设置图表的 X 轴标签和 Y 轴标签
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

    # 设置 Y 轴的数值范围
    ax.set_ylim(70, 100)

    # 添加图例到图表中，2 列，位置在右下角
    fig.legend(ncol=2, loc="lower right")

    # 调整图表布局
    fig.tight_layout()

    # 指定保存图片的文件名
    fname = "maml-accs.png"

    # 打印保存信息
    print(f"--- Plotting accuracy to {fname}")

    # 将图表保存为文件
    fig.savefig(fname)

    # 关闭图表对象，释放资源
    plt.close(fig)


# 此类定义了一个自定义的 PyTorch 模型层，用于将输入扁平化为二维张量
# 在以下 PR 合并后不再需要这个类：https://github.com/pytorch/pytorch/pull/22245
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# 主程序入口，执行主函数 main()
if __name__ == "__main__":
    main()
```