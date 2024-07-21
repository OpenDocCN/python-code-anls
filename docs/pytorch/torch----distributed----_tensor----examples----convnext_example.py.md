# `.\pytorch\torch\distributed\_tensor\examples\convnext_example.py`

```
"""
The following example demonstrates how to train a ConvNeXt model
with intermediate activations sharded across mutliple GPUs via DTensor
"""

import os  # 导入操作系统接口模块
import time  # 导入时间模块

import torch  # 导入PyTorch深度学习库
import torch.distributed as dist  # 导入PyTorch分布式训练库
import torch.multiprocessing as mp  # 导入PyTorch多进程库
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch.distributed._tensor import (  # 导入分布式张量相关模块
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    Replicate,
    Shard,
)

WORLD_SIZE = 4  # 定义全局变量，表示分布式训练中的总进程数
ITER_TIME = 20  # 定义全局变量，表示迭代次数

class LayerNorm(nn.Module):
    """
    实现Layer Normalization的模块
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format=torch.contiguous_format):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 初始化归一化的权重参数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # 初始化归一化的偏置参数
        self.eps = eps  # 归一化常数epsilon
        self.data_format = data_format  # 数据格式
        if self.data_format not in [torch.contiguous_format]:
            raise NotImplementedError  # 如果数据格式不支持，抛出未实现错误
        self.normalized_shape = (normalized_shape,)  # 归一化形状

    def forward(self, x):
        u = x.mean(1, keepdim=True)  # 计算均值
        s = (x - u).pow(2).mean(1, keepdim=True)  # 计算方差
        x = (x - u) / torch.sqrt(s + self.eps)  # 标准化
        x = self.weight[:, None, None] * x + self.bias[:, None, None]  # 应用权重和偏置
        return x  # 返回归一化后的张量

class Block(nn.Module):
    """
    ConvNeXt模型的一个基本块
    """
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # 深度卷积，使用7x7的卷积核，分组卷积
        self.norm = LayerNorm(dim, eps=1e-6, data_format=torch.contiguous_format)  # 应用Layer Normalization
        self.pwconv1 = nn.Conv2d(
            dim, 4 * dim, kernel_size=1, stride=1
        )  # 1x1卷积，将维度扩展为4倍
        self.act = nn.GELU()  # 使用GELU作为激活函数
        self.pwconv2 = nn.Conv2d(
            4 * dim, dim, kernel_size=1, stride=1
        )  # 1x1卷积，将维度还原回dim
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )  # 可学习的缩放因子gamma，用于调整块的输出
        self.drop_path = nn.Identity()  # 用于执行可选的dropout路径

    def forward(self, x):
        input_x = x  # 保存输入张量
        x = self.dwconv(x)  # 深度卷积操作

        x = self.norm(x)  # 应用Layer Normalization
        x = self.pwconv1(x)  # 第一个1x1卷积
        x = self.act(x)  # 激活函数
        x = self.pwconv2(x)  # 第二个1x1卷积

        if self.gamma is not None:
            x = self.gamma * self.drop_path(x)  # 使用gamma调整块的输出
        x = input_x + x  # 残差连接
        return x  # 返回块的输出

class DownSampling(nn.Module):
    """
    下采样模块的定义
    """
    # 定义一个类的初始化方法，初始化函数参数和属性
    def __init__(self, dim_in=3, dim_out=2, down_scale=4, norm_first=False):
        super().__init__()  # 调用父类的初始化方法

        self.norm_first = norm_first  # 设定是否先进行归一化的标志位

        # 根据是否先进行归一化选择不同的网络层顺序
        if norm_first:
            # 如果先归一化，则创建 LayerNorm 对象，对输入进行归一化处理
            self.norm = LayerNorm(dim_in, eps=1e-6, data_format=torch.contiguous_format)
            # 然后使用卷积层对归一化后的数据进行卷积操作
            self.conv = nn.Conv2d(
                dim_in, dim_out, kernel_size=down_scale, stride=down_scale
            )
        else:
            # 如果先卷积，则创建卷积层对象，对输入进行卷积操作
            self.conv = nn.Conv2d(
                dim_in, dim_out, kernel_size=down_scale, stride=down_scale
            )
            # 然后对卷积后的数据进行归一化处理
            self.norm = LayerNorm(
                dim_out, eps=1e-6, data_format=torch.contiguous_format
            )

    # 定义类的前向传播方法，用于网络的正向计算
    def forward(self, x):
        if self.norm_first:
            # 如果在初始化时设定了先进行归一化，则先对输入数据进行归一化处理，再进行卷积
            return self.conv(self.norm(x))
        else:
            # 否则先进行卷积操作，再对卷积结果进行归一化处理
            return self.norm(self.conv(x))
@torch.no_grad()
def init_weights(m):
    # 使用 @torch.no_grad() 装饰器，确保在权重初始化时没有梯度计算
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        # 如果是 Conv2d 或者 Linear 模块，初始化权重为全1，偏置为全0
        nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=10,
        depths=[1, 1],  # noqa: B006
        dims=[2, 4],  # noqa: B006
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
    ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        # 创建下采样层，首先是 stem 层
        stem = DownSampling(in_chans, dims[0], 4, norm_first=False)
        self.downsample_layers.append(stem)
        # 根据 dims 列表创建其余的下采样层
        for i in range(len(dims) - 1):
            downsample_layer = DownSampling(dims[i], dims[i + 1], 2, norm_first=True)
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        # 根据 depths 和 dims 创建多个阶段，每个阶段包含若干个 Block
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        # 创建头部分类器
        self.head = nn.Linear(dims[-1], num_classes)
        # 应用权重初始化函数 init_weights
        self.apply(init_weights)

    def forward(self, x):
        # 模型前向传播过程
        for i in range(len(self.stages)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1])
        x = self.head(x)
        return x


def _conv_fn(
    name: str,
    module: nn.Module,
    device_mesh: DeviceMesh,
) -> None:
    # 遍历模块的参数，并分发到设备网格上
    for name, param in module.named_parameters():
        dist_spec = [Replicate()]
        dist_param = torch.nn.Parameter(
            distribute_tensor(param, device_mesh, dist_spec)
        )
        # 注册梯度钩子，确保在梯度计算时重新分发梯度
        dist_param.register_hook(lambda grad: grad.redistribute(placements=dist_spec))
        name = "_".join(name.split("."))
        module.register_parameter(name, dist_param)


def test_tp_convnext_train(rank, world_size):
    # 设置环境变量
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # 初始化分布式进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 定义输入和输出的形状
    in_shape = [7, 3, 512, 1024]
    output_shape = [7, 1000]
    # 将当前进程设备设为 CUDA
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    # 设置每个进程的 GPU 使用全部显存
    torch.cuda.set_per_process_memory_fraction(1.0, device)
    # 创建设备网格
    mesh = DeviceMesh("cuda", torch.arange(world_size))

    # 设置随机种子并创建 ConvNeXt 模型
    torch.manual_seed(12)
    model = ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[256, 512, 1024, 2048],
        drop_path_rate=0.0,
        num_classes=1000,
    ).to(device)
    # 将模型分发到设备网格上，并使用 _conv_fn 函数进行参数分发
    model = distribute_module(model, mesh, _conv_fn, input_fn=None, output_fn=None)

    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 使用 Adam 优化器对模型的参数进行优化，设置学习率为 1e-4，禁用 AMSGrad
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=False)

    # 生成一个形状为 in_shape 的随机张量 x，并将其移动到指定设备上，同时要求计算梯度
    x = torch.randn(*in_shape).to(device).requires_grad_()

    # 生成一个形状为 [output_shape[0]] 的长整型张量 y_target，随机填充在 [0, output_shape[1]) 范围内的整数，并移动到指定设备上
    y_target = (
        torch.empty(output_shape[0], dtype=torch.long)
        .random_(output_shape[1])
        .to(device)
    )

    # 使用 distribute_tensor 函数将张量 x 按照指定策略（这里是 Shard(3)）分发到指定的 mesh 上
    x = distribute_tensor(x, mesh, [Shard(3)])

    # 使用 distribute_tensor 函数将张量 y_target 按照指定策略（这里是 Replicate()）分发到指定的 mesh 上
    y_target = distribute_tensor(y_target, mesh, [Replicate()])

    # 模型预热阶段
    y = model(x)  # 对输入 x 进行模型预测
    loss = criterion(y, y_target)  # 计算预测结果 y 与目标 y_target 的损失
    optimizer.zero_grad()  # 清空优化器当前存储的梯度值
    loss.backward()  # 反向传播，计算梯度
    optimizer.step()  # 执行优化器的一步更新
    torch.cuda.synchronize(device)  # 同步所有 CUDA 设备上的流，确保前面的操作完成

    # 初始化计时器
    forward_time = 0.0
    backward_time = 0.0
    start = time.time()  # 记录开始时间
    for i in range(ITER_TIME):
        t1 = time.time()
        y = model(x)  # 对输入 x 进行模型预测
        torch.cuda.synchronize(device)  # 同步所有 CUDA 设备上的流，确保前面的操作完成
        t2 = time.time()

        loss = criterion(y, y_target)  # 计算预测结果 y 与目标 y_target 的损失
        optimizer.zero_grad()  # 清空优化器当前存储的梯度值

        t3 = time.time()
        loss.backward()  # 反向传播，计算梯度
        torch.cuda.synchronize(device)  # 同步所有 CUDA 设备上的流，确保前面的操作完成
        t4 = time.time()

        optimizer.step()  # 执行优化器的一步更新

        # 更新前向传播和反向传播的时间统计
        forward_time += t2 - t1
        backward_time += t4 - t3

    torch.cuda.synchronize(device)  # 同步所有 CUDA 设备上的流，确保前面的操作完成
    end = time.time()  # 记录结束时间

    max_reserved = torch.cuda.max_memory_reserved(device)  # 获取 CUDA 设备上的最大内存保留量
    max_allocated = torch.cuda.max_memory_allocated(device)  # 获取 CUDA 设备上的最大内存分配量

    # 打印输出性能统计信息
    print(
        f"rank {rank}, {ITER_TIME} iterations, average latency {(end - start)/ITER_TIME*1000:10.2f} ms"
    )
    print(
        f"rank {rank}, forward {forward_time/ITER_TIME*1000:10.2f} ms, backward {backward_time/ITER_TIME*1000:10.2f} ms"
    )
    print(
        f"rank {rank}, max reserved {max_reserved/1024/1024/1024:8.2f} GiB, max allocated {max_allocated/1024/1024/1024:8.2f} GiB"
    )

    # 销毁进程组，释放资源
    dist.destroy_process_group()
# 如果当前脚本作为主程序运行（而不是作为模块导入），则执行以下代码块
if __name__ == "__main__":
    # 使用多进程（mp.spawn）来并行执行 test_tp_convnext_train 函数
    # args=(WORLD_SIZE,) 将 WORLD_SIZE 作为参数传递给 test_tp_convnext_train 函数
    # nprocs=WORLD_SIZE 指定使用的进程数为 WORLD_SIZE
    # join=True 等待所有进程完成后再继续执行后续代码
    mp.spawn(test_tp_convnext_train, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
```