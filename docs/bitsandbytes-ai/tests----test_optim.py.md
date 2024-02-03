# `bitsandbytes\tests\test_optim.py`

```py
# 导入必要的库
import os
from os.path import join
import shutil
import time
import uuid

# 导入测试相关的库
import pytest
import torch

# 导入自定义的库
import bitsandbytes as bnb
import bitsandbytes.functional as F
from tests.helpers import describe_dtype, id_formatter

# 定义常量 k
k = 20

# 自定义函数，用于比较两个张量是否近似相等
def assert_most_approx_close(a, b, rtol=1e-3, atol=1e-3, max_error_count=0):
    # 判断两个张量是否近似相等
    idx = torch.isclose(a, b, rtol=rtol, atol=atol)
    # 统计不相等的元素个数
    error_count = (idx == 0).sum().item()
    # 如果不相等的元素个数超过阈值，则输出错误信息
    if error_count > max_error_count:
        print(f"Too many values not close: assert {error_count} < {max_error_count}")
        # 断言两个张量近似相等
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

# 获取临时目录路径
def get_temp_dir():
    # 生成唯一的临时目录路径
    path = f"/tmp/autoswap/{uuid.uuid4()}"
    # 创建临时目录
    os.makedirs(path, exist_ok=True)
    return path

# 删除指定路径的文件或目录
def rm_path(path):
    shutil.rmtree(path)

# 定义不同优化器的映射关系
str2optimizers = {}
str2optimizers["adam_pytorch"] = (None, torch.optim.Adam, bnb.optim.Adam)
str2optimizers["lion_pytorch"] = (None, Lion, bnb.optim.Lion)
str2optimizers["momentum_pytorch"] = (
    None,
    lambda pxx: torch.optim.SGD(pxx, 0.01, 0.9),
    bnb.optim.Adam,
)
str2optimizers["adam"] = (torch.optim.Adam, bnb.optim.Adam)
str2optimizers["paged_adamw"] = (torch.optim.AdamW, bnb.optim.PagedAdamW)
str2optimizers["paged_adam"] = (torch.optim.Adam, bnb.optim.PagedAdam)
str2optimizers["lion"] = (Lion, bnb.optim.Lion)
str2optimizers["paged_lion"] = (Lion, bnb.optim.PagedLion)
str2optimizers["momentum"] = (
    lambda pxx: torch.optim.SGD(pxx, 0.01, 0.9),
    lambda pxx: bnb.optim.SGD(pxx, 0.01, 0.9, block_wise=False),
)
str2optimizers["rmsprop"] = (
    lambda pxx: torch.optim.RMSprop(pxx, 0.01, 0.9),
    lambda pxx: bnb.optim.RMSprop(pxx, 0.01, 0.9, block_wise=False),
)
str2optimizers["adam8bit"] = (torch.optim.Adam, lambda pxx: bnb.optim.Adam8bit(pxx, block_wise=False))
str2optimizers["lion8bit"] = (Lion, lambda pxx: bnb.optim.Lion8bit(pxx, block_wise=False))
str2optimizers["momentum8bit"] = (
    lambda pxx: torch.optim.SGD(pxx, 0.01, 0.9),
    # 使用 lambda 表达式定义一个函数，参数为 pxx
    # 调用 bnb.optim 模块中的 SGD8bit 函数，传入参数 pxx, 0.01, 0.9, block_wise=False
    lambda pxx: bnb.optim.SGD8bit(pxx, 0.01, 0.9, block_wise=False),
# 定义字符串到优化器的映射字典，包含不同优化器的初始化函数
str2optimizers["rmsprop8bit"] = (
    lambda pxx: torch.optim.RMSprop(pxx, 0.01, 0.9),  # 使用 torch.optim.RMSprop 初始化 RMSprop8bit 优化器
    lambda pxx: bnb.optim.RMSprop8bit(pxx, 0.01, 0.9, block_wise=False),  # 使用 bnb.optim.RMSprop8bit 初始化 RMSprop8bit 优化器
)

# 定义字符串到优化器的映射字典，包含不同优化器的初始化函数
str2optimizers["adam8bit_blockwise"] = (torch.optim.Adam, lambda pxx: bnb.optim.Adam8bit(pxx, block_wise=True))
str2optimizers["paged_adamw8bit_blockwise"] = (torch.optim.AdamW, lambda pxx: bnb.optim.PagedAdamW8bit(pxx, block_wise=True))
str2optimizers["paged_adam8bit_blockwise"] = (torch.optim.Adam, lambda pxx: bnb.optim.PagedAdam8bit(pxx, block_wise=True))
str2optimizers["lion8bit_blockwise"] = (Lion, lambda pxx: bnb.optim.Lion8bit(pxx, block_wise=True))
str2optimizers["paged_lion8bit_blockwise"] = (Lion, lambda pxx: bnb.optim.PagedLion8bit(pxx, block_wise=True))
str2optimizers["momentum8bit_blockwise"] = (
    lambda pxx: torch.optim.SGD(pxx, 0.01, 0.9),  # 使用 torch.optim.SGD 初始化 momentum8bit 优化器
    lambda pxx: bnb.optim.SGD8bit(pxx, 0.01, 0.9, block_wise=True),  # 使用 bnb.optim.SGD8bit 初始化 momentum8bit 优化器
)
str2optimizers["rmsprop8bit_blockwise"] = (
    lambda pxx: torch.optim.RMSprop(pxx, 0.01, 0.9),  # 使用 torch.optim.RMSprop 初始化 rmsprop8bit 优化器
    lambda pxx: bnb.optim.RMSprop8bit(pxx, 0.01, 0.9, block_wise=True),  # 使用 bnb.optim.RMSprop8bit 初始化 rmsprop8bit 优化器
)

# 定义字符串到状态名称的映射字典，包含不同优化器的状态名称
str2statenames = {}
str2statenames["adam"] = [("exp_avg", "state1"), ("exp_avg_sq", "state2")]
str2statenames["paged_adamw"] = [("exp_avg", "state1"), ("exp_avg_sq", "state2")]
str2statenames["paged_adam"] = [("exp_avg", "state1"), ("exp_avg_sq", "state2")]
str2statenames["lion"] = [("exp_avg", "state1")]
str2statenames["paged_lion"] = [("exp_avg", "state1")]
str2statenames["momentum"] = [("momentum_buffer", "state1")]
str2statenames["lamb"] = [("exp_avg", "state1"), ("exp_avg_sq", "state2")]
str2statenames["rmsprop"] = [("square_avg", "state1")]
str2statenames["adam8bit"] = [("exp_avg", "state1", "qmap1", "max1"), ("exp_avg_sq", "state2", "qmap2", "max2")]
str2statenames["lamb8bit"] = [("exp_avg", "state1", "qmap1", "max1"), ("exp_avg_sq", "state2", "qmap2", "max2")]
str2statenames["adam8bit_blockwise"] = [("exp_avg", "state1", "qmap1", "absmax1"), ("exp_avg_sq", "state2", "qmap2", "absmax2")]
# 将字符串键值对应的值设置为包含元组的列表，每个元组包含四个字符串
str2statenames["paged_adam8bit_blockwise"] = [("exp_avg", "state1", "qmap1", "absmax1"), ("exp_avg_sq", "state2", "qmap2", "absmax2")]
str2statenames["paged_adamw8bit_blockwise"] = [("exp_avg", "state1", "qmap1", "absmax1"), ("exp_avg_sq", "state2", "qmap2", "absmax2")]
str2statenames["momentum8bit"] = [("momentum_buffer", "state1", "qmap1", "max1")]
str2statenames["lion8bit"] = [("exp_avg", "state1", "qmap1", "max1")]
str2statenames["momentum8bit_blockwise"] = [("momentum_buffer", "state1", "qmap1", "absmax1")]
str2statenames["rmsprop8bit"] = [("square_avg", "state1", "qmap1", "max1")]
str2statenames["rmsprop8bit_blockwise"] = [("square_avg", "state1", "qmap1", "absmax1")]
str2statenames["lion8bit_blockwise"] = [("exp_avg", "state1", "qmap1", "absmax1")]
str2statenames["paged_lion8bit_blockwise"] = [("exp_avg", "state1", "qmap1", "absmax1")]

# 定义包含字符串的列表
optimizer_names_32bit = ["adam", "momentum", "rmsprop", 'paged_adamw', 'paged_adam', 'lion', 'paged_lion']

# 使用参数化测试，对优化器名称、数据类型、维度1、维度2进行参数化测试
@pytest.mark.parametrize("optim_name", optimizer_names_32bit, ids=id_formatter("opt"))
@pytest.mark.parametrize("gtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
@pytest.mark.parametrize("dim1", [1024], ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", [32, 1024, 4097, 1], ids=id_formatter("dim2"))
def test_optimizer32bit(dim1, dim2, gtype, optim_name):
    # 如果数据类型为 torch.bfloat16 且优化器名称为 'momentum' 或 'rmsprop'，则跳过测试
    if gtype == torch.bfloat16 and optim_name in ['momentum', 'rmsprop']:
        pytest.skip()
    # 如果维度1和维度2均为1，则直接返回
    if dim1 == 1 and dim2 == 1:
        return
    # 在 CUDA 设备上生成随机数据 p1，并复制给 p2
    p1 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1
    p2 = p1.clone()
    # 将 p1 转换为 float 类型
    p1 = p1.float()

    # 根据优化器名称选择对应的 Torch 优化器和 BNB 优化器
    torch_optimizer = str2optimizers[optim_name][0]([p1])
    bnb_optimizer = str2optimizers[optim_name][1]([p2])

    # 根据数据类型设置不同的绝对误差和相对误差
    if gtype == torch.float32:
        atol, rtol = 1e-6, 1e-5
    elif gtype == torch.bfloat16:
        atol, rtol = 1e-3, 1e-2
    else:
        atol, rtol = 1e-4, 1e-3
# 使用 pytest.mark.parametrize 装饰器为 test_global_config 函数添加参数化测试
@pytest.mark.parametrize("dim2", [32, 1024, 4097], ids=id_formatter("dim2"))
# 使用 pytest.mark.parametrize 装饰器为 test_global_config 函数添加参数化测试
@pytest.mark.parametrize("gtype", [torch.float32, torch.float16], ids=describe_dtype)
# 定义 test_global_config 函数，测试全局配置
def test_global_config(dim1, dim2, gtype):
    # 如果 dim1 和 dim2 都为 1，则直接返回
    if dim1 == 1 and dim2 == 1:
        return
    # 生成随机张量 p1, p2, p3
    p1 = torch.randn(dim1, dim2, device="cpu", dtype=gtype) * 0.1
    p2 = torch.randn(dim1, dim2, device="cpu", dtype=gtype) * 0.1
    p3 = torch.randn(dim1, dim2, device="cpu", dtype=gtype) * 0.1
    # 生成掩码 mask
    mask = torch.rand_like(p2) < 0.1
    # 初始化 beta1, beta2, lr, eps
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.001
    eps = 1e-8

    # 初始化全局优化管理器
    bnb.optim.GlobalOptimManager.get_instance().initialize()
    # 覆盖 p3 的优化位数配置
    bnb.optim.GlobalOptimManager.get_instance().override_config(
        p3, "optim_bits", 8
    )
    # 注册参数 p1, p2, p3
    bnb.optim.GlobalOptimManager.get_instance().register_parameters(
        [p1, p2, p3]
    )
    # 将 p1, p2, p3 移动到 GPU
    p1 = p1.cuda()
    p2 = p2.cuda()
    p3 = p3.cuda()

    # 初始化 Adam 优化器
    adam2 = bnb.optim.Adam([p1, p2, p3], lr, (beta1, beta2), eps)

    # 根据 gtype 设置不同的 atol 和 rtol
    if gtype == torch.float32:
        atol, rtol = 1e-6, 1e-5
    else:
        atol, rtol = 1e-4, 1e-3

    # 进行 50 次优化迭代
    for i in range(50):
        # 生成随机梯度 g1, g2, g3
        g1 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1 + 0.001
        g2 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1 + 0.001
        g3 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1 + 0.001
        # 设置参数 p1, p2, p3 的梯度
        p1.grad = g1
        p2.grad = g2
        p3.grad = g3

        # 执行一步优化
        adam2.step()

        # 断言 p3 的状态1数据类型为 torch.uint8
        assert adam2.state[p3]["state1"].dtype == torch.uint8
        # 断言 p3 的状态2数据类型为 torch.uint8

        assert adam2.state[p3]["state2"].dtype == torch.uint8


# 定义 8 位优化器名称列表
optimizer_names_8bit = [
    "adam8bit",
    "lion8bit",
    "momentum8bit",
    "rmsprop8bit",
    "adam8bit_blockwise",
    "lion8bit_blockwise",
    "momentum8bit_blockwise",
    "rmsprop8bit_blockwise",
]

# 使用 pytest.mark.parametrize 装饰器为 optimizer_names_8bit 添加参数化测试
@pytest.mark.parametrize("optim_name", optimizer_names_8bit, ids=id_formatter("opt"))
# 使用 pytest.mark.parametrize 装饰器为 test_global_config 函数添加参数化测试
@pytest.mark.parametrize("gtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
# 使用 pytest.mark.parametrize 装饰器为 test_global_config 函数添加参数化测试
@pytest.mark.parametrize("dim2", [32, 1024, 4097], ids=id_formatter("dim2"))
# 使用 pytest.mark.parametrize 注册测试参数，dim1 固定为 1024，ids 为 id_formatter("dim1") 返回的标识
@pytest.mark.parametrize("dim1", [1024], ids=id_formatter("dim1"))
def test_optimizer8bit(dim1, dim2, gtype, optim_name):
    # 如果 gtype 为 torch.bfloat16 且 optim_name 不在指定列表中，则跳过测试
    if gtype == torch.bfloat16 and optim_name not in ['adam8bit_blockwise', 'lion8bit_blockwise']: pytest.skip()
    # 如果 dim1 和 dim2 均为 1，则直接返回
    if dim1 == 1 and dim2 == 1:
        return
    # 在 GPU 上生成随机张量 p1，并克隆给 p2
    p1 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1
    p2 = p1.clone()
    # 将 p1 转换为 float 类型
    p1 = p1.float()
    # 设置 blocksize 为 2048
    blocksize = 2048

    # 根据 optim_name 选择对应的优化器，并传入 p1 创建 torch_optimizer 和 bnb_optimizer
    torch_optimizer = str2optimizers[optim_name][0]([p1])
    bnb_optimizer = str2optimizers[optim_name][1]([p2])

    # 根据 gtype 设置不同的误差容限
    if gtype == torch.float32:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-5, 1e-3
    elif gtype == torch.bfloat16:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-4, 1e-2
    else:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-5, 1e-3

    # 初始化错误列表 errors 和相对误差列表 relerrors

    # 打印平均错误和平均相对误差
    # print(sum(errors)/len(errors))
    # print(sum(relerrors)/len(relerrors))


# 使用 pytest.mark.parametrize 注册测试参数，optim_bits 为 32 或 8，ids 为 id_formatter("optim_bits") 返回的标识
@pytest.mark.parametrize("optim_bits", [32, 8], ids=id_formatter("optim_bits"))
# 使用 pytest.mark.parametrize 注册测试参数，gtype 为 torch.float32，ids 为 describe_dtype 返回的标识
@pytest.mark.parametrize("gtype", [torch.float32], ids=describe_dtype)
# 使用 pytest.mark.parametrize 注册测试参数，dim2 固定为 32, 1024, 4097，ids 为 id_formatter("dim2") 返回的标识
@pytest.mark.parametrize("dim2", [32, 1024, 4097], ids=id_formatter("dim2"))
# 使用 pytest.mark.parametrize 注册测试参数，dim1 固定为 1024，ids 为 id_formatter("dim1") 返回的标识
@pytest.mark.parametrize("dim1", [1024], ids=id_formatter("dim1"))
def test_adam_percentile_clipping(dim1, dim2, gtype, optim_bits):
    # 如果 dim1 和 dim2 均为 1，则直接返回
    if dim1 == 1 and dim2 == 1:
        return
    # 在 CPU 上生成随机张量 p1，并设置 beta1、beta2、lr、eps
    p1 = torch.randn(dim1, dim2, device="cpu", dtype=gtype) * 0.1
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.001
    eps = 1e-8
    # 将 p1 移动到 GPU，并克隆给 p2
    p1 = p1.cuda()
    p2 = p1.clone()
    # 创建两个 Adam 优化器，分别设置 optim_bits 和 percentile_clipping 参数

    # 在 GPU 上创建 gnorm_vec 张量，初始化为 0，设置步数为 0
    gnorm_vec = torch.zeros(100).cuda()
    step = 0

# 定义优化器名称列表 optimizer_names_benchmark
optimizer_names_benchmark = [
    "adam8bit_blockwise",
    "paged_adam8bit_blockwise",
    "paged_adamw8bit_blockwise",
    "paged_lion8bit_blockwise",
]
# 使用 pytest.mark.parametrize 注册测试参数，dim1 固定为 4096
# ids 参数用于指定参数的标识符格式
@pytest.mark.parametrize("dim1", [4096], ids=id_formatter("dim1"))
# 使用 pytest.mark.parametrize 注册测试参数，dim2 固定为 4096
# ids 参数用于指定参数的标识符格式
@pytest.mark.parametrize("dim2", [4096], ids=id_formatter("dim2"))
# 使用 pytest.mark.parametrize 注册测试参数，gtype 可选为 torch.float32 或 torch.float16
# ids 参数用于描述数据类型
@pytest.mark.parametrize("gtype", [torch.float32, torch.float16], ids=describe_dtype)
# 使用 pytest.mark.parametrize 注册测试参数，optim_name 从 optimizer_names_benchmark 中选择
# ids 参数用于指定参数的标识符格式
@pytest.mark.parametrize("optim_name", optimizer_names_benchmark, ids=id_formatter("opt"))
# 使用 pytest.mark.benchmark 注册性能测试
@pytest.mark.benchmark
def test_benchmark_blockwise(dim1, dim2, gtype, optim_name):
    # 如果 dim1 和 dim2 都为 1，则直接返回
    if dim1 == 1 and dim2 == 1:
        return
    # 生成随机张量 p1，设备为 cuda，数据类型为 gtype
    p1 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1

    # 根据优化器名称创建优化器对象 bnb_optimizer
    bnb_optimizer = str2optimizers[optim_name][1]([p1])

    # 生成随机梯度张量 g，设备为 cuda，数据类型为 gtype
    g = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.01
    # 将梯度张量 g 赋值给 p1 的梯度
    p1.grad = g
    # 迭代 k 次
    for i in range(k):
        # 当 i 等于 k 的五分之一时
        if i == k // 5:
            # 进行 100 次迭代以进行 burn-in
            torch.cuda.synchronize()
            t0 = time.time()

        # 执行优化器的一步操作
        bnb_optimizer.step()

    # 同步 CUDA 设备
    torch.cuda.synchronize()
    # 计算时间差
    s = time.time() - t0
    print("")
    # 计算参数数量
    params = (k - k // 5) * dim1 * dim2
    # 打印优化器名称、数据类型、每个参数的时间
    print(optim_name, gtype, s / params)
    # 断言时间小于 3.9

# 使用 pytest.mark.parametrize 注册测试参数，dim1 固定为 2 * 1024
# ids 参数用于指定参数的标识符格式
@pytest.mark.parametrize("dim1", [2 * 1024], ids=id_formatter("dim1"))
# 使用 pytest.mark.parametrize 注册测试参数，gtype 固定为 torch.float16
# ids 参数用于描述数据类型
@pytest.mark.parametrize("gtype", [torch.float16], ids=describe_dtype)
# 使用 pytest.mark.parametrize 注册测试参数，optim_name 固定为 'paged_adamw'
# ids 参数用于指定参数的标识符格式
@pytest.mark.parametrize("optim_name", ['paged_adamw'], ids=id_formatter("optim_name"))
# 使用 pytest.mark.parametrize 注册测试参数，mode 固定为 'bnb'
# ids 参数用于指定参数的标识符格式
@pytest.mark.parametrize("mode", ['bnb'], ids=id_formatter("mode"))
# 使用 pytest.mark.benchmark 注册性能测试
@pytest.mark.benchmark
def test_stream_optimizer_bench(dim1, gtype, optim_name, mode):
    # 创建包含 10 个线性层的神经网络 layers1
    layers1 = torch.nn.Sequential(*torch.nn.ModuleList([torch.nn.Linear(dim1, dim1) for i in range(10)]))
    # 将神经网络 layers1 转移到 gtype 数据类型
    layers1 = layers1.to(gtype)
    # 将神经网络 layers1 移动到 CUDA 设备
    layers1 = layers1.cuda()

    large_tensor = None
    # 如果 mode 为 'torch'
    if mode == 'torch':
        # 使用 'paged_adamw' 优化器优化 layers1 的参数
        optim = str2optimizers[optim_name][0](layers1.parameters())
    else:
        # 使用 'paged_adamw' 优化器优化 layers1 的参数
        optim = str2optimizers[optim_name][1](layers1.parameters())
        # 创建大小为 12 GB 的空张量 large_tensor，设备为 cuda
        large_tensor = torch.empty((int(4.5e9),), device='cuda')

    # 同步 CUDA 设备
    torch.cuda.synchronize()
    # 等待 5 秒
    time.sleep(5)

    # 定义批次数量为 5
    num_batches = 5
    # 生成随机批次张量 batches，形状为 (num_batches, 128, dim1)，设备为 cuda，数据类型为 gtype
    batches = torch.randn(num_batches, 128, dim1, device='cuda').to(gtype)
    # 生成一个包含随机整数的张量，范围在[0, 10)，大小为(num_batches, 128)，并将其移动到GPU上
    lbls = torch.randint(0, 10, size=(num_batches, 128)).cuda()

    # 遍历num_batches次
    for i in range(num_batches):
        # 打印当前循环的索引i
        print(i)
        # 获取batches中索引为i的元素
        b = batches[i]
        
        # 如果i等于2，则进行以下操作
        if i == 2:
            # 在GPU上同步所有流
            torch.cuda.synchronize()
            # 记录当前时间
            t0 = time.time()

        # 对输入b执行layers1函数
        out1 = layers1(b)

        # 计算out1和对应标签lbls[i]之间的交叉熵损失，并求均值
        loss1 = torch.nn.functional.cross_entropy(out1, lbls[i]).mean()
        # 反向传播计算梯度
        loss1.backward()
        # 更新优化器的参数
        optim.step()
    
    # 在GPU上同步所有流
    torch.cuda.synchronize()
    # 打印mode和从t0开始到当前时间的时间差
    print(mode, time.time() - t0)
```