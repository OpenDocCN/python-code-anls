# `bitsandbytes\bitsandbytes\functional.py`

```py
# 导入 ctypes 库，用于调用 C 语言函数
import ctypes as ct
# 导入 reduce 函数，用于对可迭代对象进行累积计算，Python 3 中必需
from functools import reduce
import itertools
import operator
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

# 导入自定义模块中的函数
from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict

# 导入自定义模块中的 C 扩展
from .cextension import COMPILED_WITH_CUDA, lib

# 定义一个函数 prod，用于计算可迭代对象的乘积
def prod(iterable):
    return reduce(operator.mul, iterable, 1)

# 创建一个空字典 name2qmap
name2qmap = {}

# 如果编译时启用了 CUDA
if COMPILED_WITH_CUDA:
    """C FUNCTIONS FOR OPTIMIZERS"""
    # 定义一个字典，将字符串映射到不同的优化器函数
    str2optimizer32bit = {
        "adam": (
            lib.cadam32bit_grad_fp32,
            lib.cadam32bit_grad_fp16,
            lib.cadam32bit_grad_bf16,
        ),
        "momentum": (
            lib.cmomentum32bit_grad_32,
            lib.cmomentum32bit_grad_16,
        ),
        "rmsprop": (
            lib.crmsprop32bit_grad_32,
            lib.crmsprop32bit_grad_16,
        ),
        "lion": (
            lib.clion32bit_grad_fp32,
            lib.clion32bit_grad_fp16,
            lib.clion32bit_grad_bf16,
        ),
        "adagrad": (
            lib.cadagrad32bit_grad_32,
            lib.cadagrad32bit_grad_16,
        ),
    }
    # 定义一个字典，将字符串优化器名称映射到对应的静态8位梯度计算函数元组
    str2optimizer8bit = {
        "adam": (
            lib.cadam_static_8bit_grad_32,
            lib.cadam_static_8bit_grad_16,
        ),
        "momentum": (
            lib.cmomentum_static_8bit_grad_32,
            lib.cmomentum_static_8bit_grad_16,
        ),
        "rmsprop": (
            lib.crmsprop_static_8bit_grad_32,
            lib.crmsprop_static_8bit_grad_16,
        ),
        "lion": (
            lib.clion_static_8bit_grad_32,
            lib.clion_static_8bit_grad_16,
        ),
        "lamb": (
            lib.cadam_static_8bit_grad_32,
            lib.cadam_static_8bit_grad_16,
        ),
        "lars": (
            lib.cmomentum_static_8bit_grad_32,
            lib.cmomentum_static_8bit_grad_16,
        ),
    }

    # 定义一个字典，将字符串优化器名称映射到对应的块状8位梯度计算函数元组
    str2optimizer8bit_blockwise = {
        "adam": (
            lib.cadam_8bit_blockwise_grad_fp32,
            lib.cadam_8bit_blockwise_grad_fp16,
            lib.cadam_8bit_blockwise_grad_bf16,
        ),
        "momentum": (
            lib.cmomentum_8bit_blockwise_grad_fp32,
            lib.cmomentum_8bit_blockwise_grad_fp16,
        ),
        "rmsprop": (
            lib.crmsprop_8bit_blockwise_grad_fp32,
            lib.crmsprop_8bit_blockwise_grad_fp16,
        ),
        "lion": (
            lib.clion_8bit_blockwise_grad_fp32,
            lib.clion_8bit_blockwise_grad_fp16,
            lib.clion_8bit_blockwise_grad_bf16,
        ),
        "adagrad": (
            lib.cadagrad_8bit_blockwise_grad_fp32,
            lib.cadagrad_8bit_blockwise_grad_fp16,
        ),
    }
# 定义全局页面管理器类
class GlobalPageManager:
    # 单例模式，存储唯一实例
    _instance = None

    # 初始化方法，抛出运行时错误
    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    # 初始化页面张量列表
    def initialize(self):
        self.paged_tensors = []

    # 类方法，获取全局页面管理器实例
    @classmethod
    def get_instance(cls):
        # 如果实例为空，创建新实例并初始化
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    # 预取所有页面张量到 CPU 或 GPU
    def prefetch_all(self, to_cpu=False):
        # 假设先添加的张量会先被使用，所以逆序遍历并预取
        for t in self.paged_tensors[::-1]:
            prefetch_tensor(t, to_cpu)


# 定义 CUBLAS 上下文类
class CUBLAS_Context:
    # 单例模式，存储唯一实例
    _instance = None

    # 初始化方法，抛出运行时错误
    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    # 初始化 CUBLAS 上下文
    def initialize(self):
        self.context = {}

    # 类方法，获取 CUBLAS 上下文实例
    @classmethod
    def get_instance(cls):
        # 如果实例为空，创建新实例并初始化
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    # 获取指定设备的上下文
    def get_context(self, device):
        # 如果设备索引不在上下文中，设置当前设备并获取上下文
        if device.index not in self.context:
            prev_device = torch.cuda.current_device()
            torch.cuda.set_device(device)
            self.context[device.index] = ct.c_void_p(lib.get_context())
            torch.cuda.set_device(prev_device)
        return self.context[device.index]


# 定义 Cusparse 上下文类
class Cusparse_Context:
    # 单例模式，存储唯一实例
    _instance = None

    # 初始化方法，抛出运行时错误
    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    # 初始化 Cusparse 上下文
    def initialize(self):
        self.context = ct.c_void_p(lib.get_cusparse())

    # 类方法，获取 Cusparse 上下文实例
    @classmethod
    def get_instance(cls):
        # 如果实例为空，创建新实例并初始化
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

# 数据类型到字节数的映射
dtype2bytes = {}
dtype2bytes[torch.float32] = 4
dtype2bytes[torch.float16] = 2
dtype2bytes[torch.bfloat16] = 2
dtype2bytes[torch.uint8] = 1
dtype2bytes[torch.int8] = 1

# 第一个 CUDA 设备
FIRST_CUDA_DEVICE = torch.device('cuda', index=0)
# 创建一个张量，可以指定形状、数据类型、设备，默认为第一个 CUDA 设备
def get_paged(*shape, dtype=torch.float32, device=FIRST_CUDA_DEVICE):
    # 计算张量所需的字节数
    num_bytes = dtype2bytes[dtype]*prod(shape)
    # 获取 CUDA 管理的指针
    cuda_ptr = lib.cget_managed_ptr(ct.c_size_t(num_bytes))
    # 将指针转换为整型指针
    c_ptr = ct.cast(cuda_ptr, ct.POINTER(ct.c_int))
    # 将指针转换为 NumPy 数组
    new_array = np.ctypeslib.as_array(c_ptr, shape=shape)
    # 从 NumPy 数组创建 PyTorch 张量
    out = torch.frombuffer(new_array, dtype=dtype, count=prod(shape)).view(shape)
    # 设置张量为分页状态
    out.is_paged = True
    # 设置张量所在设备的 ID
    out.page_deviceid = device.index
    return out

# 预取张量数据到 CPU 或 GPU
def prefetch_tensor(A, to_cpu=False):
    # 断言张量为分页状态
    assert A.is_paged, 'Only paged tensors can be prefetched!'
    # 根据参数决定设备 ID
    if to_cpu:
        deviceid = -1
    else:
        deviceid = A.page_deviceid
    # 计算张量所需的字节数
    num_bytes = dtype2bytes[A.dtype]*A.numel()
    # 调用 C 函数进行预取
    lib.cprefetch(get_ptr(A), ct.c_size_t(num_bytes), ct.c_int32(deviceid))

# 执行元素级函数操作
def elementwise_func(func_name, A, B, value, prefetch=True):
    # 根据数据类型选择对应的函数和数值类型
    func = None
    if A.dtype == torch.float32:
        func = getattr(lib, f'c{func_name}_fp32', None)
        cvalue = ct.c_float(value)
    elif A.dtype == torch.uint8:
        func = getattr(lib, f'c{func_name}_uint8', None)
        cvalue = ct.c_uint8(value)
    # 如果函数未实现，则抛出异常
    if func is None: raise NotImplementedError(f'Function not implemented: {func_name}')
    # 检查是否为管理状态，并根据参数进行预取
    is_managed = getattr(A, 'is_managed', False)
    if is_managed and prefetch:
        prefetch_tensor(A)
        if B is not None: prefetch_tensor(B)
    # 调用 C 函数执行操作
    func(get_ptr(A), get_ptr(B), cvalue, ct.c_int64(A.numel()))
    # 如果张量为分页状态，则同步 CUDA 操作
    if A.is_paged or B.is_paged:
        # 分页函数是完全异步的
        # 如果从该函数返回，我们希望张量处于正确的状态，即操作发生后的最终状态
        # 因此我们进行同步
        torch.cuda.synchronize()

# 填充张量的值
def fill(A, value, device=None, prefetch=True): elementwise_func('fill', A, None, value)
# 创建一个范围张量
def arange(A, device=None): elementwise_func('arange', A, None, 0)
# 执行元素级乘法操作
def _mul(A, B, device=None): elementwise_func('_mul', A, B, 0)

# 创建线性映射
def create_linear_map(signed=True, total_bits=8, add_zero=True):
    # 根据 signed 参数确定符号位的值，如果 signed 为 True，则符号位为 -1.0，否则为 0.0
    sign = (-1.0 if signed else 0.0)
    # 计算总共可能的取值数量，即 2 的 total_bits 次方
    total_values = 2**total_bits
    # 如果 add_zero 为 True 或者 total_bits 小于 8，则需要添加一个零值
    if add_zero or total_bits < 8:
        # 添加一个零值
        # 由于我们通过在数据类型中添加零来模拟更少的位数，我们需要将量化围绕零中心化，因此会损失一个值
        total_values = (2**total_bits if not signed else 2**total_bits-1)

    # 生成一个从 sign 到 1.0 的均匀间隔的张量
    values = torch.linspace(sign, 1.0, total_values)
    # 计算差值，即 256 与 values 的元素数量之差
    gap = 256 - values.numel()
    # 如果差值为 0，则直接返回 values
    if gap == 0:
        return values
    else:
        # 否则，将 values 分成两部分，中间插入 gap 个零值，然后返回新的张量
        l = values.numel()//2  # noqa: E741
        return torch.Tensor(values[:l].tolist() + [0]*gap + values[l:].tolist())
# 创建一个正态分布映射，可以设置偏移量和是否使用额外值
def create_normal_map(offset=0.9677083, use_extra_value=True):
    try:
        # 导入 scipy 库中的 norm 模块
        from scipy.stats import norm
    except ImportError as ie:
        # 如果导入失败，抛出 ImportError 异常
        raise ImportError(
            "Scipy is required for `create_normal_map`. "
            "Install `bitsandbytes` with the `[test]` extra."
        ) from ie

    if use_extra_value:
        # 如果使用额外值，生成一个不对称类型的正态分布映射
        v1 = norm.ppf(torch.linspace(offset, 0.5, 9)[:-1]).tolist()
        v2 = [0]*(256-15) ## we have 15 non-zero values in this data type
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1])).tolist()
    else:
        # 如果不使用额外值，生成一个对称类型的正态分布映射
        v1 = norm.ppf(torch.linspace(offset, 0.5, 8)[:-1]).tolist()
        v2 = [0]*(256-14) ## we have 14 non-zero values in this data type
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1])).tolist()

    # 合并生成的值列表
    v = v1 + v2 + v3

    # 将值列表转换为 Tensor 类型，并按值排序
    values = torch.Tensor(v)
    values = values.sort().values
    # 将值归一化
    values /= values.max()

    # 断言值的数量为 256
    assert values.numel() == 256

    # 返回归一化后的值
    return values

# 创建一个 8 位浮点数映射，可以设置是否有符号位、指数位数、精度位数和总位数
def create_fp8_map(signed=True, exponent_bits=5, precision_bits=2, total_bits=8):
    e = exponent_bits
    p = precision_bits
    has_sign = 1 if signed else 0
    # 断言指数位数和精度位数之和等于总位数减去符号位数
    assert e+p == total_bits-has_sign
    # 指数位数偏置为 2^(e-1) -1 == 0
    evalues = []
    pvalues = []
    for i, val in enumerate(range(-(2**(exponent_bits-has_sign)), 2**(exponent_bits-has_sign), 1)):
        evalues.append(2**val)

    values = []
    lst = list(itertools.product([0, 1], repeat=precision_bits))
    #for ev in evalues:
    bias = 2**(exponent_bits-1)
    # 遍历指数位数范围内的所有可能值
    for evalue in range(2**(exponent_bits)):
        # 遍历给定的比特模式列表
        for bit_pattern in lst:
            # 初始化数值为1（如果指数值不为0）或0（如果指数值为0）
            value = (1 if evalue != 0 else 0)
            # 根据比特模式计算数值
            for i, pval in enumerate(list(bit_pattern)):
                value += pval*(2**-(i+1))
            # 如果指数值为0，处理 subnormals
            if evalue == 0:
                value = value*2**-(bias)
            # 如果指数值不为0，处理 normals
            else:
                value = value*2**-(evalue-bias-1)
            # 将计算得到的数值添加到列表中
            values.append(value)
            # 如果是有符号数，将相反数也添加到列表中
            if signed:
                values.append(-value)

    # 断言列表长度为总比特数的2次方
    assert len(values) == 2**total_bits
    # 对值列表进行排序
    values.sort()
    # 如果总比特数小于8，补齐到256个值
    if total_bits < 8:
        gap = 256 - len(values)
        for i in range(gap):
            values.append(0)
    # 再次对值列表进行排序
    values.sort()
    # 将值列表转换为 Torch 张量
    code = torch.Tensor(values)
    # 将值归一化
    code /= code.max()

    # 返回处理后的 Torch 张量
    return code
def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    """
    创建动态量化映射表。

    动态数据类型由动态指数和分数组成。随着指数从0增加到-7，用于分数的位数减少。

    这是动态类型的一般化，其中一定数量的位可以保留用于线性量化区域（分数）。n确定最大指数位数。

    有关更多详细信息，请参见
    (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561]
    """

    data = []
    # 这些是来自指数位全为零且没有指示位的情况下的额外项目
    non_sign_bits = total_bits - (1 if signed else 1)
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        fraction_items = int(2 ** (i + non_sign_bits - max_exponent_bits) + 1 if signed else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1)
        boundaries = torch.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1, additional_items + 1)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    data.append(0)
    data.append(1.0)

    assert len(data) == 2**total_bits

    gap = 256 - len(data)
    for i in range(gap):
        data.append(0)

    data.sort()
    return Tensor(data)

def create_quantile_map(A, total_bits=8):
    # 估算输入数组 A 的分位数，分为 2**total_bits-1 个分位数
    q = estimate_quantiles(A, num_quantiles=2**total_bits-1)
    # 将分位数转换为列表形式
    q = q.tolist()
    # 在列表末尾添加一个 0
    q.append(0)

    # 计算需要填充的空缺数量
    gap = 256 - len(q)
    # 循环将 0 填充到列表末尾，使列表长度达到 256
    for i in range(gap):
        q.append(0)

    # 对列表进行排序
    q.sort()

    # 将列表转换为张量
    q = Tensor(q)
    # 将张量中的值归一化到 [-1, 1] 范围内
    q = q/q.abs().max()
    # 返回处理后的张量
    return q
# 返回特殊格式字符串，根据是否有 CUDA 设备来确定返回值
def get_special_format_str():
    # 如果没有 CUDA 设备，则返回 'col_turing'
    if not torch.cuda.is_available(): return 'col_turing'
    # 获取当前 CUDA 设备的主要和次要版本号
    major, _minor = torch.cuda.get_device_capability()
    # 如果主要版本号小于等于7，则返回 "col_turing"
    if major <= 7:
        return "col_turing"
    # 如果主要版本号为8，则返回 "col_ampere"
    if major == 8:
        return "col_ampere"
    # 其他情况返回 "col_turing"


# 检查输入张量是否都在同一个 GPU 上
def is_on_gpu(tensors):
    on_gpu = True
    gpu_ids = set()
    # 遍历输入张量
    for t in tensors:
        # 如果张量为 None，则跳过
        if t is None: continue # NULL pointers are fine
        # 获取张量是否分页的属性
        is_paged = getattr(t, 'is_paged', False)
        # 判断张量是否在 GPU 上或者是否分页
        on_gpu &= (t.device.type == 'cuda' or is_paged)
        # 如果不是分页张量，则记录 GPU ID
        if not is_paged:
            gpu_ids.add(t.device.index)
    # 如果有张量不在 GPU 上，则抛出异常
    if not on_gpu:
        raise TypeError(f'All input tensors need to be on the same GPU, but found some tensors to not be on a GPU:\n {[(t.shape, t.device) for t in tensors]}')
    # 如果存在多个 GPU ID，则抛出异常
    if len(gpu_ids) > 1:
        raise TypeError(f'Input tensors need to be on the same GPU, but found the following tensor and device combinations:\n {[(t.shape, t.device) for t in tensors]}')
    return on_gpu


# 获取 PyTorch 张量的 ctypes 指针
def get_ptr(A: Optional[Tensor]) -> Optional[ct.c_void_p]:
    """
    Get the ctypes pointer from a PyTorch Tensor.

    Parameters
    ----------
    A : torch.tensor
        The PyTorch tensor.

    Returns
    -------
    ctypes.c_void_p
    """
    # 如果张量为 None，则返回 None
    if A is None:
        return None
    else:
        # 否则返回张量的数据指针
        return ct.c_void_p(A.data.data_ptr())


# 在调用之前设置当前 CUDA 设备，并返回之前的设备
def pre_call(device):
    # 获取当前 CUDA 设备，并设置新的设备
    prev_device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    return prev_device


# 在调用之后恢复之前的 CUDA 设备
def post_call(prev_device):
    torch.cuda.set_device(prev_device)


# 获取转换函数的名称，并检查是否存在对应的库函数
def get_transform_func(dtype, orderA, orderOut, transpose=False):
    # 根据数据类型、输入顺序、输出顺序和是否转置生成函数名称
    name = f'ctransform_{(8 if dtype == torch.int8 else 32)}_{orderA}_to_{orderOut}_{"t" if transpose else "n"}'
    # 如果库中不存在该函数，则抛出异常
    if not hasattr(lib, name):
        print(name)
        raise ValueError(
            f"Transform function not supported: {orderA} to {orderOut} for data type {dtype} and transpose={transpose}"
        )
    else:
        # 否则返回对应的库函数
        return getattr(lib, name)


# 获取转换缓冲区
def get_transform_buffer(
    # 定义函数参数：shape 表示形状，dtype 表示数据类型，device 表示设备，to_order 表示目标顺序，from_order 表示原始顺序，默认值为"row"，transpose 表示是否转置，默认为False
def nvidia_transform(
    A,
    to_order,
    from_order="row",
    out=None,
    transpose=False,
    state=None,
    ld=None,
):
    # 初始化函数为 torch.zeros
    init_func = torch.zeros
    # 获取形状的维度
    dims = len(shape)

    if dims == 2:
        # 如果维度为2，获取行数
        rows = shape[0]
    elif dims == 3:
        # 如果维度为3，计算行数
        rows = shape[0] * shape[1]
    # 获取列数
    cols = shape[-1]

    # 保存状态信息
    state = (shape, to_order)
    if transpose:
        # 如果需要转置，交换行列数
        tmp = rows
        rows = cols
        cols = tmp
        state = (shape[::-1], to_order)

    if to_order == "row" or to_order == "col":
        # 如果是行或列顺序，返回初始化后的数据和状态信息
        return init_func(shape, dtype=dtype, device=device), state
    elif to_order == "col32":
        # 如果是32列块（填充），计算列数
        cols = 32 * ((cols + 31) // 32)
        return init_func((rows, cols), dtype=dtype, device=device), state
    elif to_order == "col_turing":
        # 如果是32列和8行块，计算行列数
        cols = 32 * ((cols + 31) // 32)
        rows = 8 * ((rows + 7) // 8)
        return init_func((rows, cols), dtype=dtype, device=device), state
    elif to_order == "col_ampere":
        # 如果是32列和32行块，计算行列数
        cols = 32 * ((cols + 31) // 32)
        rows = 32 * ((rows + 31) // 32)
        return init_func((rows, cols), dtype=dtype, device=device), state
    else:
        # 抛出未实现的错误
        raise NotImplementedError(f"To_order not supported: {to_order}")


def nvidia_transform(
    A,
    to_order,
    from_order="row",
    out=None,
    transpose=False,
    state=None,
    ld=None,
):
    if state is None:
        # 如果状态为空，设置状态为输入矩阵的形状和原始顺序
        state = (A.shape, from_order)
    else:
        # 如果状态不为空，更新原始顺序
        from_order = state[1]
    if out is None:
        # 如果输出为空，获取变换缓冲区和新状态
        out, new_state = get_transform_buffer(
            state[0], A.dtype, A.device, to_order, state[1]
        )
    else:
        # 如果输出不为空，更新新状态
        new_state = (state[1], to_order)
    # 获取变换函数
    func = get_transform_func(A.dtype, from_order, to_order, transpose)

    shape = state[0]
    if len(shape) == 2:
        # 如果形状为2维，获取维度1和维度2
        dim1 = ct.c_int32(shape[0])
        dim2 = ct.c_int32(shape[1])
    elif ld is not None:
        # 如果存在 ld 参数，计算维度1和维度2
        n = prod(shape)
        dim1 = prod([shape[i] for i in ld])
        dim2 = ct.c_int32(n // dim1)
        dim1 = ct.c_int32(dim1)
    # 如果条件不满足，则计算第一维度的乘积并转换为32位整数
    else:
        dim1 = ct.c_int32(shape[0] * shape[1])
        # 将第二维度转换为32位整数
        dim2 = ct.c_int32(shape[2])

    # 获取 CUBLAS 上下文的指针
    ptr = CUBLAS_Context.get_instance().get_context(A.device)
    # 调用函数，传入指针、输入矩阵的指针、输出矩阵的指针、第一维度和第二维度
    func(ptr, get_ptr(A), get_ptr(out), dim1, dim2)

    # 返回输出矩阵和新状态
    return out, new_state
# 估算输入张量的256个等距分位数

def estimate_quantiles(A: Tensor, out: Optional[torch.Tensor] = None, offset: float = 1 / 512, num_quantiles=256) -> Tensor:
    '''
    通过输入张量 `A` 的 eCDF 快速估算256个等距分位数。

    使用 SRAM-Quantiles 算法通过输入张量 `A` 的 eCDF 快速估算256个等距分位数。这是一种快速但近似的算法，
    靠近0和1的极端分位数具有较大的方差/估计误差。可以通过使用修剪分布的 offset 变量来避免这些大误差。
    默认的 offset 值为 1/512 确保最小熵编码 -- 它从分布的每一侧修剪 1/512 = 0.2%。offset 值为 0.01 到 0.02
    通常具有更低的误差，但不是最小熵编码。给定 offset 为 0.02，在范围 [0.02, 0.98] 中使用等距点进行分位数。

    Parameters
    ----------
    A : torch.Tensor
        输入张量。任何形状。
    out : torch.Tensor
        256个估计分位数的张量。
    offset : float
        第一个和最后一个分位数距离0和1的偏移量。默认值：1/(2*num_quantiles)
    num_quantiles : int
        等间距分位数的数量。

    Returns
    -------
    torch.Tensor:
        float32 数据类型的256个分位数。
    '''
    
    # 如果张量 A 的元素数量小于256，则抛出异常
    if A.numel() < 256: raise NotImplementedError(f'Quantile estimation needs at least 256 values in the Tensor, but Tensor had only {A.numel()} values.')
    
    # 如果 num_quantiles 大于256，则抛出异常
    if num_quantiles > 256: raise NotImplementedError(f"Currently only a maximum of 256 equally spaced quantiles are supported, but the argument num_quantiles={num_quantiles}")
    
    # 如果 num_quantiles 小于256 并且 offset 等于 1/(512)
    if num_quantiles < 256 and offset == 1/(512):
        # 覆盖默认参数
        offset = 1/(2*num_quantiles)

    # 如果 out 为 None，则创建一个与 A.device 相同的 float32 数据类型的全零张量
    if out is None: out = torch.zeros((256,), dtype=torch.float32, device=A.device)
    
    # 检查 A 和 out 是否在 GPU 上
    is_on_gpu([A, out])
    
    # 在调用之前获取设备
    device = pre_call(A.device)
    # 如果输入张量 A 的数据类型为 torch.float32
    if A.dtype == torch.float32:
        # 调用 C 库中的函数，估计分位数，传入 A 的指针、输出的指针、偏移量和元素数量
        lib.cestimate_quantiles_fp32(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
    # 如果输入张量 A 的数据类型为 torch.float16
    elif A.dtype == torch.float16:
        # 调用 C 库中的函数，估计分位数，传入 A 的指针、输出的指针、偏移量和元素数量
        lib.cestimate_quantiles_fp16(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
    # 如果输入张量 A 的数据类型不是 torch.float32 或 torch.float16
    else:
        # 抛出未实现的错误，显示不支持的数据类型
        raise NotImplementedError(f"Not supported data type {A.dtype}")
    # 调用后处理函数，传入设备信息
    post_call(device)

    # 如果分位数数量小于 256
    if num_quantiles < 256:
        # 计算步长，使得输出的数量为 256，然后生成索引，将输出按索引重新排列
        step = round(256/num_quantiles)
        idx = torch.linspace(0, 255, num_quantiles).long().to(A.device)
        out = out[idx]

    # 返回输出张量
    return out
class QuantState:
    """定义一个用于量化状态组件的容器，以便与Params4bit和类似类一起使用"""
    valid_quant_types = ('fp4', 'nf4')
    valid_qs_type_keys = [f"bitsandbytes__{x}" for x in valid_quant_types]
    valid_qs_keys = ['absmax', 'quant_map', 'nested_absmax', 'nested_quant_map', 'quant_state', 'quant_type',
                     'blocksize', 'dtype', 'shape', 'nested_blocksize', 'nested_dtype', 'nested_offset']

    def __init__(self, absmax, shape=None, code=None, blocksize=None, quant_type=None, dtype=None, offset=None, state2=None):
        self.absmax = absmax
        self.shape = shape
        self.code = code
        self.dtype = dtype
        self.blocksize = blocksize
        self.quant_type = quant_type
        self.offset = offset
        self.state2 = state2
        self.nested = state2 is not None

    def __get_item__(self, idx):
        """
        确保与旧的量化状态方案的嵌套列表兼容。
        假设以下布局：
        state = [qabsmax, input_shape, A.dtype, blocksize, [offset, state2], quant_type]
        state2 = [absmax, input_shape, A.dtype, blocksize, None, quant_type]
        """
        if self.nested:
            list_repr = [self.absmax, self.shape, self.dtype, self.blocksize, [self.offset, self.state2], self.quant_type]
        else:
            list_repr = [self.absmax, self.shape, self.dtype, self.blocksize, None, self.quant_type]
        return list_repr[idx]

    @classmethod
    # 将量化状态转换为字典形式，用于通过 _save_to_state_dict() 进行序列化
    def as_dict(self, packed=False):
        """
        returns dict of tensors and strings to use in serialization via _save_to_state_dict()
        param: packed -- returns dict[str, torch.Tensor] for state_dict fit for safetensors saving
        """
        # 创建包含量化状态信息的字典
        qs_dict = {
            'quant_type': self.quant_type,
            'absmax': self.absmax,
            'blocksize': self.blocksize,
            'quant_map': self.code,
            'dtype': str(self.dtype).strip('torch.'),
            'shape': tuple(self.shape),
        }
        # 如果存在嵌套量化状态，添加嵌套状态信息到字典中
        if self.nested:
            qs_dict.update({
                'nested_absmax': self.state2.absmax,
                'nested_blocksize': self.state2.blocksize,
                'nested_quant_map': self.state2.code.clone(),  # un-shared to avoid restoring it after shared tensors are removed by safetensors
                'nested_dtype': str(self.state2.dtype).strip('torch.'),
                'nested_offset': self.offset.item(),
            })
        # 如果不需要打包，直接返回量化状态字典
        if not packed:
            return qs_dict

        # 打包格式允许序列化非张量组件，对于在 safetensors 格式中保存非常关键
        # 创建打包后的字典，只包含张量类型的值
        qs_packed_dict = {k: v for k, v in qs_dict.items() if isinstance(v, torch.Tensor)}
        # 创建非张量类型的字典
        non_tensor_dict = {k: v for k, v in qs_dict.items() if not isinstance(v, torch.Tensor)}
        # 将非张量类型的字典打包成张量，并添加到打包后的字典中
        qs_packed_dict["quant_state." + "bitsandbytes__" + self.quant_type] = pack_dict_to_tensor(non_tensor_dict)
        return qs_packed_dict

    # 将量化状态转移到指定设备
    def to(self, device):
        # 确保量化状态在正确的设备上
        self.absmax = self.absmax.to(device)
        if self.nested:
            self.offset = self.offset.to(device)
            self.state2.absmax = self.state2.absmax.to(device)
            self.state2.code = self.state2.code.to(device)
# 在块大小为4096的情况下，对张量A进行分块量化

# 定义函数，接受输入张量A，可选的量化映射code，可选的绝对最大值absmax，可选的输出张量out，块大小为4096，默认不嵌套
def quantize_blockwise(
    A: Tensor,
    code: Optional[torch.Tensor] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize=4096,
    nested=False,
) -> Tuple[Tensor, QuantState]:
    """
    Quantize tensor A in blocks of size 4096 values.

    Quantizes tensor A by dividing it into blocks of 4096 values.
    Then the absolute maximum value within these blocks is calculated
    for the non-linear quantization.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor.
    code : torch.Tensor
        The quantization map.
    absmax : torch.Tensor
        The absmax values.
    out : torch.Tensor
        The output tensor (8-bit).

    Returns
    -------
    torch.Tensor:
        The 8-bit tensor.
    tuple(torch.Tensor, torch.Tensor):
        The quantization state to undo the quantization.
    """

    # 如果未提供code，则根据name2qmap中的动态映射创建动态映射
    if code is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]

    # 如果未提供absmax，则计算绝对最大值
    if absmax is None:
        n = A.numel()
        blocks = n // blocksize
        blocks += 1 if n % blocksize > 0 else 0
        absmax = torch.zeros((blocks,), device=A.device, dtype=torch.float32)

    # 如果未提供out，则创建与A相同形状的8位无符号整数张量
    if out is None:
        out = torch.zeros_like(A, dtype=torch.uint8)
    # 如果 A 的设备类型不是 CPU
    if A.device.type != 'cpu':
        # 断言块大小在指定范围内
        assert blocksize in [4096, 2048, 1024, 512, 256, 128, 64]
        # 将块大小转换为 ctypes 的 c_int32 类型
        cblocksize = ct.c_int32(blocksize)
        # 在调用之前记录当前设备
        prev_device = pre_call(A.device)
        # 将 code 转移到 A 的设备上
        code = code.to(A.device)
        # 检查 code, A, out, absmax 是否在 GPU 上
        is_on_gpu([code, A, out, absmax])
        # 根据 A 的数据类型进行不同的处理
        if A.dtype == torch.float32:
            lib.cquantize_blockwise_fp32(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), cblocksize, ct.c_int(A.numel()))
        elif A.dtype == torch.float16:
            lib.cquantize_blockwise_fp16(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), cblocksize, ct.c_int(A.numel()))
        elif A.dtype == torch.bfloat16:
            lib.cquantize_blockwise_bf16(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), cblocksize, ct.c_int(A.numel()))
        else:
            # 抛出数值错误异常
            raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")
        # 在调用之后恢复设备
        post_call(A.device)
    else:
        # 如果 A 的设备类型是 CPU
        # 将 code 转移到 CPU 上
        code = code.cpu()
        # 调用 CPU 上的函数
        lib.cquantize_blockwise_cpu_fp32(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_longlong(blocksize), ct.c_longlong(A.numel()))

    # 如果嵌套为真
    if nested:
        # 计算 absmax 的均值作为偏移量
        offset = absmax.mean()
        # 减去偏移量
        absmax -= offset
        # 对 absmax 进行块状量化，嵌套为假
        qabsmax, state2 = quantize_blockwise(absmax, blocksize=blocksize, nested=False)
        # 创建量化状态对象
        quant_state = QuantState(absmax=qabsmax, code=code, blocksize=blocksize, dtype=A.dtype, offset=offset, state2=state2)
    else:
        # 创建量化状态对象
        quant_state = QuantState(absmax=absmax, code=code, blocksize=blocksize, dtype=A.dtype)

    # 返回输出和量化状态
    return out, quant_state
def dequantize_blockwise(
    A: Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    code: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: int = 4096,
    nested=False
) -> Tensor:
    """
    Dequantizes blockwise quantized values.

    Dequantizes the tensor A with maximum absolute values absmax in
    blocks of size 4096.

    Parameters
    ----------
    A : torch.Tensor
        The input 8-bit tensor.
    quant_state : QuantState
        Object with code, absmax and other quantization state components.
    absmax : torch.Tensor
        The absmax values.
    code : torch.Tensor
        The quantization map.
    out : torch.Tensor
        Dequantized output tensor (default: float32)


    Returns
    -------
    torch.Tensor:
        Dequantized tensor (default: float32)
    """
    # 确保 quant_state 或 absmax 不为空
    assert quant_state is not None or absmax is not None
    # 如果 code 为空且 quant_state 为空
    if code is None and quant_state is None:
        # 如果名为 "dynamic" 的映射不存在于 name2qmap 中，则创建一个动态映射
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        # 将 code 设置为名为 "dynamic" 的映射
        code = name2qmap["dynamic"]

    # 如果 quant_state 为空
    if quant_state is None:
        # 创建一个 QuantState 对象
        quant_state = QuantState(absmax=absmax, code=code, blocksize=blocksize, dtype=torch.float32)

    # 获取 quant_state 的 absmax 值
    absmax = quant_state.absmax
    # 如果 quant_state 是嵌套的
    if quant_state.nested:
        # 对 quant_state 的 absmax 进行块状去量化
        absmax = dequantize_blockwise(quant_state.absmax, quant_state.state2)
        # 加上 quant_state 的偏移量
        absmax += quant_state.offset
        # 如果 absmax 的数据类型不是 float32，则转换为 float32
        if absmax.dtype != torch.float32: absmax = absmax.float()

    # 如果 out 为空
    if out is None:
        # 创建一个与 A 形状相同的空张量，数据类型为 quant_state 的数据类型，设备为 A 的设备
        out = torch.empty(A.shape, dtype=quant_state.dtype, device=A.device)
    # 检查张量 A 是否在 CPU 上运行，如果不是，则进行预处理
    if A.device.type != 'cpu':
        # 对 A 的设备进行预处理
        device = pre_call(A.device)
        # 将 quant_state.code 转移到 A 所在的设备上
        code = quant_state.code.to(A.device)
        # 检查 quant_state.blocksize 是否在支持的范围内，如果不在范围内则抛出数值错误
        if quant_state.blocksize not in [2048, 4096, 1024, 512, 256, 128, 64]:
            raise ValueError(f"The blockwise of {quant_state.blocksize} is not supported. Supported values: [2048, 4096, 1024, 512, 256, 128, 64]")
        # 检查 A, absmax, out 是否在 GPU 上
        is_on_gpu([A, absmax, out])
        # 根据 out 的数据类型进行不同的量化解码操作
        if out.dtype == torch.float32:
            lib.cdequantize_blockwise_fp32(get_ptr(quant_state.code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(A.numel()))
        elif out.dtype == torch.float16:
            lib.cdequantize_blockwise_fp16(get_ptr(quant_state.code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(A.numel()))
        elif out.dtype == torch.bfloat16:
            lib.cdequantize_blockwise_bf16(get_ptr(quant_state.code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(A.numel()))
        else:
            raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")
        # 对 A 的设备进行后处理
        post_call(A.device)
    else:
        # 将 quant_state.code 转移到 CPU 上
        code = quant_state.code.cpu()
        # 在 CPU 上进行量化解码操作
        lib.cdequantize_blockwise_cpu_fp32(get_ptr(code), get_ptr(A), get_ptr(quant_state.absmax), get_ptr(out), ct.c_longlong(quant_state.blocksize), ct.c_longlong(A.numel()))

    # 返回解码后的张量 out
    return out
def get_4bit_type(typename, device=None, blocksize=64):
    # 如果设备为空，则默认为 'cuda'
    if device is None: device = 'cuda'
    data = None
    # 如果数据类型为 'nf4'
    if typename == 'nf4':
        ''' Implements the NF4 data type.

            Constructs a quantization data type where each bin has equal area under a standard normal distribution N(0, 1) that
            is normalized into the range [-1, 1].

            For more information read the paper: QLoRA: Efficient Finetuning of Quantized LLMs (https://arxiv.org/abs/2305.14314)

            Implementation of the NF4 data type in bitsandbytes can be found in the `create_normal_map` function in
            the `functional.py` file: https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L236.
        '''
        # 定义 NF4 数据类型的数据
        data = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                0.7229568362236023, 1.0]
    # 如果数据类型为 'fp4'
    elif typename == 'fp4':
        # 定义 FP4 数据类型的数据
        data = [0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, -0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0]
    # 如果数据类型为 'int4'
    elif typename == 'int4':
        # 定义 INT4 数据类型的数据
        data = [7, 6, 5, 4, 3, 2, 1, 0, -0, -1, -2, -3, -4, -5, -6, -7]
    # 如果类型名为 'af4'
    elif typename == 'af4':
        # 从论文中引用的数据，表示4位AbnormalFloats的值
        # https://arxiv.org/abs/2306.06965
        # 如果块大小为64
        if blocksize == 64:
            # 初始化数据为一组特定数值
            data = [-1., -0.69441008, -0.51243739, -0.3736951, -0.25607552, -0.14982478,
                    -0.04934812,  0., 0.04273164, 0.12934483, 0.21961274, 0.31675666,
                    0.42563882,  0.55496234,  0.72424863,  1.][::-1]
        else:
            # 如果块大小不为64，则抛出未实现的错误
            raise NotImplementedError('4-bit AbnormalFloats currently only support blocksize 64.')

    # 如果数据为空
    if data is None:
        # 抛出未实现的错误，指明不支持该类型名
        raise NotImplementedError(f'Typename {typename} not supported')

    # 将数据转换为张量
    data = Tensor(data)
    # 将数据归一化
    data /= data.abs().max()
    # 断言数据元素数量为16
    assert data.numel() == 16

    # 返回移至指定设备后的数据
    return data.to(device)
# 定义一个函数，将输入张量 A 以 4 位值的块进行量化
def quantize_fp4(A: Tensor, absmax: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None, blocksize=64, compress_statistics=False, quant_storage=torch.uint8):
    # 调用 quantize_4bit 函数，使用 'fp4' 类型进行量化
    return quantize_4bit(A, absmax, out, blocksize, compress_statistics, 'fp4', quant_storage)

# 定义一个函数，将输入张量 A 以 4 位值的块进行量化
def quantize_nf4(A: Tensor, absmax: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None, blocksize=64, compress_statistics=False, quant_storage=torch.uint8):
    # 调用 quantize_4bit 函数，使用 'nf4' 类型进行量化
    return quantize_4bit(A, absmax, out, blocksize, compress_statistics, 'nf4', quant_storage)

# 定义一个函数，将输入张量 A 以 4 位值的块进行量化
def quantize_4bit(
    A: Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize=64,
    compress_statistics=False,
    quant_type='fp4',
    quant_storage=torch.uint8,
) -> Tuple[Tensor, QuantState]:
    """
    Quantize tensor A in blocks of 4-bit values.

    Quantizes tensor A by dividing it into blocks which are independently quantized to FP4.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor.
    absmax : torch.Tensor
        The absmax values.
    out : torch.Tensor
        The output tensor.
    blocksize : int
        The blocksize used in quantization.
    quant_type : str
        The 4-bit quantization data type {fp4, nf4}

    Returns
    -------
    torch.Tensor:
        Tensor with packed 4-bit values.
    tuple(torch.Tensor, torch.Size, torch.dtype, int):
        The quantization state to undo the quantization.
    """
    # 如果输入张量 A 不在 CUDA 设备上，抛出异常
    if A.device.type != 'cuda':
        raise NotImplementedError(f'Device type not supported for FP4 quantization: {A.device.type}')
    # 如果量化类型不是 'fp4' 或 'nf4'，抛出异常
    if quant_type not in ['fp4', 'nf4']:
        raise NotImplementedError(f'4-bit quantization data type {quant_type} is not implemented.')

    # 计算输入张量 A 的元素个数
    n = A.numel()
    # 获取输入张量 A 的形状
    input_shape = A.shape

    # 如果 absmax 未提供，则计算块数并初始化 absmax 张量
    if absmax is None:
        blocks = n // blocksize
        blocks += 1 if n % blocksize > 0 else 0
        absmax = torch.zeros((blocks,), device=A.device, dtype=torch.float32)
    # 如果输出为空，则根据量化存储类型计算模数，并创建相应形状的全零张量
    if out is None:
        mod = dtype2bytes[quant_storage] * 2
        out = torch.zeros(((n+1)//mod, 1), dtype=quant_storage, device=A.device)

    # 检查块大小是否在指定范围内
    assert blocksize in [4096, 2048, 1024, 512, 256, 128, 64]

    # 在调用之前记录当前设备
    prev_device = pre_call(A.device)
    # 检查输入张量是否在 GPU 上
    is_on_gpu([A, out, absmax])

    # 根据输入张量的数据类型进行不同的量化处理
    if A.dtype == torch.float32:
        if quant_type == 'fp4':
            lib.cquantize_blockwise_fp32_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int32(blocksize), ct.c_int(n))
        else:
            lib.cquantize_blockwise_fp32_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int32(blocksize), ct.c_int(n))
    elif A.dtype == torch.float16:
        if quant_type == 'fp4':
            lib.cquantize_blockwise_fp16_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int32(blocksize), ct.c_int(n))
        else:
            lib.cquantize_blockwise_fp16_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int32(blocksize), ct.c_int(n))
    elif A.dtype == torch.bfloat16:
        if quant_type == 'fp4':
            lib.cquantize_blockwise_bf16_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int32(blocksize), ct.c_int(n))
        else:
            lib.cquantize_blockwise_bf16_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int32(blocksize), ct.c_int(n))
    else:
        raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")
    # 调用后处理函数
    post_call(A.device)

    # 获取4位类型的编码
    code = get_4bit_type(quant_type, device=A.device)

    # 如果需要压缩统计信息
    if compress_statistics:
        # 计算绝对值的均值作为偏移量
        offset = absmax.mean()
        absmax -= offset
        # 对绝对值进行块状量化
        qabsmax, state2 = quantize_blockwise(absmax, blocksize=256)
        del absmax
        # 创建量化状态对象
        state = QuantState(absmax=qabsmax, shape=input_shape, dtype=A.dtype, blocksize=blocksize, code=code, quant_type=quant_type, offset=offset, state2=state2)
    # 如果不是第一次调用，创建一个新的量化状态对象
    else:
        state = QuantState(absmax=absmax, shape=input_shape, dtype=A.dtype, blocksize=blocksize, code=code, quant_type=quant_type, )
    
    # 返回输出和量化状态对象
    return out, state
# 将输入的 FP4 块状量化数值反量化
def dequantize_fp4(A: Tensor, quant_state: Optional[QuantState] = None, absmax: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None, blocksize: int = 64) -> Tensor:
    # 调用 dequantize_4bit 函数，指定 quant_type 为 'fp4'
    return dequantize_4bit(A, quant_state, absmax, out, blocksize, 'fp4')

# 将输入的 NF4 块状量化数值反量化
def dequantize_nf4(A: Tensor, quant_state: Optional[QuantState] = None, absmax: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None, blocksize: int = 64) -> Tensor:
    # 调用 dequantize_4bit 函数，指定 quant_type 为 'nf4'
    return dequantize_4bit(A, quant_state, absmax, out, blocksize, 'nf4')

# 反量化 4 位块状量化数值
def dequantize_4bit(A: Tensor, quant_state: Optional[QuantState] = None, absmax: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None, blocksize: int = 64, quant_type='fp4') -> Tensor:
    """
    Dequantizes FP4 blockwise quantized values.

    Dequantizes the tensor A with maximum absolute values absmax in blocks of size blocksize.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor (packed 4-bit values).
    quant_state : QuantState
        object with quantisation stats, incl. absmax values, original tensor shape and original dtype.
    absmax : torch.Tensor
        The absmax values.
    out : torch.Tensor
        Dequantized output tensor.
    blocksize : int
        The blocksize used in quantization.
    quant_type : str
        The 4-bit quantization data type {fp4, nf4}

    Returns
    -------
    torch.Tensor:
        Dequantized tensor.
    """
    # 检查 blocksize 是否为支持的值
    if blocksize not in [2048, 4096, 1024, 512, 256, 128, 64]:
        raise ValueError(f"The blockwise of {blocksize} is not supported. Supported values: [2048, 4096, 1024, 512, 256, 128, 64]")
    # 检查 quant_type 是否为支持的值
    if quant_type not in ['fp4', 'nf4']:
        raise NotImplementedError(f'4-bit quantization data type {quant_type} is not implemented.')

    # 如果 quant_state 为 None，则要求 absmax 和 out 参数不为 None
    if quant_state is None:
        assert absmax is not None and out is not None
        # 创建 QuantState 对象，包含 absmax、shape、dtype、blocksize 和 quant_type
        quant_state = QuantState(absmax=absmax, shape=out.shape, dtype=out.dtype, blocksize=blocksize, quant_type=quant_type)
    # 如果 quant_state.nested 为真，则执行以下操作
    if quant_state.nested:
        # 根据 quant_state.absmax 和 quant_state.state2 进行分块反量化
        absmax = dequantize_blockwise(quant_state.absmax, quant_state.state2)
        # 将偏移量加到 absmax 上
        absmax += quant_state.offset
        # 如果 absmax 的数据类型不是 torch.float32，则将其转换为 torch.float32
        if absmax.dtype != torch.float32: absmax = absmax.float()

    # 如果 out 为 None，则创建一个与 quant_state 相同形状和数据类型的空张量
    if out is None:
        out = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=A.device)

    # 计算 out 的元素个数
    n = out.numel()

    # 在调用之前获取设备信息
    device = pre_call(A.device)
    # 检查 A、absmax 和 out 是否在 GPU 上
    is_on_gpu([A, absmax, out])

    # 根据 out 的数据类型执行不同的反量化操作
    if out.dtype == torch.float32:
        # 根据 quant_state.quant_type 的值选择不同的反量化函数
        if quant_state.quant_type == 'fp4':
            lib.cdequantize_blockwise_fp32_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(n))
        else:
            lib.cdequantize_blockwise_fp32_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(n))
    elif out.dtype == torch.float16:
        if quant_state.quant_type == 'fp4':
            lib.cdequantize_blockwise_fp16_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(n))
        else:
            lib.cdequantize_blockwise_fp16_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(n))
    elif out.dtype == torch.bfloat16:
        if quant_state.quant_type == 'fp4':
            lib.cdequantize_blockwise_bf16_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(n))
        else:
            lib.cdequantize_blockwise_bf16_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(n))
    else:
        # 如果 out 的数据类型不是 torch.float32、torch.float16 或 torch.bfloat16，则抛出异常
        raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")
    
    # 在调用之后处理设备信息
    post_call(A.device)

    # 检查 A 的形状是否为 (1, ...)，如果是则返回 out 的转置，否则返回 out
    is_transposed = (True if A.shape[0] == 1 else False)
    if is_transposed: return out.t()
    else: return out
# 定义一个量化函数，将输入张量 A 量化为 8 位
def quantize(
    A: Tensor,
    code: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    # 如果未提供量化映射表，则创建一个动态映射表
    if code is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]
        code = code.to(A.device)

    # 计算输入张量 A 的绝对值的最大值
    absmax = torch.abs(A).max()
    # 如果最大值的数据类型不是 float32，则转换为 float32
    if absmax.dtype != torch.float32: absmax = absmax.float()
    # 将输入张量 A 标准化
    inp = A / absmax
    # 对标准化后的输入进行量化，得到输出张量 out
    out = quantize_no_absmax(inp, code, out)
    # 返回量化后的输出张量和元组 (绝对值最大值, 量化映射表)
    return out, (absmax, code)


# 定义一个反量化函数，将输入张量 A 反量化为原始值
def dequantize(
    A: Tensor,
    state: Optional[Tuple[Tensor, Tensor]] = None,
    absmax: Optional[torch.Tensor] = None,
    code: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tensor:
    # 断言必须提供状态元组或绝对值最大值
    assert state is not None or absmax is not None
    # 如果未提供量化映射表和状态元组，则创建一个动态映射表
    if code is None and state is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]
        code = code.to(A.device)

    # 如果未提供状态元组，则使用提供的绝对值最大值和量化映射表
    if state is None:
        state = (absmax, code)
    # 对输入张量 A 进行反量化，得到输出张量 out
    out = dequantize_no_absmax(A, state[1], out)
    # 返回反量化后的输出张量乘以绝对值最大值
    return out * state[0]


# 定义一个不考虑绝对值最大值的量化函数
def quantize_no_absmax(A: Tensor, code: Tensor, out: Optional[torch.Tensor] = None) -> Tensor:
    '''
    Quantizes input tensor to 8-bit.

    Quantizes the 32-bit input tensor `A` to the 8-bit output tensor
    `out` using the quantization map `code`.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor.
    code : torch.Tensor
        The quantization map.
    out : torch.Tensor, optional
        The output tensor. Needs to be of type byte.

    Returns
    -------
    torch.Tensor:
        Quantized 8-bit tensor.
    '''
    # 保存调用前的设备信息
    prev_device = pre_call(A.device)
    # 如果未提供输出张量，则创建一个与输入张量 A 相同形状的零张量
    if out is None: out = torch.zeros_like(A, dtype=torch.uint8)
    # 检查输入张量 A 和输出张量 out 是否在 GPU 上
    is_on_gpu([A, out])
    # 调用 C 函数进行量化操作
    lib.cquantize(get_ptr(code), get_ptr(A), get_ptr(out), ct.c_int(A.numel()))
    # 恢复调用前的设备信息
    post_call(prev_device)
    # 返回量化后的输出张量
    return out
# 将8位张量去量化为32位张量

def dequantize_no_absmax(A: Tensor, code: Tensor, out: Optional[torch.Tensor] = None) -> Tensor:
    '''
    Dequantizes the 8-bit tensor to 32-bit.

    Dequantizes the 8-bit tensor `A` to the 32-bit tensor `out` via
    the quantization map `code`.

    Parameters
    ----------
    A : torch.Tensor
        The 8-bit input tensor.
    code : torch.Tensor
        The quantization map.
    out : torch.Tensor
        The 32-bit output tensor.

    Returns
    -------
    torch.Tensor:
        32-bit output tensor.
    '''
    # 保存之前的设备信息
    prev_device = pre_call(A.device)
    # 如果输出张量为空，则创建一个与输入张量A相同形状的全零张量
    if out is None: out = torch.zeros_like(A, dtype=torch.float32)
    # 检查张量是否在GPU上
    is_on_gpu([code, A, out])
    # 调用C函数进行去量化操作
    lib.cdequantize(get_ptr(code), get_ptr(A), get_ptr(out), ct.c_int(A.numel()))
    # 恢复之前的设备信息
    post_call(prev_device)
    # 返回32位输出张量
    return out


# 执行带有一个或两个优化器状态的原地优化器更新

def optimizer_update_32bit(
    optimizer_name: str,
    g: Tensor,
    p: Tensor,
    state1: Tensor,
    beta1: float,
    eps: float,
    step: int,
    lr: float,
    state2: Optional[torch.Tensor] = None,
    beta2: float = 0.0,
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    unorm_vec: Optional[torch.Tensor] = None,
    max_unorm: float = 0.0,
    skip_zeros=False,
) -> None:
    """
    Performs an inplace optimizer update with one or two optimizer states.

    Universal optimizer update for 32-bit state and 32/16-bit gradients/weights.

    Parameters
    ----------
    optimizer_name : str
        The name of the optimizer: {adam}.
    g : torch.Tensor
        Gradient tensor.
    p : torch.Tensor
        Parameter tensor.
    state1 : torch.Tensor
        Optimizer state 1.
    beta1 : float
        Optimizer beta1.
    eps : float
        Optimizer epsilon.
    weight_decay : float
        Weight decay.
    step : int
        Current optimizer step.
    lr : float
        The learning rate.
    state2 : torch.Tensor
        Optimizer state 2.
    beta2 : float
        Optimizer beta2.
    gnorm_scale : float
        梯度重新缩放的因子，使其不超过最大剪切值。
    unorm_vec : torch.Tensor
        用于更新规范的张量。
    max_unorm : float
        相对于权重规范的最大更新规范。
    skip_zeros : bool
        是否跳过零值梯度（默认为False）。

    param_norm = 0.0
    如果最大更新规范大于0.0，则计算参数的规范。
    if max_unorm > 0.0:
        param_norm = torch.norm(p.data.float())

    optim_func = None
    根据梯度的数据类型选择相应的优化器函数
    if g.dtype == torch.float32:
        optim_func = str2optimizer32bit[optimizer_name][0]
    elif g.dtype == torch.float16:
        optim_func = str2optimizer32bit[optimizer_name][1]
    elif (g.dtype == torch.bfloat16 and len(str2optimizer32bit[optimizer_name])==3):
        optim_func = str2optimizer32bit[optimizer_name][2]
    else:
        raise ValueError(f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}")

    is_on_gpu([g, p, state1, state2, unorm_vec])
    在GPU上执行操作
    prev_device = pre_call(g.device)
    调用优化器函数进行参数更新
    optim_func(
        get_ptr(g),
        get_ptr(p),
        get_ptr(state1),
        get_ptr(state2),
        get_ptr(unorm_vec),
        ct.c_float(max_unorm),
        ct.c_float(param_norm),
        ct.c_float(beta1),
        ct.c_float(beta2),
        ct.c_float(eps),
        ct.c_float(weight_decay),
        ct.c_int32(step),
        ct.c_float(lr),
        ct.c_float(gnorm_scale),
        ct.c_bool(skip_zeros),
        ct.c_int32(g.numel()))
    恢复之前的设备状态
    post_call(prev_device)
# 定义一个函数，用于更新8位状态下的优化器参数
def optimizer_update_8bit(
    optimizer_name: str,  # 优化器的名称
    g: Tensor,  # 梯度张量
    p: Tensor,  # 参数张量
    state1: Tensor,  # Adam状态1
    state2: Tensor,  # Adam状态2
    beta1: float,  # Adam的beta1
    beta2: float,  # Adam的beta2
    eps: float,  # Adam的epsilon
    step: int,  # 当前优化器步数
    lr: float,  # 学习率
    qmap1: Tensor,  # 第一个Adam状态的量化映射
    qmap2: Tensor,  # 第二个Adam状态的量化映射
    max1: Tensor,  # 第一个Adam状态更新的最大值
    max2: Tensor,  # 第二个Adam状态更新的最大值
    new_max1: Tensor,  # 下一个第一个Adam状态更新的最大值
    new_max2: Tensor,  # 下一个第二个Adam状态更新的最大值
    weight_decay: float = 0.0,  # 权重衰减，默认为0.0
    gnorm_scale: float = 1.0,  # 梯度缩放因子，默认为1.0
    unorm_vec: Optional[torch.Tensor] = None,  # 更新范数的张量，默认为None
    max_unorm: float = 0.0,  # 最大更新范数相对于权重范数的值，默认为0.0
) -> None:
    """
    执行原地Adam更新。

    适用于32/8位状态和32/16位梯度/权重的通用Adam更新。
    如果权重衰减>0.0，则使用AdamW公式。

    参数
    ----------
    optimizer_name : str
        优化器的名称。选择 {adam, momentum}
    g : torch.Tensor
        梯度张量。
    p : torch.Tensor
        参数张量。
    state1 : torch.Tensor
        Adam状态1。
    state2 : torch.Tensor
        Adam状态2。
    beta1 : float
        Adam的beta1。
    beta2 : float
        Adam的beta2。
    eps : float
        Adam的epsilon。
    weight_decay : float
        权重衰减。
    step : int
        当前优化器步数。
    lr : float
        学习率。
    qmap1 : torch.Tensor
        第一个Adam状态的量化映射。
    qmap2 : torch.Tensor
        第二个Adam状态的量化映射。
    max1 : torch.Tensor
        第一个Adam状态更新的最大值。
    max2 : torch.Tensor
        第二个Adam状态更新的最大值。
    new_max1 : torch.Tensor
        下一个第一个Adam状态更新的最大值。
    new_max2 : torch.Tensor
        下一个第二个Adam状态更新的最大值。
    gnorm_scale : float
        将梯度重新缩放为最大剪辑值的因子。
    unorm_vec : torch.Tensor
        用于更新范数的张量。
    max_unorm : float
        相对于权重范数的最大更新范数。
    """

    param_norm = 0.0  # 参数范数初始化为0.0
    # 如果最大无符号数大于0.0，则计算参数的范数
    if max_unorm > 0.0:
        param_norm = torch.norm(p.data.float())

    # 在调用之前获取设备信息
    prev_device = pre_call(g.device)
    # 检查是否所有变量都在GPU上
    is_on_gpu([g, p, state1, state2, unorm_vec, qmap1, qmap2, max1, max2, new_max1, new_max2])
    
    # 如果梯度和状态数据类型分别为torch.float32和torch.uint8
    if g.dtype == torch.float32 and state1.dtype == torch.uint8:
        # 调用对应的优化器函数，传入参数指针和其他参数
        str2optimizer8bit[optimizer_name][0](
            get_ptr(p),
            get_ptr(g),
            get_ptr(state1),
            get_ptr(state2),
            get_ptr(unorm_vec),
            ct.c_float(max_unorm),
            ct.c_float(param_norm),
            ct.c_float(beta1),
            ct.c_float(beta2),
            ct.c_float(eps),
            ct.c_int32(step),
            ct.c_float(lr),
            get_ptr(qmap1),
            get_ptr(qmap2),
            get_ptr(max1),
            get_ptr(max2),
            get_ptr(new_max1),
            get_ptr(new_max2),
            ct.c_float(weight_decay),
            ct.c_float(gnorm_scale),
            ct.c_int32(g.numel()),
        )
    # 如果梯度和状态数据类型分别为torch.float16和torch.uint8
    elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
        # 调用对应的优化器函数，传入参数指针和其他参数
        str2optimizer8bit[optimizer_name][1](
            get_ptr(p),
            get_ptr(g),
            get_ptr(state1),
            get_ptr(state2),
            get_ptr(unorm_vec),
            ct.c_float(max_unorm),
            ct.c_float(param_norm),
            ct.c_float(beta1),
            ct.c_float(beta2),
            ct.c_float(eps),
            ct.c_int32(step),
            ct.c_float(lr),
            get_ptr(qmap1),
            get_ptr(qmap2),
            get_ptr(max1),
            get_ptr(max2),
            get_ptr(new_max1),
            get_ptr(new_max2),
            ct.c_float(weight_decay),
            ct.c_float(gnorm_scale),
            ct.c_int32(g.numel()),
        )
    else:
        # 抛出数值错误，表示不支持梯度和优化器位数据类型组合
        raise ValueError(
            f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}"
        )
    
    # 在调用之后恢复设备信息
    post_call(prev_device)
# 定义一个函数，用于在8位块状优化器更新时更新优化器状态
def optimizer_update_8bit_blockwise(
    optimizer_name: str,  # 优化器名称
    g: Tensor,  # 梯度张量
    p: Tensor,  # 参数张量
    state1: Tensor,  # 状态1张量
    state2: Tensor,  # 状态2张量
    beta1: float,  # beta1 参数
    beta2: float,  # beta2 参数
    eps: float,  # epsilon 参数
    step: int,  # 步数
    lr: float,  # 学习率
    qmap1: Tensor,  # qmap1 张量
    qmap2: Tensor,  # qmap2 张量
    absmax1: Tensor,  # absmax1 张量
    absmax2: Tensor,  # absmax2 张量
    weight_decay: float = 0.0,  # 权重衰减，默认为0.0
    gnorm_scale: float = 1.0,  # 梯度范数缩放，默认为1.0
    skip_zeros=False,  # 是否跳过零，默认为False
) -> None:  # 返回空值

    # 初始化优化器函数
    optim_func = None
    # 保存梯度张量的设备信息
    prev_device = pre_call(g.device)
    # 检查张量是否在GPU上
    is_on_gpu([g, p, state1, state2, qmap1, qmap2, absmax1, absmax2])

    # 根据梯度和状态1的数据类型选择相应的优化器函数
    if g.dtype == torch.float32 and state1.dtype == torch.uint8:
        optim_func = str2optimizer8bit_blockwise[optimizer_name][0]
    elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
        optim_func = str2optimizer8bit_blockwise[optimizer_name][1]
    elif (g.dtype == torch.bfloat16 and state1.dtype == torch.uint8 and
          len(str2optimizer8bit_blockwise[optimizer_name])==3):
        optim_func = str2optimizer8bit_blockwise[optimizer_name][2]
    else:
        raise ValueError(
            f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}"
        )
    # 恢复之前保存的设备信息
    post_call(prev_device)

    # 再次检查张量是否在GPU上
    is_on_gpu([p, g, state1, state2, qmap1, qmap2, absmax1, absmax2])

    # 保存梯度张量的设备信息
    prev_device = pre_call(g.device)
    # 调用优化器函数更新参数
    optim_func(
        get_ptr(p),
        get_ptr(g),
        get_ptr(state1),
        get_ptr(state2),
        ct.c_float(beta1),
        ct.c_float(beta2),
        ct.c_float(eps),
        ct.c_int32(step),
        ct.c_float(lr),
        get_ptr(qmap1),
        get_ptr(qmap2),
        get_ptr(absmax1),
        get_ptr(absmax2),
        ct.c_float(weight_decay),
        ct.c_float(gnorm_scale),
        ct.c_bool(skip_zeros),
        ct.c_int32(g.numel()),
    )
    # 恢复之前保存的设备信息
    post_call(prev_device)

# 定义一个函数，用于应用百分位剪裁
def percentile_clipping(
    grad: Tensor,  # 梯度张量
    gnorm_vec: Tensor,  # 梯度范数向量
    step: int,  # 步数
    percentile: int = 5  # 百分位，默认为5
):
    """Applies percentile clipping

    grad: torch.Tensor
        The gradient tensor.
    # 定义梯度范数向量，预期有100个元素
    gnorm_vec: torch.Tensor
        Vector of gradient norms. 100 elements expected.
    # 当前优化步骤（过去梯度范数的数量）
    step: int
        The current optimiation steps (number of past gradient norms).

    """
    # 保存调用前的设备
    prev_device = pre_call(grad.device)
    # 检查梯度和梯度范数向量是否在 GPU 上
    is_on_gpu([grad, gnorm_vec])
    # 根据梯度类型执行不同的百分位裁剪操作
    if grad.dtype == torch.float32:
        lib.cpercentile_clipping_g32(
            get_ptr(grad),
            get_ptr(gnorm_vec),
            ct.c_int32(step),
            ct.c_int32(grad.numel()),
        )
    elif grad.dtype == torch.float16:
        lib.cpercentile_clipping_g16(
            get_ptr(grad),
            get_ptr(gnorm_vec),
            ct.c_int32(step),
            ct.c_int32(grad.numel()),
        )
    else:
        # 抛出异常，不支持的梯度类型
        raise ValueError(f"Gradient type {grad.dtype} not supported!")
    # 恢复调用前的设备
    post_call(prev_device)

    # 计算当前梯度范数
    current_gnorm = torch.sqrt(gnorm_vec[step % 100])
    # 对梯度范数向量进行排序，并获取指定百分位的值和索引
    vals, idx = torch.sort(gnorm_vec)
    # 计算裁剪值
    clip_value = torch.sqrt(vals[percentile])
    # 初始化梯度范数缩放因子
    gnorm_scale = 1.0

    # 如果当前梯度范数大于裁剪值，则更新梯度范数缩放因子
    if current_gnorm > clip_value:
        gnorm_scale = clip_value / current_gnorm

    # 返回当前梯度范数、裁剪值和梯度范数缩放因子
    return current_gnorm, clip_value, gnorm_scale
# 定义一个函数，用于在二维直方图上进行散点加法操作
def histogram_scatter_add_2d(
    histogram: Tensor, index1: Tensor, index2: Tensor, source: Tensor
):
    # 断言直方图的维度为2
    assert len(histogram.shape) == 2
    # 断言直方图的数据类型为torch.float32
    assert histogram.dtype == torch.float32
    # 断言source的数据类型为torch.float32
    assert source.dtype == torch.float32
    # 断言index1的数据类型为torch.int32
    assert index1.dtype == torch.int32
    # 断言index2的数据类型为torch.int32
    assert index2.dtype == torch.int32

    # 断言直方图、index1、index2、source都在cuda设备上
    assert histogram.device.type == "cuda"
    assert index1.device.type == "cuda"
    assert index2.device.type == "cuda"
    assert source.device.type == "cuda"

    # 将直方图的第一维度大小转换为c_int32类型
    maxdim1 = ct.c_int32(histogram.shape[0])
    # 获取index1的元素数量
    n = ct.c_int32(index1.numel())
    # 检查直方图、index1、index2、source是否在GPU上
    is_on_gpu([histogram, index1, index2, source])
    # 调用C扩展库中的函数进行二维直方图的散点加法操作
    lib.chistogram_scatter_add_2d(get_ptr(histogram), get_ptr(index1), get_ptr(index2), get_ptr(source), maxdim1, n)

# 定义一个函数，用于检查矩阵乘法的输入是否符合要求
def check_matmul(A, B, out, transposed_A, transposed_B, expected_type=torch.int8):
    # 如果CUDA未初始化，则初始化CUDA
    if not torch.cuda.is_initialized(): torch.cuda.init()
    # 如果A或B的数据类型不是torch.int8，则抛出类型错误
    if A.dtype != expected_type or B.dtype != expected_type:
        raise TypeError(
            f"Expected torch.int8 input tensors A and B, but got {A.dtype} and {B.dtype}"
        )

    # 获取A和B的形状
    sA = A.shape
    sB = B.shape
    tA = transposed_A
    tB = transposed_B

    # 初始化correct为True
    correct = True

    # 检查A和B的维度是否符合矩阵乘法的要求
    if len(sA) == 2 and len(sB) == 2:
        if not tA and not tB and A.shape[1] != B.shape[0]:
            correct = False
        elif tA and not tB and A.shape[0] != B.shape[0]:
            correct = False
        elif tA and tB and A.shape[0] != B.shape[1]:
            correct = False
        elif not tA and tB and A.shape[1] != B.shape[1]:
            correct = False
    elif len(sA) == 3 and len(sB) == 2:
        if not tA and not tB and A.shape[2] != B.shape[0]:
            correct = False
        elif tA and not tB and A.shape[1] != B.shape[0]:
            correct = False
        elif tA and tB and A.shape[1] != B.shape[1]:
            correct = False
        elif not tA and tB and A.shape[2] != B.shape[1]:
            correct = False
    # 如果两个张量的维度都为3
    elif len(sA) == 3 and len(sB) == 3:
        # 如果 A 和 B 都不需要转置，并且 A 的第三维不等于 B 的第二维
        if not tA and not tB and A.shape[2] != B.shape[1]:
            correct = False
        # 如果 A 需要转置，B 不需要转置，并且 A 的第二维不等于 B 的第二维
        elif tA and not tB and A.shape[1] != B.shape[1]:
            correct = False
        # 如果 A 和 B 都需要转置，并且 A 的第二维不等于 B 的第三维
        elif tA and tB and A.shape[1] != B.shape[2]:
            correct = False
        # 如果 A 不需要转置，B 需要转置，并且 A 的第三维不等于 B 的第三维
        elif not tA and tB and A.shape[2] != B.shape[2]

    # 如果输出张量不为空
    if out is not None:
        # 获取输出张量的维度
        sout = out.shape
        # 在反向传播中常见的特殊情况
        if not correct and len(sA) == 3 and len(sB) == 3:
            # 如果满足特定条件，则设置 correct 为 True
            if (
                sout[0] == sA[2]
                and sout[1] == sB[2]
                and sA[0] == sB[0]
                and sA[1] == sB[1]
            ):
                correct = True
    else:
        # 如果 A 和 B 的维度都为2
        if len(sA) == 2 and len(sB) == 2:
            # 根据转置情况设置输出张量的维度
            if not tA and not tB:
                sout = (sA[0], sB[1])
            elif tA and tB:
                sout = (sA[1], sB[0])
            elif tA and not tB:
                sout = (sA[1], sB[1])
            elif not tA and tB:
                sout = (sA[0], sB[0])
        # 如果 A 的维度为3，B 的维度为2
        elif len(sA) == 3 and len(sB) == 2:
            # 根据转置情况设置输出张量的维度
            if not tA and not tB:
                sout = (sA[0], sA[1], sB[1])
            elif tA and tB:
                sout = (sA[0], sA[2], sB[0])
            elif tA and not tB:
                sout = (sA[0], sA[2], sB[1])
            elif not tA and tB:
                sout = (sA[0], sA[1], sB[0])
        # 如果 A 和 B 的维度都为3
        elif len(sA) == 3 and len(sB) == 3:
            # 根据转置情况设置输出张量的维度
            if not tA and not tB:
                sout = (sA[0], sA[1], sB[2])
            elif tA and tB:
                sout = (sA[0], sA[2], sB[1])
            elif tA and not tB:
                sout = (sA[0], sA[2], sB[2])
            elif not tA and tB:
                sout = (sA[0], sA[1], sB[1])

    # 如果 correct 为 False，则抛出 ValueError 异常
    if not correct:
        raise ValueError(
            f"Tensor dimensions incorrect for matrix mulitiplication: A x B: {sA} x {sB} with transpose for A x B: {tA} x {tB}."
        )

    # 返回输出张量的维度
    return sout
def gemv_4bit(
    A: Tensor,
    B: Tensor,
    out: Optional[torch.Tensor] = None,
    transposed_A=False,
    transposed_B=False,
    state=None
):
    # 保存之前的设备信息
    prev_device = pre_call(A.device)
    # 检查矩阵乘法的输入和输出是否符合要求
    #sout = check_matmul(A, B, out, transposed_A, transposed_B, expected_type=A.dtype)
    
    # 如果状态为空，则抛出数值错误
    if state is None:
        raise ValueError('state cannot None. gem_4bit( ) requires the state from quantize_4bit( )')

    # 如果 A 的元素数量不等于最后一个维度的大小，则抛出数值错误
    if A.numel() != A.shape[-1]:
        raise ValueError('Dimensions of A are invalid. Must be a vector with the leading dimensions of "1", e.g. [1, 1, 2048]')

    # 获取状态的形状信息
    Bshape = state.shape
    bout = Bshape[0]
    absmax = state.absmax
    # 如果状态是嵌套的，则解量化 absmax，并加上偏移量
    if state.nested:
        absmax = dequantize_blockwise(state.absmax, state.state2)
        absmax += state.offset

    # 如果输出为空，则根据 A 的形状创建一个新的输出张量
    if out is None:
        if len(A.shape) == 3:
            out = torch.empty(size=(A.shape[0], A.shape[1], bout), dtype=A.dtype, device=A.device)
        else:
            out = torch.empty(size=(A.shape[0], bout), dtype=A.dtype, device=A.device)

    # 初始化矩阵乘法的参数
    n = 1
    m = Bshape[0]
    k = Bshape[1]
    lda = Bshape[0]
    ldc = Bshape[0]
    ldb = (A.shape[-1]+1)//2
    # 检查 B, A, out, absmax, state.code 是否在 GPU 上
    is_on_gpu([B, A, out, absmax, state.code])
    # 将参数转换为 ctypes.c_int32 类型
    m = ct.c_int32(m)
    n = ct.c_int32(n)
    k = ct.c_int32(k)
    lda = ct.c_int32(lda)
    ldb = ct.c_int32(ldb)
    ldc = ct.c_int32(ldc)
    # 检查 B 的数据类型是否在指定的数据类型列表中
    if B.dtype in [torch.uint8, torch.bfloat16, torch.float16, torch.float32]:
        # 如果 A 的数据类型是 torch.float16
        if A.dtype == torch.float16:
            # 调用底层库中的函数进行矩阵乘法运算，数据类型为 torch.float16
            lib.cgemm_4bit_inference_naive_fp16(m, n, k, get_ptr(A), get_ptr(B), get_ptr(absmax), get_ptr(state.code), get_ptr(out), lda, ldb, ldc, ct.c_int32(state.blocksize))
        # 如果 A 的数据类型是 torch.bfloat16
        elif A.dtype == torch.bfloat16:
            # 调用底层库中的函数进行矩阵乘法运算，数据类型为 torch.bfloat16
            lib.cgemm_4bit_inference_naive_bf16(m, n, k, get_ptr(A), get_ptr(B), get_ptr(absmax), get_ptr(state.code), get_ptr(out), lda, ldb, ldc, ct.c_int32(state.blocksize))
        # 如果 A 的数据类型是 torch.float32
        elif A.dtype == torch.float32:
            # 调用底层库中的函数进行矩阵乘法运算，数据类型为 torch.float32
            lib.cgemm_4bit_inference_naive_fp32(m, n, k, get_ptr(A), get_ptr(B), get_ptr(absmax), get_ptr(state.code), get_ptr(out), lda, ldb, ldc, ct.c_int32(state.blocksize))
        else:
            # 如果 A 的数据类型不在指定的数据类型列表中，则抛出异常
            raise NotImplementedError(f'Matmul not implemented for data type {A.dtype}')

    else:
        # 如果 B 的数据类型不在指定的数据类型列表中，则抛出异常
        raise NotImplementedError(f'Matmul not implemented for data type {A.dtype}')

    # 调用 post_call 函数，传入 prev_device 参数
    post_call(prev_device)

    # 返回结果矩阵 out
    return out
# 定义一个函数 igemm，用于执行整数矩阵乘法
def igemm(
    A: Tensor,  # 输入矩阵 A
    B: Tensor,  # 输入矩阵 B
    out: Optional[torch.Tensor] = None,  # 输出矩阵，可选
    transposed_A=False,  # 是否对矩阵 A 进行转置，默认为 False
    transposed_B=False,  # 是否对矩阵 B 进行转置，默认为 False
):
    # 检查矩阵乘法的参数，返回输出矩阵的形状
    sout = check_matmul(A, B, out, transposed_A, transposed_B)
    # 如果输出矩阵为空，则创建一个全零矩阵作为输出
    if out is None:
        out = torch.zeros(size=sout, dtype=torch.int32, device=A.device)
    # 如果输入矩阵 A 和 B 的维度均为 3，并且第一个维度和第三个维度匹配，则调用 batched_igemm 函数
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1]:
            return batched_igemm(A, B, out)

    # 获取输入矩阵 A 和 B 的形状
    sA = A.shape
    sB = B.shape
    # 如果需要对矩阵 A 进行转置，并且矩阵 A 的维度为 2，则调整形状
    if transposed_A and len(sA) == 2:
        sA = (sA[1], sA[0])
    # 如果需要对矩阵 A 进行转置，并且矩阵 A 的维度为 3，则调整形状
    elif transposed_A and len(sA) == 3:
        sA = (sA[0], sA[2], sA[0])
    # 如果需要对矩阵 B 进行转置，并且矩阵 B 的维度为 2，则调整形状
    if transposed_B and len(sB) == 2:
        sB = (sB[1], sB[0])
    # 如果需要对矩阵 B 进行转置，并且矩阵 B 的维度为 3，则调整形状
    elif transposed_B and len(sB) == 3:
        sB = (sB[0], sB[2], sB[0])
    
    # 以下注释解释了对矩阵乘法中矩阵维度的处理方式
    # cuBLAS 需要按列主序进行计算，而 PyTorch 使用行主序
    # 因此，为了执行矩阵乘法，需要对 A、B 和 C 矩阵进行处理
    # （行主序的转置是列主序）
    # 这意味着我们计算 B^T A^T = C^T，并且明确地交换每个矩阵的维度

    # 输入参数中矩阵的维度处理方式
    # 列主序：A @ B = C: [m, k] @ [k, n] = [m, n]
    # 行主序：B^T @ A^T = C^T: [m, k] @ [k, n] = [m, n]
    # 列主序与行主序布局：B^T @ A^T = C^T: [k, m] @ [n, k] = [n, m]
    # 检查输入张量 B 的维度是否为 2
    if len(sB) == 2:
        # 检查 B 的步长是否与第二维的长度相等，确定是否需要转置
        if B.stride()[0] == B.shape[1]:
            transposed_B = False
        elif B.stride()[1] == B.shape[0]:
            transposed_B = True
        # 检查输入张量 A 的维度
        if len(A.shape) == 2:
            # 检查 A 的步长是否与第二维的长度相等，确定是否需要转置
            if A.stride()[0] == A.shape[1]:
                transposed_A = False
            elif A.stride()[1] == A.shape[0]:
                transposed_A = True
        else:
            # 检查 A 的步长是否与第三维的长度相等，确定是否需要转置
            if A.stride()[1] == A.shape[2]:
                transposed_A = False
            elif A.stride()[2] == A.shape[1]:
                transposed_A = True

        # 根据输入张量 A 的维度确定 n 和 ldb
        if len(sA) == 2:
            n = sA[0]
            ldb = A.stride()[1 if transposed_A else 0]
        elif len(sA) == 3 and len(sB) == 2:
            n = sA[0] * sA[1]
            ldb = sA[2]

        # 确定 m, k, lda, ldc
        m = sB[1]
        k = sB[0]
        lda = B.stride()[(1 if transposed_B else 0)]
        ldc = sB[1]
    # 处理维度为 3 的情况
    elif len(sB) == 3:
        # 特殊情况，要求输入张量 A 和 B 的维度必须相等
        assert len(sA) == 3
        if not (sA[0] == sB[0] and sA[1] == sB[1]):
            raise ValueError(
                f"Only bsi,bso->io supported for tensor contractions, but dims for A x B were: {sA} x {sB}"
            )

        transposed_A = True
        transposed_B = False

        m = sB[2]
        n = sA[2]
        k = sB[0] * sB[1]

        lda = m
        ldb = sA[2]
        ldc = m

    # 获取 CUBLAS 上下文的指针
    ptr = CUBLAS_Context.get_instance().get_context(A.device)

    # 进行矩阵乘法运算
    # B^T @ A^T = C^T
    # [km, nk -> mn]
    is_on_gpu([B, A, out])
    lib.cigemm(ptr, ct.c_bool(transposed_B), ct.c_bool(transposed_A), ct.c_int32(m), ct.c_int32(n), ct.c_int32(k),
               get_ptr(B), get_ptr(A), get_ptr(out), ct.c_int32(lda), ct.c_int32(ldb), ct.c_int32(ldc))
    # 返回结果张量 out
    return out
# 批量进行整数矩阵乘法操作，支持输入参数 A、B、out，以及是否转置的标志
def batched_igemm(
    A: Tensor,
    B: Tensor,
    out: Optional[torch.Tensor] = None,
    transposed_A=False,
    transposed_B=False,
):
    # 检查输入的 A 和 B 张量是否为 3 维，如果不是则抛出数值错误
    if not len(A.shape) == 3 or not len(B.shape) == 3:
        raise ValueError(
            f"Expected 3-dimensional tensors for bmm, but got shapes A and B: {A.shape} and {B.shape}"
        )
    # 检查矩阵乘法的结果形状是否符合要求
    sout = check_matmul(A, B, out, transposed_A, transposed_B)
    # 如果输出张量 out 为空，则创建一个形状为 sout 的全零张量
    if out is None:
        out = torch.zeros(size=sout, dtype=torch.int32, device=A.device)

    # 检查 B 张量是否是连续的，根据不同情况设置 lda 和 transposed_A
    if B.is_contiguous():
        lda = B.stride()[1]
        transposed_A = False
    else:
        s = B.stride()
        if s[0] != B.shape[0]:
            B = B.contiguous()
            lda = B.stride()[1]
        elif s[2] == B.shape[1]:
            transposed_A = True
            lda = B.stride()[2]
        else:
            if s[2] == 1:
                B = B.contiguous()
                lda = B.stride()[1]
            elif s[1] == 1:
                B = B.contiguous()
                lda = B.stride()[1]
            else:
                B = B.contiguous()
                lda = B.stride()[1]

    # 检查 A 张量是否是连续的，根据不同情况设置 ldb 和 transposed_B
    if A.is_contiguous():
        ldb = A.stride()[1]
        transposed_B = False
    else:
        s = A.stride()
        if s[0] != A.shape[0]:
            A = A.contiguous()
            ldb = A.stride()[1]
            transposed_B = False
        elif s[2] == A.shape[1]:
            ldb = A.stride()[2]
            transposed_B = True
        else:
            A = A.contiguous()
            ldb = A.stride()[1]
            transposed_B = False

    # 这里是一个混乱的地方：cuBLAS 期望列优先，但 PyTorch 是行优先。
    # 因此，为了执行矩阵乘法，我们必须处理 A、B 和 C 矩阵（行优先的转置是列优先）
    # 这意味着我们计算 B^T A^T = C^T，并在输入参数中明确切换这些矩阵的维度
    # 这意味着我们计算 B^T A^T = C^T，并在输入参数中明确切换这些矩阵的维度

    # 列优先：A @ B = C: [batch, m, k] @ [batch, k, n] = [batch, m, n]
    # 获取输入张量 A 的批次数
    num_batch = A.shape[0]
    # 获取输入张量 A 的列数
    n = A.shape[1]
    # 获取输入张量 B 的行数
    m = B.shape[2]
    # 获取输入张量 B 的列数
    k = B.shape[1]

    # 设置输出张量 C 的列数
    ldc = m

    # 计算张量 B 在内存中的步长
    strideA = B.shape[1] * B.shape[2]
    # 计算张量 A 在内存中的步长
    strideB = A.shape[1] * A.shape[2]
    # 计算输出张量 C 在内存中的步长
    strideC = A.shape[1] * B.shape[2]

    # 获取 CUBLAS 上下文实例，并获取设备上下文
    ptr = CUBLAS_Context.get_instance().get_context(A.device)

    # 检查张量 B、A、out 是否在 GPU 上
    is_on_gpu([B, A, out])
    # 调用底层 C 函数进行批量整数矩阵乘法运算
    lib.cbatched_igemm(ptr, ct.c_bool(transposed_B), ct.c_bool(transposed_A), ct.c_int32(m), ct.c_int32(n), ct.c_int32(k),
               get_ptr(B), get_ptr(A), get_ptr(out), ct.c_int32(lda), ct.c_int32(ldb), ct.c_int32(ldc),
               ct.c_long(strideA), ct.c_long(strideB), ct.c_long(strideC), ct.c_uint32(num_batch))
    # 返回计算结果张量 out
    return out
# 定义一个函数，用于在 GPU 上执行整数矩阵乘法
def igemmlt(A, B, SA, SB, out=None, Sout=None, dtype=torch.int32):
    # 获取输入矩阵 A 的形状
    shapeA = SA[0]
    # 获取输入矩阵 B 的形状
    shapeB = SB[0]
    # 获取矩阵 A 的维度
    dimsA = len(shapeA)
    # 获取矩阵 B 的维度
    dimsB = len(shapeB)
    # 断言矩阵 B 的维度为 2，只支持二维矩阵
    assert dimsB == 2, 'Only two dimensional matrices are supported for argument B'
    # 根据矩阵 A 的维度确定 m 的值
    if dimsA == 2:
        m = shapeA[0]
    elif dimsA == 3:
        m = shapeA[0] * shapeA[1]

    # 获取矩阵 B 的行数
    rows = n = shapeB[0]
    # 断言输入矩阵 A 的所有维度都大于 0
    assert prod(list(shapeA)) > 0, f'Input tensor dimensions need to be > 0: {shapeA}'

    # 如果输入矩阵 A 为空，则返回一个空的张量，维度与输出张量相同
    if shapeA[0] == 0 and dimsA == 2:
        return torch.empty((0, shapeB[0]), device=A.device, dtype=torch.float16)
    elif shapeA[1] == 0 and dimsA == 3:
        return torch.empty(tuple(shapeA[:2] + [shapeB[0]]), device=A.device, dtype=torch.float16)

    # 根据输入矩阵 A 的维度和输出张量是否为空，获取变换缓冲区
    if dimsA == 2 and out is None:
        out, Sout = get_transform_buffer(
            (shapeA[0], shapeB[0]), dtype, A.device, "col32", "row"
        )
    elif dimsA == 3 and out is None:
        out, Sout = get_transform_buffer(
            (shapeA[0], shapeA[1], shapeB[0]), dtype, A.device, "col32", "row"
        )

    # 断言矩阵 B 的维度不为 3
    assert dimsB != 3, "len(B.shape)==3 not supported"
    # 断言输入矩阵 A 和 B 的设备类型为 CUDA
    assert A.device.type == "cuda"
    assert B.device.type == "cuda"
    # 断言输入矩阵 A 和 B 的数据类型为 torch.int8
    assert A.dtype == torch.int8
    assert B.dtype == torch.int8
    # 断言输出张量的数据类型为指定的数据类型
    assert out.dtype == dtype
    # 断言输入矩阵 A 的存储格式为 "col32"
    assert SA[1] == "col32"
    # 断言输入矩阵 B 的存储格式为 "col_turing" 或 "col_ampere"
    assert SB[1] in ["col_turing", "col_ampere"]
    # 断言输出张量的存储格式为 "col32"
    assert Sout[1] == "col32"
    # 断言矩阵 A 和 B 的最后一个维度相等
    assert shapeA[-1] == shapeB[-1], f"Matmullt only supports A @ B^T. Inner matrix dimensions do not match: A @ B = {shapeA} @ {shapeB}"
    # 获取矩阵 B 的存储格式
    formatB = SB[1]
    # 保存当前设备，设置当前设备为输入矩阵 A 的设备
    prev_device = A.device
    torch.cuda.set_device(A.device)

    # 获取 CUBLAS 上下文指针
    ptr = CUBLAS_Context.get_instance().get_context(A.device)
    # 获取输入矩阵 A 的指针
    ptrA = get_ptr(A)
    # 获取输入矩阵 B 的指针
    ptrB = get_ptr(B)
    # 获取输出张量的指针
    ptrC = get_ptr(out)

    # 获取矩阵 A 和 B 的最后一个维度
    k = shapeA[-1]
    # 设置 lda 为 m * 32
    lda = ct.c_int32(m * 32)
    if formatB == "col_turing":
        # 如果格式为 col_turing，则按照 turing 格式处理，即行数填充至 8 的倍数，列数为 32
        # n = 行数
        ldb = ct.c_int32(((rows + 7) // 8) * 8 * 32)
    else:
        # 如果格式不是 col_turing，则按照 ampere 格式处理，即行数填充至 32 的倍数，列数为 32
        # n = 行数
        ldb = ct.c_int32(((rows + 31) // 32) * 32 * 32)

    # 设置 ldc 为 m * 32
    ldc = ct.c_int32(m * 32)
    # 设置 m 为 ct.c_int32 类型的 m
    m = ct.c_int32(m)
    # 设置 n 为 ct.c_int32 类型的 n
    n = ct.c_int32(n)
    # 设置 k 为 ct.c_int32 类型的 k
    k = ct.c_int32(k)

    # 初始化错误标志为 0
    has_error = 0
    # 获取指向 None 的指针
    ptrRowScale = get_ptr(None)
    # 检查 A、B、out 是否在 GPU 上
    is_on_gpu([A, B, out])
    if formatB == 'col_turing':
        if dtype == torch.int32:
            # 调用 turing 格式下的 igemmlt_32 函数
            has_error = lib.cigemmlt_turing_32(
                ptr, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc
            )
        else:
            # 调用 turing 格式下的 igemmlt_8 函数
            has_error = lib.cigemmlt_turing_8(
                ptr, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc
            )
    elif formatB == "col_ampere":
        if dtype == torch.int32:
            # 调用 ampere 格式下的 igemmlt_32 函数
            has_error = lib.cigemmlt_ampere_32(
                ptr, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc
            )
        else:
            # 调用 ampere 格式下的 igemmlt_8 函数
            has_error = lib.cigemmlt_ampere_8(
                ptr, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc
            )

    # 如果 has_error 为 100（ERR_NOT_IMPLEMENTED 在 ops.cu 中定义为 100）
    if has_error == 100:
        # 抛出未实现错误
        raise NotImplementedError("igemmlt not available (probably built with NO_CUBLASLT)")

    # 如果 has_error 不为 0
    if has_error:
        # 打印相关信息并抛出异常
        print(f'A: {shapeA}, B: {shapeB}, C: {Sout[0]}; (lda, ldb, ldc): {(lda, ldb, ldc)}; (m, n, k): {(m, n, k)}')
        raise Exception('cublasLt ran into an error!')

    # 恢复之前的 GPU 设备
    torch.cuda.set_device(prev_device)

    # 返回 out 和 Sout
    return out, Sout
# 对输入矩阵进行去量化操作，返回去量化后的结果
def mm_dequant(
    A,  # 输入矩阵
    quant_state,  # 量化状态
    row_stats,  # 行统计信息
    col_stats,  # 列统计信息
    out=None,  # 输出矩阵，默认为None
    new_row_stats=None,  # 新的行统计信息，默认为None
    new_col_stats=None,  # 新的列统计信息，默认为None
    bias=None  # 偏置，默认为None
):
    assert A.dtype == torch.int32  # 断言输入矩阵数据类型为torch.int32
    if bias is not None: assert bias.dtype == torch.float16  # 如果存在偏置，则断言偏置数据类型为torch.float16
    out_shape = quant_state[0]  # 获取输出形状
    if len(out_shape) == 3:  # 如果输出形状长度为3
        out_shape = (out_shape[0] * out_shape[1], out_shape[2])  # 重置输出形状为展平后的形状

    if out is None:  # 如果输出矩阵为None
        out = torch.empty(out_shape, dtype=torch.float16, device=A.device)  # 创建与输入矩阵相同形状的空torch.float16类型的输出矩阵
    if new_row_stats is None:  # 如果新的行统计信息为None
        new_row_stats = torch.empty(  # 创建与输出矩阵行数相同的空torch.float32类型的新行统计信息
            out_shape[0], dtype=torch.float32, device=A.device
        )
    if new_col_stats is None:  # 如果新的列统计信息为None
        new_col_stats = torch.empty(  # 创建与输出矩阵列数相同的空torch.float32类型的新列统计信息
            out_shape[1], dtype=torch.float32, device=A.device
        )
    assert (
        new_row_stats.shape[0] == row_stats.shape[0]
    ), f"{new_row_stats.shape} vs {row_stats.shape}"  # 断言新的行统计信息长度与原行统计信息长度相同
    assert (
        new_col_stats.shape[0] == col_stats.shape[0]
    ), f"{new_col_stats.shape} vs {col_stats.shape}"  # 断言新的列统计信息长度与原列统计信息长度相同

    prev_device = pre_call(A.device)  # 调用pre_call函数，记录之前的设备
    ptrA = get_ptr(A)  # 获取输入矩阵的指针
    ptrOut = get_ptr(out)  # 获取输出矩阵的指针
    ptrRowStats = get_ptr(row_stats)  # 获取行统计信息的指针
    ptrColStats = get_ptr(col_stats)  # 获取列统计信息的指针
    ptrNewRowStats = get_ptr(new_row_stats)  # 获取新的行统计信息的指针
    ptrNewColStats = get_ptr(new_col_stats)  # 获取新的列统计信息的指针
    ptrBias = get_ptr(bias)  # 获取偏置的指针
    numRows = ct.c_int32(out_shape[0])  # 将输出矩阵行数转换为ct.c_int32类型
    numCols = ct.c_int32(out_shape[1])  # 将输出矩阵列数转换为ct.c_int32类型

    is_on_gpu([A, row_stats, col_stats, out, new_row_stats, new_col_stats, bias])  # 检查输入数据是否在GPU上
    lib.cdequant_mm_int32_fp16(ptrA, ptrRowStats, ptrColStats, ptrOut, ptrNewRowStats, ptrNewColStats, ptrBias, numRows, numCols)  # 调用cdequant_mm_int32_fp16函数进行矩阵去量化操作
    post_call(prev_device)  # 调用post_call函数，恢复之前的设备

    return out  # 返回输出矩阵


# 获取矩阵的列行绝对最大值
def get_colrow_absmax(
    A,  # 输入矩阵
    row_stats=None,  # 行统计信息，默认为None
    col_stats=None,  # 列统计信息，默认为None
    nnz_block_ptr=None,  # 非零块指针，默认为None
    threshold=0.0  # 阈值，默认为0.0
):
    assert A.dtype == torch.float16  # 断言输入矩阵数据类型为torch.float16
    device = A.device  # 获取输入矩阵的设备

    cols = A.shape[-1]  # 获取输入矩阵的列数
    if len(A.shape) == 3:  # 如果输入矩阵的维度为3
        rows = A.shape[0] * A.shape[1]  # 计算行数为第一维度和第二维度的乘积
    else:
        rows = A.shape[0]  # 否则行数为第一维度

    col_tiles = (cols + 255) // 256  # 计算列瓦片数
    tiled_rows = ((rows + 15) // 16) * 16  # 计算瓦片行数
    # 如果行统计为空，则创建一个指定大小的张量，并填充为-50000.0，设备为指定设备
    if row_stats is None:
        row_stats = torch.empty(
            (rows,), dtype=torch.float32, device=device
        ).fill_(-50000.0)
    # 如果列统计为空，则创建一个指定大小的张量，并填充为-50000.0，设备为指定设备
    if col_stats is None:
        col_stats = torch.empty(
            (cols,), dtype=torch.float32, device=device
        ).fill_(-50000.0)

    # 如果非零块指针为空且阈值大于0.0，则创建一个指定大小的张量，设备为指定设备
    if nnz_block_ptr is None and threshold > 0.0:
        nnz_block_ptr = torch.zeros(
            ((tiled_rows * col_tiles) + 1,), dtype=torch.int32, device=device
        )

    # 获取 A 张量的指针
    ptrA = get_ptr(A)
    # 获取行统计张量的指针
    ptrRowStats = get_ptr(row_stats)
    # 获取列统计张量的指针
    ptrColStats = get_ptr(col_stats)
    # 获取非零行指针的指针
    ptrNnzrows = get_ptr(nnz_block_ptr)
    # 创建包含行数的 c_int32 类型对象
    rows = ct.c_int32(rows)
    # 创建包含列数的 c_int32 类型对象
    cols = ct.c_int32(cols)

    # 在调用之前记录当前设备，切换到指定设备
    prev_device = pre_call(A.device)
    # 检查输入张量是否在 GPU 上
    is_on_gpu([A, row_stats, col_stats, nnz_block_ptr])
    # 调用 C 库函数，计算列和行的统计信息
    lib.cget_col_row_stats(ptrA, ptrRowStats, ptrColStats, ptrNnzrows, ct.c_float(threshold), rows, cols)
    # 调用之后恢复到之前的设备
    post_call(prev_device)

    # 如果阈值大于0.0，则对非零块指针进行累加操作
    if threshold > 0.0:
        nnz_block_ptr.cumsum_(0)

    # 返回行统计、列统计和非零块指针
    return row_stats, col_stats, nnz_block_ptr
class COOSparseTensor:
    # 定义 COO 格式的稀疏张量类
    def __init__(self, rows, cols, nnz, rowidx, colidx, values):
        # 断言确保输入的数据类型和维度符合要求
        assert rowidx.dtype == torch.int32
        assert colidx.dtype == torch.int32
        assert values.dtype == torch.float16
        assert values.numel() == nnz
        assert rowidx.numel() == nnz
        assert colidx.numel() == nnz

        # 初始化 COO 格式的稀疏张量对象
        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.rowidx = rowidx
        self.colidx = colidx
        self.values = values


class CSRSparseTensor:
    # 定义 CSR 格式的稀疏张量类
    def __init__(self, rows, cols, nnz, rowptr, colidx, values):
        # 断言确保输入的数据类型和维度符合要求
        assert rowptr.dtype == torch.int32
        assert colidx.dtype == torch.int32
        assert values.dtype == torch.float16
        assert values.numel() == nnz
        assert colidx.numel() == nnz
        assert rowptr.numel() == rows + 1

        # 初始化 CSR 格式的稀疏张量对象
        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.rowptr = rowptr
        self.colidx = colidx
        self.values = values


class CSCSparseTensor:
    # 定义 CSC 格式的稀疏张量类
    def __init__(self, rows, cols, nnz, colptr, rowidx, values):
        # 断言确保输入的数据类型和维度符合要求
        assert colptr.dtype == torch.int32
        assert rowidx.dtype == torch.int32
        assert values.dtype == torch.float16
        assert values.numel() == nnz
        assert rowidx.numel() == nnz
        assert colptr.numel() == cols + 1

        # 初始化 CSC 格式的稀疏张量对象
        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.colptr = colptr
        self.rowidx = rowidx
        self.values = values


def coo2csr(cooA):
    # 将 COO 格式的稀疏张量转换为 CSR 格式
    values, counts = torch.unique(cooA.rowidx, return_counts=True)
    values.add_(1)
    rowptr = torch.zeros(
        (cooA.rows + 1,), dtype=torch.int32, device=cooA.rowidx.device
    )
    rowptr.scatter_(index=values.long(), src=counts.int(), dim=0)
    rowptr.cumsum_(0)
    return CSRSparseTensor(
        cooA.rows, cooA.cols, cooA.nnz, rowptr, cooA.colidx, cooA.values
    )


def coo2csc(cooA):
    # 将 COO 格式的稀疏张量转换为 CSC 格式
    val, col2rowidx = torch.sort(cooA.colidx)
    rowidx = cooA.rowidx[col2rowidx]
    # 从稀疏矩阵 cooA 中获取指定列的数值
    values = cooA.values[col2rowidx]
    # 获取列中唯一值和对应出现次数
    colvalues, counts = torch.unique(val, return_counts=True)
    # 对列中的值进行加一操作
    colvalues.add_(1)
    # 创建一个全零张量作为列指针，数据类型为 int32，设备为 cooA 的列索引设备
    colptr = torch.zeros(
        (cooA.cols + 1,), dtype=torch.int32, device=cooA.colidx.device
    )
    # 将 counts 中的值根据 colvalues 的索引位置填充到 colptr 中
    colptr.scatter_(index=colvalues.long(), src=counts.int(), dim=0)
    # 对 colptr 进行累加操作
    colptr.cumsum_(0)
    # 返回一个 CSC 稀疏张量，包括行数、列数、非零元素个数、列指针、行索引和数值
    return CSCSparseTensor(
        cooA.rows, cooA.cols, cooA.nnz, colptr, rowidx, values
    )
# 创建一个稀疏张量的 COO 格式的零张量
def coo_zeros(rows, cols, nnz, device, dtype=torch.half):
    # 创建一个存储行索引的张量，全零，数据类型为 int32，存储设备为指定设备
    rowidx = torch.zeros((nnz,), dtype=torch.int32, device=device)
    # 创建一个存储列索引的张量，全零，数据类型为 int32，存储设备为指定设备
    colidx = torch.zeros((nnz,), dtype=torch.int32, device=device)
    # 创建一个存储值的张量，全零，数据类型为指定类型，存储设备为指定设备
    values = torch.zeros((nnz,), dtype=dtype, device=device)
    # 返回 COO 格式的稀疏张量
    return COOSparseTensor(rows, cols, nnz, rowidx, colidx, values)


# 对输入张量进行双精度量化
def double_quant(
    A, col_stats=None, row_stats=None, out_col=None, out_row=None, threshold=0.0
):
    # 获取输入张量的设备
    device = A.device
    # 断言输入张量的数据类型为半精度
    assert A.dtype == torch.half
    # 断言设备类型为 CUDA
    assert device.type == "cuda"
    # 在调用之前记录当前设备
    prev_device = pre_call(A.device)

    # 获取输入张量的列数
    cols = A.shape[-1]
    # 如果输入张量的维度为 3，则计算行数为第一维和第二维的乘积，否则行数为第一维
    if len(A.shape) == 3:
        rows = A.shape[0] * A.shape[1]
    else:
        rows = A.shape[0]

    # 如果未提供行统计或列统计信息，则计算绝对最大值和非零元素行指针
    if row_stats is None or col_stats is None:
        row_stats, col_stats, nnz_row_ptr = get_colrow_absmax(
            A, threshold=threshold
        )

    # 如果未提供输出列索引，则创建全零张量，数据类型为 int8，存储设备为指定设备
    if out_col is None:
        out_col = torch.zeros(A.shape, device=device, dtype=torch.int8)
    # 如果未提供输出行索引，则创建全零张量，数据类型为 int8，存储设备为指定设备
    if out_row is None:
        out_row = torch.zeros(A.shape, device=device, dtype=torch.int8)

    # 初始化 COO 张量为空
    coo_tensor = None
    # 获取输入张量的指针
    ptrA = get_ptr(A)
    # 获取列统计信息的指针
    ptrColStats = get_ptr(col_stats)
    # 获取行统计信息的指针
    ptrRowStats = get_ptr(row_stats)
    # 获取输出列索引的指针
    ptrOutCol = get_ptr(out_col)
    # 获取输出行索引的指针
    ptrOutRow = get_ptr(out_row)

    # 检查输入张量、列统计、行统计、输出列索引、输出行索引是否在 GPU 上
    is_on_gpu([A, col_stats, row_stats, out_col, out_row])
    # 如果阈值大于0.0
    if threshold > 0.0:
        # 获取非零元素的数量
        nnz = nnz_row_ptr[-1].item()
        # 如果非零元素数量大于0
        if nnz > 0:
            # 创建一个COO格式的稀疏张量，用于存储计算结果
            coo_tensor = coo_zeros(
                A.shape[0], A.shape[1], nnz_row_ptr[-1].item(), device
            )
            # 获取COO张量的行索引指针
            ptrRowIdx = get_ptr(coo_tensor.rowidx)
            # 获取COO张量的列索引指针
            ptrColIdx = get_ptr(coo_tensor.colidx)
            # 获取COO张量的值指针
            ptrVal = get_ptr(coo_tensor.values)
            # 获取非零元素行指针
            ptrRowPtr = get_ptr(nnz_row_ptr)

            # 调用C库函数进行双行列量化计算
            lib.cdouble_rowcol_quant(
                ptrA,
                ptrRowStats,
                ptrColStats,
                ptrOutCol,
                ptrOutRow,
                ptrRowIdx,
                ptrColIdx,
                ptrVal,
                ptrRowPtr,
                ct.c_float(threshold),
                ct.c_int32(rows),
                ct.c_int32(cols),
            )
            # 对COO张量的行索引进行排序
            val, idx = torch.sort(coo_tensor.rowidx)
            coo_tensor.rowidx = val
            coo_tensor.colidx = coo_tensor.colidx[idx]
            coo_tensor.values = coo_tensor.values[idx]
        else:
            # 如果非零元素数量为0，调用C库函数进行双行列量化计算
            lib.cdouble_rowcol_quant(
                ptrA,
                ptrRowStats,
                ptrColStats,
                ptrOutCol,
                ptrOutRow,
                None,
                None,
                None,
                None,
                ct.c_float(0.0),
                ct.c_int32(rows),
                ct.c_int32(cols),
            )
    else:
        # 如果阈值小于等于0.0，调用C库函数进行双行列量化计算
        lib.cdouble_rowcol_quant(
            ptrA,
            ptrRowStats,
            ptrColStats,
            ptrOutCol,
            ptrOutRow,
            None,
            None,
            None,
            None,
            ct.c_float(threshold),
            ct.c_int32(rows),
            ct.c_int32(cols),
        )
    # 调用后处理函数
    post_call(prev_device)

    # 返回计算结果
    return out_row, out_col, row_stats, col_stats, coo_tensor
# 对输入的矩阵进行转换操作，将其从指定顺序转换到另一种顺序
def transform(A, to_order, from_order='row', out=None, transpose=False, state=None, ld=None):
    # 保存函数调用前的设备状态
    prev_device = pre_call(A.device)
    # 如果状态为空，则初始化状态为输入矩阵的形状和原始顺序
    if state is None: state = (A.shape, from_order)
    else: from_order = state[1]
    # 如果输出为空，则获取转换后的缓冲区和新状态
    if out is None: out, new_state = get_transform_buffer(state[0], A.dtype, A.device, to_order, state[1], transpose)
    else: new_state = (state[0], to_order) # (shape, order)

    # 获取矩阵的形状
    shape = state[0]
    # 如果形状是二维的
    if len(shape) == 2:
        dim1 = ct.c_int32(shape[0])
        dim2 = ct.c_int32(shape[1])
    else:
        dim1 = ct.c_int32(shape[0] * shape[1])
        dim2 = ct.c_int32(shape[2])

    # 检查输入矩阵和输出矩阵是否在同一设备上
    is_on_gpu([A, out])
    # 根据目标顺序进行不同的转换操作
    if to_order == 'col32':
        if transpose:
            lib.ctransform_row2col32T(get_ptr(A), get_ptr(out), dim1, dim2)
        else:
            lib.ctransform_row2col32(get_ptr(A), get_ptr(out), dim1, dim2)
    elif to_order == "col_turing":
        if transpose:
            lib.ctransform_row2turingT(get_ptr(A), get_ptr(out), dim1, dim2)
        else:
            lib.ctransform_row2turing(get_ptr(A), get_ptr(out), dim1, dim2)
    elif to_order == "col_ampere":
        if transpose:
            lib.ctransform_row2ampereT(get_ptr(A), get_ptr(out), dim1, dim2)
        else:
            lib.ctransform_row2ampere(get_ptr(A), get_ptr(out), dim1, dim2)
    elif to_order == "row":
        if from_order == "col_turing":
            lib.ctransform_turing2row(get_ptr(A), get_ptr(out), dim1, dim2)
        elif from_order == "col_ampere":
            lib.ctransform_ampere2row(get_ptr(A), get_ptr(out), dim1, dim2)
    else:
        # 抛出未实现的错误
        raise NotImplementedError(f'Transform function not implemented: From {from_order} to {to_order}')

    # 恢复函数调用前的设备状态
    post_call(prev_device)

    # 返回输出矩阵和新状态
    return out, new_state


# 对 COO 格式的稀疏矩阵和稠密矩阵进行乘法操作
def spmm_coo(cooA, B, out=None):
    # 如果输出为空，则创建一个空的输出矩阵
    if out is None:
        out = torch.empty(
            (cooA.rows, B.shape[1]), device=B.device, dtype=B.dtype
        )
    # 获取 COO 矩阵的非零元素数量
    nnz = cooA.nnz
    # 断言 COO 矩阵的行索引长度等于非零元素数量
    assert cooA.rowidx.numel() == nnz
    # 断言 COO 矩阵的列索引长度等于非零元素数量
    assert cooA.colidx.numel() == nnz
    # 断言稀疏矩阵 cooA 的值的数量等于非零元素的数量
    assert cooA.values.numel() == nnz
    # 断言稀疏矩阵 cooA 的列数等于矩阵 B 的行数
    assert cooA.cols == B.shape[0]

    # 检查矩阵 B 是否是连续存储的，如果不是则需要转置
    transposed_B = False if B.is_contiguous() else True

    # 获取矩阵 B 的列偏移
    ldb = B.stride()[(1 if transposed_B else 0)]
    # 获取矩阵 C 的列数
    ldc = B.shape[1]

    # 获取 Cusparse_Context 的实例，并获取上下文指针
    ptr = Cusparse_Context.get_instance().context

    # 获取 cooA 的行索引、列索引、值、矩阵 B 和输出矩阵的指针
    ptrRowidx = get_ptr(cooA.rowidx)
    ptrColidx = get_ptr(cooA.colidx)
    ptrValues = get_ptr(cooA.values)
    ptrB = get_ptr(B)
    ptrC = get_ptr(out)
    # 将 cooA 的非零元素数量、行数、列数，矩阵 B 的列数、ldb、ldc 转换为 ctypes 类型
    cnnz = ct.c_int32(cooA.nnz)
    crowsA = ct.c_int32(cooA.rows)
    ccolsA = ct.c_int32(cooA.cols)
    ccolsB = ct.c_int32(B.shape[1])
    cldb = ct.c_int32(ldb)
    cldc = ct.c_int32(ldc)

    # 检查 cooA 的行索引、列索引、值、矩阵 B 和输出矩阵是否在 GPU 上
    is_on_gpu([cooA.rowidx, cooA.colidx, cooA.values, B, out])
    # 调用 C 库中的 cspmm_coo 函数进行稀疏矩阵乘法运算
    lib.cspmm_coo(ptr, ptrRowidx, ptrColidx, ptrValues, cnnz, crowsA, ccolsA, ccolsB, cldb, ptrB, cldc, ptrC, ct.c_bool(transposed_B))

    # 返回输出矩阵
    return out
def spmm_coo_very_sparse(cooA, B, dequant_stats=None, out=None):
    # 如果输出为空，则创建一个与 cooA 行数相同，列数与 B 的列数相同的全零张量
    if out is None:
        out = torch.zeros(
            (cooA.rows, B.shape[1]), device=B.device, dtype=cooA.values.dtype
        )
    # 获取 cooA 的非零元素个数
    nnz = cooA.nnz
    # 在调用函数之前记录 B 的设备
    prev_device = pre_call(B.device)
    # 断言 cooA 的行索引元素数量等于非零元素个数
    assert cooA.rowidx.numel() == nnz
    # 断言 cooA 的列索引元素数量等于非零元素个数
    assert cooA.colidx.numel() == nnz
    # 断言 cooA 的值元素数量等于非零元素个数
    assert cooA.values.numel() == nnz
    # 断言 cooA 的列数等于 B 的行数
    assert cooA.cols == B.shape[0], f"{cooA.cols} vs {B.shape}"

    # 检查 B 是否是连续的，如果不是则转置
    transposed_B = False if B.is_contiguous() else True

    # 获取 B 的 stride
    ldb = B.stride()[(1 if transposed_B else 0)]
    ldc = B.shape[1]

    # 获取 cooA 的行索引中唯一值和对应计数
    values, counts = torch.unique(cooA.rowidx, return_counts=True)
    # 计算偏移量
    offset = counts.cumsum(0).int()
    # 找到每行中最大的计数和对应的索引
    max_count, max_idx = torch.sort(counts, descending=True)
    max_idx = max_idx.int()
    max_count = max_count.int()
    # 断言每行最大计数不超过32
    assert (
        max_count[0] <= 32
    ), f"Current max count per row is 8 but found {max_count[0]}."
    # 断言 B 的数据类型为 torch.float16 或 torch.int8
    assert B.dtype in [torch.float16, torch.int8]
    # 获取偏移量的指针
    ptrOffset = get_ptr(offset)
    # 获取最大计数的指针
    ptrMaxCount = get_ptr(max_count)
    # 获取最大索引的指针
    ptrMaxIdx = get_ptr(max_idx)

    # 获取行索引、列索引、值、B、输出、dequant_stats 的指针
    ptrRowidx = get_ptr(cooA.rowidx)
    ptrColidx = get_ptr(cooA.colidx)
    ptrValues = get_ptr(cooA.values)
    ptrB = get_ptr(B)
    ptrC = get_ptr(out)
    ptrDequantStats = get_ptr(dequant_stats)
    # 获取 counts 的元素数量
    cnnz_rows = ct.c_int32(counts.numel())
    # 获取 cooA 的非零元素数量
    cnnz = ct.c_int32(cooA.nnz)
    # 将 cooA 的行数、列数、B 的行数、B 的列数转换为 c_int32 类型
    crowsA = ct.c_int32(cooA.rows)
    ccolsA = ct.c_int32(cooA.cols)
    crowsB = ct.c_int32(B.shape[1])
    ccolsB = ct.c_int32(B.shape[1])
    # 将 ldb 和 ldc 转换为 c_int32 类型
    cldb = ct.c_int32(ldb)
    cldc = ct.c_int32(ldc)

    # 检查输入数据是否在 GPU 上
    is_on_gpu([cooA.rowidx, cooA.colidx, cooA.values, B, out, dequant_stats])
    # 如果 B 的数据类型为 torch.float16，则调用相应的 C 函数
    lib.cspmm_coo_very_sparse_naive_fp16(
        ptrMaxCount,
        ptrMaxIdx,
        ptrOffset,
        ptrRowidx,
        ptrColidx,
        ptrValues,
        ptrB,
        ptrC,
        ptrDequantStats,
        cnnz_rows,
        cnnz,
        crowsA,
        crowsB,
        ccolsB,
    )
    # 如果 B 的数据类型是 torch.int8，则执行以下操作
    elif B.dtype == torch.int8:
        # 调用 C 库中的函数，对非常稀疏的 COO 格式进行稠密矩阵乘法运算，数据类型为 int8
        lib.cspmm_coo_very_sparse_naive_int8(
            ptrMaxCount,
            ptrMaxIdx,
            ptrOffset,
            ptrRowidx,
            ptrColidx,
            ptrValues,
            ptrB,
            ptrC,
            ptrDequantStats,
            cnnz_rows,
            cnnz,
            crowsA,
            crowsB,
            ccolsB,
        )
    # 如果 B 的数据类型不是 torch.int8，则抛出断言错误
    # else: assertion error
    post_call(prev_device)
    
    # 返回输出结果
    return out
# 定义一个常量 C，值为 127.0
C = 127.0

# 对输入的张量进行向量量化
def vectorwise_quant(x, dim=1, quant_type="vector"):
    # 如果量化类型为线性
    if quant_type == "linear":
        # 计算张量 x 的绝对值的最大值
        max1 = torch.abs(x).max().float()
        # 对 x 进行线性量化，将结果四舍五入为整数，并转换为 int8 类型
        xq = torch.round(x / max1 * 127).to(torch.int8)
        return xq, max1
    # 如果量化类型为向量或行
    elif quant_type in ["vector", "row"]:
        # 计算张量 x 沿指定维度的绝对值的最大值
        max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
        # 对 x 进行向量量化，将结果四舍五入为整数，并转换为 int8 类型
        xq = torch.round(x * (C / max1)).to(torch.int8)
        return xq, max1
    # 如果量化类型为零点量化
    elif quant_type == "zeropoint":
        # 将张量 x 转换为 float 类型
        dtype = x.dtype
        x = x.float()
        # 计算张量 x 的动态范围
        dyna = x.max() - x.min()
        if dyna == 0:
            dyna = 1
        # 计算量化因子 qx
        qx = 255.0 / dyna
        # 计算最小值 minx
        minx = x.min()
        # 计算零点 zpx
        zpx = torch.round(minx * qx)
        # 对 x 进行零点量化
        x = torch.round(qx * x - zpx) + zpx
        return x, qx
    # 如果量化类型为向量零点量化或行零点量化
    elif quant_type in ["vector-zeropoint", "row-zeropoint"]:
        # 将张量 x 转换为 float 类型
        dtype = x.dtype
        x = x.float()
        # 计算张量 x 沿指定维度的最大值和最小值之差
        dyna = torch.amax(x, dim=dim, keepdim=True) - torch.amin(x, dim=dim, keepdim=True)
        dyna[dyna == 0] = 1
        # 计算量化因子 qx
        qx = 255.0 / dyna
        # 计算最小值 minx
        minx = torch.amin(x, dim=dim, keepdim=True)
        # 计算零点 zpx
        zpx = torch.round(minx * qx)
        # 对 x 进行零点量化
        x = torch.round(qx * x - zpx) + zpx
        return x, qx
    # 如果量化类型为截断向量
    elif quant_type == "truncated-vector":
        # 禁用梯度计算
        with torch.no_grad():
            # 计算张量 x 的绝对值
            absx = torch.abs(x)
            # 计算张量 x 沿指定维度的绝对值的最大值
            max1 = torch.amax(absx, dim=dim, keepdim=True)
            # 将最大值缩小为原来的 70%
            max1 = max1 * 0.7
            # 找到需要截断的元素的索引
            idx = absx > max1.expand_as(absx)
            # 根据索引截断张量 x，并进行量化
            sign = torch.sign(x[idx])
            x[idx] = max1.expand_as(absx)[idx] * sign
            xq = torch.round(x / max1 * C).to(torch.int8)
        return xq, max1
    else:
        return None

# 对向量量化后的张量进行反量化
def vectorwise_dequant(xq, max1, quant_type="vector"):
    # 如果量化类型为向量
    if quant_type == "vector":
        # 对向量量化后的张量进行反量化
        x = (xq / C * max1).to(torch.float32)
        return x
    else:
        return None

# 对向量量化后的矩阵进行反量化
def vectorwise_mm_dequant(xq, S1, S2, dtype=torch.half, quant_type="vector"):
    # 如果量化类型是线性
    if quant_type == "linear":
        # 计算归一化系数
        norm = S1 * S2 / (C * C)
        # 防止溢出需要进行双重转换
        return (xq.float() * norm).to(dtype)
    # 如果量化类型是零点
    elif quant_type == "zeropoint":
        # 计算归一化系数
        norm = 1.0 / (S1 * S2)
        return (xq.float() * norm).to(dtype)
    # 如果量化类型是行零点
    elif quant_type == "row-zeropoint":
        # 计算归一化系数
        norm = 1.0 / (S1 * S2)
        x = xq.float()
        # 如果 S1 和 x 的维度匹配，则压缩 S1 的维度
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        # 如果 S2 和 x 的维度匹配，则压缩 S2 的维度
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        # 如果 S1 的维度是 2，则乘以归一化系数
        if len(S1.shape) == 2:
            x *= norm
        else:
            x *= norm
        return x.to(dtype)
    # 如果量化类型是向量零点
    elif quant_type == "vector-zeropoint":
        x = xq.float()
        # 如果 S1 和 x 的维度匹配，则压缩 S1 的维度
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        # 如果 S2 和 x 的维度匹配，则压缩 S2 的维度
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        # 如果 S1 的维度是 2，则乘以 1.0/S1
        if len(S1.shape) == 2:
            x *= 1.0 / S1
        else:
            x *= 1.0 / S1
        # 乘以 1.0/S2 转置
        x *= 1.0 / S2.t()
        return x.to(dtype)
    # 如果量化类型是行
    elif quant_type == "row":
        x = xq.float()
        # 如果 S1 和 x 的维度匹配，则压缩 S1 的维度
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        # 如果 S2 和 x 的维度匹配，则压缩 S2 的维度
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        # 如果 S1 的维度是 2，则乘以 S1 * S2 / (C * C)
        if len(S1.shape) == 2:
            x *= S1 * S2 / (C * C)
        else:
            x *= S1 * S2 / (C * C)
        return x.to(dtype)
    # 如果量化类型是截断向量或向量
    elif quant_type in ["truncated-vector", "vector"]:
        x = xq.float()
        # 如果 S1 和 x 的维度匹配，则压缩 S1 的维度
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        # 如果 S2 和 x 的维度匹配，则压缩 S2 的维度
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        # 如果 S1 的维度是 2，则乘以 S1 / C
        if len(S1.shape) == 2:
            x *= S1 / C
        else:
            x *= S1 / C
        # 乘以 S2 / C
        x *= S2 / C
        return x.to(dtype)
    else:
        return None
# 根据量化后的数据 xq、量化参数 A、B、缩放因子 SA、SB，将数据反量化为浮点数
def dequant_min_max(xq, A, B, SA, SB, dtype=torch.half):
    # 计算偏移量
    offset = B.float().t().sum(0) * (SA[0] + SA[1])
    # 将输入数据转换为浮点数
    x = xq.float()
    # 如果输入数据 xq 和缩放因子 SB 的维度分别为 2 和 3，则将 SB 的维度压缩为 2
    if len(xq.shape) == 2 and len(SB.shape) == 3:
        SB = SB.squeeze(0)
    # 根据 SB 的维度情况对输入数据进行缩放
    if len(SB.shape) == 2:
        x *= SB.t() / 127
    else:
        x *= SB / 127
    # 对数据进行进一步缩放
    x *= SA[1] / 127
    # 加上偏移量
    x += offset
    # 将数据转换为指定的数据类型
    return x.to(dtype)


# 根据输入数据 A、缩放因子 SA 和索引 idx，提取异常值
def extract_outliers(A, SA, idx):
    # 获取输入数据 A 的形状和格式
    shapeA = SA[0]
    formatA = SA[1]
    # 断言格式为 "col_turing" 或 "col_ampere"，并且数据在 CUDA 设备上
    assert formatA in ["col_turing", "col_ampere"]
    assert A.device.type == "cuda"

    # 创建与输入数据 A 相同形状的零张量 out
    out = torch.zeros(
        (shapeA[0], idx.numel()), dtype=torch.int8, device=A.device
    )

    # 获取索引的大小和 A 的行列数
    idx_size = ct.c_int32(idx.numel())
    rows = ct.c_int32(shapeA[0])
    cols = ct.c_int32(shapeA[1])
    ptrA = get_ptr(A)
    ptrIdx = get_ptr(idx)
    ptrOut = get_ptr(out)

    # 调用 C 函数前的准备工作
    prev_device = pre_call(A.device)
    # 根据数据格式调用不同的 C 函数提取异常值
    if formatA == 'col_turing':
        lib.cextractOutliers_turing(ptrA, ptrIdx, ptrOut, idx_size, rows, cols)
    elif formatA == "col_ampere":
        lib.cextractOutliers_ampere(ptrA, ptrIdx, ptrOut, idx_size, rows, cols)
    # 调用 C 函数后的清理工作
    post_call(prev_device)

    # 返回提取的异常值
    return out

# 对输入数据 A 进行管道测试，返回测试结果
def pipeline_test(A, batch_size):
    # 创建与输入数据 A 相同形状的零张量 out
    out = torch.zeros_like(A)
    # 调用 C 函数进行管道测试
    lib.cpipeline_test(get_ptr(A), get_ptr(out), ct.c_size_t(A.numel()), ct.c_size_t(batch_size))
    # 返回测试结果
    return out
```