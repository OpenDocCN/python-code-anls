# `.\pytorch\torch\testing\_internal\common_quantization.py`

```py
# mypy: ignore-errors
r"""Importing this file includes common utility methods and base clases for
checking quantization api and properties of resulting modules.
"""

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数模块
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd  # 导入动态量化的内部量化模块
import torch.ao.nn.quantized as nnq  # 导入量化模块
import torch.ao.nn.quantized.dynamic as nnqd  # 导入动态量化模块
from torch.ao.nn.intrinsic import _FusedModule  # 导入融合模块
import torch.distributed as dist  # 导入分布式训练模块
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM  # 导入测试相关的实用工具

from torch._export import capture_pre_autograd_graph  # 导入用于捕获自动求导图的函数
from torch.ao.quantization import (
    QuantType,  # 导入量化类型枚举
    default_dynamic_qat_qconfig,  # 导入默认动态量化训练后量化配置
    default_embedding_qat_qconfig,  # 导入默认嵌入量化训练后量化配置
    default_symmetric_qnnpack_qat_qconfig,  # 导入默认对称QNNPACK量化训练后量化配置
)
from torch.ao.quantization.quantize_pt2e import (
    _convert_to_reference_decomposed_fx,  # 导入将PT2E转换为参考分解FX格式的函数
    convert_pt2e,  # 导入转换PT2E格式的函数
    prepare_pt2e,  # 导入准备PT2E格式的函数
    prepare_qat_pt2e,  # 导入准备量化训练PT2E格式的函数
)
from torch.ao.quantization.backend_config import (
    get_executorch_backend_config,  # 导入获取ExecutorCh后端配置的函数
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,  # 导入XNNPACK量化器
    get_symmetric_quantization_config,  # 导入获取对称量化配置的函数
)
from torch.ao.quantization import (
    QuantWrapper, QuantStub, DeQuantStub,  # 导入量化包装器、量化存根和去量化存根
    default_qconfig, default_dynamic_qconfig, default_per_channel_qconfig, QConfig, default_observer, default_weight_observer,  # 导入默认量化配置、默认动态量化配置、默认通道量化配置、量化观察器和权重观察器
    propagate_qconfig_, convert, get_default_qconfig, quantize_dynamic_jit, quantize_jit, float_qparams_weight_only_qconfig,  # 导入配置传播、转换、获取默认量化配置、动态量化JIT、量化JIT、仅权重浮点QParams量化配置
    get_default_qat_qconfig, PerChannelMinMaxObserver, default_dynamic_quant_observer, quantize,  # 导入获取默认量化训练配置、通道最小最大观察器、默认动态量化观察器、量化函数
    QConfigMapping, get_default_qconfig_mapping, get_default_qat_qconfig_mapping,  # 导入量化配置映射、获取默认量化训练配置映射、获取默认量化训练后量化配置映射
)
from torch.ao.quantization.quantization_mappings import (
    get_default_dynamic_quant_module_mappings,  # 导入获取默认动态量化模块映射的函数
    get_default_qconfig_propagation_list,  # 导入获取默认量化配置传播列表的函数
    get_default_qat_module_mappings,  # 导入获取默认量化训练模块映射的函数
)
from torch.testing._internal.common_quantized import (
    override_quantized_engine,  # 导入重写量化引擎的函数
)
from torch.jit.mobile import _load_for_lite_interpreter  # 导入用于Lite解释器加载的函数

try:
    # graph mode quantization based on fx
    from torch.ao.quantization.quantize_fx import (
        prepare_fx,  # 导入准备FX格式的函数
        prepare_qat_fx,  # 导入准备量化训练FX格式的函数
        convert_fx,  # 导入转换FX格式的函数
        convert_to_reference_fx,  # 导入转换为参考FX格式的函数
    )
    from torch.ao.ns.fx.ns_types import NSSingleResultValuesType, NSSubgraph  # 导入FX命名空间相关类型
    from torch.fx.graph import Node  # 导入FX图节点
    from torch.fx import GraphModule  # 导入FX图模块
    HAS_FX = True  # 设置FX模块标志为True
except ImportError:
    HAS_FX = False  # 如果导入错误，设置FX模块标志为False

import copy  # 导入复制模块
import io  # 导入IO模块
import functools  # 导入函数工具模块
import time  # 导入时间模块
import os  # 导入操作系统模块

import unittest  # 导入单元测试模块
import numpy as np  # 导入NumPy模块
from torch.testing import FileCheck  # 导入用于文件检查的工具
from typing import Callable, Tuple, Dict, Any, Union, Type, Optional  # 导入类型注解

import torch._dynamo as torchdynamo  # 导入Torch的动态连接库

class NodeSpec:
    ''' Used for checking GraphModule Node
    '''
    def __init__(self, op, target):
        '''
        op: call_function | call_module
        target:
          for call_function, target would be a function
          for call_module, target would be the type of PyTorch module
        '''
        self.op = op  # 初始化操作类型（函数调用或模块调用）
        self.target = target  # 初始化目标对象（对于函数调用，目标是一个函数；对于模块调用，目标是PyTorch模块类型）

    @classmethod
    # 创建一个类方法，用于生成一个代表函数调用的 NodeSpec 对象
    def call_function(cls, target):
        return NodeSpec('call_function', target)

    # 创建一个类方法，用于生成一个代表方法调用的 NodeSpec 对象
    @classmethod
    def call_method(cls, target):
        return NodeSpec('call_method', target)

    # 创建一个类方法，用于生成一个代表模块调用的 NodeSpec 对象
    @classmethod
    def call_module(cls, target):
        return NodeSpec('call_module', target)

    # 定义对象的哈希方法，使得对象能够被哈希化
    def __hash__(self):
        return hash((self.op, self.target))

    # 定义对象的相等性方法，用于比较两个 NodeSpec 对象是否相等
    def __eq__(self, other):
        # 如果比较的对象不是 NodeSpec 类型，则返回 Not Implemented
        if not isinstance(other, NodeSpec):
            return NotImplemented
        
        # 比较操作符和目标是否相同
        return self.op == other.op and self.target == other.target

    # 定义对象的字符串表示方法，返回一个描述对象的字符串
    def __repr__(self):
        return repr(self.op) + " " + repr(self.target)
def get_supported_device_types():
    # 如果CUDA可用且不使用ROCm，则返回包含'cpu'和'cuda'的列表；否则只返回'cpu'
    return ['cpu', 'cuda'] if torch.cuda.is_available() and not TEST_WITH_ROCM else ['cpu']

def test_only_eval_fn(model, calib_data):
    r"""
    默认评估函数接受一个torch.utils.data.Dataset或输入张量列表，并在数据集上运行模型
    """
    for inp in calib_data:
        output = model(*inp)

_default_loss_fn = torch.nn.CrossEntropyLoss()
def test_only_train_fn(model, train_data, loss_fn=_default_loss_fn):
    r"""
    默认训练函数接受一个torch.utils.data.Dataset，并在数据集上训练模型
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, correct, total = 0, 0, 0
    for i in range(10):
        model.train()

        for data, target in train_data:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return train_loss, correct, total

class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """计算指定topk预测下的准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end='')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        if cnt >= ntrain_batches:
            return
    return

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    # 设置环境变量 MASTER_PORT 为 '12355'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化分布式进程组，使用 'gloo' 后端，指定当前进程的排名和总进程数
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
# 清理分布式数据并行的进程组
def ddp_cleanup():
    dist.destroy_process_group()

# 在分布式设置下运行分布式数据并行任务
def run_ddp(rank, world_size, prepared):
    # 设置分布式数据并行环境
    ddp_setup(rank, world_size)
    # 将准备好的模型移到 GPU
    prepared.cuda()
    # 使用分布式数据并行包装模型，指定设备 ID 为当前进程的 rank
    prepared = torch.nn.parallel.DistributedDataParallel(prepared, device_ids=[rank])
    prepared.to(rank)
    # 使用 SGD 优化器，学习率为 0.0001
    optimizer = torch.optim.SGD(model_with_ddp.parameters(), lr=0.0001)
    # 在数据集上训练一个 epoch
    train_one_epoch(model_with_ddp, criterion, optimizer, dataset, rank, 1)  # noqa: F821
    # 清理分布式数据并行的进程组
    ddp_cleanup()

# 将动态量化应用于模型的特定模块
def convert_dynamic(module):
    convert(module, get_default_dynamic_quant_module_mappings(), inplace=True)

# 准备动态量化，传播量化配置到模型
def prepare_dynamic(model, qconfig_dict=None):
    propagate_qconfig_(model, qconfig_dict)

# 创建一个用于卷积测试的输入张量
def _make_conv_test_input(
    batch_size, in_channels_per_group, input_feature_map_size,
    out_channels_per_group, groups, kernel_size, X_scale, X_zero_point, W_scale,
    W_zero_point, use_bias, use_channelwise,
):
    # 计算输入通道数和输出通道数
    in_channels = in_channels_per_group * groups
    out_channels = out_channels_per_group * groups
    
    # 生成随机整数张量 X_init，并将其缩放到量化值 X_q
    (X_value_min, X_value_max) = (0, 4)
    X_init = torch.randint(
        X_value_min, X_value_max,
        (batch_size, in_channels,) + input_feature_map_size)
    X = X_scale * (X_init - X_zero_point).float()
    X_q = torch.quantize_per_tensor(
        X, scale=X_scale, zero_point=X_zero_point, dtype=torch.quint8)
    
    # 缩放权重的规模和零点，确保与输出通道数相匹配
    W_scale = W_scale * out_channels
    W_zero_point = W_zero_point * out_channels
    W_scale = W_scale[:out_channels]  # 调整权重规模数组大小
    W_zero_point = W_zero_point[:out_channels]  # 调整权重零点数组大小
    
    # 为了测试，在权重和激活值中使用较小的值，避免 vpmaddubsw 指令的溢出
    (W_value_min, W_value_max) = (-5, 5)
    # 生成随机整数权重张量 W_init
    W_init = torch.randint(
        W_value_min, W_value_max,
        (out_channels, in_channels_per_group,) + kernel_size)
    # 生成随机整数偏置张量 b_init
    b_init = torch.randint(0, 10, (out_channels,))

    if use_channelwise:
        # 如果使用通道独立量化，则需要处理权重和偏置的缩放和零点
        W_shape = (-1, 1) + (1,) * len(kernel_size)
        W_scales_tensor = torch.tensor(W_scale, dtype=torch.float)
        W_zero_points_tensor = torch.tensor(W_zero_point, dtype=torch.float)
        # 缩放权重 W，并计算偏置 b
        W = W_scales_tensor.reshape(*W_shape) * (
            W_init.float() - W_zero_points_tensor.reshape(*W_shape)).float()
        b = X_scale * W_scales_tensor * b_init.float()
        # 对权重 W 进行通道独立量化得到 W_q
        W_q = torch.quantize_per_channel(
            W, W_scales_tensor.double(), W_zero_points_tensor.long(), 0,
            dtype=torch.qint8)
    else:
        # 计算量化后的权重 W_q
        W = W_scale[0] * (W_init - W_zero_point[0]).float()
        # 计算量化后的偏置 b，如果不使用偏置则为 None
        b = X_scale * W_scale[0] * b_init.float()
        # 对权重 W 进行量化，使用指定的量化参数：scale, zero_point 和数据类型 qint8
        W_q = torch.quantize_per_tensor(
            W, scale=W_scale[0], zero_point=W_zero_point[0], dtype=torch.qint8)

    # 返回 X, X_q, W, W_q 和可能的偏置 b（如果使用偏置的话）
    return (X, X_q, W, W_q, b if use_bias else None)
# 定义函数，创建一个卷积的附加输入张量
def _make_conv_add_extra_input_tensor(scale, zero_point, sizes):
    # 定义输入张量的取值范围
    (X_value_min, X_value_max) = (0, 4)
    # 生成随机整数张量，大小由sizes参数推断
    X_init = torch.randint(
        X_value_min,
        X_value_max,
        sizes  # 推断张量大小以执行加法
    )
    # 对生成的随机整数张量进行量化操作，计算量化后的张量X_q
    X = scale * (X_init - zero_point).float()
    X_q = torch.quantize_per_tensor(
        X, scale=scale, zero_point=zero_point, dtype=torch.quint8)
    return X, X_q

# 装饰器函数，如果没有FBGEMM支持，则跳过测试
def skipIfNoFBGEMM(fn):
    # 设置跳过测试的原因
    reason = 'Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs with instruction set support AVX2 or newer.'
    if isinstance(fn, type):
        # 如果fn是类，则检查支持的引擎中是否包含fbgemm，若不包含，则跳过该类
        if 'fbgemm' not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # 检查支持的引擎中是否包含fbgemm，若不包含，则引发SkipTest异常
        if 'fbgemm' not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)
    return wrapper

# 装饰器函数，如果没有QNNPACK支持，则跳过测试
def skipIfNoQNNPACK(fn):
    # 设置跳过测试的原因
    reason = 'Quantized operations require QNNPACK.'
    if isinstance(fn, type):
        # 如果fn是类，则检查支持的引擎中是否包含qnnpack，若不包含，则跳过该类
        if 'qnnpack' not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # 检查支持的引擎中是否包含qnnpack，若不包含，则引发SkipTest异常；否则使用qnnpack引擎执行测试
        if 'qnnpack' not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        with override_quantized_engine('qnnpack'):
            fn(*args, **kwargs)

    return wrapper

# 装饰器函数，如果没有ONEDNN支持，则跳过测试
def skipIfNoONEDNN(fn):
    # 设置跳过测试的原因
    reason = 'Quantized operations require ONEDNN.'
    if isinstance(fn, type):
        # 如果fn是类，则检查支持的引擎中是否包含onednn，若不包含，则跳过该类
        if 'onednn' not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # 检查支持的引擎中是否包含onednn，若不包含，则引发SkipTest异常；否则执行测试
        if 'onednn' not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)
    return wrapper

# 待补充：装饰器函数，如果没有ONEDNN BF16支持，则跳过测试
def skipIfNoONEDNNBF16(fn):
    reason = 'Quantized operations require BF16 support.'
    # 待补充
    # 检查 fn 是否为 type 类型
    if isinstance(fn, type):
        # 如果不支持 MKL-DNN BF16，设置 fn 的 __unittest_skip__ 属性为 True
        # 并且设置跳过测试的原因为给定的 reason
        if not torch.ops.mkldnn._is_mkldnn_bf16_supported():
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        # 返回 fn 函数
        return fn

    # 使用 functools 模块的 wraps 装饰器，确保 wrapper 函数与 fn 函数具有相同的文档字符串和名称等属性
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # 如果不支持 MKL-DNN BF16，抛出 unittest.SkipTest 异常，表示跳过测试，原因是 reason
        if not torch.ops.mkldnn._is_mkldnn_bf16_supported():
            raise unittest.SkipTest(reason)
        else:
            # 否则执行 fn 函数，传入参数 args 和 kwargs
            fn(*args, **kwargs)
    # 返回 wrapper 函数
    return wrapper
# 定义装饰器函数，根据系统支持情况跳过测试函数
def skipIfNoX86(fn):
    # 提示消息：量化操作需要 X86 支持
    reason = 'Quantized operations require X86.'
    # 如果被装饰的对象是类
    if isinstance(fn, type):
        # 如果 X86 不在支持的量化引擎列表中
        if 'x86' not in torch.backends.quantized.supported_engines:
            # 标记该类为跳过测试
            fn.__unittest_skip__ = True
            # 设置跳过原因
            fn.__unittest_skip_why__ = reason
        return fn
    
    # 如果被装饰的对象是函数
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # 如果 X86 不在支持的量化引擎列表中
        if 'x86' not in torch.backends.quantized.supported_engines:
            # 抛出测试跳过异常，并提供原因
            raise unittest.SkipTest(reason)
        else:
            # 否则执行原函数
            fn(*args, **kwargs)
    return wrapper

# 定义装饰器函数，根据 Dynamo 支持情况跳过测试函数
def skipIfNoDynamoSupport(fn):
    # 提示消息：Dynamo 不支持
    reason = "dynamo doesn't support."
    if isinstance(fn, type):
        # 如果 Dynamo 不支持
        if not torchdynamo.is_dynamo_supported():
            # 标记该类为跳过测试
            fn.__unittest_skip__ = True
            # 设置跳过原因
            fn.__unittest_skip_why__ = reason
        return fn
    
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # 如果 Dynamo 不支持
        if not torchdynamo.is_dynamo_supported():
            # 抛出测试跳过异常，并提供原因
            raise unittest.SkipTest(reason)
        else:
            # 否则执行原函数
            fn(*args, **kwargs)
    return wrapper

# 定义装饰器函数，根据 Inductor 支持情况跳过测试函数
def skipIfNoInductorSupport(fn):
    # 提示消息：Inductor 不支持
    reason = "inductor doesn't support."
    if isinstance(fn, type):
        # 如果 Inductor 不支持
        if not torchdynamo.is_inductor_supported():
            # 标记该类为跳过测试
            fn.__unittest_skip__ = True
            # 设置跳过原因
            fn.__unittest_skip_why__ = reason
        return fn
    
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # 如果 Inductor 不支持
        if not torchdynamo.is_inductor_supported():
            # 抛出测试跳过异常，并提供原因
            raise unittest.SkipTest(reason)
        else:
            # 否则执行原函数
            fn(*args, **kwargs)
    return wrapper

# 尝试导入 torchvision 模块，设置是否存在的标志
try:
    import torchvision  # noqa: F401
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# 根据是否导入了 torchvision 设置跳过条件的装饰器
skip_if_no_torchvision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

# 将模型转换为脚本模块，根据参数确定是否追踪执行
def get_script_module(model, tracing, data):
    return torch.jit.trace(model, data) if tracing else torch.jit.script(model)

# 将长度转换为偏移量，用于 embedding_bag
def lengths_to_offsets(t, offset_type=np.int64, use_begin_offset=True):
    """
    Convert lengths to offsets for embedding_bag
    """
    # 创建全零数组，长度比 t 长度大一，数据类型为 offset_type
    tt = np.zeros((t.shape[0] + 1,), dtype=offset_type)
    # 将 t 的值复制给 tt 的后续部分
    tt[1:] = t
    # 计算累积和，转换为 Torch 张量
    tt = torch.from_numpy(np.cumsum(tt, dtype=offset_type))
    if use_begin_offset:
        return tt[:-1]  # 返回除最后一个元素外的所有元素作为偏移量
    return tt[1:]  # 返回从第二个元素开始到最后一个元素作为偏移量

# 将张量按组进行量化
def _group_quantize_tensor(w, n_bit=4, q_group_size=16):
    assert w.dim() == 2  # 断言 w 是二维张量
    w = w.transpose(0, 1).contiguous()  # 转置张量的第一和第二维，并保持内存连续性
    assert q_group_size > 1  # 断言 q_group_size 大于 1
    assert w.shape[-1] % q_group_size == 0  # 断言 w 的最后一维可以被 q_group_size 整除

    # 将 w 重塑为形状为 (-1, q_group_size) 的张量
    to_quant = w.reshape(-1, q_group_size)
    assert torch.isnan(to_quant).sum() == 0  # 断言 to_quant 中没有 NaN 值

    # 计算每组的最大值和最小值
    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2 ** n_bit - 1  # 计算最大整数值
    min_int = 0  # 最小整数值为 0

    # 计算比例因子
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    assert torch.isnan(scales).sum() == 0  # 断言 scales 中没有 NaN 值

    # 计算偏置值
    zeros = min_val + scales * (2 ** (n_bit - 1))
    assert torch.isnan(zeros).sum() == 0  # 断言 zeros 中没有 NaN 值

    # 对输入张量进行量化操作，并保证在指定范围内
    out = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int)
    assert torch.isnan(out).sum() == 0  # 断言 out 中没有 NaN 值

    # 将结果转换为 torch.int32 数据类型，并重塑回原始形状
    out = out.to(dtype=torch.int32).reshape(w.shape)

    # Scales 和 zeros 对于同一量化组应该是连续的，因此我们可以
    # 将张量 scales 重新视图为指定形状，第一维度为 w 的第一维度，第二维度为 -1（自动推断）
    scales = scales.view(w.shape[0], -1)
    # 将张量 zeros 重新视图为指定形状，第一维度为 w 的第一维度，第二维度为 -1（自动推断）
    zeros = zeros.view(w.shape[0], -1)
    # 创建 scales 和 zeros 的组合张量，每个张量被重塑为三维张量，然后串联在一起，最后转置
    scales_and_zeros = (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),  # 将 scales 重塑为三维张量
                zeros.reshape(zeros.size(0), zeros.size(1), 1),      # 将 zeros 重塑为三维张量
            ],
            2,  # 在第二维度上串联
        ).transpose(0, 1).contiguous()  # 转置张量的维度 0 和 1，并保持内存连续性
    )

    return out, scales_and_zeros
# 定义一个函数用于按通道动态量化张量 x
def _dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # 引用来源：https://github.com/pytorch-labs/gpt-fast/blob/main/quantize.py
    # 将输入张量 x 的数据类型存储起来
    x_dtype = x.dtype
    # 将输入张量 x 转换为 float 类型
    x = x.float()
    # 获取浮点数类型的最小正数值
    eps = torch.finfo(torch.float32).eps

    # 计算张量 x 沿着第一个维度的最小值和最大值
    min_val, max_val = torch.aminmax(x, dim=1)

    # 根据最小值和最大值计算量化的尺度 scales 和零点 zero_points
    # 参考链接：https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # 参考链接：https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    # 确保 scales 与原始张量的数据类型一致
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # 根据量化的最小值、最大值、尺度 scales 和零点 zero_points 进行量化
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    # 返回量化后的结果 quant、尺度 scales（转换为原始张量的数据类型）、零点 zero_points
    return quant, scales.to(x_dtype), zero_points



# QuantizationTestCase 用作测试模块上的量化的基类
class QuantizationTestCase(TestCase):
    def setUp(self):
        super().setUp()
        # 设置用于测试的校准数据，每个元素为一个包含随机张量的列表
        self.calib_data = [[torch.rand(2, 5, dtype=torch.float)] for _ in range(2)]
        # 设置用于训练的数据，每个元素为一个包含随机张量和随机整数张量的列表
        self.train_data = [[torch.rand(2, 5, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long)] for _ in range(2)]
        # 设置用于测试的一维图像数据，每个元素为一个包含随机三维张量的列表
        self.img_data_1d = [[torch.rand(2, 3, 10, dtype=torch.float)] for _ in range(2)]
        # 设置用于测试的二维图像数据，每个元素为一个包含随机四维张量的列表
        self.img_data_2d = [[torch.rand(1, 3, 10, 10, dtype=torch.float)] for _ in range(2)]
        # 设置用于测试的三维图像数据，每个元素为一个包含随机五维张量的列表
        self.img_data_3d = [[torch.rand(1, 3, 5, 5, 5, dtype=torch.float)] for _ in range(2)]
        # 设置用于训练的一维图像数据，每个元素为一个包含随机三维张量和随机整数张量的列表
        self.img_data_1d_train = [[torch.rand(2, 3, 10, dtype=torch.float),
                                   torch.randint(0, 1, (1,), dtype=torch.long)] for _ in range(2)]
        # 设置用于训练的二维图像数据，每个元素为一个包含随机四维张量和随机整数张量的列表
        self.img_data_2d_train = [[torch.rand(1, 3, 10, 10, dtype=torch.float),
                                   torch.randint(0, 1, (1,), dtype=torch.long)] for _ in range(2)]
        # 设置用于训练的三维图像数据，每个元素为一个包含随机五维张量和随机整数张量的列表
        self.img_data_3d_train = [[torch.rand(1, 3, 5, 5, 5, dtype=torch.float),
                                   torch.randint(0, 1, (1,), dtype=torch.long)] for _ in range(2)]

        # 图像数据字典，键为整数，值为相应的图像数据列表
        self.img_data_dict = {1 : self.img_data_1d,
                              2 : self.img_data_2d,
                              3 : self.img_data_3d}

        # 用于静态量化操作的量化类型列表
        self.static_quant_types = [QuantType.STATIC, QuantType.QAT]
        # 所有用于图模式量化（基于固定点的）的量化类型列表
        self.all_quant_types = [QuantType.DYNAMIC, QuantType.STATIC, QuantType.QAT]

    def checkNoPrepModules(self, module):
        r"""检查模块不包含用于量化准备的子模块，例如 quant、dequant 和 observer"""
        self.assertFalse(hasattr(module, 'quant'))
        self.assertFalse(hasattr(module, 'dequant'))

    def checkNoQconfig(self, module):
        r"""检查模块不包含 qconfig"""
        self.assertFalse(hasattr(module, 'qconfig'))

        # 递归检查模块的所有子模块
        for child in module.children():
            self.checkNoQconfig(child)

    def checkHasPrepModules(self, module):
        r"""检查模块包含用于量化准备的子模块，例如 quant、dequant 和 observer"""
        self.assertTrue(hasattr(module, 'module'))
        self.assertTrue(hasattr(module, 'quant'))
        self.assertTrue(hasattr(module, 'dequant'))
    # 检查模块或其叶子后代是否具有观察器，为量化准备做检查
    def checkObservers(self, module, propagate_qconfig_list=None, prepare_custom_config_dict=None):
        r"""Checks the module or module's leaf descendants
            have observers in preparation for quantization
        """
        # 如果未提供传播的量化配置列表，则使用默认的量化配置传播列表
        if propagate_qconfig_list is None:
            propagate_qconfig_list = get_default_qconfig_propagation_list()
        # 如果未提供自定义配置字典，则使用空字典
        if prepare_custom_config_dict is None:
            prepare_custom_config_dict = {}
        # 从自定义配置字典中获取浮点类型到观察模块类的映射关系
        float_to_observed_module_class_mapping = prepare_custom_config_dict.get("float_to_observed_custom_module_class", {})

        # 检查模块是否为叶子模块，忽略 activation_post_process 属性
        def is_leaf_module(module):
            submodule_name_count = 0
            # 遍历模块的子模块
            for name, _ in module.named_children():
                if name != 'activation_post_process':
                    submodule_name_count += 1
            return submodule_name_count == 0

        # 如果模块具有 qconfig 属性且不为 None，并且满足以下条件之一：
        # 1. 是叶子模块，不是 torch.nn.Sequential 的实例，并且其类型在传播的量化配置列表中；
        # 2. 模块类型在浮点类型到观察模块类的映射关系的键中；
        # 并且不是 torch.ao.quantization.DeQuantStub 的实例，则执行断言
        if hasattr(module, 'qconfig') and module.qconfig is not None and \
           ((is_leaf_module(module) and not isinstance(module, torch.nn.Sequential)
            and type(module) in propagate_qconfig_list) or
           type(module) in float_to_observed_module_class_mapping.keys()) and \
           not isinstance(module, torch.ao.quantization.DeQuantStub):
            self.assertTrue(hasattr(module, 'activation_post_process'),
                            'module: ' + str(type(module)) + ' do not have observer')
        
        # 对于非量化训练模块的子模块，无需检查观察器
        if type(module) not in get_default_qat_module_mappings().values() and \
           type(module) not in float_to_observed_module_class_mapping.values() and \
           not isinstance(module, _FusedModule):
            # 遍历模块的子模块，排除特定类型如 nn.Dropout
            for child in module.children():
                if type(child) in [nn.Dropout]:
                    continue
                # 递归调用检查观察器函数
                self.checkObservers(child, propagate_qconfig_list, prepare_custom_config_dict)

    # 检查模块是否包含 nn.Quantize 和 nn.DeQuantize 子模块
    def checkQuantDequant(self, mod):
        r"""Checks that mod has nn.Quantize and
            nn.DeQuantize submodules inserted
        """
        self.assertEqual(type(mod.quant), nnq.Quantize)
        self.assertEqual(type(mod.dequant), nnq.DeQuantize)

    # 检查模块是否被替换为 nnq.Linear 模块，并且包含 Quantize 和 DeQuantize 子模块
    def checkWrappedQuantizedLinear(self, mod):
        r"""Checks that mod has been swapped for an nnq.Linear
            module, the bias is qint32, and that the module
            has Quantize and DeQuantize submodules
        """
        self.assertEqual(type(mod.module), nnq.Linear)
        self.checkQuantDequant(mod)

    # 检查模块是否为 nnq.Linear 类型
    def checkQuantizedLinear(self, mod):
        self.assertEqual(type(mod), nnq.Linear)

    # 检查动态量化的模块是否被替换为 nnqd.Linear 模块，并且偏置为指定的 dtype 类型
    def checkDynamicQuantizedLinear(self, mod, dtype):
        r"""Checks that mod has been swapped for an nnqd.Linear
            module, the bias is float.
        """
        self.assertEqual(type(mod), nnqd.Linear)
        self.assertEqual(mod._packed_params.dtype, dtype)
    def checkDynamicQuantizedLinearRelu(self, mod, dtype):
        r"""Checks that mod has been swapped for an nnqd.Linear
            module, the bias is float.
        """
        # 断言 mod 的类型为 nnqd.LinearReLU
        self.assertEqual(type(mod), nniqd.LinearReLU)
        # 断言 mod 的 _packed_params 的数据类型为指定的 dtype
        self.assertEqual(mod._packed_params.dtype, dtype)

    def check_eager_serialization(self, ref_model, loaded_model, x):
        # Check state dict serialization and torch.save APIs
        # 获取参考模型的状态字典
        model_dict = ref_model.state_dict()
        # 创建一个字节流对象
        b = io.BytesIO()
        # 将模型的状态字典保存到字节流中
        torch.save(model_dict, b)
        b.seek(0)
        # 从字节流中加载模型状态字典
        loaded_dict = torch.load(b)
        # 加载模型的状态字典到加载的模型中
        loaded_model.load_state_dict(loaded_dict)
        # 对参考模型和加载后的模型进行前向传播计算
        ref_out = ref_model(*x)
        load_out = loaded_model(*x)

        def check_outputs(ref_out, load_out):
            # 断言前向传播输出的第一个元素相等
            self.assertEqual(ref_out[0], load_out[0])
            # 如果第二个输出是元组，则分别断言其元素相等，否则直接断言相等
            if isinstance(ref_out[1], tuple):
                self.assertEqual(ref_out[1][0], load_out[1][0])
                self.assertEqual(ref_out[1][1], load_out[1][1])
            else:
                self.assertEqual(ref_out[1], load_out[1])

        # 检查前向传播输出是否一致
        check_outputs(ref_out, load_out)
        # 再次使用字节流对象保存参考模型
        b = io.BytesIO()
        torch.save(ref_model, b)
        b.seek(0)
        # 从字节流中加载模型
        loaded = torch.load(b)
        # 对加载后的模型进行前向传播计算
        load_out = loaded(*x)
        # 再次检查前向传播输出是否一致
        check_outputs(ref_out, load_out)

    def check_weight_bias_api(self, ref_model, weight_keys, bias_keys):
        # 获取参考模型的权重和偏置
        weight = ref_model.get_weight()
        bias = ref_model.get_bias()
        # 断言参考模型的权重键和给定的权重键集合相等
        self.assertEqual(weight_keys ^ weight.keys(), set())
        # 断言参考模型的偏置键和给定的偏置键集合相等
        self.assertEqual(bias_keys ^ bias.keys(), set())

    def checkDynamicQuantizedLSTM(self, mod, reference_module_type, dtype):
        r"""Checks that mod has been swapped for an nnqd.LSTM type
            module, the bias is float.
        """
        # 定义权重数据类型映射表
        wt_dtype_map = {torch.qint8: 'quantized_dynamic', torch.float16: 'quantized_fp16'}
        # 断言 mod 的类型为指定的 reference_module_type
        self.assertEqual(type(mod), reference_module_type)
        # 遍历模型的所有权重值，并断言其数据类型与指定的 dtype 对应的映射一致
        for packed_params in mod._all_weight_values:
            self.assertEqual(packed_params.param.__getstate__()[0][0], wt_dtype_map[dtype])

    def checkLinear(self, mod):
        # 断言 mod 的类型为 torch.nn.Linear
        self.assertEqual(type(mod), torch.nn.Linear)

    def checkDynamicQuantizedModule(self, mod, reference_module_type, dtype):
        r"""Checks that mod has been swapped for an nnqd.Linear
            module, the bias is float.
        """
        # 定义权重数据类型映射表
        wt_dtype_map = {torch.qint8: 'quantized_dynamic', torch.float16: 'quantized_fp16'}
        # 断言 mod 的类型为指定的 reference_module_type
        self.assertEqual(type(mod), reference_module_type)
        # 如果模型具有 _all_weight_values 属性，则继续检查其权重数据类型是否与 dtype 对应的映射一致
        if hasattr(mod, '_all_weight_values'):
            for packed_params in mod._all_weight_values:
                self.assertEqual(packed_params.param.__getstate__()[0][0], wt_dtype_map[dtype])
    # 检查是否可以进行脚本化（scriptable）处理，即将原始模型orig_mod转换为脚本化或追踪化模型
    def checkScriptable(self, orig_mod, calib_data, check_save_load=False):
        # 使用 torch.jit.script 将原始模型脚本化
        scripted = torch.jit.script(orig_mod)
        # 调用内部方法 _checkScriptable 来验证脚本化后的模型
        self._checkScriptable(orig_mod, scripted, calib_data, check_save_load)

        # 使用 torch.jit.trace 将原始模型追踪化
        traced = torch.jit.trace(orig_mod, calib_data[0])
        # 再次调用内部方法 _checkScriptable 来验证追踪化后的模型
        self._checkScriptable(orig_mod, traced, calib_data, check_save_load)

    # 内部方法：检查脚本化或追踪化后的模型是否正确
    def _checkScriptable(self, orig_mod, script_mod, calib_data, check_save_load):
        # 验证脚本化或追踪化后的模型与原始模型在给定数据上的输出是否一致
        self._checkModuleCorrectnessAgainstOrig(orig_mod, script_mod, calib_data)

        # 测试模型的保存和加载功能
        buffer = io.BytesIO()
        # 将脚本化或追踪化后的模型保存到字节流中
        torch.jit.save(script_mod, buffer)

        buffer.seek(0)
        # 从字节流中加载模型
        loaded_mod = torch.jit.load(buffer)
        # 如果需要检查保存和加载功能，则再次验证加载后的模型与原始模型的一致性
        if check_save_load:
            self._checkModuleCorrectnessAgainstOrig(orig_mod, loaded_mod, calib_data)

    # 内部方法：验证模型是否正确处理给定数据
    def _checkModuleCorrectnessAgainstOrig(self, orig_mod, test_mod, calib_data):
        # 对于每一个输入数据 inp，比较原始模型和测试模型的输出是否相等
        for inp in calib_data:
            ref_output = orig_mod(*inp)
            scripted_output = test_mod(*inp)
            # 使用断言检查两个输出是否相等
            self.assertEqual(scripted_output, ref_output)
    # 检查图模式操作的函数，用于量化模型中的操作
    def checkGraphModeOp(self, module, inputs, quantized_op, tracing=False, debug=False,
                         check=True, eval_mode=True, dynamic=False, qconfig=None):
        # 如果 debug 标志为 True，打印测试模块的信息
        if debug:
            print('Testing:', str(module))
        
        # 创建空的量化配置字典
        qconfig_dict = {'': get_default_qconfig(torch.backends.quantized.engine)}

        # 如果 eval_mode 标志为 True，将模块设为评估模式
        if eval_mode:
            module = module.eval()
        
        # 如果 dynamic 标志为 True，根据 qconfig 是否为 None，更新量化配置字典
        if dynamic:
            qconfig_dict = {'': default_dynamic_qconfig if qconfig is None else qconfig}
        
        # 使用输入数据获取脚本化的模块并设为评估模式
        model = get_script_module(module, tracing, inputs[0]).eval()

        # 如果 debug 标志为 True，打印输入图形的信息
        if debug:
            print('input graph:', model.graph)
        
        # 初始化空的模型字典和输出字典
        models = {}
        outputs = {}

        # 遍历 debug 为 True 和 False 的两种情况
        for debug in [True, False]:
            # 如果 dynamic 标志为 True，使用动态量化方法对模型进行量化
            if dynamic:
                models[debug] = quantize_dynamic_jit(model, qconfig_dict, debug=debug)
                # 确保模型成功运行
                outputs[debug] = models[debug](inputs)
            else:
                # 复制输入数据以便进行比较，因为被测试的模块可能包含原地操作
                inputs_copy = copy.deepcopy(inputs)
                models[debug] = quantize_jit(
                    model, qconfig_dict, test_only_eval_fn, [inputs_copy], inplace=False,
                    debug=debug)
                # 确保模型成功运行
                outputs[debug] = models[debug](*inputs[0])

        # 如果 debug 标志为 True，打印调试图形的信息和非调试图形的信息
        if debug:
            print('debug graph:', models[True].graph)
            print('non debug graph:', models[False].graph)

        # 如果 check 标志为 True，验证 debug 和非 debug 选项的数值结果是否相同
        if check:
            self.assertEqual(outputs[True], outputs[False])

            # 非调试图形应生成量化操作，使用 FileCheck 进行检查
            FileCheck().check(quantized_op) \
                       .run(models[False].graph)

        # 返回非 debug 模型作为结果
        return models[False]
    def checkGraphModuleNodes(
            self, graph_module,
            expected_node=None,
            expected_node_occurrence=None,
            expected_node_list=None):
        """ Check if GraphModule contains the target node
        Args:
            graph_module: the GraphModule instance we want to check
            expected_node, expected_node_occurrence, expected_node_list:
               see docs for checkGraphModeFxOp
        """
        # 创建一个空字典来存储图中的节点及其出现次数
        nodes_in_graph = {}
        # 创建一个空列表来存储图中的节点列表
        node_list = []
        # 获取图模块中所有模块的字典表示
        modules = dict(graph_module.named_modules(remove_duplicate=False))
        # 遍历图中的每个节点
        for node in graph_module.graph.nodes:
            n = None
            # 如果节点操作是函数调用或方法调用
            if node.op == 'call_function' or node.op == 'call_method':
                # 创建一个节点规范对象，表示函数或方法调用
                n = NodeSpec(node.op, node.target)
            # 如果节点操作是模块调用
            elif node.op == 'call_module':
                # 创建一个节点规范对象，表示模块调用
                n = NodeSpec(node.op, type(modules[node.target]))

            # 如果成功创建了节点规范对象
            if n is not None:
                # 将节点规范对象添加到节点列表中
                node_list.append(n)
                # 更新节点字典中该节点规范对象的出现次数
                if n in nodes_in_graph:
                    nodes_in_graph[n] += 1
                else:
                    nodes_in_graph[n] = 1

        # 如果期望存在特定节点
        if expected_node is not None:
            # 断言期望的节点存在于节点字典中
            self.assertTrue(expected_node in nodes_in_graph, 'node:' + str(expected_node) +
                            ' not found in the graph module')

        # 如果期望特定节点的出现次数
        if expected_node_occurrence is not None:
            # 遍历期望的节点及其出现次数的字典
            for expected_node, occurrence in expected_node_occurrence.items():
                # 如果期望的节点出现次数不为0
                if occurrence != 0:
                    # 断言期望的节点存在于节点字典中
                    self.assertTrue(
                        expected_node in nodes_in_graph,
                        'Check failed for node:' + str(expected_node) +
                        ' not found')
                    # 断言节点在图中的实际出现次数与期望次数一致
                    self.assertTrue(
                        nodes_in_graph[expected_node] == occurrence,
                        'Check failed for node:' + str(expected_node) +
                        ' Expected occurrence:' + str(occurrence) +
                        ' Found occurrence:' + str(nodes_in_graph[expected_node]))
                else:
                    # 断言期望的节点不在节点字典中
                    self.assertTrue(
                        expected_node not in nodes_in_graph,
                        'Check failed for node:' + str(expected_node) +
                        ' expected no occurrence but found')

        # 如果期望特定节点的列表
        if expected_node_list is not None:
            cur_index = 0
            # 遍历节点列表
            for n in node_list:
                # 如果当前索引等于期望节点列表的长度，结束循环
                if cur_index == len(expected_node_list):
                    return
                # 如果当前节点与期望节点列表中对应位置的节点相同
                if n == expected_node_list[cur_index]:
                    cur_index += 1
            # 断言当前索引等于期望节点列表的长度
            self.assertTrue(
                cur_index == len(expected_node_list),
                "Check failed for graph:" +
                self.printGraphModule(graph_module, print_str=False) +
                "Expected ordered list:" +
                str(expected_node_list))
    # 定义一个方法，用于打印图模块的信息，并可选择是否输出到标准输出
    def printGraphModule(self, graph_module, print_str=True):
        # 获取图模块中所有模块的命名字典
        modules = dict(graph_module.named_modules(remove_duplicate=False))
        # 初始化一个空列表，用于存储每个节点的信息
        node_infos = []
        # 遍历图模块中的每个节点
        for n in graph_module.graph.nodes:
            # 组装节点信息，包括操作、名称、目标、位置参数和关键字参数
            node_info = ' '.join(map(repr, [n.op, n.name, n.target, n.args, n.kwargs]))
            # 如果节点操作是调用模块
            if n.op == 'call_module':
                # 添加模块类型信息到节点信息中
                node_info += ' module type: ' + repr(type(modules[n.target]))
            # 将节点信息添加到节点信息列表中
            node_infos.append(node_info)
        # 将所有节点信息连接成一个字符串，用换行符分隔
        str_to_print = '\n'.join(node_infos)
        # 如果需要打印字符串到标准输出
        if print_str:
            # 打印连接好的节点信息字符串
            print(str_to_print)
        # 返回连接好的节点信息字符串
        return str_to_print
class QuantizationLiteTestCase(QuantizationTestCase):
    # QuantizationLiteTestCase 类，继承自 QuantizationTestCase，用于量化测试用例

    def _create_quantized_model(self, model_class: Type[torch.nn.Module], **kwargs):
        # 创建用于测试移动脚本模块的量化模型
        qengine = "qnnpack"
        with override_quantized_engine(qengine):
            # 使用指定的量化引擎 qnnpack 覆盖默认引擎
            qconfig = torch.ao.quantization.get_default_qconfig(qengine)
            model = model_class(**kwargs)
            # 创建模型实例
            model = quantize(model, test_only_eval_fn, [self.calib_data])
            # 对模型进行量化，使用 test_only_eval_fn 进行评估，self.calib_data 用于校准

        return model
        # 返回量化后的模型实例

    def _compare_script_and_mobile(self,
                                   model: torch.nn.Module,
                                   input: torch.Tensor):
        # 比较脚本模块和移动模块的数值输出
        qengine = "qnnpack"
        with override_quantized_engine(qengine):
            # 使用 qnnpack 引擎覆盖默认引擎
            script_module = torch.jit.script(model)
            # 将模型转换为 Torch 脚本模块
            script_module_result = script_module(input)
            # 对输入执行 Torch 脚本模块的前向传播并获取结果

            max_retry = 5
            for retry in range(1, max_retry + 1):
                # 最多重试 `max_retry` 次，如果成功则中断，否则抛出异常
                try:
                    buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
                    buffer.seek(0)
                    mobile_module = _load_for_lite_interpreter(buffer)
                    # 从字节流加载移动解释器所需的模块

                    mobile_module_result = mobile_module(input)
                    # 对输入执行移动模块的前向传播并获取结果

                    torch.testing.assert_close(script_module_result, mobile_module_result)
                    # 使用 Torch 的测试工具验证脚本模块和移动模块的结果近似相等

                    mobile_module_forward_result = mobile_module.forward(input)
                    torch.testing.assert_close(script_module_result, mobile_module_forward_result)
                    # 使用 Torch 的测试工具验证脚本模块和移动模块的 forward 方法结果近似相等

                    mobile_module_run_method_result = mobile_module.run_method("forward", input)
                    torch.testing.assert_close(script_module_result, mobile_module_run_method_result)
                    # 使用 Torch 的测试工具验证脚本模块和移动模块的 run_method 方法结果近似相等
                except AssertionError as e:
                    if retry == max_retry:
                        raise e
                    else:
                        continue
                break


class PT2EQuantizationTestCase(QuantizationTestCase):
    """
    Base QuantizationTestCase for PT2 with some helper methods.
    """
    # PT2EQuantizationTestCase 类，继承自 QuantizationTestCase，为 PT2 提供基础量化测试用例和一些辅助方法

    _MAP_TO_FX_TRACED_OPS = {
        torch.ops.quantized_decomposed.quantize_per_tensor: torch.ops.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.quantized_decomposed.dequantize_per_tensor: torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        torch.ops.quantized_decomposed.quantize_per_channel: torch.ops.quantized_decomposed.quantize_per_channel.default,
        torch.ops.quantized_decomposed.dequantize_per_channel: torch.ops.quantized_decomposed.dequantize_per_channel.default,
        torch.ops.quantized_decomposed.quantize_per_tensor.tensor: torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
        torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    }
    # _MAP_TO_FX_TRACED_OPS 字典，将量化操作映射到 FX 跟踪操作，用于将 Torch 操作映射到其分解版本
    # 定义一个测试函数 `_test_quantizer`，用于测试量化器在给定模型上的效果
    def _test_quantizer(
        self,
        model,  # 输入参数：模型
        example_inputs,  # 输入参数：示例输入数据
        quantizer,  # 输入参数：量化器对象
        expected_node_occurrence,  # 输入参数：预期节点出现次数的字典
        expected_node_list=None,  # 输入参数：预期节点列表，默认为 None
        check_against_fx_quant=False,  # 输入参数：是否与 FX 量化进行比较，默认为 False
        fx_qconfig_mapping=None,  # 输入参数：FX 量化配置映射，可选，默认为 None
        export_with_dynamic_shape=False,  # 输入参数：是否导出动态形状，默认为 False
        is_qat=False,  # 输入参数：是否量化感知训练，默认为 False
        is_debug_mode=False,  # 输入参数：是否调试模式，默认为 False
    ):
        # 重置 Dynamo 缓存
        torch._dynamo.reset()
        # 将模型设置为评估模式
        m_eager = model.eval()

        # 复制模型用于捕获
        m = copy.deepcopy(m_eager)
        # 设置动态形状
        dynamic_shapes = tuple(
            {0: torch.export.Dim("dim")} if i == 0 else None
            for i in range(len(example_inputs))
        )
        # 捕获预自动求导图
        m = capture_pre_autograd_graph(
            m,
            example_inputs,
            dynamic_shapes=dynamic_shapes if export_with_dynamic_shape else None,
        )

        # 如果是量化感知训练，应用准备量化感知训练到量化器
        if is_qat:
            m = prepare_qat_pt2e(m, quantizer)
        else:
            # 否则，应用准备标量至量化引擎到量化器
            m = prepare_pt2e(m, quantizer)
        
        # 校准
        m(*example_inputs)
        # 转换为标量至量化引擎
        m = convert_pt2e(m)
        
        # 如果是调试模式，打印量化模型
        if is_debug_mode:
            print("quantized model", m)

        # 使用示例输入获取量化后的输出
        pt2_quant_output = m(*example_inputs)
        
        # 定义节点规范
        ns = NodeSpec
        # 创建节点出现次数字典，从预期节点出现次数映射中生成
        node_occurrence = {
            ns.call_function(k): v for k, v in expected_node_occurrence.items()
        }
        
        # 如果预期节点列表为 None，则设置为空列表
        if expected_node_list is None:
            expected_node_list = []
        
        # 生成节点列表，将预期节点列表中的每个节点转换为节点规范对象
        node_list = [ns.call_function(n) for n in expected_node_list]
        
        # 调用检查图模块节点方法，验证量化后的模型节点
        self.checkGraphModuleNodes(
            m, expected_node_occurrence=node_occurrence, expected_node_list=node_list
        )
        
        # 如果需要与 FX 量化比较
        if check_against_fx_quant:
            # 获取 FX 量化配置映射
            qconfig_mapping = fx_qconfig_mapping
            # 获取执行后端配置
            backend_config = get_executorch_backend_config()
            # 复制原始模型
            m_copy = copy.deepcopy(m_eager)
            # 准备 FX 模型
            m_fx = prepare_fx(
                m_copy, qconfig_mapping, example_inputs, backend_config=backend_config
            )
            # 在 FX 模型上执行
            m_fx(*example_inputs)
            # 将 FX 模型转换为参考分解 FX
            m_fx = _convert_to_reference_decomposed_fx(
                m_fx, backend_config=backend_config
            )
            # 再次捕获预自动求导图
            m_fx = capture_pre_autograd_graph(
                m_fx,
                example_inputs,
                dynamic_shapes=dynamic_shapes if export_with_dynamic_shape else None,
            )
            
            # 创建节点出现次数字典
            node_occurrence = {}
            # 将 PT2EQuantizationTestCase._MAP_TO_FX_TRACED_OPS 中的节点映射到 FX 跟踪操作中
            for k, v in PT2EQuantizationTestCase._MAP_TO_FX_TRACED_OPS.items():
                if k in expected_node_occurrence:
                    node_occurrence[ns.call_function(v)] = expected_node_occurrence[k]
            
            # 检查 FX 模块节点
            self.checkGraphModuleNodes(m_fx, expected_node_occurrence=node_occurrence)
            
            # 获取 FX 量化输出
            fx_quant_output = m_fx(*example_inputs)
            
            # 断言 FX 量化输出与标量至量化引擎输出相等
            self.assertEqual(fx_quant_output, pt2_quant_output)
    # 使用给定的量化器和示例输入对模型 m 进行量化处理，返回量化后的模型
    def _quantize(self, m, quantizer, example_inputs, is_qat: bool = False):
        # 重置动态图缓存
        torch._dynamo.reset()

        # 捕获前自动求导图
        m = capture_pre_autograd_graph(
            m,
            example_inputs,
        )

        # 如果是量化训练，使用量化训练准备函数
        if is_qat:
            m = prepare_qat_pt2e(m, quantizer)
        else:
            # 否则，使用非量化训练准备函数
            m = prepare_pt2e(m, quantizer)

        # 执行模型的前向传播
        m(*example_inputs)

        # 将 PyTorch 模型转换为量化引擎支持的格式
        m = convert_pt2e(m)
        return m

    # 获取量化后的线性层模型，并返回其作为 torch.fx.GraphModule
    def _get_pt2e_quantized_linear(self, is_per_channel=False) -> torch.fx.GraphModule:
        # 定义一个简单的模型类 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        # 创建 XNNPACKQuantizer 量化器
        quantizer = XNNPACKQuantizer()

        # 获取对称量化配置
        operator_config = get_symmetric_quantization_config(is_per_channel=is_per_channel)

        # 设置全局量化配置
        quantizer.set_global(operator_config)

        # 准备示例输入
        example_inputs = (torch.randn(2, 2),)

        # 创建模型实例并设为评估模式
        m = M().eval()

        # 对模型 m 进行量化，并返回量化后的模型
        return self._quantize(m, quantizer, example_inputs)
# 以下是一系列用于测试量化的玩具模型

class SingleLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个单层线性模型，输入和输出都是5维
        self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        # 模型的前向传播，通过全连接层处理输入
        x = self.fc1(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个例子输入，形状为(1, 5)
        return (torch.rand(1, 5),)

class AnnotatedSingleLayerLinearModel(torch.nn.Module):
    def __init__(self, qengine='fbgemm'):
        super().__init__()
        # 获取指定量化引擎的默认量化配置
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        # 使用量化包装器包裹线性层，用于量化模型
        self.fc1 = QuantWrapper(torch.nn.Linear(5, 5).to(dtype=torch.float))

    def forward(self, x):
        # 模型的前向传播，通过量化后的线性层处理输入
        x = self.fc1(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个例子输入，形状为(1, 5)
        return (torch.rand(1, 5),)

class SingleLayerLinearDynamicModel(torch.nn.Module):
    def __init__(self, qengine='fbgemm'):
        super().__init__()
        # 获取指定量化引擎的默认动态量化配置
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        # 定义一个单层线性模型，输入和输出都是5维
        self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        # 模型的前向传播，通过全连接层处理输入
        x = self.fc1(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个例子输入，形状为(1, 5)
        return (torch.rand(1, 5),)

class LinearAddModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个两层的线性模型，输入是5维，中间层输出是8维，最终输出是5维
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(8, 5).to(dtype=torch.float)

    def forward(self, x):
        # 模型的前向传播，先经过第一层线性层，然后加上5，再经过第二层线性层
        x = self.fc1(x)
        x = torch.add(x, 5)
        x = self.fc2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个例子输入，形状为(1, 5)
        return (torch.rand(1, 5),)

class RNNDynamicModel(torch.nn.Module):
    def __init__(self, mod_type):
        super().__init__()
        self.qconfig = default_dynamic_qconfig
        # 根据给定的模型类型选择相应的循环神经网络模型
        if mod_type == 'GRU':
            self.mod = torch.nn.GRU(2, 2).to(dtype=torch.float)
        elif mod_type == 'LSTM':
            self.mod = torch.nn.LSTM(2, 2).to(dtype=torch.float)

    def forward(self, x):
        # 模型的前向传播，通过选择的循环神经网络模型处理输入
        x = self.mod(x)
        return x

class RNNCellDynamicModel(torch.nn.Module):
    def __init__(self, mod_type):
        super().__init__()
        self.qconfig = default_dynamic_qconfig
        # 根据给定的模型类型选择相应的循环神经网络单元模型
        if mod_type == 'GRUCell':
            self.mod = torch.nn.GRUCell(2, 2).to(dtype=torch.float)
        elif mod_type == 'LSTMCell':
            self.mod = torch.nn.LSTMCell(2, 2).to(dtype=torch.float)
        elif mod_type == 'RNNReLU':
            self.mod = torch.nn.RNNCell(2, 2, nonlinearity='relu').to(dtype=torch.float)
        elif mod_type == 'RNNTanh':
            self.mod = torch.nn.RNNCell(2, 2, nonlinearity='tanh').to(dtype=torch.float)

    def forward(self, x):
        # 模型的前向传播，通过选择的循环神经网络单元模型处理输入
        x = self.mod(x)
        return x

class LSTMwithHiddenDynamicModel(torch.nn.Module):
    def __init__(self, qengine='fbgemm'):
        super().__init__()
        # 获取指定量化引擎的默认量化配置
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        # 定义一个带隐藏状态的LSTM模型，输入和输出都是2维
        self.lstm = torch.nn.LSTM(2, 2).to(dtype=torch.float)
    # 定义一个方法 `forward`，用于执行前向传播操作，接受输入 `x` 和隐藏状态 `hid`
    def forward(self, x, hid):
        # 调用 LSTM 模型的前向传播方法 `lstm`，对输入 `x` 和隐藏状态 `hid` 进行处理
        x, hid = self.lstm(x, hid)
        # 返回处理后的输出 `x` 和更新后的隐藏状态 `hid`
        return x, hid
class ConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个2维卷积层，输入通道数为3，输出通道数为5，卷积核大小为3x3，无偏置
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)

    def forward(self, x):
        # 在输入张量x上应用定义好的卷积层
        x = self.conv(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入，这里是一个形状为(1, 3, 5, 5)的随机张量
        return (torch.rand(1, 3, 5, 5),)

class ConvTransposeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个2维转置卷积层，输入通道数为3，输出通道数为5，卷积核大小为3x3，无偏置
        self.conv = torch.nn.ConvTranspose2d(3, 5, 3, bias=False).to(dtype=torch.float)

    def forward(self, x):
        # 在输入张量x上应用定义好的转置卷积层
        x = self.conv(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入，这里是一个形状为(1, 3, 5, 5)的随机张量
        return (torch.rand(1, 3, 5, 5),)

class AnnotatedConvModel(torch.nn.Module):
    def __init__(self, qengine):
        super().__init__()
        # 设置量化配置
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        # 定义一个2维卷积层，输入通道数为3，输出通道数为5，卷积核大小为3x3，无偏置
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        # 定义量化前处理
        self.quant = QuantStub()
        # 定义量化后处理
        self.dequant = DeQuantStub()

    def forward(self, x):
        # 应用量化前处理
        x = self.quant(x)
        # 在量化后的张量上应用定义好的卷积层
        x = self.conv(x)
        # 应用量化后处理
        x = self.dequant(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入，这里是一个形状为(1, 3, 5, 5)的随机张量
        return (torch.rand(1, 3, 5, 5),)

class AnnotatedConvTransposeModel(torch.nn.Module):
    def __init__(self, qengine):
        super().__init__()
        # 设置量化配置
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        # 定义一个2维转置卷积层，输入通道数为3，输出通道数为5，卷积核大小为3x3，无偏置
        self.conv = torch.nn.ConvTranspose2d(3, 5, 3, bias=False).to(dtype=torch.float)
        # 定义量化前处理
        self.quant = QuantStub()
        # 定义量化后处理
        self.dequant = DeQuantStub()

    def forward(self, x):
        # 应用量化前处理
        x = self.quant(x)
        # 在量化后的张量上应用定义好的转置卷积层
        x = self.conv(x)
        # 应用量化后处理
        x = self.dequant(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入，这里是一个形状为(1, 3, 5, 5)的随机张量
        return (torch.rand(1, 3, 5, 5),)

class ConvBnModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个2维卷积层，输入通道数为3，输出通道数为5，卷积核大小为3x3，无偏置
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        # 定义一个批归一化层，输入通道数为5
        self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)

    def forward(self, x):
        # 在输入张量x上应用定义好的卷积层
        x = self.conv(x)
        # 在卷积输出上应用批归一化层
        x = self.bn(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入，这里是一个形状为(1, 3, 5, 5)的随机张量
        return (torch.rand(1, 3, 5, 5),)

class AnnotatedConvBnModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 设置默认量化配置
        self.qconfig = default_qconfig
        # 定义一个2维卷积层，输入通道数为3，输出通道数为5，卷积核大小为3x3，无偏置
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        # 定义一个批归一化层，输入通道数为5
        self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)
        # 定义量化前处理
        self.quant = QuantStub()
        # 定义量化后处理
        self.dequant = DeQuantStub()

    def forward(self, x):
        # 应用量化前处理
        x = self.quant(x)
        # 在量化后的张量上应用定义好的卷积层
        x = self.conv(x)
        # 在卷积输出上应用批归一化层
        x = self.bn(x)
        # 应用量化后处理
        x = self.dequant(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入，这里是一个形状为(1, 3, 5, 5)的随机张量
        return (torch.rand(1, 3, 5, 5),)
    # 初始化函数，继承父类的初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个卷积层，输入通道数为3，输出通道数为5，卷积核大小为3，不使用偏置
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        # 创建一个批归一化层，输入通道数为5
        self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)
        # 创建一个ReLU激活函数，inplace=True表示原地操作，节省内存
        self.relu = nn.ReLU(inplace=True)

    # 前向传播函数
    def forward(self, x):
        # 卷积层操作，对输入x进行卷积操作
        x = self.conv(x)
        # 批归一化层操作，对卷积结果x进行批归一化
        x = self.bn(x)
        # 使用ReLU激活函数处理批归一化结果x
        x = self.relu(x)
        # 返回激活后的结果x
        return x

    # 获取示例输入的函数
    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个包含示例输入的元组，这里示例输入是一个随机生成的1x3x5x5的张量
        return (torch.rand(1, 3, 5, 5),)
class AnnotatedConvBnReLUModel(torch.nn.Module):
    def __init__(self, qengine='fbgemm'):
        super().__init__()
        # 设置量化配置
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        # 创建卷积层，输入通道为3，输出通道为5，卷积核大小为3x3，无偏置
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        # 创建批归一化层，参数为5个通道
        self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)
        # 创建ReLU激活函数，inplace设置为True，表示原地操作
        self.relu = nn.ReLU(inplace=True)
        # 创建量化模拟器（QuantStub）用于输入
        self.quant = QuantStub()
        # 创建量化反模拟器（DeQuantStub）用于输出
        self.dequant = DeQuantStub()

    def forward(self, x):
        # 对输入进行量化
        x = self.quant(x)
        # 进行卷积操作
        x = self.conv(x)
        # 进行批归一化操作
        x = self.bn(x)
        # 使用ReLU激活函数
        x = self.relu(x)
        # 对输出进行反量化
        x = self.dequant(x)
        return x

    def fuse_model(self):
        # TODO: remove this check and define two fuse_modules function on this module
        # 如果处于训练模式，则使用量化感知训练（QAT）融合模块
        if self.training:
            torch.ao.quantization.fuse_modules_qat(self, [['conv', 'bn', 'relu']], inplace=True)
        else:
            # 否则，使用普通量化融合模块
            torch.ao.quantization.fuse_modules(self, [['conv', 'bn', 'relu']], inplace=True)

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入，形状为(1, 3, 5, 5)
        return (torch.rand(1, 3, 5, 5),)

class TwoLayerConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建第一个卷积层，输入通道为3，输出通道为5，卷积核大小为3x3，无偏置
        self.conv1 = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        # 创建第二个卷积层，输入通道为5，输出通道为5，卷积核大小为1x1，无偏置
        self.conv2 = torch.nn.Conv2d(5, 5, 1, bias=False).to(dtype=torch.float)

    def forward(self, x):
        # 第一个卷积层操作
        x = self.conv1(x)
        # 第二个卷积层操作
        x = self.conv2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入，形状为(1, 3, 5, 5)
        return (torch.rand(1, 3, 5, 5),)

class TwoLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建第一个全连接层，输入特征数为5，输出特征数为8
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        # 创建第二个全连接层，输入特征数为8，输出特征数为5
        self.fc2 = torch.nn.Linear(8, 5).to(dtype=torch.float)

    def forward(self, x):
        # 第一个全连接层操作
        x = self.fc1(x)
        # 第二个全连接层操作
        x = self.fc2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入，形状为(1, 5)
        return (torch.rand(1, 5),)

class LinearModelWithSubmodule(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建包含两层全连接层的子模块
        self.subm = TwoLayerLinearModel()
        # 创建额外的全连接层
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        # 调用子模块进行前向传播
        x = self.subm(x)
        # 对输出结果进行额外的全连接层操作
        x = self.fc(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 调用子模块的示例输入
        return self.subm.get_example_inputs()

class AnnotatedTwoLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建第一个全连接层，输入特征数为5，输出特征数为8
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        # 创建第二个全连接层，使用量化包装器QuantWrapper包裹，输入特征数为8，输出特征数为5
        self.fc2 = QuantWrapper(torch.nn.Linear(8, 5).to(dtype=torch.float))
        # 设置fc2的量化配置
        self.fc2.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")

    def forward(self, x):
        # 第一个全连接层操作
        x = self.fc1(x)
        # 第二个全连接层操作（经过量化包装器）
        x = self.fc2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入，形状为(1, 5)
        return (torch.rand(1, 5),)

class ActivationsTestModel(torch.nn.Module):
    # 初始化函数，继承父类的初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 设置量化配置为默认的 "fbgemm"
        self.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        # 创建量化的前向传递模块
        self.quant = torch.ao.quantization.QuantStub()
        # 创建硬切线激活函数模块
        self.hardswish = torch.nn.Hardswish().to(dtype=torch.float)
        # 创建指数线性单元激活函数模块
        self.elu = torch.nn.ELU().to(dtype=torch.float)
        # 创建去量化的后向传递模块
        self.dequant = torch.ao.quantization.DeQuantStub()

    # 前向传递函数，接收输入张量 x
    def forward(self, x):
        # 对输入张量 x 进行量化
        x = self.quant(x)
        # 使用硬切线激活函数处理量化后的张量 x
        x = self.hardswish(x)
        # 使用指数线性单元激活函数处理处理后的张量 x
        x = self.elu(x)
        # 对处理后的张量 x 进行去量化
        x = self.dequant(x)
        # 返回处理后的张量 x
        return x
class LinearReluModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个线性层，输入维度为5，输出维度为5，转换为浮点类型
        self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)
        # 定义ReLU激活函数
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # 前向传播函数，先通过线性层fc，再经过ReLU激活函数
        x = self.relu(self.fc(x))
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入，形状为(1, 5)
        return (torch.rand(1, 5),)


class LinearReluLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义第一个线性层，输入维度为5，输出维度为8，转换为浮点类型
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        # 定义ReLU激活函数
        self.relu = torch.nn.ReLU()
        # 定义第二个线性层，输入维度为8，输出维度为5，转换为浮点类型
        self.fc2 = torch.nn.Linear(8, 5).to(dtype=torch.float)

    def forward(self, x):
        # 前向传播函数，先通过第一个线性层fc1，再经过ReLU激活函数，
        # 然后通过第二个线性层fc2
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入，形状为(1, 5)
        return (torch.rand(1, 5),)


class LinearReluAddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义第一个线性层，输入维度为5，输出维度为5，转换为浮点类型
        self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)
        # 定义ReLU激活函数
        self.relu = torch.nn.ReLU()
        # 定义第二个线性层，输入维度为5，输出维度为5，转换为浮点类型
        self.fc2 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        # 前向传播函数，先通过第一个线性层fc1，再经过ReLU激活函数，
        # 然后加上常数5，最后通过第二个线性层fc2
        x = self.fc1(x)
        x = self.relu(x)
        x = torch.add(x, 5)
        x = self.fc2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入，形状为(1, 5)
        return (torch.rand(1, 5),)


class LinearBnLeakyReluModel(torch.nn.Module):
    def __init__(self, with_bn=True):
        super().__init__()
        # 定义一个线性层，输入维度为5，输出维度为5
        self.linear = nn.Linear(5, 5)
        # 定义批归一化层，对5维度进行归一化
        self.bn1d = nn.BatchNorm1d(5)
        # 定义LeakyReLU激活函数，负斜率为0.01
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.with_bn = with_bn  # 是否使用批归一化

    def forward(self, x):
        # 前向传播函数，先通过线性层linear，然后根据with_bn决定是否使用批归一化，
        # 最后通过LeakyReLU激活函数
        x = self.linear(x)
        if self.with_bn:
            x = self.bn1d(x)
        x = self.leaky_relu(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入，形状为(1, 5)
        return (torch.rand(1, 5),)


class LinearTanhModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个线性层，输入维度为5，输出维度为5
        self.linear = nn.Linear(5, 5)
        # 定义Tanh激活函数
        self.tanh = nn.Tanh()

    def forward(self, x):
        # 前向传播函数，先通过线性层linear，然后经过Tanh激活函数
        x = self.linear(x)
        x = self.tanh(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入，形状为(1, 5)
        return (torch.rand(1, 5),)


class ConvBnAddReluModel(torch.nn.Module):
    def __init__(self,
                 with_bn=True,
                 with_relu=True,
                 left_conv=True,
                 two_conv=True,
                 use_torch_add=True):
        super().__init__()
        # 定义一个2D卷积层，输入通道数和输出通道数都为5，卷积核大小为(2, 2)
        self.conv = nn.Conv2d(5, 5, (2, 2))
        # 定义第二个2D卷积层，输入通道数和输出通道数都为5，卷积核大小为(2, 2)
        self.conv2 = nn.Conv2d(5, 5, (2, 2))
        # 定义批归一化层，对5个通道进行归一化
        self.bn = nn.BatchNorm2d(5)
        # 定义ReLU激活函数
        self.relu = nn.ReLU()
        self.with_bn = with_bn  # 是否使用批归一化
        self.with_relu = with_relu  # 是否使用ReLU激活函数
        self.two_conv = two_conv  # 是否使用第二个卷积层
        self.left_conv = left_conv  # 是否使用第一个卷积层
        self.use_torch_add = use_torch_add  # 是否使用torch的add函数
    # 定义一个前向传播函数，接受两个输入 x1 和 x2
    def forward(self, x1, x2):
        # 如果设置了两个卷积层
        if self.two_conv:
            # 如果选择使用 torch 的 add 函数
            if self.use_torch_add:
                # 如果使用了 Batch Normalization
                if self.with_bn:
                    # 对 x1 进行卷积操作，然后进行 Batch Normalization，再与 x1 的第二个卷积结果相加
                    x = torch.add(self.bn(self.conv(x1)), self.conv2(x1))
                else:
                    # 直接将 x1 的两个卷积结果相加
                    x = torch.add(self.conv(x1), self.conv2(x1))
            else:
                # 如果使用了 Batch Normalization
                if self.with_bn:
                    # 分别对 x1 进行卷积操作并进行 Batch Normalization，然后与 x1 的第二个卷积结果相加
                    x = self.bn(self.conv(x1)) + self.conv2(x1)
                else:
                    # 直接将 x1 的两个卷积结果相加
                    x = self.conv(x1) + self.conv2(x1)
        else:
            # 如果只有一个卷积层
            if self.use_torch_add:
                # 如果左侧卷积标志为真
                if self.left_conv:
                    # 如果使用了 Batch Normalization
                    if self.with_bn:
                        # 对 x1 进行卷积操作，然后进行 Batch Normalization，再与 x2 相加
                        x = torch.add(self.bn(self.conv(x1)), x2)
                    else:
                        # 直接将 x1 和 x2 相加
                        x = torch.add(self.conv(x1), x2)
                else:
                    # 如果使用了 Batch Normalization
                    if self.with_bn:
                        # 先对 x1 进行卷积操作，然后进行 Batch Normalization，再与 x2 相加
                        x = torch.add(x2, self.bn(self.conv(x1)))
                    else:
                        # 直接将 x1 和 x2 相加
                        x = torch.add(x2, self.conv(x1))
            else:
                # 如果左侧卷积标志为真
                if self.left_conv:
                    # 如果使用了 Batch Normalization
                    if self.with_bn:
                        # 对 x1 进行卷积操作，然后进行 Batch Normalization，再与 x2 相加
                        x = self.bn(self.conv(x1)) + x2
                    else:
                        # 直接将 x1 和 x2 相加
                        x = self.conv(x1) + x2
                else:
                    # 如果使用了 Batch Normalization
                    if self.with_bn:
                        # 先对 x1 进行卷积操作，然后进行 Batch Normalization，再与 x2 相加
                        x = x2 + self.bn(self.conv(x1))
                    else:
                        # 直接将 x1 和 x2 相加
                        x = x2 + self.conv(x1)
        # 如果设置了使用 ReLU 激活函数，则对 x 应用 ReLU 操作
        if self.with_relu:
            x = self.relu(x)
        # 返回前向传播后的结果 x
        return x

    # 返回一个示例输入元组
    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 5, 3, 3), torch.rand(1, 5, 2, 2))
# TODO: self.fc should be self.conv
class ConvReluModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个2D卷积层，输入通道为3，输出通道为5，卷积核大小为3
        self.fc = torch.nn.Conv2d(3, 5, 3).to(dtype=torch.float)
        # 定义ReLU激活函数
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # 将输入x先经过卷积层self.fc，再经过ReLU激活函数self.relu
        x = self.relu(self.fc(x))
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入的元组，这里是一个形状为(1, 3, 5, 5)的随机张量
        return (torch.rand(1, 3, 5, 5),)

# TODO: self.fc should be self.conv
class ConvReluConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义两个2D卷积层，分别为self.fc1和self.fc2
        self.fc1 = torch.nn.Conv2d(3, 5, 3).to(dtype=torch.float)
        # 定义ReLU激活函数
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Conv2d(5, 5, 1).to(dtype=torch.float)

    def forward(self, x):
        # 依次对输入x进行两次卷积操作，并在两次操作中间添加ReLU激活函数
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入的元组，这里是一个形状为(1, 3, 5, 5)的随机张量
        return (torch.rand(1, 3, 5, 5),)

# TODO: self.fc should be self.conv
class ConvReluAddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义两个2D卷积层，分别为self.fc1和self.fc2
        self.fc1 = torch.nn.Conv2d(3, 5, 3).to(dtype=torch.float)
        # 定义ReLU激活函数
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Conv2d(5, 5, 1).to(dtype=torch.float)

    def forward(self, x):
        # 先进行卷积操作，再应用ReLU激活函数，接着加上一个常数5，最后再进行卷积操作
        x = self.fc1(x)
        x = self.relu(x)
        x = torch.add(x, 5)
        x = self.fc2(x)
        self.relu = torch.nn.ReLU()  # 注意：这里可能是个错误，将self.relu重新定义为ReLU激活函数
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入的元组，这里是一个形状为(1, 3, 5, 5)的随机张量
        return (torch.rand(1, 3, 5, 5),)

class NormalizationTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义量化桩（QuantStub）、线性层（Linear）、层归一化（LayerNorm）、组归一化（GroupNorm）和实例归一化（InstanceNorm）
        self.quant = torch.ao.quantization.QuantStub()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.layer_norm = torch.nn.LayerNorm(8)
        self.group_norm = torch.nn.GroupNorm(2, 8)
        self.instance_norm1d = torch.nn.InstanceNorm1d(8)
        self.instance_norm2d = torch.nn.InstanceNorm2d(8)
        self.instance_norm3d = torch.nn.InstanceNorm3d(8)

    def forward(self, x):
        # 对输入x依次进行量化、线性变换、层归一化、组归一化、1D实例归一化、2D实例归一化、3D实例归一化
        x = self.quant(x)
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = self.group_norm(x.unsqueeze(-1).repeat(1, 1, 3))
        x = self.instance_norm1d(x)
        x = self.instance_norm2d(x.unsqueeze(-1))
        x = self.instance_norm3d(x.unsqueeze(-1))
        return x

class NestedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化两个子模型，并定义一个线性层
        self.sub1 = LinearReluModel()
        self.sub2 = TwoLayerLinearModel()
        self.fc3 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        # 将输入x分别传递给两个子模型和线性层，然后返回结果
        x = self.sub1(x)
        x = self.sub2(x)
        x = self.fc3(x)
        return x

class AnnotatedNestedModel(torch.nn.Module):
    # 这里是一个空的模型类定义，没有实现任何内容，需要补充完整
    pass
    # 初始化函数，接收一个量化引擎参数 qengine
    def __init__(self, qengine):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个 LinearReluModel 类的实例，并赋值给 self.sub1
        self.sub1 = LinearReluModel()
        # 创建一个 TwoLayerLinearModel 类的实例，并赋值给 self.sub2
        self.sub2 = TwoLayerLinearModel()
        # 创建一个包装了 torch.nn.Linear(5, 5) 的 QuantWrapper 实例，并赋值给 self.fc3
        self.fc3 = QuantWrapper(torch.nn.Linear(5, 5).to(dtype=torch.float))
        # 设置 self.fc3 的量化配置为默认配置 default_qconfig
        self.fc3.qconfig = default_qconfig
        # 将 self.sub2 的第一个全连接层 fc1 也用 QuantWrapper 进行包装
        self.sub2.fc1 = QuantWrapper(self.sub2.fc1)
        # 根据 qengine 参数选择量化配置
        if qengine == 'fbgemm':
            # 如果 qengine 是 'fbgemm'，则将 self.sub2 的 fc1 层的量化配置设置为默认的分通道量化配置 default_per_channel_qconfig
            self.sub2.fc1.qconfig = default_per_channel_qconfig
        else:
            # 如果 qengine 不是 'fbgemm'，则将 self.sub2 的 fc1 层的量化配置设置为默认配置 default_qconfig
            self.sub2.fc1.qconfig = default_qconfig

    # 前向传播函数，接收输入 x
    def forward(self, x):
        # 将输入 x 经过 self.sub1 模型处理
        x = self.sub1(x)
        # 将处理后的结果 x 经过 self.sub2 模型处理
        x = self.sub2(x)
        # 将处理后的结果 x 经过 self.fc3 模型处理
        x = self.fc3(x)
        # 返回处理后的结果 x
        return x
class AnnotatedSubNestedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 实例化一个 LinearReluModel 模型作为子模型 sub1
        self.sub1 = LinearReluModel()
        # 实例化一个 TwoLayerLinearModel 模型，并用 QuantWrapper 进行包装作为子模型 sub2
        self.sub2 = QuantWrapper(TwoLayerLinearModel())
        # 定义一个具有5个输入和5个输出的线性层，并用 QuantWrapper 进行包装作为模型的第三层 fc3
        self.fc3 = QuantWrapper(torch.nn.Linear(5, 5).to(dtype=torch.float))
        # 设置 fc3 的量化配置为 default_qconfig
        self.fc3.qconfig = default_qconfig
        # 设置 sub2 的量化配置为 default_qconfig

    def forward(self, x):
        # 将输入 x 传递给子模型 sub1
        x = self.sub1(x)
        # 将经过 sub1 处理后的结果 x 传递给子模型 sub2
        x = self.sub2(x)
        # 将经过 sub2 处理后的结果 x 传递给模型的第三层 fc3
        x = self.fc3(x)
        # 返回最终的输出 x
        return x

class AnnotatedCustomConfigNestedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 实例化一个 LinearReluModel 模型作为子模型 sub1
        self.sub1 = LinearReluModel()
        # 实例化一个 TwoLayerLinearModel 模型作为子模型 sub2
        self.sub2 = TwoLayerLinearModel()
        # 定义一个具有5个输入和5个输出的线性层，并用 QuantWrapper 进行包装作为模型的第三层 fc3
        self.fc3 = QuantWrapper(torch.nn.Linear(5, 5).to(dtype=torch.float))
        # 设置 fc3 的量化配置为 default_qconfig
        self.fc3.qconfig = default_qconfig
        # 设置 sub2 的量化配置为 default_qconfig

        # 定义一个自定义的量化选项字典
        custom_options = {
            'dtype': torch.quint8,
            'qscheme': torch.per_tensor_affine
        }
        # 创建一个自定义的量化配置对象 custom_qconfig，其中包含激活函数的默认观察器和权重的默认观察器
        custom_qconfig = QConfig(activation=default_observer.with_args(**custom_options),
                                 weight=default_weight_observer)
        # 将 sub2 的第一个线性层 fc1 的量化配置设置为 custom_qconfig
        self.sub2.fc1.qconfig = custom_qconfig

        # 使用 QuantWrapper 对 sub2 的每个线性层进行包装
        self.sub2.fc1 = QuantWrapper(self.sub2.fc1)
        self.sub2.fc2 = QuantWrapper(self.sub2.fc2)

    def forward(self, x):
        # 将输入 x 传递给子模型 sub1
        x = self.sub1(x)
        # 将经过 sub1 处理后的结果 x 传递给子模型 sub2
        x = self.sub2(x)
        # 将经过 sub2 处理后的结果 x 传递给模型的第三层 fc3
        x = self.fc3(x)
        # 返回最终的输出 x
        return x

class QuantSubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 实例化一个 LinearReluModel 模型作为子模型 sub1
        self.sub1 = LinearReluModel()
        # 实例化一个 TwoLayerLinearModel 模型，并用 QuantWrapper 进行包装作为子模型 sub2
        self.sub2 = QuantWrapper(TwoLayerLinearModel())
        # 定义一个具有5个输入和5个输出的线性层作为模型的第三层 fc3
        self.fc3 = torch.nn.Linear(5, 5).to(dtype=torch.float)
        # 设置 fc3 的量化配置为 default_qconfig
        self.fc3.qconfig = default_qconfig

    def forward(self, x):
        # 将输入 x 传递给子模型 sub1
        x = self.sub1(x)
        # 将经过 sub1 处理后的结果 x 传递给子模型 sub2
        x = self.sub2(x)
        # 将经过 sub2 处理后的结果 x 传递给模型的第三层 fc3
        x = self.fc3(x)
        # 返回最终的输出 x
        return x

class InnerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个具有5个输入和8个输出的线性层 fc1
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        # 定义一个 ReLU 激活函数 relu1
        self.relu1 = torch.nn.ReLU()
        # 定义一个具有8个输入和5个输出的线性层 fc2
        self.fc2 = torch.nn.Linear(8, 5).to(dtype=torch.float)
        # 定义一个 ReLU 激活函数 relu2
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        # 将输入 x 经过线性层 fc1，然后经过 ReLU 激活函数 relu1
        out = self.fc1(x)
        out = self.relu1(out)
        # 将经过 relu1 的输出再经过线性层 fc2，然后经过 ReLU 激活函数 relu2
        out = self.fc2(out)
        out = self.relu2(out)
        # 返回最终的输出 out
        return out
    # 定义一个方法，用于融合模块
    def fuse_modules(self):
        # 初始化一个空列表，用于存储可融合的层
        fusable_layers = []
        # 获取当前模块的所有子模块，并转换为列表形式
        named_children = list(self.named_children())
        # 遍历子模块列表
        for idx, (current_name, layer) in enumerate(named_children):
            # 检查当前子模块是否为线性层（Linear）
            if isinstance(layer, torch.nn.Linear):
                # 如果当前索引大于等于列表长度减一，结束循环
                if idx >= len(named_children) - 1:
                    break
                # 检查下一个子模块是否为ReLU激活层
                if isinstance(named_children[idx + 1][1], torch.nn.ReLU):
                    # 将当前层名和下一个层名作为可融合的一对，添加到可融合层列表中
                    fusable_layers.append([current_name,
                                           named_children[idx + 1][0]])
        # 如果模型正在训练中
        if self.training:
            # 在量化感知训练（Quantization Aware Training, QAT）中，原地融合模块
            torch.ao.quantization.fuse_modules_qat(self, fusable_layers, inplace=True)
        else:
            # 在推断或验证阶段，原地融合模块
            torch.ao.quantization.fuse_modules(self, fusable_layers, inplace=True)
class FunctionalLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化权重矩阵，形状为(5, 5)，随机初始化
        self.weight = torch.rand((5, 5))
        # 初始化偏置向量，长度为5，全零初始化
        self.bias = torch.zeros(5)

    def forward(self, x):
        # 调用 torch.nn.functional 中的线性函数，对输入 x 进行线性变换
        return F.linear(x, self.weight, self.bias)

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入，形状为 (1, 5)
        return (torch.rand(1, 5),)

class SingleLayerFunctionalLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个 FunctionalLinear 类的实例
        self.linear1 = FunctionalLinear()

    def forward(self, x):
        # 调用 FunctionalLinear 实例的 forward 方法进行前向传播
        x = self.linear1(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 调用 FunctionalLinear 实例的 get_example_inputs 方法获取示例输入
        return self.linear1.get_example_inputs()

class TwoLayerFunctionalLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建两个 FunctionalLinear 类的实例
        self.linear1 = FunctionalLinear()
        self.linear2 = FunctionalLinear()

    def forward(self, x):
        # 依次调用两个 FunctionalLinear 实例的 forward 方法进行前向传播
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 调用第一个 FunctionalLinear 实例的 get_example_inputs 方法获取示例输入
        return self.linear1.get_example_inputs()

class FunctionalLinearAddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建两个 FunctionalLinear 类的实例
        self.linear1 = FunctionalLinear()
        self.linear2 = FunctionalLinear()

    def forward(self, x):
        # 先调用第一个 FunctionalLinear 实例的 forward 方法进行前向传播
        x = self.linear1(x)
        # 对输出 x 进行加法操作，增加了一个常数 5
        x = torch.add(x, 5)
        # 再调用第二个 FunctionalLinear 实例的 forward 方法进行前向传播
        x = self.linear2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 调用第一个 FunctionalLinear 实例的 get_example_inputs 方法获取示例输入
        return self.linear1.get_example_inputs()

class FunctionalLinearReluModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个 FunctionalLinear 类的实例
        self.linear = FunctionalLinear()

    def forward(self, x):
        # 调用 FunctionalLinear 实例的 forward 方法进行前向传播
        x = self.linear(x)
        # 对输出 x 进行 ReLU 激活函数操作
        x = F.relu(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 调用 FunctionalLinear 实例的 get_example_inputs 方法获取示例输入
        return self.linear.get_example_inputs()

class FunctionalLinearReluLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个 FunctionalLinear 类的实例
        self.linear1 = FunctionalLinear()
        # 创建一个 ReLU 激活函数层
        self.relu = nn.ReLU()
        # 再创建一个 FunctionalLinear 类的实例
        self.linear2 = FunctionalLinear()

    def forward(self, x):
        # 调用第一个 FunctionalLinear 实例的 forward 方法进行前向传播
        x = self.linear1(x)
        # 调用 ReLU 激活函数层的 forward 方法进行前向传播
        x = self.relu(x)
        # 调用第二个 FunctionalLinear 实例的 forward 方法进行前向传播
        x = self.linear2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 调用第一个 FunctionalLinear 实例的 get_example_inputs 方法获取示例输入
        return self.linear1.get_example_inputs()

class FunctionalConv2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化卷积核权重，形状为 (3, 3, 3, 3)，随机初始化
        self.weight = torch.rand(3, 3, 3, 3)
        # 初始化卷积核偏置，长度为 3，随机初始化
        self.bias = torch.rand(3)
        # 设置卷积步长为 (1, 1)
        self.stride = (1, 1)
        # 设置填充大小为 (0, 0)
        self.padding = (0, 0)
        # 设置膨胀系数为 (1, 1)
        self.dilation = (1, 1)
        # 设置卷积操作的分组数为 1
        self.groups = 1

    def forward(self, x):
        # 调用 torch.nn.functional 中的二维卷积函数
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回一个示例输入，形状为 (1, 3, 5, 5)
        return (torch.rand(1, 3, 5, 5),)

class SingleLayerFunctionalConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个 FunctionalConv2d 类的实例
        self.conv1 = FunctionalConv2d()

    def forward(self, x):
        # 调用 FunctionalConv2d 实例的 forward 方法进行前向传播
        x = self.conv1(x)
        return x
    # 定义一个方法 `get_example_inputs`，返回一个元组，其中包含 self.conv1.get_example_inputs() 方法的返回值
    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 调用 self.conv1 对象的 get_example_inputs 方法，获取示例输入
        return self.conv1.get_example_inputs()
class TwoLayerFunctionalConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建两个 FunctionalConv2d 实例作为模型的卷积层
        self.conv1 = FunctionalConv2d()
        self.conv2 = FunctionalConv2d()

    def forward(self, x):
        # 对输入 x 分别应用两个卷积层
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回第一个卷积层的示例输入
        return self.conv1.get_example_inputs()

class FunctionalConvReluModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个 FunctionalConv2d 实例作为模型的卷积层
        self.conv = FunctionalConv2d()

    def forward(self, x):
        # 对输入 x 应用卷积和 ReLU 激活函数
        x = self.conv(x)
        x = F.relu(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回卷积层的示例输入
        return self.conv.get_example_inputs()

class FunctionalConvReluConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建两个 FunctionalConv2d 实例和一个 ReLU 激活层
        self.conv1 = FunctionalConv2d()
        self.relu = nn.ReLU()
        self.conv2 = FunctionalConv2d()

    def forward(self, x):
        # 对输入 x 应用第一个卷积层、ReLU 和第二个卷积层
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        # 返回第一个卷积层的示例输入
        return self.conv1.get_example_inputs()

class SkipQuantModel(torch.nn.Module):
    r"""We can skip quantization by explicitly
    setting qconfig of a submodule to None
    """
    def __init__(self):
        super().__init__()
        # 创建一个 InnerModule 实例和一个线性层
        self.sub = InnerModule()
        self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        # 对输入 x 应用 InnerModule 和线性层
        return self.fc(self.sub(x))

    def fuse_modules(self):
        # 调用 InnerModule 的 fuse_modules 方法
        self.sub.fuse_modules()

class AnnotatedSkipQuantModel(torch.nn.Module):
    r"""We can skip quantization by explicitly
    setting qconfig of a submodule to None
    """
    def __init__(self, qengine):
        super().__init__()
        # 设置量化配置和创建一个 QuantWrapper 包装的 InnerModule 实例以及一个线性层
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        self.sub = QuantWrapper(InnerModule())
        self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)
        # 设置线性层不被量化
        self.fc.qconfig = None

    def forward(self, x):
        # 对输入 x 应用量化的 InnerModule 和线性层
        return self.fc(self.sub(x))

    def fuse_modules(self):
        # 调用 QuantWrapper 内部的模块的 fuse_modules 方法
        self.sub.module.fuse_modules()

class QuantStubModel(torch.nn.Module):
    r"""A Module with manually inserted `QuantStub` and `DeQuantStub`
    """
    def __init__(self):
        super().__init__()
        # 设置量化配置，并创建 QuantStub 和 DeQuantStub，以及一个线性层
        self.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        # 对输入 x 应用量化、线性层，然后反量化
        x = self.quant(x)
        x = self.fc(x)
        return self.dequant(x)

class ManualLinearQATModel(torch.nn.Module):
    r"""A Module with manually inserted `QuantStub` and `DeQuantStub`
    """
    # 初始化函数，用于创建一个新的神经网络模型实例
    def __init__(self, qengine):
        # 调用父类的初始化方法
        super().__init__()
        # 获取指定量化引擎的默认量化训练配置
        self.qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine)
        # 创建量化装饰器，用于模拟量化过程
        self.quant = QuantStub()
        # 创建反量化装饰器，用于模拟反量化过程
        self.dequant = DeQuantStub()
        # 创建一个线性层，输入维度为5，输出维度为1，并将其转换为指定的浮点数类型
        self.fc1 = torch.nn.Linear(5, 1).to(dtype=torch.float)
        # 创建另一个线性层，输入维度为1，输出维度为10，并将其转换为指定的浮点数类型
        self.fc2 = torch.nn.Linear(1, 10).to(dtype=torch.float)

    # 前向传播函数，定义了数据在模型中的流动方式
    def forward(self, x):
        # 对输入数据进行量化处理
        x = self.quant(x)
        # 将量化后的数据输入第一个线性层进行计算
        x = self.fc1(x)
        # 将第一个线性层的输出作为输入，输入第二个线性层进行计算
        x = self.fc2(x)
        # 将第二个线性层的输出进行反量化处理，并作为前向传播函数的输出结果
        return self.dequant(x)
class ManualDropoutQATModel(torch.nn.Module):
    r"""A Module with manually inserted `QuantStub` and `DeQuantStub`
    """
    def __init__(self, qengine):
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine)  # 设置量化训练配置
        self.quant = QuantStub()  # 插入量化前处理
        self.dequant = DeQuantStub()  # 插入量化后处理
        self.fc1 = torch.nn.Linear(5, 1).to(dtype=torch.float)  # 创建线性层
        self.dropout = torch.nn.Dropout(0.5)  # 创建 dropout 层

    def forward(self, x):
        x = self.quant(x)  # 应用量化前处理
        x = self.fc1(x)  # 应用线性层
        x = self.dropout(x)  # 应用 dropout
        return self.dequant(x)  # 应用量化后处理并返回结果


class ManualLinearDynamicQATModel(torch.nn.Module):
    r"""A Module that uses a dynamic QAT by default.
    """
    def __init__(self, qconfig=None):
        super().__init__()
        self.qconfig = qconfig or default_dynamic_qat_qconfig  # 设置动态量化训练配置
        self.fc1 = torch.nn.Linear(5, 1).to(dtype=torch.float)  # 创建线性层
        self.fc2 = torch.nn.Linear(1, 10).to(dtype=torch.float)  # 创建另一个线性层

    def forward(self, x):
        x = self.fc1(x)  # 应用第一个线性层
        x = self.fc2(x)  # 应用第二个线性层
        return x  # 返回结果


class ManualConvLinearQATModel(torch.nn.Module):
    r"""A module with manually inserted `QuantStub` and `DeQuantStub`
    and contains both linear and conv modules
    """
    def __init__(self, qconfig=None):
        super().__init__()
        self.qconfig = qconfig if qconfig else torch.ao.quantization.get_default_qat_qconfig("qnnpack")  # 设置量化训练配置，默认为 qnnpack
        self.quant = QuantStub()  # 插入量化前处理
        self.dequant = DeQuantStub()  # 插入量化后处理
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=3).to(dtype=torch.float)  # 创建卷积层
        self.fc1 = torch.nn.Linear(64, 10).to(dtype=torch.float)  # 创建线性层
        self.fc2 = torch.nn.Linear(10, 10).to(dtype=torch.float)  # 创建另一个线性层

    def forward(self, x):
        x = self.quant(x)  # 应用量化前处理
        x = self.conv(x)  # 应用卷积层
        x = x.view(-1, 64).contiguous()  # 重塑张量形状
        x = self.fc1(x)  # 应用第一个线性层
        x = self.fc2(x)  # 应用第二个线性层
        return self.dequant(x)  # 应用量化后处理并返回结果


class ManualConvLinearSymmQATModel(ManualConvLinearQATModel):
    r"""Same as ManualConvLinearQATModule but with Symmetric Quantization.
    Supported only with qnnpack.
    """
    def __init__(self):
        super().__init__(default_symmetric_qnnpack_qat_qconfig)  # 使用对称量化配置继承自父类


class ManualEmbeddingBagLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.EmbeddingBag(num_embeddings=10, embedding_dim=12, mode='sum')  # 创建嵌入袋层
        self.emb.qconfig = default_embedding_qat_qconfig  # 设置嵌入袋层的量化训练配置
        self.quant = QuantStub()  # 插入量化前处理
        self.dequant = DeQuantStub()  # 插入量化后处理
        self.linear = nn.Linear(12, 1).to(dtype=torch.float)  # 创建线性层
        self.qconfig = get_default_qat_qconfig("qnnpack")  # 设置量化训练配置，默认为 qnnpack

    def forward(self, input: torch.Tensor, offsets: Optional[torch.Tensor] = None,
                per_sample_weights: Optional[torch.Tensor] = None):
        x = self.emb(input, offsets, per_sample_weights)  # 应用嵌入袋层
        x = self.quant(x)  # 应用量化前处理
        x = self.linear(x)  # 应用线性层
        return self.dequant(x)  # 应用量化后处理并返回结果


class DeFusedEmbeddingBagLinear(nn.Module):
    r"""A module to simulate QAT embedding bag with a linear layer,
    """
        this module uses a separate embedding and bagging op, similar
        to that which is described in the EmbeddingBag documentation.
    
        https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html
    """
    # 定义一个新的神经网络模块，继承自 nn.Module 类
    class CustomModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # 创建一个大小为 10 的嵌入层，每个嵌入向量的维度为 12
            self.emb = nn.Embedding(num_embeddings=10, embedding_dim=12)
            # 设置嵌入层的量化配置为默认的量化训练配置
            self.emb.qconfig = default_embedding_qat_qconfig
            # 定义一个求和操作，用于嵌入袋的聚合
            self.bagging_op = torch.sum
            # 创建一个量化存根（QuantStub），用于量化输入数据
            self.quant = QuantStub()
            # 创建一个去量化存根（DeQuantStub），用于去量化输出数据
            self.dequant = DeQuantStub()
            # 创建一个线性层，将输入大小为 12 的向量映射到大小为 1 的输出
            self.linear = nn.Linear(12, 1).to(dtype=torch.float)
            # 设置模型的量化配置为 QNNPACK 的默认量化训练配置
            self.qconfig = get_default_qat_qconfig("qnnpack")
    
        # 定义前向传播函数
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            # 对输入进行嵌入，并使用指定的维度进行求和（嵌入袋操作）
            x = self.bagging_op(self.emb(input), dim=1)
            # 对结果进行量化
            x = self.quant(x)
            # 对量化后的结果应用线性层
            x = self.linear(x)
            # 返回去量化后的结果
            return self.dequant(x)
class SubModelForFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个2通道到2通道的1x1卷积层，无偏置
        self.conv = nn.Conv2d(2, 2, 1, bias=None).to(dtype=torch.float)
        # 定义一个2通道的批归一化层
        self.bn = nn.BatchNorm2d(2).to(dtype=torch.float)

    def forward(self, x):
        # 执行卷积操作
        x = self.conv(x)
        # 执行批归一化操作
        x = self.bn(x)
        return x


class SubModelWithoutFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个2通道到2通道的1x1卷积层，无偏置
        self.conv = nn.Conv2d(2, 2, 1, bias=None).to(dtype=torch.float)
        # 定义一个ReLU激活函数层
        self.relu = nn.ReLU(inplace=False).to(dtype=torch.float)

    def forward(self, x):
        # 执行卷积操作，并通过ReLU激活函数处理
        return self.relu(self.conv(x))


class ModelForFusion(nn.Module):
    def __init__(self, qconfig):
        super().__init__()
        # 定义一个3通道到2通道的1x1卷积层，无偏置
        self.conv1 = nn.Conv2d(3, 2, 1, bias=None).to(dtype=torch.float)
        # 定义一个2通道的批归一化层
        self.bn1 = nn.BatchNorm2d(2).to(dtype=torch.float)
        # 定义一个ReLU激活函数层（inplace操作）
        self.relu1 = nn.ReLU(inplace=True).to(dtype=torch.float)
        # 定义一个SubModelForFusion的子模块
        self.sub1 = SubModelForFusion()
        # 定义一个SubModelWithoutFusion的子模块
        self.sub2 = SubModelWithoutFusion()
        # 定义一个线性层，输入维度36，输出维度10
        self.fc = nn.Linear(36, 10).to(dtype=torch.float)
        # 定义量化层
        self.quant = QuantStub()
        # 定义反量化层
        self.dequant = DeQuantStub()
        # 保存量化配置
        self.qconfig = qconfig
        # 定义一个3通道到2通道的1x1x1卷积层，无偏置
        self.conv2 = nn.Conv3d(3, 2, (1, 1, 1), bias=None).to(dtype=torch.float)
        # 定义一个ReLU激活函数层（非inplace操作）
        self.relu2 = nn.ReLU(inplace=False).to(dtype=torch.float)
        # 定义一个3通道的批归一化层
        self.bn2 = nn.BatchNorm3d(2).to(dtype=torch.float)
        # 定义一个ReLU激活函数层（inplace操作）
        self.relu3 = nn.ReLU(inplace=True).to(dtype=torch.float)
        # 定义一个3通道到3通道的1x2卷积层
        self.conv3 = nn.Conv1d(3, 3, 2).to(dtype=torch.float)
        # 定义一个3通道的批归一化层
        self.bn3 = nn.BatchNorm1d(3).to(dtype=torch.float)
        # 定义一个ReLU激活函数层（inplace操作）
        self.relu4 = nn.ReLU(inplace=True).to(dtype=torch.float)
        # 不对sub2进行量化
        self.sub2.qconfig = None
        # 不对fc进行量化
        self.fc.qconfig = None

    def forward(self, x):
        # 压缩维度为2的张量
        x = x.squeeze(2)
        # 执行量化操作
        x = self.quant(x)
        # 执行1维卷积操作
        x = self.conv3(x)
        # 执行1维批归一化操作
        x = self.bn3(x)
        # 执行ReLU激活函数操作
        x = self.relu4(x)
        # 增加一个维度为2的张量
        x = x.unsqueeze(2)
        # 增加两个维度为2的张量
        y = x.unsqueeze(2)
        # 执行1x1卷积操作
        x = self.conv1(x)
        # 执行2维批归一化操作
        x = self.bn1(x)
        # 执行ReLU激活函数操作
        x = self.relu1(x)
        # 执行SubModelForFusion的前向传播操作
        x = self.sub1(x)
        # 执行反量化操作
        x = self.dequant(x)
        # 执行SubModelWithoutFusion的前向传播操作
        x = self.sub2(x)
        # 重塑张量形状为（-1, 36）
        x = x.reshape(-1, 36).contiguous()
        # 执行线性层操作
        x = self.fc(x)
        # 执行3维卷积操作
        y = self.conv2(y)
        # 执行ReLU激活函数操作
        y = self.relu2(y)
        # 执行3维批归一化操作
        y = self.bn2(y)
        # 执行ReLU激活函数操作
        y = self.relu3(y)
        # 执行反量化操作
        y = self.dequant(y)
        return x

class ConvBNReLU(nn.Sequential):
    def __init__(self):
        super().__init__(
            # 定义一个3通道到3通道的1x1卷积层，无偏置
            nn.Conv2d(3, 3, 1, 1, bias=False),
            # 定义一个3通道的批归一化层
            nn.BatchNorm2d(3),
            # 定义一个ReLU激活函数层（非inplace操作）
            nn.ReLU(inplace=False)
        )

class ModelWithSequentialFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个3通道到3通道的1x1卷积层
        self.conv1 = nn.Conv2d(3, 3, 1)
        # 定义一个ReLU激活函数层（非inplace操作）
        self.relu1 = nn.ReLU(inplace=False)
        layers = []
        for i in range(3):
            # 将3个ConvBNReLU模块添加到layers列表中
            layers.append(ConvBNReLU())
        # 定义一个Sequential层，包含3个ConvBNReLU模块
        self.features = nn.Sequential(*layers)
        head = [nn.Linear(300, 10), nn.ReLU(inplace=False)]
        # 定义一个Sequential层，包含一个线性层和一个ReLU激活函数层
        self.classifier = nn.Sequential(*head)
        # 定义一个空的Sequential层
        self.seq = nn.Sequential()
        # 定义量化层
        self.quant = QuantStub()
        # 定义反量化层
        self.dequant = DeQuantStub()
    # 定义前向传播方法，接受输入张量 x
    def forward(self, x):
        # 将输入张量 x 进行量化处理
        x = self.quant(x)
        # 将量化后的张量 x 通过第一个卷积层 conv1
        x = self.conv1(x)
        # 对卷积层输出应用 ReLU 激活函数 relu1
        x = self.relu1(x)
        # 将经过 ReLU 激活函数后的张量 x 输入到特征提取层 features
        x = self.features(x)
        # 将特征提取层的输出张量 x 进行形状重塑为 (-1, 3 * 10 * 10) 的形状
        x = torch.reshape(x, (-1, 3 * 10 * 10))
        # 将重塑后的张量 x 输入到分类器层 classifier 进行分类
        x = self.classifier(x)
        # 将分类器的输出张量 x 输入到序列处理层 seq
        x = self.seq(x)
        # 将序列处理层的输出张量 x 进行反量化处理
        x = self.dequant(x)
        # 返回最终处理后的张量 x 作为前向传播的结果
        return x
# 定义带偏置项的融合模型类
class ModelForFusionWithBias(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层卷积层，输入通道数为3，输出通道数为2，卷积核大小为5，包含偏置项
        self.conv1 = nn.Conv2d(3, 2, 5, bias=True).to(dtype=torch.float)
        # 第一层批归一化层，输入通道数为2
        self.bn1 = nn.BatchNorm2d(2).to(dtype=torch.float)
        # ReLU 激活函数，inplace=True 表示在原地执行
        self.relu1 = nn.ReLU(inplace=True).to(dtype=torch.float)
        # 第二层卷积层，输入通道数为2，输出通道数为2，卷积核大小为1，包含偏置项
        self.conv2 = nn.Conv2d(2, 2, 1, bias=True).to(dtype=torch.float)
        # 第二层批归一化层，输入通道数为2
        self.bn2 = nn.BatchNorm2d(2).to(dtype=torch.float)
        # 量化模块的存根
        self.quant = QuantStub()
        # 反量化模块的存根
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)  # 执行输入量化
        x = self.conv1(x)   # 第一层卷积操作
        x = self.bn1(x)     # 第一层批归一化
        x = self.relu1(x)   # 第一层ReLU激活
        x = self.conv2(x)   # 第二层卷积操作
        x = self.bn2(x)     # 第二层批归一化
        x = self.dequant(x) # 执行输出反量化
        return x

# 定义线性层与批归一化融合模型类
class ModelForLinearBNFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 线性层，输入特征数为20，输出特征数为10
        self.fc = nn.Linear(20, 10)
        # 批归一化层，输入特征数为10，使用均匀分布初始化权重和偏置项
        self.bn = nn.BatchNorm1d(10)
        nn.init.uniform_(self.bn.weight)  # 初始化权重
        nn.init.uniform_(self.bn.bias)    # 初始化偏置项

    def forward(self, x):
        return self.bn(self.fc(x))  # 执行线性层与批归一化的融合操作

# 定义卷积转置与批归一化融合模型类
class ModelForConvTransposeBNFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层一维卷积转置，输入通道数为3，输出通道数为3，卷积核大小为1
        self.conv1 = nn.ConvTranspose1d(3, 3, 1)
        # 第一层批归一化层，输入通道数为3
        self.bn1 = nn.BatchNorm1d(3)
        # 第二层二维卷积转置，输入通道数为3，输出通道数为3，卷积核大小为1
        self.conv2 = nn.ConvTranspose2d(3, 3, 1)
        # 第二层批归一化层，输入通道数为3
        self.bn2 = nn.BatchNorm2d(3)
        # 第三层三维卷积转置，输入通道数为3，输出通道数为3，卷积核大小为1
        self.conv3 = nn.ConvTranspose3d(3, 3, 1)
        # 第三层批归一化层，输入通道数为3
        self.bn3 = nn.BatchNorm3d(3)

    def forward(self, x):
        x = self.conv1(x)   # 第一层一维卷积转置
        x = self.bn1(x)     # 第一层批归一化
        x = x.unsqueeze(2)  # 在第二维上增加维度
        x = self.conv2(x)   # 第二层二维卷积转置
        x = self.bn2(x)     # 第二层批归一化
        x = x.unsqueeze(2)  # 在第四维上增加维度
        x = self.conv3(x)   # 第三层三维卷积转置
        x = self.bn3(x)     # 第三层批归一化
        return x

# 定义带功能模块的模型类
class ModelWithFunctionals(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义 FloatFunctional 模块
        self.mycat = nnq.FloatFunctional()
        self.myadd = nnq.FloatFunctional()
        self.myadd_relu = nnq.FloatFunctional()
        self.mymatmul = nnq.FloatFunctional()
        # Tracing doesnt work yet for c10 ops with scalar inputs
        # https://github.com/pytorch/pytorch/issues/27097
        # self.my_scalar_add = nnq.FloatFunctional()
        # self.my_scalar_mul = nnq.FloatFunctional()

    def forward(self, x):
        y = self.mycat.cat([x, x, x])             # 执行 cat 操作
        z = self.myadd.add(y, y)                  # 执行 add 操作
        w = self.myadd_relu.add_relu(z, z)        # 执行 add_relu 操作
        u = self.mymatmul.matmul(w, w.T)          # 执行 matmul 操作
        # Tracing doesnt work yet for c10 ops with scalar inputs
        # https://github.com/pytorch/pytorch/issues/27097
        # w = self.my_scalar_add.add_scalar(w, -0.5)
        # w = self.my_scalar_mul.mul_scalar(w, 0.5)
        return u
    def __init__(self):
        super().__init__()
        norm_layer = nn.BatchNorm2d  # 定义规范化层为 BatchNorm2d
        inplanes = 3  # 输入平面数为 3
        self.conv1 = nn.Conv2d(inplanes, inplanes, (1, 1), bias=False)  # 定义一个无偏置的二维卷积层
        self.bn1 = norm_layer(inplanes)  # 使用规范化层对输入平面进行规范化
        self.relu1 = nn.ReLU()  # 定义第一个 ReLU 激活函数
        self.relu2 = nn.ReLU()  # 定义第二个 ReLU 激活函数
        self.downsample = torch.nn.Identity()  # 定义一个恒等映射层
        self.myop = nn.quantized.FloatFunctional()  # 定义一个量化为浮点数的操作
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 定义一个自适应平均池化层，输出大小为 (1, 1)
        self.fc = torch.nn.Linear(inplanes, 1)  # 定义一个线性层，将输入平面映射到单一输出

    def forward(self, x):
        out = self.conv1(x)  # 进行第一次卷积操作
        out = self.bn1(out)  # 进行规范化层操作
        out = self.relu1(out)  # 进行第一个 ReLU 激活操作
        identity = self.downsample(x)  # 获取输入的恒等映射
        out = self.myop.add(out, identity)  # 使用定义的量化浮点操作进行加法操作
        out = self.relu2(out)  # 进行第二个 ReLU 激活操作
        out = self.avgpool(out)  # 进行自适应平均池化操作
        out = torch.flatten(out, 1)  # 将输出展平为一维
        out = self.fc(out)  # 进行线性层操作
        return out

    def fuse_model(self):
        # TODO: remove this check and define two fuse_model function on this module
        if self.training:
            torch.ao.quantization.fuse_modules_qat(self, [['conv1', 'bn1', 'relu1']], inplace=True)
            # 在量化训练模式下，使用 QAT 方法融合卷积、规范化和ReLU激活函数模块
        else:
            torch.ao.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu1']], inplace=True)
            # 在推理模式下，使用传统方法融合卷积、规范化和ReLU激活函数模块
# 定义一个继承自 torch.nn.Module 的类，用于多个操作模型
class ModelMultipleOps(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 设置归一化层为 BatchNorm2d
        norm_layer = nn.BatchNorm2d
        # 输入平面数为 3
        inplanes = 3
        # 第一个卷积层，输入和输出通道数都是 inplanes，卷积核大小为 (1, 1)，无偏置
        self.conv1 = nn.Conv2d(inplanes, inplanes, (1, 1), bias=False)
        # 第二个卷积层，输入和输出通道数同上，卷积核大小为 (1, 1)，无偏置
        self.conv2 = nn.Conv2d(inplanes, inplanes, (1, 1), bias=False)
        # BatchNorm2d 归一化层，输入通道数为 inplanes
        self.bn1 = norm_layer(inplanes)
        # ReLU 激活函数
        self.relu1 = nn.ReLU()
        # ReLU 激活函数
        self.relu2 = nn.ReLU()
        # 跳跃连接恒等映射
        self.downsample = torch.nn.Identity()
        # 量化相关的加法操作
        self.skip_add = nn.quantized.FloatFunctional()
        # 量化相关的连接操作
        self.cat = nn.quantized.FloatFunctional()
        # 自适应平均池化层，输出尺寸为 (4, 4)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        # 全连接层，输入特征维度为 12，输出特征维度为 6
        self.fc = nn.Linear(12, 6)

    # 前向传播函数
    def forward(self, x):
        # 第一层卷积操作
        out = self.conv1(x)
        # BatchNorm2d 归一化层
        out = self.bn1(out)
        # ReLU 激活函数
        out = self.relu1(out)
        # 恒等映射
        identity = self.downsample(x)
        # 执行量化的加法操作
        out = self.skip_add.add(out, identity)
        # ReLU 激活函数
        out = self.relu2(out)
        # 自适应平均池化
        out = self.avgpool(out)
        # 第二层卷积操作
        out = self.conv2(out)
        # 最大池化操作，核大小为 2x2，步长为 2
        out = torch.nn.functional.max_pool2d(out, 2, 2)
        # 连接操作
        out = self.cat.cat([out, out])
        # 重新整形
        out = out.reshape(-1, 3 * 2 * 2)
        # 全连接层
        out = self.fc(out)
        return out


# 一个模型类，用于确保假量化和真量化的一致性
# 平均池化和均值操作不能准确地用假量化模拟，因此此模型中不包含这些操作
class ModelMultipleOpsNoAvgPool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 设置归一化层为 BatchNorm2d
        norm_layer = nn.BatchNorm2d
        # 输入平面数为 3
        inplanes = 3
        # 第一个卷积层，输入和输出通道数都是 inplanes，卷积核大小为 (1, 1)，无偏置
        self.conv1 = nn.Conv2d(inplanes, inplanes, (1, 1), bias=False)
        # 第二个卷积层，输入和输出通道数同上，卷积核大小为 (1, 1)，无偏置
        self.conv2 = nn.Conv2d(inplanes, inplanes, (1, 1), bias=False)
        # BatchNorm2d 归一化层，输入通道数为 inplanes
        self.bn1 = norm_layer(inplanes)
        # ReLU 激活函数
        self.relu1 = nn.ReLU()
        # ReLU 激活函数
        self.relu2 = nn.ReLU()
        # 量化相关的加法操作
        self.skip_add = nn.quantized.FloatFunctional()
        # 量化相关的连接操作
        self.cat = nn.quantized.FloatFunctional()
        # 最大池化层，池化核大小为 (4, 4)
        self.maxpool = nn.MaxPool2d((4, 4))
        # 全连接层，输入特征维度为 12，输出特征维度为 6
        self.fc = nn.Linear(12, 6)

    # 前向传播函数
    def forward(self, x):
        # 第一层卷积操作
        out = self.conv1(x)
        # BatchNorm2d 归一化层
        out = self.bn1(out)
        # ReLU 激活函数
        out = self.relu1(out)
        # 第二层卷积操作，跳过第一层的输出直接输入到第二层
        skip = self.conv2(x)
        # 执行量化的加法操作
        out = self.skip_add.add(out, skip)
        # ReLU 激活函数
        out = self.relu2(out)
        # 最大池化操作
        out = self.maxpool(out)
        # 第二层卷积操作
        out = self.conv2(out)
        # 最大池化操作，核大小为 2x2，步长为 2
        out = torch.nn.functional.max_pool2d(out, 2, 2)
        # 连接操作
        out = self.cat.cat([out, out])
        # 重新整形
        out = out.reshape(-1, 3 * 2 * 2)
        # 全连接层
        out = self.fc(out)
        return out


# 一个包含 EmbeddingBag 模块的模型类
class EmbeddingBagModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 设置 EmbeddingBag 层，包含 10 个嵌入，每个嵌入维度为 12
        self.emb = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12,
                                         include_last_offset=True, scale_grad_by_freq=False, mode='sum')

    # 前向传播函数
    def forward(self, indices, offsets, per_sample_weights):
        return self.emb(indices, offsets, per_sample_weights)


# 一个包含 Embedding 模块的模型类
class EmbeddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 设置 Embedding 层，包含 10 个嵌入，每个嵌入维度为 12
        self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)

    # 前向传播函数
    def forward(self, indices):
        return self.emb(indices)
class EmbeddingWithStaticLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化一个 EmbeddingBag 模块，设置嵌入的数量和维度
        self.emb = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12)
        # 初始化一个线性层，输入维度为4，输出维度为2
        self.fc = torch.nn.Linear(4, 2)
        # 设置嵌入模块的量化配置为 float_qparams_weight_only_qconfig
        self.emb.qconfig = float_qparams_weight_only_qconfig
        # 设置当前模块的量化配置为 default_qconfig
        self.qconfig = default_qconfig
        # 初始化量化的前处理模块
        self.quant = QuantStub()
        # 初始化量化的后处理模块
        self.dequant = DeQuantStub()

    def forward(self, indices, offsets, linear_in):
        # 对输入的索引和偏移量应用嵌入模块
        emb = self.emb(indices, offsets)
        # 对线性层输入进行量化处理
        q_x = self.quant(linear_in)
        # 应用线性层
        fc = self.fc(q_x)
        # 对线性层输出进行反量化处理
        fc = self.dequant(fc)
        # 将处理后的特征拼接在一起
        features = torch.cat([fc] + [emb], dim=1)
        return features

class DenseTopMLP(nn.Module):

    def __init__(self, dense_dim, dense_out, embedding_dim, top_out_in, top_out_out) -> None:
        super().__init__()

        # 初始化稠密层的多层感知机（MLP）
        self.dense_mlp = nn.Sequential(
            nn.Linear(dense_dim, dense_out),
        )
        # 初始化顶层 MLP
        self.top_mlp = nn.Sequential(
            nn.Linear(dense_out + embedding_dim, top_out_in),
            nn.Linear(top_out_in, top_out_out),
        )

    def forward(
        self,
        sparse_feature: torch.Tensor,
        dense: torch.Tensor,
    ) -> torch.Tensor:
        # 应用稠密层的 MLP
        dense_feature = self.dense_mlp(dense)
        # 将稠密特征和稀疏特征拼接在一起
        features = torch.cat([dense_feature] + [sparse_feature], dim=1)

        # 应用顶层 MLP
        out = self.top_mlp(features)
        return out

# 包装嵌入袋的薄包装器，因为目前不支持在 nn.EmbeddingBag 内部进行跟踪
class EmbBagWrapper(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        # 初始化嵌入袋模块
        self.emb_bag = nn.EmbeddingBag(num_embeddings, embedding_dim, mode='sum')

    def forward(self, indices, offsets):
        # 应用嵌入袋模块
        return self.emb_bag(indices, offsets)

class SparseNNModel(nn.Module):
    _NUM_EMBEDDINGS = 10
    _EMBEDDING_DIM = 5
    _DENSE_DIM = 4
    _DENSE_OUTPUT = 2
    _TOP_OUT_IN = 2
    _TOP_OUT_OUT = 2
    _TOP_MLP_DIM = 1

    def __init__(self) -> None:
        super().__init__()

        # 初始化稀疏神经网络模型的嵌入袋部分
        self.model_sparse = EmbBagWrapper(self._NUM_EMBEDDINGS, self._EMBEDDING_DIM)
        # 初始化稀疏神经网络模型的顶层 MLP 部分
        self.dense_top = DenseTopMLP(
            self._DENSE_DIM, self._DENSE_OUTPUT, self._EMBEDDING_DIM, self._TOP_OUT_IN,
            self._TOP_OUT_OUT)

    def forward(
        self,
        sparse_indices: torch.Tensor,
        sparse_offsets: torch.Tensor,
        dense: torch.Tensor,
    ) -> torch.Tensor:

        # 应用嵌入袋模块
        sparse_feature = self.model_sparse(sparse_indices, sparse_offsets)
        # 应用顶层 MLP
        out = self.dense_top(sparse_feature, dense)

        return out

class TestHelperModules:
    # 卷积和线性层的模型，用于测试
    class Conv2dPropAnnotaton(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 初始化卷积层，输入通道数为3，输出通道数为3，卷积核大小为3x3
            self.conv = torch.nn.Conv2d(3, 3, 3)
            # 初始化线性层，输入维度为3，输出维度为3
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            # 应用卷积层
            x = self.conv(x)
            # 将输出张量视图重塑为二维张量
            x = x.view(-1, 3)
            # 应用硬切线性激活函数
            x = torch.nn.functional.hardtanh(x, -0.5, 0.5)
            # 应用线性层
            x = self.linear(x)
            return x
    # 定义一个继承自 torch.nn.Module 的类 Conv2dWithObsSharingOps
    class Conv2dWithObsSharingOps(torch.nn.Module):
        # 初始化函数，用于设置网络结构
        def __init__(self):
            super().__init__()  # 调用父类的初始化函数
            self.conv = torch.nn.Conv2d(3, 3, 3)  # 定义一个2维卷积层，输入通道数为3，输出通道数为3，卷积核大小为3x3
            self.hardtanh = torch.nn.Hardtanh()  # 定义一个硬切线激活函数
            self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))  # 定义一个自适应平均池化层，输出大小为1x1

        # 前向传播函数，定义了数据从输入到输出的流程
        def forward(self, x):
            x = self.conv(x)  # 对输入数据进行卷积操作
            x = self.adaptive_avg_pool2d(x)  # 对卷积后的数据进行自适应平均池化
            x = self.hardtanh(x)  # 对平均池化后的数据进行硬切线激活函数处理
            x = torch.mean(x)  # 对处理后的数据取平均值
            return x  # 返回处理后的结果

    # 定义一个继承自 torch.nn.Module 的类 Conv2dWithTwoLinearPermute
    class Conv2dWithTwoLinearPermute(torch.nn.Module):
        # 初始化函数，用于设置网络结构
        def __init__(self):
            super().__init__()  # 调用父类的初始化函数
            self.conv = torch.nn.Conv2d(3, 16, 3)  # 定义一个2维卷积层，输入通道数为3，输出通道数为16，卷积核大小为3x3
            self.linear1 = torch.nn.Linear(16, 8, bias=False)  # 定义一个全连接层，输入大小为16，输出大小为8，无偏置
            self.linear2 = torch.nn.Linear(8, 8)  # 定义另一个全连接层，输入大小为8，输出大小为8

        # 前向传播函数，定义了数据从输入到输出的流程
        def forward(self, x):
            conv_out = self.conv(x)  # 对输入数据进行卷积操作
            permute_out = torch.permute(conv_out, (0, 2, 3, 1))  # 对卷积结果进行维度置换
            return self.linear2(self.linear1(permute_out))  # 分别对维度置换后的数据进行两次全连接层操作

    # 定义一个继承自 torch.nn.Module 的类 Conv2dWithTwoLinear
    class Conv2dWithTwoLinear(torch.nn.Module):
        # 初始化函数，用于设置网络结构
        def __init__(self):
            super().__init__()  # 调用父类的初始化函数
            self.conv = torch.nn.Conv2d(3, 16, 3)  # 定义一个2维卷积层，输入通道数为3，输出通道数为16，卷积核大小为3x3
            self.linear1 = torch.nn.Linear(64, 8, bias=False)  # 定义一个全连接层，输入大小为64，输出大小为8，无偏置
            self.linear2 = torch.nn.Linear(8, 8)  # 定义另一个全连接层，输入大小为8，输出大小为8

        # 前向传播函数，定义了数据从输入到输出的流程
        def forward(self, x):
            conv_out = self.conv(x)  # 对输入数据进行卷积操作
            reshape_out = torch.reshape(conv_out, (2, 64))  # 对卷积结果进行重塑操作
            return self.linear2(self.linear1(reshape_out))  # 分别对重塑后的数据进行两次全连接层操作

    # 定义一个继承自 torch.nn.Module 的类 ConvLinearWPermute
    class ConvLinearWPermute(torch.nn.Module):
        # 初始化函数，用于设置网络结构
        def __init__(self):
            super().__init__()  # 调用父类的初始化函数
            self.conv = torch.nn.Conv2d(3, 8, 3)  # 定义一个2维卷积层，输入通道数为3，输出通道数为8，卷积核大小为3x3
            self.linear1 = torch.nn.Linear(8, 8)  # 定义一个全连接层，输入大小为8，输出大小为8

        # 前向传播函数，定义了数据从输入到输出的流程
        def forward(self, x):
            conv_out = self.conv(x)  # 对输入数据进行卷积操作
            permute_out = torch.permute(conv_out, (0, 2, 3, 1))  # 对卷积结果进行维度置换
            return self.linear1(permute_out)  # 对维度置换后的数据进行全连接层操作

    # 定义一个继承自 torch.nn.Module 的类 TwoLinearModule
    class TwoLinearModule(torch.nn.Module):
        # 初始化函数，用于设置网络结构
        def __init__(self):
            super().__init__()  # 调用父类的初始化函数
            self.linear1 = torch.nn.Linear(8, 16, bias=False)  # 定义一个全连接层，输入大小为8，输出大小为16，无偏置
            self.linear2 = torch.nn.Linear(16, 8)  # 定义另一个全连接层，输入大小为16，输出大小为8

        # 前向传播函数，定义了数据从输入到输出的流程
        def forward(self, x):
            return self.linear2(self.linear1(x))  # 分别对输入数据进行两次全连接层操作

    # 定义一个继承自 torch.nn.Module 的类 ConvMaxPool2d
    class ConvMaxPool2d(torch.nn.Module):
        # 初始化函数，用于设置网络结构
        def __init__(self):
            super().__init__()  # 调用父类的初始化函数
            self.conv = torch.nn.Conv2d(2, 2, 1)  # 定义一个2维卷积层，输入通道数为2，输出通道数为2，卷积核大小为1x1
            self.pool = torch.nn.MaxPool2d(1, 1)  # 定义一个最大池化层，池化核大小为1x1，步幅为1

        # 前向传播函数，定义了数据从输入到输出的流程
        def forward(self, x):
            x = self.conv(x)  # 对输入数据进行卷积操作
            x = self.pool(x)  # 对卷积结果进行最大池化操作
            return x  # 返回池化后的结果

    # 定义一个继承自 torch.nn.Module 的类 ConvWithAdaptiveAvgPool2d
    class ConvWithAdaptiveAvgPool2d(torch.nn.Module):
        # 初始化函数，用于设置网络结构
        def __init__(self):
            super().__init__()  # 调用父类的初始化函数
            self.conv = torch.nn.Conv2d(3, 3, 3)  # 定义一个2维卷积层，输入通道数为3，输出通道数为3，卷积核大小为3x3
            self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))  # 定义一个自适应平均池化层，输出大小为1x1

        # 前向传播函数，定义了数据从输入到输出的流程
        def forward(self, x):
            x = self.conv(x)  # 对输入数据进行卷积操作
            x = self.adaptive_avg_pool2d(x)  # 对卷积结果进行自适应平均池化操作
            return x  # 返回池化后的结果
    # 定义一个带有批量归一化、ReLU激活函数的卷积模块类
    class ConvWithBNRelu(torch.nn.Module):
        def __init__(self, relu, dim=2, bn=True, bias=True):
            super().__init__()
            # 根据维度选择合适的卷积层和批量归一化层
            convs = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d}
            bns = {1: torch.nn.BatchNorm1d, 2: torch.nn.BatchNorm2d}
            # 创建卷积层，输入通道数、输出通道数、卷积核大小
            self.conv = convs[dim](3, 3, 3, bias=bias)

            # 根据参数决定是否添加批量归一化层
            if bn:
                self.bn = bns[dim](3            else:
                self.bn = torch.nn.Identity()
            # 根据参数决定是否添加ReLU激活函数
            if relu:
                self.relu = torch.nn.ReLU()
            else:
                self.relu = torch.nn.Identity()

        # 前向传播函数
        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            return self.relu(x)

    # 定义一个带有批量归一化、ReLU激活函数的反卷积模块类
    class ConvTWithBNRelu(torch.nn.Module):
        def __init__(self, relu, dim=2, bn=True, bias=True):
            super().__init__()
            # 根据维度选择合适的反卷积层和批量归一化层
            convts = {1: torch.nn.ConvTranspose1d, 2: torch.nn.ConvTranspose2d}
            bns = {1: torch.nn.BatchNorm1d, 2: torch.nn.BatchNorm2d}
            # 创建反卷积层，输入通道数、输出通道数、卷积核大小
            self.convt = convts[dim](3, 3, 3, bias=bias)

            # 根据参数决定是否添加批量归一化层
            if bn:
                self.bn = bns[dim](3            else:
                self.bn = torch.nn.Identity()
            # 根据参数决定是否添加ReLU激活函数
            if relu:
                self.relu = torch.nn.ReLU()
            else:
                self.relu = torch.nn.Identity()

        # 前向传播函数
        def forward(self, x):
            x = self.convt(x)
            x = self.bn(x)
            return self.relu(x)

    # 定义一个先经过2D卷积再经过1D卷积的模块类
    class Conv2dThenConv1d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 创建一个2D卷积层和一个1D卷积层
            self.conv1d = torch.nn.Conv1d(3, 3, 3)
            self.conv2d = torch.nn.Conv2d(3, 3, 3)

        # 前向传播函数
        def forward(self, x):
            x = self.conv2d(x)
            x = x.squeeze(0)  # 去除批次维度
            x = self.conv1d(x)
            return x

        # 返回一个示例输入的函数
        def example_inputs(self):
            return (torch.randn(1, 3, 5, 5),)

    # 定义一个带有通道拼接的2D卷积模块类
    class Conv2dWithCat(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 创建两个2D卷积层
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.conv2 = torch.nn.Conv2d(3, 3, 3)

        # 前向传播函数，输入两个张量x和y
        def forward(self, x, y):
            x = self.conv1(x)
            y = self.conv2(y)
            z = torch.cat([x, y], dim=1)  # 在通道维度上拼接x和y
            return z

    # 定义一个带有两次通道拼接的2D卷积模块类
    class Conv2dWithTwoCat(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 创建两个2D卷积层
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.conv2 = torch.nn.Conv2d(3, 3, 3)

        # 前向传播函数，输入四个张量x1、x2、x3、x4
        def forward(self, x1, x2, x3, x4):
            x1 = self.conv1(x1)
            x2 = self.conv2(x2)
            y = torch.cat([x1, x2], dim=1)  # 在通道维度上拼接x1和x2
            z = x3 + x4  # 直接对x3和x4进行相加
            w = torch.cat([z, y])  # 在通道维度上拼接z和y
            return w

    # 定义一个进行三次张量相加的模块类
    class ThreeAdd(torch.nn.Module):
        # 前向传播函数，输入四个张量x1、x2、x3、x4
        def forward(self, x1, x2, x3, x4):
            y = x1 + x2  # 第一次相加
            z = x3 + x4  # 第二次相加
            w = y + z  # 第三次相加
            return w
    # 定义一个继承自torch.nn.Module的嵌入模块类
    class EmbeddingModule(torch.nn.Module):
        # 初始化方法，设置一个大小为10，维度为12的嵌入层
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)

        # 前向传播方法，接收索引作为输入，并返回对应的嵌入向量
        def forward(self, indices):
            return self.emb(indices)

    # 定义一个继承自torch.nn.Module的嵌入、卷积、线性组合模块类
    class EmbeddingConvLinearModule(torch.nn.Module):
        # 初始化方法，设置一个大小为10，维度为8的嵌入层，一个卷积层(输入通道8，输出通道16，卷积核大小为(1, 3))，一个线性层(输入大小16，输出大小8)
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=8)
            self.conv = torch.nn.Conv2d(8, 16, (1, 3))
            self.linear = torch.nn.Linear(16, 8)

        # 前向传播方法，接收索引作为输入，并返回线性层的输出结果
        def forward(self, indices):
            embeddings = self.emb(indices)  # 获取嵌入向量
            embeddings = torch.unsqueeze(embeddings, dim=0)  # 在第0维上增加一个维度
            embeddings = torch.permute(embeddings, (0, 3, 1, 2))  # 将维度顺序重新排列
            conv_out = self.conv(embeddings)  # 进行卷积操作
            conv_out = torch.permute(conv_out, (0, 2, 3, 1))  # 将维度顺序重新排列
            conv_out = torch.squeeze(conv_out, dim=0)  # 去除第0维
            return self.linear(conv_out)  # 返回线性层的输出结果

    # 定义一个继承自torch.nn.Module的加法、原地加法组合模块类
    class AddInplaceAdd(torch.nn.Module):
        # 前向传播方法，接收两个输入x和y，进行加法和原地加法操作，并返回结果
        def forward(self, x, y):
            x = x + y  # 加法操作
            x += y  # 原地加法操作
            return x

    # 定义一个继承自torch.nn.Module的乘法、原地乘法组合模块类
    class MulInplaceMul(torch.nn.Module):
        # 前向传播方法，接收两个输入x和y，进行乘法和原地乘法操作，并返回结果
        def forward(self, x, y):
            x = x * y  # 乘法操作
            x *= y  # 原地乘法操作
            return x

    # 定义一个继承自torch.nn.Module的加法、乘法标量组合模块类
    class AddMulScalar(torch.nn.Module):
        # 前向传播方法，接收一个输入x，进行加法、乘法和相应的原地操作，并返回结果
        def forward(self, x):
            x = x + 3  # 加法操作
            x = x * 3  # 乘法操作
            x += 3  # 原地加法操作
            x *= 3  # 原地乘法操作
            return x

    # 定义一个继承自torch.nn.Module的卷积、批归一化、ReLU组合模块类
    class ConvBnReLU2dAndLinearReLU(torch.nn.Module):
        # 初始化方法，设置一个包含ReLU的卷积、批归一化组合对象和一个线性层
        def __init__(self):
            super().__init__()
            self.conv_bn_relu = TestHelperModules.ConvWithBNRelu(relu=True)  # 创建一个带有ReLU的卷积、批归一化组合对象
            self.linear = torch.nn.Linear(3, 8, bias=False)  # 创建一个线性层，输入大小为3，输出大小为8，无偏置项
            self.relu = torch.nn.ReLU()  # 创建一个ReLU激活函数对象

        # 前向传播方法，接收输入x，通过卷积、批归一化、ReLU操作和线性层操作后返回结果
        def forward(self, x):
            x = self.conv_bn_relu(x)  # 经过卷积、批归一化和ReLU操作
            permute_out = torch.permute(x, (0, 2, 3, 1))  # 将维度顺序重新排列
            linear_out = self.linear(permute_out)  # 线性层操作
            return linear_out  # 返回线性层的输出结果

    # 定义一个继承自torch.nn.Module的分组卷积模块类
    class GroupwiseConv2d(torch.nn.Module):
        # 初始化方法，设置一个分组为2的卷积层，输入通道为4，输出通道为4，卷积核大小为3x3
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 4, 3, groups=2)

        # 前向传播方法，接收输入x，经过分组卷积后返回结果
        def forward(self, x):
            return self.conv(x)

        # 返回一个示例输入，形状为(2, 4, 10, 10)
        def example_inputs(self):
            return (torch.randn(2, 4, 10, 10),)

    # 定义一个继承自torch.nn.Module的线性、ReLU组合模块类
    class LinearReluModel(torch.nn.Module):
        # 初始化方法，设置一个线性层，输入大小为5，输出大小为5，并创建一个ReLU激活函数对象
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)  # 创建一个线性层，输入大小为5，输出大小为5
            self.relu = torch.nn.ReLU()  # 创建一个ReLU激活函数对象

        # 前向传播方法，接收输入x，通过线性层和ReLU激活函数操作后返回结果
        def forward(self, x):
            x = self.relu(self.fc(x))  # 线性层和ReLU操作
            return x  # 返回ReLU激活后的结果
```