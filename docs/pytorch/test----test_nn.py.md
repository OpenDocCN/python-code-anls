# `.\pytorch\test\test_nn.py`

```py
# Owner(s): ["module: nn"]

# 导入必要的库和模块
import contextlib  # 上下文管理工具
import math  # 数学函数库
import random  # 随机数生成库
import unittest  # 单元测试框架
import io  # 输入输出流
import itertools  # 迭代工具库
import warnings  # 警告控制工具
import pickle  # Python对象序列化库
import re  # 正则表达式库
from copy import deepcopy  # 深度拷贝函数
from itertools import product  # 迭代器工具库中的笛卡尔积函数
from functools import partial  # 函数工具库中的函数部分应用工具
from collections import OrderedDict  # 有序字典
from unittest import SkipTest  # 单元测试框架中的跳过测试用例工具

import torch  # PyTorch主库
from torch import inf, nan  # 正无穷和NaN常量
import torch.autograd.forward_ad as fwAD  # 自动求导库的前向自动求导模块
import torch.backends.cudnn as cudnn  # cuDNN后端接口
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数模块
import torch.nn.utils.rnn as rnn_utils  # 神经网络工具库中的RNN工具
from torch.nn.utils import clip_grad_norm_, clip_grad_value_  # 梯度裁剪工具函数
from torch.nn.utils import parameters_to_vector, vector_to_parameters  # 参数向量化和反向还原工具
from torch.nn.utils.fusion import fuse_conv_bn_weights  # 融合卷积和批归一化权重工具
from torch.nn.utils.fusion import fuse_linear_bn_weights  # 融合线性层和批归一化权重工具
from torch.nn import Parameter  # 神经网络参数类
from torch.nn.parallel._functions import Broadcast  # 广播函数
from torch.testing._internal.common_dtype import integral_types, get_all_math_dtypes, floating_types  # 数据类型相关工具
from torch.testing._internal.common_utils import freeze_rng_state, run_tests, TestCase, skipIfNoLapack, skipIfRocm, \
    TEST_NUMPY, TEST_SCIPY, TEST_WITH_CROSSREF, TEST_WITH_ROCM, \
    download_file, get_function_arglist, load_tests, skipIfMps, \
    IS_PPC, \
    parametrize as parametrize_test, subtest, instantiate_parametrized_tests, \
    skipIfTorchDynamo, gcIfJetson, set_default_dtype  # 测试工具和辅助函数
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU, TEST_CUDNN, PLATFORM_SUPPORTS_FLASH_ATTENTION  # CUDA相关测试工具
from torch.testing._internal.common_nn import NNTestCase, NewModuleTest, CriterionTest, \
    module_tests, criterion_tests, loss_reference_fns, _create_basic_net, \
    ctcloss_reference, new_module_tests, single_batch_reference_fn, _test_bfloat16_ops, _test_module_empty_input  # 神经网络测试相关工具
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, \
    dtypesIfCUDA, precisionOverride, skipCUDAIfCudnnVersionLessThan, onlyCUDA, onlyCPU, \
    skipCUDAIfRocm, skipCUDAIf, skipCUDAIfNotRocm, \
    onlyNativeDeviceTypes, deviceCountAtLeast, largeTensorTest, expectedFailureMeta, skipMeta, get_all_device_types  # 设备类型测试工具

from hypothesis import given  # 基于假设的测试工具
import torch.testing._internal.hypothesis_utils as hu  # 假设测试工具
from torch.testing._internal.common_utils import _assertGradAndGradgradChecks, gradcheck, gradgradcheck, \
    GRADCHECK_NONDET_TOL  # 梯度检查相关工具
from torch.testing._internal.common_utils import dtype2prec_DONTUSE  # 数据类型到精度映射（不建议使用）
from torch.testing._internal.common_cuda import tf32_on_and_off, tf32_is_not_fp32, tf32_off, tf32_on  # CUDA相关工具
from torch.types import _TensorOrTensors  # 张量或张量列表类型
from torch.testing._internal.common_mkldnn import bf32_on_and_off  # MKL-DNN相关工具

AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()  # 检查是否在ROCM环境下或者开启了TF32模式

# load_tests函数从common_utils中导入，用于自动过滤测试用例以在沙堡上进行分片。这行代码用于抑制flake警告。
load_tests = load_tests

if TEST_SCIPY:
    import scipy.signal  # 科学计算库中的信号处理模块
    import scipy.ndimage  # 科学计算库中的图像处理模块

if TEST_NUMPY:
    import numpy as np  # 数值计算库
# 定义一个测试类 TestNN，继承自 NNTestCase，用于测试神经网络模块
class TestNN(NNTestCase):
    # 启用 CUDA 内存泄漏检查
    _do_cuda_memory_leak_check = True
    # 启用非默认 CUDA 流检查
    _do_cuda_non_default_stream = True

    # 定义一个辅助方法 _forward，用于执行模块的前向传播
    def _forward(self, module, input: _TensorOrTensors):
        # 冻结随机数生成器状态
        with freeze_rng_state():
            # 如果输入是一个元组，则调用模块的 __call__ 方法
            if isinstance(input, tuple):
                return module(*input)
            else:
                return module(input)

    # 定义一个辅助方法 _backward，用于执行模块的反向传播
    def _backward(self, module, input: _TensorOrTensors, output, grad_output, create_graph=False):
        # 对输出执行反向传播，保留计算图
        output.backward(grad_output, retain_graph=True, create_graph=create_graph)
        # 如果输入是一个元组，则返回每个输入的梯度数据，否则返回输入的梯度数据
        if isinstance(input, tuple):
            return tuple(i.grad.data if i.grad is not None else None for i in input)
        else:
            return input.grad.data if input.grad is not None else None

    # 定义一个辅助方法 _forward_criterion，用于执行标准的前向传播
    def _forward_criterion(self, criterion, input, target, extra_args=None):
        # 如果额外参数为空，则初始化为一个空元组
        if extra_args is None:
            extra_args = tuple()
        # 如果输入是一个元组，则将输入、目标和额外参数合并传入标准
        if isinstance(input, tuple):
            args = input + (target,) + extra_args
            output = criterion(*args)
        else:
            output = criterion(input, target, *extra_args)
        return output

    # 定义一个辅助方法 _backward_criterion，用于执行标准的反向传播
    def _backward_criterion(self, criterion, input, output, target, gradOutput=None, extra_args=None):
        # 如果额外参数为空，则初始化为一个空元组
        if extra_args is None:
            extra_args = tuple()
        # 如果输入是一个元组，则将输入和输出转为元组
        input_tuple = input if isinstance(input, tuple) else (input,)
        output_tuple = output if isinstance(output, tuple) else (output,)
        # 将所有输入的梯度数据清零
        for i in input_tuple:
            if i.grad is not None:
                i.grad.data.zero_()
        # 构造参数列表，并根据 gradOutput 执行反向传播
        args = input_tuple + (target,) + extra_args
        if gradOutput is None:
            gradOutput = torch.ones(())
        criterion(*args).backward(gradOutput.to(output_tuple[0]))
        # 如果输入是一个元组，则返回每个输入的梯度数据，否则返回输入的梯度数据
        if isinstance(input, tuple):
            return tuple(i.grad.data for i in input)
        else:
            return input.grad.data

    # 定义一个辅助方法 _zero_grad_parameters，用于将模块的梯度参数清零
    def _zero_grad_parameters(self, module):
        # 遍历模块的所有参数，如果梯度不为空，则清零梯度
        for p in module.parameters():
            if p.grad is not None:
                with torch.no_grad():
                    p.grad.zero_()
                p.grad.detach_()

    # 定义一个辅助方法 _get_parameters，用于获取模块的参数及其梯度
    def _get_parameters(self, module):
        # 定义两个空列表，分别存储参数和参数的梯度
        params = []
        d_params = []
        # 遍历模块的所有参数，分别存储参数和参数的梯度
        for p in module.parameters():
            params.append(p)
            d_params.append(p.grad)
        return params, d_params

    # 定义一个测试方法 test_parse_to，用于测试 THPMemoryFormat_New 的错误使用
    def test_parse_to(self):
        # 断言 THPMemoryFormat_New 的 repr 是 "torch.contiguous_format"
        self.assertEqual(
            repr(torch._C._nn._parse_to(memory_format=torch.contiguous_format)[3]),
            "torch.contiguous_format"
        )
    # 测试函数，验证是否正确设置了 requires_grad 属性
    def test_requires_grad_(self):
        # 创建一个基础网络模型，选择最后一层
        m = _create_basic_net()[-1]
        # 断言缓冲区数量大于零，否则测试失败
        assert len(list(m.buffers())) > 0, 'invalid test'
        # 断言所有缓冲区的 requires_grad 属性均为 False，否则测试失败
        assert all(not b.requires_grad for b in m.buffers()) > 0, 'invalid test'
        # 断言参数数量大于零，否则测试失败
        assert len(list(m.parameters())) > 0, 'invalid test'
        # 断言所有参数的 requires_grad 属性均为 True，否则测试失败
        assert all(p.requires_grad for p in m.parameters()) > 0, 'invalid test'
        # 针对 requires_grad 属性为 False 和 True 两种情况进行测试
        for requires_grad in (False, True):
            # 验证设置 requires_grad 后返回的对象是自身
            self.assertIs(m.requires_grad_(requires_grad), m)
            # 验证所有参数的 requires_grad 属性是否正确设置
            for p in m.parameters():
                self.assertEqual(p.requires_grad, requires_grad)
            # 验证所有缓冲区的 requires_grad 属性应为 False
            for b in m.buffers():
                self.assertFalse(b.requires_grad)

    # 测试函数，验证模块在反序列化时的兼容性
    def test_module_backcompat(self):
        # 忽略来自 torch.serialization 的 SourceChangeWarning
        from torch.serialization import SourceChangeWarning
        # 下载模型文件并加载
        path = download_file('https://download.pytorch.org/test_data/linear.pt')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SourceChangeWarning)
            m = torch.load(path)
        # 创建输入张量
        input = torch.randn(2, 3, dtype=torch.float)
        # 验证模型对输入的输出尺寸是否符合预期
        self.assertEqual(m(input).size(), (2, 5))

    # 测试函数，验证模块在使用多重继承时的初始化顺序
    def test_module_super_init(self):
        # 定义一个带有初始化的混合类 MyMixin
        class MyMixin:
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.mixin_init = True

        # 定义一个继承自 MyMixin 和 nn.Module 的类 MyModuleWithMixinBefore
        class MyModuleWithMixinBefore(MyMixin, nn.Module):
            pass

        # 定义一个继承自 nn.Module 和 MyMixin 的类 MyModuleWithMixinAfter
        class MyModuleWithMixinAfter(nn.Module, MyMixin):
            pass

        # 验证 MyModuleWithMixinBefore 实例是否具有 mixin_init 属性
        self.assertTrue(hasattr(MyModuleWithMixinBefore(), 'mixin_init'))
        # 验证 MyModuleWithMixinAfter 实例是否不具有 mixin_init 属性
        self.assertFalse(hasattr(MyModuleWithMixinAfter(), 'mixin_init'))

        # 启用 nn.Module 的 call_super_init 属性
        nn.Module.call_super_init = True
        # 验证 MyModuleWithMixinBefore 实例是否具有 mixin_init 属性
        self.assertTrue(hasattr(MyModuleWithMixinBefore(), 'mixin_init'))
        # 验证 MyModuleWithMixinAfter 实例是否具有 mixin_init 属性
        self.assertTrue(hasattr(MyModuleWithMixinAfter(), 'mixin_init'))
        # 关闭 nn.Module 的 call_super_init 属性
        nn.Module.call_super_init = False

        # 启用 MyModuleWithMixinBefore 的 call_super_init 属性
        MyModuleWithMixinBefore.call_super_init = True
        # 启用 MyModuleWithMixinAfter 的 call_super_init 属性
        MyModuleWithMixinAfter.call_super_init = True
        # 验证 MyModuleWithMixinBefore 实例是否具有 mixin_init 属性
        self.assertTrue(hasattr(MyModuleWithMixinBefore(), 'mixin_init'))
        # 验证 MyModuleWithMixinAfter 实例是否具有 mixin_init 属性
        self.assertTrue(hasattr(MyModuleWithMixinAfter(), 'mixin_init'))
        # 关闭 MyModuleWithMixinBefore 的 call_super_init 属性
        MyModuleWithMixinBefore.call_super_init = False
        # 关闭 MyModuleWithMixinAfter 的 call_super_init 属性
        MyModuleWithMixinAfter.call_super_init = False

    # 测试函数，验证模块是否能正确共享内存
    def test_share_memory(self):
        # 定义一个简单的神经网络类 Net
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个参数张量 p 和一个参数列表 par
                self.p = nn.Parameter(torch.eye(5))
                self.par = nn.ParameterList()
                self.par.append(nn.Parameter(torch.randn(10)))

            def forward(self, inp):
                # 注意：这是一个死代码注释，因为该方法不会被调用
                # 返回输入张量的克隆
                return inp.clone()

        # 创建一个 Net 类的实例 net
        net = Net()
        # 验证所有参数张量的 storage 是否未被共享
        for p in net.parameters():
            self.assertFalse(p.storage().is_shared())
        # 验证所有缓冲区的 storage 是否未被共享
        for b in net.buffers():
            self.assertFalse(b.storage().is_shared())
        # 启用共享内存功能
        net.share_memory()
        # 验证所有参数张量的 storage 是否被正确共享
        for p in net.parameters():
            self.assertTrue(p.storage().is_shared())
        # 验证所有缓冲区的 storage 是否被正确共享
        for b in net.buffers():
            self.assertTrue(b.storage().is_shared())
    # 定义测试方法，验证 PyTorch 模型转换功能
    def test_to(self):
        # 创建一个线性层模型，输入维度为3，输出维度为5
        m = nn.Linear(3, 5)
        # 断言模型对象在转换到CPU后仍是同一个对象
        self.assertIs(m, m.to('cpu'))
        # 断言模型对象在转换到CPU和指定数据类型后仍是同一个对象
        self.assertIs(m, m.to('cpu', dtype=torch.float32))
        # 断言模型的双精度版本等价于转换到64位浮点数后的模型
        self.assertEqual(m.double(), m.to(torch.float64))
        # 使用 lambda 表达式断言在指定参数下会引发 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: m.to('cpu', copy=True))

        # 如果CUDA可用，测试模型在不同CUDA设备上的转换
        if torch.cuda.is_available():
            # 遍历可用的CUDA设备
            for cuda in ['cuda', 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1']:
                # 将模型移动到指定CUDA设备
                m2 = m.cuda(device=cuda)
                # 断言模型在转换到指定CUDA设备后仍是同一个对象
                self.assertIs(m2, m2.to(cuda))
                # 断言模型转换到CPU后与原始模型相等
                self.assertEqual(m, m2.to('cpu'))
                # 断言模型在CUDA设备上的转换后与原始模型相等
                self.assertEqual(m2, m.to(cuda))
                # 断言模型在转换到指定数据类型后仍是同一个对象
                self.assertIs(m2, m2.to(dtype=torch.float32))
                # 断言模型的双精度版本在CUDA设备上转换到64位浮点数后仍是同一个对象
                self.assertEqual(m2.double(), m2.to(dtype=torch.float64))

    # 定义测试方法，验证梯度清零功能
    def test_zero_grad(self):
        # 创建一个具有随机初始化的张量输入，需要梯度计算
        i = torch.randn(2, 5, requires_grad=True)
        # 创建一个线性层模型，输入维度为5，输出维度为5
        module = nn.Linear(5, 5)
        # 关闭所有模型参数的梯度计算
        for p in module.parameters():
            p.requires_grad = False
        # 清除模型的梯度信息
        module.zero_grad()

        # 设置模型权重参数需要梯度计算
        module.weight.requires_grad = True
        # 再次清除模型的梯度信息
        module.zero_grad()
        # 断言模型的权重参数的梯度为未初始化状态
        self.assertIsNone(module.weight.grad)  # uninitialized grad

        # 对模型进行前向传播和反向传播
        module(i).sum().backward()
        # 断言模型的权重参数的梯度不为None
        self.assertIsNotNone(module.weight.grad)
        # 断言模型的权重参数的梯度绝对值之和大于0
        self.assertGreater(module.weight.grad.data.abs().sum(), 0)
        # 再次清除模型的梯度信息
        module.zero_grad()
        # 断言模型的权重参数的梯度为None
        self.assertIsNone(module.weight.grad)

        # 设置模型偏置参数需要梯度计算
        module.bias.requires_grad = True
        # 再次清除模型的梯度信息
        module.zero_grad()
        # 断言模型的权重参数和偏置参数的梯度都为None
        self.assertIsNone(module.weight.grad)
        self.assertIsNone(module.bias.grad)
        # 对模型进行前向传播和反向传播
        module(i).sum().backward()
        # 断言模型的权重参数和偏置参数的梯度不为None
        self.assertIsNotNone(module.weight.grad)
        self.assertIsNotNone(module.bias.grad)
        # 断言模型的权重参数和偏置参数的梯度绝对值之和大于0
        self.assertGreater(module.weight.grad.data.abs().sum(), 0)
        self.assertGreater(module.bias.grad.data.abs().sum(), 0)
        # 强制清零模型的梯度信息
        module.zero_grad(set_to_none=False)   # Force set to zeros.
        # 断言模型的权重参数和偏置参数的梯度被设置为零
        self.assertEqual(module.weight.grad.data, module.weight.data.clone().zero_())
        self.assertEqual(module.bias.grad.data, module.bias.data.clone().zero_())

        # 再次清除模型的梯度信息
        module.zero_grad()
        # 断言模型的权重参数和偏置参数的梯度为None
        self.assertIsNone(module.weight.grad)
        self.assertIsNone(module.bias.grad)

    # 定义测试方法，验证不需要梯度计算时的行为
    def test_no_grad(self):
        # 遍历不同的数据类型
        for dtype in [torch.bfloat16, torch.float, torch.double]:
            # 创建一个二维卷积模型，输入通道数为2，输出通道数为5，卷积核大小为3x3
            module = nn.Conv2d(2, 5, kernel_size=3, padding=1).to(dtype)
            # 创建一个随机初始化的张量输入，并转换到指定数据类型
            input = torch.randn(1, 2, 10, 10).to(dtype)
            x = input
            y = input.clone()

            # 对输入进行前向传播
            output = module(x)
            # 断言输出需要计算梯度
            self.assertTrue(output.requires_grad)
            # 对输出进行反向传播
            output.backward(torch.ones(1, 5, 10, 10))

            # 使用 torch.no_grad 上下文管理器
            with torch.no_grad():
                # 对克隆的输入进行前向传播
                output2 = module(y)
                # 断言输出不需要计算梯度
                self.assertFalse(output2.requires_grad)
                # 断言在不允许计算梯度的情况下调用反向传播会引发 RuntimeError 异常
                self.assertRaises(RuntimeError, lambda: output2.backward(torch.ones(1, 5, 10, 10)))
    def test_parameters_and_named_parameters(self):
        # 定义一个内部函数 names，用于从命名参数中提取参数名列表
        def names(named_parameters):
            return [k for k, _ in named_parameters]

        # 调用 _create_basic_net() 创建模型组件 l, n, s
        l, n, s = _create_basic_net()

        # 断言模型 l 的参数列表长度为 1
        self.assertEqual(len(list(l.parameters())), 1)
        # 断言模型 l 的命名参数列表为 ['layer_dummy_param']
        self.assertEqual(
            names(l.named_parameters()),
            ['layer_dummy_param'])

        # 断言模型 n 的参数列表长度为 2
        self.assertEqual(len(list(n.parameters())), 2)
        # 断言模型 n 的命名参数列表为 ['dummy_param', 'l1.layer_dummy_param']
        self.assertEqual(
            names(n.named_parameters()),
            ['dummy_param', 'l1.layer_dummy_param'])

        # 断言模型 n 的参数列表长度为 1（不递归）
        self.assertEqual(len(list(n.parameters(recurse=False))), 1)
        # 断言模型 n 的命名参数列表为 ['dummy_param']（不递归）
        self.assertEqual(
            names(n.named_parameters(recurse=False)),
            ['dummy_param'])

        # 断言模型 s 的参数列表长度为 2
        self.assertEqual(len(list(s.parameters())), 2)
        # 断言模型 s 的命名参数列表为 ['0.dummy_param', '0.l1.layer_dummy_param']
        self.assertEqual(
            names(s.named_parameters()),
            ['0.dummy_param', '0.l1.layer_dummy_param'])

    def test_named_parameters_remove_duplicate(self):
        # 定义一个内部函数 names，用于从命名参数中提取参数名列表
        def names(named_parameters):
            return [k for k, _ in named_parameters]

        # 定义模型 M1，包含参数 param1 和 param2，其中 param2 与 param1 引用相同的对象
        class M1(nn.Module):
            def __init__(self):
                super().__init__()
                self.param1 = nn.Parameter(torch.empty(3, 3))
                self.param2 = self.param1

        # 创建模型 M1 的实例 m1
        m1 = M1()
        # 断言模型 m1 的命名参数列表为 ["param1"]
        self.assertEqual(names(m1.named_parameters()),
                         ["param1"])
        # 断言模型 m1 的命名参数列表（保留重复项）为 ["param1", "param2"]
        self.assertEqual(names(m1.named_parameters(remove_duplicate=False)),
                         ["param1", "param2"])

        # 定义模型 M2，包含模块 mod1 和 mod2，其中 mod2 与 mod1 引用相同的对象
        class M2(nn.Module):
            def __init__(self):
                super().__init__()
                self.mod1 = nn.Linear(3, 4, bias=False)
                self.mod2 = self.mod1

        # 创建模型 M2 的实例 m2
        m2 = M2()
        # 断言模型 m2 的命名参数列表为 ["mod1.weight"]
        self.assertEqual(names(m2.named_parameters()),
                         ["mod1.weight"])
        # 断言模型 m2 的命名参数列表（保留重复项）为 ["mod1.weight", "mod2.weight"]
        self.assertEqual(names(m2.named_parameters(remove_duplicate=False)),
                         ["mod1.weight", "mod2.weight"])
    # 测试函数：验证模型的缓冲区和命名缓冲区相关功能
    def test_buffers_and_named_buffers(self):
        # 内部辅助函数：从命名缓冲区中获取所有缓冲区的名称
        def names(named_buffers):
            return [k for k, _ in named_buffers]

        # 创建基础网络对象
        l, n, s = _create_basic_net()

        # 断言：l 的缓冲区数量为1
        self.assertEqual(len(list(l.buffers())), 1)
        # 断言：l 的命名缓冲区应包含一个名为 'layer_dummy_buf' 的缓冲区
        self.assertEqual(
            names(l.named_buffers()),
            ['layer_dummy_buf'])

        # 断言：n 的缓冲区数量为2
        self.assertEqual(len(list(n.buffers())), 2)
        # 断言：n 的命名缓冲区应包含两个缓冲区，分别为 'dummy_buf' 和 'l1.layer_dummy_buf'
        self.assertEqual(
            names(n.named_buffers()),
            ['dummy_buf', 'l1.layer_dummy_buf'])

        # 断言：在不递归查找子模块的情况下，n 的缓冲区数量为1
        self.assertEqual(len(list(n.buffers(recurse=False))), 1)
        # 断言：在不递归查找子模块的情况下，n 的命名缓冲区应包含一个名为 'dummy_buf' 的缓冲区
        self.assertEqual(
            names(n.named_buffers(recurse=False)),
            ['dummy_buf'])

        # 断言：s 的缓冲区数量为2
        self.assertEqual(len(list(s.buffers())), 2)
        # 断言：s 的命名缓冲区应包含两个缓冲区，分别为 '0.dummy_buf' 和 '0.l1.layer_dummy_buf'
        self.assertEqual(
            names(s.named_buffers()),
            ['0.dummy_buf', '0.l1.layer_dummy_buf'])

        # 测试移除重复项功能
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer1", torch.empty(3, 5))
                self.register_buffer("buffer2", self.buffer1)

        # 创建 M 类的实例 m
        m = M()
        # 断言：m 的命名缓冲区应包含一个名为 'buffer1' 的缓冲区
        self.assertEqual(names(m.named_buffers()),
                         ["buffer1"])
        # 断言：在不移除重复项的情况下，m 的命名缓冲区应包含两个缓冲区，分别为 'buffer1' 和 'buffer2'
        self.assertEqual(names(m.named_buffers(remove_duplicate=False)),
                         ["buffer1", "buffer2"])

    # 测试函数：验证模型的 forward 方法是否支持 Python 字典作为输出
    def test_call_supports_python_dict_output(self):
        # 定义一个简单的网络类 Net
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(10, 20)
                # 注册 backward hook 方法
                self.register_backward_hook(self.hook)
                # 初始化检查 backward hook 标志
                self.check_backward_hook_flag = False

            # backward hook 方法的实现
            def hook(self, module, grad_out, grad_in):
                self.check_backward_hook_flag = True

            # 前向传播方法的实现
            def forward(self, inputs):
                return {"output": self.l1(inputs).sum()}

        # 创建一个 Net 类的实例 net
        net = Net()
        # 对模型进行前向传播
        model_output = net(torch.randn([5, 10]))
        # 对输出字典中的 'output' 键进行反向传播
        model_output["output"].backward()
        # 断言：验证 backward hook 标志已被设置为 True
        self.assertTrue(net.check_backward_hook_flag)

    # 测试函数：验证模型的子模块列表生成
    def test_children(self):
        # 创建几个简单的线性层对象
        l1 = nn.Linear(2, 2)
        l2 = nn.Linear(2, 2)
        l3 = nn.Linear(2, 2)
        l4 = nn.Linear(2, 2)
        # 构建一个包含 l3 和 l4 的子模块序列 subnet
        subnet = nn.Sequential(l3, l4)
        # 构建一个包含 l1、l2、l1、l2 和 subnet 的顺序模块序列 s
        s = nn.Sequential(l1, l2, l1, l2, subnet)
        # 断言：验证 s 的子模块列表应为 [l1, l2, subnet]
        self.assertEqual(list(s.children()), [l1, l2, subnet])

    # 测试函数：验证对无效模式进行训练时是否会抛出错误
    def test_train_errors_for_invalid_mode(self):
        # 定义一个简单的子类网络 SubclassNet
        class SubclassNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(2, 2)

            # 前向传播方法的实现
            def forward(self, inputs):
                return self.l1(inputs)

        # 创建 SubclassNet 类的实例 subclass_net
        subclass_net = SubclassNet()
        # 创建一个包含两个线性层的顺序模块 sequential_net
        sequential_net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

        # 定义要检查的错误模式列表
        error_modes = ["invalid_str", torch.device('cpu')]
        # 定义要检查的模块列表
        modules_to_check = [subclass_net, sequential_net]

        # 使用 itertools.product 生成错误模式和模块的组合
        for error_mode, module in itertools.product(error_modes, modules_to_check):
            # 断言：对于每个组合，确保在训练模式时会抛出 ValueError 异常
            with self.assertRaises(ValueError):
                module.train(error_mode)
    # 定义一个测试方法，用于测试线性层对象的属性和方法
    def test_dir(self):
        # 创建一个输入维度为2，输出维度为2的线性层对象
        linear = nn.Linear(2, 2)
        # 在线性层对象上添加一个名为'_test_submodule'的子模块，也是一个线性层对象
        linear._test_submodule = nn.Linear(2, 2)
        # 在线性层对象上添加一个名为'_test_parameter'的参数，是一个2x2的Tensor
        linear._test_parameter = Parameter(torch.empty(2, 2))
        # 在线性层对象上注册一个名为'_test_buffer'的缓冲区，是一个2x2的Tensor
        linear.register_buffer('_test_buffer', torch.empty(2, 2))
        # 获取线性层对象的所有属性名列表
        keys = dir(linear)
        # 断言'_test_submodule'属性名存在于keys列表中
        self.assertIn('_test_submodule', keys)
        # 断言'_test_parameter'属性名存在于keys列表中
        self.assertIn('_test_parameter', keys)
        # 断言'_test_buffer'属性名存在于keys列表中
        self.assertIn('_test_buffer', keys)

        # 遍历线性层对象的所有属性名
        for key in keys:
            # 断言线性层对象具有名为key的属性
            self.assertTrue(hasattr(linear, key))

    # 定义一个测试方法，用于测试不同情况下序列模块的字符串表示
    def test_repr(self):
        # 创建一个空的序列模块对象
        empty_sequential = nn.Sequential()
        # 期望的空序列模块的字符串表示
        expected_repr_empty = 'Sequential()'
        # 断言空序列模块的字符串表示与期望值相等
        self.assertEqual(repr(empty_sequential), expected_repr_empty)

        # 创建一个输入维度为1，输出维度为1的线性层对象
        linear = nn.Linear(1, 1)
        # 期望的线性层对象的字符串表示
        expected_repr_linear = 'Linear(in_features=1, out_features=1, bias=True)'
        # 断言线性层对象的字符串表示与期望值相等
        self.assertEqual(repr(linear), expected_repr_linear)

        # 创建一个包含一个线性层对象的序列模块对象
        sequential = nn.Sequential(linear)
        # 期望的包含子模块的序列模块对象的字符串表示
        expected_repr_sequential = 'Sequential(\n' \
            '  (0): Linear(in_features=1, out_features=1, bias=True)\n' \
            ')'
        # 断言包含子模块的序列模块对象的字符串表示与期望值相等
        self.assertEqual(repr(sequential), expected_repr_sequential)

    # 定义一个测试方法，用于测试序列模块对象的属性是否包含数字
    def test_dir_digit(self):
        # 创建一个包含一个线性层对象的序列模块对象
        model = nn.Sequential(nn.Linear(2, 2))
        # 获取序列模块对象的所有属性名列表
        keys = dir(model)
        # 断言属性名列表中不包含'0'
        self.assertNotIn('0', keys)

    # 定义一个测试方法，用于测试序列模块对象的命名子模块
    def test_named_children(self):
        # 创建四个线性层对象
        l1 = nn.Linear(2, 2)
        l2 = nn.Linear(2, 2)
        l3 = nn.Linear(2, 2)
        l4 = nn.Linear(2, 2)
        # 创建一个包含l3和l4线性层对象的子模块序列模块对象
        subnet = nn.Sequential(l3, l4)
        # 创建一个空的序列模块对象
        s = nn.Sequential()
        # 预期此处抛出KeyError异常
        with self.assertRaises(KeyError):
            s.add_module('', l1)
        # 预期此处抛出KeyError异常
        with self.assertRaises(KeyError):
            s.add_module('name.with.dot', l1)
        # 向序列模块对象添加名为'layer1'的l1线性层对象作为命名子模块
        s.add_module('layer1', l1)
        # 向序列模块对象添加名为'layer2'的l2线性层对象作为命名子模块
        s.add_module('layer2', l2)
        # 向序列模块对象添加名为'layer3'的l1线性层对象作为命名子模块
        s.add_module('layer3', l1)
        # 向序列模块对象添加名为'layer4'的l2线性层对象作为命名子模块
        s.add_module('layer4', l2)
        # 向序列模块对象添加名为'subnet'的子模块序列模块对象作为命名子模块
        s.add_module('subnet', subnet)
        # 断言序列模块对象的命名子模块列表与预期值相等
        self.assertEqual(list(s.named_children()), [('layer1', l1), ('layer2', l2), ('subnet', subnet)])

    # 定义一个测试方法，用于测试模块对象及其子模块的模块列表
    def test_modules(self):
        # 定义一个包含一个线性层对象的类Net
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                # 在Net类的初始化方法中创建两个线性层对象
                self.l1 = l
                self.l2 = l
                self.param = torch.empty(3, 5)

        # 创建一个输入维度为10，输出维度为20的线性层对象
        l = nn.Linear(10, 20)
        # 创建Net类的实例n
        n = Net()
        # 创建一个包含四个n实例的序列模块对象
        s = nn.Sequential(n, n, n, n)
        # 断言序列模块对象的模块列表与预期值相等
        self.assertEqual(list(s.modules()), [s, n, l])
    # 定义测试方法 test_named_modules
    def test_named_modules(self):
        # 定义内部类 Net，继承自 nn.Module
        class Net(nn.Module):
            # 构造方法
            def __init__(self):
                super().__init__()
                # 创建线性层实例 l，并分配给 self.l1 和 self.l2
                self.l1 = l
                self.l2 = l
                # 创建一个形状为 (3, 5) 的空张量并分配给 self.param
                self.param = torch.empty(3, 5)
                # 创建一个空的 Sequential 容器并分配给 self.block
                self.block = block
        
        # 创建一个线性层实例 l，形状为 (10, 20)
        l = nn.Linear(10, 20)
        # 创建另外两个线性层实例 l1 和 l2，形状均为 (10, 20)
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(10, 20)
        # 创建一个空的 Sequential 容器实例并分配给 block
        block = nn.Sequential()
        # 向 block 中添加一个名为 'linear1' 的 l1 线性层
        block.add_module('linear1', l1)
        # 向 block 中添加一个名为 'linear2' 的 l2 线性层
        
        # 创建一个 Net 类的实例 n
        n = Net()
        # 创建一个包含 n 两次的 Sequential 容器实例 s
        s = nn.Sequential(n, n)
        
        # 断言测试，验证 s 中所有命名模块的列表是否符合预期
        self.assertEqual(list(s.named_modules()), [('', s), ('0', n), ('0.l1', l),
                                                   ('0.block', block), ('0.block.linear1', l1),
                                                   ('0.block.linear2', l2)])
        
        # 再次断言测试，验证不移除重复模块实例的选项下的命名模块列表是否符合预期
        self.assertEqual(list(s.named_modules(remove_duplicate=False)), [
            ('', s), ('0', n), ('0.l1', l), ('0.l2', l),
            ('0.block', block), ('0.block.linear1', l1),
            ('0.block.linear2', l2),
            ('1', n), ('1.l1', l), ('1.l2', l),
            ('1.block', block), ('1.block.linear1', l1),
            ('1.block.linear2', l2)])

    # 定义测试方法 test_register_buffer_raises_error_if_name_is_not_string
    def test_register_buffer_raises_error_if_name_is_not_string(self):
        # 创建一个空的 nn.Module 实例 m
        m = nn.Module()
        # 预期的错误信息前缀
        expected_error = 'buffer name should be a string. Got '
        
        # 使用断言检查是否捕获到 TypeError 异常，且错误信息符合预期
        with self.assertRaisesRegex(TypeError, expected_error + 'int'):
            m.register_buffer(1, torch.rand(5))
        
        # 使用断言检查是否捕获到 TypeError 异常，且错误信息符合预期
        with self.assertRaisesRegex(TypeError, expected_error + 'NoneType'):
            m.register_buffer(None, torch.rand(5))

    # 定义测试方法 test_register_buffer_raises_error_if_attr_exists
    def test_register_buffer_raises_error_if_attr_exists(self):
        # 创建一个空的 nn.Module 实例 m
        m = nn.Module()
        # 设置 m 的 attribute_name 属性为整数 5
        m.attribute_name = 5
        # 使用断言检查是否捕获到 KeyError 异常，因为同名属性已存在
        with self.assertRaises(KeyError):
            m.register_buffer('attribute_name', torch.rand(5))
        
        # 删除 m 的 attribute_name 属性
        del m.attribute_name
        # 将参数注册为 attribute_name 属性后再注册缓冲区，预期捕获 KeyError 异常
        m.register_parameter('attribute_name', nn.Parameter())
        with self.assertRaises(KeyError):
            m.register_buffer('attribute_name', torch.rand(5))
        
        # 删除 m 的 attribute_name 属性
        del m.attribute_name
        # 向 m 的模块添加一个名为 'attribute_name' 的 nn.Module 实例后再注册缓冲区，预期捕获 KeyError 异常
        m.add_module('attribute_name', nn.Module())
        with self.assertRaises(KeyError):
            m.register_buffer('attribute_name', torch.rand(5))

    # 定义测试方法 test_register_buffer_raises_error_if_not_tensor
    def test_register_buffer_raises_error_if_not_tensor(self):
        # 创建一个空的 nn.Module 实例 m
        m = nn.Module()
        # 使用断言检查是否捕获到 TypeError 异常，因为传入的不是张量而是整数 5
        with self.assertRaises(TypeError):
            m.register_buffer('attribute_name', 5)

    # 定义测试方法 test_register_buffer_allows_overwriting_with_same_name
    def test_register_buffer_allows_overwriting_with_same_name(self):
        # 创建一个空的 nn.Module 实例 m
        m = nn.Module()
        # 创建三个不同的张量缓冲区实例
        buffer1 = torch.rand(5)
        buffer2 = buffer1 + 5
        buffer3 = None
        
        # 注册 buffer1 为名为 'buffer_name' 的缓冲区
        m.register_buffer('buffer_name', buffer1)
        # 使用断言检查是否成功注册 buffer1
        self.assertEqual(m.buffer_name, buffer1)
        
        # 再次注册 buffer2 为名为 'buffer_name' 的缓冲区，覆盖原有缓冲区
        m.register_buffer('buffer_name', buffer2)
        # 使用断言检查是否成功覆盖为 buffer2
        self.assertEqual(m.buffer_name, buffer2)
        
        # 注册 buffer3 为名为 'buffer_name' 的缓冲区，此时为 None
        m.register_buffer('buffer_name', buffer3)
        # 使用断言检查是否成功注册 buffer3
        self.assertEqual(m.buffer_name, buffer3)
    # 定义测试函数，用于测试获取缓冲区的功能
    def test_get_buffer(self):
        # 创建一个空的神经网络模块
        m = nn.Module()
        # 创建两个随机张量作为缓冲区
        buffer1 = torch.randn(2, 3)
        buffer2 = torch.randn(4, 5)
        # 将第一个缓冲区注册到模块中，并命名为'foo'
        m.register_buffer('foo', buffer1)
        # 将第二个缓冲区注册到模块中，并命名为'bar'
        m.register_buffer('bar', buffer2)
        # 断言获取到的'foo'缓冲区与buffer1相等
        self.assertEqual(buffer1, m.get_buffer('foo'))
        # 断言获取到的'bar'缓冲区与buffer2相等
        self.assertEqual(buffer2, m.get_buffer('bar'))

    # 定义测试函数，用于测试从子模块获取缓冲区的功能
    def test_get_buffer_from_submodules(self):
        # 定义一个自定义的神经网络模块MyModule，接收两个参数foo和bar
        class MyModule(nn.Module):
            def __init__(self, foo, bar):
                super().__init__()
                # 创建一个名为'sub'的子模块，并将foo和bar传递给它
                self.sub = Sub(foo, bar)

        # 定义一个名为Sub的子模块，接收foo和bar作为参数
        class Sub(nn.Module):
            def __init__(self, foo, bar):
                super().__init__()
                # 将foo作为缓冲区注册到当前子模块中
                self.register_buffer('foo', foo)
                # 创建一个名为'subsub'的子子模块，并将bar传递给它
                self.subsub = SubSub(bar)

        # 定义一个名为SubSub的子子模块，接收bar作为参数
        class SubSub(nn.Module):
            def __init__(self, bar):
                super().__init__()
                # 将bar作为缓冲区注册到当前子子模块中
                self.register_buffer('bar', bar)

        # 创建两个随机张量作为foo和bar
        foo = torch.randn(2, 3)
        bar = torch.randn(4, 5)
        # 创建一个MyModule实例，传入foo和bar作为参数
        m = MyModule(foo, bar)
        # 断言从子模块'sub'获取的foo缓冲区与foo相等
        self.assertEqual(foo, m.get_buffer('sub.foo'))
        # 断言从子模块'sub.subsub'获取的bar缓冲区与bar相等
        self.assertEqual(bar, m.get_buffer('sub.subsub.bar'))

    # 定义测试函数，用于测试非持久性缓冲区
    def test_buffer_not_persistent(self):
        # 创建一个空的神经网络模块
        m = nn.Module()
        # 注册一个非持久性的随机张量作为缓冲区，并命名为'buf'
        m.register_buffer('buf', torch.rand(5), persistent=False)
        # 断言模块中的缓冲区数量为1
        self.assertTrue(len(list(m.buffers())) == 1)
        # 断言模块的状态字典为空
        self.assertTrue(len(m.state_dict()) == 0)

    # 定义测试函数，用于测试删除非持久性缓冲区
    def test_buffer_not_persistent_del(self):
        # 创建一个空的神经网络模块
        m = nn.Module()
        # 注册一个非持久性的随机张量作为缓冲区，并命名为'buf'
        m.register_buffer('buf', torch.rand(5), persistent=False)
        # 删除名为'buf'的缓冲区
        del m.buf
        # 断言模块中的缓冲区数量为0
        self.assertTrue(len(list(m.buffers())) == 0)

    # 定义测试函数，用于测试覆盖非持久性缓冲区
    def test_buffer_not_persistent_overwrite(self):
        # 创建一个空的神经网络模块
        m = nn.Module()
        # 注册一个非持久性的随机张量作为缓冲区，并命名为'buf'
        m.register_buffer('buf', torch.rand(5), persistent=False)
        # 再次注册一个名为'buf'的持久性随机张量作为缓冲区
        m.register_buffer('buf', torch.rand(5))
        # 断言模块中的缓冲区数量为1
        self.assertTrue(len(list(m.buffers())) == 1)
        # 断言模块的状态字典中的条目数量为1
        self.assertTrue(len(m.state_dict()) == 1)
        
        # 再次注册一个非持久性的随机张量作为名为'buf'的缓冲区
        m.register_buffer('buf', torch.rand(5), persistent=False)
        # 断言模块中的缓冲区数量为1
        self.assertTrue(len(list(m.buffers())) == 1)
        # 断言模块的状态字典为空
        self.assertTrue(len(m.state_dict()) == 0)

    # 定义测试函数，用于测试分配非持久性缓冲区
    def test_buffer_not_persistent_assign(self):
        # 创建一个空的神经网络模块
        m = nn.Module()
        # 注册一个非持久性的随机张量作为缓冲区，并命名为'buf'
        m.register_buffer('buf', torch.rand(5), persistent=False)

        # 将'buf'属性分配为None，移除缓冲区，但如果再分配一个Tensor到同一属性，它应该仍然被标记为缓冲区
        m.buf = None
        # 断言模块中的缓冲区数量为0
        self.assertTrue(len(list(m.buffers())) == 0)
        # 断言模块的状态字典为空
        self.assertTrue(len(m.state_dict()) == 0)
        # 再次将一个随机张量分配给'buf'属性
        m.buf = torch.rand(5)
        # 断言模块中的缓冲区数量为1
        self.assertTrue(len(list(m.buffers())) == 1)
        # 断言模块的状态字典为空
        self.assertTrue(len(m.state_dict()) == 0)

        # 将'buf'属性分配为一个参数，移除缓冲区
        m.buf = nn.Parameter(torch.rand(5))
        # 断言模块中的缓冲区数量为0
        self.assertTrue(len(list(m.buffers())) == 0)
        # 断言模块的状态字典中的条目数量为1
        self.assertTrue(len(m.state_dict()) == 1)
    # 定义一个单元测试方法，测试注册的缓冲区不是持久化加载
    def test_buffer_not_persistent_load(self):
        # 创建一个空的神经网络模块
        m = nn.Module()
        # 注册一个非持久化的缓冲区 'buf'，其值为随机生成的5维张量
        m.register_buffer('buf', torch.rand(5), persistent=False)
        # 加载一个空的状态字典，用于模拟加载状态
        m.load_state_dict({})

    # 定义一个单元测试方法，测试如果参数名不是字符串会引发错误
    def test_register_parameter_raises_error_if_name_is_not_string(self):
        # 创建一个空的神经网络模块
        m = nn.Module()
        # 预期的错误消息前缀
        expected_error = 'parameter name should be a string. Got '
        # 测试注册整数参数名引发 TypeError 错误
        with self.assertRaisesRegex(TypeError, expected_error + 'int'):
            m.register_parameter(1, nn.Parameter())
        # 测试注册 None 类型参数名引发 TypeError 错误
        with self.assertRaisesRegex(TypeError, expected_error + 'NoneType'):
            m.register_parameter(None, nn.Parameter())

    # 定义一个单元测试方法，测试如果属性名称已存在会引发错误
    def test_register_parameter_raises_error_if_attr_exists(self):
        # 创建一个空的神经网络模块
        m = nn.Module()
        # 设置一个名为 'attribute_name' 的属性
        m.attribute_name = 5
        # 测试注册已经存在的属性名称引发 KeyError 错误
        with self.assertRaises(KeyError):
            m.register_parameter('attribute_name', nn.Parameter())

        # 删除 'attribute_name' 属性
        del m.attribute_name
        # 注册一个名为 'attribute_name' 的缓冲区
        m.register_buffer('attribute_name', torch.rand(5))
        # 测试注册已经存在的缓冲区名称引发 KeyError 错误
        with self.assertRaises(KeyError):
            m.register_parameter('attribute_name', nn.Parameter())

        # 删除 'attribute_name' 缓冲区
        del m.attribute_name
        # 添加一个名为 'attribute_name' 的子模块
        m.add_module('attribute_name', nn.Module())
        # 测试注册已经存在的模块名称引发 KeyError 错误
        with self.assertRaises(KeyError):
            m.register_parameter('attribute_name', nn.Parameter())

    # 定义一个单元测试方法，测试允许用相同名称覆盖注册参数
    def test_register_parameter_allows_overwriting_with_same_name(self):
        # 创建一个空的神经网络模块
        m = nn.Module()
        # 创建参数 param1 和 param2
        param1 = nn.Parameter(torch.rand(5))
        param2 = nn.Parameter(param1.data + 5)
        param3 = None
        # 注册参数 'param_name' 为 param1
        m.register_parameter('param_name', param1)
        # 断言 'param_name' 等于 param1
        self.assertEqual(m.param_name, param1)
        # 用 param2 覆盖 'param_name'
        m.register_parameter('param_name', param2)
        # 断言 'param_name' 等于 param2
        self.assertEqual(m.param_name, param2)
        # 用 param3 覆盖 'param_name'
        m.register_parameter('param_name', param3)
        # 断言 'param_name' 等于 param3
        self.assertEqual(m.param_name, param3)

    # 定义一个单元测试方法，测试如果属性名称已存在会引发错误
    def test_add_module_raises_error_if_attr_exists(self):
        # 需要测试的方法列表
        methods_to_test = ['add_module', 'register_module']
        for fn in methods_to_test:
            # 创建一个空的神经网络模块
            m = nn.Module()
            # 设置一个名为 'attribute_name' 的属性
            m.attribute_name = 5
            # 测试添加已经存在的属性名称引发 KeyError 错误
            with self.assertRaises(KeyError):
                getattr(m, fn)('attribute_name', nn.Module())

            # 删除 'attribute_name' 属性
            del m.attribute_name
            # 注册一个名为 'attribute_name' 的缓冲区
            m.register_buffer('attribute_name', torch.rand(5))
            # 测试添加已经存在的缓冲区名称引发 KeyError 错误
            with self.assertRaises(KeyError):
                getattr(m, fn)('attribute_name', nn.Module())

            # 删除 'attribute_name' 缓冲区
            del m.attribute_name
            # 注册一个名为 'attribute_name' 的参数
            m.register_parameter('attribute_name', nn.Parameter())
            # 测试添加已经存在的参数名称引发 KeyError 错误
            with self.assertRaises(KeyError):
                getattr(m, fn)('attribute_name', nn.Module())

    # 定义一个单元测试方法，测试带有属性的 getattr 方法
    @unittest.expectedFailure
    def test_getattr_with_property(self):
        # 定义一个带有属性的神经网络模块
        class Model(nn.Module):
            @property
            def some_property(self):
                return self.something_that_doesnt_exist

        # 创建一个 Model 实例
        model = Model()
        # 测试访问不存在的属性引发 AttributeError 错误
        with self.assertRaisesRegex(
                AttributeError,
                r"'Model' object has no attribute 'something_that_doesnt_exist'"):
            model.some_property
    # 测试 Sequential 对象的索引功能
    def test_Sequential_getitem(self):
        # 创建四个线性层对象
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        # 创建 Sequential 对象 n，包含上述四个线性层
        n = nn.Sequential(l1, l2, l3, l4)
        # 检查索引为 0 的元素是否为 l1
        self.assertIs(n[0], l1)
        # 检查索引为 1 的元素是否为 l2
        self.assertIs(n[1], l2)
        # 检查索引为 2 的元素是否为 l3
        self.assertIs(n[2], l3)
        # 检查索引为 3 的元素是否为 l4
        self.assertIs(n[3], l4)
        # 使用整数张量作为索引，检查索引为 3 的元素是否为 l4
        self.assertIs(n[torch.tensor(3, dtype=torch.int64)], l4)
        # 检查切片索引 [1:] 是否返回正确的 Sequential 对象
        self.assertEqual(n[1:], nn.Sequential(l2, l3, l4))
        # 检查切片索引 [3:] 是否返回正确的 Sequential 对象
        self.assertEqual(n[3:], nn.Sequential(l4))
        # 检查切片索引 [:-1] 是否返回正确的 Sequential 对象
        self.assertEqual(n[:-1], nn.Sequential(l1, l2, l3))
        # 检查切片索引 [:-3] 是否返回正确的 Sequential 对象
        self.assertEqual(n[:-3], nn.Sequential(l1))
        # 检查逆序索引 [::-1] 是否返回正确的 Sequential 对象
        self.assertEqual(n[::-1], nn.Sequential(l4, l3, l2, l1))

    # 测试 Sequential 对象的赋值功能
    def test_Sequential_setitem(self):
        # 创建四个线性层对象
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        # 创建 Sequential 对象 n，包含前三个线性层
        n = nn.Sequential(l1, l2, l3)
        # 修改索引为 0 的元素为 l4
        n[0] = l4
        # 修改索引为 -1 的元素为 l4
        n[-1] = l4
        # 使用整数张量作为索引，修改索引为 1 的元素为 l1
        n[torch.tensor(1, dtype=torch.int16)] = l1
        # 检查索引为 0 的元素是否为 l4
        self.assertIs(n[0], l4)
        # 检查索引为 1 的元素是否为 l1
        self.assertIs(n[1], l1)
        # 检查索引为 2 的元素是否为 l4
        self.assertIs(n[2], l4)

    # 测试 Sequential 对象的命名赋值功能
    def test_Sequential_setitem_named(self):
        # 创建四个线性层对象
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        # 创建带命名的 Sequential 对象 n
        n = nn.Sequential(OrderedDict([
            ('linear1', l1),
            ('linear2', l2),
            ('linear3', l3),
        ]))
        # 修改索引为 0 的元素为 l4
        n[0] = l4
        # 修改索引为 -1 的元素为 l4
        n[-1] = l4
        # 检查 linear1 属性是否等于 l4
        self.assertEqual(n.linear1, l4)
        # 检查 linear3 属性是否等于 l4
        self.assertEqual(n.linear3, l4)

    # 测试 Sequential 对象的删除功能
    def test_Sequential_delitem(self):
        # 创建四个线性层对象
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        # 创建 Sequential 对象 n，包含上述四个线性层
        n = nn.Sequential(l1, l2, l3, l4)
        # 删除索引为 -1 的元素
        del n[-1]
        # 检查是否正确删除了最后一个元素
        self.assertEqual(n, nn.Sequential(l1, l2, l3))
        # 删除切片索引 [1::2] 的元素
        del n[1::2]
        # 检查是否正确删除了索引为 1 和 3 的元素
        self.assertEqual(n, nn.Sequential(l1, l3))

    # 测试 Sequential 对象的加法功能
    def test_Sequential_add(self):
        # 创建四个线性层对象
        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 4)
        l4 = nn.Linear(4, 5)
        # 创建两个 Sequential 对象 n 和 other
        n = nn.Sequential(l1, l2)
        other = nn.Sequential(l3, l4)
        # 检查两个 Sequential 对象相加的结果是否正确
        self.assertEqual(n + other, nn.Sequential(l1, l2, l3, l4))

    # 测试 Sequential 对象的增强加法功能
    def test_Sequential_iadd(self):
        # 创建四个线性层对象
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        # 创建 Sequential 对象 n 和 n2
        n = nn.Sequential(l1, l2, l3)
        n2 = nn.Sequential(l4)
        # 将 n2 增强加到 n 中
        n += n2
        # 将 n 增强加到 n2 中
        n2 += n
        # 检查增强加法后 n 的内容是否正确
        self.assertEqual(n, nn.Sequential(l1, l2, l3, l4))
        # 检查增强加法后 n2 的内容是否正确
        self.assertEqual(n2, nn.Sequential(l4, l1, l2, l3, l4))

    # 测试 Sequential 对象的乘法功能
    def test_Sequential_mul(self):
        # 创建四个线性层对象
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        # 创建 Sequential 对象 n
        n = nn.Sequential(l1, l2, l3, l4)
        # 将 n 乘以 2
        n2 = n * 2
        # 检查乘法后 n2 的内容是否正确
        self.assertEqual(n2, nn.Sequential(l1, l2, l3, l4, l1, l2, l3, l4))
    def test_Sequential_rmul(self):
        # 创建四个线性层对象，分别将输入维度10映射到20，20映射到30，30映射到40，40映射到50
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        # 创建一个包含上述四个线性层的序列模型
        n = nn.Sequential(l1, l2, l3, l4)
        # 使用右乘操作符将该序列模型复制一次
        n2 = 2 * n
        # 断言两个序列模型相等，即n2包含两个n的拷贝
        self.assertEqual(n2, nn.Sequential(l1, l2, l3, l4, l1, l2, l3, l4))

    def test_Sequential_imul(self):
        # 创建四个线性层对象，分别将输入维度10映射到20，20映射到30，30映射到40，40映射到50
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        # 创建一个包含上述四个线性层的序列模型
        n = nn.Sequential(l1, l2, l3, l4)
        # 使用扩展赋值操作符乘以2，即将序列模型中的层重复一次
        n *= 2
        # 断言序列模型n等于包含四个重复层的新序列模型
        self.assertEqual(n, nn.Sequential(l1, l2, l3, l4, l1, l2, l3, l4))
        # 再次使用扩展赋值操作符乘以2，即将序列模型中的层再次重复一次
        n *= 2
        # 断言序列模型n等于包含八个重复层的新序列模型
        self.assertEqual(
            n,
            nn.Sequential(l1, l2, l3, l4, l1, l2, l3, l4, l1, l2, l3, l4, l1, l2, l3, l4)
        )

    def test_Sequential_append(self):
        # 创建四个线性层对象，分别将输入维度10映射到20，20映射到30，30映射到40，40映射到50
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        # 创建一个包含前三个线性层的序列模型
        n = nn.Sequential(l1, l2, l3)
        # 在序列模型末尾追加第四个线性层，修改原序列模型n，并赋值给n2
        n2 = n.append(l4)
        # 断言n等于包含四个线性层的新序列模型
        self.assertEqual(n, nn.Sequential(l1, l2, l3, l4))
        # 断言n2等于包含四个线性层的新序列模型
        self.assertEqual(n2, nn.Sequential(l1, l2, l3, l4))
        # 断言链式操作后的序列模型等于包含三个线性层和一个第二个线性层的新序列模型
        self.assertEqual(nn.Sequential(l1).append(l2).append(l4), nn.Sequential(l1, l2, l4))

    def test_Sequential_pop(self):
        # 创建四个线性层对象，分别将输入维度1映射到2，2映射到3，3映射到4，4映射到5
        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 4)
        l4 = nn.Linear(4, 5)
        # 创建一个包含所有四个线性层的序列模型n1
        n1 = nn.Sequential(l1, l2, l3, l4)
        # 断言从序列模型n1中弹出最后一个线性层l4，并返回l4
        self.assertEqual(l4, n1.pop(3))
        # 创建一个包含前三个线性层的序列模型n2
        n2 = nn.Sequential(l1, l2, l3)
        # 断言序列模型n1等于序列模型n2，即n1和n2包含相同的线性层
        self.assertEqual(n1, n2)
        # 检查索引的顺序是否与模块的顺序一致
        for k, mod in zip(range(len(n1)), n1):
            # 断言序列模型n1中的第k个模块与mod相同
            self.assertIs(n1[k], mod)

    def test_Sequential_insert(self):
        # 创建三个线性层对象，分别将输入维度1映射到2，2映射到3，3映射到4
        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 4)

        # 创建一个包含前三个线性层的序列模型n1
        n1 = nn.Sequential(l1, l2, l3)
        # 创建一个新的线性层对象，将输入维度4映射到5
        module_1 = nn.Linear(4, 5)
        # 创建一个包含插入module_1后的四个线性层的序列模型n2
        n2 = nn.Sequential(l1, module_1, l2, l3)
        # 断言在n1中索引1处插入module_1后，n1等于n2
        self.assertEqual(n1.insert(1, module_1), n2)

        # 测试负索引支持的情况
        # 创建一个包含前三个线性层的序列模型n3
        n3 = nn.Sequential(l1, l2, l3)
        # 创建一个新的线性层对象，将输入维度5映射到6
        module_2 = nn.Linear(5, 6)
        # 创建一个包含在n3中倒数第二个位置插入module_2后的四个线性层的序列模型n4
        n4 = nn.Sequential(l1, module_2, l2, l3)
        # 断言在n3中负索引-2处插入module_2后，n3等于n4

        self.assertEqual(n3.insert(-2, module_2), n4)

    def test_Sequential_insert_fail_case(self):
        # 创建三个线性层对象，分别将输入维度1映射到2，2映射到3，3映射到4
        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 4)

        # 创建一个新的线性层对象，将输入维度5映射到6
        module = nn.Linear(5, 6)

        # 创建一个包含前三个线性层的序列模型n1
        n1 = nn.Sequential(l1, l2, l3)
        # 测试错误情况
        # 断言在n1中插入module时引发IndexError
        with self.assertRaises(IndexError):
            n1.insert(-5, module)

        # 断言在n1中插入包含线性层[nn.Linear(6, 7)]时引发AssertionError
        with self.assertRaises(AssertionError):
            n1.insert(1, [nn.Linear(6, 7)])

    def test_Sequential_extend(self):
        # 创建四个线性层对象，分别将输入维度10映射到20，20映射到30，30映射到40，40映射到50
    # 定义一个测试方法，用于测试 nn.ModuleDict 的各种功能
    def test_ModuleDict(self):
        # 创建一个有序字典，包含不同的 nn.Module 对象
        modules = OrderedDict([
            ('act', nn.ReLU()),
            ('conv', nn.Conv2d(10, 10, 5)),
            ('fc', nn.Linear(5, 5)),
        ])

        # 使用 OrderedDict 创建 nn.ModuleDict 对象
        module_dict = nn.ModuleDict(modules)

        # 定义一个内部函数 check()，用于检查 module_dict 的状态
        def check():
            # 断言 ModuleDict 的长度与 modules 字典的长度相等
            self.assertEqual(len(module_dict), len(modules))
            # 使用 zip 遍历 modules 字典和 module_dict 的 children()，验证每个模块是否一致
            for k1, m2 in zip(modules, module_dict.children()):
                self.assertIs(modules[k1], m2)
            # 使用 zip 遍历 modules 字典和 module_dict，验证每个键对应的值是否一致
            for k1, k2 in zip(modules, module_dict):
                self.assertIs(modules[k1], module_dict[k2])
            # 遍历 module_dict，验证每个键对应的值与 modules 字典中的是否一致
            for k in module_dict:
                self.assertIs(module_dict[k], modules[k])
            # 遍历 module_dict.keys()，验证每个键对应的值与 modules 字典中的是否一致
            for k in module_dict.keys():
                self.assertIs(module_dict[k], modules[k])
            # 遍历 module_dict.items()，验证每个键值对是否与 modules 字典中的一致
            for k, v in module_dict.items():
                self.assertIs(modules[k], v)
            # 使用 zip 遍历 modules 字典和 module_dict 的 values()，验证每个值是否一致
            for k1, m2 in zip(modules, module_dict.values()):
                self.assertIs(modules[k1], m2)
            # 遍历 modules 字典的键，验证每个键是否存在于 module_dict 中
            for k in modules.keys():
                self.assertTrue(k in module_dict)

        # 调用 check() 函数，检查初始状态下的 module_dict
        check()

        # 修改 modules 字典中 'conv' 对应的模块，并同步更新 module_dict
        modules['conv'] = nn.Conv2d(3, 4, 3)
        module_dict['conv'] = modules['conv']
        # 再次调用 check() 函数，验证更新后的 module_dict
        check()

        # 添加新的模块到 modules 字典，并同步更新 module_dict
        next_modules = [
            ('fc2', nn.Linear(5, 5)),
            ('act', nn.Sigmoid()),
        ]
        modules.update(next_modules)
        module_dict.update(next_modules)
        # 再次调用 check() 函数，验证更新后的 module_dict
        check()

        # 使用 OrderedDict 添加新的模块到 modules 字典，并同步更新 module_dict
        next_modules = OrderedDict([
            ('fc3', nn.Linear(5, 5)),
            ('act2', nn.Sigmoid()),
        ])
        modules.update(next_modules)
        module_dict.update(next_modules)
        # 再次调用 check() 函数，验证更新后的 module_dict
        check()

        # 使用字典添加新的模块到 modules 字典，并同步更新 module_dict
        next_modules = {
            'fc4': nn.Linear(5, 5),
            'act3': nn.Sigmoid()
        }
        modules.update(next_modules.items())
        module_dict.update(next_modules)
        # 再次调用 check() 函数，验证更新后的 module_dict
        check()

        # 使用 nn.ModuleDict 添加新的模块到 modules 字典，并同步更新 module_dict
        next_modules = nn.ModuleDict([
            ('fc5', nn.Linear(5, 5)),
            ('act4', nn.Sigmoid()),
        ])
        modules.update(next_modules)
        module_dict.update(next_modules)
        # 再次调用 check() 函数，验证更新后的 module_dict
        check()

        # 从 module_dict 和 modules 中删除 'fc' 模块，并调用 check() 函数验证状态
        del module_dict['fc']
        del modules['fc']
        check()

        # 使用 assertRaises 检查当试图传入错误类型数据时是否抛出 TypeError 异常
        with self.assertRaises(TypeError):
            module_dict.update(nn.ReLU())

        with self.assertRaises(TypeError):
            module_dict.update([nn.ReLU()])

        with self.assertRaises(ValueError):
            module_dict.update([[nn.ReLU()]])

        # 使用 assertRaises 检查当试图使用非键值对形式更新时是否抛出 TypeError 异常
        with self.assertRaises(TypeError):
            module_dict[1] = nn.ReLU()

        # 创建 nn.Sequential 对象并使用 named_children() 初始化 module_dict
        s = nn.Sequential(modules)
        module_dict = nn.ModuleDict(s.named_children())
        # 调用 check() 函数，验证初始化后的 module_dict
        check()

        # 从 module_dict 和 modules 中弹出 'conv' 模块，并调用 check() 函数验证状态
        c = module_dict.pop('conv')
        self.assertIs(c, modules['conv'])
        modules.pop('conv')
        check()

        # 清空 module_dict 和 modules，并使用 assertEqual 验证长度是否为 0
        module_dict.clear()
        self.assertEqual(len(module_dict), 0)
        modules.clear()
        check()

        # 使用 assertRaises 检查当试图调用未实现的方法时是否抛出 NotImplementedError 异常
        # 验证尝试通过 ModuleDict 对象调用时是否正确抛出异常
        self.assertRaises(NotImplementedError, module_dict)
        self.assertRaises(NotImplementedError, module_dict, torch.rand(1, 3))
    # 使用 @skipIfTorchDynamo() 装饰器标记这个测试函数，表示在满足特定条件（可能是与 Torch Dynamo 相关的条件）下跳过执行
    @skipIfTorchDynamo()
    # 定义一个名为 test_ParameterList_meta 的测试函数，测试 torch.nn.Parameter 类的 meta 数据处理
    def test_ParameterList_meta(self):
        # 创建一个 torch.nn.Parameter 对象 p，使用空的 tensor 初始化，指定设备为 'meta'
        p = torch.nn.Parameter(torch.empty(1, device='meta'))
        # 使用 self.assertExpectedInline 方法断言 p 的字符串表示是否符合预期输出
        self.assertExpectedInline(str(p), """\
    def test_ParameterList_replication(self):
        # The actual replication code from DP cannot be used on CPU so doing it manually here
        # 定义一个函数，用于生成包含随机张量的 Parameter 对象
        def make_param():
            return Parameter(torch.randn(2, 2))
        # 创建包含两个 Parameter 对象的列表
        parameters = [make_param(), make_param()]
        # 创建 ParameterList 对象，用于管理 Parameter 对象的列表
        param_list = nn.ParameterList(parameters)

        # 在数据并行时，复制 ParameterList 对象
        new_param_list = param_list._replicate_for_data_parallel()

        # 遍历原始 ParameterList 中的参数
        for n, p in param_list.named_parameters():
            # 使用 view_as 方法创建视图，以便稍后检查其基本对象
            setattr(new_param_list, n, p.view_as(p))

        # 检查新旧 ParameterList 中的参数是否相等
        for p, p2 in zip(param_list, new_param_list):
            self.assertEqual(p, p2)
            self.assertIsNotNone(p2.grad_fn)
            # 检查新参数的基本对象是否与原参数相同
            self.assertIs(p2._base, p)

    def test_ParameterDict_replication(self):
        # The actual replication code from DP cannot be used on CPU so doing it manually here
        # 定义一个函数，用于生成包含随机张量的 Parameter 对象
        def make_param():
            return Parameter(torch.randn(2, 2))
        # 创建包含两个 Parameter 对象的字典
        parameters = {"foo": make_param(), "bar": make_param()}
        # 创建 ParameterDict 对象，用于管理 Parameter 对象的字典
        param_dict = nn.ParameterDict(parameters)

        # 在数据并行时，复制 ParameterDict 对象
        new_param_dict = param_dict._replicate_for_data_parallel()

        # 遍历原始 ParameterDict 中的参数
        for n, p in param_dict.named_parameters():
            # 使用 view_as 方法创建视图，以便稍后检查其基本对象
            setattr(new_param_dict, n, p.view_as(p))

        # 检查新旧 ParameterDict 中的参数是否相等
        for (k, p), (k2, p2) in zip(param_dict.items(), new_param_dict.items()):
            self.assertEqual(k, k2)
            self.assertEqual(p, p2)
            self.assertIsNotNone(p2.grad_fn)
            # 检查新参数的基本对象是否与原参数相同
            self.assertIs(p2._base, p)

        # 检查 ParameterDict 中指定键的参数是否相等
        self.assertEqual(param_dict["foo"], new_param_dict["foo"])

    def test_add_module(self):
        methods_to_test = ['add_module', 'register_module']
        for fn in methods_to_test:
            # 创建一个线性层对象
            l = nn.Linear(10, 20)
            # 创建一个空的神经网络模型
            net = nn.Module()
            # 将线性层对象作为模型的属性添加两次
            net.l = l
            net.l2 = l
            # 使用指定的方法向模型添加一个空的模块
            getattr(net, fn)('empty', None)
            # 检查模型的属性是否正确
            self.assertEqual(net.l, l)
            self.assertEqual(net.l2, l)
            self.assertEqual(net.empty, None)
            # 使用指定的方法向模型添加一个新的线性层模块
            getattr(net, fn)('l3', l)
            self.assertEqual(net.l3, l)
            # 创建一个新的线性层对象
            l3 = nn.Linear(20, 10)
            # 使用指定的方法向模型添加一个新的线性层模块
            getattr(net, fn)('l', l3)
            # 检查模型的属性是否正确
            self.assertEqual(net.l, l3)
            # 检查传入非模块对象时是否引发了 TypeError 异常
            self.assertRaises(TypeError, lambda: getattr(net, fn)('x', 'non-module'))
            # 检查传入整数作为模块名时是否引发了 TypeError 异常，并检查异常消息
            self.assertRaisesRegex(TypeError, 'module name should be a string. Got int',
                                   lambda: getattr(net, fn)(1, l))
            # 检查传入 None 作为模块名时是否引发了 TypeError 异常，并检查异常消息
            self.assertRaisesRegex(TypeError, 'module name should be a string. Got NoneType',
                                   lambda: getattr(net, fn)(None, l))
    # 测试设置子模块的方法
    def test_set_submodule(self):
        # 创建一个空的神经网络模块
        net = nn.Module()
        # 在模块中创建一个子模块 t
        net.t = nn.Module()
        # 创建一个线性层，输入维度为 1，输出维度为 2
        l = nn.Linear(1, 2)
        # 指定目标子模块路径
        target = "t.l"
        # 将线性层 l 设置为 net 的子模块
        net.set_submodule(target, l)
        # 断言获取指定路径下的子模块是否为 l
        self.assertEqual(net.get_submodule(target), l)
        # 创建另一个线性层，输入维度为 2，输出维度为 1
        l2 = nn.Linear(2, 1)
        # 将新的线性层 l2 设置为 net 的子模块
        net.set_submodule(target, l2)
        # 再次断言获取指定路径下的子模块是否为 l2
        self.assertEqual(net.get_submodule(target), l2)
        # 测试空字符串作为路径设置子模块，预期引发 ValueError 异常
        self.assertRaises(ValueError, net.set_submodule, "", l)
        # 测试无效路径 "a.l" 设置子模块，预期引发 AttributeError 异常
        self.assertRaises(AttributeError, net.set_submodule, "a.l", l)
        # 测试深层级路径 "t.l.l2" 设置子模块，预期引发 AttributeError 异常
        self.assertRaises(AttributeError, net.set_submodule, "t.l.l2", l2)

    # 测试将模块转换为 argparse 参数的方法
    def test_module_to_argparse(self):
        # 创建一个包含一个线性层的序列模块
        net = nn.Sequential(nn.Linear(3, 3))
        # 创建一个表示 CPU 设备的对象
        cpu = torch.device('cpu')
        # 测试将模块转移到 CPU 设备，同时传入了不支持的参数，预期引发 TypeError 异常
        with self.assertRaises(TypeError):
            net.to(cpu, True)
        # 测试将模块转移到 torch.long 数据类型，预期引发 TypeError 异常
        with self.assertRaises(TypeError):
            net.to(torch.long)
        # 测试将模块转移到 None 设备，预期引发 TypeError 异常
        with self.assertRaises(TypeError):
            net.to(None, True)
        # 测试将模块转移到 CPU 设备，同时传入了多余的参数 torch.long，预期引发 TypeError 异常
        with self.assertRaises(TypeError):
            net.to(cpu, torch.long, True)
        # 测试将模块转移到 CPU 设备，同时传入了不支持的参数 dtype=torch.long 和 non_blocking=True，预期引发 TypeError 异常
        with self.assertRaises(TypeError):
            net.to(cpu, dtype=torch.long, non_blocking=True)
        # 测试将模块转移到空列表，预期引发 TypeError 异常
        with self.assertRaises(TypeError):
            net.to([])
        # 测试将模块转移到空字典，预期引发 TypeError 异常
        with self.assertRaises(TypeError):
            net.to({}, non_blocking=True)
        # 测试将模块转移到 torch.tensor(3, dtype=torch.long)，预期引发 TypeError 异常
        with self.assertRaises(TypeError):
            net.to(torch.tensor(3, dtype=torch.long), non_blocking=True)
        # 测试将模块转移到 CPU 设备，同时传入了不支持的参数 torch.tensor(3, dtype=torch.long) 和 non_blocking=True，预期引发 TypeError 异常
        with self.assertRaises(TypeError):
            net.to(cpu, torch.tensor(3, dtype=torch.long), non_blocking=True)

    # 测试 RNN 的非线性函数设置
    def test_RNN_nonlinearity(self):
        # 创建一个 RNN 模型，输入维度为 1，隐藏状态维度为 10，默认使用 tanh 作为非线性函数
        rnn = torch.nn.RNN(1, 10)
        # 断言 RNN 模型的非线性函数为 'tanh'
        self.assertEqual(rnn.nonlinearity, 'tanh')

        # 创建一个 RNN 模型，输入维度为 1，隐藏状态维度为 10，指定使用 relu 作为非线性函数
        rnn = torch.nn.RNN(1, 10, nonlinearity='relu')
        # 断言 RNN 模型的非线性函数为 'relu'
        self.assertEqual(rnn.nonlinearity, 'relu')

        # 测试创建 RNN 模型时指定未知的非线性函数 'garbage'，预期引发 ValueError 异常
        with self.assertRaisesRegex(ValueError, 'Unknown nonlinearity'):
            rnn = torch.nn.RNN(1, 10, nonlinearity='garbage')

    # 测试通过参数传递设置 RNN 的非线性函数
    def test_RNN_nonlinearity_passed_as_arg(self):
        # 创建一个 RNN 模型，输入维度为 2，隐藏状态维度为 3，层数为 1，指定使用 relu 作为非线性函数
        rnn = torch.nn.RNN(2, 3, 1, 'relu')
        # 断言 RNN 模型的非线性函数为 'relu'
        self.assertEqual(rnn.nonlinearity, 'relu')
    # 定义一个测试方法，用于验证在模块上应用原地操作的情况
    def test_module_apply_inplace_op(self):
        
        # 定义一个原地加一操作的函数
        def add_one_inplace(t):
            return t.add_(1.0)
        
        # 创建一个线性层模块，输入维度为20，输出维度为10
        m = nn.Linear(20, 10)
        
        # 计算模块权重的逐元素乘积
        pvm = m.weight.mul(m.weight)
        
        # 保存当前模块权重的版本号
        m_weight_version_saved = m.weight._version
        
        # 在模块上应用原地加一操作
        m = m._apply(add_one_inplace)
        
        # 断言模块权重的版本号已经增加
        self.assertGreater(m.weight._version, m_weight_version_saved)
        
        # 使用断言检查是否抛出 RuntimeError，预期消息为"modified by an inplace operation"
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            pvm.backward(torch.randn(10, 20))
        
        # 重新创建一个线性层模块，输入维度为20，输出维度为10
        m = nn.Linear(20, 10)
        
        # 设置模块权重的梯度为随机张量，并要求梯度计算
        m.weight.grad = torch.randn(10, 20).requires_grad_()
        
        # 计算模块权重梯度的逐元素乘积
        pgm = m.weight.grad.mul(m.weight.grad)
        
        # 保存当前模块权重梯度的版本号
        m_weight_grad_version_saved = m.weight.grad._version
        
        # 在模块上应用原地加一操作
        m = m._apply(add_one_inplace)
        
        # 断言模块权重梯度的版本号已经增加
        self.assertGreater(m.weight.grad._version, m_weight_grad_version_saved)
        
        # 使用断言检查是否抛出 RuntimeError，预期消息为"modified by an inplace operation"
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            pgm.backward(torch.randn(10, 20))

    # 定义一个测试方法，用于验证交换模块参数后对梯度积累的影响
    def test_swap_module_params_poisons_acc_grad(self):
        try:
            # 设置将来的特性，允许在模型转换时交换模块参数
            torch.__future__.set_swap_module_params_on_conversion(True)
            
            # (1) 测试 backward 在 _apply 后不能运行
            # forward 运算会初始化累积梯度节点，增加参数张量的使用计数
            # 另外，如果有张量被保存用于反向传播，它们的使用计数也会增加
            m = torch.nn.Linear(2, 3)
            inp = torch.randn(2, 2)
            out = m(inp)
            m.half()
            
            # 断言所有参数张量的数据类型为 torch.float16
            self.assertTrue(all(p.dtype == torch.float16 for p in m.parameters()))
            
            # 使用断言检查是否抛出 RuntimeError，预期消息为"Trying to execute AccumulateGrad node that was poisoned by swap_tensors"
            with self.assertRaisesRegex(RuntimeError, "Trying to execute AccumulateGrad node that was poisoned by swap_tensors"):
                out.sum().backward()
            
            # (2) 测试 backward() 后可以运行 _apply
            # 运行 backward 后，所有保存用于反向传播的引用将被清除
            # 所以使用计数将为2（来自张量本身的1和累积梯度节点的1），swap_tensors 应该允许这样的操作
            inp2 = torch.randn(2, 2, dtype=torch.half)
            out2 = m(inp2)
            out2.sum().backward()
            m.float()
            
            # 断言所有参数张量的数据类型为 torch.float32
            self.assertTrue(all(p.dtype == torch.float32 for p in m.parameters()))
            
            # 使用线性层模块进行额外的前向传播
            out3 = m(inp)
        
        finally:
            # 最终，还原将来的特性，禁止在模型转换时交换模块参数
            torch.__future__.set_swap_module_params_on_conversion(False)
    # 测试神经网络层的类型和设备转换

    # 创建一个线性层，输入维度为10，输出维度为20
    l = nn.Linear(10, 20)

    # 创建一个空的神经网络模块
    net = nn.Module()

    # 将创建的线性层l添加到神经网络模块net中，作为其属性l
    net.l = l

    # 再次将线性层l添加到神经网络模块net中，作为属性l2
    net.l2 = l

    # 添加一个空的子模块'empty'到神经网络模块net中
    net.add_module('empty', None)

    # 在神经网络模块net中注册一个缓冲区'indices'，其中存储一个包含单个元素的长整型张量
    net.register_buffer('indices', torch.LongTensor(1))

    # 将整个神经网络模块net转换为单精度浮点数类型
    net.float()

    # 验证线性层l的权重数据类型为单精度浮点数张量
    self.assertIsInstance(l.weight.data, torch.FloatTensor)

    # 验证线性层l的偏置数据类型为单精度浮点数张量
    self.assertIsInstance(l.bias.data, torch.FloatTensor)

    # 验证神经网络模块net中缓冲区'indices'的数据类型为长整型张量
    self.assertIsInstance(net.indices, torch.LongTensor)

    # 将整个神经网络模块net转换为双精度浮点数类型
    net.double()

    # 验证线性层l的权重数据类型为双精度浮点数张量
    self.assertIsInstance(l.weight.data, torch.DoubleTensor)

    # 验证线性层l的偏置数据类型为双精度浮点数张量
    self.assertIsInstance(l.bias.data, torch.DoubleTensor)

    # 验证神经网络模块net中缓冲区'indices'的数据类型仍为长整型张量
    self.assertIsInstance(net.indices, torch.LongTensor)

    # 将整个神经网络模块net转换为半精度浮点数类型
    net.to(torch.half)

    # 验证线性层l的权重数据类型为半精度浮点数张量
    self.assertIsInstance(l.weight.data, torch.HalfTensor)

    # 验证线性层l的偏置数据类型为半精度浮点数张量
    self.assertIsInstance(l.bias.data, torch.HalfTensor)

    # 验证神经网络模块net中缓冲区'indices'的数据类型仍为长整型张量
    self.assertIsInstance(net.indices, torch.LongTensor)

    # 如果在测试环境中启用了CUDA，将整个神经网络模块net转换为单精度浮点数类型并移动到GPU上
    if TEST_CUDA:
        net.float().cuda()

        # 验证线性层l的权重数据类型为CUDA上的单精度浮点数张量
        self.assertIsInstance(l.weight.data, torch.cuda.FloatTensor)

        # 验证线性层l的偏置数据类型为CUDA上的单精度浮点数张量
        self.assertIsInstance(l.bias.data, torch.cuda.FloatTensor)

        # 验证神经网络模块net中缓冲区'indices'的数据类型为CUDA上的长整型张量
        self.assertIsInstance(net.indices, torch.cuda.LongTensor)

        # 将整个神经网络模块net从GPU移动回CPU
        net.cpu()

        # 验证线性层l的权重数据类型恢复为单精度浮点数张量（在CPU上）
        self.assertIsInstance(l.weight.data, torch.FloatTensor)

        # 验证线性层l的偏置数据类型恢复为单精度浮点数张量（在CPU上）
        self.assertIsInstance(l.bias.data, torch.FloatTensor)

        # 验证神经网络模块net中缓冲区'indices'的数据类型为长整型张量（在CPU上）
        self.assertIsInstance(net.indices, torch.LongTensor)

        # 将整个神经网络模块net转换为双精度浮点数类型并移动到CUDA上
        net.to("cuda", torch.double, True)

        # 验证线性层l的权重数据类型为CUDA上的双精度浮点数张量
        self.assertIsInstance(l.weight.data, torch.cuda.DoubleTensor)

        # 验证线性层l的偏置数据类型为CUDA上的双精度浮点数张量
        self.assertIsInstance(l.bias.data, torch.cuda.DoubleTensor)

        # 验证神经网络模块net中缓冲区'indices'的数据类型为CUDA上的长整型张量
        self.assertIsInstance(net.indices, torch.cuda.LongTensor)

        # 将整个神经网络模块net转换为指定设备和半精度浮点数类型
        net.to(torch.empty(1, device="cuda:0", dtype=torch.half))

        # 验证线性层l的权重数据类型为CUDA上的半精度浮点数张量
        self.assertIsInstance(l.weight.data, torch.cuda.HalfTensor)

        # 验证线性层l的偏置数据类型为CUDA上的半精度浮点数张量
        self.assertIsInstance(l.bias.data, torch.cuda.HalfTensor)

        # 验证神经网络模块net中缓冲区'indices'的数据类型为CUDA上的长整型张量
        self.assertIsInstance(net.indices, torch.cuda.LongTensor)

    # 将整个神经网络模块net转换回CPU上，并指定为非阻塞操作
    net.to(torch.device("cpu"), non_blocking=True)

    # 验证线性层l的权重数据类型恢复为半精度浮点数张量（在CPU上）
    self.assertIsInstance(l.weight.data, torch.HalfTensor)

    # 验证线性层l的偏置数据类型恢复为半精度浮点数张量（在CPU上）
    self.assertIsInstance(l.bias.data, torch.HalfTensor)

    # 验证神经网络模块net中缓冲区'indices'的数据类型为长整型张量（在CPU上）
    self.assertIsInstance(net.indices, torch.LongTensor)

    # 将整个神经网络模块net转换为单精度浮点数类型
    net.to(torch.float)

    # 验证线性层l的权重数据类型为单精度浮点数张量
    self.assertIsInstance(l.weight.data, torch.FloatTensor)

    # 验证线性层l的偏置数据类型为单精度浮点数张量
    self.assertIsInstance(l.bias.data, torch.FloatTensor)

    # 将整个神经网络模块net转换为指定双精度浮点数类型（无效操作）
    net.to(torch.DoubleTensor(1))

    # 验证线性层l的权重数据类型仍为单精度浮点数张量（未变化）
    self.assertIsInstance(l.weight.data, torch.FloatTensor)

    # 验证线性层l的偏置数据类型仍为单精度浮点数张量（未变化）
    self.assertIsInstance(l.bias.data, torch.FloatTensor)

    # 如果在测试环境中启用了CUDA，将整个神经网络模块net转换为指定CUDA设备和单精度浮点数类型
    if TEST_CUDA:
        net.to(device='cuda', dtype=torch.float)

        # 验证线性层l的权重数据类型为CUDA上的单精度浮点数张量
        self.assertIsInstance(l.weight.data, torch.cuda.FloatTensor)

        # 验证线性层l的偏置数据类型为CUDA上的单精度浮点数张量
        self.assertIsInstance(l.bias.data, torch.cuda.FloatTensor)
    # 定义一个测试函数，用于测试将模型参数转换为向量的功能
    def test_parameters_to_vector(self):
        # 创建一个二维卷积层，输入通道数为3，输出通道数为10，卷积核大小为5
        conv1 = nn.Conv2d(3, 10, 5)
        # 创建一个全连接层，输入特征数为10，输出特征数为20
        fc1 = nn.Linear(10, 20)
        # 创建一个序列模型，包含先前定义的卷积层和全连接层
        model = nn.Sequential(conv1, fc1)

        # 调用函数 parameters_to_vector，将模型参数转换为一个向量
        vec = parameters_to_vector(model.parameters())
        # 断言转换后的向量大小为980
        self.assertEqual(vec.size(0), 980)

    # 定义一个测试函数，用于测试将向量转换回模型参数的功能
    def test_vector_to_parameters(self):
        # 创建一个二维卷积层，输入通道数为3，输出通道数为10，卷积核大小为5
        conv1 = nn.Conv2d(3, 10, 5)
        # 创建一个全连接层，输入特征数为10，输出特征数为20
        fc1 = nn.Linear(10, 20)
        # 创建一个序列模型，包含先前定义的卷积层和全连接层
        model = nn.Sequential(conv1, fc1)

        # 创建一个长度为980的张量，用于设置模型参数的值
        vec = torch.arange(0., 980)
        # 调用函数 vector_to_parameters，将给定向量设置为模型的参数
        vector_to_parameters(vec, model.parameters())

        # 获取模型中第一个参数的第一个元素，作为样本值
        sample = next(model.parameters())[0, 0, 0]
        # 断言样本值与向量的前5个元素是否相等
        self.assertTrue(torch.equal(sample.data, vec.data[:5]))

    # 定义一个测试函数，用于测试对循环神经网络权重应用权重归一化的功能
    def test_rnn_weight_norm(self):
        # 定义一个内部函数，用于检查给定层的权重归一化效果
        def check_weight_norm(l, name, num_params):
            # 对给定层 l 应用权重归一化，指定归一化的参数名为 name
            l = torch.nn.utils.weight_norm(l, name=name)
            # 断言归一化后的参数数量减少1个
            self.assertEqual(
                sum(isinstance(p, torch.nn.Parameter) for p in l._flat_weights),
                num_params - 1,
            )

            # 移除权重归一化效果，恢复原始的参数表示
            l = torch.nn.utils.remove_weight_norm(l, name=name)
            # 断言恢复后的参数数量与初始数量一致
            self.assertEqual(
                sum(isinstance(p, torch.nn.Parameter) for p in l._flat_weights),
                num_params,
            )

            # 确保在移除权重归一化后，._parameters 和 .named_parameters 包含正确的参数
            # 具体来说，原始权重 ('weight_ih_l0') 应该放回参数中，
            # 而归一化组件 ('weight_ih_l0_v' 和 'weight_ih_l0_g') 应该被移除
            self.assertTrue(name in l._parameters)
            self.assertIsNotNone(l._parameters[name])
            self.assertTrue(name + '_v' not in l._parameters)
            self.assertTrue(name + '_g' not in l._parameters)
            self.assertTrue(name in dict(l.named_parameters()))
            self.assertIsNotNone(dict(l.named_parameters())[name])
            self.assertTrue(name + '_v' not in dict(l.named_parameters()))
            self.assertTrue(name + '_g' not in dict(l.named_parameters()))

        # 测试对输入大小和隐藏大小都为32的 LSTM 层应用权重归一化
        check_weight_norm(torch.nn.LSTM(32, 32), 'weight_ih_l0', 4)
        # 测试对输入大小和隐藏大小都为32，投影大小为16的 LSTM 层应用权重归一化
        check_weight_norm(torch.nn.LSTM(32, 32, proj_size=16), 'weight_hr_l0', 5)
    # 定义一个测试函数，用于测试权重归一化功能
    def test_weight_norm(self):
        # 对于每种数据类型，执行以下测试
        for dtype in [torch.float, torch.bfloat16]:
            # 创建一个随机输入张量
            input = torch.randn(3, 4, dtype=dtype)
            # 创建一个线性层，并指定数据类型
            m = nn.Linear(4, 5).to(dtype=dtype)
            # 计算预期输出
            expected_output = m(input)

            # 添加权重归一化
            m = torch.nn.utils.weight_norm(m)
            # 断言权重归一化后的权重v的大小与原始权重一致
            self.assertEqual(m.weight_v.size(), m.weight.size())
            # 断言权重归一化后的权重g的大小为(5, 1)
            self.assertEqual(m.weight_g.size(), (5, 1))
            # 断言调用模型后输出与预期输出一致，设置误差限
            self.assertEqual(m(input), expected_output, atol=dtype2prec_DONTUSE[dtype], rtol=0)

            # 移除权重归一化
            m = torch.nn.utils.remove_weight_norm(m)
            # 断言模型不再具有权重g和权重v属性
            self.assertFalse(hasattr(m, 'weight_g'))
            self.assertFalse(hasattr(m, 'weight_v'))
            # 再次断言调用模型后输出与预期输出一致，设置误差限
            self.assertEqual(m(input), expected_output, atol=dtype2prec_DONTUSE[dtype], rtol=0)

            # 使用dim=1进行权重归一化测试
            m = torch.nn.utils.weight_norm(m, dim=1)
            # 断言权重归一化后的权重v的大小与原始权重一致
            self.assertEqual(m.weight_v.size(), m.weight.size())
            # 断言权重归一化后的权重g的大小为(1, 4)
            self.assertEqual(m.weight_g.size(), (1, 4))
            # 再次断言调用模型后输出与预期输出一致，设置误差限
            self.assertEqual(m(input), expected_output, atol=dtype2prec_DONTUSE[dtype], rtol=0)

            # 使用dim=None进行权重归一化测试
            m = nn.Linear(4, 5).to(dtype=dtype)
            expected_output = m(input)
            m = torch.nn.utils.weight_norm(m, dim=None)
            # 断言调用模型后输出与预期输出一致
            self.assertEqual(m(input), expected_output)

            # 使用assertRaisesRegex断言运行时错误，确保不能注册两次权重归一化钩子
            with self.assertRaisesRegex(RuntimeError, 'register two weight_norm hooks'):
                m = torch.nn.utils.weight_norm(m)
                m = torch.nn.utils.weight_norm(m)

        # 对于float16数据类型，模块的前向传播不工作，但仍应能够注册权重归一化，因为通常在发送模块到CUDA之前进行此操作
        m = nn.Linear(4, 5, dtype=torch.float16)
        m = torch.nn.utils.weight_norm(m)

    # 定义测试函数，用于测试ParameterList和ParameterDict设置属性时的行为
    def test_parameterlistdict_setting_attributes(self):
        # 使用catch_warnings记录警告信息
        with warnings.catch_warnings(record=True) as w:
            # 创建包含两个参数张量的ParameterList
            mod = nn.ParameterList(map(nn.Parameter, [torch.rand(2), torch.rand(2)]))
        # 断言没有警告信息被记录
        self.assertTrue(len(w) == 0)

        with warnings.catch_warnings(record=True) as w:
            # 将模块设为训练模式
            mod.train()
            # 将模块设为评估模式
            mod.eval()
        # 断言没有警告信息被记录
        self.assertTrue(len(w) == 0)

        with warnings.catch_warnings(record=True) as w:
            # 创建包含两个参数的ParameterDict
            mod = nn.ParameterDict({"a": nn.Parameter(torch.rand(2)), "b": nn.Parameter(torch.rand(2))})
        # 断言没有警告信息被记录
        self.assertTrue(len(w) == 0)

        with warnings.catch_warnings(record=True) as w:
            # 将模块设为训练模式
            mod.train()
            # 将模块设为评估模式
            mod.eval()
        # 断言没有警告信息被记录
        self.assertTrue(len(w) == 0)
    def test_parameterlistdict_pickle(self):
        # 定义警告信息字符串，用于未来可能触发的警告
        WEIGHTS_ONLY_WARN = "You are using `torch.load` with `weights_only=False`"
        
        # 创建包含两个随机张量的参数列表，并将其转换为参数对象
        m = nn.ParameterList(map(nn.Parameter, [torch.rand(2), torch.rand(2)]))
        
        # 使用 assertWarnsRegex 检查是否会触发 FutureWarning，并加载/保存参数列表
        with self.assertWarnsRegex(FutureWarning, WEIGHTS_ONLY_WARN):
            m = pickle.loads(pickle.dumps(m))

        # 再次创建包含两个随机张量的参数列表，清空相关 hooks 和 buffers
        m = nn.ParameterList(map(nn.Parameter, [torch.rand(2), torch.rand(2)]))
        del m._forward_pre_hooks, m._state_dict_hooks, m._load_state_dict_pre_hooks, m._non_persistent_buffers_set
        
        # 使用 assertWarnsRegex 再次检查是否会触发 FutureWarning，并加载/保存参数列表
        with self.assertWarnsRegex(FutureWarning, WEIGHTS_ONLY_WARN):
            m = pickle.loads(pickle.dumps(m))

        # 创建包含两个随机张量的参数字典
        m = nn.ParameterDict({"a": nn.Parameter(torch.rand(2)), "b": nn.Parameter(torch.rand(2))})
        
        # 使用 assertWarnsRegex 检查是否会触发 FutureWarning，并加载/保存参数字典
        with self.assertWarnsRegex(FutureWarning, WEIGHTS_ONLY_WARN):
            m = pickle.loads(pickle.dumps(m))

        # 再次创建包含两个随机张量的参数字典，清空相关 hooks 和 buffers
        m = nn.ParameterDict({"a": nn.Parameter(torch.rand(2)), "b": nn.Parameter(torch.rand(2))})
        del m._forward_pre_hooks, m._state_dict_hooks, m._load_state_dict_pre_hooks, m._non_persistent_buffers_set
        
        # 使用 assertWarnsRegex 再次检查是否会触发 FutureWarning，并加载/保存参数字典
        with self.assertWarnsRegex(FutureWarning, WEIGHTS_ONLY_WARN):
            m = pickle.loads(pickle.dumps(m))

    def test_weight_norm_pickle(self):
        # 创建具有权重归一化的线性层，并加载/保存它
        m = torch.nn.utils.weight_norm(nn.Linear(5, 7))
        m = pickle.loads(pickle.dumps(m))
        self.assertIsInstance(m, nn.Linear)

    @skipIfTorchDynamo("TorchDynamo fails here for unknown reasons")
    @set_default_dtype(torch.double)
    @skipIfNoLapack
    def test_spectral_norm_dim(self):
        # 创建输入张量，并使用具有谱归一化的转置卷积层
        inp = torch.randn(2, 3, 10, 12)
        m = nn.ConvTranspose2d(3, 4, (5, 6))
        m = torch.nn.utils.spectral_norm(m)
        
        # 执行前向传播，并验证输出形状兼容性
        x = m(inp)
        
        # 检查权重向量 u 的维度是否与原始权重的某个切片维度相同
        self.assertEqual(m.weight_u.shape, m.weight_orig[0, :, 0, 0].shape)

    def test_spectral_norm_forward(self):
        # 创建输入张量，并使用具有谱归一化的线性层
        input = torch.randn(3, 5)
        m = nn.Linear(5, 7)
        m = torch.nn.utils.spectral_norm(m)
        
        # 执行简单的前向传播
        _weight, _bias, _u = m.weight_orig, m.bias, m.weight_u
        _weight_mat = _weight.view(_weight.size(0), -1)
        _v = torch.mv(_weight_mat.t(), _u)
        _v = F.normalize(_v, dim=0, eps=1e-12)
        _u = torch.mv(_weight_mat, _v)
        _u = F.normalize(_u, dim=0, eps=1e-12)
        _weight.data /= torch.dot(_u, torch.matmul(_weight_mat, _v))
        
        # 使用谱归一化的线性层进行前向传播，并验证预期输出
        out_hat = torch.nn.functional.linear(input, _weight, _bias)
        expect_out = m(input)
        self.assertEqual(expect_out, out_hat)

    def test_spectral_norm_pickle(self):
        # 创建具有谱归一化的线性层，并加载/保存它
        m = torch.nn.utils.spectral_norm(nn.Linear(5, 7))
        m = pickle.loads(pickle.dumps(m))
        self.assertIsInstance(m, nn.Linear)
    # 定义测试阈值为整数的功能测试方法
    def test_threshold_int(self):
        # 创建一个包含整数的张量
        x = torch.tensor([-3, -2, -1, 0, 1, 2, 3])
        # 期望的输出张量，对于非正数元素阈值设置为99
        expected = torch.tensor([99, 99, 99, 99, 1, 2, 3])
        # 断言阈值函数的输出与期望值相等
        self.assertEqual(F.threshold(x, 0, 99), expected)

    # 定义测试阈值为bfloat16和half精度的功能测试方法
    def test_threshold_bfloat16_half(self):
        # 创建一个包含随机数据的张量
        x = torch.randn(100)
        # 遍历bfloat16和half两种精度类型
        for dtype in [torch.bfloat16, torch.half]:
            # 遍历不同的阈值设置
            for threshold in [0, -0.5, 0.5, float('inf'), float('-inf'), float('nan')]:
                # 计算预期的阈值处理结果，并转换为指定精度类型的张量
                expected = F.threshold(x, threshold, 0).to(dtype=dtype).float()
                # 使用指定精度类型处理输入数据，并转换为float类型的结果
                res_bf16 = F.threshold(x.to(dtype=dtype), threshold, 0).float()
                # 断言处理结果与预期值相等
                self.assertEqual(res_bf16, expected)

    # 根据条件跳过测试，依赖于torch.backends.quantized.supported_engines是否包含'fbgemm'
    @unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines,
                         'Linear_FP16_weight requires FBGEMM. FBGEMM is only optimized for CPUs'
                         ' with instruction set support avx2 or newer.')
    def test_fb_fc_packed(self):
        # 创建随机的输入、权重和偏置张量
        X = np.random.rand(16, 16).astype(np.float32) - 0.5
        W = np.random.rand(16, 16).astype(np.float32) - 0.5
        b = np.random.rand(16).astype(np.float32) - 0.5

        # 定义仿射操作函数
        def fc_op(X, W, b):
            return np.dot(X, W.T) + b

        # 将numpy数组转换为torch张量
        x_tensor = torch.tensor(X)
        w_tensor = torch.tensor(W)
        b_tensor = torch.tensor(b)
        # 使用fbgemm优化的线性仿射操作，期望输出与numpy仿射操作结果接近
        packed_w_tensor = torch.fbgemm_pack_gemm_matrix_fp16(w_tensor)
        actual_output = torch.fbgemm_linear_fp16_weight(x_tensor, packed_w_tensor, b_tensor)
        expected_output = fc_op(X, W, b)
        # 使用torch.testing.assert_close函数断言期望输出与实际输出在指定的容差范围内相等
        torch.testing.assert_close(torch.from_numpy(expected_output), actual_output.cpu(), atol=1e-3, rtol=1e-3)

    # 测试在标量张量上使用F.pad函数是否会引发运行时错误
    def test_pad_scalar_error(self):
        # 创建一个标量张量，并标记为需要梯度计算
        inputs = torch.tensor(0., requires_grad=True)
        # 断言调用F.pad函数时会引发运行时错误
        self.assertRaises(RuntimeError, lambda: F.pad(inputs, (1, 1)))
        # 断言调用F.pad函数时会引发运行时错误
        self.assertRaises(RuntimeError, lambda: F.pad(inputs, (1,)))

    # 测试从掩码生成嵌套张量的功能
    def test_nested_tensor_from_mask(self):
        # 定义三维随机数据的维度
        N, L, D = 10, 12, 14

        # 创建三维随机数据张量和掩码张量
        input = torch.rand(N, L, D)
        mask = torch.ones(N, L, dtype=torch.bool)
        # 保持第一行全为True，以保持nt的大小不变
        for i in range(1, N):
            # 随机选择一个位置，将之后的元素设置为False
            end = torch.randint(1, L, size=()).item()
            mask[i, end:] = False

        # 根据掩码生成嵌套张量
        nt = torch._nested_tensor_from_mask(input, mask)
        # 将嵌套张量转换为填充的张量，未填充部分用0填充
        input_convert = nt.to_padded_tensor(0.)
        # 使用掩码将原始输入数据中对应位置的元素置为0
        input.masked_fill_(mask.reshape(N, L, 1).logical_not(), 0.)

        # 断言转换后的填充张量与原始输入数据在所有位置上相等
        self.assertEqual(input, input_convert)
    def test_nested_tensor_from_mask_error(self):
        N, L, D = 10, 12, 14

        input = torch.rand(N, L, D)
        # Mask is not bool
        mask = torch.zeros(N, L, dtype=torch.float)
        self.assertRaises(RuntimeError, lambda: torch._nested_tensor_from_mask(input, mask))

        # Mask size is not 2
        mask = torch.zeros(N, L, D, dtype=torch.bool)
        self.assertRaises(RuntimeError, lambda: torch._nested_tensor_from_mask(input, mask))

        # Input size is not 3
        mask = torch.zeros(N, L, dtype=torch.bool)
        input = torch.rand(N, L)
        self.assertRaises(RuntimeError, lambda: torch._nested_tensor_from_mask(input, mask))

        # Mask size does not match input
        mask = torch.zeros(N + 1, L + 1, dtype=torch.bool)
        input = torch.rand(N, L, D)
        self.assertRaises(RuntimeError, lambda: torch._nested_tensor_from_mask(input, mask))

        # Mask is not padding format
        mask = torch.ones(N, L, dtype=torch.bool)
        mask[0, 0] = False
        mask[0, 2] = False
        self.assertRaises(RuntimeError, lambda: torch._nested_tensor_from_mask(input, mask))



        inputs = torch.randn(1, 3, 4, 4, requires_grad=True, dtype=torch.double)
        # Check gradient for normalization with L1 norm along the last dimension
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=1, dim=-1), (inputs,)))
        # Check gradient for normalization with L2 norm along the second-to-last dimension
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=2, dim=-2), (inputs,)))

        inputs = torch.randn((), requires_grad=True)
        # Check gradient for normalization of a scalar tensor with L1 norm along the last dimension
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=1, dim=-1), (inputs,)))



    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    # Skip the test for ROCm as per https://github.com/pytorch/pytorch/issues/53190
    @skipIfRocm
    def test_broadcast_double_backwards_gpu(self):
        tensors = (torch.randn(4, 4, device='cuda', requires_grad=True, dtype=torch.double),
                   torch.randn(4, 4, device='cuda', requires_grad=True, dtype=torch.double),
                   torch.randn(4, 4, device='cuda', requires_grad=True, dtype=torch.double))
        # Assertion for gradient and double gradient checks with broadcasting on GPU
        _assertGradAndGradgradChecks(self, lambda *i: Broadcast.apply((0, 1), *i), tensors,
                                     check_batched_grad=False)



    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    # Skip the test for ROCm as per https://github.com/pytorch/pytorch/issues/53190
    @skipIfRocm
    # 定义一个测试方法，用于测试不需要梯度的广播操作
    def test_broadcast_not_requiring_grad(self):
        # 创建包含不同要素的变量列表，其中一些需要梯度，一些不需要
        variables = [
            torch.randn(1, 2, device='cuda', requires_grad=True),   # 生成一个需要梯度的 CUDA 张量
            torch.randn(1, 2, device='cuda', requires_grad=False),  # 生成一个不需要梯度的 CUDA 张量
            torch.randn(1, 2, device='cuda', requires_grad=False),  # 生成一个不需要梯度的 CUDA 张量
            torch.randn(1, 2, device='cuda', requires_grad=True),   # 生成一个需要梯度的 CUDA 张量
            torch.randn(1, 2, device='cuda', requires_grad=True),   # 生成一个需要梯度的 CUDA 张量
        ]
        # 调用自定义的广播操作函数 Broadcast.apply，将 variables 列表作为参数传入
        broadcasted_variables = Broadcast.apply((0, 1), *variables)
        # 遍历广播后的变量列表，对比每个广播后的变量与其对应输入变量的梯度需求
        for output_idx, broadcasted_var in enumerate(broadcasted_variables):
            input_var = variables[output_idx % len(variables)]  # 获取当前广播变量对应的输入变量
            # 断言当前广播变量的梯度需求与其对应的输入变量一致
            self.assertEqual(input_var.requires_grad, broadcasted_var.requires_grad)

    # 如果不支持多GPU测试，则跳过该测试方法
    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_broadcast_no_grad(self):
        # 创建一个需要梯度的 CUDA 张量 x
        x = torch.randn(1, 2, dtype=torch.float32, requires_grad=True, device='cuda')
        # 使用 torch.no_grad() 上下文管理器，对 x 进行广播操作
        with torch.no_grad():
            broadcasted = Broadcast.apply((0, 1), x)
        # 断言原始张量 x 依然需要梯度
        self.assertTrue(x.requires_grad)
        # 遍历广播后的结果张量列表，断言所有广播后的张量都不需要梯度
        for output in broadcasted:
            self.assertFalse(output.requires_grad)
    # 定义一个测试方法，用于测试神经网络模型的状态字典
    def test_state_dict(self):
        # 创建一个线性层，输入和输出都是5维
        l = nn.Linear(5, 5)
        # 创建一个空的模块块
        block = nn.Module()
        # 给模块块添加一个没有偏置的二维卷积层
        block.conv = nn.Conv2d(3, 3, 3, bias=False)
        # 创建一个新的神经网络模型
        net = nn.Module()
        # 将之前创建的线性层l作为net的第一个线性层
        net.linear1 = l
        # 将同一个线性层l再次作为net的第二个线性层
        net.linear2 = l
        # 添加一个二维批量归一化层
        net.bn = nn.BatchNorm2d(2)
        # 将之前创建的模块块block作为net的一个子模块
        net.block = block
        # 添加一个名为'empty'的空子模块
        net.add_module('empty', None)

        # 获取整个网络模型net的状态字典
        state_dict = net.state_dict()
        # 断言状态字典的长度为10，即包含10个键值对
        self.assertEqual(len(state_dict), 10)
        # 断言状态字典的_metadata属性的长度为6，即包含6个元数据项
        self.assertEqual(len(state_dict._metadata), 6)
        # 断言空字符串''在状态字典的_metadata中
        self.assertIn('', state_dict._metadata)
        # 断言'linear1'在状态字典的_metadata中
        self.assertIn('linear1', state_dict._metadata)
        # 断言'linear1.weight'在状态字典中
        self.assertIn('linear1.weight', state_dict)
        # 断言'linear1.bias'在状态字典中
        self.assertIn('linear1.bias', state_dict)
        # 断言'linear2'在状态字典的_metadata中
        self.assertIn('linear2', state_dict._metadata)
        # 断言'linear2.weight'在状态字典中
        self.assertIn('linear2.weight', state_dict)
        # 断言'linear2.bias'在状态字典中
        self.assertIn('linear2.bias', state_dict)
        # 断言'block'在状态字典的_metadata中
        self.assertIn('block', state_dict._metadata)
        # 断言'block.conv'在状态字典的_metadata中
        self.assertIn('block.conv', state_dict._metadata)
        # 断言'block.conv.weight'在状态字典中
        self.assertIn('block.conv.weight', state_dict)
        # 再次断言'block.conv.weight'在状态字典中，确保没有重复
        self.assertIn('block.conv.weight', state_dict)
        # 断言'block.conv.bias'不在状态字典中
        self.assertNotIn('block.conv.bias', state_dict)
        # 断言'bn'在状态字典的_metadata中
        self.assertIn('bn', state_dict._metadata)
        # 断言'bn.weight'在状态字典中
        self.assertIn('bn.weight', state_dict)
        # 断言'bn.bias'在状态字典中
        self.assertIn('bn.bias', state_dict)
        # 断言'bn.running_var'在状态字典中
        self.assertIn('bn.running_var', state_dict)
        # 断言'bn.running_mean'在状态字典中
        self.assertIn('bn.running_mean', state_dict)
        # 断言'bn.num_batches_tracked'在状态字典中
        self.assertIn('bn.num_batches_tracked', state_dict)
        # 断言状态字典中不存在以'empty'开头的任何键
        self.assertFalse(any(k.startswith('empty') for k in state_dict.keys()))
        
        # 遍历状态字典中的每个键值对
        for k, v in state_dict.items():
            # 初始化param为net
            param = net
            # 根据点号分割键k，并逐级获取param的属性
            for component in k.split('.'):
                param = getattr(param, component)
                # 如果param是Parameter类型，则获取其数据
                if isinstance(param, Parameter):
                    param = param.data
            # 断言v的数据指针与param的数据指针相同
            self.assertEqual(v.data_ptr(), param.data_ptr())

        # 再次创建一个新的线性层l
        l = nn.Linear(5, 5)
        # 获取线性层l的状态字典
        state_dict = l.state_dict()
        # 断言状态字典的长度为2，即包含2个键值对
        self.assertEqual(len(state_dict), 2)
        # 断言状态字典的_metadata属性的长度为1，即包含1个元数据项
        self.assertEqual(len(state_dict._metadata), 1)
        # 断言空字符串''在状态字典的_metadata中
        self.assertIn('', state_dict._metadata)
        # 断言状态字典的_metadata中的''对应的版本号version大于等于0
        self.assertTrue(state_dict._metadata['']['version'] >= 0)
        # 断言状态字典中'weight'对应的数据指针与线性层l的权重数据指针相同
        self.assertEqual(state_dict['weight'].data_ptr(), l.weight.data_ptr())
        # 断言状态字典中'bias'对应的数据指针与线性层l的偏置数据指针相同
        self.assertEqual(state_dict['bias'].data_ptr(), l.bias.data_ptr())

        # 引用网址 https://github.com/pytorch/pytorch/pull/75507#issuecomment-1110291545
        # 使用self.assertNotWarn检查调用l.state_dict(destination=dict())时是否会警告，验证kwarg destination在没有_metadata的情况下不应该产生警告
        self.assertNotWarn(lambda: l.state_dict(destination=dict()), "Should not warn kwarg destination w/o _metadata")
    # 定义一个测试方法，用于测试带有额外状态的模块
    def test_extra_state(self):

        # 定义一个子模块类，继承自torch.nn.Module
        class SubModule(torch.nn.Module):
            # 构造方法，初始化实例属性foo
            def __init__(self, foo):
                super().__init__()
                self.foo = foo

            # 获取额外状态的方法，返回包含foo属性的字典
            def get_extra_state(self):
                return {
                    'foo': self.foo
                }

            # 设置额外状态的方法，根据给定的状态字典设置foo属性
            def set_extra_state(self, state):
                self.foo = state['foo']

        # 定义一个主模块类，继承自torch.nn.Module
        class MyModule(torch.nn.Module):
            # 构造方法，初始化实例属性sub和bar
            def __init__(self, foo, bar):
                super().__init__()
                self.sub = SubModule(foo)
                self.bar = bar

            # 获取额外状态的方法，返回包含bar属性的字典
            def get_extra_state(self):
                return {
                    'bar': self.bar
                }

            # 设置额外状态的方法，根据给定的状态字典设置bar属性
            def set_extra_state(self, state):
                self.bar = state['bar']

        # 创建MyModule的实例m和m2，分别用不同的参数初始化
        m = MyModule(3, 'something')
        m2 = MyModule(5, 'something else')
        
        # 加载m的状态字典到m2中，确保状态字典包含额外的状态信息
        m2.load_state_dict(m.state_dict())
        
        # 断言两个模块的状态字典相等
        self.assertEqual(m.state_dict(), m2.state_dict())
        
        # 断言m2的bar属性与m的bar属性相等
        self.assertEqual(m2.bar, m.bar)
        
        # 断言m2的sub模块的foo属性与m的sub模块的foo属性相等
        self.assertEqual(m2.sub.foo, m.sub.foo)

    # 定义一个测试方法，用于测试不同类型额外状态的情况
    def test_extra_state_non_dict(self):

        # 定义一个模块类，继承自torch.nn.Module，初始化时接受foo参数
        class MyModule(torch.nn.Module):
            def __init__(self, foo):
                super().__init__()
                self.foo = foo

            # 获取额外状态的方法，直接返回foo属性而不是字典
            def get_extra_state(self):
                return self.foo

            # 设置额外状态的方法，直接设置foo属性为给定的状态
            def set_extra_state(self, state):
                self.foo = state

        # 测试不同类型的额外状态：字符串'something'，整数5，以及MyModule类的实例
        for state in ('something', 5, MyModule(3)):
            m = MyModule(state)
            m2 = MyModule('something else')
            m2.load_state_dict(m.state_dict())
            
            # 断言两个模块的状态字典相等
            self.assertEqual(m.state_dict(), m2.state_dict())
            
            # 断言m2的foo属性与m的foo属性相等
            self.assertEqual(m.foo, m2.foo)

    # 定义一个测试方法，用于测试没有set_extra_state方法时的情况
    def test_extra_state_missing_set_extra_state(self):

        # 定义一个模块类，继承自torch.nn.Module，但没有set_extra_state方法
        class MyModule(torch.nn.Module):
            def get_extra_state(self):
                return {
                    'foo': 5
                }

        # 创建MyModule的实例m，并尝试加载其状态字典
        m = MyModule()
        
        # 断言加载状态字典时会引发RuntimeError，并且错误信息包含'Unexpected key'
        with self.assertRaisesRegex(RuntimeError, 'Unexpected key'):
            m.load_state_dict(m.state_dict())

    # 定义一个测试方法，用于测试没有get_extra_state方法时的情况
    def test_extra_state_missing_get_extra_state(self):

        # 定义一个模块类，继承自torch.nn.Module，但没有get_extra_state方法
        class MyModule(torch.nn.Module):
            def set_extra_state(self):
                pass

        # 创建MyModule的实例m，并尝试加载其状态字典
        m = MyModule()
        
        # 断言加载状态字典时会引发RuntimeError，并且错误信息包含'Missing key'
        with self.assertRaisesRegex(RuntimeError, 'Missing key'):
            m.load_state_dict(m.state_dict())

    # 跳过TorchDynamo失败的测试，用于注解测试方法
    @skipIfTorchDynamo("TorchDynamo fails here for unknown reasons")
    # 定义一个测试方法，用于测试参数赋值的行为
    def test_parameter_assignment(self):
        # 创建一个线性层，输入和输出都是大小为5
        l = nn.Linear(5, 5)

        # 定义一个内部函数，用于获取线性层参数的数量
        def num_params():
            return len(list(l.parameters()))

        # 断言初始参数数量为2
        self.assertEqual(num_params(), 2)

        # 创建一个新的参数，并将其赋给线性层的一个属性
        new_param = Parameter(torch.randn(5, 5))
        l.param_name = new_param
        # 断言参数数量增加到3
        self.assertEqual(num_params(), 3)
        # 断言新参数在线性层的参数列表中
        self.assertObjectIn(new_param, l.parameters())

        # 创建一个新的变量，并将其赋给线性层的一个属性
        var = torch.randn(5, 5)
        l.var_name = var
        # 断言参数数量仍为3（变量不应该作为参数保存）
        self.assertEqual(num_params(), 3)
        # 断言新变量的id不在线性层参数列表中
        self.assertNotIn(id(var), map(id, l.parameters()))

        # 确保变量不会被保存为参数
        l.variable_attr = torch.empty(5, 5)
        # 断言参数数量仍为3
        self.assertEqual(num_params(), 3)
        # 将一个空的张量作为参数赋给线性层的一个属性
        l.param_attr = Parameter(torch.empty(5, 5))
        # 断言参数数量增加到4
        self.assertEqual(num_params(), 4)

        # 尝试用一个张量替换一个参数应该引发 TypeError 异常
        def assign_var():
            l.param_attr = torch.empty(5, 5)

        # 断言替换参数为变量会引发 TypeError 异常
        self.assertRaises(TypeError, assign_var)
        # 但是用 None 替换参数应该是可以的
        l.param_attr = None
        # 断言参数数量减少到3
        self.assertEqual(num_params(), 3)
    # 定义一个测试方法，测试属性的赋值和删除操作
    def test_assignment(self):
        # 创建一个空的神经网络模块实例
        l = nn.Module()
        # 创建三个张量作为参数，随机初始化其值
        a = nn.Parameter(torch.randn(2))
        b = nn.Parameter(torch.randn(3))
        c = nn.Parameter(torch.randn(4))
        # 创建三个线性层实例，分别为输入输出均为4、5、6的线性映射
        q = nn.Linear(4, 4)
        r = nn.Linear(5, 5)
        w = nn.Linear(6, 6)

        # 定义一个内部函数，测试属性赋值的影响
        def test_assignments(get_list, a, b, c):
            # 将属性 l.a 设为 None，检查是否为 None
            l.a = None
            self.assertIsNone(l.a)
            # 检查属性 'a' 是否在 l 的字典属性中
            self.assertIn('a', l.__dict__)
            # 将属性 l.a 设为 a，检查是否相等
            l.a = a
            self.assertIs(l.a, a)
            # 检查 get_list() 返回的列表是否包含 a
            self.assertEqual(get_list(), [a])
            # 确保属性 'a' 已经从 l 的字典属性中移除
            self.assertNotIn('a', l.__dict__)

            # 重复上述过程，用属性 'b' 和 'c' 进行测试
            l.b = None
            self.assertIsNone(l.b)
            self.assertIn('b', l.__dict__)
            l.b = b
            self.assertIs(l.b, b)
            self.assertEqual(get_list(), [a, b])
            self.assertNotIn('b', l.__dict__)

            # 移除并重新添加属性 'a'，顺序应保持不变
            l.a = None
            self.assertIsNone(l.a)
            self.assertEqual(get_list(), [b])
            l.a = a
            self.assertIs(l.a, a)
            self.assertEqual(get_list(), [a, b])

            # 用属性 'c' 替换属性 'a'，顺序应保持不变
            l.a = c
            self.assertIs(l.a, c)
            self.assertEqual(get_list(), [c, b])

            # 移除并重新分配属性，它应该出现在列表的末尾
            del l.a
            self.assertFalse(hasattr(l, 'a'))
            l.a = a
            self.assertIs(l.a, a)
            self.assertEqual(get_list(), [b, a])

        # 使用 lambda 函数调用 test_assignments，获取参数列表，并传入 a、b、c
        test_assignments(lambda: list(l.parameters()), a, b, c)
        # 删除属性 'a' 和 'b'，确保参数列表为空
        del l.a, l.b
        self.assertEqual(list(l.parameters()), [])

        # 使用 lambda 函数调用 test_assignments，获取子模块列表，并传入 q、r、w
        test_assignments(lambda: list(l.children()), q, r, w)
        # 再次删除属性 'a' 和 'b'，确保子模块列表为空
        del l.a, l.b
        self.assertEqual(list(l.children()), [])

        # 创建一个长度为 10 的随机张量 buf
        buf = torch.randn(10)
        # 将 buf 注册为 l 的缓冲区，检查其是否被正确存储
        l.register_buffer('buf', buf)
        self.assertIs(l.buf, buf)
        # 将 l.buf 设为 None，检查其是否为 None，并确保不在 l 的字典属性中
        l.buf = None
        self.assertIs(l.buf, None)
        self.assertNotIn('buf', l.__dict__)  # 应该存储在 l._buffers 中
        # 再次将 l.buf 设为 buf，检查其是否在 l 的状态字典中
        l.buf = buf
        self.assertIn('buf', l.state_dict())
        self.assertEqual(l.state_dict()['buf'], buf)

    # 定义一个测试容器复制的方法
    def test_container_copy(self):
        # 定义一个简单的神经网络模型，包含一个线性层
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 5)

            def forward(self, input):
                return self.linear(input)

        # 创建一个输入张量 input，形状为 (2, 4)
        input = torch.randn(2, 4)

        # 创建模型实例 model 和其深拷贝 model_cp
        model = Model()
        model_cp = deepcopy(model)
        # 检查模型和其拷贝在输入 input 下的输出是否相等
        self.assertEqual(model(input).data, model_cp(input).data)

        # 修改 model_cp 的参数，检查其输出是否不再与 model 相等
        model_cp.linear.weight.data[:] = 2
        self.assertNotEqual(model(input).data, model_cp(input).data)
    def test_RNN_cell(self):
        # 这只是一个烟雾测试；这些模块是通过 autograd 实现的，因此不需要进行雅可比测试
        # 循环遍历 RNNCell 和 GRUCell 模块，分别测试是否带有偏置和不带偏置的情况
        for module in (nn.RNNCell, nn.GRUCell):
            for bias in (True, False):
                # 创建随机输入和隐藏状态
                input = torch.randn(3, 10)
                hx = torch.randn(3, 20)
                # 根据给定参数创建 RNNCell 或 GRUCell 模块
                cell = module(10, 20, bias=bias)
                # 运行 6 次模块，更新隐藏状态
                for _ in range(6):
                    hx = cell(input, hx)

                # 对隐藏状态的和进行反向传播
                hx.sum().backward()

    def test_RNN_cell_forward_zero_hidden_size(self):
        # 创建随机输入和零维隐藏状态
        input = torch.randn(3, 10)
        hx = torch.randn(3, 0)
        cell_shared_param = (10, 0)
        # 针对不同的 RNNCell 和 GRUCell，测试其输出形状是否为 [3, 0]
        for cell in (nn.RNNCell(*cell_shared_param, nonlinearity="relu"),
                     nn.RNNCell(*cell_shared_param, nonlinearity="tanh"),
                     nn.GRUCell(*cell_shared_param)):
            self.assertEqual(cell(input, hx).shape, torch.Size([3, 0]))

    def _test_loss_equal_input_target_shape(self, cast):
        # 测试那些输入应具有相同大小的损失函数
        losses = {
            'mse_loss': lambda x, y: F.mse_loss(x, y),
            'l1_loss': lambda x, y: F.l1_loss(x, y),
            'smooth_l1_loss': lambda x, y: F.smooth_l1_loss(x, y),
            'huber_loss': lambda x, y: F.huber_loss(x, y),
            'kl_div': lambda x, y: F.kl_div(x, y),
            'poisson_nll_loss': lambda x, y: F.poisson_nll_loss(x, y),
        }

        # 创建随机输入和目标张量，进行类型转换
        input = cast(torch.randn(3, 5))
        target = cast(torch.randn(5, 3))
        # 对每个损失函数进行测试，检查是否引发异常
        for fn in losses.values():
            self.assertRaises(Exception, lambda: fn(input, target))

    def test_loss_equal_input_target_shape(self):
        # 调用 _test_loss_equal_input_target_shape 方法，传入类型转换函数 lambda x: x
        self._test_loss_equal_input_target_shape(lambda x: x)

    def test_mse_loss_size_warning(self):
        # 创建具有随机数据的张量 i 和 t，并设置需要梯度计算
        i = torch.randn((10, 1), requires_grad=True)
        t = torch.randn((10,))
        with warnings.catch_warnings(record=True) as w:
            # 确保显示警告信息
            warnings.simplefilter("always")
            # 触发均方误差损失函数
            F.mse_loss(i, t)
            # 检查是否出现了警告
            self.assertEqual(len(w), 1)
            self.assertIn('Please ensure they have the same size.', str(w[0]))
    # 测试高斯负对数似然损失函数的广播特性
    def test_gaussian_nll_loss_broadcasting(self):
        # 创建输入张量，包含两个样本和三个特征
        input = torch.tensor([[0.5, 1.5, 2.5], [2., 4., 6.]])
        # 创建完整目标张量，每个样本对应的目标值
        target_full = torch.tensor([[1., 2., 3.], [1., 2., 3.]])
        # 创建部分目标张量，只包含一个样本的目标值
        target_part = torch.tensor([[1., 2., 3.]])
        # 创建完整方差张量，每个样本对应的方差值
        var_full = torch.tensor([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
        # 创建部分方差张量，每个样本对应一个方差值
        var_part1 = torch.tensor([[0.5], [1.5]])
        # 创建部分方差张量，所有样本共用一个方差值
        var_part2 = torch.tensor([0.5, 1.5])
        
        # 计算逐分量的损失值，根据高斯负对数似然的定义
        component_wise_loss = 0.5 * (torch.log(var_full) + (input - target_full)**2 / var_full)
        
        # 测试不同参数组合下函数的返回值是否符合预期
        self.assertEqual(component_wise_loss,
                         F.gaussian_nll_loss(input, target_part, var_full, reduction='none'))
        self.assertEqual(component_wise_loss,
                         F.gaussian_nll_loss(input, target_full, var_part1, reduction='none'))
        self.assertEqual(component_wise_loss,
                         F.gaussian_nll_loss(input, target_full, var_part2, reduction='none'))
        self.assertEqual(component_wise_loss,
                         F.gaussian_nll_loss(input, target_part, var_part1, reduction='none'))
        self.assertEqual(component_wise_loss,
                         F.gaussian_nll_loss(input, target_part, var_part2, reduction='none'))

    # 测试高斯负对数似然损失函数的参数检查
    def test_gaussian_nll_loss_args(self):
        # 创建随机输入张量
        input = torch.randn(3, 5)
        # 测试当方差张量大小不符合预期时是否会抛出异常
        with self.assertRaisesRegex(ValueError, 'var is of incorrect size'):
            target = torch.randn(3, 5)
            var = torch.ones(3, 3)
            torch.nn.functional.gaussian_nll_loss(input, target, var)
        # 测试当方差张量包含负值时是否会抛出异常
        with self.assertRaisesRegex(ValueError, 'var has negative entry/entries'):
            var = -1 * torch.ones(3, 5)
            torch.nn.functional.gaussian_nll_loss(input, target, var)

    # 测试KL散度损失函数的批次平均计算
    def test_KLDivLoss_batch_mean(self):
        # 定义输入张量形状
        input_shape = (2, 5)
        # 创建log_softmax后的张量作为输入概率分布
        log_prob1 = F.log_softmax(torch.randn(input_shape), 1)
        # 创建softmax后的张量作为目标概率分布
        prob2 = F.softmax(torch.randn(input_shape), 1)

        # 创建计算损失的对象，使用批次均值作为减少方式
        loss = nn.KLDivLoss(reduction='batchmean')
        l = loss(log_prob1, prob2)

        # 计算不减少损失的总和并除以批次大小，作为期望的批次均值
        loss_none_reduce = nn.KLDivLoss(reduction='sum')(log_prob1, prob2)
        expected = loss_none_reduce / input_shape[0]

        # 断言批次平均损失与预期值相等
        self.assertEqual(l, expected)

    # 测试带有log目标的KL散度损失函数的批次平均计算
    def test_KLDivLoss_batch_mean_log_target(self):
        # 定义输入张量形状
        input_shape = (2, 5)
        # 创建log_softmax后的张量作为输入概率分布
        log_prob1 = F.log_softmax(torch.randn(input_shape), 1)
        # 创建另一个log_softmax后的张量作为目标概率分布
        log_prob2 = F.log_softmax(torch.randn(input_shape), 1)

        # 创建计算损失的对象，使用批次均值作为减少方式，并指定目标为log概率
        loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        l = loss(log_prob1, log_prob2)

        # 计算不减少损失的总和并除以批次大小，作为期望的批次均值
        loss_none_reduce = nn.KLDivLoss(reduction='sum', log_target=True)(log_prob1, log_prob2)
        expected = loss_none_reduce / input_shape[0]

        # 断言批次平均损失与预期值相等
        self.assertEqual(l, expected)
    # 测试函数，用于验证 CTC 损失函数对输入类型的检查
    def test_CTCLoss_typechecks(self):
        # 创建目标长度张量，包含三个值：30, 25, 20
        target_lengths = torch.tensor([30, 25, 20])
        # 创建输入长度张量，包含三个值：50, 50, 50
        input_lengths = torch.tensor([50, 50, 50])
        # 创建随机目标张量，其长度为目标长度张量之和，类型为整数
        targets = torch.randint(1, 15, (sum(target_lengths),), dtype=torch.int)
        # 创建随机对数概率张量，形状为 (50, 3, 15)，类型为浮点数，然后进行 log_softmax 处理
        log_probs = torch.randn(50, 3, 15, dtype=torch.float).log_softmax(2)
        # 断言运行时错误，测试输入长度转换为浮点数时是否触发错误
        with self.assertRaises(RuntimeError):
            _input_lengths = input_lengths.to(dtype=torch.float)
            torch.nn.functional.ctc_loss(log_probs, targets, _input_lengths, target_lengths)
        # 断言运行时错误，测试目标长度转换为浮点数时是否触发错误
        with self.assertRaises(RuntimeError):
            target_lengths = target_lengths.to(dtype=torch.float)
            torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)

    # 如果 CUDA 不可用，则跳过 CUDA 测试
    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    # 测试函数，用于验证 CTC 损失函数在 CUDA 设备上对长度的检查
    def test_CTCLoss_lengthchecks_cuda(self):
        # 遍历目标长度和输入长度的组合：[[30, 25, 20], [-1, -1, -1]]
        for target_lengths in [[30, 25, 20], [-1, -1, -1]]:
            # 遍历目标长度和输入长度的组合：[[50, 50, 50], [-1, -1, -1]]
            for input_lengths in [[50, 50, 50], [-1, -1, -1]]:
                # 在 CUDA 设备上创建随机目标张量，形状为 (3, 29)，类型为长整型
                targets = torch.randint(1, 15, (3, 29), dtype=torch.long, device='cuda')
                # 在 CUDA 设备上创建随机对数概率张量，形状为 (50, 3, 15)，类型为浮点数，然后进行 log_softmax 处理
                log_probs = torch.randn(50, 3, 15, dtype=torch.float, device='cuda').log_softmax(2)
                # 断言运行时错误，测试在 CUDA 设备上是否正确处理输入长度和目标长度
                with self.assertRaises(RuntimeError):
                    torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)

    # 测试函数，用于验证 CTC 损失函数在 CPU 设备上对长度的检查
    def test_CTCLoss_lengthchecks_cpu(self):
        # 遍历目标长度和输入长度的组合：[[30, 25, 20], [-1, -1, -1]]
        for target_lengths in [[30, 25, 20], [-1, -1, -1]]:
            # 遍历目标长度和输入长度的组合：[[50, 50, 50], [-1, -1, -1]]
            for input_lengths in [[50, 50, 50], [-1, -1, -1]]:
                # 创建随机目标张量，形状为 (3, 29)，类型为整数
                targets = torch.randint(1, 15, (3, 29), dtype=torch.int)
                # 创建随机对数概率张量，形状为 (50, 3, 15)，类型为浮点数，然后进行 log_softmax 处理
                log_probs = torch.randn(50, 3, 15, dtype=torch.float).log_softmax(2)
                # 断言运行时错误，测试在 CPU 设备上是否正确处理输入长度和目标长度
                with self.assertRaises(RuntimeError):
                    torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)

    # 如果 CUDA 不可用，则跳过 CUDA 测试
    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    # 定义一个测试函数，用于测试 CTC loss 对长目标序列的处理
    def test_CTCLoss_long_targets(self):
        # 设置输入长度、词汇表大小、批量大小和目标长度
        input_length = 4000
        vocab_size = 3
        batch_size = 4
        target_length = 1200

        # 生成服从正态分布的对数概率，计算对数 softmax 并标记为需要梯度
        log_probs = torch.randn(input_length, batch_size, vocab_size, dtype=torch.double).log_softmax(2).requires_grad_()
        # 随机生成整数张量作为目标，范围为 [1, vocab_size-1]
        targets = torch.randint(low=1, high=vocab_size - 1, size=(batch_size, target_length), dtype=torch.long)
        # 每个样本的输入长度为 input_length
        input_lengths = batch_size * [input_length]
        # 每个样本的目标长度为 target_length
        target_lengths = batch_size * [target_length]

        # 在 CPU 上计算 CTC loss，设置 reduction 为 'sum'，允许无穷大值
        res_cpu = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                                               reduction='sum', zero_infinity=True)
        # 随机生成与 res_cpu 形状相同的梯度
        grad_out = torch.randn_like(res_cpu)
        # 计算 log_probs 相对于 res_cpu 的梯度
        grad_cpu, = torch.autograd.grad(res_cpu, log_probs, grad_out)

        # 在 GPU 上计算 CTC loss，前提是 CUDA 可用，设置 reduction 为 'sum'，允许无穷大值
        with torch.backends.cudnn.flags(enabled=False):
            res_gpu = torch.nn.functional.ctc_loss(log_probs.cuda(), targets.cuda(), input_lengths, target_lengths,
                                                   reduction='sum', zero_infinity=True)
            # 计算 log_probs 相对于 res_gpu 的梯度
            grad_gpu, = torch.autograd.grad(res_gpu, log_probs, grad_out.cuda())
        # 断言 CPU 和 GPU 计算结果的近似相等性，使用绝对和相对容差
        self.assertEqual(res_cpu, res_gpu, atol=1e-4, rtol=0)
        self.assertEqual(grad_cpu, grad_gpu, atol=1e-4, rtol=0)

    # 如果 CUDA 不可用，则跳过此测试
    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_CTCLoss_critical_target_len(self):
        # 定义参数 N、S、C、T
        N = 1
        S = 256
        C = 10
        T = 500
        # 生成一个长度为 S 的随机整数张量作为目标，范围为 [1, C-1]
        target = torch.randint(low=1, high=C, size=(S,), dtype=torch.int)
        # 创建长度为 N 的整数张量，填充值为 T，作为输入长度
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int)
        # 创建长度为 1 的整数张量，值为 S，作为目标长度
        target_lengths = torch.tensor(S, dtype=torch.int)
        # 在 CUDA 设备上生成服从正态分布的对数概率，计算对数 softmax 并标记为需要梯度
        inp = torch.randn(T, N, C, dtype=torch.float, device='cuda').log_softmax(2).requires_grad_()
        # 使用 cudnn 加速标志，计算在 CUDA 设备上的 CTC loss，设置 reduction 为 'none'
        with cudnn.flags(enabled=True):
            res_gpu = torch.nn.functional.ctc_loss(inp, target, input_lengths, target_lengths, reduction='none')
        # 在 CPU 上计算相同的 CTC loss，设置 reduction 为 'none'
        res_cpu = torch.nn.functional.ctc_loss(inp.cpu(), target, input_lengths, target_lengths, reduction='none')
        # 断言 CPU 和 GPU 计算结果的近似相等性，使用绝对容差
        self.assertEqual(res_cpu, res_gpu, atol=1e-3, rtol=0)
    # 定义一个测试函数，测试在输入长度为零或目标长度为零时的CTC损失函数行为
    def test_CTCLoss_zero_lengths(self):
        # 定义设备列表，包含CPU
        devices = ['cpu']
        # 如果测试CUDA可用，添加'cuda'到设备列表中
        devices += ['cuda'] if TEST_CUDA else []
        # 定义样本数N，序列长度S，类别数C，时间步长T
        N = 3
        S = 2
        C = 200
        T = 1
        # 生成随机的目标张量，形状为(N, S)，数值在1到C之间
        target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.int)
        # 创建输入长度张量，全零，大小为(N,)
        input_lengths = torch.full(size=(N,), fill_value=0, dtype=torch.int)
        # 创建目标长度张量，全零，大小为(N,)
        target_lengths = torch.full(size=(N,), fill_value=0, dtype=torch.int)
        
        # 遍历设备列表
        for device in devices:
            # 在指定设备上生成随机输入张量，形状为(T, N, C)，进行log_softmax，同时要求计算梯度
            inp = torch.randn(T, N, C, dtype=torch.float, device=device).log_softmax(2).requires_grad_()
            # 计算CTC损失，传入输入张量、目标张量、输入长度、目标长度，设置reduction为'none'
            res = torch.nn.functional.ctc_loss(inp, target, input_lengths, target_lengths, reduction='none')
            # 断言所有损失值为零
            self.assertTrue((res == 0).all().item())
            # 对总损失值进行反向传播
            res.sum().backward()
            # 断言输入张量的梯度全为零
            self.assertTrue((inp.grad == 0).all().item())
        
        # 更新目标长度张量为全为1，大小为(N,)
        target_lengths = torch.full(size=(N,), fill_value=1, dtype=torch.int)
        
        # 再次遍历设备列表
        for device in devices:
            # 在指定设备上生成随机输入张量，形状为(T, N, C)，进行log_softmax，同时要求计算梯度
            inp = torch.randn(T, N, C, dtype=torch.float, device=device).log_softmax(2).requires_grad_()
            # 计算CTC损失，传入输入张量、目标张量、输入长度、目标长度，设置reduction为'none'
            res = torch.nn.functional.ctc_loss(inp, target, input_lengths, target_lengths, reduction='none')
            # 断言所有损失值为无穷大
            self.assertTrue((res == torch.inf).all().item())
            # 对总损失值进行反向传播
            res.sum().backward()
            # 断言输入张量的梯度全为零
            self.assertTrue((inp.grad == 0).all().item())

    # 标记为unittest跳过，如果CUDA不可用
    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    # 定义测试函数，测试在零无穷大情况下的CTC损失函数行为
    def test_CTCLoss_zero_infinity(self):
        # 定义目标长度列表
        target_lengths = [60, 25, 20]
        # 定义输入长度列表
        input_lengths = [50, 50, 50]
        # 在CUDA设备上生成随机目标张量，总长度为所有目标长度之和，数值在1到15之间
        targets = torch.randint(1, 15, (sum(target_lengths),), dtype=torch.int, device='cuda')
        # 在CUDA设备上生成随机输入张量，形状为(50, 3, 15)，进行log_softmax，同时要求计算梯度
        log_probs = torch.randn(50, 3, 15, dtype=torch.float, device='cuda').log_softmax(2).requires_grad_()
        
        # 计算CTC损失，传入输入张量、目标张量、输入长度、目标长度，设置reduction为'sum'，允许无穷大为零
        res = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                                           reduction='sum', zero_infinity=True)
        # 禁用cuDNN后再次计算CTC损失，传入输入张量、目标张量、输入长度、目标长度，设置reduction为'sum'，允许无穷大为零
        with torch.backends.cudnn.flags(enabled=False):
            res2 = torch.nn.functional.ctc_loss(log_probs, targets.cuda().long(), input_lengths, target_lengths,
                                                reduction='sum', zero_infinity=True)
        # 在CPU上计算CTC损失，传入输入张量、目标张量、输入长度、目标长度，设置reduction为'sum'，允许无穷大为零
        res_cpu = torch.nn.functional.ctc_loss(log_probs.cpu(), targets.cpu(), input_lengths, target_lengths,
                                               reduction='sum', zero_infinity=True)

        # 断言两种计算方式得到的损失结果近似相等
        self.assertEqual(res2, res, atol=1e-4, rtol=0)
        # 断言CPU计算的损失结果与CUDA计算的CPU结果近似相等
        self.assertEqual(res_cpu, res.cpu(), atol=1e-4, rtol=0)
        # 计算损失相对于输入张量的梯度
        g1, = torch.autograd.grad(res, log_probs)
        g2, = torch.autograd.grad(res2, log_probs)
        g3, = torch.autograd.grad(res_cpu, log_probs)
        # 断言两种方式计算得到的梯度近似相等
        self.assertEqual(g2, g3, atol=1e-4, rtol=0)
        self.assertEqual(g1, g2, atol=1e-4, rtol=0)
        # 断言梯度中没有NaN值
        self.assertTrue((g1 == g1).all().item())  # check that we don't have NaN
    # 定义测试函数 test_RNN_cell_no_broadcasting
    def test_RNN_cell_no_broadcasting(self):
        # 定义内部测试函数 test，用于测试指定的 RNNCell 模块是否会抛出 RuntimeError 异常
        def test(cell_module, input, hx, input_size, hidden_size):
            # 创建指定类型的 RNN cell 对象
            cell = cell_module(input_size, hidden_size)
            # 断言调用 cell(input, hx) 会抛出 RuntimeError 异常
            self.assertRaises(RuntimeError, lambda: cell(input, hx))

        # 定义内部测试函数 test_all，测试多种 RNN cell 模块在给定的条件下是否会抛出 RuntimeError 异常
        def test_all(hidden_size, bad_hx, good_hx, input_size, input):
            # 测试 nn.RNNCell 模块
            test(nn.RNNCell, input, bad_hx, input_size, hidden_size)
            # 测试 nn.GRUCell 模块
            test(nn.GRUCell, input, bad_hx, input_size, hidden_size)
            # 测试 nn.LSTMCell 模块，使用 (bad_hx, good_hx) 作为隐藏状态
            test(nn.LSTMCell, input, (bad_hx, good_hx), input_size, hidden_size)
            # 测试 nn.LSTMCell 模块，使用 (good_hx, bad_hx) 作为隐藏状态
            test(nn.LSTMCell, input, (good_hx, bad_hx), input_size, hidden_size)

        # 设置隐藏状态和输入大小
        hidden_size = 20
        input_size = 10
        # 生成随机输入数据和隐藏状态
        input = torch.randn(3, input_size)
        bad_hx = torch.randn(1, hidden_size)
        good_hx = torch.randn(3, hidden_size)

        # 测试隐藏状态和输入的批处理大小广播
        test_all(hidden_size, bad_hx, good_hx, input_size, input)

        # 测试隐藏状态的 hidden_size 与模块的 hidden_size 的广播
        bad_hx = torch.randn(3, 1)
        test_all(hidden_size, bad_hx, good_hx, input_size, input)

        # 测试输入的 input_size 与模块的 input_size 的广播
        bad_input = torch.randn(3, 1)
        test_all(hidden_size, good_hx, good_hx, input_size, bad_input)

    # 定义测试函数 test_LSTM_cell
    def test_LSTM_cell(self):
        # 这只是一个简单的烟雾测试；这些模块已经通过 autograd 实现，因此不需要雅可比测试
        for bias in (True, False):
            # 生成随机输入数据和隐藏状态
            input = torch.randn(3, 10)
            hx = torch.randn(3, 20)
            cx = torch.randn(3, 20)
            # 创建 LSTMCell 模块
            lstm = nn.LSTMCell(10, 20, bias=bias)
            # 多次调用 LSTMCell，并更新隐藏状态和细胞状态
            for _ in range(6):
                hx, cx = lstm(input, (hx, cx))

            # 对 (hx + cx) 求和并进行反向传播
            (hx + cx).sum().backward()

    # 定义测试函数 test_LSTM_cell_forward_input_size
    def test_LSTM_cell_forward_input_size(self):
        # 生成随机输入数据和隐藏状态
        input = torch.randn(3, 11)
        hx = torch.randn(3, 20)
        cx = torch.randn(3, 20)
        # 创建 LSTMCell 模块
        lstm = nn.LSTMCell(10, 20)
        # 断言调用 lstm(input, (hx, cx)) 会抛出异常
        self.assertRaises(Exception, lambda: lstm(input, (hx, cx)))

    # 定义测试函数 test_LSTM_cell_forward_hidden_size
    def test_LSTM_cell_forward_hidden_size(self):
        # 生成随机输入数据和隐藏状态
        input = torch.randn(3, 10)
        hx = torch.randn(3, 21)
        cx = torch.randn(3, 20)
        # 创建 LSTMCell 模块
        lstm = nn.LSTMCell(10, 20)
        # 断言调用 lstm(input, (hx, cx)) 会抛出异常
        self.assertRaises(Exception, lambda: lstm(input, (hx, cx)))
        # 断言调用 lstm(input, (cx, hx)) 会抛出异常
        self.assertRaises(Exception, lambda: lstm(input, (cx, hx)))

    # 根据 CUDA 是否可用选择性跳过测试函数 test_pack_sequence_batch_sizes_throw
    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_pack_sequence_batch_sizes_throw(self):
        # 使用 assertRaisesRegex 断言捕获 ValueError 异常，并检查错误消息中是否包含特定文本
        with self.assertRaisesRegex(ValueError, r"batch_sizes should always be on CPU"):
            # 将 LSTM 模块移至 CUDA 设备
            m = nn.LSTM(3, 4, bidirectional=True, num_layers=2).to('cuda')
            # 创建输入数据和序列长度张量，并封装成 PackedSequence 对象
            a = torch.rand(5, 3, device='cuda')
            b = torch.tensor([1, 1, 1, 1, 1], device='cuda')
            input = nn.utils.rnn.PackedSequence(a, b)
    # 定义一个名为 test_Transformer_cell 的测试函数
    def test_Transformer_cell(self):
        # 这只是一个简单的烟雾测试；这些模块通过自动求导实现，因此不需要雅可比测试
        d_model = 512
        nhead = 16
        num_encoder_layers = 4
        num_decoder_layers = 3
        dim_feedforward = 256
        dropout = 0.3
        bsz = 8
        seq_length = 35
        tgt_length = 15
        
        # 遍历两种不同的数据格式
        for batch_first, src_size, tgt_size in zip((True, False),
                                                   [(bsz, seq_length, d_model),
                                                    (seq_length, bsz, d_model)],
                                                   [(bsz, tgt_length, d_model),
                                                    (tgt_length, bsz, d_model)]):
            # 创建一个 Transformer 模型对象
            transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                         dim_feedforward, dropout, batch_first=batch_first,
                                         dtype=torch.double)
            # 生成随机的输入数据 src 和 tgt
            src = torch.randn(src_size, dtype=torch.double)
            # 生成用于源序列的掩码
            src_mask = transformer.generate_square_subsequent_mask(seq_length).double()
            # 生成随机的目标数据 tgt
            tgt = torch.randn(tgt_size, dtype=torch.double)
            # 生成用于目标序列的掩码
            tgt_mask = transformer.generate_square_subsequent_mask(tgt_length).double()
            # 生成随机的记忆掩码 memory_mask
            memory_mask = torch.randn(tgt_length, seq_length).double()
            # 生成随机的源序列键掩码 src_key_padding_mask
            src_key_padding_mask = torch.rand(bsz, seq_length) >= 0.5
            # 生成随机的目标序列键掩码 tgt_key_padding_mask
            tgt_key_padding_mask = torch.rand(bsz, tgt_length) >= 0.5
            # 生成随机的记忆键掩码 memory_key_padding_mask
            memory_key_padding_mask = torch.rand(bsz, seq_length) >= 0.5

            # 使用 Transformer 模型进行前向传播计算
            output = transformer(src, tgt,
                                 src_mask=src_mask,
                                 tgt_mask=tgt_mask,
                                 memory_mask=memory_mask,
                                 src_key_padding_mask=src_key_padding_mask,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask)
            # 对输出结果进行求和并进行反向传播
            output.sum().backward()

    # 设置默认的张量类型为 torch.double，并且标记为跳过测试如果不满足条件（CUDNN 和多GPU可用）
    @set_default_dtype(torch.double)
    @unittest.skipIf(not (TEST_CUDNN and TEST_MULTIGPU), 'CUDNN or multi-gpu not available')
    # 定义一个名为 test_cudnn_rnn_dropout_states_device 的测试函数
    def test_cudnn_rnn_dropout_states_device(self):
        # 创建一个具有 dropout 的双层 RNN 模型
        rnn = nn.RNN(10, 20, num_layers=2, dropout=.5)
        device = 1
        # 生成随机输入张量并将其移到指定设备上
        input = torch.randn(5, 4, 10).cuda(device)
        # 将 RNN 模型也移到指定设备上
        rnn.cuda(device)
        # 生成随机的隐藏状态张量 hx 并将其移到指定设备上
        hx = torch.randn(2, 4, 20).cuda(device)
        # 使用输入和隐藏状态进行 RNN 模型的前向传播计算
        output = rnn(input, hx)
    # 定义一个测试方法，用于测试 cuDNN 前向传播时的异常情况处理
    def test_cudnn_forward_exception(self):
        # 定义一组 RNN 模型和对应的输入数据，每个元组包含一个 RNN 模型和输入数据的元组或单个张量
        rnns = [
            (nn.LSTM(10, 20, batch_first=True), (torch.zeros(1, 2, 19), torch.zeros(1, 2, 19))),
            (nn.LSTM(10, 20, batch_first=True, proj_size=10), (torch.zeros(1, 2, 19), torch.zeros(1, 2, 19))),
            (nn.GRU(10, 20, batch_first=True), torch.zeros(1, 2, 19)),
            (nn.RNN(10, 20, batch_first=True), torch.zeros(1, 2, 19)),
        ]
        # 创建一个不符合输入尺寸的张量作为错误的输入数据
        x_wrong = torch.randn(2, 3, 3)
        # 创建一个符合输入尺寸的张量作为正确的输入数据
        x_right = torch.randn(2, 3, 10)
        # 遍历每个 RNN 模型及其对应的隐藏状态
        for rnn, hidden in rnns:
            # 断言捕获到 RuntimeError，并匹配特定的错误消息来测试正确的输入数据情况
            self.assertRaisesRegex(RuntimeError, "Expected hidden.*size.*got", rnn, x_right, hidden)
            # 断言捕获到 RuntimeError，并匹配特定的错误消息来测试错误的输入数据情况
            self.assertRaisesRegex(RuntimeError, re.escape("input.size(-1) must be equal to input_size"), rnn, x_wrong)

    # 如果不支持 cuDNN 测试，则跳过该测试
    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    @skipIfRocm
    # 定义一个测试方法，用于测试不同的 RNN 模型在 CUDA 环境下的权重格式
    def test_cudnn_weight_format(self):
        # 定义几个 RNN 模型，包括不同的 LSTM 变种和 GRU、RNN
        rnns = [
            nn.LSTM(10, 20, batch_first=True),  # 创建一个 LSTM 模型
            nn.LSTM(10, 20, batch_first=True, proj_size=10),  # 创建一个带投影的 LSTM 模型
            nn.GRU(10, 20, batch_first=True),  # 创建一个 GRU 模型
            nn.RNN(10, 20, batch_first=True)   # 创建一个 RNN 模型
        ]
        # 初始化一个布尔变量，用于记录是否是第一次出现警告
        first_warn = True
        # 遍历每个 RNN 模型
        for rnn in rnns:
            # 将当前 RNN 模型移动到 CUDA 上执行
            rnn.cuda()
            # 创建一个随机输入张量，也移动到 CUDA 上
            input = torch.randn(5, 4, 10, requires_grad=True, device="cuda")
            # 创建一个随机隐藏状态张量 hx，也移动到 CUDA 上
            hx = torch.randn(1, 5, 20, requires_grad=True, device="cuda")
            # 将当前 RNN 模型的所有参数（包括权重和偏置等）加入到一个列表中
            all_vars = [input, hx] + list(rnn.parameters())
            # 如果当前 RNN 模型是 LSTM 并且带有投影
            if isinstance(rnn, nn.LSTM):
                # LSTM 模型带投影时，隐藏状态 hx 的大小不同
                if rnn.proj_size > 0:
                    hx = torch.randn(1, 5, 10, requires_grad=True, device="cuda")
                    all_vars[1] = hx
                # 创建一个随机细胞状态张量 cx，也移动到 CUDA 上
                cx = torch.randn(1, 5, 20, requires_grad=True, device="cuda")
                # 将细胞状态加入到参数列表的适当位置
                all_vars[2:2] = [cx]
                # 将隐藏状态和细胞状态组合成一个元组，作为 LSTM 的输入
                hx = (hx, cx)

            # 使用当前 RNN 模型处理输入 input 和隐藏状态 hx，得到输出 output
            output = rnn(input, hx)
            # 对输出的第一个元素求和并进行反向传播
            output[0].sum().backward()
            # 复制所有参数的梯度数据到 grads 列表中
            grads = [v.grad.data.clone() for v in all_vars]
            # 将所有参数的梯度数据清零
            for v in all_vars:
                v.grad.data.zero_()

            # 权重将不再视图同一块内存的块
            # 获取权重变量
            weight = all_vars[4]
            # 复制权重的数据
            weight_data = weight.data.clone()
            # 使用 torch.no_grad() 块，设置权重为其数据的副本
            with torch.no_grad():
                weight.set_(weight_data)

            # 对于两次循环
            for _ in range(2):
                # 捕获警告信息
                with warnings.catch_warnings(record=True) as w:
                    # 使用当前 RNN 模型处理输入 input 和隐藏状态 hx，得到非连续的输出 output_noncontig
                    output_noncontig = rnn(input, hx)
                # 如果是第一次警告，检查是否有一条警告信息
                if first_warn:
                    self.assertEqual(len(w), 1)
                    # 检查警告信息中是否包含特定的消息内容
                    self.assertIn('weights are not part of single contiguous chunk of memory', w[0].message.args[0])
                    first_warn = False
                    # 重置警告状态
                    warnings.resetwarnings()
                # 对非连续输出的第一个元素求和并进行反向传播
                output_noncontig[0].sum().backward()
                # 复制非连续情况下所有参数的梯度数据到 grads_noncontig 列表中
                grads_noncontig = [v.grad.data.clone() for v in all_vars]
                # 将所有参数的梯度数据清零
                for v in all_vars:
                    v.grad.data.zero_()
                # 检查连续输出和非连续输出是否相等
                self.assertEqual(output, output_noncontig)
                # 检查非连续情况下的梯度数据和连续情况下的梯度数据是否相等
                self.assertEqual(grads_noncontig, grads)

            # 确保这些仍然共享存储
            # 将权重数据设置为统一值
            weight_data[:] = 4
            # 检查权重数据是否与 all_vars 中的权重数据相等
            self.assertEqual(weight_data, all_vars[4].data)

    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_cudnn_weight_tying(self):
        # 定义四种不同类型的循环神经网络模型
        rnns = [
            nn.LSTM(10, 20, batch_first=True, bidirectional=True),
            nn.LSTM(10, 20, batch_first=True, bidirectional=True, proj_size=10),
            nn.GRU(10, 20, batch_first=True, bidirectional=True),
            nn.RNN(10, 20, batch_first=True, bidirectional=True)
        ]
        # 遍历每种循环神经网络模型
        for rnn in rnns:
            # 将反向传播偏置项设置为正向传播偏置项的副本
            rnn.bias_ih_l0_reverse = rnn.bias_ih_l0
            # 将模型移动到 CUDA 设备
            rnn.cuda()
            # 创建输入张量，形状为 (5, 4, 10)，在 CUDA 设备上
            input = torch.randn(5, 4, 10, requires_grad=True, device="cuda")
            # 创建初始隐藏状态张量 hx，形状为 (2, 5, 20)，在 CUDA 设备上
            hx = torch.randn(2, 5, 20, requires_grad=True, device="cuda")
            # 将所有需要优化的变量收集到列表 all_vars 中
            all_vars = [input, hx] + list(rnn.parameters())
            # 使用 SGD 优化器，学习率为 0.1
            opt = torch.optim.SGD(rnn.parameters(), lr=0.1)
            # 将优化器梯度清零
            opt.zero_grad()
            # 对于 LSTM 模型，根据是否有投影层选择不同的初始隐藏状态结构
            if isinstance(rnn, nn.LSTM):
                if rnn.proj_size > 0:
                    # 如果有投影层，重新创建 hx 张量
                    hx = torch.randn(2, 5, 10, requires_grad=True, device="cuda")
                    all_vars[1] = hx
                # 创建细胞状态张量 cx，形状为 (2, 5, 20)，在 CUDA 设备上
                cx = torch.randn(2, 5, 20, requires_grad=True, device="cuda")
                # 将 cx 插入到 all_vars 列表的正确位置
                all_vars[2:2] = [cx]
                # 更新 hx 为元组 (hx, cx)
                hx = (hx, cx)

            # 捕获可能的警告信息
            with warnings.catch_warnings(record=True) as w:
                # 执行模型前向传播
                output = rnn(input, hx)
            # 对输出张量的所有元素求和并执行反向传播
            output[0].sum().backward()

            # 执行优化步骤
            opt.step()
            # 再次执行模型前向传播，捕获可能的警告信息
            with warnings.catch_warnings(record=True) as w:
                output_cuda = rnn(input, hx)
            # 将模型移动回 CPU
            rnn.cpu()
            # 根据模型类型调整隐藏状态 hx 的设备位置
            hx = (hx[0].cpu(), hx[1].cpu()) if isinstance(rnn, nn.LSTM) else hx.cpu()
            # 在 CPU 上执行模型前向传播
            output_cpu = rnn(input.cpu(), hx)
            # 使用断言检查 CUDA 和 CPU 上的输出是否相等
            self.assertEqual(output_cuda, output_cpu)


    def test_transformer_layer_args_check(self):
        # 定义需要测试的 Transformer 模型类名列表
        model_names = ['TransformerEncoderLayer', 'TransformerDecoderLayer']
        d_model = 128
        nhead = 4
        dim_feedforward = 65
        dropout = 0.3
        bsz = 3
        seq_len = 35
        tgt_len = 15
        # 定义激活函数列表
        activations = [F.relu, F.gelu]

        # 不正确的激活函数字符串
        wrong_activation = "abc"

        # 定义编码器和解码器输入张量的形状
        encoder_input_shape = (seq_len, bsz, d_model)
        decoder_input_shape = (tgt_len, bsz, d_model)

        # 创建编码器和解码器输入张量
        encoder_input = torch.randn(encoder_input_shape)
        decoder_input = torch.randn(decoder_input_shape)

        # 遍历 Transformer 模型类名列表和激活函数列表
        for model_name in model_names:
            for activation in activations:
                # 根据模型名和参数创建 Transformer 模型对象
                model = getattr(nn, model_name)(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        # 使用不正确的激活函数字符串创建 Transformer 模型对象，并期望引发 RuntimeError 异常
        for model_name in model_names:
            with self.assertRaises(RuntimeError):
                model = getattr(nn, model_name)(d_model, nhead, dim_feedforward,
                                                dropout, wrong_activation)
    def test_rnn_args_check(self):
        # 定义各种输入参数的大小
        input_size = 3
        hidden_size = 5
        num_layers = 2
        batch_size = 4
        seq_len = 6
        num_directions = 1
        bad_size = 7  # 质数，无法被其他大小整除

        def test(input_shape, hidden_shape, mode):
            # 对于给定的输入形状和隐藏状态形状，获取模型输入和隐藏状态的生成器
            for input, hidden in get_inputs(input_shape, hidden_shape, mode):
                # 根据模式创建相应的循环神经网络模型对象
                model = getattr(nn, mode)(input_size, hidden_size, num_layers)
                # 断言模型调用时会触发 RuntimeError 异常
                self.assertRaises(RuntimeError, lambda: model(input, hidden))

        correct_input_shape = (seq_len, batch_size, input_size)
        correct_hidden_shape = (num_layers * num_directions, batch_size, hidden_size)

        def update_shape(shape, dim, new_dim_size):
            # 更新形状的指定维度为新的维度大小
            new_shape = list(shape)
            new_shape[dim] = new_dim_size
            return tuple(new_shape)

        def get_inputs(input_shape, hidden_shape, mode):
            '''返回一个列表，包含元组(input, hidden)，这些元组是模型的输入和隐藏状态'''
            input = torch.randn(input_shape)
            hidden = torch.randn(hidden_shape)
            if mode != 'LSTM':
                return [(input, hidden)]
            if hidden_shape == correct_hidden_shape:
                return [(input, (hidden, hidden))]
            good_hidden = torch.randn(correct_hidden_shape)
            return [
                (input, (hidden, good_hidden)),
                (input, (good_hidden, hidden)),
            ]

        rnn_modes = ['RNN', 'GRU', 'LSTM']
        for mode in rnn_modes:
            # 测试不正确的输入批处理大小
            input_shape = update_shape(correct_input_shape, 1, bad_size)
            hidden_shape = correct_hidden_shape
            test(input_shape, hidden_shape, mode)

            # 测试不正确的隐藏状态批处理大小
            input_shape = correct_input_shape
            hidden_shape = update_shape(correct_hidden_shape, 1, bad_size)
            test(input_shape, hidden_shape, mode)

            # 测试不正确的输入大小
            input_shape = update_shape(correct_input_shape, 2, bad_size)
            hidden_shape = correct_hidden_shape
            test(input_shape, hidden_shape, mode)

            # 测试不正确的隐藏状态大小
            input_shape = correct_input_shape
            hidden_shape = update_shape(correct_hidden_shape, 2, bad_size)
            test(input_shape, hidden_shape, mode)

            # 测试不正确的隐藏状态第一个元素大小
            input_shape = correct_input_shape
            hidden_shape = update_shape(correct_hidden_shape, 0, bad_size)
            test(input_shape, hidden_shape, mode)
    # 定义一个名为 test_rnn_check_device 的测试方法
    def test_rnn_check_device(self):
        # 导入 copy 模块，用于深拷贝模型
        import copy
        # 设置 RNN 模型的输入大小、隐藏层大小、层数、批次大小和序列长度
        input_size = 3
        hidden_size = 5
        num_layers = 2
        batch_size = 4
        seq_len = 6
        num_directions = 1

        # 期望的输入张量形状和隐藏状态张量形状
        correct_input_shape = (seq_len, batch_size, input_size)
        correct_hidden_shape = (num_layers * num_directions, batch_size, hidden_size)
        # RNN 模型的三种模式：标准 RNN、GRU 和 LSTM
        rnn_modes = ['RNN', 'GRU', 'LSTM']

        # 遍历三种模式
        for mode in rnn_modes:
            # 动态创建对应模式的 RNN 模型
            model = getattr(nn, mode)(input_size, hidden_size, num_layers)
            # 深拷贝模型并将其移动到 CUDA 设备上
            model_cuda = copy.deepcopy(model).to('cuda:0')
            # 创建符合期望形状的随机输入张量和隐藏状态张量
            input = torch.randn(correct_input_shape)
            hidden = torch.randn(correct_hidden_shape)

            # 断言错误：输入张量和模型参数张量不在同一设备上
            with self.assertRaisesRegex(RuntimeError,
                                        "Input and parameter tensors are not at the same device"):
                model(input.to('cuda:0'))
            with self.assertRaisesRegex(RuntimeError,
                                        "Input and parameter tensors are not at the same device"):
                model_cuda(input)

            # 断言错误：输入张量和隐藏状态张量不在同一设备上
            with self.assertRaisesRegex(RuntimeError,
                                        r"Input and hidden tensors are not at the same device"):
                if mode == 'LSTM':
                    model(input, (hidden.to('cuda:0'), hidden.to('cuda:0')))
                else:
                    model(input, (hidden.to('cuda:0')))
            with self.assertRaisesRegex(RuntimeError,
                                        r"Input and hidden tensors are not at the same device"):
                if mode == 'LSTM':
                    model_cuda(input.to('cuda:0'), (hidden, hidden))
                else:
                    model_cuda(input.to('cuda:0'), (hidden))

            # 断言错误：隐藏状态张量不在同一 CUDA 设备上
            if mode == 'LSTM':
                with self.assertRaisesRegex(RuntimeError,
                                            "Input and hidden tensors are not at the same device"):
                    model(input.to('cuda:0'), (hidden.to('cuda:0'), hidden.to('cuda:1')))

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    # 测试 LSTM 模型的设备匹配情况
    def test_projections_lstm_check_device(self):
        # 定义输入大小、隐藏状态大小、投影大小、层数、批次大小和序列长度等参数
        input_size = 3
        hidden_size = 5
        proj_size = 2
        num_layers = 2
        batch_size = 4
        seq_len = 6
        num_directions = 1

        # 预期的输入张量形状和隐藏状态张量形状
        correct_input_shape = (seq_len, batch_size, input_size)
        correct_hidden_h_shape = (num_layers * num_directions, batch_size, proj_size)
        correct_hidden_c_shape = (num_layers * num_directions, batch_size, hidden_size)

        # 创建 LSTM 模型对象，指定输入大小、隐藏状态大小、层数和投影大小
        model = nn.LSTM(input_size, hidden_size, num_layers, proj_size=proj_size)
        # 生成符合预期形状的随机输入张量和隐藏状态张量
        input = torch.randn(correct_input_shape)
        hidden_h = torch.randn(correct_hidden_h_shape)
        hidden_c = torch.randn(correct_hidden_c_shape)

        # 检查输入张量和模型参数张量是否在同一个设备上，并抛出匹配错误
        with self.assertRaisesRegex(RuntimeError,
                                    "Input and parameter tensors are not at the same device"):
            model(input.to('cuda:0'))

        # 检查输入张量和隐藏状态张量是否在同一个设备上，并抛出匹配错误
        with self.assertRaisesRegex(RuntimeError,
                                    r"Input and hidden tensors are not at the same device"):
            model(input, (hidden_h.to('cuda:0'), hidden_c.to('cuda:0')))

        # 检查隐藏状态张量的两部分是否在同一个 CUDA 设备上，并抛出匹配错误
        with self.assertRaisesRegex(RuntimeError,
                                    "Input and hidden tensors are not at the same device"):
            model(input.to('cuda:0'), (hidden_h.to('cuda:0'), hidden_c.to('cuda:1')))

    # 测试 RNN 初始隐藏状态
    def test_rnn_initial_hidden_state(self):
        # 支持的 RNN 模型：RNN、GRU、LSTM
        rnn_modes = ['RNN', 'GRU', 'LSTM']
        for mode in rnn_modes:
            # 根据模式获取对应的 RNN 类，并初始化模型对象
            rnn = getattr(nn, mode)(30, 20, 2)
            # 生成符合预期形状的随机输入张量和全零隐藏状态张量
            input = torch.randn(10, 32, 30)
            hidden = torch.zeros(2, 32, 20)

            # 如果是 LSTM 模式，隐藏状态使用两个相同的全零张量元组
            if mode == 'LSTM':
                hidden = (hidden, hidden)
            
            # 运行 RNN 模型，并获取输出和新的隐藏状态
            output1, hidden1 = rnn(input, hidden)
            # 重新运行 RNN 模型，获取另一组输出和隐藏状态
            output2, hidden2 = rnn(input)
            
            # 断言两次运行的输出和隐藏状态是否相等
            self.assertEqual(output1, output2)
            self.assertEqual(hidden1, hidden2)

    # 测试带投影的 LSTM 初始隐藏状态
    def test_projections_lstm_initial_hidden_state(self):
        # 遍历是否双向的标志位列表：False 表示单向，True 表示双向
        for bidir in [False, True]:
            # 创建带投影的 LSTM 模型对象，根据是否双向设定输出通道数
            rnn = nn.LSTM(30, 20, 2, bidirectional=bidir, proj_size=10)
            # 根据是否双向设定隐藏状态张量的通道数
            num_dirs = 2 if bidir else 1
            # 生成符合预期形状的随机输入张量和全零隐藏状态张量（包含投影部分）
            input = torch.randn(10, 32, 30)
            hidden_h = torch.zeros(2 * num_dirs, 32, 10)
            hidden_c = torch.zeros(2 * num_dirs, 32, 20)
            hidden = (hidden_h, hidden_c)
            
            # 运行带投影的 LSTM 模型，并获取输出和新的隐藏状态
            output1, hidden1 = rnn(input, hidden)
            # 重新运行带投影的 LSTM 模型，获取另一组输出和隐藏状态
            output2, hidden2 = rnn(input)
            
            # 断言两次运行的输出和隐藏状态是否相等
            self.assertEqual(output1, output2)
            self.assertEqual(hidden1, hidden2)

    # 测试在 RNN 和 GRU 上使用投影参数是否引发错误
    def test_projections_errors_on_gru_and_rnn(self):
        # 错误消息：proj_size 参数仅支持 LSTM 模型，不支持 RNN 或 GRU
        error_msg = "proj_size argument is only supported for LSTM, not RNN or GRU"
        # 遍历 RNN 和 GRU 两种模型
        for mode in ['RNN', 'GRU']:
            # 使用 getattr 根据字符串名称获取对应的 RNN 或 GRU 模型类
            with self.assertRaisesRegex(ValueError, error_msg):
                rnn = getattr(nn, mode)(30, 20, 2, proj_size=10)

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    def test_RNN_cpu_vs_cudnn_no_dropout(self):
        # 定义数据类型为双精度的 Torch 张量
        dtype = torch.double
        # 调用 _test_RNN_cpu_vs_cudnn 函数进行测试，使用 CPU
        self._test_RNN_cpu_vs_cudnn(0, dtype)

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    def test_RNN_cpu_vs_cudnn_with_dropout(self):
        # 由于 dropout 的随机性，只能比较 dropout=0 和 dropout=1 的情况
        self._test_RNN_cpu_vs_cudnn(1)

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    def test_RNN_cudnn_weight_norm(self):
        # 设置输入大小、隐藏层大小、层数、序列长度和批处理大小
        input_size = 10
        hidden_size = 6
        num_layers = 2
        seq_length = 7
        batch = 6

        # 在 CPU 上运行以获取预期输出
        def check_weight_norm(m, name):
            # 创建随机输入张量
            input = torch.randn(seq_length, batch, input_size)
            expected_output = m(input)

            # 添加权重归一化
            m = torch.nn.utils.weight_norm(m, name=name)

            # 将模型转移到 CUDA
            m = m.cuda()
            input = input.cuda()

            # 否则，后续的警告将被隐藏，进一步的测试依赖于它们
            warnings.simplefilter("always")
            # 断言模型在 CUDA 上的输出与预期输出相等
            self.assertEqual(m(input), expected_output)

            # 移除权重归一化
            m = torch.nn.utils.remove_weight_norm(m, name=name)
            # 断言移除权重归一化后模型的输出与预期输出相等
            self.assertEqual(m(input), expected_output)

        # 检查 LSTM 模型的权重归一化
        check_weight_norm(nn.LSTM(input_size, hidden_size, num_layers), 'weight_hh_l0')
        check_weight_norm(nn.LSTM(input_size, hidden_size, num_layers, proj_size=3), 'weight_hr_l0')

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_partial_flat_weights(self):
        # 设置输入大小、隐藏层大小和层数
        input_size = 10
        hidden_size = 6
        num_layers = 2

        # 创建 LSTM 模型
        m = nn.LSTM(input_size, hidden_size, num_layers)
        # 创建随机输入张量
        inp = torch.randn(3, 2, 10)
        # 获取预期输出
        out_expected = m(inp)
        
        # 删除原始 LSTM 的一个属性
        weight_orig = m.weight_hh_l0
        del m.weight_hh_l0
        # 断言原始 LSTM 不再有 'weight_hh_l0' 属性
        self.assertFalse(hasattr(m, "weight_hh_l0"))
        
        # 将模型转移到 CUDA，验证只有部分属性定义的情况下不会抛出错误
        m.cuda()
        # 重新计算权重，并确保模块可以使用
        m.weight_hh_l0 = weight_orig.cuda()
        inp = inp.cuda()
        
        # 否则，后续的警告将被隐藏，进一步的测试依赖于它们
        warnings.simplefilter("always")
        # 断言模型在 CUDA 上的输出与预期输出相等
        self.assertEqual(m(inp)[0].cpu(), out_expected[0])

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    @set_default_dtype(torch.double)
    def test_RNN_dropout(self):
        # 检查 cuDNN 是否将 dropout 放置在 RNN 层之间的假设
        for p in (0, 0.276, 0.731, 1):
            for train in (True, False):
                for cuda in (True, False):
                    # 创建一个具有特定参数的 RNN 模型
                    rnn = nn.RNN(10, 1000, 2, bias=False, dropout=p, nonlinearity='relu')
                    if cuda:
                        rnn.cuda()

                    if train:
                        rnn.train()
                    else:
                        rnn.eval()
                    # 初始化 RNN 模型的权重
                    rnn.weight_ih_l0.data.fill_(1)
                    rnn.weight_hh_l0.data.fill_(1)
                    rnn.weight_ih_l1.data.fill_(1)
                    rnn.weight_hh_l1.data.fill_(1)
                    # 创建输入和隐藏状态
                    input = torch.ones(1, 1, 10)
                    hx = torch.zeros(2, 1, 1000)
                    if cuda:
                        input = input.cuda()
                        hx = hx.cuda()

                    # 运行 RNN 模型
                    output, hy = rnn(input, hx)
                    # 断言输出张量的最小值和最大值相等，以确保输出一致
                    self.assertEqual(output.data.min(), output.data.max())
                    output_val = output.data[0][0][0]
                    # 根据 dropout 的不同取值，进行不同的输出值断言
                    if p == 0 or not train:
                        self.assertEqual(output_val, 10000)
                    elif p == 1:
                        self.assertEqual(output_val, 0)
                    else:
                        self.assertGreater(output_val, 8000)
                        self.assertLess(output_val, 12000)
                        # 针对输出值进行归一化修正的检查
                        denorm_mod = (output_val * (1 - p)) % 10
                        self.assertLess(min(denorm_mod, 10 - denorm_mod), 1e-2)

                    # 断言隐藏状态张量的最小值和最大值相等
                    self.assertEqual(hy[0].data.min(), hy[0].data.max())
                    self.assertEqual(hy[1].data.min(), hy[1].data.max())
                    # 断言隐藏状态的特定值
                    self.assertEqual(hy.data[0][0][0], 10)
                    self.assertEqual(hy.data[1][0][0], output_val)

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    @set_default_dtype(torch.double)
    def test_error_RNN_seq_len_zero(self):
        # 检查当 RNN 的 seq_len = 0 时的错误消息
        for module in (nn.RNN, nn.LSTM, nn.GRU):
            for bidirectional in [True, False]:
                for device in get_all_device_types():
                    # 创建一个输入张量，seq_len = 0
                    input = torch.ones(0, 10, 5)
                    # 创建特定类型的 RNN 模型
                    rnn = module(5, 6, bidirectional=bidirectional)
                    if device == 'cuda':
                        rnn.cuda()
                        input = input.cuda()

                    # 使用断言捕获期望的运行时错误消息
                    with self.assertRaisesRegex(RuntimeError, "Expected sequence length to be larger than 0 in RNN"):
                        rnn(input)
    # 定义测试函数，测试输入序列长度为零的情况
    def test_RNN_input_size_zero(self):
        # 遍历三种循环神经网络模型：RNN、LSTM、GRU
        for module in (nn.RNN, nn.LSTM, nn.GRU):
            # 遍历所有设备类型（包括 CPU 和 CUDA）
            for device in get_all_device_types():
                # 创建一个形状为 (5, 0, 3) 的全零输入张量
                input = torch.zeros((5, 0, 3))
                # 根据当前循环神经网络模型创建 RNN 对象，输入大小为 3，隐藏状态大小为 4
                rnn = module(input_size=3, hidden_size=4)
                # 如果当前设备为 CUDA，将输入张量和 RNN 对象移动到 CUDA 上
                if device == 'cuda':
                    rnn.cuda()
                    input = input.cuda()
                # 将输入张量输入到 RNN 中，获取输出
                outs = rnn(input)
                # 断言输出的第一个元素的形状为 [5, 0, 4]
                self.assertEqual(outs[0].shape, torch.Size([5, 0, 4]))
                # 检查反向传播不会导致严重错误
                outs[0].sum().backward()

    # 如果不支持 cudnn，则跳过该测试用例
    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    # 定义测试函数，测试 RNN 的 dropout 状态
    def test_RNN_dropout_state(self):
        # 遍历 dropout 概率为 0 和 0.1234 两种情况
        for p in (0, 0.1234):
            # 遍历训练状态为 True 和 False 两种情况
            for train in (True, False):
                # 遍历 CUDA 是否可用的两种情况
                for cuda in (True, False):
                    # 创建一个具有指定参数的 RNN 对象
                    rnn = nn.RNN(100, 100, 2, bias=False, dropout=p, nonlinearity='relu')
                    # 如果 CUDA 可用，将 RNN 对象移动到 CUDA 上
                    if cuda:
                        rnn.cuda()

                    # 设置 RNN 对象的训练状态
                    if train:
                        rnn.train()
                    else:
                        rnn.eval()
                    
                    # 创建随机输入和隐藏状态张量
                    input = torch.rand(1, 1, 100)
                    hx = torch.rand(2, 1, 100)
                    
                    # 如果 CUDA 可用，将输入张量和隐藏状态张量移动到 CUDA 上
                    if cuda:
                        input = input.cuda()
                        hx = hx.cuda()

                    # 调用 RNN 对象，获取输出和新的隐藏状态
                    output1, hy1 = rnn(input, hx)
                    output2, hy2 = rnn(input, hx)

                    # 将 RNN 对象序列化后，再次调用并获取输出和隐藏状态
                    buf = io.BytesIO()
                    rnn_pickle = torch.save(rnn, buf)
                    buf.seek(0)
                    rnn2 = torch.load(buf)
                    rnn2.flatten_parameters()
                    output3, hy3 = rnn2(input, hx)

                    # 根据 dropout 概率是否为 0 或训练状态是否为 False，进行断言
                    if p == 0 or not train:
                        self.assertEqual(output1, output2)
                        self.assertEqual(output1, output3)
                        self.assertEqual(hy1, hy2)
                        self.assertEqual(hy1, hy3)
                    else:
                        self.assertNotEqual(output1, output2)
                        self.assertNotEqual(output1, output3)
                        self.assertNotEqual(hy1, hy2)
                        self.assertNotEqual(hy1, hy3)
    # 定义测试函数，测试在修改RNN的dropout时的行为
    def test_RNN_change_dropout(self):
        # 对train和cuda两个参数进行组合，每个参数取True和False各一次
        for train, cuda in product((True, False), repeat=2):
            # 创建一个具有100个输入特征，100个隐藏单元，2层的RNN模型，dropout为0，激活函数为ReLU
            rnn = nn.RNN(100, 100, 2, dropout=0, nonlinearity='relu')
            # 创建一个形状为(3, 2, 100)的随机输入张量
            input = torch.rand(3, 2, 100)
            # 如果cuda为True，则将输入数据移动到GPU上，并将RNN模型也移动到GPU上
            if cuda:
                input.data = input.data.cuda()
                rnn.cuda()

            # 如果train为True，则设置RNN模型为训练模式，否则设置为评估模式
            if train:
                rnn.train()
            else:
                rnn.eval()

            # 初始化上一个输出为None
            prev_output = None
            # 遍历不同的dropout值进行测试
            for p in (0, 0.5, 0, 0.7, 0.2, 1, 0.2, 0):
                # 设置当前RNN模型的dropout值为p
                rnn.dropout = p
                # 对输入进行RNN前向传播，得到输出和隐藏状态
                output1, hy1 = rnn(input)
                output2, hy2 = rnn(input)

                # 如果当前的dropout为0或1，或者模型处于评估模式，则输出应该相等，隐藏状态也应该相等
                if p == 0 or p == 1 or not train:
                    self.assertEqual(output1, output2)
                    self.assertEqual(hy1, hy2)
                else:
                    # 否则输出和隐藏状态不应该相等
                    self.assertNotEqual(output1, output2)
                    self.assertNotEqual(hy1, hy2)

                # 如果有上一个输出，则根据模型状态进行断言
                if prev_output is not None:
                    if not train:
                        self.assertEqual(output1.data, prev_output)
                        self.assertEqual(output2.data, prev_output)
                    else:
                        self.assertNotEqual(output1.data, prev_output)
                        self.assertNotEqual(output2.data, prev_output)
                prev_output = output1.data

    # 定义测试函数，测试inplace=True情况下各种ReLU函数的反向传播
    def test_inplace_thnn(self):
        # 定义需要测试的ReLU变种
        modules = [nn.ReLU, nn.ELU, nn.SELU, nn.CELU, nn.RReLU]
        # 遍历每种ReLU变种
        for mod in modules:
            # 创建一个指定inplace=True的ReLU实例
            r = mod(inplace=True)
            # 创建一个形状为(5, 5)的随机张量，并设置requires_grad为True
            input = torch.randn(5, 5, requires_grad=True)
            # 对输入应用ReLU函数
            output = r(input + 0)
            # 创建一个形状相同的随机梯度张量
            grad_output = torch.randn(5, 5)
            # 克隆梯度张量
            grad_output_clone = grad_output.clone()
            # 计算ReLU函数的反向传播
            output.backward(grad_output)
            # 断言梯度张量没有被修改
            self.assertEqual(grad_output, grad_output_clone)

    # 设置默认数据类型为torch.double的装饰器，测试PixelShuffle在NHWC布局下的CPU行为
    @set_default_dtype(torch.double)
    def test_pixel_shuffle_nhwc_cpu(self):
        # 创建一个形状为(3, 18, 4, 4)的随机输入张量，并在内存布局上设置为channels_last，并且设置requires_grad=True
        input = torch.randn(3, 18, 4, 4, device='cpu')
        input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
        # 创建一个形状相同的随机梯度张量
        grad = torch.randn(3, 18, 4, 4, device='cpu')
        # 创建PixelShuffle和PixelUnshuffle实例
        ps = torch.nn.PixelShuffle(3)
        pus = torch.nn.PixelUnshuffle(3)

        # 克隆输入张量并设置requires_grad=True
        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        # 克隆梯度张量
        ref_grad = grad.detach().clone().contiguous()
        # 创建PixelShuffle和PixelUnshuffle实例
        ref_ps = torch.nn.PixelShuffle(3)
        ref_pus = torch.nn.PixelUnshuffle(3)

        # 对输入进行PixelShuffle和PixelUnshuffle操作
        out = pus(ps(input))
        # 对结果进行反向传播
        out.backward(grad)
        # 对参考输入进行PixelShuffle和PixelUnshuffle操作
        ref_out = ref_pus(ref_ps(ref_input))
        # 对参考结果进行反向传播
        ref_out.backward(ref_grad)

        # 断言输出结果保持内存布局为channels_last
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        # 断言输出结果与参考输出结果相等
        self.assertEqual(out, ref_out)
        # 断言输入的梯度与参考输入的梯度相等
        self.assertEqual(input.grad, ref_input.grad)

    # 这些测试应该被标记为OpInfo
    # 定义一个测试函数，用于测试 ELU 激活函数在视图上的原地操作
    def test_elu_inplace_on_view(self):
        # 创建一个双精度张量，指定梯度计算，初始值为 [1.0, -1.0, 1.0, -1.0]
        v = torch.tensor([1.0, -1.0, 1.0, -1.0], requires_grad=True, dtype=torch.double)

        # 定义内部函数 func，接受一个根节点参数 root
        def func(root):
            # 克隆根节点，以确保不改变原始数据
            x = root.clone()
            # 从索引 1 开始，长度为 2，创建 x 的视图
            view = x.narrow(0, 1, 2)
            # 在视图上应用 ELU 激活函数，并指定原地操作
            res = F.elu(view, inplace=True)
            # 断言 ELU 操作的结果对象与视图对象是同一个对象
            self.assertIs(res, view)
            # 返回经 ELU 处理后的张量 x
            return x

        # 对 func 函数进行梯度检查
        gradcheck(func, [v])
        # 对 func 函数进行二阶梯度检查
        gradgradcheck(func, [v])

    # 定义一个测试函数，用于测试 ELU 激活函数在视图上的原地操作，并检查二阶梯度
    def test_elu_inplace_gradgrad(self):
        # 创建一个形状为 (8,) 的随机张量，指定梯度计算，双精度
        v = torch.randn(8, requires_grad=True, dtype=torch.double)

        # 定义内部函数 func，接受一个根节点参数 root
        def func(root):
            # 克隆根节点，以确保不改变原始数据
            x = root.clone()
            # 在张量 x 上应用 ELU 激活函数，并指定原地操作
            return F.elu(x, inplace=True)

        # 对 func 函数进行梯度检查
        gradcheck(func, [v])
        # 对 func 函数进行二阶梯度检查
        gradgradcheck(func, [v])

    # 定义一个测试函数，用于测试 ReLU 激活函数在视图上的原地操作
    def test_relu_inplace_on_view(self):
        # 创建一个双精度张量，指定梯度计算，初始值为 [1.0, -1.0, 1.0, -1.0]
        v = torch.tensor([1.0, -1.0, 1.0, -1.0], requires_grad=True, dtype=torch.double)

        # 定义内部函数 func，接受一个根节点参数 root
        def func(root):
            # 克隆根节点，以确保不改变原始数据
            x = root.clone()
            # 从索引 1 开始，长度为 2，创建 x 的视图
            view = x.narrow(0, 1, 2)
            # 在视图上应用 ReLU 激活函数，并指定原地操作
            res = F.relu(view, inplace=True)
            # 断言 ReLU 操作的结果对象与视图对象是同一个对象
            self.assertIs(res, view)
            # 返回经 ReLU 处理后的张量 x
            return x

        # 对 func 函数进行梯度检查
        gradcheck(func, [v])
        # 对 func 函数进行二阶梯度检查
        gradgradcheck(func, [v])

    # 定义一个测试函数，用于验证 PReLU 在不需要梯度的情况下是否能正确反向传播
    def test_PReLU_backward_requires_grad_false(self):
        # 创建设备列表，初始为 CPU
        devices = ['cpu']
        # 如果测试 CUDA，则将设备列表添加 CUDA
        devices += ['cuda'] if TEST_CUDA else []
        # 遍历设备列表
        for d in devices:
            # 创建 PReLU 模块并移动到指定设备上
            m = nn.PReLU().to(d)
            # 创建一个形状为 (2, 3, 4, 5) 的随机张量 x，设备为 d，不需要梯度
            x = torch.randn(2, 3, 4, 5, device=d, requires_grad=False)
            # 在模块 m 上应用张量 x，得到输出 y
            y = m(x)
            # 计算 y 的均值并进行反向传播
            y.mean().backward()
            # 断言张量 x 的梯度为 None
            self.assertEqual(x.grad, None)

    # 定义一个测试函数，用于验证 BCELoss 计算结果始终为非负数
    def test_bce_loss_always_nonnegative(self):
        # 创建全为 1 的目标张量和输入张量
        target = torch.ones(5)
        input = torch.ones(5)
        # 断言 BCELoss 计算的结果中小于 0 的数量为 0
        self.assertEqual((nn.BCELoss()(input, target) < 0).sum(), 0)

        # 创建全为 0 的目标张量和输入张量
        target = torch.zeros(5)
        input = torch.zeros(5)
        # 断言 BCELoss 计算的结果中小于 0 的数量为 0
        self.assertEqual((nn.BCELoss()(input, target) < 0).sum(), 0)

    # 定义一个测试函数，用于验证 BCEWithLogitsLoss 在目标张量和输入张量尺寸不匹配时是否能抛出 ValueError 异常
    def test_bce_with_logits_raises_if_target_and_input_are_different_size(self):
        # 创建形状为 (5,) 的随机目标张量和形状为 (5, 1) 的随机输入张量
        target = torch.rand(5)
        input = torch.rand(5, 1)
        # 使用 self.assertRaises 检查 BCEWithLogitsLoss 在尺寸不匹配时是否抛出 ValueError 异常
        with self.assertRaises(ValueError):
            nn.BCEWithLogitsLoss()(input, target)

        # 创建形状为 (5, 1) 的随机目标张量和形状为 (5,) 的随机输入张量
        target = torch.rand(5, 1)
        input = torch.rand(5)
        # 使用 self.assertRaises 检查 BCEWithLogitsLoss 在尺寸不匹配时是否抛出 ValueError 异常
        with self.assertRaises(ValueError):
            nn.BCEWithLogitsLoss()(input, target)
    # 定义一个测试方法，验证 nn.BCEWithLogitsLoss() 是否与 sigmoid 和 nn.BCELoss() 的组合给出相同的结果

    # 创建 Sigmoid 激活函数对象
    sigmoid = nn.Sigmoid()

    # 生成一个大小为 [64, 4] 的随机目标张量
    target = torch.rand(64, 4)
    # 生成一个大小为 [64, 4] 的随机输出张量，并对其进行偏移，使其值范围在 -0.5 到 0.5 之间
    output = torch.rand(64, 4) - 0.5

    # 使用 nn.BCEWithLogitsLoss() 计算输出和目标的损失，并与使用 sigmoid 和 nn.BCELoss() 计算的结果进行比较
    self.assertEqual(nn.BCEWithLogitsLoss()(output, target), nn.BCELoss()(sigmoid(output), target))

    # 生成一个大小为 [4] 的随机权重张量，并用它们验证带权重版本的损失函数
    weight = torch.rand(4)
    self.assertEqual(nn.BCEWithLogitsLoss(weight)(output, target), nn.BCELoss(weight)(sigmoid(output), target))

    # 生成一个大小为 [4, 1] 的全零目标张量，并生成一个大小相同的张量，并填充 -100 的输出张量
    target = torch.zeros(4, 1, dtype=torch.float)
    output = torch.empty(4, 1, dtype=torch.float).fill_(-100)

    # 测试处理极端输出的损失计算
    self.assertEqual(nn.BCEWithLogitsLoss()(output, target), nn.BCELoss()(sigmoid(output), target))

    # 测试带有 'none' 选项的损失计算，用于比较输出和目标之间的逐元素损失
    self.assertEqual(nn.BCEWithLogitsLoss(reduction='none')(output, target),
                     nn.BCELoss(reduction='none')(sigmoid(output), target))

    # 生成一个大小为 [1] 的随机权重张量，并验证带权重版本的损失函数
    weight = torch.rand(1, dtype=torch.float)
    self.assertEqual(nn.BCEWithLogitsLoss(weight)(output, target), nn.BCELoss(weight)(sigmoid(output), target))



    # 定义一个测试方法，验证 nn.BCELoss() 的输入范围

    # 创建 BCELoss 对象
    bceloss = nn.BCELoss()

    # 生成一个大小为 [25, 25] 的随机目标张量和一个相同大小的有效输出张量
    target = torch.rand(25, 25)
    output_valid = torch.rand(25, 25)

    # 生成一个比有效输出张量低 1.0 的输出张量和一个比有效输出张量高 1.0 的输出张量，用于测试损失函数
    output_too_negative = output_valid - 1.0
    output_too_positive = output_valid + 1.0

    # 计算有效输出张量的损失值
    loss_valid = bceloss(output_valid, target)

    # 验证对于过低和过高的输出张量，损失计算是否会引发 RuntimeError
    with self.assertRaisesRegex(RuntimeError, 'between 0 and 1'):
        loss_too_negative = bceloss(output_too_negative, target)
    with self.assertRaisesRegex(RuntimeError, 'between 0 and 1'):
        loss_too_positive = bceloss(output_too_positive, target)



    # 定义一个测试方法，验证当目标张量的大小与输出张量的大小不匹配时，是否会引发 ValueError

    # 创建 BCELoss 对象
    bceloss = nn.BCELoss()

    # 生成一个大小为 [25] 的随机张量和一个大小为 [25, 1] 的随机张量
    a = torch.rand(25)
    b = torch.rand(25, 1)

    # 验证损失计算是否会引发 ValueError，因为目标张量和输出张量的维度不匹配
    with self.assertRaisesRegex(ValueError, r'Using a target size \('):
        bceloss(a, b)



    # 定义一个测试方法，验证当处理大张量并进行梯度计算时，nn.BCEWithLogitsLoss() 是否与 sigmoid 和 nn.BCELoss() 的组合给出相同的结果

    # 设置张量的大小
    x_size = 1024
    y_size = 256

    # 生成一个大小为 [1024, 256] 的随机目标张量
    target = torch.rand(x_size, y_size)

    # 遍历不同的减少选项：'none', 'mean', 'sum'
    for reduction in ['none', 'mean', 'sum']:
        # 生成一个大小为 [1024, 256] 的随机输出张量，并从中克隆一个不带梯度的版本
        output_sig = torch.rand(x_size, y_size) - 0.5
        output_logits = output_sig.clone().detach()

        # 设置输出张量的梯度计算标志
        output_sig.requires_grad = True
        output_logits.requires_grad = True

        # 生成一个大小为 [256] 的随机权重张量
        weight = torch.rand(y_size)

        # 计算使用 nn.BCELoss() 和 sigmoid 输出的损失值，并使用 nn.BCEWithLogitsLoss() 和 logits 输出进行比较
        loss_sig = nn.BCELoss(weight, reduction=reduction)(
            torch.sigmoid(output_sig), target
        )
        loss_logits = nn.BCEWithLogitsLoss(weight, reduction=reduction)(
            output_logits, target
        )

        # 验证计算的损失是否相等
        self.assertEqual(loss_logits, loss_sig)

        # 如果减少选项为 'none'，则生成一个与输出张量相同大小的随机梯度张量，并进行反向传播
        if reduction == 'none':
            grad = torch.rand(x_size, y_size)
            loss_sig.backward(grad)
            loss_logits.backward(grad)
        else:
            # 否则，只进行 logits 输出的反向传播
            loss_sig.backward()
            loss_logits.backward()

        # 验证输出张量的梯度是否相等
        self.assertEqual(output_sig.grad, output_logits.grad)
    # 测试 BCEWithLogitsLoss 在前向传播时具有正确的梯度
    def test_bce_with_logits_has_correct_forward_grad(self):
        # 生成一个随机张量作为输出，要求计算其梯度，数据类型为双精度浮点数
        output = torch.randn(3, 5, requires_grad=True, dtype=torch.double)
        # 生成一个随机张量作为目标值，数据类型为双精度浮点数
        target = torch.randn(3, 5, dtype=torch.double)
        # 对不同的减少方式（sum、mean、none）进行梯度检查
        for reduction in ('sum', 'mean', 'none'):
            # 使用 gradcheck 函数检查前向自动求导的正确性，lambda 函数定义了 BCEWithLogitsLoss 的调用方式
            gradcheck(lambda self, target: nn.BCEWithLogitsLoss(reduction=reduction)(self, target),
                      (output, target), check_forward_ad=True)

    # 测试 BCEWithLogitsLoss 在零值输入时的梯度是否正确
    def test_bce_with_logits_has_correct_grad_at_zero(self):
        # 创建一个全零张量作为输出，要求计算其梯度
        output = torch.zeros(3, 1, requires_grad=True)
        # 创建一个全零张量作为目标值
        target = torch.zeros(3, 1)
        # 对 BCEWithLogitsLoss 的 sum 形式进行反向传播
        nn.BCEWithLogitsLoss(reduction='sum')(output, target).backward()
        # 创建一个预期梯度的张量，值为0.5
        expected_grad = torch.empty(3, 1).fill_(0.5)
        # 使用断言检查输出张量的梯度是否与预期梯度一致
        self.assertEqual(output.grad, expected_grad)

    # 测试 BCEWithLogitsLoss 在广播权重时的行为
    def test_bce_with_logits_broadcasts_weights(self):
        # 创建一个随机张量作为目标值
        target = torch.rand(16, 4)
        # 创建一个随机张量作为输出，并将其值减去0.5
        output = torch.rand(16, 4) - 0.5

        # 创建一个长度为4的随机权重张量
        weight = torch.rand(4)
        # 使用权重进行 BCEWithLogitsLoss 的计算
        out1 = nn.BCEWithLogitsLoss(weight)(output, target)

        # 将权重张量扩展为与输出张量相同的大小，并确保其连续性
        weight = weight.expand(16, 4).contiguous()
        # 使用扩展后的权重进行 BCEWithLogitsLoss 的计算
        out2 = nn.BCEWithLogitsLoss(weight)(output, target)

        # 使用断言检查两次计算结果是否一致
        self.assertEqual(out1, out2)

        # 创建一个长度为16的随机权重张量
        weight = torch.rand(16, 1)
        # 使用权重进行 BCEWithLogitsLoss 的计算
        out1 = nn.BCEWithLogitsLoss(weight)(output, target)

        # 将权重张量扩展为与输出张量相同的大小，并确保其连续性
        weight = weight.expand(16, 4).contiguous()
        # 使用扩展后的权重进行 BCEWithLogitsLoss 的计算
        out2 = nn.BCEWithLogitsLoss(weight)(output, target)

        # 使用断言检查两次计算结果是否一致
        self.assertEqual(out1, out2)

    # 测试 BCEWithLogitsLoss 在使用正权重时是否与无权重时结果一致
    def test_bce_with_logits_ones_in_pos_weights_are_the_same_as_none(self):
        # 创建一个随机张量作为目标值
        target = torch.rand(64, 4)
        # 创建一个随机张量作为输出，并将其值减去0.5
        output = torch.rand(64, 4) - 0.5
        # 创建一个全为1的正权重张量
        pos_weight = torch.ones(64, 4)

        # 使用断言检查有正权重与无权重时 BCEWithLogitsLoss 的计算结果是否一致
        self.assertEqual(nn.BCEWithLogitsLoss()(output, target),
                         nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target))

    # 测试 BCEWithLogitsLoss 在广播正权重时的行为
    def test_bce_with_logits_broadcasts_pos_weights(self):
        # 创建一个随机张量作为目标值
        target = torch.rand(64, 4)
        # 创建一个随机张量作为输出，并将其值减去0.5
        output = torch.rand(64, 4) - 0.5
        # 创建一个长度为4的随机正权重张量
        pos_weight = torch.rand(4)
        
        # 使用正权重张量进行 BCEWithLogitsLoss 的计算
        out1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target)

        # 将正权重张量扩展为与输出张量相同的大小，并确保其连续性
        pos_weight1 = pos_weight.expand(1, 4)
        # 使用扩展后的正权重张量进行 BCEWithLogitsLoss 的计算
        out2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight1)(output, target)

        # 将正权重张量扩展为与输出张量相同的大小，并确保其连续性
        pos_weight2 = pos_weight.expand(64, 4)
        # 使用扩展后的正权重张量进行 BCEWithLogitsLoss 的计算
        out3 = nn.BCEWithLogitsLoss(pos_weight=pos_weight2)(output, target)

        # 使用断言检查三次计算结果是否一致
        self.assertEqual(out1, out2)
        self.assertEqual(out1, out3)

    # 测试 BCEWithLogitsLoss 在使用正权重时零值输入的梯度是否正确
    def test_bce_with_logits_with_pos_weight_has_correct_grad_at_zero(self):
        # 创建一个全零张量作为输出，要求计算其梯度
        output = torch.zeros(3, 1, requires_grad=True)
        # 创建一个全零张量作为目标值
        target = torch.zeros(3, 1)
        # 创建一个全为1的正权重张量
        pos_weight = torch.ones(3, 1)
        # 对 BCEWithLogitsLoss 的 sum 形式进行反向传播
        nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum')(output, target).backward()
        # 创建一个预期梯度的张量，值为0.5
        expected_grad = torch.empty(3, 1).fill_(0.5)
        # 获取输出张量的梯度
        grad = output.grad
        # 使用断言检查输出张量的梯度是否与预期梯度一致
        self.assertEqual(grad, expected_grad)
    # 测试 BCEWithLogitsLoss 在稳定性上的表现
    def test_bce_with_logits_stability(self):
        # 创建模拟的输出张量和目标张量
        output = torch.tensor([0., -120.])
        target = torch.tensor([0., 1.])
        pos_weight = torch.tensor([1., 1.])

        # 使用默认参数计算 BCEWithLogitsLoss
        out1 = nn.BCEWithLogitsLoss()(output, target)
        # 断言输出的所有值都是有限的
        self.assertTrue(torch.isfinite(out1).all().item())

        # 使用自定义的正权重参数计算 BCEWithLogitsLoss
        out2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target)
        # 断言输出的所有值都是有限的
        self.assertTrue(torch.isfinite(out2).all().item())

    # 测试 BCELoss 在权重广播上的行为
    def test_bce_loss_broadcasts_weights(self):
        # 创建 Sigmoid 激活函数
        sigmoid = nn.Sigmoid()
        # 创建随机目标张量和输出张量
        target = torch.rand(16, 4)
        output = torch.rand(16, 4) - 0.5

        # 创建单一维度的权重张量
        weight = torch.rand(4)
        # 使用权重计算 BCELoss
        out1 = nn.BCELoss(weight)(sigmoid(output), target)

        # 将权重张量扩展为与输出张量相同的形状
        weight = weight.expand(16, 4).contiguous()
        # 再次使用扩展后的权重计算 BCELoss
        out2 = nn.BCELoss(weight)(sigmoid(output), target)

        # 断言两次计算结果相等
        self.assertEqual(out1, out2)

        # 创建单列的权重张量
        weight = torch.rand(16, 1)
        # 使用单列权重计算 BCELoss
        out1 = nn.BCELoss(weight)(sigmoid(output), target)

        # 将单列权重张量扩展为与输出张量相同的形状
        weight = weight.expand(16, 4).contiguous()
        # 再次使用扩展后的权重计算 BCELoss
        out2 = nn.BCELoss(weight)(sigmoid(output), target)

        # 断言两次计算结果相等
        self.assertEqual(out1, out2)

    # 测试 inplace=True 的情况下 Hardtanh 的梯度二阶导数
    def test_hardtanh_inplace_gradgrad(self):
        # 创建一个双精度浮点数类型的需要梯度的随机张量
        v = torch.randn(8, requires_grad=True, dtype=torch.double)

        # 定义函数 func，它在 inplace=True 的情况下使用 Hardtanh
        def func(root):
            x = root.clone()
            return F.hardtanh(x, inplace=True)

        # 检查函数 func 的梯度
        gradcheck(func, [v])
        # 检查函数 func 的二阶导数
        gradgradcheck(func, [v])

    # 测试大张量下 Hardtanh 的反向传播
    def test_hardtanh_backward(self):
        # 创建一个形状为 (128, 10000) 的需要梯度的随机张量
        x = torch.randn(128, 10000, requires_grad=True)
        # 创建一个形状相同的随机梯度张量
        grad = torch.randn(128, 10000)
        # 创建一个与 x 形状相同的零张量
        z = torch.zeros(128, 10000)
        # 使用 Hardtanh 计算 y
        y = F.hardtanh(x)
        # 对 y 执行反向传播，使用 grad 作为梯度
        y.backward(grad)
        # 计算 Hardtanh 的反向传播参考路径
        mask = (x > -1) & (x < 1)
        x_grad_ref = torch.where(mask, grad, z)
        # 断言计算得到的梯度与参考路径一致
        self.assertEqual(x.grad, x_grad_ref)
    # 定义一个测试方法，用于测试批归一化在 NHWC 格式下在 CPU 上的表现
    def test_batchnorm_nhwc_cpu(self):
        # 定义内部辅助函数 helper，用于执行具体的批归一化测试
        def helper(self, mod, size, dtype, mixed_dtype=False, format=torch.channels_last, precision=None):
            # 获取输入张量的通道数
            channels = size[1]
            # 生成一个随机的输入张量，指定数据类型、设备为 CPU，并标记需要梯度计算
            input = torch.randn(size, dtype=dtype, device='cpu', requires_grad=True)
            # 将输入张量转换为指定的内存格式（NHWC）并指定数据类型
            input = input.contiguous(memory_format=format).to(dtype)
            # 保持输入张量的梯度信息
            input.retain_grad()
            # 生成一个随机的梯度张量，指定数据类型和设备为 CPU
            grad = torch.randn(size, dtype=dtype, device='cpu')
            # 将梯度张量转换为指定的内存格式（NHWC）
            grad = grad.contiguous(memory_format=format)
            # 创建批归一化层，指定通道数，并将其移到 CPU 上，并指定数据类型
            bn = mod(channels).cpu().to(dtype)
            # 初始化批归一化层的权重为均匀分布
            bn.weight.data.uniform_()
            # 初始化批归一化层的偏置为均匀分布
            bn.bias.data.uniform_()

            # 创建参考输入张量，从输入张量分离出来，并克隆其数据并保持梯度
            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            # 创建参考梯度张量，从梯度张量分离出来，并克隆其数据
            ref_grad = grad.detach().clone().contiguous()
            # 创建参考批归一化层，与原批归一化层保持一致的状态
            ref_bn = mod(channels).cpu().to(dtype)
            # 从原批归一化层加载状态到参考批归一化层

            ref_bn.load_state_dict(bn.state_dict())

            # 如果需要混合数据类型，则将批归一化层和参考批归一化层转换为 float 类型
            if mixed_dtype:
                bn.float()
                ref_bn.float()

            # 对批归一化层进行前向传播
            out = bn(input)
            # 对输出进行反向传播
            out.backward(grad)
            # 对参考批归一化层进行前向传播
            ref_out = ref_bn(ref_input)
            # 对参考批归一化层的输出进行反向传播
            ref_out.backward(ref_grad)

            # 断言输出张量保持指定的内存格式（NHWC）
            self.assertTrue(out.is_contiguous(memory_format=format))
            # 断言参考输出张量保持连续性
            self.assertTrue(ref_out.is_contiguous())
            # 断言输出张量与参考输出张量相等
            self.assertEqual(out, ref_out)
            # 断言批归一化层的权重梯度与参考批归一化层的权重梯度相等，允许一定的数值误差范围
            self.assertEqual(bn.weight.grad, ref_bn.weight.grad, atol=precision, rtol=precision)
            # 断言批归一化层的偏置梯度与参考批归一化层的偏置梯度相等
            self.assertEqual(bn.bias.grad, ref_bn.bias.grad)
            # 断言输入张量的梯度与参考输入张量的梯度相等
            self.assertEqual(input.grad, ref_input.grad)

        # 对不同形状的输入进行测试：NC11 和 N1HW；同时测试混合数据类型
        for shape in [(4, 8, 10, 10), (4, 1, 9, 9), (4, 9, 1, 1)]:
            # 针对不同的数据类型进行测试：float、bfloat16、float16
            for dtype in [torch.float, torch.bfloat16, torch.float16]:
                # 对是否混合数据类型进行测试
                for mixed_dtype in [False, True]:
                    # 如果数据类型为 float，则强制 mixed_dtype 为 False
                    if dtype == torch.float:
                        mixed_dtype = False
                    # 调用辅助函数 helper 进行测试
                    helper(self, nn.BatchNorm2d, shape, dtype, mixed_dtype, torch.channels_last)

        # 定义不同数据类型的精度要求
        precisons = {torch.float: 1e-4, torch.bfloat16: 1e-4, torch.float16: None}
        # 对不同形状的 3D 输入进行测试，同时测试混合数据类型
        for shape in [(4, 8, 2, 10, 10), (4, 1, 2, 9, 9), (4, 9, 1, 1, 1)]:
            # 针对不同的数据类型进行测试：float、bfloat16、float16
            for dtype in [torch.float, torch.bfloat16, torch.float16]:
                # 对是否混合数据类型进行测试
                for mixed_dtype in [False, True]:
                    # 如果数据类型为 float，则强制 mixed_dtype 为 False
                    if dtype == torch.float:
                        mixed_dtype = False
                    # 调用辅助函数 helper 进行测试，同时传入精度要求
                    helper(self, nn.BatchNorm3d, shape, dtype, mixed_dtype, torch.channels_last_3d, precisons[dtype])
    # 定义一个测试函数，用于测试非连续数据的批归一化在 CPU 上的行为，参数 bn_module 是批归一化模块
    def test_batchnorm_non_contig_cpu(self, bn_module):
        # 定义内部辅助函数 helper，接受 self 和数据类型 dtype 作为参数
        def helper(self, dtype):
            # 创建一个大小为 (1, 3, 2, 1) 的浮点数张量 input，并移到 CPU 上
            input = torch.arange(6, dtype=torch.float).reshape(1, 3, 2, 1).cpu()
            # 将 input 的维度重新排列为 (1, 2, 3, 1)
            input = input.permute(0, 2, 1, 3)

            # 创建一个大小为 2 的 bn_module 批归一化模块，并移到 CPU 上，设为浮点数类型，并设置为评估模式
            bn = bn_module(2).cpu().float().eval()
            # 随机初始化 bn 的权重和偏置
            bn.weight.data.uniform_()
            bn.bias.data.uniform_()

            # 创建 ref_input，为 input 的分离克隆，并保证其是连续的
            ref_input = input.detach().clone().contiguous()
            # 创建一个标准的 nn.BatchNorm2d 批归一化模块，移到 CPU 上，设为浮点数类型，并设置为评估模式
            ref_bn = nn.BatchNorm2d(2).cpu().float().eval()
            # 从 bn 的状态字典加载状态到 ref_bn
            ref_bn.load_state_dict(bn.state_dict())

            # 对 input 应用 bn 批归一化
            out = bn(input)
            # 对 ref_input 应用 ref_bn 批归一化
            ref_out = ref_bn(ref_input)

            # 断言 out 是连续的，使用通道最后的内存格式
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            # 断言 ref_out 是连续的
            self.assertTrue(ref_out.is_contiguous())
            # 断言 out 等于 ref_out
            self.assertEqual(out, ref_out)

            # 创建一个大小为 (1, 3, 2, 4) 的 dtype 数据类型张量 input_bf
            input_bf = torch.arange(24, dtype=dtype).reshape(1, 3, 2, 4)
            # 将 input_bf 的维度重新排列为 (1, 2, 3, 4)
            input_bf = input_bf.permute(0, 2, 1, 3)
            # 将 input_bf 转换为浮点数类型
            input_f = input_bf.float()
            # 创建一个大小为 2 的 bn_module 批归一化模块，设为浮点数类型，并设置为评估模式
            bn_mix = bn_module(2).float().eval()
            # 深度复制 bn_mix 到 ref_bn_f
            ref_bn_f = deepcopy(bn_mix)
            # 对 input_bf 应用 bn_mix 批归一化
            out_bf = bn_mix(input_bf)
            # 对 input_f 应用 ref_bn_f 批归一化
            ref_out_bf = ref_bn_f(input_f)
            # 断言 ref_out_bf 等于 out_bf 的浮点数形式，允许的绝对误差和相对误差分别为 0.05 和 0.05
            self.assertEqual(ref_out_bf, out_bf.float(), atol=0.05, rtol=0.05)

        # 调用 helper 函数，分别传入 self 和 torch.bfloat16 作为参数
        helper(self, torch.bfloat16)
        # 调用 helper 函数，分别传入 self 和 torch.float16 作为参数
        helper(self, torch.float16)

    # 如果 TEST_CUDA 为假，则跳过该测试
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    # 如果 TEST_CUDNN 为假，则跳过该测试
    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    def test_batchnorm_cudnn_nhwc(self):
        # 定义内部函数 run_test，用于执行单个测试用例
        def run_test(input, grad_output):
            # 获取输入张量的通道数
            c = input.size(1)
            # 创建并配置一个在 CUDA 上运行的 BatchNorm2d 模块，数据类型为 float
            mod = nn.BatchNorm2d(c).cuda().float()
            # 初始化权重和偏置
            mod.weight.data.uniform_()
            mod.bias.data.uniform_()
            # 创建参考输入张量，确保其是可导的
            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            # 创建参考梯度张量
            ref_grad = grad.detach().clone().contiguous()
            # 创建另一个在 CUDA 上运行的 BatchNorm2d 模块作为参考模块
            ref_mod = nn.BatchNorm2d(c).cuda().float()
            # 加载当前模块的状态到参考模块中
            ref_mod.load_state_dict(mod.state_dict())
            # 对当前模块应用输入张量，并计算输出
            out = mod(input)
            # 反向传播当前模块的梯度
            out.backward(grad_output)
            # 对参考模块应用参考输入张量，并计算输出
            ref_out = ref_mod(ref_input)
            # 对参考模块进行反向传播
            ref_out.backward(ref_grad)
            # 断言当前模块的输出是通道优先内存格式
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            # 断言参考模块的输出是连续的
            self.assertTrue(ref_out.is_contiguous())
            # 断言当前模块的输出与参考模块的输出相等
            self.assertEqual(out, ref_out)
            # 断言当前模块的权重梯度与参考模块的权重梯度相等
            self.assertEqual(mod.weight.grad, ref_mod.weight.grad)
            # 断言当前模块的偏置梯度与参考模块的偏置梯度相等
            self.assertEqual(mod.bias.grad, ref_mod.bias.grad)
            # 断言当前模块的输入张量梯度与参考输入张量的梯度相等
            self.assertEqual(input.grad, ref_input.grad)

        # 创建 CUDA 上的随机整数张量作为输入，并标记为通道优先内存格式，并确保可以计算梯度
        input = torch.randint(1, 10, (4, 8, 2, 2), dtype=torch.float32, device="cuda")
        input = input.contiguous(memory_format=torch.channels_last).detach().requires_grad_()

        # 创建 CUDA 上的随机整数张量作为梯度，并标记为通道优先内存格式
        grad = torch.randint(1, 10, (4, 8, 2, 2), dtype=torch.float32, device="cuda")
        grad = grad.contiguous(memory_format=torch.channels_last)
        # 运行测试函数，传入输入张量和梯度张量
        run_test(input, grad)
        # 见 issue #42588，虽然 grad 是通道优先内存格式的，但 grad.suggest_memory_format 正确返回 "contiguous" 而非 channels_last

        # 创建 CUDA 上的随机整数张量作为输入，并标记为通道优先内存格式，并确保可以计算梯度
        input = torch.randint(1, 10, (2, 8, 8, 1), dtype=torch.float32, device="cuda")
        input = input.contiguous(memory_format=torch.channels_last).detach().requires_grad_()
        # 创建 CUDA 上的随机整数张量作为梯度，并进行维度置换
        grad = torch.randint(1, 10, (2, 8, 8, 1), dtype=torch.float32, device="cuda")
        grad = grad.permute(0, 2, 1, 3)
        # 运行测试函数，传入输入张量和梯度张量
        run_test(input, grad)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_batchnorm_cudnn_half(self):
        # 使用半精度张量在 CUDA 上创建随机整数张量作为输入，并标记为可计算梯度
        input = torch.randint(1, 10, (2, 3, 2, 2), dtype=torch.half, device="cuda", requires_grad=True)
        # 创建并配置在 CUDA 上运行的 BatchNorm2d 模块，数据类型为 half
        m = nn.BatchNorm2d(3).half().cuda()
        # 应用 THNN 算法计算输出
        thnn_output = m(input)
        # 计算输出的和并执行反向传播
        thnn_output.sum().backward()
        # 复制 THNN 算法的输入梯度
        thnn_input_grad = input.grad.data.clone()
        # 断言 THNN 算法的输出与输入的数据类型和形状相等
        self.assertEqualTypeString(thnn_output, input)

        # 如果支持 cuDNN，继续执行以下代码
        if TEST_CUDNN:
            # 清空输入的梯度
            input.grad = None
            # 将 BatchNorm2d 模块转换为 float 类型
            m = m.float()
            # 应用 cuDNN 算法计算输出
            cudnn_output = m(input)
            # 计算输出的和并执行反向传播
            cudnn_output.sum().backward()
            # 复制 cuDNN 算法的输入梯度
            cudnn_input_grad = input.grad.data.clone()
            # 断言 cuDNN 算法的输出与 THNN 算法的输出相等
            self.assertEqual(cudnn_output, thnn_output)
            # 断言 cuDNN 算法的输入梯度与 THNN 算法的输入梯度相等，允许的绝对误差为 1e-3，相对误差为 0
            self.assertEqual(cudnn_input_grad, thnn_input_grad, atol=1e-3, rtol=0)
    # 定义一个测试函数，测试在使用半精度输入时批归一化的行为
    def test_batchnorm_nonaffine_cuda_half_input(self):
        # 创建一个大小为 (16, 3, 24, 24) 的随机张量，数据类型为半精度，存储在 CUDA 设备上
        input = torch.randn(16, 3, 24, 24, dtype=torch.half, device="cuda")
        # 创建一个批归一化层对象，对每个通道不进行仿射变换，且在 FP32 中保留运行时统计数据，存储在 CUDA 设备上
        m = nn.BatchNorm2d(3, affine=False).cuda().float()  # keep running stats in FP32
        # 对输入进行批归一化处理
        output = m(input)
        # 断言输出张量的类型与输入张量相同
        self.assertEqualTypeString(output, input)
        # 将批归一化层设置为评估模式
        m.eval()
        # 再次对输入进行批归一化处理
        output = m(input)
        # 断言输出张量的类型与输入张量相同
        self.assertEqualTypeString(output, input)

    # 定义一个测试函数，测试如果每个通道的值少于一个时，批归一化是否会引发错误
    def test_batchnorm_raises_error_if_less_than_one_value_per_channel(self):
        # 创建一个形状为 (1, 10, 1) 的随机张量
        x = torch.rand(10)[None, :, None]
        # 使用批归一化层对该张量进行处理，预期会引发 ValueError 异常
        with self.assertRaises(ValueError):
            torch.nn.BatchNorm1d(10)(x)

    # 定义一个测试函数，测试如果运行时均值的大小与输入不同，批归一化是否会引发错误
    def test_batchnorm_raises_error_if_running_mean_is_not_same_size_as_input(self):
        # 创建一个形状为 (2, 10) 的随机张量作为输入
        input = torch.rand(2, 10)
        # 创建一个形状为 (10,) 的随机张量作为运行时方差
        running_var = torch.rand(10)
        # 创建一个错误大小的列表 [9, 11]
        wrong_sizes = [9, 11]
        # 遍历错误大小列表
        for size in wrong_sizes:
            # 使用 F.batch_norm 函数对输入进行处理，预期会引发 RuntimeError 异常
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, torch.rand(size), running_var)

    # 定义一个测试函数，测试如果运行时方差的大小与输入不同，批归一化是否会引发错误
    def test_batchnorm_raises_error_if_running_var_is_not_same_size_as_input(self):
        # 创建一个形状为 (2, 10) 的随机张量作为输入
        input = torch.rand(2, 10)
        # 创建一个形状为 (10,) 的随机张量作为运行时均值
        running_mean = torch.rand(10)
        # 创建一个错误大小的列表 [9, 11]
        wrong_sizes = [9, 11]
        # 遍历错误大小列表
        for size in wrong_sizes:
            # 使用 F.batch_norm 函数对输入进行处理，预期会引发 RuntimeError 异常
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, running_mean, torch.rand(size))

    # 定义一个测试函数，测试如果权重的大小与输入不同，批归一化是否会引发错误
    def test_batchnorm_raises_error_if_weight_is_not_same_size_as_input(self):
        # 创建一个形状为 (2, 10) 的随机张量作为输入
        input = torch.rand(2, 10)
        # 创建一个形状为 (10,) 的随机张量作为运行时均值
        running_mean = torch.rand(10)
        # 创建一个形状为 (10,) 的随机张量作为运行时方差
        running_var = torch.rand(10)
        # 创建一个错误大小的列表 [9, 11]
        wrong_sizes = [9, 11]
        # 遍历错误大小列表
        for size in wrong_sizes:
            # 使用 F.batch_norm 函数对输入进行处理，预期会引发 RuntimeError 异常
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, running_mean, running_var, weight=Parameter(torch.rand(size)))

    # 定义一个测试函数，测试如果偏置的大小与输入不同，批归一化是否会引发错误
    def test_batchnorm_raises_error_if_bias_is_not_same_size_as_input(self):
        # 创建一个形状为 (2, 10) 的随机张量作为输入
        input = torch.rand(2, 10)
        # 创建一个形状为 (10,) 的随机张量作为运行时均值
        running_mean = torch.rand(10)
        # 创建一个形状为 (10,) 的随机张量作为运行时方差
        running_var = torch.rand(10)
        # 创建一个错误大小的列表 [9, 11]
        wrong_sizes = [9, 11]
        # 遍历错误大小列表
        for size in wrong_sizes:
            # 使用 F.batch_norm 函数对输入进行处理，预期会引发 RuntimeError 异常
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, running_mean, running_var, bias=Parameter(torch.rand(size)))
    # 定义一个测试函数，用于检查如果运行的方差或均值有前向梯度，则批量归一化会引发错误
    def test_batchnorm_raises_error_if_running_var_or_running_mean_have_forward_grad(self):
        # 设置输入参数和运行时的均值、方差
        args = (
            torch.randn(3, 2, 5),  # input，随机生成的输入数据
            torch.randn(2),  # running_mean，随机生成的运行时均值
            torch.randn(2),  # running_var，随机生成的运行时方差
        )
        # 设置关键字参数
        kwargs = {'training': False, 'momentum': -1.2}
        # 创建一个偏函数，用于调用 F.batch_norm 函数
        fn = partial(F.batch_norm, **kwargs)

        # 对于每一个不同的索引组合进行遍历
        for dual_indices in ((0,), (1,), (1, 2), (0, 1), (0, 1, 2),):
            # 为每个参数创建一个相似的随机张量作为切线
            tangents = tuple(torch.rand_like(x) for x in args)

            # 在自动微分的双重级别上下文中
            with fwAD.dual_level():
                # 如果索引在 dual_indices 中，则使用切线创建双重对象；否则使用原始值
                duals = [fwAD.make_dual(primal, tangent) if i in dual_indices else primal
                         for i, (primal, tangent) in enumerate(zip(args, tangents))]
                # 错误消息，用于测试是否批量归一化对于 running_mean 和 running_var 不可微分
                msg = "batch_norm is not differentiable wrt running_mean and running_var"
                # 如果 running_mean 或 running_var 在 dual_indices 中，并且 input 在其中，则应该引发 RuntimeError
                if (1 in dual_indices or 2 in dual_indices) and 0 in dual_indices:
                    with self.assertRaisesRegex(RuntimeError, msg):
                        fn(*duals)
                else:
                    fn(*duals)

    # 测试函数：验证当未跟踪统计数据时，批量归一化缓冲区是否更新
    def test_batchnorm_buffer_update_when_stats_are_not_tracked(self):
        # 输入大小
        input_size = (32, 4)
        # 创建一个具有跟踪运行时统计数据的批量归一化层
        bn = nn.BatchNorm1d(input_size[1], track_running_stats=True)
        # 禁止跟踪运行时统计数据
        bn.track_running_stats = False
        # 存储初始值
        num_batches = bn.num_batches_tracked.clone()
        running_mean = bn.running_mean.clone()
        running_var = bn.running_var.clone()
        # 前向传播随机张量
        _ = bn(torch.rand(input_size))
        # 确保没有更新任何缓冲区
        self.assertTrue(torch.equal(num_batches, bn.num_batches_tracked))
        self.assertTrue(torch.equal(running_mean, bn.running_mean))
        self.assertTrue(torch.equal(running_var, bn.running_var))

    # 在 CUDA 可用时跳过测试：验证在 NHWC 格式下的批量归一化在 CUDA 上是否正常工作
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_batchnorm_nhwc_cuda(self):
        # 对于每种数据类型（半精度和单精度）
        for dtype in (torch.half, torch.float):
            # 设置输入的尺寸
            (N, C, H, W) = 2, 64, 50, 50
            # 创建一个具有跟踪运行时统计数据的二维批量归一化模型
            model = torch.nn.BatchNorm2d(C, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            # 将模型设置为评估模式，并将其移到 CUDA 并设置数据类型
            model = model.eval().cuda().to(dtype)
            # 创建随机输入张量，并确保在 CUDA 设备上
            inp1 = torch.randn(N, C, H, W, device=torch.device('cuda'), dtype=dtype)
            # 将输入张量转换为连续内存格式的通道最后格式
            inp2 = inp1.contiguous(memory_format=torch.channels_last)
            # 分别对 inp1 和 inp2 进行模型前向传播
            out1 = model(inp1)
            out2 = model(inp2)
            # 断言两次前向传播的结果相等
            self.assertTrue(torch.equal(out1, out2))
    # 定义一个测试方法，用于验证 BatchNorm2d 类的状态字典加载功能
    def test_batchnorm_load_state_dict(self):
        # 创建一个具有3个特征通道的 BatchNorm2d 实例
        bn = torch.nn.BatchNorm2d(3)
        # 断言状态字典中 "num_batches_tracked" 键对应的值为0的张量
        self.assertEqual(bn.state_dict()["num_batches_tracked"], torch.tensor(0))

        # 修改 bn 实例的 num_batches_tracked 属性为张量值 10
        bn.num_batches_tracked = torch.tensor(10)
        # 再次断言状态字典中 "num_batches_tracked" 键对应的值为10的张量
        self.assertEqual(bn.state_dict()["num_batches_tracked"], torch.tensor(10))

        # 创建一个空的有序字典
        empty_dict = OrderedDict()
        # 使用 strict=False 的方式加载空状态字典到 bn 实例中
        bn.load_state_dict(empty_dict, strict=False)
        # 断言状态字典中 "num_batches_tracked" 键对应的值仍为10的张量
        self.assertEqual(bn.state_dict()["num_batches_tracked"], torch.tensor(10))

        # 在 'meta' 设备上创建一个 BatchNorm2d 实例
        with torch.device('meta'):
            meta_bn = torch.nn.BatchNorm2d(3)
        # 断言 meta_bn 实例的 num_batches_tracked 属性位于 'meta' 设备上
        self.assertTrue(meta_bn.num_batches_tracked.device == torch.device('meta'))
        # 使用 assign=True 和 strict=False 的方式加载空状态字典到 meta_bn 实例中
        meta_bn.load_state_dict(empty_dict, assign=True, strict=False)
        # 断言状态字典中 "num_batches_tracked" 键对应的值为单一的0张量
        self.assertEqual(meta_bn.state_dict()["num_batches_tracked"], torch.tensor(0))

    # 定义一个测试方法，用于验证 batch_norm_update_stats 函数的异常情况处理
    def test_batch_norm_update_stats(self):
        # 创建一个空的输入张量 input
        input = torch.rand(0, 1)
        # 创建随机的 running_mean 和 running_var 张量
        running_mean = torch.rand(1)
        running_var = torch.rand(1)
        # 断言调用 torch.batch_norm_update_stats 函数时抛出 RuntimeError 异常，
        # 并且异常消息包含指定的错误信息
        with self.assertRaisesRegex(RuntimeError,
                                    re.escape("input tensor must have at least one element, but got input_sizes = [0, 1]")):
            torch.batch_norm_update_stats(input=input, momentum=0.0, running_mean=running_mean, running_var=running_var)

    # 定义一个测试方法，用于验证 F.pairwise_distance 函数的梯度检查
    def test_pairwise_distance(self):
        # 创建两个形状为 (4, 4) 的双精度浮点类型的张量，并且要求计算梯度
        input1 = torch.randn(4, 4, requires_grad=True, dtype=torch.double)
        input2 = torch.randn(4, 4, requires_grad=True, dtype=torch.double)
        # 断言对 lambda 函数 F.pairwise_distance 进行梯度检查通过
        self.assertTrue(gradcheck(lambda x, y: F.pairwise_distance(x, y), (input1, input2)))

    # TODO: Create an OpInfo for pdist
    # 定义一个测试方法，用于验证 F.pdist 函数在不同设备上的操作
    def test_pdist(self):
        # 遍历所有设备和转置选项
        for device, trans in itertools.product(device_(), [False, True]):
            # 创建形状为 (4, 5) 的双精度浮点类型的随机张量，并且要求计算梯度
            inp = torch.randn(4, 5, dtype=torch.double, device=device, requires_grad=True)
            # 如果 trans 为真，则对 inp 进行转置操作
            if trans:
                inp = inp.transpose(0, 1)
            # 遍历不同的距离参数 p
            for p in [0, 1, 2, 0.5, 1.5, 2.5, float('inf')]:
                # 断言对 lambda 函数 F.pdist 进行梯度检查通过
                self.assertTrue(gradcheck(lambda x: F.pdist(x, p), (inp,)))

    # 定义一个测试方法，测试当输入的距离为零时，F.pdist 函数的梯度是否有效
    def test_pdist_zeros(self):
        # 遍历所有设备
        for device in device_():
            # 创建形状为 (1, 3) 的双精度浮点类型的随机张量，并且要求计算梯度，重复两次
            inp = torch.randn(1, 3, dtype=torch.double, device=device, requires_grad=True).repeat([2, 1])
            # 遍历不同的距离参数 p
            for p in [0, 1, 2, 0.5, 1.5, 2.5, float('inf')]:
                # 断言对 lambda 函数 F.pdist 进行梯度检查通过
                self.assertTrue(gradcheck(lambda x: F.pdist(x, p), (inp,)))

    # 定义一个测试方法，用于验证当输入张量有空行时，F.pdist 函数的梯度检查
    def test_pdist_empty_row(self):
        # 遍历所有设备
        for device in device_():
            # 创建形状为 (1, 3) 的双精度浮点类型的随机张量，并且要求计算梯度
            inp = torch.randn(1, 3, dtype=torch.double, device=device, requires_grad=True)
            # 断言对 F.pdist 函数进行梯度检查通过
            self.assertTrue(gradcheck(F.pdist, (inp,)))

    # 定义一个测试方法，用于验证当输入张量有空列时，F.pdist 函数的梯度检查
    def test_pdist_empty_col(self):
        # 遍历所有设备
        for device in device_():
            # 创建形状为 (4, 0) 的双精度浮点类型的随机张量，并且要求计算梯度
            inp = torch.randn(4, 0, dtype=torch.double, device=device, requires_grad=True)
            # 断言对 F.pdist 函数进行梯度检查通过
            self.assertTrue(gradcheck(F.pdist, (inp,)))

    @unittest.expectedFailure
    # 定义一个测试函数，测试 torch.pdist 函数在 CPU 上的梯度梯度检查
    def test_pdist_cpu_gradgrad_unimplemented(self):
        # 生成一个大小为 (4, 5) 的随机张量，并声明需要计算梯度
        inp = torch.randn(4, 5, requires_grad=True)
        # 调用 gradgradcheck 函数，检查 torch.pdist 的梯度梯度
        gradgradcheck(F.pdist, (inp,))

    # 标记为预期失败的测试函数，测试 torch.pdist 函数在 CUDA 上的梯度梯度检查
    @unittest.expectedFailure
    def test_pdist_cuda_gradgrad_unimplemented(self):
        # 生成一个在 CUDA 上的大小为 (4, 5) 的随机张量，并声明需要计算梯度
        inp = torch.randn(4, 5, device='cuda', requires_grad=True)
        # 调用 gradgradcheck 函数，检查 torch.pdist 的梯度梯度
        gradgradcheck(F.pdist, (inp,))

    # 定义一个测试函数，用于测试 torch.pdist 函数在大规模数据上的运行情况
    def test_pdist_large(self):
        # 遍历不同的设备
        for device in device_():
            # 定义一个简单的函数，计算输入张量的 p=2 范数距离
            def func(x):
                return torch.pdist(x, p=2)

            # 设置一个大规模的张量形状 (1000, 1)
            shape = (1000, 1)
            # 生成一个在指定设备上的随机张量，并声明需要计算梯度
            x = torch.randn(shape, device=device).requires_grad_()
            # 计算张量 x 的 p=2 范数距离
            output = torch.pdist(x, p=2)
            # 对输出结果进行求和，并进行反向传播计算梯度
            output.sum().backward()

    # 定义一个测试函数，测试 torch.nn.functional.cosine_embedding_loss 函数在不同数据类型下的表现
    def test_cosine_embedding_loss_with_diff_type(self):
        # 遍历不同的设备
        for device in device_():
            # 定义两个输入张量和目标张量，分别为双精度浮点型和整型，在指定设备上
            input1 = torch.tensor([[2, 3, 4], [6, 2, 4]], dtype=torch.double, device=device)
            input2 = torch.tensor([[2, 3, 5], [3, 2, 1]], dtype=torch.double, device=device)
            target = torch.tensor([1, -1], dtype=torch.int, device=device)
            # 计算预期的 cosine embedding loss
            expected = torch.nn.functional.cosine_embedding_loss(input1, input2, target)
            # 遍历所有数学数据类型，测试函数的输入和目标张量的数据类型组合
            for dt1 in get_all_math_dtypes(device):
                for dt2 in get_all_math_dtypes(device):
                    for dt3 in get_all_math_dtypes(device):
                        # 跳过无符号整型，因为目标张量使用的是有符号整型
                        if dt3 == torch.uint8:
                            continue
                        # 跳过复数类型的数据
                        if dt1.is_complex or dt2.is_complex or dt3.is_complex:
                            continue
                        # 将输入张量和目标张量转换为当前的数据类型
                        input1 = input1.to(dt1)
                        input2 = input2.to(dt2)
                        target = target.to(dt3)
                        # 计算当前数据类型下的 cosine embedding loss
                        result = torch.nn.functional.cosine_embedding_loss(input1, input2, target)
                        # 使用断言检查计算结果是否与预期接近
                        self.assertEqual(result.item(), expected.item(), atol=0.001, rtol=0)

    # 定义一个测试函数，测试 torch.nn.functional.cosine_embedding_loss 函数对不同形状输入的错误处理
    def test_cosine_embedding_loss_error_on_diff_shapes(self):
        # 遍历不同的设备
        for device in device_():
            # 创建空的输入张量和目标张量，数据类型为双精度浮点型和整型，在指定设备上
            input1 = torch.empty((0, 0), dtype=torch.double, device=device)
            input2 = torch.empty((0,), dtype=torch.double, device=device)
            target = torch.empty((0,), dtype=torch.int, device=device)
            # 使用断言检查是否抛出 RuntimeError 异常，提示期望 2 维输入张量
            with self.assertRaisesRegex(RuntimeError, ".*expects 2D.*"):
                torch.nn.functional.cosine_embedding_loss(input1, input2, target)
    # 测试函数：验证在非可扩展形状上的余弦嵌入损失的错误
    def test_cosine_embedding_loss_error_on_nonexpandable_shapes(self):
        # 遍历所有设备类型
        for device in device_():
            # 创建一个空的张量 input1，形状为 (1, 5)，数据类型为双精度，放置在指定设备上
            input1 = torch.empty((1, 5), dtype=torch.double, device=device)
            # 创建一个空的张量 input2，形状为 (1, 6)，数据类型为双精度，放置在指定设备上
            input2 = torch.empty((1, 6), dtype=torch.double, device=device)
            # 创建目标张量 target，形状为 (1,)，数据类型为整数，放置在指定设备上
            target = torch.ones((1,), dtype=torch.int, device=device)
            # 使用断言检查是否抛出运行时错误，并验证错误消息是否包含"must match the size"
            with self.assertRaisesRegex(RuntimeError, ".*must match the size.*"):
                # 调用余弦嵌入损失函数，验证输入的张量形状不匹配时是否会引发异常
                torch.nn.functional.cosine_embedding_loss(input1, input2, target)

    # 测试函数：验证不同类型的 KL 散度计算
    def test_kl_div_with_diff_type(self):
        # 遍历所有设备类型
        for device in device_():
            # 创建输入张量 input，形状为 (2, 3)，数据类型为双精度，放置在指定设备上
            input = torch.tensor([[2, 3, 5], [3, 2, 1]], dtype=torch.double, device=device)
            # 创建目标张量 target，形状为 (2, 3)，数据类型为双精度，放置在指定设备上
            target = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.double, device=device)
            # 计算预期的 KL 散度
            expected = torch.nn.functional.kl_div(input, target)
            # 定义真实数据类型的组合
            real_dtypes = (torch.float32, torch.float64, torch.float16)
            # 对真实数据类型的组合进行排列组合
            for input_dtype, target_dtype in product(real_dtypes, repeat=2):
                # 如果设备类型为 CPU 并且目标数据类型为 torch.float16，则跳过当前循环
                if (torch.device(device).type == 'cpu' and target_dtype == torch.float16):
                    continue
                # 转换输入和目标张量的数据类型
                input = input.to(input_dtype)
                target = target.to(target_dtype)
                # 调用 KL 散度函数，计算结果
                result = torch.nn.functional.kl_div(input, target)
                # 使用断言验证计算结果与预期结果的近似程度
                self.assertEqual(result.item(), expected.item(), atol=0.001, rtol=0)

    # 测试函数：验证带有不同类型日志目标的 KL 散度计算
    def test_kl_div_with_diff_type_log_target(self):
        # 遍历所有设备类型
        for device in device_():
            # 创建输入张量 input，形状为 (2, 3)，数据类型为双精度，放置在指定设备上
            input = torch.tensor([[2, 3, 5], [3, 2, 1]], dtype=torch.double, device=device)
            # 创建目标张量 target，形状为 (2, 3)，数据类型为双精度，放置在指定设备上，并对目标张量取对数
            target = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.double, device=device).log()
            # 计算预期的 KL 散度，设定 log_target=True
            expected = torch.nn.functional.kl_div(input, target, log_target=True)
            # 定义真实数据类型的组合
            real_dtypes = (torch.float32, torch.float64, torch.float16)
            # 对真实数据类型的组合进行排列组合
            for input_dtype, target_dtype in product(real_dtypes, repeat=2):
                # 如果设备类型为 CPU 并且目标数据类型为 torch.float16，则跳过当前循环
                if (torch.device(device).type == 'cpu' and target_dtype == torch.float16):
                    continue
                # 转换输入和目标张量的数据类型
                input = input.to(input_dtype)
                target = target.to(target_dtype)
                # 调用 KL 散度函数，计算结果，设定 log_target=True
                result = torch.nn.functional.kl_div(input, target, log_target=True)
                # 使用断言验证计算结果与预期结果的近似程度
                self.assertEqual(result.item(), expected.item(), atol=0.001, rtol=0)

    # 测试函数：验证带有 log_softmax 目标的 KL 散度计算
    def test_kl_div_log_softmax_target(self):
        # 遍历所有设备类型
        for device in device_():
            # 创建张量 a，形状为 (2, 3)，放置在指定设备上
            a = torch.tensor([[1.0, 2, 3], [5.0, 5, 5]], device=device)
            # 创建张量 b，形状为 (2, 3)，放置在指定设备上
            b = torch.tensor([[1.0, 2, 3], [5.0, 5, 5]], device=device)
            # 使用断言验证 log_softmax 函数与 KL 散度函数计算结果是否为零张量
            self.assertEqual(
                F.kl_div(F.log_softmax(a, 1), F.log_softmax(b, 1), reduction='none', log_target=True),
                torch.zeros_like(a)
            )
    # 定义测试函数，用于测试不带降维的余弦嵌入损失函数
    def test_cosine_embedding_loss_no_reduce(self):
        # 创建随机张量作为输入1，形状为(15, 10)，需要梯度，数据类型为双精度
        input1 = torch.randn(15, 10, requires_grad=True, dtype=torch.double)
        # 创建随机张量作为输入2，形状为(15, 10)，需要梯度，数据类型为双精度
        input2 = torch.randn(15, 10, requires_grad=True, dtype=torch.double)
        # 创建随机张量作为目标，形状为(15)，数据类型为双精度，值为随机数的符号
        target = torch.randn(15, dtype=torch.double).sign()
        # 使用梯度检查确保函数的可导性，lambda函数定义了余弦嵌入损失函数，reduction参数为'none'
        self.assertTrue(gradcheck(lambda x, y, z: F.cosine_embedding_loss(
            x, y, z, reduction='none'), (input1, input2, target)))
        # 断言函数与参考函数的结果相等，用于验证余弦嵌入损失函数的计算正确性，reduction参数为'none'
        self.assertEqual(F.cosine_embedding_loss(input1, input2, target, reduction='none'),
                         loss_reference_fns['CosineEmbeddingLoss'](input1, input2, target, reduction='none'))

    # 定义测试函数，用于测试带边界(margin)的余弦嵌入损失函数，且不降维
    def test_cosine_embedding_loss_margin_no_reduce(self):
        # 创建随机张量作为输入1，形状为(15, 10)，需要梯度，数据类型为双精度
        input1 = torch.randn(15, 10, requires_grad=True, dtype=torch.double)
        # 创建随机张量作为输入2，形状为(15, 10)，需要梯度，数据类型为双精度
        input2 = torch.randn(15, 10, requires_grad=True, dtype=torch.double)
        # 创建随机张量作为目标，形状为(15)，数据类型为双精度，值为随机数的符号
        target = torch.randn(15, dtype=torch.double).sign()
        # 使用梯度检查确保函数的可导性，lambda函数定义了带边界(margin=0.5)的余弦嵌入损失函数，reduction参数为'none'
        self.assertTrue(gradcheck(lambda x, y, z: F.cosine_embedding_loss(
            x, y, z, margin=0.5, reduction='none'), (input1, input2, target)))
        # 断言函数与参考函数的结果相等，用于验证带边界(margin=0.5)的余弦嵌入损失函数的计算正确性，reduction参数为'none'
        self.assertEqual(F.cosine_embedding_loss(input1, input2, target, margin=0.5, reduction='none'),
                         loss_reference_fns['CosineEmbeddingLoss'](input1, input2, target,
                                                                   margin=0.5, reduction='none'))

    # 定义测试函数，用于测试输入张量形状不合法的余弦嵌入损失函数
    def test_cosine_embedding_loss_invalid_shape(self):
        # 创建随机张量作为输入1，形状为(15, 10)
        input1 = torch.randn(15, 10)
        # 创建随机张量作为输入2，形状为(15, 10)
        input2 = torch.randn(15, 10)
        # 创建随机张量作为目标，形状为(15, 1)，值为随机数的符号
        target = torch.randn(15, 1).sign()

        # 使用断言确保运行时错误被触发，提示信息为"1D target tensor expected"
        with self.assertRaisesRegex(RuntimeError, "1D target tensor expected"):
            F.cosine_embedding_loss(input1, input2, target)

        # 使用断言确保运行时错误被触发，提示信息为"1D target tensor expects 2D input tensors"
        with self.assertRaisesRegex(RuntimeError, "1D target tensor expects 2D input tensors"):
            F.cosine_embedding_loss(torch.randn(10), torch.randn(10), torch.randn(10))

        # 使用断言确保运行时错误被触发，提示信息为"0D target tensor expects 1D input tensors"
        with self.assertRaisesRegex(RuntimeError, "0D target tensor expects 1D input tensors"):
            F.cosine_embedding_loss(torch.randn(2, 5), torch.randn(2, 5), torch.randn(()))

    # 定义测试函数，用于测试不带降维的排名损失函数
    def test_margin_ranking_loss_no_reduce(self):
        # 创建随机张量作为输入1，形状为(15)，数据类型为双精度，每个元素乘以10，并需要梯度
        input1 = torch.randn(15, dtype=torch.double).mul_(10).requires_grad_()
        # 创建随机张量作为输入2，形状为(15)，数据类型为双精度，每个元素乘以10，并需要梯度
        input2 = torch.randn(15, dtype=torch.double).mul_(10).requires_grad_()
        # 创建随机张量作为目标，形状为(15)，数据类型为双精度，值为随机数的符号
        target = torch.randn(15, dtype=torch.double).sign()
        # 使用梯度检查确保函数的可导性，lambda函数定义了排名损失函数，reduction参数为'none'
        self.assertTrue(gradcheck(lambda x, y, z: F.margin_ranking_loss(
            x, y, z, reduction='none'), (input1, input2, target)))
        # 断言函数与参考函数的结果相等，用于验证排名损失函数的计算正确性，reduction参数为'none'
        self.assertEqual(F.margin_ranking_loss(input1, input2, target, reduction='none'),
                         loss_reference_fns['MarginRankingLoss'](input1, input2, target, reduction='none'))
    # 定义一个测试函数，用于测试 margin ranking loss 函数在不进行减少操作时的边界情况
    def test_margin_ranking_loss_margin_no_reduce(self):
        # 创建输入张量 input1 和 input2，形状为 (15,)，数据类型为 double，并且需要计算梯度
        input1 = torch.randn(15, dtype=torch.double).mul_(10).requires_grad_()
        # 创建输入张量 input2，形状和性质与 input1 相同
        input2 = torch.randn(15, dtype=torch.double).mul_(10).requires_grad_()
        # 创建目标张量 target，形状为 (15,)，数据类型为 double，值为随机数的符号
        target = torch.randn(15, dtype=torch.double).sign()
        # 使用 gradcheck 函数检查 lambda 函数的梯度，lambda 函数调用 F.margin_ranking_loss 函数
        self.assertTrue(gradcheck(lambda x, y, z: F.margin_ranking_loss(
            x, y, z, margin=0.5, reduction='none'), (input1, input2, target)))
        # 断言调用 F.margin_ranking_loss 函数的结果与参考函数 loss_reference_fns['MarginRankingLoss'] 相等
        self.assertEqual(F.margin_ranking_loss(input1, input2, target, margin=0.5, reduction='none'),
                         loss_reference_fns['MarginRankingLoss'](input1, input2, target, margin=0.5, reduction='none'))

    # 定义一个测试函数，用于测试 triplet margin loss 函数
    def test_triplet_margin_loss(self):
        # 创建输入张量 input1、input2 和 input3，形状为 (5, 10)，需要计算梯度，数据类型为 double
        input1 = torch.randn(5, 10, requires_grad=True, dtype=torch.double)
        input2 = torch.randn(5, 10, requires_grad=True, dtype=torch.double)
        input3 = torch.randn(5, 10, requires_grad=True, dtype=torch.double)
        # 使用 gradcheck 函数检查 lambda 函数的梯度，lambda 函数调用 F.triplet_margin_loss 函数
        self.assertTrue(gradcheck(lambda x1, x2, x3: F.triplet_margin_loss(
            x1, x2, x3), (input1, input2, input3)))
        # 断言调用 F.triplet_margin_loss 函数的结果与参考函数 loss_reference_fns['TripletMarginLoss'] 相等
        self.assertEqual(F.triplet_margin_loss(input1, input2, input3),
                         loss_reference_fns['TripletMarginLoss'](input1, input2, input3))

    # 定义一个测试函数，用于测试 triplet margin loss 函数中的 swap 参数
    def test_triplet_margin_loss_swap(self):
        # 创建输入张量 input1、input2 和 input3，形状为 (5, 10)，需要计算梯度，数据类型为 double
        input1 = torch.randn(5, 10, requires_grad=True, dtype=torch.double)
        input2 = torch.randn(5, 10, requires_grad=True, dtype=torch.double)
        input3 = torch.randn(5, 10, requires_grad=True, dtype=torch.double)
        # 使用 gradcheck 函数检查 lambda 函数的梯度，lambda 函数调用 F.triplet_margin_loss 函数，并设置 swap=True
        self.assertTrue(gradcheck(lambda x1, x2, x3: F.triplet_margin_loss(
            x1, x2, x3, swap=True), (input1, input2, input3)))
        # 断言调用 F.triplet_margin_loss 函数的结果与参考函数 loss_reference_fns['TripletMarginLoss'] 相等，并设置 swap=True
        self.assertEqual(F.triplet_margin_loss(input1, input2, input3, swap=True),
                         loss_reference_fns['TripletMarginLoss'](input1, input2, input3, swap=True))

    # 定义一个测试函数，用于测试 triplet margin loss 函数在不进行减少操作时的情况
    def test_triplet_margin_loss_no_reduce(self):
        # 创建输入张量 input1、input2 和 input3，形状为 (5, 10)，需要计算梯度，数据类型为 double
        input1 = torch.randn(5, 10, requires_grad=True, dtype=torch.double)
        input2 = torch.randn(5, 10, requires_grad=True, dtype=torch.double)
        input3 = torch.randn(5, 10, requires_grad=True, dtype=torch.double)
        # 使用 gradcheck 函数检查 lambda 函数的梯度，lambda 函数调用 F.triplet_margin_loss 函数，并设置 reduction='none'
        self.assertTrue(gradcheck(lambda x1, x2, x3: F.triplet_margin_loss(
            x1, x2, x3, reduction='none'), (input1, input2, input3)))
        # 断言调用 F.triplet_margin_loss 函数的结果与参考函数 loss_reference_fns['TripletMarginLoss'] 相等，并设置 reduction='none'
        self.assertEqual(F.triplet_margin_loss(input1, input2, input3, reduction='none'),
                         loss_reference_fns['TripletMarginLoss'](input1, input2, input3, reduction='none'))
    # 定义一个测试函数，用于测试 triplet_margin_loss 函数的 swap 和 no reduction 参数组合
    def test_triplet_margin_loss_swap_no_reduce(self):
        # 创建三个随机张量作为输入，需计算梯度，数据类型为双精度浮点型
        input1 = torch.randn(5, 10, requires_grad=True, dtype=torch.double)
        input2 = torch.randn(5, 10, requires_grad=True, dtype=torch.double)
        input3 = torch.randn(5, 10, requires_grad=True, dtype=torch.double)
        # 使用 gradcheck 验证 triplet_margin_loss 函数的梯度
        self.assertTrue(gradcheck(lambda x1, x2, x3: F.triplet_margin_loss(
            x1, x2, x3, swap=True, reduction='none'), (input1, input2, input3)))
        # 使用 loss_reference_fns 字典中的 TripletMarginLoss 函数验证 triplet_margin_loss 的结果
        self.assertEqual(F.triplet_margin_loss(input1, input2, input3, swap=True, reduction='none'),
                         loss_reference_fns['TripletMarginLoss'](input1, input2, input3, swap=True, reduction='none'))

    # 测试 pointwise loss 函数的目标梯度和减少参数设置为 'none' 的情况
    def test_pointwise_loss_target_grad_none_reduction(self):
        # 创建两个随机张量，一个不需要梯度，一个需要梯度
        i = torch.randn(5, 10)
        t = torch.randn(5, 10, requires_grad=True)
        # 验证 mse_loss 和 l1_loss 的 'none' 减少参数，输出大小应与目标张量 t 的大小相同
        self.assertEqual(F.mse_loss(i, t, reduction='none').size(), t.size())
        self.assertEqual(F.l1_loss(i, t, reduction='none').size(), t.size())

    # 测试 pointwise loss 函数的广播特性
    def test_pointwise_loss_broadcast(self):
        # 定义包含不同损失函数的字典
        losses = {
            'mse_loss': lambda x, y, r: F.mse_loss(x, y, reduction=r),
            'l1_loss': lambda x, y, r: F.l1_loss(x, y, reduction=r),
            'smooth_l1_loss': lambda x, y, r: F.smooth_l1_loss(x, y, reduction=r),
            'huber_loss': lambda x, y, r: F.huber_loss(x, y, reduction=r),
        }
        # 创建一个双精度浮点型的随机输入张量
        input = torch.randn(2, 1, requires_grad=True, dtype=torch.double)
        # 遍历损失函数字典
        for fn in losses.values():
            # 遍历目标张量是否需要梯度的情况
            for requires_grad in [True, False]:
                # 当目标张量 target.requires_grad=True 时，其实现在 Python 中，而另一个在 TH 中
                target = torch.randn(2, 10, requires_grad=requires_grad, dtype=torch.double)
                # 遍历减少参数的情况
                for reduction in ['none', 'mean', 'sum']:
                    # 计算损失 l
                    l = fn(input, target, reduction)
                    # 如果减少参数为 'none'，验证 l 的大小应与目标张量 target 的大小相同
                    if reduction == 'none':
                        self.assertEqual(l.size(), target.size())
                    # 使用 gradcheck 验证损失函数的梯度
                    self.assertTrue(gradcheck(fn, (input, target, reduction)))

    # 测试 L1 损失函数在大批量大小下的正确性
    # https://github.com/pytorch/pytorch/issues/27692 报告 l1_loss 在大批量大小下得到错误结果的问题
    def test_l1_loss_correct(self):
        # 遍历数据类型和批量大小范围
        for dtype in [torch.float, torch.cfloat]:
            for N in range(1, 50, 10):
                # 创建具有指定数据类型和大小的随机输入张量
                input = torch.rand(N, 3, 1024, 1024, dtype=dtype)
                # 验证 L1Loss 函数对输入张量和全零张量的输出，应为输入张量的绝对值的均值
                self.assertEqual(
                    torch.nn.L1Loss()(input, torch.zeros_like(input)),
                    input.abs().mean())
    # 定义测试函数，验证 smooth_l1_loss 对于整数类型的目标的梯度计算是否正确
    def test_smoothl1loss_intergral_target(self):
        
        # 定义内部函数 _input_grad，计算 smooth_l1_loss 的梯度并返回输入的梯度
        def _input_grad(input, target, reduction):
            # 计算 smooth_l1_loss，并设置 beta 参数为 0.5
            output = F.smooth_l1_loss(input, target, reduction=reduction, beta=0.5)
            # 对 loss 进行求和并反向传播
            output.sum().backward()
            # 返回输入的梯度
            return input.grad

        # 使用 product 生成设备、数据类型和减少方式的组合
        for device, dtype, reduction in product(device_(),
                                                integral_types(),
                                                ('none', 'sum', 'mean')):
            # 创建随机输入张量和整数类型的目标张量
            input = torch.randn(2, 2, device=device, requires_grad=True)
            target = torch.randint(0, 9, (2, 2), device=device, dtype=dtype)

            # 使用浮点类型的目标计算梯度
            input_grad_with_float_target = _input_grad(input, target.float(), reduction)

            # 使用整数类型的目标计算梯度
            input_grad = _input_grad(input.detach().clone().requires_grad_(True),
                                     target,
                                     reduction)
            
            # 断言两种方式计算得到的梯度相等
            self.assertEqual(input_grad, input_grad_with_float_target)

    # 测试当 beta 参数为负数时，smooth_l1_loss 是否会引发 RuntimeError
    def test_smoothl1loss_negative_beta_not_supported(self):
        # 使用 assertRaises 确认运行时异常被引发
        with self.assertRaises(RuntimeError):
            F.smooth_l1_loss(torch.randn(2, 2), torch.randn(2, 2), beta=-1.0)

    # 测试当 delta 参数为负数或零时，HuberLoss 是否会引发 RuntimeError
    def test_huber_loss_invalid_delta(self):
        
        # 定义内部辅助函数 _test_huber_loss_delta_error_helper，用于测试不同的 delta 值
        def _test_huber_loss_delta_error_helper(delta):
            # 创建随机输入张量和目标张量
            input, target = torch.randn(2, 2), torch.randn(2, 2)
            # 创建 HuberLoss 实例，设置 delta 参数
            loss = torch.nn.HuberLoss(delta=delta)
            # 使用 assertRaises 确认运行时异常被引发
            with self.assertRaises(RuntimeError):
                loss(input, target)

        # 定义测试负 delta 值的函数
        def test_huber_loss_negative_delta():
            _test_huber_loss_delta_error_helper(delta=-0.5)

        # 定义测试零 delta 值的函数
        def test_huber_loss_zero_delta():
            _test_huber_loss_delta_error_helper(delta=0.0)

        # 执行负 delta 值的测试
        test_huber_loss_negative_delta()
        # 执行零 delta 值的测试
        test_huber_loss_zero_delta()

    # 将默认的张量数据类型设置为 torch.double
    @set_default_dtype(torch.double)
    # 定义测试方法：检查余弦相似度函数的输入输出形状是否正确
    def test_cosine_similarity(self):
        # 定义输入的大小
        input_size = (1, 3, 2, 1)
        # 定义预期的输出大小
        expected_size = (1, 2, 1)
        # 生成随机张量作为输入1和输入2，并标记需要计算梯度
        input1 = torch.randn(input_size, requires_grad=True)
        input2 = torch.randn(input_size, requires_grad=True)
        # 使用 torch 的余弦相似度函数计算输入1和输入2在维度1上的余弦相似度，并检查输出大小是否符合预期
        self.assertEqual(F.cosine_similarity(input1, input2, dim=1).size(), expected_size)

        # 检查数值精度，解决问题 #18057
        # 创建两个包含从0到83的浮点数的张量，并在0维度上增加一个维度
        vv1 = torch.tensor([float(i) for i in range(84)]).unsqueeze(0)
        vv2 = torch.tensor([float(i) for i in range(84)]).unsqueeze(0)
        # 计算两个张量的余弦相似度
        out = F.cosine_similarity(vv1, vv2)
        # 检查计算的余弦相似度是否小于等于1.0
        self.assertLessEqual(out, 1.0)

        # 检查除以0的情况
        # 先前的行为: <x,y>/max(eps, ||x|| * ||y||)
        # 现在的行为: <x/max(eps, ||x||), y/max(eps,||y||)>
        # 如果 f(x,y) 是余弦相似度，则
        # df/dx = y/(||x|| * ||y||) - (x * <x,y> * ||y||/||x||)/(||x|| * ||y||)^2
        # 下面的测试检查在反向传播公式中当 x := input2 = 0, y := input1 != 0 时除以零的情况
        # 对于这些输入，关于 x 的梯度简化为 g(x,y) := y/(||x|| * ||y||)
        # 先前的测试检查 g(x,y) == y/eps，
        # 现在的测试检查 g(x,y) == (y/||y||)/eps。
        # 检查 input1 和 input2 张量的梯度是否如预期一样
        input1 = torch.randn(10).requires_grad_()
        input2 = torch.zeros_like(input1).requires_grad_()
        torch.cosine_similarity(input1, input2, 0).sum().backward()
        self.assertEqual(input1.grad, torch.zeros_like(input1))
        self.assertEqual(input2.grad, input1 / input1.norm() * 1e8)

        # 检查类型提升，解决问题 #61454
        # 创建一个标量张量 input，并计算其与转换为 torch.int8 类型的 input 的余弦相似度
        input = torch.tensor(12.)
        out = F.cosine_similarity(input.to(torch.int8), input, dim=-1)
        # 检查计算结果是否等于1.0
        self.assertEqual(out, 1.)

        # 检查广播功能 #109333
        # 创建两个形状为 (2, 3) 的全1张量 a 和形状为 (1,) 的全1张量 b，并计算它们的余弦相似度
        a = torch.ones(2, 3, dtype=torch.float)
        b = torch.ones(1, 1, dtype=torch.float)
        out = F.cosine_similarity(a, b)
        # 检查计算结果是否等于形状为 (2,) 的全1张量
        self.assertEqual(out, torch.ones(2, dtype=torch.float))

        # 创建形状为 (2, 3) 的全1张量 a 和形状为 (1,) 的全1张量 b，并计算它们的余弦相似度
        a = torch.ones(2, 3, dtype=torch.float)
        b = torch.ones(1, dtype=torch.float)
        out = F.cosine_similarity(a, b)
        # 检查计算结果是否等于形状为 (2,) 的全1张量
        self.assertEqual(out, torch.ones(2, dtype=torch.float))
    # 定义一个测试方法，用于检查 grid_sample 函数的错误处理能力
    def test_grid_sample_error_checking(self):
        # 创建一个空的 2x2 的张量作为输入
        input = torch.empty(1, 1, 2, 2)
        # 创建一个空的 1x2 的张量作为网格参数
        grid = torch.empty(1, 1, 1, 2)

        # 测试没有错误的情况，使用 grid_sample 函数对输入和网格进行采样，不进行角点对齐
        F.grid_sample(input, grid, align_corners=False)

        # 测试 mode 参数为 'garbage' 时是否会引发 ValueError 异常
        with self.assertRaisesRegex(ValueError, "but got: 'garbage'"):
            F.grid_sample(input, grid, mode='garbage', align_corners=False)

        # 测试 padding_mode 参数为 'garbage' 时是否会引发 ValueError 异常
        with self.assertRaisesRegex(ValueError, "but got: 'garbage'"):
            F.grid_sample(input, grid, padding_mode='garbage', align_corners=False)

        # 测试输入的最后一维度不为 2 时是否会引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "expected grid to have size 1 in last dimension"):
            F.grid_sample(input[0], grid, align_corners=False)

        # 测试网格的最后一维度不为 2 时是否会引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "expected grid to have size 2 in last dimension"):
            F.grid_sample(input, torch.empty(1, 1, 1, 1, 3), align_corners=False)

        # 测试输入和网格的批处理大小不同是否会引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "expected grid and input to have same batch size"):
            F.grid_sample(input, torch.empty(2, 1, 1, 2), align_corners=False)

        # 测试网格的最后一维度不为 2 时是否会引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "expected grid to have size 2 in last dimension"):
            F.grid_sample(input, torch.empty(1, 1, 1, 3), align_corners=False)

        # 测试输入的空间维度为零时是否会引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "expected input to have non-empty spatial dimensions"):
            F.grid_sample(torch.empty(1, 1, 0, 2), grid, align_corners=False)

        # 测试 bicubic 插值仅支持 4D 输入时是否会引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "bicubic interpolation only supports 4D input"):
            F.grid_sample(torch.empty(1, 1, 2, 2, 2), torch.empty(1, 1, 1, 1, 3), mode='bicubic')

        # 如果支持 CUDA 测试，检查所有张量是否位于同一设备上时是否会引发 RuntimeError 异常
        if TEST_CUDA:
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                F.grid_sample(input.cuda(), grid, align_corners=False)

    # 使用参数化测试，测试 affine_grid 的反向传播时通道顺序的一致性问题
    @parametrize_test('device', ['cpu'] + (['cuda'] if TEST_CUDA else []))
    @parametrize_test('nd', [2, 3])
    def test_affine_grid_backward_cl_cf_consistency(self, device, nd):
        # 根据报告的问题进行测试：https://github.com/pytorch/pytorch/issues/124154

        # 创建一个随机的 theta 张量，用于仿射变换，要求梯度，指定设备类型
        theta = torch.rand([6, nd, nd + 1], requires_grad=True, device=device)
        # 根据不同的维度创建 size 参数
        size = [6, 3, 4, 5] if nd == 2 else [6, 3, 4, 5, 5]
        # 使用 affine_grid 函数根据 theta 和 size 创建网格
        grid = torch.nn.functional.affine_grid(theta, size, align_corners=False)

        # 创建一个与 grid 相同形状的随机梯度张量
        grad_tensor = torch.rand(grid.shape, device=device)

        # 根据通道顺序创建 channels_last 内存格式的梯度张量
        memory_format_cl = torch.channels_last if nd == 2 else torch.channels_last_3d
        grad_tensor_cl = grad_tensor.contiguous(memory_format=memory_format_cl)

        # 断言 theta 的梯度为 None
        assert theta.grad is None
        # 对 channels_last 内存格式的梯度进行反向传播
        grid.backward(grad_tensor_cl)
        # 克隆并转换成连续的 theta 梯度
        theta_grad_cl = theta.grad.clone().contiguous()

        # 将 theta 的梯度重置为零
        theta.grad.zero_()
        # 对 channels_first 内存格式的梯度进行反向传播
        grid.backward(grad_tensor)
        # 获取 channels_first 内存格式的 theta 梯度
        theta_grad_cf = theta.grad

        # 断言 channels_first 内存格式和 channels_last 内存格式的 theta 梯度相等
        self.assertEqual(theta_grad_cf, theta_grad_cl)

    # 使用 double 类型设置默认的数据类型，三次调用确保设置有效
    @set_default_dtype(torch.double)
    @set_default_dtype(torch.double)
    @set_default_dtype(torch.double)
    def test_affine_grid(self):
        # 定义测试函数 test_affine_grid，用于测试 F.affine_grid 函数的不同输入情况

        # 测试已知输入在 CPU 上的结果
        input = torch.arange(1., 7).view(1, 2, 3)
        output = F.affine_grid(input, torch.Size([1, 1, 2, 2]), align_corners=True)
        groundtruth = torch.tensor(
            [[[0., -3.], [2., 5.]], [[4., 7.], [6., 15.]]]).view(1, 2, 2, 2)
        self.assertEqual(output, groundtruth)

        # 再次测试 align_corners=False 的情况
        output = F.affine_grid(input, torch.Size([1, 1, 2, 2]), align_corners=False)
        groundtruth = torch.tensor(
            [[[1.5, 1.5], [2.5, 5.5]], [[3.5, 6.5], [4.5, 10.5]]]).view(1, 2, 2, 2)
        self.assertEqual(output, groundtruth)

        # 对 align_corners=True 和 align_corners=False 都进行梯度检查
        for align_corners in (True, False):
            N = random.randint(1, 8)
            C = random.randint(1, 8)
            H = random.randint(1, 8)
            W = random.randint(1, 8)
            sz = torch.Size([N, C, H, W])
            inp = torch.randn(N, 2, 3, requires_grad=True)
            
            # 使用 gradcheck 函数检查梯度
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")  # python2 需要这个设置以便其他测试可以触发
                self.assertTrue(gradcheck(
                    lambda inp: F.affine_grid(inp, sz, align_corners=align_corners),
                    (inp,)))

        # 在 CPU 和 CUDA 上进行测试比较
        if TEST_CUDA:
            N = random.randint(1, 8)
            C = random.randint(1, 8)
            H = random.randint(1, 8)
            W = random.randint(1, 8)
            sz = torch.Size([N, C, H, W])
            
            # 对 align_corners=True 和 align_corners=False 都进行测试
            for align_corners in (True, False):
                input_cpu = torch.randn(N, 2, 3, requires_grad=True)
                
                # 使用 warnings 捕捉警告
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")  # python2 需要这个设置以便其他测试可以触发
                    out_cpu = F.affine_grid(input_cpu, sz, align_corners=align_corners)
                
                gradients = torch.randn(out_cpu.size())
                out_cpu.backward(gradients)
                
                # 将 input_cpu 转移到 GPU 上测试
                input_gpu = input_cpu.detach().cuda().requires_grad_()
                
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")  # python2 需要这个设置以便其他测试可以触发
                    out_cuda = F.affine_grid(input_gpu, sz, align_corners=align_corners)
                
                out_cuda.backward(gradients.cuda())
                
                # 断言 CPU 和 CUDA 上的输出应该相等
                self.assertEqual(out_cpu, out_cuda)
                self.assertEqual(input_cpu.grad, input_gpu.grad)

    @set_default_dtype(torch.double)
    def test_affine_grid_3d(self):
        # 在 CPU 上测试已知输入数据
        input = torch.arange(1., 13).view(1, 3, 4)
        # 使用 F.affine_grid 函数生成输出，指定输出尺寸和 align_corners=True
        output = F.affine_grid(input, torch.Size([1, 1, 2, 2, 2]), align_corners=True)
        # 预期的输出结果
        groundtruth = torch.tensor(
            [[[[[-2., -10., -18.], [0., 0., 0.]], [[2., 2., 2.], [4., 12., 20.]]],
              [[[4., 4., 4.], [6., 14., 22.]], [[8., 16., 24.], [10., 26., 42.]]]]]).view(1, 2, 2, 2, 3)
        # 使用断言检查输出与预期结果是否相等
        self.assertEqual(output, groundtruth)
        
        # 使用 F.affine_grid 函数生成输出，指定输出尺寸和 align_corners=False
        output = F.affine_grid(input, torch.Size([1, 1, 2, 2, 2]), align_corners=False)
        # 预期的输出结果
        groundtruth = torch.tensor(
            [[[[[1., -1., -3.], [2., 4., 6.]], [[3., 5., 7.], [4., 10., 16.]]],
              [[[4., 6., 8.], [5., 11., 17.]], [[6., 12., 18.], [7., 17., 27.]]]]]).view(1, 2, 2, 2, 3)
        # 使用断言检查输出与预期结果是否相等
        self.assertEqual(output, groundtruth)

        # 对于 align_corners=True 和 align_corners=False，进行梯度检查
        for align_corners in (True, False):
            # 随机生成张量的尺寸
            N = random.randint(1, 8)
            C = random.randint(1, 8)
            D = random.randint(1, 8)
            H = random.randint(1, 8)
            W = random.randint(1, 8)
            sz = torch.Size([N, C, D, H, W])
            # 随机生成具有梯度的输入张量
            inp = torch.randn(N, 3, 4, requires_grad=True)
            # 捕获可能的警告信息
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")  # python2 需要这个设置以便其他测试可以触发
                # 使用 gradcheck 函数检查梯度
                self.assertTrue(gradcheck(
                    lambda inp: F.affine_grid(inp, sz, align_corners=align_corners),
                    (inp,)))

        # 在 CPU 和 CUDA 上进行测试比较
        if TEST_CUDA:
            # 随机生成张量的尺寸
            N = random.randint(1, 8)
            C = random.randint(1, 8)
            D = random.randint(1, 8)
            H = random.randint(1, 8)
            W = random.randint(1, 8)
            sz = torch.Size([N, C, D, H, W])
            for align_corners in (True, False):
                # 随机生成具有梯度的输入张量在 CPU 上
                input_cpu = torch.randn(N, 3, 4, requires_grad=True)
                # 捕获可能的警告信息
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")  # python2 需要这个设置以便其他测试可以触发
                    # 使用 F.affine_grid 函数在 CPU 上生成输出
                    out_cpu = F.affine_grid(input_cpu, sz, align_corners=align_corners)
                # 随机生成相同尺寸的梯度张量
                gradients = torch.randn(out_cpu.size())
                # 计算 CPU 输出的梯度
                out_cpu.backward(gradients)
                # 将 CPU 上的输入张量转移到 GPU，并确保需要梯度
                input_gpu = input_cpu.detach().cuda().requires_grad_()
                # 捕获可能的警告信息
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")  # python2 需要这个设置以便其他测试可以触发
                    # 使用 F.affine_grid 函数在 CUDA 上生成输出
                    out_cuda = F.affine_grid(input_gpu, sz, align_corners=align_corners)
                # 在 CUDA 上计算输出的梯度
                out_cuda.backward(gradients.cuda())
                # 使用断言检查 CPU 和 CUDA 输出是否相等
                self.assertEqual(out_cpu, out_cuda)
                # 使用断言检查 CPU 上输入的梯度是否等于 GPU 上输入的梯度
                self.assertEqual(input_cpu.grad, input_gpu.grad)
    def test_channel_shuffle_return_alias_of_self(self):
        # 测试通道洗牌函数的返回是否是输入张量的别名，当输入张量为空时
        # 定义洗牌的分组数
        groups = 3
        # 创建一个空的输入张量
        input_tensor = torch.rand([0, 9, 4, 4])
        # 对输入张量应用通道洗牌操作
        output = torch.nn.ChannelShuffle(groups)(input_tensor)
        # 断言输出张量和输入张量应该相等
        torch.testing.assert_close(output, input_tensor)

    @skipIfTorchDynamo("TorchDynamo fails here for unknown reasons")
    def test_native_channel_shuffle_return_alias_of_self(self):
        # 测试本地的通道洗牌函数的返回是否是输入张量的别名
        # 定义洗牌的分组数
        groups = 3
        # 创建一个空的输入张量
        input_tensor = torch.rand([0, 9, 4, 4])
        # 调用本地的通道洗牌函数
        output = torch.native_channel_shuffle(input_tensor, groups)
        # 断言输出张量和输入张量应该相等
        torch.testing.assert_close(output, input_tensor)

    @set_default_dtype(torch.double)
    def test_upsamplingLinear1d(self):
        # 测试线性插值的一维上采样操作
        # 针对是否对齐角点和是否重新计算缩放因子进行循环测试
        for align_corners in [True, False]:
            for recompute_scale_factor in [True, False]:
                # 设置线性插值参数
                kwargs = dict(
                    mode='linear', align_corners=align_corners, recompute_scale_factor=recompute_scale_factor
                )
                # 测试不同的浮点数缩放因子进行上下采样
                for scale_factor in [0.5, 1.5, 2]:
                    # 创建线性插值对象
                    m = nn.Upsample(scale_factor=scale_factor, **kwargs)
                    # 创建输入张量
                    in_t = torch.ones(1, 1, 2)
                    # 计算预期的输出尺寸
                    out_size = int(math.floor(in_t.shape[-1] * scale_factor))
                    # 在捕获警告时测试线性插值函数
                    with warnings.catch_warnings(record=True) as w:
                        out_t = m(in_t)
                    # 断言输出张量的数据应为全1
                    self.assertEqual(torch.ones(1, 1, out_size), out_t.data)

                    # 创建具有梯度的随机输入张量
                    input = torch.randn(1, 1, 2, requires_grad=True)
                    # 根据是否重新计算缩放因子进行梯度检查
                    if not recompute_scale_factor:
                        gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), (input,))
                    else:
                        gradcheck(lambda x: F.interpolate(x, scale_factor=scale_factor, **kwargs), (input,))

    def test_upsamplingLinear1d_spatial_invariance(self):
        # 测试线性插值一维上采样的空间不变性
        # 创建线性插值对象，设置参数
        m = nn.Upsample(scale_factor=3, mode='linear', align_corners=False)
        # 创建长度为9的零张量输入
        in_t_9 = torch.zeros(1, 1, 9)
        # 随机初始化部分输入张量
        in_t_9[:, :, :4].normal_()
        # 在捕获警告时测试线性插值函数
        with warnings.catch_warnings(record=True) as w:
            out_t_9 = m(in_t_9)
            out_t_5 = m(in_t_9[:, :, :5])
        # 断言输出张量的前15个元素应该相等
        self.assertEqual(out_t_9[:, :, :15], out_t_5)

    @set_default_dtype(torch.double)
    def test_upsampling_not_recompute_scale_factor(self):
        # 测试输出与已知输入是否匹配：结果必须与 OpenCV 相符合
        in_t = torch.arange(8.).view(1, 2, 2, 2)
        # 期望的输出张量
        expected_out_t = torch.tensor(
            [[[[-0.32725, -0.08843, 0.37933, 0.79744],
              [0.15039, 0.38921, 0.85697, 1.27508],
              [1.08591, 1.32473, 1.79249, 2.21060],
              [1.92213, 2.16095, 2.62871, 3.04682]],

             [[3.67275, 3.91157, 4.37933, 4.79744],
              [4.15039, 4.38921, 4.85697, 5.27508],
              [5.08591, 5.32473, 5.79249, 6.21060],
              [5.92213, 6.16095, 6.62871, 7.04682]]]])
        if IS_PPC:
            # 在 PPC 上，OpenCV 和 PyTorch 的结果略有不同
            expected_out_t = torch.tensor(
                [[[[-0.32725, -0.08843, 0.37933, 0.79744],
                  [0.15039, 0.38921, 0.85697, 1.27508],
                  [1.08591, 1.32473, 1.79249, 2.21060],
                  [1.92212, 2.16094, 2.62870, 3.04681]],

                 [[3.67275, 3.91157, 4.37933, 4.79743],
                  [4.15039, 4.38921, 4.85697, 5.27508],
                  [5.08591, 5.32473, 5.79249, 6.21059],
                  [5.92212, 6.16094, 6.62870, 7.04680]]]])
        # 使用 F.interpolate 进行上采样，不重新计算比例因子
        out_t = F.interpolate(in_t, scale_factor=2.3, mode='bicubic', align_corners=False, recompute_scale_factor=False)
        # 设置打印精度
        torch.set_printoptions(precision=5)
        # 断言输出张量与期望的输出张量是否接近
        self.assertEqual(out_t, expected_out_t, atol=1e-4, rtol=0)

        device_list = ['cpu']
        if TEST_CUDA:
            device_list.append('cuda')

        for align_corners in [True, False]:
            kwargs = dict(mode='bicubic', align_corners=align_corners)
            # 测试浮点比例因子的上采样和下采样
            for device in device_list:
                for scale_factor in [0.6, 1.6, 2.3]:
                    # 创建大小为 (2, 2, 2, 2) 的输入张量并移到指定设备
                    in_t = torch.ones(2, 2, 2, 2).to(device)
                    # 使用 F.interpolate 进行上采样
                    out_t = F.interpolate(in_t, scale_factor=scale_factor, **kwargs)
                    # 计算输出大小
                    out_size = int(math.floor(in_t.shape[-1] * scale_factor))
                    # 断言输出张量与预期的全一张量是否接近
                    self.assertEqual(torch.ones(2, 2, out_size, out_size), out_t.data, atol=1e-5, rtol=0)

                    # 创建具有梯度的输入张量
                    input = torch.randn(2, 2, 2, 2, requires_grad=True)
                    # 使用 gradcheck 验证插值函数的梯度
                    gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [input])

    def test_upsamplingBilinear2d_spatial_invariance(self):
        # 创建双线性上采样层对象 m
        m = nn.Upsample(scale_factor=3, mode='bilinear', align_corners=False)
        # 创建大小为 (1, 1, 9, 9) 的输入张量，并随机初始化部分子区域
        in_t_9 = torch.zeros(1, 1, 9, 9)
        in_t_9[:, :, :4, :4].normal_()
        # 使用 m 进行双线性上采样
        with warnings.catch_warnings(record=True) as w:
            out_t_9 = m(in_t_9)
            # 对部分子区域进行双线性上采样
            out_t_5 = m(in_t_9[:, :, :5, :5])
        # 断言输出张量的部分是否相等
        self.assertEqual(out_t_9[:, :, :15, :15], out_t_5)
    # 定义一个测试函数，用于测试三维三线性插值的空间不变性
    def test_upsamplingTrilinear3d_spatial_invariance(self):
        # 创建一个三维上采样对象，缩放因子为3，插值模式为三线性，关闭角落对齐
        m = nn.Upsample(scale_factor=3, mode='trilinear', align_corners=False)
        # 创建一个形状为 (1, 1, 9, 9, 9) 的零张量
        in_t_9 = torch.zeros(1, 1, 9, 9, 9)
        # 在部分区域填充正态分布随机值
        in_t_9[:, :, :4, :4, :4].normal_()
        # 使用上采样对象 m 对输入张量进行插值操作，生成输出张量 out_t_9
        with warnings.catch_warnings(record=True) as w:
            out_t_9 = m(in_t_9)
            # 对输入张量的部分区域进行插值操作，生成输出张量 out_t_5
            out_t_5 = m(in_t_9[:, :, :5, :5, :5])
        # 断言两个输出张量的前部分相等
        self.assertEqual(out_t_9[:, :, :15, :15, :15], out_t_5)

    # 定义一个测试函数，用于测试小规模尺度的上采样
    def test_upsampling_small_scale(self):
        # 创建一个尺度因子为0.5的双线性上采样对象
        m = torch.nn.Upsample(scale_factor=0.5, mode="bilinear")
        # 创建一个形状为 (1, 1, 2, 2) 的张量，内容为从1到4的序列
        in_t = torch.arange(1, 5, dtype=torch.get_default_dtype()).reshape(1, 1, 2, 2)
        # 使用上采样对象 m 对输入张量进行插值操作，生成输出张量 out_t
        out_t = m(in_t)
        # 预期的输出张量，应为 [[[[2.5]]]]
        expected_out_t = torch.tensor([[[[2.5]]]])
        # 断言预期的输出张量与实际输出张量相等
        self.assertEqual(expected_out_t, out_t)
    # 定义一个测试函数，用于测试 bfloat16 数据类型的上采样功能
    def test_upsampling_bfloat16(self, dtype=torch.bfloat16):
        # 定义一个辅助函数，用于执行具体的上采样测试
        def helper(size, scale_factor, mode, device, memory_format=torch.contiguous_format):
            # 生成指定大小的随机张量作为输入数据，指定设备和数据类型，并将其从计算图中分离并设置为需要梯度
            input = torch.randn(size, device=device, dtype=dtype).to(memory_format=memory_format).detach().requires_grad_(True)
            # 将输入张量转换为 float32 类型，并指定内存格式为连续，也将其从计算图中分离并设置为需要梯度
            inputf = input.to(torch.float32).to(memory_format=torch.contiguous_format).detach().requires_grad_(True)
            # 创建一个上采样模块对象，指定上采样的比例和模式
            m = nn.Upsample(scale_factor=scale_factor, mode=mode)

            # 对 float32 类型的输入数据进行上采样操作
            outf = m(inputf)
            # 对 bfloat16 类型的输入数据进行上采样操作
            out = m(input)
            # 使用断言检查上采样后的输出是否与预期的 float32 类型输出相等，指定允许的绝对误差和相对误差
            self.assertEqual(out.to(torch.float32), outf, atol=0.05, rtol=0)

            # 生成指定大小的随机梯度输入张量，与输入数据形状相同，并指定设备和数据类型
            ginput = torch.randn(out.shape, device=device, dtype=dtype).to(memory_format=memory_format)
            # 将梯度输入张量转换为 float32 类型，并指定内存格式为连续
            ginputf = ginput.to(torch.float32).to(memory_format=torch.contiguous_format)
            # 对上采样操作后的输出进行反向传播，计算梯度
            out.backward(ginput)
            # 对 float32 类型的输出进行反向传播，计算梯度
            outf.backward(ginputf)
            # 使用断言检查 bfloat16 类型输入的梯度是否与 float32 类型输入的梯度相等，指定允许的绝对误差和相对误差
            self.assertEqual(input.grad.to(torch.float32), inputf.grad, atol=0.01, rtol=0.01)

        # 遍历设备列表，对每种设备执行上采样测试
        for device in ['cpu']:
            # 调用辅助函数进行不同参数设置下的上采样测试
            helper([3, 20, 11, 7], 2, 'nearest', device)
            helper([3, 20, 11, 7], 2, 'nearest', device, torch.channels_last)
            helper([3, 20, 11, 7, 3], 2, 'nearest', device)
            helper([3, 20, 30], 2, 'linear', device)
            helper([3, 20, 11, 7], 2, 'bilinear', device)
            helper([3, 20, 11, 7], 2, 'bilinear', device, torch.channels_last)
            helper([1, 3, 11, 7], 2, 'bicubic', device)
            helper([1, 3, 11, 7], 2, 'bicubic', device, torch.channels_last)
            helper([3, 20, 11, 7, 3], 2, 'trilinear', device)

            helper([3, 5, 5], 257., 'nearest', device)
            helper([3, 20, 11, 7], 20, 'nearest', device)
            helper([3, 20, 11, 7, 3], 20, 'nearest', device)
            helper([1, 2, 11, 7], 257, 'nearest', device, torch.channels_last)
            helper([1, 2, 2000, 2000], 1 / 377., 'nearest', device)
            helper([1, 2, 2000, 2000], 1 / 257., 'nearest', device, torch.channels_last)
            helper([3, 2, 11, 7, 3], 20, 'nearest', device, torch.channels_last_3d)
            helper([3, 5, 5], 10, 'linear', device)
            helper([3, 5, 5], 257, 'linear', device)
            helper([1, 2, 11, 7], 257, 'bilinear', device)
            helper([1, 2, 11, 7], 257, 'bilinear', device, torch.channels_last)
            helper([1, 3, 11, 7], 10, 'bicubic', device)
            helper([1, 3, 11, 7], 10, 'bicubic', device, torch.channels_last)
            helper([1, 1, 11, 7], 257, 'bicubic', device)
            helper([3, 2, 11, 7, 3], 20, 'trilinear', device)
            helper([3, 2, 11, 7, 3], 20, 'trilinear', device, torch.channels_last_3d)

    # 如果 CUDA 不可用，则跳过这个测试
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    # 定义测试函数，检验非法内存访问的插值操作
    def test_interpolate_illegal_memory_access(self):
        # 设置输入和输出的尺寸
        in_s = 45
        out_s = 14

        # 创建需要梯度的 CUDA 张量输入
        input = torch.ones((1, 1, in_s), device='cuda', requires_grad=True)
        # 分配比输出尺寸更大的梯度，以便在梯度计算中观察越界访问
        grad = torch.ones((1, 1, out_s * 2), device='cuda', requires_grad=True)
        grad = grad[:, :, :out_s]

        # 生成不需要梯度的 CPU 引用版本
        input_ref = input.detach().cpu().requires_grad_()
        grad_ref = grad.cpu()

        # 执行插值操作，使用最近邻模式调整大小
        out = F.interpolate(input, size=(out_s,), mode='nearest')
        out.backward(grad)

        # CPU 引用版本的插值操作和反向传播
        out_ref = F.interpolate(input_ref, size=(out_s,), mode='nearest')
        out_ref.backward(grad_ref)

        # 断言两种方法的输出相等
        self.assertEqual(out_ref, out)
        # 断言两种方法的输入梯度相等
        self.assertEqual(input_ref.grad, input.grad)

    # 测试插值中的未定义行为转换
    def test_interpolate_undefined_behavior_casting(self):
        # 创建大小为 [1, 1, 16, 16] 的张量 x
        x = torch.ones([1, 1, 16, 16])
        # 断言在使用超出合理范围的缩放因子时引发运行时错误
        self.assertRaises(RuntimeError, lambda: F.interpolate(x, scale_factor=-1e20, mode="bilinear"))
        self.assertRaises(RuntimeError, lambda: F.interpolate(x, scale_factor=1e20, mode="bilinear"))

    # 设置默认数据类型为双精度浮点数的线性广播测试
    @set_default_dtype(torch.double)
    def test_linear_broadcasting(self):
        # 创建输入维度为 [2, 3, 5] 的线性模型 m
        m = nn.Linear(5, 8)
        inp = torch.randn(2, 3, 5)
        # 计算预期输出，将输入展平后进行线性变换再重塑回原始形状
        expected = m(inp.view(6, 5)).view(2, 3, 8)
        # 断言预期输出与模型应用于输入后的输出相等
        self.assertEqual(expected, m(inp))

    # 测试线性层对标量输入的异常抛出
    def test_linear_raise_on_scalar_input(self):
        # 创建输入维度为 [1] 的线性模型 m
        m = nn.Linear(1, 1)
        inp = torch.ones(1).squeeze()
        # 使用断言捕获 RuntimeError，并验证错误消息中包含特定文本
        with self.assertRaisesRegex(RuntimeError, ".*both arguments.*1D.*"):
            m(inp)

    # 参数化测试函数，测试不同设备和偏置情况下的线性层权重布局
    @parametrize_test('device', ['cpu'] + (['cuda'] if TEST_CUDA else []))
    @parametrize_test('bias', [
        subtest(False, name='nobias'), subtest(True, name='bias')])
    @parametrize_test('weight_layout', [
        subtest(torch.strided, name='weightStrided'),
        subtest(torch.sparse_coo, name='weightCOO'),
        subtest(torch.sparse_csr, name='weightCSR'),
        subtest(torch.sparse_csc, name='weightCSC'),
        # TODO: addmm: computation on CPU is not implemented for Strided + Strided @ SparseBsr
        # subtest(torch.sparse_bsr, name='weightBSR'),
        # subtest(torch.sparse_bsc, name='weightBSC'),
    ])
    # 定义一个测试线性自动求导的方法，接受设备类型、是否包含偏置和权重布局作为参数
    def test_linear_autograd(self, device, bias, weight_layout):
        # 创建一个线性模块，输入和输出都是4维，可以选择是否包含偏置，并指定设备类型
        module = nn.Linear(4, 4, bias=bias, device=device)
        
        # 根据权重布局类型进行不同的处理
        if weight_layout == torch.strided:
            pass
        elif weight_layout == torch.sparse_csr:
            # 将权重转换为压缩稀疏行格式并设为模块参数
            module.weight = nn.Parameter(module.weight.to_sparse_csr())
        elif weight_layout == torch.sparse_csc:
            # 将权重转换为压缩稀疏列格式并设为模块参数
            module.weight = nn.Parameter(module.weight.to_sparse_csc())
        elif weight_layout == torch.sparse_bsr:
            # 将权重转换为块压缩稀疏行格式并设为模块参数
            module.weight = nn.Parameter(module.weight.to_sparse_bsr((2, 2)))
        elif weight_layout == torch.sparse_bsc:
            # 将权重转换为块压缩稀疏列格式并设为模块参数
            module.weight = nn.Parameter(module.weight.to_sparse_bsc((2, 2)))
        elif weight_layout == torch.sparse_coo:
            # 将权重转换为坐标格式的稀疏张量并设为模块参数
            module.weight = nn.Parameter(module.weight.to_sparse_coo())
        else:
            # 若权重布局类型不在预期范围内，则引发断言错误
            raise AssertionError
        
        # 创建一个需要梯度的输入张量，形状为(4,)
        inp = torch.randn(4, requires_grad=True, device=device)
        
        # 将输入传入模块并计算输出结果
        res = module(inp)
        
        # 根据是否包含偏置，计算预期的输出结果
        if bias:
            expected = (torch.einsum("i,ji->j", inp, module.weight.to_dense())) + module.bias
        else:
            expected = (torch.einsum("i,ji->j", inp, module.weight.to_dense()))
        
        # 使用断言检查模块计算的结果是否与预期一致
        self.assertEqual(res, expected)
        
        # 创建一个用于梯度计算的输出张量
        grad_output = torch.randn(4, device=device)
        
        # 使用自动求导计算模块参数和输入张量的梯度
        grads = torch.autograd.grad(res, [module.weight, inp], grad_output)
        
        # 使用自动求导计算预期的模块参数和输入张量的梯度
        grads_expected = torch.autograd.grad(expected, [module.weight, inp], grad_output)
        
        # 使用断言检查模块参数的预期梯度布局是否与指定的权重布局一致
        self.assertEqual(grads_expected[0].layout, weight_layout)
        
        # 使用断言逐个检查计算得到的梯度和预期梯度是否一致
        for g, ge in zip(grads, grads_expected):
            self.assertEqual(g, ge)

    # 定义一个测试双线性模块的方法
    def test_bilinear(self):
        # 创建一个双线性模块，输入维度为10，输出维度为8
        module = nn.Bilinear(10, 10, 8)
        
        # 创建两个需要梯度的输入张量，形状为(4, 10)
        input1 = torch.randn(4, 10, requires_grad=True)
        input2 = torch.randn(4, 10, requires_grad=True)
        
        # 创建一个用于梯度计算的输出张量，形状为(4, 8)
        grad_output = torch.randn(4, 8)
        
        # 将输入传入双线性模块并计算输出结果
        res = module(input1, input2)
        
        # 根据公式计算预期的输出结果
        expected = (torch.einsum("bi,kij,bj->bk", input1, module.weight, input2) +
                    module.bias)
        
        # 使用断言检查模块计算的结果是否与预期一致
        self.assertEqual(res, expected)
        
        # 使用自动求导计算模块参数、偏置以及输入张量的梯度
        grads = torch.autograd.grad(res, [module.weight, module.bias, input1, input2], grad_output)
        
        # 使用自动求导计算预期的模块参数、偏置以及输入张量的梯度
        grads_expected = torch.autograd.grad(expected, [module.weight, module.bias, input1, input2], grad_output)
        
        # 使用断言逐个检查计算得到的梯度和预期梯度是否一致
        for g, ge in zip(grads, grads_expected):
            self.assertEqual(g, ge)
    # 定义测试函数，测试 nn.Bilinear 模块在非连续内存情况下的行为
    def test_bilinear_non_contiguous(self):
        # 创建一个 Bilinear 模块，输入特征维度为 7 和 7，输出特征维度为 5
        module = nn.Bilinear(7, 7, 5)
        # 生成随机张量作为输入1和输入2，形状为 (4, 7, 10)，并要求计算梯度
        input1 = torch.randn(4, 7, 10, requires_grad=True)
        input2 = torch.randn(4, 7, 10, requires_grad=True)
        # 对输入1和输入2进行维度转置，交换第1和第2维度
        input1_tp = input1.transpose(1, 2)
        input2_tp = input2.transpose(1, 2)

        # 创建一个随机梯度输出张量，形状为 (4, 10, 5)
        grad_output = torch.randn(4, 10, 5)

        # 定义运行函数，接收经过转置的输入1和输入2作为参数
        def run(input1_tp, input2_tp):
            # 清空输入1和输入2的梯度信息
            input1.grad = input2.grad = None
            # 使用 Bilinear 模块计算输出
            output = module(input1_tp, input2_tp)
            # 对输出进行反向传播梯度计算
            output.backward(grad_output)

            return output.data, input1.grad.data, input2.grad.data

        # 第一次运行，获取非连续内存情况下的输出、输入1梯度、输入2梯度
        out_nc, g1_nc, g2_nc = run(input1_tp, input2_tp)
        # 将输入1和输入2转换为连续内存
        input1_tp = input1_tp.contiguous()
        input2_tp = input2_tp.contiguous()
        # 第二次运行，获取连续内存情况下的输出、输入1梯度、输入2梯度
        out, g1, g2 = run(input1_tp, input2_tp)

        # 断言连续内存和非连续内存情况下的输出、输入1梯度、输入2梯度应该相等
        self.assertEqual(out, out_nc)
        self.assertEqual(g1, g1_nc)
        self.assertEqual(g2, g2_nc)

    # 测试不带偏置的 Bilinear 模块
    def test_bilinear_no_bias(self):
        # 创建一个输入特征维度为 10 和 10，输出特征维度为 8 的双精度 Bilinear 模块
        module = nn.Bilinear(10, 10, 8, dtype=torch.double)
        # 创建一个相同参数但不带偏置的 Bilinear 模块
        module_no_bias = nn.Bilinear(10, 10, 8, False, dtype=torch.double)

        # 将带偏置的模块的偏置数据清零，并复制不带偏置模块的权重数据
        module.bias.data.zero_()
        module.weight.data.copy_(module_no_bias.weight)

        # 生成随机张量作为输入1和输入2，形状为 (4, 10)，并要求计算梯度，使用双精度
        input1 = torch.randn(4, 10, requires_grad=True, dtype=torch.double)
        input2 = torch.randn(4, 10, requires_grad=True, dtype=torch.double)
        # 创建一个随机梯度输出张量，形状为 (4, 8)，使用双精度
        grad_output = torch.randn(4, 8, dtype=torch.double)

        # 定义运行函数，接收 Bilinear 模块作为参数
        def run(net):
            # 清空输入1和输入2的梯度信息
            input1.grad = input2.grad = None
            # 使用给定的 Bilinear 模块计算输出
            output = net(input1, input2)
            # 对输出进行反向传播梯度计算
            output.backward(grad_output)

            return output.data, input1.grad.data, input2.grad.data

        # 第一次运行，获取带偏置和不带偏置情况下的输出、输入1梯度、输入2梯度
        out, g1, g2 = run(module)
        out_nb, g1_nb, g2_nb = run(module_no_bias)

        # 断言带偏置和不带偏置情况下的输出、输入1梯度、输入2梯度应该相等
        self.assertEqual(out, out_nb)
        self.assertEqual(g1, g1_nb)
        self.assertEqual(g2, g2_nb)

        # 使用自定义的函数进行梯度和梯度二阶检查
        _assertGradAndGradgradChecks(self,
                                     lambda x1, x2: F.bilinear(x1, x2, module_no_bias.weight, module_no_bias.bias),
                                     (input1, input2))

    # 测试 Bilinear 模块的广播性质
    def test_bilinear_broadcasting(self):
        # 创建一个输入特征维度为 5 和 6，输出特征维度为 8 的 Bilinear 模块
        m = nn.Bilinear(5, 6, 8)
        # 生成随机张量作为输入1和输入2，形状为 (2, 3, 5) 和 (2, 3, 6)
        input1 = torch.randn(2, 3, 5)
        input2 = torch.randn(2, 3, 6)
        # 将输入1和输入2的形状进行调整以便进行广播操作，期望的输出形状为 (2, 3, 8)
        expected = m(input1.view(6, 5), input2.view(6, 6)).view(2, 3, 8)
        # 断言模块计算的实际输出与预期输出相等
        self.assertEqual(expected, m(input1, input2))
    def test_fold_invalid_arg(self):
        # 测试折叠操作中的无效参数情况

        # 创建 Fold 对象，设置输出大小为 (4, 5)，核大小为 (2, 3)
        fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 3))
        # 断言抛出 RuntimeError，并检查错误信息是否包含特定字符串
        with self.assertRaisesRegex(RuntimeError, r"be divisible by the product of kernel_size"):
            # 对输入大小 (1, 5, 9) 进行折叠操作
            fold(torch.randn(1, 5, 9))

        with self.assertRaisesRegex(RuntimeError, r"be divisible by the product of kernel_size"):
            # 对输入大小 (1, 19, 9) 进行折叠操作
            fold(torch.randn(1, 19, 9))

        # 输入维度 size(2) 与滑动块总数不匹配的情况
        with self.assertRaisesRegex(RuntimeError, r"match the calculated number of sliding blocks"):
            # 创建 Fold 对象，设置输出大小为 (4, 5)，核大小为 (2, 3)
            fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 3))
            # 对输入大小 (1, 6, 10) 进行折叠操作
            fold(torch.randn(1, 6, 10))

        with self.assertRaisesRegex(RuntimeError, r"match the calculated number of sliding blocks"):
            # 创建 Fold 对象，设置输出大小为 (4, 5)，核大小为 (2, 3)，步长为 (2, 2)
            fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 3), stride=(2, 2))
            # 对输入大小 (1, 6, 5) 进行折叠操作
            fold(torch.randn(1, 6, 5))

        with self.assertRaisesRegex(RuntimeError, r"match the calculated number of sliding blocks"):
            # 创建 Fold 对象，设置输出大小为 (4, 5)，核大小为 (2, 3)，步长为 (2, 2)，扩张为 (1, 2)，填充为 (2, 0)
            fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 3), stride=(2, 2), dilation=(1, 2), padding=(2, 0))
            # 对输入大小 (1, 6, 5) 进行折叠操作，期望有 4 个滑动块
            fold(torch.randn(1, 6, 5))

        # 创建 Fold 对象，设置输出大小为 (4, 5)，核大小为 (2, 2)，步长为 1，扩张为 8，填充为 0
        fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 2), stride=1, dilation=8, padding=0)
        # 断言抛出 RuntimeError，并检查错误信息是否包含特定字符串
        with self.assertRaisesRegex(RuntimeError, r"calculated shape of the array of sliding blocks as"):
            # 对输入大小 (1, 12, 12) 进行折叠操作
            fold(torch.randn(1, 12, 12))

    def test_unfold_invalid_arg(self):
        # 测试展开操作中的无效参数情况

        # 创建 Unfold 对象，设置核大小为 (2, 3)
        unfold = nn.Unfold(kernel_size=(2, 3))

        # 计算的输出形状过小的情况
        with self.assertRaisesRegex(RuntimeError, r"its components must be at least one"):
            # 创建 Unfold 对象，设置核大小为 (2, 3)
            unfold = nn.Unfold(kernel_size=(2, 3))
            # 对输入大小 (1, 2, 2, 2) 进行展开操作
            unfold(torch.randn(1, 2, 2, 2))

        with self.assertRaisesRegex(RuntimeError, r"its components must be at least one"):
            # 创建 Unfold 对象，设置核大小为 (5, 3)，填充为 (1, 1)
            unfold = nn.Unfold(kernel_size=(5, 3), padding=(1, 1))
            # 对输入大小 (1, 2, 2, 3) 进行展开操作
            unfold(torch.randn(1, 2, 2, 3))

        with self.assertRaisesRegex(RuntimeError, r"its components must be at least one"):
            # 创建 Unfold 对象，设置核大小为 (1, 3)，填充为 (1, 1)，扩张为 (1, 2)
            unfold = nn.Unfold(kernel_size=(1, 3), padding=(1, 1), dilation=(1, 2))
            # 对输入大小 (1, 2, 2, 2) 进行展开操作
            unfold(torch.randn(1, 2, 2, 2))

    def test_softmin(self):
        # 测试 softmin 函数的功能

        # 创建大小为 (2, 16) 的随机张量
        x = torch.randn(2, 16)
        # 断言 softmin 函数在维度 1 上的输出与 softmax(-x) 相等
        self.assertEqual(F.softmin(x, 1), F.softmax(-x, 1))
        # 断言 softmin 函数在维度 0 上的输出与 softmax(-x) 相等
        self.assertEqual(F.softmin(x, 0), F.softmax(-x, 0))
    def test_cross_entropy_loss(self, dtype=torch.bfloat16):
        # 创建一个在 CPU 上的交叉熵损失函数对象
        loss_cpu = nn.CrossEntropyLoss().cpu()
        # 生成一个形状为 (15, 10) 的随机张量，在 CPU 上，数据类型为 torch.float，并要求梯度
        inputf = torch.randn(15, 10, device="cpu", dtype=torch.float, requires_grad=True)
        # 将 inputf 张量转换为指定的数据类型 dtype，并且保留其梯度信息
        input = inputf.to(dtype).detach().requires_grad_(True)
        # 生成一个形状为 (15,) 的长整型张量，其元素随机取自 [0, 10)
        target = torch.empty(15, dtype=torch.long).random_(10)

        # 对 inputf 和 input 张量分别计算交叉熵损失值
        outf = loss_cpu(inputf, target)
        out = loss_cpu(input, target)
        # 断言两个损失值的近似相等性，允许的绝对误差为 1e-1，相对误差为 0
        self.assertEqual(out, outf.to(dtype=dtype), atol=1e-1, rtol=0)

        # 对 outf 和 out 损失值分别进行反向传播
        outf.backward()
        out.backward()
        # 断言 input 和 inputf 张量的梯度在指定的数据类型 dtype 下的近似相等性，允许的绝对误差为 1e-1，相对误差为 0
        self.assertEqual(input.grad, inputf.grad.to(dtype=dtype), atol=1e-1, rtol=0)

    def test_cross_entropy_loss_precision(self):
        # 回归测试，用于问题 #55657
        # 创建一个在 CPU 上的交叉熵损失函数对象
        loss_cpu = nn.CrossEntropyLoss().cpu()
        # 生成一个形状为 (128, 2, 768, 768) 的随机张量，在 CPU 上，数据类型为 torch.float
        inputf = torch.randn(128, 2, 768, 768, device="cpu", dtype=torch.float)
        # 将 inputf 张量转换为双精度浮点类型
        inputd = inputf.double()
        # 生成一个形状为 (128, 768, 768) 的长整型张量，其元素随机取自 [0, 2)
        target = torch.randint(2, (128, 768, 768), dtype=torch.long)

        # 对 inputf 和 inputd 张量分别计算交叉熵损失值
        outf = loss_cpu(inputf, target)
        outd = loss_cpu(inputd, target)
        # 断言 outf 和 outd 的交叉熵损失值相等，不要求完全相等的数据类型
        self.assertEqual(outf, outd, exact_dtype=False)

    def test_cross_entropy_loss_zero_div(self):
        # 测试问题 #73165
        # 创建两个形状为 [5, 0] 的随机张量，数据类型为 torch.float32
        input_1 = torch.rand([5, 0], dtype=torch.float32)
        input_2 = torch.rand([5, 0], dtype=torch.float32)
        # 对这两个张量应用交叉熵损失函数
        torch.nn.CrossEntropyLoss()(input_1, input_2)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_convert_sync_batchnorm(self):
        # 创建一个包含 BatchNorm1d 和 InstanceNorm1d 层的 Sequential 模块，并将其移到 GPU 上
        module = torch.nn.Sequential(
            torch.nn.BatchNorm1d(100),
            torch.nn.InstanceNorm1d(100)
        ).cuda()

        # 用于比较的模块，加载与 module 相同的状态字典
        comp_module = torch.nn.Sequential(
            torch.nn.BatchNorm1d(100),
            torch.nn.InstanceNorm1d(100)
        ).cuda()
        comp_module.load_state_dict(module.state_dict())

        # 将 module 中的 BatchNorm1d 层转换为 SyncBatchNorm 层
        sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
        children = list(sync_bn_module.children())
        # 断言第一个子模块为 SyncBatchNorm 类型
        self.assertEqual(children[0].__class__, torch.nn.SyncBatchNorm)
        # 断言第二个子模块为 InstanceNorm1d 类型
        self.assertEqual(children[1].__class__, torch.nn.InstanceNorm1d)

        # 逐层比较 comp_module 和 sync_bn_module 的状态字典
        for layer, converted_layer in zip(comp_module.children(), sync_bn_module.children()):
            for key in layer.state_dict().keys():
                # 断言每一层的状态字典的设备相同
                self.assertEqual(layer.state_dict()[key].device, converted_layer.state_dict()[key].device)
                # 断言每一层的状态字典的值相等
                self.assertEqual(layer.state_dict()[key], converted_layer.state_dict()[key])

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_sync_batchnorm_backward_elemt(self):
        device = 'cuda'
        saved_input = torch.rand(2, 3, 2, 1, device=device)  # 创建一个指定设备上随机数填充的张量 saved_input
        grad_output = torch.rand(2, 3, 2, 1, device=device)  # 创建一个指定设备上随机数填充的张量 grad_output
        mean = torch.rand(3, device=device)  # 创建一个指定设备上随机数填充的张量 mean
        invstd = torch.rand(3, device=device)  # 创建一个指定设备上随机数填充的张量 invstd
        weight = torch.rand(3, device=device)  # 创建一个指定设备上随机数填充的张量 weight
        sum_dy = torch.rand(3, device=device)  # 创建一个指定设备上随机数填充的张量 sum_dy
        sum_dy_xmu = torch.rand(3, device=device)  # 创建一个指定设备上随机数填充的张量 sum_dy_xmu
        count_tensor = torch.tensor([5, 5, 5], dtype=torch.int32, device=device)  # 创建一个指定设备上的整数张量 count_tensor

        # 调用 torch.batch_norm_backward_elemt 函数计算梯度
        gI_contiguous = torch.batch_norm_backward_elemt(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            sum_dy,
            sum_dy_xmu,
            count_tensor
        )

        # 测试在不同的内存格式下，torch.batch_norm_backward_elemt 函数的结果是否一致
        for a, b in [
                (torch.channels_last, torch.contiguous_format),
                (torch.contiguous_format, torch.channels_last),
                (torch.channels_last, torch.channels_last),
        ]:
            # 使用不同内存格式的数据调用 torch.batch_norm_backward_elemt 函数
            gI_actual = torch.batch_norm_backward_elemt(
                grad_output.contiguous(memory_format=a),
                saved_input.contiguous(memory_format=b),
                mean,
                invstd,
                weight,
                sum_dy,
                sum_dy_xmu,
                count_tensor
            )
            # 断言两种调用方式得到的结果 gI_actual 和 gI_contiguous 相等
            self.assertEqual(gI_actual, gI_contiguous)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_sync_batchnorm_accuracy_cuda(self):
        # 该测试的目标是测试 SyncBatchNorm 中使用的单GPU CUDA 内核的功能和准确性
        # 它们包括：
        #   前向：torch.batch_norm_stats, torch.batch_norm_gather_stats_with_counts, torch.batch_norm_elemt
        #   反向：torch.batch_norm_backward_reduce, torch.batch_norm_backward_elemt

        def _batch_norm_stats(data, memory_format, mean_axes):
            # 调用 torch.batch_norm_stats 函数计算统计量 mean1
            mean1, _ = torch.batch_norm_stats(data, 1e-5)
            # 使用指定的内存格式调用 torch.batch_norm_stats 函数计算统计量 mean2
            mean2, _ = torch.batch_norm_stats(data.to(memory_format=memory_format), 1e-5)
            # 计算数据在指定轴上的平均值 mean_ref
            mean_ref = torch.mean(data, mean_axes, keepdim=False)

            # 断言 mean_ref 与 mean1 相等
            self.assertEqual(mean_ref, mean1)
            # 断言 mean_ref 与 mean2 相等
            self.assertEqual(mean_ref, mean2)

        # 测试在 CUDA 设备上使用 channels_last 内存格式的 _batch_norm_stats 函数
        _batch_norm_stats(torch.randn(1, 96, 112, 112, dtype=torch.float, device='cuda'), torch.channels_last, (0, 2, 3))
        # 测试在 CUDA 设备上使用 channels_last_3d 内存格式的 _batch_norm_stats 函数
        _batch_norm_stats(torch.randn(1, 96, 112, 112, 112, dtype=torch.float, device='cuda'), torch.channels_last_3d, (0, 2, 3, 4))

    def test_flatten(self):
        tensor_input = torch.randn(2, 1, 2, 3)

        # 展平张量
        flatten = nn.Flatten(start_dim=1, end_dim=-1)
        tensor_output = flatten(tensor_input)
        # 断言展平后的张量大小是否符合预期
        self.assertEqual(tensor_output.size(), torch.Size([2, 6]))
    def test_unflatten(self):
        tensor_input = torch.randn(2, 50)

        # Unflatten Tensor (unflattened_size as a tuple of ints and list of ints)

        # 循环测试不同的 unflattened_size 参数格式
        for us in ((2, 5, 5), [2, 5, 5]):
            # 创建 Unflatten 模块对象，指定维度和 unflattened_size 参数
            unflatten = nn.Unflatten(dim=1, unflattened_size=us)
            # 对输入的 tensor 进行 unflatten 操作
            tensor_output = unflatten(tensor_input)
            # 断言输出 tensor 的尺寸是否正确
            self.assertEqual(tensor_output.size(), torch.Size([2, 2, 5, 5]))

        # Unflatten NamedTensor

        # 创建 Unflatten 模块对象，使用命名的维度和 unflattened_size 参数
        unflatten = nn.Unflatten(dim='features', unflattened_size=(('C', 2), ('H', 5), ('W', 5)))
        # 将 tensor_input 转换为命名 tensor
        named_tensor_input = tensor_input.refine_names('N', 'features')
        # 对命名 tensor 进行 unflatten 操作
        named_tensor_output = unflatten(named_tensor_input)
        # 断言输出命名 tensor 的尺寸是否正确
        self.assertEqual(named_tensor_output.size(), torch.Size([2, 2, 5, 5]))

    def test_unflatten_invalid_arg(self):
        # Wrong type for unflattened_size (tuple of floats)

        # 测试 unflattened_size 参数为浮点数的情况
        with self.assertRaisesRegex(
                TypeError,
                r"unflattened_size must be tuple of ints, but found element of type float at pos 2"):
            nn.Unflatten(dim=1, unflattened_size=(2, 5, 5.0))

        # Wrong type for unflattened_size (list of lists and list of tuples)
        for us in ([['C', 2], ['W', 5], ['H', 5]], [('C', 2), ('W', 5), ('H', 5)]):
            # 测试 unflattened_size 参数为列表或混合类型的情况
            with self.assertRaisesRegex(
                    TypeError,
                    r"unflattened_size must be a tuple of tuples, but found type list"):
                nn.Unflatten(dim='features', unflattened_size=us)

        # Wrong type for unflattened_size (tuple of lists)

        # 测试 unflattened_size 参数内部元素为列表的情况
        with self.assertRaisesRegex(
                TypeError,
                r"unflattened_size must be tuple of tuples, but found element of type list at pos 0"):
            nn.Unflatten(dim='features', unflattened_size=(['C', 2], ['W', 5], ['H', 5]))

        # Wrong type for unflattened_size (tuple of dicts)

        # 测试 unflattened_size 参数内部元素为字典的情况
        with self.assertRaisesRegex(
                TypeError,
                r"unflattened_size must be tuple of tuples, but found element of type dict at pos 0"):
            nn.Unflatten(dim='features', unflattened_size=({'C': 2}, {'W': 5}, {'H': 5}))
    # 定义测试方法：验证带有 create_graph 标志的 LayerNorm 的梯度计算是否一致
    def test_layer_norm_grads_with_create_graph_flag(self):
        atol = 1e-5  # 允许的绝对误差
        rtol = 1e-3  # 允许的相对误差

        # 创建一个随机张量 x，需要计算梯度
        x = torch.randn((4, 4, 16), requires_grad=True)
        # 创建一个 LayerNorm 层，指定归一化的维度和参数
        layer_norm = nn.LayerNorm((16,), eps=1e-5, elementwise_affine=True)
        # 设置 layer_norm 的权重参数为固定值
        with torch.no_grad():
            layer_norm.weight = torch.nn.Parameter(0.1 * torch.ones_like(layer_norm.weight))

        # 计算不同 create_graph 标志下的梯度
        grads1 = torch.autograd.grad(layer_norm(x).sum(), x, create_graph=False)[0]
        grads2 = torch.autograd.grad(layer_norm(x).sum(), x, create_graph=True)[0]

        # 断言两种梯度计算方式是否一致
        self.assertEqual(grads1, grads2, rtol=rtol, atol=atol)

        # 如果支持 CUDA，再次进行相同的测试，但在 CUDA 设备上进行
        if TEST_CUDA:
            x = x.to('cuda')
            layer_norm = layer_norm.to('cuda')

            grads1 = torch.autograd.grad(layer_norm(x).sum(), x, create_graph=False)[0]
            grads2 = torch.autograd.grad(layer_norm(x).sum(), x, create_graph=True)[0]

            self.assertEqual(grads1, grads2, rtol=rtol, atol=atol)

    # 定义测试方法：验证 LayerNorm 的 eps 参数设置是否正常工作
    def test_layer_norm_eps(self):
        # 测试 https://github.com/pytorch/pytorch/issues/108072 的问题
        x = torch.Tensor([[[2.0, 2.0], [14.0, 14.0]], [[2.0, 2.0], [14.0, 14.0]]])
        ln = torch.nn.LayerNorm(2, eps=1e-6, elementwise_affine=False)
        # 断言 LayerNorm 应用于 x 后的结果是否全为零张量
        self.assertEqual(ln.forward(x), torch.zeros_like(x))

    # 定义测试方法：验证卷积转置层的 padding 参数可以是列表或元组
    def test_padding_list(self):
        x = torch.randn(4, 8, 32, 32)
        # 使用列表作为 padding 参数创建卷积转置层
        net = torch.nn.ConvTranspose2d(8, 16, kernel_size=3, padding=[3, 3])
        y = net(x)

        # 使用元组作为 padding 参数创建卷积转置层
        net = torch.nn.ConvTranspose2d(8, 16, kernel_size=3, padding=(3, 3))
        y = net(x)

    # 定义测试方法：验证 FractionalMaxPool2d 的 output_ratio 参数设置是否正常工作
    def test_fractional_max_pool2d_invalid_output_ratio(self):
        arg_1 = [2, 1]
        arg_2 = [0.5, 0.5, 0.6]
        arg_class = torch.nn.FractionalMaxPool2d(kernel_size=arg_1, output_ratio=arg_2,)
        arg_3_0_tensor = torch.rand([20, 16, 50, 32], dtype=torch.float32)
        arg_3_0 = arg_3_0_tensor.clone()
        arg_3 = [arg_3_0,]

        # 使用不合法的 output_ratio 参数值，应抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError,
                                    "fractional_max_pool2d requires output_ratio to either be a single Int or tuple of Ints."):
            res = arg_class(*arg_3)

    # 定义测试方法：验证 MaxPool1d 的输出大小参数设置是否正常工作
    def test_max_pool1d_invalid_output_size(self):
        arg_1 = 3
        arg_2 = 255
        arg_3 = False
        arg_class = torch.nn.MaxPool1d(kernel_size=arg_1, stride=arg_2, return_indices=arg_3)
        arg_4_0 = torch.as_tensor([[0.3204]])
        arg_4 = [arg_4_0,]

        # 使用不合法的输出大小参数，应抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            res = arg_class(*arg_4)
class TestFusionEval(TestCase):
    # 定义测试类 TestFusionEval，继承自 TestCase
    @set_default_dtype(torch.double)
    @given(X=hu.tensor(shapes=((5, 3, 5, 5),), dtype=np.double),
           running_mean=hu.tensor(shapes=(6,), dtype=np.double),
           running_var=hu.tensor(shapes=(6,), dtype=np.double))
    # 使用 set_default_dtype 设置默认的张量数据类型为双精度浮点数
    # 使用 @given 定义测试参数 X, running_mean, running_var 的生成规则
    def test_fuse_module_eval_numerics(self, X, running_mean, running_var):
        # 定义测试方法 test_fuse_module_eval_numerics，接受参数 X, running_mean, running_var
        inputs, _ = X
        # 解包 X，获取 inputs

        iC, oC = inputs.shape[1], len(running_mean[0])
        # 获取输入张量的通道数 iC 和 running_mean 的长度 oC

        inputs = torch.from_numpy(inputs)
        # 将 NumPy 数组 inputs 转换为 PyTorch 张量

        kernel_size = (3, 3)
        # 定义卷积核大小 kernel_size

        conv_ref = torch.nn.Conv2d(iC, oC, bias=True, kernel_size=kernel_size)
        # 创建卷积层 conv_ref，指定输入通道数 iC，输出通道数 oC，使用 3x3 的卷积核

        bn_ref = torch.nn.BatchNorm2d(oC)
        # 创建批归一化层 bn_ref，指定特征通道数 oC

        bn_ref.running_mean = torch.from_numpy(running_mean[0])
        # 设置 bn_ref 的 running_mean 属性为 running_mean 的第一个元素转换的张量

        bn_ref.running_var = torch.from_numpy(running_var[0])
        # 设置 bn_ref 的 running_var 属性为 running_var 的第一个元素转换的张量

        conv_ref.eval()
        # 将卷积层 conv_ref 设置为评估模式（eval mode）

        bn_ref.eval()
        # 将批归一化层 bn_ref 设置为评估模式（eval mode）

        Y_ref = bn_ref(conv_ref(inputs))
        # 对输入 inputs 先进行卷积 conv_ref，然后通过 bn_ref 进行批归一化得到 Y_ref

        conv_bn_fused = torch.nn.utils.fusion.fuse_conv_bn_eval(conv_ref,
                                                                bn_ref)
        # 使用工具函数 fuse_conv_bn_eval 对 conv_ref 和 bn_ref 进行融合得到融合后的模块 conv_bn_fused

        Y_hat = conv_bn_fused(inputs)
        # 使用融合后的模块 conv_bn_fused 处理输入 inputs 得到 Y_hat

        self.assertEqual(Y_ref, Y_hat, msg="Conv+BN fusion results are off")
        # 使用断言检查 Y_ref 和 Y_hat 是否相等，如果不相等则输出指定的错误信息

        na_bn_ref = torch.nn.BatchNorm2d(oC, affine=False)
        # 创建不带仿射变换的批归一化层 na_bn_ref，指定特征通道数 oC

        na_bn_ref.running_mean = torch.from_numpy(running_mean[0])
        # 设置 na_bn_ref 的 running_mean 属性为 running_mean 的第一个元素转换的张量

        na_bn_ref.running_var = torch.from_numpy(running_var[0])
        # 设置 na_bn_ref 的 running_var 属性为 running_var 的第一个元素转换的张量

        na_bn_ref.eval()
        # 将 na_bn_ref 设置为评估模式（eval mode）

        Y_ref = na_bn_ref(conv_ref(inputs))
        # 对输入 inputs 先进行卷积 conv_ref，然后通过 na_bn_ref 进行批归一化得到 Y_ref

        conv_na_bn_fused = torch.nn.utils.fusion.fuse_conv_bn_eval(conv_ref,
                                                                   na_bn_ref)
        # 使用工具函数 fuse_conv_bn_eval 对 conv_ref 和 na_bn_ref 进行融合得到融合后的模块 conv_na_bn_fused

        Y_hat = conv_na_bn_fused(inputs)
        # 使用融合后的模块 conv_na_bn_fused 处理输入 inputs 得到 Y_hat

        self.assertEqual(Y_ref, Y_hat, msg="Conv+BN(non-affine) fusion results are off")
        # 使用断言检查 Y_ref 和 Y_hat 是否相等，如果不相等则输出指定的错误信息


class TestConstantPadNd(TestCase):
    # 定义测试类 TestConstantPadNd，继承自 TestCase
    def test_constant_pad_nd(self):
        # 定义测试方法 test_constant_pad_nd
        a = torch.tensor([[1, 2], [3, 4]])
        # 创建张量 a，包含二维数组

        res = torch.constant_pad_nd(a, [1, 2, 1, 0], 9)
        # 使用 constant_pad_nd 函数对张量 a 进行常数填充，指定填充边界为 [1, 2, 1, 0]，填充值为 9

        expected = torch.tensor([
            [9, 9, 9, 9, 9],
            [9, 1, 2, 9, 9],
            [9, 3, 4, 9, 9]
        ])
        # 创建预期的填充结果张量 expected

        self.assertEqual(res, expected)
        # 使用断言检查 res 和 expected 是否相等

    def test_preserves_memory_format(self):
        # 定义测试方法 test_preserves_memory_format
        nchw_tensor = torch.rand((1, 2, 5, 3))
        # 创建 NCHW 格式的随机张量 nchw_tensor

        nchw_padded = torch.constant_pad_nd(nchw_tensor, [1, 2], 0.5)
        # 使用 constant_pad_nd 函数对 nchw_tensor 进行常数填充，指定填充边界为 [1, 2]，填充值为 0.5

        self.assertTrue(nchw_padded.is_contiguous(memory_format=torch.contiguous_format))
        # 使用断言检查 nchw_padded 是否是连续的内存格式（contiguous format）

        nhwc_tensor = nchw_tensor.contiguous(memory_format=torch.channels_last)
        # 将 nchw_tensor 转换为 NHWC 格式的连续张量 nhwc_tensor

        nhwc_padded = torch.constant_pad_nd(nhwc_tensor, [1, 2], 0.5)
        # 使用 constant_pad_nd 函数对 nhwc_tensor 进行常数填充，指定填充边界为 [1, 2]，填充值为 0.5

        self.assertTrue(nhwc_padded.is_contiguous(memory_format=torch.channels_last))
        # 使用断言检查 nhwc_padded 是否是通道优先的内存格式（channels_last format）


class TestAddRelu(TestCase):
    # 定义测试类 TestAddRelu，继承自 TestCase
    def test_add_relu(self):
        # 定义测试方法 test_add_relu
        a = torch.rand((7, 11))
        # 创建形状为 (7, 11) 的随机张量 a

        b = torch.rand((7, 11))
        # 创建形状与 a 相同的随机张量 b

        a = a.float()
        # 将张量 a 转换为单精度浮点数类型

        b = b.float()
        # 将张量 b 转换为单精度浮点数类型

        a = a * -10
        # 将张量 a 中的每个元素乘以 -10

        a = a + 5
        # 将张量 a 中的每个元素加上 5

        add_res = a + b
        # 计算张量 a 和 b 的逐元素和，结果存储在 add_res 中

        relu_res = torch.relu(add_res)
        # 对 add_res 中的每个元素应用 ReLU 函数，结果存储在 relu_res 中

        add_relu_res = torch._VF._add_relu(a, b)
        # 调用内部函数 _add_relu 对张量 a 和 b 进行逐元素相加并应用 ReLU 函数

        self.assertEqual(add_relu_res, relu_res)
        # 使用断言检查 add_relu_res 和 relu_res 是否相等
    # 定义一个测试方法，用于测试加法和ReLU激活的广播操作
    def test_add_relu_broadcasting(self):
        # 创建一个形状为(1, 32)的随机张量a
        a = torch.rand((1, 32))
        # 创建一个标量b，值为1
        b = 1
        # 创建一个形状为(1, 32)的张量b_scalar，每个元素均为1
        b_scalar = torch.ones(1, 32)
        # 使用torch._VF._add_relu函数对张量a和标量b进行加法并应用ReLU激活，得到结果res
        res = torch._VF._add_relu(a, b)
        # 使用torch._VF._add_relu函数对张量a和张量b_scalar进行加法并应用ReLU激活，得到广播后的结果broadcasted_res
        broadcasted_res = torch._VF._add_relu(a, b_scalar)

        # 断言广播后的结果broadcasted_res等于未广播的结果res
        self.assertEqual(broadcasted_res, res)
# 定义函数 `add_test`，用于将测试添加到 `TestNN` 类中
def add_test(test, decorator=None):
    # 定义嵌套函数 `add`，用于将单个测试函数添加到 `TestNN` 类中
    def add(test_name, fn):
        # 检查是否已经存在同名的测试函数，如果存在则抛出运行时错误
        if hasattr(TestNN, test_name):
            raise RuntimeError('Found two tests with the same name: ' + test_name)
        # 如果指定了装饰器，则对测试函数进行装饰
        if decorator is not None:
            fn = decorator(fn)
        # 将测试函数添加为 `TestNN` 类的属性，属性名为 `test_name`
        setattr(TestNN, test_name, fn)

    # 获取测试的名称
    test_name = test.get_name()
    # 如果测试对象中不存在 `test_cpu` 属性，或者 `test_cpu` 为 True，则添加 CPU 测试
    if not hasattr(test, 'test_cpu') or test.test_cpu:
        # 使用 lambda 表达式定义并添加 CPU 测试函数
        add(test_name, lambda self, test=test: test(self))
    
    # 构造 CUDA 测试函数名
    cuda_test_name = test_name + '_cuda'

    # 检查测试函数是否支持额外的参数
    kwargs = {}
    if 'extra_args' in get_function_arglist(test.test_cuda):
        kwargs['extra_args'] = test.extra_args

    # 检查测试函数是否支持 `dtype` 参数
    if 'dtype' in get_function_arglist(test.test_cuda):
        # 如果当前环境不支持 TF32 且测试需要使用 TF32
        if tf32_is_not_fp32() and test.with_tf32:
            # 定义关闭 TF32 的测试函数，并添加到 `TestNN` 类中
            def with_tf32_off(self, test=test, kwargs=kwargs):
                with tf32_off():
                    test.test_cuda(self, dtype=torch.float, **kwargs)
            add(cuda_test_name + '_fp32', with_tf32_off)

            # 定义打开 TF32 的测试函数，并添加到 `TestNN` 类中
            def with_tf32_on(self, test=test, kwargs=kwargs):
                with tf32_on(self, test.tf32_precision):
                    test.test_cuda(self, dtype=torch.float, **kwargs)
            add(cuda_test_name + '_tf32', with_tf32_on)
        else:
            # 添加使用单精度浮点数进行 CUDA 测试的函数
            add(cuda_test_name + '_float', lambda self,
                test=test, kwargs=kwargs: test.test_cuda(self, dtype=torch.float, **kwargs))
        
        # 添加使用双精度浮点数进行 CUDA 测试的函数
        add(cuda_test_name + '_double', lambda self,
            test=test, kwargs=kwargs: test.test_cuda(self, dtype=torch.double, **kwargs))

        # 定义使用半精度浮点数进行 CUDA 测试的函数，并根据 `check_half` 属性决定是否添加
        def test_half(self, test=test, kwargs=kwargs):
            test.test_cuda(self, dtype=torch.half, **kwargs)
        if getattr(test, 'check_half', True):
            add(cuda_test_name + '_half', test_half)

        # 定义使用 BF16 进行 CUDA 测试的函数，并根据 `check_bfloat16` 属性决定是否添加
        def test_bfloat16(self, test=test, kwargs=kwargs):
            test.test_cuda(self, dtype=torch.bfloat16, **kwargs)
        if getattr(test, 'check_bfloat16', True):
            add(cuda_test_name + '_bfloat16', test_bfloat16)

        # 如果测试函数支持复数类型，定义相应的测试函数并添加到 `TestNN` 类中
        def test_cfloat(self, test=test, kwargs=kwargs):
            test.test_cuda(self, dtype=torch.cfloat, **kwargs)

        def test_cdouble(self, test=test, kwargs=kwargs):
            test.test_cuda(self, dtype=torch.cdouble, **kwargs)
        if getattr(test, 'check_complex', False):
            add(cuda_test_name + '_cfloat', test_cfloat)
            add(cuda_test_name + '_cdouble', test_cdouble)

    else:
        # 定义关闭 TF32 的测试函数，并添加到 `TestNN` 类中
        def with_tf32_off(self, test=test, kwargs=kwargs):
            with tf32_off():
                test.test_cuda(self, **kwargs)
        
        # 如果当前环境不支持 TF32 且测试需要使用 TF32
        if tf32_is_not_fp32() and test.with_tf32:
            add(cuda_test_name + '_fp32', with_tf32_off)

            # 定义打开 TF32 的测试函数，并添加到 `TestNN` 类中
            def with_tf32_on(self, test=test, kwargs=kwargs):
                with tf32_on(self, test.tf32_precision):
                    test.test_cuda(self, **kwargs)
            add(cuda_test_name + '_tf32', with_tf32_on)
        else:
            add(cuda_test_name, with_tf32_off)
for test_params in module_tests + new_module_tests:
    # 遍历所有的测试参数，包括原有的模块测试和新模块测试
    # TODO: CUDA is not implemented yet
    # 如果测试参数中没有 'constructor' 键
    if 'constructor' not in test_params:
        # 弹出 'module_name' 键作为模块名
        name = test_params.pop('module_name')
        # 使用 getattr 函数获取 nn 模块中的构造函数，并赋给 'constructor' 键
        test_params['constructor'] = getattr(nn, name)
    # 弹出 'decorator' 键作为装饰器，如果没有则为 None
    decorator = test_params.pop('decorator', None)
    # 创建一个 NewModuleTest 实例，传入所有的测试参数
    test = NewModuleTest(**test_params)
    # 将新创建的测试实例添加到测试集中，应用可能存在的装饰器
    add_test(test, decorator)
    # 如果测试参数中包含 'check_eval' 键
    if 'check_eval' in test_params:
        # 创建一个与当前测试相同的新测试，但将 module.training 设置为 False
        desc = test_params.get('desc', None)
        # 如果原描述为 None，则设置为 'eval'；否则在原描述后加上 '_eval'
        test_params['desc'] = 'eval' if desc is None else desc + '_eval'

        # 定义一个生成 eval 构造函数的函数，保持与原构造函数一致
        def gen_eval_constructor(constructor):
            def eval_constructor(*args, **kwargs):
                # 调用原构造函数，并将 module.training 设置为 False
                cons = constructor(*args, **kwargs)
                cons.training = False
                return cons
            # 设置 eval 构造函数的名称与原构造函数相同
            eval_constructor.__name__ = constructor.__name__
            return eval_constructor

        # 使用 gen_eval_constructor 函数生成 eval 构造函数，并将其赋给 'constructor' 键
        test_params['constructor'] = gen_eval_constructor(test_params['constructor'])
        # 创建一个新的 NewModuleTest 实例，传入修改后的测试参数
        test = NewModuleTest(**test_params)
        # 将新创建的 eval 测试实例添加到测试集中，应用可能存在的装饰器
        add_test(test, decorator)
    # 检查是否在测试参数中存在 'check_with_long_tensor' 键
    if 'check_with_long_tensor' in test_params:
        # 获取测试参数中的 'fullname'，默认为 None
        fullname = test_params.get('fullname', None)
        # 如果 fullname 存在
        if fullname:
            # 在 fullname 后面添加 '_with_long_tensor'
            test_params['fullname'] = fullname + '_with_long_tensor'
        else:
            # 获取测试参数中的 'desc'，默认为 None
            desc = test_params.get('desc', None)
            # 如果 desc 不存在，则设置为 'with_long_tensor'，否则在 desc 后面添加 '_with_long_tensor'
            test_params['desc'] = 'with_long_tensor' if desc is None else desc + '_with_long_tensor'

        # 定义一个函数，返回一个大小为 size 的随机整数张量，类型为双精度
        def double_equivalent_of_long_tensor(size):
            return torch.randint(-1000, 1000, size=size).double()

        # 应用于构造函数的函数，将浮点型张量转换为双精度型张量
        def apply_to_cons(t):
            # 如果张量是浮点型
            if t.is_floating_point():
                # 如果 t 是 Parameter 对象，则返回双精度等价的长整型张量作为 Parameter 对象
                if isinstance(t, Parameter):
                    return Parameter(double_equivalent_of_long_tensor(t.size()))
                # 如果 t 是普通的 torch.Tensor 对象，则返回双精度等价的长整型张量
                elif isinstance(t, torch.Tensor):
                    return double_equivalent_of_long_tensor(t.size())
            else:
                return t

        # 生成长整型张量构造函数的构造器，用于将给定构造函数的输出张量转换为长整型张量
        def gen_long_tensor_constructor(constructor):
            def long_tensor_constructor(*args, **kwargs):
                # 调用原始的构造函数 constructor，并将其输出张量应用于 apply_to_cons 函数
                cons = constructor(*args, **kwargs)
                cons._apply(apply_to_cons)
                return cons
            # 设置构造器的名称为原始构造函数的名称
            long_tensor_constructor.__name__ = constructor.__name__
            return long_tensor_constructor

        # 生成长整型张量输入的函数，返回一个大小为 input_size 的双精度型长整型张量
        def gen_long_tensor_input(input_size):
            def input_func():
                return double_equivalent_of_long_tensor(input_size)
            return input_func

        # 参考函数，用于生成需要梯度的长整型张量
        def reference_fn(i, p, m):
            # 将模型 m 的所有参数的 requires_grad 属性设为 False，避免创建需要梯度的长整型张量
            for p in m.parameters():
                p.requires_grad_(False)
            # 将模型 m 中所有的浮点型张量转换为长整型张量
            m._apply(lambda t: t.long())
            # 将输入张量 i 转换为长整型张量
            input = i.long()
            # 使用模型 m 进行前向计算，得到输出张量 out
            out = m.forward(input)
            return out

        # 将构造函数替换为生成长整型张量的构造函数
        test_params['constructor'] = gen_long_tensor_constructor(test_params['constructor'])
        # 将输入函数替换为生成长整型张量输入的函数
        test_params['input_fn'] = gen_long_tensor_input(test_params['input_size'])
        # 将参考函数设置为上面定义的 reference_fn
        test_params['reference_fn'] = reference_fn
        # 设置检查仅前向传播的标志为 True
        test_params['check_forward_only'] = True
        # 目前不支持在 CUDA 下使用长整型张量进行 conv2d/conv3d 测试
        test_params['test_cuda'] = False
        # 创建一个 NewModuleTest 对象，传入上述修改后的测试参数
        test = NewModuleTest(**test_params)

        # 添加测试 test 到测试集中，使用给定的装饰器
        add_test(test, decorator)
for test_params in criterion_tests:
    # 遍历 criterion_tests 中的每个测试参数字典
    if 'constructor' not in test_params:
        # 如果测试参数字典中没有 'constructor' 键
        name = test_params.pop('module_name')
        # 弹出 'module_name' 键的值并赋给 name
        test_params['constructor'] = getattr(nn, name)
        # 将 nn 模块中名为 name 的属性作为 'constructor' 的值赋给测试参数字典
    # 创建 CriterionTest 实例，传入 test_params 中的参数
    test = CriterionTest(**test_params)
    # 从 test_params 中弹出 'decorator' 键的值作为 decorator，若无则为 None
    decorator = test_params.pop('decorator', None)
    # 将创建的 test 实例和 decorator 添加到测试集中
    add_test(test, decorator)
    if 'check_sum_reduction' in test_params:
        # 如果测试参数字典中有 'check_sum_reduction' 键
        desc = test_params.get('desc', None)
        # 获取 'desc' 键的值，若无则为 None
        test_params['desc'] = 'sum_reduction' if desc is None else desc + '_sum_reduction'

        # 定义一个生成 sum reduction 构造函数的函数
        def gen_sum_reduction_constructor(constructor):
            def sum_reduction_constructor(*args, **kwargs):
                # 在 constructor 基础上创建一个新的构造函数，设置 reduction='sum'
                cons = constructor(*args, reduction='sum', **kwargs)
                return cons
            sum_reduction_constructor.__name__ = constructor.__name__
            return sum_reduction_constructor

        # 将生成的 sum reduction 构造函数应用于 'constructor' 键对应的值
        test_params['constructor'] = gen_sum_reduction_constructor(test_params['constructor'])
        # 创建新的 CriterionTest 实例，传入更新后的 test_params
        test = CriterionTest(**test_params)
        # 将新创建的 test 实例和 decorator 添加到测试集中
        add_test(test, decorator)


class UnpoolingNet(nn.Module):
    def __init__(self, pool, unpool):
        super().__init__()
        self.pool = pool
        self.unpool = unpool

    def forward(self, input):
        # 在 forward 方法中执行 unpool 操作和 pool 操作
        return self.unpool(*self.pool(input))


# 添加一个 NewModuleTest 实例，构造函数为 UnpoolingNet 实例化对象
add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool1d(2, return_indices=True),
        nn.MaxUnpool1d(2)),
    input_size=(1, 1, 4),
    fullname='MaxUnpool1d_net',
    default_dtype=torch.double,))
# 添加一个 NewModuleTest 实例，构造函数为 UnpoolingNet 实例化对象
add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool2d(2, return_indices=True),
        nn.MaxUnpool2d(2)),
    input_size=(1, 1, 2, 4),
    fullname='MaxUnpool2d_net',
    default_dtype=torch.double,))
# 添加一个 NewModuleTest 实例，构造函数为 UnpoolingNet 实例化对象
add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool3d(2, return_indices=True),
        nn.MaxUnpool3d(2)),
    input_size=(1, 1, 2, 4, 6),
    fullname='MaxUnpool3d_net',
    check_gradgrad=False,
    default_dtype=torch.double,))

# 添加一个 NewModuleTest 实例，构造函数为 UnpoolingNet 实例化对象，无批量维度
add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool1d(2, return_indices=True),
        nn.MaxUnpool1d(2)),
    input_size=(1, 4),
    reference_fn=single_batch_reference_fn,
    fullname='MaxUnpool1d_net_no_batch_dim',
    default_dtype=torch.double,))
# 添加一个 NewModuleTest 实例，构造函数为 UnpoolingNet 实例化对象，无批量维度
add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool2d(2, return_indices=True),
        nn.MaxUnpool2d(2)),
    input_size=(1, 2, 4),
    reference_fn=single_batch_reference_fn,
    fullname='MaxUnpool2d_net_no_batch_dim',
    default_dtype=torch.double,))

# 添加一个 NewModuleTest 实例，构造函数为 UnpoolingNet 实例化对象，无批量维度
add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool3d(2, return_indices=True),
        nn.MaxUnpool3d(2)),
    input_size=(1, 2, 4, 6),
    reference_fn=single_batch_reference_fn,
    fullname='MaxUnpool3d_net_no_batch_dim',
    check_gradgrad=False,
    default_dtype=torch.double,))

class _AdaptiveLogSoftmaxWithLoss(nn.AdaptiveLogSoftmaxWithLoss):
    # 继承自 nn.AdaptiveLogSoftmaxWithLoss 的类
    # 定义一个类的方法 `__call__`，用于执行对象的调用操作
    def __call__(self, input):
        # 创建一个张量 `t`，包含指定的数值，并将其移动到与 `input` 相同的设备上
        t = torch.tensor([0, 1, 4, 8]).to(input.device)
        # 调用父类 `nn.AdaptiveLogSoftmaxWithLoss` 的 `__call__` 方法，传入 `input` 和 `t` 参数，并返回其输出
        return nn.AdaptiveLogSoftmaxWithLoss.__call__(self, input, t).output
# 创建一个新的测试对象，并添加到测试套件中
add_test(NewModuleTest(
    constructor=lambda: _AdaptiveLogSoftmaxWithLoss(16, 10, [2, 6]),
    input_size=(4, 16),
    fullname='AdaptiveLogSoftmax',
    with_tf32=True,
    tf32_precision=0.005,
    default_dtype=torch.double))

# 下面是用于 TestNN.test_affine_* 的辅助函数

# 如果 CUDA 可用，则返回 ['cpu', 'cuda']，否则返回 ['cpu']
if torch.cuda.is_available():
    def device_():
        return ['cpu', 'cuda']
else:
    def device_():
        return ['cpu']

# 返回一个包含多个角度的列表，每个角度都转换为弧度
def angle_rad_():
    return [r * math.pi * 2 for r in [0.0, 0.5, 0.25, 0.125, random.random()]]

# 返回一个包含随机向量的列表，每个向量都是单位向量
def axis_vector_():
    t = (random.random(), random.random(), random.random())
    l = sum(x ** 2 for x in t) ** 0.5
    return [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), tuple(x / l for x in t)]

# 返回一个包含多个二维输入大小的列表
def input_size2d_():
    return [[1, 1, 3, 5], [1, 1, 3, 3], [1, 1, 4, 4], [1, 1, 3, 4]]

# 返回一个包含多个二维输出大小的列表
def output_size2d_():
    return [[1, 1, 5, 3], [1, 1, 3, 5], [1, 1, 4, 3], [1, 1, 5, 5], [1, 1, 6, 6]]

# 返回一个包含多个二维输入大小（正方形）的列表
def input_size2dsq_():
    return [[1, 1, 2, 2], [1, 1, 3, 3], [1, 1, 4, 4], [1, 1, 6, 6]]

# 返回一个包含多个二维输出大小（正方形）的列表
def output_size2dsq_():
    return [[1, 1, 2, 2], [1, 1, 3, 3], [1, 1, 4, 4], [1, 1, 5, 5], [1, 1, 6, 6]]

# 返回一个包含多个三维输入大小的列表
def input_size3d_():
    return [[1, 1, 2, 2, 2], [1, 1, 2, 3, 4], [1, 1, 3, 3, 3], [1, 1, 4, 4, 4], [1, 1, 3, 4, 5]]

# 返回一个包含多个三维输入大小（正方形）的列表
def input_size3dsq_():
    return [[1, 1, 2, 2, 2], [1, 1, 3, 3, 3], [1, 1, 4, 4, 4], [1, 1, 6, 6, 6]]

# 返回一个包含多个三维输出大小（正方形）的列表
def output_size3dsq_():
    return [[1, 1, 2, 2, 2], [1, 1, 3, 3, 3], [1, 1, 4, 4, 4], [1, 1, 5, 5, 5], [1, 1, 6, 6, 6]]

# 返回一个包含多个三维输出大小的列表
def output_size3d_():
    return [[1, 1, 2, 2, 2], [1, 1, 3, 3, 3], [1, 1, 3, 4, 5], [1, 1, 4, 3, 2], [1, 1, 5, 5, 5], [1, 1, 6, 6, 6]]

# 构建等效仿射变换的函数，参数包括设备、输入大小、输出大小和角度（弧度）
def _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad):
    # 计算输入和输出中心
    input_center = [(x - 1) / 2.0 for x in input_size]
    output_center = [(x - 1) / 2.0 for x in output_size]

    # 计算角度的正弦和余弦值
    s = math.sin(angle_rad)
    c = math.cos(angle_rad)

    # 构建输入变换矩阵
    intrans_ary = np.array([
        [1, 0, input_center[2]],
        [0, 1, input_center[3]],
        [0, 0, 1],
    ], dtype=np.float64)

    # 构建输入缩放矩阵
    inscale_ary = np.array([
        [input_center[2], 0, 0],
        [0, input_center[3], 0],
        [0, 0, 1],
    ], dtype=np.float64)

    # 构建旋转矩阵
    rotation_ary = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    # 构建输出缩放矩阵
    outscale_ary = np.array([
        [1.0 / output_center[2], 0, 0],
        [0, 1.0 / output_center[3], 0],
        [0, 0, 1],
    ], dtype=np.float64)

    # 构建输出变换矩阵
    outtrans_ary = np.array([
        [1, 0, -output_center[2]],
        [0, 1, -output_center[3]],
        [0, 0, 1],
    ], dtype=np.float64)

    # 构建重新排序矩阵
    reorder_ary = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    # 计算最终变换矩阵和网格矩阵
    transform_ary = np.dot(np.dot(np.dot(np.dot(
        intrans_ary,
        inscale_ary),
        rotation_ary.T),
        outscale_ary),
        outtrans_ary)
    grid_ary = np.dot(np.dot(np.dot(reorder_ary, rotation_ary.T), outscale_ary), outtrans_ary)
    # 将 NumPy 数组 rotation_ary 转换为 PyTorch 张量，并将其移动到指定的设备上，数据类型为 32 位浮点数
    transform_tensor = torch.from_numpy(rotation_ary).to(device, torch.float32)
    
    # 从 transform_tensor 中选择前两个元素，并在第 0 维上增加一个维度，得到一个形状为 (1, 2) 的张量
    transform_tensor = transform_tensor[:2].unsqueeze(0)
    
    # 返回三个变量 transform_tensor, transform_ary, grid_ary，此处假设这三个变量是函数的输出
    return transform_tensor, transform_ary, grid_ary
def _buildEquivalentAffineTransforms3d(device, input_size, output_size, angle_rad, axis_vector):
    # 计算输入中心点坐标，使用了输入尺寸减1再除以2
    input_center = [(x - 1) / 2.0 for x in input_size]
    # 计算输出中心点坐标，使用了输出尺寸减1再除以2
    output_center = [(x - 1) / 2.0 for x in output_size]

    # 计算旋转角度的正弦值和余弦值
    s = math.sin(angle_rad)
    c = math.cos(angle_rad)
    c1 = 1 - c

    # 创建输入变换矩阵，包括平移和缩放
    intrans_ary = np.array([
        [1, 0, 0, input_center[2]],
        [0, 1, 0, input_center[3]],
        [0, 0, 1, input_center[4]],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    # 创建输入缩放矩阵
    inscale_ary = np.array([
        [input_center[2], 0, 0, 0],
        [0, input_center[3], 0, 0],
        [0, 0, input_center[4], 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    # 解构旋转向量
    l, m, n = axis_vector
    # 创建科学旋转矩阵
    scipyRotation_ary = np.array([
        [l * l * c1 + c, m * l * c1 - n * s, n * l * c1 + m * s, 0],
        [l * m * c1 + n * s, m * m * c1 + c, n * m * c1 - l * s, 0],
        [l * n * c1 - m * s, m * n * c1 + l * s, n * n * c1 + c, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    # 解构旋转向量
    z, y, x = axis_vector
    # 创建PyTorch旋转矩阵
    torchRotation_ary = np.array([
        [x * x * c1 + c, y * x * c1 - z * s, z * x * c1 + y * s, 0],
        [x * y * c1 + z * s, y * y * c1 + c, z * y * c1 - x * s, 0],
        [x * z * c1 - y * s, y * z * c1 + x * s, z * z * c1 + c, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    # 创建输出缩放矩阵
    outscale_ary = np.array([
        [1.0 / output_center[2], 0, 0, 0],
        [0, 1.0 / output_center[3], 0, 0],
        [0, 0, 1.0 / output_center[4], 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    # 创建输出平移矩阵
    outtrans_ary = np.array([
        [1, 0, 0, -output_center[2]],
        [0, 1, 0, -output_center[3]],
        [0, 0, 1, -output_center[4]],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    # 创建重新排序矩阵，将坐标轴重新排列为ZYX
    reorder_ary = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    # 计算总的仿射变换矩阵
    transform_ary = np.dot(np.dot(np.dot(np.dot(
        intrans_ary,
        inscale_ary),
        np.linalg.inv(scipyRotation_ary)),
        outscale_ary),
        outtrans_ary)

    # 计算网格变换矩阵
    grid_ary = np.dot(np.dot(np.dot(reorder_ary, np.linalg.inv(scipyRotation_ary)), outscale_ary), outtrans_ary)

    # 创建PyTorch的旋转矩阵张量，并移动到指定设备
    transform_tensor = torch.from_numpy(torchRotation_ary).to(device, torch.float32)
    transform_tensor = transform_tensor[:3].unsqueeze(0)

    # 返回三个计算结果：PyTorch旋转矩阵张量、总的仿射变换矩阵、网格变换矩阵
    return transform_tensor, transform_ary, grid_ary
# end TestNN.test_affine_* helpers
    # 定义测试方法：_test_InstanceNorm_general，用于测试 InstanceNorm 类的通用功能
    def _test_InstanceNorm_general(self, cls, input, device, dtype=torch.float):
        # 获取输入张量的批次数 b 和通道数 c
        b, c = input.size(0), input.size(1)
        # 将输入张量移动到指定设备并设置数据类型，并标记需要计算梯度
        input_var = input.to(device=device, dtype=dtype).requires_grad_()

        # 创建 InstanceNorm 实例 IN，设置 epsilon 为 0，并移动到指定设备上
        IN = cls(c, eps=0).to(device, dtype)

        # 将输入张量输入 InstanceNorm 实例，获取输出
        output = IN(input_var)
        # 将输出重塑为形状为 (b*c, -1) 的张量
        out_reshaped = output.view(b * c, -1)

        # 计算输出张量各行的均值和方差（不进行无偏估计）
        mean = out_reshaped.mean(1)
        var = out_reshaped.var(1, unbiased=False)

        # 断言输出张量各行的绝对均值接近于 0，允许误差为 1e-5
        self.assertEqual(torch.abs(mean.data).mean(), 0, atol=1e-5, rtol=0)
        # 断言输出张量各行的绝对方差接近于 1，允许误差为 1e-5
        self.assertEqual(torch.abs(var.data).mean(), 1, atol=1e-5, rtol=0)

        # 检查评估模式下是否不改变行为
        grad_out = torch.randn_like(output)
        res1 = output.data.clone()
        # 计算输出关于输入的梯度，并克隆数据
        output.backward(grad_out)
        grad1 = input_var.grad.data.clone()

        # 将 InstanceNorm 实例切换到评估模式
        IN.eval()
        output = IN(input_var)
        # 清空输入张量的梯度
        input_var.grad = None
        # 再次计算输出关于输入的梯度
        output.backward(grad_out)
        res2 = output.data
        grad2 = input_var.grad.data
        # 断言两次计算的输出数据相等
        self.assertEqual(res1, res2)
        # 断言两次计算的梯度数据相等
        self.assertEqual(grad1, grad2)

        # 如果 track_running_stats=True 且 momentum=1，则 running_mean/var 应与输入的均值/方差（进行无偏估计）相等
        IN = cls(c, momentum=1, eps=0, track_running_stats=True).to(device, dtype)

        output = IN(input_var)

        # 将输入张量重新排列为 (c, -1)，计算每行的均值
        input_reshaped = input_var.transpose(1, 0).reshape(c, -1)
        mean = input_reshaped.mean(1)

        # 将输入张量重新排列为 (c, b, -1)，计算每行的方差（进行无偏估计）
        input_reshaped = input_var.transpose(1, 0).reshape(c, b, -1)
        var = input_reshaped.var(2, unbiased=True)[:, :]

        # 断言输出张量各行的绝对均值与 InstanceNorm 实例的 running_mean 接近，允许误差为 1e-5
        self.assertEqual(torch.abs(mean.data - IN.running_mean).mean(), 0, atol=1e-5, rtol=0)
        # 断言输出张量各行方差的平均值与 InstanceNorm 实例的 running_var 接近，允许误差为 1e-5
        self.assertEqual(torch.abs(var.data.mean(1) - IN.running_var).mean(), 0, atol=1e-5, rtol=0)

        # 在评估模式下，向输入的每个通道添加 X * std，应使输出的相应通道具有均值 X
        IN.eval()
        delta = IN.running_var.sqrt() * torch.arange(c, device=device, dtype=dtype)
        delta = delta.view(-1, *[1 for _ in range(2, input.dim())])
        output = IN(input_var + delta)
        # 断言输出张量重新排列后的每行的均值与 torch.arange(c) 的数据类型接近
        self.assertEqual(output.transpose(0, 1).reshape(c, -1).mean(1), torch.arange(c, dtype=dtype))
    # 测试 InstanceNorm 在 CUDA 半精度下的功能
    def _test_InstanceNorm_cuda_half(self, cls, input, device):
        # 将输入转换为指定的设备和半精度类型，并随机填充数据
        input = input.to(device=device, dtype=torch.half).random_(1, 10).requires_grad_(True)
        # 创建 InstanceNorm 模块，并设置为半精度类型
        m = cls(input.size(1), affine=True, track_running_stats=True).to(device, torch.half)
        # 使用 InstanceNorm 进行前向计算
        thnn_output = m(input)
        # 计算前向输出的梯度
        thnn_output.sum().backward()
        # 复制输入梯度数据
        thnn_input_grad = input.grad.data.clone()
        # 断言前向输出类型与输入类型相同
        self.assertEqualTypeString(thnn_output, input)

        # 如果启用了 cuDNN 测试
        if TEST_CUDNN:
            # 清空输入的梯度
            input.grad = None
            # 将模块转换为 float 类型
            m = m.float()
            # 使用 cuDNN 进行前向计算
            cudnn_output = m(input)
            # 计算 cuDNN 前向输出的梯度
            cudnn_output.sum().backward()
            # 复制 cuDNN 前向输入的梯度数据
            cudnn_input_grad = input.grad.data.clone()
            # 断言 cuDNN 前向输出与 THNN 前向输出相等，允许误差范围为 1e-4
            self.assertEqual(cudnn_output, thnn_output, atol=1e-4, rtol=0)
            # 断言 cuDNN 前向输入梯度与 THNN 前向输入梯度相等，允许误差范围为 1e-3
            self.assertEqual(cudnn_input_grad, thnn_input_grad, atol=1e-3, rtol=0)

    # 测试 LayerNorm 的通用功能
    def _test_LayerNorm_general(self, device, dtype=torch.float):
        # 遍历不同维度的测试情况
        for i in range(2, 6):
            # 随机生成张量的形状
            shape = torch.randint(3, 6, (i,), dtype=torch.long).tolist()
            # 在指定设备和数据类型下创建随机填充的张量
            x = torch.empty(*shape, device=device, dtype=dtype).uniform_(0, 10)
            # 随机选择要进行归一化的维度
            normalized_ndim = random.randint(1, i - 1)  # inclusive
            normalized_shape = shape[-normalized_ndim:]
            unnormalized_shape = shape[:-normalized_ndim]

            # 创建 LayerNorm 模块，设置 epsilon 为 0，并转换为指定的设备和数据类型
            ln = nn.LayerNorm(normalized_shape, eps=0).to(device, dtype)
            ln.weight.data.fill_(1)  # 设置权重为 1
            ln.bias.data.fill_(0)    # 设置偏置为 0
            # 应用 LayerNorm
            output = ln(x)
            # 将输出重新形状为未归一化的形状
            out_reshaped = output.view(*(unnormalized_shape + [-1]))
            # 计算每个维度上的均值和方差
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)

            # 根据数据类型设置不同的误差阈值
            delta = 1e-1 if (dtype == torch.bfloat16 or dtype == torch.half) else 1e-5
            # 断言均值的绝对值的平均值接近 0，允许误差范围为 delta
            self.assertEqual(torch.abs(mean.data).mean(), 0, atol=delta, rtol=0)
            # 断言方差的绝对值的平均值接近 1，允许误差范围为 delta
            self.assertEqual(torch.abs(var.data).mean(), 1, atol=delta, rtol=0)

            # 随机生成权重和偏置
            scale, bias = torch.empty(2).uniform_(0.2, 2).tolist()
            # 设置 LayerNorm 的权重和偏置
            ln.weight.data.fill_(scale)
            ln.bias.data.fill_(bias)
            # 再次应用 LayerNorm
            output = ln(x)
            # 将输出重新形状为未归一化的形状
            out_reshaped = output.view(*(unnormalized_shape + [-1]))
            # 计算每个维度上的均值和方差
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)
            # 断言均值的绝对值的平均值接近设置的偏置，允许误差范围为 delta
            self.assertEqual(torch.abs(mean.data).mean(), bias, atol=delta, rtol=0)
            # 断言方差的绝对值的平均值接近设置的权重的平方，允许误差范围为 delta
            self.assertEqual(torch.abs(var.data).mean(), scale ** 2, atol=delta, rtol=0)

        # 测试不良的归一化形状输入的情况
        bad_norm_shape_input_shape = {
            (): (),
            (2, 3): (3,),
            (2,): (1, 2, 3),
            (10,): (2, 3),
            10: (2, 3),
        }
        for norm_shape, input_shape in bad_norm_shape_input_shape.items():
            # 创建 LayerNorm 模块
            ln = nn.LayerNorm(norm_shape)
            # 在指定设备和数据类型下创建随机填充的张量
            input = torch.empty(input_shape, device=device, dtype=dtype).uniform_(0, 10)
            # 断言运行时错误被触发，因为传递了不良的归一化形状
            self.assertRaises(RuntimeError, lambda: ln(input))
    # 测试使用 CUDA 半精度进行 LayerNorm 操作
    def _test_LayerNorm_cuda_half(self, device):
        # 创建一个形状为 [2, 3, 3, 2] 的半精度空张量，填充随机数据，并要求计算梯度
        input = torch.empty(2, 3, 3, 2, device=device, dtype=torch.half).random_(1, 10).requires_grad_(True)
        # 创建一个 LayerNorm 模块，对形状为 [3, 2] 的输入进行标准化，使用半精度，并移动到指定设备上
        m = nn.LayerNorm([3, 2]).to(device, torch.half)
        # 对输入进行 LayerNorm 操作
        output = m(input)
        # 计算输出的所有元素之和的梯度
        output.sum().backward()
        # 断言输出类型与输入类型相同
        self.assertEqualTypeString(output, input)

    # 测试在 CPU 上混合数据类型进行 LayerNorm 操作
    def _test_LayerNorm_cpu_mixed_dtype(self, device, dtype):
        for elementwise_affine in [True, False]:
            # LayerNorm 输入形状规范化为 m x n，CPU 矢量化在 n 上进行，因此确保 n 超过矢量长度
            # 创建一个形状为 [2, 3, 11, 3] 的张量，填充随机数据，并将数据类型设置为指定的 dtype
            input = torch.empty(2, 3, 11, 3, device=device, dtype=dtype).random_(1, 10)
            # 创建一个 LayerNorm 模块，对形状为 [11, 3] 的输入进行标准化，支持元素级仿射变换，并移动到指定设备上
            m = nn.LayerNorm([11, 3], elementwise_affine=elementwise_affine).to(device, dtype)

            # 深拷贝一个 fp32 类型的 LayerNorm 模块，并移动到指定设备上
            m_fp32 = deepcopy(m).to(device, torch.float)
            # 克隆并分离输入张量，转换为 float 类型，并要求计算梯度
            x_fp32 = input.clone().detach().float().requires_grad_()
            # 对 fp32 类型的输入进行 LayerNorm 操作
            out_fp32 = m_fp32(x_fp32)
            # 计算输出的所有元素之和的梯度
            out_fp32.sum().backward()

            # 深拷贝一个 bf16/half 类型的 LayerNorm 模块
            m_bf16 = deepcopy(m)
            # 克隆并分离输入张量，并要求计算梯度
            x_bf16 = input.clone().detach().requires_grad_()
            # 对 bf16/half 类型的输入进行 LayerNorm 操作
            out_bf16 = m_bf16(x_bf16)
            # 计算输出的所有元素之和的梯度
            out_bf16.sum().backward()

            # 混合类型 bf16/half
            # 深拷贝一个 fp32 类型的 LayerNorm 模块，并移动到指定设备上
            m_mix = deepcopy(m).to(device, torch.float)
            # 克隆并分离输入张量，并要求计算梯度
            x_mix = input.clone().detach().requires_grad_()
            # 对混合类型的输入进行 LayerNorm 操作
            out_mix = m_mix(x_mix)
            # 计算输出的所有元素之和的梯度
            out_mix.sum().backward()

            # 断言 fp32 类型的输出转换为指定 dtype 后与 bf16 类型的输出相等
            self.assertEqual(out_fp32.to(dtype=dtype), out_bf16)
            # 断言 fp32 类型的输出转换为指定 dtype 后与混合类型的输出相等
            self.assertEqual(out_fp32.to(dtype=dtype), out_mix)
            # 断言 fp32 类型的输入梯度转换为指定 dtype 后与 bf16 类型的输入梯度相等，允许的绝对误差为 1e-1，相对误差为 1e-1
            self.assertEqual(x_fp32.grad.to(dtype=dtype), x_bf16.grad, atol=1e-1, rtol=1e-1)
            # 断言 fp32 类型的输入梯度转换为指定 dtype 后与混合类型的输入梯度相等，允许的绝对误差为 1e-1，相对误差为 1e-1
            self.assertEqual(x_fp32.grad.to(dtype=dtype), x_mix.grad, atol=1e-1, rtol=1e-1)
    # 定义一个测试函数，用于测试 GroupNorm 模块在通用情况下的行为
    def _test_GroupNorm_general(self, device, dtype=torch.float):
        # 定义符合规范的形状与分组数的字典
        good_shape_g = {
            (1, 2, 3, 4): 2,
            (2, 3, 10): 3,
            (3, 1, 1, 1, 2): 1,
            (2, 6, 4, 2, 2): 3,
            (1, 256, 1, 1): 32,
        }
        # 遍历字典中的每个形状与分组数组合
        for shape_g, grad in product(good_shape_g.items(), [True, False]):
            shape, g = shape_g
            # 创建一个指定设备和数据类型的空张量，填充为均匀分布的随机数
            x = torch.empty(*shape, device=device, dtype=dtype).uniform_(0, 10)
            x.requires_grad_(grad)
            b = shape[0]
            c = shape[1]

            # 创建一个 GroupNorm 层，指定分组数 g 和通道数 c，epsilon 设为 0
            gn = nn.GroupNorm(g, c, eps=0).to(device, dtype)
            # 设置 GroupNorm 层的权重为全 1，偏置为全 0
            gn.weight.data.fill_(1)
            gn.bias.data.fill_(0)
            # 将输入 x 输入到 GroupNorm 层中，并获取输出
            output = gn(x)
            # 将输出 reshape 为 (b, g, -1) 的形式
            out_reshaped = output.view(b, g, -1)
            # 计算输出在最后一个维度上的均值和方差
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)
            # 断言均值的绝对值的平均值接近 0，允许的误差为 1e-5
            self.assertEqual(torch.abs(mean).mean(), 0, atol=1e-5, rtol=0)
            # 断言方差的绝对值的平均值接近 1，允许的误差为 1e-5
            self.assertEqual(torch.abs(var).mean(), 1, atol=1e-5, rtol=0)

            # 对输出进行反向传播
            output.backward(torch.randn_like(output))
            if output.is_cuda:
                torch.cuda.synchronize()

            # 再次测试 GroupNorm 层应用权重和偏置的正确性
            # 创建新的权重和偏置张量，均匀分布于指定设备和数据类型上
            scale = torch.empty(c, device=device, dtype=dtype).uniform_(0.2, 2)
            bias = torch.empty(c, device=device, dtype=dtype).uniform_(0.2, 2)
            # 将新的权重和偏置值复制到 GroupNorm 层的权重和偏置中
            gn.weight.data.copy_(scale)
            gn.bias.data.copy_(bias)
            # 重新计算 GroupNorm 层的输出
            output = gn(x)
            out_reshaped = output.view(b, c, -1)
            out_normed = (out_reshaped - bias.view(c, 1)) / scale.view(c, 1)
            out_normed_reshaped = out_normed.view(b, g, -1)
            mean = out_normed_reshaped.mean(-1)
            var = out_normed_reshaped.var(-1, unbiased=False)
            # 再次断言均值的绝对值的平均值接近 0，允许的误差为 1e-5
            self.assertEqual(torch.abs(mean).mean(), 0, atol=1e-5, rtol=0)
            # 再次断言方差的绝对值的平均值接近 1，允许的误差为 1e-5
            self.assertEqual(torch.abs(var).mean(), 1, atol=1e-5, rtol=0)

        # 定义不符合规范的形状与分组数的字典
        bad_shape_g = {
            (1, 2, 3, 4): 3,
            (2, 3, 10): 2,
            (3, 1, 1, 1, 2): 10,
            (2, 6, 4, 2, 2): 4,
        }
        # 遍历字典中的每个形状与分组数组合
        for shape, g in bad_shape_g.items():
            # 断言创建 GroupNorm 层时应会引发 ValueError 异常
            with self.assertRaises(ValueError):
                gn = nn.GroupNorm(g, shape[1])

    # 定义一个测试函数，测试在 CUDA 环境下的半精度浮点数计算
    def _test_GroupNorm_cuda_half(self):
        # 创建一个在 CUDA 设备上的零张量，并将其转换为半精度浮点数
        input = torch.zeros(2, 4, 3, 2, requires_grad=True).cuda().half().random_(1, 10)
        # 创建一个 GroupNorm 层，设备为 "cuda"，数据类型为 torch.half
        m = nn.GroupNorm(2, 4).to("cuda", torch.half)
        # 将输入传入 GroupNorm 层，并获取输出
        output = m(input)
        # 对输出求和并进行反向传播
        output.sum().backward()
        # 断言输出的类型与输入的类型相同
        self.assertEqualTypeString(output, input)

    # 定义一个测试函数，测试模块处理空输入的情况
    def _test_module_empty_inputs(self, module, inputs):
        # 将输入列表中的每个输入张量设置为需要梯度
        for _inp in inputs:
            _inp.requires_grad_(True)
        # 将输入传入指定的模块，获取输出
        out = module(*inputs)
        # 创建一个与输出相同形状的随机梯度张量 gO，并对输出进行反向传播
        gO = torch.rand_like(out)
        out.backward(gO)

        # 遍历模块中的每个参数
        for p in module.parameters():
            # 如果参数需要梯度，断言其梯度张量为全零张量
            if p.requires_grad:
                self.assertEqual(p.grad, torch.zeros_like(p.grad))

        # 遍历输入列表中的每个输入张量
        for _inp in inputs:
            # 断言每个输入张量的梯度为全零张量
            self.assertEqual(_inp.grad, torch.zeros_like(_inp))
    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")

# 如果条件不满足，则跳过当前单元测试。条件包括未启用测试numpy或scipy，或者scipy版本低于1.0.0。


    @tf32_on_and_off()
    @bf32_on_and_off()

# 调用两个装饰器，用于在测试期间临时启用和禁用TensorFlow的32位浮点数精度（tf32）和混合精度（bf32）。


    def test_affine_2d_rotate0(self, device):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.

# 定义一个测试方法`test_affine_2d_rotate0`，测试2D仿射变换（角度为0），传入`device`作为参数。在scipy版本低于1.0.0时，由于不支持齐次坐标，需要跳过测试。


        input_size = [1, 1, 3, 3]
        input_ary = np.array(np.random.random(input_size), dtype=np.float32)
        output_size = [1, 1, 5, 5]
        angle_rad = 0.

# 定义输入数据大小为`[1, 1, 3, 3]`，创建随机浮点数数组`input_ary`，输出大小为`[1, 1, 5, 5]`，角度`angle_rad`为0。


        transform_tensor, transform_ary, offset = \
            _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

# 调用`_buildEquivalentAffineTransforms2d`函数生成2D仿射变换所需的变换张量`transform_tensor`、变换数组`transform_ary`和偏移量`offset`。


        scipy_ary = torch.from_numpy(scipy.ndimage.affine_transform(
            input_ary[0, 0],
            transform_ary,
            offset=offset,
            output_shape=output_size[2:],
            order=1,
            mode='nearest',
            prefilter=False))

# 使用scipy对输入数组的第一个通道进行仿射变换，生成`scipy_ary`，用于比较后续的格点采样结果。


        affine_tensor = torch.nn.functional.affine_grid(
            transform_tensor,
            torch.Size(output_size),
            align_corners=True
        )

# 使用PyTorch的`affine_grid`函数创建仿射变换所需的网格张量`affine_tensor`，输出大小为`output_size`，角点对齐方式为True。


        gridsample_ary = torch.nn.functional.grid_sample(
            torch.tensor(input_ary, device=device).to(device),
            affine_tensor,
            padding_mode='border',
            align_corners=True
        ).to('cpu')

# 使用PyTorch的`grid_sample`函数对输入数据进行格点采样，生成`gridsample_ary`，采用边界填充模式'border'，角点对齐方式为True。


        self.assertEqual(scipy_ary.mean(), gridsample_ary.mean())
        self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))

# 断言两个仿射变换结果的均值和形状是否相等，以验证PyTorch实现的仿射变换功能与scipy的结果一致。
    def test_affine_2d_rotate90(self, device):
        # scipy 在 1.0.0 版本之前不支持齐次坐标
        # scipy.ndimage.affine_transform，因此我们需要跳过这部分。

        # 使用 itertools.product 遍历输入和输出尺寸的组合
        for input_size2dsq, output_size2dsq in \
                itertools.product(input_size2dsq_(), output_size2dsq_()):
            # 设置输入尺寸
            input_size = input_size2dsq
            # 创建指定尺寸的随机浮点数数组作为输入数组
            input_ary = np.array(np.random.random(input_size), dtype=np.float32)
            # 设置输出尺寸
            output_size = output_size2dsq
            # 设置旋转角度为 0.25 * π * 2 弧度
            angle_rad = 0.25 * math.pi * 2

            # 使用 _buildEquivalentAffineTransforms2d 函数构建等效的二维仿射变换
            transform_tensor, transform_ary, offset = \
                _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

            # 使用 scipy.ndimage.affine_transform 对输入数组进行仿射变换
            scipy_ary = torch.from_numpy(scipy.ndimage.affine_transform(
                input_ary[0, 0],  # 输入的第一个通道的第一个平面
                transform_ary,    # 变换矩阵
                offset=offset,    # 偏移量
                output_shape=output_size[2:],  # 输出的形状
                order=1,          # 插值阶数
                mode='nearest',   # 插值模式
                prefilter=True    # 是否对输入进行预滤波
            ))

            # 如果输入和输出尺寸相同，则检查仿射变换后的数组的平均值是否与输入数组相同
            if input_size2dsq == output_size2dsq:
                self.assertEqual(scipy_ary.mean(), input_ary.mean())
            # 检查仿射变换后的数组的指定位置的值是否与输入数组对应位置的值相同
            self.assertEqual(scipy_ary[0, 0], input_ary[0, 0, 0, -1])
            self.assertEqual(scipy_ary[0, -1], input_ary[0, 0, -1, -1])
            self.assertEqual(scipy_ary[-1, -1], input_ary[0, 0, -1, 0])
            self.assertEqual(scipy_ary[-1, 0], input_ary[0, 0, 0, 0])

            # 使用 torch.nn.functional.affine_grid 函数生成仿射变换的网格
            affine_tensor = torch.nn.functional.affine_grid(
                transform_tensor,     # 仿射变换矩阵
                torch.Size(output_size),  # 输出的尺寸
                align_corners=True   # 是否对齐角点
            )

            # 使用 torch.nn.functional.grid_sample 对输入数组进行网格采样
            gridsample_ary = torch.nn.functional.grid_sample(
                torch.tensor(input_ary, device=device).to(device),  # 将输入数组转换为张量并移动到指定设备上
                affine_tensor,      # 仿射变换的网格
                padding_mode='border',  # 填充模式
                align_corners=True  # 是否对齐角点
            ).to('cpu')

            # 检查仿射变换后的数组的平均值是否与网格采样后的数组的平均值相同
            self.assertEqual(scipy_ary.mean(), gridsample_ary.mean())
            # 检查仿射变换后的数组是否与网格采样后的数组形状相同
            self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    @tf32_on_and_off(0.005)
    @bf32_on_and_off(0.005)
    def test_affine_2d_rotate45(self, device):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        # 定义输入张量的大小
        input_size = [1, 1, 3, 3]
        # 创建一个全零的输入数组，并将其类型设置为 float32
        input_ary = np.array(np.zeros(input_size), dtype=np.float32)
        # 设置输入数组的特定元素值
        input_ary[0, 0, 0, :] = 0.5
        input_ary[0, 0, 2, 2] = 1.0
        # 定义输出张量的大小
        output_size = [1, 1, 3, 3]
        # 定义旋转角度（弧度）
        angle_rad = 0.125 * math.pi * 2

        # 调用 _buildEquivalentAffineTransforms2d 函数获取仿射变换所需的张量、数组和偏移量
        transform_tensor, transform_ary, offset = \
            _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

        # 使用 scipy.ndimage.affine_transform 对输入数组进行仿射变换
        scipy_ary = torch.from_numpy(scipy.ndimage.affine_transform(
            input_ary[0, 0],
            transform_ary,
            offset=offset,
            output_shape=output_size[2:],
            order=1,
            mode='nearest',
            prefilter=False))

        # 使用 torch.nn.functional.affine_grid 生成仿射变换后的网格
        affine_tensor = torch.nn.functional.affine_grid(
            transform_tensor,
            torch.Size(output_size),
            align_corners=True
        )

        # 使用 torch.nn.functional.grid_sample 对输入数组进行网格采样
        gridsample_ary = torch.nn.functional.grid_sample(
            torch.tensor(input_ary, device=device).to(device),
            affine_tensor,
            padding_mode='border',
            align_corners=True
        ).to('cpu')

        # 断言两个数组是否相等
        self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))

    @onlyCUDA
    @largeTensorTest("60GB", "cpu")
    @largeTensorTest("16GB", "cuda")
    def test_avg_pool_large_tensor(self, device):
        # test for https://github.com/pytorch/pytorch/issues/113833
        # 创建一个指定设备上的随机张量，用于测试大张量平均池化
        a = torch.randn(128, 256, 256, 256, dtype=torch.half, device=device, requires_grad=True)
        # 将 a 的副本转移到 CPU 上，并转换为 float 类型
        a_cpu = a.detach().cpu().float()
        # 创建 AvgPool2d 池化层
        m = torch.nn.AvgPool2d(2)
        # 对输入张量进行平均池化
        o = m(a)
        # 将 a_cpu 设置为需要梯度计算
        a_cpu.requires_grad = True
        # 对 o 进行求和并反向传播
        o.sum().backward()
        # 对 a_cpu 进行平均池化
        o_cpu = m(a_cpu)
        # 对 o_cpu 进行求和并反向传播
        o_cpu.sum().backward()
        # 断言两个 CPU 上梯度的 half 类型的张量是否近似相等
        self.assertTrue(torch.allclose(a.grad.cpu(), a_cpu.grad.half()))

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    @tf32_on_and_off(0.005)
    @bf32_on_and_off(0.005)
    # 定义一个测试函数，用于测试二维仿射变换的随机旋转操作
    def test_affine_2d_rotateRandom(self, device):
        # 在早于1.0.0版本的 scipy 中，不支持齐次坐标的 scipy.ndimage.affine_transform，
        # 因此我们需要跳过这部分测试。
        for angle_rad, input_size2d, output_size2d in \
                itertools.product(angle_rad_(), input_size2d_(), output_size2d_()):

            input_size = input_size2d
            # 创建一个指定大小的随机浮点数数组作为输入数据
            input_ary = np.array(np.random.random(input_size), dtype=np.float32).round(3)
            output_size = output_size2d

            # 在输入数组的指定位置设置特定的值
            input_ary[0, 0, 0, 0] = 2
            input_ary[0, 0, 0, -1] = 4
            input_ary[0, 0, -1, 0] = 6
            input_ary[0, 0, -1, -1] = 8

            # 调用 _buildEquivalentAffineTransforms2d 函数，构建等效的二维仿射变换
            transform_tensor, transform_ary, grid_ary = \
                _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

            # 使用 scipy.ndimage.affine_transform 函数对输入数组进行仿射变换
            scipy_ary = torch.from_numpy(scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                output_shape=output_size[2:],
                order=1,
                mode='nearest',
                prefilter=False))

            # 使用 torch.nn.functional.affine_grid 函数创建仿射变换的输出网格
            affine_tensor = torch.nn.functional.affine_grid(
                transform_tensor,
                torch.Size(output_size),
                align_corners=True
            )

            # 使用 torch.nn.functional.grid_sample 函数对输入数据进行网格采样
            gridsample_ary = torch.nn.functional.grid_sample(
                torch.tensor(input_ary, device=device).to(device),
                affine_tensor,
                padding_mode='border',
                align_corners=True
            ).to('cpu')

            # 将仿射变换的输出网格转移到 CPU 上
            affine_tensor = affine_tensor.to('cpu')

            # 遍历仿射变换的输出网格，验证其与预期的网格输出一致性
            for r in range(affine_tensor.size(1)):
                for c in range(affine_tensor.size(2)):
                    grid_out = np.dot(grid_ary, [r, c, 1])
                    self.assertEqual(affine_tensor[0, r, c], grid_out[:2], exact_dtype=False)

            # 使用断言函数验证 grid_sample 函数的输出与 scipy 仿射变换的输出一致
            self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))

    # 使用 unittest.skipIf 装饰器，根据条件跳过测试，条件包括未安装 numpy 或 scipy 或 scipy 版本低于 '1.0.0'
    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    @tf32_on_and_off(0.005)
    @bf32_on_and_off(0.005)
    # 定义一个测试方法，用于测试三维仿射变换中随机旋转的情况
    def test_affine_3d_rotateRandom(self, device):
        # 在 scipy 版本低于 1.0.0 时，不支持齐次坐标，因此需要跳过这部分测试
        for angle_rad, axis_vector, input_size3d, output_size3d in \
                itertools.product(angle_rad_(), axis_vector_(), input_size3d_(), output_size3d_()):
            # 设置输入大小为三维大小
            input_size = input_size3d
            # 创建一个随机数值填充的指定大小的浮点数 NumPy 数组
            input_ary = np.array(np.random.random(input_size), dtype=np.float32)
            # 设置输出大小为三维大小
            output_size = output_size3d

            # 修改输入数组的特定索引位置的数值
            input_ary[0, 0, 0, 0, 0] = 2
            input_ary[0, 0, 0, 0, -1] = 3
            input_ary[0, 0, 0, -1, 0] = 4
            input_ary[0, 0, 0, -1, -1] = 5
            input_ary[0, 0, -1, 0, 0] = 6
            input_ary[0, 0, -1, 0, -1] = 7
            input_ary[0, 0, -1, -1, 0] = 8
            input_ary[0, 0, -1, -1, -1] = 9

            # 调用 _buildEquivalentAffineTransforms3d 函数生成仿射变换的 tensor、数组和网格数组
            transform_tensor, transform_ary, grid_ary = \
                _buildEquivalentAffineTransforms3d(device, input_size, output_size, angle_rad, axis_vector)

            # 使用 scipy 进行三维仿射变换，输出结果转换为 PyTorch tensor
            scipy_ary = torch.from_numpy(scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                output_shape=output_size[2:],
                order=1,
                mode='nearest',
                prefilter=False))

            # 使用 PyTorch 的 affine_grid 函数生成仿射变换的 tensor
            affine_tensor = torch.nn.functional.affine_grid(
                transform_tensor,
                torch.Size(output_size),
                align_corners=True
            )

            # 使用 PyTorch 的 grid_sample 函数进行仿射变换操作，将结果存储在 gridsample_ary 中
            gridsample_ary = torch.nn.functional.grid_sample(
                torch.tensor(input_ary, device=device).to(device),
                affine_tensor,
                padding_mode='border',
                align_corners=True
            ).to('cpu')

            # 将 affine_tensor 转移到 CPU 上
            affine_tensor = affine_tensor.to('cpu')

            # 遍历 affine_tensor 的通道、行、列维度，计算仿射变换后的网格输出
            for i in range(affine_tensor.size(1)):
                for r in range(affine_tensor.size(2)):
                    for c in range(affine_tensor.size(3)):
                        grid_out = np.dot(grid_ary, [i, r, c, 1])
                        # 使用断言检查仿射变换后的 tensor 是否与计算的网格输出匹配
                        self.assertEqual(affine_tensor[0, i, r, c], grid_out[:3], exact_dtype=False)

            # 使用断言检查 grid_sample 函数得到的 gridsample_ary 是否与 scipy_ary 形状一致
            self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))


    # 定义一个测试方法，用于测试在 CUDA 下进行大批量数据的批归一化操作
    @onlyCUDA
    @dtypes(torch.float, torch.half)
    def test_batchnorm_large_batch(self, device, dtype):
        # 在指定设备和数据类型下创建一个 BatchNorm2d 实例
        bn = nn.BatchNorm2d(1).to(device, dtype)
        # 生成一个指定大小的随机数张量作为输入数据，用于批归一化的计算
        data = torch.rand(880801, 1, 1, 1, device=device, dtype=dtype)
        # 对批数据进行归一化操作，并计算其梯度
        out = bn(data).sum().backward()

    # 根据 CUDA 的情况选择数据类型进行测试
    @dtypesIfCUDA(torch.float, torch.double, torch.half, torch.complex128)
    @dtypes(torch.float, torch.double, torch.bfloat16, torch.complex128)
    # 定义测试方法，用于测试卷积操作处理空输入的情况
    def test_conv_empty_input(self, device, dtype):
        # 定义帮助函数，用于执行卷积操作，并验证结果
        def help(input, conv, memory_format):
            # 执行常规卷积操作
            ref_out = conv(input)
            # 将卷积层转换为指定的内存格式
            conv_cl = conv.to(memory_format=memory_format)
            # 使用转换后的卷积层执行卷积操作
            out_cl = conv_cl(input)
            # 断言常规卷积输出与转换后卷积输出相等
            self.assertEqual(ref_out, out_cl)
            # 将输入数据转换为指定的内存格式
            input_cl = input.to(memory_format=memory_format)
            # 使用转换后的输入数据执行卷积操作
            out_cl2 = conv(input_cl)
            # 断言转换后卷积输出与转换后输入数据的卷积输出相等
            self.assertEqual(out_cl, out_cl2)
            # 使用转换后的卷积层再次执行卷积操作
            out_cl3 = conv_cl(input_cl)
            # 断言第二次转换后卷积输出与第一次转换后卷积输出相等
            self.assertEqual(out_cl, out_cl3)

        # channels_last 情况下的测试
        input2d = torch.randn((0, 4, 20, 20)).to(device=device, dtype=dtype)
        conv2d = torch.nn.Conv2d(4, 4, 3, 1).to(device=device, dtype=dtype)
        help(input2d, conv2d, torch.channels_last)

        # channels_last_3d 情况下的测试
        input3d = torch.randn((0, 4, 20, 20, 20)).to(device=device, dtype=dtype)
        conv3d = torch.nn.Conv3d(4, 4, 3, 1).to(device=device, dtype=dtype)
        help(input3d, conv3d, torch.channels_last_3d)

        # 非连续内存情况下的测试
        weight = torch.rand(4, 8, 3, 3)[:, ::2, :, :].to(device=device, dtype=dtype)
        bias = torch.rand(4).to(device=device, dtype=dtype)
        # 执行非连续内存情况下的卷积操作
        out = F.conv2d(input2d, weight, bias, (1, 1), 0, (1, 1), 1)
        # 将权重张量转换为连续内存
        weight = weight.contiguous()
        # 使用连续内存的权重张量执行卷积操作
        out_ref = F.conv2d(input2d, weight, bias, (1, 1), 0, (1, 1), 1)
        # 断言连续内存和非连续内存情况下的卷积结果相等
        self.assertEqual(out_ref, out)

        # 在空输入情况下报告 sigfpe 错误，参见 https://github.com/pytorch/pytorch/issues/94125
        with self.assertRaises(RuntimeError):
            # 创建空输入张量和权重张量，调用慢速 3D 卷积操作
            inp = torch.empty([1, 1, 1, 0], dtype=dtype, device=device)
            weight = torch.empty([1, 0, 1], dtype=dtype, device=device)
            torch._C._nn.slow_conv3d(inp, weight, 1)

        # 断言引发的 RuntimeError 包含特定的错误信息，指示期望 2D kernel_size
        with self.assertRaisesRegex(RuntimeError, re.escape("2D kernel_size expected")):
            # 调用 THNN 2D 卷积操作，使用不合法的 kernel_size 参数
            torch._C._nn.thnn_conv2d(torch.rand([1, 1, 1, 1]), kernel_size=[], padding=[1, 1], stride=[1, 1],
                                     weight=torch.rand([1, 1]))

        # 断言引发的 RuntimeError 包含特定的错误信息，指示期望 2D stride
        with self.assertRaisesRegex(RuntimeError, re.escape("2D stride expected")):
            # 调用 THNN 2D 卷积操作，使用不合法的 stride 参数
            torch._C._nn.thnn_conv2d(torch.rand([1, 1, 1, 1]), kernel_size=[1, 1], padding=[1, 1], stride=[],
                                     weight=torch.rand([1, 1]))

        # 断言引发的 RuntimeError 包含特定的错误信息，指示期望 2D padding
        with self.assertRaisesRegex(RuntimeError, re.escape("2D padding expected")):
            # 调用 THNN 2D 卷积操作，使用不合法的 padding 参数
            torch._C._nn.thnn_conv2d(torch.rand([1, 1, 1, 1]), kernel_size=[1, 1], padding=[], stride=[1, 1],
                                     weight=torch.rand([1, 1]))

    # 测试 InstanceNorm1d 的一般情况
    def test_InstanceNorm1d_general(self, device):
        # 随机生成 batch size、channel 数和长度
        b = random.randint(3, 5)
        c = random.randint(3, 5)
        d = random.randint(8, 10)

        # 生成随机输入张量
        input = torch.rand(b, c, d)
        # 调用 _test_InstanceNorm_general 方法测试 InstanceNorm1d 的一般情况
        self._test_InstanceNorm_general(nn.InstanceNorm1d, input, device)

        # 如果设备类型为 'cuda'，调用 _test_InstanceNorm_cuda_half 方法测试 InstanceNorm1d 在 CUDA 半精度下的情况
        if self.device_type == 'cuda':
            self._test_InstanceNorm_cuda_half(nn.InstanceNorm1d, input, device)
    # 测试 InstanceNorm2d 的通用功能
    def test_InstanceNorm2d_general(self, device):
        # 随机生成 batch size
        b = random.randint(3, 5)
        # 随机生成通道数
        c = random.randint(3, 5)
        # 随机生成宽度
        w = random.randint(3, 6)
        # 随机生成高度
        h = random.randint(6, 8)

        # 创建随机数据张量
        input = torch.rand(b, c, h, w)
        # 调用 _test_InstanceNorm_general 方法，测试 InstanceNorm2d 的通用功能
        self._test_InstanceNorm_general(nn.InstanceNorm2d, input, device)

        # 如果设备类型为 'cuda'，则测试 InstanceNorm2d 的半精度 CUDA 功能
        if self.device_type == 'cuda':
            self._test_InstanceNorm_cuda_half(nn.InstanceNorm2d, input, device)

    # 测试 InstanceNorm3d 的通用功能
    def test_InstanceNorm3d_general(self, device):
        # 随机生成 batch size
        b = random.randint(3, 5)
        # 随机生成通道数
        c = random.randint(3, 5)
        # 随机生成宽度
        w = random.randint(2, 5)
        # 随机生成高度
        h = random.randint(2, 5)
        # 随机生成深度
        d = random.randint(2, 5)

        # 创建随机数据张量
        input = torch.rand(b, c, h, w, d)
        # 调用 _test_InstanceNorm_general 方法，测试 InstanceNorm3d 的通用功能
        self._test_InstanceNorm_general(nn.InstanceNorm3d, input, device)

        # 如果设备类型为 'cuda'，则测试 InstanceNorm3d 的半精度 CUDA 功能
        if self.device_type == 'cuda':
            self._test_InstanceNorm_cuda_half(nn.InstanceNorm3d, input, device)

    # 使用参数化测试的方式，测试当输入通道数不等于 num_features 时，InstanceNorm 抛出错误
    @parametrize_test("instance_norm_cls", [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d], name_fn=lambda c: c.__name__)
    @parametrize_test("no_batch_dim", [True, False])
    @parametrize_test("affine", [True, False])
    def test_instancenorm_raises_error_if_input_channels_is_not_num_features(self, device, instance_norm_cls, no_batch_dim, affine):
        # 创建 instance_norm_cls 的实例，设置通道数为 4，affine 参数取决于参数化
        inst_norm = instance_norm_cls(4, affine=affine)
        # 根据是否有 batch 维度，调整输入数据的大小
        size = [2] * inst_norm._get_no_batch_dim()
        if not no_batch_dim:
            size = [3] + size
        # 创建随机数据张量
        t = torch.randn(size)
        # 如果 affine 为 True，预期 inst_norm(t) 抛出 ValueError
        if affine:
            with self.assertRaisesRegex(ValueError, "expected input's size at dim="):
                inst_norm(t)
        # 如果 affine 为 False，预期 inst_norm(t) 会发出警告，警告信息表明因 affine=False 而未使用
        else:
            with warnings.catch_warnings(record=True) as w:
                inst_norm(t)
            self.assertIn("which is not used because affine=False", str(w[0].message))

    # 测试当输入张量的每个通道值少于一时，InstanceNorm 抛出 ValueError
    def test_instancenorm_raises_error_if_less_than_one_value_per_channel(self, device):
        # 创建一个张量 x，形状为 [1, 10, 1]
        x = torch.rand(10)[None, :, None]
        # 预期 InstanceNorm1d(10)(x) 抛出 ValueError
        with self.assertRaises(ValueError):
            torch.nn.InstanceNorm1d(10)(x).to(device)

    # 测试训练时，当输入张量的某个空间元素为单一值时，InstanceNorm 抛出 ValueError
    def test_instancenorm_raises_error_for_single_spatial_element_during_training(self, device):
        # 定义 BATCH_SIZE 和 NUM_CHANNELS
        BATCH_SIZE = 10
        NUM_CHANNELS = 3
        norms = [torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d]
        for i, norm in enumerate(norms):
            # 创建指定类型的 InstanceNorm 实例 m
            m = norm(NUM_CHANNELS, track_running_stats=True)
            m.to(device)

            # 创建一个具有单一空间元素的适当大小的输入张量
            input = torch.randn(BATCH_SIZE, NUM_CHANNELS, *[1 for _ in range(i + 1)],
                                device=device)
            # 预期 m(input) 抛出 ValueError
            with self.assertRaises(ValueError):
                m(input)

            # 在评估模式下，单一空间元素应该是正常的
            m.eval()
            m(input)
    # 测试 LayerNorm 的通用功能，对指定设备进行测试
    def test_LayerNorm_general(self, device):
        # 调用内部函数测试 LayerNorm 的通用功能
        self._test_LayerNorm_general(device)

        # 如果设备类型是 'cuda' 或 'cpu'，则进入以下条件分支
        if self.device_type == 'cuda' or self.device_type == 'cpu':
            # 遍历 torch.half 和 torch.bfloat16 两种数据类型
            for dtype in [torch.half, torch.bfloat16]:
                # 调用内部函数测试 LayerNorm 的通用功能，指定数据类型
                self._test_LayerNorm_general(device, dtype=dtype)

        # 如果设备类型是 'cuda'，则进入以下条件分支
        if self.device_type == 'cuda':
            # 调用内部函数测试 LayerNorm 在半精度数据类型下的 CUDA 功能
            self._test_LayerNorm_cuda_half(device)

        # 如果设备类型是 'cpu'，则进入以下条件分支
        if self.device_type == 'cpu':
            # 遍历 torch.half 和 torch.bfloat16 两种数据类型
            for dtype in [torch.half, torch.bfloat16]:
                # 调用内部函数测试 LayerNorm 在混合数据类型的 CPU 功能
                self._test_LayerNorm_cpu_mixed_dtype(device, dtype=dtype)

    # 仅适用于原生设备类型的测试函数修饰器
    @onlyNativeDeviceTypes
    def test_LayerNorm_numeric(self, device):
        # 定义用于参考的 LayerNorm 函数
        def layer_norm_ref(X, gamma, beta, normalized_shape, eps):
            # 计算特征大小
            feature_size = np.prod(normalized_shape)
            # 将输入张量 X 重塑为二维视图
            X_view = X.view(-1, feature_size)
            # 计算均值和方差
            mean = X_view.mean(dim=-1, keepdim=True)
            var = X_view.var(dim=-1, unbiased=False, keepdim=True)
            # 计算 LayerNorm 后的输出 Y
            Y = (X_view - mean) / torch.sqrt(var + eps)
            Y = Y * gamma.view(-1) + beta.view(-1)
            return Y.view(*X.size())

        # 指定规范化尺寸
        normalized_shape = [256, 256, 144]
        # 创建并移动到指定设备的 LayerNorm 层
        layer_norm = nn.LayerNorm(normalized_shape).float().to(device)
        # 创建随机张量 X，指定数据类型和设备
        X = torch.rand(2, *normalized_shape, dtype=torch.float32,
                       device=device)

        # 对张量 X 应用 LayerNorm
        Y = layer_norm(X)
        # 调用参考函数计算 LayerNorm 的参考结果 Y_ref
        Y_ref = layer_norm_ref(X, layer_norm.weight.data, layer_norm.bias.data,
                               normalized_shape, layer_norm.eps)
        # 断言 LayerNorm 的计算结果 Y 与参考结果 Y_ref 相等
        self.assertEqual(Y, Y_ref, rtol=0, atol=1e-5)

        # 如果设备类型是 'cuda'，则进入以下条件分支
        if self.device_type == 'cuda':
            # 将 LayerNorm 层移动到 CPU 上
            layer_norm.cpu()
            # 在 CPU 上应用 LayerNorm，并进行断言
            Y_cpu = layer_norm(X.cpu())
            self.assertEqual(Y_cpu, Y, rtol=0, atol=1e-5)

    # 仅适用于 CPU 设备的测试函数修饰器
    @onlyCPU
    def test_glu_bfloat16(self, device):
        # 定义测试数据类型的内部函数
        def test_dtype(fn, input, dtype):
            # 将输入张量转换为指定数据类型，并启用梯度追踪
            input = input.detach().clone().to(dtype=dtype).requires_grad_(True)
            # 克隆输入张量并转换为 float 类型，并启用梯度追踪
            input2 = input.detach().clone().float().requires_grad_(True)
            # 调用给定函数计算输出
            out = fn(input)
            # 对输出求和并反向传播梯度
            out.sum().backward()
            # 再次调用给定函数计算输出
            out2 = fn(input2)
            # 对输出求和并反向传播梯度
            out2.sum().backward()
            # 断言输出的数据类型为指定的 dtype
            self.assertEqual(out.dtype, dtype)
            # 断言输入张量的梯度数据类型为指定的 dtype
            self.assertEqual(input.grad.dtype, dtype)
            # 断言两次运算的输出结果相等，允许数据类型不完全相同
            self.assertEqual(out, out2, exact_dtype=False)
            # 断言两次运算的输入张量梯度相等，允许数据类型不完全相同
            self.assertEqual(input.grad, input2.grad, atol=1e-2, rtol=0, exact_dtype=False)

        # 定义返回指定维度的 GLU 层函数
        def func(device):
            return torch.nn.GLU(dim=-1).to(device)

        # 定义不同形状的输入张量
        shapes = [[1, 3, 1, 6], [1, 3, 1, 128], [1, 3, 256, 256]]
        # 遍历不同形状的输入张量
        for shape in shapes:
            # 在指定设备上创建随机张量 x
            x = torch.randn(shape, device=device)
            # 调用测试数据类型函数，测试 GLU 层的 bfloat16 数据类型
            test_dtype(func(device), x, torch.bfloat16)

    # 仅适用于原生设备类型的测试函数修饰器
    @onlyNativeDeviceTypes
    def test_GroupNorm_general(self, device):
        # 调用内部函数测试 GroupNorm 的通用功能
        self._test_GroupNorm_general(device)

        # 如果设备类型是 'cuda'，则进入以下条件分支
        if self.device_type == 'cuda':
            # 调用内部函数测试 GroupNorm 在半精度数据类型下的 CUDA 功能
            self._test_GroupNorm_cuda_half()

        # 如果设备类型是 'cpu'，则进入以下条件分支
        if self.device_type == 'cpu':
            # 调用内部函数测试 GroupNorm 在混合数据类型的 CPU 功能
            self._test_GroupNorm_cpu_mixed_dtype()
    # 定义一个测试方法，验证如果每个分组中只有一个值，GroupNorm 是否会抛出 ValueError 异常
    def test_GroupNorm_raises_error_if_one_value_per_group(self, device):
        # 生成一个形状为 (1, 10, 1) 的随机张量 x
        x = torch.rand(10)[None, :, None]
        # 使用 assertRaises 验证在调用 GroupNorm(10, 10) 时会抛出 ValueError 异常
        with self.assertRaises(ValueError):
            torch.nn.GroupNorm(10, 10)(x).to(device)

    # 定义一个测试方法，验证在输入为空的情况下 GroupNorm 的行为
    def test_GroupNorm_empty(self, device):
        # 创建一个 GroupNorm 模块，分组数为 2，特征数为 4，将其移动到指定设备上
        mod = torch.nn.GroupNorm(2, 4).to(device)
        # 创建一个形状为 (0, 4, 2, 2) 的随机输入张量 inp，移动到指定设备上
        inp = torch.randn(0, 4, 2, 2, device=device)
        # 调用 _test_module_empty_input 方法测试空输入的模块行为
        _test_module_empty_input(self, mod, inp)
        
        # 如果当前设备类型为 'cuda' 并且支持 cuDNN 加速
        if self.device_type == 'cuda' and self.has_cudnn():
            # 在禁用 cuDNN 加速的情况下，再次调用 _test_module_empty_input 方法测试模块行为
            with torch.backends.cudnn.flags(enabled=False):
                _test_module_empty_input(self, mod, inp)

    # 应用装饰器，限制以下测试方法只在 CPU 上运行，并指定支持的数据类型
    @onlyCPU
    @dtypes(torch.float, torch.double, torch.bfloat16, torch.half)
    # 定义一个测试方法，用于测试 GroupNorm 在 NHWC 格式下的行为
    def test_groupnorm_nhwc(self, device, dtype):
        # 定义一个辅助函数，用于执行具体的测试
        def helper(self, size, groups, memory_format, is_mixed):
            # 获取输入张量的通道数
            channels = size[1]
            # 生成一个随机张量作为输入，指定数据类型和设备，并开启梯度追踪
            input = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
            # 将输入张量转换为指定的内存布局格式
            input = input.contiguous(memory_format=memory_format)
            # 保留输入张量的梯度
            input.retain_grad()
            # 生成一个随机张量作为梯度，指定数据类型和设备，并转换为指定的内存布局格式
            grad = torch.randn(size, dtype=dtype, device=device)
            grad = grad.contiguous(memory_format=memory_format)
            # 如果数据类型是 torch.bfloat16 并且 is_mixed 为 True，则使用 float 类型的 GroupNorm
            if dtype == torch.bfloat16 and is_mixed:
                gn = nn.GroupNorm(groups, channels).to(device).to(torch.float)
            else:
                # 否则使用指定的数据类型创建 GroupNorm
                gn = nn.GroupNorm(groups, channels).to(device).to(dtype)
            # 随机初始化权重和偏置
            gn.weight.data.uniform_()
            gn.bias.data.uniform_()
    
            # 创建参考输入，从输入中分离数据并克隆，保持连续性，指定内存布局格式，并开启梯度追踪
            ref_input = input.detach().clone().contiguous(memory_format=torch.contiguous_format).requires_grad_(True)
            # 从梯度中分离数据并克隆，保持连续性，指定内存布局格式
            ref_grad = grad.detach().clone().contiguous(memory_format=torch.contiguous_format)
            # 如果数据类型是 torch.bfloat16 并且 is_mixed 为 True，则使用 float 类型的 GroupNorm
            if dtype == torch.bfloat16 and is_mixed:
                ref_gn = nn.GroupNorm(groups, channels).to(device).to(torch.float)
            else:
                # 否则使用指定的数据类型创建 GroupNorm
                ref_gn = nn.GroupNorm(groups, channels).to(device).to(dtype)
            # 加载当前 GroupNorm 的状态字典到参考 GroupNorm
            ref_gn.load_state_dict(gn.state_dict())
            # 运行当前 GroupNorm 对输入进行正向传播
            out = gn(input)
            # 对输出进行反向传播，使用当前梯度
            out.backward(grad)
            # 运行参考 GroupNorm 对参考输入进行正向传播
            ref_out = ref_gn(ref_input)
            # 对参考输出进行反向传播，使用参考梯度
            ref_out.backward(ref_grad)
    
            # 断言当前输出在指定的内存布局格式下是连续的
            self.assertTrue(out.is_contiguous(memory_format=memory_format))
            # 断言参考输出在默认的内存布局格式下是连续的
            self.assertTrue(ref_out.is_contiguous(memory_format=torch.contiguous_format))
            # 断言当前输出与参考输出相等
            self.assertEqual(out, ref_out)
            # 参数在 bfloat16/Half 类型下使用不推荐
            atol = 5e-4
            rtol = 8e-3
            # 断言当前 GroupNorm 的权重梯度与参考 GroupNorm 的权重梯度相等
            self.assertEqual(gn.weight.grad, ref_gn.weight.grad, atol=atol, rtol=rtol)
            # 断言当前 GroupNorm 的偏置梯度与参考 GroupNorm 的偏置梯度相等
            self.assertEqual(gn.bias.grad, ref_gn.bias.grad, atol=atol, rtol=rtol)
            # 断言当前输入的梯度与参考输入的梯度相等
            self.assertEqual(input.grad, ref_input.grad, atol=atol, rtol=rtol)
    
        # 遍历混合类型的列表进行测试
        for is_mixed in [True, False]:
            # 使用辅助函数测试不同大小和参数的情况
            helper(self, (4, 8, 10, 10), 4, torch.channels_last, is_mixed)
            helper(self, (2, 30, 9, 9), 3, torch.channels_last, is_mixed)
            helper(self, (4, 8, 40, 40), 4, torch.channels_last, is_mixed)
            helper(self, (4, 40, 40, 40), 2, torch.channels_last, is_mixed)
            helper(self, (2, 30, 50, 50), 3, torch.channels_last, is_mixed)
            helper(self, (2, 60, 50, 50), 3, torch.channels_last, is_mixed)
            helper(self, (2, 9, 7, 11, 15), 3, torch.channels_last_3d, is_mixed)
            helper(self, (2, 9, 7, 200, 15), 3, torch.channels_last_3d, is_mixed)
            helper(self, (2, 60, 7, 200, 15), 3, torch.channels_last_3d, is_mixed)
    def test_GroupNorm_memory_format(self, device):
        # Tests for regression reported in https://github.com/pytorch/pytorch/issues/92166

        def helper(input_format, grad_format, B=2, C=4, W=4, H=4):
            # 导入深拷贝工具模块
            import copy
            # 创建原始的 GroupNorm 模型并移动到指定设备
            net_orig = torch.nn.GroupNorm(B, C).to(device=device)
            # 深拷贝原始模型以备后续比较
            net = copy.deepcopy(net_orig)
            # 生成随机输入张量并设置 requires_grad=True
            x_orig = torch.rand(B, C, W, H, device=device, requires_grad=True)
            # 生成随机梯度张量
            grad_orig = torch.rand(B, C, W, H, device=device)
            # 根据指定的内存格式创建输入张量副本，并设置 requires_grad=True
            x = x_orig.clone().detach().to(memory_format=input_format).requires_grad_(True)
            # 根据指定的梯度格式剥离梯度张量
            grad = grad_orig.detach().to(memory_format=grad_format)

            # 计算模型输出及其反向传播
            y = net(x)
            y.backward(grad)

            # 计算原始模型输出及其反向传播
            y_orig = net_orig(x_orig)
            y_orig.backward(grad_orig)

            # 断言两次计算结果一致
            self.assertEqual(y, y_orig)
            # 断言输入张量的梯度与原始输入张量的梯度一致
            self.assertEqual(x.grad, x_orig.grad)

        # 遍历不同的输入格式和梯度格式进行测试
        for input_format in [torch.contiguous_format, torch.channels_last]:
            for grad_format in [torch.contiguous_format, torch.channels_last]:
                helper(input_format, grad_format)

    @onlyNativeDeviceTypes
    def test_GroupNorm_numeric(self, device):
        def group_norm_ref(X, gamma, beta, groups, channels, eps):
            # 计算输入张量的维度信息
            batch_size = X.size()[0]
            # 将输入张量重塑为指定的分组形式
            X_view = X.view(batch_size, groups, -1)
            # 计算分组内的均值
            mean = X_view.mean(dim=-1, keepdim=True)
            # 计算分组内的方差
            var = X_view.var(dim=-1, unbiased=False, keepdim=True)
            # 应用 GroupNorm 公式并将结果重塑为原始形状
            Y = ((X_view - mean) / torch.sqrt(var + eps)).view(
                batch_size, channels, -1)
            # 应用 gamma 和 beta 进行归一化和偏移
            Y = Y * gamma.view(channels, 1) + beta.view(channels, 1)
            # 将结果重塑为与输入张量相同的形状并返回
            return Y.view(*X.size())

        # 设置批处理大小、分组数和通道数
        batch_size = 1
        groups = 2
        channels = 8
        # 创建 GroupNorm 模型并转换为指定设备
        group_norm = nn.GroupNorm(groups, channels).float().to(device)
        # 生成随机输入张量并设置数据类型和设备
        X = torch.rand(batch_size, channels, 256, 256, 72,
                       dtype=torch.float32, device=device)

        # 计算 GroupNorm 模型的输出
        Y = group_norm(X)
        # 计算参考输出并与 GroupNorm 模型输出进行比较
        Y_ref = group_norm_ref(
            X, group_norm.weight.data, group_norm.bias.data, groups,
            channels, group_norm.eps)
        # 断言两者在指定的容差范围内相等
        self.assertEqual(Y, Y_ref, rtol=0, atol=1e-5)

        # 如果设备类型是 'cuda'，则在 CPU 上重新计算并与 GPU 结果比较
        if self.device_type == 'cuda':
            group_norm.cpu()
            Y_cpu = group_norm(X.cpu())
            # 断言 CPU 计算结果与 GPU 计算结果一致
            self.assertEqual(Y_cpu, Y, rtol=0, atol=1e-5)

    @onlyNativeDeviceTypes
    @dtypes(torch.float64, torch.complex128)
    # 定义一个测试函数，用于测试填充操作
    def test_pad(self, device, dtype):
        # 断言对于无效的循环填充值会引发断言错误
        inputs = torch.randn(1, 1, 4, device=device, dtype=dtype, requires_grad=True)
        
        # 当尝试多次环绕时应该引发错误
        self.assertRaises(RuntimeError, lambda: F.pad(inputs, (5, 4), mode='circular'))
        self.assertRaises(RuntimeError, lambda: F.pad(inputs, (3, 6), mode='circular'))
        
        # 当负填充导致输出形状为负数时应该引发错误
        self.assertRaises(RuntimeError, lambda: F.pad(inputs, (-3, -2), mode='circular'))

        # 断言反射填充在填充量 >= 输入大小时会引发错误
        expected_err_msg = r"Padding size should be less than the corresponding input dimension"
        inputs = torch.randn(1, 1, 2, 3, device=device, dtype=dtype)
        self.assertRaisesRegex(RuntimeError, expected_err_msg,
                               lambda: F.pad(inputs, (1, 1, 3, 0), mode='reflect'))
        inputs = torch.randn(1, 1, 2, device=device, dtype=dtype)
        self.assertRaisesRegex(RuntimeError, expected_err_msg,
                               lambda: F.pad(inputs, (2, 1), mode='reflect'))

        inputs = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        
        # 断言填充不会返回输入张量的视图
        for mode in 'constant', 'reflect', 'replicate', 'circular':
            out = F.pad(inputs, (0, 0, 0, 0), mode=mode)
            out.fill_(4)
            self.assertTrue(torch.all(torch.abs(inputs) < 2))

            out = F.pad(inputs, (0, 0, -1, -1), mode=mode)
            out.fill_(4)
            self.assertTrue(torch.all(torch.abs(inputs) < 2))

    @onlyNativeDeviceTypes
    @dtypes(torch.float64, torch.complex128)
    # 测试空输入情况下的 ReplicationPad1d、ReplicationPad2d 和 ReplicationPad3d 模块
    def test_ReplicationPad_empty(self, device, dtype):
        # 对于每个模块和输入组合，执行空输入的测试
        for mod, inp in [
                (torch.nn.ReplicationPad1d(3), torch.randn(0, 3, 10, device=device, dtype=dtype)),
                (torch.nn.ReplicationPad2d(3), torch.randn(0, 3, 10, 10, device=device, dtype=dtype)),
                (torch.nn.ReplicationPad3d(3), torch.randn(0, 3, 10, 10, 10, device=device, dtype=dtype))]:
            _test_module_empty_input(self, mod, inp, check_size=False)

        # 检查期望引发运行时错误的情况：ReplicationPad1d 期望输入为 2D 或 3D
        with self.assertRaisesRegex(RuntimeError, 'Expected 2D or 3D'):
            mod = torch.nn.ReplicationPad1d(2)
            inp = torch.randn(3, 0, 10, device=device, dtype=dtype)
            mod(inp)

        # 检查期望引发运行时错误的情况：ReplicationPad2d 期望输入为 3D 或 4D
        with self.assertRaisesRegex(RuntimeError, 'Expected 3D or 4D'):
            mod = torch.nn.ReplicationPad2d((2, 2, 2, 2))
            inp = torch.randn(43, 0, 10, 10, device=device, dtype=dtype)
            mod(inp)

        # 检查期望引发运行时错误的情况：ReplicationPad3d 期望输入为 4D 或 5D
        with self.assertRaisesRegex(RuntimeError, 'Expected 4D or 5D'):
            mod = torch.nn.ReplicationPad3d((2, 2, 2, 2, 2, 2))
            inp = torch.randn(3, 0, 10, 10, 10, device=device, dtype=dtype)
            mod(inp)

        # 检查期望引发运行时错误的情况：torch._C._nn.replication_pad1d 要求 padding 的长度为 2
        with self.assertRaisesRegex(RuntimeError, 'padding size is expected to be 2'):
            torch._C._nn.replication_pad1d(torch.randn([2]), padding=[])

        # 检查期望引发运行时错误的情况：torch._C._nn.replication_pad2d 要求 padding 的长度为 4
        with self.assertRaisesRegex(RuntimeError, 'padding size is expected to be 4'):
            torch._C._nn.replication_pad2d(torch.randn([2]), padding=[])

        # 检查期望引发运行时错误的情况：torch._C._nn.replication_pad3d 要求 padding 的长度为 6
        with self.assertRaisesRegex(RuntimeError, 'padding size is expected to be 6'):
            torch._C._nn.replication_pad3d(torch.randn([2]), padding=[])


    # 测试 ReplicationPad1d 在大尺寸输入下的功能
    def test_ReplicationPad1d_large(self, device):
        # 定义不同形状的输入
        shapes = ([2, 65736, 4], [65736, 2, 4])
        pl, pr = 3, 4
        for shape in shapes:
            # 创建随机张量，设备为指定设备，需要梯度
            x = torch.randn(shape, device=device, requires_grad=True)
            # 创建 ReplicationPad1d 模块，设置左右填充大小
            model = torch.nn.ReplicationPad1d((pl, pr))

            # 前向传播
            out = model(x)
            # 断言输出张量去掉填充部分与输入张量相等
            self.assertEqual(out[:, :, pl : -pr], x)

            # 检查左填充部分是否与输入的第一个元素扩展后的结果相等
            left_padding = out[:, :, : pl]
            self.assertEqual(left_padding, x[:, :, :1].expand_as(left_padding))
            # 检查右填充部分是否与输入的最后一个元素扩展后的结果相等
            right_padding = out[:, :, -pr :]
            self.assertEqual(right_padding, x[:, :, -1:].expand_as(right_padding))

            # 反向传播
            g = torch.randn_like(out)
            out.backward(g)
            # 检查梯度的正确性，确保去掉填充后的梯度与输入张量对应部分的梯度相等
            self.assertEqual(x.grad[:, :, 1 : -1], g[:, :, pl + 1 : -pr - 1])

            # 检查第一个元素的梯度是否等于填充部分的梯度之和
            self.assertEqual(x.grad[:, :, 0], g[:, :, : pl + 1].sum(-1))
            # 检查最后一个元素的梯度是否等于填充部分的梯度之和
            self.assertEqual(x.grad[:, :, -1], g[:, :, -pr - 1:].sum(-1))
    # 定义名为 test_ReplicationPad2d_large 的测试方法，接受参数 device
    def test_ReplicationPad2d_large(self, device):
        # 定义不同形状的张量作为测试数据
        shapes = ([2, 65736, 4, 4], [65736, 2, 4, 4])
        # 设置左、右、上、下各自的填充大小
        pl, pr, pt, pb = 3, 4, 5, 6
        # 遍历不同形状的测试数据
        for shape in shapes:
            # 生成随机数据张量 x，并将其移到指定的设备上，并设置需要计算梯度
            x = torch.randn(shape, device=device, requires_grad=True)
            # 创建 ReplicationPad2d 模型对象，使用指定的填充参数
            model = torch.nn.ReplicationPad2d((pl, pr, pt, pb))

            # 前向传播测试：中心和边缘
            out = model(x)
            # 断言中心区域的输出与原始输入 x 相等
            self.assertEqual(out[:, :, pt : -pb, pl : -pr], x)

            # 断言左侧填充区域的输出与 x 的左侧列的扩展相等
            left_padding = out[:, :, pt : -pb, : pl]
            self.assertEqual(left_padding, x[:, :, :, :1].expand_as(left_padding))
            # 断言右侧填充区域的输出与 x 的右侧列的扩展相等
            right_padding = out[:, :, pt : -pb, -pr :]
            self.assertEqual(right_padding, x[:, :, :, -1:].expand_as(right_padding))
            # 断言顶部填充区域的输出与 x 的顶部行的扩展相等
            top_padding = out[:, :, : pt, pl : -pr]
            self.assertEqual(top_padding, x[:, :, :1, :].expand_as(top_padding))
            # 断言底部填充区域的输出与 x 的底部行的扩展相等
            bottom_padding = out[:, :, -pb : , pl : -pr]
            self.assertEqual(bottom_padding, x[:, :, -1:, :].expand_as(bottom_padding))

            # 前向传播测试：角落区域
            tl_padding = out[:, :, : pt + 1, : pl + 1]
            self.assertEqual(tl_padding, x[:, :, :1, :1].expand_as(tl_padding))
            tr_padding = out[:, :, : pt + 1, -pr - 1:]
            self.assertEqual(tr_padding, x[:, :, :1, -1:].expand_as(tr_padding))
            bl_padding = out[:, :, -pb - 1:, : pl + 1]
            self.assertEqual(bl_padding, x[:, :, -1:, :1].expand_as(bl_padding))
            br_padding = out[:, :, -pb - 1:, -pr - 1:]
            self.assertEqual(br_padding, x[:, :, -1:, -1:].expand_as(br_padding))

            # 反向传播测试：中心和边缘
            g = torch.randn_like(out)
            out.backward(g)
            # 断言 x 的梯度在中心区域的计算结果与 g 的对应区域相等
            self.assertEqual(x.grad[:, :, 1:-1, 1:-1], g[:, :, pt + 1 : -pb - 1, pl + 1 : -pr - 1])

            # 断言 x 的梯度在左侧边缘区域的计算结果与 g 的对应区域相等
            self.assertEqual(x.grad[:, :, 1:-1, 0], g[:, :, pt + 1 : -pb - 1, : pl + 1].sum(-1))
            # 断言 x 的梯度在右侧边缘区域的计算结果与 g 的对应区域相等
            self.assertEqual(x.grad[:, :, 1:-1, -1], g[:, :, pt + 1 : -pb - 1, -pr - 1 :].sum(-1))
            # 断言 x 的梯度在顶部边缘区域的计算结果与 g 的对应区域相等
            self.assertEqual(x.grad[:, :, 0, 1:-1], g[:, :, : pt + 1, pl + 1 : -pr - 1].sum(-2))
            # 断言 x 的梯度在底部边缘区域的计算结果与 g 的对应区域相等
            self.assertEqual(x.grad[:, :, -1, 1:-1], g[:, :, -pb - 1 :, pl + 1 : -pr - 1].sum(-2))

            # 反向传播测试：角落区域
            # 断言 x 的梯度在左上角区域的计算结果与 g 的对应区域相等
            self.assertEqual(x.grad[:, :, 0, 0], g[:, :, : pt + 1, : pl + 1].sum((-2, -1)))
            # 断言 x 的梯度在右上角区域的计算结果与 g 的对应区域相等
            self.assertEqual(x.grad[:, :, 0, -1], g[:, :, : pt + 1, -pr - 1 :].sum((-2, -1)))
            # 断言 x 的梯度在左下角区域的计算结果与 g 的对应区域相等
            self.assertEqual(x.grad[:, :, -1, 0], g[:, :, -pb - 1 :, : pl + 1].sum((-2, -1)))
            # 断言 x 的梯度在右下角区域的计算结果与 g 的对应区域相等
            self.assertEqual(x.grad[:, :, -1, -1], g[:, :, -pb - 1 :, -pr - 1 :].sum((-2, -1)))

    @largeTensorTest("6GB")
    # 定义测试函数，测试 ReplicationPad3d 在大尺寸输入上的行为，使用指定的设备
    def test_ReplicationPad3d_large(self, device):
        # 定义两种不同形状的输入张量
        shapes = ([1, 65736, 2, 2, 2], [65736, 1, 2, 2, 2])
        # 定义六个填充参数
        pl, pr, pt, pbt, pf, pbk = 3, 4, 5, 6, 7, 8

        # 对每种形状进行循环测试
        for shape in shapes:
            # 创建随机张量 x，使用指定设备，并开启梯度追踪
            x = torch.randn(shape, device=device, requires_grad=True)
            # 创建 ReplicationPad3d 模型，使用定义好的填充参数
            model = torch.nn.ReplicationPad3d((pl, pr, pt, pbt, pf, pbk))

            # 对模型进行前向传播计算
            out = model(x)
            # 断言模型输出与输入在指定边界内的区域相等
            self.assertEqual(out[:, :, pf : -pbk, pt : -pbt, pl : -pr], x)

            # 对模型进行反向传播计算中心区域
            g = torch.randn_like(out)
            out.backward(g)
            # 断言输入的梯度与输出梯度在中心区域内相等
            self.assertEqual(x.grad[:, :, 1:-1, 1:-1, 1:-1], g[:, :, pf + 1 : -pbk - 1, pt + 1 : -pbt - 1, pl + 1 : -pr - 1])

    # 标记为只在原生设备类型上运行的测试函数
    @onlyNativeDeviceTypes
    def test_Bilinear_empty(self, device):
        # 创建 Bilinear 模型，输入大小为 (20, 30)，并将其转移到指定设备上
        mod = torch.nn.Bilinear(20, 30, 40).to(device)
        # 创建大小为 (0, 10, 20) 的随机输入张量，开启梯度追踪，将其移到指定设备上
        inp1 = torch.randn(0, 10, 20, requires_grad=True, device=device)
        # 创建大小为 (0, 10, 30) 的随机输入张量，开启梯度追踪，将其移到指定设备上
        inp2 = torch.randn(0, 10, 30, requires_grad=True, device=device)

        # 对模型进行前向传播计算
        output = mod(inp1, inp2)
        # 对输出进行求和并进行反向传播
        output.sum().backward()

        # 断言 inp1 和 inp2 的值为零张量
        self.assertEqual(inp1, torch.zeros_like(inp1))
        self.assertEqual(inp2, torch.zeros_like(inp2))

        # 断言 inp1 和 inp2 的梯度为零张量
        self.assertEqual(inp1.grad, torch.zeros_like(inp1))
        self.assertEqual(inp2.grad, torch.zeros_like(inp2))

    # 标记为预期失败的元数据，因为可能会抛出 RuntimeError
    @expectedFailureMeta  # RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1]
    @onlyNativeDeviceTypes
    # 定义测试方法，用于测试 TransformerEncoderLayer 在空输入情况下的行为
    def test_TransformerEncoderLayer_empty(self, device):
        # 遍历训练和非训练模式
        for training in (True, False):
            # 遍历不同的 batch_first 和输入形状组合
            for batch_first, input_shape in [(True, (0, 10, 512)),
                                             (False, (10, 0, 512))]:
                # 创建指定形状的随机输入张量
                input = torch.rand(*input_shape, device=device, dtype=torch.double)
                # 创建 TransformerEncoderLayer 实例，指定参数并移动到指定设备
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=512, nhead=8, batch_first=batch_first, dtype=torch.double).to(device)
                
                # 如果不是训练模式，切换为评估模式并执行无梯度操作
                if not training:
                    encoder_layer = encoder_layer.eval()
                    with torch.no_grad():
                        # 调用测试函数，测试空输入情况下模块的行为，忽略尺寸检查，推理模式为真
                        _test_module_empty_input(self, encoder_layer, input, check_size=False, inference=True)
                    
                    # 对于 batch_first 为真且未使用跨参考测试，使用断言检查异常情况
                    if batch_first and not TEST_WITH_CROSSREF:
                        with torch.no_grad():
                            # 测试 MultiheadAttention 是否支持空 NestedTensor
                            with self.assertRaisesRegex(
                                    AssertionError, 'MultiheadAttention does not support NestedTensor outside'):
                                nt = torch.nested.nested_tensor([], device=device)
                                _test_module_empty_input(self, encoder_layer, nt, check_size=False, inference=True)
                            
                            # 创建包含一个空张量的 NestedTensor，测试其行为
                            nt = torch.nested.nested_tensor([torch.rand(0, 512, device=device, dtype=torch.double)], device=device)
                            _test_module_empty_input(self, encoder_layer, nt, check_size=False, inference=True)
                else:
                    # 如果是训练模式，直接调用测试函数测试空输入情况下模块的行为，忽略尺寸检查
                    _test_module_empty_input(self, encoder_layer, input, check_size=False)

    # 标记为预期失败的测试用例装饰器，因为会抛出 RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1]
    @expectedFailureMeta
    # 标记仅适用于原生设备类型的测试用例装饰器
    @onlyNativeDeviceTypes
    # 定义测试方法，用于测试 TransformerEncoder 在空输入情况下的行为
    def test_TransformerEncoder_empty(self, device):
        # 遍历不同的 batch_first 和输入形状组合
        for batch_first, input_shape in [(True, (0, 10, 512)),
                                         (False, (10, 0, 512))]:
            # 创建指定形状的随机输入张量
            input = torch.rand(*input_shape, device=device, dtype=torch.double)
            # 创建 TransformerEncoderLayer 实例，指定参数并移动到指定设备
            encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=batch_first, dtype=torch.double).to(device)
            # 创建 TransformerEncoder 实例，指定参数并移动到指定设备
            transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6).to(device)
            # 调用测试函数，测试空输入情况下模块的行为，忽略尺寸检查
            _test_module_empty_input(self, transformer_encoder, input, check_size=False)
    # 定义测试方法，测试TransformerDecoderLayer类的空输入情况
    def test_TransformerDecoderLayer_empty(self, device):
        # 针对不同的参数组合进行循环测试
        for batch_first, memory_shape, tgt_shape in [(True, (0, 10, 512), (0, 20, 512)),
                                                     (False, (10, 0, 512), (20, 0, 512))]:
            # 创建指定形状的随机数据张量，设备为指定设备，数据类型为双精度浮点型
            memory = torch.rand(*memory_shape, device=device, dtype=torch.double)
            # 创建指定形状的随机数据张量，设置梯度计算，设备为指定设备，数据类型为双精度浮点型
            tgt = torch.rand(*tgt_shape, requires_grad=True, device=device, dtype=torch.double)
            # 创建一个TransformerDecoderLayer实例，指定模型参数和设备
            decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=batch_first, dtype=torch.double).to(device)
            # 调用测试方法，验证模块处理空输入的情况
            self._test_module_empty_inputs(decoder_layer, [tgt, memory])

    # 标记为预期失败的测试用例元信息，测试TransformerDecoder类的空输入情况
    @expectedFailureMeta  # RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1]
    # 仅在本地设备类型下执行测试用例
    @onlyNativeDeviceTypes
    def test_TransformerDecoder_empty(self, device):
        # 针对不同的参数组合进行循环测试
        for batch_first, memory_shape, tgt_shape in [(True, (0, 10, 512), (0, 20, 512)),
                                                     (False, (10, 0, 512), (20, 0, 512))]:
            # 创建指定形状的随机数据张量，设备为指定设备，数据类型为双精度浮点型
            memory = torch.rand(*memory_shape, device=device, dtype=torch.double)
            # 创建指定形状的随机数据张量，设置梯度计算，设备为指定设备，数据类型为双精度浮点型
            tgt = torch.rand(*tgt_shape, requires_grad=True, device=device, dtype=torch.double)
            # 创建一个TransformerDecoderLayer实例，指定模型参数和设备
            decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=batch_first, dtype=torch.double).to(device)
            # 创建一个TransformerDecoder实例，包含指定的解码层和层数
            transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6).to(device)
            # 调用测试方法，验证模块处理空输入的情况
            self._test_module_empty_inputs(transformer_decoder, [tgt, memory])

    # 标记为预期失败的测试用例元信息，测试Transformer类的空输入情况
    @expectedFailureMeta  # RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1]
    # 仅在本地设备类型下执行测试用例
    @onlyNativeDeviceTypes
    def test_Transformer_empty(self, device):
        # 针对不同的参数组合进行循环测试
        for batch_first, src_shape, tgt_shape in [(True, (10, 0, 512), (20, 0, 512))]:
            # 创建一个Transformer实例，指定模型参数和设备
            transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, dtype=torch.double).to(device)
            # 创建指定形状的随机数据张量，设置梯度计算，设备为指定设备，数据类型为双精度浮点型
            src = torch.rand(*src_shape, requires_grad=True, device=device, dtype=torch.double)
            # 创建指定形状的随机数据张量，设置梯度计算，设备为指定设备，数据类型为双精度浮点型
            tgt = torch.rand(*tgt_shape, requires_grad=True, device=device, dtype=torch.double)
            # 调用测试方法，验证模块处理空输入的情况
            self._test_module_empty_inputs(transformer_model, [src, tgt])

    # 仅在本地设备类型下执行测试用例，同时设置数据类型为浮点型和复数型
    @onlyNativeDeviceTypes
    @dtypes(torch.float32, torch.complex64)
    # 测试 ReflectionPad1d、ReflectionPad2d 和 ReflectionPad3d 对空输入的行为
    def test_ReflectionPad_empty(self, device, dtype):
        # 针对每个模块和输入进行迭代测试
        for mod, inp in [
                (torch.nn.ReflectionPad1d(2), torch.randn(0, 3, 10, device=device, dtype=dtype)),
                (torch.nn.ReflectionPad2d(2), torch.randn(0, 3, 10, 10, device=device, dtype=dtype)),
                (torch.nn.ReflectionPad3d(3), torch.randn(0, 3, 10, 10, 10, device=device, dtype=dtype))]:
            # 调用辅助函数 _test_module_empty_input 测试空输入的模块行为
            _test_module_empty_input(self, mod, inp, check_size=False)

        # 测试 ReflectionPad1d 在不合法的输入上抛出异常
        with self.assertRaisesRegex(RuntimeError, '2D or 3D'):
            mod = torch.nn.ReflectionPad1d(2)
            inp = torch.randn(3, 0, 10, device=device, dtype=dtype)
            mod(inp)

        # 测试 ReflectionPad2d 在不合法的输入上抛出异常
        with self.assertRaisesRegex(RuntimeError, '3D or 4D'):
            mod = torch.nn.ReflectionPad2d(2)
            inp = torch.randn(3, 0, 10, 10, device=device, dtype=dtype)
            mod(inp)

        # 测试 ReflectionPad3d 在不合法的输入上抛出异常
        with self.assertRaisesRegex(RuntimeError, '4D or 5D'):
            mod = torch.nn.ReflectionPad3d(3)
            inp = torch.randn(3, 0, 10, 10, 10, device=device, dtype=dtype)
            mod(inp)

    @onlyCUDA   # 测试 CPU 和 GPU 的结果是否一致
    def test_ReflectionPad2d_large(self, device):
        # 定义不同形状的输入张量和填充
        shapes = ([2, 65736, 6, 6], [65736, 2, 6, 6])
        pad = (1, 2, 3, 4)
        for shape in shapes:
            # 在给定设备上创建随机张量并设置梯度需求
            x = torch.randn(shape, device=device, requires_grad=True)
            ref_x = x.detach().cpu().requires_grad_()

            # 使用反射模式填充张量
            out = F.pad(x, pad, mode='reflect')
            ref_out = F.pad(ref_x, pad, mode='reflect')

            # 断言填充后的结果张量是否相等
            self.assertEqual(out, ref_out)

            # 创建相同形状的随机梯度张量并移到 CPU 上
            g = torch.randn_like(out)
            ref_g = g.cpu()

            # 反向传播并断言梯度是否相等
            out.backward(g)
            ref_out.backward(ref_g)

            self.assertEqual(x.grad, ref_x.grad)

    @onlyNativeDeviceTypes
    def test_LocalResponseNorm_empty(self, device):
        # 在给定设备上创建 LocalResponseNorm 模块和空输入
        mod = torch.nn.LocalResponseNorm(2).to(device)
        inp = torch.ones(0, 5, 24, 24, device=device)
        # 调用辅助函数 _test_module_empty_input 测试空输入的模块行为
        _test_module_empty_input(self, mod, inp, check_size=False)

    @onlyCUDA   # 测试 CPU 和 GPU 的结果是否一致
    def test_ReflectionPad3d_large(self, device):
        # 定义不同形状的输入张量和填充
        shapes = ([2, 1000, 7, 7, 7], [1000, 2, 7, 7, 7])
        pad = (1, 2, 3, 4, 5, 6)
        for shape in shapes:
            # 在给定设备上创建随机张量并设置梯度需求
            x = torch.randn(shape, device=device, requires_grad=True)
            ref_x = x.detach().cpu().requires_grad_()

            # 使用反射模式填充张量
            out = F.pad(x, pad, mode='reflect')
            ref_out = F.pad(ref_x, pad, mode='reflect')

            # 断言填充后的结果张量是否相等
            self.assertEqual(out, ref_out)

            # 创建相同形状的随机梯度张量并移到 CPU 上
            g = torch.randn_like(out)
            ref_g = g.cpu()

            # 反向传播并断言梯度是否相等
            out.backward(g)
            ref_out.backward(ref_g)

            self.assertEqual(x.grad, ref_x.grad)

    @onlyNativeDeviceTypes
    @dtypes(torch.float, torch.double)
    # 测试空输入情况下 MultiMarginLoss 的行为
    def test_MarginLoss_empty(self, device, dtype):
        for mod, x, y in [
                # 使用 MultiMarginLoss 测试，输入为零行零列的随机张量和空标签张量
                (torch.nn.MultiMarginLoss().to(device),
                 torch.randn(0, 10, requires_grad=True, device=device, dtype=dtype),
                 torch.ones(0, device=device).type(torch.long)),
                # 使用 MultiLabelMarginLoss 测试，输入为零行零列的随机张量和空标签张量
                (torch.nn.MultiLabelMarginLoss().to(device),
                 torch.randn(0, 10, requires_grad=True, device=device, dtype=dtype),
                 torch.ones(0, 10, device=device).type(torch.long))]:

            # 计算损失并进行反向传播
            out = mod(x, y)
            out.sum().backward()

            # 断言：期望 x 的值为全零
            self.assertEqual(x, torch.zeros_like(x))
            # 断言：期望 x 的梯度为全零
            self.assertEqual(x.grad, torch.zeros_like(x))

            # 断言：使用随机张量 x 和空标签 y 时，期望抛出 RuntimeError 异常并提示 'Expected'
            with self.assertRaisesRegex(RuntimeError, 'Expected'):
                x = torch.randn(0, requires_grad=True, device=device, dtype=dtype)
                y = torch.ones(10, device=device).type(torch.long)
                mod(x, y)

            # 断言：使用随机张量 x 和零行零列的标签 y 时，期望抛出 RuntimeError 异常并提示 'Expected'
            with self.assertRaisesRegex(RuntimeError, 'Expected'):
                x = torch.randn(10, 0, requires_grad=True, device=device, dtype=dtype)
                y = torch.ones(10, 0, device=device).type(torch.long)
                mod(x, y)

    # 在 CUDA 设备上测试 MultiMarginLoss 的警告输出情况
    @onlyCUDA
    def test_MarginLoss_warnings(self, device):
        model = torch.nn.Linear(128, 22, device=device)
        loss = torch.nn.MultiMarginLoss()
        x = torch.rand((56, 128), device=device)
        targets = torch.randint(22, (56,), device=device)
        # 用于捕获标准错误输出的流
        f = io.StringIO()
        with contextlib.redirect_stderr(f):
            # 正向传播
            out = model(x)
            # 计算损失
            l = loss(out, targets)
            # 反向传播
            l.backward()
        # 断言：没有警告信息输出
        self.assertTrue(len(f.getvalue()) == 0)

    # 测试输入为空时 Unfold 模块的行为
    @onlyNativeDeviceTypes
    def test_Unfold_empty(self, device):
        # 创建零行三维张量作为输入
        inp = torch.randn(0, 3, 3, 4, device=device)
        # 创建 Unfold 模块，设置卷积核大小为 (2, 3)
        unfold = torch.nn.Unfold(kernel_size=(2, 3)).to(device)
        # 使用通用的测试函数测试空输入情况下 Unfold 模块的行为
        _test_module_empty_input(self, unfold, inp, check_size=False)

        # 断言：使用非法形状的输入时，期望抛出 RuntimeError 异常并提示 'Expected 3D or 4D'
        with self.assertRaisesRegex(RuntimeError, 'Expected 3D or 4D'):
            inp = torch.randn(3, 0, 3, 4, device=device)
            unfold = torch.nn.Unfold(kernel_size=(2, 3)).to(device)
            unfold(inp)

    # 在 CUDA 设备上测试 BatchNorm2d 模块在空输入下的行为
    @onlyCUDA
    @dtypes(torch.float, torch.double)
    @tf32_on_and_off(0.005)
    def test_BatchNorm_empty(self, device):
        # 创建 BatchNorm2d 模块
        mod = torch.nn.BatchNorm2d(3).to(device)
        # 创建零行四维张量作为输入
        inp = torch.randn(0, 3, 2, 2, device=device)
        # 使用通用的测试函数测试空输入情况下 BatchNorm2d 模块的行为
        _test_module_empty_input(self, mod, inp)
        
        # 如果设备类型为 'cuda' 并且支持 cuDNN，则禁用 cuDNN 后再次测试 BatchNorm2d 模块的空输入行为
        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                _test_module_empty_input(self, mod, inp)

        # 断言：检查 BatchNorm2d 模块的运行均值是否为全零
        self.assertEqual(mod.running_mean, torch.tensor([0., 0, 0], device=device))
        # 断言：检查 BatchNorm2d 模块的运行方差是否为全一
        self.assertEqual(mod.running_var, torch.tensor([1., 1, 1], device=device))
        # 断言：检查 BatchNorm2d 模块的权重梯度是否为全零
        self.assertEqual(mod.weight.grad, torch.tensor([0., 0, 0], device=device))
        # 断言：检查 BatchNorm2d 模块的偏置梯度是否为全零
        self.assertEqual(mod.bias.grad, torch.tensor([0., 0, 0], device=device))
    # 测试 PReLU 反向传播函数，使用 32 位索引
    def test_prelu_backward_32bit_indexing(self, device):
        # 创建 PReLU 激活函数模型，移动到 CUDA 并使用半精度
        m = torch.nn.PReLU().cuda().half()
        # 创建输入张量，全为 1，形状为 (1024, 1024, 1024, 2)，使用半精度，放置在指定设备上
        input_ = torch.ones((1024, 1024, 1024, 2), dtype=torch.half, device=device)
        # 将输入张量传递给模型，获取输出
        output = m(input_)
        # 对输出进行反向传播
        output.backward(input_)

    # 测试空输入的线性模型
    def test_linear_empty(self, device):
        # 创建具有输入和输出尺寸为 7 的线性模型，并移动到指定设备
        mod = torch.nn.Linear(7, 7).to(device)
        # 创建空的输入张量，形状为 (0, 7)，放置在指定设备上
        inp = torch.randn(0, 7, device=device)
        # 调用帮助函数来测试模型在空输入上的行为
        _test_module_empty_input(self, mod, inp)

    # 测试 one_hot 函数的不同用例
    def test_one_hot(self, device):
        # 在 CUDA 设备上，如果设备类型不是 'cuda' 或 'xla'，则期望引发 RuntimeError
        if self.device_type != 'cuda' and self.device_type != 'xla':
            with self.assertRaises(RuntimeError):
                torch.nn.functional.one_hot(torch.tensor([3, 4, -1, 0], device=device), -1)

            with self.assertRaises(RuntimeError):
                torch.nn.functional.one_hot(torch.tensor([3, 4, 1, 0], device=device), 3)

        # 测试默认参数下的 one_hot 转换
        t = torch.nn.functional.one_hot(torch.tensor([3, 4, 1, 0], device=device))
        expected = torch.tensor([[0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 1],
                                 [0, 1, 0, 0, 0],
                                 [1, 0, 0, 0, 0]], device=device)
        self.assertEqual(t, expected)

        # 测试使用负值参数的 one_hot 转换
        t = torch.nn.functional.one_hot(torch.tensor([3, 4, 1, 0], device=device), -1)
        self.assertEqual(t, expected)

        # 测试指定深度参数的 one_hot 转换
        t = torch.nn.functional.one_hot(torch.tensor([3, 4, 1, 0], device=device), 6)
        expected = torch.tensor([[0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 1, 0],
                                 [0, 1, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 0]], device=device)
        self.assertEqual(t, expected)

        # 测试处理二维输入的 one_hot 转换
        t = torch.nn.functional.one_hot(torch.tensor([[3, 4], [1, 0]], device=device))
        expected = torch.tensor([[[0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 1]],
                                 [[0, 1, 0, 0, 0],
                                  [1, 0, 0, 0, 0]]], device=device)
        self.assertEqual(t, expected)

        # 测试处理标量输入的 one_hot 转换
        t = torch.nn.functional.one_hot(torch.tensor(4, device=device))
        expected = torch.tensor([0, 0, 0, 0, 1], device=device)
        self.assertEqual(t, expected)

        # 测试处理空张量输入的 one_hot 转换，预期输出为全零张量
        t = torch.nn.functional.one_hot(torch.empty([4, 0], dtype=torch.long, device=device), 100)
        expected = torch.empty([4, 0, 100], dtype=torch.long)
        self.assertEqual(t, expected)

        # 使用空的长整型张量引发 RuntimeError
        with self.assertRaises(RuntimeError):
            torch.nn.functional.one_hot(torch.empty([4, 0], dtype=torch.long, device=device))

        # 使用超出范围的深度值引发 RuntimeError
        with self.assertRaises(RuntimeError):
            torch.nn.functional.one_hot(torch.tensor([3, 4, 1, 0], device=device), -2)
    # 定义一个测试方法，用于测试空输入的情况
    def test_nn_empty(self, device):
        # 用于验证标量是否正确应用于 nn.yaml 中的模块
        def verify_scalars(input, output):
            self.assertEqual(input.shape, output.shape)
            self.assertEqual(0, output.numel())

        # 针对不同的输入形状进行迭代测试
        for input_shape in [(0), (0, 2)]:
            # 遍历不同的模块，如 ELU、Hardtanh 等
            for module in [torch.nn.ELU, torch.nn.Hardtanh, torch.nn.LeakyReLU, torch.nn.LogSigmoid,
                           torch.nn.RReLU, torch.nn.Softshrink, torch.nn.Softplus, torch.nn.Sigmoid,
                           torch.nn.Tanh]:
                # 创建具有指定设备和梯度的随机输入
                input = torch.randn(input_shape, device=device, requires_grad=True)
                # 实例化当前的模块
                m = module()
                # 对输入应用模块，获取输出
                output = m(input)
                # 调用验证函数，检查输出是否符合预期标量特性
                verify_scalars(input, output)

    # 定义一个测试方法，用于测试标量是否正确应用于 nn.yaml 中的模块
    def test_nn_scalars(self, device):
        # 用于验证标量是否正确应用于 nn.yaml 中的模块
        def verify_scalars(input, output):
            if input.dim() == 0:
                self.assertEqual((), output.shape)
            else:
                self.assertNotEqual((), output.shape)
            # 计算输出的和的梯度
            output.sum().backward()
            # 检查输入张量和梯度张量的形状是否相同
            self.assertEqual(input.shape, input.grad.shape)

        # 针对不同的输入形状进行迭代测试
        for input_shape in [(5, 6), ()]:
            # 遍历不同的模块，如 ELU、Hardtanh 等
            for module in [torch.nn.ELU, torch.nn.Hardtanh, torch.nn.LeakyReLU, torch.nn.LogSigmoid,
                           torch.nn.RReLU, torch.nn.Softshrink, torch.nn.Softplus, torch.nn.Sigmoid,
                           torch.nn.Tanh]:
                # 创建具有指定设备和梯度的随机输入
                input = torch.randn(input_shape, device=device, requires_grad=True)
                # 实例化当前的模块
                m = module()
                # 对输入应用模块，获取输出
                output = m(input)
                # 调用验证函数，检查输出是否符合预期标量特性
                verify_scalars(input, output)

    # 定义一个测试方法，用于测试标量在减少过程中的应用情况
    def test_nn_scalars_reductions(self, device):
        # 用于验证标量是否正确应用于 nn.yaml 中的模块
        def verify_reduction_scalars(input, reduction, output):
            if reduction != 'none' or input.dim() == 0:
                self.assertEqual((), output.shape)
            else:
                self.assertNotEqual((), output.shape)
            # 计算输出的和的梯度
            output.sum().backward()
            # 检查输入张量和梯度张量的形状是否相同
            self.assertEqual(input.shape, input.grad.shape)

        # 针对不同的输入形状进行迭代测试
        for input_shape in [(5, 6), ()]:
            # 针对不同的减少方式进行迭代测试，如 'none'、'mean'、'sum'
            for reduction in ['none', 'mean', 'sum']:
                # 遍历不同的损失函数模块，如 BCELoss、L1Loss 等
                for module in [torch.nn.BCELoss, torch.nn.L1Loss, torch.nn.MSELoss,
                               torch.nn.SmoothL1Loss, torch.nn.SoftMarginLoss]:
                    # 创建具有指定设备和梯度的随机输入
                    input = torch.randn(input_shape, device=device, requires_grad=True)
                    # 创建随机目标数据，并应用 sigmoid 函数
                    target = torch.empty(input_shape, device=device).random_(2)
                    sigmoid = nn.Sigmoid()

                    input = torch.randn(input_shape, device=device, requires_grad=True)
                    # 实例化当前的损失函数模块，设置减少方式
                    m = module(reduction=reduction)
                    # 对输入应用模块，获取输出
                    output = m(sigmoid(input), target)
                    # 调用验证函数，检查输出是否符合预期标量特性
                    verify_reduction_scalars(input, reduction, output)
    # 定义一个测试函数，用于测试无效的减少（reduction）字符串
    def test_invalid_reduction_strings(self, device):
        # 创建一个大小为 (3, 5) 的张量，要求梯度计算，指定设备
        input = torch.randn(3, 5, requires_grad=True, device=device)
        # 创建一个大小为 (3, 5) 的复数张量，要求梯度计算，指定设备和数据类型为复数
        cinput = torch.randn(3, 5, requires_grad=True, device=device, dtype=torch.cfloat)
        # 创建一个在指定设备上的张量，内容为 [1, 0, 4]
        target = torch.tensor([1, 0, 4], device=device)
        # 创建一个与 input 大小相同的张量，要求梯度计算，指定设备
        var = torch.ones(size=input.size(), requires_grad=True, device=device)

        # 遍历每个指定的减少方式（'none' 和 'invalid'）
        for reduction in ['none', 'invalid']:
            # 定义一个辅助函数 v，接受一个函数作为参数 fn
            def v(fn):
                # 如果 reduction 为 'invalid'，则断言该函数抛出 ValueError 异常
                if reduction == 'invalid':
                    self.assertRaises(ValueError, lambda: fn())
                else:
                    # 否则直接执行该函数
                    fn()

            # 对一系列损失函数调用 v 函数进行测试
            v(lambda: F.nll_loss(input, target, reduction=reduction))
            v(lambda: F.cross_entropy(input, target, reduction=reduction))

            v(lambda: F.kl_div(input, input, reduction=reduction))
            v(lambda: F.huber_loss(input, input, reduction=reduction))
            v(lambda: F.smooth_l1_loss(input, input, reduction=reduction))
            v(lambda: F.l1_loss(input, input, reduction=reduction))
            v(lambda: F.l1_loss(cinput, cinput, reduction=reduction))
            v(lambda: F.mse_loss(input, input, reduction=reduction))
            v(lambda: F.hinge_embedding_loss(input, input, reduction=reduction))
            v(lambda: F.poisson_nll_loss(input, input, reduction=reduction))
            v(lambda: F.gaussian_nll_loss(input, input, var, reduction=reduction))
            v(lambda: F.binary_cross_entropy(torch.sigmoid(input), input.gt(0).to(torch.get_default_dtype()), reduction=reduction))
            v(lambda: F.binary_cross_entropy_with_logits(input, input, reduction=reduction))

            zeros = torch.zeros_like(input).to(torch.int64)
            v(lambda: F.multilabel_soft_margin_loss(input, zeros, reduction=reduction))

            v(lambda: F.triplet_margin_loss(input, input, input, reduction=reduction))
            v(lambda: F.triplet_margin_with_distance_loss(input, input, input, reduction=reduction))
            v(lambda: F.margin_ranking_loss(input, input, input.sign(), reduction=reduction))
            v(lambda: F.cosine_embedding_loss(input, input, input[:, 0].sign(), reduction=reduction))

            log_probs = torch.randn(50, 16, 20, requires_grad=True, device=device).log_softmax(2)
            targets = torch.randint(1, 20, (16, 30), dtype=torch.long, device=device)
            input_lengths = torch.full((16,), 50, dtype=torch.long, device=device)
            target_lengths = torch.randint(10, 30, (16,), dtype=torch.long, device=device)
            v(lambda: F.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction=reduction))

            # FIXME: should we allow derivatives on these?
            v(lambda: F.soft_margin_loss(input, input.sign().detach(), reduction=reduction))
    # 定义一个测试方法，用于测试 Smooth L1 Loss 在 bfloat16 数据类型下的表现
    def test_smooth_l1_loss_bfloat16(self, device):
        
        # 定义一个内部辅助函数，用于测试不同数据类型下的函数行为
        def test_dtype(fn, input, target, dtype):
            # 将输入数据转换为指定的数据类型，并保留梯度信息
            input = input.detach().clone().to(dtype=dtype).requires_grad_(True)
            # 创建一个浮点数版本的输入数据，同样保留梯度信息
            input2 = input.detach().clone().float().requires_grad_(True)
            # 将目标数据转换为指定的数据类型
            target = target.detach().clone().to(dtype=dtype)
            # 创建一个浮点数版本的目标数据
            target2 = target.detach().clone().float()
            # 使用给定的函数计算输入和目标的损失
            out = fn(input, target)
            # 对损失值进行求和并反向传播
            out.sum().backward()
            # 使用浮点数版本的输入和目标再次计算损失值
            out2 = fn(input2, target2)
            # 对其损失值进行求和并反向传播
            out2.sum().backward()
            # 断言输出的数据类型与预期的数据类型相符
            self.assertEqual(out.dtype, dtype)
            # 断言输入的梯度数据类型与预期的数据类型相符
            self.assertEqual(input.grad.dtype, dtype)
            # 断言两次计算的损失值应该相等（允许小数点精度的差异）
            self.assertEqual(out, out2, exact_dtype=False)
            # 断言两次计算的输入梯度应该相等（允许小数点精度的差异）
            self.assertEqual(input.grad, input2.grad, exact_dtype=False)

        # 定义一个函数，返回一个 Smooth L1 Loss 对象，设定在指定的设备上
        def func(device):
            return nn.SmoothL1Loss().to(device=device)

        # 定义不同形状的输入数据
        shapes = [[1, 3, 1, 6], [1, 3, 1, 128], [1, 3, 128, 128]]
        # 遍历不同的形状
        for shape in shapes:
            # 生成随机输入数据，并设定其需要计算梯度
            x = torch.randn(shape, device=device, requires_grad=True)
            # 生成随机目标数据
            t = torch.randn(shape, device=device)
            # 对指定形状的数据进行类型测试
            test_dtype(func(device), x, t, torch.bfloat16)

    # 我们不希望将 NaN 的传播作为操作的硬要求，但是对于这些简单的操作，我们应该这样做。
    # 定义一个方法，用于测试非线性函数在给定设备上是否能正确传播 NaN 值
    def test_nonlinearity_propagate_nan(self, device):
        
        # 定义一个内部测试函数，用于测试指定非线性函数在处理 NaN 值时的行为
        def test(nonlinearity, *args, **kwargs):
            # 创建一个包含 NaN 值的张量，并放置在指定设备上
            x = torch.tensor([nan], device=device)
            # 获取对应的非线性函数
            fn = getattr(F, nonlinearity)
            try:
                # 断言使用该函数处理包含 NaN 值的张量时，结果应该也是 NaN
                self.assertTrue(math.isnan(fn(x, *args, **kwargs).item()))
            except Exception as e:
                # 如果出现未实现的异常，则跳过该测试
                if 'not implemented' not in str(e):
                    raise

        # 对一系列非线性函数进行 NaN 传播测试
        test('relu')
        test('relu', inplace=True)
        test('relu6')
        test('elu')
        test('selu')
        test('celu')
        test('rrelu')
        test('rrelu', inplace=True)
        test('hardtanh')
        test('tanh')
        test('sigmoid')
        test('logsigmoid')
        test('hardshrink')
        test('tanhshrink')
        test('softsign')
        test('softmin', 0)
        test('softmax', 0)
        test('log_softmax', 0)
        test('leaky_relu', 0.2)
        test('threshold', 3, 2)
        test('threshold', 3, 2, inplace=True)

    # 对于给定的模式进行参数化测试，模式包括 "nearest-exact" 和 "nearest"
    # 定义测试函数 test_upsamplingNearest1d，接受设备和模式作为参数
    def test_upsamplingNearest1d(self, device, mode):
        # 检查是否支持自动求导，排除 XLA 设备因为 XLA 张量没有存储
        check_forward_ad = torch.device(device).type != 'xla'

        # 创建 Upsample 模块对象 m，设置目标大小为 4，使用指定的插值模式
        m = nn.Upsample(size=4, mode=mode)

        # 创建输入张量 in_t，大小为 1x1x2，设备为指定的 device，数据类型为双精度浮点型
        in_t = torch.ones(1, 1, 2, device=device, dtype=torch.double)

        # 创建输入 uint8 类型的张量 in_uint8_t，大小为 1x1x2，数据类型为 uint8，设备同样为指定的 device
        in_uint8_t = torch.ones(1, 1, 2, dtype=torch.uint8, device=device)

        # 使用上面创建的 Upsample 模块进行前向传播，捕获警告信息
        with warnings.catch_warnings(record=True) as w:
            out_t = m(in_t)  # 对 in_t 进行上采样
            out_uint8_t = m(in_uint8_t)  # 对 in_uint8_t 进行上采样

        # 断言输出张量 out_t 的数据与预期的全为1的 1x1x4 双精度浮点型张量相等
        self.assertEqual(torch.ones(1, 1, 4, device=device, dtype=torch.double), out_t.data)

        # 断言输出张量 out_uint8_t 的数据与预期的全为1的 1x1x4 uint8 类型张量相等
        self.assertEqual(torch.ones(1, 1, 4, dtype=torch.uint8, device=device), out_uint8_t.data)

        # 检查上采样功能
        # 创建具有梯度的输入张量 input，大小为 1x1x2，数据类型为双精度浮点型，设备为指定的 device
        input = torch.randn(1, 1, 2, requires_grad=True, device=device, dtype=torch.double)

        # 使用 gradcheck 函数检查 F.interpolate 函数在输入 input 上的梯度是否正确计算
        gradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_forward_ad=check_forward_ad)

        # 使用 gradgradcheck 函数检查 F.interpolate 函数在输入 input 上的梯度是否正确计算，同时验证正向和反向传播的一致性
        gradgradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_fwd_over_rev=check_forward_ad)

        # 检查下采样功能
        # 创建具有梯度的输入张量 input，大小为 1x1x20，数据类型为双精度浮点型，设备为指定的 device
        input = torch.randn(1, 1, 20, requires_grad=True, device=device, dtype=torch.double)

        # 使用 gradcheck 函数检查 F.interpolate 函数在输入 input 上的梯度是否正确计算
        gradcheck(lambda x: F.interpolate(x, 11, mode=mode), [input], check_forward_ad=check_forward_ad)

        # 使用 gradgradcheck 函数检查 F.interpolate 函数在输入 input 上的梯度是否正确计算，同时验证正向和反向传播的一致性
        gradgradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_fwd_over_rev=check_forward_ad)

        # 检查 CUDA 和 CPU 之间的一致性
        if torch.device(device).type == 'cuda':
            # 创建具有梯度的输入张量 input_cuda，大小为 1x1x20，数据类型为双精度浮点型，设备为指定的 device
            input_cuda = torch.randn(1, 1, 20, device=device, dtype=torch.double)

            # 将 input_cuda 复制到 CPU
            input_cpu = input_cuda.cpu()

            # 在 CUDA 设备上使用 F.interpolate 函数进行插值，得到 output_cuda
            output_cuda = F.interpolate(input_cuda, 4, mode=mode)

            # 在 CPU 上使用 F.interpolate 函数进行插值，得到 output_cpu
            output_cpu = F.interpolate(input_cpu, 4, mode=mode)

            # 断言 output_cuda 在 CPU 上的结果与 output_cpu 相等
            self.assertEqual(output_cuda.cpu(), output_cpu)

            # 再次进行插值，调整目标大小为 24
            output_cuda = F.interpolate(input_cuda, 24, mode=mode)
            output_cpu = F.interpolate(input_cpu, 24, mode=mode)

            # 断言 output_cuda 在 CPU 上的结果与 output_cpu 相等
            self.assertEqual(output_cuda.cpu(), output_cpu)

    # 使用参数化测试装饰器，定义测试函数 test_upsamplingNearest1d_correctness，接受设备、输入大小 isize 和目标大小 osize 作为参数
    @parametrize_test("isize, osize", [(20, 11), (10, 15)])
    def test_upsamplingNearest1d_correctness(self, device, isize, osize):
        # 在这里检查输出是否与 OpenCV 的 INTER_NEAREST 类似
        # 创建输入张量 in_t，大小为 isize，数据类型为浮点型，设备为指定的 device
        in_t = torch.arange(isize, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

        # 使用 F.interpolate 函数进行插值，目标大小为 osize，禁用重新计算比例因子，使用最近邻插值模式
        out_t = F.interpolate(
            in_t, size=(osize, ), recompute_scale_factor=False, mode="nearest"
        )

        # 计算期望输出，类似于 OpenCV 的计算方式
        expected_out = torch.zeros(osize, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        scale = 1.0 * isize / osize
        for o in range(osize):
            i_f32 = o * scale
            i = int(i_f32)
            expected_out[0, 0, o] = in_t[0, 0, i]
        expected_out = expected_out.to(device=device)

        # 断言输出张量 out_t 与期望输出 expected_out 相等
        self.assertEqual(out_t, expected_out)
    # 定义一个测试方法，用于验证上采样最近邻插值在特定情况下的正确性
    def test_upsamplingNearestExact1d_rescale(self, device):
        # 检查 https://github.com/pytorch/pytorch/issues/62237
        isize = 20
        # 创建一个设备相关的张量，表示从0到isize-1的序列，扩展为三维张量
        in_t = torch.arange(isize, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
        
        # 针对不同的缩放因子s进行迭代测试
        for s in [1.00001, ]:
            # 使用最近邻精确插值对输入张量进行插值操作
            out_t = F.interpolate(
                in_t, scale_factor=s, recompute_scale_factor=False, mode="nearest-exact"
            )
            # 预期输出应该等于输入张量本身，用于验证插值的正确性
            expected_out = in_t
            self.assertEqual(out_t, expected_out, msg=f"scale: {s}")

        # 检查如果输出大小为输入大小的两倍时，数据是否正确重复
        for s in [2.00001, ]:
            # 使用最近邻精确插值对输入张量进行插值操作
            out_t = F.interpolate(
                in_t, scale_factor=s, recompute_scale_factor=False, mode="nearest-exact"
            )
            # 输入是 [[[0, 1, 2, 3, ..., 9]]]
            # 预期输出是 [[[0, 0, 1, 1, 2, 2, ..., 9, 9]]]
            expected_out = in_t.repeat_interleave(2, dim=-1)
            self.assertEqual(out_t, expected_out)

    # 参数化测试方法，验证最近邻精确插值在不同参数下的正确性
    @parametrize_test("isize, osize", [(20, 11), (10, 15)])
    def test_upsamplingNearestExact1d_correctness(self, device, isize, osize):
        # 在这里我们验证输出是否与Scikit-Image/Scipy类似
        # 检查 https://github.com/pytorch/pytorch/issues/34808
        # 创建一个设备相关的张量，表示从0到isize-1的序列，扩展为三维张量
        in_t = torch.arange(isize, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
        # 使用最近邻精确插值对输入张量进行插值操作，目标输出大小为(osize, )
        out_t = F.interpolate(
            in_t, size=(osize, ), recompute_scale_factor=False, mode="nearest-exact"
        )
        # 计算期望输出，以便与Scikit-Image/Scipy进行比较
        expected_out = torch.zeros(osize, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        scale = 1.0 * isize / osize
        for o in range(osize):
            i_f32 = (o + 0.5) * scale
            i = int(i_f32)
            expected_out[0, 0, o] = in_t[0, 0, i]
        expected_out = expected_out.to(device=device)
        self.assertEqual(out_t, expected_out)

    # 参数化测试方法，验证最近邻和最近邻精确插值在不同内存格式下的正确性
    @parametrize_test("memory_format", [torch.contiguous_format, torch.channels_last])
    @parametrize_test("mode", ["nearest", "nearest-exact"])
    @parametrize_test("memory_format", [torch.contiguous_format, torch.channels_last])
    @parametrize_test("isize, osize", [(20, 11), (10, 15)])
    # 定义一个测试函数，用于测试上采样最近邻方法的正确性
    def test_upsamplingNearest2d_correctness(self, device, memory_format, isize, osize):
        # 创建一个输入张量，包含从0到isize*isize的数字，设备为指定设备，形状为(1, 1, isize, isize)
        in_t = torch.arange(isize * isize, dtype=torch.float, device=device).reshape(1, 1, isize, isize)
        # 根据指定的内存格式重排输入张量的内存布局
        in_t = in_t.contiguous(memory_format=memory_format)
        # 使用最近邻插值方式，对输入张量进行大小调整到(osize, osize)，不重新计算缩放因子
        out_t = F.interpolate(
            in_t, size=(osize, osize), recompute_scale_factor=False, mode="nearest"
        )
        # 计算期望输出，模仿OpenCV的行为
        expected_out = torch.zeros(1, 1, osize, osize, dtype=torch.float)
        scale = 1.0 * isize / osize
        for o1 in range(osize):
            i1_f32 = o1 * scale
            i1 = int(i1_f32)
            for o2 in range(osize):
                i2_f32 = o2 * scale
                i2 = int(i2_f32)
                expected_out[0, 0, o1, o2] = in_t[0, 0, i1, i2]
        expected_out = expected_out.to(device=device)
        # 断言输出张量与期望输出张量相等
        self.assertEqual(out_t, expected_out)

    # 使用参数化测试装饰器，测试不同内存格式的上采样最近邻方法的正确性
    @parametrize_test("memory_format", [torch.contiguous_format, torch.channels_last])
    # 使用参数化测试装饰器，测试不同输入输出大小的上采样最近邻方法的正确性
    @parametrize_test("isize, osize", [(20, 11), (10, 15)])
    def test_upsamplingNearestExact2d_correctness(self, device, memory_format, isize, osize):
        # 在这里我们检查输出是否与Scikit-Image/Scipy的结果匹配
        # 参考 https://github.com/pytorch/pytorch/issues/34808
        # 创建一个输入张量，包含从0到isize*isize的数字，设备为指定设备，形状为(1, 1, isize, isize)
        in_t = torch.arange(isize * isize, dtype=torch.float, device=device).reshape(1, 1, isize, isize)
        # 根据指定的内存格式重排输入张量的内存布局
        in_t = in_t.contiguous(memory_format=memory_format)
        # 使用最近邻插值方式，对输入张量进行大小调整到(osize, osize)，不重新计算缩放因子
        out_t = F.interpolate(
            in_t, size=(osize, osize), recompute_scale_factor=False, mode="nearest-exact"
        )
        # 计算期望输出，模仿Scikit-Image/Scipy的行为
        expected_out = torch.zeros(1, 1, osize, osize, dtype=torch.float)
        scale = 1.0 * isize / osize
        for o1 in range(osize):
            i1_f32 = (o1 + 0.5) * scale
            i1 = int(i1_f32)
            for o2 in range(osize):
                i2_f32 = (o2 + 0.5) * scale
                i2 = int(i2_f32)
                expected_out[0, 0, o1, o2] = in_t[0, 0, i1, i2]
        expected_out = expected_out.to(device=device)
        # 断言输出张量与期望输出张量相等
        self.assertEqual(out_t, expected_out)

    # 使用参数化测试装饰器，测试不同内存格式和插值模式的上采样方法
    @parametrize_test("memory_format", [torch.contiguous_format, torch.channels_last_3d])
    @parametrize_test("mode", ["nearest", "nearest-exact"])
   `
# 定义一个测试方法，用于测试三维最近邻插值操作
def test_upsamplingNearest3d(self, device, memory_format, mode):
    # 检查是否支持自动微分，XLA 引擎不支持自动微分因为 XLA 张量没有存储
    check_forward_ad = torch.device(device).type != 'xla'

    # 创建一个 Upsample 模块实例，设置插值大小和插值模式
    m = nn.Upsample(size=4, mode=mode)
    
    # 创建一个双精度张量，全为1，设置设备和内存格式，并标记需要梯度
    in_t = torch.ones(1, 2, 2, 2, 2, device=device, dtype=torch.double).contiguous(memory_format=memory_format).requires_grad_()
    
    # 创建一个无符号8位整数张量，全为1，设置设备和内存格式
    in_uint8_t = torch.ones(
        1, 2, 2, 2, 2, dtype=torch.uint8, device=device
    ).contiguous(memory_format=memory_format)
    
    # 捕获警告信息，执行 Upsample 操作并得到输出张量
    with warnings.catch_warnings(record=True) as w:
        out_t = m(in_t)
        out_uint8_t = m(in_uint8_t)
    
    # 预期的输出张量，全为1，设置设备和双精度类型
    expected_output = torch.ones(1, 2, 4, 4, 4, device=device, dtype=torch.double)
    
    # 断言输出张量与预期输出相等
    self.assertEqual(expected_output, out_t)
    # 断言输出张量转换为无符号8位整数与预期输出相等
    self.assertEqual(expected_output.to(torch.uint8), out_uint8_t)
    
    # 断言内存格式在输出中被保持
    self.assertTrue(out_t.is_contiguous(memory_format=memory_format))
    
    # 对输出张量进行反向传播
    out_t.backward(torch.randn_like(out_t))
    
    # 断言输入张量的梯度在相同内存格式下保持连续
    self.assertTrue(in_t.grad.is_contiguous(memory_format=memory_format))

    # 创建一个双精度随机张量，设置设备和内存格式，标记需要梯度
    input = torch.randn(
        1, 2, 2, 2, 2, requires_grad=True, device=device, dtype=torch.double
    ).contiguous(memory_format=memory_format)
    
    # 使用 gradcheck 方法检查插值函数的梯度
    gradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_forward_ad=check_forward_ad)
    
    # 使用 gradgradcheck 方法检查插值函数的二阶梯度
    gradgradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_fwd_over_rev=check_forward_ad)

    # 断言 CPU 和 CUDA 在处理 channels_last 内存格式时的一致性
    # 参考：https://github.com/pytorch/pytorch/issues/54590
    if torch.device(device).type == 'cuda':
        # 创建一个双精度张量，全为1，设置设备和内存格式为 channels_last_3d，并标记需要梯度
        a = torch.ones(
            2, 2, 2, 3, 4, device=device, requires_grad=True, dtype=torch.double
        ).contiguous(memory_format=torch.channels_last_3d)
        
        # 使数据不对称，确保 CUDA/CPU 适当处理 channels_last
        a[1][1][1][2][2] = a[1][1][1][2][3] = 0
        
        # 使用 interpolate 函数对 CUDA 张量进行插值操作
        out_cuda = torch.nn.functional.interpolate(a, scale_factor=2, mode=mode)
        
        # 使用 interpolate 函数对 CPU 张量进行插值操作
        out_cpu = torch.nn.functional.interpolate(a.to('cpu'), scale_factor=2, mode=mode)
        
        # 断言 CPU 输出与 CUDA 输出相等
        self.assertEqual(out_cpu, out_cuda.to('cpu'))

        # 使用 gradcheck 方法检查插值函数在 CUDA 下的梯度
        gradcheck(lambda x: F.interpolate(x, 4, mode=mode), [a], check_forward_ad=check_forward_ad)
        
        # 使用 gradgradcheck 方法检查插值函数在 CUDA 下的二阶梯度
        gradgradcheck(lambda x: F.interpolate(x, 4, mode=mode), [a], check_fwd_over_rev=check_forward_ad)

        # 使用 gradcheck 方法检查插值函数在 CPU 下的梯度
        gradcheck(lambda x: F.interpolate(x, 4, mode=mode), [a.to('cuda')], check_forward_ad=check_forward_ad)
        
        # 使用 gradgradcheck 方法检查插值函数在 CPU 下的二阶梯度
        gradgradcheck(lambda x: F.interpolate(x, 4, mode=mode), [a.to('cuda')], check_fwd_over_rev=check_forward_ad)
    # 定义测试方法，用于验证三维最近邻插值的正确性
    def test_upsamplingNearest3d_correctness(self, device, memory_format, isize, osize):
        # 检查输出是否与OpenCV的INTER_NEAREST类似的结果匹配

        # 创建输入张量，包含从0到isize*isize*isize的连续浮点数，放在设备上
        in_t = torch.arange(isize * isize * isize, dtype=torch.float, device=device)
        # 将输入张量重新形状为1x1xisizexisizexisize
        in_t = in_t.reshape(1, 1, isize, isize, isize)
        # 根据指定的内存格式（memory_format），使输入张量连续化
        in_t = in_t.contiguous(memory_format=memory_format)
        # 使用最近邻插值方法，调整输入张量的尺寸为(osize, osize, osize)，不重新计算缩放因子
        out_t = F.interpolate(
            in_t, size=(osize, osize, osize), recompute_scale_factor=False, mode="nearest"
        )

        # 计算期望输出，模拟OpenCV的行为
        expected_out = torch.zeros(1, 1, osize, osize, osize, dtype=torch.float)
        scale = 1.0 * isize / osize
        for o1 in range(osize):
            i1_f32 = o1 * scale
            i1 = int(i1_f32)
            for o2 in range(osize):
                i2_f32 = o2 * scale
                i2 = int(i2_f32)
                for o3 in range(osize):
                    i3_f32 = o3 * scale
                    i3 = int(i3_f32)
                    expected_out[0, 0, o1, o2, o3] = in_t[0, 0, i1, i2, i3]

        # 将期望输出也放在设备上，并使用断言检查实际输出是否与期望输出相等
        expected_out = expected_out.to(device=device)
        self.assertEqual(out_t, expected_out)

    # 使用参数化测试框架，对内存格式和输入输出尺寸进行测试
    @parametrize_test("memory_format", [torch.contiguous_format, torch.channels_last_3d])
    @parametrize_test("isize, osize", [(20, 11), (10, 15)])
    def test_upsamplingNearestExact3d_correctness(self, device, memory_format, isize, osize):
        # 检查输出是否与Scikit-Image/Scipy类似的结果匹配
        # 参考 https://github.com/pytorch/pytorch/issues/34808

        # 创建输入张量，包含从0到isize*isize*isize的连续浮点数，放在设备上
        in_t = torch.arange(isize * isize * isize, dtype=torch.float, device=device)
        # 将输入张量重新形状为1x1xisizexisizexisize
        in_t = in_t.reshape(1, 1, isize, isize, isize)
        # 根据指定的内存格式（memory_format），使输入张量连续化
        in_t = in_t.contiguous(memory_format=memory_format)
        # 使用最近邻精确插值方法，调整输入张量的尺寸为(osize, osize, osize)，不重新计算缩放因子
        out_t = F.interpolate(
            in_t, size=(osize, osize, osize), recompute_scale_factor=False, mode="nearest-exact"
        )

        # 计算期望输出，模拟Scikit-Image/Scipy的行为
        expected_out = torch.zeros(1, 1, osize, osize, osize, dtype=torch.float)
        scale = 1.0 * isize / osize
        for o1 in range(osize):
            i1_f32 = (o1 + 0.5) * scale
            i1 = int(i1_f32)
            for o2 in range(osize):
                i2_f32 = (o2 + 0.5) * scale
                i2 = int(i2_f32)
                for o3 in range(osize):
                    i3_f32 = (o3 + 0.5) * scale
                    i3 = int(i3_f32)
                    expected_out[0, 0, o1, o2, o3] = in_t[0, 0, i1, i2, i3]

        # 将期望输出也放在设备上，并使用断言检查实际输出是否与期望输出相等
        expected_out = expected_out.to(device=device)
        self.assertEqual(out_t, expected_out)

    # 使用参数化测试框架，对抗锯齿、对齐角点、插值模式和内存格式进行测试，仅限原生设备类型
    @parametrize_test("antialias", [True, False])
    @parametrize_test("align_corners", [True, False])
    @parametrize_test("mode", ["bilinear", "bicubic"])
    @parametrize_test("memory_format", [torch.contiguous_format, torch.channels_last])
    @onlyNativeDeviceTypes
    @parametrize_test("antialias", [True, False])
    @parametrize_test("num_channels", [3, 5])
    # 参数化测试装饰器，为每个模式和数据类型组合生成测试
    @parametrize_test("mode", ["nearest", "nearest-exact", "bilinear", "bicubic"])
    @parametrize_test("dtype", integral_types() + floating_types())
    # 只在本地设备类型上运行测试
    @onlyNativeDeviceTypes
    def test_upsamplingBiMode2d_nonsupported_dtypes(self, device, antialias, num_channels, mode, dtype):
        # 创建一个形状为 (1, num_channels, 32, 32) 的张量，元素为1，指定设备和数据类型
        x = torch.ones(1, num_channels, 32, 32, dtype=dtype, device=device)

        # 是否应该引发运行时错误的标志
        should_raise_runtime_error = True

        # 如果模式包含 "nearest"
        if "nearest" in mode:
            # 如果启用了抗锯齿
            if antialias:
                # 抛出跳过测试的异常，因为最近邻插值模式不支持抗锯齿
                raise SkipTest("Nearest mode does not have antialiasing")
            # 如果数据类型为 uint8 或浮点类型之一
            if dtype in (torch.uint8, ) + floating_types():
                # 不应该引发运行时错误
                should_raise_runtime_error = False

        # 否则，如果模式是 "bilinear" 或 "bicubic"
        elif mode in ("bilinear", "bicubic"):
            # 如果数据类型为浮点类型或者在 CPU 上且数据类型为 uint8
            if dtype in floating_types() or (device == "cpu" and dtype == torch.uint8):
                # 不应该引发运行时错误
                should_raise_runtime_error = False

        # 如果应该引发运行时错误
        if should_raise_runtime_error:
            # 使用断言上下文检查是否引发了带有特定消息的 RuntimeError 异常
            with self.assertRaisesRegex(RuntimeError, "not implemented for"):
                # 调用 F.interpolate 函数，期望引发异常，不存储返回值
                F.interpolate(x, (12, 12), mode=mode, antialias=antialias)
        else:
            # 调用 F.interpolate 函数，不期望引发异常，不存储返回值
            _ = F.interpolate(x, (12, 12), mode=mode, antialias=antialias)

    # 参数化测试装饰器，为每种内存布局格式生成测试
    @parametrize_test("memory_format", [torch.contiguous_format, torch.channels_last])
    def test_upsamplingBilinear2d_aa_correctness(self, device, memory_format):
        # 注意：我们扩展了批量维度，使得 `b*c` 超过了 CUDA 网格 z 维度的最大尺寸 (2**16)
        shape = [23000, 3, 8, 8]
        # 创建一个张量，包含从0到 3*8*8 的浮点数序列，指定设备和数据类型
        t_in = torch.arange(3 * 8 * 8, dtype=torch.float, device=device).reshape(1, *shape[1:])
        # 扩展张量的形状为 shape
        t_in = t_in.expand(shape)
        # 以指定的内存格式使张量连续
        t_in = t_in.contiguous(memory_format=memory_format)
        # 预期的输出结果，是使用 PIL.Image.resize 获得的
        expected_out = torch.tensor([
            17.035713, 20.25, 42.75, 45.964287, 81.03572, 84.25,
            106.75, 109.96428, 145.0357, 148.25, 170.75, 173.9643
        ], device=device, dtype=t_in.dtype).reshape(1, 3, 2, 2)
        # 调用 F.interpolate 函数，使用双线性插值模式，不对齐角点，启用抗锯齿
        t_out = F.interpolate(t_in, size=(2, 2), mode="bilinear", align_corners=False, antialias=True)
        # 使用断言检查 t_out 是否与预期输出匹配
        self.assertEqual(expected_out.expand([*shape[:2], 2, 2]), t_out)

    # 参数化测试装饰器，为每种内存布局格式和插值模式组合生成测试
    @parametrize_test("memory_format", [torch.contiguous_format, torch.channels_last])
    @parametrize_test("mode", ["bilinear", "bicubic"])
    @parametrize_test("antialias", [True, False])
    @parametrize_test("align_corners", [True, False])
    @parametrize_test("num_channels", [3, 5])
    @parametrize_test("output_size", [32, 600])
    @parametrize_test("check_as_unsqueezed_3d_tensor", [True, False])
    @parametrize_test("non_contig", [False, "sliced", "restrided"])
    @parametrize_test("batch_size", [1, 5])
    # 定义测试函数，用于验证双线性插值的一致性
    def test_upsamplingBiMode2d_consistency(
        self,
        device,  # 设备参数，指定测试运行的设备
        memory_format,  # 内存格式参数，影响张量存储方式
        mode,  # 插值模式参数，指定插值算法
        antialias,  # 抗锯齿参数，控制是否使用抗锯齿
        align_corners,  # 对齐角点参数，影响插值结果的角点对齐方式
        num_channels,  # 通道数参数，指定输入张量的通道数量
        output_size,  # 输出尺寸参数，指定期望的输出张量尺寸
        check_as_unsqueezed_3d_tensor,  # 三维张量展开检查参数，用于验证张量形状
        non_contig,  # 非连续张量参数，用于检查非连续存储的情况
        batch_size,  # 批处理大小参数，指定输入张量的批次大小
        # Check output value consistency between resized_input_uint8 and resized input_float
        # 检查 resized_input_uint8 和 resized input_float 之间输出值的一致性

        if torch.device(device).type == "cuda":
            # 如果使用 CUDA 设备，抛出跳过测试的异常，因为暂不支持 uint8
            raise SkipTest("CUDA implementation is not yet supporting uint8")

        torch.manual_seed(0)

        # - input range is set to [30, 220] for bicubic mode, because the bicubic kernel may create
        #   [intermediate] values outside of the [0, 255] range, which need
        #   to be clipped in uint8 path, but not in float path. This isn't
        #   an issue with bilinear kernel.
        #   对于 bicubic 模式，输入范围设置为 [30, 220]，因为 bicubic 核可能会创建超出 [0, 255] 范围的中间值，
        #   这些值在 uint8 路径中需要剪裁，但在 float 路径中不需要。对于 bilinear 核来说，这不是问题。
        input_range = (30, 220) if mode == "bicubic" else (0, 256)
        input_ui8 = torch.randint(*input_range, size=(batch_size, num_channels, 400, 400), dtype=torch.uint8, device=device)
        input_ui8 = input_ui8.contiguous(memory_format=memory_format)

        if non_contig == "sliced":
            # 如果 non_contig 为 "sliced"，则对 input_ui8 进行切片操作
            input_ui8 = input_ui8[:, :, 10:-10, 10:-10]
        elif non_contig == "restrided":
            # 如果 non_contig 为 "restrided"，则对 input_ui8 进行步幅重排操作
            input_ui8 = input_ui8[:, :, ::2, ::2]

        if batch_size == 1 and check_as_unsqueezed_3d_tensor:
            # 如果 batch_size 为 1 并且 check_as_unsqueezed_3d_tensor 为真，则对 input_ui8 进行维度扩展操作
            input_ui8 = input_ui8[0, ...]
            input_ui8 = input_ui8[None, ...]

        input_f32 = input_ui8.float()

        output_f32 = F.interpolate(
            input_f32, size=(output_size, output_size), mode=mode, align_corners=align_corners, antialias=antialias
        ).round().clip(0, 255)
        output_ui8 = F.interpolate(
            input_ui8, size=(output_size, output_size), mode=mode, align_corners=align_corners, antialias=antialias
        )

        if non_contig is False:
            # 如果 non_contig 为 False，则断言 input_ui8 是连续的内存格式
            self.assertTrue(input_ui8.is_contiguous(memory_format=memory_format))

        # FIXME if-clause shows the current behaviour which is definitely unexpected.
        # Ideally we want to fix it such that both the ui8 and f32 outputs are also channels_last
        # See for more details: https://github.com/pytorch/pytorch/pull/100373
        # FIXME if-语句显示了当前的行为，这显然是意外的。
        # 理想情况下，我们希望修复它，使得 ui8 和 f32 输出也是 channels_last 格式
        # 更多细节请参见：https://github.com/pytorch/pytorch/pull/100373
        if batch_size == 1 and check_as_unsqueezed_3d_tensor and memory_format == torch.channels_last:
            # 如果 batch_size 为 1，check_as_unsqueezed_3d_tensor 为真，并且 memory_format 是 channels_last 格式，
            # 则断言 output_ui8 和 output_f32 都是连续的
            self.assertTrue(output_ui8.is_contiguous())
            self.assertTrue(output_f32.is_contiguous())
        else:
            # 否则，断言 output_ui8 和 output_f32 是连续的，但要符合指定的内存格式
            self.assertTrue(output_ui8.is_contiguous(memory_format=memory_format))
            self.assertTrue(output_f32.is_contiguous(memory_format=memory_format))

        if mode == "bilinear":
            # 如果 mode 是 "bilinear"，则使用测试函数检查 output_f32 和 output_ui8.float() 的值是否接近
            torch.testing.assert_close(output_f32, output_ui8.float(), rtol=0, atol=1)
        else:
            # 否则，计算输出 f32 和 ui8 之间的差异，并进行多个断言测试
            diff = (output_f32 - output_ui8.float()).abs()
            self.assertLess(diff.max(), 15)

            threshold = 2
            percent = 3
            self.assertLess((diff > threshold).float().mean(), percent / 100)

            threshold = 5
            percent = 1
            self.assertLess((diff > threshold).float().mean(), percent / 100)

            self.assertLess(diff.mean(), 0.4)
    @parametrize_test("input_size, output_size", [(399, 437), (403, 377)])
    # 使用参数化测试装饰器，定义输入大小和输出大小的参数化测试
    def test_upsamplingBiLinear2d_consistency_interp_size_bug(self, device, memory_format, align_corners, input_size, output_size):
        # 针对 https://github.com/pytorch/pytorch/pull/101403 的非回归测试

        if torch.device(device).type == "cuda":
            # 如果设备是 CUDA，则跳过测试，因为尚不支持 uint8 类型
            raise SkipTest("CUDA implementation is not yet supporting uint8")

        mode = "bilinear"
        # 创建一个 uint8 类型的随机张量作为输入
        input_ui8 = torch.randint(0, 256, size=(1, 3, input_size, input_size), dtype=torch.uint8, device=device)
        input_ui8 = input_ui8.contiguous(memory_format=memory_format)
        # 将 uint8 类型的输入张量转换为 float32 类型
        input_f32 = input_ui8.float()

        # 使用双线性插值方法对 float32 输入进行上采样，并将结果四舍五入到 uint8 类型
        output_f32 = F.interpolate(
            input_f32, size=(output_size, output_size), mode=mode, align_corners=align_corners, antialias=False
        ).round().to(torch.uint8)
        # 使用双线性插值方法对 uint8 输入进行上采样
        output_ui8 = F.interpolate(
            input_ui8, size=(output_size, output_size), mode=mode, align_corners=align_corners, antialias=False
        )
        # 断言两个张量的值在一定的绝对和相对误差范围内相等
        torch.testing.assert_close(output_f32, output_ui8, atol=1, rtol=0)

    def test_upsamplingBicubic2d_correctness(self, device):
        # 对已知输入进行测试：当 align_corners=False 时，结果必须与 OpenCV 的结果匹配
        in_t = torch.arange(8., device=device).view(1, 2, 2, 2)
        # 预期输出结果，使用双三次插值方法
        expected_out_t = torch.tensor(
            [[[[-0.31641, 0.01562, 0.56250, 0.89453],
              [0.34766, 0.67969, 1.22656, 1.55859],
              [1.44141, 1.77344, 2.32031, 2.65234],
              [2.10547, 2.43750, 2.98438, 3.31641]],

             [[3.68359, 4.01562, 4.56250, 4.89453],
              [4.34766, 4.67969, 5.22656, 5.55859],
              [5.44141, 5.77344, 6.32031, 6.65234],
              [6.10547, 6.43750, 6.98438, 7.31641]]]], device=device)
        # 使用双三次插值方法对输入张量进行上采样
        out_t = F.interpolate(in_t, scale_factor=2, mode='bicubic', align_corners=False)
        # 设置打印精度，以便更精细地查看结果
        torch.set_printoptions(precision=5)
        # 断言两个张量在指定的绝对和相对误差范围内相等
        self.assertEqual(out_t, expected_out_t, atol=1e-5, rtol=0)

    @parametrize_test("memory_format", [torch.contiguous_format, torch.channels_last])
    # 使用参数化测试装饰器，定义内存格式参数化测试
    def test_upsamplingBicubic2d_aa_correctness(self, device, memory_format):
        # 创建一个 3 通道的输入张量，并设置其形状为 8x8
        t_in = torch.arange(3 * 8 * 8, dtype=torch.float, device=device).reshape(1, 3, 8, 8)
        t_in = t_in.contiguous(memory_format=memory_format)
        # 预期的输出结果，使用带抗锯齿的双三次插值方法
        expected_out = torch.tensor([
            15.1205635, 18.760439, 44.23956, 47.879436, 79.12056, 82.76044,
            108.23956, 111.87944, 143.12057, 146.76044, 172.23956, 175.87943
        ], device=device, dtype=t_in.dtype).reshape(1, 3, 2, 2)
        # 使用带抗锯齿的双三次插值方法对输入张量进行上采样
        t_out = F.interpolate(t_in, size=(2, 2), mode="bicubic", align_corners=False, antialias=True)
        # 断言两个张量的值相等
        self.assertEqual(expected_out, t_out)

    @parametrize_test("align_corners", [True, False])
    @parametrize_test("memory_format", [torch.contiguous_format, torch.channels_last_3d])
    # 使用 parametrize_test 装饰器，为 test_upsamplingTrilinear3d 方法参数化测试
    def test_upsamplingTrilinear3d(self, device, align_corners, memory_format):
        # 根据 align_corners 和 memory_format 构建参数字典
        kwargs = dict(mode='trilinear', align_corners=align_corners)

        # 测试浮点型缩放因子的上采样和下采样
        for scale_factor in [0.5, 1.5, 2]:
            # 创建 Upsample 模块 m
            m = nn.Upsample(scale_factor=scale_factor, **kwargs)
            # 创建输入张量 in_t，全为1，指定设备和数据类型
            in_t = torch.ones(1, 2, 4, 4, 4, device=device, dtype=torch.double)
            # 根据 memory_format 进行连续化处理，并标记需要梯度
            in_t = in_t.contiguous(memory_format=memory_format).requires_grad_()
            # 计算输出尺寸
            out_size = int(math.floor(in_t.shape[-1] * scale_factor))
            # 使用警告捕获记录
            with warnings.catch_warnings(record=True) as w:
                # 进行模块 m 的前向传播计算
                out_t = m(in_t)
            # 期望的输出张量 expected_out，全为1，与 out_size 和设备、数据类型对应
            expected_out = torch.ones(1, 2, out_size, out_size, out_size, device=device, dtype=torch.double)
            # 断言期望输出与实际输出 out_t 相等
            self.assertEqual(expected_out, out_t)
            # 断言输出张量 out_t 的连续性符合指定的 memory_format
            self.assertTrue(out_t.is_contiguous(memory_format=memory_format))

            # 创建与 out_t 形状相同的梯度张量 grad_out，并使其连续化处理
            grad_out = torch.randn_like(out_t).contiguous(memory_format=memory_format)
            # 清空输入张量的梯度
            in_t.grad = None
            # 对 out_t 进行反向传播计算
            out_t.backward(grad_out)
            # 获取输入张量的梯度 grad_in
            grad_in = in_t.grad
            # 断言输入张量的梯度 grad_in 的连续性符合指定的 memory_format
            self.assertTrue(grad_in.is_contiguous(memory_format=memory_format))

            if memory_format == torch.channels_last_3d:
                # 如果 memory_format 是 channels_last_3d，则检查梯度输入的 CF 和 CL 是否匹配
                in_t.grad = None
                out_t.backward(grad_out.contiguous())
                self.assertEqual(in_t.grad, grad_in)

            # 创建形状为 (1, 2, 4, 4, 4) 的随机张量 input，指定需要梯度
            input = torch.randn(1, 2, 4, 4, 4, requires_grad=True, dtype=torch.double)
            # 使用 F.interpolate 进行插值操作，比较两种不同方式的结果
            self.assertEqual(
                F.interpolate(input, (out_size, out_size, out_size), **kwargs),
                F.interpolate(input, scale_factor=scale_factor, **kwargs))
            # 对插值操作函数进行梯度检查
            gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [input])
            # 对插值操作函数进行二阶梯度检查
            gradgradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [input])

    @onlyCUDA
    @dtypes(torch.half)
    @largeTensorTest('40GB')
    # 使用 onlyCUDA、dtypes 和 largeTensorTest 装饰器，为 test_upsampling_64bit_indexing_channels_last 方法添加限制条件和测试信息
    def test_upsampling_64bit_indexing_channels_last(self, device, dtype):
        # 创建随机张量 x，形状为 (32, 64, 512, 512)，指定设备和数据类型
        x = torch.rand((32, 64, 512, 512), dtype=dtype, device=device)
        # 使用 channels_last 内存格式对 x 进行操作，进行双线性插值，比例因子为 2
        out = torch.nn.functional.interpolate(x.to(memory_format=torch.channels_last), scale_factor=2, mode='nearest')
        # 使用普通内存格式对 x 进行双线性插值，比例因子为 2
        out_ref = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        # 删除原始张量 x
        del x
        # 断言两种插值结果 out 和 out_ref 的近似程度
        self.assertTrue(torch.allclose(out, out_ref))

    @onlyCUDA
    @dtypes(torch.half)
    @largeTensorTest('40GB')
    # 使用 onlyCUDA、dtypes 和 largeTensorTest 装饰器，为 test_replicatepad_64bit_indexing 方法添加限制条件和测试信息
    def test_replicatepad_64bit_indexing(self, device, dtype):
        # 创建 1 维卷积层 conv，输入通道和输出通道均为 128，内核大小为 3，填充为 1，填充模式为 replicate，指定设备和数据类型
        conv = torch.nn.Conv1d(128, 128, 3, 1, 1, padding_mode="replicate", device=device, dtype=dtype)
        # 创建随机张量 x，形状为 (256 * 448 * 2, 128, 96)，指定设备和数据类型
        x = torch.randn(size=(256 * 448 * 2, 128, 96), dtype=dtype, device=device)
        # 对输入张量 x 进行卷积计算，得到输出张量 y
        y = conv(x)
        # 对输出张量 y 的均值进行反向传播
        torch.mean(y).backward()
    # 定义一个测试函数，用于测试上采样最近邻插值的反向传播，支持64位索引
    def test_upsamplingnearest2d_backward_64bit_indexing(self, device, dtype):
        # 创建一个随机张量 x，形状为 (36, 128, 512, 512)，位于指定设备上，数据类型为 dtype，并且需要梯度
        x = torch.randn(size=(36, 128, 512, 512), device=device, dtype=dtype).requires_grad_()
        # 对张量 x 进行上采样，使用最近邻插值，放大倍数为 2
        y = F.interpolate(x, scale_factor=2, mode="nearest")
        # 对 y 进行反向传播，传入一个与 y 同样形状的随机张量作为梯度
        y.backward(torch.randn_like(y))

    # 定义一个函数，执行较慢的带掩码的 softmax 操作
    def _slow_masked_softmax(self, input, mask):
        # 计算输入张量 input 的指数
        exp = torch.exp(input)
        # 将指数张量与掩码相乘，以实现只对掩码内的部分进行 softmax
        exp = exp * mask
        # 对第3维求和，保持维度不变，扩展到与 exp 相同的形状
        s = exp.sum(dim=3, keepdim=True).expand(exp.size())
        # 执行 softmax 操作，将每个元素除以对应位置的求和结果 s
        return exp / s
    # 定义一个测试方法，用于验证不同类型的掩码在快速路径上的处理是否正确，并且与显式的慢速计算结果匹配。
    def test_masked_softmax_mask_types(self, device):
        # 定义不同大小的测试用例，每个元组包含(B, num_heads, L)，分别表示批次大小、头数和长度。
        sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]

        # 遍历不同大小的测试用例
        for (B, num_heads, L) in sizes:

            # mask_type == 0 => 形状为 LxL 的注意力掩码
            src_mask_orig = torch.randint(0, 2, (L, L)).bool()
            src_mask = src_mask_orig.reshape(1, 1, L, L).expand(B, num_heads, L, L).bool()

            # mask_type == 1 => 形状为 BxL 的填充掩码
            src_key_padding_mask_orig = torch.randint(0, 2, (B, L)).bool()
            src_key_padding_mask = src_key_padding_mask_orig.reshape(B, 1, 1, L).expand(B, num_heads, L, L).bool()

            # mask_type == 2 => 形状为 BxHxLxL 的通用掩码
            generic_mask = torch.randint(0, 2, (B, num_heads, L, L)).bool()

            # 构建掩码列表，每个元素包含原始掩码、扩展后的掩码和掩码类型
            masks = [(src_mask_orig, src_mask, 0),
                     (src_key_padding_mask_orig, src_key_padding_mask, 1),
                     (generic_mask, generic_mask, 2)
                     ]

            # 遍历维度列表，包括0和3
            for dim in [0, 3]:
                for mask_orig, mask, mask_type in masks:
                    # 如果运行在 CUDA 设备上且头数为奇数且掩码类型为1，则跳过当前循环
                    if (self.device_type == "cuda") and (num_heads % 2) and (mask_type == 1):
                        continue

                    # 生成随机输入张量
                    input = torch.randn((B, num_heads, L, L))

                    # 如果运行在 CUDA 设备上，将输入张量和掩码转移到 CUDA
                    if (self.device_type == "cuda"):
                        input = input.cuda()
                        mask = mask.cuda()
                        mask_orig = mask_orig.cuda()

                    # 调用 PyTorch 内置的 _masked_softmax 函数得到本地计算结果
                    native_res = torch._masked_softmax(input, mask_orig, dim, mask_type)

                    # 取反掩码，用于慢速 softmax 函数
                    mask = ~mask

                    # 定义慢速 softmax 函数，输入张量乘以掩码后取指数，计算归一化结果
                    def slow_masked_softmax(input, mask):
                        exp = torch.exp(input)
                        exp = exp * mask
                        s = exp.sum(dim=dim, keepdim=True).expand(exp.size())
                        return exp / s

                    # 调用慢速 softmax 函数得到 PyTorch 计算结果，并将 NaN 替换为 0
                    pt_res = slow_masked_softmax(input, mask)
                    pt_res = torch.nan_to_num(pt_res)

                    # 计算掩码的逻辑非，用于确定应该填充的位置
                    mask_not = mask.logical_not()

                    # 创建掩码输出，将所有 True 行填充为 False
                    mask_out = mask_not.all(dim, keepdim=True).expand(mask_not.shape)

                    # 断言本地计算结果与 PyTorch 计算结果匹配
                    self.assertEqual(
                        pt_res.masked_fill(mask_out, 0),
                        native_res.masked_fill(mask_out, 0),
                        exact_dtype=True
                    )

    @onlyCUDA
    @gcIfJetson
    def test_masked_softmax_devices_parity(self):
        # 测试 softmax 函数对三种类型的掩码（0：LxL 注意力掩码，1：BxL 填充掩码，2：BxHxLxL 通用掩码）在 CPU 和 CUDA 上的结果是否一致。

        sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]
        for (B, num_heads, L) in sizes:
            # mask_type == 0 => LxL 注意力掩码
            src_mask = torch.randint(0, 2, (L, L)).bool()
            # mask_type == 1 => BxL 填充掩码
            src_key_padding_mask = torch.randint(0, 2, (B, L)).bool()
            # mask_type == 2 => BxHxLxL 通用掩码
            generic_mask = torch.randint(0, 2, (B, num_heads, L, L)).bool()
            masks = [(src_mask, 0), (src_key_padding_mask, 1), (generic_mask, 2)]
            input = torch.randn((B, num_heads, L, L))
            for dim in [0, 3]:
                for mask, mask_type in masks:
                    if (num_heads % 2) and (mask_type == 1):
                        # 当头数为奇数时，CUDA 不支持填充掩码
                        continue

                    def softmax_on_device(mask, input, device):
                        # 在指定设备上计算 softmax
                        input_device = input.to(device)
                        mask_device = mask.to(device)
                        softmax_res = torch._masked_softmax(input_device, mask_device, dim, mask_type)
                        if mask_type == 0:
                            mask_expanded = mask_device.reshape(1, 1, L, L).expand(B, num_heads, L, L).bool()
                        elif mask_type == 1:
                            mask_expanded = mask_device.reshape(B, 1, 1, L).expand(B, num_heads, L, L).bool()
                        else:
                            mask_expanded = mask_device
                        # 在结果中，仅填充所有行都为 True 的部分，因为这些是非确定性的（可能为 0）
                        # 将所有行都为 True 的部分填充为 0
                        mask_out = mask_expanded.all(dim, keepdim=True).expand(mask_expanded.shape)
                        softmax_res = softmax_res.masked_fill(mask_out, 0)
                        return softmax_res

                    cpu_res = softmax_on_device(mask, input, "cpu")
                    cuda_res = softmax_on_device(mask, input, "cuda")
                    self.assertEqual(cpu_res, cuda_res, exact_dtype=True)
    # 定义一个测试函数，用于测试带有遮罩的 softmax 函数
    def test_masked_softmax(self, device):
        # 定义不同的输入大小
        sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]
        
        # 遍历不同的大小组合
        for (B, num_heads, L) in sizes:
            # 对于每个大小组合，遍历两个维度：0 和 3
            for dim in [0, 3]:
                # 生成随机输入张量，形状为 (B, num_heads, L, L)
                input = torch.randn((B, num_heads, L, L))
                
                # 生成随机的二进制遮罩，形状为 (B, L)，并扩展为 (B, num_heads, L, L)
                mask = torch.randint(0, 2, (B, L))
                mask = mask.reshape(B, 1, 1, L).expand(B, num_heads, L, L).bool()
                
                # 设置遮罩类型为 1，这里暂时未看到代码中如何使用该变量
                mask_type = 1
                
                # 如果设备类型为 cuda，则将输入和遮罩张量移动到 GPU 上
                if (self.device_type == "cuda"):
                    input = input.cuda()
                    mask = mask.cuda()
                
                # 调用 PyTorch 内部的带遮罩 softmax 函数
                native_res = torch._masked_softmax(input, mask, dim, mask_type)
                
                # 反转遮罩（将 True 变为 False，将 False 变为 True）
                mask = ~mask

                # 定义一个慢速的带遮罩 softmax 函数
                def slow_masked_softmax(input, mask):
                    exp = torch.exp(input)
                    exp = exp * mask
                    s = exp.sum(dim=dim, keepdim=True).expand(exp.size())
                    return exp / s
                
                # 使用慢速 softmax 函数计算参考结果
                pt_res = slow_masked_softmax(input, mask)
                
                # 将结果中的 NaN 值替换为 0
                pt_res = torch.nan_to_num(pt_res)
                
                # 取反转后的遮罩
                mask_not = mask.logical_not()
                
                # 生成一个遮罩，用于标记完全遮罩掉的行（即所有元素为 True 的行）
                mask_out = mask_not.all(dim, keepdim=True).expand(mask_not.shape)
                
                # 断言慢速和快速 softmax 函数的结果是否相同
                self.assertEqual(
                    pt_res.masked_fill(mask_out, 0),
                    native_res.masked_fill(mask_out, 0),
                    exact_dtype=True
                )

    # 根据不同的数据类型和精度要求，测试低精度下的带遮罩 softmax 函数
    @dtypes(torch.bfloat16, torch.half)
    @precisionOverride({torch.bfloat16: 2e-2, torch.half: 3e-3})
    def test_masked_softmax_lowp(self, dtype):
        # 定义不同的输入大小
        sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]
        
        # 遍历不同的大小组合
        for (B, num_heads, L) in sizes:
            # 对于每个大小组合，遍历两个维度：0 和 3
            for dim in [0, 3]:
                # 生成低精度随机输入张量，形状为 (B, num_heads, L, L)，并设置 requires_grad=True
                input_lowp = torch.randn((B, num_heads, L, L), dtype=dtype).requires_grad_()
                
                # 生成浮点数精度的输入张量作为参考，并且将梯度计算分离出来
                input_ref = input_lowp.float().detach().requires_grad_()
                
                # 生成随机的二进制遮罩，形状为 (B, L)，并扩展为 (B, num_heads, L, L)
                mask = torch.randint(0, 2, (B, L))
                mask = mask.reshape(B, 1, 1, L).expand(B, num_heads, L, L).bool()

                # 对于每种遮罩类型，分别计算低精度和参考的带遮罩 softmax 结果
                for mask_type in [1, 2]:
                    res_ref = torch._masked_softmax(input_ref, mask, dim, mask_type)
                    res = torch._masked_softmax(input_lowp, mask, dim, mask_type)
                    
                    # 断言低精度结果与参考结果相等
                    self.assertEqual(res_ref.to(dtype), res)

                    # 生成一个与 res_ref 相同大小的随机梯度，并将其转换为低精度
                    grad_lowp = torch.randn_like(res_ref).to(dtype=dtype)
                    grad_ref = grad_lowp.float()

                    # 分别计算低精度和参考结果的梯度
                    res_ref.backward(grad_ref)
                    res.backward(grad_lowp)
                    
                    # 断言低精度输入的梯度与参考输入的梯度相等
                    self.assertEqual(input_ref.grad.to(dtype), input_lowp.grad)
    # 定义一个辅助函数用于测试带有掩码的 softmax 操作
    def _test_masked_softmax_helper(self, input, dim, mask, mask_type):
        # 创建一个输入的副本，将其从计算图中分离并克隆，同时要求计算梯度
        input_ref = input.detach().clone().requires_grad_()
        # 调用 PyTorch 中的 _masked_softmax 函数进行 softmax 操作，根据给定的掩码和维度
        result = torch._masked_softmax(input, mask, dim, mask_type)

        # 根据预期结果计算未掩码部分的 softmax，同时要求梯度
        expected = torch._softmax(input_ref.masked_fill(mask, float('-inf')), dim, False)
        grad = torch.randn_like(expected).to(dtype=expected.dtype)

        # 计算结果的反向传播梯度
        result.backward(grad)
        # 计算预期结果的反向传播梯度
        expected.backward(grad)

        # 确保可选参数也能正常工作
        if dim == input.dim() - 1:
            # 创建输入的另一个副本，将其从计算图中分离并克隆，同时要求计算梯度
            input_ref_default = input.detach().clone().requires_grad_()
            # 调用 _masked_softmax 函数，不指定维度，使用默认参数
            result_default = torch._masked_softmax(input_ref_default, mask, None, mask_type)
            result_default.backward(grad)
            # 使用断言确保结果和默认结果一致
            self.assertEqual(result, result_default)
            # 使用断言确保输入的梯度和默认输入的梯度一致
            self.assertEqual(input.grad, input_ref_default.grad)

        # 在结果中，只填充完全掩码的行，因为这些行是非确定性的（可能为 0）
        # 将所有元素都为 True 的行转换为 False
        mask_out = mask.all(dim, keepdim=True).expand(mask.shape)
        # 使用断言确保结果中完全掩码的行被填充为 0，与预期的填充一致
        self.assertEqual(result.masked_fill(mask_out, 0), expected.masked_fill(mask_out, 0))

        # 使用断言确保输入的梯度被转换为数值，以避免 NaN
        self.assertEqual(input.grad, torch.nan_to_num(input_ref.grad))
        # 使用断言确保输入的梯度中被掩码的部分被填充为 0.0
        self.assertEqual(input.grad, input.grad.masked_fill(mask, 0.0))

    # 测试 softmax 操作的梯度
    def test_masked_softmax_grad(self, device):
        # 不同形状的输入数据
        shapes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]
        for shape in shapes:
            # 确定要测试的维度
            dims = [0, len(shape) - 1] if len(shape) > 0 else [0]
            for dim in dims:
                # 测试不同的掩码类型
                for mask_type in [1, 2]:  # 1 = BxL => src_key_padding_mask
                    # 创建随机输入数据，要求计算梯度
                    input = torch.randn(shape, requires_grad=True)
                    # 创建随机掩码
                    mask = torch.randint(0, 2, shape).bool()
                    # 如果设备类型是 CUDA，则将输入和掩码移动到 CUDA 上，并分离计算梯度
                    if (self.device_type == "cuda"):
                        input = input.cuda().detach().requires_grad_()
                        mask = mask.cuda()
                    # 调用 _test_masked_softmax_helper 函数进行测试
                    self._test_masked_softmax_helper(input, dim, mask, mask_type)

    # 在此测试中，预期前向传播将产生 NaN，因为当 dim=0 时，我们只有未指定的值
    def test_masked_softmax_forward_with_nans(self, device):
        # 设置维度为 0
        dim = 0
        # 不同形状的输入数据对
        shapes = [(4, 5), (50, 100), (1500, 1200)]
        for (x, y) in shapes:
            # 测试不同的掩码类型
            for mask_type in [1, 2]:  # 1 = BxL => src_key_padding_mask
                # 创建随机输入数据，要求计算梯度
                input = torch.randn((x, y), requires_grad=True)
                # 创建掩码，使用一种特定模式
                mask = torch.tensor([i % 2 for i in range(y)]).expand((x, y)).bool()
                # 如果设备类型是 CUDA，则将输入和掩码移动到 CUDA 上，并分离计算梯度
                if (self.device_type == "cuda"):
                    input = input.cuda().detach().requires_grad_()
                    mask = mask.cuda()
                # 调用 _test_masked_softmax_helper 函数进行测试
                self._test_masked_softmax_helper(input, dim, mask, mask_type)

    # 仅在 CUDA 设备上执行的装饰器
    @onlyCUDA
    # 定义一个测试方法，用于测试带掩码的 softmax 在 Transformer 布局下的情况
    def test_masked_softmax_transformer_layout(self, device):
        # 设置批量大小 B
        B = 211
        # 设置注意力头数
        num_heads = 16
        # 设置序列长度 L
        L = 42
        # 生成随机输入张量，形状为 (B, num_heads, L, L)
        input = torch.randn((B, num_heads, L, L))
        # 确定操作维度 dim，为输入张量的最后一个维度的索引
        dim = input.dim() - 1
        # 创建一个随机的二元掩码张量 mask，形状为 (B, L)
        mask = torch.randint(0, 2, (B, L))
        # 设置掩码类型 mask_type = 1，用于 src_key_padding_mask
        mask_type = 1   # BxL => src_key_padding_mask
        # 如果设备类型为 CUDA，则将输入张量和掩码张量移动到 CUDA 设备上
        if (self.device_type == "cuda"):
            input = input.cuda()
            mask = mask.cuda()
        # 将掩码张量转换为布尔型
        mask = mask.bool()
        # 调用 PyTorch 内部函数 _masked_softmax，对输入张量进行带掩码的 softmax 操作
        native_res = torch._masked_softmax(input, mask, dim, mask_type)
        # 将掩码张量调整形状为 (B, 1, 1, L)，并在第二和第三维度上扩展为 (B, num_heads, L, L)
        mask = mask.reshape(B, 1, 1, L).expand(B, num_heads, L, L)
        # 取反掩码张量，将其转换为浮点型
        mask = ~mask
        mask = mask.float()

        # 调用类中的 _slow_masked_softmax 方法，对输入张量执行带掩码的 softmax 操作
        pt_res = self._slow_masked_softmax(input, mask)
        # 使用断言验证 PyTorch 原生实现与自定义实现的结果是否相等
        self.assertEqual(pt_res, native_res, exact_dtype=True)

    # 只在 CUDA 设备上执行的测试方法标记
    @onlyCUDA
    def test_masked_softmax_TxT_layout(self, device):
        # 设置批量大小 B
        B = 211
        # 设置注意力头数
        num_heads = 16
        # 设置序列长度 L
        L = 42
        # 生成随机输入张量，形状为 (B, num_heads, L, L)
        input = torch.randn((B, num_heads, L, L))
        # 确定操作维度 dim，为输入张量的最后一个维度的索引
        dim = input.dim() - 1
        # 创建一个随机的二元掩码张量 mask，形状为 (L, L)
        mask = torch.randint(0, 2, (L, L))
        # 设置掩码类型 mask_type = 0，用于 src_mask
        mask_type = 0   # LxL => src_mask
        # 如果设备类型为 CUDA，则将输入张量和掩码张量移动到 CUDA 设备上
        if (self.device_type == "cuda"):
            input = input.cuda()
            mask = mask.cuda()
        # 将掩码张量转换为布尔型
        mask = mask.bool()
        # 调用 PyTorch 内部函数 _masked_softmax，对输入张量进行带掩码的 softmax 操作
        native_res = torch._masked_softmax(input, mask, dim, mask_type)
        # 将掩码张量扩展为 (B, num_heads, L, L)
        mask = mask.expand(B, num_heads, L, L)
        # 取反掩码张量，将其转换为浮点型
        mask = ~mask
        mask = mask.float()

        # 调用类中的 _slow_masked_softmax 方法，对输入张量执行带掩码的 softmax 操作
        pt_res = self._slow_masked_softmax(input, mask)
        # 使用断言验证 PyTorch 原生实现与自定义实现的结果是否相等
        self.assertEqual(pt_res, native_res, exact_dtype=True)

    # 在 CPU 上执行的测试方法，用于验证 log_softmax 方法
    @onlyCPU
    @dtypes(torch.bfloat16, torch.half)
    def test_log_softmax_cpu(self, device, dtype):
        # 遍历维度列表 [0, 1]
        for dim in [0, 1]:
            # 生成随机浮点型输入张量 inputf，形状为 (200, 200)，在指定设备和数据类型上
            inputf = torch.rand(200, 200, device=device, dtype=torch.float, requires_grad=True)
            # 将 inputf 转换为指定数据类型 dtype，且不再保留梯度
            input = inputf.to(dtype).detach().requires_grad_(True)
            # 使用 PyTorch 中的 F.log_softmax 计算 log softmax，维度为 dim
            outf = F.log_softmax(inputf, dim=dim)
            out = F.log_softmax(input, dim=dim)
            # 使用断言验证 log softmax 的结果是否相等，允许一定的数值误差
            self.assertEqual(out, outf.to(dtype=dtype), atol=0.1, rtol=0)

            # 计算输出张量的梯度和
            out.sum().backward()
            outf.sum().backward()
            # 使用断言验证输入张量的梯度是否与 inputf 的梯度在指定数据类型下相等
            self.assertEqual(input.grad, inputf.grad.to(dtype), atol=0.1, rtol=0)

    # 在 CPU 上执行的测试方法，用于验证 softmax 方法
    @onlyCPU
    @dtypes(torch.bfloat16, torch.half)
    def test_softmax_cpu(self, device, dtype):
        # 遍历维度列表 [0, 1]
        for dim in [0, 1]:
            # 生成随机浮点型输入张量 inputf，形状为 (200, 200)，在指定设备和数据类型上
            inputf = torch.rand(200, 200, device=device, dtype=torch.float, requires_grad=True)
            # 将 inputf 转换为指定数据类型 dtype，且不再保留梯度
            input = inputf.to(dtype).detach().requires_grad_(True)
            # 使用 PyTorch 中的 F.softmax 计算 softmax，维度为 dim
            outf = F.softmax(inputf, dim=dim)
            out = F.softmax(input, dim=dim)
            # 使用断言验证 softmax 的结果是否相等，允许一定的数值误差
            self.assertEqual(out, outf.to(dtype), atol=1e-3, rtol=0)

            # 计算输出张量的梯度和
            out.sum().backward()
            outf.sum().backward()
            # 使用断言验证输入张量的梯度是否与 inputf 的梯度在指定数据类型下相等
            self.assertEqual(input.grad, inputf.grad.to(dtype), atol=1e-3, rtol=0)

    # 根据当前环境选择数据类型，在 CUDA 上选择 torch.float 或 torch.half，在 CPU 上选择 torch.float
    @dtypesIfCUDA(torch.half, torch.float)
    @dtypes(torch.float)
    # 测试 softmax 和 log_softmax 函数的结果
    def test_softmax_results(self, device, dtype):
        # 非均匀大小和非零偏移测试向量化内核的回退路径
        # 注意: 当 dim1 > 1024 时，需要使用向量化（非持久化）路径，(16, 30576) 类似于 BERT
        sizes = [(0, 10), (32, 20), (10, 0), (31, 20), (32, 21), (31, 23), (32, 1536), (31, 2048), (33, 2049), (16, 30576)]
        shifts = [(0, 0), (1, 0), (0, 1), (1, 1)]
        
        # 对于 softmax 和 log_softmax 两种函数，以及不同的 sizes 和 shifts 进行循环测试
        for fn in [F.softmax, F.log_softmax]:
            for size in sizes:
                for shift in shifts:
                    # 创建随机数作为输入，指定设备和数据类型
                    input = torch.rand(size, device=device, dtype=dtype)
                    
                    # 注意: 对于最大的测试，我们可能会达到 fp16 的上限，因此将输入缩小以保持在较好的范围内
                    if dtype == torch.float16:
                        input = input / 100.
                    
                    # 对输入进行切片操作，并确保不通过切片操作反向传播
                    input = input[shift[0]:, shift[1]:]
                    input = input.detach().requires_grad_(True)
                    ref_input = input.clone().cpu().detach().requires_grad_(True)
                    
                    # 对每个维度进行循环
                    for dim in [0, 1]:
                        # 计算当前函数的输出和梯度
                        ref_output = fn(ref_input, dtype=torch.float, dim=dim)
                        output = fn(input, dtype=torch.float, dim=dim)
                        
                        # 创建随机梯度输出，并对其进行与输入相同的切片操作
                        grad_output = torch.rand(size, device=device, dtype=dtype)
                        grad_output = grad_output[shift[0]:, shift[1]:]
                        ref_grad_output = grad_output.clone().cpu().detach()
                        
                        # 计算输入的梯度和参考输入的梯度
                        grad_input, = torch.autograd.grad(output, input, grad_outputs=(grad_output), create_graph=True)
                        ref_grad_input, = torch.autograd.grad(ref_output, ref_input,
                                                              grad_outputs=(ref_grad_output), create_graph=True)
                        
                        # 对梯度求和并进行反向传播
                        grad_input.sum().backward()
                        ref_grad_input.sum().backward()
                        
                        # 断言当前输出与参考输出相等
                        self.assertEqual(output, ref_output)
                        # 断言当前梯度与参考梯度相等
                        self.assertEqual(grad_input, ref_grad_input)
                        # 断言输入的梯度与参考输入的梯度相等
                        self.assertEqual(input.grad, ref_input.grad)
    # 定义一个测试函数，测试 warp softmax 对 64 位索引的支持，接受设备和数据类型作为参数
    def test_warp_softmax_64bit_indexing(self, device, dtype):
        # 定义内部运行测试的函数，接受任意形状的张量作为参数
        def run_test(*shape):
            # 生成一个在 CUDA 设备上的随机张量 x，数据类型为 torch.float16，需要梯度
            x = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
            # 计算 x 的 log softmax，沿着最后一个维度，数据类型为传入的 dtype
            y = F.log_softmax(x, dim=-1, dtype=dtype)
            # 反向传播 y
            y.backward(y)
            # 将 x 转移到 CPU，并设置需要梯度计算
            with torch.no_grad():
                xx = x.cpu().requires_grad_()
            # 对 xx 进行 float 类型的 log softmax，沿着最后一个维度，转换为 dtype 类型
            yy = F.log_softmax(xx.float(), dim=-1).to(dtype)
            # 对 yy 进行反向传播
            yy.backward(yy)
            # 通过 torch.allclose 比较 y 和 yy 是否接近，使用 dtype 的容差设置
            rtol, atol = torch.testing._comparison.get_tolerances(dtype, rtol=None, atol=None)
            self.assertTrue(torch.allclose(y.cpu(), yy, rtol=rtol, atol=atol))
            # 断言 x 的梯度与 xx 的梯度是否接近，使用 torch.half 的容差设置
            rtol, _ = torch.testing._comparison.get_tolerances(torch.half, rtol=None, atol=None)
            self.assertTrue(torch.allclose(x.grad.cpu(), xx.grad, rtol=rtol, atol=1e-3))

        # 运行测试函数，传入超出内存限制的形状参数，引发异常情况
        run_test(1100000000, 2)  # Illegal memory access https://github.com/pytorch/pytorch/issues/52715
        # 运行测试函数，传入无效的配置参数形状，引发异常情况
        run_test(2200000000, 1)  # invalid configuration argument https://github.com/pytorch/pytorch/issues/52716

    # 用于 CUDA 设备的单元测试装饰器，测试 softmax 对 64 位索引的支持
    @onlyCUDA
    # 使用 torch.half 数据类型的单元测试装饰器
    @dtypes(torch.half)
    # 执行大型张量测试，限制大小为 "20GB"
    @largeTensorTest("20GB")
    # 执行大型张量测试，限制大小为 "2GB"，仅在 CPU 上执行
    @largeTensorTest("2GB", "cpu")
    # 覆盖 torch.half 数据类型的精度设置为 0.001
    @precisionOverride({torch.half: 0.001})
    # 定义测试函数，测试 softmax 对 64 位索引的支持，接受设备和数据类型作为参数
    def test_softmax_64bit_indexing(self, device, dtype):
        # 定义内部运行测试的函数，接受任意形状的张量作为参数
        def run_test(*shape):
            # 生成一个在指定设备上的全为 1 的张量 x，数据类型为传入的 dtype，需要梯度
            x = torch.ones(shape, device=device, dtype=dtype, requires_grad=True)
            # 计算 x 的 log softmax，沿着最后一个维度，数据类型为传入的 dtype
            y = F.log_softmax(x, dim=-1, dtype=dtype)
            # 反向传播 y
            y.backward(y)
            # 断言 y 的第一个元素与最后一个元素相等
            self.assertEqual(y[0], y[-1])
            # 断言 x 的梯度的第一个元素与最后一个元素的梯度相等
            self.assertEqual(x.grad[0], x.grad[-1])

        # 运行测试函数，传入特定形状参数，测试是否出现特定问题 https://github.com/pytorch/pytorch/issues/84144
        run_test(1024 * 256 + 1, 8192)

    # 使用 torch.float 数据类型的单元测试装饰器
    @dtypes(torch.float)
    # 如果在 CUDA 设备上，还使用 torch.float 和 torch.half 数据类型的单元测试装饰器
    @dtypesIfCUDA(torch.float, torch.half)
    # 定义测试函数，测试对大型输入进行 log softmax 的计算
    def test_log_softmax_big(self, device, dtype):
        # 定义辅助测试函数，接受形状作为参数
        def _test_helper(shape):
            # 生成一个在指定设备上的具有大数值且与 dtype 中小数值相差固定偏移量的随机张量 x_small
            x_small = torch.randint(100, shape, dtype=dtype, device=device)
            offset = 1.5e3 if dtype == torch.half else 1e7
            x_big = x_small + offset
            # 断言 x_small 和 x_big 的 logsoftmax 是否相等
            self.assertEqual(F.log_softmax(x_small, -1), F.log_softmax(x_big, -1))

        # 运行辅助测试函数，传入形状参数 (16, 4)
        _test_helper((16, 4))
        # 如果设备类型为 CUDA，还测试非持久 softmax 内核
        if self.device_type == 'cuda':
            _test_helper((4, 1536))
    # 定义一个测试方法，用于验证在不同版本的PyTorch中保存的LSTM模型的兼容性
    def test_save_lstm_compatibility(self, device):
        # 测试在PyTorch 1.7及更早版本保存的LSTM模型能否在新版本中正确加载
        model = nn.LSTM(2, 3)  # 创建一个输入维度为2，隐藏单元数为3的LSTM模型
        x = torch.randn(32, 5, 2)  # 生成一个形状为(32, 5, 2)的随机张量作为输入
        expected = model(x)  # 使用模型进行预测

        # 获取PyTorch 1.7 LSTM模型的状态字典。在PyTorch 1.8之前，proj_size属性不存在。
        assert model.proj_size == 0  # 断言模型的proj_size属性为0
        state_dict = model.__dict__  # 获取当前模型的状态字典
        del state_dict['proj_size']  # 删除状态字典中的proj_size键

        # 加载一个新的LSTM模型
        loaded_model = nn.LSTM(2, 3)
        loaded_model.__setstate__(state_dict)  # 使用先前保存的状态字典设置加载的模型状态
        result = loaded_model(x)  # 使用加载的模型进行预测
        self.assertEqual(result, expected)  # 断言加载模型的预测结果与原模型一致

    @onlyCUDA
    @tf32_on_and_off(0.005)
    # 定义一个测试函数，用于测试 grid_sample 函数在大尺寸数据上的行为
    def test_grid_sample_large(self, device):
        
        # 定义问题 35202，测试输入数据与坐标数据的 grid_sample 行为
        def issue_35202():
            # 创建一个随机的输入张量，形状为 (1, 1, 480, 640)，数据类型为 float，存储在指定设备上，需要梯度计算
            input_tensor = torch.rand(1, 1, 480, 640, dtype=torch.float, device=device, requires_grad=True)
            # 创建坐标张量，包含两个坐标点，数据类型为 float，存储在指定设备上
            coords = torch.tensor([[-10059144, 67680944], [67680944, 67680944]], dtype=torch.float, device=device)
            # 将坐标张量进行扩展，形状变为 (1, 1, 2, 2)，并重复一次，形状变为 (1, 1, 2, 2)
            coords = coords.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
            # 对输入张量应用 grid_sample 函数，使用给定的坐标点进行采样
            result = torch.nn.functional.grid_sample(input_tensor, coords)
            # 断言采样结果与预期的张量相等，形状为 (1, 1, 1, 1)，数据类型为 float，存储在指定设备上
            self.assertEqual(result, torch.tensor([[[[0., 0.]]]], dtype=torch.float, device=device))
            # 对采样结果进行反向传播
            result.backward(torch.ones_like(result))
            # 同步 CUDA 设备
            torch.cuda.synchronize()
        
        # 调用问题 35202 函数
        issue_35202()

        # 定义问题 24823_1，测试在不同数据类型下的 grid_sample 行为
        def issue_24823_1(dtype):
            # 创建一个张量，包含数字 27 到 1，形状为 (1, 1, 3, 3, 3)，数据类型为指定的 dtype，存储在指定设备上
            image = torch.arange(27, 0, -1, dtype=dtype, device=device).view(1, 1, 3, 3, 3)
            # 设置张量需要梯度计算
            image.requires_grad_()
            # 使用 affine_grid 函数创建一个网格张量，形状为 (1, 1, 3, 3, 3)
            grid = torch.nn.functional.affine_grid(
                torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]], dtype=dtype, device=device),
                (1, 1, 3, 3, 3))
            # 将网格张量中第 (1, 1, 1, 0) 位置的值设置为正无穷
            grid[:, 1, 1, 1, 0] = float('inf')
            # 对输入张量应用 grid_sample 函数，使用给定的网格进行采样，使用 'zeros' 填充模式
            result = torch.nn.functional.grid_sample(image, grid, padding_mode='zeros')
            # 如果数据类型为 torch.half，则覆盖公差设置为 {'atol': 0.005, 'rtol': 0}，否则使用空字典
            tol_override = {'atol': 0.005, 'rtol': 0} if dtype == torch.half else {}
            # 断言采样结果与预期的张量相等，同时考虑公差设置
            self.assertEqual(result, torch.tensor([[[[[27., 26., 25.], [24., 23., 22.], [21., 20., 19.]],
                                                     [[18., 17., 16.], [15., 0., 13.], [12., 11., 10.]],
                                                     [[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]]]]],
                                                  device=device, dtype=dtype), **tol_override)
            # 对采样结果进行反向传播
            result.backward(torch.ones_like(result))
            # 创建预期的梯度张量，形状与 image 相同，所有元素初始化为 1，将第 (0, 0, 1, 1, 1) 位置的元素设为 0
            expected_grad = torch.ones_like(image)
            expected_grad[0, 0, 1, 1, 1] = 0
            # 断言 image 的梯度与预期的梯度张量相等，公差设置为 {'atol': 0.005, 'rtol': 0}
            self.assertEqual(image.grad, expected_grad, atol=0.005, rtol=0)
        
        # 分别调用问题 24823_1 函数，使用 torch.half、torch.float 和 torch.double 作为数据类型参数
        issue_24823_1(torch.half)
        issue_24823_1(torch.float)
        issue_24823_1(torch.double)

        # 定义问题 24823_2，测试在特定参数情况下的 grid_sample 行为
        def issue_24823_2():
            # 创建一个参数张量，包含元素 [-1.0e+20, 0.0, 0.0]，数据类型为 float，存储在指定设备上
            param = torch.tensor([[[-1.0e+20, 0.0, 0.0], [0.0, -1.0e+20, 0.0]]], dtype=torch.float, device=device)
            # 创建一个全零张量作为图像，形状为 (1, 1, 4, 4)，数据类型为 float，存储在指定设备上，需要梯度计算
            img = torch.zeros((1, 1, 4, 4), dtype=torch.float, device=device, requires_grad=True)
            # 使用 affine_grid 函数根据参数张量和图像张量的大小创建网格张量
            grid = torch.nn.functional.affine_grid(param, img.size())
            # 对输入张量应用 grid_sample 函数，使用给定的网格进行采样
            result = torch.nn.functional.grid_sample(img, grid)
            # 断言采样结果与预期的全零张量相等，形状为 (1, 1, 4, 4)，存储在指定设备上，数据类型为 float
            self.assertEqual(result, torch.zeros(1, 1, 4, 4, device=device, dtype=torch.float))
            # 对采样结果进行反向传播
            result.backward(torch.ones_like(result))
            # 同步 CUDA 设备
            torch.cuda.synchronize()
        
        # 调用问题 24823_2 函数
        issue_24823_2()
    def test_grid_sample_large_index_2d(self, device, dtype):
        # 测试使用 grid_sample 进行64位索引 (gh-41656)
        # 尝试访问角点，不应该出现段错误

        # 创建包含坐标的张量，设备和数据类型由参数指定
        coords = torch.tensor([[[-1., -1.],
                                [+1., -1.]],
                               [[-1., +1.],
                                [+1., +1.]]], device=device, dtype=dtype)
        coords = coords.expand(1, 2, 2, 2)

        # 创建一个指定设备和数据类型的全零张量，用于大视图操作
        im = torch.zeros([1, 1, 32769, 65536], device=device, dtype=dtype)

        # 比较使用大步长对大视图进行采样与同一操作在连续张量上的效果
        # 创建随机张量，与大视图共享内存
        coords = torch.rand(1, 4, 4, 2, device=device, dtype=dtype)
        large_view = im[..., 127::128]  # 获取大视图的一个切片
        small_image = torch.rand_like(large_view)  # 创建与大视图形状相同的随机张量
        large_view[...] = small_image  # 将随机张量的内容复制到大视图上
        large_view.requires_grad, small_image.requires_grad = True, True

        # 断言大视图使用64位索引
        self.assertTrue(
            sum(i * s for i, s in zip(large_view.size(), large_view.stride())) >= 2 ** 31,
            msg="View must use 64-bit indexing")

        # 使用 itertools 的 product 生成各种采样模式的组合进行测试
        for mode, padding_mode, align_corners in itertools.product(
                ('nearest', 'bilinear', 'bicubic'), ('zeros', 'border', 'reflection'), (True, False)):
            # 在小视图上进行 grid_sample 操作
            a = F.grid_sample(
                small_image, coords, mode=mode,
                padding_mode=padding_mode, align_corners=align_corners)
            a.sum().backward()

            # 在大视图上进行 grid_sample 操作
            b = F.grid_sample(
                large_view, coords, mode=mode,
                padding_mode=padding_mode, align_corners=align_corners)
            b.sum().backward()

            # 断言两个采样结果相等
            self.assertEqual(a, b)

            # 断言小视图的梯度与大视图的梯度相等
            self.assertEqual(small_image.grad, large_view.grad)

            # 清零梯度，为下一轮迭代做准备
            small_image.grad.zero_()
            large_view.grad.zero_()
    # 定义一个测试方法，用于测试在三维场景下使用大索引进行 grid_sample (gh-41656)
    def test_grid_sample_large_index_3d(self, device, dtype):
        # 测试在 grid_sample 中使用64位索引是否正常工作 (gh-41656)
        # 尝试访问角落像素，不应该发生段错误
        coords = torch.full((1, 2, 2, 2, 3), 1., device=device, dtype=dtype)
        # 创建一个形状为 [1, 1, 2, 32769, 32768] 的全零张量
        im = torch.zeros([1, 1, 2, 32769, 32768], device=device, dtype=dtype)

        # 使用 grid_sample 函数对 im 进行采样，不使用角落对齐
        result = F.grid_sample(im, coords, align_corners=False)
        self.assertEqual(result, torch.zeros((1, 1, 2, 2, 2), device=device, dtype=dtype))

        # 将大步长下的采样与相同操作在连续张量上的结果进行比较
        coords = torch.rand(1, 1, 4, 4, 3, device=device, dtype=dtype)
        large_view = im[..., 127::128]  # 获取 im 的大视图
        small_image = torch.rand_like(large_view)
        large_view[...] = small_image
        small_image.requires_grad, large_view.requires_grad = True, True
        # 检查大视图是否使用了64位索引
        self.assertTrue(
            sum(i * s for i, s in zip(large_view.size(), large_view.stride())) >= 2 ** 31,
            msg="View must use 64-bit indexing")
        
        # 使用 product 函数遍历所有可能的模式、填充模式和对齐方式的组合
        for mode, padding_mode, align_corners in itertools.product(
                ('nearest', 'bilinear'), ('zeros', 'border', 'reflection'), (True, False)):
            # 对小图像进行 grid_sample 操作
            a = F.grid_sample(
                small_image, coords, mode=mode,
                padding_mode=padding_mode, align_corners=align_corners)
            a.sum().backward()

            # 对大视图进行 grid_sample 操作
            b = F.grid_sample(
                large_view, coords, mode=mode,
                padding_mode=padding_mode, align_corners=align_corners)
            b.sum().backward()

            # 断言小图像和大视图的结果应该一致
            self.assertEqual(a, b)
            # 断言小图像和大视图的梯度应该一致
            self.assertEqual(small_image.grad, large_view.grad)

            # 清空梯度，为下一次迭代做准备
            small_image.grad.zero_()
            large_view.grad.zero_()
    # 定义测试函数，测试 grid_sample 函数在 bfloat16 精度下的行为
    def test_grid_sample_bfloat16_precision(self):
        # 定义辅助函数，用于测试不同参数组合下的 grid_sample 函数
        def helper(shape_in, shape_out, align_corners):
            # 遍历三种插值模式：双线性插值、最近邻插值、双三次插值
            for mode in ('bilinear', 'nearest', 'bicubic'):
                # 如果输入形状不是四维且模式是双三次插值，则跳过本次循环
                if len(shape_in) != 4 and mode == 'bicubic':
                    continue
                # 生成随机数据张量，使用 torch.bfloat16 类型，在 CUDA 设备上
                data = torch.randn(shape_in, device='cuda', dtype=torch.bfloat16)
                # 生成随机网格张量，使用 torch.bfloat16 类型，在 CUDA 设备上
                grid = torch.rand(shape_out, device='cuda', dtype=torch.bfloat16) * 2.0 - 1.0

                # 使用 grid_sample 函数进行插值计算，半精度输出
                out_half = F.grid_sample(data, grid, mode=mode, padding_mode='zeros', align_corners=align_corners)
                # 使用 grid_sample 函数进行插值计算，双精度输出，再转为半精度
                out_double = F.grid_sample(data.double(), grid.double(), mode=mode, padding_mode='zeros',
                                           align_corners=align_corners)

                # 断言半精度输出与双精度转为半精度后的输出一致
                self.assertEqual(out_half, out_double.bfloat16(), msg=f"grid_sample with mode = {mode} doesn't match")

        # 测试不同参数下的 grid_sample 函数行为
        helper((32, 64, 16, 16), (32, 8, 8, 2), True)
        helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), True)
        helper((32, 64, 16, 16), (32, 8, 8, 2), False)
        helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), False)

    # 测试 Gumbel Softmax 函数在指定形状下的行为（形状固定）
    def _test_gumbel_softmax_st_shapes(self, device, dtype, shape, dim, count_expected):
        # 生成具有正态分布的随机 logits 张量，指定设备和数据类型
        logits = torch.randn(shape, dtype=torch.float, device=device)
        logits = logits.to(dtype)

        # 使用 Gumbel Softmax 函数生成带硬采样的 y_draw 张量
        y_draw = F.gumbel_softmax(logits, hard=True, dim=dim)

        # 断言 y_draw 张量的最小值不小于 0
        self.assertGreaterEqual(y_draw.min(), 0)
        # 断言 y_draw 张量的形状与 logits 张量的形状相同
        self.assertTrue(y_draw.shape == logits.shape)
        # 断言每个 draw 的选择数量等于 count_expected，使用指定的容差
        self.assertEqual(y_draw.sum(), count_expected, atol=torch.finfo(y_draw.dtype).eps, rtol=0)

    # 测试 Gumbel Softmax 函数在指定设备和数据类型下的直通硬采样行为
    def _test_gumbel_softmax_straight_through(self, device, dtype):
        # 定义实验次数
        num_draws = 100

        # 创建指定 logits 张量，形状为 [1, 3]，设备为指定设备
        logits = torch.tensor([[0.2, 0.8, 0.1]], device=device)
        logits = logits.reshape([1, 3])
        logits = logits.to(dtype).requires_grad_()
        probs = logits.softmax(dim=-1)

        # 创建用于统计每个 logits 的 counts 张量
        counts = torch.zeros_like(logits)
        # 执行 num_draws 次实验
        for _ in range(num_draws):
            # 使用 Gumbel Softmax 函数生成带硬采样的 y_draw 张量
            y_draw = F.gumbel_softmax(logits, hard=True)
            # 更新 counts 张量，统计每个 logits 的计数
            counts = counts + y_draw

        # 断言 counts 张量的最小值不小于 0
        self.assertGreaterEqual(y_draw.min(), 0)
        # 断言每次实验中总计数等于 num_draws，使用指定的容差
        self.assertEqual(counts.sum(), num_draws, atol=torch.finfo(counts.dtype).eps, rtol=0)

        # 检查结果是否接近预期结果
        expected = probs * num_draws
        # 计算 z 值，用于评估结果的偏差
        z = (counts - expected) / (expected * (1 - probs)).sqrt()
        # 断言 z 值的绝对值最大不超过 2.58，用于大致估计 99% 的双侧检验
        self.assertLess(z.abs().max().item(), 2.58)
    # 测试函数：测试 Gumbel Softmax 梯度计算是否正确
    def _test_gumbel_softmax_grad(self, device, dtype):
        # 创建一个形状为 (10, 10) 的全零张量，用于梯度计算，需要在指定设备上进行，并标记为需要梯度计算
        logits_soft = torch.zeros(10, 10, dtype=dtype, device=device, requires_grad=True)
        logits_hard = torch.zeros(10, 10, dtype=dtype, device=device, requires_grad=True)

        # 保存随机数生成器的状态，以确保两次调用生成相同的随机数
        seed = torch.random.get_rng_state()
        # 使用 Gumbel Softmax 函数生成软化版本的输出，不使用硬化方式
        y_soft = F.gumbel_softmax(logits_soft, hard=False)
        # 恢复随机数生成器状态
        torch.random.set_rng_state(seed)
        # 使用 Gumbel Softmax 函数生成硬化版本的输出
        y_hard = F.gumbel_softmax(logits_hard, hard=True)

        # 对软化版本的输出求和并反向传播
        y_soft.sum().backward()
        # 对硬化版本的输出求和并反向传播
        y_hard.sum().backward()

        # 定义容差，用于比较梯度是否一致
        tol = 2 * torch.finfo(dtype).eps
        # 使用断言检查两种方式计算的梯度是否一致
        self.assertEqual(logits_soft.grad, logits_hard.grad, atol=tol, rtol=0)

    # 测试函数：测试 Gumbel Softmax 在不同形状和维度下的输出
    @skipIfMps
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_gumbel_softmax(self, device, dtype):
        # 测试不同形状和维度下的 Gumbel Softmax 输出
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5], dim=0, count_expected=1)
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5], dim=-1, count_expected=1)
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5, 4], dim=1, count_expected=5)
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5, 4, 3], dim=1, count_expected=5 * 3)
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5, 4, 3], dim=-1, count_expected=5 * 4)
        # 测试直通 Gumbel Softmax 的输出
        self._test_gumbel_softmax_straight_through(device, dtype)
        # 测试梯度计算是否正确
        self._test_gumbel_softmax_grad(device, dtype)

    # 测试函数：测试 RNN 是否能正确保留变量用于多次梯度计算
    def _test_rnn_retain_variables(self, device, dtype):
        # 创建包含 LSTM、GRU 和 RNN 的循环神经网络列表
        rnns = [nn.LSTM(10, 20, num_layers=2).to(device, dtype),
                nn.GRU(10, 20, num_layers=2).to(device, dtype),
                nn.RNN(10, 20, num_layers=2).to(device, dtype)]
        # 对每个 RNN 进行测试
        for rnn in rnns:
            # 创建一个随机张量作为输入，标记为需要梯度计算
            input = torch.randn(5, 6, 10, device=device, dtype=dtype, requires_grad=True)
            # 运行 RNN 模型，并获取输出
            output = rnn(input)
            # 对输出的第一个元素求和并进行反向传播，保留计算图用于多次梯度计算
            output[0].sum().backward(retain_graph=True)
            # 复制所有参数的梯度和输入的梯度数据
            grads = [input.grad.data.clone()] + [p.grad.data.clone() for p in rnn.parameters()]
            # 进行多次循环，验证多次梯度计算结果是否一致
            for _ in range(4):
                rnn.zero_grad()
                input.grad.data.zero_()
                output[0].sum().backward(retain_graph=True)
                grads2 = [input.grad.data] + [p.grad.data for p in rnn.parameters()]
                # 使用断言检查两次梯度计算结果是否一致
                self.assertEqual(grads, grads2)

    # 测试函数：测试 RNN 在 CUDA 上是否正确保留变量用于多次梯度计算
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.double)
    def test_rnn_retain_variables(self, device, dtype):
        # 调用测试函数来测试 RNN 是否能正确保留变量
        self._test_rnn_retain_variables(device, dtype)

        # 如果设备类型是 CUDA 且支持 cuDNN
        if self.device_type == 'cuda' and self.has_cudnn():
            # 禁用 cuDNN 后再次测试 RNN 是否能正确保留变量
            with torch.backends.cudnn.flags(enabled=False):
                self._test_rnn_retain_variables(device, dtype)

    # 仅在 CUDA 设备上测试
    @onlyCUDA
    @dtypes(torch.double)
    # 检查只有一个输出梯度时未定义梯度是否会影响反向传播
    # 参见 issue #11872
    def test_lstmcell_backward_only_one_output_grad(self, device, dtype):
        # 创建一个 LSTMCell 模型，输入维度为 2，输出维度为 3，并移到指定设备，设置数据类型
        l = torch.nn.LSTMCell(2, 3).to(device).to(dtype=dtype)
        # 创建一个随机张量作为输入，形状为 (1, 2)，移到指定设备，并标记需要梯度计算
        s = torch.randn(1, 2, device=device, dtype=dtype, requires_grad=True)
        # 对 LSTMCell 模型进行两次迭代
        for i in range(2):
            # 调用 LSTMCell 模型进行前向传播，获取输出
            out = l(s)[i]
            # 对输出求和，并执行反向传播
            out.sum().backward()
            # 断言输入张量的梯度不为 None，且梯度的绝对值之和不为 0
            self.assertFalse(s.grad is None or s.grad.abs().sum().item() == 0)

    # 测试 rnn 模型的通用函数
    def _test_rnn_mod(self, mod, inp):
        # 定义一个函数，用于执行模型前向传播并扁平化输出
        def flatten_out(mod, inp):
            out = mod(inp)
            return tuple([t if isinstance(t, torch.Tensor) else tt for t in out for tt in t])
        # 使用 functools.partial 创建一个用于梯度检查的函数
        gradcheckfunc = partial(flatten_out, mod)
        # 禁用 cudnn 后，执行梯度检查，不检查批次梯度
        with torch.backends.cudnn.flags(enabled=False):
            gradcheck(gradcheckfunc, inp, check_batched_grad=False)
            gradgradcheck(gradcheckfunc, inp, check_batched_grad=False)

        # 如果输入在 CUDA 上且不是 ROCm 环境
        if inp.is_cuda and not TEST_WITH_ROCM:
            # 断言对于不支持的 CuDNN 双向传播有良好的错误消息
            # 由于问题 https://github.com/pytorch/pytorch/issues/37874，我们使用 .backward() 触发双向传播
            with torch.backends.cudnn.flags(enabled=True):
                result = gradcheckfunc(inp)
                result[0].sum().backward(create_graph=True)
                # 获取第一个参数的梯度，并断言抛出 RuntimeError，提示暂时禁用 CuDNN 后端
                grad0 = next(mod.parameters()).grad
                with self.assertRaisesRegex(RuntimeError,
                                            "please disable the CuDNN backend temporarily"):
                    grad0.sum().backward()

                # 避免在这里出现的 backward(create_graph=True) 内存泄漏
                for param in mod.parameters():
                    param.grad = None
                inp.grad = None

    # 合并到 OpInfo 中？
    @skipMeta  # LSTM cell reuses output which was resized
    @dtypes(torch.double)
    def test_LSTM_grad_and_gradgrad(self, device, dtype):
        # 定义隐藏状态大小
        hsize = 4
        # 创建一个随机张量作为输入，形状为 (1, 3, hsize)，移到指定设备，并标记需要梯度计算
        inp = torch.rand(1, 3, hsize, device=device, dtype=dtype, requires_grad=True)
        # 对于有偏置和无偏置的情况分别测试 LSTM 模型的梯度和双向梯度
        for bias in [True, False]:
            mod = torch.nn.LSTM(hsize, hsize, bias=bias).to(device).to(dtype)
            self._test_rnn_mod(mod, inp)

    @skipMeta  # GRU cell reuses output which was resized
    @dtypes(torch.double)
    def test_GRU_grad_and_gradgrad(self, device, dtype):
        # 定义隐藏状态大小
        hsize = 4
        # 创建一个随机张量作为输入，形状为 (1, 3, hsize)，移到指定设备，并标记需要梯度计算
        inp = torch.rand(1, 3, hsize, device=device, dtype=dtype, requires_grad=True)
        # 对于有偏置和无偏置的情况分别测试 GRU 模型的梯度和双向梯度
        for bias in [True, False]:
            mod = torch.nn.GRU(hsize, hsize, bias=bias).to(device).to(dtype)
            self._test_rnn_mod(mod, inp)

    @skipMeta
    @dtypes(torch.float32, torch.bfloat16)
    @onlyCPU
    # 定义一个测试方法，用于验证在不同的数据类型（dtype）下，LSTM 后向传播是否可导（即梯度可计算）
    def test_LSTM_differentiable_backward_using_oneDNN(self, dtype):
        # 定义测试批次大小、序列长度和输入维度
        batch = 10
        seq_len = 12
        input = 3
        
        # 创建一个具有指定输入、隐藏单元和层数的 LSTM 模型
        Net = nn.LSTM(input, 3, 20, batch_first=True)
        
        # 深度复制 LSTM 模型以备后续比较
        import copy
        Net_clone = copy.deepcopy(Net)
        
        # 生成随机输入数据张量 x，并克隆为 x1 和 x2，并标记允许计算梯度
        x = torch.rand(batch, seq_len, input)
        x1 = x.clone().requires_grad_(True)
        x2 = x.clone().requires_grad_(True)
        
        # 禁用 MKLDNN 加速
        torch._C._set_mkldnn_enabled(False)
        
        # 在 Net 上执行前向传播，得到输出 out1，并计算相对于 x1 的梯度 der_out1
        out1, _ = Net(x1)
        der_out1 = torch.autograd.grad(out1, x1,
                                       grad_outputs=torch.ones_like(out1),
                                       retain_graph=True,
                                       create_graph=True)[0]
        # 计算 loss1 作为 der_out1 的和，并执行反向传播
        loss1 = der_out1.sum()
        loss1.backward(retain_graph=True)
        
        # 启用 MKLDNN 加速
        torch._C._set_mkldnn_enabled(True)
        
        # 在 Net 上执行前向传播，得到输出 out2，并计算相对于 x2 的梯度 der_out2
        out2, _ = Net(x2)
        der_out2 = torch.autograd.grad(out2, x2,
                                       grad_outputs=torch.ones_like(out2),
                                       retain_graph=True,
                                       create_graph=True)[0]
        # 计算 loss2 作为 der_out2 的和，并执行反向传播
        loss2 = der_out2.sum()
        loss2.backward(retain_graph=True)
        
        # 断言两次前向传播的梯度 der_out1 和 der_out2 应该相等
        assert torch.allclose(der_out1, der_out2)
        # 断言两次反向传播得到的梯度 x1.grad 和 x2.grad 应该相等
        assert torch.allclose(x1.grad, x2.grad)

    # 在 CUDA 设备上执行的测试方法，验证 Upsample 在 1 维情况下的启动配置
    @onlyCUDA
    def test_upsamplingNearest1d_launch_config(self, device):
        # 创建一个 Upsample 模块，并生成指定设备上的随机输入张量 inp
        m = nn.Upsample(scale_factor=2)
        inp = torch.rand(2**25, 1, 1, device=device)
        # 对输入张量进行 Upsample 操作，并记录结果 out
        out = m(inp)
        # 将输入张量 inp 在 CPU 上的参考版本 inp_ref 进行 Upsample 操作，并记录结果 out_ref
        inp_ref = inp.cpu()
        out_ref = m(inp_ref)
        # 使用断言验证 GPU 和 CPU 上的 Upsample 结果应该相等
        self.assertEqual(out_ref, out)

    # 在 CUDA 设备上执行的测试方法，验证 Upsample 在 2 维情况下的启动配置
    @onlyCUDA
    def test_upsamplingNearest2d_launch_config(self, device):
        # 创建一个 Upsample 模块，并生成指定设备上的随机输入张量 inp
        m = nn.Upsample(scale_factor=2)
        inp = torch.rand(2**25, 1, 1, 1, device=device)
        # 对输入张量进行 Upsample 操作，并记录结果 out
        out = m(inp)
        # 将输入张量 inp 在 CPU 上的参考版本 inp_ref 进行 Upsample 操作，并记录结果 out_ref
        inp_ref = inp.cpu()
        out_ref = m(inp_ref)
        # 使用断言验证 GPU 和 CPU 上的 Upsample 结果应该相等
        self.assertEqual(out_ref, out)

    # 在 CUDA 设备上执行的测试方法，验证 Upsample 在 3 维情况下的启动配置
    @onlyCUDA
    @gcIfJetson
    def test_upsamplingNearest3d_launch_config(self, device):
        # 创建一个 Upsample 模块，并生成指定设备上的随机输入张量 inp
        m = nn.Upsample(scale_factor=2)
        inp = torch.rand(2**25, 1, 1, 1, 1, device=device)
        # 对输入张量进行 Upsample 操作，并记录结果 out
        out = m(inp)
        # 将输入张量 inp 在 CPU 上的参考版本 inp_ref 进行 Upsample 操作，并记录结果 out_ref
        inp_ref = inp.cpu()
        out_ref = m(inp_ref)
        # 使用断言验证 GPU 和 CPU 上的 Upsample 结果应该相等
        self.assertEqual(out_ref, out)

    # 在 CUDA 设备上执行的测试方法，预期测试 Upsample 在 2 维情况下的启动失败
    @unittest.expectedFailure
    @skipIfRocm
    @onlyCUDA
    def test_upsamplingNearest2d_launch_fail(self, device):
        # 创建一个 Upsample 模块，并生成指定设备上的大型输入张量 inp
        # 这里的输入将会导致启动的 Grid_y 维度超过 CUDA 的最大限制，预期会失败
        inp = torch.rand(1, 1, 2**15, 2**8, device=device)
        m = nn.Upsample(scale_factor=2)
        # 执行 Upsample 操作，但预期会触发启动失败的情况
        out = m(inp)

    # 在 CUDA 设备上执行的测试方法，只在 ROCm 平台下执行，验证 Upsample 在 2 维情况下的启动配置
    @onlyCUDA
    @skipCUDAIfNotRocm
    def test_upsamplingNearest2d_launch_rocm(self, device):
        # 创建一个 Upsample 模块，并生成指定设备上的随机输入张量 inp
        m = nn.Upsample(scale_factor=2)
        inp = torch.rand(1, 1, 2**15, 2**8, device=device)
        # 对输入张量进行 Upsample 操作，并记录结果 out
        out = m(inp)

    # 在 CUDA 设备上执行的测试方法，只在 CUDA CuDNN 版本大于等于 7600 时执行
    @onlyCUDA
    @skipCUDAIfCudnnVersionLessThan(7600)
    # 定义测试函数 test_CTCLoss_cudnn，接受一个设备参数 device
    def test_CTCLoss_cudnn(self, device):
        # 定义内部辅助函数 _helper，用于执行 CTC 损失函数的测试
        def _helper(zero_infinity):
            # 设置目标长度和输入长度的列表
            target_lengths = [30, 25, 20]
            input_lengths = [50, 50, 50]
            # 生成随机目标张量，包含指定长度的随机整数，数据类型为 torch.int
            targets = torch.randint(1, 15, (sum(target_lengths),), dtype=torch.int)
            # 生成随机对数概率张量 log_probs，形状为 (50, 3, 15)，数据类型为 torch.float，在第二维上进行 log_softmax 处理，并标记为需要梯度
            log_probs = torch.randn(50, 3, 15, dtype=torch.float, device=device).log_softmax(2).requires_grad_()

            # 克隆 log_probs 并标记为需要梯度
            log_probs_ref = log_probs.detach().clone().requires_grad_()

            # 启用 cudnn 后端标志，执行 CTC 损失计算和反向传播
            with torch.backends.cudnn.flags(enabled=True):
                res = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, zero_infinity=zero_infinity)
                res.backward()

            # 使用参考实现计算预期结果
            expected = ctcloss_reference(log_probs, targets.cuda(), input_lengths, target_lengths).float()

            # 禁用 cudnn 后端标志，执行第二次 CTC 损失计算和反向传播
            with torch.backends.cudnn.flags(enabled=False):
                res2 = torch.nn.functional.ctc_loss(log_probs_ref, targets.cuda().long(), input_lengths, target_lengths,
                                                    zero_infinity=zero_infinity)
                res2.backward()

            # 断言两次计算的损失结果一致
            self.assertEqual(res, expected)
            # 断言两次损失计算的结果一致
            self.assertEqual(res2, res)
            # 断言 log_probs 和 log_probs_ref 的梯度一致
            self.assertEqual(log_probs.grad, log_probs_ref.grad)

        # 分别调用 _helper 函数，测试 zero_infinity 为 True 和 False 的情况
        _helper(zero_infinity=True)
        _helper(zero_infinity=False)
    # 定义一个私有方法用于生成CTC损失的计算
    def _CTCLoss_gen_losses(self, device, input_length, vocab_size, target_length, reduction, use_module_form):
        batch_size = 1
        # 生成随机的对数概率张量，形状为(input_length, batch_size, vocab_size)，并进行log_softmax处理
        log_probs = torch.randn(input_length, batch_size, vocab_size, dtype=torch.float, device=device) \
                         .log_softmax(2).requires_grad_()
        # 生成随机的目标张量，形状为(batch_size, target_length)，值在1到vocab_size-1之间
        targets = torch.randint(low=1, high=vocab_size - 1, size=(batch_size, target_length),
                                dtype=torch.int, device=device)
        # 生成输入长度列表，每个元素为input_length，表示每个输入的长度
        input_lengths = batch_size * [input_length]
        # 生成目标长度列表，每个元素为target_length，表示每个目标的长度

        # 从log_probs中去除batch维度并克隆，同时需要梯度计算
        log_probs_no_bd = log_probs.squeeze(1).detach().clone().requires_grad_()
        # 从targets中去除batch维度并克隆
        targets_no_bd = targets.squeeze(0).detach().clone()
        # 生成一个仅包含单个元素的tensor，表示input_lengths_no_bd
        input_lengths_no_bd = torch.tensor(input_length)
        # 生成一个仅包含单个元素的tensor，表示target_lengths_no_bd
        target_lengths_no_bd = torch.tensor(target_length)

        # 创建长度为2的log_probs_refs列表，每个元素都是log_probs的克隆，并需要梯度计算
        log_probs_refs = [log_probs.detach().clone().requires_grad_() for _ in range(2)]
        # 创建长度为1的log_probs_no_bd_refs列表，每个元素都是log_probs_no_bd的克隆，并需要梯度计算
        log_probs_no_bd_refs = [log_probs_no_bd.detach().clone().requires_grad_() for _ in range(1)]

        # 检查当前设备是否支持CUDA，并且是否具有CuDNN支持
        has_cuda = torch.cuda.is_available()
        has_cudnn = has_cuda and 'cuda' in device and self.has_cudnn()
        # 如果使用CuDNN，则需要将目标张量转移到CPU上
        if has_cuda and has_cudnn:
            targets = targets.cpu()
            targets_no_bd = targets_no_bd.cpu()

        # 根据use_module_form参数选择性地创建CTC损失函数对象或函数
        ctc_loss = (
            nn.CTCLoss(reduction=reduction, zero_infinity=True)
            if use_module_form
            else partial(torch.nn.functional.ctc_loss, reduction=reduction, zero_infinity=True)
        )

        # 根据是否使用CuDNN设置相应的标志
        with torch.backends.cudnn.flags(enabled=has_cudnn):
            # 对第一个log_probs_refs计算CTC损失，形状为(T, N, C)，targets形状为(N, S)，input_lengths/target_lengths形状为(N,)
            losses.append(ctc_loss(log_probs_refs[0], targets, input_lengths, target_lengths))
            # 对第二个log_probs_refs计算CTC损失，形状为(T, N, C)，targets_no_bd形状为(S,)，input_lengths/target_lengths形状为(N,)
            losses.append(ctc_loss(log_probs_refs[1], targets_no_bd, input_lengths, target_lengths))
            # 对log_probs_no_bd_refs计算CTC损失，形状为(T, C)，targets_no_bd形状为(S,)，input_lengths_no_bd/target_lengths_no_bd形状为(1,)
            losses_no_bd.append(ctc_loss(log_probs_no_bd_refs[0], targets_no_bd,
                                         input_lengths_no_bd, target_lengths_no_bd))

            # 对所有损失进行反向传播
            for loss in losses + losses_no_bd:
                loss.backward()

        # 返回损失列表、无边界损失列表、log_probs_refs列表和log_probs_no_bd_refs列表
        return losses, losses_no_bd, log_probs_refs, log_probs_no_bd_refs

    # 定义一个私有方法，用于比较期望的列表和要比较的列表中的元素是否相等
    def _assertEqual_list(self, expected, list_to_compare, atol=None, rtol=None):
        for ele in list_to_compare:
            self.assertEqual(expected, ele, atol=atol, rtol=rtol)

    # 使用@parametrize_test装饰器标记测试方法的参数化测试，对reduction和use_module_form参数进行测试
    @parametrize_test("reduction", ['none', 'mean', 'sum'])
    @parametrize_test("use_module_form", [True, False])
    # 测试CTC损失函数在没有批量维度的情况下的行为
    def test_CTCLoss_no_batch_dim(self, device, reduction, use_module_form):
        # 输入序列的长度
        input_length = 40
        # 词汇表大小
        vocab_size = 3
        # 目标序列的长度
        target_length = 12

        # 生成损失函数所需的参数
        args = self._CTCLoss_gen_losses(device, input_length, vocab_size, target_length, reduction, use_module_form)
        # 解包参数
        losses, losses_no_bd, log_probs_refs, log_probs_no_bd_refs = args

        # 测试输出值是否相等
        self._assertEqual_list(losses[0], losses[1:], atol=1e-4, rtol=0)
        # 测试没有批量维度时的损失值是否正确
        self._assertEqual_list(losses[0].squeeze(0), losses_no_bd, atol=1e-4, rtol=0)

        # 测试梯度值是否相等
        self._assertEqual_list(log_probs_refs[0].grad, [t.grad for t in log_probs_refs[1:]], atol=1e-4, rtol=0)
        # 测试没有批量维度时的梯度值是否正确
        self._assertEqual_list(
            log_probs_refs[0].grad.squeeze(1),
            [t.grad for t in log_probs_no_bd_refs],
            atol=1e-4,
            rtol=0,
        )

        # 检查输出的形状是否正确
        # 如果 reduction 参数为 'none'，应该是 (N,)，否则应该是 ()
        self._assertEqual_list((1,) if reduction == 'none' else (), [loss.shape for loss in losses])
        # 检查没有批量维度时输出的形状是否正确，应该是 ()
        self._assertEqual_list((), [loss.shape for loss in losses_no_bd])

        # 检查梯度的形状是否正确
        # 如果有批量维度，应该是 (T, N, C)，否则应该是 (T, C)
        self._assertEqual_list((input_length, 1, vocab_size), [t.grad.shape for t in log_probs_refs])
        # 检查没有批量维度时梯度的形状是否正确，应该是 (T, C)
        self._assertEqual_list((input_length, vocab_size), [t.grad.shape for t in log_probs_no_bd_refs])

    # 创建一个有序列表的随机序列
    def _ordered_sequence(self, device, dtype):
        seqs = [torch.empty(random.randint(1, 6), device=device, dtype=dtype)
                for _ in range(5)]
        # 对序列进行填充为随机数并排序
        seqs = [s.random_(-128, 128) for s in seqs]
        ordered = sorted(seqs, key=len, reverse=True)
        return ordered

    # 创建一个包含随机填充序列的张量
    def _padded_sequence(self, device, dtype):
        # 创建有序的随机序列
        ordered = self._ordered_sequence(device, dtype)
        # 计算每个序列的长度
        lengths = [len(i) for i in ordered]
        # 对序列进行填充，并返回填充后的张量及其长度
        padded_tensor = rnn_utils.pad_sequence(ordered)
        return padded_tensor, lengths

    # 用于只在CUDA设备上执行的装饰器
    @onlyCUDA
    # 测试设备掩码功能
    def test_device_mask(self, device):
        # 针对 enforce_sorted 参数为 True 和 False 进行循环测试
        for enforce_sorted in [True, False]:
            # 创建随机填充序列及其长度列表（在CPU上）
            padded, lengths = self._padded_sequence('cpu', torch.float)
            # 打包填充后的序列
            packed = rnn_utils.pack_padded_sequence(
                padded, lengths, enforce_sorted=enforce_sorted)
            # 断言打包后的张量不在CUDA上
            self.assertFalse(packed.is_cuda)
            # 将打包后的张量移到指定的设备上
            packed = packed.to(device)
            # 断言打包后的张量在CUDA上
            self.assertTrue(packed.is_cuda)
            # 解包填充打包序列
            unpacked, _ = rnn_utils.pad_packed_sequence(packed)
            # 断言解包后的张量在CUDA上且数据类型为 torch.float
            self.assertTrue(unpacked.is_cuda)
            self.assertEqual(unpacked.dtype, torch.float)
    def test_overwrite_module_params_on_conversion_cpu_device(self, device):
        # 测试在当前默认设置下，
        # (`torch.__future__.get_overwrite_module_params_on_conversion() == False`)，
        # 将模块参数转换后，模块参数的视图不再指向与基础变量相同的存储空间。

        # 创建一个线性层模块
        m = nn.Linear(20, 10)
        # 复制模块的权重参数作为 mw
        mw = m.weight[:]
        # 将模块移动到指定的设备上
        m.to(device)
        
        with torch.no_grad():
            # 使用 `torch.no_grad()` 是因为否则会泄漏 CUDA 内存。
            # (问题记录在 https://github.com/pytorch/pytorch/issues/21875)
            mw[0][0] = 5
            # 断言修改后的权重在 CPU 设备上
            self.assertTrue(mw[0][0].device.type == "cpu")
            # 断言 _base 中的权重在 CUDA 设备上
            self.assertTrue(mw._base[0][0].device.type == "cuda")

        try:
            torch.__future__.set_overwrite_module_params_on_conversion(True)

            # 测试如果 `torch.__future__.get_overwrite_module_params_on_conversion() == True`，
            # 模块参数的视图仍然指向与基础变量相同的存储空间
            m = nn.Linear(20, 10)
            mw = m.weight[:]
            m.to(device)
            with torch.no_grad():
                mw[0][0] = 5
            # 断言两者相等
            self.assertTrue(mw[0][0] == mw._base[0][0])

            # 测试如果 `torch.__future__.get_overwrite_module_params_on_conversion() == True`，
            # `cpu_module.to("cuda")` 不会保留对 `cpu_module` 参数或梯度的先前引用。
            m = nn.Linear(20, 10)
            m.weight.grad = torch.randn(10, 20)
            # 保留权重参数的引用
            weight_ref = m.weight
            # 保留权重梯度的引用
            weight_grad_ref = m.weight.grad
            m.to(device)
            # 断言设备不相同
            self.assertNotEqual(weight_ref.device, m.weight.device)
            # 断言梯度设备不相同
            self.assertNotEqual(weight_grad_ref.device, m.weight.grad.device)
        finally:
            torch.__future__.set_overwrite_module_params_on_conversion(False)

    @onlyCUDA
    @dtypes(torch.half, torch.float)
    def test_softmax(self, device, dtype):
        # 创建一个具有指定设备和数据类型的随机张量
        input = torch.rand(32, 100, device=device, dtype=dtype, requires_grad=True)
        # 将输入张量转换为 float 类型，并保留梯度
        inputf = input.to(torch.float).detach().requires_grad_(True)
        # 计算输入张量的 softmax，并指定维度和数据类型
        out = F.softmax(input, dim=-1, dtype=torch.float)
        # 计算输入张量的 softmax，并指定维度
        outf = F.softmax(inputf, dim=-1)
        # 断言两者在位上相等
        self.assertEqual(out, outf, atol=0, rtol=0)
        # 创建一个与 outf 相同形状的随机张量
        gO = torch.empty_like(outf).uniform_()
        # 对 out 执行反向传播
        out.backward(gO)
        # 对 outf 执行反向传播
        outf.backward(gO)
        # 断言输入张量的梯度与转换后的 dtype 后的输入张量的梯度在位上相等
        self.assertEqual(input.grad, inputf.grad.to(dtype), atol=0, rtol=0)
    # 定义一个测试函数，用于检查 BatchNorm 的梯度计算
    def _test_batchnorm_grad(self, device, dtype=torch.double):
        # 定义测试数据的维度和大小
        bs, n_feat, size_feat = 4, 5, 6
        # 创建一个张量作为输入数据，需要计算梯度
        input = torch.arange(bs * n_feat * size_feat, device=device,
                             requires_grad=True, dtype=dtype).view(bs, n_feat, size_feat)
        # 创建一个张量作为权重，需要计算梯度
        weight = torch.arange(1, n_feat + 1, device=device, requires_grad=True, dtype=dtype)
        # 创建一个张量作为偏置，需要计算梯度
        bias = torch.arange(n_feat, device=device, requires_grad=True, dtype=dtype)
        # 创建一个张量作为运行时均值
        running_mean = 1 - torch.arange(n_feat, device=device, dtype=dtype)
        # 创建一个张量作为运行时方差
        running_var = 2 * torch.arange(n_feat, device=device, dtype=dtype)
        # 对于两种训练模式（训练/评估），执行梯度和梯度二阶检查
        for training in [False, True]:
            # 调用函数执行梯度和梯度二阶检查
            _assertGradAndGradgradChecks(self, F.batch_norm, (input, running_mean, running_var, weight, bias,
                                                              training, 0.1, 0.0001))

    # 测试 BatchNorm 的梯度计算，可以在不同设备上进行
    def test_batchnorm_grad(self, device):
        # 在给定设备上执行 BatchNorm 梯度测试
        self._test_batchnorm_grad(device)

        # 如果设备类型是 CUDA 且支持 cuDNN 加速
        if self.device_type == 'cuda' and self.has_cudnn():
            # 禁用 cuDNN 后，在相同设备上再次执行 BatchNorm 梯度测试
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_grad(device)

    # 只在 CUDA 设备上执行的测试，检查半精度 LayerNorm 的计算
    @onlyCUDA
    def test_layernorm_half_precision(self):
        # 定义输入张量的宽度
        width = 128
        # 创建一个随机数填充的张量作为输入，数据类型为半精度
        input = torch.rand(1, 5, width, device="cuda", dtype=torch.half) * 0.1
        # 定义 LayerNorm 的标准化形状
        normalized_shape = (width,)
        # 创建一个全为 1 的张量作为权重，数据类型为半精度
        weight = torch.ones(width, device="cuda", dtype=torch.half)
        # 创建一个全为 0 的张量作为偏置，数据类型为半精度
        bias = torch.zeros(width, device="cuda", dtype=torch.half)
        # 定义小数值 eps 作为 epsilon 参数
        eps = 1e-5

        # 使用半精度数据执行 LayerNorm，并获取输出
        output_fp16 = torch.layer_norm(input, normalized_shape, weight, bias, eps)
        # 将输入转换为单精度数据，并使用单精度权重和偏置执行 LayerNorm，然后转换为半精度输出
        output_fp32 = torch.layer_norm(input.float(), normalized_shape, weight.float(), bias.float(), eps).half()
        # 断言半精度和单精度计算的输出应该非常接近
        self.assertEqual(output_fp16, output_fp32, atol=0, rtol=0)

    # 只在 CUDA 设备上执行的测试，检查 LayerNorm 在有权重和偏置情况下的计算
    @onlyCUDA
    def test_layernorm_weight_bias(self):
        # 定义输入张量的宽度
        width = 128
        # 创建一个随机数填充的张量作为输入，数据类型为单精度
        input = torch.rand(1, 5, width, device="cuda", dtype=torch.float32) * 0.1
        # 定义 LayerNorm 的标准化形状
        normalized_shape = (width,)
        # 创建一个随机数填充的张量作为数据
        data = torch.randn(width, device="cuda", dtype=torch.float32)
        # 创建一个全为 1 的张量作为权重，数据类型为单精度
        weight = torch.ones(width, device="cuda", dtype=torch.float32)
        # 创建一个全为 0 的张量作为偏置，数据类型为单精度
        bias = torch.zeros(width, device="cuda", dtype=torch.float32)
        # 定义小数值 eps 作为 epsilon 参数
        eps = 1e-5

        # 执行 LayerNorm 操作，传入 None 作为权重，使用 data 作为偏置
        out_none_weight = torch.layer_norm(input, normalized_shape, None, data, eps)
        # 执行 LayerNorm 操作，传入 weight 作为权重，使用 data 作为偏置
        out_one_weight = torch.layer_norm(input, normalized_shape, weight, data, eps)
        # 断言两种情况下的输出应该相等
        self.assertEqual(out_none_weight, out_one_weight)

        # 执行 LayerNorm 操作，传入 data 作为权重，使用 None 作为偏置
        out_none_bias = torch.layer_norm(input, normalized_shape, data, None, eps)
        # 执行 LayerNorm 操作，传入 data 作为权重，使用 bias 作为偏置
        out_zero_bias = torch.layer_norm(input, normalized_shape, data, bias, eps)
        # 断言两种情况下的输出应该相等
        self.assertEqual(out_none_bias, out_zero_bias)

    # 测试 Hardsigmoid 函数的梯度
    def test_hardsigmoid_grad(self, device):
        # 创建一个随机数填充的张量作为输入，数据类型为双精度，减去 0.5 并乘以 10
        inputs = (torch.randn(4, 16, 16, device=device, dtype=torch.double) - 0.5) * 10
        # 设置输入张量需要计算梯度
        inputs.requires_grad = True
        # 断言 Hardsigmoid 函数的梯度计算是成功的
        self.assertTrue(gradcheck(F.hardsigmoid, (inputs,)))

    # 只在原生设备类型上执行的测试
    @onlyNativeDeviceTypes
    # 测试对 hardswish 函数的梯度是否正确
    def test_hardswish_grad(self, device):
        # 创建一个随机张量作为输入，形状为 (4, 16, 16)，并在指定设备上使用双精度浮点数
        inputs = (torch.randn(4, 16, 16, device=device, dtype=torch.double) - 0.5) * 10
        # 设置输入张量需要计算梯度
        inputs.requires_grad = True
        # 使用 gradcheck 函数验证 hardswish 函数在给定输入上的梯度是否正确
        self.assertTrue(gradcheck(F.hardswish, (inputs,)))


    # 测试批量归一化在评估模式下的行为
    def _test_batchnorm_eval(self, ndim, device, dtype, module_dtype=None):
        # 如果未指定模块数据类型，则使用与数据类型相同的数据类型
        module_dtype = module_dtype or dtype
        # 创建一个 BatchNorm1d 模块，设置为评估模式，并移动到指定设备和数据类型
        module = nn.BatchNorm1d(3).to(device, module_dtype)
        module.eval()

        # 创建一个随机张量作为输入数据，形状为 [3] * ndim，设备和数据类型与参数指定相同，同时需要计算梯度
        data = torch.rand([3] * ndim, device=device, dtype=dtype, requires_grad=True)
        # 创建一个与 data 相同形状的随机张量作为梯度
        grad = torch.rand([3] * ndim, device=device, dtype=dtype)

        # 第一次前向传播
        res1 = module(data)
        # 对第一次前向传播结果进行反向传播
        res1.backward(grad)
        # 复制第一次反向传播后数据的梯度
        grad1 = data.grad.clone()

        # 第二次前向传播
        if data.grad is not None:
            data.grad.data.zero_()

        res2 = module(data)
        # 对第二次前向传播结果进行反向传播
        res2.backward(grad)
        # 复制第二次反向传播后数据的梯度
        grad2 = data.grad.clone()
        # 断言第一次和第二次前向传播的结果相等
        self.assertEqual(res1, res2)
        # 断言第一次和第二次反向传播后数据的梯度相等
        self.assertEqual(grad1, grad2)

        # 设置 track_running_stats=False
        module = nn.BatchNorm1d(3, track_running_stats=False).to(device, module_dtype)

        # 创建一个随机张量作为输入数据，形状为 (4, 3)，设备和数据类型与参数指定相同，同时需要计算梯度
        data = torch.rand(4, 3, device=device, dtype=dtype, requires_grad=True)
        # 创建一个与 data 相同形状的随机张量作为梯度
        grad = torch.rand(4, 3, device=device, dtype=dtype)

        # 第一次前向传播
        res1 = module(data)
        # 对第一次前向传播结果进行反向传播
        res1.backward(grad)
        # 复制第一次反向传播后数据的梯度
        grad1 = data.grad.clone()

        # 设置为评估模式
        module.eval()

        # 第二次前向传播
        if data.grad is not None:
            data.grad.data.zero_()

        res2 = module(data)
        # 对第二次前向传播结果进行反向传播
        res2.backward(grad)
        # 复制第二次反向传播后数据的梯度
        grad2 = data.grad.clone()
        # 断言第一次和第二次前向传播的结果相等
        self.assertEqual(res1, res2)
        # 断言第一次和第二次反向传播后数据的梯度相等
        self.assertEqual(grad1, grad2)


    # 使用指定设备和数据类型测试批量归一化在评估模式下的行为
    @dtypes(torch.float)
    @dtypesIfCUDA(torch.float, torch.bfloat16)
    def test_batchnorm_eval(self, device, dtype):
        # 测试二维数据情况下的批量归一化评估模式行为
        self._test_batchnorm_eval(2, device, dtype)
        # 测试三维数据情况下的批量归一化评估模式行为
        self._test_batchnorm_eval(3, device, dtype)

        # 如果设备类型为 CUDA 并且支持 cuDNN
        if self.device_type == 'cuda' and self.has_cudnn():
            # 禁用 cuDNN 后，再次测试二维数据情况下的批量归一化评估模式行为
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_eval(2, device, dtype)
                # 禁用 cuDNN 后，再次测试三维数据情况下的批量归一化评估模式行为
                self._test_batchnorm_eval(3, device, dtype)


    # 仅在 CUDA 上执行，测试混合精度下批量归一化评估模式行为
    @onlyCUDA
    @dtypes(torch.bfloat16, torch.half)
    def test_batchnorm_eval_mixed(self, device, dtype):
        # 测试 bfloat16 输入与 float 模块的批量归一化评估模式行为
        self._test_batchnorm_eval(2, device, dtype, torch.float)
        self._test_batchnorm_eval(3, device, dtype, torch.float)

        # 如果设备类型为 CUDA 并且支持 cuDNN
        if self.device_type == 'cuda' and self.has_cudnn():
            # 禁用 cuDNN 后，再次测试 bfloat16 输入与 float 模块的批量归一化评估模式行为
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_eval(2, device, dtype, torch.float)
                self._test_batchnorm_eval(3, device, dtype, torch.float)
    # 定义一个测试方法，用于测试批归一化层的仿射变换
    def _test_batchnorm_affine(self, ndim, device, dtype, module_dtype=None):
        # 如果未指定模块数据类型，则使用默认的数据类型
        module_dtype = module_dtype or dtype
        # 创建一个批归一化层，不启用仿射变换，设备为指定的设备，数据类型为指定的模块数据类型
        module = nn.BatchNorm1d(3, affine=False).to(device, module_dtype)
        # 创建一个批归一化层，启用仿射变换，设备为指定的设备，数据类型为指定的模块数据类型
        module_affine = nn.BatchNorm1d(3, affine=True).to(device, module_dtype)
        # 使用 torch.no_grad() 上下文管理器，设置仿射变换的权重为全1，偏置为0
        with torch.no_grad():
            module_affine.weight.fill_(1.0)
            module_affine.bias.zero_()

        # 创建一个指定维度的随机数据张量，设备和数据类型与输入参数一致，需要计算梯度
        data = torch.rand([3] * ndim, device=device, dtype=dtype, requires_grad=True)
        # 创建一个全1的梯度张量，与 data 张量形状一致，不需要计算梯度
        grad = torch.ones_like(data, requires_grad=False)

        # 对于仿射变换后的批归一化层，计算输出
        res1 = module_affine(data)
        # 对 res1 进行反向传播，计算梯度
        res1.backward(grad)
        # 复制 data 的梯度，并清空原梯度
        grad1 = data.grad.clone()
        data.grad.zero_()

        # 对于不启用仿射变换的批归一化层，计算输出
        res2 = module(data)
        # 对 res2 进行反向传播，计算梯度
        res2.backward(grad)
        # 获取 data 的梯度
        grad2 = data.grad

        # 断言仿射变换后和未启用仿射变换的输出结果应该相等
        self.assertEqual(res1, res2)
        # 断言仿射变换后和未启用仿射变换的梯度应该相等
        self.assertEqual(grad1, grad2)

    # 使用 torch.float 数据类型装饰器，定义批归一化仿射变换的测试方法
    @dtypes(torch.float)
    # 如果是在 CUDA 设备上，则同时支持 torch.float 和 torch.bfloat16 数据类型
    @dtypesIfCUDA(torch.float, torch.bfloat16)
    def test_batchnorm_affine(self, device, dtype):
        # 分别测试二维和三维数据的批归一化仿射变换
        self._test_batchnorm_affine(2, device, dtype)
        self._test_batchnorm_affine(3, device, dtype)

        # 如果设备类型为 'cuda' 并且支持 cuDNN
        if self.device_type == 'cuda' and self.has_cudnn():
            # 使用禁用 cuDNN 的上下文管理器
            with torch.backends.cudnn.flags(enabled=False):
                # 分别测试二维和三维数据的批归一化仿射变换
                self._test_batchnorm_affine(2, device, dtype)
                self._test_batchnorm_affine(3, device, dtype)

    # 只在 CUDA 设备上测试，使用 torch.bfloat16 和 torch.half 数据类型装饰器
    @onlyCUDA
    @dtypes(torch.bfloat16, torch.half)
    def test_batchnorm_affine_mixed(self, device, dtype):
        # 初始化一个 cudnn_enabled 列表，初始值为 [False]
        cudnn_enabled = [False]
        # 如果设备类型为 'cuda' 并且支持 cuDNN
        if self.device_type == 'cuda' and self.has_cudnn():
            # TODO: 测试在 cuDNN 下失败，请参见 gh-62034
            # cudnn_enabled = [False, True]
            pass

        # 针对 bfloat16 输入和 float 模块的测试
        for enabled in cudnn_enabled:
            # 使用指定的 cudnn_enabled 值来禁用或启用 cuDNN
            with torch.backends.cudnn.flags(enabled=enabled):
                # 分别测试二维和三维数据的批归一化仿射变换，模块数据类型为 torch.float
                self._test_batchnorm_affine(2, device, dtype, torch.float)
                self._test_batchnorm_affine(3, device, dtype, torch.float)
    # 定义一个测试方法，用于验证简单平均的批量归一化操作
    def _test_batchnorm_simple_average(self, device, dtype, module_dtype=None):
        # 如果未指定模块数据类型，则默认与主数据类型一致
        module_dtype = module_dtype or dtype
        # 创建一个具有3个特征的批量归一化模块，设置动量参数为None，并移动到指定设备和数据类型
        module = nn.BatchNorm1d(3, momentum=None).to(dtype=module_dtype, device=device)
        # 创建一个全零张量，与模块数据类型和设备类型相匹配
        zeros = torch.zeros(3, dtype=module_dtype, device=device)
        # 创建一个全1张量，与模块数据类型和设备类型相匹配
        ones = torch.ones(3, dtype=module_dtype, device=device)
        # 断言当前模块的运行均值为全零张量
        self.assertEqual(module.running_mean, zeros)
        # 断言当前模块的运行方差为全1张量
        self.assertEqual(module.running_var, ones)

        # 创建两个随机数据张量，形状为(4, 3)，数据类型为dtype，设备类型为device
        data1 = torch.rand(4, 3, dtype=dtype, device=device)
        data2 = torch.rand(4, 3, dtype=dtype, device=device)

        # 第一轮操作
        res1 = module(data1)
        # 克隆当前模块的运行均值和方差
        running_mean1 = module.running_mean.clone()
        running_var1 = module.running_var.clone()
        # 断言当前模块的运行均值不再是全零张量
        self.assertNotEqual(running_mean1, zeros)
        # 断言当前模块的运行方差不再是全1张量
        self.assertNotEqual(running_var1, ones)

        # 重置模块的统计信息
        module.reset_running_stats()
        # 断言重置后当前模块的运行均值为全零张量
        self.assertEqual(module.running_mean, zeros)
        # 断言重置后当前模块的运行方差为全1张量
        self.assertEqual(module.running_var, ones)

        # 第二轮操作
        res2 = module(data2)
        # 克隆当前模块的运行均值和方差
        running_mean2 = module.running_mean.clone()
        running_var2 = module.running_var.clone()
        # 断言当前模块的运行均值不再是全零张量
        self.assertNotEqual(running_mean2, zeros)
        # 断言当前模块的运行方差不再是全1张量
        self.assertNotEqual(running_var2, ones)

        # 重置模块的统计信息
        module.reset_running_stats()
        # 断言重置后当前模块的运行均值为全零张量
        self.assertEqual(module.running_mean, zeros)
        # 断言重置后当前模块的运行方差为全1张量
        self.assertEqual(module.running_var, ones)

        # 第三轮（综合）操作
        res3 = module(data1)
        res4 = module(data2)
        # 断言第三轮的结果与第一轮一致
        self.assertEqual(res3, res1)
        # 断言第四轮的结果与第二轮一致
        self.assertEqual(res4, res2)
        # 断言当前模块的运行均值为第一轮和第二轮均值的平均值
        self.assertEqual(module.running_mean, (running_mean1 + running_mean2) / 2)
        # 断言当前模块的运行方差为第一轮和第二轮方差的平均值
        self.assertEqual(module.running_var, (running_var1 + running_var2) / 2)

    # 使用torch.float数据类型调用_test_batchnorm_simple_average方法进行测试
    @dtypes(torch.float)
    # 如果设备类型为CUDA，则使用torch.float和torch.bfloat16数据类型进行测试
    @dtypesIfCUDA(torch.float, torch.bfloat16)
    # 定义测试简单平均批量归一化的方法，针对不同设备和数据类型进行测试
    def test_batchnorm_simple_average(self, device, dtype):
        # 调用_test_batchnorm_simple_average方法进行测试
        self._test_batchnorm_simple_average(device, dtype)

        # 如果设备类型为CUDA且支持cuDNN加速
        if self.device_type == 'cuda' and self.has_cudnn():
            # 禁用cuDNN加速后，再次调用_test_batchnorm_simple_average方法进行测试
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_simple_average(device, dtype)

    # 仅针对本地设备类型进行测试
    @onlyCUDA
    # 使用torch.bfloat16和torch.half数据类型进行测试
    @dtypes(torch.bfloat16, torch.half)
    # 定义测试混合简单平均批量归一化的方法，针对不同设备和数据类型进行测试
    def test_batchnorm_simple_average_mixed(self, device, dtype):
        # 调用_test_batchnorm_simple_average方法进行测试，模块数据类型为torch.float
        self._test_batchnorm_simple_average(device, dtype, torch.float)

        # 如果设备类型为CUDA且支持cuDNN加速
        if self.device_type == 'cuda' and self.has_cudnn():
            # 禁用cuDNN加速后，再次调用_test_batchnorm_simple_average方法进行测试，模块数据类型为torch.float
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_simple_average(device, dtype, torch.float)

    # 仅针对本地设备类型进行测试
    @onlyNativeDeviceTypes
    # 使用torch.float和torch.double数据类型进行测试
    @dtypes(torch.float, torch.double)
    # 定义一个测试函数，用于测试 grid_sample 函数处理 NaN 和 Inf 的情况
    def test_grid_sample_nan_inf(self, device, dtype):
        # 创建一个全零的张量作为输入
        input = torch.zeros([1, 1, 3, 3], device=device, dtype=dtype)
        # 创建一个包含 NaN 和 Inf 的网格张量作为输入的网格
        grid = torch.tensor([[[[nan, 0], [0, inf]]]], device=device, dtype=dtype)
        
        # 遍历三种 padding_mode 参数值：reflection、border、zeros
        for padding_mode in ('reflection', 'border', 'zeros'):
            # 调用 grid_sample 函数，使用 nearest 模式进行采样
            sample = torch.nn.functional.grid_sample(input=input, grid=grid, mode='nearest',
                                                     padding_mode=padding_mode, align_corners=False)
            # 断言采样结果与预期的全零张量相等
            self.assertEqual(sample, torch.zeros([1, 1, 1, 2], device=device, dtype=dtype))

    # 定义一个测试函数，用于测试 CTC 损失函数处理空目标序列的情况
    def test_CTCLoss_empty_target(self, device):
        # 第一个测试案例：空的目标长度列表和空的目标张量
        target_lengths = [0, 0, 0]
        input_lengths = [50, 50, 50]
        targets = torch.randint(1, 15, (0,), dtype=torch.long, device=device)
        log_probs = torch.randn(50, 3, 15, dtype=torch.double, device=device).log_softmax(2)
        # 计算 CTC 损失，设置 reduction='none' 以便得到每个样本的损失值
        loss = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
        # 断言损失值非负
        self.assertTrue((loss >= 0).all().item())
        # 断言损失值等于-log_probs 沿第一维的总和（第一列）
        self.assertEqual(-log_probs.sum(0)[:, 0], loss)

        # 第二个测试案例：一个非空的目标长度列表和一个非空的目标张量
        target_lengths = [0, 9, 0]
        input_lengths = [50, 50, 50]
        targets = torch.randint(1, 15, (9,), dtype=torch.long, device=device)
        log_probs = torch.randn(50, 3, 15, dtype=torch.double, device=device).log_softmax(2)
        # 计算 CTC 损失，设置 reduction='none' 以便得到每个样本的损失值
        loss = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
        # 断言损失值非负
        self.assertTrue((loss >= 0).all().item())
        # 断言损失值等于-log_probs 沿第一维的总和，仅包括第一列和第三列的损失值
        self.assertEqual(-log_probs.sum(0)[[0, 2], 0], loss[[0, 2]])

    # 注释：这个装饰器函数用于跳过在 Linux 和 Windows 上容易出错的情况下运行的测试
    @skipCUDAIf(True, """Test is flaky on Linux and Windows, typical error message:
                          https://github.com/pytorch/pytorch/issues/34870""")
    # 定义一个测试函数，用于测试 CTC 损失函数
    def test_ctc_loss(self, device):
        # 定义批量大小
        batch_size = 64
        # 定义标签类别数
        num_labels = 101
        # 定义目标长度
        target_length = 15
        # 定义梯度检查输入大小
        gradcheck_input_size = 10

        # 定义零模式常量
        ZERO_NONE = 0
        ZERO_SOME = 1
        ZERO_ALL = 2

        # 不同测试用例的设置：输入长度、是否变化长度、零模式
        tests = [(150, False, ZERO_NONE),
                 (150, True, ZERO_NONE),
                 (50, True, ZERO_SOME),
                 (50, True, ZERO_ALL)]

        # 如果设备是 CUDA，添加额外的测试用例
        if 'cuda' in device:
            tests += [(50, False, ZERO_NONE),
                      (50, True, ZERO_NONE),
                      (150, True, ZERO_SOME),
                      (150, True, ZERO_ALL)]

        # 遍历每个测试用例
        for input_length, vary_lengths, zero_mode in tests:
            # 生成随机目标张量
            targets = torch.randint(1, num_labels, (batch_size, target_length),
                                    device=device, dtype=torch.long)
            # 生成随机输入张量 x，用于梯度检查
            x = torch.randn(gradcheck_input_size, dtype=torch.double, device=device, requires_grad=True)
            # 生成随机 tile 因子张量
            tile_factors = torch.randn(input_length * batch_size * num_labels // gradcheck_input_size + 1,
                                       device=device)
            # 生成输入长度列表
            input_lengths = [(torch.randint(input_length // 2, input_length + 1, ()).item()
                              if vary_lengths or i == 0 else input_length) for i in range(batch_size)]
            # 根据零模式生成目标长度列表
            if zero_mode == ZERO_ALL:
                target_lengths = [0 for _ in range(batch_size)]
            else:
                target_lengths = [(torch.randint(target_length // 2, target_length + 1, ()).item()
                                   if vary_lengths else target_length) for _ in range(batch_size)]
                if zero_mode == ZERO_SOME:
                    idxes = torch.randint(0, batch_size, (10,))
                    for i in idxes:
                        target_lengths[i] = 0

            # 定义 CTC 在 softmax 后的函数
            def ctc_after_softmax(x):
                # 计算完整的 x_full 张量
                x_full = ((x[:, None] * tile_factors[None, :]).view(-1)[:input_length * batch_size * num_labels]
                          .view(input_length, batch_size, num_labels))
                # 计算对数概率
                log_probs = torch.log_softmax(x_full, 2)
                # 计算 CTC 损失
                return torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)

            # 运行梯度检查函数
            gradcheck(ctc_after_softmax, [x])

    # 以下装饰器用于设置 CUDA 相关的测试环境
    @onlyCUDA
    @skipCUDAIfRocm(msg="skipped Cudnn test on ROCm")
    @skipCUDAIfCudnnVersionLessThan(7600)
    # 定义测试函数，用于测试 CTC 损失在 CUDA 上的计算
    def test_ctc_loss_cudnn(self, device):
        # 设置批量大小为 16
        batch_size = 16
        # 输入序列长度为 30
        input_length = 30
        # 类别数为 101
        num_labels = 101
        # 目标序列长度为 15
        target_length = 15
        # 生成在 1 到 num_labels 之间的随机目标张量，设备为 CUDA，数据类型为长整型
        targets = torch.randint(1, num_labels, (batch_size * target_length,),
                                device='cuda', dtype=torch.long)
        # 生成一个形状为 (input_length, batch_size, num_labels) 的随机 log_softmax 张量，设备为 CUDA，数据类型为浮点型
        log_probs = torch.log_softmax(torch.randn(input_length, batch_size, num_labels, device='cuda', dtype=torch.float), 2)
        # 设置 log_probs 张量需要梯度
        log_probs.requires_grad_()

        # 设置每个输入序列的长度为 input_length 的列表
        input_lengths = batch_size * [input_length]
        # 设置每个目标序列的长度为 target_length 的列表
        target_lengths = batch_size * [target_length]
        # 生成一个形状为 (batch_size,) 的随机梯度张量，设备为 CUDA，数据类型为浮点型
        grad_out = torch.randn(batch_size, device='cuda', dtype=torch.float)
        
        # 禁用 cuDNN 加速上下文
        with torch.backends.cudnn.flags(enabled=False):
            # 使用 native 模式计算 CTC 损失
            loss_native = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
            # 计算 native 模式下损失相对于 log_probs 的梯度
            grad_native, = torch.autograd.grad(loss_native, log_probs, grad_out)
        
        # 使用 cuDNN 加速模式计算 CTC 损失
        loss_cudnn = torch.nn.functional.ctc_loss(log_probs, targets.to('cpu', torch.int32),
                                                  input_lengths, target_lengths, reduction='none')
        # 断言损失函数中包含 "Cudnn" 字符串，验证使用了 cuDNN 加速
        self.assertTrue("Cudnn" in str(loss_cudnn.grad_fn))
        # 计算 cuDNN 加速模式下损失相对于 log_probs 的梯度
        grad_cudnn, = torch.autograd.grad(loss_cudnn, log_probs, grad_out)
        # 断言 cuDNN 加速模式下的梯度与 native 模式下的梯度在一定容差范围内相等
        self.assertEqual(grad_cudnn, grad_native, atol=1e-4, rtol=0)

    # 装饰器声明仅在 CUDA 环境下运行
    @onlyCUDA
    # 装饰器声明若 cuDNN 版本低于 8000 则跳过测试
    @skipCUDAIfCudnnVersionLessThan(8000)
    # 定义一个测试函数，用于测试 CTC 损失函数在使用 CuDNN Tensor 时的行为
    def test_ctc_loss_cudnn_tensor(self, device):
        # 设置批量大小为 16
        batch_size = 16
        # 输入序列长度为 30
        input_length = 30
        # 标签类别数为 101
        num_labels = 101
        # 目标序列长度为 15
        target_length = 15

        # 生成随机整数标签，形状为 (batch_size * target_length)，放在 GPU 上
        targets = torch.randint(1, num_labels, (batch_size * target_length,),
                                device='cuda', dtype=torch.long)

        # 生成随机的 log_softmax 概率，形状为 (input_length, batch_size, num_labels)，放在 GPU 上，并且需要梯度
        log_probs = torch.log_softmax(torch.randn(input_length, batch_size, num_labels, device='cuda', dtype=torch.float), 2)
        log_probs.requires_grad_()

        # 设置每个输入序列的长度为 input_length
        input_lengths = batch_size * [input_length]
        input_lengths = torch.linspace(start=15, end=input_length, steps=batch_size, dtype=torch.long, device='cuda')

        # 设置每个目标序列的长度为 target_length
        target_lengths = torch.tensor(batch_size * [target_length], dtype=torch.long, device='cuda')

        # 生成一个随机梯度，形状为 (batch_size)，放在 GPU 上
        grad_out = torch.randn(batch_size, device='cuda', dtype=torch.float)

        # 禁用 CuDNN 后计算原生的 CTC 损失和梯度
        with torch.backends.cudnn.flags(enabled=False):
            # 计算原生的 CTC 损失，不进行减少
            loss_native = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
            # 计算原生 CTC 损失对 log_probs 的梯度
            grad_native, = torch.autograd.grad(loss_native, log_probs, grad_out)

        # 使用 CuDNN 加速计算 CTC 损失
        loss_cudnn = torch.nn.functional.ctc_loss(log_probs,
                                                  targets.to('cuda', torch.int32),
                                                  input_lengths.to('cuda', torch.int32),
                                                  target_lengths.to('cuda', torch.int32),
                                                  reduction='none')

        # 断言在损失函数的计算图中包含 'Cudnn'
        self.assertTrue("Cudnn" in str(loss_cudnn.grad_fn))

        # 计算 CuDNN 加速的 CTC 损失对 log_probs 的梯度
        grad_cudnn, = torch.autograd.grad(loss_cudnn, log_probs, grad_out)

        # 断言 CuDNN 加速和原生计算的梯度相等，允许的绝对误差为 1e-4，相对误差为 0
        self.assertEqual(grad_cudnn, grad_native, atol=1e-4, rtol=0)

    # 根据设备和数据类型，测试批归一化层更新统计信息的行为
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    @tf32_on_and_off(0.005)
    @skipIfTorchDynamo("TorchDynamo fails here for unknown reasons")
    def _test_batchnorm_update_stats(self, device, dtype=torch.float):
        # 创建一个批归一化层，应用在指定设备上，并设置数据类型
        module = nn.BatchNorm1d(3).to(device, dtype)

        # 创建一个随机数据张量，形状为 (4, 3)，放在指定设备上，使用指定数据类型
        data = torch.rand(4, 3, device=device, dtype=dtype)

        # 训练阶段的前，保存当前运行均值、方差和追踪批次数
        old_running_mean = module.running_mean.clone()
        old_running_var = module.running_var.clone()
        old_num_batches_tracked = module.num_batches_tracked.clone()

        # 将数据传递给批归一化层，进行训练
        module(data)

        # 断言运行均值和方差已经被更新
        self.assertNotEqual(old_running_mean, module.running_mean)
        self.assertNotEqual(old_running_var, module.running_var)

        # 断言追踪批次数增加了 1
        self.assertEqual(old_num_batches_tracked + 1, module.num_batches_tracked)

        # 评估阶段的前，保存当前运行均值、方差和追踪批次数
        module.eval()
        old_running_mean = module.running_mean.clone()
        old_running_var = module.running_var.clone()
        old_num_batches_tracked = module.num_batches_tracked.clone()

        # 将数据传递给批归一化层，进行评估
        module(data)

        # 断言运行均值和方差没有变化
        self.assertEqual(old_running_mean, module.running_mean)
        self.assertEqual(old_running_var, module.running_var)

        # 断言追踪批次数没有变化
        self.assertEqual(old_num_batches_tracked, module.num_batches_tracked)
    # 测试批归一化层更新统计数据的方法，在指定设备上进行测试

    # 调用内部方法 _test_batchnorm_update_stats 对指定设备进行测试
    self._test_batchnorm_update_stats(device)

    # 如果设备类型为 'cuda' 并且支持 cuDNN 加速，则在 cuDNN flags 禁用状态下再次测试
    if self.device_type == 'cuda' and self.has_cudnn():
        with torch.backends.cudnn.flags(enabled=False):
            self._test_batchnorm_update_stats(device)

    # 使用装饰器 onlyCPU 和 dtypes(torch.bfloat16, torch.float16) 来测试 bfloat16 和 float16 数据类型的激活函数

    # 定义测试辅助函数 test_helper，用于测试给定函数在不同输入维度下的行为
    def test_helper(fn, device, inp_dims, prec=None):
        torch.manual_seed(37)

        # 将函数 fn 转换为指定的数据类型（bfloat16 或 float16）
        fn = fn.to(dtype=dtype)

        # 创建指定设备上的随机输入数据，并设置 requires_grad=True 支持梯度计算
        input = torch.randn(inp_dims, dtype=dtype, device=device, requires_grad=True)

        # 使用函数 fn 处理输入数据
        out = fn(input)

        # 创建与 out 形状相同的随机梯度数据 grad_input
        grad_input = torch.randn_like(out, dtype=dtype, device=device)

        # 计算反向传播
        out.backward(grad_input)

        # 对比使用 float32 数据类型的计算

        # 将输入数据 input 转换为 float32 类型的副本，并设置 requires_grad=True
        input2 = input.detach().clone().float().requires_grad_(True)

        # 使用 float32 类型的数据进行计算
        out2 = fn.float()(input2)

        # 创建与 out2 形状相同的 float32 类型的梯度数据 grad_input2
        grad_input2 = grad_input.detach().clone().float()

        # 计算 float32 类型数据的反向传播
        out2.backward(grad_input2)

        # 断言：验证数据类型是否符合预期
        self.assertEqual(out.dtype, dtype)
        self.assertEqual(input.grad.dtype, dtype)

        # 断言：验证 bfloat16/half 和 float32 计算结果的数值接近程度
        self.assertEqual(out, out2.to(dtype=dtype), atol=prec, rtol=prec)
        self.assertEqual(input.grad.data, input2.grad.data.to(dtype=dtype), atol=prec, rtol=prec)

    # 针对不同形状的输入进行测试
    shapes = [[1, 3, 1, 6], [1, 3, 1, 128], [1, 3, 256, 256]]
    for shape in shapes:
        # 依次对各种激活函数进行测试
        test_helper(torch.nn.LogSigmoid(), device, shape)
        test_helper(torch.nn.Hardsigmoid(), device, shape)
        test_helper(torch.nn.Hardshrink(), device, shape)
        test_helper(torch.nn.Softshrink(), device, shape)
        test_helper(torch.nn.Hardswish(), device, shape)
        test_helper(torch.nn.Softplus(), device, shape)
        test_helper(torch.nn.SiLU(), device, shape)
        test_helper(torch.nn.Hardtanh(), device, shape)
        test_helper(torch.nn.Mish(), device, shape)
        test_helper(torch.nn.ELU(), device, shape)
        test_helper(torch.nn.PReLU(), device, shape)
        test_helper(torch.nn.GLU(), device, shape, prec=1e-2)
        test_helper(torch.nn.Threshold(0.1, 20), device, shape)
        test_helper(torch.nn.GELU(), device, shape)
        test_helper(torch.nn.Hardtanh(), device, shape)
        test_helper(torch.nn.LeakyReLU(), device, shape)
    # 测试不同激活函数对 bfloat16 数据类型的操作
    def test_activations_bfloat16(self, device):
        # 测试 ReLU 激活函数在 bfloat16 下的操作
        _test_bfloat16_ops(self, torch.nn.ReLU(), device, inp_dims=(5), prec=1e-2)
        # 测试带阈值的 Threshold 激活函数在 bfloat16 下的操作
        _test_bfloat16_ops(self, torch.nn.Threshold(0.1, 20), device, inp_dims=(5), prec=1e-2)
        # 测试 ELU 激活函数在 bfloat16 下的操作
        _test_bfloat16_ops(self, torch.nn.ELU(), device, inp_dims=(5), prec=1e-2)
        # 测试 Softplus 激活函数在 bfloat16 下的操作
        _test_bfloat16_ops(self, torch.nn.Softplus(), device, inp_dims=(5), prec=1e-2)
        # 测试 Hardshrink 激活函数在 bfloat16 下的操作
        _test_bfloat16_ops(self, torch.nn.Hardshrink(), device, inp_dims=(5), prec=1e-2)
        # 测试 Softshrink 激活函数在 bfloat16 下的操作
        _test_bfloat16_ops(self, torch.nn.Softshrink(), device, inp_dims=(5), prec=1e-2)
        # 测试 LeakyReLU 激活函数在 bfloat16 下的操作
        _test_bfloat16_ops(self, torch.nn.LeakyReLU(), device, inp_dims=(5), prec=1e-2)

    # 仅适用于本机设备类型的测试函数装饰器
    @onlyNativeDeviceTypes
    # 测试 bfloat16 数据类型下的 Softmax 操作
    def test_softmax_bfloat16(self, device):
        # 对不同维度进行 Softmax 操作的测试
        for dim in [0, 1, 2, 3]:
            _test_bfloat16_ops(self, torch.nn.Softmax(dim=dim), device, inp_dims=(16, 33, 15, 16), prec=1e-2)
            # 测试输入值较大导致 exp() 溢出的情况下的 Softmax 操作
            _test_bfloat16_ops(self, torch.nn.Softmax(dim=dim), device, inp_dims=(16, 33, 15, 16), prec=0.05, scale_factor=1000.0)

    # 测试在目标尺寸不匹配时的负对数似然损失函数
    def test_nll_loss_mismatched_batch(self, device):
        # 生成设备上随机数据张量 x
        x = torch.randn((10, 3), requires_grad=True, device=device)
        # t 应该是大小为 (10,) 的张量，但这里错误地创建了大小为 (3,) 的张量
        t = torch.zeros((3,), dtype=torch.int64, device=device)
        # 使用断言检查是否引发预期的 ValueError 异常，指示批次大小不匹配
        with self.assertRaisesRegex(ValueError, 'Expected.*batch_size'):
            F.nll_loss(x, t)

    # 测试超出边界的忽略索引值在负对数似然损失函数中的影响
    def test_nll_loss_out_of_bounds_ignore_index(self, device):
        # 生成设备上随机数据张量 x
        x = torch.randn(6, 3, requires_grad=True, device=device)
        # t 包含一个超出范围的忽略索引值 (255)，会影响损失计算
        t = torch.tensor([0, 1, 255, 0, 1, 2], dtype=torch.int64, device=device)
        # 对不同的缩减方式 ('mean', 'none') 测试负对数似然损失函数
        for reduction in ['mean', 'none']:
            # 计算损失并执行反向传播
            F.nll_loss(x, t, ignore_index=255, reduction=reduction).sum().backward()

    # 测试目标维度无效时的负对数似然损失函数
    def test_nll_loss_invalid_target_dim(self, device):
        # 生成设备上随机数据张量 x
        x = torch.randn((10, 3), device=device)
        # t 误创建为无效的二维张量 (10, 2)，但负对数似然损失函数要求是一维张量
        t = torch.zeros((10, 2), dtype=torch.int64, device=device)
        # 使用断言检查是否引发预期的 RuntimeError 异常，指示需要一维目标张量
        with self.assertRaisesRegex(RuntimeError, "1D target tensor expected"):
            F.nll_loss(x, t)

    # 测试无效权重参数时的负对数似然损失函数
    def test_nll_loss_invalid_weights(self, device):
        # 生成设备上随机数据张量 x
        x = torch.randn((10, 3), device=device)
        # 生成包含随机整数的目标张量 t
        t = torch.empty(10, dtype=torch.int64, device=device).random_(0, 3)
        # 无效的权重张量列表，包括维度不匹配的情况
        invalid_weights = [
            torch.randn(4, device=device),
            torch.randn(1, 3, device=device),
        ]
        # 错误消息，指示权重张量应同时定义所有类别或不定义任何类别
        msg = "weight tensor should be defined either for all 3 classes or no classes"
        # 对每个无效权重张量进行测试，检查是否引发预期的 RuntimeError 异常
        for weight in invalid_weights:
            with self.assertRaisesRegex(RuntimeError, msg):
                F.nll_loss(x, t, weight=weight)

    # 引用：https://github.com/pytorch/pytorch/issue/85005
    # 仅适用于 CUDA 的大张量测试装饰器，测试不同缩减方式的负对数似然损失函数
    @onlyCUDA
    @largeTensorTest("120GB", "cpu")
    @largeTensorTest("45GB", "cuda")
    @parametrize_test("reduction", ("none", "mean", "sum"))
    # 测试用例，用于测试在大张量上的负对数似然损失（NLL loss）
    def test_nll_loss_large_tensor(self, device, reduction):
        # 创建一个形状为 [2^16, 2^16 + 1] 的张量
        shape = [int(2 ** 16), int(2 ** 16) + 1]

        # 生成一个在指定设备上的随机输入张量，数据类型为浮点型，需要梯度计算
        input = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
        
        # 生成一个在指定设备上的随机标签张量，数据类型为长整型
        labels = torch.randint(shape[0], (shape[0],), dtype=torch.long, device=device)

        # 计算输入张量和标签张量之间的负对数似然损失
        out = F.nll_loss(input, labels, reduction=reduction)

        # 在不计算梯度的上下文中，将输入张量移至 CPU，并转换为浮点型并要求计算梯度
        with torch.no_grad():
            input_cpu = input.cpu().float().requires_grad_()
            labels_cpu = labels.cpu()
        
        # 计算 CPU 上相同数据的负对数似然损失
        out_cpu = F.nll_loss(input_cpu, labels_cpu, reduction=reduction)
        
        # 通过比较 workaround 函数减少内存使用，而不是使用 self.assertEqual，见 issue #84944
        rtol, atol = torch.testing._comparison.get_tolerances(torch.float32, rtol=None, atol=None)
        
        # 如果 reduction 是 "sum"，则设置更宽松的相对误差和绝对误差
        if reduction == "sum":
            orig_rtol, orig_atol = rtol, atol
            rtol, atol = 7 * rtol, 3 * atol
        
        # 在不计算梯度的上下文中，确保两个张量在给定的误差范围内全部相等
        with torch.no_grad():
            self.assertTrue(torch.allclose(out.cpu(), out_cpu, rtol=rtol, atol=atol))
        
        # 如果 reduction 是 "sum"，则恢复原始的相对误差和绝对误差
        if reduction == "sum":
            rtol, atol = orig_rtol, orig_atol
        
        # 如果 reduction 不是 "none"，则计算输出张量关于输入张量的梯度
        if reduction != "none":
            out.backward()
            out_cpu.backward()
            
            # 在不计算梯度的上下文中，确保两个输入张量的梯度在给定的误差范围内全部相等
            with torch.no_grad():
                self.assertTrue(torch.allclose(input.grad.cpu(), input_cpu.grad, rtol=rtol, atol=atol))

    # 参考链接：https://github.com/pytorch/pytorch/issue/108345
    # 仅限 CUDA 测试
    @onlyCUDA
    # 执行大张量测试，分别在 "cpu" 和 "cuda" 设备上测试，张量大小为 "20GB"
    @largeTensorTest("20GB", "cpu")
    @largeTensorTest("20GB", "cuda")
    # 参数化测试，使用 "none", "mean", "sum" 三种 reduction 模式进行测试
    @parametrize_test("reduction", ("none", "mean", "sum"))
    # 测试 64 位精度下的交叉熵损失函数
    def test_cross_entropy_64bit(self, device, reduction):
        # 创建一个在指定设备上的零张量，数据类型为长整型
        labels = torch.zeros(190, 50, dtype=torch.long, device=device)
        
        # 创建一个在指定设备上的全一张量，数据类型为浮点型
        logits = torch.ones(190, 229000, 50, dtype=torch.float, device=device)
        
        # 计算 logits 和 labels 之间的交叉熵损失
        loss = torch.nn.functional.cross_entropy(logits, labels)
        
        # 在 CPU 上计算相同数据的交叉熵损失
        loss_cpu = torch.nn.functional.cross_entropy(logits.cpu(), labels.cpu())
        
        # 打印 logits、labels 和 loss 的元素数量
        print(logits.numel(), labels.numel(), loss.numel())
        
        # 确保 CPU 上计算的损失与 GPU 上计算的损失在指定的相对误差和绝对误差范围内全部相等
        self.assertTrue(torch.allclose(loss_cpu, loss.cpu(), rtol=1e-4, atol=1e-4))

    # 辅助函数，用于测试负对数似然损失函数（NLL loss）
    def _nll_loss_helper(self, input_size, reduction, expected, device):
        # 创建一个指定大小、需要梯度计算的随机输入张量，设备为指定设备
        input = torch.rand(input_size, requires_grad=True, device=device)
        
        # 获取输入张量的通道数
        num_channels = input_size[1]
        
        # 创建一个与输入张量相同大小的随机目标张量，数据类型为长整型，设备为指定设备
        target_size = (input_size[0], ) + tuple(input_size[2:])
        target = torch.randint(num_channels, target_size, device=device)

        # 计算输入张量和目标张量之间的负对数似然损失
        output = F.nll_loss(input, target, reduction=reduction)
        
        # 使用 self.assertEqual 断言输出张量的值与期望值相等，精确到数据类型
        self.assertEqual(output, expected, exact_dtype=False)

        # 计算输出张量的和，并计算关于输入张量的梯度
        output.sum().backward()
        
        # 使用 self.assertEqual 断言输入张量的梯度大小与输入张量本身大小相同
        self.assertEqual(input.grad.size(), input.size())
    # 测试空张量情况下的负对数似然损失，不进行缩减操作
    def test_nll_loss_empty_tensor_reduction_none(self, device):
        # 调用辅助函数，传入索引、缩减方式、空的张量和设备信息
        self._nll_loss_helper([0, 3], "none", torch.empty([0], device=device), device)
        self._nll_loss_helper([0, 3, 5, 7], "none", torch.empty([0, 5, 7], device=device), device)
        self._nll_loss_helper([2, 3, 0, 7], "none", torch.empty([2, 0, 7], device=device), device)
        self._nll_loss_helper([2, 3, 5, 0], "none", torch.empty([2, 5, 0], device=device), device)
        self._nll_loss_helper([2, 3, 5, 7, 0], "none", torch.empty([2, 5, 7, 0], device=device), device)

    # 测试空张量情况下的负对数似然损失，进行均值缩减操作
    def test_nll_loss_empty_tensor_reduction_mean(self, device):
        # 创建一个 NaN 的张量
        nan = torch.tensor(float('nan'), device=device)
        # 调用辅助函数，传入索引、缩减方式、NaN张量和设备信息
        self._nll_loss_helper([0, 3], "mean", nan, device)
        self._nll_loss_helper([0, 3, 5, 7], "mean", nan, device)
        self._nll_loss_helper([2, 3, 0, 7], "mean", nan, device)
        self._nll_loss_helper([2, 3, 5, 0], "mean", nan, device)
        self._nll_loss_helper([2, 3, 5, 7, 0], "mean", nan, device)

    # 测试空张量情况下的负对数似然损失，进行总和缩减操作
    def test_nll_loss_empty_tensor_reduction_sum(self, device):
        # 创建一个零张量
        zero = torch.tensor(0, device=device)
        # 调用辅助函数，传入索引、缩减方式、零张量和设备信息
        self._nll_loss_helper([0, 3], "sum", zero, device)
        self._nll_loss_helper([0, 3, 5, 7], "sum", zero, device)
        self._nll_loss_helper([2, 3, 0, 7], "sum", zero, device)
        self._nll_loss_helper([2, 3, 5, 0], "sum", zero, device)
        self._nll_loss_helper([2, 3, 5, 7, 0], "sum", zero, device)

    # 测试总权重为零的情况下的负对数似然损失
    def test_nll_loss_total_weight_is_zero(self, device):

        # 定义辅助函数，创建全为1的输入张量，目标张量全为0，权重全为0
        def helper(input_size):
            input = torch.ones(input_size, requires_grad=True, device=device)
            num_channels = input_size[1]
            target_size = (input_size[0], ) + tuple(input_size[2:])
            target = torch.zeros(target_size, dtype=torch.long, device=device)
            weight = torch.zeros([num_channels], device=device)
            # 断言不同缩减方式下的负对数似然损失的值
            self.assertEqual(F.nll_loss(input, target, weight, reduction="sum").item(), 0.)
            self.assertEqual(F.nll_loss(input, target, weight, reduction="mean").item(), float("nan"))
            self.assertEqual(F.nll_loss(input, target, weight, reduction="none"), torch.zeros(target.shape, device=device))

        # 调用辅助函数，传入不同的输入大小
        helper([2, 3])
        helper([2, 3, 5, 7])
        helper([2, 3, 5, 7, 9])
    # 定义测试函数，测试在忽略索引为0时的负对数似然损失
    def test_nll_loss_all_ignored(self, device):

        # 辅助函数，用于生成测试数据并进行验证
        def helper(input_size):
            # 创建全为1的张量作为输入数据，指定设备
            input = torch.ones(input_size, device=device)
            # 确定输入数据的通道数
            num_channels = input_size[1]
            # 构建目标张量，全为0，除了第一个维度外保持与输入相同的形状
            target_size = (input_size[0], ) + tuple(input_size[2:])
            target = torch.zeros(target_size, dtype=torch.long, device=device)
            # 验证不同损失计算方式下的预期结果
            self.assertEqual(F.nll_loss(input, target, ignore_index=0, reduction="sum").item(), 0)
            self.assertEqual(F.nll_loss(input, target, ignore_index=0, reduction="mean").item(), float("nan"))
            self.assertEqual(F.nll_loss(input, target, ignore_index=0, reduction="none"), torch.zeros(target.shape, device=device))

        # 分别使用不同的输入形状调用辅助函数进行测试
        helper([2, 3])
        helper([2, 3, 5, 7])
        helper([2, 3, 5, 7, 9])

    # 定义测试函数，验证字节类型目标张量与长整型目标张量的负对数似然损失是否一致
    def test_nll_loss_byte_target_matches_long(self, device):
        N, C = 10, 4
        # 创建随机输入张量，并声明需要梯度
        input = torch.randn(N, C, device=device, requires_grad=True)
        # 创建长整型目标张量，取值范围为[0, C)
        target = torch.empty(N, dtype=torch.long, device=device).random_(0, C)

        # 定义计算结果和梯度的函数，支持不同的减少(reduction)方式和目标数据类型
        def compute_result_and_gradient(reduction, target_dtype):
            # 分离输入张量并声明需要梯度
            input_ = input.detach()
            input_.requires_grad_()

            # 计算对数softmax，并定义NLL损失
            prob = F.log_softmax(input_, dim=-1)
            loss = nn.NLLLoss(reduction=reduction)
            # 计算损失结果
            result = loss(prob, target.to(target_dtype))
            # 对结果进行求和并反向传播梯度
            result.sum().backward()

            return result, input_.grad

        # 针对不同的减少(reduction)方式进行测试
        for reduction in ["none", "mean", "sum"]:
            # 使用长整型目标数据类型进行计算
            result_long, grad_long = compute_result_and_gradient(reduction, torch.long)
            # 使用字节类型目标数据类型进行计算
            result_byte, grad_byte = compute_result_and_gradient(reduction, torch.uint8)
            # 验证两种数据类型计算得到的结果和梯度是否一致
            self.assertEqual(result_long, result_byte)
            self.assertEqual(grad_long, grad_byte)

    # 定义测试函数，针对CUDA环境，测试二维交叉熵损失中超出边界类索引的情况
    @onlyCUDA
    @skipIfRocm
    @dtypes(torch.float16, torch.float32)
    def test_cross_entropy_loss_2d_out_of_bounds_class_index(self, device, dtype):
        # 测试问题 #117532
        # 为了避免设备端的断言影响其他测试，使用不同进程运行
        stderr = TestCase.runWithPytorchAPIUsageStderr(f"""\
#!/usr/bin/env python3

import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的函数式接口模块
from torch.testing._internal.common_utils import (run_tests, TestCase)  # 导入测试相关的函数和类

class TestThatContainsCUDAAssert(TestCase):  # 定义一个测试类，继承自TestCase类

    def test_cross_entropy_loss_2d_out_of_bounds_class_index(self):
        device = '{str(device)}'  # 设备类型，这里是一个占位符
        dtype = {str(dtype).strip("'")}  # 数据类型，去除字符串的单引号
        ignore_index = 255  # 忽略的类别索引
        b = 10  # 批量大小
        n_classes = 3  # 类别数量
        w = 768  # 图像宽度
        h = 1024  # 图像高度
        pred = torch.randn(b, n_classes, w, h, dtype=dtype, device=device)  # 预测张量，随机初始化
        labels = torch.zeros(b, w, h, dtype=torch.int64, device=device)  # 标签张量，全零初始化
        labels[5, 200, 200] = ignore_index  # 在指定位置设置忽略索引
        labels[5, 200, 200] = 254  # 设置无效的类别索引

        x = F.cross_entropy(  # 计算交叉熵损失
            pred, labels, reduction="none", ignore_index=ignore_index
        )
        torch.cuda.synchronize()  # 同步CUDA流

if __name__ == '__main__':
    run_tests()  # 运行测试
        """)
        self.assertIn('CUDA error: device-side assert triggered', stderr)  # 检查是否触发了CUDA设备端断言错误



    def test_cross_entropy_loss_prob_target_all_reductions(self, device):
        # Test with k-dimensional loss.
        for k in range(5):
            N, C = 5, 4  # 样本数量和类别数量
            other_dims = [torch.randint(2, 5, size=(1,)).item() for _ in range(k)]  # 其他维度随机生成
            input = torch.randn(N, C, *other_dims, device=device, requires_grad=True)  # 输入张量，随机初始化
            target = torch.randn(N, C, *other_dims, device=device, requires_grad=True)  # 目标张量，随机初始化
            weight = torch.randn(C, device=device).abs()  # 权重张量，随机初始化并取绝对值

            for reduction, w in product(['none', 'mean', 'sum'], [None, weight]):
                m = torch.nn.CrossEntropyLoss(weight=w, reduction=reduction)  # 创建交叉熵损失函数对象
                output = m(input, target)  # 计算损失
                output_ref = loss_reference_fns['CrossEntropyLoss'](  # 计算参考损失
                    input, target, reduction=reduction, weight=w)
                self.assertEqual(output, output_ref)  # 断言损失值相等

    def test_cross_entropy_loss_prob_target_unit_weights(self, device):
        # Test with k-dimensional loss.
        for k in range(5):
            N, C = 5, 4  # 样本数量和类别数量
            other_dims = [torch.randint(2, 5, size=(1,)).item() for _ in range(k)]  # 其他维度随机生成
            input = torch.randn(N, C, *other_dims, device=device, requires_grad=True)  # 输入张量，随机初始化
            target = torch.randn(N, C, *other_dims, device=device, requires_grad=True)  # 目标张量，随机初始化

            for reduction in ['none', 'mean', 'sum']:
                # Ensure result with unit weights is equivalent to result without weights.
                m = torch.nn.CrossEntropyLoss(reduction=reduction)  # 创建交叉熵损失函数对象
                unit_weight = torch.ones(C, device=device, dtype=target.dtype)  # 单位权重张量
                m_unit = torch.nn.CrossEntropyLoss(weight=unit_weight, reduction=reduction)  # 创建带单位权重的损失函数对象
                output = m(input, target)  # 计算原始损失
                output_unit = m_unit(input, target)  # 计算单位权重损失
                self.assertEqual(output, output_unit)  # 断言损失值相等

    @parametrize_test('reduction', ['none', 'mean', 'sum'])
    @parametrize_test('weighted', [False, True])
    # 测试函数：测试在没有批次维度的情况下，交叉熵损失函数的计算
    def test_cross_entropy_loss_prob_target_no_batch_dim(self, device, reduction, weighted):
        # 类别数目
        C = 5
        # 生成一个随机的输入张量，并对最后一个维度进行 log_softmax 操作
        input = torch.randn(C, device=device).log_softmax(dim=-1)
        # 生成一个随机的目标张量，并对最后一个维度进行 softmax 操作
        target = torch.randn(C, device=device).softmax(dim=-1)
        # 根据是否使用权重来生成权重张量
        weight = torch.randn(C, device=device) if weighted else None
        # 创建交叉熵损失函数对象
        m = nn.CrossEntropyLoss(reduction=reduction, weight=weight)
        # 计算没有批次维度的损失
        loss_no_batch = m(input, target)
        # 在输入张量的第0维度上增加批次维度，并计算对应的损失
        loss_batch = m(input.unsqueeze(0), target.unsqueeze(0))
        # 如果 reduction 参数为 'none'，则去除 loss_batch 的批次维度
        if reduction == 'none':
            loss_batch = loss_batch.squeeze(0)
        # 断言没有批次维度和有批次维度的损失应该相等
        self.assertEqual(loss_no_batch, loss_batch)

    # 测试函数：测试在指数索引目标和单位权重下，交叉熵损失函数的计算
    def test_cross_entropy_loss_index_target_unit_weights(self, device):
        # 循环测试多维度损失
        for k in range(5):
            N, C = 5, 4
            # 生成除了(N, C)之外的其他随机维度
            other_dims = [torch.randint(2, 5, size=(1,)).item() for _ in range(k)]
            # 生成随机输入张量，并标签化目标张量
            input = torch.randn(N, C, *other_dims, device=device, requires_grad=True)
            target = torch.empty(N, *other_dims, dtype=torch.long, device=device).random_(0, C)

            # 循环不同的 reduction 模式
            for reduction in ['none', 'mean', 'sum']:
                # 确保单位权重下的结果与没有权重下的结果相等
                m = torch.nn.CrossEntropyLoss(reduction=reduction)
                unit_weight = torch.ones(C, device=device, dtype=input.dtype)
                m_unit = torch.nn.CrossEntropyLoss(weight=unit_weight, reduction=reduction)
                # 计算使用当前设置的损失
                output = m(input, target)
                output_unit = m_unit(input, target)
                # 断言两种设置下的损失应该相等
                self.assertEqual(output, output_unit)
    # 定义一个测试函数，用于测试交叉熵损失函数对于使用独热编码的目标的情况
    def test_cross_entropy_loss_one_hot_target(self, device):
        # 测试多维度的损失情况
        for k in range(5):
            N, C = 5, 4
            # 生成除了N和C之外的其他维度的随机整数
            other_dims = [torch.randint(2, 5, size=(1,)).item() for _ in range(k)]
            # 生成随机输入张量，并将其设备化并要求计算梯度
            input = torch.randn(N, C, *other_dims, device=device, requires_grad=True)
            # 生成随机目标张量，以长整型存储类型，并设备化
            target = torch.empty(N, *other_dims, dtype=torch.long, device=device).random_(0, C)
            # 生成随机权重张量，设备化，并取绝对值
            weight = torch.randn(C, device=device).abs()

            # 获取目标的独热编码表示
            target_one_hot = F.one_hot(target, num_classes=C).to(input.dtype)
            # 需要将维度C放置到索引1的位置
            target_one_hot = target_one_hot.permute(0, -1, *range(1, target_one_hot.dim() - 1))

            # 对于每一种减少方式和权重选项的组合进行迭代
            for reduction, w in product(['none', 'mean', 'sum'], [None, weight]):
                # 目前跳过此用例，因为软标签和硬标签的交叉熵在应用类权重时不一致（参见问题 #61309）
                if reduction == 'mean' and weight is not None:
                    continue

                # 确保使用类索引计算的损失与使用独热类概率计算的损失一致
                m = torch.nn.CrossEntropyLoss(weight=w, reduction=reduction)
                output = m(input, target)
                output_one_hot = m(input, target_one_hot)
                # 断言两种计算方式的损失值相等
                self.assertEqual(output, output_one_hot)

    # 定义一个测试函数，用于测试交叉熵损失函数的标签平滑错误情况
    def test_cross_entropy_label_smoothing_errors(self, device):
        N, C = 3, 4
        # 定义输入参数列表，包括两种不同的输入
        input_args = [
            (torch.randn((N, C), device=device), torch.arange(0, C, device=device)),
            (torch.randn((N, C), device=device), torch.randn(N, C, device=device))
        ]
        # 对于每一个输入参数进行迭代
        for input_arg in input_args:
            # 创建具有标签平滑参数的交叉熵损失函数
            loss = nn.CrossEntropyLoss(label_smoothing=1.2)
            # 使用断言检查损失函数是否引发了预期的运行时错误
            with self.assertRaisesRegex(RuntimeError,
                                        r"label_smoothing must be between 0\.0"):
                loss(*input_arg)

    # 将默认的张量数据类型设置为双精度浮点型
    @set_default_dtype(torch.double)
    def test_cross_entropy_label_smoothing_consistent_index_target_and_probs(self, device):
        # 设置数据维度和类别数
        N, C = 10, 4
        # 定义 k 的取值范围
        ks = range(5)
        # 定义损失函数的减少方式
        reductions = ['none', 'mean', 'sum']
        # 定义标签平滑的取值范围
        label_smoothings = [0.05, 0.15]

        # 使用 product 函数迭代所有参数组合
        for k, reduction, label_smoothing in product(ks, reductions, label_smoothings):
            # 生成其他维度的随机整数列表
            other_dims = [torch.randint(2, 5, size=(1,)).item() for _ in range(k)]
            # 生成随机输入数据张量
            input = torch.randn(N, C, *other_dims, device=device, requires_grad=True)
            # 生成随机目标张量
            target = torch.empty(N, *other_dims, dtype=torch.long, device=device).random_(0, C)

            # 构建应该与标签平滑具有相同结果的目标概率
            target_proba = F.one_hot(target, num_classes=C)
            # 将 C 维度放在索引 1 处
            target_proba = target_proba.permute(0, -1, *range(1, target_proba.dim() - 1))
            # 创建目标掩码
            target_mask = (target_proba == 1)
            target_proba = target_proba.to(dtype=input.dtype)

            # 计算平滑后的目标概率
            target_proba.masked_fill_(target_mask, 1 - label_smoothing + label_smoothing / C)
            target_proba.masked_fill_(~target_mask, label_smoothing / C)

            # 使用 CrossEntropyLoss 函数计算损失
            loss = nn.CrossEntropyLoss(reduction=reduction)
            output_with_prob = loss(input, target_proba)

            # 使用带有 label_smoothing 参数的 CrossEntropyLoss 函数计算损失
            loss = nn.CrossEntropyLoss(
                reduction=reduction, label_smoothing=label_smoothing)
            output_with_index = loss(input, target)

            # 断言两种计算方式的输出应该相等
            self.assertEqual(output_with_prob, output_with_index,
                             rtol=1e-07, atol=1e-05)

    def test_cross_entropy_label_smoothing_with_probs(self, device):
        # 设置数据维度和类别数
        N, C = 10, 4
        # 定义 k 的取值范围
        ks = range(5)
        # 定义损失函数的减少方式
        reductions = ['none', 'mean', 'sum']
        # 定义标签平滑的取值范围
        label_smoothings = [0.05, 0.15]

        # 在 k 维度上进行测试
        for k, label_smoothing in product(ks, label_smoothings):
            # 生成其他维度的随机整数列表
            other_dims = [torch.randint(2, 5, size=(1,)).item() for _ in range(k)]
            # 生成随机输入数据张量
            input = torch.randn(N, C, *other_dims, device=device, requires_grad=True)
            # 生成经过 log_softmax 处理的随机目标张量
            target = F.log_softmax(torch.randn(N, C, *other_dims, device=device), dim=1)

            # 对每种减少方式进行测试
            for reduction in reductions:
                # 使用带有 label_smoothing 参数的 CrossEntropyLoss 函数计算损失
                loss = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)
                output_with_smoothing = loss(input, target)

                # 手动平滑目标
                # class_proba^ls = class_proba * (1 - label_smoothing) +
                #                  label_smoothing / n_classes
                target_with_smoothing = target * (1 - label_smoothing) + label_smoothing / C
                # 使用 CrossEntropyLoss 函数计算手动平滑后的损失
                loss = nn.CrossEntropyLoss(reduction=reduction)
                output_with_manual_smoothing = loss(input, target_with_smoothing)

                # 断言两种计算方式的输出应该相等
                self.assertEqual(output_with_smoothing, output_with_manual_smoothing)
    def test_cross_entropy_label_smoothing_weight_ignore_indices(self, device):
        # 定义减少方式的列表
        reductions = ['none', 'sum', 'mean']
        # 定义标签平滑参数的列表
        label_smoothings = [0.05, 0.15]

        # 定义权重张量
        wgt = torch.tensor([0.3, 0.6], device=device)
        # 定义输入张量1和2
        inp1 = torch.tensor([[0.3, 0.4], [1, 2]], device=device)
        inp2 = torch.tensor([[0.3, 0.6], [1, 2]], device=device)

        # 定义默认的忽略索引目标张量
        targ_default_ignore_index = torch.tensor([-100, 1], device=device)
        # 定义负忽略索引目标张量
        targ_negative_ignore_index = torch.tensor([-2, 1], device=device)
        # 定义正忽略索引目标张量
        targ_positive_ignore_index = torch.tensor([2, 1], device=device)

        # 遍历减少方式、标签平滑参数和权重的笛卡尔积
        for reduction, label_smoothing, weight in product(reductions, label_smoothings, (None, wgt)):
            # 定义一个函数来检查两个输入-目标对产生的损失是否相等
            def check_equal(loss, inp_targ_1, inp_targ_2):
                inp1, targ1 = inp_targ_1
                inp2, targ2 = inp_targ_2
                l1 = loss(inp1, targ1)
                l2 = loss(inp2, targ2)
                self.assertEqual(l1, l2)

            # 使用默认的忽略索引
            loss = nn.CrossEntropyLoss(reduction=reduction,
                                       label_smoothing=label_smoothing,
                                       weight=weight)
            check_equal(loss, (inp1, targ_default_ignore_index), (inp2, targ_default_ignore_index))
            if reduction != 'none':
                # 检查在计算均值时是否正确统计分母
                # 即完全不计入被忽略的索引
                check_equal(loss, (inp1, targ_default_ignore_index), (inp2[1:], targ_default_ignore_index[1:]))

            # 使用负的忽略索引
            loss = nn.CrossEntropyLoss(reduction=reduction,
                                       label_smoothing=label_smoothing,
                                       ignore_index=-2,
                                       weight=weight)
            check_equal(loss, (inp1, targ_negative_ignore_index), (inp2, targ_negative_ignore_index))
            if reduction != 'none':
                # 检查在计算均值时是否正确统计分母
                # 即完全不计入被忽略的索引
                check_equal(loss, (inp1, targ_negative_ignore_index), (inp2[1:], targ_negative_ignore_index[1:]))

            # 使用正的忽略索引
            loss = nn.CrossEntropyLoss(reduction=reduction,
                                       label_smoothing=label_smoothing,
                                       ignore_index=2,
                                       weight=weight)
            check_equal(loss, (inp1, targ_positive_ignore_index), (inp2, targ_positive_ignore_index))
            if reduction != 'none':
                # 检查在计算均值时是否正确统计分母
                # 即完全不计入被忽略的索引
                check_equal(loss, (inp1, targ_positive_ignore_index), (inp2[1:], targ_positive_ignore_index[1:]))

    # Ref: https://github.com/pytorch/pytorch/issues/85005
    @onlyCUDA
    @largeTensorTest("45GB", "cpu")
    @largeTensorTest("70GB", "cuda")
    @parametrize_test("reduction", ("none", "mean", "sum"))
    # 定义测试函数，用于测试在大张量上的交叉熵损失计算
    def test_cross_entropy_large_tensor(self, device, reduction):
        # 创建一个随机的 CUDA 张量 logits，形状为 (2^16, 2^16 + 1)，数据类型为 float32
        logits = torch.randn(int(2 ** 16), int(2 ** 16) + 1, dtype=torch.float32, device='cuda', requires_grad=True)
        # 创建一个全零的 CUDA 张量 labels，与 logits 形状相同，数据类型为 long
        labels = torch.zeros(logits.size(0), dtype=torch.long, device='cuda')
        # 计算 logits 和 labels 之间的交叉熵损失，指定 reduction 方法
        loss = F.cross_entropy(logits, labels, reduction=reduction)
        # 如果 reduction 不为 "none"，则进行反向传播
        if reduction != "none":
            loss.backward()

        with torch.no_grad():
            # 将 logits 转移到 CPU，并保留其梯度计算信息
            logits_cpu = logits.cpu().detach().requires_grad_()
            # 将 labels 转移到 CPU，并取消其梯度计算信息
            labels_cpu = labels.cpu().detach()
        # 计算 logits_cpu 和 labels_cpu 之间的交叉熵损失，指定 reduction 方法
        loss_cpu = F.cross_entropy(logits_cpu, labels_cpu, reduction=reduction)
        # 如果 reduction 不为 "none"，则进行反向传播
        if reduction != "none":
            loss_cpu.backward()

        # 通过 torch.allclose 方法验证 CUDA 和 CPU 上计算的损失是否接近
        # 用于减少内存使用，参见 issue #84944
        rtol, atol = torch.testing._comparison.get_tolerances(torch.float32, rtol=None, atol=None)
        self.assertTrue(torch.allclose(loss.cpu(), loss_cpu, rtol=rtol, atol=atol))
        # 如果 reduction 不为 "none"，则通过 torch.allclose 方法验证梯度是否接近
        if reduction != "none":
            self.assertTrue(torch.allclose(logits.grad.cpu(), logits_cpu.grad, rtol=rtol, atol=atol))

    # 定义测试函数，用于测试 SmoothL1 损失函数在 beta=0.0 时的反向传播
    def test_smoothl1loss_backward_zero_beta(self, device):
        # 创建一个随机的设备张量 input，并克隆其数据，用作 target
        input = torch.randn(300, 256, requires_grad=True, device=device)
        target = input.detach()

        # 计算 input 和 target 之间的 SmoothL1 损失，设置 beta=0.0，并进行反向传播
        loss = F.smooth_l1_loss(input, target, beta=0.0, reduction='sum')
        loss.backward()

        # 计算 input 的梯度的最大绝对值，并断言其小于等于 1.0
        grad_max_abs = input.grad.abs().max().item()
        self.assertLessEqual(grad_max_abs, 1.0)

    # 定义测试函数，用于测试 Softshrink 激活函数 lambda 为负数时的异常情况
    def test_softshrink_negative(self, device):
        # 创建一个随机的设备张量 input，并设置其需要梯度计算
        input = torch.randn(5, device=device, requires_grad=True)
        # 创建 Softshrink 激活函数对象 m，lambda 设置为 -1，预期会抛出 RuntimeError 异常
        m = torch.nn.Softshrink(-1)
        # 断言调用 m(input) 会抛出指定异常信息的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError,
                                    r'lambda must be greater or equal to 0, but found to be -1\.'):
            m(input)

    # 定义测试函数，用于测试 F.fold 函数在不同数据类型下的行为
    def test_fold(self, device):
        # 定义嵌套函数 test_dtype，用于测试指定数据类型下的 F.fold 行为
        def test_dtype(fn, input, dtype):
            # 克隆 input，并将其转换为指定数据类型，并设置需要梯度计算
            input = input.detach().clone().to(dtype=dtype).requires_grad_(True)
            # 克隆 input，并将其转换为 float 类型，并设置需要梯度计算
            input2 = input.detach().clone().float().requires_grad_(True)
            # 调用函数 fn 计算 out，并进行反向传播
            out = fn(input)
            out.sum().backward()
            # 再次调用函数 fn 计算 out2，并进行反向传播
            out2 = fn(input2)
            out2.sum().backward()
            # 断言 out 的数据类型为 dtype
            self.assertEqual(out.dtype, dtype)
            # 断言 input 的梯度数据类型为 dtype
            self.assertEqual(input.grad.dtype, dtype)
            # 使用 torch.allclose 方法验证 out 和 out2 在指定的误差范围内是否相等
            self.assertEqual(out, out2.to(dtype=dtype), atol=0.05, rtol=0)
            # 使用 torch.allclose 方法验证 input 的梯度和 input2 的梯度在指定的误差范围内是否相等
            self.assertEqual(input.grad, input2.grad.to(dtype=dtype))

        # 定义嵌套函数 func，用于对输入 x 执行 F.fold 操作
        def func(x):
            return F.fold(x, output_size=(4, 5), kernel_size=(2, 2))

        # 定义随机种子列表 seeds
        seeds = (44, 83, 71, 25, 999)
        # 遍历 seeds 列表进行测试
        for sd in seeds:
            torch.manual_seed(sd)
            # 创建一个随机的设备张量 x，形状为 (1, 12, 12)，数据类型为 double，并设置需要梯度计算
            x = torch.randn(1, 12, 12, device=device, requires_grad=True, dtype=torch.double)
            # 调用 gradcheck 和 gradgradcheck 函数，验证 func 对 x 的梯度计算正确性
            gradcheck(func, [x], check_forward_ad=True)
            gradgradcheck(func, [x], check_fwd_over_rev=True)
            # 如果设备为 CPU，则使用 test_dtype 测试不同数据类型下的 F.fold 行为
            if device == 'cpu':
                test_dtype(func, x, torch.bfloat16)
    # 测试 logsigmoid 函数的输出
    def test_logsigmoid_out(self, device):
        # 这个问题实际上没有文档记录，但以前有问题：
        # https://github.com/pytorch/pytorch/issues/36499
        # 创建一个形状为 (2, 3) 的随机张量 x，并转置
        x = torch.randn(2, 3, device=device).t()
        # 创建一个形状为空的随机张量 empty_out
        empty_out = torch.randn(0, device=device)
        # 检查 logsigmoid(x) 和 logsigmoid(x, out=empty_out) 是否相等
        self.assertEqual(F.logsigmoid(x), F.logsigmoid(x, out=empty_out))

        # 创建一个非连续内存的形状为 (2, 3) 的随机张量 noncontig_out
        noncontig_out = torch.randn(2, 3, device=device).t()
        # 检查 logsigmoid(x) 和 logsigmoid(x, out=noncontig_out) 是否相等
        self.assertEqual(F.logsigmoid(x), F.logsigmoid(x, out=noncontig_out))

    # 检查 clip_grad_norm_ 函数是否在参数梯度的总范数非有限时引发错误
    @onlyCUDA
    @deviceCountAtLeast(2)
    @parametrize_test('foreach', (False, True))
    def test_clip_grad_norm_multi_device(self, devices, foreach):
        # 定义一个测试用的模型 TestModel
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 10)
                self.layer2 = nn.Linear(10, 10)

        # 创建测试用的模型实例 test_model 和参考模型实例 ref_model
        test_model = TestModel()
        test_model.layer1.to(devices[0])
        test_model.layer2.to(devices[1])
        ref_model = TestModel().to(devices[0])

        # 对于给定的 norm_type 进行循环测试
        for norm_type in [2., math.inf]:
            # 将测试模型和参考模型的所有参数的梯度设置为全为1
            for p in test_model.parameters():
                p.grad = torch.ones_like(p)
            for p in ref_model.parameters():
                p.grad = torch.ones_like(p)
            # 调用 clip_grad_norm_ 函数计算梯度总范数，检查是否与预期一致
            norm = clip_grad_norm_(test_model.parameters(), 0.5, norm_type=norm_type, foreach=foreach)
            expected = clip_grad_norm_(ref_model.parameters(), 0.5, norm_type=norm_type, foreach=foreach)
            self.assertEqual(norm, expected)
            # 检查测试模型和参考模型的每个参数的梯度是否相等
            for p, pe in zip(test_model.parameters(), ref_model.parameters()):
                self.assertEqual(p.grad.to(devices[0]), pe.grad)

    # 测试 elu 函数在 inplace 模式下的重叠
    def test_elu_inplace_overlap(self, device):
        # 创建一个形状为 (1, 6) 的随机张量 x，使用 torch.bfloat16 类型，并在第二维度上进行扩展
        x = torch.randn((1, 6), dtype=torch.bfloat16, device=device).expand((6, 6))
        # 使用断言检查是否会抛出 RuntimeError，且错误信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.elu(x, inplace=True)
        # 使用断言检查是否会抛出 RuntimeError，且错误信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.elu_(x)

    # 合并到 OpInfo 中？
    @onlyNativeDeviceTypes
    def test_elu_inplace_with_neg_alpha(self, device):
        # 创建一个形状为 (2,) 的张量 a，在指定设备上，且需要梯度计算
        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        # 使用 torch.nn.functional.elu_ 函数应用 alpha=-2，创建一个张量 b
        b = torch.nn.functional.elu_(a.clone(), alpha=-2)
        # 使用断言检查是否会抛出 RuntimeError，且错误信息包含 "call out-of-place version"
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

        # 创建一个形状为 (2,) 的张量 a，在指定设备上，且需要梯度计算
        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        # 使用 torch.nn.functional.celu_ 函数应用 alpha=-2，创建一个张量 b
        b = torch.nn.functional.celu_(a.clone(), alpha=-2)
        # 使用断言检查是否会抛出 RuntimeError，且错误信息包含 "call out-of-place version"
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

    # 预期的失败元信息
    @expectedFailureMeta  # https://github.com/pytorch/pytorch/issues/54897
    # 测试在硬件混合中使用 inplace 模式是否会导致运行时错误
    def test_hardswish_inplace_overlap(self, device):
        # 创建一个形状为 (1, 6) 的张量，并在设备上扩展为 (6, 6)
        x = torch.randn((1, 6), device=device).expand((6, 6))
        # 使用断言确保调用 F.hardswish 时会抛出 RuntimeError，并且错误信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.hardswish(x, inplace=True)

    # 测试在硬件混合中使用 inplace 模式是否会导致运行时错误
    def test_silu_inplace_overlap(self, device):
        # 创建一个形状为 (1, 6) 的张量，并在设备上扩展为 (6, 6)
        x = torch.randn((1, 6), device=device).expand((6, 6))
        # 使用断言确保调用 F.silu 时会抛出 RuntimeError，并且错误信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.silu(x, inplace=True)

    # 测试在硬件混合中使用 inplace 模式是否会导致运行时错误
    @onlyNativeDeviceTypes
    def test_mish_inplace_overlap(self, device):
        # 创建一个形状为 (1, 6) 的张量，并在设备上扩展为 (6, 6)
        x = torch.randn((1, 6), device=device).expand((6, 6))
        # 使用断言确保调用 F.mish 时会抛出 RuntimeError，并且错误信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.mish(x, inplace=True)

    # 测试在软正切函数中使用 out 参数是否会导致运行时错误
    def test_softplus_inplace_overlap(self, device):
        # 创建一个形状为 (1, 6) 的张量，并在设备上扩展为 (6, 6)
        x = torch.randn((1, 6), device=device).expand((6, 6))
        # 使用断言确保调用 F.softplus 时会抛出 RuntimeError，并且错误信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.softplus(x, out=x)

    # 测试在软收缩函数中使用 out 参数是否会导致运行时错误
    def test_softshrink_inplace_overlap(self, device):
        # 创建一个形状为 (1, 6) 的张量，并在设备上扩展为 (6, 6)
        x = torch.randn((1, 6), device=device).expand((6, 6))
        # 使用断言确保调用 F.softshrink 时会抛出 RuntimeError，并且错误信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.softshrink(x, out=x)

    # 测试在泄漏修正线性单元中使用 inplace 模式和 out-of-place 方法是否会导致运行时错误
    def test_leaky_relu_inplace_overlap(self, device):
        # 创建一个形状为 (1, 6) 的张量，并在设备上扩展为 (6, 6)
        x = torch.randn((1, 6), device=device).expand((6, 6))
        # 使用断言确保调用 F.leaky_relu 时会抛出 RuntimeError，并且错误信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.leaky_relu(x, inplace=True)
        # 使用断言确保调用 F.leaky_relu_ 时会抛出 RuntimeError，并且错误信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.leaky_relu_(x)

    # 合并到 OpInfo 中？
    # 测试在泄漏修正线性单元中使用 inplace 模式和 out-of-place 方法是否会导致运行时错误
    def test_leaky_relu_inplace_with_neg_slope(self, device):
        # 创建一个张量 a，包含值 [-1., 1.]，并标记需要计算梯度
        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        # 使用负斜率 -2 调用 F.leaky_relu_ 的 out-of-place 版本，并期望抛出 RuntimeError
        b = torch.nn.functional.leaky_relu_(a.clone(), -2)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

        # 创建一个张量 a，包含值 [-1., 1.]，并标记需要计算梯度
        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        # 使用随机泄漏修正线性单元版本 rrelu_，并期望抛出 RuntimeError
        b = torch.nn.functional.rrelu_(a.clone(), -5.0, 1.0)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))
    def`
    # 测试 Leaky ReLU 激活函数在零斜率情况下的原地操作
    def test_leaky_relu_inplace_with_zero_slope(self, device):
        # 创建一个张量 a，包含值 [-2., 0., 2.]，放置在指定的设备上，并设置其需要计算梯度
        a = torch.tensor([-2., 0., 2.], device=device, requires_grad=True)
        # 对张量 a 进行 Leaky ReLU 操作，斜率为 0.0，返回结果 b，使用 inplace 操作
        b = torch.nn.functional.leaky_relu_(a.clone(), 0.0)
        # 执行反向传播，传入梯度全为 1 的张量
        b.backward(torch.ones(3, device=device))
        # 定义期望的梯度值
        expected = torch.tensor([0., 0., 1.], device=device)
        # 检查张量 a 的梯度是否等于期望值
        self.assertEqual(a.grad, expected)

        # 创建一个 bfloat16 类型的张量 a_bf16，包含值 [-2., 0., 2.]，放置在指定的设备上，并设置其需要计算梯度
        a_bf16 = torch.tensor([-2., 0., 2.], device=device, dtype=torch.bfloat16, requires_grad=True)
        # 对张量 a_bf16 进行 Leaky ReLU 操作，斜率为 0.0，返回结果 b_bf16，使用 inplace 操作
        b_bf16 = torch.nn.functional.leaky_relu_(a_bf16.clone(), 0.0)
        # 执行反向传播，传入梯度全为 1 的张量
        b_bf16.backward(torch.ones(3, device=device))
        # 定义期望的梯度值，数据类型为 bfloat16
        expected_bf16 = torch.tensor([0., 0., 1.], device=device, dtype=torch.bfloat16)
        # 检查张量 a_bf16 的梯度是否等于期望值
        self.assertEqual(a_bf16.grad, expected_bf16)

    # 仅在 CPU 上运行的测试
    @onlyCPU
    def test_softshrink(self, device):
        # 创建一个张量 x，包含二维数据，放置在指定的设备上
        x = torch.tensor([[1.21, 0.56, 0.5001, 0.4999, 1.2357, -0.4999, -0.5001, -1.154,
                           0.254, -0.24, -0.225, 0.104, 0.002, -0.001, 0.0574, 1.2344,
                           0.1748, -0.1797, -0.8125, 0.2051, -1.1328, 1.2344, -0.1562, 2.3554,
                           -0.1953, 0.0304, -0.3613, -1.3047, 1.0312, 0.1436, -0.6953, 0.5664,
                           -0.5820, -0.3301, 0.8203, 0.6133, 0.5938],
                          [-0.8203, -1.2344, -0.5234, 2.5312, -0.4551, -0.6875, -1.5547, -0.2217,
                           -0.3027, 2.6406, 1.3047, 0.2344, -1.6719, 0.2773, -1.3516, 3.4575,
                           0.4414, 0.2656, 2.1094, -1.5156, 1.2344, -0.4336, 0.6797, -3.5486,
                           0.9766, -0.4062, 1.4844, 0.7500, -1.7578, 0.7461, 1.6094, 8.5458,
                           0.3730, -0.3477, -1.0625, 0.3848, 0.0557]], device=device)
        # 定义期望的结果张量
        expected = torch.tensor([[0.71, 0.06, 0.0001, 0., 0.7357, 0., -0.0001, -0.654,
                                  0., 0., 0., 0., 0., 0., 0., 0.7344,
                                  0., 0., -0.3125, 0., -0.6328, 0.7344, 0., 1.8554,
                                  0., 0., 0., -0.8047, 0.5312, 0., -0.1953, 0.0664,
                                  -0.0820, 0.0, 0.3203, 0.1133, 0.0938],
                                 [-0.3203, -0.7344, -0.0234, 2.0312, 0.0, -0.1875, -1.0547, 0.,
                                  0.0, 2.1406, 0.8047, 0., -1.1719, 0., -0.8516, 2.9575,
                                  0., 0., 1.6094, -1.0156, 0.7344, 0., 0.1797, -3.0486,
                                  0.4766, 0., 0.9844, 0.2500, -1.2578, 0.2461, 1.1094, 8.0458,
                                  0., 0., -0.5625, 0., 0.]])
        # 创建 Softshrink 层的实例
        softshrink = torch.nn.Softshrink()
        # 对张量 x 进行 Softshrink 操作
        out = softshrink(x)
        # 检查输出张量是否与期望值相等，容差设为 atol=1e-2 和 rtol=0
        self.assertEqual(out, expected, atol=1e-2, rtol=0)

    # 测试阈值操作在原地操作时的重叠问题，验证原地阈值操作是 idempotent 的
    def test_threshold_inplace_overlap(self, device):
        # 创建一个随机张量 x，形状为 (1, 6)，扩展为 (6, 6)，放置在指定设备上
        x = torch.randn((1, 6), device=device).expand((6, 6))
        # 对张量 x 进行阈值操作，阈值为 0.5，阈值后值为 0.5，使用 inplace 操作
        F.threshold(x, 0.5, 0.5, inplace=True)
        # 进行原地阈值操作，阈值为 0.5，阈值后值为 0.5，使用 inplace 操作
        F.threshold_(x, 0.5, 0.5)

    # 仅在支持的原生设备类型上运行的测试
    @onlyNativeDeviceTypes
    def test_triplet_margin_with_distance_loss_default_parity(self, device):
        # 测试 `nn.TripletMarginWithDistanceLoss` 和 `F.triplet_margin_with_distance_loss`。
        # 检查在默认参数下与相应的非距离敏感实现的三元组边界损失（`nn.TripletMarginLoss` 和 `F.triplet_margin_loss`）的一致性。

        # 使用 itertools.product 生成所有可能的参数组合进行测试
        for extra_args in \
                itertools.product((0.5, 1, 1.5), (True, False), ('none', 'mean', 'sum')):
            kwargs = {'margin': extra_args[0], 'swap': extra_args[1], 'reduction': extra_args[2]}

            # 生成随机张量作为锚点、正例和负例
            anchor = torch.randn(5, 10, device=device, requires_grad=True, dtype=torch.double)
            positive = torch.randn(5, 10, device=device, requires_grad=True, dtype=torch.double)
            negative = torch.randn(5, 10, device=device, requires_grad=True, dtype=torch.double)

            # 测试 forward，使用函数形式计算期望值
            expected = F.triplet_margin_loss(anchor, positive, negative, **kwargs)
            actual = F.triplet_margin_with_distance_loss(anchor, positive, negative, **kwargs)
            self.assertEqual(actual, expected, rtol=1e-6, atol=1e-6)

            # 测试 forward，使用模块形式计算损失
            loss_ref = nn.TripletMarginLoss(**kwargs)
            loss_op = nn.TripletMarginWithDistanceLoss(**kwargs)
            self.assertEqual(loss_op(anchor, positive, negative),
                             loss_ref(anchor, positive, negative),
                             rtol=1e-6, atol=1e-6)

            # 测试 backward，检查梯度
            self.assertTrue(gradcheck(lambda a, p, n: F.triplet_margin_with_distance_loss(
                a, p, n, **kwargs), (anchor, positive, negative)))
            self.assertTrue(gradcheck(lambda a, p, n: loss_op(a, p, n),
                            (anchor, positive, negative)))
    # 定义一个测试函数，用于测试 `nn.TripletMarginWithDistanceLoss` 和 `F.triplet_margin_with_distance_loss` 的一致性
    def test_triplet_margin_with_distance_loss(self, device):
        # Test for parity between `nn.TripletMarginWithDistanceLoss` and
        # `F.triplet_margin_with_distance_loss`.

        # 创建一个用于计算两点间距离的对象
        pairwise_distance = nn.PairwiseDistance()

        # 定义一个计算余弦距离的函数
        def cosine_distance(x, y):
            return 1.0 - F.cosine_similarity(x, y)

        # 定义不同的距离函数组合
        distance_functions = (pairwise_distance, cosine_distance,
                              lambda x, y: 1.0 - F.cosine_similarity(x, y))

        # 定义不同的减少方式（reduction）、边界（margin）、交换标志（swap）的组合
        reductions = ('mean', 'none', 'sum')
        margins = (1.0, 1.5, 0.5)
        swaps = (True, False)

        # 对每一种距离函数、减少方式、边界、交换标志的组合进行迭代测试
        for distance_fn, reduction, margin, swap \
                in itertools.product(distance_functions, reductions, margins, swaps):
            # 创建随机张量作为锚点、正样本、负样本
            anchor = torch.randn(5, 10, device=device, requires_grad=True, dtype=torch.double)
            positive = torch.randn(5, 10, device=device, requires_grad=True, dtype=torch.double)
            negative = torch.randn(5, 10, device=device, requires_grad=True, dtype=torch.double)

            # 测试反向传播是否正确
            self.assertTrue(gradcheck(lambda a, p, n: F.triplet_margin_with_distance_loss(
                a, p, n, distance_function=distance_fn, reduction=reduction, margin=margin, swap=swap),
                (anchor, positive, negative)))
            
            # 创建 `nn.TripletMarginWithDistanceLoss` 的实例
            loss_op = nn.TripletMarginWithDistanceLoss(distance_function=distance_fn,
                                                       reduction=reduction, margin=margin, swap=swap)
            # 再次测试反向传播是否正确
            self.assertTrue(gradcheck(lambda a, p, n: loss_op(
                a, p, n), (anchor, positive, negative)))
            
            # 对 `loss_op` 进行跟踪（tracing）
            traced_loss_op = torch.jit.trace(loss_op, (anchor, positive, negative))
            # 再次测试反向传播是否正确
            self.assertTrue(gradcheck(lambda a, p, n: traced_loss_op(
                a, p, n), (anchor, positive, negative)))

            # 测试前向计算的一致性
            functional = F.triplet_margin_with_distance_loss(anchor, positive, negative,
                                                             distance_function=distance_fn,
                                                             reduction=reduction, margin=margin, swap=swap)
            modular = loss_op(anchor, positive, negative)
            traced = traced_loss_op(anchor, positive, negative)
            # 断言前向计算的结果一致性
            self.assertEqual(functional, modular, atol=1e-6, rtol=1e-6)
            self.assertEqual(traced, modular, atol=1e-6, rtol=1e-6)
    # 测试将模块转换为复杂数据类型的功能
    def test_to_complex(self, device):
        # 创建一个线性层模型，输入维度为3，输出维度为5，并移到指定设备上
        m = nn.Linear(3, 5).to(device)
        # 断言模型m和其移动到设备后返回的结果是同一个对象
        self.assertIs(m, m.to(device))
        # 将模型m的数据类型转换为复数浮点数
        m.to(torch.cfloat)
        # 断言模型m的权重数据类型已转换为复数浮点数
        self.assertIs(m.weight.dtype, torch.cfloat)
        # 将模型m的数据类型转换为双精度复数浮点数
        m.to(torch.cdouble)
        # 断言模型m的权重数据类型已转换为双精度复数浮点数
        self.assertIs(m.weight.dtype, torch.cdouble)
        # 将模型m的数据类型转换回标准浮点数
        m.to(torch.float)
        # 断言模型m的权重数据类型已转换回标准浮点数
        self.assertIs(m.weight.dtype, torch.float)
        
        # 使用警告捕获上下文，触发警告信息
        with warnings.catch_warnings(record=True) as w:
            m.to(torch.cfloat)
            # 断言警告信息是否发生
            self.assertEqual(len(w), 1)
            self.assertTrue("Complex modules are a new feature" in str(w[-1].message))

    # 装饰器标记此测试方法为跳过的元数据测试，仅对浮点数类型为float32和float64进行测试
    @skipMeta
    @dtypes(torch.float32, torch.float64)
    # 测试模块转换为空的功能
    def test_module_to_empty(self, device, dtype):
        # 定义一个自定义模块类MyModule
        class MyModule(nn.Module):
            # 模块初始化函数，定义权重参数
            def __init__(self, in_features, out_features, device=None, dtype=None):
                super().__init__()
                factory_kwargs = {"device": device, "dtype": dtype}
                # 初始化权重参数为随机张量
                self.weight = nn.Parameter(torch.randn(in_features, out_features, **factory_kwargs))

            # 前向传播函数，实现输入张量与权重的矩阵乘法
            def forward(self, x):
                return x @ self.weight

        # 测试元数据模块实例化
        input = torch.randn(5, 10, device=device, dtype=dtype)
        # 创建一个MyModule类的实例m，指定设备为'meta'，数据类型为dtype
        m = MyModule(10, 1, device='meta', dtype=dtype)
        # 将输入张量input传入模型m进行前向传播计算
        m(input)

        # 测试使用torch.nn.Module.to()时空元模块抛出异常
        with self.assertRaisesRegex(
            NotImplementedError,
            re.escape(
                "Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() "
                "instead of torch.nn.Module.to() when moving module from meta to a different "
                "device."
            ),
        ):
            # 尝试将模型m移动到指定设备，预期会抛出NotImplementedError异常
            m.to(device)

        # 测试在真实设备上材料化空元模块
        m.to_empty(device=device)
        m(input)
        # 使用torch.nn.init.kaiming_uniform_函数初始化权重参数
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(m.weight)
        m(input)

        # 测试从材料化模块创建空元模块
        m.to_empty(device='meta')
        m(input)
    # 定义一个测试类的方法，测试模块在不递归的情况下将其子模块置为空
    def test_module_to_empty_non_recursive(self, device):
        # 定义一个简单的神经网络层模块
        class Layer(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                # 随机初始化权重参数并将其设为可学习的参数
                self.weight = nn.Parameter(torch.randn(in_features, out_features))
                # 注册一个缓冲区，并用随机数填充
                self.register_buffer('buf', torch.randn(out_features))

            def forward(self, x):
                # 前向传播函数，返回输入 x 与权重相乘后加上缓冲区的结果
                return x @ self.weight + self.buf

        # 定义一个包含神经网络层的主模块
        class MyModule(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                # 随机初始化权重参数并将其设为可学习的参数
                self.weight = nn.Parameter(torch.randn(in_features, out_features))
                # 注册一个缓冲区，并用随机数填充
                self.register_buffer('buf1', torch.randn(out_features))
                # 创建一个 Layer 类的实例作为子模块
                self.layer = Layer(out_features, out_features)

            def forward(self, x):
                # 主模块的前向传播函数，调用子模块的前向传播函数
                return self.layer(x @ self.weight + self.buf1)

        # 将当前环境设置为 'meta' 设备
        with torch.device('meta'):
            # 创建 MyModule 的实例 m，输入参数为 3 和 5
            m = MyModule(3, 5)

        # 将模块 m 上的参数置为空，不进行递归操作
        m.to_empty(device=device, recurse=False)

        # 断言：父模块的参数和缓冲区已经在指定设备上
        self.assertTrue(not m.weight.is_meta)
        self.assertTrue(not m.buf1.is_meta)

        # 断言：子模块的参数和缓冲区仍然在 'meta' 设备上
        for p in (*m.layer.parameters(), *m.layer.buffers()):
            self.assertTrue(p.is_meta)

    # 装饰器函数，跳过使用 meta 设备初始化的测试函数
    @skipMeta
    def test_skip_init(self, device):
        # 设置随机数种子为 1，创建一个已初始化的线性模块
        torch.manual_seed(1)
        m_initialized = torch.nn.Linear(5, 1)
        # 将已初始化的模块移动到指定设备上
        m_initialized.to(device)

        # 设置随机数种子为 1，使用 skip_init 函数创建一个未初始化的线性模块
        torch.manual_seed(1)
        m_uninitialized = torch.nn.utils.skip_init(torch.nn.Linear, 5, 1, device=device)

        # 断言：已初始化和未初始化模块的权重参数应该在相同的设备上
        self.assertEqual(m_initialized.weight.device, m_uninitialized.weight.device)
        # 断言：已初始化和未初始化模块的权重参数不应该接近
        self.assertFalse(torch.allclose(m_initialized.weight, m_uninitialized.weight))

    # 装饰器函数，指定仅在 CPU 上运行的测试函数
    @dtypes(torch.float)
    @dtypesIfCUDA(torch.double, torch.float, torch.half)
    @onlyCPU
    @dtypes(torch.double)
    # 定义一个测试方法，用于测试 TransformerEncoderLayer 的快速路径
    def test_transformerencoderlayer_fast_path(self, device, dtype):
        """
        Test transformer fast path on CPU with different valid mask types and shapes
        """
        # 定义模型的维度、注意力头数、批量大小和源序列长度
        d_model = 512
        nhead = 8
        batch_size = 32
        src_len = 10

        # 创建 TransformerEncoderLayer 模型实例，并设为评估模式
        model = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True,
                                                 device=device, dtype=dtype, dropout=0)
        model.eval()

        # 创建一个随机源序列张量，形状为 (batch_size, src_len, 512)
        src = torch.rand(batch_size, src_len, 512, dtype=dtype)

        # 创建形状为 (src_len, src_len) 的注意力掩码张量，初始化为全零
        src_mask = torch.zeros(src_len, src_len).to(torch.bool)
        with torch.no_grad():
            model(src, src_mask=src_mask)

        # 创建形状为 (batch_size, src_len) 的填充掩码张量，初始化为全零
        src_key_padding_mask = torch.zeros(batch_size, src_len).to(torch.bool)
        with torch.no_grad():
            model(src, src_key_padding_mask=src_key_padding_mask)

        # 同时提供两种掩码
        with torch.no_grad():
            model(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)


    @dtypes(torch.float)
    @dtypesIfCUDA(torch.half, torch.float)
    @parametrize_test('foreach', (False, True))
    # 定义一个测试方法，用于测试 clip_grad_value 函数的功能
    def test_clip_grad_value(self, foreach, device):
        if torch.device(device).type == 'xla' and foreach:
            # 如果设备是 'xla' 并且 foreach 为 True，则跳过测试
            raise SkipTest('foreach not supported on XLA')

        # 创建一个在指定设备上的线性层实例
        l = nn.Linear(10, 10).to(device)
        clip_value = 2.5

        # 创建梯度张量 grad_w 和 grad_b
        grad_w, grad_b = torch.arange(-50., 50, device=device).view(10, 10).div_(5), torch.ones(10, device=device).mul_(2)
        for grad_list in [[grad_w, grad_b], [grad_w, None]]:
            # 将梯度分别应用到线性层的参数上
            for p, g in zip(l.parameters(), grad_list):
                p._grad = g.clone().view_as(p.data) if g is not None else g

            # 对线性层参数应用 clip_grad_value_ 函数
            clip_grad_value_(l.parameters(), clip_value, foreach=foreach)
            # 断言每个参数的梯度值都在指定的剪切范围内
            for p in filter(lambda p: p.grad is not None, l.parameters()):
                self.assertLessEqual(p.grad.data.max(), clip_value)
                self.assertGreaterEqual(p.grad.data.min(), -clip_value)

        # 测试函数应接受单个张量作为输入
        p1, p2 = torch.randn(10, 10, device=device), torch.randn(10, 10, device=device)
        g = torch.arange(-50., 50, device=device).view(10, 10).div_(5)
        p1._grad = g.clone()
        p2._grad = g.clone()
        # 对 p1 和 [p2] 分别应用 clip_grad_value_ 函数
        clip_grad_value_(p1, clip_value, foreach=foreach)
        clip_grad_value_([p2], clip_value, foreach=foreach)
        # 断言 p1 和 p2 的梯度值相等
        self.assertEqual(p1.grad, p2.grad)


    @parametrize_test('foreach', (False, True))
    @parametrize_test('norm_type', (0.5, 1.5, 2, 4, 'inf'))
    # 定义测试函数，用于测试梯度裁剪的效果
    def test_clip_grad_norm(self, norm_type, foreach, device):
        # 如果设备是 'xla' 并且 foreach 参数为真，则抛出跳过测试的异常
        if torch.device(device).type == 'xla' and foreach:
            raise SkipTest('foreach not supported on XLA')

        # 创建一个在指定设备上的线性层对象
        l = nn.Linear(10, 10).to(device)
        max_norm = 2

        # 定义计算梯度范数的函数
        def compute_norm(norm_type):
            norm_type = float(norm_type)
            # 如果范数类型不是无穷大
            if norm_type != inf:
                total_norm = 0
                # 遍历线性层参数的梯度
                for p in l.parameters():
                    total_norm += p.grad.data.abs().pow(norm_type).sum()
                return pow(total_norm, 1. / norm_type)
            else:
                # 如果范数类型是无穷大，返回每个参数梯度的最大值
                return max(p.grad.data.abs().max() for p in l.parameters())

        # 定义比较梯度缩放比例的函数
        def compare_scaling(grads):
            # 计算每个参数梯度相对于全局梯度的缩放比例
            p_scale = [p.grad.data.div(g).view(-1) for p, g in zip(l.parameters(), grads)]
            scale = torch.cat(p_scale)
            # 断言缩放比例的标准差为零
            self.assertEqual(scale.std(), 0)
            return scale[0]

        # 创建两个梯度张量，一个递增序列，一个全部为1/1000
        grads = torch.arange(1., 101, device=device).view(10, 10), torch.ones(10, device=device).div(1000)
        # 将梯度分别复制给线性层的参数
        for p, g in zip(l.parameters(), grads):
            p._grad = g.clone().view_as(p.data)
        # 计算裁剪前的梯度范数
        norm_before = compute_norm(norm_type)
        # 执行梯度裁剪操作，并获取裁剪后的范数
        norm = clip_grad_norm_(l.parameters(), max_norm, norm_type=norm_type, foreach=foreach)
        # 再次计算裁剪后的梯度范数
        norm_after = compute_norm(norm_type)
        # 断言裁剪后的范数等于裁剪前的范数
        self.assertEqual(norm, norm_before)
        # 断言裁剪后的范数等于最大范数限制
        self.assertEqual(norm_after, max_norm)
        # 断言裁剪后的范数小于或等于裁剪前的范数
        self.assertLessEqual(norm_after, norm_before)
        # 比较裁剪前后的梯度缩放比例
        compare_scaling(grads)

        # 创建两个小梯度张量，一个随机，一个全部为1/500
        grads = torch.rand(10, 10, device=device).div(10000), torch.ones(10, device=device).div(500)
        # 将梯度分别复制给线性层的参数
        for p, g in zip(l.parameters(), grads):
            p.grad.data.copy_(g)
        # 计算裁剪前的梯度范数
        norm_before = compute_norm(norm_type)
        # 执行梯度裁剪操作，并获取裁剪后的范数
        norm = clip_grad_norm_(l.parameters(), max_norm, norm_type=norm_type, foreach=foreach)
        # 再次计算裁剪后的梯度范数
        norm_after = compute_norm(norm_type)
        # 断言裁剪后的范数等于裁剪前的范数
        self.assertEqual(norm, norm_before)
        # 断言裁剪前后的范数相等
        self.assertEqual(norm_before, norm_after)
        # 断言裁剪后的范数小于或等于最大范数限制
        self.assertLessEqual(norm_after, max_norm)
        # 比较裁剪前后的梯度缩放比例，并断言缩放比例为1
        scale = compare_scaling(grads)
        self.assertEqual(scale, 1)

        # 创建两个张量，并将其设为参数的梯度
        p1, p2 = torch.randn(10, 10, device=device), torch.randn(10, 10, device=device)
        g = torch.arange(1., 101, device=device).view(10, 10)
        p1._grad = g.clone()
        p2._grad = g.clone()
        # 对单个张量或张量列表执行梯度裁剪操作
        clip_grad_norm_(p1, max_norm, norm_type=norm_type, foreach=foreach)
        clip_grad_norm_([p2], max_norm, norm_type=norm_type, foreach=foreach)
        # 断言两个张量的梯度相等
        self.assertEqual(p1.grad, p2.grad)

    # 引用问题链接：https://github.com/pytorch/pytorch/issues/111484
    @onlyCUDA
    @largeTensorTest("42GB", "cuda")
    # 定义一个测试方法，用于测试在64位索引情况下的Softmax前向传播
    def test_softmax_forward_64bit_indexing(self, device):
        # 定义批处理大小、序列长度和词汇表大小
        batch_size = 70
        seq_len = 2048
        vocab_size = 50000

        # 创建一个全零的张量作为标签，大小为(batch_size, seq_len-1)，数据类型为长整型，在指定设备上
        shift_labels = torch.zeros(batch_size, seq_len - 1, dtype=torch.long, device=device)
        
        # 创建一个全为1的张量作为Logits，大小为(batch_size, seq_len-1, vocab_size)，数据类型为半精度浮点数，在指定设备上
        logits = torch.ones(batch_size, seq_len - 1, vocab_size, dtype=torch.float16, device=device)
        
        # 定义交叉熵损失函数，设置为不进行减少（reduction="none"）
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        
        # 计算交叉熵损失，需要对Logits进行维度置换，然后与标签计算损失，结果转换为单精度浮点数
        nll = loss_fct(logits.permute(0, 2, 1), shift_labels).float()
        
        # 获取在半精度浮点数下的比较容差
        rtol, atol = torch.testing._comparison.get_tolerances(torch.float16, rtol=None, atol=None)
        
        # 断言计算的损失与期望的损失（全部为词汇表大小的对数）相等，根据容差进行比较
        self.assertEqual(nll, torch.ones_like(nll) * torch.log(torch.tensor(vocab_size)), rtol=rtol, atol=atol)

    # 引用的问题链接：https://github.com/pytorch/pytorch/issues/68248
    # 仅在CUDA环境下运行的测试方法，用于测试在64位索引情况下的Softmax反向传播
    @onlyCUDA
    @largeTensorTest("20GB", "cuda")
    def test_softmax_backward_64bit_indexing(self, device):
        # 遍历两个张量大小（2147483650和2147483651）
        for numel in (2147483650, 2147483650 + 1):
            # 创建一个空张量，大小为[1, 1, numel]，数据类型为半精度浮点数，在指定设备上，填充为1/numel
            x = torch.empty([1, 1, numel], device=device, dtype=torch.float16)
            x.fill_(1.0 / numel)
            
            # 调用内部函数进行Softmax反向传播的数据处理，返回结果out
            out = torch._softmax_backward_data(x, x, 2, x.dtype)
            
            # 断言输出结果的第一个元素与1/numel相等
            self.assertEqual(out[0, 0, 0], 1 / numel)

    # 仅在CUDA环境下运行的测试方法，测试自适应1D平均池化在共享内存上的表现
    @onlyCUDA
    def test_adaptiveavg_pool1d_shmem(self, device):
        # 创建一个随机张量x，大小为[1, 256, 1, 5000]，在指定设备上，并设置存储格式为通道最后
        x = torch.randn(1, 256, 1, 5000, device=device).to(memory_format=torch.channels_last)
        
        # 将x的CPU版本赋值给x_cpu，并要求计算梯度
        x_cpu = x.cpu()
        x_cpu.requires_grad_()
        
        # 要求计算x的梯度
        x.requires_grad_()
        
        # 对x进行自适应平均池化，输出大小为(1, 256)，并对x_cpu进行相同操作
        y = torch.nn.functional.adaptive_avg_pool2d(x, (1, 256))
        y_cpu = torch.nn.functional.adaptive_avg_pool2d(x_cpu, (1, 256))
        
        # 创建一个随机梯度grad，并将其CPU版本赋值给grad_cpu
        grad = torch.randn_like(y)
        grad_cpu = grad.cpu()
        
        # 对y和y_cpu进行反向传播，计算梯度
        y.backward(grad)
        y_cpu.backward(grad_cpu)
        
        # 断言x的梯度与x_cpu的梯度相等
        self.assertEqual(x.grad, x_cpu.grad)

    # 标记为跳过元信息的测试方法
    @skipMeta
    # 定义一个测试方法，用于测试通道混洗功能，接受设备参数
    def test_channel_shuffle(self, device):
        # 创建一个 3D 张量 x，设备为指定的设备
        x = torch.tensor(
            [[[1, 2],
              [5, 6],
              [9, 10],
              [13, 14],
              ]], device=device
        )
        # 创建预期的输出张量 y_ref，设备为指定的设备
        y_ref = torch.tensor(
            [[[1, 2],
              [9, 10],
              [5, 6],
              [13, 14],
              ]], device=device
        )
        # ChannelsFirst 模式下的通道混洗
        with warnings.catch_warnings(record=True) as w:
            # 调用通道混洗函数 F.channel_shuffle，传入参数 x 和 2，并将结果转换到指定设备
            y = F.channel_shuffle(x, 2).to(device)
            # 断言警告列表长度为 0
            self.assertEqual(len(w), 0)
        # 断言 y 与预期的 y_ref 相等
        self.assertEqual(y, y_ref)
        
        # 4D 张量
        x = torch.tensor(
            [[[[1, 2],
               [3, 4]],
              [[5, 6],
               [7, 8]],
              [[9, 10],
               [11, 12]],
              [[13, 14],
               [15, 16]],
              ]], device=device
        )
        y_ref = torch.tensor(
            [[[[1, 2],
               [3, 4]],
              [[9, 10],
               [11, 12]],
              [[5, 6],
               [7, 8]],
              [[13, 14],
               [15, 16]],
              ]], device=device
        )
        # ChannelsFirst NCHW 模式下的通道混洗
        with warnings.catch_warnings(record=True) as w:
            y = F.channel_shuffle(x, 2).to(device)
            self.assertEqual(len(w), 0)
        self.assertEqual(y, y_ref)
        
        # ChannelsLast NHWC 模式下的通道混洗
        with warnings.catch_warnings(record=True) as w:
            # 将 x 转换为 ChannelsLast 内存格式，然后进行通道混洗
            y = F.channel_shuffle(x.contiguous(memory_format=torch.channels_last), 2).to(device)
            self.assertEqual(len(w), 0)
        # 将 y 转换为连续内存格式
        y = y.contiguous(memory_format=torch.contiguous_format)
        # 断言 y 与预期的 y_ref 相等
        self.assertEqual(y, y_ref)

        # 5D 张量
        x = torch.tensor(
            [[[[[1, 2],
               [3, 4]]],
              [[[5, 6],
               [7, 8]]],
              [[[9, 10],
               [11, 12]]],
              [[[13, 14],
               [15, 16]]],
              ]], device=device
        )
        y_ref = torch.tensor(
            [[[[[1, 2],
               [3, 4]]],
              [[[9, 10],
               [11, 12]]],
              [[[5, 6],
               [7, 8]]],
              [[[13, 14],
               [15, 16]]],
              ]], device=device
        )
        # ChannelsFirst NCHW 模式下的通道混洗
        with warnings.catch_warnings(record=True) as w:
            y = F.channel_shuffle(x, 2).to(device)
            self.assertEqual(len(w), 0)
        self.assertEqual(y, y_ref)
        
        # ChannelsLast NHWC 模式下的通道混洗
        with warnings.catch_warnings(record=True) as w:
            # 将 x 转换为 ChannelsLast_3D 内存格式，然后进行通道混洗
            y = F.channel_shuffle(x.contiguous(memory_format=torch.channels_last_3d), 2).to(device)
            self.assertEqual(len(w), 0)
        # 将 y 转换为连续内存格式
        y = y.contiguous(memory_format=torch.contiguous_format)
        # 断言 y 与预期的 y_ref 相等
        self.assertEqual(y, y_ref)
class TestFunctionalPickle(TestCase):

    # issue gh-38137
    # 测试 pickle 序列化函数 F.softsign 是否能正常执行，不会抛出异常
    def test_pickle_softsign(self):
        s = pickle.dumps(F.softsign)


class TestFusionUtils(TestCase):
    
    # 测试融合卷积和批归一化操作时权重是否正确设置梯度要求
    def test_fuse_conv_bn_requires_grad(self):
        conv = torch.nn.Conv2d(3, 3, 3)
        bn = torch.nn.BatchNorm2d(3)
        cases = itertools.product([True, False], [True, False])
        for w_rg, b_rg in cases:
            conv.weight.requires_grad = w_rg
            conv.bias.requires_grad = b_rg
            weight, bias = \
                fuse_conv_bn_weights(conv.weight, conv.bias,
                                     bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
            self.assertEqual(weight.requires_grad, w_rg)
            self.assertEqual(bias.requires_grad, b_rg)

    # 测试融合线性层和批归一化操作时权重是否正确设置梯度要求
    def test_fuse_linear_bn_requires_grad(self):
        linear = torch.nn.Linear(3, 3)
        bn = torch.nn.BatchNorm1d(3)
        cases = itertools.product([True, False], [True, False])
        for w_rg, b_rg in cases:
            linear.weight.requires_grad = w_rg
            linear.bias.requires_grad = b_rg
            weight, bias = \
                fuse_linear_bn_weights(linear.weight, linear.bias,
                                       bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
            self.assertEqual(weight.requires_grad, w_rg)
            self.assertEqual(bias.requires_grad, b_rg)

class TestUtils(TestCase):
    pass
    # 定义一个测试方法，验证在状态字典中是否存在指定前缀，并在存在时移除该前缀
    def test_consume_prefix_in_state_dict_if_present(self):
        # 定义一个简单的神经网络模块
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, 3, bias=True)
                self.conv2 = nn.Conv2d(3, 3, 3, bias=False)

        # 定义一个包含神经网络模块的更大的神经网络
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(5, 5)
                self.linear2 = nn.Linear(5, 5)
                net.bn = nn.BatchNorm2d(2)  # 错误：应为 self.bn = nn.BatchNorm2d(2)
                self.block = Block()

        # 0. 针对非分布式数据并行（non-DDP）模型的空状态字典情况
        net = nn.Module()
        state_dict = net.state_dict()
        # 尝试消费状态字典中的指定前缀（这里是空前缀），期望不会改变其内容
        nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, 'module.')
        # 检查消费前后状态字典的键和元数据保持顺序不变
        self.assertEqual(list(state_dict.keys()), list(net.state_dict().keys()))
        self.assertEqual(list(state_dict._metadata.keys()), list(net.state_dict()._metadata.keys()))

        # 1. 针对非-DDP模型的测试示例状态字典情况
        net = Net()
        state_dict = net.state_dict()
        # 尝试消费状态字典中的指定前缀（这里是 'module.'），期望不会改变其内容
        nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, 'module.')
        # 检查消费前后状态字典的键和元数据保持顺序不变
        self.assertEqual(list(state_dict.keys()), list(net.state_dict().keys()))
        self.assertEqual(list(state_dict._metadata.keys()), list(net.state_dict()._metadata.keys()))

        # 2. 针对分布式数据并行（DDP）模型的测试示例状态字典情况
        state_dict = net.state_dict()
        metadata = state_dict._metadata
        # 将状态字典中的每个键前加 'module.' 前缀，创建新的状态字典
        ddp_state_dict = OrderedDict((f'module.{k}', v) for k, v in state_dict.items())
        # 复制元数据，并将空键对应的元数据移动到 'module.' 下
        ddp_state_dict._metadata = OrderedDict({'': metadata['']})
        ddp_state_dict._metadata.update(('module' if k == '' else f'module.{k}', v) for k, v in metadata.items())
        # 尝试消费状态字典中的指定前缀（这里是 'module.'），期望不会改变其内容
        nn.modules.utils.consume_prefix_in_state_dict_if_present(ddp_state_dict, 'module.')
        # 检查消费前后状态字典的键和元数据保持顺序不变
        self.assertEqual(list(state_dict.keys()), list(ddp_state_dict.keys()))
        self.assertEqual(list(state_dict._metadata.keys()), list(ddp_state_dict._metadata.keys()))
# 实例化设备类型相关的测试，使用 TestNNDeviceType 类和全局变量进行实例化
instantiate_device_type_tests(TestNNDeviceType, globals())

# 实例化参数化的测试，使用 TestNN 类进行实例化
instantiate_parametrized_tests(TestNN)

# 如果当前脚本作为主程序运行，则执行以下代码块
if __name__ == '__main__':
    # 启用 TestCase 的默认数据类型检查
    TestCase._default_dtype_check_enabled = True
    # 运行测试套件
    run_tests()
```