# `.\pytorch\test\test_meta.py`

```
# Owner(s): ["module: decompositions"]

import itertools  # 导入 itertools 模块，提供迭代工具函数
import torch  # 导入 PyTorch 深度学习框架
import os  # 导入操作系统相关的功能
import numpy as np  # 导入 NumPy 数组计算库，用于科学计算
from enum import Enum  # 导入枚举类型 Enum
from torch.overrides import resolve_name  # 导入 resolve_name 函数，用于解析名称
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten  # 导入树操作相关函数
from torch.utils import _pytree as pytree  # 导入 pytree 模块
from torch._subclasses.meta_utils import MetaConverter, assert_metadata_eq, is_sparse_any  # 导入元数据相关函数和类
import torch.utils._python_dispatch  # 导入 Python 调度相关功能
from torch._dispatch.python import enable_python_dispatcher  # 导入 Python 调度器启用函数
from torch._ops import OpOverload, OpOverloadPacket  # 导入运算符重载相关类
from torch.testing import make_tensor  # 导入用于创建测试张量的函数
from torch.testing._internal.common_utils import unMarkDynamoStrictTest  # 导入内部测试工具函数
from torch.testing._internal.common_utils import (  # 导入内部测试常用工具函数和类
    TestCase,
    skipIfCrossRef,
    skipIfTorchDynamo,
    suppress_warnings,
    TEST_WITH_ASAN,
    TEST_WITH_TORCHDYNAMO,
    run_tests,
    dtype_abbrs,
    parametrize,
)
from torch.testing._internal.common_device_type import (  # 导入设备类型相关测试工具
    ops,
    instantiate_device_type_tests,
    onlyCUDA,
    onlyCPU,
    OpDTypes,
)
from torch.testing._internal.common_methods_invocations import (  # 导入方法调用相关测试工具
    binary_ufuncs,
    op_db,
    foreach_unary_op_db,
    foreach_binary_op_db,
    foreach_pointwise_op_db,
    foreach_reduce_op_db,
    foreach_other_op_db,
)
from torch.testing._internal.opinfo.core import S, SampleInput  # 导入操作信息核心类和样本输入类
from torchgen.yaml_utils import YamlLoader  # 导入 YAML 加载器
from torchgen.model import OperatorName  # 导入运算符名称模块

import copy  # 导入复制相关函数
import sys  # 导入系统相关的函数和变量
import yaml  # 导入 YAML 序列化和反序列化模块
import atexit  # 导入退出函数注册模块
import re  # 导入正则表达式模块
from collections import defaultdict  # 导入默认字典
from collections.abc import Iterable  # 导入 Iterable 抽象基类
import unittest  # 导入单元测试框架
import warnings  # 导入警告相关模块
import weakref  # 导入弱引用支持模块
from functools import partial, wraps  # 导入部分函数和包装器

bf16 = torch.bfloat16  # 定义 bf16 为 torch.bfloat16 类型
f64 = torch.float64  # 定义 f64 为 torch.float64 类型
f32 = torch.float32  # 定义 f32 为 torch.float32 类型
f16 = torch.float16  # 定义 f16 为 torch.float16 类型
c32 = torch.complex32  # 定义 c32 为 torch.complex32 类型
c64 = torch.complex64  # 定义 c64 为 torch.complex64 类型
c128 = torch.complex128  # 定义 c128 为 torch.complex128 类型
i8 = torch.int8  # 定义 i8 为 torch.int8 类型
i16 = torch.int16  # 定义 i16 为 torch.int16 类型
i32 = torch.int32  # 定义 i32 为 torch.int32 类型
i64 = torch.int64  # 定义 i64 为 torch.int64 类型
b8 = torch.bool  # 定义 b8 为 torch.bool 类型
u8 = torch.uint8  # 定义 u8 为 torch.uint8 类型
u16 = torch.uint16  # 定义 u16 为 torch.uint16 类型
u32 = torch.uint32  # 定义 u32 为 torch.uint32 类型
u64 = torch.uint64  # 定义 u64 为 torch.uint64 类型

foreach_op_db = (  # 定义 foreach_op_db 变量，包含所有的操作数据库
    foreach_unary_op_db +  # 包括一元操作数据库
    foreach_binary_op_db +  # 包括二元操作数据库
    foreach_pointwise_op_db +  # 包括逐点操作数据库
    foreach_reduce_op_db +  # 包括减少操作数据库
    foreach_other_op_db  # 包括其他操作数据库
)


class TestMetaConverter(TestCase):
    def assertSameVersionCounter(self, m1, m2):
        # 断言 m1 和 m2 的版本计数器相同，用于元数据比较
        vc = m1._version
        self.assertEqual(m2._version, vc)
        # 使用 torch.no_grad() 确保即使在叶子上也会进行版本计数器增加
        with torch.no_grad():
            m1._base.add_(3)
        self.assertNotEqual(m1._version, vc)  # 断言 m1 的版本计数器已经改变
        self.assertEqual(m2._version, m1._version)  # 断言 m2 的版本计数器与 m1 相同

    def assertMetadataMatches(self, m1, m2):
        assert_metadata_eq(self.assertEqual, m1, m2)  # 使用 assert_metadata_eq 函数断言 m1 和 m2 的元数据匹配
    # 测试非叶节点的视图
    def test_view_of_non_leaf(self):
        # 创建一个形状为 (4,) 的张量 x，并设置 requires_grad=True
        x = torch.randn(4, requires_grad=True)
        # 对 x 取负数并赋给 y
        y = x.neg()
        # 使用切片操作创建 y 的视图 z1 和 z2
        z1 = y[:]
        z2 = y[:]
        # 创建 MetaConverter 的实例
        to_meta = MetaConverter()
        # 将 z1 转换为 Meta 对象 m1
        m1 = to_meta(z1)
        # 将 z2 转换为 Meta 对象 m2
        m2 = to_meta(z2)

        # 检查测试是否按其声明进行
        self.assertTrue(m1._is_view())  # 确保 m1 是一个视图
        self.assertFalse(m1._base.is_leaf)  # 确保 m1 的基张量不是叶节点

        # 断言 m1 和 m2 是不同的对象
        self.assertIsNot(m1, m2)
        # 断言 m1 和 z1 的元数据匹配
        self.assertMetadataMatches(m1, z1)
        # 断言 m2 和 z2 的元数据匹配
        self.assertMetadataMatches(m2, z2)
        # 断言 m1 和 m2 的版本计数相同
        self.assertSameVersionCounter(m1, m2)

    # 测试叶节点的视图
    def test_view_of_leaf(self):
        # 创建一个形状为 (4,) 的张量 x，并设置 requires_grad=True
        x = torch.randn(4, requires_grad=True)
        # 使用切片操作创建 x 的视图 z1 和 z2
        z1 = x[:]
        z2 = x[:]
        # 创建 MetaConverter 的实例
        to_meta = MetaConverter()
        # 将 z1 转换为 Meta 对象 m1
        m1 = to_meta(z1)
        # 将 z2 转换为 Meta 对象 m2
        m2 = to_meta(z2)

        # 检查测试是否按其声明进行
        self.assertTrue(m1._is_view())  # 确保 m1 是一个视图
        self.assertTrue(m1._base.is_leaf)  # 确保 m1 的基张量是叶节点

        # 断言 m1 和 m2 是不同的对象
        self.assertIsNot(m1, m2)
        # 断言 m1 和 z1 的元数据匹配
        self.assertMetadataMatches(m1, z1)
        # 断言 m2 和 z2 的元数据匹配
        self.assertMetadataMatches(m2, z2)
        # 断言 m1 和 m2 的版本计数相同
        self.assertSameVersionCounter(m1, m2)

    # 测试叶节点的视图的视图
    def test_view_of_view_of_leaf(self):
        # 创建形状为 (8,) 的张量 x
        x = torch.randn(8)
        # 将 x 变形为 (2, 4) 的张量 y
        y = x.view(2, 4)
        # 设置 y 的 requires_grad=True
        y.requires_grad = True
        # 将 y 变形为 (2, 2, 2) 的张量 z
        z = y.view(2, 2, 2)

        # 创建 MetaConverter 的实例
        to_meta = MetaConverter()
        # 将 x 转换为 Meta 对象 mx
        mx = to_meta(x)
        # 将 z 转换为 Meta 对象 mz
        mz = to_meta(z)

        # 断言 z 不是叶节点
        self.assertFalse(z.is_leaf)

        # 断言 mx 和 x 的元数据匹配
        self.assertMetadataMatches(mx, x)
        # 断言 mz 和 z 的元数据匹配
        self.assertMetadataMatches(mz, z)

    # 测试叶节点
    def test_leaf(self):
        # 创建一个形状为 (4,) 的张量 x，并设置 requires_grad=True
        x = torch.randn(4, requires_grad=True)
        # 创建 MetaConverter 的实例
        to_meta = MetaConverter()
        # 将 x 转换为 Meta 对象 m
        m = to_meta(x)

        # 检查测试是否按其声明进行
        self.assertTrue(m.is_leaf)  # 确保 m 是一个叶节点
        self.assertTrue(m.requires_grad)  # 确保 m 的 requires_grad=True

        # 断言 m 和 x 的元数据匹配
        self.assertMetadataMatches(m, x)

    # 测试非叶节点
    def test_non_leaf(self):
        # 创建一个形状为 (4,) 的张量 x，并设置 requires_grad=True
        x = torch.randn(4, requires_grad=True)
        # 对 x 取负数并赋给 y
        y = x.neg()
        # 创建 MetaConverter 的实例
        to_meta = MetaConverter()
        # 将 y 转换为 Meta 对象 m
        m = to_meta(y)

        # 检查测试是否按其声明进行
        self.assertFalse(m.is_leaf)  # 确保 m 不是叶节点
        self.assertTrue(m.requires_grad)  # 确保 m 的 requires_grad=True

        # 断言 m 和 y 的元数据匹配
        self.assertMetadataMatches(m, y)

    # 测试 requires_grad=False 的情况
    def test_requires_grad_false(self):
        # 创建一个形状为 (4,) 的张量 x，并设置 requires_grad=False
        x = torch.randn(4, requires_grad=False)
        # 创建 MetaConverter 的实例
        to_meta = MetaConverter()
        # 将 x 转换为 Meta 对象 m
        m = to_meta(x)

        # 检查测试是否按其声明进行
        self.assertFalse(m.requires_grad)  # 确保 m 的 requires_grad=False

        # 断言 m 和 x 的元数据匹配
        self.assertMetadataMatches(m, x)

    # 测试使用 channels_last 内存格式的情况
    def test_channels_last(self):
        # 创建一个形状为 (2, 3, 4, 5) 的张量 x，使用 channels_last 内存格式
        x = torch.empty(2, 3, 4, 5, memory_format=torch.channels_last)
        # 创建 MetaConverter 的实例
        to_meta = MetaConverter()
        # 将 x 转换为 Meta 对象 m
        m = to_meta(x)

        # 检查测试是否按其声明进行
        self.assertTrue(m.is_leaf)  # 确保 m 是一个叶节点

        # 断言 m 和 x 的元数据匹配
        self.assertMetadataMatches(m, x)
    def test_channels_last_leaf(self):
        # 创建一个空的张量，形状为 (2, 3, 4, 5)，使用 channels_last 内存格式，并要求梯度计算
        x = torch.empty(2, 3, 4, 5, memory_format=torch.channels_last, requires_grad=True)
        # 创建 MetaConverter 实例
        to_meta = MetaConverter()
        # 将张量 x 转换为元数据 m
        m = to_meta(x)

        # 检查测试是否确实测试了其声称的内容
        self.assertTrue(m.requires_grad)  # 断言 m 需要梯度计算
        self.assertTrue(m.is_leaf)         # 断言 m 是叶子节点

        # 检查元数据 m 是否与原始张量 x 匹配
        self.assertMetadataMatches(m, x)

    def test_channels_last_non_leaf(self):
        # 创建一个空的张量 x，形状为 (2, 3, 4, 5)，使用 channels_last 内存格式，并要求梯度计算
        x = torch.empty(2, 3, 4, 5, memory_format=torch.channels_last, requires_grad=True)
        # 创建一个新的张量 y，通过对 x 加 2 得到
        y = x + 2

        # 检查 x 和 y 的步幅是否相等
        self.assertEqual(x.stride(), y.stride())
        # 断言 y 不是叶子节点
        self.assertFalse(y.is_leaf)

        # 创建 MetaConverter 实例
        to_meta = MetaConverter()
        # 将张量 y 转换为元数据 m
        m = to_meta(y)

        # 检查测试是否确实测试了其声称的内容
        self.assertTrue(m.requires_grad)  # 断言 m 需要梯度计算
        self.assertFalse(m.is_leaf)        # 断言 m 不是叶子节点

        # 检查元数据 m 是否与张量 y 匹配
        self.assertMetadataMatches(m, y)

        # 检查是否可以使用 m 作为输入进行自动求导，而不出错
        loss = m.sum()
        torch.autograd.grad(loss, m)

    def test_empty_strided_non_dense_leaf(self):
        # 创建一个空的步幅张量 x，形状为 (2, 2)，步幅为 (4, 2)，并要求梯度计算
        x = torch.empty_strided((2, 2), (4, 2), requires_grad=True)

        # 创建 MetaConverter 实例
        to_meta = MetaConverter()
        # 将步幅张量 x 转换为元数据 m
        m = to_meta(x)

        # 检查测试是否确实测试了其声称的内容
        self.assertTrue(m.requires_grad)  # 断言 m 需要梯度计算
        self.assertTrue(m.is_leaf)         # 断言 m 是叶子节点

        # 检查元数据 m 是否与步幅张量 x 匹配
        self.assertMetadataMatches(m, x)

    def test_view_mutate(self):
        # 创建一个全零张量 x，形状为 (4,)
        x = torch.zeros(4)
        # 将张量 x 变形为形状为 (2, 2) 的张量 y
        y = x.view(2, 2)

        # 创建 MetaConverter 实例
        to_meta = MetaConverter()
        # 将张量 y 转换为元数据 m
        m = to_meta(y)

        # 对张量 y 进行就地加法操作
        y.add_(torch.randn(2, 2, requires_grad=True))
        # 对元数据 m 进行就地加法操作，指定设备为 'meta'，并要求梯度计算
        m.add_(torch.randn(2, 2, device='meta', requires_grad=True))

    def test_non_leaf_torture(self):
        # 创建一个空的张量 x，形状为 (20,)，并要求梯度计算
        x = torch.empty(20, requires_grad=True)
        # 使用 torch.no_grad 上下文管理器，以非梯度计算方式对张量 x 进行操作
        with torch.no_grad():
            x.set_(x.storage(), 10, (2,), (2,))

        # 创建 MetaConverter 实例
        to_meta = MetaConverter()
        # 将张量 x 转换为元数据 m
        m = to_meta(x)

        # 检查测试是否确实测试了其声称的内容
        self.assertTrue(m.requires_grad)  # 断言 m 需要梯度计算
        self.assertTrue(m.is_leaf)         # 断言 m 是叶子节点

        # 检查元数据 m 是否与张量 x 匹配
        self.assertMetadataMatches(m, x)

    # 注意：目前复杂的转换未被实际测试，因为我们对复杂类型转换有统一的排除

    def test_view_as_real(self):
        # 创建一个复杂类型张量 x，形状为 (4,)，数据类型为 torch.complex64
        x = torch.randn(4, dtype=torch.complex64)
        # 将张量 x 视为其实部张量 y
        y = torch.view_as_real(x)
        # 使用 MetaConverter 将张量 y 转换为元数据 m
        m = MetaConverter()(y)
        # 检查元数据 m 是否与张量 y 匹配
        self.assertMetadataMatches(m, y)

    def test_complex_noncontiguous_bug(self):
        # 创建一个复杂类型张量 x，形状为 (2, 4, 9)，数据类型为 torch.complex32，选择部分切片
        x = torch.randn((2, 2, 4, 9), dtype=torch.complex32)[:, 0, :, :]
        # 使用 MetaConverter 将张量 x 转换为元数据 m
        m = MetaConverter()(x)
        # 检查元数据 m 是否与张量 x 匹配
        self.assertMetadataMatches(m, x)

    def test_view_as_complex(self):
        # 创建一个浮点类型张量 x，形状为 (4, 2)
        x = torch.randn((4, 2), dtype=torch.float32)
        # 将张量 x 视为复数类型张量 y
        y = torch.view_as_complex(x)
        # 使用 MetaConverter 将张量 y 转换为元数据 m
        m = MetaConverter()(y)
        # 检查元数据 m 是否与张量 y 匹配
        self.assertMetadataMatches(m, y)

    def test_view_dtype(self):
        # 创建一个浮点类型张量 x，形状为 (4,)
        x = torch.randn(4, dtype=torch.float32)
        # 将张量 x 变换为指定数据类型 dtype=torch.int32 的张量 y
        y = x.view(dtype=torch.int32)
        # 使用 MetaConverter 将张量 y 转换为元数据 m
        m = MetaConverter()(y)
        # 检查元数据 m 是否与张量 y 匹配
        self.assertMetadataMatches(m, y)
    # 测试复数张量的虚部提取
    def test_imag(self):
        # 创建一个形状为 (4,) 的随机复数张量 x
        x = torch.randn(4, dtype=torch.complex64)
        # 提取 x 的虚部作为 y
        y = x.imag
        # 使用 MetaConverter 对象转换 y 的元数据
        m = MetaConverter()(y)
        # 断言转换后的元数据匹配 y 的预期元数据
        self.assertMetadataMatches(m, y)

    # 测试 inplace 设置存储
    def test_inplace_set_storage(self):
        # 创建一个包含 [0, 1] 的整数张量 x
        x = torch.tensor([0, 1], dtype=torch.int64)
        # 获取 x 的未类型化存储
        storage = x.untyped_storage()
        # 获取存储的大小
        ssize = storage.size()
        # 创建一个空的整数张量 meta
        meta = torch.empty((), dtype=torch.int64)
        # 使用 inplace 方式设置 meta 的存储为 storage
        meta.set_(storage, 0, (), ())
        # 断言设置后存储的大小不变
        self.assertEqual(storage.size(), ssize)

    # 跳过由于 Torch Dynamo 问题导致的测试失败
    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1991")
    def test_weakref(self):
        # 创建一个形状为 (4, 4, 4) 的随机张量 x
        x = torch.randn(4, 4, 4)
        # 创建 MetaConverter 对象 m
        m = MetaConverter()
        # 使用 m 转换 x 得到 y 和 z
        y = m(x)
        z = m(x)
        # 断言 y 和 z 是同一个对象
        self.assertIs(y, z)
        # 断言 m 中记录的张量和存储的数量为 1
        self.assertEqual(len(m.tensor_memo), 1)
        self.assertEqual(len(m.storage_memo), 1)
        self.assertEqual(len(m.describer.lookup_tensor), 1)
        self.assertEqual(len(m.describer.lookup_storage), 1)
        # 删除张量 x
        del x
        # 断言 m 中记录的张量和存储的数量为 0
        self.assertEqual(len(m.describer.lookup_tensor), 0)
        self.assertEqual(len(m.describer.lookup_storage), 0)
        # 删除 y 和 z
        del y
        del z
        # 断言 m 中记录的张量和存储的数量为 0
        self.assertEqual(len(m.tensor_memo), 0)
        self.assertEqual(len(m.storage_memo), 0)
        # 创建一个空列表 li 和结果列表 r
        li = []
        r = []
        # 循环创建 4 个不同形状的随机张量，并使用 m 进行转换，将结果存入 r
        for i in range(4):
            li.append(torch.rand([i]))
            r.append(m(li[-1]))
        # 断言 m 中记录的张量和存储的数量为 4
        self.assertEqual(len(m.tensor_memo), 4)
        self.assertEqual(len(m.storage_memo), 4)
        self.assertEqual(len(m.describer.lookup_tensor), 4)
        self.assertEqual(len(m.describer.lookup_storage), 4)
        # 删除列表 li
        del li
        # 断言 m 中记录的张量和存储的数量为 0
        self.assertEqual(len(m.describer.lookup_tensor), 0)
        self.assertEqual(len(m.describer.lookup_storage), 0)
        # 删除列表 r
        del r
        # 断言 m 中记录的张量和存储的数量为 0
        self.assertEqual(len(m.tensor_memo), 0)
        self.assertEqual(len(m.storage_memo), 0)

    # 跳过由于 Torch Dynamo 问题导致的测试失败
    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1991")
    def test_tensor_outlives_converter(self):
        # 创建 MetaConverter 对象 m
        m = MetaConverter()
        # 创建 m 的弱引用 ref
        ref = weakref.ref(m)
        # 创建一个形状为 [4, 4] 的随机张量 x
        x = torch.randn([4, 4])
        # 使用 m 转换 x 得到 y
        y = m(x)
        # 删除 m
        del m
        # 断言 m 的弱引用 ref 已经为 None
        self.assertIs(ref(), None)
# 导入 torch 的 aten 操作命名空间
aten = torch.ops.aten

# 定义应该检查步长的操作集合，初始为 torch.Tensor.__getitem__ 方法
CHECK_STRIDES = {
    torch.Tensor.__getitem__,
}

# 定义应该检查所有步长的操作集合，初始为 aten.unsqueeze.default 方法
CHECK_ALL_STRIDES = {
    aten.unsqueeze.default
}

# 定义应该跳过步长检查的操作集合
CHECK_STRIDES_SKIPS = {
    # 下面列出的操作会跳过步长检查
    aten._conj_physical.default,
    aten._fft_c2c.default,
    aten._fft_c2r.default,
    aten._fft_r2c.default,
    aten._linalg_svd.default,
    aten.binary_cross_entropy.default,
    aten.complex.default,
    aten.polar.default,
    aten.copysign.Tensor,
    aten.div.Tensor_mode,
    aten.floor_divide.default,
    aten.heaviside.default,
    aten.lerp.Scalar,
    aten.lerp.Tensor,
    aten.logaddexp.default,
    aten.logical_and.default,
    aten.logical_or.default,
    aten.logical_xor.default,
    aten.pow.Scalar,
    aten.prelu.default,
    aten.special_xlog1py.default,
    aten.xlogy.Tensor,
    aten.nll_loss2d_forward.default,

    # 以下操作与 channel_last 和 channel_last_3d 相关的失败
    aten.convolution.default,

    # 下面的操作在 include_storage_offset = True 时可能失败，但这些情况较少
    # 仍然需要修复，暂时保留以进行跟踪。
    # aten._reshape_alias.default,  # 用于 test_dispatch_symbolic_meta_outplace_all_strides_matmul_cuda_float32
    # aten.view.default,  # 用于 test_dispatch_symbolic_meta_outplace_all_strides_unflatten_cuda_float32
}

# 定义应该跳过共轭检查的操作集合
CHECK_CONJ_SKIPS = {
    # 这些操作的共轭位未复制，详见：
    # https://github.com/pytorch/pytorch/pull/101836
    aten.linalg_lu_solve.out,
}

# 定义 CheckStrides 枚举类，表示步长检查的级别
class CheckStrides(Enum):
    NONE = 0  # 不需要检查步长
    SIGNIFICANT = 1  # 需要检查重要的步长
    ALL = 2  # 需要检查所有步长

# 定义一个函数，确定是否需要检查给定函数的步长
def should_check_strides(func):
    if func in CHECK_ALL_STRIDES:
        return CheckStrides.ALL
    if func in CHECK_STRIDES:
        return CheckStrides.SIGNIFICANT
    if func in CHECK_STRIDES_SKIPS:
        return CheckStrides.NONE
    if not isinstance(func, torch._ops.OpOverload):
        return CheckStrides.NONE
    # 原语（prims）预期正确建模步长
    if func.namespace == "prims":
        return CheckStrides.SIGNIFICANT
    # 检查是否为视图，通过检查返回值中是否有非空别名集合来判断
    if any(r.alias_info.before_set for r in func._schema.returns if r.alias_info):
        return CheckStrides.SIGNIFICANT
    # TODO: 检查 TensorIterator
    return CheckStrides.SIGNIFICANT

# 定义一个函数，用于断言参考元数据和实际结果是否相等
def assert_ref_meta_equal(test_case, func, meta_rs, rs, msg_callable):
    flat_meta_rs = pytree.tree_leaves(meta_rs)
    flat_rs = pytree.tree_leaves(rs)
    test_case.assertEqual(len(flat_meta_rs), len(flat_rs))
    # 遍历元组 (i, meta_r, r)，其中 i 是索引，meta_r 和 r 是 torch.Tensor 对象
    for i, meta_r, r in zip(range(len(flat_rs)), flat_meta_rs, flat_rs):
        # 定义一个用于测试断言的函数 test_assert，如果条件不满足则抛出 RuntimeError 异常
        def test_assert(cond, msg):
            if not cond:
                raise RuntimeError(f"output {i}: {msg_callable(msg)}")
        
        # 如果 r 不是 torch.Tensor 对象，则跳过当前循环，继续处理下一个元素
        if not isinstance(r, torch.Tensor):
            continue
        
        # 断言 meta_r 是 torch.Tensor 对象，如果不是则抛出异常
        test_assert(isinstance(meta_r, torch.Tensor), f"but real {i}th result is Tensor")
        
        # 断言 meta_r 的数据类型与 r 的数据类型相同，如果不同则抛出异常
        test_assert(meta_r.dtype == r.dtype, f"for element {i}, was {meta_r.dtype} but real dtype was {r.dtype}")
        
        # 断言 meta_r 的形状与 r 的形状相同，如果不同则抛出异常
        test_assert(meta_r.shape == r.shape, f"for element {i}, was {meta_r.shape} but real shape was {r.shape}")
        
        # 检查是否需要检查张量的步长，根据函数 should_check_strides(func) 的返回值进行不同的处理
        if should_check_strides(func) == CheckStrides.ALL:
            # 检查所有步长是否相同
            same_strides, _ = torch._prims_common.check_all_strides(meta_r, r)
            test_assert(same_strides, f"for element {i}, was {meta_r.stride()} but real stride was {r.stride()}")
        elif should_check_strides(func) == CheckStrides.SIGNIFICANT:
            # 检查重要步长是否相同
            same_strides, _ = torch._prims_common.check_significant_strides(meta_r, r)
            test_assert(same_strides, f"for element {i}, was {meta_r.stride()} but real stride was {r.stride()}")
        
        # 断言 meta_r 的存储偏移与 r 的存储偏移相同，如果不同则抛出异常
        test_assert(
            meta_r.storage_offset() == r.storage_offset(),
            f"for element {i}, was {meta_r.storage_offset()} but real storage_offset was {r.storage_offset()}")
        
        # 断言 meta_r 的 requires_grad 属性与 r 的 requires_grad 属性相同，如果不同则抛出异常
        test_assert(meta_r.requires_grad == r.requires_grad,
                    f"for element {i}, was {meta_r.requires_grad} but real requires_grad was {r.requires_grad}")
        
        # 如果 func 不在 CHECK_CONJ_SKIPS 中，则断言 meta_r 的共轭属性与 r 的共轭属性相同，如果不同则抛出异常
        if func not in CHECK_CONJ_SKIPS:
            test_assert(meta_r.is_conj() == r.is_conj(),
                        f"for element {i}, was {meta_r.is_conj()} but real is_conj was {r.is_conj()}")
        
        # 断言 meta_r 的负号属性与 r 的负号属性相同，如果不同则抛出异常
        test_assert(meta_r.is_neg() == r.is_neg(), f"for element {i}, was {meta_r.is_neg()} but real is_neg was {r.is_neg()}")
# 环境变量控制是否在测试套件运行结束时打印预期失败列表。使用方法如下：
#
# 1. 在安装了 LAPACK/MAGMA 的 CUDA 版 PyTorch 上运行命令：
#    `PYTORCH_COLLECT_EXPECT=1 python test/test_meta.py`。
#    可以使用 `-k test_meta` 或 `-k test_dispatch_meta` 进行过滤，只关注其中一个列表。
# 2. 根据打印出的跳过/预期失败列表，将 torch.* 条目加入 meta_function，aten.* 条目加入 meta_dispatch。
#    如果已存在条目，需合并这些条目。
#
# 这个过程有些手动，通常不需要频繁进行，除非进行了重大更改（例如，向 PyTorch 添加了新的数据类型）并且需要更新列表。
# 如果要从头开始操作，只需在运行之前清空预先存在的列表即可。
#
# 警告：Python 字典字面量会静默忽略重复的键
COLLECT_EXPECT = os.getenv('PYTORCH_COLLECT_EXPECT', '0') == '1'

# 记录已成功和失败的操作及其对应数据类型
seen_succeeded = {}
seen_failed = {}
failed_reasons = defaultdict(set)

# 打印函数，用于输出已记录的失败和跳过列表
def print_seen():
    expected_failures = []
    skips = []

    # 格式化数据类型列表
    def fmt_dtypes(dtypes):
        r = ', '.join(sorted(dtype_abbrs[d] for d in dtypes))
        return '{' + r + '}'

    # 遍历记录的失败操作及其数据类型
    for op, failed_dtypes in seen_failed.items():
        ops = resolve_name(op)  # 解析操作名称
        succeeded_dtypes = seen_succeeded.get(op, set())
        expected_failures_dtypes = failed_dtypes - succeeded_dtypes
        skips_dtypes = failed_dtypes & succeeded_dtypes
        reasons = ""
        if failed_reasons[op]:
            reasons = "  # " + ", ".join(sorted(failed_reasons[op]))
        # 将预期失败和跳过的操作及数据类型格式化并添加到相应列表中
        if expected_failures_dtypes:
            expected_failures.append(f"    {ops}: {fmt_dtypes(expected_failures_dtypes)},{reasons}")
        if skips_dtypes:
            skips.append(f"    {ops}: {fmt_dtypes(skips_dtypes)},")
    
    # 对列表进行排序
    expected_failures.sort()
    skips.sort()
    nl = '\n'
    # 打印格式化的预期失败和跳过列表
    print(f"""\
expected_failures = {{
{nl.join(expected_failures)}
}}

skips = {{
{nl.join(skips)}
}}
""")

# 如果设置了 COLLECT_EXPECT 环境变量，则在退出时注册打印函数 print_seen
if COLLECT_EXPECT:
    atexit.register(print_seen)

# 枚举类型，用于描述测试期望的结果，包括成功、预期失败和跳过
TestExpect = Enum("TestExpect", ("SUCCESS", "XFAILURE", "SKIP"))

# verbose_print 函数定义，用于详细打印参数 e 的信息
def verbose_print(e):
    # Lit 类定义
    class Lit:
        def __init__(self, s):
            self.s = s

        def __repr__(self):
            return self.s

    # go 函数，根据类型 t 返回相应的表示形式
    def go(t):
        if is_sparse_any(t):
            return t
        elif isinstance(t, torch.Tensor):
            return Lit(f"{t} stride={t.stride()}")
        else:
            return t

    return repr(tree_map(go, e))

# run_meta_crossref 函数定义，用于执行元数据交叉引用测试
def run_meta_crossref(
    test_case,
    test_expect,
    func,
    args,
    kwargs,
    *,
    dtype,
    device_type,
    run_symbolic_meta: bool
):
    # MetaConverter 实例化
    to_meta = MetaConverter()
    # 若测试期望不是跳过，则执行元数据转换操作
    do_meta = test_expect is not TestExpect.SKIP
    # 如果需要处理元数据，尝试将函数参数和关键字参数转换为元数据
    if do_meta:
        try:
            # 使用 tree_map 将 args 中的每个元素转换为元数据形式
            meta_args = tree_map(to_meta, args)
            # 使用 tree_map 将 kwargs 中的每个值转换为元数据形式
            meta_kwargs = tree_map(to_meta, kwargs)
        except Exception as e:
            # 如果转换失败，抛出运行时错误，包含原始的参数和关键字参数信息
            raise RuntimeError(
                f"failed to convert args to meta; "
                f"originally (*{args}, **{kwargs})") from e
    
    try:
        # 调用函数 func，传入原始的参数和关键字参数，并接收返回值 rs
        rs = func(*args, **kwargs)
    except Exception as e:
        # 如果函数调用过程中出现异常，抛出断言错误，指示原始 OpInfo 出现问题
        raise AssertionError("Original OpInfo is broken") from e

    # TODO: 也需要处理函数 func 抛出异常的情况

    # 目前只在所有张量类型成功转换的情况下尝试执行后续操作
    # （如果有任何转换失败，说明处于混合设备状态，这种情况支持有限）
# 定义一个异常类，用于表示测试失败的情况
class TestFailedError(Exception):
    pass

# 定义一个正则表达式对象，用于匹配实现未完成的错误信息
RE_NOT_IMPLEMENTED_MSG = re.compile(r"Could not run '([^']+)' with arguments ")

# 定义一个字典，记录各个 Torch 函数预期的数据类型失败情况
meta_function_expected_failures = {
    torch.Tensor.to_sparse : {f64, i32, c128, i64, i16, f16, u8, c64, bf16, b8, i8, f32},
    torch.allclose : {f64, f16, c128, c64, bf16, f32},
    torch.argwhere : {f64, i32, c128, i64, i16, f16, u8, c64, bf16, b8, i8, f32},
    torch.combinations : {f64, i32, c128, i64, i16, f16, u8, c64, bf16, b8, i8, f32},
    torch.corrcoef : {f64, i32, c128, i64, i16, u8, c64, bf16, f16, i8, f32},
    torch.cov : {f64, i32, c128, i64, i16, u8, c64, bf16, i8, f32, f16},
    torch.functional.istft : {f64, c64, c128, f32},
    torch.geqrf : {f64, c64, c128, f32},
    torch.masked_select : {f64, i32, c128, i64, i16, f16, u8, c64, bf16, b8, i8, f32},
    torch.nonzero : {f64, i32, c128, i64, i16, c32, f16, u8, c64, bf16, b8, i8, f32},
    torch.Tensor.nonzero : {f64, i32, c128, i64, i16, c32, f16, u8, c64, bf16, b8, i8, f32},
    torch.Tensor.item : {f64, i32, c128, i64, i16, f16, u8, c32, c64, bf16, b8, i8, f32},
    torch.bincount : {i32, i64, u8, i16, i8},
    torch.functional.unique : {f64, i32, i64, u8, i16, f16, bf16, b8, i8, f32, u16, u32, u64},
    torch.functional.unique_consecutive : {f64, i32, i64, u8, i16, f16, bf16, b8, i8, f32, u16, u32, u64},
    torch.histogram : {f64, f32},
    torch.histogramdd : {f64, f32},
    torch.kthvalue : {f64, i32, i64, u8, i16, f16, bf16, i8, f32},
    torch.nn.functional.ctc_loss : {f64, f32},
    torch.nn.functional.gaussian_nll_loss : {f16, f64, bf16, f32},
    torch.linalg.lstsq : {f64, f32, c128, c64},
}

# 定义一个字典，记录特定 Torch 函数在特定条件下预期的数据类型失败情况
meta_function_expected_failures_conditional = {
    torch.repeat_interleave : (lambda dtype, *args, **kwargs: not isinstance(kwargs.get("repeats", None), int)),
}

# 以下代码块以 YAML 格式导出上述两个字典，以便进行更易读写的处理
import yaml
# 将元数据函数的预期失败项映射为相应的跳过项，使用元数据函数名称解析器来解析键值
meta_function_skips = {
    # 对于 torch.Tensor.__rmatmul__，设置其跳过的数据类型集合
    torch.Tensor.__rmatmul__: {bf16, c128, f64, f32, f16, c64},
    # 对于 torch.Tensor.matmul，设置其跳过的数据类型集合
    torch.Tensor.matmul: {f64, f32, c128, c64},
    # 对于 torch.functional.atleast_2d，设置其跳过的数据类型集合
    torch.functional.atleast_2d: {bf16, i8, c32, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    # 对于 torch.functional.atleast_3d，设置其跳过的数据类型集合
    torch.functional.atleast_3d: {bf16, i8, c32, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    # 对于 torch.functional.cartesian_prod，设置其跳过的数据类型集合
    torch.functional.cartesian_prod: {bf16, i8, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    # 对于 torch.functional.einsum，设置其跳过的数据类型集合
    torch.functional.einsum: {bf16, c128, f64, f32, f16, c64},
    # 对于 torch.inner，设置其跳过的数据类型集合
    torch.inner: {f16, bf16, i8, i64, u8, c128, f64, i16, f32, i32, c64},
    # 对于 torch.linalg.matrix_norm，设置其跳过的数据类型集合
    torch.linalg.matrix_norm: {c128, f32, c64, f64},
    # 对于 torch.linalg.matrix_rank，设置其跳过的数据类型集合
    torch.linalg.matrix_rank: {c128, c64},
    # 对于 torch.linalg.svd，设置其跳过的数据类型集合
    torch.linalg.svd: {c128, c64},
    # 对于 torch.matmul，设置其跳过的数据类型集合
    torch.matmul: {bf16, c128, f64, f32, f16, c64},
    # 对于 torch.nanquantile，设置其跳过的数据类型集合
    torch.nanquantile: {f64, f32},
    # 对于 torch.narrow，设置其跳过的数据类型集合
    torch.narrow: {bf16, i8, i64, u8, c128, b8, f64, i16, i32, f32, f16, c32, c64},
    # 对于 torch.nn.functional.batch_norm，设置其跳过的数据类型集合
    torch.nn.functional.batch_norm: {f64, f32},
    # 对于 torch.nn.functional.binary_cross_entropy，设置其跳过的数据类型集合
    torch.nn.functional.binary_cross_entropy: {bf16, f64, f32, f16},
    # 对于 torch.nn.functional.dropout3d，设置其跳过的数据类型集合
    torch.nn.functional.dropout3d: {bf16, f64, f32, f16},
    # 对于 torch.nn.functional.local_response_norm，设置其跳过的数据类型集合
    torch.nn.functional.local_response_norm: {bf16, f64, f32, f16},
    # 对于 torch.svd，设置其跳过的数据类型集合
    torch.svd: {c128, c64},
    # 对于 torch.take_along_dim，设置其跳过的数据类型集合
    torch.take_along_dim: {bf16, i8, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    # 对于 torch.diff，设置其跳过的数据类型集合
    torch.diff: {b8},
    # 对于 torch.equal，设置其跳过的数据类型集合
    torch.equal: {bf16, i8, c32, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    # 对于 torch.nanmean，设置其跳过的数据类型集合
    torch.nanmean: {bf16, f64, f32, f16, c32, c64, c128},
    # 对于 torch.nn.functional.cross_entropy，设置其跳过的数据类型集合
    torch.nn.functional.cross_entropy: {bf16, f64, f32},
    # 对于 torch.nn.functional.nll_loss，设置其跳过的数据类型集合
    torch.nn.functional.nll_loss: {bf16, f64, f32},
    # 对于 torch.linalg.cond，设置其跳过的数据类型集合
    torch.linalg.cond: {c128, c64, f32, f64},
    # 对于 torch.linalg.vecdot，设置其跳过的数据类型集合
    torch.linalg.vecdot: {bf16, f64, f32, f16},
    # 对于 torch.empty，设置其跳过的数据类型集合
    torch.empty: {bf16, i8, c32, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    # 对于 torch.Tensor.addbmm_，设置其跳过的数据类型集合
    torch.Tensor.addbmm_: {bf16, c128, c64, f32, f64, i16, i32, i64, i8, u8},
    # 对于 torch.nn.functional.one_hot，设置其跳过的数据类型集合
    torch.nn.functional.one_hot: {i64},
}

# 创建元数据函数设备预期失败项和只限于非就地操作的默认字典
meta_function_device_expected_failures = defaultdict(dict)
meta_function_device_expected_failures_only_outplace = defaultdict(dict)
meta_function_device_skips = defaultdict(dict)

# 在 'cpu' 设备下设置元数据函数的预期失败项
meta_function_device_expected_failures['cpu'] = {
    # 对于 torch.native_batch_norm，设置其在 'cpu' 设备上预期失败的数据类型集合
    torch.native_batch_norm: {bf16, f16},
    # 对于 torch._native_batch_norm_legit，设置其在 'cpu' 设备上预期失败的数据类型集合
    torch._native_batch_norm_legit: {bf16, f16},
    # 对于 torch.ops.aten._batch_norm_with_update，设置其在 'cpu' 设备上预期失败的数据类型集合
    torch.ops.aten._batch_norm_with_update: {bf16, f16},
    # 对于 torch.native_layer_norm，设置其在 'cpu' 设备上预期失败的数据类型集合
    torch.native_layer_norm: {bf16, f16},
}

# 在 'cuda' 设备下设置元数据函数的预期失败项
meta_function_device_expected_failures['cuda'] = {
    # 对于 torch.corrcoef，设置其在 'cuda' 设备上预期失败的数据类型集合
    torch.corrcoef: {bf16, f16},  # aten::_local_scalar_dense
    # 对于 torch.cov，设置其在 'cuda' 设备上预期失败的数据类型集合
    torch.cov: {f16},  # aten::_local_scalar_dense
    # 对于 torch.functional.unique，设置其在 'cuda' 设备上预期失败的数据类型集合
    torch.functional.unique: {f16},  # aten::_unique2, aten::unique_dim
    # 对于 torch.functional.unique_consecutive，设置其在 'cuda' 设备上预期失败的数据类型集合
    torch.functional.unique_consecutive: {f16},  # aten::unique_consecutive
}
    torch.geqrf: {f32, f64},  # aten::geqrf
    torch.kthvalue: {f16},  # aten::kthvalue.values


    # 定义了两个键值对，分别表示 torch 模块中的函数和它们支持的数据类型
    # torch.geqrf: {f32, f64}, 表示 torch 模块中的 geqrf 函数支持 f32 和 f64 类型的数据
    # torch.kthvalue: {f16}, 表示 torch 模块中的 kthvalue 函数支持 f16 类型的数据
}

# 将 CPU 设备下的一些批量归一化操作添加到跳过列表中
meta_function_device_skips['cpu'] = {
    # TODO: 这些批量归一化操作的解压缩返回的数据类型依赖于设备。我们应该通过元张量更好地处理它们。
    torch.native_batch_norm: {f32, f64},
    torch._native_batch_norm_legit: {f32, f64},
    torch.ops.aten._batch_norm_with_update: {f32, f64},
}

# 将 CUDA 设备下的一些操作添加到跳过列表中
meta_function_device_skips['cuda'] = {
    torch.inner: {f16},
    torch.linalg.matrix_rank: {f32, f64},
    torch.linalg.svd: {f32, f64},
    torch.nn.functional.cross_entropy: {f16},
    torch.nn.functional.interpolate: {f16},
    torch.nn.functional.nll_loss: {f16},
    torch.svd: {f32, f64},
}

# 这是一个 __torch_function__ 模式，当启用时，会介入每个 Torch API 调用，并按正常方式运行运算符，
# 然后使用元输入重新运行它，最后检查输出的一致性。
# 大部分逻辑用于忠实地复制原始张量作为元张量，这并不简单，因为可能涉及许多子系统。
#
# 尽管如此，对于这个测试文件来说，这个类有点过于复杂（因为我本可以在 OpInfo 调用上内联 __torch_function__，
# 而 OpInfos 通常具有非常规则的输入），但它将在更全面的测试中非常有用，例如
# https://github.com/pytorch/pytorch/pull/75994。其主要好处是它比 torch dispatch 模式高效得多（尽管覆盖范围较小）。
class MetaCrossRefFunctionMode(torch.overrides.TorchFunctionMode):
    test_case: TestCase
    device_type: str
    dtype: torch.dtype

    def __init__(self, test_case, *, device, dtype, inplace):
        self.test_case = test_case
        self.device_type = torch.device(device).type
        self.dtype = dtype
        self.inplace = inplace
    # 实现自定义的 torch 函数调用协议 __torch_function__
    def __torch_function__(self, func, types, args=(), kwargs=None):
        # 确保 kwargs 不为 None
        kwargs = kwargs or {}

        # 如果处于 Torch 的跟踪模式下，或者 func 是 torch 脚本方法，或者
        # 当 no_dispatch() 开启时元转换器无法正确工作，因此在这种情况下跳过运行交叉引用测试
        if (
            torch.jit.is_tracing() or isinstance(func, torch.ScriptMethod) or
            torch._C._dispatch_tls_local_exclude_set().has(torch._C.DispatchKey.Python)
        ):
            return func(*args, **kwargs)

        # 根据数据类型和函数属性确定测试期望结果
        if self.dtype in meta_function_skips.get(func, set()):
            test_expect = TestExpect.SKIP
        elif self.dtype in meta_function_device_skips[self.device_type].get(func, set()):
            test_expect = TestExpect.SKIP
        elif self.dtype in meta_function_expected_failures.get(func, set()):
            test_expect = TestExpect.XFAILURE
        elif self.dtype in meta_function_device_expected_failures[self.device_type].get(func, set()):
            test_expect = TestExpect.XFAILURE
        elif meta_function_expected_failures_conditional.get(func, lambda *_, **__: False)(self.dtype, *args, **kwargs):
            test_expect = TestExpect.XFAILURE
        elif not self.inplace and \
                self.dtype in meta_function_device_expected_failures_only_outplace[self.device_type].get(func, set()):
            test_expect = TestExpect.XFAILURE
        else:
            test_expect = TestExpect.SUCCESS

        # 运行元交叉引用测试，并返回结果
        return run_meta_crossref(
            self.test_case, test_expect, func, args,
            kwargs, dtype=self.dtype, device_type=self.device_type, run_symbolic_meta=False
        )
# 定义了预期会失败的元分发函数集合，每个键值对表示一个函数和其失败的输入数据类型集合
meta_dispatch_expected_failures = {
    aten.allclose.default: {f16, bf16, f32, f64, c64, c128},  # NotImplementedError: 'aten::_local_scalar_dense'
    aten.geqrf.default : {c64, c128, f64, f32},
    aten.linalg_lstsq.default : {c64, c128, f64, f32},
    aten.masked_select.default : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten.masked_select.out : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten.nonzero.default : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, c32, b8, i16, u8},
    aten.nonzero.out : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, c32, b8, i16, u8},
    aten._to_sparse.default : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten._to_sparse.sparse_dim : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten._ctc_loss.Tensor : {f32, f64},  # Shape of second output depends on data.
    aten._histogramdd_bin_edges.default : {f32, f64},
    aten._histogramdd_from_bin_cts.default : {f32, f64},
    aten._histogramdd_from_bin_tensors.default : {f32, f64},
    aten._local_scalar_dense.default : {c32, c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten._unique2.default : {i8, f64, i64, f16, bf16, f32, i32, b8, i16, u8, u16, u32, u64},
    aten.bincount.default : {i64, i8, i32, i16, u8},
    aten.equal.default : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten.histogram.bin_ct : {f32, f64},
    aten.histogram.bins_tensor : {f32, f64},
    aten.kthvalue.default : {i8, f64, i64, f16, bf16, f32, i32, i16, u8},
    aten.unique_consecutive.default : {i8, f64, i64, f16, bf16, f32, i32, b8, i16, u8, u16, u32, u64},
    aten.unique_dim.default : {i8, f64, i64, f16, bf16, f32, i32, b8, i16, u8, u16, u32, u64},
    aten.upsample_nearest3d.vec : {bf16, f32, f64, u8},
}

# 定义了有时成功有时失败的元分发函数集合，每个键值对表示一个函数和其可能跳过的输入数据类型集合
meta_dispatch_skips = {
    aten.index.Tensor: {i64, bf16, f16, u8, b8, f32, i8, f64, i16, i32, c32, c64, c128},  # at::nonzero doesn't have a Meta function
    aten._to_copy.default: {i64, bf16, f16, u8, b8, f32, i8, f64, i16, i32, c32, c64, c128},
    aten.empty.memory_format: {b8, bf16, c128, c64, c32, f16, f32, f64, i16, i32, i64, i8, u8},
    aten.addbmm_.default: {bf16, c128, c64, f32, f64, i16, i32, i64, i8, u8},
}

# 对于在进入模式前可能失败的 CompositeImplicitAutograd 函数的集合
meta_dispatch_early_skips = set({
    torch.Tensor.float_power_,
    # Errors out in one of the tests, while ProxyTensor passes...
    torch.Tensor.cumprod_,
    torch.Tensor.cumsum_,
})

# 表示可能在原地操作中跳过的函数集合
meta_inplace_skips = set({
    # Errors out in one of the tests, while ProxyTensor passes...
    torch.Tensor.cumprod_,
    torch.Tensor.cumsum_,
})

# 初始化设备相关预期失败的元分发函数字典
meta_dispatch_device_expected_failures = defaultdict(dict)
meta_dispatch_device_skips = defaultdict(dict)

# 定义在 CPU 设备上预期会失败的元分发函数
meta_dispatch_device_expected_failures['cpu'] = {
    # TODO: The decomps for these batch norm ops return different dtypes depending
    # TODO: 这些批量归一化操作的分解返回的数据类型各不相同
}
    # 在设备上执行批量归一化操作。需要更好地支持元张量（meta tensors）。
    aten.native_batch_norm.default: {bf16, f16},
    # 在设备上执行合法的批量归一化操作。
    aten._native_batch_norm_legit.default: {bf16, f16},
    # 在设备上执行合法的批量归一化操作，但不计算统计信息。
    aten._native_batch_norm_legit.no_stats: {bf16, f16},
    # 使用更新执行批量归一化操作。
    aten._batch_norm_with_update.default: {bf16, f16},

    # 在设备上执行层归一化操作。
    aten.native_layer_norm.default: {bf16, f16},
}

# 将预期在 CUDA 设备上失败的操作及其对应的数据类型列表添加到字典中
meta_dispatch_device_expected_failures['cuda'] = {
    aten._unique2.default: {f16},  # aten::_unique2
    aten._use_cudnn_ctc_loss.default: {f32, f64},  # aten::_use_cudnn_ctc_loss
    aten._use_cudnn_ctc_loss.Tensor: {f32, f64},  # aten::_use_cudnn_ctc_loss.Tensor
    aten.cudnn_grid_sampler.default: {f16, f32, f64},  # aten::cudnn_grid_sampler
    aten.geqrf.default: {f32, f64},  # aten::geqrf
    aten.kthvalue.default: {f16},  # aten::kthvalue.values
    aten.linalg_eigvalsh.out: {f32, f64},  # aten::linalg_eigvalsh.out
    aten.log_sigmoid_forward.default: {bf16, f16, f64, f32},  # aten::log_sigmoid_forward.default
    aten.log_sigmoid_forward.output : {bf16, f16, f64, f32},  # aten::log_sigmoid_forward.output
    aten.unique_consecutive.default: {f16},  # aten::unique_consecutive
    aten.unique_dim.default: {f16},  # aten::unique_dim
    aten.upsample_nearest3d.vec: {f16},  # aten::upsample_nearest3d.vec
}

# 将在 CPU 设备上跳过的操作及其对应的数据类型列表添加到字典中
meta_dispatch_device_skips['cpu'] = {
    aten._embedding_bag_forward_only.default: {bf16, f16, f32, f64},  # aten::_embedding_bag_forward_only.default

    # 以下操作的批归一化运算在不同设备上返回的数据类型可能不同，需要与元张量更好地配合
    aten.native_batch_norm.default: {f32, f64},
    aten._native_batch_norm_legit.default: {f32, f64},
    aten._native_batch_norm_legit.no_stats: {f32, f64},
    aten._batch_norm_with_update.default: {f32, f64},

    # 如果计算的数据类型与输入数据类型不同，可能导致失败。CPU 执行的结果也可能与其他设备不同。
    aten.native_batch_norm.out: {bf16, f16, f32, f64},  # aten::native_batch_norm.out
}

# 将在 CUDA 设备上跳过的操作及其对应的数据类型列表添加到字典中
meta_dispatch_device_skips['cuda'] = {
    aten._conj.default: {c32, f16},  # file issue
    aten._linalg_svd.default: {c64, c128},  # aten::linalg_eigvalsh.out
    aten.cudnn_batch_norm.default: {f32, f64},
    aten.log_softmax.int : {c32, c64},  # aten::log_softmax.int
    aten.softmax.int : {c32, c64},  # aten::softmax.int

    # ROCm 相关内容；理论上应该是预期失败，但这不值得做；这些应该被统一
    aten.miopen_batch_norm.default: {f32},  # aten::miopen_batch_norm.default
}

# 定义一个函数，用于从参数列表中获取分步参数
def get_strided_args(args):
    # 定义一个函数，用于生成给定张量的各种步长变体
    def get_strided_variants(t, include_storage_offset=False):
        variants = []

        # 将原始张量添加到变体列表中（保持连续性）
        variants.append(t)

        # 如果张量的维度大于1，则生成其转置
        if t.ndim > 1:
            # 创建与转置后形状相反的空张量，设备和数据类型与原张量相同，允许梯度跟踪
            perm = list(reversed(range(t.ndim)))
            transposed = torch.empty(
                t.shape[::-1], device=t.device, dtype=t.dtype, requires_grad=t.requires_grad
            ).permute(perm).copy_(t)
            variants.append(transposed)

        # 如果张量的维度大于0，则生成非密集张量
        if t.ndim > 0:
            nondense = torch.repeat_interleave(t, 2, dim=-1)[..., ::2]
            variants.append(nondense)

        # 如果张量是4维的，则生成按通道优先（channels_last）的连续张量
        if t.ndim == 4:
            variants.append(t.contiguous(memory_format=torch.channels_last))

        # 如果张量是5维的，则生成按通道优先（channels_last_3d）的连续张量
        if t.ndim == 5:
            variants.append(t.contiguous(memory_format=torch.channels_last_3d))

        # 如果需要包含存储偏移，则生成具有指定存储偏移的缓冲区
        if include_storage_offset:
            buffer = torch.empty(t.numel() + 1, device=t.device, dtype=t.dtype, requires_grad=t.requires_grad)
            buffer = buffer.as_strided(t.shape, t.stride(), storage_offset=1)
            buffer.copy_(t)
            variants.append(buffer)

        return variants

    # 初始化一个空列表，用于存储所有参数的各种步长变体
    strided_args = []

    # 遍历每个参数
    for arg in args:
        # 如果参数是张量且非稀疏 CSR 格式且是连续的
        if isinstance(arg, torch.Tensor) and not arg.is_sparse_csr and arg.is_contiguous():
            # 获取该张量的各种步长变体
            strided_arg_variants = get_strided_variants(arg)
        else:
            # 否则，将参数本身作为其变体
            strided_arg_variants = [arg]

        # 将变体列表添加到参数列表中
        strided_args.append(strided_arg_variants)

    # 生成参数列表的笛卡尔积，并以生成器形式返回
    yield from itertools.product(*strided_args)
class MetaCrossRefDispatchMode(torch.utils._python_dispatch.TorchDispatchMode):
    # 定义元类 MetaCrossRefDispatchMode，继承自 TorchDispatchMode
    test_case: TestCase  # 声明实例变量 test_case，类型为 TestCase
    device: torch.device  # 声明实例变量 device，类型为 torch.device
    dtype: torch.dtype  # 声明实例变量 dtype，类型为 torch.dtype
    aten_olp_no_out_overload: set = set()  # 声明实例变量 aten_olp_no_out_overload，初始化为空集合

    def __init__(self, test_case, *, device, dtype, symbolic_meta: bool, inplace: bool, supports_out: bool):
        self.test_case = test_case  # 初始化实例变量 test_case，保存测试用例
        # 保存 TLS（Thread Local Storage，线程局部存储）
        self.precision = test_case.precision  # 初始化实例变量 precision，保存测试用例的精度
        self.rel_tol = test_case.rel_tol  # 初始化实例变量 rel_tol，保存测试用例的相对容差
        self.device_type = torch.device(device).type  # 初始化实例变量 device_type，保存设备类型
        self.dtype = dtype  # 初始化实例变量 dtype，保存数据类型
        self.symbolic_meta = symbolic_meta  # 初始化实例变量 symbolic_meta，标记是否使用符号元数据
        self.inplace = inplace  # 初始化实例变量 inplace，标记是否原地操作
        self.supports_out = supports_out  # 初始化实例变量 supports_out，标记是否支持输出参数

    @staticmethod
    # 尝试解析给定的操作重载函数，以确定是否有匹配的输出参数版本
    def try_resolve_aten_out_overload(ol, args, kwargs, num_outputs):
        # 获取操作重载函数的参数模式
        ol_args = ol._schema.arguments
        # 获取操作重载函数的包装信息
        olp: OpOverloadPacket = ol._overloadpacket

        # 如果该操作重载函数在禁止无输出参数的跨引用调度模式中，则返回空结果
        if olp in MetaCrossRefDispatchMode.aten_olp_no_out_overload:
            return (None, None, None)

        # 用于存储候选的带有输出参数的操作重载函数列表
        candidate_ols = []
        # 遍历所有候选的操作重载函数
        for candidate_ol_name in olp.overloads():
            candidate_ol = getattr(olp, candidate_ol_name)
            # 如果候选操作重载函数的任一参数被标记为输出参数，则将其添加到候选列表中
            if any(arg.is_out for arg in candidate_ol._schema.arguments):
                candidate_ols.append(candidate_ol)

        # 如果没有找到任何带有输出参数的候选操作重载函数，则将当前操作重载包添加到禁止列表中，并返回空结果
        if not candidate_ols:
            MetaCrossRefDispatchMode.aten_olp_no_out_overload.add(olp)
            return (None, None, None)

        # 现在基于传入参数、关键字参数以及所需输出数量进行匹配
        candidate_ol: OpOverload = None
        for candidate_ol in candidate_ols:
            candidate_ol_args = candidate_ol._schema.arguments

            # 如果传入参数的数量大于或等于候选操作重载函数的参数数量，则继续下一个候选函数的匹配
            if (len(args) >= len(candidate_ol_args)):
                continue

            # 对于位置参数，必须保证类型一致
            if not all(
                ol_args[pos_arg_ind].type == candidate_ol_args[pos_arg_ind].type
                for pos_arg_ind in range(len(args))
            ):
                continue

            # 输出参数的数量必须匹配
            candidate_out_names = [out_arg.name for out_arg in candidate_ol_args[-num_outputs:] if out_arg.is_out]
            if len(candidate_out_names) != num_outputs:
                continue

            # 尝试匹配关键字参数。只需确保剩余的关键字参数允许调用带有输出参数的重载函数
            new_kwargs = {}
            kwargs_match = True
            for arg in candidate_ol_args[len(args):-num_outputs]:
                if arg.name not in kwargs:
                    # 如果参数有默认值，则使用默认值
                    if arg.has_default_value():
                        new_kwargs[arg.name] = arg.default_value
                    # 对于可选类型的参数，使用默认策略赋值
                    elif isinstance(arg.type, torch.OptionalType):
                        if isinstance(arg.type.getElementType(), torch.BoolType):
                            new_kwargs[arg.name] = False
                        else:
                            new_kwargs[arg.name] = None
                    else:
                        kwargs_match = False
                        break
                else:
                    new_kwargs[arg.name] = kwargs[arg.name]

            # 如果关键字参数匹配成功，则返回匹配的操作重载函数、输出参数名称列表和新的关键字参数
            if kwargs_match:
                return candidate_ol, candidate_out_names, new_kwargs

        # 如果没有找到匹配的操作重载函数，则返回空结果
        return None, None, None
    # 获取预期的测试结果类型，根据当前数据类型和函数类型确定
    def _get_expected_test_result(self, func: OpOverload):
        # 如果当前数据类型在跳过的函数集合中，则测试期望为跳过
        if self.dtype in meta_dispatch_skips.get(func, set()):
            test_expect = TestExpect.SKIP
        # 如果当前数据类型在当前设备类型的跳过函数集合中，则测试期望为跳过
        elif self.dtype in meta_dispatch_device_skips[self.device_type].get(func, set()):
            test_expect = TestExpect.SKIP
        # 如果当前数据类型在预期失败的函数集合中，则测试期望为预期失败
        elif self.dtype in meta_dispatch_expected_failures.get(func, set()):
            test_expect = TestExpect.XFAILURE
        # 如果当前数据类型在当前设备类型的预期失败函数集合中，则测试期望为预期失败
        elif self.dtype in meta_dispatch_device_expected_failures[self.device_type].get(func, set()):
            test_expect = TestExpect.XFAILURE
        else:
            # 否则，测试期望为成功
            test_expect = TestExpect.SUCCESS
        return test_expect

    # Torch dispatch方法，处理torch操作的分发逻辑
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # 设置测试用例的精度和相对误差
        self.test_case.precision = self.precision
        self.test_case.rel_tol = self.rel_tol

        # 获取当前操作的预期测试结果
        test_expect = self._get_expected_test_result(func)

        # 运行元信息交叉引用，获取预期结果
        expected = run_meta_crossref(
            self.test_case,
            test_expect,
            func,
            args,
            kwargs,
            dtype=self.dtype,
            device_type=self.device_type,
            run_symbolic_meta=self.symbolic_meta,
        )

        # 对没有out参数但有aten op重载的torch操作进行测试
        if (
            not self.inplace and
            not self.supports_out and
            test_expect == TestExpect.SUCCESS and
            (torch.is_tensor(expected) or isinstance(expected, Iterable))
        ):
            # 尝试解析aten op的out参数重载
            num_outputs = 1 if torch.is_tensor(expected) else len(expected)
            func_out_overload, out_param_names, kwargs = self.try_resolve_aten_out_overload(func, args, kwargs, num_outputs)

            if func_out_overload:
                # 如果有out参数重载，则根据预期结果设置kwargs中的参数
                if num_outputs == 1:
                    kwargs[out_param_names[0]] = expected
                else:
                    for ind, out_param_name in enumerate(out_param_names):
                        kwargs[out_param_name] = expected[ind]

                # 获取out参数重载函数的预期测试结果
                test_expect = self._get_expected_test_result(func_out_overload)

                # 再次运行元信息交叉引用，验证out参数重载的预期结果
                run_meta_crossref(
                    self.test_case,
                    test_expect,
                    func_out_overload,
                    args,
                    kwargs,
                    dtype=self.dtype,
                    device_type=self.device_type,
                    run_symbolic_meta=self.symbolic_meta,
                )

        # 返回预期的测试结果
        return expected
# NB: we're running these tests only on CUDA because there are some
# inconsistencies between CUDA and CPU, and running on CUDA makes it easier
# to ignore the CPU case when inconsistencies arise.  Ideally we deal
# with the inconsistencies but this takes time.
@unMarkDynamoStrictTest
class TestMeta(TestCase):
    # Copies inputs to inplace operations to avoid inplace modifications
    # to leaves requiring gradient
    def _get_safe_inplace(self, inplace_variant):
        @wraps(inplace_variant)
        def _fn(t, *args, **kwargs):
            # If input t is a list, clone each element to avoid inplace modifications
            if isinstance(t, list):
                return inplace_variant([x.clone() for x in t], *args, **kwargs)
            else:
                # Clone tensor t to avoid inplace modifications
                return inplace_variant(t.clone(), *args, **kwargs)

        return _fn

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipIfCrossRef
    @suppress_warnings
    @ops(itertools.chain(op_db, foreach_op_db))
    def test_meta_outplace(self, device, dtype, op):
        # List of operation names to skip due to flakiness when testing with TorchDynamo
        skip_op_names = (
            "fft.ihfft",
            "fft.ihfft2",
            "linalg.lu_solve",
        )
        # Skip the test if TEST_WITH_TORCHDYNAMO is True and op.name is in skip_op_names
        if TEST_WITH_TORCHDYNAMO and op.name in skip_op_names:
            raise unittest.SkipTest("flaky")
        # Obtain the function implementing the operation
        func = op.get_op()
        # Generate sample inputs for the operation
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        # Iterate through each sample input
        for sample_input in samples:
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            # Run the function in MetaCrossRefFunctionMode to check consistency with regular mode
            with MetaCrossRefFunctionMode(self, dtype=dtype, device=device, inplace=False):
                # Compute the expected output
                expected = func(*args, **kwargs)
                # If the operation supports out parameter, also test with it
                if isinstance(expected, torch.Tensor) and op.supports_out:
                    func(*args, **kwargs, out=expected)

            # Special test for functions taking "device" kwarg and ending with "_like"
            # Ensure compatibility of these functions with "meta" tensors and their original device argument
            if "device" in kwargs and "_like" in op.name:
                with torch.random.fork_rng():
                    torch.manual_seed(123)
                    # Compute reference output
                    ref = func(*args, **kwargs)
                # Ensure args[0] is a Tensor for *_like functions
                assert isinstance(args[0], torch.Tensor)
                with torch.random.fork_rng():
                    torch.manual_seed(123)
                    # Switch args[0] to "meta" device and compute output
                    args[0] = args[0].to(device="meta")
                    meta = func(*args, **kwargs)

                # Special case: empty_like is not deterministic, so skip comparison
                if op.name != "empty_like":
                    # Assert equality of reference and meta outputs
                    self.assertEqual(ref, meta)
    # 定义测试方法，测试元数据的原位操作
    def test_meta_inplace(self, device, dtype, op):
        # 从操作中获取原位操作函数
        func = op.get_inplace()
        # 如果没有原位操作函数，则跳过测试
        if not func:
            self.skipTest("No inplace variable for this op")
        # 如果操作将整数提升为浮点数，但数据类型不是浮点数，则跳过测试
        if op.promotes_int_to_float and not dtype.is_floating_point:
            self.skipTest("Op promotes to float, which is impossible for inplace with non-float input")
        # 如果原位操作函数在元数据的原位跳过列表中，则跳过测试
        if func in meta_inplace_skips:
            self.skipTest("Skipped")
        # 获取安全的原位操作函数
        func = self._get_safe_inplace(func)
        # 生成操作的样本输入，不需要梯度
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        # 遍历每个样本输入
        for sample_input in samples:
            # 如果样本输入可以广播，则继续下一个循环
            if sample_input.broadcasts_input:
                continue
            # 构建函数调用所需的参数列表
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            # 使用元数据交叉引用函数模式执行函数调用
            with MetaCrossRefFunctionMode(self, dtype=dtype, device=device, inplace=True):
                expected = func(*args, **kwargs)

    # 运行元数据分派测试
    def _run_dispatch_meta_test(self, device, dtype, op, symbolic_meta, inplace, all_stride_variants=False):
        # 如果是原位操作
        if inplace:
            # 获取操作的原位函数
            func = op.get_inplace()
            # 如果没有原位操作函数，则跳过测试
            if not func:
                self.skipTest("No inplace variable for this op")
            # 如果操作将整数提升为浮点数，但数据类型不是浮点数，则跳过测试
            if op.promotes_int_to_float and not dtype.is_floating_point:
                self.skipTest("Op promotes to float, which is impossible for inplace with non-float input")
        else:
            # 否则获取操作的一般函数
            func = op.get_op()

        # 如果函数在元数据分派早期跳过列表中，则跳过测试
        if func in meta_dispatch_early_skips:
            self.skipTest("Function is in dispatch early skips")

        # 如果是原位操作，则获取安全的原位操作函数
        if inplace:
            func = self._get_safe_inplace(func)

        # 生成操作的样本输入，不需要梯度
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        # 遍历每个样本输入
        for sample_input in samples:
            # 如果是原位操作且样本输入可以广播，则继续下一个循环
            if inplace and sample_input.broadcasts_input:
                continue

            # 构建函数调用所需的参数列表
            sample_args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs

            # 如果启用所有步幅变体，并且样本参数中的张量数小于或等于5，则获取步幅参数的变体
            if all_stride_variants and sum(isinstance(arg, torch.Tensor) for arg in sample_args) <= 5:
                # 避免组合爆炸，获取步幅参数的变体
                strided_args = get_strided_args(sample_args)
            else:
                # 否则将样本参数作为单一变体
                strided_args = [sample_args]

            # 遍历所有的步幅参数变体
            for args in strided_args:
                # 使用元数据交叉引用分派模式推入堆栈，执行函数调用
                with MetaCrossRefDispatchMode.push(
                    self, dtype=dtype, device=device,
                    symbolic_meta=symbolic_meta, inplace=inplace,
                     supports_out=op.supports_out):
                    expected = func(*args, **kwargs)

                    # 如果不是原位操作且返回值是张量，并且操作支持输出，则再次调用函数以输出期望的张量
                    if not inplace and isinstance(expected, torch.Tensor) and op.supports_out:
                        func(*args, **kwargs, out=expected)


    # 跳过 ASAN 环境下的测试
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    # 跨引用测试跳过装饰器
    @skipIfCrossRef
    # 抑制警告的装饰器
    @suppress_warnings
    # 对操作数据库和每个操作数据库的测试
    @ops(itertools.chain(op_db, foreach_op_db))
    # 测试分派元数据的非原位操作
    def test_dispatch_meta_outplace(self, device, dtype, op):
        # 运行元数据分派测试，不使用符号元数据，不进行原位操作
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=False, inplace=False)

    # 跳过 ASAN 环境下的测试
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    # 使用装饰器跳过指定条件的测试：跳过交叉引用的测试
    @skipIfCrossRef
    # 使用装饰器抑制警告
    @suppress_warnings
    # 使用装饰器注册操作函数，使用itertools.chain将两个操作列表连接起来
    @ops(itertools.chain(op_db, foreach_op_db))
    # 测试方法：测试在指定设备上以原地操作的方式分发元信息
    def test_dispatch_meta_inplace(self, device, dtype, op):
        # 调用内部方法执行分发元信息测试，设置为非符号化元信息，以原地操作的方式
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=False, inplace=True)

    # 使用装饰器跳过特定条件下的测试：在AddressSanitizer下跳过测试
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    # 使用装饰器跳过指定条件的测试：跳过交叉引用的测试
    @skipIfCrossRef
    # 使用装饰器抑制警告
    @suppress_warnings
    # 使用装饰器注册操作函数，使用itertools.chain将两个操作列表连接起来
    @ops(itertools.chain(op_db, foreach_op_db))
    # 测试方法：测试在指定设备上以非原地操作的方式分发符号化元信息
    def test_dispatch_symbolic_meta_outplace(self, device, dtype, op):
        # 调用内部方法执行分发元信息测试，设置为符号化元信息，以非原地操作的方式
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=True, inplace=False)

    # 使用装饰器跳过特定条件下的测试：在AddressSanitizer下跳过测试
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    # 使用装饰器跳过指定条件的测试：跳过交叉引用的测试
    @skipIfCrossRef
    # 使用装饰器抑制警告
    @suppress_warnings
    # 使用装饰器注册操作函数，使用itertools.chain将两个操作列表连接起来
    @ops(itertools.chain(op_db, foreach_op_db))
    # 测试方法：测试在指定设备上以原地操作的方式分发符号化元信息
    def test_dispatch_symbolic_meta_inplace(self, device, dtype, op):
        # 调用内部方法执行分发元信息测试，设置为符号化元信息，以原地操作的方式
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=True, inplace=True)

    # 使用装饰器跳过特定条件下的测试：在AddressSanitizer下跳过测试
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    # 使用装饰器跳过指定条件的测试：跳过交叉引用的测试
    @skipIfCrossRef
    # 使用装饰器抑制警告
    @suppress_warnings
    # 仅测试一个数据类型，因为所有数据类型的输出步幅行为相同
    @ops(itertools.chain(op_db, foreach_op_db), dtypes=OpDTypes.any_common_cpu_cuda_one)
    # 仅在CUDA上进行测试，因为CUDA核心的步幅是参考值
    @onlyCUDA
    # 测试方法：测试在指定设备上以非原地操作的方式分发符号化元信息的所有步幅变体
    def test_dispatch_symbolic_meta_outplace_all_strides(self, device, dtype, op):
        # 调用内部方法执行分发元信息测试，设置为符号化元信息，以非原地操作的方式，同时测试所有步幅变体
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=True, inplace=False, all_stride_variants=True)

    # 使用装饰器跳过特定条件下的测试：在AddressSanitizer下跳过测试
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    # 使用装饰器跳过指定条件的测试：跳过交叉引用的测试
    @skipIfCrossRef
    # 使用装饰器抑制警告
    @suppress_warnings
    # 仅测试一个数据类型，因为所有数据类型的输出步幅行为相同
    @ops(itertools.chain(op_db, foreach_op_db), dtypes=OpDTypes.any_common_cpu_cuda_one)
    # 仅在CUDA上进行测试，因为CUDA核心的步幅是参考值
    @onlyCUDA
    # 测试方法：测试在指定设备上以原地操作的方式分发符号化元信息的所有步幅变体
    def test_dispatch_symbolic_meta_inplace_all_strides(self, device, dtype, op):
        # 调用内部方法执行分发元信息测试，设置为符号化元信息，以原地操作的方式，同时测试所有步幅变体
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=True, inplace=True, all_stride_variants=True)

    # 使用装饰器跳过特定条件下的测试：在AddressSanitizer下跳过测试
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    # 使用装饰器跳过指定条件的测试：跳过交叉引用的测试
    @skipIfCrossRef
    # 使用装饰器抑制警告
    @suppress_warnings
    # 仅测试一个数据类型，因为所有数据类型的输出步幅行为相同
    @ops(binary_ufuncs, allowed_dtypes=(torch.float32,))
    # 仅在CUDA上进行测试，因为CUDA核心的步幅是参考值
    @onlyCUDA
    # 测试方法：测试混合数据类型的二元通用函数
    def test_binary_ufuncs_mixed_dtype(self, device, dtype, op):
        # 部分函数生成器，生成输入样例
        make_arg = partial(
            make_tensor,
            device=device,
        )

        # 定义样例输入函数
        def sample_input(op, device, dtype, requires_grad, **kwargs):
            # 生成并返回输入样例
            yield SampleInput(
                make_arg((S,), dtype=dtype), make_arg((S,), dtype=torch.float16)
            )

        # 复制操作对象
        op = copy.copy(op)
        # 设置操作对象的样例输入生成函数
        op.sample_inputs_func = sample_input

        # 调用内部方法执行分发元信息测试，设置为符号化元信息，以非原地操作的方式
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=True, inplace=False)

    # 测试方法：测试创建一个空的量化张量
    def test_empty_quantized(self):
        # 创建一个空的量化张量，设备为'meta'，数据类型为torch.qint8
        r = torch.empty(2 ** 52, device='meta', dtype=torch.qint8)
        # 断言：验证返回的张量设备类型为'meta'
        self.assertEqual(r.device.type, 'meta')
    # 定义测试函数，测试 torch.tensor.nan_to_num() 方法
    def test_nan_to_num(self):
        # 创建包含 NaN、Infinity 和 -Infinity 的张量，并指定设备为 'meta'
        t = torch.tensor([float('nan'), float('inf'), -float('inf'), 3.14], device='meta')
        # 对张量进行 nan_to_num 处理
        r = t.nan_to_num()
        # 断言处理后张量的设备类型为 'meta'
        self.assertEqual(r.device.type, 'meta')

    # 定义测试函数，测试 torch.Tensor.masked_fill_() 方法在不匹配广播时抛出 RuntimeError
    def test_inplace_masked_fill_error(self):
        # 创建形状为 (3, 3) 的随机张量，并指定设备为 'meta'
        t = torch.randn(3, 3, device='meta')
        # 使用断言捕获 RuntimeError 异常，检查错误消息是否包含指定字符串
        with self.assertRaisesRegex(RuntimeError, "doesn't match the broadcast"):
            # 尝试在不匹配广播条件下使用 masked_fill_() 方法
            t.masked_fill_((t > 0).unsqueeze(0), 0.1)

    # 定义测试函数，测试多种 inplace 二元操作方法在不匹配广播时抛出 RuntimeError
    def test_inplace_bin_ops_error(self):
        # 创建形状为 (3, 3) 的随机张量，并指定设备为 'meta'
        t = torch.randn(3, 3, device='meta')
        # 遍历多个二元操作方法
        for op in (torch.Tensor.add_, torch.Tensor.sub_, torch.Tensor.mul_, torch.Tensor.div_,
                   torch.Tensor.logical_and_, torch.Tensor.logical_or_, torch.Tensor.logical_xor_):
            # 使用断言捕获 RuntimeError 异常，检查错误消息是否包含指定字符串
            with self.assertRaisesRegex(RuntimeError, "doesn't match the broadcast"):
                # 尝试在不匹配广播条件下使用当前二元操作方法
                op(t, t.clone().unsqueeze(0))

    # 标记仅在 CPU 上运行的测试函数
    @onlyCPU
    def test_meta_autograd_no_error(self):
        # 使用 torch.library._scoped_library 在测试中注册自定义库和实现
        with torch.library._scoped_library("meta_test", "DEF") as lib:
            with torch.library._scoped_library("meta_test", "IMPL", "CPU") as impl_cpu:
                with torch.library._scoped_library("meta_test", "IMPL", "Meta") as impl_meta:
                    # 定义一个简单的函数实现，用于注册到自定义库中
                    def foo_impl(x):
                        return x + 1

                    # 在库中定义自定义函数 foo(Tensor a) -> Tensor
                    lib.define("foo(Tensor a) -> Tensor")
                    # 在 Meta 实现中注册 foo_impl 函数
                    impl_meta.impl("foo", foo_impl)
                    # 在 CPU 实现中注册 foo_impl 函数
                    impl_cpu.impl("foo", foo_impl)

                    # 创建形状为 (2,) 的张量 a，并指定设备为 'meta'
                    a = torch.ones(2, device='meta')
                    # 测试点是确认以下操作不会引发错误：
                    # 我们已经注册了一个 fallback 内核到 AutogradMeta 关键字，所以即使 `foo()` 没有自动求导内核也没问题。
                    # 调用注册的自定义操作 foo.default(a)
                    b = torch.ops.meta_test.foo.default(a)

    # 定义测试函数，测试 torch.ops.aten.huber_loss_backward() 函数的反向传播
    def test_huber_loss_backward(self):
        # 创建三个形状为 (2^52,) 的随机张量列表，并指定设备为 'meta'
        inps = [torch.rand(2**52, device='meta') for _ in range(3)]
        # 调用 torch.ops.aten.huber_loss_backward() 函数计算损失反向传播
        r = torch.ops.aten.huber_loss_backward(*inps, 0, 1.0)
        # 断言结果张量的设备类型为 'meta' 和形状与第一个输入张量相同
        self.assertEqual(r.device.type, 'meta')
        self.assertEqual(r.shape, inps[0].shape)
    # 定义一个辅助方法用于测试反向传播的规范化操作
    def _norm_backwards_test_helper(self, op, args, output_mask, expected_shapes):

        # 设置数据类型为 float32
        dtype = torch.float32
        # 设置设备为 "meta"
        device = "meta"

        # 测试函数调用
        grads = op(*args, output_mask)

        # 定义一个断言函数，用于验证结果张量的形状是否与期望一致
        def assertEqualShapes(res, exp):
            self.assertIsNone(res) if exp is None else self.assertEqual(exp, res.shape)

        # 验证每个梯度张量的形状
        assertEqualShapes(grads[0], expected_shapes[0])
        assertEqualShapes(grads[1], expected_shapes[1])
        assertEqualShapes(grads[2], expected_shapes[2])

        # 准备用于输出参数的 kwargs 字典，每个输出参数初始化为空张量
        out_kwargs = {
            f"out{i}": torch.empty(0, device=device, dtype=dtype)
            for i in range(len(output_mask))
        }

        # 测试带有输出参数的函数调用
        grads = op(*args, output_mask, **out_kwargs)

        # 定义另一个断言函数，用于验证输出参数的形状是否与期望一致
        def assertEqualShapes(res, exp):
            self.assertEqual(exp, res.shape) if exp is not None else True

        # 验证每个输出参数的形状
        assertEqualShapes(out_kwargs["out0"], expected_shapes[0])
        assertEqualShapes(out_kwargs["out1"], expected_shapes[1])
        assertEqualShapes(out_kwargs["out2"], expected_shapes[2])

    # 使用 onlyCPU 装饰器和参数化装饰器 parametrize 对 layer_norm_backward 函数进行测试
    @onlyCPU
    @parametrize("output_mask", list(itertools.product([True, False], [True, False], [True, False])))
    def test_layer_norm_backward(self, output_mask):
        # 导入需要的函数
        from torch.testing._internal.common_methods_invocations import sample_inputs_layer_norm

        # 设置设备为 "meta"
        device = "meta"
        # 设置数据类型为 float32
        dtype = torch.float32

        # 从样本函数中获取规范化输入样本，不需要梯度
        samples = sample_inputs_layer_norm(None, device, dtype, requires_grad=False)

        # 遍历每个样本
        for sample in samples:
            with self.subTest(sample=sample):
                # 处理可选的权重和偏置参数
                if len(sample.args) != 3:
                    sample.args = (*sample.args, *([None] * (3 - len(sample.args))))

                # 创建梯度输出张量，与输入样本形状相同
                grad_out = torch.ones_like(sample.input)
                normalized_shape, weight, bias = sample.args
                ndims_after_reduction = sample.input.ndim - len(normalized_shape)
                mean_shape = grad_out.shape[:ndims_after_reduction]
                mean = torch.zeros(mean_shape, device=device, dtype=dtype)
                rstd = torch.zeros(mean_shape, device=device, dtype=dtype)

                # 预期的输出形状，根据输出掩码和是否存在权重和偏置来确定
                expected_shapes = (
                    sample.input.shape if output_mask[0] else None,
                    weight.shape if output_mask[1] and weight is not None else None,
                    bias.shape if output_mask[2] and bias is not None else None)

                # 组装参数列表
                args = [grad_out, sample.input, normalized_shape, mean, rstd, weight, bias]

                # 调用 _norm_backwards_test_helper 方法进行测试
                self._norm_backwards_test_helper(torch.ops.aten.native_layer_norm_backward,
                                                 args, output_mask, expected_shapes)
    # 定义一个测试方法，用于测试组归一化反向传播函数
    def test_group_norm_backward(self, output_mask):
        # 导入测试所需的方法和模块
        from torch.testing._internal.common_methods_invocations import sample_inputs_group_norm

        # 设置设备为"meta"
        device = "meta"
        # 设置数据类型为torch.float32
        dtype = torch.float32
        # 生成一组输入样本，不需要梯度计算
        samples = sample_inputs_group_norm(None, device, dtype, requires_grad=False)

        # 遍历每个样本
        for sample in samples:
            # 使用子测试来执行每个样本的测试
            with self.subTest(sample=sample):
                # 创建一个梯度为1的tensor，形状与输入样本相同
                grad_out = torch.ones_like(sample.input)
                # 获取输入样本的形状信息
                N, C = sample.input.shape[:2]
                # 计算输入样本除了batch和channel维度之外的所有元素的乘积
                HxW = torch.prod(torch.as_tensor(sample.input.shape[2:]), dtype=torch.int32).item()
                # 获取样本参数中的组数
                group = sample.args[0]
                # 创建一个全零tensor，用作均值，设备和数据类型与输入一致
                mean = torch.zeros((N, group), device=device, dtype=dtype)
                # 创建一个全零tensor，用作逆标准差，设备和数据类型与输入一致
                rstd = torch.zeros((N, group), device=device, dtype=dtype)
                # 创建一个全零tensor，用作权重，设备和数据类型与输入一致
                weight = torch.zeros((C), device=device, dtype=dtype)

                # 构造参数列表
                args = [grad_out, sample.input, mean, rstd, weight, N, C, HxW, group]

                # 期望的输出形状，根据output_mask确定
                expected_shapes = (
                    sample.input.shape if output_mask[0] else None,
                    weight.shape if output_mask[1] else None,
                    weight.shape if output_mask[2] else None)

                # 调用_norm_backwards_test_helper方法，测试函数调用
                self._norm_backwards_test_helper(torch.ops.aten.native_group_norm_backward,
                                                 args, output_mask, expected_shapes)

    # 仅在CPU上运行此测试
    @onlyCPU
    # 参数化装饰器，生成不同的output_mask组合进行测试
    @parametrize("output_mask", list(itertools.product([True], [True, False], [True, False])))
    # 定义测试方法：验证批标准化的反向传播
    def test_batch_norm_backward(self, output_mask):
        # 导入批标准化的样本输入生成器
        from torch.testing._internal.common_methods_invocations import sample_inputs_batch_norm

        # 指定设备为"meta"
        device = "meta"
        # 指定数据类型为32位浮点数
        dtype = torch.float32
        # 生成批标准化的样本输入
        samples = sample_inputs_batch_norm(None, device, dtype, requires_grad=False)

        # 遍历每个样本
        for sample in samples:
            # 使用子测试对当前样本进行测试
            with self.subTest(sample=sample):

                # 如果样本输入的维度小于2，则跳过本次循环
                if sample.input.dim() < 2:
                    continue

                # 创建梯度输出，其形状与样本输入相同
                grad_out = torch.ones_like(sample.input)
                # 解包样本的参数：running_mean, running_var, weight, bias
                running_mean, running_var, weight, bias = sample.args
                # 检查是否处于训练模式
                train = sample.kwargs.get("training", True)
                # 如果处于训练模式，创建保存均值和标准差的张量
                save_mean = torch.zeros((sample.input.shape[1],), device=device, dtype=dtype) if train else None
                save_invstd = torch.zeros((sample.input.shape[1],), device=device, dtype=dtype) if train else None

                # 组装参数列表
                args = [grad_out, sample.input, weight, running_mean, running_var,
                        save_mean, save_invstd, train, sample.kwargs.get("eps", 1e-5)]

                # 预期的输出形状
                expected_shapes = (
                    sample.input.shape,
                    torch.Size([sample.input.shape[1]]) if output_mask[1] else None,
                    torch.Size([sample.input.shape[1]]) if output_mask[2] else None)

                # 调用_norm_backwards_test_helper进行批标准化的反向传播测试
                self._norm_backwards_test_helper(torch.ops.aten.native_batch_norm_backward,
                                                 args, output_mask, expected_shapes)

    # 定义测试方法：验证aten.fill_的别名关系
    def test_fill__alias_relationship(self):
        # 生成一个非常大的随机张量，指定设备为'meta'
        inps = torch.rand(2**52, device='meta')
        # 调用aten.fill_进行填充操作，返回结果张量r
        r = torch.ops.aten.fill_(inps, 1.0)
        # 断言：aten.fill_返回的是输入张量的别名
        self.assertEqual(id(inps), id(r))

        # 调用aten.fill进行填充操作，返回结果张量r2
        r2 = torch.ops.aten.fill(inps, 1.0)
        # 断言：aten.fill返回的是一个新的张量
        self.assertNotEqual(id(inps), id(r2))
    def test_meta__fused_moving_avg_obs_fq_helper(self, device):
        # 导入 FusedMovingAvgObsFakeQuantize 类
        from torch.ao.quantization import FusedMovingAvgObsFakeQuantize
        # 创建 MetaConverter 实例
        to_meta = MetaConverter()

        # 生成一个随机张量 x，形状为 5x5，位于指定设备上
        x = torch.randn(5, 5, device=device)
        # 创建一个表示正无穷大的张量 running_min_op，位于指定设备上
        running_min_op = torch.tensor(float("inf"), device=device)
        # 创建一个表示负无穷大的张量 running_max_op，位于指定设备上
        running_max_op = torch.tensor(float("-inf"), device=device)
        # 创建一个标量表示 avg_const 为 0.01
        avg_const = 0.01
        # 创建一个张量 scale，值为 [1.0]，位于指定设备上
        scale = torch.tensor([1.0], device=device)
        # 创建一个整数张量 zero_point，值为 [0]，数据类型为 torch.int，位于指定设备上
        zero_point = torch.tensor([0], dtype=torch.int, device=device)

        # 创建 FusedMovingAvgObsFakeQuantize 实例 mod
        mod = FusedMovingAvgObsFakeQuantize()
        # 启用 mod 的伪量化
        torch.ao.quantization.enable_fake_quant(mod)
        # 启用 mod 的观察器
        torch.ao.quantization.enable_observer(mod)
        # 将 mod 移动到指定设备上
        mod.to(device)

        # 将输入张量 x 转换为元数据形式
        meta_x = to_meta(x)

        # 构建参数列表 args
        args = [
            x,
            mod.observer_enabled,
            mod.fake_quant_enabled,
            running_min_op,
            running_max_op,
            scale,
            zero_point,
            avg_const,
            0,
            255,
            0,
        ]

        # 复制参数列表 args 到 meta_args
        meta_args = args.copy()
        # 将 meta_args 的第一个元素替换为 meta_x
        meta_args[0] = meta_x

        # 构建关键字参数列表 kwargss
        kwargss = [
            {},
            {"per_row_fake_quant": False, "symmetric_quant": False},
            {"per_row_fake_quant": False, "symmetric_quant": True},
        ]

        # 遍历关键字参数列表 kwargss
        for kwargs in kwargss:
            # 调用 aten._fused_moving_avg_obs_fq_helper.default 方法，使用 args 和 kwargs
            ref_out = aten._fused_moving_avg_obs_fq_helper.default(*args, **kwargs)
            # 调用 aten._fused_moving_avg_obs_fq_helper.default 方法，使用 meta_args 和 kwargs
            meta_out = aten._fused_moving_avg_obs_fq_helper.default(*meta_args, **kwargs)

            # 断言 meta_out 的第一个张量形状与 ref_out 的第一个张量形状相同
            self.assertEqual(ref_out[0].size(), meta_out[0].size())
            # 断言 meta_out 的第一个张量步长与 ref_out 的第一个张量步长相同
            self.assertEqual(ref_out[0].stride(), meta_out[0].stride())
            # 断言 meta_out 的第二个张量形状与 ref_out 的第二个张量形状相同
            self.assertEqual(ref_out[1].size(), meta_out[1].size())
            # 断言 meta_out 的第二个张量步长与 ref_out 的第二个张量步长相同
            self.assertEqual(ref_out[1].stride(), meta_out[1].stride())

    def test_cdist_forward(self, device):
        # 创建 MetaConverter 实例
        to_meta = MetaConverter()
        # 生成一个随机张量 x1，形状为 [3, 2]，位于指定设备上
        x1 = torch.rand([3, 2], device=device)
        # 生成一个随机张量 x2，形状为 [2, 2]，位于指定设备上
        x2 = torch.rand([2, 2], device=device)
        # 设置 p 为 2.0
        p = 2.0
        # 遍历计算模式 compute_mode 的值 (None, 1, 2)
        for compute_mode in (None, 1, 2):
            # 调用 aten._cdist_forward.default 方法，使用 x1, x2, p 和 compute_mode
            ref = aten._cdist_forward.default(x1, x2, p, compute_mode)
            # 调用 aten._cdist_forward.default 方法，使用 to_meta(x1), to_meta(x2), p 和 compute_mode
            res = aten._cdist_forward.default(to_meta(x1), to_meta(x2), p, compute_mode)
            # 断言 res 的设备类型为 'meta'
            self.assertEqual(res.device.type, 'meta')
            # 断言 ref 的形状与 res 的形状相同
            self.assertEqual(ref.shape, res.shape)
    # 定义测试函数，用于测试量化的嵌入包功能
    def test_quantized_embedding_bag(self):
        # 定义表的形状
        tab_shape = [8, 128]
        # 从表的形状中提取嵌入大小、索引长度和偏移长度
        emb_size, ind_len, off_len = tab_shape[0], 32, 33
        # 创建一个随机填充的浮点数表格，并转换为张量
        f_table = torch.from_numpy((np.random.random_sample(tab_shape) + 1).astype(np.float32))
        # 使用量化的嵌入包字节预打包函数对表格进行量化
        q_table = torch.ops.quantized.embedding_bag_byte_prepack(f_table)
        # 创建随机整数索引，并转换为张量
        indices = torch.from_numpy(np.random.randint(low=0, high=emb_size, size=ind_len)).int()
        # 计算索引长度除以偏移长度减一得到的最大长度
        max_length = len(indices) // (off_len - 1)
        # 如果最大长度超过20，则限制为20
        if max_length > 20:
            max_length = 20
        # 创建随机整数数组作为偏移长度，并转换为张量
        np_lengths = np.random.randint(0, max_length + 1, size=off_len - 1).astype(np.int32)
        offsets = torch.cat([torch.zeros([1]), torch.cumsum(torch.from_numpy(np_lengths), 0)]).int()

        # 使用量化的按行偏移量嵌入包字节函数计算嵌入向量
        eb = torch.ops.quantized.embedding_bag_byte_rowwise_offsets(
            q_table.to(device="meta"),  # 量化表格，转换到特定设备
            indices.to(device="meta"),  # 索引，转换到特定设备
            offsets.to(device="meta"),  # 偏移，转换到特定设备
            mode=0,  # 操作模式，0代表求和
            per_sample_weights=None,  # 每个样本的权重，默认为None
            include_last_offset=True,  # 是否包含最后一个偏移
        )
        # 断言嵌入向量的形状为[32, 128]
        self.assertEqual(eb.shape, [32, 128])
        # 断言嵌入向量的数据类型为torch.float32
        self.assertEqual(eb.dtype, torch.float32)
        # 断言嵌入向量的未命名存储数据指针为0
        self.assertEqual(eb.untyped_storage().data_ptr(), 0)

    # 测试均值和最大值。
    # 无法轻松测试求和，因为求和有一个快速路径，可能导致不分配offset2bag...但是反向函数需要它，
    # 并且offset2bag计算直接位于derivatives.yaml公式中，因此无法访问它。
    # 要测试求和，需要手动计算offset2bag
    @parametrize("mode", [1, 2])
    # 定义测试嵌入包稠密反向计算函数，参数为mode
    def test_embedding_bag_dense_backward(self, mode):
        # 创建随机张量作为权重，并启用梯度跟踪
        weight = torch.randn(4, 3, requires_grad=True)
        # 创建示例索引张量
        indices = torch.tensor([1, 0, 2, 1, 3])
        # 创建示例偏移张量
        offsets = torch.tensor([0, 2, 3, 5])
        # 指定梯度按频率缩放为False
        scale_grad_by_freq = False
        # 指定稀疏性为False
        sparse = False
        # 指定每个样本的权重为None
        per_sample_weights = None
        # 指定是否包含最后一个偏移为False
        include_last_offset = False
        # 指定填充索引为-1
        padding_idx = -1

        # 调用默认的嵌入包函数，计算输出、offset2bag、袋子大小和最大索引
        output, offset2bag, bag_size, maximum_indices = torch.ops.aten._embedding_bag.default(
            weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx
        )
        # 创建与输出相同形状的随机梯度张量
        grad = torch.randn_like(output)

        # 调用默认的稠密嵌入包反向计算函数，计算权重的梯度
        grad_weight = torch.ops.aten._embedding_bag_dense_backward.default(
            grad, indices, offset2bag, bag_size, maximum_indices, weight.size(0),
            scale_grad_by_freq, mode, per_sample_weights, padding_idx
        )
        # 转换到meta设备后再次调用函数，计算权重的梯度
        meta_grad_weight = torch.ops.aten._embedding_bag_dense_backward.default(
            grad.to('meta'), indices.to('meta'), offset2bag.to('meta'), bag_size.to('meta'),
            maximum_indices.to('meta'), weight.size(0),
            scale_grad_by_freq, mode, per_sample_weights, padding_idx
        )
        # 断言meta设备的梯度与原始梯度相等
        self.assertEqual(grad_weight.to('meta'), meta_grad_weight)
    def test_embedding_bag_dense_backward_per_sample_weights(self):
        # 创建一个形状为 (4, 3) 的随机张量，需要计算梯度
        weight = torch.randn(4, 3, requires_grad=True)
        # 指定索引张量，表示要提取的嵌入向量的索引
        indices = torch.tensor([1, 0, 2, 1, 3])
        # 指定偏移张量，表示每个样本在输出中的起始偏移
        offsets = torch.tensor([0, 2, 3, 5])
        # 设置是否按频率缩放梯度，默认为 False
        scale_grad_by_freq = False
        # 设置稀疏模式，默认为 False
        sparse = False
        # 设置嵌入操作模式，这里为 0 表示使用累加
        mode = 0
        # 创建一个形状为 (5,) 的随机张量，需要计算梯度，表示每个样本的权重
        per_sample_weights = torch.randn(5, requires_grad=True)
        # 是否包含最后一个偏移，默认为 False
        include_last_offset = False
        # 指定填充索引，默认为 -1
        padding_idx = -1

        # 调用自定义的嵌入包操作，返回输出、偏移到袋子的映射、每个袋子的大小和最大索引
        output, offset2bag, bag_size, maximum_indices = torch.ops.aten._embedding_bag.default(
            weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx
        )
        # 创建一个与 output 形状相同的随机张量，用作梯度输入
        grad = torch.randn_like(output)

        # 调用带有样本权重的嵌入包反向传播函数，计算权重的梯度
        grad_weight = torch.ops.aten._embedding_bag_per_sample_weights_backward.default(
            grad, weight, indices, offsets, offset2bag, mode, padding_idx
        )
        # 使用 meta 设备调用函数，计算元数据权重的梯度
        meta_grad_weight = torch.ops.aten._embedding_bag_per_sample_weights_backward.default(
            grad.to('meta'), weight.to('meta'), indices.to('meta'),
            offsets.to('meta'), offset2bag.to('meta'), mode, padding_idx
        )
        # 断言两种方法计算得到的权重梯度相等
        self.assertEqual(grad_weight.to('meta'), meta_grad_weight)

    # opinfo test is using aten.fill_, it's not testing aten.fill
    @onlyCUDA
    def test_fill_stride(self):
        # 创建 MetaConverter 实例
        to_meta = MetaConverter()
        # 创建一个样本参数列表
        sample_args = [torch.rand(2, 2, 2, 2), 1.0]

        # 遍历使用步幅参数的参数列表
        for args in get_strided_args(sample_args):
            # 将参数转换为 meta 类型
            meta_args = to_meta(args)
            # 使用普通设备调用 fill 函数，得到参考输出
            ref_out = torch.ops.aten.fill(*args)
            # 使用 meta 设备调用 fill 函数，得到 meta 输出
            meta_out = torch.ops.aten.fill(*meta_args)
            # 断言参考输出和 meta 输出的形状相同
            self.assertEqual(ref_out.size(), meta_out.size())
            # 断言参考输出和 meta 输出的步幅相同
            self.assertEqual(ref_out.stride(), meta_out.stride())

    def test_map_location_deserialize(self):
        # 导入 io 模块
        import io

        # 创建一个形状为 (10,) 的随机张量
        t = torch.rand(10)
        # 创建一个字节流对象
        b = io.BytesIO()

        # 将张量 t 保存到字节流中
        torch.save(t, b)
        # 将字节流指针移动到起始位置
        b.seek(0)
        # 从字节流中加载张量 r，使用 meta 设备映射位置
        r = torch.load(b, map_location=torch.device("meta"))
        # 断言加载的张量 r 的设备类型为 'meta'
        self.assertEqual(r.device.type, 'meta')
        # 断言加载的张量 r 的形状与原始张量 t 的形状相同
        self.assertEqual(r.shape, t.shape)
        # 断言加载的张量 r 的数据类型与原始张量 t 的数据类型相同
        self.assertEqual(r.dtype, t.dtype)
        # 断言加载的张量 r 的存储指针为 0
        self.assertEqual(r.storage().data_ptr(), 0)

    def test_embedding_bag_byte_prepack(self):
        # 定义批量大小
        batch_size = 10
        # 定义嵌入数量
        num_embeddings = 80
        # 定义嵌入维度列表
        embedding_dim = [128, 256, 512]
        # 定义预期输出形状列表
        res_shape = [[batch_size, num_embeddings, ed + 8] for ed in embedding_dim]

        # 遍历嵌入维度和预期形状的组合
        for ed, rs in zip(embedding_dim, res_shape):
            # 创建形状为 (batch_size, num_embeddings, ed) 的随机权重张量
            weight = torch.randn(batch_size, num_embeddings, ed, dtype=torch.float32)
            # 调用量化的嵌入包字节预打包操作，返回预期形状的结果张量
            res = torch.ops.quantized.embedding_bag_byte_prepack(weight.to(device="meta"))
            # 断言结果张量的形状与预期形状相同
            self.assertEqual(res.shape, rs)
            # 断言结果张量的数据类型为 torch.float32
            self.assertEqual(res.dtype, torch.float32)
            # 断言结果张量的未类型化存储指针为 0
            self.assertEqual(res.untyped_storage().data_ptr(), 0)
    # 定义测试函数，用于测试 `embedding_bag_byte_unpack` 方法
    def test_embedding_bag_byte_unpack(self):
        # 设置批处理大小为10，嵌入数量为80
        batch_size = 10
        num_embeddings = 80
        # 设置嵌入维度的不同值
        embedding_dim = [128, 256, 512]
        # 根据不同的嵌入维度生成预期的输出形状列表
        res_shape = [[batch_size, num_embeddings, ed] for ed in embedding_dim]
        # 遍历嵌入维度和预期形状列表
        for ed, rs in zip(embedding_dim, res_shape):
            # 创建随机张量作为 packed_weight，包含特定的维度
            packed_weight = torch.randn(batch_size, num_embeddings, ed + 8, dtype=torch.float32)
            # 调用 quantized.embedding_bag_byte_unpack 方法，转换到指定设备上
            res = torch.ops.quantized.embedding_bag_byte_unpack(packed_weight.to(device="meta"))
            # 断言输出张量的形状符合预期
            self.assertEqual(res.shape, rs)
            # 断言输出张量的数据类型为 torch.float32
            self.assertEqual(res.dtype, torch.float32)
            # 断言未命名存储的数据指针为零
            self.assertEqual(res.untyped_storage().data_ptr(), 0)

    # 定义测试函数，用于测试 `index_select` 方法的输出
    def test_index_select_out(self):
        # 定义内部函数 f，生成随机输入张量和索引张量
        def f():
            input = torch.randn([8, 16], device='meta')
            index = torch.tensor([2, 1, 6, 7, 3, 1, 7, 5, 6, 7], device='meta')
            out = torch.empty([10, 16], device='meta')  # 创建输出张量
            # 调用 index_select 方法，将结果存储到预先创建的输出张量中
            return torch.index_select(input=input, dim=0, index=index, out=out)
        # 启用 Python 调度器，确保调用的是 Python 实现
        with enable_python_dispatcher():
            # 调用函数 f，获取输出结果
            out = f()
            # 断言输出张量的形状为 [10, 16]
            self.assertEqual(out.shape, [10, 16])

    # 定义测试函数，用于测试在 meta 张量上调用 `item()` 方法时的异常情况
    def test_local_scalar_dense_call(self):
        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，并验证异常消息包含特定文本
        with self.assertRaisesRegex(RuntimeError, "cannot be called on meta tensors"):
            # 创建一个 meta 张量
            meta_tensor = torch.randn(1, device='meta')
            # 尝试调用 meta 张量的 item() 方法，预期引发异常
            meta_tensor.item()
# 使用给定的 TestMeta 类来实例化设备类型测试，将结果添加到当前的全局变量空间中
instantiate_device_type_tests(TestMeta, globals())

# 定义一个函数，如果操作名字不受支持，则打印操作字符串
def print_op_str_if_not_supported(op_str):
    # 解析操作字符串为 OperatorName 对象
    op = OperatorName.parse(op_str)
    # 获取 torch.ops.aten 中对应操作名的包
    packet = getattr(torch.ops.aten, str(op.name))
    # 获取对应包中重载的函数，若未指定重载则使用默认
    overload = getattr(packet, op.overload_name if op.overload_name else "default")
    # 检查当前重载是否在跳过列表中（meta_dispatch_skips 或 meta_dispatch_device_skips['cuda']），如果是则打印跳过信息
    if any(overload in d for d in [meta_dispatch_skips, meta_dispatch_device_skips['cuda']]):
        print(f"{overload}  # SKIP")
    # 检查当前重载是否在预期失败列表中（meta_dispatch_expected_failures 或 meta_dispatch_device_expected_failures['cuda']），如果是则打印重载信息
    if any(overload in d for d in [meta_dispatch_expected_failures, meta_dispatch_device_expected_failures['cuda']]):
        print(overload)

# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 获取环境变量 PYTORCH_COMPARE_XLA 的值
    COMPARE_XLA = os.getenv('PYTORCH_COMPARE_XLA', None)
    # 如果环境变量值不为空
    if COMPARE_XLA is not None:
        # 打开指定文件并加载 YAML 格式内容
        with open(COMPARE_XLA) as f:
            d = yaml.load(f, Loader=YamlLoader)
            # 从加载的内容中获取操作列表
            ops = d.get("full_codegen", []) + d.get("supported", []) + d.get("autograd", [])
            # 遍历操作列表并调用打印函数
            for op_str in ops:
                print_op_str_if_not_supported(op_str)
        # 退出程序
        sys.exit(0)

    # 获取环境变量 PYTORCH_COMPARE_TEXT 的值
    COMPARE_TEXT = os.getenv('PYTORCH_COMPARE_TEXT', None)
    # 如果环境变量值不为空
    if COMPARE_TEXT is not None:
        # 打开指定文件并逐行读取内容
        with open(COMPARE_TEXT) as f:
            for op_str in f:
                # 调用打印函数，去除每行末尾的空白字符
                print_op_str_if_not_supported(op_str.strip())
        # 退出程序
        sys.exit(0)

    # 运行测试函数
    run_tests()
```