# `.\pytorch\test\jit\test_dtype_analysis.py`

```
# Owner(s): ["oncall: jit"]

# 导入所需的模块和类
from itertools import product  # 提供迭代器工具，用于生成笛卡尔积
from typing import Tuple  # 引入类型提示中的元组类型
from unittest.case import expectedFailure  # 导入测试用例中的预期失败模块

import torch  # 引入PyTorch深度学习库
from torch import complex32, float32, float64, int32, int64  # 导入不同数据类型的张量类型
from torch.jit._passes import _property_propagation  # 导入JIT编译器的属性传播模块
from torch.testing._internal.common_device_type import (  # 导入设备类型测试相关工具
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import (  # 导入通用方法调用的测试工具
    op_db,
    sample_inputs_adaptive_avg_pool2d,
    sample_inputs_conv2d,
    SampleInput,
)
from torch.testing._internal.common_utils import first_sample, set_default_dtype  # 导入测试的实用工具函数
from torch.testing._internal.jit_metaprogramming_utils import create_traced_fn  # 导入用于元编程的跟踪函数创建工具
from torch.testing._internal.jit_utils import JitTestCase  # 导入用于JIT测试的测试用例基类

"""
Dtype Analysis relies on symbolic shape analysis, which is still in beta
"""

# 如果文件被直接运行，抛出运行时错误提醒
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 自定义规则生效的函数列表
custom_rules_works_list = {
    "nn.functional.adaptive_avg_pool1d",
    "nn.functional.adaptive_avg_pool2d",
    "nn.functional.adaptive_avg_pool3d",
    "nn.functional.adaptive_max_pool1d",
    "nn.functional.adaptive_max_pool2d",
    "avg_pool1d",
    "avg_pool3d",
    "conv_transpose2d",
    "conv1d",
    "conv2d",
    "hardswish",
    "avg_pool2d",
    "max_pool1d",
    "max_pool2d",
    "max_pool3d",
    "nn.functional.prelu",
    "batch_norm",
}

# 预期失败的自定义规则函数列表
custom_rules_expected_failure_list = {
    # create_traced_fn generates prim::NumToTensor nodes in graph (not supported yet)
    "nn.functional.adaptive_max_pool3d",
}

# 这些操作似乎不在 opinfos 中，不测试的自定义规则函数列表
custom_rules_not_tested_list = [
    "conv3d",
    "conv_tbc",
    "conv_transpose1d",
    "conv_transpose3d",
    "convolution",
    "_convolution",
    "max_unpool2d",
    "max_unpool3d",
    "reflection_pad1d",
    "reflection_pad2d",
    "reflection_pad3d",
    "replication_pad1d",
    "replication_pad2d",
    "replication_pad3d",
    "upsample_bilinear2d",
    "upsample_linear1d",
    "upsample_nearest1d",
    "upsample_nearest2d",
    "upsample_nearest3d",
    "upsample_trilinear3d",
    "flatten",
]


class TestDtypeBase(JitTestCase):
    SCALAR = "SCALAR"  # 用于标记标量与零维张量的常量

    def setUp(self):
        self.prev_symbolic_shapes_test_enabled = (
            torch._C._jit_symbolic_shapes_test_mode_enabled()
        )
        torch._C._jit_set_symbolic_shapes_test_mode(True)  # 启用符号形状测试模式

    def tearDown(self):
        torch._C._jit_set_symbolic_shapes_test_mode(
            self.prev_symbolic_shapes_test_enabled
        )  # 恢复之前的符号形状测试模式设置

    @staticmethod
    def node_output_dtypes(graph):
        dtypes = []
        for out in graph.outputs():
            if isinstance(out.type(), torch._C.TensorType):
                dtypes.append(out.type().dtype())
            else:
                dtypes.append(None)
        return dtypes

    @staticmethod
    # 返回图中节点的单一输出数据类型
    def node_output_dtype_single(graph):
        # 获取图中节点的输出数据类型列表
        dtypes = TestDtypeBase.node_output_dtypes(graph)
        # 断言输出数据类型列表长度为1
        assert len(dtypes) == 1
        # 返回唯一的输出数据类型
        return dtypes[0]

    # 在图上进行属性传播，基于示例输入来应用输入属性
    def prop_dtype_on_graph(self, graph, example_inputs):
        # 清除图中的形状信息，因为 torch.jit.script
        # 如果函数被脚本化两次，将返回缓存的图
        torch._C._jit_pass_erase_shape_information(graph)
        # 使用示例输入来应用输入属性
        _property_propagation.apply_input_props_using_example(graph, example_inputs)
        # 在图上传播形状
        torch._C._jit_pass_propagate_shapes_on_graph(graph)
        # 在图上传播数据类型
        torch._C._jit_pass_propagate_dtype(graph)

    # 断言函数的输出数据类型与预期的数据类型相等
    def assert_dtype_equal(self, fn, in_shapes, in_dtypes):
        # 生成随机张量作为输入
        inputs = [self.get_rand_tensor(s, d) for s, d in zip(in_shapes, in_dtypes)]
        try:
            # 调用自定义函数进行数据类型比较
            self.assert_dtype_equal_custom_args(fn, inputs)
        except Exception as e:
            fail_text = f"Failed for shapes {in_shapes}, and dtypes {in_dtypes}"
            # 如果断言失败，则抛出 AssertionError
            raise AssertionError(fail_text) from e

    # 使用自定义参数进行数据类型比较
    def assert_dtype_equal_custom_args(self, fn, args):
        try:
            # 立即执行函数
            expected_res = fn(*args)
        except RuntimeError as e:
            return

        # 获取函数的图表达式
        graph = torch.jit.script(fn).graph  # 注意这是一个已缓存的图表达式
        # 在图上进行属性传播，使用给定的参数
        self.prop_dtype_on_graph(graph, args)
        # 获取图中节点的单一输出数据类型
        actual_dtype = self.node_output_dtype_single(graph)

        # 断言实际数据类型与预期数据类型相等
        self.assertEqual(actual_dtype, expected_res.dtype, "Failed Verification")

    # 获取指定形状和数据类型的随机张量
    def get_rand_tensor(self, shape, dtype):
        # 如果形状是标量
        if shape is self.SCALAR:
            # 根据数据类型返回特定的标量值
            if dtype is float32:
                return 1.1
            elif dtype is int64:
                return 2
            else:
                raise RuntimeError(
                    "Testing of scalars only supported for fp32 and int64"
                )

        # 如果数据类型是 int32 或 int64
        if dtype in (int32, int64):
            # 生成随机整数张量
            rand_tensor = torch.randint(0, 10, shape, dtype=dtype)
        else:
            # 生成随机浮点数张量
            rand_tensor = torch.rand(shape, dtype=dtype)

        # 断言生成的张量数据类型与指定的数据类型相等
        self.assertEqual(rand_tensor.dtype, dtype)
        # 返回生成的随机张量
        return rand_tensor
class TestDtypeAnalysis(TestDtypeBase):
    def test_unary(self):
        # Testing the Unary Implementation that uses metatensors
        
        # 定义一个原地执行ReLU操作的函数
        def relu_inplace(x):
            return x.relu_()
        
        # 定义对数操作的函数
        def log(x):
            return torch.log(x)
        
        # 函数列表包括relu_inplace和log
        functions = [relu_inplace, log]
        
        # 输入的形状列表
        input_shapes = [
            ((2, 2),),  # 简单情况
            ((0, 2),),  # 大小为0的张量
            ((),),      # 零维张量
        ]
        
        # 输入的数据类型列表
        input_dtypes = [
            (float32,),    # 简单情况，浮点数类型
            (int64,),      # 测试一些一元操作隐式转换为浮点数的情况
            (complex32,),  # 展示我们也可以处理复数值
        ]
        
        # 对于每个函数、输入形状和输入数据类型的组合，执行断言
        for fn, in_shapes, in_dtypes in product(functions, input_shapes, input_dtypes):
            self.assert_dtype_equal(fn, in_shapes, in_dtypes)

    def test_binary_tensors(self):
        # Testing using Metatensors
        
        # 定义加法操作的函数
        def add(x, y):
            return x + y
        
        # 定义除法操作的函数
        def div(x, y):
            return x / y
        
        # 函数列表包括add和div
        functions = [add, div]
        
        # 输入的形状列表
        input_shapes = [
            ((1, 1, 2), (1, 2)),  # 不同维度，非零维
            ((), (1, 2)),         # 其中一个是零维
            ((1, 2), ()),         # 另一个是零维
            ((2, 0, 3), (1, 3)),  # 测试一个有零维的张量
            ((), ()),             # 都是零维
        ]
        
        # 输入的数据类型列表
        input_dtypes = [
            (float32, float32),    # 简单情况，浮点数类型
            (int32, int64),        # 大小提升（对于零维张量是复杂的情况）
            (float32, int32),      # 类型提升
            (int64, float32),      # 类型提升并改变大小
            (float64, complex32),  # 展示我们也可以处理复数值
        ]
        
        # 对于每个函数、输入形状和输入数据类型的组合，执行断言
        for fn, in_shapes, in_dtypes in product(functions, input_shapes, input_dtypes):
            self.assert_dtype_equal(fn, in_shapes, in_dtypes)

    def test_binary_scalar(self):
        # Test the mixing of scalar and non-scalar args
        
        # 输入的形状列表
        input_shapes = [
            ((2, 2), self.SCALAR),  # 非零维张量 vs 标量
            ((), self.SCALAR),      # 零维张量 vs 标量
            # 标量 vs 标量会自动推断
        ]
        
        # 输入的数据类型列表
        input_dtypes = [
            (float32, float32),  # 简单情况，浮点数类型
            (int32, int64),      # 大小提升（对于零维张量是复杂的情况）
            (int32, float32),    # 类型提升
        ]
        
        # 使用默认的数据类型float32进行测试
        with set_default_dtype(float32):
            for in_shapes, in_dtypes in product(input_shapes, input_dtypes):
                scalar_type = in_dtypes[1]
                
                # 根据标量类型定义不同的加法函数
                if scalar_type == float32:
                    
                    # 定义一个使用浮点数进行加法的函数
                    def add(x, y: float):
                        return x + y
                    
                else:
                    
                    # 定义一个使用整数进行加法的函数
                    def add(x, y: int):
                        return x + y
                
                # 执行断言
                self.assert_dtype_equal(add, in_shapes, in_dtypes)
    def test_custom_rules(self):
        # 测试一些 Metatensors 没有覆盖的操作

        # 注意，与 Conv2d 模块不同，函数 conv2d
        # 不接受 dtype/device 参数。

        def conv2d_fn(input, weight, bias):
            return torch.nn.functional.conv2d(input, weight, bias)

        def adaptive_avg_pool2d_fn(input, output_size: Tuple[int]):
            return torch._C._nn.adaptive_avg_pool2d(input, output_size)

        for fn, inputs_fn in (
            (conv2d_fn, sample_inputs_conv2d),
            (adaptive_avg_pool2d_fn, sample_inputs_adaptive_avg_pool2d),
        ):
            for dtype in (torch.int8, torch.float64):
                # 获取 conv2d 的默认版本
                sample_input: SampleInput = list(inputs_fn(None, "cpu", dtype, False))[
                    -1
                ]
                input_args = [sample_input.input, *sample_input.args]
                self.assert_dtype_equal_custom_args(fn, input_args)

    def test_conv_no_mixed_args(self):
        def conv2d_fn(input, weight, bias):
            return torch.nn.functional.conv2d(input, weight, bias)

        # 确保 conv2d 不支持混合参数
        conv_ins = sample_inputs_conv2d(None, "cpu", torch.float, False)
        conv_in = list(conv_ins)[-1]
        weight, bias = conv_in.args
        weight = weight.type(torch.long)

        with self.assertRaises(RuntimeError):
            conv2d_fn(conv_in.input, weight, bias)

        # 检查我们也不会传播
        graph = torch.jit.script(conv2d_fn).graph  # 注意这是一个缓存的图
        self.prop_dtype_on_graph(graph, [conv_in.input, weight, bias])
        actual_dtype = self.node_output_dtype_single(graph)
        self.assertEqual(actual_dtype, None)

    def test_combined(self):
        # 测试包含自定义规则和 Metatensors 的情况

        def func(input, weight, bias, y):
            conv_out = torch.nn.functional.conv2d(input, weight, bias)
            conv_2 = conv_out + y
            flattened = torch.flatten(conv_2, start_dim=2)
            add_res = flattened + y
            return add_res

        conv_ins = sample_inputs_conv2d(None, "cpu", torch.int8, False)
        conv_in = list(conv_ins)[-1]
        y_val = torch.rand((1,), dtype=torch.float32)
        input_args = [conv_in.input, *conv_in.args, y_val]
        self.assert_dtype_equal_custom_args(func, input_args)
class TestDtypeCustomRules(TestDtypeBase):
    # 定义一个测试类，继承自TestDtypeBase

    def assert_output_dtype_equal(self, expected_res, prop_graph):
        # 断言输出的数据类型与预期结果相等
        actual_dtype = self.node_output_dtypes(prop_graph)
        # 获取节点输出的数据类型
        if len(actual_dtype) == 1:
            # 如果长度为1，表示没有对expected_res进行元组打包
            self.assert_tensor_dtype_equal(expected_res, actual_dtype[0])
            # 断言单个张量的数据类型与实际输出数据类型相等
        else:
            self.assertEqual(len(expected_res), len(actual_dtype))
            # 断言预期结果列表的长度与实际输出数据类型列表的长度相等
            for expected, actual in zip(expected_res, actual_dtype):
                self.assert_tensor_dtype_equal(expected, actual)
                # 逐个断言每个预期结果张量与实际输出数据类型张量的数据类型相等

    def assert_tensor_dtype_equal(self, tensor_output, graph_dtype):
        # 断言张量的数据类型与图中的数据类型相等
        if not isinstance(tensor_output, torch.Tensor):
            return
            # 如果输出不是torch.Tensor类型，则直接返回
        self.assertEqual(tensor_output.dtype, graph_dtype)
        # 断言张量的数据类型与图中指定的数据类型相等

    def custom_rules_test_base(self, device, dtype, op, allow_eager_fail=False):
        # 自定义规则测试基础函数
        try:
            samples = op.sample_inputs(device, dtype, requires_grad=False)
            # 从操作op中获取样本输入
            sample_input = first_sample(self, samples)
            # 获取第一个样本输入
            input_args = [sample_input.input, *sample_input.args]
            # 构建输入参数列表
            expected_res = op(*input_args, **sample_input.kwargs)
            # 执行操作op，获取预期结果

        except Exception as e:
            if allow_eager_fail:
                return
                # 如果允许快速失败，则直接返回
            else:
                raise e
                # 否则抛出异常

        func = op.get_op()
        # 获取操作op的函数
        traced_fn = create_traced_fn(self, func)
        # 创建跟踪函数

        # Have to run the traced function to actually generate the trace
        traced_fn(sample_input.input, *sample_input.args, **sample_input.kwargs)
        # 必须运行跟踪函数来实际生成跟踪

        # Run the Dtype Analysis
        graph = traced_fn.graph  # Note this is a cached graph
        # 运行数据类型分析，获取跟踪函数的图（注意这是一个缓存的图）
        input_tensors = [t for t in input_args if isinstance(t, torch.Tensor)]
        # 收集输入参数中的张量
        input_tensors += [
            v for v in sample_input.kwargs.values() if isinstance(v, torch.Tensor)
        ]
        # 将样本输入的关键字参数中的张量也加入到输入张量列表中
        self.prop_dtype_on_graph(graph, input_tensors)
        # 在图上应用数据类型属性
        self.assert_output_dtype_equal(expected_res, graph)
        # 断言输出数据类型与预期结果相等

    @ops([op for op in op_db if op.aten_name in custom_rules_works_list])
    # 应用自定义操作装饰器，选择包含在custom_rules_works_list中的操作
    def test_custom_rules(self, device, dtype, op):
        # 测试自定义规则
        self.custom_rules_test_base(device, dtype, op)

    @ops([op for op in op_db if op.aten_name in custom_rules_works_list])
    # 应用自定义操作装饰器，选择包含在custom_rules_works_list中的操作
    def test_custom_rules_ints(self, device, dtype, op):
        # 测试整数类型的自定义规则
        # This is done because opinfos currently only runs on floats.
        # Return fn, inputs_fn for all
        # 这是因为opinfos目前仅在浮点数上运行。

        if dtype == torch.float32:
            dtype = torch.int32
            # 如果dtype是torch.float32，则将其设置为torch.int32
        else:
            dtype = torch.int64
            # 否则将其设置为torch.int64

        # Because ints are not always implemented, we need to allow for eager to fail
        # 因为整数并不总是被实现，所以我们需要允许快速失败
        self.custom_rules_test_base(device, dtype, op, allow_eager_fail=True)

    @expectedFailure
    # 标记为预期失败的测试用例
    @ops([op for op in op_db if op.aten_name in custom_rules_expected_failure_list])
    # 应用自定义操作装饰器，选择包含在custom_rules_expected_failure_list中的操作
    def test_custom_rules_expected_failure(self, device, dtype, op):
        # 测试预期失败的自定义规则
        self.custom_rules_test_base(device, dtype, op)

TestDtypeCustomRulesCPU = None
# 定义TestDtypeCustomRulesCPU为None

instantiate_device_type_tests(TestDtypeCustomRules, globals(), only_for=("cpu",))
# 实例化设备类型测试，仅限于CPU
```