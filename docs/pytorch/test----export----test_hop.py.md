# `.\pytorch\test\export\test_hop.py`

```
# Owner(s): ["oncall: export"]
# flake8: noqa

# 导入必要的模块
import copy  # 导入深拷贝模块
import io  # 导入IO模块
import unittest  # 导入单元测试模块

import torch  # 导入PyTorch模块
import torch._dynamo as torchdynamo  # 导入私有模块torch._dynamo
import torch.utils._pytree as pytree  # 导入私有模块torch.utils._pytree
from torch._dynamo.test_case import TestCase  # 从私有模块导入TestCase类
from torch.export import export, load, save  # 导入导出相关函数
from torch.export._trace import _export  # 导入私有模块torch.export._trace中的_export函数
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,  # 导入设备类型测试相关函数
    ops,  # 导入操作函数
)
from torch.testing._internal.common_utils import (
    IS_WINDOWS,  # 导入Windows平台判断变量
    run_tests,  # 导入运行测试函数
    TestCase as TorchTestCase,  # 导入测试用例类并重命名为TorchTestCase
)
from torch.testing._internal.hop_db import (
    hop_db,  # 导入hop_db
    hop_that_doesnt_have_opinfo_test_allowlist,  # 导入不包含opinfo测试白名单
)

hop_tests = []  # 初始化hop_tests列表

# 遍历hop_db中的操作信息
for op_info in hop_db:
    op_info_hop_name = op_info.name
    # 如果操作信息在不包含opinfo测试白名单中，则继续下一次循环
    if op_info_hop_name in hop_that_doesnt_have_opinfo_test_allowlist:
        continue
    # 将操作信息添加到hop_tests列表中
    hop_tests.append(op_info)

# 定义TestHOPGeneric类，继承自TestCase类
class TestHOPGeneric(TestCase):

    # 测试所有高阶操作是否具有操作信息
    def test_all_hops_have_op_info(self):
        from torch._ops import _higher_order_ops  # 导入私有模块torch._ops中的_higher_order_ops

        # 获取具有操作信息的高阶操作名称集合
        hops_that_have_op_info = set([k.name for k in hop_db])
        # 获取所有高阶操作的名称集合
        all_hops = _higher_order_ops.keys()

        missing_ops = []  # 初始化缺失操作列表

        # 遍历所有高阶操作
        for op in all_hops:
            # 如果操作既不在具有操作信息的高阶操作集合中，也不在不包含opinfo测试白名单中，则将其添加到缺失操作列表中
            if (
                op not in hops_that_have_op_info
                and op not in hop_that_doesnt_have_opinfo_test_allowlist
            ):
                missing_ops.append(op)

        # 断言缺失操作列表的长度为0，否则输出缺失操作信息
        self.assertTrue(len(missing_ops) == 0, f"Missing op info for {missing_ops}")


# 根据平台决定是否跳过测试
@unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
# 根据是否支持dynamo决定是否跳过测试
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
# 定义TestHOP类，继承自TestCase类
class TestHOP(TestCase):

    # 比较两个模型的输出结果
    def _compare(self, eager_model, export, args, kwargs):
        # 深拷贝输入参数
        eager_args = copy.deepcopy(args)
        eager_kwargs = copy.deepcopy(kwargs)
        export_args = copy.deepcopy(args)
        export_kwargs = copy.deepcopy(kwargs)

        # 获取eager_model和export的输出结果并展平
        flat_orig_outputs = pytree.tree_leaves(eager_model(*eager_args, **eager_kwargs))
        flat_loaded_outputs = pytree.tree_leaves(
            export.module()(*export_args, **export_kwargs)
        )

        # 逐个比较展平后的输出结果
        for orig, loaded in zip(flat_orig_outputs, flat_loaded_outputs):
            self.assertEqual(type(orig), type(loaded))
            self.assertEqual(orig, loaded)

    # 装饰器定义测试操作
    @ops(hop_tests, allowed_dtypes=(torch.float,))
    def test_aot_export(self, device, dtype, op):
        # 定义Foo类，继承自torch.nn.Module
        class Foo(torch.nn.Module):
            # 定义前向传播函数
            def forward(self, *args):
                return op.op(*args)

        # 获取操作op的样本输入迭代器
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        # 遍历样本输入
        for inp in sample_inputs_itr:
            model = Foo()  # 创建Foo类的实例model
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            args = (*input, *inp.args)
            kwargs = inp.kwargs
            ep = export(model, args, kwargs)  # 导出模型
            self._compare(model, ep, args, kwargs)  # 比较模型输出结果

    # 装饰器定义测试操作
    @ops(hop_tests, allowed_dtypes=(torch.float,))
    # 定义一个测试方法，用于测试导出功能，接受设备、数据类型和操作对象作为参数
    def test_pre_dispatch_export(self, device, dtype, op):
        # 定义一个内部类 Foo，继承自 torch.nn.Module，重载 forward 方法执行操作对象的操作
        class Foo(torch.nn.Module):
            def forward(self, *args):
                return op.op(*args)

        # 生成操作对象的样本输入迭代器，要求梯度为 True
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        # 遍历样本输入迭代器
        for inp in sample_inputs_itr:
            # 创建 Foo 类的实例 model
            model = Foo()
            # 确定输入数据是元组还是单个值，并构造参数 args
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            args = (*input, *inp.args)
            kwargs = inp.kwargs
            # 调用 _export 函数导出模型，pre_dispatch 参数设为 True
            ep = _export(model, args, kwargs, pre_dispatch=True)
            # 使用 self._compare 方法比较模型和导出结果的效果
            self._compare(model, ep, args, kwargs)

    # 将该方法标记为操作测试，并指定允许的数据类型为 torch.float
    @ops(hop_tests, allowed_dtypes=(torch.float,))
    # 定义一个测试方法，用于测试导出功能，接受设备、数据类型和操作对象作为参数
    def test_retrace_export(self, device, dtype, op):
        # 定义一个内部类 Foo，继承自 torch.nn.Module，重载 forward 方法执行操作对象的操作
        class Foo(torch.nn.Module):
            def forward(self, *args):
                return op.op(*args)

        # 生成操作对象的样本输入迭代器，要求梯度为 True
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        # 遍历样本输入迭代器
        for inp in sample_inputs_itr:
            # 创建 Foo 类的实例 model
            model = Foo()
            # 确定输入数据是元组还是单个值，并构造参数 args
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            args = (*input, *inp.args)
            kwargs = inp.kwargs
            # 调用 _export 函数导出模型，pre_dispatch 参数设为 True
            ep = _export(model, args, kwargs, pre_dispatch=True)
            # 运行导出的模型的分解过程
            ep = ep.run_decompositions()
            # 使用 self._compare 方法比较模型和导出结果的效果
            self._compare(model, ep, args, kwargs)

    # 将该方法标记为操作测试，并指定允许的数据类型为 torch.float
    @ops(hop_tests, allowed_dtypes=(torch.float,))
    # 定义一个测试方法，用于测试导出功能，接受设备、数据类型和操作对象作为参数
    def test_serialize_export(self, device, dtype, op):
        # 定义一个内部类 Foo，继承自 torch.nn.Module，重载 forward 方法执行操作对象的操作
        class Foo(torch.nn.Module):
            def forward(self, *args):
                return op.op(*args)

        # 生成操作对象的样本输入迭代器，要求梯度为 True
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        # 遍历样本输入迭代器
        for inp in sample_inputs_itr:
            # 创建 Foo 类的实例 model
            model = Foo()
            # 确定输入数据是元组还是单个值，并构造参数 args
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            args = (*input, *inp.args)
            kwargs = inp.kwargs
            # 调用 _export 函数导出模型，pre_dispatch 参数设为 True
            ep = _export(model, args, kwargs, pre_dispatch=True)
            # 运行导出的模型的分解过程
            ep = ep.run_decompositions()
            # 创建一个字节流缓冲区
            buffer = io.BytesIO()
            # 将导出的模型 ep 序列化到缓冲区
            save(ep, buffer)
            buffer.seek(0)
            # 从缓冲区加载反序列化模型 ep
            ep = load(buffer)
            # 如果操作对象中包含 "while_loop" 字符串
            if "while_loop" in str(op):
                # 断言 RuntimeError 中包含 "carried_inputs must be a tuple"
                with self.assertRaisesRegex(
                    RuntimeError, "carried_inputs must be a tuple"
                ):
                    # 使用 self._compare 方法比较模型和导出结果的效果
                    self._compare(model, ep, args, kwargs)
            else:
                # 使用 self._compare 方法比较模型和导出结果的效果
                self._compare(model, ep, args, kwargs)
# 调用函数 instantiate_device_type_tests，用于实例化与设备类型相关的测试用例，将 TestHOP 作为参数传递，并将结果赋给全局变量
instantiate_device_type_tests(TestHOP, globals())

# 检查当前模块是否作为主程序运行
if __name__ == "__main__":
    # 如果是，则执行测试函数 run_tests()
    run_tests()
```