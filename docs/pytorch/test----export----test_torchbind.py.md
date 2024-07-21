# `.\pytorch\test\export\test_torchbind.py`

```py
# Owner(s): ["oncall: export"]

# 导入单元测试模块
import unittest

# 导入PyTorch及相关模块
import torch
import torch.utils._pytree as pytree
from torch._dynamo.testing import EagerAndRecordGraphs
from torch._functorch.aot_autograd import aot_export_module
from torch._higher_order_ops.torchbind import enable_torchbind_tracing

# 导入包装函数和虚拟脚本对象相关模块
from torch._higher_order_ops.wrap import wrap
from torch._library.fake_class_registry import FakeScriptObject

# 导入导出和跟踪相关模块
from torch.export import export
from torch.export._trace import _export

# 导入实验性代理张量相关模块
from torch.fx.experimental.proxy_tensor import make_fx

# 导入测试工具函数
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)

# 导入Torchbind实现相关模块
from torch.testing._internal.torchbind_impls import (
    _empty_tensor_queue,
    init_torchbind_implementations,
)


def _assertEqualSkipScriptObject(test_case, exp, actual):
    # 将期望值和实际值展平为列表
    flat_exp = pytree.tree_leaves(exp)
    flat_actual = pytree.tree_leaves(actual)
    # 断言展平后的列表长度相等
    test_case.assertEqual(len(flat_exp), len(flat_actual))
    # 逐一比较列表中的每一对值
    for a, b in zip(flat_exp, flat_actual):
        # 如果都是torch.ScriptObject类型，则跳过比较
        if isinstance(a, torch.ScriptObject) and isinstance(b, torch.ScriptObject):
            continue
        # 否则比较两个值是否相等
        test_case.assertEqual(a, b)


def _check_script_obj_equal(test_case, a: torch.ScriptObject, b: torch.ScriptObject):
    # 断言两个脚本对象的类型和展平后的值是否相等
    return test_case.assertEqual(
        a._type().qualified_name(), b._type().qualified_name()
    ) and test_case.assertEqual(a.__obj_flatten__(), b.__obj_flatten__())


def _assertEqualScriptObject(
    test_case, exp, actual, check_obj_eq=_check_script_obj_equal
):
    # 将期望值和实际值展平为列表
    flat_exp = pytree.tree_leaves(exp)
    flat_actual = pytree.tree_leaves(actual)
    # 断言展平后的列表长度相等
    test_case.assertEqual(len(flat_exp), len(flat_actual))
    # 逐一比较列表中的每一对值
    for a, b in zip(flat_exp, flat_actual):
        # 如果都是torch.ScriptObject类型，则调用自定义的脚本对象比较函数
        if isinstance(a, torch.ScriptObject) and isinstance(b, torch.ScriptObject):
            check_obj_eq(test_case, a, b)
        else:
            # 否则比较两个值是否相等
            test_case.assertEqual(a, b)


# 使用装饰器跳过对torchbind不支持的测试
@skipIfTorchDynamo("torchbind not supported with dynamo yet")
class TestExportTorchbind(TestCase):
    python
        # 设置测试环境的准备工作，在每个测试方法执行前调用
        def setUp(self):
            # 初始化 TorchBind 实现
            init_torchbind_implementations()
    
            # 将 self 赋值给 test 变量
            test = self
            # 初始化计数器
            test.tq_push_counter = 0
            test.tq_pop_counter = 0
            test.tq_size_counter = 0
            test.foo_add_tensor_counter = 0
    
            # 注册虚拟类 _TorchScriptTesting::_Foo
            @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
            class FakeFoo:
                # 初始化方法，接受两个整数参数 x 和 y
                def __init__(self, x: int, y: int):
                    self.x = x
                    self.y = y
    
                # 类方法，将扁平化的 Foo 对象转换为原始对象
                @classmethod
                def __obj_unflatten__(cls, flattend_foo):
                    return cls(**dict(flattend_foo))
    
                # 添加张量方法，增加计数器并返回计算结果
                def add_tensor(self, z):
                    test.foo_add_tensor_counter += 1
                    return (self.x + self.y) * z
    
            # 注册虚拟类 _TorchScriptTesting::_TensorQueue
            @torch._library.register_fake_class("_TorchScriptTesting::_TensorQueue")
            class FakeTensorQueue:
                # 初始化方法，接受一个队列参数
                def __init__(self, queue):
                    self.queue = queue
    
                # 类方法，将扁平化的上下文对象转换为原始对象
                @classmethod
                def __obj_unflatten__(cls, flattened_ctx):
                    return cls(**dict(flattened_ctx))
    
                # 入队方法，增加计数器并将元素加入队列
                def push(self, x):
                    test.tq_push_counter += 1
                    self.queue.append(x)
    
                # 出队方法，增加计数器并返回队列中的第一个元素
                def pop(self):
                    test.tq_pop_counter += 1
                    return self.queue.pop(0)
    
                # 队列大小方法，增加计数器并返回队列长度
                def size(self):
                    test.tq_size_counter += 1
                    return len(self.queue)
    
                # 判断队列是否为空方法
                def is_empty(self):
                    return len(self.queue) == 0
    
                # 返回队列长度的浮点数表示方法
                def float_size(self):
                    return float(len(self.queue))
    
            # 初始化 TorchBind 操作列表
            self.torch_bind_ops = [
                torch.ops._TorchScriptTesting.takes_foo,
                torch.ops._TorchScriptTesting.takes_foo_python_meta,
                torch.ops._TorchScriptTesting.takes_foo_list_return,
                torch.ops._TorchScriptTesting.takes_foo_tuple_return,
                torch.ops._TorchScriptTesting.take_an_instance,
                torch.ops._TorchScriptTesting.take_an_instance_inferred,
                torch.ops._TorchScriptTesting.takesch_bind_ops = [
                torch.ops._TorchScriptTesting.takes_foo,
                torch.ops._TorchScriptTesting.takes_foo_python_meta,
                torch.ops._TorchScriptTesting.takes_foo_list_return,
                torch.ops._TorchScriptTesting.takes_foo_tuple_return,
                torch.ops._TorchScriptTesting.take_an_instance,
                torch.ops._TorchScriptTesting.take_an_instance_inferred,
                torch.ops._TorchScriptTesting.takes_foo_cia,
                torch.ops._TorchScriptTesting.queue_pop,
                torch.ops._TorchScriptTesting.queue_push,
                torch.ops._TorchScriptTesting.queue_size,
            ]
    
        # 在测试结束时，取消注册 FakeFoo 和 FakeTensorQueue 假类
        def tearDown(self):
            torch._library.fake_class_registry.deregister_fake_class(
                "_TorchScriptTesting::_Foo"
            )
            torch._library.fake_class_registry.deregister_fake_class(
                "_TorchScriptTesting::_TensorQueue"
            )
        ):
        kwargs = kwargs or {}

        # 定义一个内部函数export_wrapper，用于封装导出逻辑并进行 TorchBind 跟踪
        def export_wrapper(f, args, kwargs, strcit, pre_dispatch):
            # 启用 TorchBind 跟踪上下文
            with enable_torchbind_tracing():
                if pre_dispatch:
                    # 如果预调度开启，使用_strict参数调用_export进行导出
                    exported_program = _export(
                        f, args, kwargs, strict=strict, pre_dispatch=True
                    )
                else:
                    # 否则使用_strict参数调用export进行导出
                    exported_program = export(f, args, kwargs, strict=strict)
            return exported_program

        # 调用export_wrapper函数，导出模型并获取导出的程序
        exported_program = export_wrapper(f, args, kwargs, strict, pre_dispatch)
        
        # 创建一个反转kwargs键值的字典
        reversed_kwargs = {key: kwargs[key] for key in reversed(kwargs)}
        
        # 获取未解封的模型
        unlifted = exported_program.module()
        
        # 调用原始函数f，获取其结果
        exp = f(*args, **kwargs)
        
        # 断言未解封模型应用于相同的参数时的输出与原始函数的输出相等
        _assertEqualScriptObject(self, unlifted(*args, **kwargs), exp)
        
        # 断言未解封模型应用于反转kwargs的参数时的输出与原始函数的输出相等
        _assertEqualScriptObject(
            self,
            unlifted(*args, **reversed_kwargs),
            exp,
        )

        # 检查重新跟踪
        retraced_ep = export_wrapper(unlifted, args, kwargs, strict, pre_dispatch)
        
        # 断言重新跟踪后的模型应用于相同的参数时的输出与原始函数的输出相等
        _assertEqualScriptObject(self, retraced_ep.module()(*args, **kwargs), exp)
        
        # 返回导出的程序对象
        return exported_program

    # 使用@parametrize装饰器对test_none方法进行参数化，参数为pre_dispatch=True和False
    @parametrize("pre_dispatch", [True, False])
    def test_none(self, pre_dispatch):
        # 定义一个MyModule类，继承自torch.nn.Module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个torch.classes._TorchScriptTesting._Foo对象，赋值给self.attr
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            # 定义前向传播函数forward，接受x和n两个参数，返回x加上self.attr.add_tensor(x)的结果
            def forward(self, x, n):
                return x + self.attr.add_tensor(x)

        # 调用self._test_export_same_as_eager方法，导出MyModule实例，strict=False，pre_dispatch参数化值
        ep = self._test_export_same_as_eager(
            MyModule(),
            (torch.ones(2, 3), None),
            strict=False,
            pre_dispatch=pre_dispatch,
        )
        
        # 断言导出的模型的代码与预期的代码一致
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
# 定义一个方法 forward，用于模型的前向传播
def forward(self, x, n):
    # 将输入 x 和 n 通过 fx_pytree.tree_flatten_spec() 转换成扁平化的数据结构
    x, n, = fx_pytree.tree_flatten_spec(([x, n], {}), self._in_spec)
    # 获取对象的 attr 属性
    attr = self.attr
    # 调用 torch.ops.higher_order.call_torchbind 方法，使用 attr 对象和字符串 'add_tensor' 来执行计算，结果存入 call_torchbind
    call_torchbind = torch.ops.higher_order.call_torchbind(attr, 'add_tensor', x);  attr = None
    # 调用 torch.ops.aten.add.Tensor 方法，将 x 和 call_torchbind 相加的结果存入 add，同时将 x 和 call_torchbind 设为 None
    add = torch.ops.aten.add.Tensor(x, call_torchbind);  x = call_torchbind = None
    # 使用 pytree.tree_unflatten 方法，将 add 转换回原始结构，并返回结果
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
    def test_attribute_as_custom_op_argument(self, pre_dispatch):
        # 定义一个测试方法，接受参数 pre_dispatch
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在模块初始化中，创建一个名为 attr 的自定义对象 _Foo
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                # 在前向传播中，调用自定义操作 takes_foo，将 self.attr 和输入 x 作为参数
                return x + torch.ops._TorchScriptTesting.takes_foo(self.attr, x)

        # 创建一个 MyModule 实例，并使用 _test_export_same_as_eager 方法进行导出测试
        ep = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3),), strict=False, pre_dispatch=pre_dispatch
        )
        # 断言生成的导出代码是否符合预期
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
# 定义一个名为 forward 的方法，接受 self 和 x 两个参数
def forward(self, x):
    # 将 x 扁平化并符合指定的规范，返回一个包含 x 的元组
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    # 从 self.attr 中获取属性值并赋给 attr
    attr = self.attr
    # 调用 torch.ops._TorchScriptTesting.takes_foo.default 方法，传入 attr 和 x，将结果赋给 takes_foo_default，然后将 attr 置为 None
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(attr, x);  attr = None
    # 调用 torch.ops.aten.add.Tensor 方法，传入 x 和 takes_foo_default，将结果赋给 add，并将 x 和 takes_foo_default 置为 None
    add = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
    # 返回一个包含 add 的元组，符合 self._out_spec 指定的规范
    return pytree.tree_unflatten((add,), self._out_spec)



@parametrize("pre_dispatch", [True, False])
# 定义一个名为 test_input 的测试方法，接受 self 和 pre_dispatch 两个参数
def test_input(self, pre_dispatch):
    # 创建一个 _Foo 类的实例 cc，传入参数 10 和 20
    cc = torch.classes._TorchScriptTesting._Foo(10, 20)

    # 定义一个名为 MyModule 的类，继承自 torch.nn.Module
    class MyModule(torch.nn.Module):
        # 初始化方法，无需额外参数
        def __init__(self):
            super().__init__()

        # 定义一个名为 forward 的方法，接受 self、x 和 cc 三个参数
        def forward(self, x, cc):
            # 返回 x 和 cc.add_tensor(x) 相加的结果
            return x + cc.add_tensor(x)

    # 调用 self._test_export_same_as_eager 方法，传入 MyModule 的实例、torch.ones(2, 3) 和 cc 作为参数，严格性为 False，pre_dispatch 参数取决于传入的 pre_dispatch
    ep = self._test_export_same_as_eager(
        MyModule(), (torch.ones(2, 3), cc), strict=False, pre_dispatch=pre_dispatch
    )
    # 断言 ep.module().code 的结果去掉首尾空格后等于以下内容
    self.assertExpectedInline(
        ep.module().code.strip(),
        """\
def forward(self, x, cc):
    # 将 x 和 cc 扁平化并符合指定的规范，返回一个包含 x 和 cc 的元组
    x, cc, = fx_pytree.tree_flatten_spec(([x, cc], {}), self._in_spec)
    # 调用 torch.ops.higher_order.call_torchbind 方法，传入 cc、'add_tensor' 和 x，将结果赋给 call_torchbind，并将 cc 置为 None
    call_torchbind = torch.ops.higher_order.call_torchbind(cc, 'add_tensor', x);  cc = None
    # 调用 torch.ops.aten.add.Tensor 方法，传入 x 和 call_torchbind，将结果赋给 add，并将 x 和 call_torchbind 置为 None
    add = torch.ops.aten.add.Tensor(x, call_torchbind);  x = call_torchbind = None
    # 返回一个包含 add 的元组，符合 self._out_spec 指定的规范
    return (add,)""",
    )
    # 断言 ep.graph_module.code 的结果去掉首尾空格后等于以下内容
    self.assertExpectedInline(
        ep.graph_module.code.strip(),
        """\
def forward(self, x, cc):
    # 调用 torch.ops.higher_order.call_torchbind 方法，传入 cc、'add_tensor' 和 x，将结果赋给 call_torchbind，并将 cc 置为 None
    call_torchbind = torch.ops.higher_order.call_torchbind(cc, 'add_tensor', x);  cc = None
    # 调用 torch.ops.aten.add.Tensor 方法，传入 x 和 call_torchbind，将结果赋给 add，并将 x 和 call_torchbind 置为 None
    add = torch.ops.aten.add.Tensor(x, call_torchbind);  x = call_torchbind = None
    # 返回一个包含 add 的元组
    return (add,)""",  # noqa: B950
    )
    # 断言 self.foo_add_tensor_counter 的值等于 4
    self.assertEqual(self.foo_add_tensor_counter, 4)
    # 定义一个测试函数，用于测试将 pre_dispatch 参数作为自定义操作的输入
    def test_input_as_custom_op_argument(self, pre_dispatch):
        # 创建一个名为 cc 的 TorchScript 类 _Foo 的实例对象，传入参数为 10 和 20
        cc = torch.classes._TorchScriptTesting._Foo(10, 20)

        # 定义一个继承自 torch.nn.Module 的自定义模块 MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 定义模块的前向传播方法，接受输入 x 和 cc，并返回 x 与 torch.ops._TorchScriptTesting.takes_foo(cc, x) 的加法结果
            def forward(self, x, cc):
                return x + torch.ops._TorchScriptTesting.takes_foo(cc, x)

        # 删除 takes_foo.default.py_kernels 中 torch._C.DispatchKey.Meta 键对应的 Python 实现
        del torch.ops._TorchScriptTesting.takes_foo.default.py_kernels[
            torch._C.DispatchKey.Meta
        ]
        # 清空 takes_foo.default._dispatch_cache 缓存
        torch.ops._TorchScriptTesting.takes_foo.default._dispatch_cache.clear()

        # 断言 RuntimeError 异常被抛出且异常信息包含 "no python implementation is found"
        with self.assertRaisesRegex(RuntimeError, "no python implementation is found"):
            # 调用 _test_export_same_as_eager 方法，测试导出和即时执行的一致性
            self._test_export_same_as_eager(
                MyModule(),
                (torch.ones(2, 3), cc),
                strict=False,
                pre_dispatch=pre_dispatch,
            )

        # 注册一个 lambda 函数作为 takes_foo.default 的 Python 实现，处理 torch._C.DispatchKey.Meta 类型的输入
        torch.ops._TorchScriptTesting.takes_foo.default.py_impl(
            torch._C.DispatchKey.Meta
        )(lambda cc, x: cc.add_tensor(x))

        # 测试导出和即时执行的一致性，验证预期的内联代码
        ep = self._test_export_same_as_eager(
            MyModule(),
            (torch.ones(2, 3), cc),
            strict=False,
            pre_dispatch=pre_dispatch,
        )

        # 断言预期的内联代码与生成的模块的代码（去除首尾空白字符）相匹配
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
    # 定义一个方法 `forward`，接受参数 `x` 和 `cc`
    def forward(self, x, cc):
        # 将 `x` 和 `cc` 扁平化为一个元组，并符合输入规范
        x, cc, = fx_pytree.tree_flatten_spec(([x, cc], {}), self._in_spec)
        # 调用 `torch.ops._TorchScriptTesting.takes_foo.default` 方法，使用 `cc` 和 `x` 的默认值
        takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(cc, x);  cc = None
        # 调用 `torch.ops.aten.add.Tensor` 方法，使用 `x` 和 `takes_foo_default` 执行张量加法
        add = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
        # 使用 `pytree.tree_unflatten` 方法，根据输出规范将 `add` 还原成原始形状
        return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        # 断言期望的内联输出代码，检查模块的 `forward` 方法是否返回预期的内容
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, x, cc):
    # 调用 `torch.ops._TorchScriptTesting.takes_foo.default` 方法，使用 `cc` 和 `x` 的默认值
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(cc, x);  cc = None
    # 调用 `torch.ops.aten.add.Tensor` 方法，使用 `x` 和 `takes_foo_default` 执行张量加法
    add = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
    # 返回包含 `add` 的元组
    return (add,)""",  # noqa: B950
        )

    @parametrize("pre_dispatch", [True, False])
    # 测试使用 `pre_dispatch` 参数的 `torchbind_alias` 方法
    def test_torchbind_alias(self, pre_dispatch):
        # 定义模块 `F2`，接受 `foo` 作为参数
        class F2(torch.nn.Module):
            def __init__(self, foo):
                super().__init__()
                self.foo = foo

            # 定义 `forward` 方法，接受 `x` 作为参数
            def forward(self, x):
                # 返回 `x` 加上 `torch.ops._TorchScriptTesting.takes_foo` 方法的结果
                return x + torch.ops._TorchScriptTesting.takes_foo(self.foo, x)

        # 定义模块 `F1`
        class F1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建 `_Foo` 类的实例 `alpha`，并将其作为模块属性
                self.alpha = torch.classes._TorchScriptTesting._Foo(10, 20)
                self.beta = self.alpha
                self.gamma = self.alpha
                # 创建 `F2` 类的实例 `foo`，使用 `gamma` 作为参数
                self.foo = F2(self.gamma)

            # 定义 `forward` 方法，接受 `x` 作为参数
            def forward(self, x):
                # 返回 `x` 加上 `torch.ops._TorchScriptTesting.takes_foo` 方法的结果，再加上 `self.foo` 的结果
                return (
                    x
                    + torch.ops._TorchScriptTesting.takes_foo(self.gamma, x)
                    + self.foo(x)
                )

        # 调用 `_test_export_same_as_eager` 方法，测试 `F1` 模块的导出与即时模式的一致性
        self._test_export_same_as_eager(
            F1(), (torch.ones(2, 3),), strict=False, pre_dispatch=pre_dispatch
        )

    # TODO(pianpwk): look into this
    @unittest.expectedFailure
    @parametrize("pre_dispatch", [True, False])
    # 测试 `torchbind_input_and_alias` 方法，使用 `pre_dispatch` 参数
    def test_torchbind_input_and_alias(self, pre_dispatch):
        # 定义模块 `F3`
        class F3(torch.nn.Module):
            # 定义 `forward` 方法，接受 `x` 和 `foo` 作为参数
            def forward(self, x, foo):
                # 将 `foo` 设置为模块属性
                self.foo = foo
                # 返回 `x` 加上 `self.foo.add_tensor` 方法的结果
                return x + self.foo.add_tensor(x)

        # 创建 `_Foo` 类的实例 `foo`
        foo = torch.classes._TorchScriptTesting._Foo(10, 20)
        # 调用 `_test_export_same_as_eager` 方法，测试 `F3` 模块的导出与即时模式的一致性
        self._test_export_same_as_eager(
            F3(), (torch.ones(2, 3), foo), strict=False, pre_dispatch=pre_dispatch
        )

    @parametrize("pre_dispatch", [True, False])
    # 测试 `unlift_custom_obj` 方法，使用 `pre_dispatch` 参数
    def test_unlift_custom_obj(self, pre_dispatch):
        # 定义 `MyModule` 类
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建 `_Foo` 类的实例 `attr`，并将其作为模块属性
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            # 定义 `forward` 方法，接受 `x` 作为参数
            def forward(self, x):
                # 调用 `torch.ops._TorchScriptTesting.takes_foo` 方法两次，并返回 `x` 加上最后一次调用的结果 `b`
                a = torch.ops._TorchScriptTesting.takes_foo(self.attr, x)
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, a)
                return x + b

        # 创建 `input` 张量
        input = torch.ones(2, 3)
        # 调用 `_test_export_same_as_eager` 方法，测试 `MyModule` 模块的导出与即时模式的一致性
        ep = self._test_export_same_as_eager(
            MyModule(), (input,), strict=False, pre_dispatch=pre_dispatch
        )
        # 断言期望的内联输出代码，检查模块的代码是否符合预期
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, x):
    # 将输入 x 打平并结合规范，准备传递给模块
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    # 获取当前对象的属性
    attr = self.attr
    # 调用 torch 操作，获取默认的 takes_foo 函数结果
    takes_foo_default_1 = torch.ops._TorchScriptTesting.takes_foo.default(attr, x)
    # 再次调用 takes_foo 函数获取默认结果，并清空相关变量
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(attr, takes_foo_default_1);  attr = takes_foo_default_1 = None
    # 对输入 x 和 takes_foo_default 执行张量加法
    add = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
    # 返回未打平的张量树，包含添加的结果
    return pytree.tree_unflatten((add,), self._out_spec)""",  # noqa: B950
        )



    @parametrize("pre_dispatch", [True, False])
    def test_custom_obj_list_out(self, pre_dispatch):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化自定义属性 attr，使用 TorchScriptTesting._Foo 类
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                # 调用 torch 操作，以列表形式返回 takes_foo_list_return 的结果
                takes_foo_list_return_default = torch.ops._TorchScriptTesting.takes_foo_list_return.default(self.attr, x)
                # 获取列表中的元素并命名
                getitem_2 = takes_foo_list_return_default[0]
                getitem_3 = takes_foo_list_return_default[1]
                getitem_4 = takes_foo_list_return_default[2];  takes_foo_list_return_default = None
                # 对列表中的元素执行张量加法
                add = torch.ops.aten.add.Tensor(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
                add_1 = torch.ops.aten.add.Tensor(add, getitem_4);  add = getitem_4 = None
                # 调用 takes_foo 函数获取默认结果，并清空相关变量
                takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(self.attr, add_1);  self.attr = add_1 = None
                # 对输入 x 和 takes_foo_default 执行张量加法
                add_2 = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
                # 返回未打平的张量树，包含添加的结果
                return pytree.tree_unflatten((add_2,), self._out_spec)



        input = torch.ones(2, 3)
        ep = self._test_export_same_as_eager(
            MyModule(), (input,), strict=False, pre_dispatch=pre_dispatch
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    attr = self.attr
    takes_foo_list_return_default = torch.ops._TorchScriptTesting.takes_foo_list_return.default(attr, x)
    getitem_2 = takes_foo_list_return_default[0]
    getitem_3 = takes_foo_list_return_default[1]
    getitem_4 = takes_foo_list_return_default[2];  takes_foo_list_return_default = None
    add = torch.ops.aten.add.Tensor(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
    add_1 = torch.ops.aten.add.Tensor(add, getitem_4);  add = getitem_4 = None
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(attr, add_1);  attr = add_1 = None
    add_2 = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
    return pytree.tree_unflatten((add_2,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, obj_attr, x):
    takes_foo_list_return_default = torch.ops._TorchScriptTesting.takes_foo_list_return.default(obj_attr, x)
    getitem_2 = takes_foo_list_return_default[0]
    getitem_3 = takes_foo_list_return_default[1]
    # 从 takes_foo_list_return_default 取第三个元素，并将 takes_foo_list_return_default 设为 None
    getitem_4 = takes_foo_list_return_default[2];  takes_foo_list_return_default = None
    # 使用 torch.ops.aten.add.Tensor 函数将 getitem_2 和 getitem_3 相加，并将 getitem_2 和 getitem_3 设为 None
    add = torch.ops.aten.add.Tensor(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
    # 使用 torch.ops.aten.add.Tensor 函数将 add 和 getitem_4 相加，并将 add 和 getitem_4 设为 None
    add_1 = torch.ops.aten.add.Tensor(add, getitem_4);  add = getitem_4 = None
    # 调用 torch.ops._TorchScriptTesting.takes_foo.default 方法，传入 obj_attr 和 add_1，并将 obj_attr 和 add_1 设为 None
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(obj_attr, add_1);  obj_attr = add_1 = None
    # 使用 torch.ops.aten.add.Tensor 函数将 x 和 takes_foo_default 相加，并将 x 和 takes_foo_default 设为 None
    add_2 = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
    # 返回一个元组，包含 add_2
    return (add_2,)
# 定义一个类方法 forward，接收两个参数 self 和 x
def forward(self, x):
    # 使用 fx_pytree.tree_flatten_spec 将 x 扁平化成特定规范的结构
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    # 获取 self 的属性 attr
    attr = self.attr
    # 调用 torch.ops._TorchScriptTesting.takes_foo_tuple_return.default 方法，传入 attr 和 x，返回结果
    takes_foo_tuple_return_default = torch.ops._TorchScriptTesting.takes_foo_tuple_return.default(attr, x)
    # 获取 takes_foo_tuple_return_default 的第一个元素
    getitem_1 = takes_foo_tuple_return_default[0]
    # 获取 takes_foo_tuple_return_default 的第二个元素，并将 takes_foo_tuple_return_default 设为 None
    getitem_2 = takes_foo_tuple_return_default[1];  takes_foo_tuple_return_default = None
    # 调用 torch.ops.aten.add.Tensor 方法，使用 getitem_1 和 getitem_2 进行张量加法，结果赋给 add
    add = torch.ops.aten.add.Tensor(getitem_1, getitem_2);  getitem_1 = getitem_2 = None
    # 调用 torch.ops._TorchScriptTesting.takes_foo.default 方法，传入 attr 和 add，返回结果赋给 takes_foo_default
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(attr, add);  attr = add = None
    # 再次调用 torch.ops.aten.add.Tensor 方法，使用 x 和 takes_foo_default 进行张量加法，结果赋给 add_1
    add_1 = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
    # 使用 pytree.tree_unflatten 方法将 add_1 打包成一个元组，返回结果给调用者
    return pytree.tree_unflatten((add_1,), self._out_spec)""",
        )
        # 调用 self 的 assertExpectedInline 方法，断言 ep.graph_module.code 的内容与预期的结果一致
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, obj_attr, x):
    # 调用 torch.ops._TorchScriptTesting.takes_foo_tuple_return.default 方法，传入 obj_attr 和 x，返回结果赋给 takes_foo_tuple_return_default
    takes_foo_tuple_return_default = torch.ops._TorchScriptTesting.takes_foo_tuple_return.default(obj_attr, x)
    # 获取 takes_foo_tuple_return_default 的第一个元素
    getitem_1 = takes_foo_tuple_return_default[0]
    # 获取 takes_foo_tuple_return_default 的第二个元素，并将 takes_foo_tuple_return_default 设为 None
    getitem_2 = takes_foo_tuple_return_default[1];  takes_foo_tuple_return_default = None
    # 调用 torch.ops.aten.add.Tensor 方法，使用 getitem_1 和 getitem_2 进行张量加法，结果赋给 add
    add = torch.ops.aten.add.Tensor(getitem_1, getitem_2);  getitem_1 = getitem_2 = None
    # 调用 torch.ops._TorchScriptTesting.takes_foo.default 方法，传入 obj_attr 和 add，返回结果赋给 takes_foo_default
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(obj_attr, add);  obj_attr = add = None
    # 再次调用 torch.ops.aten.add.Tensor 方法，使用 x 和 takes_foo_default 进行张量加法，结果赋给 add_1
    add_1 = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
    # 返回一个包含 add_1 的元组
    return (add_1,)""",  # noqa: B950
        )

    # 使用 parametrize 装饰器为 test_make_fx_tensor_queue_methods 方法添加多个参数化测试
    @parametrize("make_fx_tracing_mode", ["fake", "symbolic"])
    # 定义 test_make_fx_tensor_queue_methods 方法，接收 self 和 make_fx_tracing_mode 参数
    def test_make_fx_tensor_queue_methods(self, make_fx_tracing_mode):
        # 将 self 赋给 test 变量
        test = self

        # 定义一个内部类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 定义 Model 类的初始化方法
            def __init__(self):
                # 调用父类的初始化方法
                super().__init__()
                # 创建一个线性层，输入维度为 3，输出维度为 2
                self.linear = torch.nn.Linear(3, 2)
                # 设置一个检查标志 check_tq_is_fake 为 True
                self.check_tq_is_fake = True

            # 定义 Model 类的前向传播方法，接收 tq 和 x 两个参数
            def forward(self, tq, x):
                # 如果 self.check_tq_is_fake 为真，则断言 tq 是 FakeScriptObject 的实例
                if self.check_tq_is_fake:
                    test.assertTrue(isinstance(tq, FakeScriptObject))
                # 将 x 的余弦值推送到 tq 中
                tq.push(x.cos())
                # 将 x 的正弦值推送到 tq 中
                tq.push(x.sin())
                # 弹出 tq 中的一个元素，并加上 tq 的大小
                x_cos = tq.pop() + tq.size()
                # 弹出 tq 中的一个元素，并减去 tq 的大小
                x_sin = tq.pop() - tq.size()
                # 返回 x_sin, x_cos 和 tq 三个值
                return x_sin, x_cos, tq

        # 创建一个 Model 类的实例 mod
        mod = Model()
        # 创建一个 torch.classes._TorchScriptTesting._TensorQueue 类的实例 tq，并初始化为一个填充了 -1 的张量
        tq = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        # 创建一个 torch.classes._TorchScriptTesting._TensorQueue 类的实例 tq1，并初始化为一个填充了 -1 的张量
        tq1 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        # 创建一个大小为 (2, 3) 的全 1 张量 x
        x = torch.ones(2, 3)
        # 使用 make_fx 方法将 mod 转换为一个函数图，并传入 tq 和 x 作为参数
        gm = make_fx(mod, tracing_mode=make_fx_tracing_mode)(tq, x)
        # 断言 self.tq_push_counter 等于 2
        self.assertEqual(self.tq_push_counter, 2)
        # 断言 self.tq_pop_counter 等于 2
        self.assertEqual(self.tq_pop_counter, 2)
        # 断言 self.tq_size_counter 等于 2
        self.assertEqual(self.tq_size_counter, 2)
        # 断言 tq 的大小为 0
        self.assertEqual(tq.size(), 0)
        # 断言 gm.code 去除首尾换行符后的内容与预期的结果一致
        self.assertExpectedInline(
            gm.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1):
    # 调用 torch.ops.aten.cos.default 方法，传入 arg1_1，返回结果赋给 cos
    cos = torch.ops.aten.cos.default(arg1_1)
    # 调用 torch.ops.higher_order.call_torchbind 方法，传入 arg0_1, 'push', cos，结果赋给 call_torchbind
    call_torchbind = torch.ops.higher_order.call_torchbind(arg0_1, 'push', cos);  cos = None
    # 调用 torch 操作符 aten.sin.default，计算默认输入 arg1_1 的正弦值，并将结果赋给 sin；清空 arg1_1
    sin = torch.ops.aten.sin.default(arg1_1);  arg1_1 = None
    # 调用 torchbind 中的高阶函数，将 sin 压入 arg0_1 所指定的对象中
    call_torchbind_1 = torch.ops.higher_order.call_torchbind(arg0_1, 'push', sin);  sin = None
    # 再次调用 torchbind 中的高阶函数，从 arg0_1 对象中弹出一个元素
    call_torchbind_2 = torch.ops.higher_order.call_torchbind(arg0_1, 'pop')
    # 再次调用 torchbind 中的高阶函数，获取 arg0_1 对象当前的大小（元素数量）
    call_torchbind_3 = torch.ops.higher_order.call_torchbind(arg0_1, 'size')
    # 使用 torch 操作符 aten.add.Tensor，将 call_torchbind_2 中的值与标量 1 相加，结果赋给 add；清空 call_torchbind_2
    add = torch.ops.aten.add.Tensor(call_torchbind_2, 1);  call_torchbind_2 = None
    # 再次调用 torchbind 中的高阶函数，从 arg0_1 对象中弹出一个元素
    call_torchbind_4 = torch.ops.higher_order.call_torchbind(arg0_1, 'pop')
    # 再次调用 torchbind 中的高阶函数，获取 arg0_1 对象当前的大小（元素数量）
    call_torchbind_5 = torch.ops.higher_order.call_torchbind(arg0_1, 'size')
    # 使用 torch 操作符 aten.sub.Tensor，将 call_torchbind_4 中的值与标量 0 相减，结果赋给 sub；清空 call_torchbind_4
    sub = torch.ops.aten.sub.Tensor(call_torchbind_4, 0);  call_torchbind_4 = None
    # 返回 sub（subtraction 结果）、add（addition 结果）以及 arg0_1（修改后的对象）
    return (sub, add, arg0_1)
def forward(self, arg0_1, arg1_1):
    # 调用 Torchbind 操作，使用 'pop' 参数，返回结果赋值给 call_torchbind
    call_torchbind = torch.ops.higher_order.call_torchbind(arg0_1, 'pop')
    # 调用 Torchbind 操作，使用 'size' 参数，返回结果赋值给 call_torchbind_1
    call_torchbind_1 = torch.ops.higher_order.call_torchbind(arg0_1, 'size')
    # 将 call_torchbind 和 1 相加，结果赋值给 add，释放 call_torchbind
    add = torch.ops.aten.add.Tensor(call_torchbind, 1);  call_torchbind = None
    # 将 add 和 arg1_1 相加，结果赋值给 add_1，释放 add
    add_1 = torch.ops.aten.add.Tensor(add, arg1_1);  add = None
    # 再次调用 Torchbind 操作，使用 'pop' 参数，返回结果赋值给 call_torchbind_2
    call_torchbind_2 = torch.ops.higher_order.call_torchbind(arg0_1, 'pop')
    # 再次调用 Torchbind 操作，使用 'size' 参数，返回结果赋值给 call_torchbind_3
    call_torchbind_3 = torch.ops.higher_order.call_torchbind(arg0_1, 'size')
    # 将 call_torchbind_2 和 0 相减，结果赋值给 sub，释放 call_torchbind_2
    sub = torch.ops.aten.sub.Tensor(call_torchbind_2, 0);  call_torchbind_2 = None
    # 将 sub 和 arg1_1 相加，结果赋值给 add_2，释放 sub 和 arg1_1
    add_2 = torch.ops.aten.add.Tensor(sub, arg1_1);  sub = arg1_1 = None
    # 返回元组 (add_2, add_1, arg0_1)
    return (add_2, add_1, arg0_1)
    # 调用 torch._higher_order_ops.effects.with_effects 函数，生成一个带有副作用的对象 with_effects_4
    with_effects_4 = torch._higher_order_ops.effects.with_effects(getitem_6, torch.ops.higher_order.call_torchbind, tq, 'size');  getitem_6 = None
    # 从 with_effects_4 中获取第一个元素，存储在 getitem_8 中，并释放 with_effects_4 对象
    getitem_8 = with_effects_4[0];  with_effects_4 = None
    # 调用 torch.ops.aten.add.Tensor 函数，将 0 添加到 getitem_7 中，然后释放 getitem_7 对象
    add_2 = torch.ops.aten.add.Tensor(getitem_7, 0);  getitem_7 = None
    # 调用 torch.ops.aten.add.Tensor 函数，将 x 添加到 add_2 中，然后释放 add_2 和 x 对象
    add_3 = torch.ops.aten.add.Tensor(add_2, x);  add_2 = x = None
    # 返回包含 getitem_8、add_3、add_1 和 tq 的元组作为结果
    return (getitem_8, add_3, add_1, tq)""",  # noqa: B950
    # 定义一个测试方法，用于测试生成 FX 模式检查脚本对象的功能
    def test_make_fx_schema_checking_script_object(self):
        # 定义一个继承自 torch.nn.Module 的内部模型类 Model
        class Model(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, tq, x, foo):
                # 调用 torch.ops._TorchScriptTesting.queue_push 方法将 x 的余弦值推送到 foo 中
                torch.ops._TorchScriptTesting.queue_push(foo, x.cos())
                # 返回给定的 tq 张量
                return tq

        # 定义一个继承自 torch.nn.Module 的内部模型类 ModelCallByKW
        class ModelCallByKW(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, tq, x, foo):
                # 调用 torch.ops._TorchScriptTesting.queue_push 方法将 x 的余弦值和 foo 推送到相应位置
                torch.ops._TorchScriptTesting.queue_push(x=x.cos(), foo=foo)
                # 返回给定的 tq 张量
                return tq

        # 实例化 Model 和 ModelCallByKW 类
        mod = Model()
        modkw = ModelCallByKW()

        # 创建一个 torch.classes._TorchScriptTesting._Foo 实例 foo，参数为 10 和 20
        foo = torch.classes._TorchScriptTesting._Foo(10, 20)
        # 创建一个形状为 (3, 3) 的全为 1 的张量 x
        x = torch.ones(3, 3)
        # 创建一个空的 torch.classes._TorchScriptTesting._TensorQueue 实例 tq
        tq = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        # 命名空间字符串 ns
        ns = "_TorchScriptTesting"
        # 使用 torch.library._scoped_library 方法创建名为 lib 的上下文管理器
        with torch.library._scoped_library(ns, "FRAGMENT") as lib:
            # 获取 torch.ops._TorchScriptTesting.queue_push 方法的操作符 op
            op = torch.ops._TorchScriptTesting.queue_push
            # 使用 lib.impl 方法实现 op 方法的三种内核处理方式
            lib.impl(op.__name__, torch.library.fallthrough_kernel, "AutogradCPU")
            lib.impl(op.__name__, torch.library.fallthrough_kernel, "ADInplaceOrView")
            lib.impl(
                op.__name__,
                torch.library.fallthrough_kernel,
                "PythonTLSSnapshot",
            )

            # 在捕获 RuntimeError 异常的上下文中，验证 make_fx 方法对于 mod 的调用抛出预期异常
            with self.assertRaisesRegex(
                RuntimeError, "is expected to be a FakeScriptObject"
            ):
                _ = make_fx(mod, tracing_mode="fake")(tq, x, foo)

            # 在捕获 RuntimeError 异常的上下文中，验证 make_fx 方法对于 modkw 的调用抛出预期异常
            with self.assertRaisesRegex(
                RuntimeError, "is expected to be a FakeScriptObject"
            ):
                _ = make_fx(modkw, tracing_mode="fake")(tq, x, foo)

    @parametrize("fallthrough_via", ["lib_impl", "py_impl"])
    def test_make_fx_tensor_queue_operators(self, fallthrough_via):
        # 定义一个模型类，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 前向传播方法，接受两个参数 tq 和 x
            def forward(self, tq, x):
                # 使用 torch.autocast 在 CUDA 上进行自动混合精度计算，数据类型为 torch.bfloat16
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # 将 x 的余弦值推送到队列 tq 中
                    torch.ops._TorchScriptTesting.queue_push(tq, x.cos())
                    # 将 x 的正弦值推送到队列 tq 中
                    torch.ops._TorchScriptTesting.queue_push(tq, x.sin())
                    # 从队列 tq 中弹出一个元素，减去当前队列大小
                    x_sin = torch.ops._TorchScriptTesting.queue_pop(
                        tq
                    ) - torch.ops._TorchScriptTesting.queue_size(tq)
                    # 从队列 tq 中弹出一个元素，加上当前队列大小
                    x_cos = torch.ops._TorchScriptTesting.queue_pop(
                        tq
                    ) + torch.ops._TorchScriptTesting.queue_size(tq)
                    # 返回 x_sin, x_cos 和更新后的队列 tq
                    return x_sin, x_cos, tq

        # 创建模型实例
        mod = Model()

        # 创建两个 TensorQueue 实例 tq1 和 tq2
        tq1 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        tq2 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        # 创建一个形状为 (2, 3) 的全 1 张量 x
        x = torch.ones(2, 3)

        # 调用模型的前向传播方法
        mod(tq1, x)

        # 定义操作列表 ops，包含三个 TorchScriptTesting 模块中的操作
        ops = [
            torch.ops._TorchScriptTesting.queue_push,
            torch.ops._TorchScriptTesting.queue_pop,
            torch.ops._TorchScriptTesting.queue_size,
        ]
        # 根据 fallthrough_via 的值选择不同的操作实现方式
        if fallthrough_via == "lib_impl":
            ns = "_TorchScriptTesting"
            # 使用 torch.library._scoped_library 进行库作用域的设置
            with torch.library._scoped_library(ns, "FRAGMENT") as lib:
                # 针对 ops 中的每个操作，使用 lib.impl 将其实现为 fallthrough_kernel
                for op in ops:
                    lib.impl(
                        op.__name__, torch.library.fallthrough_kernel, "AutocastCUDA"
                    )

                # 使用 make_fx 生成模型的函数化表示，tracing_mode 设置为 "fake"
                gm = make_fx(mod, tracing_mode="fake")(tq1, x)
        else:
            # 对于 ops 中的每个操作，使用 op.default.py_impl 设置 AutocastCUDA 的实现方式
            for op in ops:
                op.default.py_impl(torch._C.DispatchKey.AutocastCUDA)(
                    torch.library.fallthrough_kernel
                )
            # 使用 make_fx 生成模型的函数化表示，tracing_mode 设置为 "fake"
            gm = make_fx(mod, tracing_mode="fake")(tq1, x)
            # 清除操作的分发缓存和 py_kernels 字典
            for op in ops:
                op.default._dispatch_cache.clear()
                del op.default.py_kernels[torch._C.DispatchKey.AutocastCUDA]

        # 使用 self.assertExpectedInline 断言生成的模型代码与预期代码一致
        self.assertExpectedInline(
            gm.code.strip(),
            """\
# 定义模型的前向传播方法，接受三个参数 arg0_1, arg1_1, arg2_1
def forward(self, arg0_1, arg1_1, arg2_1):
    # 计算输入张量 arg2_1 的余弦值
    cos = torch.ops.aten.cos.default(arg2_1)
    # 将余弦值推入指定队列 arg0_1 中，并获取效果（可能是状态或返回值）
    with_effects = torch._higher_order_ops.effects.with_effects(arg0_1, torch.ops._TorchScriptTesting.queue_push.default, arg1_1, cos);  arg0_1 = cos = None
    # 从 with_effects 中获取第一个元素
    getitem = with_effects[0];  with_effects = None
    # 计算输入张量 arg2_1 的正弦值
    sin = torch.ops.aten.sin.default(arg2_1);  arg2_1 = None
    # 将正弦值推入指定队列 arg0_1 中，并获取效果（可能是状态或返回值）
    with_effects_1 = torch._higher_order_ops.effects.with_effects(getitem, torch.ops._TorchScriptTesting.queue_push.default, arg1_1, sin);  getitem = sin = None
    # 从 with_effects_1 中获取第一个元素
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    # 从指定队列 arg0_1 中弹出一个元素，并与标量 1 相减
    with_effects_2 = torch._higher_order_ops.effects.with_effects(getitem_2, torch.ops._TorchScriptTesting.queue_pop.default, arg1_1);  getitem_2 = None
    # 从 with_effects_2 中获取第一个元素
    getitem_4 = with_effects_2[0]
    # 从 with_effects_2 中获取索引为 1 的元素，然后将 with_effects_2 置为 None
    getitem_5 = with_effects_2[1];  with_effects_2 = None
    # 使用 getitem_4 和 arg1_1 作为参数调用 torch._higher_order_ops.effects.with_effects 函数，得到 with_effects_3
    with_effects_3 = torch._higher_order_ops.effects.with_effects(getitem_4, torch.ops._TorchScriptTesting.queue_size.default, arg1_1);  getitem_4 = None
    # 从 with_effects_3 中获取索引为 0 的元素，然后将 with_effects_3 置为 None
    getitem_6 = with_effects_3[0];  with_effects_3 = None
    # 使用 getitem_5 和 1 作为参数调用 torch.ops.aten.sub.Tensor 函数，得到 sub
    sub = torch.ops.aten.sub.Tensor(getitem_5, 1);  getitem_5 = None
    # 使用 getitem_6 和 arg1_1 作为参数调用 torch._higher_order_ops.effects.with_effects 函数，得到 with_effects_4
    with_effects_4 = torch._higher_order_ops.effects.with_effects(getitem_6, torch.ops._TorchScriptTesting.queue_pop.default, arg1_1);  getitem_6 = None
    # 从 with_effects_4 中获取索引为 0 的元素，然后将 with_effects_4 置为 None
    getitem_8 = with_effects_4[0]
    # 从 with_effects_4 中获取索引为 1 的元素，然后将 with_effects_4 置为 None
    getitem_9 = with_effects_4[1];  with_effects_4 = None
    # 使用 getitem_8 和 arg1_1 作为参数调用 torch._higher_order_ops.effects.with_effects 函数，得到 with_effects_5
    with_effects_5 = torch._higher_order_ops.effects.with_effects(getitem_8, torch.ops._TorchScriptTesting.queue_size.default, arg1_1);  getitem_8 = None
    # 从 with_effects_5 中获取索引为 0 的元素，然后将 with_effects_5 置为 None
    getitem_10 = with_effects_5[0];  with_effects_5 = None
    # 使用 getitem_9 和 0 作为参数调用 torch.ops.aten.add.Tensor 函数，得到 add
    add = torch.ops.aten.add.Tensor(getitem_9, 0);  getitem_9 = None
    # 返回元组 (getitem_10, sub, add, arg1_1)
    return (getitem_10, sub, add, arg1_1)""",  # noqa: B950
        )
    def forward(self, tq, x):
        # 使用 fx_pytree 模块的 tree_flatten_spec 函数将 tq 和 x 扁平化为列表，符合输入规范
        tq, x, = fx_pytree.tree_flatten_spec(([tq, x], {}), self._in_spec)
        # 调用 TorchScript 测试库中的 queue_push.default 方法，将 tq 和 x 推送到队列中
        queue_push_default = torch.ops._TorchScriptTesting.queue_push.default(tq, x);  x = None
        # 使用 pytree 模块的 tree_unflatten 函数将 tq 还原为原始结构，符合输出规范
        return pytree.tree_unflatten((tq,), self._out_spec)""",
        )
        # 断言验证生成的内联代码是否符合预期
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, tq, x):
    # 调用 TorchScript 测试库中的 queue_push.default 方法，将 tq 和 x 推送到队列中
    queue_push_default = torch.ops._TorchScriptTesting.queue_push.default(tq, x);  x = None
    # 返回 tq 对象的元组
    return (tq,)""",
        )
        # 断言验证生成的内联图形代码是否符合预期
        self.assertExpectedInline(
            str(ep.graph_module.graph).strip(),
            """\
graph():
    %tq : [num_users=2] = placeholder[target=tq]
    %x : [num_users=1] = placeholder[target=x]
    %queue_push_default : [num_users=0] = call_function[target=torch.ops._TorchScriptTesting.queue_push.default](args = (%tq, %x), kwargs = {})
    return (tq,)""",  # noqa: B950
        )
    # 定义一个测试方法，用于编译脚本对象作为输入参数
    def test_compile_script_object_input(self, backend):
        # 如果 backend 是 "eager"，则使用 EagerAndRecordGraphs() 替代
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        # 定义一个简单的神经网络模型类
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个名为 check_tq_is_fake 的属性，并设置为 True
                self.check_tq_is_fake = True

            # 定义神经网络模型的前向传播方法
            def forward(self, tq, x):
                # 调用 tq 对象的 push 方法，将 x 的余弦值推入队列
                tq.push(x.cos())
                # 调用 tq 对象的 push 方法，将 x 的正弦值推入队列
                tq.push(x.sin())
                # 从 tq 对象中弹出一个值，并减去当前队列的大小
                x_sin = tq.pop() - tq.size()
                # 返回 x_sin 值和 tq 对象
                return x_sin, tq

        # 创建一个 Model 类的实例 mod
        mod = Model()

        # 创建四个 _TensorQueue 对象实例 tq1, tq2, tq3, tq4，初始化为填充值为 -1 的空张量
        tq1 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        tq2 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        tq3 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        tq4 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )

        # 创建一个形状为 (2, 3) 的随机张量 x
        x = torch.randn(2, 3)

        # 调用 torch.compile 函数编译模型 mod，传入参数 tq1 和 x，并使用指定的 backend
        ret = torch.compile(mod, backend=backend)(tq1, x)

        # 直接调用模型 mod 的 forward 方法，传入参数 tq2 和 x，得到返回值 eager_ret
        eager_ret = mod(tq2, x)

        # 使用自定义的 _assertEqualSkipScriptObject 方法比较 ret 和 eager_ret 的值
        _assertEqualSkipScriptObject(self, ret, eager_ret)

        # 断言 ret 和 eager_ret 的第二个返回值的大小相等
        self.assertEqual(ret[1].size(), eager_ret[1].size())

        # 断言 ret 和 eager_ret 的第二个返回值中弹出的值相等
        self.assertEqual(ret[1].pop(), eager_ret[1].pop())

        # 如果 backend 是 "eager"，则执行以下断言
        if backend == "eager":
            # 断言 captured graph 中不返回 L_tq_ 作为输出，因为它被检测为输入，因此不返回为图输出
            # 相关逻辑可查看 dynamo/codegen.py 文件
            self.assertExpectedInline(
                backend.graphs[0].code.strip(),
                """\
    def forward(self, L_tq_ : torch.ScriptObject, L_x_ : torch.Tensor):
        l_tq_ = L_tq_
        l_x_ = L_x_
        cos = l_x_.cos()
        call_torchbind = torch.ops.higher_order.call_torchbind(l_tq_, 'push', cos);  cos = None
        sin = l_x_.sin();  l_x_ = None
        call_torchbind_1 = torch.ops.higher_order.call_torchbind(l_tq_, 'push', sin);  sin = None
        call_torchbind_2 = torch.ops.higher_order.call_torchbind(l_tq_, 'pop')
        call_torchbind_3 = torch.ops.higher_order.call_torchbind(l_tq_, 'size');  l_tq_ = None
        x_sin = call_torchbind_2 - 1;  call_torchbind_2 = None
        return (x_sin,)""",
            )
    # 定义一个测试方法，用于验证编译脚本对象的输入保护机制，接受一个后端参数
    def test_compile_script_object_input_guards(self, backend):
        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            # 模型初始化函数
            def __init__(self):
                super().__init__()
                self.check_tq_is_fake = True  # 设置一个成员变量

            # 前向传播函数
            def forward(self, tq, x):
                tq.push(x.cos())  # 在输入队列 tq 中推入 x 的余弦值
                tq.push(x.sin())  # 在输入队列 tq 中推入 x 的正弦值
                x_sin = tq.pop() - tq.size()  # 从队列 tq 中弹出一个值，并计算 x_sin
                return x_sin, tq  # 返回 x_sin 和更新后的队列 tq

        mod = Model()  # 创建 Model 类的实例
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)  # 使用指定后端创建编译计数器实例
        x = torch.randn(2, 3)  # 创建一个大小为 (2, 3) 的随机张量 x

        tq1 = _empty_tensor_queue()  # 创建一个空的张量队列 tq1
        torch.compile(mod, backend=cnt)(tq1, x)  # 编译模型 mod，使用 cnt 后端，对输入 tq1 和 x 进行操作
        self.assertEqual(cnt.frame_count, 1)  # 断言编译帧数为 1

        tq2 = _empty_tensor_queue()  # 创建另一个空的张量队列 tq2
        for _ in range(10):
            tq2.push(torch.randn(4, 5, requires_grad=False))  # 向队列 tq2 中推入 10 个大小为 (4, 5) 的随机张量
        torch.compile(mod, backend=cnt)(tq2, x)  # 再次编译模型 mod，使用 cnt 后端，对输入 tq2 和 x 进行操作
        # 队列长度变化导致重新编译
        self.assertEqual(cnt.frame_count, 2)  # 断言编译帧数为 2

        tq3 = _empty_tensor_queue()  # 创建另一个空的张量队列 tq3
        tq3.push(torch.randn(2, 3, requires_grad=False))  # 向队列 tq3 中推入一个大小为 (2, 3) 的随机张量
        torch.compile(mod, backend=cnt)(tq3, x)  # 再次编译模型 mod，使用 cnt 后端，对输入 tq3 和 x 进行操作
        # 队列中张量形状变化导致重新编译
        self.assertEqual(cnt.frame_count, 3)  # 断言编译帧数为 3

        tq4 = _empty_tensor_queue()  # 创建另一个空的张量队列 tq4
        tq4.push(torch.randn(2, 3, requires_grad=False))  # 向队列 tq4 中推入一个大小为 (2, 3) 的随机张量
        torch.compile(mod, backend=cnt)(tq4, x)  # 再次编译模型 mod，使用 cnt 后端，对输入 tq4 和 x 进行操作
        # 没有重新编译
        self.assertEqual(cnt.frame_count, 3)  # 断言编译帧数为 3

        tq5 = _empty_tensor_queue()  # 创建另一个空的张量队列 tq5
        tq5.push(torch.randn(2, 3, requires_grad=True))  # 向队列 tq5 中推入一个大小为 (2, 3) 的随机张量，要求梯度
        torch.compile(mod, backend=cnt)(tq5, x)  # 再次编译模型 mod，使用 cnt 后端，对输入 tq5 和 x 进行操作
        # 队列中张量的调度键变化导致重新编译
        self.assertEqual(cnt.frame_count, 4)  # 断言编译帧数为 4

        tq6 = _empty_tensor_queue()  # 创建另一个空的张量队列 tq6
        tq6.push(torch.randn(2, 3, requires_grad=True, dtype=torch.float64))  # 向队列 tq6 中推入一个大小为 (2, 3)、要求梯度、数据类型为 torch.float64 的随机张量
        torch.compile(mod, backend=cnt)(tq6, x)  # 再次编译模型 mod，使用 cnt 后端，对输入 tq6 和 x 进行操作
        # 队列中张量的数据类型变化导致重新编译
        self.assertEqual(cnt.frame_count, 5)  # 断言编译帧数为 5
    def test_compile_script_object_input_automatic_dynamic_shape(self):
        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.check_tq_is_fake = True

            # 定义模型的前向传播方法
            def forward(self, tq, x):
                # 将 x 的余弦值推入 tensor queue (tq)
                tq.push(x.cos())
                # 将 x 的正弦值推入 tensor queue (tq)
                tq.push(x.sin())
                # 弹出 tensor queue (tq) 中的值，并减去当前队列的大小
                x_sin = tq.pop() - tq.size()
                return x_sin, tq

        # 创建 Model 类的实例
        mod = Model()
        # 创建一个计数器对象
        cnt = torch._dynamo.testing.CompileCounter()
        # 生成一个形状为 (2, 3) 的随机张量
        x = torch.randn(2, 3)

        # 创建一个空的 tensor queue 对象
        tq1 = _empty_tensor_queue()
        # 将一个形状为 (2, 3) 的随机张量推入 tensor queue (tq1)，不需要梯度
        tq1.push(torch.randn(2, 3, requires_grad=False))
        # 使用 torch.compile 函数对 mod 进行编译，使用 cnt 作为后端，并传入 tq1 和 x 作为输入
        torch.compile(mod, backend=cnt)(tq1, x)
        # 断言编译帧数为 1
        self.assertEqual(cnt.frame_count, 1)

        # 创建另一个空的 tensor queue 对象
        tq2 = _empty_tensor_queue()
        # 将一个形状为 (2, 4) 的随机张量推入 tensor queue (tq2)，不需要梯度
        tq2.push(torch.randn(2, 4, requires_grad=False))
        # 使用 torch.compile 函数对 mod 进行编译，使用 cnt 作为后端，并传入 tq2 和 x 作为输入
        torch.compile(mod, backend=cnt)(tq2, x)
        # 断言编译帧数为 2
        self.assertEqual(cnt.frame_count, 2)

        # 创建另一个空的 tensor queue 对象
        tq3 = _empty_tensor_queue()
        # 将一个形状为 (2, 5) 的随机张量推入 tensor queue (tq3)，不需要梯度
        tq3.push(torch.randn(2, 5, requires_grad=False))
        # 使用 torch.compile 函数对 mod 进行编译，使用 cnt 作为后端，并传入 tq3 和 x 作为输入
        torch.compile(mod, backend=cnt)(tq3, x)
        # 断言编译帧数仍为 2，因为没有重新编译
        self.assertEqual(cnt.frame_count, 2)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_compile_error_on_input_aliasing_contents(self, backend):
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.check_tq_is_fake = True

            # 定义模型的前向传播方法
            def forward(self, tq, x):
                # 返回 x 的正弦值和 tensor queue (tq) 的弹出值的余弦值
                return x.sin(), tq.pop().cos()

        # 生成一个形状为 (2, 3) 的随机张量
        x = torch.randn(2, 3)
        # 创建 Model 类的实例
        mod = Model()

        # 创建一个空的 tensor queue 对象
        tq1 = _empty_tensor_queue()
        # 将 x 推入 tensor queue (tq1)
        tq1.push(x)
        # 使用 torch.compile 函数对 mod 进行编译，使用 backend 作为后端，并传入 tq1 和 x 作为输入
        # 断言会抛出 RuntimeError 异常，内容包含 "is alising"
        with self.assertRaisesRegex(RuntimeError, "is alising"):
            torch.compile(mod, backend=backend)(tq1, x)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_compile_error_on_script_obj_setattr(self, backend):
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        # 定义一个设置属性的函数 setattr_f
        def setattr_f(tq):
            # 给 tensor queue (tq) 设置属性 a = 1
            tq.a = 1
            return tq

        # 断言会抛出 RuntimeError 异常，内容包含 "call method __setattr__ on script object is not safe"
        with self.assertRaisesRegex(
            RuntimeError, "call method __setattr__ on script object is not safe"
        ):
            # 使用 torch.compile 函数对 setattr_f 进行编译，使用 backend 作为后端，并传入空的 tensor queue 对象作为输入
            torch.compile(setattr_f, backend=backend)(_empty_tensor_queue())

    @parametrize("backend", ["eager", "aot_eager"])
    def test_compile_error_on_script_obj_missing_attr(self, backend):
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        # 定义一个获取不存在属性的函数 setattr_f
        def setattr_f(tq):
            # 返回 tensor queue (tq) 的一个未定义的属性 _not_defined_attr
            return tq._not_defined_attr

        # 断言会抛出 RuntimeError 异常，内容包含 "doesn't define method _not_defined_attr"
        with self.assertRaisesRegex(
            RuntimeError, "doesn't define method _not_defined_attr"
        ):
            # 使用 torch.compile 函数对 setattr_f 进行编译，使用 backend 作为后端，并传入空的 tensor queue 对象作为输入
            torch.compile(setattr_f, backend=backend)(_empty_tensor_queue())

    @parametrize("backend", ["eager", "aot_eager"])
    # 定义测试函数，用于测试编译后的函数与原函数行为是否一致，接受两个参数：tq（tensor queue）和 x（输入张量）
    def test_compile_body_aliasing_contents(self, backend):
        # 如果后端是 "eager"，则将 backend 替换为 EagerAndRecordGraphs 的实例
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        # 定义内部函数 f，接受参数 tq 和 x
        def f(tq, x):
            # x1 是 x 的展平视图
            x1 = x.view(-1)
            # x2 是 x 的转置，维度顺序为 (1, 0)
            x2 = x.permute(1, 0)
            # 将 x1 和 x2 推入 tq 中
            tq.push(x1)
            tq.push(x2)
            # 返回操作后的 x1、x2 以及更新后的 tq
            return x1 - tq.size(), x2 + tq.size(), tq

        # 生成一个形状为 (2, 3) 的随机张量 x
        x = torch.randn(2, 3)
        # 使用自定义函数 _assertEqualScriptObject 断言编译后的函数与原函数行为一致
        _assertEqualScriptObject(
            self,
            f(_empty_tensor_queue(), x),  # 调用原始函数 f
            torch.compile(f, backend=backend)(_empty_tensor_queue(), x),  # 编译并调用 f
        )
        # 如果不处于 Torch 动态编译模式且后端为 "eager"
        if not torch._dynamo.is_compiling() and backend == "eager":
            # 使用 self.assertExpectedInline 断言 backend 中第一个图形的代码与预期一致
            self.assertExpectedInline(
                backend.graphs[0].code.strip(),
                """\
    def forward(self, L_x_ : torch.Tensor, L_tq_ : torch.ScriptObject):
        # 将输入参数赋值给局部变量
        l_x_ = L_x_
        l_tq_ = L_tq_
        # 将 l_x_ 展平为一维张量
        x1 = l_x_.view(-1)
        # 将 l_x_ 进行维度置换
        x2 = l_x_.permute(1, 0);  l_x_ = None
        # 调用 torchbind 的 'push' 方法，将 x1 推送到 l_tq_ 中
        call_torchbind = torch.ops.higher_order.call_torchbind(l_tq_, 'push', x1)
        # 调用 torchbind 的 'push' 方法，将 x2 推送到 l_tq_ 中
        call_torchbind_1 = torch.ops.higher_order.call_torchbind(l_tq_, 'push', x2)
        # 调用 torchbind 的 'size' 方法，获取 l_tq_ 的大小
        call_torchbind_2 = torch.ops.higher_order.call_torchbind(l_tq_, 'size')
        # 计算 x1 减去 2
        sub = x1 - 2;  x1 = None
        # 调用 torchbind 的 'size' 方法，获取 l_tq_ 的大小，并释放 l_tq_
        call_torchbind_3 = torch.ops.higher_order.call_torchbind(l_tq_, 'size');  l_tq_ = None
        # 计算 x2 加上 2，并释放 x2
        add = x2 + 2;  x2 = None
        # 返回 sub 和 add 作为结果
        return (sub, add)
    # 定义一个测试函数，用于测试编译对象闭包
    def test_compile_obj_closure(self, backend):
        # 定义一个内部函数 f(x)，其中 x 是输入参数
        def f(x):
            # 定义内部函数 inner_f(x)，接收 x 作为参数，并将 x 的正弦值推送到全局变量 tq 中
            def inner_f(x):
                tq.push(x.sin())

            # 调用 inner_f(x) 函数
            inner_f(x)
            # 返回从全局变量 tq 中弹出的值以及 tq 对象本身
            return tq.pop(), tq

        # 使用 torch.compile 函数将 f 编译为优化后的函数 opt_f，使用指定的后端 backend
        opt_f = torch.compile(f, backend="eager")

        # 初始化一个空的 tensor 队列 tq
        tq = _empty_tensor_queue()
        # 生成一个形状为 (3, 2) 的随机张量 x
        x = torch.randn(3, 2)
        # 使用自定义的断言函数 _assertEqualScriptObject 检查 f(x) 和 opt_f(x) 的输出是否一致
        _assertEqualScriptObject(self, f(x), opt_f(x))

    # 使用不同的后端参数来参数化测试函数
    @parametrize("backend", ["eager", "aot_eager"])
    # 定义一个测试函数，用于测试全局对象的编译
    def test_compile_global_obj(self, backend):
        # 将全局变量 _TENSOR_QUEUE_GLOBAL_TEST 初始化为一个空的 tensor 队列
        global _TENSOR_QUEUE_GLOBAL_TEST
        _TENSOR_QUEUE_GLOBAL_TEST = _empty_tensor_queue()

        # 定义一个函数 f(x)，接收 x 作为参数
        def f(x):
            # 将 x 的正弦值推送到全局变量 _TENSOR_QUEUE_GLOBAL_TEST 中
            _TENSOR_QUEUE_GLOBAL_TEST.push(x.sin())
            # 返回从全局变量 _TENSOR_QUEUE_GLOBAL_TEST 中弹出的值以及 _TENSOR_QUEUE_GLOBAL_TEST 对象本身
            return _TENSOR_QUEUE_GLOBAL_TEST.pop(), _TENSOR_QUEUE_GLOBAL_TEST

        # 使用 torch.compile 函数将 f 编译为优化后的函数 opt_f，使用指定的后端 backend
        opt_f = torch.compile(f, backend=backend)
        # 生成一个形状为 (3, 2) 的随机张量 x
        x = torch.randn(3, 2)
        # 使用自定义的断言函数 _assertEqualScriptObject 检查 f(x) 和 opt_f(x) 的输出是否一致
        eager_ret = f(x)
        opt_ret = opt_f(x)
        _assertEqualScriptObject(self, eager_ret, opt_ret)

    # 定义一个测试函数，用于测试对象图中断的编译情况
    def test_compile_obj_graph_breaks(self):
        # 初始化一个计数器 cnt，用于统计编译过程中的帧数
        cnt = torch._dynamo.testing.CompileCounter()

        # 定义一个函数 f(tq, x)，接收 tensor 队列 tq 和输入张量 x 作为参数
        def f(tq, x):
            # 将 x 的正弦值推送到 tensor 队列 tq 中两次
            tq.push(x.sin())
            tq.push(x.sin())
            # 调用 torch._dynamo.graph_break() 来表示图中断
            torch._dynamo.graph_break()
            # 从 tensor 队列 tq 中弹出一个元素
            tq.pop()
            torch._dynamo.graph_break()
            # 将 x 的余弦值加上 tensor 队列 tq 的大小并推送到 tq 中
            tq.push(x.cos() + tq.size())
            torch._dynamo.graph_break()
            # 将 x 的余弦值减去 tensor 队列 tq 的大小并推送到 tq 中
            tq.push(x.cos() - tq.size())
            # 返回输入张量 x 以及从 tensor 队列 tq 中弹出的值以及 tq 对象本身
            return x, tq.pop(), tq

        # 使用 torch.compile 函数将 f 编译为优化后的函数 opt_f，使用计数器 cnt 作为后端
        opt_f = torch.compile(f, backend=cnt)
        # 生成一个形状为 (3, 2) 的随机张量 x
        x = torch.randn(3, 2)
        # 使用自定义的断言函数 _assertEqualScriptObject 检查 f(_empty_tensor_queue(), x) 和 opt_f(_empty_tensor_queue(), x) 的输出是否一致
        _assertEqualScriptObject(
            self, f(_empty_tensor_queue(), x), opt_f(_empty_tensor_queue(), x)
        )
        # 使用内置断言函数 self.assertEqual 检查 cnt.frame_count 是否等于 4
        self.assertEqual(cnt.frame_count, 4)

    # 使用不同的后端参数来参数化测试函数
    @parametrize("backend", ["eager", "aot_eager"])
    # 定义一个测试函数，用于测试对象属性的编译情况
    def test_compile_obj_attributes(self, backend):
        # 如果后端参数为 "eager"，则将 backend 设置为 EagerAndRecordGraphs() 对象
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个 tensor 队列 self.tq
                self.tq = _empty_tensor_queue()

            # 定义模型的前向传播函数 forward，接收输入张量 x 作为参数
            def forward(self, x):
                # 将输入张量 x 推送到 tensor 队列 self.tq 中
                self.tq.push(x)
                # 返回从 tensor 队列 self.tq 中弹出的值
                return self.tq.pop()

        # 生成一个形状为 (2, 3) 的随机张量 x
        x = torch.randn(2, 3)
        # 使用 torch.compile 函数将 Model 类实例编译为优化后的函数 opt_f，使用指定的后端 backend
        opt_f = torch.compile(Model(), backend=backend)
        # 使用自定义的断言函数 _assertEqualScriptObject 检查 Model()(x) 和 opt_f(x) 的输出是否一致
        _assertEqualScriptObject(self, Model()(x), opt_f(x))
        # 如果后端参数为 "eager"，则进一步检查 backend.graphs 中生成的图的信息是否符合预期
        if backend == "eager":
            self.assertEqual(len(backend.graphs), 1)
            # 使用 self.assertExpectedInline 检查 backend.graphs[0].code 的代码是否符合预期
            self.assertExpectedInline(
                backend.graphs[0].code.strip(),
                """\
    def forward(self, L_self_tq : torch.ScriptObject, L_x_ : torch.Tensor):
        l_self_tq = L_self_tq
        l_x_ = L_x_
        call_torchbind = torch.ops.higher_order.call_torchbind(l_self_tq, 'push', l_x_);  l_x_ = None
        call_torchbind_1 = torch.ops.higher_order.call_torchbind(l_self_tq, 'pop');  l_self_tq = None
        return (call_torchbind_1,)""",
            )

    # 使用不同的后端参数来参数化测试函数
    @parametrize("backend", ["eager", "aot_eager"])
    # 定义一个测试函数，用于编译并测试 TorchScript 操作
    def test_compile_obj_torchbind_op(self, backend):
        # 定义内部函数 f，接受队列 tq 和张量 x 作为参数
        def f(tq, x):
            # 将 x 的余弦值推送到队列 tq 中
            torch.ops._TorchScriptTesting.queue_push(tq, x.cos())
            # 将 x 的余弦值加 1 推送到队列 tq 中
            torch.ops._TorchScriptTesting.queue_push(tq, x.cos() + 1)
            # 从队列 tq 中弹出一个元素
            torch.ops._TorchScriptTesting.queue_pop(tq)
            # 将 x 的正弦值推送到队列 tq 中
            torch.ops._TorchScriptTesting.queue_push(tq, x.sin())
            # 返回队列 tq 的弹出结果、队列大小加弹出结果、以及队列 tq 本身
            return tq.pop(), tq.pop() + tq.size(), tq

        # 使用指定的后端 backend 编译函数 f，返回优化后的函数 opt_f
        opt_f = torch.compile(f, backend=backend)
        # 创建一个包含随机数据的张量 x
        x = torch.randn(2)
        # 使用自定义断言函数 _assertEqualScriptObject 检查原始函数 f 和优化后函数 opt_f 的输出是否一致
        _assertEqualScriptObject(
            self, f(_empty_tensor_queue(), x), opt_f(_empty_tensor_queue(), x)
        )
# 如果 Torch Dynamo 不支持 torchbind，则跳过这个测试类
@skipIfTorchDynamo("torchbind not supported with dynamo yet")
# 定义一个测试类 TestRegisterFakeClass，继承自 TestCase
class TestRegisterFakeClass(TestCase):
    # 在每个测试方法执行前调用，初始化 TorchBind 实现
    def setUp(self):
        init_torchbind_implementations()

    # 在每个测试方法执行后调用，清空全局假类注册表
    def tearDown(self):
        torch._library.fake_class_registry.global_fake_class_registry.clear()

    # 测试注册假类，当没有对应的 TorchBind 类时应抛出 RuntimeError 异常
    def test_register_fake_class_no_torch_bind_class(self):
        with self.assertRaisesRegex(RuntimeError, "Tried to instantiate class"):

            # 注册一个名为 Invalid 的假类，应当引发异常
            @torch._library.register_fake_class("_TorchScriptTesting::NOT_A_VALID_NAME")
            class Invalid:
                pass

    # 测试注册假类，当实现中没有定义类方法 __obj_unflatten__ 时应抛出 RuntimeError 异常
    def test_register_fake_class_no_from_real(self):
        with self.assertRaisesRegex(
            RuntimeError, "define a classmethod __obj_unflatten__"
        ):

            # 注册一个名为 InvalidFakeFoo 的假类，缺少类方法 __obj_unflatten__
            @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
            class InvalidFakeFoo:
                def __init__(self):
                    pass

    # 测试注册假类，当 __obj_unflatten__ 方法不是类方法时应抛出 RuntimeError 异常
    def test_register_fake_class_from_real_not_classmethod(self):
        with self.assertRaisesRegex(RuntimeError, "is not a classmethod"):

            # 注册一个名为 FakeFoo 的假类，__obj_unflatten__ 方法不是类方法
            @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
            class FakeFoo:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y

                # 缺少 @classmethod 装饰器，不是类方法
                def __obj_unflatten__(cls, flattend_foo):  # noqa: B902
                    return cls(**dict(flattend_foo))

    # 测试注册有效的假类 FakeFoo
    def test_register_fake_class_valid(self):
        # 定义一个名为 FakeFoo 的假类
        class FakeFoo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            # 定义类方法 __obj_unflatten__，用于从扁平化对象还原
            @classmethod
            def __obj_unflatten__(cls, flattend_foo):
                return cls(**dict(flattend_foo))

        # 注册假类 FakeFoo 到 "_TorchScriptTesting::_Foo"
        torch._library.register_fake_class("_TorchScriptTesting::_Foo", FakeFoo)

# 实例化参数化测试 TestExportTorchbind 和 TestCompileTorchbind
instantiate_parametrized_tests(TestExportTorchbind)
instantiate_parametrized_tests(TestCompileTorchbind)

# 如果是主程序入口，运行测试
if __name__ == "__main__":
    run_tests()
```