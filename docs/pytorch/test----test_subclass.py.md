# `.\pytorch\test\test_subclass.py`

```py
# Owner(s): ["module: nn"]

import tempfile  # 导入用于创建临时文件和目录的模块
from copy import deepcopy  # 导入深拷贝函数
from functools import partial  # 导入偏函数支持
from unittest import expectedFailure  # 导入用于标记预期失败测试的装饰器

import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块
from torch.nn.modules.lazy import LazyModuleMixin  # 导入懒加载模块的混合类
from torch.nn.utils.parametrize import (  # 导入参数化工具函数
    register_parametrization,
    remove_parametrizations,
)
from torch.testing._internal.common_subclass import (  # 导入用于测试的自定义子类和数据库
    DiagTensorBelow,
    subclass_db,
)
from torch.testing._internal.common_utils import (  # 导入通用测试工具函数和类
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
)
from torch.testing._internal.logging_tensor import LoggingTensor  # 导入用于记录日志的张量
from torch.utils._pytree import tree_map  # 导入用于操作树形数据结构的映射函数

# The current test methodology in this file is to test a variety of real use cases
# with a set of fully-fledged tensor subclasses. In the future, this may change
# to more narrowly specify toy subclasses for each of the specific invariants under
# test, avoiding the need to maintain the set of fully-fledged tensor subclasses.


# Decorator for parametrizing tests across the various tensor classes.
parametrize_tensor_cls = parametrize("tensor_cls", [
    subtest(tensor_cls, name=info.name) for tensor_cls, info in subclass_db.items()])

# 测试用例类，继承自unittest.TestCase，用于定义测试方法
class TestSubclass(TestCase):

    # 辅助方法：根据给定的tensor_cls创建一个张量实例
    def _create_tensor(self, tensor_cls):
        return subclass_db[tensor_cls].create_fn(3)

    # 参数化测试方法，测试不同的张量类和requires_grad参数组合
    @parametrize_tensor_cls
    @parametrize("tensor_requires_grad", [False, True])
    def test_param_invariants(self, tensor_cls, tensor_requires_grad):
        x = self._create_tensor(tensor_cls).requires_grad_(tensor_requires_grad)
        param = nn.Parameter(x, requires_grad=(not tensor_requires_grad))

        self.assertIsInstance(param, nn.Parameter)
        # Ensure requires_grad passed to Parameter's constructor takes precedence.
        self.assertEqual(param.requires_grad, not tensor_requires_grad)

        # Ensure original tensor is not mutated by Parameter construction.
        self.assertNotIsInstance(x, nn.Parameter)
        self.assertEqual(x.requires_grad, tensor_requires_grad)

    # 跳过Torch Dynamo时执行的测试方法装饰器
    @skipIfTorchDynamo()
    @parametrize_tensor_cls
    @parametrize("as_param", [False, True])
    def test_deepcopy(self, tensor_cls, as_param):
        x = self._create_tensor(tensor_cls)
        if as_param:
            x = nn.Parameter(x)
        x_copy = deepcopy(x)
        self.assertEqual(x, x_copy)
        self.assertEqual(x.__class__, x_copy.__class__)
        self.assertIsNot(x, x_copy)
        self.assertIsInstance(x_copy, tensor_cls)
        if as_param:
            # Deepcopy should preserve both custom type and "parameter-ness".
            self.assertIsInstance(x_copy, nn.Parameter)

    # 参数化测试方法，测试不同的张量类和as_param参数组合
    @parametrize_tensor_cls
    @parametrize("as_param", [False, True])
    # 测试张量序列化功能，接受张量类和参数化标志作为输入
    def test_serialization(self, tensor_cls, as_param):
        # 使用临时文件对象创建上下文
        with tempfile.TemporaryFile() as f:
            # 创建一个张量对象
            x = self._create_tensor(tensor_cls)
            # 如果as_param标志为真，则将张量转换为参数
            if as_param:
                x = nn.Parameter(x)
            # 将张量对象序列化并保存到临时文件中
            torch.save(x, f)
            # 将文件指针移动到文件开头
            f.seek(0)
            # 从临时文件中加载并反序列化张量对象
            x_loaded = torch.load(f)

            # 断言序列化前后的张量对象相等
            self.assertEqual(x, x_loaded)
            # 断言序列化前后的张量对象不是同一个实例
            self.assertIsNot(x, x_loaded)
            # 断言反序列化后的对象类型是指定的张量类的实例
            self.assertIsInstance(x_loaded, tensor_cls)
            # 如果as_param标志为真，则断言反序列化后的对象同时也是nn.Parameter的实例
            if as_param:
                self.assertIsInstance(x_loaded, nn.Parameter)

    # 在functorch作为functorch修补程序时跳过测试
    @skipIfTorchDynamo("Visible only with functorch as functorch monkeypatches tensor str")
    # 参数化测试函数，接受张量类和as_param标志作为输入参数
    @parametrize_tensor_cls
    @parametrize("as_param", [False, True])
    def test_repr(self, tensor_cls, as_param):
        # 创建指定张量类的张量对象
        x = self._create_tensor(tensor_cls)
        # 如果as_param标志为真，则将张量转换为参数
        if as_param:
            x = nn.Parameter(x)
        # 获取张量对象的字符串表示
        str_repr = x.__repr__()
        # 断言字符串表示中张量类名出现的次数为1次
        if tensor_cls is not torch.Tensor:
            self.assertEqual(str_repr.count(f"{tensor_cls.__name__}("), 1)
        # 断言字符串表示中出现"Parameter"的次数为1次（如果as_param为真）
        self.assertEqual(str_repr.count("Parameter"), 1 if as_param else 0)

    # 参数化测试函数，测试自定义类型在张量操作中的传播
    @parametrize_tensor_cls
    @parametrize("as_param", [False, True])
    def test_type_propagation(self, tensor_cls, as_param):
        # 创建指定张量类的张量对象
        x = self._create_tensor(tensor_cls)
        # 如果as_param标志为真，则将张量转换为参数
        if as_param:
            x = nn.Parameter(x)

        # 调用加法操作生成一个输出张量
        output = x + self._create_tensor(torch.Tensor)

        # 如果subclass_db中记录了张量类的封闭性，则断言输出类型是张量类的实例
        if subclass_db[tensor_cls].closed_under_ops:
            self.assertIsInstance(output, tensor_cls)
        else:
            self.assertIsInstance(output, torch.Tensor)
        # 断言输出对象不是nn.Parameter的实例
        self.assertNotIsInstance(output, nn.Parameter)

    # 参数化测试函数，测试张量类和as_param标志的组合
    def test_module_optimization(self, tensor_cls):
        # 创建一个部分应用的函数，用于在类内部创建张量对象
        create_fn = partial(self._create_tensor, tensor_cls)

        # 定义一个继承自 nn.Module 的内部类 MyModule
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建并设置一个参数化张量对象 self.p1
                self.p1 = nn.Parameter(create_fn())

                # 创建一个参数化张量对象列表 self.p_list 包含三个张量
                self.p_list = nn.ParameterList([create_fn() for _ in range(3)])
                # 向 self.p_list 中追加一个额外的参数化张量对象
                self.p_list.append(create_fn())

                # 创建一个参数化张量对象字典 self.p_dict 包含 'foo' 和 'bar' 两个键值对
                self.p_dict = nn.ParameterDict({
                    'foo': create_fn(),
                    'bar': create_fn(),
                })
                # 向 self.p_dict 中添加一个名为 'baz' 的额外参数化张量对象
                self.p_dict['baz'] = create_fn()

                # 使用 torch.no_grad() 上下文管理器初始化参数化张量对象
                with torch.no_grad():
                    nn.init.normal_(self.p1)  # 对 self.p1 进行正态分布初始化
                    for p in self.p_list:
                        nn.init.uniform_(p)  # 对 self.p_list 中的每个张量进行均匀分布初始化
                    for p in self.p_dict.values():
                        nn.init.uniform_(p)  # 对 self.p_dict 中的每个值（张量）进行均匀分布初始化

            # 定义模型的前向传播方法
            def forward(self, x):
                out = self.p1 + x  # 将输入张量 x 加到 self.p1 上
                for p in self.p_list:
                    out = p + out  # 将 self.p_list 中的每个张量加到 out 上

                for v in self.p_dict.values():
                    out = v + out  # 将 self.p_dict 中的每个值（张量）加到 out 上

                return out  # 返回最终的输出张量

        m = MyModule()  # 创建 MyModule 的实例 m
        self.assertEqual(len(m.state_dict()), 8)  # 断言模型的状态字典中有 8 个元素

        # 使用 SGD 优化器优化模型 m 的参数
        optimizer = torch.optim.SGD(m.parameters(), lr=0.1)
        # 对模型 m 进行前向传播，计算输出的和并进行反向传播
        m(create_fn()).sum().backward(torch.tensor(1))
        optimizer.step()  # 执行优化步骤

    @parametrize_tensor_cls
    @parametrize("leave_parametrized", [False, True])
    def test_parametrization(self, tensor_cls, leave_parametrized):
        # TODO: Either implement set_() properly for these tensor subclasses or apply a
        # more general fix to avoid the need for special set_() handling. For now, skip
        # testing these as they're expected to fail.
        if tensor_cls in [LoggingTensor, DiagTensorBelow]:
            return  # 如果 tensor_cls 是 LoggingTensor 或 DiagTensorBelow，跳过测试

        create_fn = partial(self._create_tensor, tensor_cls)

        # 定义一个继承自 nn.Module 的内部类 MyModule
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建并设置一个参数化张量对象 self.weight
                self.weight = nn.Parameter(create_fn())

            # 定义模型的前向传播方法
            def forward(self, x):
                return self.weight + x  # 返回 self.weight 与输入张量 x 的和

        # 定义一个继承自 nn.Module 的内部类 MyParametrization
        class MyParametrization(nn.Module):
            def forward(self, X):
                return -X  # 返回输入张量 X 的负值

        m = MyModule()  # 创建 MyModule 的实例 m
        self.assertEqual(len(m.state_dict()), 1)  # 断言模型的状态字典中有 1 个元素
        register_parametrization(m, 'weight', MyParametrization())  # 将 MyParametrization 注册到模型 m 的 'weight' 参数上
        self.assertIsInstance(m.weight, tensor_cls)  # 断言模型 m 的 weight 参数是 tensor_cls 类型的实例
        output = m(self._create_tensor(torch.Tensor))
        self.assertIsInstance(output, tensor_cls)  # 断言模型 m 的输出是 tensor_cls 类型的实例
        remove_parametrizations(m, 'weight', leave_parametrized=leave_parametrized)  # 移除模型 m 的 'weight' 参数上的参数化设置

    # Lazy modules with custom tensors are not supported yet.
    @expectedFailure
    @parametrize_tensor_cls
    def test_lazy_module(self, tensor_cls):
        # 检查传入的张量类是否为 torch.Tensor，如果是则测试失败，直到子类的测试通过为止
        if tensor_cls is torch.Tensor:
            self.fail('dummy fail for base tensor until the test passes for subclasses')

        # 定义一个继承了 LazyModuleMixin 和 nn.Module 的懒加载模块类
        class MyLazyModule(LazyModuleMixin, nn.Module):
            def __init__(self):
                super().__init__()
                # 使用未初始化的参数创建一个参数对象
                self.param = nn.UninitializedParameter()

            def initialize_parameters(self, input) -> None:  # type: ignore[override]
                # 如果模块中存在未初始化的参数
                if self.has_uninitialized_params():
                    # 使用输入的形状来实例化参数
                    with torch.no_grad():
                        self.param.materialize(input.shape)
                        # 对参数进行均匀分布的初始化
                        nn.init.uniform_(self.param)

            def forward(self, x):
                # 模块的前向传播，返回参数加上输入张量 x 的结果
                return self.param + x

        # 创建 MyLazyModule 的实例
        m = MyLazyModule()
        # 断言模块中存在未初始化的参数
        self.assertTrue(m.has_uninitialized_params())
        # 使用给定的张量类创建一个张量，并传递给模块进行前向传播
        output = m(self._create_tensor(tensor_cls))
        # 断言模块中不存在未初始化的参数
        self.assertFalse(m.has_uninitialized_params())
        # 断言参数 param 的类型是给定的张量类 tensor_cls
        self.assertIsInstance(m.param, tensor_cls)

    def test_non_rewrapping_torch_dispatch_subclass_as_parameter_throws_for_detach(self):

        # 定义一个不会为其 __torch_dispatch__ 方法中的任何函数重新包装的子类
        class NonRewrappingTensor(torch.Tensor):
            @staticmethod
            def __new__(
                cls, t: torch.Tensor
            ):
                # 使用 _make_wrapper_subclass 方法创建一个不同类型的子类
                r = super()._make_wrapper_subclass(
                    cls, t.shape, dtype=t.dtype, requires_grad=t.requires_grad, device=t.device)
                return r

            def __init__(self, t) -> None:
                # 初始化时接受一个 torch.Tensor 类型的参数
                self.tensor: torch.Tensor = t

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

                def unwrap(e) -> torch.Tensor:
                    # 如果参数 e 是 NonRewrappingTensor 类的实例，则返回其内部的 tensor
                    if isinstance(e, NonRewrappingTensor):
                        t = e.tensor
                        return t
                    else:
                        return e

                # 对传入的参数 args 和 kwargs 中的 NonRewrappingTensor 类型进行解包操作
                r = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
                # 返回一个不再是原始子类类型的未包装张量
                return r

        # 测试期望抛出 RuntimeError，消息为 "requires that detach() returns an instance of the same type"
        with self.assertRaisesRegex(RuntimeError, r"requires that detach\(\) returns an instance of the same type"):
            # 创建一个 nn.Parameter 对象，传入一个 NonRewrappingTensor 类型的张量作为参数
            param = nn.Parameter(NonRewrappingTensor(torch.randn(3)))
    def test_tensor_subclass_storage_data_accesses_throw(self):
        # 导入 LoggingTensor 类
        from torch.testing._internal.logging_tensor import LoggingTensor
        # 创建一个包含两个元素的全1张量
        x = torch.ones(2)
        # 使用 LoggingTensor 包装张量 x
        x_log = LoggingTensor(x)
        # 访问张量子类的存储器是有效的
        storage = x_log.untyped_storage()
        # 这包括访问存储器上的元数据
        sz = storage.size()
        # 但是访问数据的存储器方法会抛出异常
        with self.assertRaisesRegex(RuntimeError, "on an invalid python storage"):
            storage.data_ptr()
        with self.assertRaisesRegex(RuntimeError, "on an invalid python storage"):
            storage.resize_(0)
        with self.assertRaisesRegex(RuntimeError, "on an invalid python storage"):
            storage.copy_(storage)
        with self.assertRaisesRegex(RuntimeError, "on an invalid python storage"):
            storage.fill_(0)
        with self.assertRaisesRegex(RuntimeError, "on an invalid python storage"):
            storage._write_file("file")
# 实例化一个带参数的测试类，并传入 TestSubclass 作为参数
instantiate_parametrized_tests(TestSubclass)

# 如果当前脚本作为主程序运行，则执行测试函数
if __name__ == '__main__':
    run_tests()
```