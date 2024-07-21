# `.\pytorch\test\nn\test_parametrization.py`

```
# Owner(s): ["module: nn"]
# 导入 pickle 库，用于对象的序列化和反序列化操作
import pickle
# 从 copy 模块中导入 deepcopy 函数，用于深度复制对象
from copy import deepcopy
# 从 itertools 模块中导入 product 函数，用于计算笛卡尔积
from itertools import product

# 导入 PyTorch 库
import torch

# 导入 torch.nn 模块，提供神经网络相关的函数和类
import torch.nn as nn
# 导入 torch.nn.functional 模块，提供神经网络的函数接口
import torch.nn.functional as F
# 导入 torch.nn.init 模块，提供初始化神经网络权重的函数
import torch.nn.init as init
# 导入 torch.nn.utils.parametrize 模块，提供参数化测试相关的函数
import torch.nn.utils.parametrize as parametrize
# 从 torch 模块中导入 Tensor 类型
from torch import Tensor
# 从 torch.__future__ 模块中导入 get_swap_module_params_on_conversion 函数
from torch.__future__ import get_swap_module_params_on_conversion
# 从 torch.nn 模块中导入 Parameter 类
from torch.nn import Parameter
# 导入 torch.testing._internal.common_cuda 模块中的 TEST_MULTIGPU 符号
from torch.testing._internal.common_cuda import TEST_MULTIGPU
# 导入 torch.testing._internal.common_device_type 模块中的 instantiate_device_type_tests 函数
from torch.testing._internal.common_device_type import instantiate_device_type_tests
# 导入 torch.testing._internal.common_nn 模块中的 NNTestCase 类
from torch.testing._internal.common_nn import NNTestCase
# 导入 torch.testing._internal.common_utils 模块中的一系列函数和类
from torch.testing._internal.common_utils import (
    gradcheck,
    instantiate_parametrized_tests,
    run_tests,
    set_default_dtype,
    skipIfNoLapack,
    skipIfTorchDynamo,
    swap,
    TemporaryFileName,
)
# 导入 torch.testing._internal.two_tensor 模块中的 TwoTensor 类
from torch.testing._internal.two_tensor import TwoTensor


# 定义 TestNNParametrization 类，继承自 NNTestCase 类
class TestNNParametrization(NNTestCase):
    # 设置类属性，用于检查 CUDA 内存泄漏
    _do_cuda_memory_leak_check = True
    # 设置类属性，指示是否使用非默认 CUDA 流
    _do_cuda_non_default_stream = True

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    # torch/nn/utils/parametrize
    # 标记为跳过测试，条件是没有 LAPACK 库支持
    @skipIfNoLapack
    # 交换装饰器参数的顺序，参数为 True 和 False
    @swap([True, False])
    # 交换装饰器参数的顺序，参数为 True 和 False
    @swap([True, False])
    def test_register_and_remove_nested_parametrization(self):
        r"""Test that it is possible to nest the parametrizations
        meaning that the original param is parametrized again
        """
        
        # 定义一个名为 `test_register_and_remove_nested_parametrization` 的测试方法，用于测试嵌套参数化的功能。
        # 这里的 docstring 解释了测试的目的和功能。

        class Skew(nn.Module):
            def forward(self, X):
                X = X.tril(-1)
                return X - X.T
        
        # 定义一个名为 `Skew` 的 nn.Module 子类，实现了一个对称矩阵的下三角部分截断和转置的操作。
        
        model = nn.Linear(8, 8)
        # 创建一个输入输出都是8维的线性模型。

        # Add top level parametrization
        parametrize.register_parametrization(model, "weight", Skew())
        # 将 `model` 的 `weight` 参数注册为一个 `Skew` 类的实例，即对 `weight` 参数进行参数化操作。
        
        self.assertTrue(hasattr(model, "parametrizations"))
        # 断言 `model` 对象具有 `parametrizations` 属性。
        
        self.assertTrue(parametrize.is_parametrized(model))
        # 断言 `model` 对象已经被参数化。
        
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        # 断言 `model` 对象的 `weight` 参数已经被参数化。
        
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        # 断言 `model` 对象的 `bias` 参数没有被参数化。
        
        self.assertNotIn("weight", model._parameters)
        # 断言 `model` 对象的 `_parameters` 中不包含 `weight` 参数，因为它被参数化后会移除原参数。
        
        # Result should be skew-symmetric
        A = model.weight
        # 获取经过参数化后的 `weight` 参数。
        
        self.assertEqual(A, -A.T)
        # 断言参数化后的 `weight` 参数应当是一个对称矩阵的下三角部分截断和转置的结果。

        if get_swap_module_params_on_conversion():
            # 当使用 swap_tensors 路径时，需要确保自动求导图不再存在。
            del A
        
        # Add nested parametrization
        param_mod = model.parametrizations.weight
        # 获取 `model` 对象的 `weight` 参数的参数化模块。
        
        self.assertFalse(hasattr(param_mod, "parametrizations"))
        # 断言 `param_mod` 对象的 `parametrizations` 属性不存在。
        
        self.assertFalse(parametrize.is_parametrized(param_mod))
        # 断言 `param_mod` 对象没有被参数化。
        
        self.assertFalse(parametrize.is_parametrized(param_mod, "original"))
        # 断言 `param_mod` 对象的 `original` 参数没有被参数化。
        
        parametrize.register_parametrization(param_mod, "original", Skew())
        # 将 `param_mod` 对象的 `original` 参数注册为一个 `Skew` 类的实例，进行参数化操作。
        
        self.assertTrue(hasattr(param_mod, "parametrizations"))
        # 断言 `param_mod` 对象的 `parametrizations` 属性存在。
        
        self.assertTrue(parametrize.is_parametrized(param_mod))
        # 断言 `param_mod` 对象已经被参数化。
        
        self.assertTrue(parametrize.is_parametrized(param_mod, "original"))
        # 断言 `param_mod` 对象的 `original` 参数已经被参数化。
        
        self.assertNotIn("original", param_mod._parameters)
        # 断言 `param_mod` 对象的 `_parameters` 中不包含 `original` 参数，因为它被参数化后会移除原参数。
        
        # Result should be skew-symmetric
        A = param_mod.original
        # 获取经过参数化后的 `original` 参数。
        
        self.assertEqual(A, -A.T)
        # 断言参数化后的 `original` 参数应当是一个对称矩阵的下三角部分截断和转置的结果。

        # Remove nested param and check consistency
        parametrize.remove_parametrizations(
            param_mod, "original", leave_parametrized=False
        )
        # 移除 `param_mod` 对象的 `original` 参数的参数化，并检查一致性。
        
        self.assertFalse(hasattr(param_mod, "parametrizations"))
        # 断言 `param_mod` 对象的 `parametrizations` 属性不存在。
        
        self.assertEqual(param_mod.__class__, parametrize.ParametrizationList)
        # 断言 `param_mod` 对象的类为 `parametrize.ParametrizationList`，表示参数化被正确移除。

        # Remove top level and check consistency
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        # 移除 `model` 对象的 `weight` 参数的参数化，并检查一致性。
        
        self.assertFalse(hasattr(model, "parametrizations"))
        # 断言 `model` 对象的 `parametrizations` 属性不存在。
        
        self.assertEqual(model.__class__, nn.Linear)
        # 断言 `model` 对象的类为 `nn.Linear`，表示参数化被正确移除。

    @swap([True, False])
    def test_register_and_remove_buffer_parametrization(self):
        r"""Test that it is possible to add and remove parametrizations on buffers"""

        # Define a couple vector parametrizations
        # 定义两个向量的参数化类
        class FirstZero(nn.Module):
            def forward(self, x):
                return torch.cat([x.new_zeros(1), x[1:]])

        class LastZero(nn.Module):
            def forward(self, x):
                return torch.cat([x[:-1], x.new_zeros(1)])

        model = nn.Linear(8, 8)

        # Instantiate parametrizations on buffers. It should work as expected
        # 删除模型的 "bias" 属性
        delattr(model, "bias")
        # 注册一个名为 "bias" 的缓冲区，并初始化为全 1 的张量
        model.register_buffer("bias", torch.ones(8))
        # 在 "bias" 缓冲区上注册 FirstZero 参数化方法
        parametrize.register_parametrization(model, "bias", FirstZero())
        # 在 "bias" 缓冲区上注册 LastZero 参数化方法
        parametrize.register_parametrization(model, "bias", LastZero())
        # 断言模型是否被参数化
        self.assertTrue(parametrize.is_parametrized(model))
        # 断言 "bias" 缓冲区是否被参数化
        self.assertTrue(parametrize.is_parametrized(model, "bias"))
        # 断言 "bias" 缓冲区的第一个元素是否为 0.0
        self.assertEqual(model.bias[0].item(), 0.0)
        # 断言 "bias" 缓冲区的最后一个元素是否为 0.0
        self.assertEqual(model.bias[-1].item(), 0.0)
        # 断言 "bias" 缓冲区除了第一个和最后一个元素外，其余元素是否为 1.0
        self.assertTrue((model.bias[1:-1] == torch.ones(6)).all())
        # 断言模型参数数量是否为 1
        self.assertEqual(len(list(model.parameters())), 1)

        # Remove parametrizations on buffers. It should work as expected
        # 移除 "bias" 缓冲区上的所有参数化方法，但保留参数化后的值
        parametrize.remove_parametrizations(model, "bias", leave_parametrized=True)
        # 断言模型是否未被参数化
        self.assertFalse(parametrize.is_parametrized(model))
        # 断言 "bias" 缓冲区是否未被参数化
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        # 断言 "bias" 缓冲区的第一个元素是否为 0.0
        self.assertEqual(model.bias[0].item(), 0.0)
        # 断言 "bias" 缓冲区的最后一个元素是否为 0.0
        self.assertEqual(model.bias[-1].item(), 0.0)
        # 断言 "bias" 缓冲区除了第一个和最后一个元素外，其余元素是否为 1.0
        self.assertTrue((model.bias[1:-1] == torch.ones(6)).all())
        # 断言模型参数数量是否为 1
        self.assertEqual(len(list(model.parameters())), 1)
    def test_serialization_parametrization(self):
        r"""Test that it is possible to serialize a parametrized model via state_dict"""

        # A stateful parametrization
        class Orthogonal(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.register_buffer("id", torch.eye(n))  # 注册单位矩阵作为缓冲区
                self.register_buffer("B", torch.empty(n, n))  # 注册空的 n x n 张量作为缓冲区
                init.orthogonal_(self.B)  # 对 B 执行正交初始化

            def forward(self, X):
                A = X.triu(1)  # 获取 X 的上三角部分
                A = A - A.T  # 计算上三角矩阵和其转置之差
                return self.B @ torch.linalg.solve(self.id + A, self.id - A)  # 返回计算结果

        def get_model():
            model = torch.nn.Sequential(
                torch.nn.Linear(5, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 1),
            )

            parametrize.register_parametrization(model[0], "weight", Orthogonal(5))  # 注册参数化方法到模型的第一个线性层的权重上
            return model

        model = get_model()

        prev_weight = model[0].weight  # 保存模型初始权重
        prev_B = model[0].parametrizations.weight[0].B  # 保存参数化对象的 B 属性

        new_model = get_model()
        with TemporaryFileName() as fname:
            torch.save(model.state_dict(), fname)  # 将模型的状态字典保存到临时文件中
            new_model.load_state_dict(torch.load(fname))  # 加载保存的状态字典到新模型中

        # Integrity tests
        self.assertTrue(parametrize.is_parametrized(new_model[0], "weight"))  # 检查新模型的第一个层的权重是否参数化
        self.assertEqual(prev_weight, new_model[0].weight)  # 检查加载后的模型权重与原始模型权重是否相等
        self.assertEqual(prev_B, new_model[0].parametrizations.weight[0].B)  # 检查加载后的模型参数化对象的 B 属性是否与原始相等

        # Trying to save the whole parametrized model raises
        with self.assertRaisesRegex(RuntimeError, "state_dict"):
            with TemporaryFileName() as fname:
                torch.save(model, fname)  # 尝试保存整个参数化模型，预期会引发 RuntimeError 异常

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    @swap([True, False])
    def test_initialization_parametrization(self):
        r"""Test that it is possible to initialize a parametrization when it
        implements a `right_inverse` method
        """

        # 定义一个名为 Skew 的自定义神经网络模块类
        class Skew(nn.Module):
            # 定义模块的前向传播方法
            def forward(self, X):
                # 计算输入矩阵 X 的上三角部分
                A = X.triu(1)
                # 返回计算结果与其转置的差值，得到一个反对称矩阵
                return A - A.T

            # 检查矩阵是否为反对称矩阵
            def is_skew(self, A):
                return torch.allclose(A, -A.T, atol=1e-6)

            # 右逆操作，如果不是反对称矩阵则抛出值错误异常
            def right_inverse(self, X):
                if not self.is_skew(X):
                    raise ValueError("The matrix is not skew-symmetric.")
                # 返回输入矩阵的上三角部分
                return X.triu(1)

        # 定义一个名为 Orthogonal 的自定义神经网络模块类
        class Orthogonal(nn.Module):
            # 初始化方法，注册一个缓冲区 B 为单位矩阵
            def __init__(self, n):
                super().__init__()
                self.register_buffer("B", torch.eye(n))

            # 前向传播方法
            def forward(self, X):
                Id = torch.eye(X.size(0))
                # 返回一个正交矩阵的变换结果
                return self.B @ torch.linalg.solve(Id + X, Id - X)

            # 检查矩阵是否为正交矩阵
            def is_orthogonal(self, X):
                Id = torch.eye(X.size(0))
                return torch.allclose(X.T @ X, Id, atol=1e-4)

            # 右逆操作，如果不是正交矩阵则抛出值错误异常
            def right_inverse(self, X):
                if not self.is_orthogonal(X):
                    raise ValueError("The input is not orthogonal.")
                # 将类的缓冲区 B 更新为输入矩阵 X
                self.B = X
                # 返回与输入矩阵 X 形状相同的零矩阵
                return torch.zeros_like(X)

        N = 5
        # 创建一个线性神经网络模型，输入输出维度为 N
        model = nn.Linear(N, N)
        
        # 注册 Skew 类的参数化约束到模型的权重上，使其成为反对称矩阵
        skew = Skew()
        with torch.no_grad():
            # 设置模型的权重为经过 Skew 约束处理后的结果
            model.weight.set_(skew(model.weight))
        parametrize.register_parametrization(model, "weight", skew)
        
        # 创建一个随机矩阵 X
        X = torch.rand(N, N)
        # X 不是反对称矩阵，因此应该引发值错误异常
        with self.assertRaises(ValueError):
            model.weight = X
        
        # 将 X 转换为反对称矩阵
        X = X - X.T
        model.weight = X
        
        # 断言模型的 parametrizations 属性中的 weight 原始值与 X 的上三角部分相等
        self.assertEqual(model.parametrizations.weight.original, X.triu(1))
        # 断言模型的权重与 X 相等
        self.assertEqual(model.weight, X)

        # 向模型的权重注册 Orthogonal 类的参数化约束
        parametrize.register_parametrization(model, "weight", Orthogonal(N))
        
        # 创建一个随机矩阵 X
        X = torch.rand(N, N)
        # X 不是正交矩阵，因此应该引发值错误异常
        with self.assertRaises(ValueError):
            model.weight = X
        
        # 使用 PyTorch 的初始化函数将 X 初始化为正交矩阵
        init.orthogonal_(X)
        model.weight = X
        
        # 断言模型的权重与 X 相等
        self.assertEqual(model.weight, X)
        # 断言模型的 parametrizations 属性中的 weight 原始值为与 X 形状相同的零矩阵
        self.assertEqual(model.parametrizations.weight.original, torch.zeros_like(X))
    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    # 警告：重写此测试，使用不依赖于 LAPACK 的函数，并移除 `@skipIfNoLapack`（参见 #70995）
    @skipIfNoLapack
    @swap([True, False])
    def test_caching_parametrization(self):
        r"""Test the caching system of a parametrization"""
        
        # Define a couple matrix parametrizations
        # 定义两个矩阵参数化类
        class Skew(nn.Module):
            def forward(self, X):
                X = X.tril(-1)
                return X - X.T

        class Orthogonal(nn.Module):
            def forward(self, X):
                Id = torch.eye(X.size(0), device=X.device)
                return torch.linalg.solve(Id + X, Id - X)

        # 创建一个线性模型
        model = nn.Linear(5, 5)
        # 向模型的权重注册扭曲参数化
        parametrize.register_parametrization(model, "weight", Skew())
        # 向模型的权重注册正交参数化
        parametrize.register_parametrization(model, "weight", Orthogonal())

        # 测试缓存系统是否正常工作
        with parametrize.cached():
            # 获取模型的权重
            X = model.weight
            # 再次获取模型的权重
            Y = model.weight
            # 断言两次获取的对象的内存地址相同
            self.assertEqual(id(X), id(Y))

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    # 警告：重写此测试，使用不依赖于 LAPACK 的函数，并移除 `@skipIfNoLapack`（参见 #70995）
    @skipIfNoLapack
    @swap([True, False])
    def test_caching_parametrization_with_transfer_parametrizations_and_params(self):
        r"""Test that transferring parametrizations doesn't cause issues with caching"""
        
        # 定义一个扭曲参数化类
        class Skew(nn.Module):
            def forward(self, X):
                X = X.tril(-1)
                return X - X.T

        # 定义一个正交参数化类
        class Orthogonal(nn.Module):
            def forward(self, X):
                Id = torch.eye(X.size(0), device=X.device)
                return torch.linalg.solve(Id + X, Id - X)

        # 创建一个线性模型
        model = nn.Linear(5, 5)
        # 向模型的权重注册扭曲参数化
        parametrize.register_parametrization(model, "weight", Skew())
        # 向模型的权重注册正交参数化
        parametrize.register_parametrization(model, "weight", Orthogonal())

        # 创建另一个线性模型，用于参数传递测试
        to_model = nn.Linear(5, 5)
        # 将原模型的参数化传递到新模型
        parametrize.transfer_parametrizations_and_params(model, to_model)

        # 测试缓存系统是否正常工作
        with parametrize.cached():
            # 获取原模型的权重
            X = model.weight
            # 再次获取原模型的权重
            Y = model.weight
            # 断言原模型两次获取的对象的内存地址相同
            self.assertEqual(id(X), id(Y))

            # 获取新模型的权重
            A = to_model.weight
            # 再次获取新模型的权重
            B = to_model.weight
            # 断言新模型两次获取的对象的内存地址相同
            self.assertEqual(id(A), id(B))

            # 断言原模型和新模型的权重对象内存地址不同
            # 以验证它们是不同的对象
            self.assertNotEqual(id(A), id(X))

    @swap([True, False])
    def test_parametrization_same_training_mode(self):
        r"""Test training mode updated on parametrization registration"""

        class Identity(nn.Module):
            def forward(self, X):
                return X

        # 创建一个线性模块，输入和输出维度都是4
        module = nn.Linear(4, 4)
        # 将模块设为评估模式（非训练模式）
        module.eval()
        # 使用 Identity 类注册 "weight" 参数的参数化
        parametrize.register_parametrization(module, "weight", Identity())
        # 断言 "weight" 参数的第一个参数化对象不在训练模式
        self.assertFalse(module.parametrizations.weight[0].training)
        # 将模块设为训练模式
        module.train()
        # 使用 Identity().eval() 类重新注册 "weight" 参数的参数化
        parametrize.register_parametrization(module, "weight", Identity().eval())
        # 断言 "weight" 参数的第一个和第二个参数化对象都在训练模式
        self.assertTrue(module.parametrizations.weight[0].training)
        self.assertTrue(module.parametrizations.weight[1].training)

    @swap([True, False])
    def test_type_before_parametrizations(self):
        r"""Test that type_before_parametrizations always retrieves original type"""

        class Identity(nn.Module):
            def forward(self, X):
                return X

        # 创建一个线性模块，输入和输出维度都是5
        model = nn.Linear(5, 5)
        # 记录原始模块的类型
        original_type = type(model)
        # 断言在注册参数化之前，type_before_parametrizations 函数能够正确返回原始类型
        self.assertTrue(
            parametrize.type_before_parametrizations(model) == original_type
        )
        # 使用 Identity 类注册 "weight" 参数的参数化
        parametrize.register_parametrization(model, "weight", Identity())
        # 再次断言在注册参数化之后，type_before_parametrizations 函数能够正确返回原始类型
        self.assertTrue(
            parametrize.type_before_parametrizations(model) == original_type
        )

    @swap([True, False])
    def test_deepcopy_after_parametrization(self):
        r"""Test that we are able to create a deepcopy of the module when it's parametrized."""

        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1.0

        class ModelWithoutDeepcopy(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.tensor([1.0, 1.0, 1.0, 1.0]), requires_grad=True
                )
                self.bias = nn.Parameter(
                    torch.tensor([0.0, 0.0, 0.0, 0.0]), requires_grad=True
                )
                self.attr = [1.0, 2.0, 3.0, 4.0]

        class ActualModel(ModelWithoutDeepcopy):
            # Emulate custom implementation of the deepcopying.
            def __deepcopy__(self, memo):
                result = self.__new__(self.__class__)
                memo[id(self)] = result
                result.__dict__ = deepcopy(self.__dict__, memo)
                return result

        def check_deepcopy(m1: nn.Module, m2: nn.Module):
            w1 = m1.parametrizations.weight.original
            w2 = m2.parametrizations.weight.original
            b1 = (
                m1.parametrizations.bias.original
                if parametrize.is_parametrized(m1, "bias")
                else m1.bias
            )
            b2 = (
                m2.parametrizations.bias.original
                if parametrize.is_parametrized(m2, "bias")
                else m2.bias
            )
            # Weights, biases and attributes should be equal but they must be different objects.
            self.assertEqual(m1.__dict__.keys(), m2.__dict__.keys())  # 检查两个模型的字典键是否相同
            self.assertIsNot(m1, m2)  # 检查两个模型对象是否不同
            self.assertEqual(w1, w2)  # 检查两个权重参数是否相同
            self.assertIsNot(w1, w2)  # 检查两个权重参数对象是否不同
            self.assertEqual(b1, b2)  # 检查两个偏置参数是否相同
            self.assertIsNot(b1, b2)  # 检查两个偏置参数对象是否不同
            self.assertEqual(m1.attr, m2.attr)  # 检查两个模型属性是否相同
            self.assertIsNot(m1.attr, m2.attr)  # 检查两个模型属性对象是否不同

        for model in (ModelWithoutDeepcopy(), ActualModel()):
            # General check that we are able to create deepcopy.
            parametrize.register_parametrization(model, "weight", AddOne())  # 注册权重参数的参数化
            check_deepcopy(model, deepcopy(model))  # 检查模型深拷贝是否成功
            # Check that this works on models with several parametrized tensors.
            parametrize.register_parametrization(model, "bias", AddOne())  # 注册偏置参数的参数化
            check_deepcopy(model, deepcopy(model))  # 再次检查模型深拷贝是否成功
            # Check that this works on models where tensors have more than one parametrization.
            parametrize.register_parametrization(model, "weight", AddOne())  # 再次注册权重参数的参数化
            check_deepcopy(model, deepcopy(model))  # 再次检查模型深拷贝是否成功
    def test_transfer_parametrizations_and_params(self):
        r"""Test that all parametrizations and their associated parameters are transferred."""

        # 定义一个简单的添加操作的神经网络模块
        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1.0

        # 定义一个简单的乘以2操作的神经网络模块
        class Double(nn.Module):
            def forward(self, x):
                return 2.0 * x

            # 定义一个乘以0.5操作的右逆操作
            def right_inverse(self, x):
                return 0.5 * x

        # 定义一个简单的减去1操作的神经网络模块
        class MinusOne(nn.Module):
            def forward(self, x):
                return x - 1.0

        # 创建一个线性层模型，输入输出维度为5
        model = nn.Linear(5, 5)
        # 向模型的权重注册AddOne()的参数化操作
        parametrize.register_parametrization(model, "weight", AddOne())
        # 向模型的权重注册Double()的参数化操作
        parametrize.register_parametrization(model, "weight", Double())
        # 向模型的权重注册MinusOne()的参数化操作
        parametrize.register_parametrization(model, "weight", MinusOne())
        # 保存模型的权重
        hold_weight = model.weight

        # 创建一个带量化配置的量化训练后（QAT）的线性层模型，输入输出维度为5
        to_model = torch.ao.nn.qat.Linear(
            5, 5, qconfig=torch.ao.quantization.get_default_qconfig()
        )
        # 将模型参数化和参数从model转移到to_model
        parametrize.transfer_parametrizations_and_params(model, to_model)

        # 检查to_model的权重是否已被参数化
        self.assertTrue(torch.nn.utils.parametrize.is_parametrized(to_model, "weight"))
        # 检查模型的权重是否等于to_model的权重
        self.assertEqual(model.weight, to_model.weight)
        # 检查参数化操作是否正确传输到to_model的原始权重
        self.assertEqual(
            model.parametrizations.weight.original,
            to_model.parametrizations.weight.original,
        )

        # 检查转移是否未影响原始权重值
        self.assertEqual(hold_weight, model.weight)
        # 如果使用交换张量路径，则需要删除hold_weight以释放自动求导图
        if get_swap_module_params_on_conversion():
            del hold_weight

        # 测试一个参数化操作的更改不会影响另一个参数化操作的情况
        parametrize.remove_parametrizations(to_model, "weight")
        self.assertFalse(torch.nn.utils.parametrize.is_parametrized(to_model, "weight"))
        self.assertTrue(torch.nn.utils.parametrize.is_parametrized(model, "weight"))

        # 同时测试那些在to_model中不存在的参数是否也被正确转移
        model.test_param = Parameter(torch.randn(5, 5))

        # 检查to_model中是否没有test_param属性
        self.assertTrue(not hasattr(to_model, "test_param"))
        # 向model的test_param注册Double()的参数化操作
        parametrize.register_parametrization(model, "test_param", Double())
        # 保存model的test_param
        hold_test_param = model.test_param
        # 将test_param的参数化操作和参数从model转移到to_model
        parametrize.transfer_parametrizations_and_params(model, to_model, "test_param")

        # 检查之前缺失的参数是否被正确转移
        self.assertEqual(model.test_param, to_model.test_param)
        self.assertEqual(
            model.parametrizations.test_param.original,
            to_model.parametrizations.test_param.original,
        )

        # 检查新的转移是否未改变from_module的值
        self.assertEqual(hold_test_param, model.test_param)
    def test_transfer_parametrizations_and_params_right_inverse(self):
        r"""Test that all parametrizations and their associated parameters are transferred."""
        
        # 定义一个内部模型类 Double，用于在正向传播中对输入 x 进行乘以 2 的操作
        class Double(nn.Module):
            def forward(self, x):
                return 2.0 * x
            
            # 定义 right_inverse 方法，用于对输入 x 进行乘以 0.5 的操作，作为逆操作
            def right_inverse(self, x):
                return 0.5 * x
        
        # 创建一个包含 5 个输入和 5 个输出的线性模型
        model = nn.Linear(5, 5)
        
        # 注册 Double 类作为模型的权重参数化方式
        parametrize.register_parametrization(model, "weight", Double())
        
        # 记录当前模型的权重
        hold_weight = model.weight
        
        # 创建一个新的量化感知训练（QAT）线性模型，与输入和输出均为 5
        to_model = torch.ao.nn.qat.Linear(
            5, 5, qconfig=torch.ao.quantization.get_default_qconfig()
        )
        
        # 将模型 model 的参数化方式和参数传递到 to_model
        parametrize.transfer_parametrizations_and_params(model, to_model)
        
        # 检查转移是否成功
        self.assertEqual(model.weight, to_model.weight)
        
        # 检查权重参数化是否也成功转移
        self.assertEqual(
            model.parametrizations.weight.original,
            to_model.parametrizations.weight.original,
        )
        
        # 检查转移操作是否不影响原模型的权重
        self.assertEqual(hold_weight, model.weight)

    @swap([True, False])
    def test_transfer_parametrizations_and_params_single_param(self):
        r"""Test that all parametrizations and their associated parameters are transferred."""
        
        # 定义一个内部模型类 AddOne，用于在正向传播中对输入 x 加 1 的操作
        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1.0
        
        # 定义一个内部模型类 Double，用于在正向传播中对输入 x 进行乘以 2 的操作
        class Double(nn.Module):
            def forward(self, x):
                return 2.0 * x
        
        # 定义一个内部模型类 MinusOne，用于在正向传播中对输入 x 减 1 的操作
        class MinusOne(nn.Module):
            def forward(self, x):
                return x - 1.0
        
        # 创建一个包含 5 个输入和 5 个输出的线性模型，并包含偏置项
        model = nn.Linear(5, 5, bias=True)
        
        # 分别注册 AddOne、Double 和 MinusOne 作为模型的权重参数化方式
        parametrize.register_parametrization(model, "weight", AddOne())
        parametrize.register_parametrization(model, "weight", Double())
        parametrize.register_parametrization(model, "weight", MinusOne())
        
        # 注册 AddOne、Double 和 MinusOne 作为模型的偏置参数化方式
        parametrize.register_parametrization(model, "bias", AddOne())
        parametrize.register_parametrization(model, "bias", Double())
        parametrize.register_parametrization(model, "bias", MinusOne())
        
        # 创建一个新的量化感知训练（QAT）线性模型，与输入和输出均为 5，并包含偏置项
        to_model = torch.ao.nn.qat.Linear(
            5, 5, bias=True, qconfig=torch.ao.quantization.get_default_qconfig()
        )
        
        # 将模型 model 的权重参数化方式和参数传递到 to_model，仅传递权重相关的参数化方式
        parametrize.transfer_parametrizations_and_params(model, to_model, "weight")
        
        # 检查权重参数和仅权重参数化方式是否成功转移
        self.assertEqual(model.weight, to_model.weight)
        self.assertEqual(
            model.parametrizations.weight.original,
            to_model.parametrizations.weight.original,
        )
        
        # 检查转移操作是否不影响 to_model 的偏置参数化方式
        self.assertTrue("bias" not in to_model.parametrizations)
    # 定义一个测试方法，用于测试参数化和参数的多对一转移功能
    def test_transfer_parametrizations_and_params_many_to_one(self):
        # 定义一个具有多个输出的参数化模块
        class RankOne(nn.Module):
            # 前向传播方法，将两个向量形成一个秩为1的矩阵
            def forward(self, x, y):
                return x.unsqueeze(-1) @ y.unsqueeze(-2)

            # 计算给定矩阵在秩1矩阵上的投影
            def right_inverse(self, Y):
                # 对给定的矩阵进行奇异值分解，仅保留主要的奇异值和向量
                U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
                # 取奇异值中的最大值并开方，然后转换成列向量
                s0_sqrt = S[0].sqrt().unsqueeze(-1)
                return U[..., :, 0] * s0_sqrt, Vh[..., 0, :] * s0_sqrt

        # 定义一个简单的模块，将输入乘以2
        class Double(nn.Module):
            def forward(self, x):
                return 2.0 * x

        # 创建一个线性模型
        model = nn.Linear(3, 3)
        # 将模型的权重参数注册为RankOne类的参数化
        parametrize.register_parametrization(model, "weight", RankOne())
        # 将模型的权重参数再次注册为Double类的参数化，覆盖前一个注册
        parametrize.register_parametrization(model, "weight", Double())
        # 保存模型当前的权重参数
        hold_weight = model.weight

        # 创建一个新的量化感知训练模型
        to_model = torch.ao.nn.qat.Linear(
            3, 3, qconfig=torch.ao.quantization.get_default_qconfig()
        )

        # 将原模型的参数化及其参数转移到新模型中
        parametrize.transfer_parametrizations_and_params(model, to_model)

        # 检查转移后的模型参数是否被正确设置为参数化状态
        self.assertTrue(torch.nn.utils.parametrize.is_parametrized(to_model, "weight"))
        # 检查转移后的模型权重参数是否与原模型相同
        self.assertEqual(model.weight, to_model.weight)
        # 检查模型参数化是否正确转移
        self.assertEqual(
            model.parametrizations.weight.original0,
            to_model.parametrizations.weight.original0,
        )
        self.assertEqual(
            model.parametrizations.weight.original1,
            to_model.parametrizations.weight.original1,
        )

        # 检查转移操作未影响原模型的权重参数
        self.assertEqual(hold_weight, model.weight)

        # 向原模型添加一个测试参数，并注册为RankOne类的参数化
        model.test_param = Parameter(torch.randn(3, 3))
        parametrize.register_parametrization(model, "test_param", RankOne())
        # 保存添加后的测试参数
        hold_test_param = model.test_param
        # 将原模型的测试参数化及其参数转移到新模型中
        parametrize.transfer_parametrizations_and_params(model, to_model, "test_param")

        # 检查转移后的测试参数是否正确传递
        self.assertEqual(model.test_param, to_model.test_param)
        # 检查测试参数化是否正确转移
        self.assertEqual(
            model.parametrizations.test_param.original0,
            to_model.parametrizations.test_param.original0,
        )
        self.assertEqual(
            model.parametrizations.test_param.original1,
            to_model.parametrizations.test_param.original1,
        )

        # 检查转移操作未影响原测试参数
        self.assertEqual(hold_test_param, model.test_param)
    def test_register_parametrization_no_grad(self):
        r"""Test that it is possible to register a parametrization without gradient"""

        # 定义一个继承自 nn.Module 的子类 SplitAndCat
        class SplitAndCat(nn.Module):
            # 定义反向操作函数 right_inverse，将张量分割成两半并返回
            def right_inverse(self, x):
                return torch.split(x, x.shape[1] // 2)

            # 前向传播函数 forward，将两个输入张量连接起来返回
            def forward(self, x0, x1):
                return torch.cat([x0, x1])

        # 创建一个线性模型，输入维度为 8，输出维度为 8
        model = nn.Linear(8, 8)

        # 设置模型的权重参数不需要梯度
        model.weight.requires_grad = False

        # 使用 parametrize.register_parametrization 函数注册权重参数的参数化方法 SplitAndCat()
        parametrize.register_parametrization(model, "weight", SplitAndCat())

        # 确保被参数化和分解后的张量都不需要梯度
        self.assertFalse(model.weight.requires_grad)
        self.assertFalse(model.parametrizations.weight.original0.requires_grad)
        self.assertFalse(model.parametrizations.weight.original1.requires_grad)

    @swap([True, False])
    @swap([True, False])
    def test_new_spectral_norm_dim(self):
        # 创建一个输入张量 inp，形状为 (2, 3, 10, 12)
        inp = torch.randn(2, 3, 10, 12)
        
        # 创建一个反卷积层 m，输入通道数为 3，输出通道数为 4，卷积核大小为 (5, 6)
        m = nn.ConvTranspose2d(3, 4, (5, 6))
        
        # 将 m 应用谱范数参数化
        m = torch.nn.utils.parametrizations.spectral_norm(m)
        
        # 获取谱范数参数化后的权重 parametrizations.weight[0]
        snm = m.parametrizations.weight[0]
        
        # 对 m 执行前向传播
        x = m(inp)
        
        # 检查 u 是否与相同维度的参数相同
        self.assertEqual(
            snm._u.shape, m.parametrizations.weight.original[0, :, 0, 0].shape
        )

    @swap([True, False])
    def test_new_spectral_norm_forward(self):
        # 创建一个输入张量 input，形状为 (3, 5)
        input = torch.randn(3, 5)
        
        # 创建一个线性层 m，输入维度为 5，输出维度为 7
        m = nn.Linear(5, 7)
        
        # 将 m 应用谱范数参数化
        m = torch.nn.utils.parametrizations.spectral_norm(m)
        
        # 获取谱范数参数化后的权重 parametrizations.weight[0]
        snm = m.parametrizations.weight[0]
        
        # 简单的前向传播
        _weight = m.parametrizations.weight.original
        _bias, _v = m.bias, snm._v
        _weight_mat = _weight.view(_weight.size(0), -1)
        _u = torch.mv(_weight_mat, _v)
        _u = F.normalize(_u, dim=0, eps=1e-12)
        _v = torch.mv(_weight_mat.t(), _u)
        _v = F.normalize(_v, dim=0, eps=1e-12)
        _weight.data /= torch.dot(_u, torch.matmul(_weight_mat, _v))
        
        # 使用新的权重和偏置进行线性变换计算 out_hat
        out_hat = torch.nn.functional.linear(input, _weight, _bias)
        
        # 期望的输出 expect_out
        expect_out = m(input)
        
        # 断言计算的 out_hat 和期望的 expect_out 相等
        self.assertEqual(expect_out, out_hat)

    @swap([True, False])
    @skipIfTorchDynamo("Test does not work with TorchDynamo")
    def test_new_spectral_norm_value(self):
        # 测试谱范数（即最大奇异值）是否正确计算，使用简单对角矩阵作为示例。
        for dtype in (torch.float, torch.cfloat):
            # 创建一个线性层对象，输入输出维度为2，使用指定的数据类型
            m = nn.Linear(2, 2, dtype=dtype)
            with torch.no_grad():
                # 设置权重为对角矩阵
                x = torch.diagonal(m.weight)
                m.weight = nn.Parameter(torch.diag(x))
                # 对权重应用谱范数
                torch.nn.utils.parametrizations.spectral_norm(m)
                # 权重应该被谱范数（即最大对角元素的范数）重新缩放
                expected = torch.diag(x / x.abs().max())
                self.assertEqual(m.weight.data, expected)

    @skipIfNoLapack
    @swap([True, False])
    @skipIfNoLapack
    @swap([True, False])
    def test_orthogonal_errors(self):
        # 测试正交矩阵参数化的错误处理
        m = nn.Linear(3, 4)
        with self.assertRaisesRegex(ValueError, "has to be one of"):
            # 对权重应用正交矩阵参数化，并期望出现特定错误消息
            torch.nn.utils.parametrizations.orthogonal(m, "weight", "foo")

        with self.assertRaisesRegex(ValueError, "Expected a matrix"):
            # 对偏置应用正交矩阵参数化，并期望出现特定错误消息
            torch.nn.utils.parametrizations.orthogonal(m, "bias")

        # 对权重应用正交矩阵参数化
        torch.nn.utils.parametrizations.orthogonal(m, "weight")
        with self.assertRaisesRegex(ValueError, "matrices of shape"):
            # 设置权重为一个非预期形状的矩阵，并期望出现特定错误消息
            m.weight = torch.randn(5, 5)
        # 移除权重的参数化
        torch.nn.utils.parametrize.remove_parametrizations(m, "weight")

    @swap([True, False])
    def test_weight_norm_state_dict_compat(self):
        # 测试权重范数在状态字典兼容性方面的表现
        m = nn.Linear(4, 5)
        m = torch.nn.utils.weight_norm(m)
        old_dict = m.state_dict()

        m2 = nn.Linear(4, 5)
        m2 = torch.nn.utils.parametrizations.weight_norm(m2)
        # 加载旧的状态字典到新的模型
        m2.load_state_dict(old_dict)

        input = torch.randn(3, 4)
        # 断言两个模型在给定输入下的输出是否一致
        self.assertEqual(m(input), m2(input))

    @swap([True, False])
    def test_weight_norm_pickle(self):
        # 测试权重范数的 pickle 兼容性
        m = nn.Linear(4, 5)
        m = torch.nn.utils.parametrizations.weight_norm(m)
        with self.assertRaisesRegex(RuntimeError, "state_dict"):
            # 尝试对带有权重范数的模型进行序列化，期望抛出特定的运行时错误
            pickle.dumps(m)

    @swap([True, False])
    def test_weight_norm_deepcopy(self):
        # 测试权重范数的深拷贝功能
        m = nn.Linear(4, 5)
        m = torch.nn.utils.parametrizations.weight_norm(m)
        m2 = deepcopy(m)
        input = torch.randn(3, 4)
        # 断言两个模型在给定输入下的输出是否一致
        self.assertEqual(m(input), m2(input))

    @swap([True])
# 定义一个测试类 TestNNParametrizationDevice，继承自 NNTestCase
class TestNNParametrizationDevice(NNTestCase):

    # 使用装饰器 @swap([True, False])，为测试函数 test_weight_norm_parametrization 提供两次运行，分别设定 device 参数为 True 和 False
    @swap([True, False])
    # 定义测试函数 test_weight_norm_parametrization，接受 device 参数
    def test_weight_norm_parametrization(self, device):
        
        # 遍历数据类型列表 [torch.float, torch.bfloat16]
        for dtype in [torch.float, torch.bfloat16]:
            
            # 创建一个在指定设备上的随机张量 input，形状为 (3, 4)，数据类型为 dtype
            input = torch.randn(3, 4, dtype=dtype, device=device)
            
            # 创建一个在指定设备上的线性层 m，输入维度为 4，输出维度为 5，数据类型为 dtype
            m = nn.Linear(4, 5, dtype=dtype, device=device)
            
            # 计算线性层 m 对输入 input 的期望输出
            expected_output = m(input)

            # 添加权重归一化操作到线性层 m
            m = torch.nn.utils.parametrizations.weight_norm(m)
            
            # 断言权重归一化后的参数形状与原始权重相同
            self.assertEqual(
                m.parametrizations.weight.original1.size(), m.weight.size()
            )
            
            # 断言权重归一化后的第一个原始参数形状为 (5, 1)
            self.assertEqual(m.parametrizations.weight.original0.size(), (5, 1))
            
            # 断言使用权重归一化后，m 对输入 input 的输出与期望输出相同
            self.assertEqual(m(input), expected_output)

            # 移除权重归一化
            torch.nn.utils.parametrize.remove_parametrizations(m, "weight")
            
            # 断言 m 不再具有 parametrizations 属性
            self.assertFalse(hasattr(m, "parametrizations"))
            
            # 再次断言 m 对输入 input 的输出与期望输出相同（即移除权重归一化后，m 恢复到原始状态）
            self.assertEqual(m(input), expected_output)

            # 使用维度 dim=1 进行权重归一化
            m = torch.nn.utils.parametrizations.weight_norm(m, dim=1)
            
            # 断言权重归一化后的参数形状与原始权重相同
            self.assertEqual(
                m.parametrizations.weight.original1.size(), m.weight.size()
            )
            
            # 断言权重归一化后的第一个原始参数形状为 (1, 4)
            self.assertEqual(m.parametrizations.weight.original0.size(), (1, 4))
            
            # 断言使用权重归一化后，m 对输入 input 的输出与期望输出相同
            self.assertEqual(m(input), expected_output)

            # 使用 dim=None 进行权重归一化
            m = nn.Linear(4, 5, dtype=dtype, device=device)
            expected_output = m(input)
            m = torch.nn.utils.parametrizations.weight_norm(m, dim=None)
            
            # 断言使用权重归一化后，m 对输入 input 的输出与期望输出相同
            self.assertEqual(m(input), expected_output)

# 设定测试的设备类型为 ("cpu", "cuda")，并将实例化的设备类型测试函数注入到全局命名空间中
only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestNNParametrizationDevice, globals(), only_for=only_for)

# 实例化 parametrized_tests 函数，以便在全局命名空间中生成 parametrized_tests
instantiate_parametrized_tests(TestNNParametrization)

# 如果脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```