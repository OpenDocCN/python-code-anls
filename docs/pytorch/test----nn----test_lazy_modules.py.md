# `.\pytorch\test\nn\test_lazy_modules.py`

```py
# 导入pickle模块，用于对象序列化和反序列化
import pickle
# 导入unittest模块，用于编写和运行单元测试
import unittest

# 导入torch模块及其子模块
import torch
import torch.nn as nn
# 从torch.nn模块导入Parameter类
from torch.nn import Parameter
# 从torch.nn.parameter模块导入UninitializedBuffer和UninitializedParameter类
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter
# 导入测试CUDA相关功能的模块
from torch.testing._internal.common_cuda import TEST_CUDA
# 导入通用的测试工具函数和类
from torch.testing._internal.common_utils import (
    run_tests,
    suppress_warnings,
    TEST_PRIVATEUSE1,
    TestCase,
)


# 创建一个继承自LazyModuleMixin和torch.nn.Module的类LazyModule
class LazyModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
    pass


# 创建一个测试类TestLazyModules，继承自unittest模块的TestCase类
class TestLazyModules(TestCase):
    
    # 装饰器，用于在测试函数中抑制特定的警告
    @suppress_warnings
    # 定义一个测试函数test_lazy_module_parameter
    def test_lazy_module_parameter(self):
        # 创建一个LazyModule对象
        module = LazyModule()
        # 注册一个名为"test_param"的未初始化参数UninitializedParameter
        module.register_parameter("test_param", UninitializedParameter())
        # 断言：检查模块是否有未初始化的参数
        self.assertTrue(module.has_uninitialized_params())
        # 获取模块的状态字典
        state_dict = module.state_dict()
        # 断言：检查状态字典中的"test_param"参数是否为UninitializedParameter类型
        self.assertIsInstance(state_dict["test_param"], UninitializedParameter)
        
        # 创建一个新的LazyModule对象
        new_module = LazyModule()
        # 尝试用一个已经初始化的参数替换已有的未初始化参数时，应该会引发错误
        new_module.register_parameter("test_param", nn.Parameter(torch.ones(5, 5)))
        # 使用断言检查是否引发了RuntimeError，并且错误信息包含"shape of an uninitialized"
        with self.assertRaisesRegex(RuntimeError, "shape of an uninitialized"):
            new_module.load_state_dict(state_dict)
        
        # 创建另一个新的LazyModule对象
        new_module = LazyModule()
        # 注册一个名为"test_param"的新参数，值为已初始化的nn.Parameter对象
        new_module.register_parameter("test_param", nn.Parameter(torch.ones(5, 5)))
        # 加载new_module的状态字典到当前模块module中
        module.load_state_dict(new_module.state_dict())
        # 断言：检查module的"test_param"参数是否与加载的状态匹配
        self.assertEqual(module.test_param, torch.ones((5, 5)))
        
        # 创建一个新的LazyModule对象
        module = LazyModule()
        # 注册一个名为"test_param"的未初始化参数UninitializedParameter
        module.register_parameter("test_param", UninitializedParameter())
        # 断言：检查模块是否有未初始化的参数
        self.assertTrue(module.has_uninitialized_params())
        
        # 创建另一个新的LazyModule对象
        new_module = LazyModule()
        # 注册一个名为"test_param"的未初始化参数UninitializedParameter
        new_module.register_parameter("test_param", UninitializedParameter())
        # 加载new_module的状态字典到当前模块module中
        module.load_state_dict(new_module.state_dict())
        # 断言：检查模块是否仍然有未初始化的参数
        self.assertTrue(module.has_uninitialized_params())

    @suppress_warnings
    def test_lazy_module_buffer(self):
        # 创建 LazyModule 实例
        module = LazyModule()
        # 向模块注册一个未初始化的缓冲区 "test_buffer"
        module.register_buffer("test_buffer", UninitializedBuffer())
        # 检查模块是否有未初始化的参数
        self.assertTrue(module.has_uninitialized_params())
        # 获取模块的状态字典
        state_dict = module.state_dict()
        # 断言 "test_buffer" 在状态字典中确实是 UninitializedBuffer 类型
        self.assertIsInstance(state_dict["test_buffer"], UninitializedBuffer)

        # 创建新的 LazyModule 实例
        new_module = LazyModule()
        # 尝试用一个具有有效值的张量替换已存在的未初始化参数会引发错误
        new_module.register_buffer("test_buffer", torch.ones(5, 5))
        with self.assertRaisesRegex(RuntimeError, "shape of an uninitialized"):
            new_module.load_state_dict(state_dict)

        # 当加载的状态字典中包含有效值时，未初始化的参数将被覆盖
        new_module = LazyModule()
        new_module.register_buffer("test_buffer", torch.ones(5, 5))
        module.load_state_dict(new_module.state_dict())
        # 断言加载后的模块中的 "test_buffer" 参数与预期的张量值相等
        self.assertEqual(module.test_buffer, torch.ones((5, 5)))

        # 如果状态字典中未初始化的参数会保持不变
        module = LazyModule()
        module.register_buffer("test_buffer", UninitializedBuffer())
        self.assertTrue(module.has_uninitialized_params())

        new_module = LazyModule()
        new_module.register_buffer("test_buffer", UninitializedBuffer())
        module.load_state_dict(new_module.state_dict())
        module.load_state_dict(new_module.state_dict())
        self.assertTrue(module.has_uninitialized_params())

    @suppress_warnings
    def test_lazy_module_jit_param(self):
        module = LazyModule()
        module.register_parameter("test_param", UninitializedParameter())
        self.assertTrue(module.has_uninitialized_params())
        # 当尝试对包含未初始化参数的模块执行 JIT 编译时会抛出异常
        with self.assertRaisesRegex(RuntimeError, "run a forward pass"):
            torch.jit.script(module)

    @suppress_warnings
    def test_lazy_module_jit_buffer(self):
        module = LazyModule()
        module.register_buffer("test_buffer", UninitializedBuffer())
        self.assertTrue(module.has_uninitialized_params())
        # 当尝试对包含未初始化缓冲区的模块执行 JIT 编译时会抛出异常
        with self.assertRaisesRegex(RuntimeError, "run a forward pass"):
            torch.jit.script(module)

    @suppress_warnings
    def test_lazy_share_memory_param(self):
        module = LazyModule()
        module.register_parameter("test_param", UninitializedParameter())
        self.assertTrue(module.has_uninitialized_params())
        # 当尝试在未初始化参数上共享内存时会抛出异常
        with self.assertRaisesRegex(RuntimeError, "share memory on an uninitialized"):
            module.share_memory()

    @suppress_warnings
    def test_lazy_share_memory_buffer(self):
        module = LazyModule()
        module.register_buffer("test_buffer", UninitializedBuffer())
        self.assertTrue(module.has_uninitialized_params())
        # 当尝试在未初始化缓冲区上共享内存时会抛出异常
        with self.assertRaisesRegex(RuntimeError, "share memory on an uninitialized"):
            module.share_memory()

    @suppress_warnings
    # 测试 LazyLinear 类的功能
    def test_linear(self):
        # 创建一个 LazyLinear 模块，指定输出维度为 10
        module = nn.LazyLinear(10)
        # 断言模块的权重是未初始化的参数
        self.assertIsInstance(module.weight, UninitializedParameter)
        # 断言模块的偏置是未初始化的参数
        self.assertIsInstance(module.bias, UninitializedParameter)
        # 创建一个全为1的5x5张量作为输入
        input = torch.ones(5, 5)
        # 将输入传递给模块，触发模块的初始化
        module(input)
        # 断言模块现在是 nn.Linear 类的实例
        self.assertIsInstance(module, nn.Linear)
        # 断言模块现在不再是 nn.LazyLinear 类的实例
        self.assertNotIsInstance(module, nn.LazyLinear)
        # 断言模块的权重形状为 (10, 5)
        self.assertTrue(module.weight.shape == (10, 5))
        # 断言模块的偏置形状为 (10,)
        self.assertTrue(module.bias.shape == (10,))
        # 使用模块进行前向传播，计算输出
        y = module(input)
        # 断言模块的输出与使用 functional.linear 函数的输出相等
        self.assertTrue(
            torch.equal(
                torch.nn.functional.linear(input, module.weight, module.bias), y
            )
        )

    # 测试 LazyLinear 模块的序列化和反序列化
    @suppress_warnings
    def test_lazy_linear_pickle(self):
        # 创建一个 LazyLinear 模块，指定输出维度为 10
        module = nn.LazyLinear(10)
        # 断言模块的权重是未初始化的参数
        self.assertIsInstance(module.weight, UninitializedParameter)
        # 断言模块的偏置是未初始化的参数
        self.assertIsInstance(module.bias, UninitializedParameter)
        # 使用 pickle 序列化并反序列化模块
        module = pickle.loads(pickle.dumps(module))
        # 断言反序列化后模块仍然是 nn.LazyLinear 类的实例
        self.assertIsInstance(module, nn.LazyLinear)
        # 断言反序列化后模块的权重仍然是未初始化的参数
        self.assertIsInstance(module.weight, UninitializedParameter)
        # 断言反序列化后模块的偏置仍然是未初始化的参数
        self.assertIsInstance(module.bias, UninitializedParameter)
        # 创建一个全为1的5x5张量作为输入
        input = torch.ones(5, 5)
        # 将输入传递给模块，完全实例化模块
        module(input)  # fully materialized
        # 使用 pickle 序列化并反序列化模块
        new_module = pickle.loads(pickle.dumps(module))
        # 断言反序列化后模块现在是 nn.Linear 类的实例
        self.assertIsInstance(new_module, nn.Linear)
        # 断言反序列化后模块不再是 nn.LazyLinear 类的实例
        self.assertNotIsInstance(new_module, nn.LazyLinear)
        # 断言反序列化后模块的权重形状为 (10, 5)
        self.assertTrue(new_module.weight.shape == (10, 5))
        # 断言反序列化后模块的权重不再是未初始化的参数
        self.assertNotIsInstance(new_module.weight, UninitializedParameter)
        # 断言反序列化后模块的偏置形状为 (10,)
        self.assertTrue(new_module.bias.shape == (10,))
        # 断言反序列化后模块的偏置不再是未初始化的参数
        self.assertNotIsInstance(new_module.bias, UninitializedParameter)

    # 测试 LazyLinear 模块与 nn.Linear 模块的状态字典加载
    def test_linear_state(self):
        # 创建一个 nn.Linear 模块，输入维度为 5，输出维度为 10
        module = nn.Linear(5, 10)
        # 创建一个 LazyLinear 模块，指定输出维度为 10
        lazy_module = nn.LazyLinear(10)
        # 加载 nn.Linear 模块的状态字典到 LazyLinear 模块
        lazy_module.load_state_dict(module.state_dict())
        # 断言 LazyLinear 模块现在没有未初始化的参数
        self.assertFalse(lazy_module.has_uninitialized_params())
        # 断言 LazyLinear 模块的权重形状为 (10, 5)
        self.assertTrue(lazy_module.weight.shape == (10, 5))
        # 断言 LazyLinear 模块的偏置形状为 (10,)
        self.assertTrue(lazy_module.bias.shape == (10,))

        # 再次创建一个 nn.Linear 模块，输入维度为 5，输出维度为 10
        module = nn.Linear(5, 10)
        # 再次创建一个 LazyLinear 模块，指定输出维度为 10
        lazy_module = nn.LazyLinear(10)
        # 使用断言捕获加载状态字典时抛出的 RuntimeError 异常，错误消息包含 "shape of an uninitialized"
        with self.assertRaisesRegex(RuntimeError, "shape of an uninitialized"):
            module.load_state_dict(lazy_module.state_dict())

    # 检查 LazyConv 模块的功能
    def _check_lazy_conv(
        self,
        cls,
        lazy_cls,
        func,
        init_args,
        input_shape,
        expected_weight_shape,
        expected_bias_shape,
        *forward_args,
        **forward_kwargs,
    ):
        # 使用延迟初始化类和初始参数创建模块实例
        module = lazy_cls(*init_args)
        # 断言模块的权重参数为未初始化状态
        self.assertIsInstance(module.weight, UninitializedParameter)
        # 如果存在偏置，则断言偏置参数为未初始化状态
        if module.bias is not None:
            self.assertIsInstance(module.bias, UninitializedParameter)
        # 创建输入张量，全为1，并传递给模块进行前向计算
        input = torch.ones(*input_shape)
        module(input, *forward_args, **forward_kwargs)
        # 断言模块类型为预期的类
        self.assertIsInstance(module, cls)
        # 断言模块不是延迟初始化类的实例
        self.assertNotIsInstance(module, lazy_cls)
        # 断言模块的权重参数形状与预期形状相符
        self.assertEqual(module.weight.shape, expected_weight_shape)
        # 如果存在偏置，则断言偏置参数形状与预期形状相符
        if module.bias is not None:
            self.assertEqual(module.bias.shape, expected_bias_shape)
        # 对模块进行前向计算，将计算结果与预期结果进行比较
        y = module(input)
        self.assertTrue(torch.equal(func(input, module.weight, module.bias), y))

    def _check_lazy_conv_pickle(
        self,
        cls,
        lazy_cls,
        init_args,
        input_shape,
        expected_weight_shape,
        expected_bias_shape,
    ):
        # 使用延迟初始化类和初始参数创建模块实例
        module = lazy_cls(*init_args)
        # 断言模块的权重参数为未初始化状态
        self.assertIsInstance(module.weight, UninitializedParameter)
        # 如果存在偏置，则断言偏置参数为未初始化状态
        if module.bias is not None:
            self.assertIsInstance(module.bias, UninitializedParameter)
        # 对模块进行序列化和反序列化，以测试其可持久化性
        module = pickle.loads(pickle.dumps(module))
        # 断言反序列化后的模块仍为延迟初始化类的实例
        self.assertIsInstance(module, lazy_cls)
        # 断言反序列化后的模块的权重参数为未初始化状态
        self.assertIsInstance(module.weight, UninitializedParameter)
        # 如果存在偏置，则断言反序列化后的模块的偏置参数为未初始化状态
        if module.bias is not None:
            self.assertIsInstance(module.bias, UninitializedParameter)
        # 创建输入张量，全为1，并传递给模块进行前向计算，完全实例化模块
        input = torch.ones(*input_shape)
        module(input)
        # 对完全实例化后的模块进行序列化和反序列化，以测试状态保持
        new_module = pickle.loads(pickle.dumps(module))
        # 断言反序列化后的新模块为预期的类
        self.assertIsInstance(new_module, cls)
        # 断言新模块不是延迟初始化类的实例
        self.assertNotIsInstance(new_module, lazy_cls)
        # 断言新模块的权重参数形状与预期形状相符
        self.assertEqual(new_module.weight.shape, expected_weight_shape)
        # 断言新模块的权重参数已被初始化
        self.assertNotIsInstance(new_module.weight, UninitializedParameter)
        # 如果存在偏置，则断言新模块的偏置参数形状与预期形状相符
        if new_module.bias is not None:
            self.assertEqual(new_module.bias.shape, expected_bias_shape)
            # 断言新模块的偏置参数已被初始化
            self.assertNotIsInstance(new_module.bias, UninitializedParameter)

    def _check_lazy_conv_state(
        self, gen_module, gen_lazy_module, expected_weight_shape, expected_bias_shape
    ):
        # 生成实际模块和延迟初始化模块的实例
        module = gen_module()
        lazy_module = gen_lazy_module()
        # 从实际模块加载状态到延迟初始化模块，验证参数已被初始化
        lazy_module.load_state_dict(module.state_dict())
        # 断言延迟初始化模块不再具有未初始化的参数
        self.assertFalse(lazy_module.has_uninitialized_params())
        # 断言延迟初始化模块的权重参数形状与预期形状相符
        self.assertEqual(lazy_module.weight.shape, expected_weight_shape)
        # 如果延迟初始化模块有偏置，则断言其偏置参数形状与预期形状相符
        if lazy_module.bias is not None:
            self.assertEqual(lazy_module.bias.shape, expected_bias_shape)

        # 再次生成实际模块和延迟初始化模块的实例
        module = gen_module()
        lazy_module = gen_lazy_module()
        # 使用断言验证在加载状态时发生的运行时错误，指示未初始化参数的形状
        with self.assertRaisesRegex(RuntimeError, "shape of an uninitialized"):
            module.load_state_dict(lazy_module.state_dict())
    @suppress_warnings
    # 使用装饰器 @suppress_warnings 来忽略测试中的警告信息
    def test_lazy_pre_forward_hook(self):
        """
        This test is to test whether lazymodule can register other pre-forward hook
        functions successfully.
        """
        # 定义一个测试模块 TestModule，继承自 LazyModuleMixin 和 Module 类
        class TestModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
            # 定义初始化参数的方法，但未实际使用
            def initialize_parameters(self, input):
                return None

            # 定义前向传播的方法，简单地返回输入
            def forward(self, input):
                return input

        # 定义一个钩子函数 hook_function，用于在前向传播之前执行
        def hook_function(module, input):
            return input[0] + 1

        # 创建 TestModule 的实例 module
        module = TestModule()
        # 注册前向传播的预钩子，使用 hook_function
        module.register_forward_pre_hook(hook_function)
        # 对模块进行前向传播，传入一个 2x2 的零张量
        output = module(torch.zeros(2, 2))
        # 断言输出是否为全为1的 2x2 张量
        self.assertEqual(output, torch.ones(2, 2))

    def test_lazy_forward_hook(self):
        """
        This test is to test whether lazymodule can register other forward hook
        functions successfully.
        """
        # 定义一个测试模块 TestModule，继承自 LazyModuleMixin 和 Module 类
        class TestModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
            # 定义初始化参数的方法，但未实际使用
            def initialize_parameters(self, input):
                return None

            # 定义前向传播的方法，简单地返回输入
            def forward(self, input):
                return input

        # 定义一个钩子函数 hook_function，用于在前向传播期间执行
        def hook_function(module, input, output):
            return input[0] + 1

        # 创建 TestModule 的实例 module
        module = TestModule()
        # 注册前向传播的钩子，使用 hook_function
        module.register_forward_hook(hook_function)
        # 对模块进行前向传播，传入一个 2x2 的零张量
        output = module(torch.zeros(2, 2))
        # 断言输出是否为全为1的 2x2 张量
        self.assertEqual(output, torch.ones(2, 2))

    @suppress_warnings
    # 使用装饰器 @suppress_warnings 来忽略测试中的警告信息
    def test_lazy_conv1d(self):
        # 调用 _check_lazy_conv 方法测试 nn.Conv1d 和 nn.LazyConv1d 类
        self._check_lazy_conv(
            nn.Conv1d,
            nn.LazyConv1d,
            torch.nn.functional.conv1d,
            (32, 2),  # 输入大小
            (192, 16, 50),  # 卷积核大小
            (32, 16, 2),  # 期望输出大小
            (32,),  # 批次大小
        )

    @suppress_warnings
    # 使用装饰器 @suppress_warnings 来忽略测试中的警告信息
    def test_lazy_conv1d_pickle(self):
        # 调用 _check_lazy_conv_pickle 方法测试 nn.Conv1d 和 nn.LazyConv1d 类的序列化和反序列化
        self._check_lazy_conv_pickle(
            nn.Conv1d, nn.LazyConv1d, (32, 2), (192, 16, 50), (32, 16, 2), (32,)
        )

    @suppress_warnings
    # 使用装饰器 @suppress_warnings 来忽略测试中的警告信息
    def test_lazy_conv1d_state(self):
        # 调用 _check_lazy_conv_state 方法测试 nn.Conv1d 和 nn.LazyConv1d 类的状态
        self._check_lazy_conv_state(
            lambda: nn.Conv1d(16, 32, 2),
            lambda: nn.LazyConv1d(32, 2),
            (32, 16, 2),  # 输入大小
            (32,),  # 批次大小
        )

    @suppress_warnings
    # 使用装饰器 @suppress_warnings 来忽略测试中的警告信息
    def test_lazy_conv2d(self):
        # 调用 _check_lazy_conv 方法测试 nn.Conv2d 和 nn.LazyConv2d 类
        self._check_lazy_conv(
            nn.Conv2d,
            nn.LazyConv2d,
            torch.nn.functional.conv2d,
            (32, 2),  # 输入大小
            (192, 16, 8, 6),  # 卷积核大小
            (32, 16, 2, 2),  # 期望输出大小
            (32,),  # 批次大小
        )

    @suppress_warnings
    # 使用装饰器 @suppress_warnings 来忽略测试中的警告信息
    def test_lazy_conv2d_pickle(self):
        # 调用 _check_lazy_conv_pickle 方法测试 nn.Conv2d 和 nn.LazyConv2d 类的序列化和反序列化
        self._check_lazy_conv_pickle(
            nn.Conv2d, nn.LazyConv2d, (32, 2), (192, 16, 8, 6), (32, 16, 2, 2), (32,)
        )

    @suppress_warnings
    # 使用装饰器 @suppress_warnings 来忽略测试中的警告信息
    def test_lazy_conv2d_state(self):
        # 调用 _check_lazy_conv_state 方法测试 nn.Conv2d 和 nn.LazyConv2d 类的状态
        self._check_lazy_conv_state(
            lambda: nn.Conv2d(16, 32, 2),
            lambda: nn.LazyConv2d(32, 2),
            (32, 16, 2, 2),  # 输入大小
            (32,),  # 批次大小
        )
    def test_lazy_conv3d(self):
        # 调用 _check_lazy_conv 方法，测试 Conv3d 和 LazyConv3d 的懒惰版本
        self._check_lazy_conv(
            nn.Conv3d,  # 原始 Conv3d 类
            nn.LazyConv3d,  # 懒惰版本的 Conv3d 类
            torch.nn.functional.conv3d,  # torch 中的 conv3d 函数
            (32, 2),  # 输入张量的形状
            (192, 16, 8, 7, 6),  # 卷积核的形状
            (32, 16, 2, 2, 2),  # 步幅 (stride)
            (32,),  # 批处理大小
        )

    @suppress_warnings
    def test_lazy_conv3d_pickle(self):
        # 调用 _check_lazy_conv_pickle 方法，测试 Conv3d 和 LazyConv3d 的懒惰版本（包含 pickle 操作）
        self._check_lazy_conv_pickle(
            nn.Conv3d,  # 原始 Conv3d 类
            nn.LazyConv3d,  # 懒惰版本的 Conv3d 类
            (32, 2),  # 输入张量的形状
            (192, 16, 8, 7, 6),  # 卷积核的形状
            (32, 16, 2, 2, 2),  # 步幅 (stride)
            (32,),  # 批处理大小
        )

    @suppress_warnings
    def test_lazy_conv3d_state(self):
        # 调用 _check_lazy_conv_state 方法，测试 Conv3d 和 LazyConv3d 的懒惰版本的状态
        self._check_lazy_conv_state(
            lambda: nn.Conv3d(16, 32, 2),  # 创建 Conv3d 对象的 lambda 函数
            lambda: nn.LazyConv3d(32, 2),  # 创建 LazyConv3d 对象的 lambda 函数
            (32, 16, 2, 2, 2),  # 卷积核的形状
            (32,),  # 批处理大小
        )

    @suppress_warnings
    def test_lazy_conv_transposed1d(self):
        # 调用 _check_lazy_conv 方法，测试 ConvTranspose1d 和 LazyConvTranspose1d 的懒惰版本
        self._check_lazy_conv(
            nn.ConvTranspose1d,  # 原始 ConvTranspose1d 类
            nn.LazyConvTranspose1d,  # 懒惰版本的 ConvTranspose1d 类
            torch.nn.functional.conv_transpose1d,  # torch 中的 conv_transpose1d 函数
            (32, 2),  # 输入张量的形状
            (192, 16, 50),  # 卷积核的形状
            (16, 32, 2),  # 步幅 (stride)
            (32,),  # 批处理大小
        )

    @suppress_warnings
    def test_lazy_conv_transpose1d_kwargs(self):
        # 调用 _check_lazy_conv 方法，测试 ConvTranspose1d 和 LazyConvTranspose1d 的懒惰版本（包含 kwargs 参数）
        self._check_lazy_conv(
            nn.ConvTranspose1d,  # 原始 ConvTranspose1d 类
            nn.LazyConvTranspose1d,  # 懒惰版本的 ConvTranspose1d 类
            torch.nn.functional.conv_transpose1d,  # torch 中的 conv_transpose1d 函数
            (32, 2),  # 输入张量的形状
            (192, 16, 50),  # 卷积核的形状
            (16, 32, 2),  # 步幅 (stride)
            (32,),  # 批处理大小
            output_size=(51,),  # 输出大小参数
        )

    @suppress_warnings
    def test_lazy_conv_transpose1d_pickle(self):
        # 调用 _check_lazy_conv_pickle 方法，测试 ConvTranspose1d 和 LazyConvTranspose1d 的懒惰版本（包含 pickle 操作）
        self._check_lazy_conv_pickle(
            nn.ConvTranspose1d,  # 原始 ConvTranspose1d 类
            nn.LazyConvTranspose1d,  # 懒惰版本的 ConvTranspose1d 类
            (32, 2),  # 输入张量的形状
            (192, 16, 50),  # 卷积核的形状
            (16, 32, 2),  # 步幅 (stride)
            (32,),  # 批处理大小
        )

    @suppress_warnings
    def test_lazy_conv_transpose1d_state(self):
        # 调用 _check_lazy_conv_state 方法，测试 ConvTranspose1d 和 LazyConvTranspose1d 的懒惰版本的状态
        self._check_lazy_conv_state(
            lambda: nn.ConvTranspose1d(16, 32, 2),  # 创建 ConvTranspose1d 对象的 lambda 函数
            lambda: nn.LazyConvTranspose1d(32, 2),  # 创建 LazyConvTranspose1d 对象的 lambda 函数
            (16, 32, 2),  # 卷积核的形状
            (32,),  # 批处理大小
        )

    @suppress_warnings
    def test_lazy_conv_transpose2d(self):
        # 调用 _check_lazy_conv 方法，测试 ConvTranspose2d 和 LazyConvTranspose2d 的懒惰版本
        self._check_lazy_conv(
            nn.ConvTranspose2d,  # 原始 ConvTranspose2d 类
            nn.LazyConvTranspose2d,  # 懒惰版本的 ConvTranspose2d 类
            torch.nn.functional.conv_transpose2d,  # torch 中的 conv_transpose2d 函数
            (32, 2),  # 输入张量的形状
            (192, 16, 8, 6),  # 卷积核的形状
            (16, 32, 2, 2),  # 步幅 (stride)
            (32,),  # 批处理大小
        )

    @suppress_warnings
    def test_lazy_conv_transpose2d_kwargs(self):
        # 调用 _check_lazy_conv 方法，测试 ConvTranspose2d 和 LazyConvTranspose2d 的懒惰版本（包含 kwargs 参数）
        self._check_lazy_conv(
            nn.ConvTranspose2d,  # 原始 ConvTranspose2d 类
            nn.LazyConvTranspose2d,  # 懒惰版本的 ConvTranspose2d 类
            torch.nn.functional.conv_transpose2d,  # torch 中的 conv_transpose2d 函数
            (32, 2),  # 输入张量的形状
            (192, 16, 8, 6),  # 卷积核的形状
            (16, 32, 2, 2),  # 步幅 (stride)
            (32,),  # 批处理大小
            output_size=(9, 7),  # 输出大小参数
        )

    @suppress_warnings
    def test_lazy_conv_transpose2d_pickle(self):
        # 调用 _check_lazy_conv_pickle 方法，测试 ConvTranspose2d 和 LazyConvTranspose2d 的懒惰版本（包含 pickle 操作）
        self._check_lazy_conv_pickle(
            nn.ConvTranspose2d,  # 原始 ConvTranspose2d 类
            nn.LazyConvTranspose2d,  # 懒惰版本的 ConvTranspose2d 类
            (32, 2),  # 输入张量的形状
            (192, 16, 8, 6),  # 卷积核的形状
            (16, 32, 2, 2),  # 步幅 (stride)
            (32,),  # 批处理大小
        )
    def test_lazy_conv_transpose2d_state(self):
        # 测试函数：验证 lazy ConvTranspose2d 的状态
        self._check_lazy_conv_state(
            # 使用 lambda 表达式创建 ConvTranspose2d 对象
            lambda: nn.ConvTranspose2d(16, 32, 2),
            # 使用 lambda 表达式创建 LazyConvTranspose2d 对象
            lambda: nn.LazyConvTranspose2d(32, 2),
            # 输入参数：(16, 32, 2, 2)
            (16, 32, 2, 2),
            # 目标形状：(32,)
            (32,),
        )

    @suppress_warnings
    def test_lazy_conv_transpose3d(self):
        # 测试函数：验证 lazy ConvTranspose3d
        self._check_lazy_conv(
            # 使用 nn.ConvTranspose3d 进行测试
            nn.ConvTranspose3d,
            # 使用 nn.LazyConvTranspose3d 进行测试
            nn.LazyConvTranspose3d,
            # 使用 torch.nn.functional.conv_transpose3d 进行测试
            torch.nn.functional.conv_transpose3d,
            # 输入参数：(32, 2)
            (32, 2),
            # 目标形状：(192, 16, 8, 7, 6)
            (192, 16, 8, 7, 6),
            # 输入参数：(16, 32, 2, 2, 2)
            (16, 32, 2, 2, 2),
            # 目标形状：(32,)
            (32,),
        )

    @suppress_warnings
    def test_lazy_conv_transpose3d_kwargs(self):
        # 测试函数：验证带有关键字参数的 lazy ConvTranspose3d
        self._check_lazy_conv(
            # 使用 nn.ConvTranspose3d 进行测试
            nn.ConvTranspose3d,
            # 使用 nn.LazyConvTranspose3d 进行测试
            nn.LazyConvTranspose3d,
            # 使用 torch.nn.functional.conv_transpose3d 进行测试
            torch.nn.functional.conv_transpose3d,
            # 输入参数：(32, 2)
            (32, 2),
            # 目标形状：(192, 16, 8, 7, 6)
            (192, 16, 8, 7, 6),
            # 输入参数：(16, 32, 2, 2, 2)
            (16, 32, 2, 2, 2),
            # 目标形状：(32,)
            (32,),
            # 输出尺寸：(9, 8, 7)
            output_size=(9, 8, 7),
        )

    @suppress_warnings
    def test_lazy_conv_transpose3d_pickle(self):
        # 测试函数：验证 pickle 的 lazy ConvTranspose3d
        self._check_lazy_conv_pickle(
            # 使用 nn.ConvTranspose3d 进行测试
            nn.ConvTranspose3d,
            # 使用 nn.LazyConvTranspose3d 进行测试
            nn.LazyConvTranspose3d,
            # 输入参数：(32, 2)
            (32, 2),
            # 目标形状：(192, 16, 8, 7, 6)
            (192, 16, 8, 7, 6),
            # 输入参数：(16, 32, 2, 2, 2)
            (16, 32, 2, 2, 2),
            # 目标形状：(32,)
            (32,),
        )

    @suppress_warnings
    def test_lazy_conv_transpose3d_state(self):
        # 测试函数：验证 lazy ConvTranspose3d 的状态
        self._check_lazy_conv_state(
            # 使用 lambda 表达式创建 ConvTranspose3d 对象
            lambda: nn.ConvTranspose3d(16, 32, 2),
            # 使用 lambda 表达式创建 LazyConvTranspose3d 对象
            lambda: nn.LazyConvTranspose3d(32, 2),
            # 输入参数：(16, 32, 2, 2, 2)
            (16, 32, 2, 2, 2),
            # 目标形状：(32,)
            (32,),
        )
    # 定义一个方法用于检查懒加载模块的规范性
    def _check_lazy_norm(self, cls, lazy_cls, input_shape):
        # 遍历是否包含仿射变换和是否跟踪运行统计的两种情况
        for affine in [False, True]:
            for track_running_stats in [False, True]:
                # 根据当前仿射和跟踪运行统计的状态创建对应的懒加载模块
                lazy_module = lazy_cls(
                    affine=affine, track_running_stats=track_running_stats
                )

                # 如果仿射为真，断言懒加载模块的权重和偏置为未初始化参数类型
                if affine:
                    self.assertIsInstance(lazy_module.weight, UninitializedParameter)
                    self.assertIsInstance(lazy_module.bias, UninitializedParameter)
                # 如果跟踪运行统计为真，断言懒加载模块的运行均值和方差为未初始化缓冲区类型
                if track_running_stats:
                    self.assertIsInstance(lazy_module.running_mean, UninitializedBuffer)
                    self.assertIsInstance(lazy_module.running_var, UninitializedBuffer)

                # 创建输入数据为全为1的张量
                input = torch.ones(*input_shape)
                # 通过懒加载模块计算输出
                lazy_output = lazy_module(input)
                # 断言懒加载模块的类型为传入的基础类（cls），并且不是懒加载模块本身的类型（lazy_cls）
                self.assertIsInstance(lazy_module, cls)
                self.assertNotIsInstance(lazy_module, lazy_cls)

                # 获取输入形状中的特征数
                num_features = input_shape[1]
                # 根据基础类（cls）、仿射和跟踪运行统计的状态创建模块
                module = cls(
                    num_features, affine=affine, track_running_stats=track_running_stats
                )
                # 计算预期输出
                expected_output = module(input)

                # 断言懒加载模块的输出与预期输出相等
                self.assertEqual(lazy_output, expected_output)
                # 如果模块有权重，断言懒加载模块的权重形状和值与模块的权重相等
                if module.weight is not None:
                    self.assertEqual(lazy_module.weight.shape, module.weight.shape)
                    self.assertEqual(lazy_module.weight, module.weight)
                # 如果模块有偏置，断言懒加载模块的偏置形状和值与模块的偏置相等
                if module.bias is not None:
                    self.assertEqual(lazy_module.bias.shape, module.bias.shape)
                    self.assertEqual(lazy_module.bias, module.bias)
                # 如果模块有运行均值，断言懒加载模块的运行均值形状和值与模块的运行均值相等
                if module.running_mean is not None:
                    self.assertEqual(
                        lazy_module.running_mean.shape, module.running_mean.shape
                    )
                    self.assertEqual(lazy_module.running_mean, module.running_mean)
                # 如果模块有运行方差，断言懒加载模块的运行方差形状和值与模块的运行方差相等
                if module.running_var is not None:
                    self.assertEqual(
                        lazy_module.running_var.shape, module.running_var.shape
                    )
                    self.assertEqual(lazy_module.running_var, module.running_var)
                # 如果模块有跟踪批次的数量，断言懒加载模块的跟踪批次形状和值与模块的跟踪批次相等
                if module.num_batches_tracked is not None:
                    self.assertEqual(
                        lazy_module.num_batches_tracked.shape,
                        module.num_batches_tracked.shape,
                    )
                    self.assertEqual(
                        lazy_module.num_batches_tracked, module.num_batches_tracked
                    )
    # 检查懒加载的规范化和序列化：验证通过序列化和反序列化后的模块状态
    def _check_lazy_norm_pickle(self, cls, lazy_cls, input_shape):
        # 对于每一种仿射变换状态和统计追踪状态的组合进行迭代
        for affine in [False, True]:
            for track_running_stats in [False, True]:
                # 创建懒加载模块实例
                module = lazy_cls(
                    affine=affine, track_running_stats=track_running_stats
                )
                # 序列化并反序列化模块，测试其是否保持 Lazy 特性
                module = pickle.loads(pickle.dumps(module))

                # 断言反序列化后的模块类型为 lazy_cls
                self.assertIsInstance(module, lazy_cls)
                # 如果仿射为真，则验证权重和偏置是未初始化的参数
                if affine:
                    self.assertIsInstance(module.weight, UninitializedParameter)
                    self.assertIsInstance(module.bias, UninitializedParameter)
                # 如果追踪统计状态为真，则验证运行均值和方差是未初始化的缓冲区
                if track_running_stats:
                    self.assertIsInstance(module.running_mean, UninitializedBuffer)
                    self.assertIsInstance(module.running_var, UninitializedBuffer)

                # 创建输入张量并将其输入到模块中，使模块完全实例化
                input = torch.ones(*input_shape)
                module(input)  # fully materialized
                # 再次序列化并反序列化模块
                module = pickle.loads(pickle.dumps(module))

                # 断言反序列化后的模块类型不再是 lazy_cls，而是 cls
                self.assertNotIsInstance(module, lazy_cls)
                self.assertIsInstance(module, cls)
                # 如果仿射为真，则验证权重和偏置不再是未初始化的参数
                if affine:
                    self.assertNotIsInstance(module.weight, UninitializedParameter)
                    self.assertNotIsInstance(module.bias, UninitializedParameter)
                # 如果追踪统计状态为真，则验证运行均值和方差不再是未初始化的缓冲区
                if track_running_stats:
                    self.assertNotIsInstance(module.running_mean, UninitializedBuffer)
                    self.assertNotIsInstance(module.running_var, UninitializedBuffer)

    # 检查懒加载的批归一化状态：验证懒加载模块的状态是否正确加载到普通模块中
    def _check_lazy_batchnorm_state(self, cls, lazy_cls):
        # 创建普通模块实例并加载其状态到懒加载模块中
        module = cls(10)
        lazy_module = lazy_cls(affine=True, track_running_stats=True)
        lazy_module.load_state_dict(module.state_dict())
        
        # 断言懒加载模块没有未初始化的参数
        self.assertFalse(lazy_module.has_uninitialized_params())
        # 验证懒加载模块的权重、偏置、运行均值和方差的形状
        self.assertEqual(lazy_module.weight.shape, (10,))
        self.assertEqual(lazy_module.bias.shape, (10,))
        self.assertEqual(lazy_module.running_mean.shape, (10,))
        self.assertEqual(lazy_module.running_var.shape, (10,))

        # 再次创建普通模块实例和懒加载模块实例
        module = cls(10)
        lazy_module = lazy_cls()
        # 使用断言捕获运行时错误，验证加载懒加载模块状态字典时抛出异常
        with self.assertRaisesRegex(RuntimeError, "shape of an uninitialized"):
            module.load_state_dict(lazy_module.state_dict())
    # 检查延迟实例归一化模块的状态
    def _check_lazy_instancenorm_state(self, cls, lazy_cls):
        # 循环遍历仿射参数和运行统计参数的组合
        for affine in [False, True]:
            for track_running_stats in [False, True]:
                # 创建普通归一化模块实例
                module = cls(10, affine=affine, track_running_stats=track_running_stats)
                # 创建延迟加载的归一化模块实例
                lazy_module = lazy_cls(
                    affine=affine, track_running_stats=track_running_stats
                )
                # 加载普通模块的状态到延迟加载模块
                lazy_module.load_state_dict(module.state_dict())
                # 断言延迟加载模块没有未初始化的参数
                self.assertFalse(lazy_module.has_uninitialized_params())
                # 如果使用了仿射参数，断言权重和偏置的形状正确
                if affine:
                    self.assertEqual(lazy_module.weight.shape, (10,))
                    self.assertEqual(lazy_module.bias.shape, (10,))
                # 如果使用了运行统计参数，断言运行均值和方差的形状正确
                if track_running_stats:
                    self.assertEqual(lazy_module.running_mean.shape, (10,))
                    self.assertEqual(lazy_module.running_var.shape, (10,))

        # 对于仿射参数和运行统计参数都为True的情况，进行额外的状态加载测试
        module = cls(10, affine=True, track_running_stats=True)
        lazy_module = lazy_cls(affine=True, track_running_stats=True)
        # 使用断言异常捕获来测试状态加载的异常情况
        with self.assertRaisesRegex(RuntimeError, "shape of an uninitialized"):
            module.load_state_dict(lazy_module.state_dict())

    # 使用字典输入检查延迟归一化模块的状态
    def _check_lazy_norm_with_dict_input(self, cls, lazy_cls, input_shape):
        # 创建输入字典
        input = {"input": torch.ones(*input_shape)}

        # 创建延迟加载的归一化模块实例
        lazy_module = lazy_cls()
        # 使用输入字典进行延迟加载模块的前向计算
        lazy_output = lazy_module(**input)

        # 获取输入特征数，并创建普通归一化模块实例
        num_features = input_shape[1]
        module = cls(num_features)
        # 使用输入字典进行普通归一化模块的前向计算
        expected_output = module(**input)

        # 断言延迟加载模块的输出与普通模块的输出一致
        self.assertEqual(lazy_output, expected_output)

    # 测试延迟加载BatchNorm1d模块
    def test_lazy_batchnorm1d(self):
        self._check_lazy_norm(nn.BatchNorm1d, nn.LazyBatchNorm1d, (16, 3, 6))
        self._check_lazy_norm(nn.BatchNorm1d, nn.LazyBatchNorm1d, (16, 6))

    # 测试延迟加载BatchNorm1d模块的pickle操作
    def test_lazy_batchnorm1d_pickle(self):
        self._check_lazy_norm_pickle(nn.BatchNorm1d, nn.LazyBatchNorm1d, (16, 3, 6))
        self._check_lazy_norm_pickle(nn.BatchNorm1d, nn.LazyBatchNorm1d, (16, 6))

    # 测试延迟加载BatchNorm1d模块的状态
    def test_lazy_batchnorm1d_state(self):
        self._check_lazy_batchnorm_state(nn.BatchNorm1d, nn.LazyBatchNorm1d)
        self._check_lazy_batchnorm_state(nn.BatchNorm1d, nn.LazyBatchNorm1d)

    # 测试延迟加载BatchNorm2d模块
    def test_lazy_batchnorm2d(self):
        self._check_lazy_norm(nn.BatchNorm2d, nn.LazyBatchNorm2d, (16, 3, 6, 7))

    # 测试延迟加载BatchNorm2d模块的pickle操作
    def test_lazy_batchnorm2d_pickle(self):
        self._check_lazy_norm_pickle(nn.BatchNorm2d, nn.LazyBatchNorm2d, (16, 3, 6, 7))

    # 测试延迟加载BatchNorm2d模块的状态
    def test_lazy_batchnorm2d_state(self):
        self._check_lazy_batchnorm_state(nn.BatchNorm2d, nn.LazyBatchNorm2d)
        self._check_lazy_batchnorm_state(nn.BatchNorm2d, nn.LazyBatchNorm2d)

    # 测试延迟加载BatchNorm3d模块
    def test_lazy_batchnorm3d(self):
        self._check_lazy_norm(nn.BatchNorm3d, nn.LazyBatchNorm3d, (16, 3, 6, 7, 8))
    def test_lazy_batchnorm3d_pickle(self):
        # 测试 nn.BatchNorm3d 到 nn.LazyBatchNorm3d 的批处理规范化的序列化和反序列化
        self._check_lazy_norm_pickle(
            nn.BatchNorm3d, nn.LazyBatchNorm3d, (16, 3, 6, 7, 8)
        )

    def test_lazy_batchnorm3d_state(self):
        # 检查 nn.BatchNorm3d 到 nn.LazyBatchNorm3d 的状态匹配
        self._check_lazy_batchnorm_state(nn.BatchNorm3d, nn.LazyBatchNorm3d)
        self._check_lazy_batchnorm_state(nn.BatchNorm3d, nn.LazyBatchNorm3d)

    def test_lazy_instancenorm1d(self):
        # 测试 nn.InstanceNorm1d 到 nn.LazyInstanceNorm1d 的惰性实例归一化
        self._check_lazy_norm(nn.InstanceNorm1d, nn.LazyInstanceNorm1d, (16, 3, 6))

    def test_lazy_instancenorm1d_pickle(self):
        # 测试 nn.InstanceNorm1d 到 nn.LazyInstanceNorm1d 的序列化和反序列化
        self._check_lazy_norm_pickle(
            nn.InstanceNorm1d, nn.LazyInstanceNorm1d, (16, 3, 6)
        )

    def test_lazy_instancenorm1d_state(self):
        # 检查 nn.InstanceNorm1d 到 nn.LazyInstanceNorm1d 的状态匹配
        self._check_lazy_instancenorm_state(nn.InstanceNorm1d, nn.LazyInstanceNorm1d)
        self._check_lazy_instancenorm_state(nn.InstanceNorm1d, nn.LazyInstanceNorm1d)

    def test_lazy_instancenorm2d(self):
        # 测试 nn.InstanceNorm2d 到 nn.LazyInstanceNorm2d 的惰性实例归一化
        self._check_lazy_norm(nn.InstanceNorm2d, nn.LazyInstanceNorm2d, (16, 3, 6, 7))

    def test_lazy_instancenorm2d_pickle(self):
        # 测试 nn.InstanceNorm2d 到 nn.LazyInstanceNorm2d 的序列化和反序列化
        self._check_lazy_norm_pickle(
            nn.InstanceNorm2d, nn.LazyInstanceNorm2d, (16, 3, 6, 7)
        )

    def test_lazy_instancenorm2d_state(self):
        # 检查 nn.InstanceNorm2d 到 nn.LazyInstanceNorm2d 的状态匹配
        self._check_lazy_instancenorm_state(nn.InstanceNorm2d, nn.LazyInstanceNorm2d)
        self._check_lazy_instancenorm_state(nn.InstanceNorm2d, nn.LazyInstanceNorm2d)

    def test_lazy_instancenorm3d(self):
        # 测试 nn.InstanceNorm3d 到 nn.LazyInstanceNorm3d 的惰性实例归一化
        self._check_lazy_norm(
            nn.InstanceNorm3d, nn.LazyInstanceNorm3d, (16, 3, 6, 7, 8)
        )

    def test_lazy_instancenorm3d_pickle(self):
        # 测试 nn.InstanceNorm3d 到 nn.LazyInstanceNorm3d 的序列化和反序列化
        self._check_lazy_norm_pickle(
            nn.InstanceNorm3d, nn.LazyInstanceNorm3d, (16, 3, 6, 7, 8)
        )

    def test_lazy_instancenorm3d_state(self):
        # 检查 nn.InstanceNorm3d 到 nn.LazyInstanceNorm3d 的状态匹配
        self._check_lazy_instancenorm_state(nn.InstanceNorm3d, nn.LazyInstanceNorm3d)
        self._check_lazy_instancenorm_state(nn.InstanceNorm3d, nn.LazyInstanceNorm3d)

    def test_lazy_batchnorm_with_dict_input(self):
        # 测试接受字典输入的 nn.BatchNorm1d、nn.BatchNorm2d 和 nn.BatchNorm3d 到 nn.LazyBatchNorm1d、nn.LazyBatchNorm2d 和 nn.LazyBatchNorm3d 的批处理规范化
        self._check_lazy_norm_with_dict_input(
            nn.BatchNorm1d, nn.LazyBatchNorm1d, (16, 3, 6)
        )
        self._check_lazy_norm_with_dict_input(
            nn.BatchNorm2d, nn.LazyBatchNorm2d, (16, 3, 6, 7)
        )
        self._check_lazy_norm_with_dict_input(
            nn.BatchNorm3d, nn.LazyBatchNorm3d, (16, 3, 6, 7, 8)
        )

    @suppress_warnings
    def test_materialize_dtype(self):
        # 测试惰性模块的数据类型材料化
        module = LazyModule()
        module.register_parameter("test_param", UninitializedParameter())
        module.test_param.materialize(10)
        self.assertTrue(module.test_param.dtype == torch.get_default_dtype())
        module = LazyModule()
        module.register_parameter("test_param", UninitializedParameter())
        module.half()
        module.test_param.materialize(10)
        self.assertTrue(module.test_param.dtype == torch.float16)

    @unittest.skipIf(
        not (TEST_CUDA or TEST_PRIVATEUSE1), "CUDA and PRIVATEUSE1 not available"
    )
    @suppress_warnings
    def test_materialize_device(self):
        # 创建一个 LazyModule 实例
        module = LazyModule()
        # 注册一个未初始化的参数 "test_param" 到模块中
        module.register_parameter("test_param", UninitializedParameter())
        # 执行参数的实例化操作，设置参数大小为 10
        module.test_param.materialize(10)
        # 断言测试：验证模块参数所在设备类型为 CPU
        self.assertTrue(module.test_param.device.type == "cpu")
        # 如果开启了 CUDA 测试选项
        if TEST_CUDA:
            device = "cuda"
        # 否则，如果开启了私有使用1的测试选项
        elif TEST_PRIVATEUSE1:
            device = torch._C._get_privateuse1_backend_name()
        # 创建一个新的 LazyModule 实例
        module = LazyModule()
        # 注册一个未初始化的参数 "test_param" 到模块中
        module.register_parameter("test_param", UninitializedParameter())
        # 将模块移动到特定设备上
        module.to(device)
        # 执行参数的实例化操作，设置参数大小为 10
        module.test_param.materialize(10)
        # 断言测试：验证模块参数所在设备类型与指定设备类型一致
        self.assertTrue(module.test_param.device.type == device)

    @suppress_warnings
    def test_chained_initialization(self):
        # 定义一个名为 MyNetwork 的子类，继承自 torch.nn.Module
        class MyNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 声明一个延迟加载的线性层 linear_1，输入大小为 15
                self.linear_1 = torch.nn.LazyLinear(15)
                # 声明一个延迟加载的线性层 linear_2，输入大小为 10
                self.linear_2 = torch.nn.LazyLinear(10)

            # 前向传播函数定义
            def forward(self, x):
                # 应用第一个延迟加载的线性层 linear_1
                y = self.linear_1(x)
                # 返回第二个延迟加载的线性层 linear_2 的结果
                return self.linear_2(y)

        # 创建 MyNetwork 类的实例 net
        net = MyNetwork()
        # 对 net 执行前向传播，传入大小为 (5, 10) 的张量
        net(torch.ones(5, 10))
        # 断言测试：验证 linear_1 的权重形状为 (15, 10)
        self.assertTrue(net.linear_1.weight.shape == (15, 10))
        # 断言测试：验证 linear_1 的偏置形状为 (15,)
        self.assertTrue(net.linear_1.bias.shape == (15,))
        # 断言测试：验证 linear_2 的权重形状为 (10, 15)
        self.assertTrue(net.linear_2.weight.shape == (10, 15))
        # 断言测试：验证 linear_2 的偏置形状为 (10,)
        self.assertTrue(net.linear_2.bias.shape == (10,))
    
    @suppress_warnings
    # 定义一个测试优化器的方法
    def test_optimizer_pass(self):
        # 定义一组优化器类列表
        optimizers = [
            torch.optim.Adadelta,
            torch.optim.Adagrad,
            torch.optim.Adamax,
            torch.optim.Adam,
            torch.optim.AdamW,
            torch.optim.ASGD,
            torch.optim.SGD,
            torch.optim.Rprop,
            torch.optim.RMSprop,
            torch.optim.LBFGS,
            torch.optim.NAdam,
            torch.optim.RAdam,
        ]

        # 定义运行优化步骤的函数，接受模块和优化器作为参数
        def run_step(module, optim):
            # 断言第一个参数组的第一个参数是未初始化的参数对象
            self.assertIsInstance(
                optim.param_groups[0]["params"][0], UninitializedParameter
            )
            # 将模块的测试参数实例化
            module.test_param.materialize(10)
            # 断言第一个参数组的第一个参数是已初始化的参数对象
            self.assertIsInstance(optim.param_groups[0]["params"][0], Parameter)
            # 断言第一个参数组的第一个参数不再是未初始化的参数对象
            self.assertNotIsInstance(
                optim.param_groups[0]["params"][0], UninitializedParameter
            )
            # 为模块的所有参数设置随机梯度
            for p in module.parameters():
                p.grad = torch.rand_like(p)
            # 如果优化器是 torch.optim.LBFGS 类型，则执行步骤并传入 lambda 函数
            if isinstance(optim, torch.optim.LBFGS):
                optim.step(lambda: 1.0)
            else:
                # 否则，直接执行优化步骤
                optim.step()

        # 遍历优化器类列表
        for optim_cls in optimizers:
            # 创建一个 LazyModule 实例
            module = LazyModule()
            # 注册一个名为 "test_param" 的未初始化参数
            module.register_parameter("test_param", UninitializedParameter())
            # 根据优化器类选择初始化方式
            if optim_cls is torch.optim.SGD:
                optim = optim_cls(module.parameters(), lr=0.0)
            elif optim_cls is torch.optim.Adagrad:
                # 对于 Adagrad 类型的优化器，使用断言捕获异常并跳过当前循环
                with self.assertRaisesRegex(ValueError, "uninitialized parameter"):
                    optim = optim_cls(module.parameters())
                continue
            else:
                # 对于其他类型的优化器，直接初始化
                optim = optim_cls(module.parameters())
            # 运行优化步骤
            run_step(module, optim)

    # 装饰器，用于测试权重归一化
    @suppress_warnings
    def test_weight_norm(self):
        # 创建一个 LazyLinear 模块
        m = nn.LazyLinear(7)
        # 使用断言捕获异常，断言异常信息包含 "have uninitialized parameters."
        with self.assertRaisesRegex(ValueError, "have uninitialized parameters."):
            # 对模块应用权重归一化
            m = torch.nn.utils.weight_norm(m)

    # 装饰器，用于测试谱归一化
    @suppress_warnings
    def test_spectral_norm(self):
        # 创建一个 LazyLinear 模块
        m = nn.LazyLinear(7)
        # 使用断言捕获异常，断言异常信息包含 "have uninitialized parameters."
        with self.assertRaisesRegex(ValueError, "have uninitialized parameters."):
            # 对模块应用谱归一化
            m = torch.nn.utils.spectral_norm(m)

    # 装饰器，用于测试无效函数操作
    @suppress_warnings
    def test_invalid_functions(self):
        # 创建一个未初始化参数对象
        param = torch.nn.parameter.UninitializedParameter()
        # 使用断言捕获异常，断言异常信息包含 "uninitialized parameter"
        with self.assertRaisesRegex(ValueError, "uninitialized parameter"):
            # 使用未初始化参数对象创建一个空张量
            torch.empty_like(param)

        # 使用断言捕获异常，断言异常信息包含 "uninitialized parameter"
        with self.assertRaisesRegex(ValueError, "uninitialized parameter"):
            # 对两个未初始化参数对象执行加法操作
            torch.add(param, param)

        # 使用断言捕获异常，断言异常信息包含 "uninitialized parameter"
        with self.assertRaisesRegex(ValueError, "uninitialized parameter"):
            # 对未初始化参数对象执行加法操作
            param + param
# 如果当前模块被直接运行（而不是被导入到另一个模块中执行），则执行下面的代码块
if __name__ == "__main__":
    # 调用名为 run_tests 的函数，用于执行测试
    run_tests()
```