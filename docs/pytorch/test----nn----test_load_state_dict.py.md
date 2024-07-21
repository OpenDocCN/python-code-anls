# `.\pytorch\test\nn\test_load_state_dict.py`

```
# 导入正则表达式模块
import re
# 导入单元测试模块
import unittest
# 导入深拷贝函数
from copy import deepcopy
# 导入笛卡尔积生成函数
from itertools import product

# 导入PyTorch深度学习框架
import torch
# 导入PyTorch的神经网络模块
import torch.nn as nn
# 导入PyTorch内部用于测试的通用神经网络测试类
from torch.testing._internal.common_nn import NNTestCase
# 导入PyTorch内部用于测试的通用工具函数
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfCrossRef,
    skipIfTorchDynamo,
    swap,
    TEST_NUMPY,
    TestCase,
)
# 导入PyTorch内部用于树形结构映射的函数
from torch.utils._pytree import tree_map

# 如果测试使用了NumPy，则导入NumPy
if TEST_NUMPY:
    import numpy as np

# 定义一个测试类，继承自NNTestCase
class TestLoadStateDict(NNTestCase):
    # 设置标志，用于检查CUDA内存泄漏
    _do_cuda_memory_leak_check = True
    # 设置标志，用于检查CUDA非默认流
    _do_cuda_non_default_stream = True

    # 标记为跳过测试，如果没有安装NumPy
    @unittest.skipIf(not TEST_NUMPY, "numpy not found")
    # 装饰器，交换测试参数True和False
    @swap([True, False])
    def test_load_state_dict_invalid(self):
        # 创建一个线性层模型
        m = torch.nn.Linear(2, 2, bias=False)

        # 定义一个包含随机数据的状态字典
        state_dict = {"weight": np.random.randn(2, 2)}
        # 使用断言检查是否引发了RuntimeError异常，错误信息中包含特定字符串
        with self.assertRaisesRegex(
            RuntimeError,
            "expected torch.Tensor or Tensor-like object from checkpoint but received",
        ):
            # 加载模型状态字典
            m.load_state_dict(state_dict)

        # 定义一个包含元组数据的状态字典
        state_dict = {"weight": ((1.0, 1.0), (2.0, 2.0))}
        # 使用断言检查是否引发了RuntimeError异常，错误信息中包含特定字符串
        with self.assertRaisesRegex(
            RuntimeError,
            "expected torch.Tensor or Tensor-like object from checkpoint but received",
        ):
            # 加载模型状态字典
            m.load_state_dict(state_dict)

    # 装饰器，交换测试参数True和False
    @swap([True, False])
    def test_load_state_dict_type(self):
        # 创建一个空的神经网络模块
        m = nn.Module()

        # 使用断言检查是否引发了TypeError异常，错误信息中包含特定字符串
        with self.assertRaisesRegex(
            TypeError, "Expected state_dict to be dict-like, got"
        ):
            # 加载模型状态字典
            m.load_state_dict("")
        # 使用断言检查是否引发了TypeError异常，错误信息中包含特定字符串
        with self.assertRaisesRegex(
            TypeError, "Expected state_dict to be dict-like, got"
        ):
            # 加载模型状态字典
            m.load_state_dict(2)

    # 装饰器，交换测试参数True和False；跳过条件为Torch Dynamo安装了弱引用参数
    @swap([True, False])
    @skipIfTorchDynamo("dynamo installs weakrefs on some params")
    @swap([True, False])
    def test_load_state_dict_BC(self):
        # 创建一个BatchNorm2d层模型
        bn = nn.BatchNorm2d(3)
        # 获取BatchNorm2d层的状态字典
        state_dict = bn.state_dict()
        # 删除状态字典中的num_batches_tracked键
        del state_dict["num_batches_tracked"]
        # 设置状态字典的元数据中的版本号为1
        state_dict._metadata[""]["version"] = 1  # version 1
        # 加载模型状态字典
        bn.load_state_dict(state_dict)
        # 使用断言检查num_batches_tracked属性的数据类型是否为torch.long
        self.assertEqual(bn.num_batches_tracked.dtype, torch.long)
        # 使用断言检查num_batches_tracked属性的值是否为0
        self.assertEqual(bn.num_batches_tracked.item(), 0)
        # 删除状态字典的元数据中的版本号
        del state_dict._metadata[""]["version"]  # no version
        # 加载模型状态字典
        bn.load_state_dict(state_dict)
        # 使用断言检查num_batches_tracked属性的数据类型是否为torch.long
        self.assertEqual(bn.num_batches_tracked.dtype, torch.long)
        # 使用断言检查num_batches_tracked属性的值是否为0
        self.assertEqual(bn.num_batches_tracked.item(), 0)
    # 测试加载子模型的状态字典
    def test_load_state_dict_child(self):
        # 创建一个线性模型
        base_module = nn.Linear(1, 1)
        model = base_module
        # 多次嵌套创建序列模型
        for _ in range(3):
            model = nn.Sequential(*[deepcopy(model) for _ in range(10])

        # 定义钩子函数，用于检查加载状态字典前后的状态
        def hook_fn(
            module,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            # 获取模型的状态字典
            module_state_dict = module.state_dict()
            # 检查状态字典的键数量是否一致
            self.assertEqual(len(module_state_dict.keys()), len(state_dict.keys()))

        # 注册加载状态字典前的钩子函数
        model[0][0]._register_load_state_dict_pre_hook(hook_fn, with_module=True)
        # 加载模型的状态字典
        model.load_state_dict(model.state_dict(), strict=True)

    # 测试加载状态字典时是否会导致循环引用
    # 由于 LSTM 在参数上安装了弱引用，所以会失败
    @swap([False])
    @skipIfTorchDynamo("TorchDynamo fails here for unknown reasons")
    def test_load_state_dict_ref_cycle(self):
        # 加载状态字典不应导致涉及张量的循环引用
        import gc

        # 创建一个 LSTM 模型
        m = torch.nn.LSTM(16, 16, bidirectional=True)

        gc.collect()
        # 加载深拷贝模型的状态字典
        m.load_state_dict(deepcopy(m).state_dict())
        refcycles = gc.collect()

        # 检查循环引用数量是否为 0
        self.assertEqual(refcycles, 0)

    # 测试自定义加载状态字典的情况
    @swap([True, False])
    def test_load_state_dict_custom(self):
        # 定义一个自定义模型
        class CustomState(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.ones(1))
                self.sub = torch.nn.Linear(5, 5)

            # 自定义保存到状态字典的方法
            def _save_to_state_dict(self, destination, prefix, keep_vars):
                destination[prefix + "serialized"] = self.param.data + 1

            # 自定义从状态字典加载的方法
            def _load_from_state_dict(
                self,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                # 省略部分错误处理
                self.param.data.copy_(state_dict[prefix + "serialized"] - 1)

        # 使用 Sequential 模型验证嵌套
        m = nn.Sequential(CustomState())
        with torch.no_grad():
            m[0].param[0] = 10
            m[0].sub.weight[0, 0] = 555
        state_dict = m.state_dict()
        # 检查状态字典中的值
        self.assertEqual(state_dict["0.serialized"].item(), 11)
        self.assertIn("0.sub.weight", state_dict)
        self.assertNotIn("0.param", state_dict)
        del m
        mm = nn.Sequential(CustomState())
        self.assertEqual(mm[0].param[0].item(), 1)
        # 加载状态字典
        mm.load_state_dict(state_dict)
        self.assertEqual(mm[0].param[0].item(), 10)
        self.assertEqual(mm[0].sub.weight[0, 0].item(), 555)

    @swap([True, False])
    @parametrize("keep_vars", [True, False])
    @swap([True, False])
    def test_load_state_dict_assign_with_optimizer(self):
        # 定义一个简单的神经网络模型
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 5)  # 添加一个全连接层，输入维度为3，输出维度为5
                self.bn = nn.BatchNorm1d(5)  # 添加一个批标准化层，输入维度为5

            def forward(self, input):
                return self.bn(self.fc1(input))  # 前向传播，先通过全连接层再通过批标准化层

        net = MyModule()  # 创建一个 MyModule 的实例 net
        opt = torch.optim.Adam(net.parameters(), lr=1000)  # 使用 Adam 优化器优化 net 的参数，学习率为 1000
        x = torch.randn(4, 3)  # 生成一个大小为 4x3 的随机张量 x
        num_iters = 3  # 设置迭代次数为 3

        for i in range(num_iters):
            opt.zero_grad()  # 梯度清零
            out = net(x)  # 前向传播
            out.sum().backward()  # 计算损失并反向传播
            opt.step()  # 更新参数

        opt_state_dict = deepcopy(opt.state_dict())  # 深拷贝优化器的状态字典
        net_state_dict = deepcopy(net.state_dict())  # 深拷贝网络模型的状态字典

        with torch.device("meta"):  # 将环境设为 "meta"
            net_meta = MyModule()  # 创建一个新的 MyModule 实例 net_meta

        net_meta.load_state_dict(net_state_dict, assign=True)
        # 使用 assign=True 时，加载状态字典到 net_meta，必须在加载状态字典后创建优化器

        opt2 = torch.optim.Adam(net_meta.parameters(), lr=1000)  # 创建新的 Adam 优化器 opt2
        opt2.load_state_dict(opt_state_dict)  # 加载优化器状态字典到 opt2

        y = x.clone()  # 克隆张量 x 到 y
        for i in range(num_iters):
            opt.zero_grad()  # 梯度清零
            out = net(x)  # 前向传播
            out.sum().backward()  # 计算损失并反向传播
            opt.step()  # 更新原始网络参数

            opt2.zero_grad()  # 梯度清零
            out2 = net_meta(y)  # 前向传播
            out2.sum().backward()  # 计算损失并反向传播
            opt2.step()  # 更新 meta 网络参数

        self.assertEqual(opt.state_dict(), opt2.state_dict())  # 断言原始优化器和 meta 优化器状态字典相等
        self.assertEqual(net.state_dict(), net_meta.state_dict())  # 断言原始网络和 meta 网络状态字典相等

    @swap([True, False])
    def test_load_state_dict_assign_shape_stride(self):
        # 允许分配张量具有不同于初始张量的其他属性，除了形状
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 5)  # 添加一个全连接层，输入维度为3，输出维度为5
                self.bn = nn.BatchNorm1d(5)  # 添加一个批标准化层，输入维度为5

            def forward(self, input):
                return self.bn(self.fc1(input))  # 前向传播，先通过全连接层再通过批标准化层

        net = MyModule()  # 创建一个 MyModule 的实例 net
        state_dict = net.state_dict()  # 获取网络模型的状态字典
        state_dict["fc1.weight"] = torch.randn(3, 5).transpose(0, 1)  # 修改权重张量的形状
        net2 = MyModule()  # 创建一个新的 MyModule 实例 net2
        net2.load_state_dict(state_dict, strict=False, assign=True)  # 使用非严格模式加载状态字典到 net2

        state_dict["fc1.weight"] = torch.randn(2, 4)  # 修改权重张量的形状
        with self.assertRaisesRegex(
            RuntimeError, "size mismatch for fc1.weight: copying a param with shape"
        ):
            net2.load_state_dict(state_dict, strict=False, assign=True)  # 抛出形状不匹配的运行时错误

    @swap([True, False])
    def test_load_state_dict_warn_assign(self):
        with torch.device("meta"):  # 将环境设为 "meta"
            m = torch.nn.Linear(3, 5)  # 创建一个线性层 m，输入维度为3，输出维度为5
        state_dict = m.state_dict()  # 获取线性层的状态字典
        state_dict["weight"] = torch.empty_like(state_dict["weight"], device="cpu")  # 修改权重张量的设备为 CPU
        with self.assertWarnsRegex(
            UserWarning,
            "for weight: copying from a non-meta parameter in the checkpoint to a meta",
        ):
            m.load_state_dict(state_dict)  # 加载状态字典到线性层，并发出警告
    # 定义一个测试函数，用于测试加载模型状态字典时出现意外键的情况
    def test_load_state_dict_with_unexpected_key(self):
        # 定义一个继承自 torch.nn.Module 的子类 MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在构造函数中定义一个线性层
                self.fc1 = torch.nn.Linear(5, 10)

        # 创建 MyModule 的实例
        m = MyModule()

        # 测试意外键且 strict = True 的情况
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            # 获取当前模型的状态字典
            state_dict = m.state_dict()
            # 向状态字典中加入一个意外的键值对
            state_dict["fc1.bad_suffix"] = torch.randn(5, 10)
            # 加载修改后的状态字典到模型
            m.load_state_dict(state_dict)

        # 测试意外键且 strict = False 的情况
        state_dict = m.load_state_dict(state_dict, strict=False)
        # 断言检查在加载过程中是否发现了意外的键
        self.assertIn("fc1.bad_suffix", state_dict.unexpected_keys)

        # 测试意外键的前缀与有效键匹配且 strict = True 的情况
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            # 再次获取当前模型的状态字典
            state_dict = m.state_dict()
            # 向状态字典中加入一个前缀与有效键匹配的意外键值对
            state_dict["fc1.weight.bad_suffix"] = torch.randn(5, 10)
            # 加载修改后的状态字典到模型
            m.load_state_dict(state_dict)

        # 测试意外键的前缀与有效键匹配且 strict = False 的情况
        state_dict = m.load_state_dict(state_dict, strict=False)
        # 断言检查在加载过程中是否发现了意外的键
        self.assertIn("fc1.weight.bad_suffix", state_dict.unexpected_keys)
def load_torch_function_handler(cls, func, types, args=(), kwargs=None):
    kwargs = {} if kwargs is None else kwargs

    # 定义一个内部函数 module_load，用于处理加载操作
    def module_load(dest, src, assign=False):
        # 如果目标对象是 cls 的实例
        if isinstance(dest, cls):
            # 如果需要赋值
            if assign:
                # 返回源张量的分离版本
                return src.detach()
            else:
                # 如果源是 torch.Tensor 类型
                if type(src) is torch.Tensor:
                    # 使用 cls 类创建一个新的对象
                    return cls(src)
                # 如果源是 cls 类型的对象
                elif type(src) is cls:
                    # 返回源对象的分离版本
                    return src.detach()
                else:
                    # 如果源是 MyWrapperLoadTensor 类的实例
                    if isinstance(src, MyWrapperLoadTensor):
                        # 使用 cls 类创建一个新对象，使用 src 的数据
                        return cls(src._data)
                    # 否则，使用 cls 类创建一个新对象，使用 src 的数据
                    return cls(src)
        else:
            # 断言检查，确保 src 是 cls 类的实例
            assert isinstance(
                src, cls
            ), f"Expected isinstance(src, {cls}) but got {type(src)}"
            # 断言检查，确保 dest 是 torch.Tensor 或 torch.nn.Parameter 类型，或者是 cls 的子类
            assert (
                type(dest) == torch.Tensor
                or type(dest) == torch.nn.Parameter
                or issubclass(cls, type(dest))
            )
            # 如果需要赋值
            if assign:
                # 返回源张量的分离版本
                return src.detach()
            else:
                # 如果源是 MyWrapperLoadTensor 类的实例
                if isinstance(src, MyWrapperLoadTensor):
                    # 如果 dest 的类型不是 torch.Tensor 或 torch.nn.Parameter
                    if type(dest) not in {torch.Tensor, torch.nn.Parameter}:
                        # 使用 dest 的类型创建一个新对象，使用 src 的数据
                        return type(dest)(src._data)
                    else:
                        # 返回 src 的数据的分离版本
                        return src._data.detach()
                else:
                    # 使用 torch.Tensor 创建一个新对象，使用 src 的数据
                    return torch.Tensor(src)

    # 如果 func 是 torch.Tensor.module_load
    if func is torch.Tensor.module_load:
        # 调用 module_load 函数处理参数并返回结果
        return module_load(*args, **kwargs)
    else:
        # 禁用 torch 函数子类化
        with torch._C.DisableTorchFunctionSubclass():
            # 如果 func 是 torch.Tensor.detach
            if func == torch.Tensor.detach:
                # 调用 func 处理参数并返回结果
                ret = func(*args, **kwargs)
                # 如果返回的结果不是 cls 的实例
                if not isinstance(ret, cls):
                    # 使用 cls 类创建一个新对象，使用 ret 的数据
                    return cls(ret)
                # 返回 ret 对象
                return ret
            # 否则，调用 func 处理参数并返回结果
            return func(*args, **kwargs)


# 创建一个自定义的张量类 MyLoadTensor，扩展自 torch.Tensor
class MyLoadTensor(torch.Tensor):
    @classmethod
    # 实现 __torch_function__ 方法
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # 调用 load_torch_function_handler 处理 torch 函数调用
        return load_torch_function_handler(cls, func, types, args, kwargs)


# 使用 MyLoadTensor2 测试张量子类，包装张量子类
# 其中二者都不互相继承
class MyLoadTensor2(torch.Tensor):
    @classmethod
    # 实现 __torch_function__ 方法
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # 调用 load_torch_function_handler 处理 torch 函数调用
        return load_torch_function_handler(cls, func, types, args, kwargs)


# 创建一个自定义的张量类 MyBrokenLoadTensor，扩展自 torch.Tensor
class MyBrokenLoadTensor(torch.Tensor):
    @classmethod
    # 实现 __torch_function__ 方法
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        # 如果 func 是 torch.Tensor.module_load
        if func is torch.Tensor.module_load:
            # 错误的示例，这里没有分离操作！
            return args[1]
        else:
            # 禁用 torch 函数子类化
            with torch._C.DisableTorchFunctionSubclass():
                # 如果 func 是 torch.Tensor.detach
                if func == torch.Tensor.detach:
                    # 使用 cls 创建一个新对象，处理 func 的参数并返回结果
                    return cls(func(*args, **kwargs))
                # 否则，调用 func 处理参数并返回结果
                return func(*args, **kwargs)


# 创建一个自定义的张量类 MyWrapperLoadTensor，扩展自 MyLoadTensor
class MyWrapperLoadTensor(MyLoadTensor):
    @staticmethod
    # 定义一个新的类方法 `__new__`，用于创建包装了 torch.Tensor 的子类实例
    def __new__(cls, data: torch.Tensor):
        # 使用 torch.Tensor._make_wrapper_subclass 方法创建子类实例
        t = torch.Tensor._make_wrapper_subclass(
            cls,  # 子类类型
            data.size(),  # 数据大小
            dtype=data.dtype,  # 数据类型
            layout=data.layout,  # 数据布局
            device=data.device,  # 数据设备
            requires_grad=data.requires_grad,  # 是否需要梯度
            strides=data.stride(),  # 数据步长
            storage_offset=data.storage_offset(),  # 存储偏移量
        )
        return t  # 返回创建的子类实例

    # 初始化方法，将传入的 torch.Tensor 数据保存在 _data 属性中
    def __init__(self, data: torch.Tensor):
        self._data = data

    # 返回对象的字符串表示形式，包含类名和 _data 的字符串表示
    def __repr__(self):
        return f"MyWrapperLoadTensor({self._data.__repr__()})"

    # 类方法，用于 Torch 分发功能，将 MyWrapperLoadTensor 实例包装或者解包
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # 解包输入参数中的 MyWrapperLoadTensor 实例，得到其 _data 属性
        def unwrap(t):
            return t._data if isinstance(t, MyWrapperLoadTensor) else t

        # 将输入参数中的 torch.Tensor 实例包装为 MyWrapperLoadTensor 实例
        def wrap(t):
            return MyWrapperLoadTensor(t) if isinstance(t, torch.Tensor) else t

        # 如果 kwargs 为 None，则初始化为空字典
        kwargs = {} if kwargs is None else kwargs
        # 对传入的函数及其参数应用 unwrap 函数解包 MyWrapperLoadTensor 实例
        out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        # 对结果应用 wrap 函数，将 torch.Tensor 实例包装为 MyWrapperLoadTensor 实例
        return tree_map(wrap, out)
class TestLoadStateDictSwap(TestCase):
    # 定义测试类 TestLoadStateDictSwap，继承自 TestCase 类

    @skipIfCrossRef
    # 如果跨引用测试，则跳过该测试函数
    @skipIfTorchDynamo("Can't swap with dynamo as dynamo installs weakrefs")
    # 如果使用 Torch Dynamo，则跳过该测试函数，因为 Dynamo 安装了弱引用

    @swap([True])
    # 使用 @swap 装饰器标记该测试函数可以进行对象交换，传入参数 [True]

    @parametrize("assign", [True, False])
    # 参数化测试，assign 参数可以是 True 或 False
    def test_swap_subclass(self, assign):
        # 定义测试函数 test_swap_subclass，接受参数 assign

        def _create_model(subclass=None):
            # 定义内部函数 _create_model，用于创建模型
            m = torch.nn.Linear(2, 3, bias=False)
            # 创建一个线性层模型，输入维度为 2，输出维度为 3，无偏置
            m.register_buffer("buf", torch.randn(2, 3))
            # 向模型注册一个名为 "buf" 的缓冲区，内容为形状为 (2, 3) 的随机张量
            if subclass is not None:
                # 如果传入了子类参数
                m.weight = torch.nn.Parameter(subclass(m.weight))
                # 使用子类对模型的权重进行包装
                m.buf = subclass(m.buf)
                # 使用子类对模型的缓冲区进行包装
            return m
            # 返回创建的模型对象

        def _test(m_subclass=None, sd_subclass=None):
            # 定义内部函数 _test，用于测试加载状态字典的行为，接受模型子类和状态字典子类作为参数
            m = _create_model(m_subclass)
            # 创建模型 m，使用指定的模型子类
            sd = _create_model(sd_subclass).state_dict()
            # 创建状态字典 sd，使用指定的状态字典子类，并获取其状态字典表示
            m.load_state_dict(sd, assign=assign)
            # 载入状态字典 sd 到模型 m，使用 assign 参数指定的赋值方式
            self.assertEqual(m.weight, sd["weight"])
            # 断言模型 m 的权重与状态字典中的权重一致
            self.assertEqual(m.buf, sd["buf"])
            # 断言模型 m 的缓冲区与状态字典中的缓冲区一致
            self.assertTrue(isinstance(m.weight, torch.nn.Parameter))
            # 断言模型 m 的权重是 torch.nn.Parameter 类型
            self.assertTrue(not isinstance(m.buf, torch.nn.Parameter))
            # 断言模型 m 的缓冲区不是 torch.nn.Parameter 类型

            weight_type, buf_type = (torch.nn.Parameter, torch.Tensor)
            # 定义权重类型和缓冲区类型，默认分别为 torch.nn.Parameter 和 torch.Tensor
            if assign:
                # 如果 assign 为 True
                if sd_subclass is not None:
                    # 如果状态字典子类不为空
                    weight_type, buf_type = (sd_subclass, sd_subclass)
                    # 更新权重类型和缓冲区类型为状态字典子类
            else:
                # 如果 assign 不为 True
                if m_subclass is not None:
                    # 如果模型子类不为空
                    weight_type, buf_type = (m_subclass, m_subclass)
                    # 更新权重类型和缓冲区类型为模型子类

            self.assertTrue(type(m.weight) is weight_type)
            # 断言模型 m 的权重类型与预期的权重类型相符
            self.assertTrue(type(m.buf) is buf_type)
            # 断言模型 m 的缓冲区类型与预期的缓冲区类型相符

        # 使用 product 函数生成模型子类和状态字典子类的组合，进行测试
        subclasses = [None, MyLoadTensor, MyLoadTensor2, MyWrapperLoadTensor]
        for m_s, sd_s in product(subclasses, subclasses):
            _test(m_s, sd_s)

        # 对于 MyBrokenLoadTensor 应当抛出 RuntimeError 异常，因为其 module_load 方法未调用 .detach()
        with self.assertRaisesRegex(
            RuntimeError, re.escape("Error(s) in loading state_dict for Linear:")
        ):
            _test(None, MyBrokenLoadTensor)


instantiate_parametrized_tests(TestLoadStateDict)
# 实例化参数化测试 TestLoadStateDict

instantiate_parametrized_tests(TestLoadStateDictSwap)
# 实例化参数化测试 TestLoadStateDictSwap

if __name__ == "__main__":
    # 如果当前脚本作为主程序运行
    TestCase._default_dtype_check_enabled = True
    # 启用 TestCase 默认数据类型检查
    run_tests()
    # 运行测试函数
```