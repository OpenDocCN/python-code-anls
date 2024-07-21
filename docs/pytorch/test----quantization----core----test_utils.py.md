# `.\pytorch\test\quantization\core\test_utils.py`

```
# Owner(s): ["oncall: quantization"]

import torch
from torch.testing._internal.common_utils import TestCase  # 导入 TestCase 类，用于编写测试用例
from torch.ao.quantization.utils import get_fqn_to_example_inputs  # 导入获取完全限定名到示例输入映射的函数
from torch.ao.nn.quantized.modules.utils import _quantize_weight  # 导入权重量化函数
from torch.ao.quantization import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver  # 导入量化相关的观察者类


class TestUtils(TestCase):
    def _test_get_fqn_to_example_inputs(self, M, example_inputs, expected_fqn_to_dim):
        m = M().eval()  # 实例化模型 M 并设置为评估模式
        fqn_to_example_inputs = get_fqn_to_example_inputs(m, example_inputs)  # 获取模型中每个模块的完全限定名到示例输入的映射
        for fqn, expected_dims in expected_fqn_to_dim.items():
            assert fqn in fqn_to_example_inputs  # 断言确保完全限定名在映射中存在
            example_inputs = fqn_to_example_inputs[fqn]  # 获取完全限定名对应的示例输入列表
            for example_input, expected_dim in zip(example_inputs, expected_dims):
                assert example_input.dim() == expected_dim  # 断言示例输入的维度与预期维度一致

    def test_get_fqn_to_example_inputs_simple(self):
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)  # 子模块包含一个线性层
                self.linear2 = torch.nn.Linear(5, 5)  # 子模块包含另一个线性层

            def forward(self, x):
                x = self.linear1(x)  # 子模块的前向传播中使用第一个线性层
                x = self.linear2(x)  # 继续在子模块的前向传播中使用第二个线性层
                return x

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)  # 主模块包含一个线性层
                self.linear2 = torch.nn.Linear(5, 5)  # 主模块包含另一个线性层
                self.sub = Sub()  # 主模块包含子模块

            def forward(self, x):
                x = self.linear1(x)  # 主模块的前向传播中使用第一个线性层
                x = self.linear2(x)  # 继续在主模块的前向传播中使用第二个线性层
                x = self.sub(x)  # 在主模块的前向传播中使用子模块
                return x

        expected_fqn_to_dim = {
            "": (2,),  # 主模块和子模块共有2个示例输入
            "linear1": (2,),  # 主模块的第一个线性层有2个示例输入
            "linear2": (2,),  # 主模块的第二个线性层有2个示例输入
            "sub": (2,),  # 子模块有2个示例输入
            "sub.linear1": (2,),  # 子模块的第一个线性层有2个示例输入
            "sub.linear2": (2,)  # 子模块的第二个线性层有2个示例输入
        }
        example_inputs = (torch.rand(1, 5),)  # 创建一个示例输入元组
        self._test_get_fqn_to_example_inputs(M, example_inputs, expected_fqn_to_dim)  # 调用测试函数来验证映射关系
    def test_get_fqn_to_example_inputs_default_kwargs(self):
        """ Test that we can get example inputs for functions with default keyword arguments
        """
        # 定义一个名为 `Sub` 的内部类，继承自 `torch.nn.Module`
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化两个线性层，输入和输出维度都是 5
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)

            # 定义前向传播函数，接受 `x` 和两个默认关键字参数 `key1` 和 `key2`
            def forward(self, x, key1=torch.rand(1), key2=torch.rand(1)):
                # 将输入 `x` 通过第一个线性层
                x = self.linear1(x)
                # 再将结果通过第二个线性层
                x = self.linear2(x)
                return x

        # 定义一个名为 `M` 的内部类，继承自 `torch.nn.Module`
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化两个线性层，输入和输出维度都是 5
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)
                # 初始化 `Sub` 类的实例
                self.sub = Sub()

            # 定义前向传播函数，接受 `x` 作为输入
            def forward(self, x):
                # 将输入 `x` 通过第一个线性层
                x = self.linear1(x)
                # 再将结果通过第二个线性层
                x = self.linear2(x)
                # 调用 `Sub` 类的前向传播函数，并仅覆盖 `key2`，`key1` 保留默认值
                x = self.sub(x, key2=torch.rand(1, 2))
                return x

        # 预期的全限定名到维度的映射字典
        expected_fqn_to_dim = {
            "": (2,),  # 主模块本身，输出维度为 2
            "linear1": (2,),  # 主模块的 `linear1` 层，输出维度为 2
            "linear2": (2,),  # 主模块的 `linear2` 层，输出维度为 2
            # `sub` 模块，包含三个输出维度：2（主输出），1（`key1` 使用默认参数），2（`key2` 被调用端覆盖）
            "sub": (2, 1, 2),
            "sub.linear1": (2,),  # `sub` 模块的 `linear1` 层，输出维度为 2
            "sub.linear2": (2,)  # `sub` 模块的 `linear2` 层，输出维度为 2
        }
        example_inputs = (torch.rand(1, 5),)  # 示例输入数据，1 行 5 列的张量
        # 调用测试函数 `_test_get_fqn_to_example_inputs`，验证模型 `M` 的输出与预期的全限定名到维度映射是否一致
        self._test_get_fqn_to_example_inputs(M, example_inputs, expected_fqn_to_dim)

    def test_get_fqn_to_example_inputs_complex_args(self):
        """ Test that we can record complex example inputs such as lists and dicts
        """
        # 定义一个名为 `Sub` 的内部类，继承自 `torch.nn.Module`
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化两个线性层，输入和输出维度都是 5
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)

            # 定义前向传播函数，接受 `x`、列表类型参数 `list_arg` 和字典类型参数 `dict_arg`
            def forward(self, x, list_arg, dict_arg):
                # 将输入 `x` 通过第一个线性层
                x = self.linear1(x)
                # 再将结果通过第二个线性层
                x = self.linear2(x)
                return x

        # 定义一个名为 `M` 的内部类，继承自 `torch.nn.Module`
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化两个线性层，输入和输出维度都是 5
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)
                # 初始化 `Sub` 类的实例
                self.sub = Sub()

            # 定义前向传播函数，接受 `x` 作为输入
            def forward(self, x):
                # 将输入 `x` 通过第一个线性层
                x = self.linear1(x)
                # 再将结果通过第二个线性层
                x = self.linear2(x)
                # 调用 `Sub` 类的前向传播函数，并传递列表 `[x]` 和字典 `{"3": x}` 作为参数
                x = self.sub(x, [x], {"3": x})
                return x

        example_inputs = (torch.rand(1, 5),)  # 示例输入数据，1 行 5 列的张量
        m = M().eval()  # 创建 `M` 类的实例，并设置为评估模式
        # 调用 `get_fqn_to_example_inputs` 函数，记录模型 `m` 的全限定名到示例输入映射
        fqn_to_example_inputs = get_fqn_to_example_inputs(m, example_inputs)
        # 断言检查，确保 `sub` 模块在映射字典中，且其第二个输出为列表类型，第三个输出为字典类型并包含键值 `"3"`
        assert "sub" in fqn_to_example_inputs
        assert isinstance(fqn_to_example_inputs["sub"][1], list)
        assert isinstance(fqn_to_example_inputs["sub"][2], dict) and \
            "3" in fqn_to_example_inputs["sub"][2]
    def test_quantize_weight_clamping_per_tensor(self):
        """ Test quant_{min, max} from per tensor observer is honored by `_quantize_weight` method
        """
        # 定义浮点数的最小值和最大值
        fp_min, fp_max = -1000.0, 1000.0
        # 定义量化后的最小值和最大值
        q8_min, q8_max = -10, 10

        # 创建包含浮点数的张量
        float_tensor = torch.tensor([fp_min, fp_max])

        # 创建移动平均最小-最大值观察器对象
        observer = MovingAverageMinMaxObserver(
            averaging_constant=1.0,
            dtype=torch.qint8,
            quant_min=q8_min,
            quant_max=q8_max,
            qscheme=torch.per_tensor_symmetric,
        )

        # 对浮点数张量进行观察
        observer(float_tensor)
        # 断言观察器对象记录的最小值和最大值与预期一致
        assert observer.min_val == fp_min
        assert observer.max_val == fp_max

        # 使用 `_quantize_weight` 方法量化浮点数张量
        quantized_tensor = _quantize_weight(float_tensor, observer)
        # 断言量化后张量的最大整数表示值等于预期的量化最大值
        assert quantized_tensor.int_repr().max().item() == q8_max
        # 断言量化后张量的最小整数表示值等于预期的量化最小值
        assert quantized_tensor.int_repr().min().item() == q8_min

        # 浮点数张量的实际权重值可能超出移动平均观察器记录的 [min_val, max_val] 范围
        float_tensor *= 1.2

        # 重新使用 `_quantize_weight` 方法量化浮点数张量
        quantized_tensor = _quantize_weight(float_tensor, observer)
        # 断言量化后张量的最大整数表示值等于预期的量化最大值
        assert quantized_tensor.int_repr().max().item() == q8_max
        # 断言量化后张量的最小整数表示值等于预期的量化最小值
        assert quantized_tensor.int_repr().min().item() == q8_min

    def test_quantize_weight_clamping_per_channel(self):
        """ Test quant_{min, max} from per channel observer is honored by `_quantize_weight` method
        """
        # 定义浮点数的最小值和最大值
        fp_min, fp_max = -1000.0, 1000.0
        # 定义量化后的最小值和最大值
        q8_min, q8_max = -10, 10

        # 创建包含浮点数的二维张量
        float_tensor = torch.tensor([[fp_min, fp_max]])

        # 创建移动平均每通道最小-最大值观察器对象
        observer = MovingAveragePerChannelMinMaxObserver(
            averaging_constant=1.0,
            dtype=torch.qint8,
            quant_min=q8_min,
            quant_max=q8_max,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )

        # 对浮点数张量进行观察
        observer(float_tensor)
        # 断言观察器对象记录的最小值和最大值与预期一致
        assert observer.min_val == fp_min
        assert observer.max_val == fp_max

        # 使用 `_quantize_weight` 方法量化浮点数张量
        quantized_tensor = _quantize_weight(float_tensor, observer)
        # 断言量化后张量的最大整数表示值等于预期的量化最大值
        assert quantized_tensor.int_repr().max().item() == q8_max
        # 断言量化后张量的最小整数表示值等于预期的量化最小值
        assert quantized_tensor.int_repr().min().item() == q8_min

        # 浮点数张量的实际权重值可能超出移动平均观察器记录的 [min_val, max_val] 范围
        float_tensor *= 1.2

        # 重新使用 `_quantize_weight` 方法量化浮点数张量
        quantized_tensor = _quantize_weight(float_tensor, observer)
        # 断言量化后张量的最大整数表示值等于预期的量化最大值
        assert quantized_tensor.int_repr().max().item() == q8_max
        # 断言量化后张量的最小整数表示值等于预期的量化最小值
        assert quantized_tensor.int_repr().min().item() == q8_min
    # 定义测试函数 test_uint1_7_dtype，用于测试特定数据类型的功能

        # 定义内部函数 up_size，用于将输入大小加倍，返回新的大小元组
        def up_size(size):
            return (*size[:-1], size[-1] * 2)

        # 定义一个继承自 torch.Tensor 的类 UInt4Tensor
        class UInt4Tensor(torch.Tensor):
            # 静态方法 __new__，用于创建新的 UInt4Tensor 实例
            @staticmethod
            def __new__(cls, elem, **kwargs):
                # 断言输入的 elem 的数据类型为 torch.uint8
                assert elem.dtype is torch.uint8
                # 确保不传递 "requires_grad" 参数或者其值为 False
                assert not kwargs.get("requires_grad", False)
                kwargs["requires_grad"] = False
                # 调用父类方法 _make_wrapper_subclass 创建一个新的 Tensor 子类实例
                return torch.Tensor._make_wrapper_subclass(cls, up_size(elem.shape), dtype=torch.uint4, **kwargs)

            # 初始化方法，接收 elem 作为参数
            def __init__(self, elem):
                self.elem = elem

            # 类方法 __torch_dispatch__，用于处理 Torch 框架的分发
            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs=None):
                pass

        # 确保代码正常运行
        # 创建一个 UInt4Tensor 实例 x，传入一个 uint8 类型的 Tensor 数据
        x = UInt4Tensor(torch.tensor([
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
        ], dtype=torch.uint8))
        # 断言实例 x 的数据类型为 torch.uint4
        assert x.dtype == torch.uint4
```