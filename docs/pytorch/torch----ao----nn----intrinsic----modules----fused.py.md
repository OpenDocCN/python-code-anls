# `.\pytorch\torch\ao\nn\intrinsic\modules\fused.py`

```py
# 引入 torch 库，包括所需的模块和函数
import torch
# 从 torch.nn 模块中导入 Conv1d、Conv2d、Conv3d、ReLU、Linear、BatchNorm1d、BatchNorm2d、BatchNorm3d 等模块
from torch.nn import Conv1d, Conv2d, Conv3d, ReLU, Linear, BatchNorm1d, BatchNorm2d, BatchNorm3d
# 从 torch.nn.utils.parametrize 模块导入 type_before_parametrizations 函数
from torch.nn.utils.parametrize import type_before_parametrizations

# 定义一个公共列表，包含所有在本模块中定义的类名，方便导出时进行统一管理
__all__ = ['ConvReLU1d', 'ConvReLU2d', 'ConvReLU3d', 'LinearReLU', 'ConvBn1d', 'ConvBn2d',
           'ConvBnReLU1d', 'ConvBnReLU2d', 'ConvBn3d', 'ConvBnReLU3d', 'BNReLU2d', 'BNReLU3d',
           'LinearBn1d', 'LinearLeakyReLU', 'LinearTanh', 'ConvAdd2d', 'ConvAddReLU2d']

# 用于识别在量化过程中使用的内置模块
class _FusedModule(torch.nn.Sequential):
    pass

# 定义一个将 Conv1d 和 ReLU 模块融合的类
class ConvReLU1d(_FusedModule):
    r"""This is a sequential container which calls the Conv1d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, relu):
        # 断言确保输入的模块类型正确
        assert type_before_parametrizations(conv) == Conv1d and type_before_parametrizations(relu) == ReLU, \
            f'Incorrect types for input modules{type_before_parametrizations(conv)}{type_before_parametrizations(relu)}'
        # 调用父类构造函数，将 Conv1d 和 ReLU 模块组成序列容器
        super().__init__(conv, relu)

# 定义一个将 Conv2d 和 ReLU 模块融合的类
class ConvReLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, relu):
        # 断言确保输入的模块类型正确
        assert type_before_parametrizations(conv) == Conv2d and type_before_parametrizations(relu) == ReLU, \
            f'Incorrect types for input modules{type_before_parametrizations(conv)}{type_before_parametrizations(relu)}'
        # 调用父类构造函数，将 Conv2d 和 ReLU 模块组成序列容器
        super().__init__(conv, relu)

# 定义一个将 Conv3d 和 ReLU 模块融合的类
class ConvReLU3d(_FusedModule):
    r"""This is a sequential container which calls the Conv3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, relu):
        # 断言确保输入的模块类型正确
        assert type_before_parametrizations(conv) == Conv3d and type_before_parametrizations(relu) == ReLU, \
            f'Incorrect types for input modules{type_before_parametrizations(conv)}{type_before_parametrizations(relu)}'
        # 调用父类构造函数，将 Conv3d 和 ReLU 模块组成序列容器
        super().__init__(conv, relu)

# 定义一个将 Linear 和 ReLU 模块融合的类
class LinearReLU(_FusedModule):
    r"""This is a sequential container which calls the Linear and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, linear, relu):
        # 断言确保输入的模块类型正确
        assert type_before_parametrizations(linear) == Linear and type_before_parametrizations(relu) == ReLU, \
            f'Incorrect types for input modules{type_before_parametrizations(linear)}{type_before_parametrizations(relu)}'
        # 调用父类构造函数，将 Linear 和 ReLU 模块组成序列容器
        super().__init__(linear, relu)

# 定义一个将 Conv1d 和 BatchNorm1d 模块融合的类
class ConvBn1d(_FusedModule):
    r"""This is a sequential container which calls the Conv 1d and Batch Norm 1d modules.
    During quantization this will be replaced with the corresponding fused module."""
    # 初始化方法，用于初始化一个对象
    def __init__(self, conv, bn):
        # 断言输入的 conv 参数类型为 Conv1d 类型，bn 参数类型为 BatchNorm1d 类型
        assert type_before_parametrizations(conv) == Conv1d and type_before_parametrizations(bn) == BatchNorm1d, \
            # 如果类型不符合预期，抛出带有详细错误信息的异常
            f'Incorrect types for input modules{type_before_parametrizations(conv)}{type_before_parametrizations(bn)}'
        # 调用父类的初始化方法
        super().__init__(conv, bn)
class ConvBn2d(_FusedModule):
    r"""This is a sequential container which calls the Conv 2d and Batch Norm 2d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn):
        assert type_before_parametrizations(conv) == Conv2d and type_before_parametrizations(bn) == BatchNorm2d, \
            f'Incorrect types for input modules{type_before_parametrizations(conv)}{type_before_parametrizations(bn)}'
        super().__init__(conv, bn)

class ConvBnReLU1d(_FusedModule):
    r"""This is a sequential container which calls the Conv 1d, Batch Norm 1d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn, relu):
        assert type_before_parametrizations(conv) == Conv1d and type_before_parametrizations(bn) == BatchNorm1d and \
            type_before_parametrizations(relu) == ReLU, f'Incorrect types for input modules{type_before_parametrizations(conv)}{type_before_parametrizations(bn)}{type_before_parametrizations(relu)}'  # noqa: B950
        super().__init__(conv, bn, relu)

class ConvBnReLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv 2d, Batch Norm 2d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn, relu):
        assert type_before_parametrizations(conv) == Conv2d and type_before_parametrizations(bn) == BatchNorm2d and \
            type_before_parametrizations(relu) == ReLU, f'Incorrect types for input modules{type_before_parametrizations(conv)}{type_before_parametrizations(bn)}{type_before_parametrizations(relu)}'  # noqa: B950
        super().__init__(conv, bn, relu)

class ConvBn3d(_FusedModule):
    r"""This is a sequential container which calls the Conv 3d and Batch Norm 3d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn):
        assert type_before_parametrizations(conv) == Conv3d and type_before_parametrizations(bn) == BatchNorm3d, \
            f'Incorrect types for input modules{type_before_parametrizations(conv)}{type_before_parametrizations(bn)}'
        super().__init__(conv, bn)

class ConvBnReLU3d(_FusedModule):
    r"""This is a sequential container which calls the Conv 3d, Batch Norm 3d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn, relu):
        assert type_before_parametrizations(conv) == Conv3d and type_before_parametrizations(bn) == BatchNorm3d and \
            type_before_parametrizations(relu) == ReLU, f'Incorrect types for input modules{type_before_parametrizations(conv)}{type_before_parametrizations(bn)}{type_before_parametrizations(relu)}'  # noqa: B950
        super().__init__(conv, bn, relu)



class BNReLU2d(_FusedModule):



# 这是一个调用 Conv 2d 和 Batch Norm 2d 模块的顺序容器。
# 在量化过程中，将使用相应的融合模块替换。
class ConvBn2d(_FusedModule):
    def __init__(self, conv, bn):
        # 确保输入模块的类型正确
        assert type_before_parametrizations(conv) == Conv2d and type_before_parametrizations(bn) == BatchNorm2d, \
            f'Incorrect types for input modules{type_before_parametrizations(conv)}{type_before_parametrizations(bn)}'
        super().__init__(conv, bn)

# 这是一个调用 Conv 1d、Batch Norm 1d 和 ReLU 模块的顺序容器。
# 在量化过程中，将使用相应的融合模块替换。
class ConvBnReLU1d(_FusedModule):
    def __init__(self, conv, bn, relu):
        # 确保输入模块的类型正确
        assert type_before_parametrizations(conv) == Conv1d and type_before_parametrizations(bn) == BatchNorm1d and \
            type_before_parametrizations(relu) == ReLU, f'Incorrect types for input modules{type_before_parametrizations(conv)}{type_before_parametrizations(bn)}{type_before_parametrizations(relu)}'  # noqa: B950
        super().__init__(conv, bn, relu)

# 这是一个调用 Conv 2d、Batch Norm 2d 和 ReLU 模块的顺序容器。
# 在量化过程中，将使用相应的融合模块替换。
class ConvBnReLU2d(_FusedModule):
    def __init__(self, conv, bn, relu):
        # 确保输入模块的类型正确
        assert type_before_parametrizations(conv) == Conv2d and type_before_parametrizations(bn) == BatchNorm2d and \
            type_before_parametrizations(relu) == ReLU, f'Incorrect types for input modules{type_before_parametrizations(conv)}{type_before_parametrizations(bn)}{type_before_parametrizations(relu)}'  # noqa: B950
        super().__init__(conv, bn, relu)

# 这是一个调用 Conv 3d 和 Batch Norm 3d 模块的顺序容器。
# 在量化过程中，将使用相应的融合模块替换。
class ConvBn3d(_FusedModule):
    def __init__(self, conv, bn):
        # 确保输入模块的类型正确
        assert type_before_parametrizations(conv) == Conv3d and type_before_parametrizations(bn) == BatchNorm3d, \
            f'Incorrect types for input modules{type_before_parametrizations(conv)}{type_before_parametrizations(bn)}'
        super().__init__(conv, bn)

# 这是一个调用 Conv 3d、Batch Norm 3d 和 ReLU 模块的顺序容器。
# 在量化过程中，将使用相应的融合模块替换。
class ConvBnReLU3d(_FusedModule):
    def __init__(self, conv, bn, relu):
        # 确保输入模块的类型正确
        assert type_before_parametrizations(conv) == Conv3d and type_before_parametrizations(bn) == BatchNorm3d and \
            type_before_parametrizations(relu) == ReLU, f'Incorrect types for input modules{type_before_parametrizations(conv)}{type_before_parametrizations(bn)}{type_before_parametrizations(relu)}'  # noqa: B950
        super().__init__(conv, bn, relu)



# 这是一个待实现的类定义，可能在后续代码中被完善和扩展。
class BNReLU2d(_FusedModule):
    # 定义一个容器类，顺序调用 BatchNorm2d 和 ReLU 模块
    # 在量化过程中，这将被替换为相应的融合模块
    def __init__(self, batch_norm, relu):
        # 确保输入的 batch_norm 和 relu 参数类型为 BatchNorm2d 和 ReLU
        assert type_before_parametrizations(batch_norm) == BatchNorm2d and type_before_parametrizations(relu) == ReLU, \
            f'Incorrect types for input modules{type_before_parametrizations(batch_norm)}{type_before_parametrizations(relu)}'
        # 调用父类的构造方法，将 batch_norm 和 relu 作为参数传入
        super().__init__(batch_norm, relu)
class BNReLU3d(_FusedModule):
    r"""This is a sequential container which calls the BatchNorm 3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, batch_norm, relu):
        # 断言确保传入的 batch_norm 和 relu 是 BatchNorm3d 和 ReLU 类型的实例
        assert type_before_parametrizations(batch_norm) == BatchNorm3d and type_before_parametrizations(relu) == ReLU, \
            f'Incorrect types for input modules{type_before_parametrizations(batch_norm)}{type_before_parametrizations(relu)}'
        # 调用父类 _FusedModule 的构造函数
        super().__init__(batch_norm, relu)


class LinearBn1d(_FusedModule):
    r"""This is a sequential container which calls the Linear and BatchNorm1d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, linear, bn):
        # 断言确保传入的 linear 和 bn 是 Linear 和 BatchNorm1d 类型的实例
        assert type_before_parametrizations(linear) == Linear and type_before_parametrizations(bn) == BatchNorm1d, \
            f'Incorrect types for input modules{type_before_parametrizations(linear)}{type_before_parametrizations(bn)}'
        # 调用父类 _FusedModule 的构造函数
        super().__init__(linear, bn)

class LinearLeakyReLU(_FusedModule):
    r"""This is a sequential container which calls the Linear and LeakyReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, linear, leaky_relu):
        # 断言确保传入的 linear 是 Linear 类型的实例，leaky_relu 是 torch.nn.LeakyReLU 类型的实例
        assert type(linear) == Linear and type(leaky_relu) == torch.nn.LeakyReLU, \
            f'Incorrect types for input modules{type(linear)}{type(leaky_relu)}'
        # 调用父类 _FusedModule 的构造函数
        super().__init__(linear, leaky_relu)

class LinearTanh(_FusedModule):
    r"""This is a sequential container which calls the Linear and Tanh modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, linear, tanh):
        # 断言确保传入的 linear 是 Linear 类型的实例，tanh 是 torch.nn.Tanh 类型的实例
        assert type(linear) == Linear and type(tanh) == torch.nn.Tanh, \
            f'Incorrect types for input modules{type(linear)}{type(tanh)}'
        # 调用父类 _FusedModule 的构造函数
        super().__init__(linear, tanh)

class ConvAdd2d(_FusedModule):
    r"""This is a sequential container which calls the Conv2d modules with extra Add.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, add):
        # 调用父类 _FusedModule 的构造函数，传入 conv 参数
        super().__init__(conv)
        # 将 add 参数保存到对象的属性中
        self.add = add

    def forward(self, x1, x2):
        # 调用 self[0]，即第一个子模块 conv 对象的 forward 方法，并与 x2 进行 add 操作后返回结果
        return self.add(self[0](x1), x2)

class ConvAddReLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv2d, add, Relu.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, add, relu):
        # 调用父类 _FusedModule 的构造函数，传入 conv 参数
        super().__init__(conv)
        # 将 add 和 relu 参数保存到对象的属性中
        self.add = add
        self.relu = relu

    def forward(self, x1, x2):
        # 调用 self[0]，即第一个子模块 conv 对象的 forward 方法，并与 x2 进行 add 操作后，再经过 relu 激活函数后返回结果
        return self.relu(self.add(self[0](x1), x2))
```