# `.\pytorch\test\dynamo\test_modules.py`

```
# Owner(s): ["module: dynamo"]

# 导入必要的模块和库
import collections  # 导入 collections 模块，用于特定的数据结构和操作
import copy  # 导入 copy 模块，用于对象的深复制操作
import itertools  # 导入 itertools 模块，用于生成迭代器的函数
import os  # 导入 os 模块，提供与操作系统相关的功能
import tempfile  # 导入 tempfile 模块，用于创建临时文件和目录
import traceback  # 导入 traceback 模块，用于提取和格式化异常的回溯信息
import types  # 导入 types 模块，用于动态创建和操作 Python 类型
import unittest  # 导入 unittest 模块，用于编写和运行单元测试
from copy import deepcopy  # 导入 deepcopy 函数，用于深度复制对象
from functools import partial  # 导入 partial 函数，用于部分应用函数
from typing import Dict, NamedTuple, Tuple  # 导入类型提示，用于静态类型检查
from unittest.mock import patch  # 导入 patch 函数，用于模拟对象和函数

import torch  # 导入 PyTorch 深度学习库

import torch._dynamo.test_case  # 导入 PyTorch 内部测试相关模块
import torch._dynamo.testing  # 导入 PyTorch 内部测试相关模块
import torch.nn.functional as F  # 导入 PyTorch 中的函数式接口模块
from torch._dynamo.debug_utils import same_two_models  # 导入 PyTorch 内部调试工具相关函数
from torch._dynamo.eval_frame import unsupported  # 导入 PyTorch 内部评估框架相关函数
from torch._dynamo.mutation_guard import GenerationTracker  # 导入 PyTorch 内部变异跟踪相关类
from torch._dynamo.testing import expectedFailureDynamic, same  # 导入 PyTorch 内部测试相关函数和装饰器
from torch.nn.modules.lazy import LazyModuleMixin  # 导入 PyTorch 模块的延迟加载混合类
from torch.nn.parameter import Parameter, UninitializedParameter  # 导入 PyTorch 参数相关类

try:
    from . import test_functions  # 尝试导入当前目录下的 test_functions 模块
except ImportError:
    import test_functions  # 如果导入失败，则导入全局的 test_functions 模块

# 全局变量定义
_variable = 0  # 定义一个全局变量 _variable，初始值为 0
_variable1 = 0  # 定义另一个全局变量 _variable1，初始值为 0


def update_global():
    global _variable, _variable1  # 声明在函数内部使用全局变量 _variable 和 _variable1
    _variable += 1  # 将 _variable 自增 1
    _variable1 += 1  # 将 _variable1 自增 1


class BasicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)  # 创建一个线性层，输入维度和输出维度均为 10
        self.scale = torch.randn(1, 10)  # 创建一个形状为 (1, 10) 的张量，包含从标准正态分布中抽取的随机数

    def forward(self, x):
        return F.relu(self.linear1(x)) * self.scale  # 执行前向传播，先通过线性层然后应用 ReLU 激活函数，最后乘以 scale 张量


class FnMember(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)  # 创建一个线性层，输入维度和输出维度均为 10
        self.activation = F.relu  # 设置一个激活函数作为成员变量

    def forward(self, x):
        x = self.linear1(x)  # 应用线性层
        if self.activation:  # 如果激活函数存在
            x = self.activation(x)  # 应用激活函数
        return x


class FnMemberCmp(torch.nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)  # 创建一个线性层，输入维度和输出维度均为 10
        self.activation = activation  # 设置激活函数作为成员变量

    def forward(self, x):
        x = self.linear1(x)  # 应用线性层
        if self.activation is not None:  # 如果激活函数不为空
            x = self.activation(x)  # 应用激活函数
        if self.activation is None:  # 如果激活函数为空
            x = torch.sigmoid(x)  # 应用 sigmoid 激活函数
        return x


class SubmoduleExample(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BasicModule()  # 创建 BasicModule 类的实例作为子模块
        self.layer2 = BasicModule()  # 创建另一个 BasicModule 类的实例作为子模块
        self.scale = torch.randn(1, 10)  # 创建一个形状为 (1, 10) 的张量，包含从标准正态分布中抽取的随机数

    def forward(self, x):
        x = self.layer1(x)  # 应用第一个子模块
        x = self.layer2(x)  # 应用第二个子模块
        return x * self.scale  # 返回结果乘以 scale 张量


class IsTrainingCheck(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)  # 创建一个线性层，输入维度和输出维度均为 10
        self.linear2 = torch.nn.Linear(10, 10)  # 创建另一个线性层，输入维度和输出维度均为 10
        self.train(True)  # 设置模型处于训练模式

    def forward(self, x):
        if self.training:  # 如果模型处于训练模式
            mod = self.linear1  # 使用第一个线性层
        else:
            mod = self.linear2  # 使用第二个线性层
        return F.relu(mod(x))  # 应用线性层后使用 ReLU 激活函数


class IsEvalCheck(IsTrainingCheck):
    def __init__(self):
        super().__init__()
        self.train(False)  # 设置模型处于评估模式


class ModuleMethodCall(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BasicModule()  # 创建 BasicModule 类的实例作为子模块
        self.layer2 = BasicModule()  # 创建另一个 BasicModule 类的实例作为子模块
        self.scale = torch.randn(1, 10)  # 创建一个形状为 (1, 10) 的张量，包含从标准正态分布中抽取的随机数

    def call_and_scale(self, mod, x):
        x = mod(x)  # 调用输入的模块并传入输入张量 x
        return x * self.scale  # 返回结果乘以 scale 张量
    # 定义神经网络前向传播函数，接受输入张量 x
    def forward(self, x):
        # 调用 self.layer1 并对结果进行调用和缩放处理，存储在 x1 中
        x1 = self.call_and_scale(self.layer1, x)
        # 调用 self.layer2 并对结果进行调用和缩放处理，存储在 x2 中
        x2 = self.call_and_scale(self.layer2, x)
        # 返回两个处理结果的和作为最终的输出
        return x1 + x2
class UnsupportedMethodCall(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个 BasicModule 实例作为该类的属性
        self.layer1 = BasicModule()
        # 初始化一个形状为 (1, 10) 的随机张量作为比例因子
        self.scale = torch.randn(1, 10)

    def call_and_scale(self, mod, x):
        # 调用传入的 mod 对象处理输入 x
        x = mod(x)
        # 将处理后的结果乘以预设的比例因子 self.scale
        x = x * self.scale
        # 调用 unsupported 函数两次，并返回结果
        return unsupported(x, x)

    def forward(self, x):
        # 调用 call_and_scale 方法处理输入 x，并将结果加到原始输入上返回
        x1 = self.call_and_scale(self.layer1, x)
        return x + x1


class UnsupportedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个 BasicModule 实例作为该类的属性
        self.layer1 = BasicModule()
        # 初始化一个形状为 (1, 10) 的随机张量作为比例因子
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        # 调用 layer1 处理输入 x，然后将结果与比例因子相乘返回
        x = self.layer1(x) * self.scale
        # 调用 unsupported 函数两次，并返回结果
        return unsupported(x, x)


class UnsupportedModuleCall(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个 UnsupportedModule 实例作为该类的属性
        self.mod = UnsupportedModule()

    def forward(self, x):
        # 将输入 x 乘以 1.5，然后传递给 mod 处理，并返回处理结果加 1
        return 1 + self.mod(x * 1.5)


class ModuleWithStaticForward(torch.nn.Module):
    @staticmethod
    def forward(x):
        # 对输入 x 执行 sigmoid 函数，并将结果与 x 相乘返回
        return x * torch.sigmoid(x)


class ModuleCallModuleWithStaticForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个 ModuleWithStaticForward 实例作为该类的属性
        self.mod = ModuleWithStaticForward()

    def forward(self, x):
        # 调用 mod 对象的 forward 静态方法处理输入 x，并返回结果
        return self.mod(x)


class ModuleStaticMethodCall(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建两个 BasicModule 实例作为该类的属性
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        # 初始化一个形状为 (1, 10) 的随机张量作为比例因子
        self.scale = torch.randn(1, 10)

    @staticmethod
    def call_and_scale(scale, mod, x):
        # 调用传入的 mod 对象处理输入 x，并将结果与比例因子 scale 相乘返回
        x = mod(x)
        return x * scale

    def forward(self, x):
        # 分别调用 call_and_scale 方法处理两个 BasicModule 对象的输入 x，并返回两者相加的结果
        x1 = self.call_and_scale(self.scale, self.layer1, x)
        x2 = self.call_and_scale(self.scale, self.layer2, x)
        return x1 + x2


class ModuleClassMethodCall(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建两个 BasicModule 实例作为该类的属性
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        # 初始化一个形状为 (1, 10) 的随机张量作为比例因子
        self.scale = torch.randn(1, 10)

    @classmethod
    def call_and_scale(cls, scale, mod, x):
        # 调用传入的 mod 对象处理输入 x，并将结果与比例因子 scale 相乘返回
        x = mod(x)
        return x * scale

    def forward(self, x):
        # 分别调用 call_and_scale 类方法处理两个 BasicModule 对象的输入 x，并返回两者相加的结果
        x1 = self.call_and_scale(self.scale, self.layer1, x)
        x2 = self.call_and_scale(self.scale, self.layer2, x)
        return x1 + x2


class ModuleProperty(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化一个形状为 (1, 10) 的随机张量作为比例因子
        self.scale = torch.randn(1, 10)

    @property
    def scale_alias(self):
        # 返回 scale 属性的别名 scale_alias
        return self.scale

    def forward(self, x):
        # 将输入 x 与 scale_alias 相乘返回
        return x * self.scale_alias


class NestedModuleList(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(3):
            # 将包含 Linear 层和 ReLU 激活函数的 ModuleList 添加到 layers 中
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        torch.nn.Linear(10, 10),
                        torch.nn.ReLU(),
                    ]
                )
            )

    def forward(self, x):
        # 依次对 layers 中的每个 ModuleList 中的层进行处理
        for layer, act in self.layers:
            x = act(layer(x))
        return x


class ConstLoop(torch.nn.Module):
    # 未完成的类定义，暂无需要添加的注释
    # 初始化方法，继承父类的初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，输入维度为10，输出维度为10
        self.linear1 = torch.nn.Linear(10, 10)
        # 设定一个计数器，值为3
        self.count = 3

    # 前向传播方法，接收输入张量 x
    def forward(self, x):
        # 循环 self.count 次，对输入张量 x 进行处理
        for i in range(self.count):
            # 将输入张量 x 经过 linear1 线性层和 sigmoid 激活函数处理
            x = torch.sigmoid(self.linear1(x))
        # 返回处理后的张量 x
        return x
# 定义一个继承自 torch.nn.Module 的模块 ViaModuleCall
class ViaModuleCall(torch.nn.Module):
    # 初始化函数，设置模块的线性层 linear1
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)

    # 前向传播函数，接收输入 x，经过线性层和 sigmoid 函数处理后传递给 constant3 函数
    def forward(self, x):
        return test_functions.constant3(torch.sigmoid(self.linear1(x)), x)


# 定义一个继承自 torch.nn.Module 的模块 IsNoneLayer
class IsNoneLayer(torch.nn.Module):
    # 初始化函数，设置两个线性层 layer1 和 layer2，并且设置为训练模式
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = None
        self.train(True)

    # 前向传播函数，根据 layer1 和 layer2 的状态决定是否使用它们对输入 x 进行处理
    def forward(self, x):
        if self.layer1 is not None:
            x = self.layer1(x)
        if self.layer2 is not None:
            x = self.layer2(x)
        return x


# 定义一个继承自 torch.nn.Module 的模块 LayerList
class LayerList(torch.nn.Module):
    # 初始化函数，创建一个包含线性层和激活函数的列表 layers
    def __init__(self):
        super().__init__()
        self.layers = [
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
        ]

    # 前向传播函数，依次对输入 x 应用列表中的每一层操作
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# 定义一个继承自 torch.nn.Module 的模块 ModuleList
class ModuleList(torch.nn.Module):
    # 初始化函数，创建一个包含线性层和激活函数的 ModuleList layers
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
            ]
        )

    # 前向传播函数，依次对输入 x 应用 ModuleList 中的每一层操作
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        for layer in self.layers:
            x = layer(x)

        for layer, val in zip(self.layers, (x, x, x, x)):
            x = layer(x) + val

        for layer, val in zip(self.layers, (1, 2, 3, 4)):
            x = layer(x) + val

        for idx, layer in enumerate(self.layers):
            x = layer(x) * idx

        for idx, layer in enumerate(self.layers[::-1]):
            x = layer(x) * idx

        return x


# 定义一个继承自 torch.nn.Module 的模块 CustomGetItemModuleList
class CustomGetItemModuleList(torch.nn.Module):
    # 初始化函数，创建一个包含线性层和激活函数的 ModuleList layers
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
            ]
        )

    # 定义 __getitem__ 方法，使得该模块可以通过下标访问 layers 中的元素
    def __getitem__(self, idx: int):
        return self.layers[idx]

    # 定义 __len__ 方法，返回 layers 的长度
    def __len__(self) -> int:
        return len(self.layers)

    # 前向传播函数，依次对输入 x 应用 layers 中的每一层操作
    def forward(self, x):
        for i in range(len(self)):
            x = self[i](x)

        return x


# 定义一个继承自 torch.nn.Module 的模块 ModuleDict
class ModuleDict(torch.nn.Module):
    # 初始化函数，创建一个包含键为 "0" 的线性层的 ModuleDict layers
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleDict(
            {
                "0": torch.nn.Linear(10, 10),
            }
        )

    # 前向传播函数，使用键为 "0" 的线性层处理输入 x
    def forward(self, x):
        # TODO(future PR): handle more logic
        x = self.layers["0"](x)
        return x


# 定义一个继承自 torch.nn.Module 的模块 ParameterDict
class ParameterDict(torch.nn.Module):
    # 初始化函数，创建一个包含键为 "0" 的参数张量的 ParameterDict layers
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ParameterDict(
            {
                "0": torch.nn.Parameter(torch.randn(10, 10)),
            }
        )
    # 定义一个前向传播方法，接受输入参数 x
    def forward(self, x):
        # 访问类属性 layers 中键为 "0" 的层，并对输入 x 执行矩阵乘法
        x = self.layers["0"].mm(x)
        # 返回执行矩阵乘法后的结果 x
        return x
class CustomGetItemParameterDict(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化一个包含单个参数的 ParameterDict，键为字符串 "0"，值为大小为 (10, 10) 的随机张量
        self.layers = torch.nn.ParameterDict(
            {
                "0": torch.nn.Parameter(torch.randn(10, 10)),
            }
        )

    def __getitem__(self, key: str) -> torch.nn.Module:
        # 通过键 key 获取对应的模块（这里是参数化的张量）
        return self.layers[key]

    def forward(self, x):
        # 使用索引运算符获取键为 "0" 的模块，并进行矩阵乘法操作
        x = self["0"].mm(x)
        return x


class CustomGetItemModuleDict(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化一个包含单个模块的 ModuleDict，键为字符串 "0"，值为 Linear 模块
        self.layers = torch.nn.ModuleDict(
            {
                "0": torch.nn.Linear(10, 10),
            }
        )

    def __getitem__(self, key: str) -> torch.nn.Module:
        # 通过键 key 获取对应的模块
        return self.layers[key]

    def forward(self, x):
        # 使用索引运算符获取键为 "0" 的模块，并将输入 x 传递给它
        x = self["0"](x)
        return x


class TensorList(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化一个包含多个张量的元组
        self.layers = (
            torch.randn((1, 10)),
            torch.randn((10, 1)),
            torch.randn((1, 10)),
            torch.randn((10, 1)),
        )

    def forward(self, x):
        # 遍历每个张量，并将输入 x 与每个张量逐元素相乘
        for layer in self.layers:
            x = x * layer
        return x


class Children(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化包含多个子模块的实例变量
        self.l1 = torch.nn.Linear(10, 10)
        self.l2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(10, 10)
        self.l4 = torch.nn.ReLU()

    def forward(self, x):
        # 遍历每个子模块，并依次将输入 x 传递给它们
        for block in self.children():
            x = block(x)
        return x


class NamedChildren(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化包含多个子模块的实例变量，每个模块都有命名
        self.l1 = torch.nn.Linear(10, 10)
        self.l2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(10, 10)
        self.l4 = torch.nn.ReLU()

    def forward(self, x):
        # 遍历每个子模块的名称和实例，并依次将输入 x 传递给它们
        for _, block in self.named_children():
            x = block(x)
        return x


class IntArg(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化一个包含单个线性层的实例变量
        self.layer1 = torch.nn.Linear(10, 10)

    def forward(self, x, offset=1):
        # 对输入 x 执行线性层、ReLU 激活和偏移操作，并返回结果
        x = F.relu(self.layer1(x)) + offset
        return x


class Seq(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化一个包含多个顺序连接的模块的 Sequential 容器
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        # 将输入 x 依次传递给 Sequential 容器内的每个模块，并返回最终结果
        return self.layers(x)


class Cfg:
    def __init__(self):
        # 初始化包含两个属性的简单配置类
        self.val = 0.5
        self.count = 3


class CfgModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化一个包含简单配置对象和单个线性层的实例变量
        self.cfg = Cfg()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        # 根据配置对象中的计数值，多次对输入 x 执行线性层和偏移操作，并返回结果
        for i in range(self.cfg.count):
            x = self.layer(x + self.cfg.val)
        return x


class StringMember(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化包含单个线性层和字符串属性的实例变量
        self.linear1 = torch.nn.Linear(10, 10)
        self.mode = "some_string"
    # 定义一个前向传播方法，接受输入参数 x
    def forward(self, x):
        # 检查模式是否为 "some_string"，如果是则执行以下操作
        if self.mode == "some_string":
            # 对输入 x 执行线性变换并应用 ReLU 激活函数，返回结果
            return F.relu(self.linear1(x))
class _Block(torch.nn.Module):
    # 定义一个简单的神经网络模块 `_Block`，用于处理输入数据
    def forward(self, x):
        # 返回输入数据 `x` 的元素连接结果的 1.5 倍
        return 1.5 * torch.cat(x, 1)


class _DenseBlock(torch.nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        # 根据传入的 `num_layers` 参数，动态添加指定数量的 `_Block` 模块作为子模块
        for i in range(num_layers):
            self.add_module("denselayer%d" % (i + 1), _Block())

    def forward(self, init_features):
        features = [init_features]
        # 遍历 `_DenseBlock` 的子模块，对输入的特征进行处理
        for layer in self.values():
            new_features = layer(features)
            features.append(new_features)
        # 返回所有处理后的特征连接结果
        return torch.cat(features, 1)


class DenseNetBlocks(torch.nn.Module):
    # 定义一个包含 `_DenseBlock` 模块的神经网络模块 `DenseNetBlocks`
    def __init__(self):
        super().__init__()
        self.layers = _DenseBlock()

    def forward(self, x):
        # 将输入数据 `x` 经过 `_DenseBlock` 处理后返回结果
        return self.layers(x)


class MaterializedModule(torch.nn.Module):
    """Once the below lazy module is initialized with its first input,
    it is transformed into this module."""

    param: Parameter

    def __init__(self):
        super().__init__()
        # 注册一个参数 `param`，初始为 `None`
        self.register_parameter("param", None)

    def forward(self, x):
        # 直接返回输入 `x`，无处理
        return x


class LazyModule(LazyModuleMixin, MaterializedModule):
    param: UninitializedParameter
    cls_to_become = MaterializedModule

    def __init__(self):
        super().__init__()
        # 初始化一个未初始化的参数 `param`
        self.param = UninitializedParameter()

    def initialize_parameters(self, x):
        # 强制图断裂，确保不会内联化此处的操作
        torch._dynamo.graph_break()
        # 根据输入 `x` 的形状初始化参数 `param`
        self.param.materialize(x.shape)


class LazyMLP(torch.nn.Module):
    # 定义一个简单的多层感知机 `LazyMLP`，包含懒加载的线性层和激活函数
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.LazyLinear(10)  # 懒加载的线性层，输入维度为 10
        self.relu1 = torch.nn.ReLU()       # ReLU 激活函数
        self.fc2 = torch.nn.LazyLinear(1)  # 懒加载的线性层，输出维度为 1
        self.relu2 = torch.nn.ReLU()       # ReLU 激活函数

    def forward(self, input):
        # 前向传播函数，依次经过线性层和激活函数处理输入 `input`
        x = self.relu1(self.fc1(input))
        y = self.relu2(self.fc2(x))
        return y


class MyInput(NamedTuple):
    x: Dict[str, Dict[str, torch.Tensor]]
    y: torch.Tensor


class LazyLayerWithNamedTupleInput(LazyModuleMixin, torch.nn.Module):
    # 定义一个接收 `NamedTuple` 类型输入的懒加载模块 `LazyLayerWithNamedTupleInput`
    def __init__(self):
        super().__init__()

    def initialize_parameters(self, input):
        # 使用 `input.x["a"][0]` 的形状创建并初始化参数 `_param`
        with torch.no_grad():
            self._param = torch.nn.Parameter(
                torch.empty(input.x["a"][0].shape).fill_(0.5)
            )

    def forward(self, input):
        # 对输入 `input.x["a"]` 进行求和处理，返回结果
        input = input.x["a"]
        x = 0
        for i in range(len(input)):
            x = x + input[i]
        return x


class LazyModuleWithNamedTupleInput(torch.nn.Module):
    # 定义一个包含 `LazyLayerWithNamedTupleInput` 的神经网络模块 `LazyModuleWithNamedTupleInput`
    def __init__(self):
        super().__init__()
        self.layer = LazyLayerWithNamedTupleInput()

    def forward(self, input):
        # 将输入 `input` 经过 `LazyLayerWithNamedTupleInput` 处理后返回结果
        return self.layer(input)


class LazyLayerWithListInput(LazyModuleMixin, torch.nn.Module):
    # 定义一个接收列表类型输入的懒加载模块 `LazyLayerWithListInput`
    def __init__(self):
        super().__init__()

    def initialize_parameters(self, input):
        # 使用 `input[0]` 的形状创建并初始化参数 `_param`
        with torch.no_grad():
            self._param = torch.nn.Parameter(torch.empty(input[0].shape).fill_(0.5))
    # 定义一个类方法 forward，用于计算输入列表 input 中所有元素的总和并返回结果
    def forward(self, input):
        # 初始化变量 x，用于存储累加和，初始值为 0
        x = 0
        # 遍历输入列表 input 的所有元素
        for i in range(len(input)):
            # 将 x 更新为当前值加上 input 列表中第 i 个元素的值
            x = x + input[i]
        # 返回累加和 x
        return x
class LazyModuleWithListInput(torch.nn.Module):
    # 延迟加载模块，接受列表输入
    def __init__(self):
        super().__init__()
        # 初始化时创建 LazyLayerWithListInput 实例作为成员变量
        self.layer = LazyLayerWithListInput()

    # 前向传播方法，接受输入并传递给内部的 LazyLayerWithListInput 实例处理
    def forward(self, input):
        return self.layer(input[:-1])


class LazyModuleWithLazySubmodule(LazyModuleMixin, torch.nn.Module):
    # 延迟加载模块，包含延迟加载子模块
    def __init__(self):
        super().__init__()

    # 初始化参数方法，使用 torch.no_grad() 上下文创建 LazyLayerWithListInput 实例
    def initialize_parameters(self, input):
        with torch.no_grad():
            self.layer = LazyLayerWithListInput()

    # 前向传播方法，传递输入到内部的 LazyLayerWithListInput 实例
    def forward(self, x):
        return self.layer(x)


class LazyLayerWithInputs(LazyModuleMixin, torch.nn.Module):
    # 延迟加载模块，接受多个输入
    def __init__(self):
        super().__init__()

    # 初始化参数方法，使用 torch.no_grad() 上下文创建两个参数：_param_x 和 _param_y
    def initialize_parameters(self, x, y):
        with torch.no_grad():
            self._param_x = torch.nn.Parameter(torch.empty(x[0].shape).fill_(0.5))
            self._param_y = torch.nn.Parameter(torch.empty(y[0].shape).fill_(0.5))

    # 前向传播方法，对输入 x 和 y 中的每个元素求和，返回求和结果的和
    def forward(self, x, y):
        res_x = 0
        for i in range(len(x)):
            res_x = res_x + x[i]
        res_y = 0
        for i in range(len(y)):
            res_y = res_y + y[i]
        return res_x + res_y


class LazyModuleKwArgs(LazyModuleMixin, torch.nn.Module):
    # 延迟加载模块，接受关键字参数输入
    def __init__(self):
        super().__init__()

    # 初始化参数方法，使用 torch.no_grad() 上下文创建 LazyLayerWithInputs 实例
    def initialize_parameters(self, *args, **kwargs):
        with torch.no_grad():
            self.layer = LazyLayerWithInputs()

    # 前向传播方法，将输入 x 和关键字参数 y 传递给内部的 LazyLayerWithInputs 实例
    def forward(self, x, y):
        return self.layer(x, y=y)


class LazyParentModule(LazyModuleMixin, torch.nn.Module):
    # 延迟加载模块，包含父类模块功能
    def __init__(self):
        super().__init__()

    # 实现方法，返回输入 x 的余弦和内部变量 _val 的和
    def impl(self, x):
        return x.cos() + self._val


class LazyChildModuleNoClsToBecome(LazyParentModule):
    # 延迟加载子模块，继承 LazyParentModule
    def __init__(self):
        super().__init__()

    # 前向传播方法，将输入 x 的正弦传递给父类的 impl 方法
    def forward(self, x):
        return super().impl(x.sin())

    # 初始化参数方法，创建一个形状为 (2, 2) 的 torch.nn.Parameter 类型的参数 _val，值为全 1
    def initialize_parameters(self, input):
        self._val = torch.nn.Parameter(torch.ones(2, 2))


def requires_grad1(module: torch.nn.Module, recurse: bool = False) -> bool:
    # 判断给定模块中是否有参数需要梯度计算
    requires_grad = any(p.requires_grad for p in module.parameters(recurse))
    return requires_grad


def requires_grad2(module: torch.nn.Module, recurse: bool = False) -> bool:
    # 判断给定模块中是否有参数需要梯度计算
    requires_grad = any(p.requires_grad for p in module.parameters(recurse))
    return requires_grad


class ParametersModule1(torch.nn.Module):
    # 参数模块1，包含一个线性层和一个可学习的 scale 参数
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.nn.Parameter(torch.randn(1, 10))

    # 前向传播方法，如果模块中没有需要梯度的参数，则返回线性层输出的 ReLU 结果乘以 scale 参数
    # 否则返回输入 x 加 1
    def forward(self, x):
        if not requires_grad1(self):
            return F.relu(self.linear1(x)) * self.scale
        else:
            return x + 1


class ParametersModule2(ParametersModule1):
    # 参数模块2，继承自参数模块1
    def forward(self, x):
        if not requires_grad2(self):
            return F.relu(self.linear1(x)) * self.scale
        else:
            return x + 1


class ParametersModule3(ParametersModule1):
    # 参数模块3，继承自参数模块1
    def forward(self, x):
        ones = torch.ones(10, dtype=next(self.parameters()).dtype)
        return F.relu(self.linear1(x)) * self.scale + ones
class ParametersModule4(ParametersModule1):
    # ParametersModule4 类继承自 ParametersModule1 类

    def forward(self, x):
        # 定义前向传播函数，接受输入 x

        # 创建一个全为 1 的张量，与模型参数的数据类型一致
        ones = torch.ones(10, dtype=next(self.parameters(recurse=False)).dtype)

        # 对输入 x 执行线性变换和 ReLU 激活，乘以 self.scale，并加上 ones
        return F.relu(self.linear1(x)) * self.scale + ones


class ParametersModule5(torch.nn.Module):
    # ParametersModule5 类继承自 torch.nn.Module 类

    def __init__(self):
        # 初始化函数
        super().__init__()

        # 定义一个线性层，输入维度为 10，输出维度为 10
        self.linear1 = torch.nn.Linear(10, 10)

        # 定义一个参数 scale，其形状为 (10, 10)，初始化为随机值
        self.scale = torch.nn.Parameter(torch.randn(10, 10))

        # 将 self.scale 的引用赋给 self.scale_dup
        self.scale_dup = self.scale

    def forward(self, x):
        # 定义前向传播函数，接受输入 x

        # 初始化计数器 counter 为 0
        counter = 0

        # 遍历模型的所有参数，每遍历一个参数，计数器加 1
        for param in self.parameters():
            counter += 1

        # 返回 x 乘以 self.scale 和 counter 的乘积
        return x * self.scale * counter


class SuperModule(BasicModule):
    # SuperModule 类继承自 BasicModule 类

    def forward(self, x):
        # 定义前向传播函数，接受输入 x

        # 调用父类 BasicModule 的 forward 方法，并将结果与 10.0 相加
        x = super().forward(x)
        return x + 10.0


class SuperModule2(BasicModule):
    # SuperModule2 类继承自 BasicModule 类

    def forward(self, x):
        # 定义前向传播函数，接受输入 x

        # 显式调用 BasicModule 类的 forward 方法
        return BasicModule.forward(self, x)


class ComplicatedSuperParent(torch.nn.Module):
    # ComplicatedSuperParent 类继承自 torch.nn.Module 类

    @classmethod
    def custom_add(cls, x):
        # 定义一个类方法 custom_add，接受参数 x

        # 将参数 x 加上自身，返回结果
        x = x + x
        return x


class SuperChildCallsClassMethod(ComplicatedSuperParent):
    # SuperChildCallsClassMethod 类继承自 ComplicatedSuperParent 类

    @classmethod
    def child_func(cls, x):
        # 定义一个类方法 child_func，接受参数 x

        # 调用父类 ComplicatedSuperParent 的类方法 custom_add 处理参数 x，并返回结果
        x = super().custom_add(x)
        return x

    def forward(self, x):
        # 定义前向传播函数，接受输入 x

        # 调用类方法 child_func 处理输入 x，并返回结果
        x = self.child_func(x)
        return x


class HasAttrModule(torch.nn.Module):
    # HasAttrModule 类继承自 torch.nn.Module 类

    def __init__(self):
        # 初始化函数
        super().__init__()

        # 定义一个参数 scale，其形状为 (1, 10)，初始化为随机值
        self.scale = torch.nn.Parameter(torch.randn(1, 10))

    def forward(self, x):
        # 定义前向传播函数，接受输入 x

        # 对输入 x 执行 ReLU 激活
        x = F.relu(x)

        # 如果当前对象具有属性 "scale"，则将输入 x 乘以 self.scale
        if hasattr(self, "scale"):
            x *= self.scale

        # 如果当前对象具有属性 "scale2"，则将输入 x 乘以 self.scale2
        if hasattr(self, "scale2"):
            x *= self.scale2

        # 返回处理后的输入 x
        return x


class EnumValues(torch.nn.ModuleDict):
    # EnumValues 类继承自 torch.nn.ModuleDict 类

    def __init__(
        self,
        num_layers: int = 3,
    ) -> None:
        # 初始化函数
        super().__init__()

        # 根据给定的 num_layers 参数，循环创建 _Block 实例，并将其作为模块添加到当前 ModuleDict 中
        for i in range(num_layers):
            self.add_module("denselayer%d" % (i + 1), _Block())

    def forward(self, init_features):
        # 定义前向传播函数，接受输入 init_features

        # 初始化 features 列表，将初始特征 init_features 添加到列表中
        features = [init_features]

        # 遍历 ModuleDict 中的每个子模块，依次对 features 执行前向传播
        for idx, layer in enumerate(self.values()):
            new_features = layer(features)
            features.append(new_features)

        # 将所有 features 拼接在一起并返回
        return torch.cat(features, 1)


class AccessByKeys(torch.nn.ModuleDict):
    # AccessByKeys 类继承自 torch.nn.ModuleDict 类

    def __init__(
        self,
        num_layers: int = 3,
    ) -> None:
        # 初始化函数
        super().__init__()

        # 根据给定的 num_layers 参数，循环创建 _Block 实例，并将其作为模块添加到当前 ModuleDict 中
        for i in range(num_layers):
            self.add_module("denselayer%d" % (i + 1), _Block())

    def forward(self, init_features):
        # 定义前向传播函数，接受输入 init_features

        # 初始化 features 列表，将初始特征 init_features 添加到列表中
        features = [init_features]

        # 遍历 ModuleDict 中的每个键，通过键访问对应的子模块，并对 features 执行前向传播
        for k in self.keys():
            new_features = self[k](features)
            features.append(new_features)

        # 将所有 features 拼接在一起并返回
        return torch.cat(features, 1)


class CallForwardDirectly(torch.nn.Module):
    # CallForwardDirectly 类继承自 torch.nn.Module 类

    def __init__(self):
        # 初始化函数
        super().__init__()

        # 定义一个 BasicModule 的实例作为 layer1
        self.layer1 = BasicModule()

        # 定义一个线性层，输入维度为 10，输出维度为 10，作为 layer2
        self.layer2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        # 定义前向传播函数，接受输入 x

        # 调用 layer1 的 forward 方法对输入 x 执行前向传播
        x = self.layer1.forward(x)

        # 调用 layer2 的 forward 方法对输入 x 执行前向传播
        x = self.layer2.forward(x)

        # 返回处理后的结果
        return x


class ConvCallForwardDirectly(torch.nn.Module):
    # ConvCallForwardDirectly 类继承自 torch.nn.Module 类

    def __init__(self):
        # 初始化函数
        super().__init__()

        # 定义一个 2D 卷积层，输入通道数为 3，输出通道数为 64，卷积核大小为 3x3
        self.layer = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    # 定义一个方法 forward，用于执行神经网络中的前向传播操作
    def forward(self, x):
        # 调用 self.layer 对象的 forward 方法，将输入 x 传递给它并返回结果
        return self.layer.forward(x)
class ConvTransposeCallForwardDirectly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个转置卷积层，输入通道数为4，输出通道数为4，卷积核大小为4
        self.layer = torch.nn.ConvTranspose2d(4, 4, 4)

    def forward(self, x):
        # 直接调用转置卷积层的forward方法，传入输入张量x
        return self.layer.forward(x)


class ConvCallSuperForwardDirectly(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs,
        )

    def forward(self, inputs, mask=None):
        # 调用父类Conv1d的forward方法，传入inputs张量
        outputs = super().forward(inputs)
        return outputs


class ConvTransposeCallSuperForwardDirectly(torch.nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs,
        )

    def forward(self, x):
        if x.numel() > 0:
            # 如果输入张量x的元素数大于0，则调用父类ConvTranspose2d的forward方法，传入x
            return super().forward(x)
        # 计算输出形状的列表推导式
        output_shape = [
            ((i - 1) * d - 2 * p + (di * (k - 1) + 1) + op)
            for i, p, di, k, d, op in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
                self.output_padding,
            )
        ]
        # 设置输出形状的维度信息
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        # 调用_NewEmptyTensorOp的apply方法，传入x和输出形状，返回新的空张量
        return _NewEmptyTensorOp.apply(x, output_shape)  # noqa: F821


class ModuleNameString(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个线性层，输入特征维度为10，输出特征维度为10
        self.linear1 = torch.nn.Linear(10, 10)

    def forward(self, x):
        if self.__class__.__name__ == "ABC":
            # 如果当前类名为"ABC"，返回值为10
            return 10
        if self.linear1.__class__.__name__ == "Linear":
            # 如果self.linear1的类名为"Linear"，返回经过ReLU激活后的结果
            return F.relu(self.linear1(x) + 10)
        # 否则返回11
        return 11


class SelfMutatingModule(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        # 初始化时接受一个层对象并赋值给self.layer
        self.layer = layer
        self.counter = 0

    def forward(self, x):
        # 计算self.layer对输入x的输出并加上计数器值，然后进行ReLU激活并返回结果
        result = self.layer(x) + self.counter
        # 每次前向传播后更新计数器值
        self.counter += 1
        return F.relu(result)


class ModuleAttributePrecedenceBase(torch.nn.Module):
    def linear(self, x, flag=None):
        if flag:
            # 如果flag为真，则返回输入x乘以2.0
            return x * 2.0
        # 否则返回输入x乘以3.0
        return x * 3.0


class ModuleAttributePrecedence(ModuleAttributePrecedenceBase):
    def __init__(self):
        super().__init__()
        # 定义激活函数ReLU层、线性层、初始化矩阵为全1的矩阵、缩放比例0.5
        self.activation = torch.nn.ReLU()
        self.linear = torch.nn.Linear(10, 10)
        self.initializer = torch.ones([10, 10])
        self.scale = 0.5

    def activation(self, x):
        # 对输入x执行激活函数ReLU并返回结果
        return x * 1.2

    def initializer(self):
        # 返回一个全零的10x10张量作为初始化矩阵
        return torch.zeros([10, 10])

    def scale(self):
        # 返回缩放比例2.0
        return 2.0

    def forward(self, x):
        # object属性优先级高，除非其不是nn.Module
        return self.activation(self.linear(self.initializer + x)) * self.scale


class ModuleForwardHasGraphBreak(torch.nn.Module):
    # 在这里的代码块中没有添加具体实现或注释的指令
    # 定义一个类，继承自基础模块 torch.nn.Module
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个名为 layer1 的 BasicModule 实例作为属性
        self.layer1 = BasicModule()
        # 创建一个名为 layer2 的 BasicModule 实例作为属性
        self.layer2 = BasicModule()
        # 创建一个包含两个 BasicModule 实例的序列，并将其作为属性 layer3
        self.layer3 = torch.nn.Sequential(BasicModule(), BasicModule())
        # 创建一个包含多个子模块的模块列表作为属性 layer4
        self.layer4 = torch.nn.ModuleList(
            [
                torch.nn.Linear(10, 10),  # 添加一个线性层
                torch.nn.ReLU(),          # 添加一个 ReLU 激活函数
                torch.nn.Linear(10, 10),  # 再次添加一个线性层
                torch.nn.ReLU(),          # 再次添加一个 ReLU 激活函数
            ]
        )
        # 创建一个包含多个子模块的模块字典作为属性 layer5
        self.layer5 = torch.nn.ModuleDict(
            {
                "0": torch.nn.Linear(10, 10),  # 在字典中添加一个线性层
            }
        )
        # 创建一个形状为 (1, 10) 的随机张量作为属性 scale
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        """
        This is used to test if the results of functions like `named_parameters`
        can be reconstructed correctly after graph break.

        https://github.com/pytorch/torchdynamo/issues/1931
        """
        # 将输入 x 传递给 layer1 模块，并将结果保存在 x 中
        x = self.layer1(x)
        # 获取当前模块的所有命名参数，并转换成字典 params1
        params1 = dict(self.named_parameters())
        # 获取当前模块的所有参数，并转换成列表 params2
        params2 = list(self.parameters())
        # 获取当前模块的所有命名缓冲区，并转换成字典 buffers1
        buffers1 = dict(self.named_buffers())
        # 获取当前模块的所有缓冲区，并转换成列表 buffers2
        buffers2 = list(self.buffers())
        # 获取当前模块的所有命名子模块，并转换成字典 modules1
        modules1 = dict(self.named_modules())
        # 获取当前模块的所有子模块，并转换成列表 modules2
        modules2 = list(self.modules())
        # 断开计算图
        torch._dynamo.graph_break()
        # 将 modules2 赋值给 y
        y = modules2
        # 将 modules1 赋值给 y
        y = modules1
        # 将 buffers2 赋值给 y
        y = buffers2
        # 将 buffers1 赋值给 y
        y = buffers1
        # 将 params2 赋值给 y
        y = params2
        # 将 params1 赋值给 y
        y = params1
        # 将 x 传递给 layer2 模块，同时加上命名参数的某些权重作为偏置，最后乘以 scale
        x = (
            self.layer2(x)
            + y["layer3.1.linear1.weight"]  # 使用命名参数 "layer3.1.linear1.weight"
            + y["layer4.2.weight"]          # 使用命名参数 "layer4.2.weight"
            + y["layer5.0.weight"]          # 使用命名参数 "layer5.0.weight"
        )
        # 返回最终的输出结果 x 乘以 scale
        return x * self.scale
class ModuleGuardNameIsValid(torch.nn.ModuleDict):
    # Guard names should be valid python identifier as we use eval() to get
    # corresponding guard value. Some guard names come from source(module path)
    # where special symbols are valid. But they are not valid python identifier,
    # we should identify these pattern and rewrite them with getattr.
    def __init__(self):
        super().__init__()
        # Add two BasicModule instances to self as "l@yer-1" and "l@yer-2"
        for i in range(2):
            self.add_module("l@yer-%d" % (i + 1), BasicModule())

    def forward(self, x):
        # Iterate over values in self (ModuleDict) and apply each layer to x
        for layer in self.values():
            x = layer(x)
        return x


class SequentialWithDuplicatedModule(torch.nn.Module):
    # Sequential module(self.layer) contains three duplicated ReLU module.
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        # Define a sequential module with Linear and ReLU layers repeated
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            self.relu,
            torch.nn.Linear(20, 20),
            self.relu,
            torch.nn.Linear(20, 10),
            self.relu,
        )

    def forward(self, x):
        # Apply the sequential layers to input x
        return self.layer(x)


class SequentialWithDuplicatedModule2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        # Define a sequential module using OrderedDict with repeated Linear and ReLU layers
        self.layer = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("linear1", torch.nn.Linear(10, 20)),
                    ("relu1", self.relu),
                    ("linear2", torch.nn.Linear(20, 20)),
                    ("relu2", self.relu),
                    ("linear3", torch.nn.Linear(20, 10)),
                    ("relu3", self.relu),
                ]
            )
        )

    def forward(self, x):
        # Apply the sequential layers to input x
        return self.layer(x)


class ModuleComparison(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define three Linear layers as attributes
        self.layer0 = torch.nn.Linear(10, 10)
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)

    @property
    def encoder_layers(self):
        # Return a list of the three Linear layers
        return [self.layer0, self.layer1, self.layer2]

    def forward(self, x):
        # Apply each layer in encoder_layers to input x with different activation functions
        for layer in self.encoder_layers:
            output = layer(x)
            # Conditional activation functions based on the current layer
            if layer is None or layer == self.layer0:
                output = F.relu6(output)
            else:
                output = F.relu(output)
        return output


class ModulePatch1(torch.nn.Module):
    pass
    # Placeholder module with no implementation


class ModulePatch2(torch.nn.Module):
    def forward(self, x):
        # Subtract 1 from input x
        return x - 1


class UnspecNonInlinableModule(torch.nn.Module):
    torchdynamo_force_dynamic = True  # forced to be a UnspecializedNNModule

    def forward(self, x):
        # If sum of x is greater than 0, add 1; otherwise, subtract 1
        if x.sum() > 0:
            return x + 1
        else:
            return x - 1


class UnspecNonInlinableToplevelModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Instantiate UnspecNonInlinableModule as self.m
        self.m = UnspecNonInlinableModule()

    def forward(self, x):
        # Apply self.m to input x
        return self.m(x)


def make_test(fn, expected_ops=None):
    # Placeholder function definition for testing purposes
    # 定义一个测试函数 test_fn，该函数调用 torch._dynamo.testing.standard_test 进行标准测试
    def test_fn(self):
        return torch._dynamo.testing.standard_test(
            self, fn=fn, nargs=1, expected_ops=expected_ops
        )

    # 执行 fn 的 eval 方法，假设 fn 是一个 torch.Tensor 对象，这将设置 fn 为评估模式
    fn.eval()

    # 返回 test_fn 函数的引用，test_fn 函数用于执行标准测试
    return test_fn
# 定义一个测试类，继承自 torch._dynamo.test_case.TestCase
class NNModuleTests(torch._dynamo.test_case.TestCase):
    # 使用 make_test 函数创建测试对象，测试一个空序列的情况
    test_seq = make_test(Seq())
    # 使用 make_test 函数创建测试对象，测试 BasicModule 的功能
    test_basicmodule1 = make_test(BasicModule())
    # 使用 make_test 函数创建测试对象，再次测试 BasicModule 的功能
    test_basicmodule2 = make_test(BasicModule())
    # 使用 make_test 函数创建测试对象，测试 SubmoduleExample 的功能
    test_submodules1 = make_test(SubmoduleExample())
    # 使用 make_test 函数创建测试对象，再次测试 SubmoduleExample 的功能
    test_submodules2 = make_test(SubmoduleExample())
    # 使用 make_test 函数创建测试对象，测试 ModuleMethodCall 的方法调用
    test_modulemethod1 = make_test(ModuleMethodCall())
    # 使用 make_test 函数创建测试对象，再次测试 ModuleMethodCall 的方法调用
    test_modulemethod2 = make_test(ModuleMethodCall())
    # 使用 make_test 函数创建测试对象，测试 ModuleCallModuleWithStaticForward 的静态前向调用
    test_module_call_module_with_static_forward = make_test(
        ModuleCallModuleWithStaticForward()
    )
    # 使用 make_test 函数创建测试对象，测试 ModuleStaticMethodCall 的静态方法调用
    test_module_static_method = make_test(ModuleStaticMethodCall())
    # 使用 make_test 函数创建测试对象，测试 FnMember 的功能
    test_fnmember = make_test(FnMember())
    # 使用 make_test 函数创建测试对象，测试带有自定义激活函数的 FnMemberCmp 的功能
    test_fnmembercmp1 = make_test(FnMemberCmp(F.relu))
    # 使用 make_test 函数创建测试对象，测试 FnMemberCmp 没有自定义激活函数的功能
    test_fnmembercmp2 = make_test(FnMemberCmp(None))
    # 使用 make_test 函数创建测试对象，测试 ConstLoop 的功能
    test_constloop = make_test(ConstLoop())
    # 使用 make_test 函数创建测试对象，测试 IsTrainingCheck 的训练状态检查
    test_istraining1 = make_test(IsTrainingCheck())
    # 使用 make_test 函数创建测试对象，再次测试 IsTrainingCheck 的训练状态检查
    test_istraining2 = make_test(IsTrainingCheck())
    # 使用 make_test 函数创建测试对象，测试 IsEvalCheck 的评估状态检查
    test_iseval1 = make_test(IsEvalCheck())
    # 使用 make_test 函数创建测试对象，再次测试 IsEvalCheck 的评估状态检查
    test_iseval2 = make_test(IsEvalCheck())
    # 使用 make_test 函数创建测试对象，测试 ViaModuleCall 的模块调用
    test_viamodulecall = make_test(ViaModuleCall())
    # 使用 make_test 函数创建测试对象，测试 IsNoneLayer 的空层检查
    test_isnonelayer = make_test(IsNoneLayer())
    # 使用 make_test 函数创建测试对象，测试 LayerList 的层列表
    test_layerlist = make_test(LayerList())
    # 使用 make_test 函数创建测试对象，测试 TensorList 的张量列表
    test_tensorlist = make_test(TensorList())
    # 使用 make_test 函数创建测试对象，测试 IntArg 的整数参数
    test_intarg = make_test(IntArg())
    # 使用 make_test 函数创建测试对象，测试 CfgModule 的配置模块
    test_cfgmod = make_test(CfgModule())
    # 使用 make_test 函数创建测试对象，测试 StringMember 的字符串成员
    test_stringmember = make_test(StringMember())
    # 使用 make_test 函数创建测试对象，测试 ModuleList 的模块列表
    test_modulelist = make_test(ModuleList())
    # 使用 make_test 函数创建测试对象，测试 NestedModuleList 的嵌套模块列表
    test_modulelist_nested = make_test(NestedModuleList())
    # 使用 make_test 函数创建测试对象，测试 CustomGetItemModuleList 的自定义项获取模块列表
    test_modulelist_custom = make_test(CustomGetItemModuleList())
    # 使用 make_test 函数创建测试对象，测试 ModuleDict 的模块字典
    test_moduledict = make_test(ModuleDict())
    # 使用 make_test 函数创建测试对象，测试 CustomGetItemModuleDict 的自定义项获取模块字典
    test_moduledict_custom = make_test(CustomGetItemModuleDict())
    # 使用 make_test 函数创建测试对象，测试 ParameterDict 的参数字典
    test_parameterdict = make_test(ParameterDict())
    # 使用 make_test 函数创建测试对象，测试 CustomGetItemParameterDict 的自定义项获取参数字典
    test_parameterdict_custom = make_test(CustomGetItemParameterDict())
    # 使用 make_test 函数创建测试对象，测试 SuperModule 的超类模块
    test_super1 = make_test(SuperModule())
    # 使用 make_test 函数创建测试对象，再次测试 SuperModule 的超类模块
    test_super2 = make_test(SuperModule2())
    # 使用 make_test 函数创建测试对象，测试 SuperChildCallsClassMethod 的子类调用超类方法
    test_super_class_method = make_test(SuperChildCallsClassMethod())
    # 使用 make_test 函数创建测试对象，测试 Children 的子模块
    test_children = make_test(Children())
    # 使用 make_test 函数创建测试对象，测试 NamedChildren 的命名子模块
    test_named_children = make_test(NamedChildren())
    # 使用 make_test 函数创建测试对象，测试 DenseNetBlocks 的 DenseNet 模块
    test_densenet = make_test(DenseNetBlocks())
    # 使用 make_test 函数创建测试对象，测试 ParametersModule1 的参数模块1
    test_parameters1 = make_test(ParametersModule1())
    # 使用 make_test 函数创建测试对象，测试 ParametersModule2 的参数模块2
    test_parameters2 = make_test(ParametersModule2())
    # 使用 make_test 函数创建测试对象，测试 ParametersModule3 的参数模块3，期望操作数为5
    test_parameters3 = make_test(ParametersModule3(), expected_ops=5)
    # 使用 make_test 函数创建测试对象，测试 ParametersModule4 的参数模块4
    test_parameters4 = make_test(ParametersModule4())
    # 使用 make_test 函数创建测试对象，测试 ParametersModule5 的参数模块5
    test_parameters5 = make_test(ParametersModule5())
    # 使用 make_test 函数创建测试对象，测试 HasAttrModule 的属性存在模块
    test_hasattr = make_test(HasAttrModule())
    # 使用 make_test 函数创建测试对象，测试 EnumValues 的枚举值
    test_enumvalues = make_test(EnumValues())
    # 使用 make_test 函数创建测试对象，测试 AccessByKeys 的键值访问
    test_access_by_keys = make_test(AccessByKeys())
    # 使用 make_test 函数创建测试对象，测试 ModuleClassMethodCall 的模块类方法调用
    test_module_class_method = make_test(ModuleClassMethodCall())
    # 使用 make_test 函数创建测试对象，测试 ModuleProperty 的模块属性
    test_module_property = make_test(ModuleProperty())
    # 使用 make_test 函数创建测试对象，测试 CallForwardDirectly 的直接前向调用
    test_forward_directly = make_test(CallForwardDirectly())
    # 使用 make_test 函数创建测试对象，测试 ModuleNameString 的模块名称字符串
    test_module_name_string = make_test(ModuleNameString())
    # 使用 make_test 函数创建测试对象，测试 ModuleAttributePrecedence 的模块属性优先级
    test_module_attribute_precedence = make_test(ModuleAttributePrecedence())
    # 使用 make_test 函数创建测试对象，测试 ModuleGuardNameIsValid 的模块保护名称是否有效
    test_module_guard_name_is_valid = make_test(ModuleGuardNameIsValid())
    # 使用 make_test 函数创建测试对象，测试 SequentialWithDuplicatedModule 的包含重复模块的序列
    test_sequential_with_duplicated_module = make_test(SequentialWithDuplicatedModule())
    test_sequential_with_duplicated_module2 = make_test(
        SequentialWithDuplicatedModule2()
    )
    # 创建一个测试，针对带有重复模块的顺序模型2

    test_module_comparison = make_test(ModuleComparison())
    # 创建一个测试，对模块比较进行测试

    def test_module_forward_has_graph_break(self):
        # 定义一个测试函数，测试模块正向传播是否有图断裂问题
        m = ModuleForwardHasGraphBreak()
        # 创建一个具有图断裂问题的模块实例
        x = torch.rand([10, 10])
        # 创建一个随机张量作为输入
        ref = m(x)
        # 调用模块的正向传播得到参考输出
        opt_m = torch._dynamo.optimize("eager")(m)
        # 使用优化策略"eager"优化模块
        res = opt_m(x)
        # 对优化后的模块进行正向传播
        self.assertTrue(torch.allclose(ref, res))
        # 断言参考输出和优化后的输出在数值上全部接近

    def test_unsupportedmethod(self):
        # 定义一个测试函数，测试不支持的方法调用
        m = UnsupportedMethodCall()
        # 创建一个不支持的方法调用的模块实例
        i = torch.randn(10)
        # 创建一个随机张量作为输入
        cnt = torch._dynamo.testing.CompileCounter()
        # 创建一个编译计数器实例
        opt_m = torch._dynamo.optimize(cnt)(m)
        # 使用编译计数器优化模块
        r = opt_m(i)
        # 对优化后的模块进行输入测试
        self.assertTrue(torch._dynamo.testing.same(r, m(i)))
        # 断言优化后的输出和未优化的输出相同
        self.assertEqual(cnt.op_count, 5)
        # 断言操作计数为5

    def test_unsupportedmodule(self):
        # 定义一个测试函数，测试不支持的模块调用
        m = UnsupportedModuleCall()
        # 创建一个不支持的模块调用的模块实例
        i = torch.randn(10)
        # 创建一个随机张量作为输入
        cnt = torch._dynamo.testing.CompileCounter()
        # 创建一个编译计数器实例
        opt_m = torch._dynamo.optimize(cnt)(m)
        # 使用编译计数器优化模块
        r = opt_m(i)
        # 对优化后的模块进行输入测试
        self.assertTrue(torch._dynamo.testing.same(r, m(i)))
        # 断言优化后的输出和未优化的输出相同
        self.assertEqual(cnt.op_count, 6)
        # 断言操作计数为6

    def test_self_mutating1(self):
        # 定义一个测试函数，测试自我变异的模块
        m1 = torch.nn.Linear(10, 10)
        # 创建一个线性变换模块实例
        m2 = SelfMutatingModule(m1)
        # 使用自我变异模块封装m1
        m3 = SelfMutatingModule(m1)
        # 使用自我变异模块封装m1
        m4 = SelfMutatingModule(m1)
        # 使用自我变异模块封装m1
        i = torch.randn(10)
        # 创建一个随机张量作为输入
        out2 = [m2(i), m2(i), m2(i)]
        # 对m2进行三次输入测试
        cnt = torch._dynamo.testing.CompileCounter()
        # 创建一个编译计数器实例
        opt_m3 = torch._dynamo.optimize_assert(cnt)(m3)
        # 使用编译计数器和断言优化m3
        opt_m4 = torch._dynamo.optimize_assert(cnt)(m4)
        # 使用编译计数器和断言优化m4
        out3 = [opt_m3(i), opt_m3(i), opt_m3(i)]
        # 对opt_m3进行三次输入测试
        out4 = [opt_m4(i), opt_m4(i), opt_m4(i)]
        # 对opt_m4进行三次输入测试
        self.assertTrue(torch._dynamo.testing.same(out2, out3))
        # 断言未优化和优化后的输出相同
        self.assertTrue(torch._dynamo.testing.same(out2, out4))
        # 断言未优化和优化后的输出相同
        self.assertEqual(cnt.frame_count, 3)
        # 断言帧计数为3

    @patch.object(torch._dynamo.config, "raise_on_ctx_manager_usage", False)
    def test_generation_tag(self):
        # 定义一个测试函数，测试生成标记
        cnt = torch._dynamo.testing.CompileCounter()
        # 创建一个编译计数器实例

        # 确保我们已经安装了生成标记函数
        with torch._dynamo.optimize_assert(cnt):
            pass
        # 使用编译计数器和断言进入上下文管理器

        m1 = torch.nn.Linear(10, 10)
        # 创建一个线性变换模块实例
        prev_generation = GenerationTracker.get_generation_value(m1)
        # 获取m1的先前生成值
        cur_generation = prev_generation + 1
        # 计算当前生成值

        with torch._dynamo.optimize_assert(cnt):
            m2 = torch.nn.Linear(10, 10)
        # 使用编译计数器和断言优化m2

        self.assertEqual(GenerationTracker.get_generation_value(m1), prev_generation)
        # 断言m1的生成值与先前生成值相同
        self.assertEqual(GenerationTracker.get_generation_value(m2), cur_generation)
        # 断言m2的生成值与当前生成值相同
        # 检查新构建的实例也具有相同的生成值（即使是从旧实例复制过来）
        m3 = deepcopy(m1)
        # 深度复制m1得到m3
        self.assertEqual(GenerationTracker.get_generation_value(m3), cur_generation)
        # 断言m3的生成值与当前生成值相同
    def test_simple_torch_function(self):
        def foo(x):
            # function call, twice to test wrapping
            x = F.sigmoid(x)  # 调用 F 模块的 sigmoid 函数对输入 x 进行处理
            x = F.sigmoid(x)  # 再次调用 sigmoid 函数进行处理
            # method call, twice to test wrapping
            x = x.sigmoid()   # 调用 x 对象的 sigmoid 方法
            x = x.sigmoid()   # 再次调用 sigmoid 方法
            return x

        class TensorProxy(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # 在这里对 __torch_function__ 方法进行重写，以支持特定的操作
                return super().__torch_function__(func, types, args, kwargs)

        # 将 TensorProxy 类添加到可追踪的张量子类列表中
        torch._dynamo.config.traceable_tensor_subclasses.add(TensorProxy)

        try:
            # 生成一个随机张量 x，并将其作为 TensorProxy 的子类
            x = torch.randn(1).as_subclass(TensorProxy)
            # 用于测试编译计数器的实例化
            cnt = torch._dynamo.testing.CompileCounter()
            # 使用 foo 函数处理 x，并将结果保存到 out1
            out1 = foo(x)
            # 对 foo 函数进行优化，并使用优化后的函数处理 x，并将结果保存到 out2
            opt_foo = torch._dynamo.optimize(cnt, nopython=True)(foo)
            out2 = opt_foo(x)

            # 断言编译操作的数量为 4
            self.assertEqual(cnt.op_count, 4)
            # 断言 out1 和 out2 相同
            self.assertTrue(torch._dynamo.testing.same(out1, out2))

        finally:
            # 从可追踪的张量子类列表中移除 TensorProxy 类
            torch._dynamo.config.traceable_tensor_subclasses.remove(TensorProxy)

    def test_torch_function_with_closure(self):
        def run():
            counter = 0

            def foo(x):
                # function call, twice to test wrapping
                x = F.sigmoid(x)  # 调用 F 模块的 sigmoid 函数对输入 x 进行处理
                x = F.sigmoid(x)  # 再次调用 sigmoid 函数进行处理
                # method call, twice to test wrapping
                x = x.sigmoid()   # 调用 x 对象的 sigmoid 方法
                x = x.sigmoid()   # 再次调用 sigmoid 方法
                return x

            class TensorProxy(torch.Tensor):
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    nonlocal counter
                    # 目前只支持从闭包单元中读取
                    # TODO(未来 PR)：也支持写入
                    counter + 1  # 增加计数器，以示例计数

                    # 调用超类的 __torch_function__ 方法
                    return super().__torch_function__(func, types, args, kwargs)

            # 将 TensorProxy 类添加到可追踪的张量子类列表中
            torch._dynamo.config.traceable_tensor_subclasses.add(TensorProxy)

            try:
                # 生成一个随机张量 x，并将其作为 TensorProxy 的子类
                x = torch.randn(1).as_subclass(TensorProxy)
                x = torch.randn(1)  # 生成另一个随机张量 x
                # 用于测试编译计数器的实例化
                cnt = torch._dynamo.testing.CompileCounter()
                # 使用 foo 函数处理 x，并将结果保存到 out1
                out1 = foo(x)
                # 对 foo 函数进行优化，并使用优化后的函数处理 x，并将结果保存到 out2
                opt_foo = torch._dynamo.optimize(cnt, nopython=True)(foo)
                out2 = opt_foo(x)

                # 断言编译操作的数量为 4
                self.assertEqual(cnt.op_count, 4)
                # 断言 out1 和 out2 相同
                self.assertTrue(torch._dynamo.testing.same(out1, out2))
            
            finally:
                # 从可追踪的张量子类列表中移除 TensorProxy 类
                torch._dynamo.config.traceable_tensor_subclasses.remove(TensorProxy)

        run()

    @patch.object(torch._dynamo.config, "raise_on_ctx_manager_usage", False)
    # 定义一个测试方法，用于测试神经网络模块字典的包含性质
    def test_nn_moduledict_contains(self):
        # 定义一个继承自torch.nn.Module的内部类M，接受一个module_dict作为参数
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self, module_dict):
                super().__init__()
                self.module_dict = module_dict  # 将module_dict保存在实例属性中

            # 前向传播方法
            def forward(self, x):
                # 如果"foo"在module_dict中，则执行乘法操作
                if "foo" in self.module_dict:
                    x = torch.mul(x, 1.0)
                # 对x执行加法操作
                x = torch.add(x, 1.0)
                return x

        # 创建一个ModuleDict对象module_dict，其中包含一个名为"foo"的Conv2d模块
        module_dict = torch.nn.ModuleDict({"foo": torch.nn.Conv2d(1, 1, 1)})
        # 创建M类的实例m，传入module_dict作为参数
        m = M(module_dict)
        # 创建一个tensor数据data，形状为(1,)
        data = torch.randn(1)
        # 调用m的前向传播方法，计算输出out1
        out1 = m(data)
        
        # 创建一个CompileCounter对象cnt，用于统计操作数
        cnt = torch._dynamo.testing.CompileCounter()
        # 对m进行动态优化，返回优化后的模型opt_m
        opt_m = torch._dynamo.optimize(cnt, nopython=True)(m)
        # 使用优化后的模型opt_m进行前向传播，计算输出out2
        out2 = opt_m(data)
        
        # 断言优化前后操作数为2
        self.assertEqual(cnt.op_count, 2)
        # 断言out1和out2在数值上相等
        self.assertTrue(torch._dynamo.testing.same(out1, out2))

        # 创建一个新的ModuleDict对象module_dict，其中包含一个名为"bar"的Conv2d模块
        module_dict = torch.nn.ModuleDict({"bar": torch.nn.Conv2d(1, 1, 1)})
        # 使用新的module_dict创建M类的实例m
        m = M(module_dict)
        # 创建新的tensor数据data，形状为(1,)
        data = torch.randn(1)
        # 调用m的前向传播方法，计算输出out1
        out1 = m(data)
        
        # 创建一个CompileCounter对象cnt，用于统计操作数
        cnt = torch._dynamo.testing.CompileCounter()
        # 重置动态计算环境
        torch._dynamo.reset()
        # 对m进行动态优化，返回优化后的模型opt_m
        opt_m = torch._dynamo.optimize(cnt, nopython=True)(m)
        # 使用优化后的模型opt_m进行前向传播，计算输出out2
        out2 = opt_m(data)

        # 断言优化后操作数为1
        self.assertEqual(cnt.op_count, 1)
        # 断言out1和out2在数值上相等
        self.assertTrue(torch._dynamo.testing.same(out1, out2))

        # 创建一个新的ModuleDict对象module_dict，其中包含一个名为"cat"的Conv2d模块
        module_dict = torch.nn.ModuleDict({"cat": torch.nn.Conv2d(1, 1, 1)})
        # 执行一次m的前向传播，将结果保存在pre中
        pre = m(data)
        # 清空计数器cnt的统计信息
        
        cnt.clear()
        
        # 进入动态优化环境，nopython设置为False
        with torch._dynamo.optimize(cnt, nopython=False):
            # 执行一次m的前向传播，将结果保存在opt_pre中
            opt_pre = m(data)
            # 使用新的module_dict创建M类的实例m
            m = M(module_dict)
            # 创建新的tensor数据data，形状为(1,)
            data = torch.randn(1)
            # 调用m的前向传播方法，计算输出out1
            out1 = m(data)

        # 再次执行m的前向传播，计算输出out_post
        out_post = m(data)
        
        # 断言帧计数为1
        self.assertEqual(cnt.frame_count, 1)
        # 断言操作数为1
        self.assertEqual(cnt.op_count, 1)
        # 断言pre和opt_pre在数值上相等
        self.assertTrue(torch._dynamo.testing.same(pre, opt_pre))
        # 断言out1和out_post在数值上相等
        self.assertTrue(torch._dynamo.testing.same(out1, out_post))

    # 该注释标识下面的测试方法是预期动态失败的，出现SymIntArrayRef只包含具体整数的运行时错误
    @expectedFailureDynamic
    # 定义测试方法 test_lazy_module1，使用 self 参数表示这是一个测试方法
    def test_lazy_module1(self):
        # 定义输入的形状为 (16, 3, 6, 7, 8)
        input_shape = (16, 3, 6, 7, 8)

        # 创建 CompileCounter 对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()

        # 创建 LazyModule 的实例 module
        module = LazyModule()

        # 定义测试静态模块的函数 test_static_module
        def test_static_module():
            # 创建输入张量，全为1，形状为 input_shape
            input = torch.ones(*input_shape)
            # 调用 LazyModule 的实例 module 处理输入
            module(input)

        # 使用 torch._dynamo.optimize 优化 test_static_module 函数，禁用 JIT
        opt_test_static_module = torch._dynamo.optimize(cnt, nopython=True)(
            test_static_module
        )
        # 执行优化后的函数
        opt_test_static_module()

        # 断言 module 是否被转换为 MaterializedModule 的实例
        self.assertTrue(
            isinstance(module, MaterializedModule),
            "Module should be transformed to an instance of MaterializedModule.",
        )
        # 断言 module 的参数形状是否与 input_shape 相同
        self.assertEqual(module.param.shape, input_shape)

        # 重新创建 LazyModule 的实例 module
        module = LazyModule()

        # 定义测试未特化情况的函数 test_unspecialized
        def test_unspecialized():
            nonlocal module  # 使用 nonlocal 关键字引用外部的 module 变量
            module = LazyModule()  # 重新创建 LazyModule 的实例
            input = torch.ones(*input_shape)  # 创建输入张量，全为1，形状为 input_shape
            module(input)  # 调用 LazyModule 的实例 module 处理输入

        # 使用 torch._dynamo.optimize 优化 test_unspecialized 函数
        opt_test_unspecialized = torch._dynamo.optimize(cnt)(test_unspecialized)
        # 执行优化后的函数
        opt_test_unspecialized()

        # 断言 module 是否被转换为 MaterializedModule 的实例
        self.assertTrue(
            isinstance(module, MaterializedModule),
            "Module should be transformed to an instance of MaterializedModule.",
        )
        # 断言 module 的参数形状是否与 input_shape 相同
        self.assertEqual(module.param.shape, input_shape)

        # 创建 torch.nn.modules.LazyBatchNorm3d 的实例 module
        module = torch.nn.modules.LazyBatchNorm3d(
            affine=False, track_running_stats=False
        )

        # 创建 CompileCounter 对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()

        # 重置 torch._dynamo 的状态
        torch._dynamo.reset()

        # 定义测试静态 torch 模块的函数 test_torch_static
        def test_torch_static():
            # 创建输入张量，全为1，形状为 input_shape
            input = torch.ones(*input_shape)
            # 返回 module 处理输入的结果，完全材料化
            return module(input)

        # 使用 torch._dynamo.optimize 优化 test_torch_static 函数，禁用 JIT
        opt_test_torch_static = torch._dynamo.optimize(cnt, nopython=True)(
            test_torch_static
        )
        # 执行优化后的函数，获取输出
        out = opt_test_torch_static()

        # 断言 out 是否与 module 处理全为1输入张量的结果相同
        self.assertTrue(same(out, module(torch.ones(*input_shape))))

        # 断言 module 是否被转换为 torch.nn.modules.batchnorm.BatchNorm3d 的实例
        self.assertTrue(
            isinstance(module, torch.nn.modules.batchnorm.BatchNorm3d),
            "Module should be transformed to an instance of BatchNorm3d.",
        )
        # 断言 CompileCounter 对象 cnt 的 frame_count 属性为 1
        self.assertEqual(cnt.frame_count, 1, "No guards should have triggered.")

    # 标记为预期失败的动态测试方法，测试 lazy_module2 方法
    @expectedFailureDynamic
    def test_lazy_module2(self):
        # 测试 FX 图 'call_module' 是否能够正确处理延迟模块参数
        m = LazyMLP()
        x = torch.rand([10, 10])
        # 使用 torch._dynamo.optimize 进行 "eager" 模式的优化，禁用 JIT
        opt_m = torch._dynamo.optimize("eager", nopython=True)(m)
        # 首先执行编译模式，否则在运行 eager 模式时模块会被初始化
        res = opt_m(x)
        ref = m(x)
        # 断言 res 和 ref 是否全部接近
        self.assertTrue(torch.allclose(ref, res))

    # 标记为预期失败的动态测试方法，如果没有 CUDA，则跳过测试
    @expectedFailureDynamic
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    # 测试 LazyMLP 类的功能，特别是对动态编译和优化的支持
    def test_lazy_module3(self):
        # 创建 LazyMLP 实例
        m = LazyMLP()
        # 创建一个 10x10 的随机张量作为输入
        x = torch.rand([10, 10])
        # 创建一个编译计数器
        cnt = torch._dynamo.testing.CompileCounter()
        # 对 LazyMLP 进行动态编译和优化
        opt_m = torch._dynamo.optimize(cnt, nopython=True)(m)

        # 第一次迭代
        res = opt_m(x)
        ref = m(x)
        # 断言优化后的结果与原始结果的近似程度
        self.assertTrue(torch.allclose(ref, res))

        # 将模型和输入移至 CUDA 并进行第二次迭代
        m = m.to("cuda")
        x = x.to("cuda")
        res = opt_m(x)
        ref = m(x)
        # 断言优化后的结果与原始结果的近似程度
        self.assertTrue(torch.allclose(ref, res))
        # 断言编译计数器中的帧数为 2
        self.assertEqual(cnt.frame_count, 2)

    # 期望此测试用例动态失败，并抛出 RuntimeError
    @expectedFailureDynamic
    def test_lazy_module4(self):
        # 创建 LazyMLP 实例
        m = LazyMLP()
        # 创建一个 10x10 的随机张量作为输入
        x = torch.rand([10, 10])
        # 创建一个编译计数器
        cnt = torch._dynamo.testing.CompileCounter()
        # 对 LazyMLP 进行动态编译和优化
        opt_m = torch._dynamo.optimize(cnt, nopython=True)(m)

        # 第一次迭代
        res = opt_m(x)
        ref = m(x)
        # 断言优化后的结果与原始结果的近似程度
        self.assertTrue(torch.allclose(ref, res))

        # 输入形状发生变化，并进行第二次迭代
        x = torch.rand([20, 20])
        # 期望捕获 RuntimeError 异常
        try:
            opt_m(x)
        except RuntimeError:
            # 断言异常信息中包含特定字符串
            self.assertIn("must have same reduction dim", traceback.format_exc())

    # 期望此测试用例动态失败，并抛出 RuntimeError
    @expectedFailureDynamic
    def test_lazy_module5(self):
        # 测试 LazyModuleWithListInput 类能够处理列表/元组输入
        m = LazyModuleWithListInput()
        # 创建一个包含随机张量的列表
        x = [torch.rand([5, 5])] * 3 + [None]
        # 对 LazyModuleWithListInput 进行动态编译和优化
        opt_m = torch._dynamo.optimize("eager", nopython=True)(m)
        # 执行优化后的模型
        res = opt_m(x)
        ref = m(x)
        # 断言优化后的结果与原始结果的近似程度
        self.assertTrue(torch.allclose(ref, res))

    # 期望此测试用例动态失败，并抛出 RuntimeError
    @expectedFailureDynamic
    def test_lazy_module6(self):
        # 测试 LazyModuleWithLazySubmodule 类的功能，特别是在其初始化参数时的惰性子模块支持
        m = LazyModuleWithLazySubmodule()
        # 创建一个包含随机张量的列表
        x = [torch.rand([5, 5])] * 3
        # 对 LazyModuleWithLazySubmodule 进行动态编译和优化
        opt_m = torch._dynamo.optimize("eager", nopython=True)(m)
        # 执行优化后的模型
        res = opt_m(x)
        ref = m(x)
        # 断言优化后的结果与原始结果的近似程度
        self.assertTrue(torch.allclose(ref, res))

    # 期望此测试用例动态失败，并抛出 RuntimeError
    @expectedFailureDynamic
    def test_lazy_module7(self):
        # 测试 LazyModuleWithNamedTupleInput 类能够处理命名元组/字典输入
        m = LazyModuleWithNamedTupleInput()
        # 创建自定义输入 MyInput 的实例
        x = MyInput(
            x={"a": [torch.rand([5, 5])] * 3, "b": torch.rand([5, 5])},
            y=torch.rand([5, 5]),
        )
        # 对 LazyModuleWithNamedTupleInput 进行动态编译和优化
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        # 执行优化后的模型
        res = opt_m(x)
        ref = m(x)
        # 断言优化后的结果与原始结果的近似程度
        self.assertTrue(torch.allclose(ref, res))

    # 测试 LazyChildModuleNoClsToBecome 类在 cls_to_become 为 None 时 super() 函数的正常工作
    def test_lazy_module_no_cls_to_become(self):
        m = LazyChildModuleNoClsToBecome()
        x = torch.rand(2, 2)
        opt_m = torch._dynamo.optimize("eager", nopython=True)(m)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))
    # 测试 LazyModuleKwArgs 类的功能
    def test_lazy_module_kwargs(self):
        # 创建 LazyModuleKwArgs 的实例
        m = LazyModuleKwArgs()
        # 创建包含三个相同随机张量的列表
        x = [torch.rand([5, 5])] * 3
        # 创建包含两个相同随机张量的列表
        y = [torch.rand([5, 5])] * 2
        # 使用 torch.compile 对象对 m 进行编译，设置 backend="eager" 和 fullgraph=True
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        # 计算 m 在输入 x 和 y 上的期望结果
        exp_res = m(x, y)
        # 断言优化后的 opt_m 在输入 x 和 y 上的计算结果与 exp_res 相近
        self.assertTrue(torch.allclose(exp_res, opt_m(x, y)))

    # 测试调用带非常量输入的函数是否安全
    def test_call_fn_with_non_const_inputs_safe(self):
        # 定义 ModuleSpecialFwd 类，继承自 torch.nn.Module
        class ModuleSpecialFwd(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个 Conv2d 对象作为该类的属性
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=20, kernel_size=(5, 5)
                )

            # 定义 _conv_forward 方法，执行卷积前向传播
            def _conv_forward(self, x):
                return self.conv._conv_forward(x, self.conv.weight, self.conv.bias)

            # 定义 forward 方法，调用 _conv_forward 方法
            def forward(self, x):
                return self._conv_forward(x)

        # 创建 ModuleSpecialFwd 的实例
        mod = ModuleSpecialFwd()
        # 创建形状为 [3, 10, 10] 的随机张量 rx
        rx = torch.randn([3, 10, 10])
        # 在模型 mod 上执行真实的前向传播
        real = mod(rx)
        # 使用 torch._dynamo.export 导出 mod 的计算图，传入输入 rx
        graph, _ = torch._dynamo.export(mod)(rx)
        # 断言真实结果 real 与计算图 graph 在输入 rx 上的结果是否一致
        self.assertTrue(torch._dynamo.testing.same(real, graph(rx)))

    # 测试直接调用 ConvCallForwardDirectly 类的前向传播功能
    def test_conv_call_forward_directly(self):
        # 创建 ConvCallForwardDirectly 的实例
        m = ConvCallForwardDirectly()
        # 创建形状为 [4, 3, 9, 9] 的随机张量 x
        x = torch.rand([4, 3, 9, 9])
        # 在实例 m 上执行前向传播，得到参考结果 ref
        ref = m(x)
        # 使用 torch.compile 对象对 m 进行编译，设置 backend="eager" 和 fullgraph=True
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        # 在优化后的 opt_m 上执行前向传播，得到结果 res
        res = opt_m(x)
        # 断言优化后的 res 与参考结果 ref 是否相近
        self.assertTrue(torch.allclose(ref, res))

    # 测试直接调用 ConvTransposeCallForwardDirectly 类的转置卷积前向传播功能
    def test_conv_transpose_call_forward_directly(self):
        # 创建 ConvTransposeCallForwardDirectly 的实例
        m = ConvTransposeCallForwardDirectly()
        # 创建形状为 [4, 4, 4, 4] 的随机张量 x
        x = torch.rand([4, 4, 4, 4])
        # 在实例 m 上执行前向传播，得到参考结果 ref
        ref = m(x)
        # 使用 torch.compile 对象对 m 进行编译，设置 backend="eager" 和 fullgraph=True
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        # 在优化后的 opt_m 上执行前向传播，得到结果 res
        res = opt_m(x)
        # 断言优化后的 res 与参考结果 ref 是否相近
        self.assertTrue(torch.allclose(ref, res))

    # 测试直接调用 ConvCallSuperForwardDirectly 类的带有超类调用的前向传播功能
    def test_conv_call_super_forward_directly(self):
        # 创建形状为 [4, 4] 的随机张量 x
        x = torch.randn(4, 4)
        # 创建 ConvCallSuperForwardDirectly 的实例 m
        m = ConvCallSuperForwardDirectly(4, 4, 4)
        # 在实例 m 上执行前向传播，得到参考结果 ref
        ref = m(x)
        # 使用 torch.compile 对象对 m 进行编译，设置 backend="eager" 和 fullgraph=True
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        # 在优化后的 opt_m 上执行前向传播，得到结果 res
        res = opt_m(x)
        # 断言优化后的 res 与参考结果 ref 是否相近
        self.assertTrue(torch.allclose(ref, res))

    # 测试直接调用 ConvTransposeCallSuperForwardDirectly 类的带有超类调用的转置卷积前向传播功能
    def test_conv_transpose_call_super_forward_directly(self):
        # 创建形状为 [4, 4, 4] 的随机张量 x
        x = torch.randn(4, 4, 4)
        # 创建 ConvTransposeCallSuperForwardDirectly 的实例 m
        m = ConvTransposeCallSuperForwardDirectly(4, 4, 4)
        # 在实例 m 上执行前向传播，得到参考结果 ref
        ref = m(x)
        # 使用 torch.compile 对象对 m 进行编译，设置 backend="eager" 和 fullgraph=True
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        # 在优化后的 opt_m 上执行前向传播，得到结果 res
        res = opt_m(x)
        # 断言优化后的 res 与参考结果 ref 是否相近
        self.assertTrue(torch.allclose(ref, res))
class MockModule(torch.nn.Module):
    # 定义一个模拟的 PyTorch 模块
    def __init__(self):
        super().__init__()
        # 初始化 ReLU 激活函数
        self.relu = torch.nn.ReLU()
        # 初始化线性层，输入和输出维度都为 10
        self.linear = torch.nn.Linear(10, 10)
        # 注册一个缓冲区 buf0，其内容为大小为 10x10 的随机张量
        self.register_buffer("buf0", torch.randn(10, 10))

    # 定义前向传播函数
    def forward(self, x):
        # 执行前向传播操作：线性层 -> ReLU -> 加上缓冲区 buf0
        return self.relu(self.linear(x) + self.buf0)


class OptimizedModuleTest(torch._dynamo.test_case.TestCase):
    # 定义优化模块的测试类，继承自 PyTorch 的测试用例类
    def test_nn_module(self):
        # 创建 MockModule 实例
        mod = MockModule()
        # 创建一个编译计数器实例
        cnt = torch._dynamo.testing.CompileCounter()
        # 对模块进行优化，使用编译计数器进行计数
        opt_mod = torch._dynamo.optimize(cnt)(mod)
        # 断言优化后的模块类型为 OptimizedModule
        self.assertIsInstance(opt_mod, torch._dynamo.OptimizedModule)

        # 创建输入张量 x，大小为 10x10
        x = torch.randn(10, 10)
        # 断言模块和优化模块输出相同
        self.assertTrue(torch._dynamo.testing.same(mod(x), opt_mod(x)))
        # 断言前向传播调用次数为 1
        self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(guard_nn_modules=True)
    # 使用装饰器定义测试方法，标记为需要 patch 的测试
    def test_attr_precedence(self):
        # 定义一个继承自 Module 的类 Mod
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 设置实例属性 a 为 3
                self.a = 3

            def forward(self, x, c=4):
                # 定义前向传播函数，返回 x 乘以参数 c
                return x * c

            def linear(self, x):
                # 定义一个线性函数 linear，返回 x
                return x

            def b(self, x):
                # 抛出运行时错误，不应该被调用
                raise RuntimeError("Should not be called")

        # 定义一个继承自 Mod 的子类 MyMod
        class MyMod(Mod):
            def __init__(self):
                super().__init__()
                # 设置实例属性 linear 为一个输入输出维度为 11 的线性层
                self.linear = torch.nn.Linear(11, 11)
                # 设置实例属性 a 为 2
                self.a = 2
                # 设置实例属性 b 为 2
                self.b = 2
                # 设置实例属性 scale 为 1
                self.scale = 1

            def scale(self, x):
                # 不应该被调用，因为被实例属性覆盖了
                raise RuntimeError("Should not be called")

            def forward(self, x, c=None):
                # 执行前向传播操作：线性层 -> self.a -> self.b -> self.scale
                return self.linear(x) * self.a * self.b * self.scale

        # 创建 MyMod 的实例 mod
        mod = MyMod()
        # 创建大小为 3x3 的全为 1 的输入张量 x
        x = torch.ones(3, 3)
        # 记录参考输出
        ref = mod(x)

        # 创建一个编译计数器实例 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对模块进行编译
        opt_mod = torch.compile(mod, backend=cnts)
        opt_mod(torch.ones(3, 3))
        # 执行模块操作并记录结果
        res = opt_mod(torch.ones(3, 3))

        # 断言前向传播调用次数为 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言结果与参考输出相等
        self.assertEqual(ref, res)

    def test_to(self):
        # 创建 MockModule 实例 mod
        mod = MockModule()
        # 创建一个编译计数器实例 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 对模块进行优化，使用编译计数器进行计数
        opt_mod = torch._dynamo.optimize(cnt)(mod)
        # 创建大小为 10x10 的随机输入张量 x
        x = torch.randn(10, 10)
        # 断言模块和优化模块输出相同
        self.assertTrue(torch._dynamo.testing.same(mod(x), opt_mod(x)))
        # 断言前向传播调用次数为 1
        self.assertEqual(cnt.frame_count, 1)

        # 确保没有重新编译
        opt_mod(x)
        # 断言前向传播调用次数为 1
        self.assertEqual(cnt.frame_count, 1)

        # 将优化模块转移到 CPU 和 torch.float64 类型
        opt_mod = opt_mod.to(device="cpu").to(dtype=torch.float64)
        # 断言优化后模块类型为 OptimizedModule
        self.assertIsInstance(opt_mod, torch._dynamo.OptimizedModule)
        # 创建大小为 10x10 的随机输入张量 x，类型为 torch.float64
        x = torch.randn(10, 10).to(dtype=torch.float64)
        opt_mod(x)
        # 确保重新编译
        self.assertEqual(cnt.frame_count, 2)

        # 确保没有重新编译
        opt_mod(x)
        # 断言前向传播调用次数为 2
        self.assertEqual(cnt.frame_count, 2)

        # 重置动态图计算环境
        torch._dynamo.reset()
        opt_mod(x)
        # 断言前向传播调用次数为 3
        self.assertEqual(cnt.frame_count, 3)
    @torch._dynamo.config.patch(guard_nn_modules=True)
    # 使用 Torch 内部的动态配置装饰器，启用神经网络模块保护
    def test_param_order(self):
        # 定义测试函数 test_param_order，其中 self 是测试类的实例

        class MyModule(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的子类 MyModule
            def __init__(self):
                # 初始化方法
                super().__init__()
                # 调用父类初始化方法
                self.param1 = torch.nn.Parameter(torch.ones([1]))
                # 创建名为 param1 的参数，值为长度为 1 的张量，作为模块的一个参数
                self.param2 = torch.nn.Parameter(torch.ones([2]))
                # 创建名为 param2 的参数，值为长度为 2 的张量，作为模块的另一个参数

            def forward(self, x):
                # 前向传播方法，接收输入 x
                return x
                # 返回输入 x

        mod = MyModule()
        # 创建 MyModule 类的实例 mod
        coeffs = [2, 3]
        # 创建一个包含系数的列表

        def fn(x):
            # 定义一个函数 fn，接收输入 x
            for idx, p in enumerate(mod.parameters()):
                # 遍历 MyModule 实例 mod 的所有参数
                x += p.sum() * coeffs[idx]
                # x 加上当前参数 p 的和乘以对应的系数 coeffs[idx]

            for idx, p in enumerate(mod.named_parameters()):
                # 遍历 MyModule 实例 mod 的所有命名参数
                x += p[1].sum() * coeffs[idx]
                # x 加上当前参数 p 的第二个元素（即参数值）的和乘以对应的系数 coeffs[idx]

            return x
            # 返回处理后的 x

        ref = fn(torch.ones(1))
        # 计算 fn 函数在输入为长度为 1 的张量 torch.ones(1) 时的结果
        cnts = torch._dynamo.testing.CompileCounter()
        # 创建编译计数器 cnts
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 使用编译计数器对 fn 进行优化，并将结果赋给 opt_fn
        res = opt_fn(torch.ones(1))
        # 计算优化后的 fn 在输入为长度为 1 的张量 torch.ones(1) 时的结果

        self.assertEqual(ref, res)
        # 断言优化前后的结果相等
        self.assertEqual(cnts.frame_count, 1)
        # 断言编译计数器的帧计数为 1

        mod._parameters["param1"] = mod._parameters.pop("param1")
        # 将模块中的 param1 参数移出 _parameters，并将其重新赋值给 _parameters 字典中的 param1
        ref = fn(torch.ones(1))
        # 再次计算 fn 函数在输入为长度为 1 的张量 torch.ones(1) 时的结果
        res = opt_fn(torch.ones(1))
        # 再次计算优化后的 fn 在输入为长度为 1 的张量 torch.ones(1) 时的结果

        self.assertEqual(ref, res)
        # 断言优化前后的结果相等
        self.assertEqual(cnts.frame_count, 2)
        # 断言编译计数器的帧计数为 2

    @torch._dynamo.config.patch(guard_nn_modules=True)
    # 使用 Torch 内部的动态配置装饰器，启用神经网络模块保护
    def test_buffer_order(self):
        # 定义测试函数 test_buffer_order，其中 self 是测试类的实例

        class MyModule(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的子类 MyModule
            def __init__(self):
                # 初始化方法
                super().__init__()
                # 调用父类初始化方法
                self.register_buffer("b1", torch.ones([1]))
                # 注册名为 b1 的缓冲区，值为长度为 1 的张量，作为模块的一个缓冲区
                self.register_buffer("b2", torch.ones([2]))
                # 注册名为 b2 的缓冲区，值为长度为 2 的张量，作为模块的另一个缓冲区

            def forward(self, x):
                # 前向传播方法，接收输入 x
                return x
                # 返回输入 x

        mod = MyModule()
        # 创建 MyModule 类的实例 mod
        coeffs = [2, 3]
        # 创建一个包含系数的列表

        def fn(x):
            # 定义一个函数 fn，接收输入 x
            for idx, p in enumerate(mod.buffers()):
                # 遍历 MyModule 实例 mod 的所有缓冲区
                x += p.sum() * coeffs[idx]
                # x 加上当前缓冲区 p 的和乘以对应的系数 coeffs[idx]

            for idx, p in enumerate(mod.named_buffers()):
                # 遍历 MyModule 实例 mod 的所有命名缓冲区
                x += p[1].sum() * coeffs[idx]
                # x 加上当前缓冲区 p 的第二个元素（即缓冲区值）的和乘以对应的系数 coeffs[idx]

            return x
            # 返回处理后的 x

        ref = fn(torch.ones(1))
        # 计算 fn 函数在输入为长度为 1 的张量 torch.ones(1) 时的结果
        cnts = torch._dynamo.testing.CompileCounter()
        # 创建编译计数器 cnts
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 使用编译计数器对 fn 进行优化，并将结果赋给 opt_fn
        res = opt_fn(torch.ones(1))
        # 计算优化后的 fn 在输入为长度为 1 的张量 torch.ones(1) 时的结果

        self.assertEqual(ref, res)
        # 断言优化前后的结果相等
        self.assertEqual(cnts.frame_count, 1)
        # 断言编译计数器的帧计数为 1

        mod._buffers["b1"] = mod._buffers.pop("b1")
        # 将模块中的 b1 缓冲区移出 _buffers，并将其重新赋值给 _buffers 字典中的 b1
        ref = fn(torch.ones(1))
        # 再次计算 fn 函数在输入为长度为 1 的张量 torch.ones(1) 时的结果
        res = opt_fn(torch.ones(1))
        # 再次计算优化后的 fn 在输入为长度为 1 的张量 torch.ones(1) 时的结果

        self.assertEqual(ref, res)
        # 断言优化前后的结果相等
        self.assertEqual(cnts.frame_count, 2)
        # 断言编译计数器的帧计数为 2
    def test_module_order(self):
        # 定义一个继承自 torch.nn.Module 的类 MyModule
        class MyModule(torch.nn.Module):
            # 类的初始化方法
            def __init__(self):
                super().__init__()
                # 定义两个线性层
                self.linear1 = torch.nn.Linear(3, 3)
                self.linear2 = torch.nn.Linear(10, 10)

            # 前向传播方法
            def forward(self, x):
                return x

        # 创建 MyModule 的实例 mod
        mod = MyModule()
        # 定义一个系数列表
        coeffs = [2, 3, 4]

        # 创建一个字典，用于存储 MyModule 及其子模块和系数的对应关系
        coeffs_for_mod = {mod: 10, mod.linear1: 20, mod.linear2: 30}

        # 检查 _modules 的顺序
        def fn(x):
            # 遍历 mod 的所有模块
            for idx, p in enumerate(mod.modules()):
                # 强制依赖于顺序的一些操作
                x += coeffs_for_mod[p] * coeffs[idx]
            # 遍历 mod 的所有命名模块
            for idx, p in enumerate(mod.named_modules()):
                x += coeffs_for_mod[p[1]] * coeffs[idx]
            # 遍历 mod 的所有子模块
            for idx, p in enumerate(mod.children()):
                x += coeffs_for_mod[p] * coeffs[idx]
            # 遍历 mod 的所有命名子模块
            for idx, p in enumerate(mod.named_children()):
                x += coeffs_for_mod[p[1]] * coeffs[idx]
            return x

        # 计算原始函数的参考值
        ref = fn(torch.ones(1))
        # 创建一个计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 fn 进行优化
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 计算优化后函数的结果
        res = opt_fn(torch.ones(1))

        # 断言优化前后函数的结果一致
        self.assertEqual(ref, res)
        # 断言帧计数为 1
        self.assertEqual(cnts.frame_count, 1)

        # 将 linear1 移动到 _modules 的末尾
        mod._modules["linear1"] = mod._modules.pop("linear1")
        # 重新计算 fn 的参考值
        ref = fn(torch.ones(1))
        # 计算优化后函数的结果
        res = opt_fn(torch.ones(1))

        # 断言优化前后函数的结果一致
        self.assertEqual(ref, res)
        # 断言帧计数为 2
        self.assertEqual(cnts.frame_count, 2)

    def test_attr(self):
        # 定义一个 MockModule 类，继承自 torch.nn.Module
        class MockModule(torch.nn.Module):
            # 类的初始化方法
            def __init__(self):
                super().__init__()
                # 定义一个线性层和一个缓冲区
                self.linear = torch.nn.Linear(10, 10)
                self.register_buffer("buf0", torch.randn(10, 10))

            # 前向传播方法
            def forward(self, x):
                return self.r(torch.sin(x)) + self.buf0

        # 创建 MockModule 的实例 mod
        mod = MockModule()
        # 对 mod 进行 eager 模式的优化
        opt_mod = torch._dynamo.optimize("eager")(mod)

        # 检查参数和缓冲区
        for p1, p2 in zip(mod.parameters(), opt_mod.parameters()):
            self.assertTrue(id(p1) == id(p2))
        for b1, b2 in zip(mod.buffers(), opt_mod.buffers()):
            self.assertTrue(id(b1) == id(b2))

        # 定义一个函数，获取模块参数的数据类型
        def get_parameter_dtype(mod: torch.nn.Module):
            parameters_and_buffers = itertools.chain(mod.parameters(), mod.buffers())
            return next(parameters_and_buffers).dtype

        # 对 get_parameter_dtype 函数进行 eager 模式的优化
        opt_mod = torch._dynamo.optimize("eager")(get_parameter_dtype)
        # 计算优化后函数的输出数据类型
        out_dtype = opt_mod(mod)
        # 断言输出数据类型为 torch.float32
        self.assertEqual(out_dtype, torch.float32)
    # 定义一个测试方法，用于测试模块的属性、参数和缓冲区
    def test_dir(self):
        # 定义一个模拟模块，包含线性层、缓冲区和参数
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.register_buffer("buf0", torch.randn(10, 10))
                self.register_parameter(
                    name="param0", param=torch.nn.Parameter(torch.randn(10, 10))
                )

            def forward(self, x):
                return self.r(torch.sin(x)) + self.buf0

        # 创建 MockModule 实例
        mod = MockModule()
        # 获取模块的属性列表
        mod_keys = dir(mod)
        # 对模块进行优化
        opt_mod = torch._dynamo.optimize("eager")(mod)
        # 获取优化后模块的属性列表
        opt_mod_keys = dir(opt_mod)

        # 检查用户定义的属性、参数和缓冲区是否存在于优化后模块的属性列表中
        self.assertIn("linear", opt_mod_keys)
        self.assertIn("buf0", opt_mod_keys)
        self.assertIn("param0", opt_mod_keys)

        # 检查所有属性、参数和缓冲区是否一致
        self.assertTrue(len(set(mod_keys).difference(opt_mod_keys)) == 0)

    # 测试在 nn 保护模块上不重新编译
    def test_no_recompile_on_nn_guarded_modules(self):
        size = (10, 10)
        cache_size_limit = 1
        num_submodules = 4
        cnts = torch._dynamo.testing.CompileCounterWithBackend("eager")

        # 定义一个子模块
        class SubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(*size)

            def forward(self, x):
                a = torch.sin(torch.cos(x))
                return self.linear(a)

        # 定义一个模拟模块，包含多个子模块
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mods = [SubModule() for _ in range(num_submodules)]
                # 对每个子模块进行编译
                self.mods = [torch.compile(mod, backend=cnts) for mod in self.mods]

            def forward(self, x):
                for mod in self.mods:
                    x = mod(x)
                return x

        # 创建 MockModule 实例
        mod = MockModule()
        # 每个子模块都单独编译，并具有不同的 nn 模块保护。确保重新编译逻辑被正确处理
        with unittest.mock.patch(
            "torch._dynamo.config.error_on_recompile", True
        ), unittest.mock.patch(
            "torch._dynamo.config.cache_size_limit",
            cache_size_limit,
        ):
            x = torch.randn(*size, requires_grad=True)
            mod(x)
            if torch._dynamo.config.inline_inbuilt_nn_modules:
                self.assertEqual(cnts.frame_count, 1)
            else:
                self.assertEqual(cnts.frame_count, num_submodules)

    # 使用 patch 对象设置 inline_inbuilt_nn_modules 为 True
    @patch.object(torch._dynamo.config, "inline_inbuilt_nn_modules", True)
    def test_inline_inbuilt_nn_modules(self):
        size = (10, 10)  # 定义输入数据的大小为 10x10
        cache_size_limit = 1  # 定义缓存大小限制为 1
        num_submodules = 4  # 定义子模块数量为 4
        cnts = torch._dynamo.testing.CompileCounterWithBackend("eager")  # 创建编译计数器对象

        class SubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(*size)  # 创建线性层模块

            def forward(self, x):
                a = torch.sin(torch.cos(x))  # 对输入数据进行 sin(cos(x)) 的计算
                return self.linear(a)  # 将结果输入线性层进行计算

        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mods = [SubModule() for _ in range(num_submodules)]  # 创建多个子模块对象列表
                self.mods = [torch.compile(mod, backend=cnts) for mod in self.mods]  # 编译每个子模块

            def forward(self, x):
                for mod in self.mods:
                    x = mod(x)  # 逐个调用每个子模块进行前向传播计算
                return x

        mod = MockModule()  # 创建模拟模块对象
        # 每个子模块都单独编译，并具有不同的 nn 模块保护。确保重新编译逻辑正确处理。
        with unittest.mock.patch(
            "torch._dynamo.config.error_on_recompile", True
        ), unittest.mock.patch(
            "torch._dynamo.config.cache_size_limit",
            cache_size_limit,
        ):
            x = torch.randn(*size, requires_grad=True)  # 创建随机输入数据张量
            mod(x)  # 调用模块进行计算
            self.assertEqual(cnts.frame_count, 1)  # 断言编译计数器的帧计数为 1

    def test_cache_size_limit_on_guarded_nn_modules(self):
        cache_size_limit = 2  # 定义缓存大小限制为 2
        num_submodules = 4  # 定义子模块数量为 4
        cnts = torch._dynamo.testing.CompileCounterWithBackend("eager")  # 创建编译计数器对象

        class SubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()  # 创建 ReLU 激活函数模块

            def forward(self, x):
                a = torch.sin(torch.cos(x))  # 对输入数据进行 sin(cos(x)) 的计算
                return self.relu(a)  # 将结果输入 ReLU 模块进行计算

        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mods = [SubModule() for _ in range(num_submodules)]  # 创建多个子模块对象列表
                self.mods = [torch.compile(mod, backend=cnts) for mod in self.mods]  # 编译每个子模块

            def forward(self, x):
                for mod in self.mods:
                    x = mod(x)  # 逐个调用每个子模块进行前向传播计算
                return x

        mod = MockModule()  # 创建模拟模块对象
        # 对于第三次迭代，将达到缓存大小限制，预期的总帧数为 2 * num_submodules。
        with unittest.mock.patch(
            "torch._dynamo.config.cache_size_limit",
            cache_size_limit,
        ):
            for size in [
                (4,),
                (4, 4),
                (4, 4, 4),
            ]:
                x = torch.randn(size)  # 创建不同尺寸的随机输入数据张量
                mod(x)  # 调用模块进行计算
        if torch._dynamo.config.inline_inbuilt_nn_modules:
            self.assertEqual(cnts.frame_count, 2)  # 断言编译计数器的帧计数为 2
        else:
            self.assertEqual(cnts.frame_count, 2 * num_submodules)  # 断言编译计数器的帧计数为 2 * num_submodules
    # 定义递归优化测试方法
    def test_recursion(self):
        # 创建 MockModule 实例
        mod = MockModule()
        # 创建 CompileCounter 实例
        cnt = torch._dynamo.testing.CompileCounter()
        # 对 mod 应用编译优化，并返回优化后的模块 opt_mod
        opt_mod = torch._dynamo.optimize(cnt)(mod)

        # 进行五次优化迭代
        for _ in range(5):
            opt_mod = torch._dynamo.optimize(cnt)(opt_mod)
        
        # 对优化后的模块 opt_mod 执行前向传播
        opt_mod(torch.randn(10, 10))
        # 断言优化次数为 1
        self.assertEqual(cnt.frame_count, 1)

    # 定义模块组合测试方法
    def test_composition(self):
        # 定义内部模块 InnerModule
        class InnerModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(torch.sin(x))

        # 创建 InnerModule 实例并应用编译优化
        opt_inner_mod = InnerModule()

        # 定义外部模块 OuterModule
        class OuterModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = opt_inner_mod

            def forward(self, x):
                return self.mod(torch.cos(x))

        # 创建 OuterModule 实例
        outer_mod = OuterModule()
        # 创建 CompileCounter 实例
        cnt = torch._dynamo.testing.CompileCounter()
        # 对 outer_mod 应用编译优化，并返回优化后的模块 opt_outer_mod
        opt_outer_mod = torch._dynamo.optimize(cnt)(outer_mod)

        # 生成输入张量 x
        x = torch.randn(4)
        # 断言 opt_outer_mod 是 OptimizedModule 类的实例
        self.assertIsInstance(opt_outer_mod, torch._dynamo.OptimizedModule)
        # 断言 outer_mod 和 opt_outer_mod 在相同输入下的输出相等
        self.assertTrue(torch._dynamo.testing.same(outer_mod(x), opt_outer_mod(x)))
        # 断言优化次数为 1
        self.assertEqual(cnt.frame_count, 1)

    # 定义带优化模块的模块组合测试方法
    def test_composition_with_opt_mod(self):
        # 定义内部模块 InnerModule
        class InnerModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(torch.sin(x))

        # 创建 InnerModule 实例
        inner_mod = InnerModule()
        # 创建 CompileCounter 实例
        cnt = torch._dynamo.testing.CompileCounter()
        # 对 inner_mod 应用编译优化，并返回优化后的模块 opt_inner_mod
        opt_inner_mod = torch._dynamo.optimize(cnt)(inner_mod)

        # 定义外部模块 OuterModule
        class OuterModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = opt_inner_mod

            def forward(self, x):
                return self.mod(torch.cos(x))

        # 创建 OuterModule 实例并对其应用编译优化
        outer_mod = OuterModule()
        opt_outer_mod = torch._dynamo.optimize(cnt)(outer_mod)

        # 生成输入张量 x
        x = torch.randn(4)
        # 断言 opt_outer_mod 是 OptimizedModule 类的实例
        self.assertIsInstance(opt_outer_mod, torch._dynamo.OptimizedModule)
        # 断言 outer_mod 和 opt_outer_mod 在相同输入下的输出相等
        self.assertTrue(torch._dynamo.testing.same(outer_mod(x), opt_outer_mod(x)))
        # 断言优化次数为 2，因为内部模块也经历了一次优化
        self.assertEqual(cnt.frame_count, 2)

    # 定义模块补丁测试方法
    def test_module_patch(self):
        # 创建 ModulePatch1 实例
        mod = ModulePatch1()
        # 将 mod 的 forward 方法替换为 ModulePatch2 的 forward 方法
        mod.forward = types.MethodType(ModulePatch2.forward, mod)

        # 定义函数 fn，使用 mod 进行前向传播
        def fn(x):
            return mod(x)

        # 断言使用 eager 模式和 nopython=True 对 fn 进行编译优化后的结果满足 torch.allclose 的条件
        self.assertTrue(
            torch.allclose(
                torch._dynamo.optimize("eager", nopython=True)(fn)(torch.ones(10)),
                torch.zeros(1),
            )
        )

    # 设置 torch._dynamo.config 的 skip_nnmodule_hook_guards 属性为 False 的对象补丁
    @patch.object(torch._dynamo.config, "skip_nnmodule_hook_guards", False)
    def test_hooks_outer(self):
        # 定义一个继承自 torch.nn.Module 的测试模块
        class TestModule(torch.nn.Module):
            # 定义模块的前向传播方法，输入一个 torch.Tensor，返回一个 torch.Tensor
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 2 * x + 1

        # 创建 TestModule 的实例
        m = TestModule()

        # 定义一个前向钩子函数，接收一个 torch.nn.Module、一个元组包含输入的 torch.Tensor 和输出的 torch.Tensor，返回一个 torch.Tensor
        def forward_hook(
            module: torch.nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor
        ) -> torch.Tensor:
            return 2 * output + 1

        # 注册前向钩子函数到 m 上，并获得一个处理句柄
        handle = m.register_forward_hook(forward_hook)

        # 创建一个需要梯度的 torch.Tensor 输入
        inp = torch.tensor(1.0, requires_grad=True)

        # 定义一个用于捕捉失败信息的函数
        failure_reason = None
        def guard_fail_fn(failure):
            nonlocal failure_reason
            failure_reason = failure[0]

        # 使用 torch._dynamo.optimize 进行模块编译，指定 guard_fail_fn 和后端为 "eager"
        compiled_m = torch._dynamo.optimize(
            guard_fail_fn=guard_fail_fn, backend="eager"
        )(m)

        # 断言编译后的模块执行和原始模块执行结果相同
        self.assertEqual(compiled_m(inp), m(inp))
        # 断言编译后的模块执行结果为 7
        self.assertEqual(compiled_m(inp).item(), 7)
        # 断言没有捕捉到失败原因
        self.assertTrue(failure_reason is None)

        # 移除之前注册的钩子函数
        handle.remove()

        # 再次断言编译后的模块执行和原始模块执行结果相同
        self.assertEqual(compiled_m(inp), m(inp))
        # 断言编译后的模块执行结果为 3
        self.assertEqual(compiled_m(inp).item(), 3)

        # 注意：以下注释部分不需要在代码块中包含，因为示例并未要求包含在注释内
        # self.assertTrue(failure_reason == "hook")

        """
        Summary:
          - 移除钩子并不会导致守卫失败，因为我们在编译时并没有将钩子（至少是相同的图形）编译进去！
            我们正确地省略了调用已移除的钩子，但由于此钩子是一个后向前钩子，从前向传播的 'RETURN' 断开了图形。
          
          Why is 'forward' the entrypoint to an InstructionTranslator, after I changed
          the eval_frame entrypoint to Module.__call__?
        """
    def test_hooks_inner(self):
        # 定义一个测试用的内部模块类 TestModule
        class TestModule(torch.nn.Module):
            # 重写 forward 方法，实现简单的计算逻辑：返回输入的两倍加一
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 2 * x + 1

        # 创建 TestModule 实例
        m = TestModule()

        # 定义一个 forward hook 函数
        def forward_hook(
            module: torch.nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor
        ) -> torch.Tensor:
            # 对输出进行简单的转换：返回输出的两倍加一
            return 2 * output + 1

        # 将 forward hook 注册到模块 m 上
        handle = m.register_forward_hook(forward_hook)

        # 定义一个外部函数 outer_func，接收一个 tensor 输入，进行简单计算并调用模块 m
        def outer_func(tensor):
            x = tensor * 2 + 1
            y = m(x)
            return y

        # 创建一个需要梯度的输入 tensor
        inp = torch.tensor(1.0, requires_grad=True)

        # 用于记录失败原因的变量
        failure_reason = None

        # 定义一个 guard_fail_fn 函数，用于处理失败情况
        def guard_fail_fn(failure):
            nonlocal failure_reason
            failure_reason = failure[0]

        # 创建一个编译计数器对象 cc，使用指定的后端 "aot_eager" 进行优化
        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        # 使用 dynamo 库对 outer_func 进行优化，同时传入 guard_fail_fn 作为失败处理函数
        compiled_func = torch._dynamo.optimize(
            guard_fail_fn=guard_fail_fn,
            backend=cc,
        )(outer_func)

        # 断言编译后的函数与原始函数在相同输入下的输出一致
        self.assertEqual(compiled_func(inp), outer_func(inp))
        # 断言编译后的函数在相同输入下的标量值与预期值相同
        self.assertEqual(compiled_func(inp).item(), 15)

        # 断言编译计数器中记录的帧数与操作数符合预期
        # 编译了一个大图，包括 3 个函数，包括 hook。
        self.assertEqual(cc.frame_count, 1)
        self.assertEqual(cc.op_count, 6)

        # 移除 hook 后，断言需要重新编译
        handle.remove()
        self.assertEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 7)
        # 断言失败原因中包含 "forward_hooks"
        self.assertTrue("forward_hooks" in failure_reason)
        # 断言编译计数器中记录的帧数与操作数符合预期（增加了一次编译）
        self.assertEqual(cc.frame_count, 1 + 1)
        self.assertEqual(cc.op_count, 6 + 4)

        # 改变 hook 函数而不是移除，重新设置 TestModule 和 hook
        torch._dynamo.reset()
        m = TestModule()
        handle = m.register_forward_hook(forward_hook)
        failure_reason = None
        self.assertEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 15)

        # 定义一个新的 forward hook 函数
        def new_forward_hook(
            module: torch.nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor
        ) -> torch.Tensor:
            # 对输出进行简单的转换：返回输出的两倍加二
            return 2 * output + 2

        # 将新的 forward hook 函数赋值给 m 的 _forward_hooks
        m._forward_hooks[handle.id] = new_forward_hook
        self.assertEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 16)
        # 断言失败原因匹配特定正则表达式
        self.assertRegex(failure_reason, r"^___check_obj_id\(L\['m'\]._forward_hooks")
    # 定义一个测试方法，用于测试 hooks 的跳过保护机制
    def test_hooks_skip_guards(self):
        # 定义一个继承自 torch.nn.Module 的测试模块
        class TestModule(torch.nn.Module):
            # 重写 forward 方法，实现简单的计算逻辑：输入乘以2再加1
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 2 * x + 1

        # 创建 TestModule 的实例
        m = TestModule()

        # 定义一个 forward_hook 函数，用作 forward 方法的钩子
        def forward_hook(
            module: torch.nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor
        ) -> torch.Tensor:
            # 对输出进行操作：乘以2再加1
            return 2 * output + 1

        # 在模块 m 上注册 forward_hook
        handle = m.register_forward_hook(forward_hook)

        # 定义一个外部函数 outer_func，接收一个 tensor 作为参数
        def outer_func(tensor):
            # 对 tensor 进行操作：乘以2再加1
            x = tensor * 2 + 1
            # 将处理后的 tensor 输入模块 m 中进行计算
            y = m(x)
            # 返回计算结果 y
            return y

        # 创建一个输入 tensor，设置 requires_grad=True 以允许梯度计算
        inp = torch.tensor(1.0, requires_grad=True)

        # 初始化失败原因为 None
        failure_reason = None

        # 定义一个 guard_fail_fn 函数，用于处理守护失败的情况
        def guard_fail_fn(failure):
            nonlocal failure_reason
            # 将失败的原因设置为第一个失败信息
            failure_reason = failure[0]

        # 创建一个编译计数器 cc，使用 "aot_eager" 后端进行优化
        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        # 使用 dynamo 进行优化，传入 guard_fail_fn 函数和编译计数器 cc
        compiled_func = torch._dynamo.optimize(
            guard_fail_fn=guard_fail_fn,
            backend=cc,
        )(outer_func)

        # 重新创建 TestModule 的实例
        m = TestModule()
        # 再次注册 forward_hook
        handle = m.register_forward_hook(forward_hook)

        # 重置失败原因为 None
        failure_reason = None

        # 断言编译后的函数和原始函数 outer_func 的输出相等
        self.assertEqual(compiled_func(inp), outer_func(inp))
        # 断言编译后的函数输出的值为 15
        self.assertEqual(compiled_func(inp).item(), 15)
        # 断言编译计数器的帧计数为 1
        self.assertEqual(cc.frame_count, 1)
        # 断言编译计数器的操作计数为 6
        self.assertEqual(cc.op_count, 6)

        # 移除 hook 后，编译函数的输出不应与原始函数相等
        handle.remove()
        self.assertNotEqual(compiled_func(inp), outer_func(inp))
        # 断言编译后的函数输出的值为 15
        self.assertEqual(compiled_func(inp).item(), 15)
        # 断言编译计数器的帧计数为 1
        self.assertEqual(cc.frame_count, 1)

    # 定义一个辅助方法，用于测试 forward hook 功能
    def _forward_hook_test_helper(self, model):
        # 初始化 forward hook 的处理字典和编译后的激活字典
        forward_handles = {}
        compiled_activations = dict()
        eager_activations = dict()
        activations = None

        # 定义一个保存 activations 的函数，用于 forward hook
        def save_activations(name, mod, inp, out):
            activations[name] = inp

        # 遍历模型的所有模块，为每个模块注册 forward hook
        for name, module in model.named_modules():
            forward_handles[name] = module.register_forward_hook(
                partial(save_activations, name)
            )

        # 使用 torch.compile 方法对模型进行编译，使用 "aot_eager" 后端
        compiled_model = torch.compile(model, backend="aot_eager")

        # 将 activations 设置为 compiled_activations，执行两次循环
        activations = compiled_activations
        for i in range(2):
            # 第二次迭代是关键，第一次迭代期间已触发 hooks
            compiled_activations.clear()
            # 创建随机输入张量 x
            x = torch.randn((20, 10))
            # 使用编译后的模型进行预测
            pred = compiled_model(x)
            # 计算损失
            loss = pred.sum()
            # 反向传播计算梯度
            loss.backward()

        # 将 activations 设置为 eager_activations，执行两次循环
        activations = eager_activations
        for i in range(2):
            # 第二次迭代是关键，第一次迭代期间已触发 hooks
            eager_activations.clear()
            # 创建随机输入张量 x
            x = torch.randn((20, 10))
            # 使用原始模型进行预测
            pred = model(x)
            # 计算损失
            loss = pred.sum()
            # 反向传播计算梯度
            loss.backward()

        # 打印编译后记录的层名
        print(f"Recorded Layers: {compiled_activations.keys()}\n\n")
        # 打印预期的层名
        print(f"Expected Layers: {eager_activations.keys()}")

        # 断言编译后的 activations 和 eager_activations 的键集合相等
        self.assertTrue(compiled_activations.keys() == eager_activations.keys())
        # 断言 activations 的键集合与 forward_handles 的键集合相等
        self.assertTrue(activations.keys() == forward_handles.keys())
    # 测试允许的模块钩子
    def test_hooks_allowed_modules(self):
        # 此测试不应关心钩子保护是否启用
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个简单的神经网络模型
                self.net = torch.nn.Sequential(
                    *[torch.nn.Linear(10, 10000), torch.nn.ReLU()]
                    + [torch.nn.Linear(10000, 5), torch.nn.ReLU()]
                )

            def forward(self, x):
                return self.net(x)

        # 创建一个ToyModel实例
        model = ToyModel()
        # 调用辅助函数测试前向传播钩子
        self._forward_hook_test_helper(model)

    # 测试允许的模块钩子编译
    def test_hooks_allowed_modules_compiles(self):
        # 定义一个简单的神经网络模型
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    *[torch.nn.Linear(10, 10000), torch.nn.ReLU()]
                    + [torch.nn.Linear(10000, 5), torch.nn.ReLU()]
                )

            def forward(self, x):
                return self.net(x)

        # 创建一个ToyModel实例
        model = ToyModel()
        # 用于存储激活的列表
        activations = []

        # 定义一个保存激活的钩子函数
        def save_activations(mod, inp, out):
            activations.append(inp)

        # 遍历模型的所有模块，并注册前向传播钩子
        for name, module in model.named_modules():
            module.register_forward_hook(save_activations)

        # 创建一个编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()
        # 优化模型，使用JIT优化，无Python加速
        model = torch._dynamo.optimize(cnt, nopython=True)(model)

        # 进行两次迭代
        for i in range(2):
            # 第二次迭代很关键，因为在第一次迭代时钩子会在aot trace期间被激活
            # 清空激活列表
            activations.clear()
            # 创建随机输入张量
            x = torch.randn((20, 10))
            # 对模型进行前向传播
            pred = model(x)
            # 计算预测的总和作为损失
            loss = pred.sum()
            # 计算损失的反向传播
            loss.backward()

        # 断言激活列表的长度为6
        self.assertEqual(len(activations), 6)
        # 断言编译计数器的帧数为1
        self.assertEqual(cnt.frame_count, 1)
    def test_hooks_allowed_modules_compiles_self_contained(self):
        # 定义一个 ToyModel 类，继承自 torch.nn.Module
        class ToyModel(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 定义神经网络模型
                self.net = torch.nn.Sequential(
                    *[torch.nn.Linear(10, 10000), torch.nn.ReLU()]
                    + [torch.nn.Linear(10000, 5), torch.nn.ReLU()]
                )

            # 前向传播方法
            def forward(self, x):
                return self.net(x) * self.net(x)

        # 创建 ToyModel 实例
        model = ToyModel()
        # 初始化空字典，用于存储 forward hook
        forward_handles = {}

        # 定义输出修改的 hook 函数
        def output_modifying_hook(mod, inp, out):
            return 2 * out + 1

        # 遍历模型中的所有模块，为每个模块注册 forward hook
        for name, module in model.named_modules():
            forward_handles[name] = module.register_forward_hook(output_modifying_hook)

        # 创建 CompileCounter 实例
        cnt = torch._dynamo.testing.CompileCounter()

        # 生成输入数据 x，并进行模型的前向传播计算
        x = torch.randn((20, 10))
        pred_eager = model(x)
        # 计算预测值的损失
        loss_eager = pred_eager.sum()
        # 计算损失的梯度
        eager_loss_bwd = loss_eager.backward()

        # 对模型进行优化和编译
        model = torch._dynamo.optimize(cnt, nopython=True)(model)
        # 使用优化后的模型进行预测
        pred = model(x)

        # 计算预测值的损失
        loss = pred.sum()
        # 计算损失的梯度
        loss_bwd = loss.backward()

        # 断言预测值损失的反向传播结果与优化后模型的损失反向传播结果相等
        self.assertEqual(eager_loss_bwd, loss_bwd)
        # 断言 CompileCounter 的帧数为 2
        self.assertEqual(cnt.frame_count, 2)

        # Ndim 改变，重新编译模型
        pred = model(torch.randn([10, 10, 10]))
        # 断言 CompileCounter 的帧数为 4
        self.assertEqual(cnt.frame_count, 4)

        # 稳定状态，再次进行预测
        pred = model(torch.randn([10, 10, 10]))
        # 断言 CompileCounter 的帧数为 4，保持不变
        self.assertEqual(cnt.frame_count, 4)

    def test_dunder_call_explicitly(self):
        # 如果显式调用 __call__ 方法，hook 应该被触发
        class ToyModel(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 定义一个线性层
                self.linear = torch.nn.Linear(10, 10000)

            # 前向传播方法
            def forward(self, x):
                # 显式调用线性层的 __call__ 方法
                return self.linear.__call__(x)

        # 创建 ToyModel 实例
        model = ToyModel()
        # 调用辅助函数进行 forward hook 测试
        self._forward_hook_test_helper(model)
    def test_backward_hooks(self):
        # 测试不应关心挂钩保护是否启用

        class CustomLinear(torch.nn.Module):
            # 不是“允许的模块”，因此不应该中断图
            def __init__(self, a, b):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(a, b))

            def forward(self, x):
                return torch.mm(x, self.weight)

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    *[CustomLinear(10, 10)]
                    + [CustomLinear(10, 10000)]
                    + [CustomLinear(10000, 5)]
                )

            def forward(self, x):
                return self.net(x)

        model = ToyModel()
        backward_hook_handles = {}
        pre_backward_hook_handles = {}

        grad_sizes = {}

        def backward_hook(name, mod, grad_inp, grad_out):
            grad_sizes[name] = (
                (gi.shape for gi in grad_inp),
                (go.shape for go in grad_out),
            )
            return None

        pre_grad_sizes = {}

        def backward_pre_hook(name, mod, grad_out):
            pre_grad_sizes[name] = (go.shape for go in grad_out)
            return None

        for name, module in model.named_modules():
            # 注册完整的反向传播后钩子函数
            backward_hook_handles[name] = module.register_full_backward_hook(
                partial(backward_hook, name)
            )

            # 注册完整的反向传播前钩子函数
            pre_backward_hook_handles[name] = module.register_full_backward_pre_hook(
                partial(backward_pre_hook, name)
            )

        # 编译模型以便即时追踪
        model = torch.compile(model, backend="aot_eager")

        for i in range(2):
            # 第二次迭代很关键，挂钩将在第一次迭代的 aot 追踪期间触发
            x = torch.randn((20, 10))
            pred = model(x)
            loss = pred.sum()
            loss.backward()

        self.assertTrue(grad_sizes.keys() == backward_hook_handles.keys())
        self.assertTrue(pre_grad_sizes.keys() == pre_backward_hook_handles.keys())
    # 定义测试函数，用于测试将实例方法作为钩子函数的场景
    def test_udo_instance_method_as_hook(self):
        # 定义自定义类 CustomClass
        class CustomClass:
            # 初始化方法，接收模块参数并注册前置钩子函数
            def __init__(self, module):
                self.module = module
                # 注册前置钩子函数 func1 到模块，使用关键字参数 prepend 和 with_kwargs
                self.handle = self.module.register_forward_pre_hook(
                    self.func1, prepend=True, with_kwargs=True
                )

            # 钩子函数 func1，接收模块、参数和关键字参数作为参数，返回修改后的参数和关键字参数
            def func1(self, module, args, kwargs):
                return (args[0] + 1,), kwargs

            # 实现 __call__ 方法，使实例对象可以像函数一样被调用，调用模块的方法
            def __call__(self, x):
                return self.module(x)

        # 定义 ToyModel 类，继承自 torch.nn.Module
        class ToyModel(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()

            # 前向传播方法，计算输入张量的平方
            def forward(self, x):
                return x * x

        # 创建 ToyModel 的实例 model
        model = ToyModel()
        # 创建形状为 (3, 4) 的全零张量 x
        x = torch.zeros((3, 4))
        # 创建 CustomClass 的实例 obj，传入 ToyModel 实例 model
        obj = CustomClass(model)
        # 编译 obj 的调用，使用完整图形式
        out = torch.compile(obj, fullgraph=True)(x)
        # 断言输出结果 out 与 (x + 1) * (x + 1) 相等
        self.assertEqual(out, (x + 1) * (x + 1))

    # 定义测试函数，测试在 ModuleDict 中迭代名称的场景
    def test_module_dict_iter_name(self):
        # 定义 MyModule 类，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建 ModuleDict activations，包含 "lrelu" 和 "prelu" 激活函数模块
                self.activations = torch.nn.ModuleDict(
                    [["lrelu", torch.nn.LeakyReLU()], ["prelu", torch.nn.PReLU()]]
                )

            # 前向传播方法，使用 ModuleDict activations 中的激活函数对输入进行处理
            def forward(self, x):
                for activation_name in self.activations:
                    x = self.activations[activation_name](x)
                return x

        # 创建 CompileCounter 的实例 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 直接执行前向传播，即时模式
        eager_res = MyModule()(torch.ones(10, 10))
        # 编译模块并执行前向传播
        optim_res = torch._dynamo.optimize(cnt)(MyModule())(torch.ones(10, 10))
        # 断言即时模式和优化模式的输出结果相等
        self.assertEqual(eager_res, optim_res)
        # 断言帧计数器的计数为 1
        self.assertEqual(cnt.frame_count, 1)

    # 定义测试函数，测试在 ModuleDict 中迭代键名的场景
    def test_module_dict_iter_keys(self):
        # 定义 MyModule 类，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建 ModuleDict activations，包含 "lrelu" 和 "prelu" 激活函数模块
                self.activations = torch.nn.ModuleDict(
                    [["lrelu", torch.nn.LeakyReLU()], ["prelu", torch.nn.PReLU()]]
                )

            # 前向传播方法，使用 ModuleDict activations 中的激活函数对输入进行处理
            def forward(self, x):
                for activation_name in self.activations.keys():
                    x = self.activations[activation_name](x)
                return x

        # 创建 CompileCounter 的实例 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 直接执行前向传播，即时模式
        eager_res = MyModule()(torch.ones(10, 10))
        # 编译模块并执行前向传播
        optim_res = torch._dynamo.optimize(cnt)(MyModule())(torch.ones(10, 10))
        # 断言即时模式和优化模式的输出结果相等
        self.assertEqual(eager_res, optim_res)
        # 断言帧计数器的计数为 1
        self.assertEqual(cnt.frame_count, 1)

    # 定义测试函数，测试设置模型属性的场景
    def test_module_setattr(self):
        # 创建包含单个线性层的 Sequential 模型 models
        models = torch.nn.Sequential(torch.nn.Linear(3, 3))
        # 设置线性层的属性 abc 为 False
        models[0].abc = False

        # 定义函数 run，设置线性层的属性 abc 为 True，并执行模型的前向传播
        def run():
            models[0].abc = True
            x = torch.randn(1, 3)
            return models(x)

        # 编译 run 函数，使用完整图形式
        run = torch.compile(run, fullgraph=True)
        # 执行编译后的 run 函数
        run()
        # 断言线性层的属性 abc 为 True
        self.assertTrue(models[0].abc)
    def test_assign_does_not_exist(self):
        # 定义一个测试方法，验证在神经网络模块中执行赋值操作后，属性不存在的情况
        class MyModule(torch.nn.Module):
            def forward(self, x):
                # 在模块的前向传播中，将输入 x 加 1，并赋值给模块的属性 text_encoding
                self.text_encoding = x + 1
                return self.text_encoding
        
        # 创建 MyModule 类的实例
        mod = MyModule()
        # 使用 Torch 的编译器对模块进行编译，fullgraph=True 用于指示生成完整的图形表示
        out = torch.compile(mod, fullgraph=True)(torch.randn(10))
        # 断言模块的 text_encoding 属性与编译结果 out 相等
        assert mod.text_encoding is out

    def test_module_dict_iter_values(self):
        # 定义一个测试方法，验证模块字典的迭代值的功能
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化模块的 activations 属性为 ModuleDict，包含两个激活函数模块
                self.activations = torch.nn.ModuleDict(
                    [["lrelu", torch.nn.LeakyReLU()], ["prelu", torch.nn.PReLU()]]
                )

            def forward(self, x):
                # 在模块的前向传播中，对 activations 字典中的每个值执行激活函数操作
                for activation in self.activations.values():
                    x = activation(x)
                return x

        # 创建一个 Torch 的计数器实例
        cnt = torch._dynamo.testing.CompileCounter()
        # 在 Eager 模式下执行 MyModule 的前向传播
        eager_res = MyModule()(torch.ones(10, 10))

        # 编译优化模块
        optim_res = torch._dynamo.optimize(cnt)(MyModule())(torch.ones(10, 10))
        # 断言 Eager 模式和编译优化模式下的输出结果一致
        self.assertEqual(eager_res, optim_res)
        # 断言计数器的帧计数为 1
        self.assertEqual(cnt.frame_count, 1)

    def test_unspecialized_seq(self):
        # 定义一个测试方法，验证未特化的序列模块的行为
        models = torch.nn.Sequential(torch.nn.Linear(3, 3))

        def fn(x):
            # 将序列模块的第一个子模块设为非训练模式
            models[0].training = False
            return models(x)

        # 使用 Torch 的编译器对 fn 函数进行优化
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 创建输入张量 x
        x = torch.randn(1, 3)
        # 分别计算原始函数和优化函数的结果
        ref = fn(x)
        res = opt_fn(x)
        # 断言原始函数和优化函数的结果相等
        self.assertEqual(ref, res)

    def test_no_op_assignment(self):
        # 定义一个测试方法，验证在模块中不会执行任何操作的赋值行为
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化模块的 buffer 属性为随机张量
                self.buffer = torch.rand([4])

            def forward(self, x):
                # 这是一个不执行操作的赋值操作，但会导致 dynamo 丢失静态输入
                x = x + 1
                # 将模块的 buffer 属性转换为输入 x 的类型
                self.buffer = self.buffer.to(x)
                return self.buffer + x

        # 初始化未使用 buffer 的编译计数器
        compiles_without_buffers = 0

        def debug_compile(gm, *args, **kwargs):
            nonlocal compiles_without_buffers
            # 检查是否编译时没有使用 buffer
            compiles_without_buffers += len(list(gm.buffers())) == 0
            return gm

        # 使用 debug_compile 函数装饰 foo 函数
        @torch.compile(backend=debug_compile)
        def foo(mod, x):
            return mod(x)

        # 创建 Mod 类的实例
        mod = Mod()
        foo(mod, torch.rand([4]))
        # 如果 inline_inbuilt_nn_modules 配置为真，则断言编译时没有使用 buffer 的次数为 1
        if torch._dynamo.config.inline_inbuilt_nn_modules:
            self.assertEqual(compiles_without_buffers, 1)
        else:
            self.assertEqual(compiles_without_buffers, 0)

        foo(mod, torch.rand([4], dtype=torch.half))
        # 如果 inline_inbuilt_nn_modules 配置为真，则断言编译时没有使用 buffer 的次数为 2
        if torch._dynamo.config.inline_inbuilt_nn_modules:
            self.assertEqual(compiles_without_buffers, 2)
        else:
            self.assertEqual(compiles_without_buffers, 1)

        class Mod2(Mod):
            def __setattr__(self, name, value):
                return super().__setattr__(name, value)

        foo(Mod2(), torch.rand([4]))
        # 断言至少有两次编译，因为 setattr 方法未实现
        self.assertTrue(compiles_without_buffers >= 2)
    def test_unspec_non_inlinable_module(self):
        # 创建一个未指定为可内联的模块实例
        mod = UnspecNonInlinableModule()
        # 使用 torch._dynamo.optimize("eager") 对模块进行优化
        opt_fn = torch._dynamo.optimize("eager")(mod)
        # 生成随机输入张量 x
        x = torch.randn(100)
        # 使用优化后的函数处理输入张量 x
        actual = opt_fn(x)
        # 获取模块处理输入张量 x 的期望输出
        expected = mod(x)
        # 断言优化后的实际输出与期望输出相等
        self.assertEqual(actual, expected)

    def test_no_guard_on_torch_nn_modules(self):
        # 创建一个模拟的 torch.nn.Module 类
        # 解决 https://github.com/pytorch/pytorch/issues/110048
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.multiplier = 10

            def forward(self, x):
                # 模块前向传播方法的实现
                return self.linear(x) * self.multiplier

        mod = MockModule()

        # 创建一个编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def generate(x, c):
            # 使用模块处理输入 x，并加上常数 c
            return mod(x) + c

        # 多次调用生成函数，检查编译计数是否增加
        for _ in range(0, 10):
            generate(torch.randn(10, 10), 0)
            generate(torch.randn(10, 10), 1)
        # 断言帧计数是否为 2
        self.assertEqual(cnt.frame_count, 2)

        # 修改模块属性 multiplier，期望导致重新编译
        mod.multiplier = 11
        generate(torch.randn(10, 10), 0)
        # 断言帧计数是否增加到 3
        self.assertEqual(cnt.frame_count, 3)

    def test_setattr_on_compiled_module(self):
        # 创建一个重播突变的 torch.nn.Module 子类
        # 解决 https://github.com/pytorch/pytorch/issues/114844
        class ReplayMutation(torch.nn.Module):
            def __init__(self, inp_size, out_size, inner_size):
                super().__init__()
                self.Linear1 = torch.nn.Linear(inp_size, inner_size)
                self.Linear2 = torch.nn.Linear(inner_size, out_size)
                self.x = None

            def forward(self, inp):
                # 前向传播方法，保存 Linear1 处理的结果到 self.x
                res = self.Linear1(inp)
                self.x = res
                return self.Linear2(res)

        N, D_in, H, D_out, inner = 2, 2, 2, 2, 4
        model = ReplayMutation(D_in, H, inner)
        model2 = copy.deepcopy(model)
        input = torch.ones(N, D_in)

        # 将一些中间值保存到 model.x
        model.x = torch.tensor([[100, 100, 100, 100], [200, 200, 200, 200]])
        model(input)

        # 编译模型为 eager 后端
        compiled_model = torch.compile(model2, backend="eager")
        # 设置编译模型的属性 x
        compiled_model.x = torch.tensor([[100, 100, 100, 100], [200, 200, 200, 200]])
        compiled_model(input)

        # 断言原始模型和编译模型的属性 x 相等
        self.assertEqual(model.x, compiled_model.x)

    def test_globals_change_in_other_file(self):
        # 使用 torch.compile 对函数 fn 进行编译，全图模式
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # 更新全局变量
            update_global()
            # 调用另一个文件中的函数 update_global，并将结果赋给 a
            a = test_functions.update_global(x)
            # 确保更新的全局值被正确读取
            return x * a * (_variable + _variable1 + test_functions._variable)

        # 调用函数 fn，并检查返回结果
        res = fn(torch.ones(10))
        # 断言全局变量 _variable 和 _variable1 的值为 1
        self.assertEqual(_variable, 1)
        self.assertEqual(_variable1, 1)
        # 确保重构的字节码更新了另一个文件中的全局变量值
        self.assertEqual(test_functions._variable, 1)
        # 断言函数 fn 的返回结果
        self.assertEqual(res, 3 * torch.ones(10))
    # 使用 unittest 模块中的 skipIf 装饰器，如果条件不满足则跳过测试
    @unittest.skipIf(
        "inductor" not in torch._dynamo.list_backends(),
        "inductor backend is not available",
    )
    # 定义测试函数，测试保存和加载 inductor 后端的模型
    def test_save_and_load_inductor(self):
        # 创建 MockModule 实例
        mod = MockModule()
        # 使用 inductor 后端编译模型
        opt_mod = torch.compile(mod, backend="inductor")
        # 创建输入张量
        inp = torch.randn(10, 10)
        # 在编译后的模型上执行输入
        opt_mod(inp)

        # 使用临时目录进行模型保存和加载
        with tempfile.TemporaryDirectory() as tmpdirname:
            # 将编译后的模型保存到临时目录中
            torch.save(opt_mod, os.path.join(tmpdirname, "model.pt"))
            # 加载保存的模型
            loaded_model = torch.load(os.path.join(tmpdirname, "model.pt"))
        
        # 在加载的模型上再次执行输入
        loaded_model(inp)
        # 断言两个模型是否相同
        self.assertTrue(same_two_models(loaded_model, mod, [inp]))
        self.assertTrue(same_two_models(loaded_model, opt_mod, [inp]))

        # 重置动态编译环境
        torch._dynamo.reset()  # 强制重新编译
        # 重置 inductor 后端的度量值
        torch._inductor.metrics.generated_kernel_count = 0
        # 再次执行加载后的模型输入
        loaded_model(inp)
        # 断言生成的内核数大于零
        self.assertGreater(torch._inductor.metrics.generated_kernel_count, 0)

    # 定义测试函数，测试保存和加载所有后端的模型
    def test_save_and_load_all_backends(self):
        # 创建 MockModule 实例
        mod = MockModule()
        # 创建输入张量
        inp = torch.randn(10, 10)
        # 遍历所有动态编译后端
        for backend in torch._dynamo.list_backends():
            try:
                # 使用当前后端编译模型
                opt_mod = torch.compile(mod, backend=backend)
                # 使用临时目录进行模型保存和加载
                with tempfile.TemporaryDirectory() as tmpdirname:
                    # 将编译后的模型保存到临时目录中
                    torch.save(opt_mod, os.path.join(tmpdirname, "model.pt"))
                    # 加载保存的模型
                    loaded_model = torch.load(os.path.join(tmpdirname, "model.pt"))
                
                # 重置动态编译环境
                torch._dynamo.reset()  # 强制重新编译
                # 重置 inductor 后端的度量值
                torch._inductor.metrics.generated_kernel_count = 0
                # 在编译后的模型上执行输入
                opt_mod(inp)
                # 检查编译后是否没有生成内核
                opt_success = torch._inductor.metrics.generated_kernel_count == 0
                # 重置动态编译环境
                torch._dynamo.reset()  # 强制重新编译
                # 重置 inductor 后端的度量值
                torch._inductor.metrics.generated_kernel_count = 0
                # 在加载后的模型上执行输入
                loaded_model(inp)
                # 检查加载后是否没有生成内核
                loaded_success = torch._inductor.metrics.generated_kernel_count == 0
                # 断言编译后和加载后的模型生成内核情况是否一致
                self.assertEqual(opt_success, loaded_success)
            except torch._dynamo.exc.BackendCompilerFailed:
                # 如果后端编译失败，则跳过当前后端的测试
                pass
    def test_monkeypatching_forward(self):
        # 定义一个名为 test_monkeypatching_forward 的测试方法
        class FakeModule(torch.nn.Module):
            # 定义一个名为 FakeModule 的类，继承自 torch.nn.Module
            def forward(self, x):
                # 重写 forward 方法，返回输入 x 的正弦值
                return torch.sin(x)

        class MyModule(torch.nn.Module):
            # 定义一个名为 MyModule 的类，继承自 torch.nn.Module
            def __init__(self, x):
                # 初始化方法，继承自父类，并接受参数 x
                super().__init__()

            def forward(self, x):
                # 重写 forward 方法，返回输入 x 的余弦值
                return torch.cos(x)

        def helper():
            # 定义一个名为 helper 的辅助函数
            torch._dynamo.reset()
            # 重置 torch._dynamo 模块状态

            mod = MyModule(3)
            # 创建 MyModule 类的实例 mod，参数为 3

            def fn(x):
                # 定义一个名为 fn 的函数，接受参数 x
                return mod(x)

            cnt = torch._dynamo.testing.CompileCounter()
            # 创建 CompileCounter 实例 cnt

            opt_fn = torch._dynamo.optimize(cnt)(fn)
            # 使用 torch._dynamo.optimize 对 fn 进行优化，返回优化后的函数 opt_fn

            x = torch.randn(10)
            # 创建一个形状为 (10,) 的随机张量 x

            opt_fn(x)
            # 调用优化后的函数 opt_fn，传入 x
            opt_fn(x)
            # 再次调用优化后的函数 opt_fn，传入 x
            self.assertEqual(cnt.frame_count, 1)
            # 使用断言检查 cnt 的帧计数是否为 1

            # Monkeypatch forward
            # Monkeypatch（猴子补丁）的方式覆盖 mod 的 forward 方法
            mod.forward = types.MethodType(FakeModule.forward, mod)

            ref = fn(x)
            # 调用 fn 函数，传入 x，将结果赋给 ref
            res = opt_fn(x)
            # 再次调用优化后的函数 opt_fn，传入 x，将结果赋给 res
            self.assertEqual(ref, res)
            # 使用断言检查 ref 和 res 是否相等
            self.assertEqual(cnt.frame_count, 2)
            # 使用断言检查 cnt 的帧计数是否为 2

        helper()
        # 调用 helper 函数

        with torch._dynamo.config.patch(inline_inbuilt_nn_modules=True):
            # 使用上下文管理器，启用内置 nn 模块的内联优化
            helper()
            # 调用 helper 函数
# 如果该模块是直接运行的主程序入口
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块中导入 run_tests 函数
    from torch._dynamo.test_case import run_tests
    
    # 执行 run_tests 函数，用于运行测试用例
    run_tests()
```