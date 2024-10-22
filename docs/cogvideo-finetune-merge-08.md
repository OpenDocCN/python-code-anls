# CogVideo & CogVideoX 微调代码源码解析（九）



# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\guiders.py`

```py
# 导入 logging 模块以进行日志记录
import logging
# 从 abc 模块导入 ABC 和 abstractmethod 以定义抽象基类和抽象方法
from abc import ABC, abstractmethod
# 从 typing 模块导入用于类型注解的各种类型
from typing import Dict, List, Optional, Tuple, Union
# 从 functools 导入 partial 用于部分函数应用
from functools import partial
# 导入 math 模块以进行数学运算
import math

# 导入 PyTorch 库
import torch
# 从 einops 导入 rearrange 和 repeat 用于重排和重复张量
from einops import rearrange, repeat

# 从上级目录的 util 模块导入 append_dims、default 和 instantiate_from_config 函数
from ...util import append_dims, default, instantiate_from_config


# 定义一个抽象基类 Guider，继承自 ABC
class Guider(ABC):
    # 定义一个抽象方法 __call__，接收张量和浮点数，并返回一个张量
    @abstractmethod
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        pass

    # 定义准备输入的方法，接收张量、浮点数和字典，并返回一个元组
    def prepare_inputs(self, x: torch.Tensor, s: float, c: Dict, uc: Dict) -> Tuple[torch.Tensor, float, Dict]:
        pass


# 定义 VanillaCFG 类，实现并行化 CFG
class VanillaCFG:
    """
    implements parallelized CFG
    """

    # 初始化方法，接收缩放因子和可选的动态阈值配置
    def __init__(self, scale, dyn_thresh_config=None):
        # 设置缩放因子
        self.scale = scale
        # 定义缩放调度函数，独立于步骤
        scale_schedule = lambda scale, sigma: scale  # independent of step
        # 使用 partial 绑定缩放调度函数
        self.scale_schedule = partial(scale_schedule, scale)
        # 根据配置实例化动态阈值
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {"target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"},
            )
        )

    # 定义调用方法，接收张量、浮点数和可选缩放因子
    def __call__(self, x, sigma, scale=None):
        # 将输入张量 x 拆分为上下文和未上下文部分
        x_u, x_c = x.chunk(2)
        # 获取缩放值，优先使用传入的缩放因子
        scale_value = default(scale, self.scale_schedule(sigma))
        # 使用动态阈值处理未上下文和上下文部分
        x_pred = self.dyn_thresh(x_u, x_c, scale_value)
        # 返回预测结果
        return x_pred

    # 定义准备输入的方法
    def prepare_inputs(self, x, s, c, uc):
        # 创建一个空字典用于存储输出上下文
        c_out = dict()

        # 遍历上下文字典 c
        for k in c:
            # 如果键在特定列表中，则连接 uc 和 c 中的对应张量
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                # 否则，确保 c 和 uc 中的值相同，并将其直接赋值
                assert c[k] == uc[k]
                c_out[k] = c[k]
        # 返回两个相同的张量 x 和 s，以及输出上下文字典
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out


# 定义 DynamicCFG 类，继承自 VanillaCFG
class DynamicCFG(VanillaCFG):
    # 初始化方法，接收缩放因子、指数、步骤数和可选的动态阈值配置
    def __init__(self, scale, exp, num_steps, dyn_thresh_config=None):
        # 调用父类的初始化方法
        super().__init__(scale, dyn_thresh_config)
        # 定义动态缩放调度函数
        scale_schedule = (
            lambda scale, sigma, step_index: 1 + scale * (1 - math.cos(math.pi * (step_index / num_steps) ** exp)) / 2
        )
        # 使用 partial 绑定缩放调度函数
        self.scale_schedule = partial(scale_schedule, scale)
        # 根据配置实例化动态阈值
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {"target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"},
            )
        )

    # 定义调用方法，接收张量、浮点数、步骤索引和可选缩放因子
    def __call__(self, x, sigma, step_index, scale=None):
        # 将输入张量 x 拆分为上下文和未上下文部分
        x_u, x_c = x.chunk(2)
        # 获取缩放值，使用动态缩放调度函数
        scale_value = self.scale_schedule(sigma, step_index.item())
        # 使用动态阈值处理未上下文和上下文部分
        x_pred = self.dyn_thresh(x_u, x_c, scale_value)
        # 返回预测结果
        return x_pred


# 定义 IdentityGuider 类
class IdentityGuider:
    # 定义调用方法，接收张量和浮点数，直接返回输入张量
    def __call__(self, x, sigma):
        return x

    # 定义准备输入的方法
    def prepare_inputs(self, x, s, c, uc):
        # 创建一个空字典用于存储输出上下文
        c_out = dict()

        # 遍历上下文字典 c
        for k in c:
            # 将上下文中的值直接赋值到输出上下文中
            c_out[k] = c[k]

        # 返回原始张量 x、s 和输出上下文字典
        return x, s, c_out
```

# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\lora.py`

```py
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 从 typing 模块导入需要的类型注解
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

# 导入 PyTorch 库
import torch
import torch.nn.functional as F
from torch import nn

# 定义一个名为 LoRALinearLayer 的类，继承自 nn.Module
class LoRALinearLayer(nn.Module):
    # 初始化方法，定义输入、输出特征及其他参数
    def __init__(self, in_features, out_features, rank=4, network_alpha=None, device=None, dtype=None):
        # 调用父类的初始化方法
        super().__init__()

        # 定义一个线性层，用于将输入特征降维到指定的秩（rank）
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        # 定义另一个线性层，用于将降维后的特征映射到输出特征
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # 保存网络缩放因子，类似于训练脚本中的 --network_alpha 选项
        self.network_alpha = network_alpha
        # 保存秩（rank）值
        self.rank = rank
        # 保存输出特征数
        self.out_features = out_features
        # 保存输入特征数
        self.in_features = in_features

        # 用正态分布初始化 down 层的权重
        nn.init.normal_(self.down.weight, std=1 / rank)
        # 将 up 层的权重初始化为零
        nn.init.zeros_(self.up.weight)

    # 前向传播方法，定义如何通过层传递数据
    def forward(self, hidden_states):
        # 保存输入数据的原始数据类型
        orig_dtype = hidden_states.dtype
        # 获取 down 层权重的数据类型
        dtype = self.down.weight.dtype

        # 将输入数据通过 down 层进行降维
        down_hidden_states = self.down(hidden_states.to(dtype))
        # 将降维后的数据通过 up 层映射到输出特征
        up_hidden_states = self.up(down_hidden_states)

        # 如果设置了网络缩放因子，则对输出进行缩放
        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        # 返回转换为原始数据类型的输出
        return up_hidden_states.to(orig_dtype)

# 定义一个名为 LoRAConv2dLayer 的类，继承自 nn.Module
class LoRAConv2dLayer(nn.Module):
    # 初始化方法，定义输入、输出特征及其他参数
    def __init__(
        self, in_features, out_features, rank=4, kernel_size=(1, 1), stride=(1, 1), padding=0, network_alpha=None
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 定义一个卷积层，用于将输入特征降维到指定的秩（rank）
        self.down = nn.Conv2d(in_features, rank, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        # 定义另一个卷积层，用于将降维后的特征映射到输出特征，kernel_size 固定为 (1, 1)
        self.up = nn.Conv2d(rank, out_features, kernel_size=(1, 1), stride=(1, 1), bias=False)

        # 保存网络缩放因子，类似于训练脚本中的 --network_alpha 选项
        self.network_alpha = network_alpha
        # 保存秩（rank）值
        self.rank = rank

        # 用正态分布初始化 down 层的权重
        nn.init.normal_(self.down.weight, std=1 / rank)
        # 将 up 层的权重初始化为零
        nn.init.zeros_(self.up.weight)
    # 定义前向传播方法，接收隐藏状态作为输入
        def forward(self, hidden_states):
            # 保存输入隐藏状态的数据类型
            orig_dtype = hidden_states.dtype
            # 获取降维层权重的数据类型
            dtype = self.down.weight.dtype
    
            # 将输入的隐藏状态转换为降维层的数据类型，并进行降维处理
            down_hidden_states = self.down(hidden_states.to(dtype))
            # 对降维后的隐藏状态进行上采样处理
            up_hidden_states = self.up(down_hidden_states)
    
            # 如果网络的alpha参数不为None，则根据rank调整上采样结果
            if self.network_alpha is not None:
                up_hidden_states *= self.network_alpha / self.rank
    
            # 将上采样结果转换回原始数据类型并返回
            return up_hidden_states.to(orig_dtype)
# 定义一个兼容 LoRA 的卷积层，继承自 nn.Conv2d
class LoRACompatibleConv(nn.Conv2d):
    """
    A convolutional layer that can be used with LoRA.
    """

    # 初始化方法，接收可变参数以及 LoRA 层和缩放因子
    def __init__(self, *args, lora_layer: Optional[LoRAConv2dLayer] = None, scale: float = 1.0, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 存储 LoRA 层
        self.lora_layer = lora_layer
        # 存储缩放因子
        self.scale = scale

    # 设置 LoRA 层的方法
    def set_lora_layer(self, lora_layer: Optional[LoRAConv2dLayer]):
        # 更新 LoRA 层
        self.lora_layer = lora_layer

    # 融合 LoRA 层的方法
    def _fuse_lora(self, lora_scale=1.0):
        # 如果没有 LoRA 层，则返回
        if self.lora_layer is None:
            return

        # 获取权重的数据类型和设备
        dtype, device = self.weight.data.dtype, self.weight.data.device

        # 将原始权重转换为浮点数
        w_orig = self.weight.data.float()
        # 获取 LoRA 层上和下的权重并转换为浮点数
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()

        # 如果网络的 alpha 值不为空，则调整上权重
        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        # 计算上和下权重的融合矩阵
        fusion = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
        # 将融合矩阵调整为原始权重的形状
        fusion = fusion.reshape((w_orig.shape))
        # 计算最终的融合权重
        fused_weight = w_orig + (lora_scale * fusion)
        # 更新当前层的权重
        self.weight.data = fused_weight.to(device=device, dtype=dtype)

        # 融合后可以丢弃 LoRA 层
        self.lora_layer = None

        # 将上和下权重矩阵移到 CPU，以节省内存
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        # 存储当前的 lora_scale
        self._lora_scale = lora_scale

    # 解除融合 LoRA 层的方法
    def _unfuse_lora(self):
        # 检查是否有上和下权重
        if not (hasattr(self, "w_up") and hasattr(self, "w_down")):
            return

        # 获取当前权重数据
        fused_weight = self.weight.data
        dtype, device = fused_weight.data.dtype, fused_weight.data.device

        # 将上和下权重转换到当前设备并转为浮点数
        self.w_up = self.w_up.to(device=device).float()
        self.w_down = self.w_down.to(device).float()

        # 计算上和下权重的融合矩阵
        fusion = torch.mm(self.w_up.flatten(start_dim=1), self.w_down.flatten(start_dim=1))
        # 将融合矩阵调整为当前权重的形状
        fusion = fusion.reshape((fused_weight.shape))
        # 计算解除融合后的权重
        unfused_weight = fused_weight.float() - (self._lora_scale * fusion)
        # 更新当前层的权重
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)

        # 清除上和下权重
        self.w_up = None
        self.w_down = None

    # 前向传播方法
    def forward(self, hidden_states, scale: float = None):
        # 如果没有提供缩放因子，则使用默认值
        if scale is None:
            scale = self.scale
        # 如果没有 LoRA 层，使用普通卷积函数
        if self.lora_layer is None:
            # 调用功能性卷积函数以避免图形破坏
            # 参考链接: https://github.com/huggingface/diffusers/pull/4315
            return F.conv2d(
                hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        # 如果有 LoRA 层，则返回结合后的输出
        else:
            return super().forward(hidden_states) + (scale * self.lora_layer(hidden_states))


# 定义一个兼容 LoRA 的线性层，继承自 nn.Linear
class LoRACompatibleLinear(nn.Linear):
    """
    A Linear layer that can be used with LoRA.
    """

    # 初始化方法，接收可变参数以及 LoRA 层和缩放因子
    def __init__(self, *args, lora_layer: Optional[LoRALinearLayer] = None, scale: float = 1.0, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 存储 LoRA 层
        self.lora_layer = lora_layer
        # 存储缩放因子
        self.scale = scale
    # 设置 LoRALinearLayer 属性
    def set_lora_layer(self, lora_layer: Optional[LoRALinearLayer]):
        self.lora_layer = lora_layer

    # 融合 LoRALinearLayer
    def _fuse_lora(self, lora_scale=1.0):
        # 如果没有 LoRALinearLayer，则直接返回
        if self.lora_layer is None:
            return

        # 获取权重数据的数据类型和设备
        dtype, device = self.weight.data.dtype, self.weight.data.device

        # 将原始权重数据转换为浮点型
        w_orig = self.weight.data.float()
        # 获取 LoRALinearLayer 的上行权重数据和下行权重数据
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()

        # 如果 LoRALinearLayer 的网络 alpha 不为空，则对上行权重数据进行缩放
        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        # 计算融合后的权重数据
        fused_weight = w_orig + (lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])
        # 将融合后的权重数据转换为指定的设备和数据类型
        self.weight.data = fused_weight.to(device=device, dtype=dtype)

        # 现在可以丢弃 LoRALinearLayer 了
        self.lora_layer = None

        # 将上行和下行矩阵转移到 CPU 上，以避免内存溢出
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self._lora_scale = lora_scale

    # 取消融合 LoRALinearLayer
    def _unfuse_lora(self):
        # 如果没有保存上行和下行矩阵，则直接返回
        if not (hasattr(self, "w_up") and hasattr(self, "w_down")):
            return

        # 获取融合后的权重数据
        fused_weight = self.weight.data
        # 获取权重数据的数据类型和设备
        dtype, device = fused_weight.dtype, fused_weight.device

        # 将上行和下行矩阵转换为指定的设备和数据类型
        w_up = self.w_up.to(device=device).float()
        w_down = self.w_down.to(device=device).float()

        # 计算取消融合后的权重数据
        unfused_weight = fused_weight.float() - (self._lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])
        # 将取消融合后的权重数据转换为指定的设备和数据类型
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)

        # 清空上行和下行矩阵
        self.w_up = None
        self.w_down = None

    # 前向传播函数
    def forward(self, hidden_states, scale: float = None):
        # 如果没有指定缩放因子，则使用默认的缩放因子
        if scale is None:
            scale = self.scale
        # 如果没有 LoRALinearLayer，则直接调用父类的前向传播函数
        if self.lora_layer is None:
            out = super().forward(hidden_states)
            return out
        else:
            # 调用父类的前向传播函数，并加上 LoRALinearLayer 的输出结果
            out = super().forward(hidden_states) + (scale * self.lora_layer(hidden_states))
            return out
# 定义一个查找子模块的函数，接受一个模型和一个可选的模块类列表
def _find_children(
    model,
    search_class: List[Type[nn.Module]] = [nn.Linear],
):
    """
    查找特定类（或类的集合）的所有模块。

    返回所有匹配的模块及其父模块和引用的名称。
    """
    # 对于模型中的每个父模块，查找所有指定类的子模块
    for parent in model.modules():
        # 获取父模块的所有命名子模块
        for name, module in parent.named_children():
            # 检查子模块是否属于指定的类
            if any([isinstance(module, _class) for _class in search_class]):
                # 生成父模块、子模块名称和子模块本身
                yield parent, name, module


# 定义一个查找模块的函数，支持更复杂的搜索条件
def _find_modules_v2(
    model,
    ancestor_class: Optional[Set[str]] = None,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [
        LoRACompatibleLinear,
        LoRACompatibleConv,
        LoRALinearLayer,
        LoRAConv2dLayer,
    ],
):
    """
    查找特定类（或类的集合）的所有模块，这些模块是其他类模块的直接或间接子孙。

    返回所有匹配的模块及其父模块和引用的名称。
    """

    # 获取需要替换的目标模块
    if ancestor_class is not None:
        # 如果指定了祖先类，筛选出符合条件的模块
        ancestors = (module for module in model.modules() if module.__class__.__name__ in ancestor_class)
    else:
        # 否则，遍历所有模块
        ancestors = [module for module in model.modules()]

    # 对于每个目标，查找所有指定类的子模块
    for ancestor in ancestors:
        # 获取祖先模块的所有命名模块
        for fullname, module in ancestor.named_modules():
            # 检查模块是否属于指定的类
            if any([isinstance(module, _class) for _class in search_class]):
                # 将完整模块名称拆分，获取其父模块
                *path, name = fullname.split(".")
                parent = ancestor
                flag = False
                # 遍历路径，找到父模块
                while path:
                    try:
                        parent = parent.get_submodule(path.pop(0))
                    except:
                        flag = True
                        break
                # 如果找不到父模块，继续下一个模块
                if flag:
                    continue
                # 检查是否为排除的子模块类型
                if exclude_children_of and any([isinstance(parent, _class) for _class in exclude_children_of]):
                    continue
                # 否则，生成父模块、子模块名称和子模块本身
                yield parent, name, module


# 将 _find_modules_v2 函数赋值给 _find_modules
_find_modules = _find_modules_v2


# 定义注入可训练的 Lora 扩展的函数
def inject_trainable_lora_extended(
    model: nn.Module,
    target_replace_module: Set[str] = None,
    rank: int = 4,
    scale: float = 1.0,
):
    # 使用 _find_modules 函数查找模块，并替换特定的模块
    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear, nn.Conv2d]
    ):
        # 检查子模块是否为线性层
        if _child_module.__class__ == nn.Linear:
            # 获取线性层的权重
            weight = _child_module.weight
            # 获取线性层的偏置
            bias = _child_module.bias
            # 创建 LoRA 线性层，指定输入输出特征和秩
            lora_layer = LoRALinearLayer(
                in_features=_child_module.in_features,
                out_features=_child_module.out_features,
                rank=rank,
            )
            # 创建兼容 LoRA 的线性层，并将其转换为权重的数据类型和设备
            _tmp = (
                LoRACompatibleLinear(
                    _child_module.in_features,
                    _child_module.out_features,
                    lora_layer=lora_layer,
                    scale=scale,
                )
                .to(weight.dtype)  # 转换为权重的数据类型
                .to(weight.device)  # 转换为权重的设备
            )
            # 将原线性层的权重赋值给新的兼容层
            _tmp.weight = weight
            # 如果有偏置，则赋值给新的兼容层
            if bias is not None:
                _tmp.bias = bias
        # 检查子模块是否为二维卷积层
        elif _child_module.__class__ == nn.Conv2d:
            # 获取卷积层的权重
            weight = _child_module.weight
            # 获取卷积层的偏置
            bias = _child_module.bias
            # 创建 LoRA 卷积层，指定输入输出通道、秩、卷积核大小、步幅和填充
            lora_layer = LoRAConv2dLayer(
                in_features=_child_module.in_channels,
                out_features=_child_module.out_channels,
                rank=rank,
                kernel_size=_child_module.kernel_size,
                stride=_child_module.stride,
                padding=_child_module.padding,
            )
            # 创建兼容 LoRA 的卷积层，并将其转换为权重的数据类型和设备
            _tmp = (
                LoRACompatibleConv(
                    _child_module.in_channels,
                    _child_module.out_channels,
                    kernel_size=_child_module.kernel_size,
                    stride=_child_module.stride,
                    padding=_child_module.padding,
                    lora_layer=lora_layer,
                    scale=scale,
                )
                .to(weight.dtype)  # 转换为权重的数据类型
                .to(weight.device)  # 转换为权重的设备
            )
            # 将原卷积层的权重赋值给新的兼容层
            _tmp.weight = weight
            # 如果有偏置，则赋值给新的兼容层
            if bias is not None:
                _tmp.bias = bias
        # 如果子模块既不是线性层也不是卷积层，则继续下一个循环
        else:
            continue

        # 将新创建的兼容层替换到模块的子模块中
        _module._modules[name] = _tmp
        # print('injecting lora layer to', _module, name)

    # 返回到调用者
    return
# 更新模型中 LoRA 模块的缩放因子
def update_lora_scale(
    model: nn.Module,  # 输入的模型对象
    target_module: Set[str] = None,  # 可选的目标模块名称集合，默认为 None
    scale: float = 1.0,  # 设置的缩放因子，默认为 1.0
):
    # 遍历找到的模块及其子模块，筛选特定类的模块
    for _module, name, _child_module in _find_modules(
        model,  # 要搜索的模型
        target_module,  # 目标模块集合
        search_class=[LoRACompatibleLinear, LoRACompatibleConv]  # 指定搜索的模块类型
    ):
        # 将子模块的缩放因子设置为指定值
        _child_module.scale = scale

    # 函数结束，不返回任何值
    return
```

# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\loss.py`

```py
# 从类型提示模块导入相关类型
from typing import List, Optional, Union

# 导入 PyTorch 库及其神经网络模块
import torch
import torch.nn as nn
# 导入 ListConfig 用于配置管理
from omegaconf import ListConfig
# 从工具模块导入辅助函数
from ...util import append_dims, instantiate_from_config
# 从 LPIPS 模块导入损失计算
from ...modules.autoencoding.lpips.loss.lpips import LPIPS
# 导入模型并行模块
from sat import mpu


# 定义标准扩散损失类，继承自 nn.Module
class StandardDiffusionLoss(nn.Module):
    # 初始化方法，设置损失类型和其他参数
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        # 调用父类初始化方法
        super().__init__()

        # 确保损失类型是合法的
        assert type in ["l2", "l1", "lpips"]

        # 从配置中实例化 sigma 采样器
        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        # 保存损失类型和偏移噪声水平
        self.type = type
        self.offset_noise_level = offset_noise_level

        # 如果损失类型是 lpips，则初始化 LPIPS 实例并设置为评估模式
        if type == "lpips":
            self.lpips = LPIPS().eval()

        # 如果没有提供 batch2model_keys，初始化为空列表
        if not batch2model_keys:
            batch2model_keys = []

        # 如果 batch2model_keys 是字符串，则转换为列表
        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        # 将 batch2model_keys 转换为集合以去重
        self.batch2model_keys = set(batch2model_keys)

    # 定义调用方法，计算损失
    def __call__(self, network, denoiser, conditioner, input, batch):
        # 使用调节器处理批次数据
        cond = conditioner(batch)
        # 从批次中提取额外的模型输入
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        # 生成 sigma 值并移至输入设备
        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        # 生成与输入形状相同的随机噪声
        noise = torch.randn_like(input)
        # 如果偏移噪声水平大于零，则调整噪声
        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )
            # 确保噪声数据类型与输入一致
            noise = noise.to(input.dtype)
        # 计算加噪输入
        noised_input = input.float() + noise * append_dims(sigmas, input.ndim)
        # 使用去噪器处理加噪输入并生成模型输出
        model_output = denoiser(network, noised_input, sigmas, cond, **additional_model_inputs)
        # 获取权重并调整维度
        w = append_dims(denoiser.w(sigmas), input.ndim)
        # 返回计算的损失
        return self.get_loss(model_output, input, w)

    # 定义获取损失的方法
    def get_loss(self, model_output, target, w):
        # 根据损失类型计算不同类型的损失
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            # 计算 LPIPS 损失并调整维度
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


# 定义视频扩散损失类，继承自标准扩散损失类
class VideoDiffusionLoss(StandardDiffusionLoss):
    # 初始化方法，设置视频相关参数
    def __init__(self, block_scale=None, block_size=None, min_snr_value=None, fixed_frames=0, **kwargs):
        # 保存固定帧数量
        self.fixed_frames = fixed_frames
        # 保存块缩放因子
        self.block_scale = block_scale
        # 保存最小信噪比值
        self.block_size = block_size
        # 保存最小信噪比值
        self.min_snr_value = min_snr_value
        # 调用父类初始化方法
        super().__init__(**kwargs)
    # 定义一个可调用对象，接收网络、去噪器、调节器、输入和批处理作为参数
    def __call__(self, network, denoiser, conditioner, input, batch):
        # 使用调节器对批处理进行处理，生成条件
        cond = conditioner(batch)
        # 从批处理中过滤出与模型输入相关的额外输入
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        # 获取累积的 alpha 的平方根及其索引
        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        # 将 alpha 的平方根移动到输入的设备上
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        # 将索引移动到输入的设备上
        idx = idx.to(input.device)

        # 生成与输入形状相同的随机噪声
        noise = torch.randn_like(input)

        # 广播噪声
        mp_size = mpu.get_model_parallel_world_size()  # 获取模型并行世界的大小
        global_rank = torch.distributed.get_rank() // mp_size  # 计算全局排名
        src = global_rank * mp_size  # 计算源节点
        # 广播索引到所有相关节点
        torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        # 广播噪声到所有相关节点
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        # 广播 alpha 的平方根到所有相关节点
        torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())

        # 将索引添加到额外的模型输入中
        additional_model_inputs["idx"] = idx

        # 如果偏移噪声级别大于 0，则调整噪声
        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )

        # 计算带噪声的输入
        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims(
            (1 - alphas_cumprod_sqrt**2) ** 0.5, input.ndim
        )

        # 如果批处理包含拼接图像，则将其添加到条件中
        if "concat_images" in batch.keys():
            cond["concat"] = batch["concat_images"]

        # 调用去噪器处理噪声输入，并获取模型输出
        model_output = denoiser(network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs)
        # 计算加权值（v-pred）
        w = append_dims(1 / (1 - alphas_cumprod_sqrt**2), input.ndim)  

        # 如果设置了最小信噪比值，则取加权值与最小信噪比值的最小值
        if self.min_snr_value is not None:
            w = min(w, self.min_snr_value)
        # 返回损失值
        return self.get_loss(model_output, input, w)

    # 定义获取损失的函数，接收模型输出、目标和权重
    def get_loss(self, model_output, target, w):
        # 如果损失类型为 L2，则计算 L2 损失
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        # 如果损失类型为 L1，则计算 L1 损失
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        # 如果损失类型为 LPIPS，则计算 LPIPS 损失
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)  # 计算 LPIPS 损失并调整形状
            return loss
```

# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\model.py`

```py
# pytorch_diffusion + derived encoder decoder
# 导入所需的数学库和类型提示
import math
from typing import Any, Callable, Optional

# 导入 numpy 和 pytorch 相关的库
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from packaging import version

# 尝试导入 xformers 模块及其操作，如果失败则设置标志为 False
try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    # 如果未找到 xformers 模块，则打印提示信息
    print("no module 'xformers'. Processing without...")

# 从自定义模块导入线性注意力和内存高效交叉注意力
from ...modules.attention import LinearAttention, MemoryEfficientCrossAttention


def get_timestep_embedding(timesteps, embedding_dim):
    """
    该函数实现了 Denoising Diffusion Probabilistic Models 中的时间步嵌入
    来自 Fairseq。
    构建正弦嵌入。
    与 tensor2tensor 中的实现相匹配，但与 "Attention Is All You Need" 中第 3.5 节的描述略有不同。
    """
    # 确保 timesteps 具有一维形状
    assert len(timesteps.shape) == 1

    # 计算半维度并获取嵌入公式中的指数项
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    # 将嵌入项移动到与 timesteps 相同的设备上
    emb = emb.to(device=timesteps.device)
    # 计算时间步长与嵌入的乘积
    emb = timesteps.float()[:, None] * emb[None, :]
    # 将正弦和余弦值拼接在一起
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    # 如果嵌入维度为奇数，则进行零填充
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    # 返回最终的嵌入
    return emb


def nonlinearity(x):
    # 实现 swish 激活函数
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    # 返回分组归一化层
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        # 初始化 Upsample 类，设置输入通道数和是否使用卷积
        super().__init__()
        self.with_conv = with_conv
        # 如果使用卷积，初始化卷积层
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 对输入张量进行上采样
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        # 如果使用卷积，则应用卷积层
        if self.with_conv:
            x = self.conv(x)
        # 返回处理后的张量
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        # 初始化 Downsample 类，设置输入通道数和是否使用卷积
        super().__init__()
        self.with_conv = with_conv
        # 如果使用卷积，初始化卷积层
        if self.with_conv:
            # PyTorch 卷积层不支持不对称填充，需手动处理
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        # 如果使用卷积，进行填充并应用卷积层
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            # 否则使用平均池化进行下采样
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        # 返回处理后的张量
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels
        # 如果没有指定输出通道数，则使用输入通道数
        out_channels = in_channels if out_channels is None else out_channels
        # 保存输出通道数
        self.out_channels = out_channels
        # 保存是否使用卷积快捷方式的标志
        self.use_conv_shortcut = conv_shortcut

        # 初始化归一化层，输入通道数作为参数
        self.norm1 = Normalize(in_channels)
        # 初始化第一个卷积层，输入通道数、输出通道数，卷积核大小为3，步幅为1，填充为1
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 如果temb_channels大于0，初始化temb的线性变换层
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        # 初始化第二个归一化层，输出通道数作为参数
        self.norm2 = Normalize(out_channels)
        # 初始化丢弃层，使用给定的丢弃率
        self.dropout = torch.nn.Dropout(dropout)
        # 初始化第二个卷积层，输入和输出通道数一致，卷积核大小为3，步幅为1，填充为1
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 如果输入和输出通道数不一致
        if self.in_channels != self.out_channels:
            # 如果使用卷积快捷方式，则初始化卷积快捷方式层
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            # 否则初始化1x1的线性变换快捷方式层
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        # 将输入赋值给临时变量h
        h = x
        # 对h进行归一化处理
        h = self.norm1(h)
        # 对h应用非线性激活函数
        h = nonlinearity(h)
        # 对h进行第一个卷积操作
        h = self.conv1(h)

        # 如果temb不为None，则对h进行temb的投影
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        # 对h进行第二个归一化处理
        h = self.norm2(h)
        # 对h应用非线性激活函数
        h = nonlinearity(h)
        # 对h进行丢弃操作
        h = self.dropout(h)
        # 对h进行第二个卷积操作
        h = self.conv2(h)

        # 如果输入和输出通道数不一致
        if self.in_channels != self.out_channels:
            # 如果使用卷积快捷方式，则对输入x进行卷积处理
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            # 否则对输入x进行1x1的线性变换
            else:
                x = self.nin_shortcut(x)

        # 返回输入x与h的相加结果
        return x + h
class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


# 创建一个名为LinAttnBlock的类，继承自LinearAttention类
class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    # 构造函数，接受输入通道数作为参数
    def __init__(self, in_channels):
        # 调用父类的构造函数，传入输入通道数、头数为1、头通道数为输入通道数
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)



class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)


# 创建一个名为AttnBlock的类，继承自nn.Module类
class AttnBlock(nn.Module):
    # 构造函数，接受输入通道数作为参数
    def __init__(self, in_channels):
        # 调用父类的构造函数
        super().__init__()
        # 将输入通道数保存到实例变量中
        self.in_channels = in_channels

        # 创建一个Normalize对象，输入通道数为in_channels
        self.norm = Normalize(in_channels)
        # 创建一个卷积层，输入通道数为in_channels，输出通道数为in_channels，卷积核大小为1x1，步长为1，填充为0
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建一个卷积层，输入通道数为in_channels，输出通道数为in_channels，卷积核大小为1x1，步长为1，填充为0
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建一个卷积层，输入通道数为in_channels，输出通道数为in_channels，卷积核大小为1x1，步长为1，填充为0
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建一个卷积层，输入通道数为in_channels，输出通道数为in_channels，卷积核大小为1x1，步长为1，填充为0
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)



    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v))
        h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # scale is dim ** -0.5 per default
        # compute attention

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)


    # 定义attention方法，接受一个torch.Tensor类型的参数h_，返回一个torch.Tensor类型的结果
    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        # 对输入张量进行归一化处理
        h_ = self.norm(h_)
        # 使用卷积层q、k、v对输入张量进行卷积操作
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 获取q的形状信息
        b, c, h, w = q.shape
        # 将q、k、v的形状进行变换
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v))
        # 使用缩放点积注意力机制计算注意力
        h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # scale is dim ** -0.5 per default
        # 返回计算得到的注意力张量，并将其形状进行变换
        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)



    def forward(self, x, **kwargs):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_


    # 定义forward方法，接受输入张量x和其他关键字参数，返回一个torch.Tensor类型的结果
    def forward(self, x, **kwargs):
        # 将输入张量赋值给局部变量h_
        h_ = x
        # 使用attention方法对h_进行处理
        h_ = self.attention(h_)
        # 使用卷积层proj_out对h_进行卷积操作
        h_ = self.proj_out(h_)
        # 返回输入张量x与处理后的张量h_的和
        return x + h_


class MemoryEfficientAttnBlock(nn.Module):
    """
    Uses xformers efficient implementation,
    see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    Note: this is a single-head self-attention operation
    """

    #
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.attention_op: Optional[Any] = None


# 创建一个名为MemoryEfficientAttnBlock的类，继承自nn.Module类
class MemoryEfficientAttnBlock(nn.Module):
    """
    Uses xformers efficient implementation,
    see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    Note: this is a single-head self-attention operation
    """

    #
    # 构造函数，接受输入通道数作为参数
    def __init__(self, in_channels):
        # 调用父类的构造函数
        super().__init__()
        # 将输入通道数保存到实例变量中
        self.in_channels = in_channels

        # 创建一个Normalize对象，输入通道数为in_channels
        self.norm = Normalize(in_channels)
        # 创建一个卷积层，输入通道数为in_channels，输出通道数为in_channels，卷积核大小为1x1，步长为1，填充为0
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建一个卷积层，输入通道数为in_channels，输出通道数为in_channels，卷积核大小为1x1，步长为1，填充为0
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建一个卷积层，输入通道数为in_channels，输出通道数为in_channels，卷积核大小为1x1，步长为1，填充为0
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建一个卷积层，输入通道数为in_channels，输出通道数为in_channels，卷积核大小为1x1，步长为1，填充为0
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建一个Optional类型的实例变量attention_op，初始值为None
        self.attention_op: Optional[Any] = None
    # 定义注意力机制方法，输入为张量 h_，输出为处理后的张量
    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        # 对输入张量进行规范化处理
        h_ = self.norm(h_)
        # 通过线性变换生成查询向量 q
        q = self.q(h_)
        # 通过线性变换生成键向量 k
        k = self.k(h_)
        # 通过线性变换生成值向量 v
        v = self.v(h_)

        # 计算注意力机制
        # 获取查询向量的形状，B: 批次大小, C: 通道数, H: 高度, W: 宽度
        B, C, H, W = q.shape
        # 将 q, k, v 的形状从 (B, C, H, W) 转换为 (B, H*W, C)
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))

        # 扩展 q, k, v 的维度并重塑形状，以便进行注意力计算
        q, k, v = map(
            lambda t: t.unsqueeze(3)  # 在第 3 维增加一个维度
            .reshape(B, t.shape[1], 1, C)  # 重塑为 (B, 经过处理的长度, 1, C)
            .permute(0, 2, 1, 3)  # 调整维度顺序为 (B, 1, 经过处理的长度, C)
            .reshape(B * 1, t.shape[1], C)  # 重塑为 (B, 经过处理的长度, C)
            .contiguous(),  # 确保内存连续性
            (q, k, v),
        )
        # 使用高效的注意力计算方法，返回注意力输出
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        # 对输出进行重塑，调整形状以匹配最终的输出要求
        out = out.unsqueeze(0).reshape(B, 1, out.shape[1], C).permute(0, 2, 1, 3).reshape(B, out.shape[1], C)
        # 将输出形状调整回 (B, C, H, W)
        return rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)

    # 定义前向传播方法，输入为张量 x，接受额外参数 kwargs
    def forward(self, x, **kwargs):
        # 将输入赋值给 h_
        h_ = x
        # 通过注意力机制处理 h_
        h_ = self.attention(h_)
        # 将处理结果通过输出线性层
        h_ = self.proj_out(h_)
        # 返回输入 x 与处理结果 h_ 的和
        return x + h_
# 定义一个内存高效的交叉注意力包装类，继承自 MemoryEfficientCrossAttention
class MemoryEfficientCrossAttentionWrapper(MemoryEfficientCrossAttention):
    # 前向传播方法，接收输入和上下文
    def forward(self, x, context=None, mask=None, **unused_kwargs):
        # 获取输入的批量大小、通道数、高度和宽度
        b, c, h, w = x.shape
        # 重排输入张量，合并高度和宽度维度
        x = rearrange(x, "b c h w -> b (h w) c")
        # 调用父类的前向方法进行注意力计算
        out = super().forward(x, context=context, mask=mask)
        # 重排输出张量，恢复为原始形状
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w, c=c)
        # 返回输入与输出的和
        return x + out


# 创建注意力模块的函数，根据给定的类型和参数
def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    # 确保提供的注意力类型是有效的
    assert attn_type in [
        "vanilla",
        "vanilla-xformers",
        "memory-efficient-cross-attn",
        "linear",
        "none",
    ], f"attn_type {attn_type} unknown"
    # 检查 PyTorch 版本以决定注意力实现
    if version.parse(torch.__version__) < version.parse("2.0.0") and attn_type != "none":
        # 确保 xformers 可用以支持较早的版本
        assert XFORMERS_IS_AVAILABLE, (
            f"We do not support vanilla attention in {torch.__version__} anymore, "
            f"as it is too expensive. Please install xformers via e.g. 'pip install xformers==0.0.16'"
        )
        # 更新注意力类型为 vanilla-xformers
        attn_type = "vanilla-xformers"
    # 打印当前创建的注意力类型及输入通道数
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    # 根据类型创建对应的注意力模块
    if attn_type == "vanilla":
        # 确保没有额外的参数
        assert attn_kwargs is None
        return AttnBlock(in_channels)
    elif attn_type == "vanilla-xformers":
        # 打印构建内存高效注意力块的信息
        print(f"building MemoryEfficientAttnBlock with {in_channels} in_channels...")
        return MemoryEfficientAttnBlock(in_channels)
    elif attn_type == "memory-efficient-cross-attn":
        # 设置查询维度为输入通道数
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)
    elif attn_type == "none":
        # 返回身份映射
        return nn.Identity(in_channels)
    else:
        # 返回线性注意力块
        return LinAttnBlock(in_channels)


# 定义模型类，继承自 nn.Module
class Model(nn.Module):
    # 初始化模型参数
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        use_timestep=True,
        use_linear_attn=False,
        attn_type="vanilla",
    # 定义前向传播函数，接受输入数据 x、时间 t 和上下文 context
        def forward(self, x, t=None, context=None):
            # 检查输入张量的空间维度是否与预设分辨率一致（注释掉）
            # assert x.shape[2] == x.shape[3] == self.resolution
            # 如果上下文不为 None，沿通道轴拼接输入和上下文
            if context is not None:
                x = torch.cat((x, context), dim=1)
            # 如果使用时间步，进行时间步嵌入
            if self.use_timestep:
                # 确保时间 t 不为 None
                assert t is not None
                # 获取时间步嵌入
                temb = get_timestep_embedding(t, self.ch)
                # 通过第一层全连接进行变换
                temb = self.temb.dense[0](temb)
                # 应用非线性激活函数
                temb = nonlinearity(temb)
                # 通过第二层全连接进行变换
                temb = self.temb.dense[1](temb)
            else:
                # 如果不使用时间步，则将 temb 设为 None
                temb = None
    
            # 进行下采样
            hs = [self.conv_in(x)]  # 初始化列表，包含输入经过卷积的结果
            for i_level in range(self.num_resolutions):  # 遍历分辨率级别
                for i_block in range(self.num_res_blocks):  # 遍历每个分辨率的残差块
                    # 将上一层的输出和时间嵌入输入到当前块
                    h = self.down[i_level].block[i_block](hs[-1], temb)
                    # 如果当前层有注意力机制，应用注意力机制
                    if len(self.down[i_level].attn) > 0:
                        h = self.down[i_level].attn[i_block](h)
                    # 将当前层的输出添加到列表中
                    hs.append(h)
                # 如果不是最后一个分辨率，进行下采样
                if i_level != self.num_resolutions - 1:
                    hs.append(self.down[i_level].downsample(hs[-1]))
    
            # 处理中间层
            h = hs[-1]  # 获取最后一层的输出
            h = self.mid.block_1(h, temb)  # 经过中间第一块处理
            h = self.mid.attn_1(h)  # 应用中间第一块的注意力机制
            h = self.mid.block_2(h, temb)  # 经过中间第二块处理
    
            # 进行上采样
            for i_level in reversed(range(self.num_resolutions)):  # 反向遍历分辨率级别
                for i_block in range(self.num_res_blocks + 1):  # 遍历每个分辨率的残差块加一
                    # 将当前层输出与上一层的输出拼接，并传入时间嵌入
                    h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                    # 如果当前层有注意力机制，应用注意力机制
                    if len(self.up[i_level].attn) > 0:
                        h = self.up[i_level].attn[i_block](h)
                # 如果不是第一个分辨率，进行上采样
                if i_level != 0:
                    h = self.up[i_level].upsample(h)
    
            # 结束处理
            h = self.norm_out(h)  # 归一化输出
            h = nonlinearity(h)  # 应用非线性激活函数
            h = self.conv_out(h)  # 最终卷积层处理
            return h  # 返回最终结果
    
        # 定义获取最后一层权重的函数
        def get_last_layer(self):
            return self.conv_out.weight  # 返回最后一层卷积的权重
# 定义一个编码器类，继承自 nn.Module
class Encoder(nn.Module):
    # 初始化函数，接收多个参数用于配置编码器
    def __init__(
        self,
        *,
        ch,  # 输入通道数
        out_ch,  # 输出通道数
        ch_mult=(1, 2, 4, 8),  # 通道数倍增因子
        num_res_blocks,  # 残差块数量
        attn_resolutions,  # 注意力机制应用的分辨率
        dropout=0.0,  # dropout 概率
        resamp_with_conv=True,  # 是否使用卷积进行下采样
        in_channels,  # 输入图像的通道数
        resolution,  # 输入图像的分辨率
        z_channels,  # 潜在空间的通道数
        double_z=True,  # 是否双倍潜在通道
        use_linear_attn=False,  # 是否使用线性注意力机制
        attn_type="vanilla",  # 注意力机制的类型
        **ignore_kwargs,  # 其他未使用的参数
    ):
        # 调用父类构造函数
        super().__init__()
        # 如果使用线性注意力，设置注意力类型为线性
        if use_linear_attn:
            attn_type = "linear"
        # 设置类的属性
        self.ch = ch
        self.temb_ch = 0  # 时间嵌入通道数
        self.num_resolutions = len(ch_mult)  # 分辨率数量
        self.num_res_blocks = num_res_blocks  # 残差块数量
        self.resolution = resolution  # 输入分辨率
        self.in_channels = in_channels  # 输入通道数

        # 下采样层
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution  # 当前分辨率
        in_ch_mult = (1,) + tuple(ch_mult)  # 输入通道倍增因子
        self.in_ch_mult = in_ch_mult  # 保存输入通道倍增因子
        self.down = nn.ModuleList()  # 下采样模块列表
        # 遍历每个分辨率级别
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()  # 残差块列表
            attn = nn.ModuleList()  # 注意力模块列表
            block_in = ch * in_ch_mult[i_level]  # 当前块的输入通道数
            block_out = ch * ch_mult[i_level]  # 当前块的输出通道数
            # 遍历每个残差块
            for i_block in range(self.num_res_blocks):
                # 添加残差块
                block.append(
                    ResnetBlock(
                        in_channels=block_in,  # 输入通道数
                        out_channels=block_out,  # 输出通道数
                        temb_channels=self.temb_ch,  # 时间嵌入通道数
                        dropout=dropout,  # dropout 概率
                    )
                )
                block_in = block_out  # 更新输入通道数
                # 如果当前分辨率在注意力分辨率中，添加注意力模块
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()  # 创建一个下采样模块
            down.block = block  # 设置残差块
            down.attn = attn  # 设置注意力模块
            # 如果不是最后一个分辨率级别，添加下采样层
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2  # 更新当前分辨率
            self.down.append(down)  # 将下采样模块添加到列表中

        # 中间层
        self.mid = nn.Module()  # 创建中间层模块
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,  # 输入通道数
            out_channels=block_in,  # 输出通道数
            temb_channels=self.temb_ch,  # 时间嵌入通道数
            dropout=dropout,  # dropout 概率
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)  # 添加第一个注意力模块
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,  # 输入通道数
            out_channels=block_in,  # 输出通道数
            temb_channels=self.temb_ch,  # 时间嵌入通道数
            dropout=dropout,  # dropout 概率
        )

        # 结束层
        self.norm_out = Normalize(block_in)  # 归一化层
        self.conv_out = torch.nn.Conv2d(
            block_in,  # 输入通道数
            2 * z_channels if double_z else z_channels,  # 输出通道数，依据是否双倍潜在通道选择
            kernel_size=3,  # 卷积核大小
            stride=1,  # 步幅
            padding=1,  # 填充
        )
    # 定义前向传播方法，接受输入 x
        def forward(self, x):
            # 初始化时间步嵌入变量为 None
            temb = None
    
            # 初始化下采样，创建输入的卷积特征
            hs = [self.conv_in(x)]
            # 遍历每个分辨率级别
            for i_level in range(self.num_resolutions):
                # 遍历每个残差块
                for i_block in range(self.num_res_blocks):
                    # 使用下采样块处理最后一层特征和时间步嵌入
                    h = self.down[i_level].block[i_block](hs[-1], temb)
                    # 如果当前层有注意力机制，则应用它
                    if len(self.down[i_level].attn) > 0:
                        h = self.down[i_level].attn[i_block](h)
                    # 将处理后的特征添加到特征列表中
                    hs.append(h)
                # 如果不是最后一个分辨率级别，则进行下采样
                if i_level != self.num_resolutions - 1:
                    hs.append(self.down[i_level].downsample(hs[-1]))
    
            # 处理中间层特征
            h = hs[-1]
            # 应用中间块1处理
            h = self.mid.block_1(h, temb)
            # 应用中间层的注意力机制1
            h = self.mid.attn_1(h)
            # 应用中间块2处理
            h = self.mid.block_2(h, temb)
    
            # 结束层处理
            h = self.norm_out(h)  # 应用输出归一化
            h = nonlinearity(h)   # 应用非线性激活函数
            h = self.conv_out(h)  # 应用输出卷积
            return h  # 返回最终输出
# 定义一个解码器类，继承自 nn.Module
class Decoder(nn.Module):
    # 初始化方法，接收多个参数用于解码器的配置
    def __init__(
        self,
        *,
        ch,  # 输入通道数
        out_ch,  # 输出通道数
        ch_mult=(1, 2, 4, 8),  # 通道数的倍增因子
        num_res_blocks,  # 残差块的数量
        attn_resolutions,  # 注意力机制应用的分辨率
        dropout=0.0,  # 丢弃率，默认为0
        resamp_with_conv=True,  # 是否使用卷积进行重采样
        in_channels,  # 输入图像的通道数
        resolution,  # 输入图像的分辨率
        z_channels,  # 潜在空间的通道数
        give_pre_end=False,  # 是否提供前置结束标志
        tanh_out=False,  # 是否使用 Tanh 激活函数作为输出
        use_linear_attn=False,  # 是否使用线性注意力机制
        attn_type="vanilla",  # 注意力机制的类型，默认为普通注意力
        **ignorekwargs,  # 其他未明确列出的参数
    ):
        # 初始化父类
        super().__init__()
        # 如果使用线性注意力，设置注意力类型为"linear"
        if use_linear_attn:
            attn_type = "linear"
        # 设置通道数
        self.ch = ch
        # 设置时间嵌入通道数为0
        self.temb_ch = 0
        # 计算分辨率的数量
        self.num_resolutions = len(ch_mult)
        # 设置残差块的数量
        self.num_res_blocks = num_res_blocks
        # 设置输入分辨率
        self.resolution = resolution
        # 设置输入通道数
        self.in_channels = in_channels
        # 设置是否在最后给出前置层
        self.give_pre_end = give_pre_end
        # 设置是否使用tanh作为输出激活函数
        self.tanh_out = tanh_out

        # 计算输入通道数乘数，块输入和当前分辨率
        in_ch_mult = (1,) + tuple(ch_mult)
        # 计算当前层的输入通道数
        block_in = ch * ch_mult[self.num_resolutions - 1]
        # 计算当前分辨率
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # 设置z的形状
        self.z_shape = (1, z_channels, curr_res, curr_res)
        # 打印z的形状和维度
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # 创建注意力类
        make_attn_cls = self._make_attn()
        # 创建残差块类
        make_resblock_cls = self._make_resblock()
        # 创建卷积类
        make_conv_cls = self._make_conv()
        # z到块输入的卷积层
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # 中间模块
        self.mid = nn.Module()
        # 创建第一个残差块
        self.mid.block_1 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        # 创建第一个注意力层
        self.mid.attn_1 = make_attn_cls(block_in, attn_type=attn_type)
        # 创建第二个残差块
        self.mid.block_2 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # 上采样模块
        self.up = nn.ModuleList()
        # 反向遍历分辨率
        for i_level in reversed(range(self.num_resolutions)):
            # 创建块和注意力列表
            block = nn.ModuleList()
            attn = nn.ModuleList()
            # 计算块输出通道数
            block_out = ch * ch_mult[i_level]
            # 创建残差块
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    make_resblock_cls(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                # 更新块输入通道数
                block_in = block_out
                # 如果当前分辨率需要注意力层，添加注意力层
                if curr_res in attn_resolutions:
                    attn.append(make_attn_cls(block_in, attn_type=attn_type))
            # 创建上采样模块
            up = nn.Module()
            up.block = block
            up.attn = attn
            # 如果不是最后一层，添加上采样层
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                # 更新当前分辨率
                curr_res = curr_res * 2
            # 将上采样模块添加到列表的开头
            self.up.insert(0, up)  # prepend to get consistent order

        # 结束模块
        # 创建归一化层
        self.norm_out = Normalize(block_in)
        # 创建输出卷积层
        self.conv_out = make_conv_cls(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    # 创建注意力函数
    def _make_attn(self) -> Callable:
        return make_attn

    # 创建残差块函数
    def _make_resblock(self) -> Callable:
        return ResnetBlock

    # 创建卷积函数
    def _make_conv(self) -> Callable:
        return torch.nn.Conv2d
    # 获取模型最后一层的权重
        def get_last_layer(self, **kwargs):
            # 返回卷积输出层的权重
            return self.conv_out.weight
    
        # 前向传播函数
        def forward(self, z, **kwargs):
            # 检查输入 z 的形状是否与预期相同（注释掉的断言）
            # assert z.shape[1:] == self.z_shape[1:]
            # 保存输入 z 的形状
            self.last_z_shape = z.shape
    
            # 初始化时间步嵌入
            temb = None
    
            # 将输入 z 传入卷积输入层
            h = self.conv_in(z)
    
            # 中间层处理
            h = self.mid.block_1(h, temb, **kwargs)  # 通过第一个中间块处理
            h = self.mid.attn_1(h, **kwargs)        # 通过第一个注意力层处理
            h = self.mid.block_2(h, temb, **kwargs)  # 通过第二个中间块处理
    
            # 向上采样过程
            for i_level in reversed(range(self.num_resolutions)):  # 反向遍历分辨率层级
                for i_block in range(self.num_res_blocks + 1):  # 遍历每个块
                    h = self.up[i_level].block[i_block](h, temb, **kwargs)  # 通过上采样块处理
                    if len(self.up[i_level].attn) > 0:  # 如果有注意力层
                        h = self.up[i_level].attn[i_block](h, **kwargs)  # 通过注意力层处理
                if i_level != 0:  # 如果不是最后一个分辨率层
                    h = self.up[i_level].upsample(h)  # 执行上采样操作
    
            # 结束处理
            if self.give_pre_end:  # 如果需要返回预处理结果
                return h
    
            h = self.norm_out(h)  # 通过输出归一化层处理
            h = nonlinearity(h)   # 应用非线性激活函数
            h = self.conv_out(h, **kwargs)  # 通过卷积输出层处理
            if self.tanh_out:  # 如果需要使用双曲正切激活函数
                h = torch.tanh(h)  # 应用双曲正切激活函数
            return h  # 返回最终输出
```

# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\openaimodel.py`

```py
# 导入操作系统相关的模块
import os
# 导入数学相关的模块
import math
# 从 abc 模块导入抽象方法装饰器
from abc import abstractmethod
# 从 functools 导入偏函数工具
from functools import partial
# 导入类型提示相关的类型
from typing import Iterable, List, Optional, Tuple, Union

# 导入 numpy 库并简写为 np
import numpy as np
# 导入 PyTorch 库并简写为 th
import torch as th
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 PyTorch 的功能模块
import torch.nn.functional as F
# 从 einops 导入重排工具
from einops import rearrange

# 从本地模块中导入 SpatialTransformer
from ...modules.attention import SpatialTransformer
# 从本地模块中导入多个实用函数
from ...modules.diffusionmodules.util import (
    avg_pool_nd,  # 导入平均池化函数
    checkpoint,  # 导入检查点函数
    conv_nd,  # 导入多维卷积函数
    linear,  # 导入线性层函数
    normalization,  # 导入归一化函数
    timestep_embedding,  # 导入时间步嵌入函数
    zero_module,  # 导入零模块函数
)
# 从本地模块中导入 LoRA 相关功能
from ...modules.diffusionmodules.lora import inject_trainable_lora_extended, update_lora_scale
# 从本地模块中导入空间视频变换器
from ...modules.video_attention import SpatialVideoTransformer
# 从本地工具模块导入默认值和存在性检查
from ...util import default, exists

# 虚函数替代
def convert_module_to_f16(x):
    pass

# 虚函数替代
def convert_module_to_f32(x):
    pass

# 定义 AttentionPool2d 类，继承自 nn.Module
class AttentionPool2d(nn.Module):
    """
    来源于 CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    # 初始化方法，定义各个参数
    def __init__(
        self,
        spacial_dim: int,  # 空间维度
        embed_dim: int,  # 嵌入维度
        num_heads_channels: int,  # 通道数对应的头数量
        output_dim: int = None,  # 输出维度，默认为 None
    ):
        # 调用父类初始化
        super().__init__()
        # 定义位置嵌入参数
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5)
        # 定义 QKV 投影卷积层
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        # 定义输出卷积层
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        # 计算头的数量
        self.num_heads = embed_dim // num_heads_channels
        # 初始化注意力机制
        self.attention = QKVAttention(self.num_heads)

    # 前向传播方法
    def forward(self, x):
        # 获取输入的批量大小和通道数
        b, c, *_spatial = x.shape
        # 将输入张量重塑为 (b, c, -1) 的形状
        x = x.reshape(b, c, -1)  # NC(HW)
        # 计算输入的均值并连接
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        # 添加位置嵌入
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        # 通过 QKV 投影层处理输入
        x = self.qkv_proj(x)
        # 通过注意力机制处理输入
        x = self.attention(x)
        # 通过输出卷积层处理输入
        x = self.c_proj(x)
        # 返回处理后的第一个通道
        return x[:, :, 0]

# 定义时间步块类，继承自 nn.Module
class TimestepBlock(nn.Module):
    """
    任何模块，其中 forward() 方法将时间步嵌入作为第二个参数。
    """

    # 抽象前向传播方法
    @abstractmethod
    def forward(self, x, emb):
        """
        根据给定的时间步嵌入对 `x` 应用模块。
        """

# 定义时间步嵌入顺序模块类，继承自 nn.Sequential 和 TimestepBlock
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    一个顺序模块，将时间步嵌入作为额外输入传递给支持的子模块。
    """

    # 前向传播方法，支持多个输入参数
    def forward(
        self,
        x: th.Tensor,  # 输入张量
        emb: th.Tensor,  # 时间步嵌入张量
        context: Optional[th.Tensor] = None,  # 可选上下文张量
        image_only_indicator: Optional[th.Tensor] = None,  # 可选图像指示器
        time_context: Optional[int] = None,  # 可选时间上下文
        num_video_frames: Optional[int] = None,  # 可选视频帧数
    # 处理模型中的层，按不同类型调用相应的前向传播方法
        ):
            # 从指定路径导入 VideoResBlock 类
            from ...modules.diffusionmodules.video_model import VideoResBlock
    
            # 遍历当前模型的每一层
            for layer in self:
                # 将当前层赋值给模块变量
                module = layer
    
                # 检查模块是否为 TimestepBlock 且不是 VideoResBlock
                if isinstance(module, TimestepBlock) and not isinstance(module, VideoResBlock):
                    # 调用当前层的前向传播方法，传入 x 和 emb
                    x = layer(x, emb)
                # 检查模块是否为 VideoResBlock
                elif isinstance(module, VideoResBlock):
                    # 调用当前层的前向传播方法，传入 x、emb、num_video_frames 和 image_only_indicator
                    x = layer(x, emb, num_video_frames, image_only_indicator)
                # 检查模块是否为 SpatialVideoTransformer
                elif isinstance(module, SpatialVideoTransformer):
                    # 调用当前层的前向传播方法，传入多个上下文参数
                    x = layer(
                        x,
                        context,
                        time_context,
                        num_video_frames,
                        image_only_indicator,
                    )
                # 检查模块是否为 SpatialTransformer
                elif isinstance(module, SpatialTransformer):
                    # 调用当前层的前向传播方法，传入 x 和 context
                    x = layer(x, context)
                # 处理其他类型的模块
                else:
                    # 调用当前层的前向传播方法，传入 x
                    x = layer(x)
            # 返回最终的输出 x
            return x
# 定义一个上采样层，具有可选的卷积功能
class Upsample(nn.Module):
    """
    一个上采样层，带有可选的卷积。
    :param channels: 输入和输出的通道数。
    :param use_conv: 布尔值，决定是否应用卷积。
    :param dims: 决定信号是 1D、2D 还是 3D。如果是 3D，则在内部两个维度进行上采样。
    """

    # 初始化方法，设置上采样层的参数
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, third_up=False):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入和输出的通道数
        self.channels = channels
        # 如果没有指定输出通道，则默认为输入通道
        self.out_channels = out_channels or channels
        # 保存是否使用卷积的标志
        self.use_conv = use_conv
        # 保存信号的维度
        self.dims = dims
        # 保存是否进行三次上采样的标志
        self.third_up = third_up
        # 如果使用卷积，则创建相应的卷积层
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    # 前向传播方法
    def forward(self, x):
        # 断言输入的通道数与初始化时的通道数相同
        assert x.shape[1] == self.channels
        # 如果信号是 3D，则进行三维上采样
        if self.dims == 3:
            # 确定时间因子，如果不进行三次上采样，则因子为 1
            t_factor = 1 if not self.third_up else 2
            # 使用最近邻插值进行上采样
            x = F.interpolate(
                x,
                (t_factor * x.shape[2], x.shape[3] * 2, x.shape[4] * 2),  # 新的形状
                mode="nearest",  # 使用最近邻插值
            )
        else:
            # 对于其他维度，按比例进行上采样
            x = F.interpolate(x, scale_factor=2, mode="nearest")  # 按比例进行上采样
        # 如果使用卷积，则应用卷积层
        if self.use_conv:
            x = self.conv(x)
        # 返回上采样后的结果
        return x


# 定义一个转置上采样层，执行 2x 上采样而不添加填充
class TransposedUpsample(nn.Module):
    "学习的 2x 上采样，无填充"

    # 初始化方法，设置转置上采样层的参数
    def __init__(self, channels, out_channels=None, ks=5):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入和输出的通道数
        self.channels = channels
        self.out_channels = out_channels or channels
        # 创建转置卷积层，进行 2x 上采样
        self.up = nn.ConvTranspose2d(self.channels, self.out_channels, kernel_size=ks, stride=2)

    # 前向传播方法
    def forward(self, x):
        # 返回经过转置卷积层的结果
        return self.up(x)


# 定义一个下采样层，具有可选的卷积功能
class Downsample(nn.Module):
    """
    一个下采样层，带有可选的卷积。
    :param channels: 输入和输出的通道数。
    :param use_conv: 布尔值，决定是否应用卷积。
    :param dims: 决定信号是 1D、2D 还是 3D。如果是 3D，则在内部两个维度进行下采样。
    """
    # 初始化函数，设置下采样层的参数
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, third_down=False):
        # 调用父类的初始化函数
        super().__init__()
        # 设置下采样层的输入通道数
        self.channels = channels
        # 如果未指定输出通道数，则输出通道数与输入通道数相同
        self.out_channels = out_channels or channels
        # 记录是否使用卷积操作
        self.use_conv = use_conv
        # 设置下采样层的维度，默认为2
        self.dims = dims
        # 根据维度设置步长
        stride = 2 if dims != 3 else ((1, 2, 2) if not third_down else (2, 2, 2))
        # 如果使用卷积操作
        if use_conv:
            # 打印下采样层的维度信息
            print(f"Building a Downsample layer with {dims} dims.")
            print(
                f"  --> settings are: \n in-chn: {self.channels}, out-chn: {self.out_channels}, "
                f"kernel-size: 3, stride: {stride}, padding: {padding}"
            )
            # 如果维度为3，打印第三个轴（时间轴）的下采样信息
            if dims == 3:
                print(f"  --> Downsampling third axis (time): {third_down}")
            # 根据维度和参数设置卷积操作
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        # 如果不使用卷积操作
        else:
            # 断言输入通道数和输出通道数相同
            assert self.channels == self.out_channels
            # 设置操作为n维平均池化
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    # 前向传播函数
    def forward(self, x):
        # 断言输入数据的通道数与下采样层的输入通道数相同
        assert x.shape[1] == self.channels
        # 返回下采样操作后的结果
        return self.op(x)
# 定义一个残差块，可以选择是否改变通道数
class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    # 初始化函数，接受多个参数
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        kernel_size=3,
        exchange_temb_dims=False,
        skip_t_emb=False,
        ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化各个属性
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        # 如果 kernel_size 是可迭代对象，则计算 padding
        if isinstance(kernel_size, Iterable):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        # 构建输入层
        self.in_layers = nn.Sequential(
            normalization(channels),  # 归一化
            nn.SiLU(),  # 激活函数
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding),  # 卷积层
        )

        # 设置上采样或下采样
        self.updown = up or down

        # 如果是上采样，则初始化上采样层
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        # 如果是下采样，则初始化下采样层
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # 设置是否跳过时间步嵌入
        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = 2 * self.out_channels if use_scale_shift_norm else self.out_channels
        # 如果跳过时间步嵌入，则设置相关属性
        if self.skip_t_emb:
            print(f"Skipping timestep embedding in {self.__class__.__name__}")
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        # 如果不跳过时间步嵌入，则初始化时间步嵌入层
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),  # 激活函数
                linear(
                    emb_channels,
                    self.emb_out_channels,
                ),
            )

        # 构建输出层
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),  # 归一化
            nn.SiLU(),  # 激活函数
            nn.Dropout(p=dropout),  # 随机失活
            zero_module(  # 零填充
                conv_nd(
                    dims,
                    self.out_channels,
                    self.out_channels,
                    kernel_size,
                    padding=padding,
                )
            ),
        )

        # 设置跳跃连接
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    # 前向传播函数
    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # 使用检查点函数进行前向传播
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)
    # 定义前向传播函数，接收输入 x 和嵌入 emb
        def _forward(self, x, emb):
            # 检查是否需要进行上下采样
            if self.updown:
                # 分离输入层中的最后一层和其他层
                in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
                # 通过其他层处理输入 x
                h = in_rest(x)
                # 更新隐藏状态 h
                h = self.h_upd(h)
                # 更新输入 x
                x = self.x_upd(x)
                # 最后一层对更新后的隐藏状态 h 进行处理
                h = in_conv(h)
            else:
                # 如果不需要上下采样，直接通过所有输入层处理 x
                h = self.in_layers(x)
    
            # 检查是否需要跳过时间嵌入
            if self.skip_t_emb:
                # 创建与 h 相同形状的全零张量
                emb_out = th.zeros_like(h)
            else:
                # 通过嵌入层处理 emb，转换为与 h 相同的数据类型
                emb_out = self.emb_layers(emb).type(h.dtype)
            # 将 emb_out 的形状扩展到与 h 的形状相同
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            # 检查是否使用缩放和偏移规范化
            if self.use_scale_shift_norm:
                # 获取输出层中的规范化层和剩余层
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                # 将 emb_out 分割为缩放和偏移
                scale, shift = th.chunk(emb_out, 2, dim=1)
                # 进行规范化，应用缩放和偏移
                h = out_norm(h) * (1 + scale) + shift
                # 处理剩余层
                h = out_rest(h)
            else:
                # 检查是否需要交换时间嵌入的维度
                if self.exchange_temb_dims:
                    # 重新排列 emb_out 的维度
                    emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
                # 将嵌入结果添加到隐藏状态 h 中
                h = h + emb_out
                # 通过输出层处理 h
                h = self.out_layers(h)
            # 返回跳过连接的结果与 h 的和
            return self.skip_connection(x) + h
# 定义一个注意力模块，允许空间位置之间相互关注
class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    # 初始化方法，设置注意力模块的参数
    def __init__(
        self,
        channels,  # 输入通道数
        num_heads=1,  # 注意力头的数量，默认为1
        num_head_channels=-1,  # 每个注意力头的通道数，默认为-1表示自动计算
        use_checkpoint=False,  # 是否使用检查点，默认为False
        use_new_attention_order=False,  # 是否使用新注意力顺序，默认为False
    ):
        super().__init__()  # 调用父类的初始化方法
        self.channels = channels  # 保存输入通道数
        if num_head_channels == -1:  # 如果没有指定每个头的通道数
            self.num_heads = num_heads  # 使用指定的头数量
        else:
            # 检查输入通道数是否可以被每个头的通道数整除
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            # 计算头的数量
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint  # 保存检查点使用标志
        self.norm = normalization(channels)  # 创建归一化层
        self.qkv = conv_nd(1, channels, channels * 3, 1)  # 创建用于计算Q、K、V的卷积层
        if use_new_attention_order:  # 如果使用新注意力顺序
            # 在拆分头之前拆分QKV
            self.attention = QKVAttention(self.num_heads)  # 创建新的QKV注意力实例
        else:
            # 在拆分QKV之前拆分头
            self.attention = QKVAttentionLegacy(self.num_heads)  # 创建旧的QKV注意力实例

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))  # 创建输出投影层

    # 前向传播方法
    def forward(self, x, **kwargs):
        # TODO add crossframe attention and use mixed checkpoint
        # 使用检查点进行前向传播
        return checkpoint(
            self._forward, (x,), self.parameters(), True  # 将输入和模型参数传递给检查点
        )  # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        # return pt_checkpoint(self._forward, x)  # pytorch

    # 实际的前向传播逻辑
    def _forward(self, x):
        b, c, *spatial = x.shape  # 解包输入形状为批次大小、通道数和空间维度
        x = x.reshape(b, c, -1)  # 将输入重新形状化为二维，方便计算
        qkv = self.qkv(self.norm(x))  # 通过归一化和卷积计算Q、K、V
        h = self.attention(qkv)  # 应用注意力机制
        h = self.proj_out(h)  # 通过投影层生成输出
        return (x + h).reshape(b, c, *spatial)  # 将结果形状恢复并返回

# 计算注意力操作中的浮点运算次数
def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape  # 解包输入形状为批次大小、通道数和空间维度
    num_spatial = int(np.prod(spatial))  # 计算空间维度的总数
    # 我们执行两个矩阵乘法，它们的运算次数相同
    # 第一个计算权重矩阵，第二个计算值向量的组合
    matmul_ops = 2 * b * (num_spatial**2) * c  # 计算矩阵乘法的操作数
    model.total_ops += th.DoubleTensor([matmul_ops])  # 将操作数累加到模型的总操作数中

# 定义旧版QKV注意力模块
class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    # 初始化方法，设置注意力头数量
    def __init__(self, n_heads):
        super().__init__()  # 调用父类的初始化方法
        self.n_heads = n_heads  # 保存注意力头数量
    # 定义前向传播方法，接受 QKV 张量作为输入
    def forward(self, qkv):
        # 文档字符串，描述函数用途及参数
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        # 获取输入张量的批次大小、宽度和长度
        bs, width, length = qkv.shape
        # 确保宽度可以被头数的三倍整除
        assert width % (3 * self.n_heads) == 0
        # 计算每个头的通道数
        ch = width // (3 * self.n_heads)
        # 将 QKV 张量重塑并分割为 Q、K 和 V
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        # 计算缩放因子
        scale = 1 / math.sqrt(math.sqrt(ch))
        # 计算注意力权重，使用爱因斯坦求和约定
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        # 对权重进行 softmax 归一化
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        # 根据权重和 V 计算加权和
        a = th.einsum("bts,bcs->bct", weight, v)
        # 将输出重塑为原始批次大小和通道数
        return a.reshape(bs, -1, length)
    
    # 定义静态方法以计算模型的浮点运算次数
    @staticmethod
    def count_flops(model, _x, y):
        # 调用辅助函数以计算注意力的浮点运算次数
        return count_flops_attn(model, _x, y)
# 定义一个 QKVAttention 类，继承自 nn.Module
class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    # 初始化方法，接收头数 n_heads 作为参数
    def __init__(self, n_heads):
        # 调用父类的初始化方法
        super().__init__()
        # 存储头数
        self.n_heads = n_heads

    # 前向传播方法，接受一个 qkv 张量
    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        # 解包 qkv 张量的形状
        bs, width, length = qkv.shape
        # 确保宽度可以被 3 * n_heads 整除
        assert width % (3 * self.n_heads) == 0
        # 计算每个头的通道数
        ch = width // (3 * self.n_heads)
        # 将 qkv 张量分成 q、k、v 三个部分
        q, k, v = qkv.chunk(3, dim=1)
        # 计算缩放因子
        scale = 1 / math.sqrt(math.sqrt(ch))
        # 计算权重，通过爱因斯坦求和表示法
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # 用 f16 进行计算更稳定，避免后续除法
        # 对权重进行 softmax 操作，归一化权重
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        # 计算注意力结果 a
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        # 返回结果，重新调整形状
        return a.reshape(bs, -1, length)

    # 静态方法，用于计算模型的浮点运算量
    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


# 定义 Timestep 类，继承自 nn.Module
class Timestep(nn.Module):
    # 初始化方法，接收维度 dim 作为参数
    def __init__(self, dim):
        # 调用父类的初始化方法
        super().__init__()
        # 存储维度
        self.dim = dim

    # 前向传播方法，接受时间 t
    def forward(self, t):
        # 计算时间嵌入并返回
        return timestep_embedding(t, self.dim)


# 定义一个字典，将字符串映射到对应的数据类型
str_to_dtype = {"fp32": th.float32, "fp16": th.float16, "bf16": th.bfloat16}


# 定义 UNetModel 类，继承自 nn.Module
class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    """
    # 参数 resblock_updown: 是否使用残差块进行上采样或下采样
    # 参数 use_new_attention_order: 是否使用不同的注意力模式以提高效率
    """

    # 初始化方法
    def __init__(
        self,
        # 输入通道数
        in_channels,
        # 模型通道数
        model_channels,
        # 输出通道数
        out_channels,
        # 残差块数量
        num_res_blocks,
        # 注意力分辨率
        attention_resolutions,
        # dropout 比例，默认为 0
        dropout=0,
        # 通道倍增参数，默认为 (1, 2, 4, 8)
        channel_mult=(1, 2, 4, 8),
        # 是否使用卷积重采样，默认为 True
        conv_resample=True,
        # 维度，默认为 2
        dims=2,
        # 类别数量，默认为 None
        num_classes=None,
        # 是否使用检查点，默认为 False
        use_checkpoint=False,
        # 是否使用半精度浮点数，默认为 False
        use_fp16=False,
        # 头部数量，默认为 -1
        num_heads=-1,
        # 每个头的通道数，默认为 -1
        num_head_channels=-1,
        # 上采样时的头部数量，默认为 -1
        num_heads_upsample=-1,
        # 是否使用缩放和位移归一化，默认为 False
        use_scale_shift_norm=False,
        # 是否在上采样和下采样中使用残差块，默认为 False
        resblock_updown=False,
        # 是否使用新的注意力顺序，默认为 False
        use_new_attention_order=False,
        # 是否使用空间变换器，支持自定义变换器
        use_spatial_transformer=False,  
        # 变换器深度，默认为 1
        transformer_depth=1,  
        # 上下文维度，默认为 None
        context_dim=None,  
        # 用于将离散 ID 预测到第一个阶段 VQ 模型的字典的自定义支持
        n_embed=None,  
        # 是否使用传统方式，默认为 True
        legacy=True,
        # 禁用自注意力，默认为 None
        disable_self_attentions=None,
        # 注意力块数量，默认为 None
        num_attention_blocks=None,
        # 禁用中间自注意力，默认为 False
        disable_middle_self_attn=False,
        # 在变换器中使用线性输入，默认为 False
        use_linear_in_transformer=False,
        # 空间变换器注意力类型，默认为 "softmax"
        spatial_transformer_attn_type="softmax",
        # ADM 输入通道数，默认为 None
        adm_in_channels=None,
        # 是否使用 Fairscale 检查点，默认为 False
        use_fairscale_checkpoint=False,
        # 是否将模型卸载到 CPU，默认为 False
        offload_to_cpu=False,
        # 中间变换器深度，默认为 None
        transformer_depth_middle=None,
        # 数据类型，默认为 "fp32"
        dtype="fp32",
        # 是否初始化 LoRA，默认为 False
        lora_init=False,
        # LoRA 等级，默认为 4
        lora_rank=4,
        # LoRA 缩放因子，默认为 1.0
        lora_scale=1.0,
        # LoRA 权重路径，默认为 None
        lora_weight_path=None,
    # 初始化 LoRA 方法
    def _init_lora(self, rank, scale, ckpt_dir=None):
        # 注入可训练的 LoRA 扩展
        inject_trainable_lora_extended(self, target_replace_module=None, rank=rank, scale=scale)

        # 如果提供了检查点目录
        if ckpt_dir is not None:
            # 打开最新文件，读取最新的检查点
            with open(os.path.join(ckpt_dir, "latest")) as latest_file:
                latest = latest_file.read().strip()
            # 构建检查点路径
            ckpt_path = os.path.join(ckpt_dir, latest, "mp_rank_00_model_states.pt")
            # 打印加载的 LoRA 路径
            print(f"loading lora from {ckpt_path}")
            # 从检查点加载模型状态字典
            sd = th.load(ckpt_path)["module"]
            # 处理模型状态字典，提取相关键
            sd = {
                key[len("model.diffusion_model") :]: sd[key] for key in sd if key.startswith("model.diffusion_model")
            }
            # 加载模型状态字典，严格模式设置为 False
            self.load_state_dict(sd, strict=False)

    # 更新 LoRA 缩放因子的函数
    def _update_scale(self, scale):
        # 调用更新缩放的方法
        update_lora_scale(self, scale)

    # 将模型的主干转换为浮点16的函数
    def convert_to_fp16(self):
        """
        将模型的主干转换为 float16。
        """
        # 应用转换函数到输入块
        self.input_blocks.apply(convert_module_to_f16)
        # 应用转换函数到中间块
        self.middle_block.apply(convert_module_to_f16)
        # 应用转换函数到输出块
        self.output_blocks.apply(convert_module_to_f16)

    # 将模型的主干转换为浮点32的函数
    def convert_to_fp32(self):
        """
        将模型的主干转换为 float32。
        """
        # 应用转换函数到输入块
        self.input_blocks.apply(convert_module_to_f32)
        # 应用转换函数到中间块
        self.middle_block.apply(convert_module_to_f32)
        # 应用转换函数到输出块
        self.output_blocks.apply(convert_module_to_f32)
    # 定义模型的前向传播方法，接受输入批次和可选参数
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        # 方法说明：对输入批次应用模型，返回输出
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # 确保只有在类条件模型时才提供标签 y
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        # 初始化隐藏状态列表
        hs = []
        # 获取时间步嵌入，用于模型的输入
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=self.dtype)
        # 通过时间嵌入生成时间特征
        emb = self.time_embed(t_emb)
    
        # 如果模型是类条件的，添加标签嵌入
        if self.num_classes is not None:
            # 确保输入和标签批次大小一致
            assert y.shape[0] == x.shape[0]
            # 将标签嵌入添加到时间特征中
            emb = emb + self.label_emb(y)
    
        # h = x.type(self.dtype)  # 将输入转换为模型的数据类型（已注释）
        h = x  # 使用输入 x 作为初始隐藏状态
        # 遍历输入块，依次处理输入数据
        for module in self.input_blocks:
            h = module(h, emb, context)  # 通过模块处理隐藏状态
            hs.append(h)  # 将当前隐藏状态添加到列表中
        # 处理中间块，更新隐藏状态
        h = self.middle_block(h, emb, context)
        # 遍历输出块，生成最终输出
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)  # 将当前隐藏状态与之前的状态合并
            h = module(h, emb, context)  # 通过模块处理合并后的隐藏状态
        # 将最终隐藏状态转换为输入的原始数据类型
        h = h.type(x.dtype)
        # 检查是否支持预测代码本ID
        if self.predict_codebook_ids:
            # 如果不支持，抛出异常
            assert False, "not supported anymore. what the f*** are you doing?"
        else:
            # 返回最终输出
            return self.out(h)
# 定义一个名为 NoTimeUNetModel 的类，继承自 UNetModel
class NoTimeUNetModel(UNetModel):
    # 定义前向传播方法，接受输入 x、时间步 timesteps、上下文 context 和 y 以及其他参数
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        # 将时间步初始化为与 timesteps 形状相同的零张量
        timesteps = th.zeros_like(timesteps)
        # 调用父类的前向传播方法并返回结果
        return super().forward(x, timesteps, context, y, **kwargs)


# 定义一个名为 EncoderUNetModel 的类，继承自 nn.Module
class EncoderUNetModel(nn.Module):
    """
    半个 UNet 模型，具有注意力和时间步嵌入功能。
    用法见 UNet。
    """

    # 定义初始化方法，接受多个参数设置模型的各项属性
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
        *args,
        **kwargs,
    ):
        # 调用父类初始化方法
        super().__init__()

    # 定义将模型转换为 float16 方法
    def convert_to_fp16(self):
        """
        将模型的主体转换为 float16。
        """
        # 对输入块和中间块应用转换函数
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    # 定义将模型转换为 float32 方法
    def convert_to_fp32(self):
        """
        将模型的主体转换为 float32。
        """
        # 对输入块和中间块应用转换函数
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    # 定义前向传播方法，接受输入 x 和时间步 timesteps
    def forward(self, x, timesteps):
        """
        将模型应用于输入批次。
        :param x: 输入的 [N x C x ...] 张量。
        :param timesteps: 一维时间步批次。
        :return: [N x K] 输出张量。
        """
        # 将时间步嵌入生成的向量传递给模型
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []  # 初始化结果列表
        # h = x.type(self.dtype)  # 可选：将输入张量转换为指定的数据类型
        h = x  # 将输入赋值给 h
        for module in self.input_blocks:
            # 逐个模块处理输入，并应用嵌入
            h = module(h, emb)
            # 如果池化方式是空间池化，添加平均值到结果列表
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        # 处理中间块
        h = self.middle_block(h, emb)
        # 如果池化方式是空间池化，添加平均值到结果列表
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            # 将结果在最后一个维度拼接
            h = th.cat(results, axis=-1)
            # 返回输出
            return self.out(h)
        else:
            # 如果不是空间池化，将 h 转换为输入的数据类型
            h = h.type(x.dtype)
            # 返回输出
            return self.out(h)


# 主程序入口
if __name__ == "__main__":

    # 定义一个名为 Dummy 的类，继承自 nn.Module
    class Dummy(nn.Module):
        # 初始化方法，接受输入通道和模型通道的参数
        def __init__(self, in_channels=3, model_channels=64):
            # 调用父类初始化方法
            super().__init__()
            # 创建一个输入块的模块列表，包含时间步嵌入的卷积层
            self.input_blocks = nn.ModuleList(
                [TimestepEmbedSequential(conv_nd(2, in_channels, model_channels, 3, padding=1))]
            )

    # 创建 UNetModel 实例，并将其移至 GPU
    model = UNetModel(
        use_checkpoint=True,  # 使用检查点
        image_size=64,  # 图像大小
        in_channels=4,  # 输入通道数
        out_channels=4,  # 输出通道数
        model_channels=128,  # 模型通道数
        attention_resolutions=[4, 2],  # 注意力分辨率
        num_res_blocks=2,  # 残差块数量
        channel_mult=[1, 2, 4],  # 通道倍增系数
        num_head_channels=64,  # 头通道数
        use_spatial_transformer=False,  # 不使用空间变换器
        use_linear_in_transformer=True,  # 在变换器中使用线性输入
        transformer_depth=1,  # 变换器深度
        legacy=False,  # 不是旧版
    ).cuda()  # 移至 GPU
    # 创建一个形状为 (11, 4, 64, 64) 的随机张量，并将其移到 GPU 上
        x = th.randn(11, 4, 64, 64).cuda()
        # 生成一个包含 11 个随机整数的张量，范围在 0 到 10 之间，设备为 GPU
        t = th.randint(low=0, high=10, size=(11,), device="cuda")
        # 使用模型处理输入张量 x 和标签 t，返回结果
        o = model(x, t)
        # 打印完成信息
        print("done.")
```