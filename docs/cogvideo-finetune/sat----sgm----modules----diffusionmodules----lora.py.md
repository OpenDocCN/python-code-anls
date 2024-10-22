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