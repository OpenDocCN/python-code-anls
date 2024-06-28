# `.\models\deprecated\van\convert_van_to_pytorch.py`

```
# coding=utf-8
# 声明编码格式为 UTF-8
# Copyright 2022 BNRist (Tsinghua University), TKLNDST (Nankai University) and The HuggingFace Inc. team. All rights reserved.
# 版权声明：2022 年 BNRist（清华大学）、TKLNDST（南开大学）及 The HuggingFace Inc. 团队保留所有权利。

# Licensed under the Apache License, Version 2.0 (the "License");
# 授权许可，使用 Apache 许可版本 2.0
# you may not use this file except in compliance with the License.
# 除非符合许可，否则不得使用此文件。
# You may obtain a copy of the License at
# 可以在以下网址获取许可的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，本软件根据“原样”分发，不附带任何明示或暗示的担保或条件。
# See the License for the specific language governing permissions and
# 请参阅许可，了解特定语言的权限和限制。
# limitations under the License.
"""Convert VAN checkpoints from the original repository.

将原始仓库的 VAN 检查点转换为特定格式。
URL: https://github.com/Visual-Attention-Network/VAN-Classification"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
# 导入模块
from huggingface_hub import cached_download, hf_hub_download
# 从 torch 模块导入 Tensor 类型
from torch import Tensor

# 从 transformers 模块导入 AutoImageProcessor, VanConfig, VanForImageClassification 类
from transformers import AutoImageProcessor, VanConfig, VanForImageClassification
# 从 transformers.models.deprecated.van.modeling_van 模块导入 VanLayerScaling 类
from transformers.models.deprecated.van.modeling_van import VanLayerScaling
# 从 transformers.utils 模块导入 logging 函数
from transformers.utils import logging

# 设置日志输出级别为 INFO
logging.set_verbosity_info()
# 获取当前模块的 logger
logger = logging.get_logger(__name__)

# 定义 Tracker 类，用于跟踪模块
@dataclass
class Tracker:
    module: nn.Module
    # 被追踪的模块列表
    traced: List[nn.Module] = field(default_factory=list)
    # 注册的钩子列表
    handles: list = field(default_factory=list)

    # 前向钩子函数
    def _forward_hook(self, m, inputs: Tensor, outputs: Tensor):
        # 检查模块是否没有子模块或者是 Conv2d 或 BatchNorm2d 类型
        has_not_submodules = len(list(m.modules())) == 1 or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)
        if has_not_submodules:
            # 排除 VanLayerScaling 类型模块
            if not isinstance(m, VanLayerScaling):
                self.traced.append(m)

    # 调用实例时执行的函数，用于模块跟踪
    def __call__(self, x: Tensor):
        # 遍历模块的所有子模块，并注册前向钩子
        for m in self.module.modules():
            self.handles.append(m.register_forward_hook(self._forward_hook))
        # 执行模块的前向传播
        self.module(x)
        # 移除所有注册的钩子
        [x.remove() for x in self.handles]
        return self

    # 返回具有参数的被追踪模块列表
    @property
    def parametrized(self):
        return list(filter(lambda x: len(list(x.state_dict().keys())) > 0, self.traced))


# 定义 ModuleTransfer 类，用于模块转移
@dataclass
class ModuleTransfer:
    src: nn.Module
    dest: nn.Module
    verbose: int = 0
    src_skip: List = field(default_factory=list)
    dest_skip: List = field(default_factory=list))
    # 定义一个方法，使对象实例可以像函数一样被调用，接受一个张量 `x` 作为参数
    def __call__(self, x: Tensor):
        """
        Transfer the weights of `self.src` to `self.dest` by performing a forward pass using `x` as input. Under the
        hood we tracked all the operations in both modules.
        """
        # 对目标模块 `self.dest` 执行跟踪操作，并获取其参数化表示
        dest_traced = Tracker(self.dest)(x).parametrized
        # 对源模块 `self.src` 执行跟踪操作，并获取其参数化表示
        src_traced = Tracker(self.src)(x).parametrized

        # 过滤掉在 `self.src_skip` 中指定的类型的参数化表示
        src_traced = list(filter(lambda x: type(x) not in self.src_skip, src_traced))
        # 过滤掉在 `self.dest_skip` 中指定的类型的参数化表示
        dest_traced = list(filter(lambda x: type(x) not in self.dest_skip, dest_traced))

        # 如果目标模块和源模块的操作数量不同，则抛出异常
        if len(dest_traced) != len(src_traced):
            raise Exception(
                f"Numbers of operations are different. Source module has {len(src_traced)} operations while"
                f" destination module has {len(dest_traced)}."
            )

        # 逐个加载源模块的状态字典到目标模块中
        for dest_m, src_m in zip(dest_traced, src_traced):
            dest_m.load_state_dict(src_m.state_dict())
            # 如果设置了详细输出模式 (`verbose == 1`)，则打印迁移信息
            if self.verbose == 1:
                print(f"Transfered from={src_m} to={dest_m}")
# 复制源模型的参数到目标模型中，确保两者结构兼容
def copy_parameters(from_model: nn.Module, our_model: nn.Module) -> nn.Module:
    # 获取源模型的状态字典
    from_state_dict = from_model.state_dict()
    # 获取目标模型的状态字典
    our_state_dict = our_model.state_dict()
    # 获取目标模型的配置信息
    config = our_model.config
    # 初始化一个空列表用于存储所有需要复制的键值对
    all_keys = []
    # 遍历配置中的隐藏层尺寸列表
    for stage_idx in range(len(config.hidden_sizes)):
        # 根据深度遍历每个阶段的块数量
        for block_id in range(config.depths[stage_idx]):
            # 构建源模型中的键名
            from_key = f"block{stage_idx + 1}.{block_id}.layer_scale_1"
            # 构建目标模型中对应的键名
            to_key = f"van.encoder.stages.{stage_idx}.layers.{block_id}.attention_scaling.weight"
            # 将源模型键名和目标模型键名作为元组加入列表
            all_keys.append((from_key, to_key))
            # 类似地，构建另一个键对应关系用于 MLP 缩放权重
            from_key = f"block{stage_idx + 1}.{block_id}.layer_scale_2"
            to_key = f"van.encoder.stages.{stage_idx}.layers.{block_id}.mlp_scaling.weight"
            # 将键名和目标键名作为元组加入列表
            all_keys.append((from_key, to_key))

    # 遍历复制源模型到目标模型的所有键值对
    for from_key, to_key in all_keys:
        our_state_dict[to_key] = from_state_dict.pop(from_key)

    # 使用复制后的状态字典更新目标模型的参数
    our_model.load_state_dict(our_state_dict)
    # 返回更新后的目标模型
    return our_model


# 下载和转换权重，并将模型推送到Hub
def convert_weight_and_push(
    name: str,
    config: VanConfig,
    checkpoint: str,
    from_model: nn.Module,
    save_directory: Path,
    push_to_hub: bool = True,
):
    # 打印正在下载权重信息
    print(f"Downloading weights for {name}...")
    # 缓存下载检查点路径
    checkpoint_path = cached_download(checkpoint)
    # 打印转换模型信息
    print(f"Converting {name}...")
    # 从检查点加载源模型的状态字典
    from_state_dict = torch.load(checkpoint_path)["state_dict"]
    # 加载源模型的状态字典到源模型
    from_model.load_state_dict(from_state_dict)
    # 设置源模型为评估模式
    from_model.eval()
    # 禁用梯度计算
    with torch.no_grad():
        # 创建用于图像分类的 VanForImageClassification 模型，并设置为评估模式
        our_model = VanForImageClassification(config).eval()
        # 创建 ModuleTransfer 实例，用于从源模型传输参数到目标模型
        module_transfer = ModuleTransfer(src=from_model, dest=our_model)
        # 创建随机输入张量
        x = torch.randn((1, 3, 224, 224))
        # 通过 module_transfer 将源模型的参数传输到目标模型
        module_transfer(x)
        # 使用 copy_parameters 函数复制源模型的参数到目标模型
        our_model = copy_parameters(from_model, our_model)

    # 检查源模型和目标模型的输出是否接近，否则引发异常
    if not torch.allclose(from_model(x), our_model(x).logits):
        raise ValueError("The model logits don't match the original one.")

    # 设置检查点名称为模型名称
    checkpoint_name = name
    # 打印检查点名称
    print(checkpoint_name)

    # 如果设置为推送到Hub，则执行以下操作
    if push_to_hub:
        # 将模型推送到Hub
        our_model.push_to_hub(
            repo_path_or_name=save_directory / checkpoint_name,
            commit_message="Add model",
            use_temp_dir=True,
        )

        # 使用预训练模型 facebook/convnext-base-224-22k-1k 创建图像处理器实例
        image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224-22k-1k")
        # 将图像处理器推送到Hub
        image_processor.push_to_hub(
            repo_path_or_name=save_directory / checkpoint_name,
            commit_message="Add image processor",
            use_temp_dir=True,
        )

        # 打印推送成功信息
        print(f"Pushed {checkpoint_name}")


# 下载权重文件并将模型参数保存在本地
def convert_weights_and_push(save_directory: Path, model_name: str = None, push_to_hub: bool = True):
    # 定义文件名
    filename = "imagenet-1k-id2label.json"
    # 类别数
    num_labels = 1000
    # Hub repo ID
    repo_id = "huggingface/label-files"
    # 从 Hub 下载类标签文件并加载为 JSON
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    # 转换 JSON 中的键值对为整数类型
    id2label = {int(k): v for k, v in id2label.items()}
    # 将 id2label 保存为新的变量
    id2label = id2label
    # 构建 label2id 字典，键为类别名称，值为类别 ID
    label2id = {v: k for k, v in id2label.items()}
    # 创建一个部分应用了 VanConfig 的函数 ImageNetPreTrainedConfig，用于配置预训练模型的参数
    ImageNetPreTrainedConfig = partial(VanConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)
    
    # 定义一个字典 names_to_config，包含不同模型名称到其对应配置的映射
    names_to_config = {
        "van-tiny": ImageNetPreTrainedConfig(
            hidden_sizes=[32, 64, 160, 256],
            depths=[3, 3, 5, 2],
            mlp_ratios=[8, 8, 4, 4],
        ),
        "van-small": ImageNetPreTrainedConfig(
            hidden_sizes=[64, 128, 320, 512],
            depths=[2, 2, 4, 2],
            mlp_ratios=[8, 8, 4, 4],
        ),
        "van-base": ImageNetPreTrainedConfig(
            hidden_sizes=[64, 128, 320, 512],
            depths=[3, 3, 12, 3],
            mlp_ratios=[8, 8, 4, 4],
        ),
        "van-large": ImageNetPreTrainedConfig(
            hidden_sizes=[64, 128, 320, 512],
            depths=[3, 5, 27, 3],
            mlp_ratios=[8, 8, 4, 4],
        ),
    }
    
    # 定义一个字典 names_to_original_models，包含不同模型名称到其原始模型的映射
    names_to_original_models = {
        "van-tiny": van_tiny,
        "van-small": van_small,
        "van-base": van_base,
        "van-large": van_large,
    }
    
    # 定义一个字典 names_to_original_checkpoints，包含不同模型名称到其原始检查点的 URL 映射
    names_to_original_checkpoints = {
        "van-tiny": (
            "https://huggingface.co/Visual-Attention-Network/VAN-Tiny-original/resolve/main/van_tiny_754.pth.tar"
        ),
        "van-small": (
            "https://huggingface.co/Visual-Attention-Network/VAN-Small-original/resolve/main/van_small_811.pth.tar"
        ),
        "van-base": (
            "https://huggingface.co/Visual-Attention-Network/VAN-Base-original/resolve/main/van_base_828.pth.tar"
        ),
        "van-large": (
            "https://huggingface.co/Visual-Attention-Network/VAN-Large-original/resolve/main/van_large_839.pth.tar"
        ),
    }
    
    # 如果指定了模型名称，则将该模型的配置和原始模型转换并推送到指定目录或 Hub
    if model_name:
        convert_weight_and_push(
            model_name,
            names_to_config[model_name],
            checkpoint=names_to_original_checkpoints[model_name],
            from_model=names_to_original_models[model_name](),
            save_directory=save_directory,
            push_to_hub=push_to_hub,
        )
    # 否则，遍历所有模型名称及其配置，并将每个模型的配置和原始模型转换并推送到指定目录或 Hub
    else:
        for model_name, config in names_to_config.items():
            convert_weight_and_push(
                model_name,
                config,
                checkpoint=names_to_original_checkpoints[model_name],
                from_model=names_to_original_models[model_name](),
                save_directory=save_directory,
                push_to_hub=push_to_hub,
            )
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块
    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必需参数
    parser.add_argument(
        "--model-name",
        default=None,
        type=str,
        help=(
            "The name of the model you wish to convert, it must be one of the supported resnet* architecture,"
            " currently: van-tiny/small/base/large. If `None`, all of them will the converted."
        ),
    )
    # 添加模型名称参数，指定要转换的模型名称

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=Path,
        required=True,
        help="Path to the output PyTorch model directory.",
    )
    # 添加参数，指定输出的 PyTorch 模型目录的路径，此参数为必需

    parser.add_argument(
        "--van_dir",
        required=True,
        type=Path,
        help=(
            "A path to VAN's original implementation directory. You can download from here:"
            " https://github.com/Visual-Attention-Network/VAN-Classification"
        ),
    )
    # 添加参数，指定 VAN（Visual Attention Network）原始实现的目录路径

    parser.add_argument(
        "--push_to_hub",
        default=True,
        type=bool,
        required=False,
        help="If True, push model and image processor to the hub.",
    )
    # 添加参数，指定是否将模型和图像处理器推送到 Hub

    args = parser.parse_args()
    # 解析命令行参数并存储到 args 对象中

    pytorch_dump_folder_path: Path = args.pytorch_dump_folder_path
    # 从 args 对象中获取 PyTorch 模型输出目录的路径，并指定其类型为 Path
    pytorch_dump_folder_path.mkdir(exist_ok=True, parents=True)
    # 如果指定的 PyTorch 模型输出目录不存在，则创建该目录，允许创建多层父目录

    van_dir = args.van_dir
    # 从 args 对象中获取 VAN 实现目录的路径

    # 将 VAN 实现目录的父目录路径添加到 sys.path 中，以便引入 maskformer 目录
    sys.path.append(str(van_dir.parent))
    from van.models.van import van_base, van_large, van_small, van_tiny
    # 从 VAN 实现中导入不同规模的 VAN 模型

    # 调用函数，将权重转换并推送到 Hub
    convert_weights_and_push(pytorch_dump_folder_path, args.model_name, args.push_to_hub)
```