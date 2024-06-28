# `.\models\regnet\convert_regnet_seer_10b_to_pytorch.py`

```py
# coding=utf-8
# 版权所有 2022 年 HuggingFace Inc. 团队
#
# 根据 Apache 许可证版本 2.0（“许可证”）许可；
# 除非符合许可证要求，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，
# 没有任何形式的明示或暗示保证或条件。
# 有关许可下的详细信息，请参阅许可证。
"""转换 RegNet 10B 检查点为 vissl 格式。"""
# 您需要安装 classy vision 的特定版本
# pip install git+https://github.com/FrancescoSaverioZuppichini/ClassyVision.git@convert_weights

import argparse
import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from classy_vision.models.regnet import RegNet, RegNetParams  # 导入 RegNet 相关模块
from huggingface_hub import cached_download, hf_hub_url  # 导入缓存下载和 HF Hub URL 相关模块
from torch import Tensor
from vissl.models.model_helpers import get_trunk_forward_outputs  # 导入 vissl 模型助手函数

from transformers import AutoImageProcessor, RegNetConfig, RegNetForImageClassification, RegNetModel
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging


logging.set_verbosity_info()  # 设置日志记录级别为 info
logger = logging.get_logger()  # 获取日志记录器


@dataclass
class Tracker:
    """
    追踪器类，用于跟踪模块的前向传播过程，并记录子模块和参数信息。
    """
    module: nn.Module  # 要追踪的模块
    traced: List[nn.Module] = field(default_factory=list)  # 记录已追踪的模块列表
    handles: list = field(default_factory=list)  # 模块注册的钩子句柄列表
    name2module: Dict[str, nn.Module] = field(default_factory=OrderedDict)  # 模块名称到模块对象的字典

    def _forward_hook(self, m, inputs: Tensor, outputs: Tensor, name: str):
        """
        前向传播钩子函数，用于处理模块的前向传播输出。
        """
        has_not_submodules = len(list(m.modules())) == 1 or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)
        if has_not_submodules:
            self.traced.append(m)
            self.name2module[name] = m

    def __call__(self, x: Tensor):
        """
        执行追踪器对象，注册前向传播钩子，并进行模块的前向传播。
        """
        for name, m in self.module.named_modules():
            self.handles.append(m.register_forward_hook(partial(self._forward_hook, name=name)))
        self.module(x)
        [x.remove() for x in self.handles]  # 移除注册的所有前向传播钩子
        return self

    @property
    def parametrized(self):
        """
        属性方法，返回具有可学习参数的模块字典。
        """
        return {k: v for k, v in self.name2module.items() if len(list(v.state_dict().keys())) > 0}


class FakeRegNetVisslWrapper(nn.Module):
    """
    模拟 vissl 操作而无需传递配置文件的 RegNet 包装器。
    """
    pass
    # 初始化函数，用于创建一个特征提取器对象
    def __init__(self, model: nn.Module):
        # 调用父类的初始化方法
        super().__init__()

        # 定义特征块列表，用于存储特征块的名称和对应的模块
        feature_blocks: List[Tuple[str, nn.Module]] = []

        # 添加模型的起始卷积层作为特征块 "conv1"
        feature_blocks.append(("conv1", model.stem))

        # 遍历模型的主干输出的每个子模块
        for k, v in model.trunk_output.named_children():
            # 断言子模块的名称以 "block" 开头，以确保符合预期
            assert k.startswith("block"), f"Unexpected layer name {k}"

            # 计算当前特征块的索引
            block_index = len(feature_blocks) + 1

            # 添加当前子模块作为特征块 "resN"，其中 N 是索引
            feature_blocks.append((f"res{block_index}", v))

        # 使用特征块列表创建 nn.ModuleDict 对象，用于管理特征块
        self._feature_blocks = nn.ModuleDict(feature_blocks)

    # 前向传播函数，接受输入张量 x，并返回特征提取器的输出
    def forward(self, x: Tensor):
        # 调用 get_trunk_forward_outputs 函数获取主干网络的前向传播输出
        return get_trunk_forward_outputs(
            x,
            out_feat_keys=None,  # 不指定输出特征键值，表示返回所有特征块的输出
            feature_blocks=self._feature_blocks,  # 使用初始化时创建的特征块字典
        )
class FakeRegNetParams(RegNetParams):
    """
    Used to instantiate a RegNet model from Classy Vision with the same depth as the 10B one but with super small
    parameters, so we can trace it in memory.
    """

    def get_expanded_params(self):
        # 返回一个列表，每个元素是一个元组，描述了不同配置的参数
        return [(8, 2, 2, 8, 1.0), (8, 2, 7, 8, 1.0), (8, 2, 17, 8, 1.0), (8, 2, 1, 8, 1.0)]


def get_from_to_our_keys(model_name: str) -> Dict[str, str]:
    """
    Returns a dictionary that maps from original model's key -> our implementation's keys
    """

    # 创建我们的模型（使用小的权重）
    our_config = RegNetConfig(depths=[2, 7, 17, 1], hidden_sizes=[8, 8, 8, 8], groups_width=8)
    if "in1k" in model_name:
        our_model = RegNetForImageClassification(our_config)
    else:
        our_model = RegNetModel(our_config)

    # 创建原始模型（使用小的权重）
    from_model = FakeRegNetVisslWrapper(
        RegNet(FakeRegNetParams(depth=27, group_width=1010, w_0=1744, w_a=620.83, w_m=2.52))
    )

    with torch.no_grad():
        from_model = from_model.eval()
        our_model = our_model.eval()

        x = torch.randn((1, 3, 32, 32))
        # 对两个模型进行追踪
        dest_tracker = Tracker(our_model)
        dest_traced = dest_tracker(x).parametrized

        pprint(dest_tracker.name2module)
        src_tracker = Tracker(from_model)
        src_traced = src_tracker(x).parametrized

    # 将模块字典转换为参数字典
    def to_params_dict(dict_with_modules):
        params_dict = OrderedDict()
        for name, module in dict_with_modules.items():
            for param_name, param in module.state_dict().items():
                params_dict[f"{name}.{param_name}"] = param
        return params_dict

    from_to_ours_keys = {}

    src_state_dict = to_params_dict(src_traced)
    dst_state_dict = to_params_dict(dest_traced)

    # 将原始模型和我们模型的键映射关系存储到字典中
    for (src_key, src_param), (dest_key, dest_param) in zip(src_state_dict.items(), dst_state_dict.items()):
        from_to_ours_keys[src_key] = dest_key
        logger.info(f"{src_key} -> {dest_key}")

    # 如果模型名中包含 "in1k"，则表明它可能有一个分类头（经过微调）
    if "in1k" in model_name:
        from_to_ours_keys["0.clf.0.weight"] = "classifier.1.weight"
        from_to_ours_keys["0.clf.0.bias"] = "classifier.1.bias"

    return from_to_ours_keys


def convert_weights_and_push(save_directory: Path, model_name: str = None, push_to_hub: bool = True):
    filename = "imagenet-1k-id2label.json"
    num_labels = 1000

    repo_id = "huggingface/label-files"
    num_labels = num_labels
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    id2label = id2label
    label2id = {v: k for k, v in id2label.items()}

    # 使用部分函数创建 ImageNetPreTrainedConfig 对象
    ImageNetPreTrainedConfig = partial(RegNetConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)
    # 定义一个字典，映射模型名称到预训练配置对象
    names_to_config = {
        "regnet-y-10b-seer": ImageNetPreTrainedConfig(
            depths=[2, 7, 17, 1], hidden_sizes=[2020, 4040, 11110, 28280], groups_width=1010
        ),
        # 在 ImageNet 上微调
        "regnet-y-10b-seer-in1k": ImageNetPreTrainedConfig(
            depths=[2, 7, 17, 1], hidden_sizes=[2020, 4040, 11110, 28280], groups_width=1010
        ),
    }

    # 添加 SEER 模型权重逻辑
    def load_using_classy_vision(checkpoint_url: str) -> Tuple[Dict, Dict]:
        # 从给定 URL 加载模型状态字典，保存在内存中，并映射到 CPU
        files = torch.hub.load_state_dict_from_url(checkpoint_url, model_dir=str(save_directory), map_location="cpu")
        # 检查是否有头部信息，如果有，则添加到模型状态字典中
        model_state_dict = files["classy_state_dict"]["base_model"]["model"]
        return model_state_dict["trunk"], model_state_dict["heads"]

    # 定义一个字典，将模型名称映射到从 URL 加载模型状态字典的部分函数
    names_to_from_model = {
        "regnet-y-10b-seer": partial(
            load_using_classy_vision,
            "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet10B/model_iteration124500_conso.torch",
        ),
        "regnet-y-10b-seer-in1k": partial(
            load_using_classy_vision,
            "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_10b_finetuned_in1k_model_phase28_conso.torch",
        ),
    }

    # 获取从原始模型到我们模型键的映射
    from_to_ours_keys = get_from_to_our_keys(model_name)

    # 检查是否已经存在模型的状态字典文件
    if not (save_directory / f"{model_name}.pth").exists():
        logger.info("Loading original state_dict.")
        # 加载模型的原始状态字典 trunk 和 head
        from_state_dict_trunk, from_state_dict_head = names_to_from_model[model_name]()
        from_state_dict = from_state_dict_trunk
        if "in1k" in model_name:
            # 如果模型名称中包含 "in1k"，则将头部信息添加到模型状态字典中
            from_state_dict = {**from_state_dict_trunk, **from_state_dict_head}
        logger.info("Done!")

        # 创建一个空字典来存储转换后的状态字典
        converted_state_dict = {}

        # 初始化未使用的键列表
        not_used_keys = list(from_state_dict.keys())
        # 定义一个正则表达式来匹配要移除的模型键中的特定字符串
        regex = r"\.block.-part."
        # 迭代处理原始模型状态字典的每个键
        for key in from_state_dict.keys():
            # 从模型键中移除特定字符串以获取源键
            src_key = re.sub(regex, "", key)
            # 使用映射表将源键转换为我们模型的目标键
            dest_key = from_to_ours_keys[src_key]
            # 将参数与目标键存储到转换后的状态字典中
            converted_state_dict[dest_key] = from_state_dict[key]
            # 从未使用的键列表中移除当前键
            not_used_keys.remove(key)
        # 检查是否所有的键都已经更新
        assert len(not_used_keys) == 0, f"Some keys where not used {','.join(not_used_keys)}"

        logger.info(f"The following keys were not used: {','.join(not_used_keys)}")

        # 将转换后的状态字典保存到磁盘
        torch.save(converted_state_dict, save_directory / f"{model_name}.pth")

        # 释放转换后的状态字典的内存
        del converted_state_dict
    else:
        logger.info("The state_dict was already stored on disk.")
    # 如果需要将模型推送到 Hub
    if push_to_hub:
        # 记录环境变量中的 HF_TOKEN
        logger.info(f"Token is {os.environ['HF_TOKEN']}")
        # 输出信息：加载我们的模型
        logger.info("Loading our model.")
        # 根据模型名称获取配置
        our_config = names_to_config[model_name]
        # 默认使用 RegNetModel 作为模型函数
        our_model_func = RegNetModel
        # 如果模型名称中包含 "in1k"，则使用 RegNetForImageClassification
        if "in1k" in model_name:
            our_model_func = RegNetForImageClassification
        # 创建我们的模型实例
        our_model = our_model_func(our_config)
        # 将我们的模型放置到 meta 设备上（移除所有权重）
        our_model.to(torch.device("meta"))
        # 输出信息：在我们的模型中加载 state_dict
        logger.info("Loading state_dict in our model.")
        # 获取我们模型当前的 state_dict 的键集合
        state_dict_keys = our_model.state_dict().keys()
        # 以低内存方式加载预训练模型
        PreTrainedModel._load_pretrained_model_low_mem(
            our_model, state_dict_keys, [save_directory / f"{model_name}.pth"]
        )
        # 输出信息：最终进行推送操作
        logger.info("Finally, pushing!")
        # 将模型推送到 Hub
        our_model.push_to_hub(
            repo_path_or_name=save_directory / model_name,
            commit_message="Add model",
            output_dir=save_directory / model_name,
        )
        # 设定图像处理器的尺寸
        size = 384
        # 输出信息：我们可以使用 convnext 模型
        logger.info("we can use the convnext one")
        # 从预训练模型 facebook/convnext-base-224-22k-1k 创建图像处理器实例
        image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224-22k-1k", size=size)
        # 将图像处理器推送到 Hub
        image_processor.push_to_hub(
            repo_path_or_name=save_directory / model_name,
            commit_message="Add image processor",
            output_dir=save_directory / model_name,
        )
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # Required parameters
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help=(
            "The name of the model you wish to convert, it must be one of the supported regnet* architecture,"
            " currently: regnetx-*, regnety-*. If `None`, all of them will the converted."
        ),
    )
    # 添加名为 `--model_name` 的参数，用于指定要转换的模型名称，必须是支持的 regnet* 架构之一

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=Path,
        required=True,
        help="Path to the output PyTorch model directory.",
    )
    # 添加名为 `--pytorch_dump_folder_path` 的参数，指定输出的 PyTorch 模型目录的路径，此参数为必选

    parser.add_argument(
        "--push_to_hub",
        default=True,
        type=bool,
        required=False,
        help="If True, push model and image processor to the hub.",
    )
    # 添加名为 `--push_to_hub` 的参数，如果设置为 True，则推送模型和图像处理器到指定的 Hub

    args = parser.parse_args()
    # 解析命令行参数并存储到 `args` 对象中

    pytorch_dump_folder_path: Path = args.pytorch_dump_folder_path
    # 从参数对象 `args` 中获取 PyTorch 模型目录路径，并赋值给变量 `pytorch_dump_folder_path`

    pytorch_dump_folder_path.mkdir(exist_ok=True, parents=True)
    # 创建 PyTorch 模型目录，如果目录已存在则忽略，同时创建必要的父目录

    convert_weights_and_push(pytorch_dump_folder_path, args.model_name, args.push_to_hub)
    # 调用函数 `convert_weights_and_push`，将 PyTorch 模型目录路径、模型名称和推送标志作为参数传递给该函数
```