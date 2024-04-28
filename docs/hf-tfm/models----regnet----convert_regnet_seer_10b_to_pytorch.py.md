# `.\transformers\models\regnet\convert_regnet_seer_10b_to_pytorch.py`

```py
# 导入必要的库和模块
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
from classy_vision.models.regnet import RegNet, RegNetParams
from huggingface_hub import cached_download, hf_hub_url
from torch import Tensor
from vissl.models.model_helpers import get_trunk_forward_outputs

from transformers import AutoImageProcessor, RegNetConfig, RegNetForImageClassification, RegNetModel
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging


# 设置日志级别为信息级别
logging.set_verbosity_info()
logger = logging.get_logger()


# 定义一个跟踪器类，用于跟踪模型中的模块
@dataclass
class Tracker:
    # 模型
    module: nn.Module
    # 跟踪到的模块列表
    traced: List[nn.Module] = field(default_factory=list)
    # 注册的钩子列表
    handles: list = field(default_factory=list)
    # 模块名称到模块实例的映射
    name2module: Dict[str, nn.Module] = field(default_factory=OrderedDict)

    # 前向传播钩子函数
    def _forward_hook(self, m, inputs: Tensor, outputs: Tensor, name: str):
        # 如果模块没有子模块或者是卷积层或者是批归一化层，则将其加入到跟踪列表中
        has_not_submodules = len(list(m.modules())) == 1 or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)
        if has_not_submodules:
            self.traced.append(m)
            self.name2module[name] = m

    # 执行前向传播并注册钩子
    def __call__(self, x: Tensor):
        for name, m in self.module.named_modules():
            self.handles.append(m.register_forward_hook(partial(self._forward_hook, name=name)))
        self.module(x)
        [x.remove() for x in self.handles]
        return self

    # 获取有可学习参数的模块
    @property
    def parametrized(self):
        return {k: v for k, v in self.name2module.items() if len(list(v.state_dict().keys())) > 0}


# 定义一个假的 RegNet VisslWrapper 类，模拟 vissl 的行为
class FakeRegNetVisslWrapper(nn.Module):
    """
    Fake wrapper for RegNet that mimics what vissl does without the need to pass a config file.
    """
    # 初始化函数，接受一个 nn.Module 对象作为参数
    def __init__(self, model: nn.Module):
        # 调用父类的初始化方法
        super().__init__()

        # 定义特征块列表，每个元素是一个元组，包含名称和 nn.Module 对象
        feature_blocks: List[Tuple[str, nn.Module]] = []
        # - 获取模型的 stem（干部）部分
        feature_blocks.append(("conv1", model.stem))
        # - 获取所有特征块
        for k, v in model.trunk_output.named_children():
            # 检查子层名称是否以 "block" 开头
            assert k.startswith("block"), f"Unexpected layer name {k}"
            # 计算特征块索引并添加到特征块列表中
            block_index = len(feature_blocks) + 1
            feature_blocks.append((f"res{block_index}", v))

        # 使用 nn.ModuleDict 将特征块列表转换为模块字典
        self._feature_blocks = nn.ModuleDict(feature_blocks)

    # 前向传播函数，接受一个张量作为输入
    def forward(self, x: Tensor):
        # 调用函数获取 trunk（主干）的前向输出
        return get_trunk_forward_outputs(
            x,
            out_feat_keys=None,
            feature_blocks=self._feature_blocks,
        )
# 定义一个 FakeRegNetParams 类,继承自 RegNetParams 类
# 这个类用于使用与 10B 模型深度相同但参数超级小的 RegNet 模型进行实例化,以便在内存中进行跟踪
class FakeRegNetParams(RegNetParams):
    def get_expanded_params(self):
        return [(8, 2, 2, 8, 1.0), (8, 2, 7, 8, 1.0), (8, 2, 17, 8, 1.0), (8, 2, 1, 8, 1.0)]


# 定义一个函数,返回一个字典,将原始模型的键映射到我们实现的键
def get_from_to_our_keys(model_name: str) -> Dict[str, str]:
    # 创建一个 RegNetConfig 对象,包含小权重的参数
    our_config = RegNetConfig(depths=[2, 7, 17, 1], hidden_sizes=[8, 8, 8, 8], groups_width=8)
    # 如果模型名包含 "in1k",则创建一个 RegNetForImageClassification 对象
    # 否则创建一个 RegNetModel 对象
    if "in1k" in model_name:
        our_model = RegNetForImageClassification(our_config)
    else:
        our_model = RegNetModel(our_config)
    
    # 创建一个 FakeRegNetVisslWrapper 对象,包含一个 RegNet 对象,它使用 FakeRegNetParams 进行实例化
    from_model = FakeRegNetVisslWrapper(
        RegNet(FakeRegNetParams(depth=27, group_width=1010, w_0=1744, w_a=620.83, w_m=2.52))
    )

    # 对两个模型进行评估
    with torch.no_grad():
        from_model = from_model.eval()
        our_model = our_model.eval()

        # 创建一个随机输入张量
        x = torch.randn((1, 3, 32, 32))
        
        # 跟踪两个模型
        dest_tracker = Tracker(our_model)
        dest_traced = dest_tracker(x).parametrized

        src_tracker = Tracker(from_model)
        src_traced = src_tracker(x).parametrized

    # 将模块字典转换为参数字典
    def to_params_dict(dict_with_modules):
        params_dict = OrderedDict()
        for name, module in dict_with_modules.items():
            for param_name, param in module.state_dict().items():
                params_dict[f"{name}.{param_name}"] = param
        return params_dict

    # 创建一个字典,将原始模型的键映射到我们实现的键
    from_to_ours_keys = {}
    src_state_dict = to_params_dict(src_traced)
    dst_state_dict = to_params_dict(dest_traced)

    for (src_key, src_param), (dest_key, dest_param) in zip(src_state_dict.items(), dst_state_dict.items()):
        from_to_ours_keys[src_key] = dest_key
        logger.info(f"{src_key} -> {dest_key}")

    # 如果模型名包含 "in1k",则添加分类头的映射
    if "in1k" in model_name:
        from_to_ours_keys["0.clf.0.weight"] = "classifier.1.weight"
        from_to_ours_keys["0.clf.0.bias"] = "classifier.1.bias"

    return from_to_ours_keys


# 定义一个函数,用于转换权重并推送到 HuggingFace Hub
def convert_weights_and_push(save_directory: Path, model_name: str = None, push_to_hub: bool = True):
    # 定义 ImageNet 标签文件的路径和标签数量
    filename = "imagenet-1k-id2label.json"
    num_labels = 1000
    
    # 从 HuggingFace Hub 下载标签文件
    repo_id = "huggingface/label-files"
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    
    # 创建一个偏函数,用于创建 RegNetConfig 对象
    ImageNetPreTrainedConfig = partial(RegNetConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)
    # 定义一个字典，将模型名称映射到预训练配置对象
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
        # 从给定的 URL 加载模型状态字典到内存
        files = torch.hub.load_state_dict_from_url(checkpoint_url, model_dir=str(save_directory), map_location="cpu")
        # 检查是否存在模型头，如果有，添加头部信息
        model_state_dict = files["classy_state_dict"]["base_model"]["model"]
        return model_state_dict["trunk"], model_state_dict["heads"]

    # 定义一个字典，将模型名称映射到加载模型权重函数的部分应用
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

    # 获取模型参数键的转换映射
    from_to_ours_keys = get_from_to_our_keys(model_name)

    # 如果模型文件不存在，则执行以下操作
    if not (save_directory / f"{model_name}.pth").exists():
        # 打印日志信息
        logger.info("Loading original state_dict.")
        # 加载来自模型的状态字典（模型主干和模型头部）
        from_state_dict_trunk, from_state_dict_head = names_to_from_model[model_name]()
        # 使用模型主干的状态字典初始化来源状态字典
        from_state_dict = from_state_dict_trunk
        # 如果模型名称包含 "in1k"，则添加头部信息到来源状态字典
        if "in1k" in model_name:
            # 添加头部信息到来源状态字典
            from_state_dict = {**from_state_dict_trunk, **from_state_dict_head}
        # 打印日志信息
        logger.info("Done!")

        # 初始化转换后的状态字典
        converted_state_dict = {}

        # 初始化未使用的键列表
        not_used_keys = list(from_state_dict.keys())
        # 定义正则表达式匹配模式，用于处理模型参数键中的特殊命名情况
        regex = r"\.block.-part."
        # 遍历来源状态字典中的每个键
        for key in from_state_dict.keys():
            # 使用正则表达式替换处理模型参数键中的特殊命名情况
            src_key = re.sub(regex, "", key)
            # 使用转换映射获取目标键
            dest_key = from_to_ours_keys[src_key]
            # 将源键和对应的值添加到转换后的状态字典中
            converted_state_dict[dest_key] = from_state_dict[key]
            # 从未使用的键列表中移除已使用的键
            not_used_keys.remove(key)
        # 检查是否所有键都已经更新
        assert len(not_used_keys) == 0, f"Some keys where not used {','.join(not_used_keys)}"

        # 打印日志信息
        logger.info(f"The following keys were not used: {','.join(not_used_keys)}")

        # 将转换后的状态字典保存到磁盘上
        torch.save(converted_state_dict, save_directory / f"{model_name}.pth")

        # 删除转换后的状态字典，释放内存
        del converted_state_dict
    else:
        # 打印日志信息
        logger.info("The state_dict was already stored on disk.")
```  
    # 如果需要推送到 Hub
    if push_to_hub:
        # 输出环境变量 HF_TOKEN 的值
        logger.info(f"Token is {os.environ['HF_TOKEN']}")
        # 输出信息：加载我们的模型
        logger.info("Loading our model.")
        # 创建我们的模型
        our_config = names_to_config[model_name]
        our_model_func = RegNetModel
        # 如果模型名包含"in1k"，则选择 RegNetForImageClassification 函数
        if "in1k" in model_name:
            our_model_func = RegNetForImageClassification
        # 根据选择的函数和配置创建模型
        our_model = our_model_func(our_config)
        # 将我们的模型放置在 meta 设备上（删除所有权重）
        our_model.to(torch.device("meta"))
        # 输出信息：加载我们模型中的 state dict
        logger.info("Loading state_dict in our model.")
        # 加载 state dict
        state_dict_keys = our_model.state_dict().keys()
        # 通过 PreTrainedModel 的 _load_pretrained_model_low_mem 方法加载模型
        PreTrainedModel._load_pretrained_model_low_mem(
            our_model, state_dict_keys, [save_directory / f"{model_name}.pth"]
        )
        # 输出信息：最后，进行推送
        logger.info("Finally, pushing!")
        # 将模型推送到 Hub
        our_model.push_to_hub(
            repo_path_or_name=save_directory / model_name,
            commit_message="Add model",
            output_dir=save_directory / model_name,
        )
        # 设置图像处理器的大小为384
        size = 384
        # 使用 convnext 的图像处理器
        image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224-22k-1k", size=size)
        # 将图像处理器推送到 Hub
        image_processor.push_to_hub(
            repo_path_or_name=save_directory / model_name,
            commit_message="Add image processor",
            output_dir=save_directory / model_name,
        )
# 如果当前脚本被直接执行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help=(
            "The name of the model you wish to convert, it must be one of the supported regnet* architecture,"
            " currently: regnetx-*, regnety-*. If `None`, all of them will the converted."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=Path,
        required=True,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        default=True,
        type=bool,
        required=False,
        help="If True, push model and image processor to the hub.",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 将参数解析结果赋值给变量
    pytorch_dump_folder_path: Path = args.pytorch_dump_folder_path
    # 如果目录不存在，则创建该目录
    pytorch_dump_folder_path.mkdir(exist_ok=True, parents=True)
    # 调用函数将权重转换为PyTorch格式，并且可选地推送到hub
    convert_weights_and_push(pytorch_dump_folder_path, args.model_name, args.push_to_hub)
```