# `.\models\regnet\convert_regnet_to_pytorch.py`

```
# coding=utf-8
# 声明编码格式为 UTF-8

# Copyright 2022 The HuggingFace Inc. team.
# 版权声明为 2022 年 HuggingFace Inc. 团队所有

# Licensed under the Apache License, Version 2.0 (the "License");
# 使用 Apache 许可证 2.0 版本进行许可

# you may not use this file except in compliance with the License.
# 您除非遵守许可证，否则不得使用此文件

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证副本

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非法律要求或书面同意，否则根据许可证分发的软件是基于"原样"分发的，
# 没有任何形式的明示或暗示担保或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查看许可证以了解具体的语言授权和限制

"""Convert RegNet checkpoints from timm and vissl."""

import argparse
import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import timm
import torch
import torch.nn as nn
from classy_vision.models.regnet import RegNet, RegNetParams, RegNetY32gf, RegNetY64gf, RegNetY128gf
from huggingface_hub import cached_download, hf_hub_url
from torch import Tensor
from vissl.models.model_helpers import get_trunk_forward_outputs

from transformers import AutoImageProcessor, RegNetConfig, RegNetForImageClassification, RegNetModel
from transformers.utils import logging

# 设置日志输出为 info 级别
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger()


@dataclass
class Tracker:
    # 跟踪器类，用于追踪模块的前向传播行为
    module: nn.Module
    traced: List[nn.Module] = field(default_factory=list)
    handles: list = field(default_factory=list)

    def _forward_hook(self, m, inputs: Tensor, outputs: Tensor):
        # 前向钩子函数，记录非子模块的模块到追踪列表
        has_not_submodules = len(list(m.modules())) == 1 or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)
        if has_not_submodules:
            self.traced.append(m)

    def __call__(self, x: Tensor):
        # 对模块进行前向传播，记录前向钩子处理
        for m in self.module.modules():
            self.handles.append(m.register_forward_hook(self._forward_hook))
        self.module(x)
        [x.remove() for x in self.handles]  # 移除注册的钩子
        return self

    @property
    def parametrized(self):
        # 检查追踪的模块列表中是否有可学习的参数
        return list(filter(lambda x: len(list(x.state_dict().keys())) > 0, self.traced))


@dataclass
class ModuleTransfer:
    # 模块传输类，用于在不同模型之间传输权重参数
    src: nn.Module
    dest: nn.Module
    verbose: int = 1
    src_skip: List = field(default_factory=list)
    dest_skip: List = field(default_factory=list)
    raise_if_mismatch: bool = True
    # 定义一个方法，使得对象可以像函数一样被调用，传入参数 x，其类型为 Tensor
    def __call__(self, x: Tensor):
        """
        Transfer the weights of `self.src` to `self.dest` by performing a forward pass using `x` as input. Under the
        hood we tracked all the operations in both modules.
        """
        # 使用 Tracker 对象跟踪 self.dest 模块的前向传播结果，并获取参数化的结果
        dest_traced = Tracker(self.dest)(x).parametrized
        # 使用 Tracker 对象跟踪 self.src 模块的前向传播结果，并获取参数化的结果
        src_traced = Tracker(self.src)(x).parametrized

        # 过滤掉 self.src_skip 中指定类型的操作，得到过滤后的 src_traced 列表
        src_traced = list(filter(lambda x: type(x) not in self.src_skip, src_traced))
        # 过滤掉 self.dest_skip 中指定类型的操作，得到过滤后的 dest_traced 列表
        dest_traced = list(filter(lambda x: type(x) not in self.dest_skip, dest_traced))

        # 如果 dest_traced 和 src_traced 的长度不同，并且设置了 raise_if_mismatch 标志，则抛出异常
        if len(dest_traced) != len(src_traced) and self.raise_if_mismatch:
            raise Exception(
                f"Numbers of operations are different. Source module has {len(src_traced)} operations while"
                f" destination module has {len(dest_traced)}."
            )

        # 逐一将 src_m 的状态字典加载到 dest_m 中
        for dest_m, src_m in zip(dest_traced, src_traced):
            dest_m.load_state_dict(src_m.state_dict())
            # 如果设置了 verbose 标志为 1，则打印详细信息表示权重转移情况
            if self.verbose == 1:
                print(f"Transfered from={src_m} to={dest_m}")
class FakeRegNetVisslWrapper(nn.Module):
    """
    Fake wrapper for RegNet that mimics what vissl does without the need to pass a config file.
    """

    def __init__(self, model: nn.Module):
        super().__init__()

        feature_blocks: List[Tuple[str, nn.Module]] = []
        # - get the stem
        feature_blocks.append(("conv1", model.stem))  # 将模型的 stem 添加到特征块列表中，命名为 'conv1'
        # - get all the feature blocks
        for k, v in model.trunk_output.named_children():
            assert k.startswith("block"), f"Unexpected layer name {k}"
            block_index = len(feature_blocks) + 1
            feature_blocks.append((f"res{block_index}", v))  # 将模型 trunk_output 中的每个块添加到特征块列表中

        self._feature_blocks = nn.ModuleDict(feature_blocks)  # 将特征块列表转换为 nn.ModuleDict，存储在对象属性 _feature_blocks 中

    def forward(self, x: Tensor):
        return get_trunk_forward_outputs(
            x,
            out_feat_keys=None,
            feature_blocks=self._feature_blocks,
        )


class NameToFromModelFuncMap(dict):
    """
    A Dictionary with some additional logic to return a function that creates the correct original model.
    """

    def convert_name_to_timm(self, x: str) -> str:
        x_split = x.split("-")
        return x_split[0] + x_split[1] + "_" + "".join(x_split[2:])

    def __getitem__(self, x: str) -> Callable[[], Tuple[nn.Module, Dict]]:
        # default to timm!
        if x not in self:
            x = self.convert_name_to_timm(x)  # 将 x 转换为 timm 模型的名称格式
            val = partial(lambda: (timm.create_model(x, pretrained=True).eval(), None))  # 创建一个 lambda 函数，返回预训练的 timm 模型和空字典
        else:
            val = super().__getitem__(x)  # 调用父类 dict 的 __getitem__ 方法获取对应项

        return val


class NameToOurModelFuncMap(dict):
    """
    A Dictionary with some additional logic to return the correct hugging face RegNet class reference.
    """

    def __getitem__(self, x: str) -> Callable[[], nn.Module]:
        if "seer" in x and "in1k" not in x:
            val = RegNetModel  # 如果 x 包含 "seer" 且不包含 "in1k"，返回 RegNetModel 类引用
        else:
            val = RegNetForImageClassification  # 否则返回 RegNetForImageClassification 类引用
        return val


def manually_copy_vissl_head(from_state_dict, to_state_dict, keys: List[Tuple[str, str]]):
    for from_key, to_key in keys:
        to_state_dict[to_key] = from_state_dict[from_key].clone()  # 复制 from_state_dict 中的权重到 to_state_dict 中，并使用 clone() 方法克隆张量
        print(f"Copied key={from_key} to={to_key}")  # 打印复制的键名和目标键名
    return to_state_dict


def convert_weight_and_push(
    name: str,
    from_model_func: Callable[[], nn.Module],
    our_model_func: Callable[[], nn.Module],
    config: RegNetConfig,
    save_directory: Path,
    push_to_hub: bool = True,
):
    print(f"Converting {name}...")  # 打印转换的模型名称
    with torch.no_grad():
        from_model, from_state_dict = from_model_func()  # 调用 from_model_func 获取源模型和其状态字典
        our_model = our_model_func(config).eval()  # 调用 our_model_func 创建我们的模型，并转换为评估模式
        module_transfer = ModuleTransfer(src=from_model, dest=our_model, raise_if_mismatch=False)  # 使用 ModuleTransfer 将源模型的权重转移到我们的模型中
        x = torch.randn((1, 3, 224, 224))  # 创建一个随机张量作为输入
        module_transfer(x)  # 执行模型权重转换
    # 如果有给定的 from_state_dict，则需要手动复制特定的头部参数
    if from_state_dict is not None:
        keys = []
        # 对于 seer - in1k finetuned 模型，需要手动复制头部参数
        if "seer" in name and "in1k" in name:
            keys = [("0.clf.0.weight", "classifier.1.weight"), ("0.clf.0.bias", "classifier.1.bias")]
        # 手动复制头部参数，并获取复制后的状态字典
        to_state_dict = manually_copy_vissl_head(from_state_dict, our_model.state_dict(), keys)
        # 使用复制后的状态字典加载我们的模型
        our_model.load_state_dict(to_state_dict)

    # 获取我们模型的输出，同时要求输出隐藏状态
    our_outputs = our_model(x, output_hidden_states=True)
    # 根据模型类型选择输出 logits 或者最后的隐藏状态
    our_output = (
        our_outputs.logits if isinstance(our_model, RegNetForImageClassification) else our_outputs.last_hidden_state
    )

    # 获取原始模型的输出
    from_output = from_model(x)
    # 如果原始模型的输出是一个列表，则选择最后一个元素作为输出
    from_output = from_output[-1] if isinstance(from_output, list) else from_output

    # 对于 vissl seer 模型，因为不使用任何配置文件，实际上没有头部，因此直接使用最后的隐藏状态
    if "seer" in name and "in1k" in name:
        our_output = our_outputs.hidden_states[-1]

    # 断言两个模型的输出是否近似相等，否则抛出异常
    assert torch.allclose(from_output, our_output), "The model logits don't match the original one."

    # 如果需要推送到 Hub
    if push_to_hub:
        # 将我们的模型推送到 Hub
        our_model.push_to_hub(
            repo_path_or_name=save_directory / name,
            commit_message="Add model",
            use_temp_dir=True,
        )

        # 根据模型名称选择图像处理器的大小
        size = 224 if "seer" not in name else 384
        # 使用预训练的 convnext-base-224-22k-1k 模型创建图像处理器
        image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224-22k-1k", size=size)
        # 将图像处理器推送到 Hub
        image_processor.push_to_hub(
            repo_path_or_name=save_directory / name,
            commit_message="Add image processor",
            use_temp_dir=True,
        )

        # 打印推送成功的消息
        print(f"Pushed {name}")
def convert_weights_and_push(save_directory: Path, model_name: str = None, push_to_hub: bool = True):
    # 定义文件名和标签数量
    filename = "imagenet-1k-id2label.json"
    num_labels = 1000
    expected_shape = (1, num_labels)

    # Hub repo ID
    repo_id = "huggingface/label-files"
    num_labels = num_labels  # 更新 num_labels 变量

    # 从 Hub 下载并加载 id 到 label 的映射关系
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}  # 转换键为整数类型

    id2label = id2label  # 重复赋值，可能是误操作
    label2id = {v: k for k, v in id2label.items()}  # 创建 label 到 id 的映射关系

    # 创建一个配置对象，使用部分函数创建 RegNet 配置
    ImageNetPreTrainedConfig = partial(RegNetConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)

    # 初始化两个映射对象
    names_to_ours_model_map = NameToOurModelFuncMap()
    names_to_from_model_map = NameToFromModelFuncMap()

    # 添加 SEER weights 逻辑

    # 定义一个函数，通过 Classy Vision 加载模型和状态字典
    def load_using_classy_vision(checkpoint_url: str, model_func: Callable[[], nn.Module]) -> Tuple[nn.Module, Dict]:
        # 从 URL 加载模型状态字典到指定目录
        files = torch.hub.load_state_dict_from_url(checkpoint_url, model_dir=str(save_directory), map_location="cpu")
        model = model_func()
        # 检查是否有头部，如果有则添加到模型
        model_state_dict = files["classy_state_dict"]["base_model"]["model"]
        state_dict = model_state_dict["trunk"]
        model.load_state_dict(state_dict)
        return model.eval(), model_state_dict["heads"]

    # 预训练模型映射

    # 添加 regnet-y-320-seer 的映射
    names_to_from_model_map["regnet-y-320-seer"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet32d/seer_regnet32gf_model_iteration244000.torch",
        lambda: FakeRegNetVisslWrapper(RegNetY32gf()),
    )

    # 添加 regnet-y-640-seer 的映射
    names_to_from_model_map["regnet-y-640-seer"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet64/seer_regnet64gf_model_final_checkpoint_phase0.torch",
        lambda: FakeRegNetVisslWrapper(RegNetY64gf()),
    )

    # 添加 regnet-y-1280-seer 的映射
    names_to_from_model_map["regnet-y-1280-seer"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_ig1b_regnet128Gf_cnstant_bs32_node16_sinkhorn10_proto16k_syncBN64_warmup8k/model_final_checkpoint_phase0.torch",
        lambda: FakeRegNetVisslWrapper(RegNetY128gf()),
    )

    # 添加 regnet-y-10b-seer 的映射
    names_to_from_model_map["regnet-y-10b-seer"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet10B/model_iteration124500_conso.torch",
        lambda: FakeRegNetVisslWrapper(
            RegNet(RegNetParams(depth=27, group_width=1010, w_0=1744, w_a=620.83, w_m=2.52))
        ),
    )

    # IN1K finetuned 映射

    # 添加 regnet-y-320-seer-in1k 的映射
    names_to_from_model_map["regnet-y-320-seer-in1k"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet32_finetuned_in1k_model_final_checkpoint_phase78.torch",
        lambda: FakeRegNetVisslWrapper(RegNetY32gf()),
    )
    # 将模型名称映射到加载模型函数的部分函数调用，使用 Classy Vision 加载模型
    names_to_from_model_map["regnet-y-640-seer-in1k"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet64_finetuned_in1k_model_final_checkpoint_phase78.torch",
        lambda: FakeRegNetVisslWrapper(RegNetY64gf()),
    )

    # 将模型名称映射到加载模型函数的部分函数调用，使用 Classy Vision 加载模型
    names_to_from_model_map["regnet-y-1280-seer-in1k"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet128_finetuned_in1k_model_final_checkpoint_phase78.torch",
        lambda: FakeRegNetVisslWrapper(RegNetY128gf()),
    )

    # 将模型名称映射到加载模型函数的部分函数调用，使用 Classy Vision 加载模型
    names_to_from_model_map["regnet-y-10b-seer-in1k"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_10b_finetuned_in1k_model_phase28_conso.torch",
        lambda: FakeRegNetVisslWrapper(
            RegNet(RegNetParams(depth=27, group_width=1010, w_0=1744, w_a=620.83, w_m=2.52))
        ),
    )

    # 如果指定了模型名称，则转换权重并推送到指定的模型名称下
    if model_name:
        convert_weight_and_push(
            model_name,
            names_to_from_model_map[model_name],  # 使用模型名称从映射中获取加载模型函数
            names_to_ours_model_map[model_name],  # 使用模型名称从映射中获取我们的模型名称
            names_to_config[model_name],  # 使用模型名称从映射中获取配置
            save_directory,  # 保存目录路径
            push_to_hub,  # 是否推送到 Hub
        )
    else:
        # 否则，对于每个模型名称和其对应的配置，转换权重并推送到对应的模型名称下
        for model_name, config in names_to_config.items():
            convert_weight_and_push(
                model_name,
                names_to_from_model_map[model_name],  # 使用模型名称从映射中获取加载模型函数
                names_to_ours_model_map[model_name],  # 使用模型名称从映射中获取我们的模型名称
                config,  # 使用当前配置
                save_directory,  # 保存目录路径
                push_to_hub,  # 是否推送到 Hub
            )
    
    # 返回配置和期望的形状
    return config, expected_shape
if __name__ == "__main__":
    # 如果当前脚本作为主程序执行，则执行以下操作

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
    # 添加一个必需的参数选项，用于指定要转换的模型名称

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=Path,
        required=True,
        help="Path to the output PyTorch model directory.",
    )
    # 添加一个必需的参数选项，用于指定输出的 PyTorch 模型目录路径

    parser.add_argument(
        "--push_to_hub",
        default=True,
        type=bool,
        required=False,
        help="If True, push model and image processor to the hub.",
    )
    # 添加一个可选的参数选项，指定是否将模型和图像处理器推送到 hub

    args = parser.parse_args()
    # 解析命令行参数并将其存储在 args 变量中

    pytorch_dump_folder_path: Path = args.pytorch_dump_folder_path
    # 从参数中获取 PyTorch 模型目录路径

    pytorch_dump_folder_path.mkdir(exist_ok=True, parents=True)
    # 创建 PyTorch 模型目录，如果目录已存在则忽略，同时创建必要的父目录

    convert_weights_and_push(pytorch_dump_folder_path, args.model_name, args.push_to_hub)
    # 调用函数，将权重转换并推送到指定的 PyTorch 模型目录
```