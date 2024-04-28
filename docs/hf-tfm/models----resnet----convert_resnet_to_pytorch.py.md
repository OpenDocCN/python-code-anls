# `.\transformers\models\resnet\convert_resnet_to_pytorch.py`

```
# 设置编码格式为 UTF-8
# 版权声明，该代码版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证版本 2.0 使用本文件
# 你只能在符合许可证的情况下使用此文件
# 你可以从以下网址获得许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"提供，
# 不提供任何明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证
"""从 timm 转换 ResNet 检查点。"""

# 导入必要的库
import argparse  # 导入用于解析命令行参数的模块
import json  # 导入用于 JSON 数据处理的模块
from dataclasses import dataclass, field  # 导入用于创建数据类的模块
from functools import partial  # 导入用于创建偏函数的模块
from pathlib import Path  # 导入用于处理文件路径的模块
from typing import List  # 导入用于类型提示的 List 类型

import timm  # 导入用于加载预训练模型的 timm 库
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入神经网络模块
from huggingface_hub import hf_hub_download  # 导入从 Hugging Face Hub 下载模型的函数
from torch import Tensor  # 导入张量类型

from transformers import AutoImageProcessor, ResNetConfig, ResNetForImageClassification  # 导入转换成 Transformers 模型所需的类
from transformers.utils import logging  # 导入用于记录日志的模块

# 设置日志记录级别为 INFO
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger()


@dataclass
class Tracker:
    # 追踪器类，用于追踪模块
    module: nn.Module  # 模块对象
    traced: List[nn.Module] = field(default_factory=list)  # 已追踪的模块列表，默认为空列表
    handles: list = field(default_factory=list)  # 处理器列表，默认为空列表

    def _forward_hook(self, m, inputs: Tensor, outputs: Tensor):
        # 前向传播钩子函数，用于追踪模块的前向传播过程
        has_not_submodules = len(list(m.modules())) == 1 or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)
        # 检查模块是否有子模块
        if has_not_submodules:
            # 如果模块没有子模块，则将其添加到已追踪列表中
            self.traced.append(m)

    def __call__(self, x: Tensor):
        # 调用追踪器实例时执行的操作
        for m in self.module.modules():
            # 遍历模块中的所有子模块
            self.handles.append(m.register_forward_hook(self._forward_hook))
            # 为每个子模块注册前向传播钩子
        self.module(x)
        # 执行模块的前向传播
        [x.remove() for x in self.handles]
        # 移除所有前向传播钩子
        return self

    @property
    def parametrized(self):
        # 获取已追踪模块中含有学习参数的模块列表
        # 通过检查状态字典的键的长度是否大于 0 来确定是否含有学习参数
        return list(filter(lambda x: len(list(x.state_dict().keys())) > 0, self.traced))


@dataclass
class ModuleTransfer:
    # 模块传输器类，用于在模型之间传输权重
    src: nn.Module  # 源模型
    dest: nn.Module  # 目标模型
    verbose: int = 0  # 详细程度，默认为 0
    src_skip: List = field(default_factory=list)  # 跳过源模型的模块列表，默认为空列表
    dest_skip: List = field(default_factory=list)  # 跳过目标模型的模块列表，默认为空列表
    # 定义一个方法，接受参数 x，类型为 Tensor
    def __call__(self, x: Tensor):
        """
        # 将 self.src 的权重转移到 self.dest，通过使用 x 作为输入执行前向传播。在内部，我们跟踪了两个模块中的所有操作。
        """
        # 对 self.dest 进行跟踪，并获取参数化后的结果
        dest_traced = Tracker(self.dest)(x).parametrized
        # 对 self.src 进行跟踪，并获取参数化后的结果
        src_traced = Tracker(self.src)(x).parametrized

        # 从 src_traced 中过滤掉 self.src_skip 中指定类型的操作
        src_traced = list(filter(lambda x: type(x) not in self.src_skip, src_traced))
        # 从 dest_traced 中过滤掉 self.dest_skip 中指定类型的操作
        dest_traced = list(filter(lambda x: type(x) not in self.dest_skip, dest_traced))

        # 如果 dest_traced 和 src_traced 的长度不相等，抛出异常
        if len(dest_traced) != len(src_traced):
            raise Exception(
                f"Numbers of operations are different. Source module has {len(src_traced)} operations while"
                f" destination module has {len(dest_traced)}."
            )

        # 对 dest_traced 和 src_traced 中的元素进行逐个处理
        for dest_m, src_m in zip(dest_traced, src_traced):
            # 将 src_m 的状态字典加载到 dest_m 中
            dest_m.load_state_dict(src_m.state_dict())
            # 如果 verbose 为 1，打印日志信息
            if self.verbose == 1:
                print(f"Transfered from={src_m} to={dest_m}")
def convert_weight_and_push(name: str, config: ResNetConfig, save_directory: Path, push_to_hub: bool = True):
    # 打印正在转换的模型名称
    print(f"Converting {name}...")
    # 禁用梯度计算
    with torch.no_grad():
        # 从 timm 库中加载预训练的模型，设为评估模式
        from_model = timm.create_model(name, pretrained=True).eval()
        # 创建我们自己的模型 ResNetForImageClassification，并设为评估模式
        our_model = ResNetForImageClassification(config).eval()
        # 创建模型转换对象，将预训练的模型参数复制到我们的模型中
        module_transfer = ModuleTransfer(src=from_model, dest=our_model)
        # 创建一个随机输入张量
        x = torch.randn((1, 3, 224, 224))
        # 将随机输入张量通过模型转换对象传递给我们的模型
        module_transfer(x)

    # 断言：检查两个模型的输出是否接近，如果不接近则抛出异常
    assert torch.allclose(from_model(x), our_model(x).logits), "The model logits don't match the original one."

    # 根据模型名称生成检查点名称
    checkpoint_name = f"resnet{'-'.join(name.split('resnet'))}"
    # 打印检查点名称
    print(checkpoint_name)

    if push_to_hub:
        # 将我们的模型保存到 Hugging Face Hub
        our_model.push_to_hub(
            repo_path_or_name=save_directory / checkpoint_name,
            commit_message="Add model",
            use_temp_dir=True,
        )

        # 创建一个图像处理器，并将其保存到 Hugging Face Hub
        image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224-22k-1k")
        image_processor.push_to_hub(
            repo_path_or_name=save_directory / checkpoint_name,
            commit_message="Add image processor",
            use_temp_dir=True,
        )

        # 打印已推送的检查点名称
        print(f"Pushed {checkpoint_name}")


def convert_weights_and_push(save_directory: Path, model_name: str = None, push_to_hub: bool = True):
    # 定义标签文件的名称和数量
    filename = "imagenet-1k-id2label.json"
    num_labels = 1000
    expected_shape = (1, num_labels)

    # 定义 Hugging Face Hub 仓库信息
    repo_id = "huggingface/label-files"
    num_labels = num_labels
    # 加载标签文件内容并转为字典形式
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    # 将标签字典赋值给变量
    id2label = id2label
    label2id = {v: k for k, v in id2label.items()}

    # 配置 ImageNet 预训练模型的参数
    ImageNetPreTrainedConfig = partial(ResNetConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)

    # 模型名称与模型参数的对应关��字典
    names_to_config = {
        "resnet18": ImageNetPreTrainedConfig(
            depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 256, 512], layer_type="basic"
        ),
        "resnet26": ImageNetPreTrainedConfig(
            depths=[2, 2, 2, 2], hidden_sizes=[256, 512, 1024, 2048], layer_type="bottleneck"
        ),
        "resnet34": ImageNetPreTrainedConfig(
            depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 256, 512], layer_type="basic"
        ),
        "resnet50": ImageNetPreTrainedConfig(
            depths=[3, 4, 6, 3], hidden_sizes=[256, 512, 1024, 2048], layer_type="bottleneck"
        ),
        "resnet101": ImageNetPreTrainedConfig(
            depths=[3, 4, 23, 3], hidden_sizes=[256, 512, 1024, 2048], layer_type="bottleneck"
        ),
        "resnet152": ImageNetPreTrainedConfig(
            depths=[3, 8, 36, 3], hidden_sizes=[256, 512, 1024, 2048], layer_type="bottleneck"
        ),
    }

    if model_name:
        # 调用 convert_weight_and_push 函数，将指定模型的权重转换并推送到 Hugging Face Hub
        convert_weight_and_push(model_name, names_to_config[model_name], save_directory, push_to_hub)
    # 如果不是模型文件，则执行以下操作
    else:
        # 遍历模型名称到配置的字典中的每个键值对
        for model_name, config in names_to_config.items():
            # 调用函数将权重进行转换并推送到指定位置
            convert_weight_and_push(model_name, config, save_directory, push_to_hub)
    # 返回配置和预期形状
    return config, expected_shape
# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加必需参数
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help=(
            "The name of the model you wish to convert, it must be one of the supported resnet* architecture,"
            " currently: resnet18,26,34,50,101,152. If `None`, all of them will the converted."
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
    # 将参数中的pytorch_dump_folder_path路径赋值给pytorch_dump_folder_path变量
    pytorch_dump_folder_path: Path = args.pytorch_dump_folder_path
    # 如果路径不存在，则创建文件夹
    pytorch_dump_folder_path.mkdir(exist_ok=True, parents=True)
    # 调用convert_weights_and_push函数，传入参数pytorch_dump_folder_path, args.model_name, args.push_to_hub
    convert_weights_and_push(pytorch_dump_folder_path, args.model_name, args.push_to_hub)
```