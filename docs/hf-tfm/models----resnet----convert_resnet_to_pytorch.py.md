# `.\models\resnet\convert_resnet_to_pytorch.py`

```py
# 设置编码格式为 UTF-8

# 版权声明，这段代码由 HuggingFace Inc. 团队所有，遵循 Apache License, Version 2.0

# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import json  # 处理 JSON 格式数据
from dataclasses import dataclass, field  # 用于创建数据类，支持默认字段
from functools import partial  # 创建偏函数
from pathlib import Path  # 处理文件和目录路径
from typing import List  # 定义类型提示

import timm  # 导入 timm 库，用于模型加载
import torch  # PyTorch 库
import torch.nn as nn  # PyTorch 神经网络模块
from huggingface_hub import hf_hub_download  # 从 Hugging Face Hub 下载模型
from torch import Tensor  # PyTorch 张量类型

# 从 transformers 库中导入必要的模块和函数
from transformers import AutoImageProcessor, ResNetConfig, ResNetForImageClassification
from transformers.utils import logging  # 导入 logging 模块

# 设置日志输出级别为 info
logging.set_verbosity_info()

# 获取日志记录器
logger = logging.get_logger()


@dataclass
class Tracker:
    # 追踪器类，用于跟踪神经网络模块的前向传播
    module: nn.Module  # 要追踪的模块
    traced: List[nn.Module] = field(default_factory=list)  # 用于存储追踪到的模块列表，默认为空列表
    handles: list = field(default_factory=list)  # 存储注册的钩子句柄列表，默认为空列表

    def _forward_hook(self, m, inputs: Tensor, outputs: Tensor):
        # 前向传播的钩子函数，用于注册到模块上
        has_not_submodules = len(list(m.modules())) == 1 or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)
        # 检查模块是否没有子模块或是卷积层或批归一化层
        if has_not_submodules:
            self.traced.append(m)  # 将当前模块添加到追踪列表中

    def __call__(self, x: Tensor):
        # 实现对象可调用功能，用于启动追踪
        for m in self.module.modules():
            self.handles.append(m.register_forward_hook(self._forward_hook))  # 注册前向传播钩子到每个模块上
        self.module(x)  # 执行模块的前向传播
        [x.remove() for x in self.handles]  # 移除所有注册的钩子
        return self

    @property
    def parametrized(self):
        # 检查追踪到的模块中是否有可学习的参数
        return list(filter(lambda x: len(list(x.state_dict().keys())) > 0, self.traced))


@dataclass
class ModuleTransfer:
    # 模块转移类，用于将一个模块的参数传输到另一个模块
    src: nn.Module  # 源模块
    dest: nn.Module  # 目标模块
    verbose: int = 0  # 控制详细程度的参数，默认为 0
    src_skip: List = field(default_factory=list)  # 跳过源模块中的特定层，默认为空列表
    dest_skip: List = field(default_factory=list)  # 跳过目标模块中的特定层，默认为空列表
    # 定义一个调用方法，接受一个张量 x 作为输入，在 self.src 和 self.dest 之间传输权重。
    def __call__(self, x: Tensor):
        """
        Transfer the weights of `self.src` to `self.dest` by performing a forward pass using `x` as input. Under the
        hood we tracked all the operations in both modules.
        """
        # 使用 Tracker 对象追踪 self.dest 模块的前向传播过程，并获取其参数化表示
        dest_traced = Tracker(self.dest)(x).parametrized
        # 使用 Tracker 对象追踪 self.src 模块的前向传播过程，并获取其参数化表示
        src_traced = Tracker(self.src)(x).parametrized

        # 根据 self.src_skip 中的过滤条件，过滤掉 src_traced 中类型为 self.src_skip 的元素
        src_traced = list(filter(lambda x: type(x) not in self.src_skip, src_traced))
        # 根据 self.dest_skip 中的过滤条件，过滤掉 dest_traced 中类型为 self.dest_skip 的元素
        dest_traced = list(filter(lambda x: type(x) not in self.dest_skip, dest_traced))

        # 如果 dest_traced 和 src_traced 的长度不相等，抛出异常
        if len(dest_traced) != len(src_traced):
            raise Exception(
                f"Numbers of operations are different. Source module has {len(src_traced)} operations while"
                f" destination module has {len(dest_traced)}."
            )

        # 遍历 dest_traced 和 src_traced，将 src_m 的状态字典加载到 dest_m 中
        for dest_m, src_m in zip(dest_traced, src_traced):
            dest_m.load_state_dict(src_m.state_dict())
            # 如果 verbose 等于 1，则打印权重转移信息
            if self.verbose == 1:
                print(f"Transfered from={src_m} to={dest_m}")
# 定义函数，将指定模型的权重转换并推送到指定目录或 Hub
def convert_weight_and_push(name: str, config: ResNetConfig, save_directory: Path, push_to_hub: bool = True):
    # 打印正在转换的模型名称
    print(f"Converting {name}...")
    
    # 在没有梯度的情况下执行以下操作
    with torch.no_grad():
        # 创建指定模型，并加载预训练权重，设置为评估模式
        from_model = timm.create_model(name, pretrained=True).eval()
        # 创建自定义的 ResNet 配置模型，也设置为评估模式
        our_model = ResNetForImageClassification(config).eval()
        # 创建模型之间的模块传输器，从原始模型到自定义模型
        module_transfer = ModuleTransfer(src=from_model, dest=our_model)
        # 创建一个随机输入张量
        x = torch.randn((1, 3, 224, 224))
        # 使用模块传输器传输输入张量，确保传输正确
        module_transfer(x)

    # 断言检查：确保两个模型的输出 logits 在数值上非常接近
    assert torch.allclose(from_model(x), our_model(x).logits), "The model logits don't match the original one."

    # 根据模型名称生成检查点名称
    checkpoint_name = f"resnet{'-'.join(name.split('resnet'))}"
    # 打印检查点名称
    print(checkpoint_name)

    # 如果需要推送到 Hub
    if push_to_hub:
        # 将自定义模型推送到指定路径或名称的 Hub 仓库，使用临时目录
        our_model.push_to_hub(
            repo_path_or_name=save_directory / checkpoint_name,
            commit_message="Add model",
            use_temp_dir=True,
        )

        # 创建一个自动图像处理器，从预训练模型加载，推送到 Hub
        image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224-22k-1k")
        image_processor.push_to_hub(
            repo_path_or_name=save_directory / checkpoint_name,
            commit_message="Add image processor",
            use_temp_dir=True,
        )

        # 打印推送成功的消息
        print(f"Pushed {checkpoint_name}")


# 定义函数，将指定模型的权重转换并推送到指定目录或 Hub
def convert_weights_and_push(save_directory: Path, model_name: str = None, push_to_hub: bool = True):
    # 定义用于 ImageNet 的标签文件名
    filename = "imagenet-1k-id2label.json"
    # ImageNet 数据集中的标签数量
    num_labels = 1000
    # 预期的输出形状
    expected_shape = (1, num_labels)

    # Hub 仓库的 ID
    repo_id = "huggingface/label-files"
    # 加载 ImageNet 标签映射文件，以字典形式存储
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    # 将字符串键转换为整数键
    id2label = {int(k): v for k, v in id2label.items()}

    # 将 id2label 赋值给自己（实际上是多余的操作）
    id2label = id2label
    # 创建从标签到 ID 的反向映射字典
    label2id = {v: k for k, v in id2label.items()}

    # 使用部分函数构造 ImageNet 预训练配置
    ImageNetPreTrainedConfig = partial(ResNetConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)

    # 各个模型名称与其配置的映射关系
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

    # 如果指定了模型名称，则执行权重转换并推送到 Hub
    if model_name:
        convert_weight_and_push(model_name, names_to_config[model_name], save_directory, push_to_hub)
    else:
        # 对于字典 names_to_config 中的每个键值对，分别赋值给 model_name 和 config
        for model_name, config in names_to_config.items():
            # 调用函数 convert_weight_and_push，将 model_name, config, save_directory, push_to_hub 作为参数传入
            convert_weight_and_push(model_name, config, save_directory, push_to_hub)
    # 返回变量 config 和 expected_shape
    return config, expected_shape
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 添加必需的参数
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help=(
            "要转换的模型名称，必须是支持的 resnet* 架构之一，"
            "目前支持的有：resnet18,26,34,50,101,152。如果为 `None`，则转换所有支持的模型。"
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=Path,
        required=True,
        help="输出 PyTorch 模型目录的路径。",
    )
    parser.add_argument(
        "--push_to_hub",
        default=True,
        type=bool,
        required=False,
        help="如果为 True，将模型和图像处理器推送到 hub。",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 获取 PyTorch 模型输出目录路径，并创建该目录（如果不存在）
    pytorch_dump_folder_path: Path = args.pytorch_dump_folder_path
    pytorch_dump_folder_path.mkdir(exist_ok=True, parents=True)

    # 调用函数将权重转换并推送到指定目录和 hub
    convert_weights_and_push(pytorch_dump_folder_path, args.model_name, args.push_to_hub)
```