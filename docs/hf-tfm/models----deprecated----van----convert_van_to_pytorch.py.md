# `.\models\deprecated\van\convert_van_to_pytorch.py`

```py
# 设置编码格式为 UTF-8
# 版权声明，指定了代码的版权信息和许可协议
# 此代码的版权归 BNRist（清华大学）、TKLNDST（南开大学）以及 HuggingFace Inc. 团队所有，保留所有权利
# 根据 Apache 许可协议 Version 2.0 使用本文件
# 你可以在符合许可协议的情况下使用本文件，你可以在下方链接获取许可协议的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则本软件是按"原样"提供的，不提供任何形式的保证或条件，无论是明示的还是暗示的
# 有关许可协议的详细信息，请参阅许可协议

"""从原始存储库中转换 VAN（Visual Attention Network）检查点。

URL: https://github.com/Visual-Attention-Network/VAN-Classification"""

# 导入所需的模块
import argparse  # 导入用于解析命令行参数的模块
import json  # 导入用于 JSON 数据解析的模块
import sys  # 导入用于与 Python 解释器进行交互的模块
from dataclasses import dataclass, field  # 导入 dataclass 用于创建数据类，field 用于定义数据类的属性
from functools import partial  # 导入 partial 用于创建函数的可调用对象
from pathlib import Path  # 导入 Path 用于操作文件路径
from typing import List  # 导入 List 用于定义列表类型

import torch  # 导入 PyTorch 深度学习框架
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块
from huggingface_hub import cached_download, hf_hub_download  # 导入从 HuggingFace Hub 下载模型的函数
from torch import Tensor  # 导入 Tensor 类型

# 从 transformers 库中导入所需的类和函数
from transformers import AutoImageProcessor, VanConfig, VanForImageClassification  
# 从 transformers 库中导入 VAN 模型配置和图像分类器
from transformers.models.deprecated.van.modeling_van import VanLayerScaling  
# 从 transformers 库中导入 VAN 模型的图层缩放类
from transformers.utils import logging  # 导入用于记录日志的模块

# 设置日志记录级别为 info
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义 Tracker 类，用于追踪模块
@dataclass
class Tracker:
    module: nn.Module  # 模块对象
    traced: List[nn.Module] = field(default_factory=list)  # 追踪到的模块列表，默认为空列表
    handles: list = field(default_factory=list)  # 模块句柄列表，默认为空列表

    # 前向传播的钩子函数
    def _forward_hook(self, m, inputs: Tensor, outputs: Tensor):
        # 判断模块是否有子模块或者是否为 Conv2d 或 BatchNorm2d 模块
        has_not_submodules = len(list(m.modules())) == 1 or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)
        if has_not_submodules:
            # 如果模块不是 VanLayerScaling 类型，则将其添加到追踪列表中
            if not isinstance(m, VanLayerScaling):
                self.traced.append(m)

    # 对模块进行追踪
    def __call__(self, x: Tensor):
        # 遍历模块中的每个子模块，并注册前向传播的钩子函数
        for m in self.module.modules():
            self.handles.append(m.register_forward_hook(self._forward_hook))
        # 进行前向传播
        self.module(x)
        # 移除注册的钩子函数
        [x.remove() for x in self.handles]
        return self

    # 返回包含可学习参数的追踪模块列表
    @property
    def parametrized(self):
        # 通过检查状态字典的键的长度来判断模块是否包含可学习参数
        return list(filter(lambda x: len(list(x.state_dict().keys())) > 0, self.traced))


# 定义 ModuleTransfer 类，用于模块迁移
@dataclass
class ModuleTransfer:
    src: nn.Module  # 源模块
    dest: nn.Module  # 目标模块
    verbose: int = 0  # 控制输出详细程度的变量，默认为 0
    src_skip: List = field(default_factory=list)  # 跳过的源模块列表，默认为空列表
    dest_skip: List = field(default_factory=list)  # 跳过的目标模块列表，默认为空列表
    # 定义一个函数，用于将self.src的权重传输到self.dest，通过对x进行前向传播来实现。在内部，我们跟踪了两个模块中的所有操作。

    # 创建Tracker对象，以对self.dest执行前向传播，并获取参数化后的结果
    dest_traced = Tracker(self.dest)(x).parametrized
    # 创建Tracker对象，以对self.src执行前向传播，并获取参数化后的结果
    src_traced = Tracker(self.src)(x).parametrized

    # 过滤掉self.src_skip中指定类型的操作
    src_traced = list(filter(lambda x: type(x) not in self.src_skip, src_traced))
    # 过滤掉self.dest_skip中指定类型的操作
    dest_traced = list(filter(lambda x: type(x) not in self.dest_skip, dest_traced))

    # 检查两个模块的操作数量是否一致，如果不一致则抛出异常
    if len(dest_traced) != len(src_traced):
        raise Exception(
            f"Numbers of operations are different. Source module has {len(src_traced)} operations while"
            f" destination module has {len(dest_traced)}."
        )

    # 将self.src中每个操作的状态字典加载到self.dest中对应的操作中
    for dest_m, src_m in zip(dest_traced, src_traced):
        dest_m.load_state_dict(src_m.state_dict())
        # 如果verbose为1，则打印每次传输的操作
        if self.verbose == 1:
            print(f"Transfered from={src_m} to={dest_m}")
def copy_parameters(from_model: nn.Module, our_model: nn.Module) -> nn.Module:
    # nn.Parameter不能被Tracker跟踪，因此我们需要手动转换它们
    from_state_dict = from_model.state_dict()
    our_state_dict = our_model.state_dict()
    config = our_model.config
    all_keys = []
    for stage_idx in range(len(config.hidden_sizes)):
        for block_id in range(config.depths[stage_idx]):
            from_key = f"block{stage_idx + 1}.{block_id}.layer_scale_1"
            to_key = f"van.encoder.stages.{stage_idx}.layers.{block_id}.attention_scaling.weight"

            all_keys.append((from_key, to_key))
            from_key = f"block{stage_idx + 1}.{block_id}.layer_scale_2"
            to_key = f"van.encoder.stages.{stage_idx}.layers.{block_id}.mlp_scaling.weight"

            all_keys.append((from_key, to_key)

    for from_key, to_key in all_keys:
        our_state_dict[to_key] = from_state_dict.pop(from_key)

    our_model.load_state_dict(our_state_dict)
    return our_model


def convert_weight_and_push(
    name: str,
    config: VanConfig,
    checkpoint: str,
    from_model: nn.Module,
    save_directory: Path,
    push_to_hub: bool = True,
):
    print(f"Downloading weights for {name}...")
    checkpoint_path = cached_download(checkpoint)
    print(f"Converting {name}...")
    from_state_dict = torch.load(checkpoint_path)["state_dict"]
    from_model.load_state_dict(from_state_dict)
    from_model.eval()
    with torch.no_grad():
        our_model = VanForImageClassification(config).eval()
        module_transfer = ModuleTransfer(src=from_model, dest=our_model)
        x = torch.randn((1, 3, 224, 224))
        module_transfer(x)
        our_model = copy_parameters(from_model, our_model)

    if not torch.allclose(from_model(x), our_model(x).logits):
        raise ValueError("The model logits don't match the original one.")
    
    checkpoint_name = name
    print(checkpoint_name)

    if push_to_hub:
        our_model.push_to_hub(
            repo_path_or_name=save_directory / checkpoint_name,
            commit_message="Add model",
            use_temp_dir=True,
        )

        # we can use the convnext one
        image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224-22k-1k")
        image_processor.push_to_hub(
            repo_path_or_name=save_directory / checkpoint_name,
            commit_message="Add image processor",
            use_temp_dir=True,
        )

        print(f"Pushed {checkpoint_name}")


def convert_weights_and_push(save_directory: Path, model_name: str = None, push_to_hub: bool = True):
    filename = "imagenet-1k-id2label.json"
    num_labels = 1000

    repo_id = "huggingface/label-files"
    num_labels = num_labels
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    id2label = id2label
    label2id = {v: k for k, v in id2label.items()}
``` 
    # 使用偏函数 partial 创建 ImageNetPreTrainedConfig，设置模型的参数和标签映射
    ImageNetPreTrainedConfig = partial(VanConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)
    
    # 定义模型名称到配置对象的映射字典
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
    
    # 定义模型名称到原始模型对象的映射字典
    names_to_original_models = {
        "van-tiny": van_tiny,
        "van-small": van_small,
        "van-base": van_base,
        "van-large": van_large,
    }
    
    # 定义模型名称到原始模型检查点 URL 的映射字典
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
    
    # 如果给定了模型名称，则转换权重并推送到 Hub
    if model_name:
        convert_weight_and_push(
            model_name,
            names_to_config[model_name],
            checkpoint=names_to_original_checkpoints[model_name],
            from_model=names_to_original_models[model_name](),
            save_directory=save_directory,
            push_to_hub=push_to_hub,
        )
    # 否则，对所有模型进行循环操作，转换权重并推送到 Hub
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
# 如果脚本作为主程序运行
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--model-name",
        default=None,
        type=str,
        help=(
            "The name of the model you wish to convert, it must be one of the supported resnet* architecture,"
            " currently: van-tiny/small/base/large. If `None`, all of them will the converted."
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
        "--van_dir",
        required=True,
        type=Path,
        help=(
            "A path to VAN's original implementation directory. You can download from here:"
            " https://github.com/Visual-Attention-Network/VAN-Classification"
        ),
    )
    parser.add_argument(
        "--push_to_hub",
        default=True,
        type=bool,
        required=False,
        help="If True, push model and image processor to the hub.",
    )

    # 解析命令行参数并返回一个命名空间，其中包含设置的参数
    args = parser.parse_args()

    # 从命名空间中获取参数并赋值给变量
    pytorch_dump_folder_path: Path = args.pytorch_dump_folder_path
    # 如果文件夹不存在，则创建它
    pytorch_dump_folder_path.mkdir(exist_ok=True, parents=True)
    van_dir = args.van_dir

    # 将 VAN 实现目录的父目录添加到 sys.path 中，以方便导入其他模块
    # 导入 VAN 模块中的各个模型
    sys.path.append(str(van_dir.parent))
    from van.models.van import van_base, van_large, van_small, van_tiny

    # 调用函数来转换权重并推送到 Hub 上
    convert_weights_and_push(pytorch_dump_folder_path, args.model_name, args.push_to_hub)
```