# `.\models\levit\convert_levit_timm_to_pytorch.py`

```
# 设置编码格式为 UTF-8
# 版权声明，这段代码由 HuggingFace Inc. 团队版权所有，遵循 Apache License, Version 2.0 授权
#
# 根据许可证规定，除非符合许可证的条件，否则不得使用此文件
# 可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，
# 没有任何形式的担保或条件，无论是明示的还是默示的。
# 详见许可证了解更多信息。
"""从 timm 转换 LeViT 检查点。"""

# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import json  # 用于处理 JSON 格式数据
from collections import OrderedDict  # 有序字典，用于按照插入顺序存储键值对
from functools import partial  # 用于创建偏函数
from pathlib import Path  # 处理文件路径的类库

import timm  # 提供预训练模型的创建和管理
import torch  # PyTorch 深度学习框架
from huggingface_hub import hf_hub_download  # 用于从 HuggingFace Hub 下载模型和文件

from transformers import LevitConfig, LevitForImageClassificationWithTeacher, LevitImageProcessor  # LeViT 模型相关类
from transformers.utils import logging  # 日志记录模块

# 设置日志输出为 info 级别
logging.set_verbosity_info()
logger = logging.get_logger()

# 定义函数：转换权重并推送到 Hub
def convert_weight_and_push(
    hidden_sizes: int, name: str, config: LevitConfig, save_directory: Path, push_to_hub: bool = True
):
    print(f"Converting {name}...")

    # 禁用梯度计算
    with torch.no_grad():
        # 根据不同的 hidden_sizes 加载不同的 LeViT 模型
        if hidden_sizes == 128:
            if name[-1] == "S":
                from_model = timm.create_model("levit_128s", pretrained=True)
            else:
                from_model = timm.create_model("levit_128", pretrained=True)
        elif hidden_sizes == 192:
            from_model = timm.create_model("levit_192", pretrained=True)
        elif hidden_sizes == 256:
            from_model = timm.create_model("levit_256", pretrained=True)
        elif hidden_sizes == 384:
            from_model = timm.create_model("levit_384", pretrained=True)

        # 设置模型为评估模式
        from_model.eval()
        our_model = LevitForImageClassificationWithTeacher(config).eval()
        huggingface_weights = OrderedDict()

        # 获取源模型的权重，并根据键的映射将其赋给新模型
        weights = from_model.state_dict()
        og_keys = list(from_model.state_dict().keys())
        new_keys = list(our_model.state_dict().keys())
        print(len(og_keys), len(new_keys))
        for i in range(len(og_keys)):
            huggingface_weights[new_keys[i]] = weights[og_keys[i]]
        our_model.load_state_dict(huggingface_weights)

        # 创建随机输入张量并计算两个模型的输出结果
        x = torch.randn((2, 3, 224, 224))
        out1 = from_model(x)
        out2 = our_model(x).logits

    # 检查两个模型输出是否相等
    assert torch.allclose(out1, out2), "The model logits don't match the original one."

    # 设置检查点名称为模型名称
    checkpoint_name = name
    print(checkpoint_name)

    # 如果指定推送到 Hub，则保存模型和相关处理器，并输出推送成功信息
    if push_to_hub:
        our_model.save_pretrained(save_directory / checkpoint_name)
        image_processor = LevitImageProcessor()
        image_processor.save_pretrained(save_directory / checkpoint_name)

        print(f"Pushed {checkpoint_name}")

# 定义函数：转换权重并推送到 Hub
def convert_weights_and_push(save_directory: Path, model_name: str = None, push_to_hub: bool = True):
    filename = "imagenet-1k-id2label.json"
    num_labels = 1000
    # 预期模型输出的形状为 (1, num_labels)
    expected_shape = (1, num_labels)

    # 定义用于下载模型配置的 Hugging Face 仓库 ID
    repo_id = "huggingface/label-files"
    # 将 num_labels 赋值给变量 num_labels
    num_labels = num_labels
    # 使用 Hugging Face Hub 下载指定文件名的数据集，并加载为 JSON 格式
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    # 将 id2label 中的键转换为整数类型，并保留原始值
    id2label = {int(k): v for k, v in id2label.items()}

    # 将 id2label 赋值给变量 id2label
    id2label = id2label
    # 创建一个将标签映射到 ID 的字典
    label2id = {v: k for k, v in id2label.items()}

    # 定义一个部分应用的函数，使用 ImageNet 预训练配置创建 LevitConfig
    ImageNetPreTrainedConfig = partial(LevitConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)

    # 定义不同 Levit 模型名称到隐藏层大小的映射关系
    names_to_hidden_sizes = {
        "levit-128S": 128,
        "levit-128": 128,
        "levit-192": 192,
        "levit-256": 256,
        "levit-384": 384,
    }

    # 定义不同 Levit 模型名称到其配置对象的映射关系
    names_to_config = {
        "levit-128S": ImageNetPreTrainedConfig(
            hidden_sizes=[128, 256, 384],
            num_attention_heads=[4, 6, 8],
            depths=[2, 3, 4],
            key_dim=[16, 16, 16],
            drop_path_rate=0,
        ),
        "levit-128": ImageNetPreTrainedConfig(
            hidden_sizes=[128, 256, 384],
            num_attention_heads=[4, 8, 12],
            depths=[4, 4, 4],
            key_dim=[16, 16, 16],
            drop_path_rate=0,
        ),
        "levit-192": ImageNetPreTrainedConfig(
            hidden_sizes=[192, 288, 384],
            num_attention_heads=[3, 5, 6],
            depths=[4, 4, 4],
            key_dim=[32, 32, 32],
            drop_path_rate=0,
        ),
        "levit-256": ImageNetPreTrainedConfig(
            hidden_sizes=[256, 384, 512],
            num_attention_heads=[4, 6, 8],
            depths=[4, 4, 4],
            key_dim=[32, 32, 32],
            drop_path_rate=0,
        ),
        "levit-384": ImageNetPreTrainedConfig(
            hidden_sizes=[384, 512, 768],
            num_attention_heads=[6, 9, 12],
            depths=[4, 4, 4],
            key_dim=[32, 32, 32],
            drop_path_rate=0.1,
        ),
    }

    # 如果给定了模型名称，则转换权重并推送到指定的 Hub
    if model_name:
        convert_weight_and_push(
            names_to_hidden_sizes[model_name], model_name, names_to_config[model_name], save_directory, push_to_hub
        )
    else:  # 否则对所有模型进行转换权重并推送操作
        for model_name, config in names_to_config.items():
            convert_weight_and_push(names_to_hidden_sizes[model_name], model_name, config, save_directory, push_to_hub)
    
    # 返回最终的配置对象和预期的输出形状
    return config, expected_shape
if __name__ == "__main__":
    # 如果当前脚本作为主程序执行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建命令行参数解析器对象

    # 必选参数
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="The name of the model you wish to convert, it must be one of the supported Levit* architecture,",
    )
    # 添加模型名称参数，指定需要转换的模型名称，必须是支持的 Levit* 架构之一

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="levit-dump-folder/",
        type=Path,
        required=False,
        help="Path to the output PyTorch model directory.",
    )
    # 添加 PyTorch 模型输出文件夹路径参数，默认为 'levit-dump-folder/'，指定输出 PyTorch 模型的目录路径

    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image processor to the hub")
    # 添加推送到 Hub 的选项参数，如果设置该参数，则推送模型和图像处理器到 Hub

    parser.add_argument(
        "--no-push_to_hub",
        dest="push_to_hub",
        action="store_false",
        help="Do not push model and image processor to the hub",
    )
    # 添加不推送到 Hub 的选项参数，设置该参数则不将模型和图像处理器推送到 Hub

    args = parser.parse_args()
    # 解析命令行参数，将参数存储在 args 对象中

    pytorch_dump_folder_path: Path = args.pytorch_dump_folder_path
    # 获取 PyTorch 模型输出文件夹路径，并将其赋值给 pytorch_dump_folder_path 变量
    pytorch_dump_folder_path.mkdir(exist_ok=True, parents=True)
    # 创建 PyTorch 模型输出文件夹，如果不存在则创建，确保存在父文件夹路径

    convert_weights_and_push(pytorch_dump_folder_path, args.model_name, args.push_to_hub)
    # 调用函数，将权重转换并推送到指定的 PyTorch 模型文件夹路径，使用指定的模型名称和推送到 Hub 的标志
```