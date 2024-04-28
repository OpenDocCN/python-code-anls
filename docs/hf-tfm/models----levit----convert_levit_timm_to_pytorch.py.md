# `.\transformers\models\levit\convert_levit_timm_to_pytorch.py`

```py
# coding=utf-8
# 导入所需的模块和库
# 加载命令行参数解析器
import argparse
# 导入 json 模块
import json
# 导入有序字典类
from collections import OrderedDict
# 导入 functools 中的 partial 函数
from functools import partial
# 导入 pathlib 中的 Path 类
from pathlib import Path

# 导入 timm 模块
import timm
# 导入 torch 库
import torch
# 导入 Hugging Face Hub 中的 hf_hub_download 函数
from huggingface_hub import hf_hub_download

# 导入 transformers 库中的相关模块和函数
# 导入 LevitConfig 类
from transformers import LevitConfig
# 导入 LevitForImageClassificationWithTeacher 类
from transformers import LevitForImageClassificationWithTeacher
# 导入 LevitImageProcessor 类
from transformers import LevitImageProcessor
# 导入 transformers 库中的 logging 模块
from transformers.utils import logging

# 设置日志输出等级为 info
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger()


# 定义函数 convert_weight_and_push，用于转换权重并推送到 Hugging Face Hub
# hidden_sizes：隐藏层大小
# name：模型名称
# config：LevitConfig 配置对象
# save_directory：保存目录
# push_to_hub：是否推送到 Hub，默认为 True
def convert_weight_and_push(
    hidden_sizes: int, name: str, config: LevitConfig, save_directory: Path, push_to_hub: bool = True
):
    # 打印转换信息
    print(f"Converting {name}...")

    # 禁用梯度计算
    with torch.no_grad():
        # 根据隐藏层大小选择对应的 Levit 模型
        if hidden_sizes == 128:
            if name[-1] == "S":
                from_model = timm.create_model("levit_128s", pretrained=True)
            else:
                from_model = timm.create_model("levit_128", pretrained=True)
        if hidden_sizes == 192:
            from_model = timm.create_model("levit_192", pretrained=True)
        if hidden_sizes == 256:
            from_model = timm.create_model("levit_256", pretrained=True)
        if hidden_sizes == 384:
            from_model = timm.create_model("levit_384", pretrained=True)

        # 设置模型为评估模式
        from_model.eval()
        # 创建我们的 LevitForImageClassificationWithTeacher 模型
        our_model = LevitForImageClassificationWithTeacher(config).eval()
        # 创建 Hugging Face 权重的有序字典
        huggingface_weights = OrderedDict()

        # 获取原始模型的权重
        weights = from_model.state_dict()
        og_keys = list(from_model.state_dict().keys())
        new_keys = list(our_model.state_dict().keys())
        print(len(og_keys), len(new_keys))
        # 将原始模型的权重复制到我们的模型中
        for i in range(len(og_keys)):
            huggingface_weights[new_keys[i]] = weights[og_keys[i]]
        our_model.load_state_dict(huggingface_weights)

        # 创建一个输入张量
        x = torch.randn((2, 3, 224, 224))
        # 获取原始模型的输出
        out1 = from_model(x)
        # 获取我们模型的输出
        out2 = our_model(x).logits

    # 检查模型输出是否一致
    assert torch.allclose(out1, out2), "The model logits don't match the original one."

    # 设置检查点名称
    checkpoint_name = name
    print(checkpoint_name)

    # 如果设置为推送到 Hub，则保存模型和处理器并推送到 Hub
    if push_to_hub:
        our_model.save_pretrained(save_directory / checkpoint_name)
        image_processor = LevitImageProcessor()
        image_processor.save_pretrained(save_directory / checkpoint_name)

        # 打印推送信息
        print(f"Pushed {checkpoint_name}")


# 定义函数 convert_weights_and_push，用于转换权重并推送到 Hugging Face Hub
# save_directory：保存目录
# model_name：模型名称，默认为 None
# push_to_hub：是否推送到 Hub，默认为 True
def convert_weights_and_push(save_directory: Path, model_name: str = None, push_to_hub: bool = True):
    # 设置文件名
    filename = "imagenet-1k-id2label.json"
    # 设置标签数量
    num_labels = 1000
    # 期望的形状为 (1, num_labels)
    expected_shape = (1, num_labels)
    
    # GitHub 仓库 ID
    repo_id = "huggingface/label-files"
    # 标签数量
    num_labels = num_labels
    # 从文件中加载 ID 到标签的映射关系
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    # 将 ID 到标签的映射关系转换为整数型的字典
    id2label = {int(k): v for k, v in id2label.items()}
    
    # ID 到标签的映射关系
    id2label = id2label
    # 标签到 ID 的映射关系
    label2id = {v: k for k, v in id2label.items()}
    
    # ImageNet 预训练配置，部分函数参数已经设置为预定义值
    ImageNetPreTrainedConfig = partial(LevitConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)
    
    # 不同模型名称对应的隐藏层大小
    names_to_hidden_sizes = {
        "levit-128S": 128,
        "levit-128": 128,
        "levit-192": 192,
        "levit-256": 256,
        "levit-384": 384,
    }
    
    # 不同模型名称对应的配置信息
    names_to_config = {
        "levit-128S": ImageNetPreTrainedConfig(
            hidden_sizes=[128, 256, 384],
            num_attention_heads=[4, 6, 8],
            depths=[2, 3, 4],
            key_dim=[16, 16, 16],
            drop_path_rate=0,
        ),
        # levit-128, levit-192, levit-256, levit-384 配置类似，仅 hidden_sizes, num_attention_heads, depths, key_dim, drop_path_rate 不同
    }
    
    # 如果有指定模型名称，则将该模型的权重转换并推送到 Hub
    if model_name:
        convert_weight_and_push(
            names_to_hidden_sizes[model_name], model_name, names_to_config[model_name], save_directory, push_to_hub
        )
    # 否则，遍历所有模型名称及其配置，将权重转换并推送到 Hub
    else:
        for model_name, config in names_to_config.items():
            convert_weight_and_push(names_to_hidden_sizes[model_name], model_name, config, save_directory, push_to_hub)
    # 返回配置和期望形状
    return config, expected_shape
# 判断当前模块是否作为主程序运行
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 必要参数的添加
    parser.add_argument(
        # 指定模型名称的参数
        "--model_name",
        # 默认值为 None
        default=None,
        # 参数类型为字符串
        type=str,
        # 参数用途说明
        help="你想转换的模型名称，它必须是支持的 Levit* 架构之一",
    )
    # 添加 PyTorch 模型输出目录路径的参数
    parser.add_argument(
        "--pytorch_dump_folder_path",
        # 默认输出目录
        default="levit-dump-folder/",
        # 参数类型为路径
        type=Path,
        # 是否为必需参数
        required=False,
        # 参数用途说明
        help="PyTorch 模型输出目录的路径",
    )
    # 添加是否将模型推送到集中式仓库的参数，作为布尔开关
    parser.add_argument("--push_to_hub", action="store_true", help="将模型和图像处理器推送到中央仓库")
    parser.add_argument(
        # 添加相反意义的参数，禁止推送到中央仓库
        "--no-push_to_hub",
        # 目标参数的指定
        dest="push_to_hub",
        # 设置操作为 'store_false'，即如果传递了此参数，`push_to_hub` 将为 False
        action="store_false",
        # 参数用途说明
        help="不将模型和图像处理器推送到中央仓库",
    )

    # 解析命令行参数并赋值给 `args`
    args = parser.parse_args()
    # 获取解析到的 PyTorch 模型输出目录路径
    pytorch_dump_folder_path: Path = args.pytorch_dump_folder_path
    # 创建目录，如果目录已存在则不会引发错误
    pytorch_dump_folder_path.mkdir(exist_ok=True, parents=True)
    # 将权重转换并推送到指定目录或中央仓库
    convert_weights_and_push(pytorch_dump_folder_path, args.model_name, args.push_to_hub)
```