# `.\models\visual_bert\convert_visual_bert_original_pytorch_checkpoint_to_pytorch.py`

```
# coding=utf-8
# 定义代码文件的字符编码为UTF-8

# 版权声明
# 2021年由HuggingFace Inc.团队版权所有。
#
# 根据Apache许可证2.0版（“许可证”）授权；
# 您只能在符合许可证的情况下使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件
# 依据“现状”分发，没有任何形式的明示或暗示的担保或条件。
# 有关许可证的详细信息，请参阅
# 许可证。

"""Convert VisualBert checkpoint."""
# 头部文档字符串，指明本文件用途为转换VisualBert检查点。

import argparse
from collections import OrderedDict
from pathlib import Path

import torch

from transformers import (
    VisualBertConfig,
    VisualBertForMultipleChoice,
    VisualBertForPreTraining,
    VisualBertForQuestionAnswering,
    VisualBertForVisualReasoning,
)
# 导入所需的库和模块

from transformers.utils import logging

# 设置日志输出级别为info
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义需要重命名的键前缀列表
rename_keys_prefix = [
    ("bert.bert", "visual_bert"),
    ("bert.cls", "cls"),
    ("bert.classifier", "cls"),
    ("token_type_embeddings_visual", "visual_token_type_embeddings"),
    ("position_embeddings_visual", "visual_position_embeddings"),
    ("projection", "visual_projection"),
]

# 可接受的检查点文件名列表
ACCEPTABLE_CHECKPOINTS = [
    "nlvr2_coco_pre_trained.th",
    "nlvr2_fine_tuned.th",
    "nlvr2_pre_trained.th",
    "vcr_coco_pre_train.th",
    "vcr_fine_tune.th",
    "vcr_pre_train.th",
    "vqa_coco_pre_trained.th",
    "vqa_fine_tuned.th",
    "vqa_pre_trained.th",
]

# 加载模型状态字典的函数
def load_state_dict(checkpoint_path):
    # 使用CPU加载检查点文件的状态字典
    sd = torch.load(checkpoint_path, map_location="cpu")
    return sd

# 根据给定的状态字典和配置信息，生成适配VisualBert模型的新字典
def get_new_dict(d, config, rename_keys_prefix=rename_keys_prefix):
    new_d = OrderedDict()
    # 创建新字典中的'visual_bert.embeddings.position_ids'键，对应的值为一个torch张量，表示位置ID
    new_d["visual_bert.embeddings.position_ids"] = torch.arange(config.max_position_embeddings).expand((1, -1))
    # 遍历原始字典的键
    for key in d:
        if "detector" in key:
            # 如果键中包含'detector'，则跳过处理
            continue
        new_key = key
        # 使用预定义的前缀对键名进行重命名
        for name_pair in rename_keys_prefix:
            new_key = new_key.replace(name_pair[0], name_pair[1])
        # 将重命名后的键值对添加到新字典中
        new_d[new_key] = d[key]
        # 特殊处理，如果键为'bert.cls.predictions.decoder.weight'，则额外添加'decoder.bias'到新字典
        if key == "bert.cls.predictions.decoder.weight":
            new_d["cls.predictions.decoder.bias"] = new_d["cls.predictions.bias"]
    return new_d

# 无梯度运行的函数装饰器，用于转换VisualBert检查点
@torch.no_grad()
def convert_visual_bert_checkpoint(checkpoint_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our VisualBERT structure.
    """
    # 断言检查，确保提供的检查点文件名在可接受的检查点文件列表中
    assert (
        checkpoint_path.split("/")[-1] in ACCEPTABLE_CHECKPOINTS
    ), f"The checkpoint provided must be in {ACCEPTABLE_CHECKPOINTS}."

    # 获取配置信息
    # 如果检查点路径中包含字符串 "pre"，则模型类型为预训练
    if "pre" in checkpoint_path:
        model_type = "pretraining"
        # 根据检查点路径中的特定字符串确定配置参数
        if "vcr" in checkpoint_path:
            config_params = {"visual_embedding_dim": 512}
        elif "vqa_advanced" in checkpoint_path:
            config_params = {"visual_embedding_dim": 2048}
        elif "vqa" in checkpoint_path:
            config_params = {"visual_embedding_dim": 2048}
        elif "nlvr" in checkpoint_path:
            config_params = {"visual_embedding_dim": 1024}
        else:
            # 如果未找到适合的实现，抛出 NotImplementedError 异常
            raise NotImplementedError(f"No implementation found for `{checkpoint_path}`.")
    else:
        # 如果检查点路径不包含 "pre"，则根据其他字符串确定模型类型和配置参数
        if "vcr" in checkpoint_path:
            config_params = {"visual_embedding_dim": 512}
            model_type = "multichoice"
        elif "vqa_advanced" in checkpoint_path:
            config_params = {"visual_embedding_dim": 2048}
            model_type = "vqa_advanced"
        elif "vqa" in checkpoint_path:
            config_params = {"visual_embedding_dim": 2048, "num_labels": 3129}
            model_type = "vqa"
        elif "nlvr" in checkpoint_path:
            config_params = {
                "visual_embedding_dim": 1024,
                "num_labels": 2,
            }
            model_type = "nlvr"

    # 根据配置参数创建 VisualBertConfig 对象
    config = VisualBertConfig(**config_params)

    # 加载模型的状态字典
    state_dict = load_state_dict(checkpoint_path)

    # 根据状态字典和配置创建新的状态字典
    new_state_dict = get_new_dict(state_dict, config)

    # 根据模型类型选择相应的模型类实例化
    if model_type == "pretraining":
        model = VisualBertForPreTraining(config)
    elif model_type == "vqa":
        model = VisualBertForQuestionAnswering(config)
    elif model_type == "nlvr":
        model = VisualBertForVisualReasoning(config)
    elif model_type == "multichoice":
        model = VisualBertForMultipleChoice(config)

    # 将新的状态字典加载到模型中
    model.load_state_dict(new_state_dict)

    # 确保 PyTorch dump 文件夹路径存在，如果不存在则创建
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)

    # 将模型保存为 PyTorch 预训练格式
    model.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 如果这个脚本是直接运行的（而不是被导入的），则执行以下代码
    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # 必选参数
    parser.add_argument("orig_checkpoint_path", type=str, help="A path to .th on local filesystem.")
    # 添加一个必选参数：原始检查点文件的路径，必须是一个本地文件系统上的 .th 文件

    parser.add_argument("pytorch_dump_folder_path", type=str, help="Path to the output PyTorch model.")
    # 添加一个必选参数：输出 PyTorch 模型的路径

    args = parser.parse_args()
    # 解析命令行参数，并将其存储在 args 对象中

    convert_visual_bert_checkpoint(args.orig_checkpoint_path, args.pytorch_dump_folder_path)
    # 调用函数 convert_visual_bert_checkpoint，传入两个参数：原始检查点路径和输出模型路径
```