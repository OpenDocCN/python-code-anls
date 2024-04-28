# `.\transformers\models\visual_bert\convert_visual_bert_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本使用此文件
# 只有在遵守许可证的情况下才能使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""将 VisualBert 检查点转换为其他格式。"""


# 导入必要的库
import argparse
from collections import OrderedDict
from pathlib import Path

import torch

# 导入 transformers 库中的相关模块
from transformers import (
    VisualBertConfig,
    VisualBertForMultipleChoice,
    VisualBertForPreTraining,
    VisualBertForQuestionAnswering,
    VisualBertForVisualReasoning,
)
from transformers.utils import logging

# 设置日志级别为 info
logging.set_verbosity_info()
# 获取 logger
logger = logging.get_logger(__name__)

# 定义需要重命名的键的前缀
rename_keys_prefix = [
    ("bert.bert", "visual_bert"),
    ("bert.cls", "cls"),
    ("bert.classifier", "cls"),
    ("token_type_embeddings_visual", "visual_token_type_embeddings"),
    ("position_embeddings_visual", "visual_position_embeddings"),
    ("projection", "visual_projection"),
]

# 可接受的检查点列表
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

# 加载状态字典
def load_state_dict(checkpoint_path):
    sd = torch.load(checkpoint_path, map_location="cpu")
    return sd

# 获取新的字典
def get_new_dict(d, config, rename_keys_prefix=rename_keys_prefix):
    new_d = OrderedDict()
    new_d["visual_bert.embeddings.position_ids"] = torch.arange(config.max_position_embeddings).expand((1, -1))
    # detector_d = OrderedDict()
    for key in d:
        if "detector" in key:
            # detector_d[key.replace('detector.','')] = d[key]
            continue
        new_key = key
        for name_pair in rename_keys_prefix:
            new_key = new_key.replace(name_pair[0], name_pair[1])
        new_d[new_key] = d[key]
        if key == "bert.cls.predictions.decoder.weight":
            # 旧的 bert 代码没有 `decoder.bias`，但是被单独添加了
            new_d["cls.predictions.decoder.bias"] = new_d["cls.predictions.bias"]
    return new_d

# 转换 VisualBert 检查点
@torch.no_grad()
def convert_visual_bert_checkpoint(checkpoint_path, pytorch_dump_folder_path):
    """
    将模型的权重复制/粘贴/调整到我们的 VisualBERT 结构中。
    """

    assert (
        checkpoint_path.split("/")[-1] in ACCEPTABLE_CHECKPOINTS
    ), f"The checkpoint provided must be in {ACCEPTABLE_CHECKPOINTS}."

    # 获取配置
    # 检查是否在检查点路径中包含 "pre" 字符串
    if "pre" in checkpoint_path:
        # 设置模型类型为 "pretraining"
        model_type = "pretraining"
        # 根据不同的检查点路径设置不同的配置参数
        if "vcr" in checkpoint_path:
            config_params = {"visual_embedding_dim": 512}
        elif "vqa_advanced" in checkpoint_path:
            config_params = {"visual_embedding_dim": 2048}
        elif "vqa" in checkpoint_path:
            config_params = {"visual_embedding_dim": 2048}
        elif "nlvr" in checkpoint_path:
            config_params = {"visual_embedding_dim": 1024}
        else:
            # 如果没有找到对应的实现，则抛出 NotImplementedError 异常
            raise NotImplementedError(f"No implementation found for `{checkpoint_path}`.")
    else:
        # 根据不同的检查点路径设置不同的配置参数和模型类型
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

    # 获取新的状态字典
    new_state_dict = get_new_dict(state_dict, config)

    # 根据模型类型选择对应的模型类实例化
    if model_type == "pretraining":
        model = VisualBertForPreTraining(config)
    elif model_type == "vqa":
        model = VisualBertForQuestionAnswering(config)
    elif model_type == "nlvr":
        model = VisualBertForVisualReasoning(config)
    elif model_type == "multichoice":
        model = VisualBertForMultipleChoice(config)

    # 加载模型的状态字典
    model.load_state_dict(new_state_dict)
    # 保存检查点
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument("orig_checkpoint_path", type=str, help="A path to .th on local filesystem.")
    parser.add_argument("pytorch_dump_folder_path", type=str, help="Path to the output PyTorch model.")
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将原始检查点路径和PyTorch模型输出路径作为参数传递
    convert_visual_bert_checkpoint(args.orig_checkpoint_path, args.pytorch_dump_folder_path)
```