# `.\models\bart\convert_bart_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置文件编码为 UTF-8
# 版权声明，指明代码版权归 HuggingFace Inc. 团队所有，遵循 Apache License 2.0
# 引入必要的库和模块
import argparse  # 引入命令行参数解析模块
import os  # 引入操作系统相关功能模块
from pathlib import Path  # 引入处理文件路径的模块

import fairseq  # 引入 fairseq 库，用于处理 BART 模型
import torch  # 引入 PyTorch 深度学习框架
from packaging import version  # 引入版本管理模块

from torch import nn  # 引入 PyTorch 的神经网络模块

from transformers import (  # 引入 Transformers 库中的相关模块和类
    BartConfig,
    BartForConditionalGeneration,
    BartForSequenceClassification,
    BartModel,
    BartTokenizer,
)
from transformers.utils import logging  # 引入 Transformers 的日志记录模块

# 定义 Fairseq 中已有的 BART 模型列表
FAIRSEQ_MODELS = ["bart.large", "bart.large.mnli", "bart.large.cnn", "bart_xsum/model.pt"]
# 额外的架构映射，用于将模型名称映射到对应的类
extra_arch = {"bart.large": BartModel, "bart.large.mnli": BartForSequenceClassification}

# 检查 Fairseq 版本是否大于等于 0.9.0，否则抛出异常
if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")

# 设置日志记录的详细程度为信息级别
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 示例文本用于后续操作
SAMPLE_TEXT = " Hello world! cécé herlolip"

# 用于 MNLI 模型的键重命名列表
mnli_rename_keys = [
    ("model.classification_heads.mnli.dense.weight", "classification_head.dense.weight"),
    ("model.classification_heads.mnli.dense.bias", "classification_head.dense.bias"),
    ("model.classification_heads.mnli.out_proj.weight", "classification_head.out_proj.weight"),
    ("model.classification_heads.mnli.out_proj.bias", "classification_head.out_proj.bias"),
]

# 移除状态字典中的特定键，这些键不被需要
def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)

# 将字典中的旧键重命名为新键
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val

# 从给定的检查点路径加载 XSum 模型
def load_xsum_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    # 使用 torch 加载模型检查点
    sd = torch.load(checkpoint_path, map_location="cpu")
    # 从 PyTorch hub 中加载 BART CNN 模型
    hub_interface = torch.hub.load("pytorch/fairseq", "bart.large.cnn").eval()
    # 加载模型的状态字典
    hub_interface.model.load_state_dict(sd["model"])
    return hub_interface

# 从给定的嵌入层创建线性层
def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer

# 转换 BART 模型检查点到 Hugging Face 的格式
@torch.no_grad()
def convert_bart_checkpoint(checkpoint_path, pytorch_dump_folder_path, hf_checkpoint_name=None):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    # 如果给定的检查点路径不存在，则从 PyTorch hub 加载 BART 模型
    if not os.path.exists(checkpoint_path):
        bart = torch.hub.load("pytorch/fairseq", checkpoint_path).eval()
    else:
        # 否则，加载 XSum 检查点
        bart = load_xsum_checkpoint(checkpoint_path)
    # 更新 BART 模型的状态字典
    bart.model.upgrade_state_dict(bart.model.state_dict())
    # 如果 hf_checkpoint_name 为 None，则使用 checkpoint_path 替换 '.' 为 '-' 作为 hf_checkpoint_name
    if hf_checkpoint_name is None:
        hf_checkpoint_name = checkpoint_path.replace(".", "-")
    # 根据 hf_checkpoint_name 加载预训练配置
    config = BartConfig.from_pretrained(hf_checkpoint_name)
    # 使用 bart 对象编码示例文本 SAMPLE_TEXT，并添加一个维度
    tokens = bart.encode(SAMPLE_TEXT).unsqueeze(0)
    # 使用 hf_checkpoint_name 加载 BartTokenizer，并编码示例文本 SAMPLE_TEXT，并返回 PyTorch 张量，并添加一个维度
    tokens2 = BartTokenizer.from_pretrained(hf_checkpoint_name).encode(SAMPLE_TEXT, return_tensors="pt").unsqueeze(0)
    # 检查 tokens 和 tokens2 是否完全相等，否则抛出异常
    if not torch.eq(tokens, tokens2).all():
        raise ValueError(
            f"converted tokenizer and pretrained tokenizer returned different output: {tokens} != {tokens2}"
        )

    # 如果 checkpoint_path 是 "bart.large.mnli"
    if checkpoint_path == "bart.large.mnli":
        # 获取当前 bart 模型的状态字典
        state_dict = bart.state_dict()
        # 移除需要忽略的键
        remove_ignore_keys_(state_dict)
        # 重命名特定键名到 mnli_rename_keys 中定义的目标键名
        state_dict["model.shared.weight"] = state_dict["model.decoder.embed_tokens.weight"]
        for src, dest in mnli_rename_keys:
            rename_key(state_dict, src, dest)
        # 加载 Sequence Classification 模型
        model = BartForSequenceClassification(config).eval()
        # 加载状态字典到模型
        model.load_state_dict(state_dict)
        # 使用 bart 预测 mnli，返回 logits
        fairseq_output = bart.predict("mnli", tokens, return_logits=True)
        # 使用新加载的模型进行推理，得到新的模型输出（logits）
        new_model_outputs = model(tokens)[0]
    else:  # 没有分类头需要担心
        # 获取当前 bart 模型的状态字典
        state_dict = bart.model.state_dict()
        # 移除需要忽略的键
        remove_ignore_keys_(state_dict)
        # 更新共享权重的键名
        state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
        # 使用 bart 提取特征
        fairseq_output = bart.extract_features(tokens)
        # 如果 hf_checkpoint_name 是 "facebook/bart-large"
        if hf_checkpoint_name == "facebook/bart-large":
            # 加载 BartModel 模型
            model = BartModel(config).eval()
            # 加载状态字典到模型
            model.load_state_dict(state_dict)
            # 使用新加载的模型进行推理，得到新的模型输出
            new_model_outputs = model(tokens).model[0]
        else:
            # 加载 BartForConditionalGeneration 模型
            model = BartForConditionalGeneration(config).eval()  # 一个现有的摘要检查点
            # 加载状态字典到模型
            model.model.load_state_dict(state_dict)
            # 如果模型有 lm_head 属性，则使用 make_linear_from_emb 函数创建线性层
            if hasattr(model, "lm_head"):
                model.lm_head = make_linear_from_emb(model.model.shared)
            # 使用新加载的模型进行推理，得到新的模型输出
            new_model_outputs = model.model(tokens)[0]

    # 检查输出结果的形状是否相等
    if fairseq_output.shape != new_model_outputs.shape:
        raise ValueError(
            f"`fairseq_output` shape and `new_model_output` shape are different: {fairseq_output.shape=}, {new_model_outputs.shape}"
        )
    # 如果 fairseq_output 和 new_model_outputs 中有任何不同的值，抛出异常
    if (fairseq_output != new_model_outputs).any().item():
        raise ValueError("Some values in `fairseq_output` are different from `new_model_outputs`")
    # 创建 PyTorch dump 文件夹路径（如果不存在）
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 将模型保存到指定的 PyTorch dump 文件夹路径中
    model.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码
    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # 必选参数
    parser.add_argument(
        "fairseq_path", type=str, help="bart.large, bart.large.cnn or a path to a model.pt on local filesystem."
    )
    # 添加一个必选参数 fairseq_path，用于指定 fairseq 模型的路径或名称

    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加一个必选参数 pytorch_dump_folder_path，用于指定 PyTorch 模型输出的路径

    parser.add_argument(
        "--hf_config", default=None, type=str, help="Which huggingface architecture to use: bart-large-xsum"
    )
    # 添加一个可选参数 --hf_config，用于指定要使用的 Hugging Face 模型架构

    args = parser.parse_args()
    # 解析命令行参数，并将其存储在 args 对象中

    convert_bart_checkpoint(args.fairseq_path, args.pytorch_dump_folder_path, hf_checkpoint_name=args.hf_config)
    # 调用 convert_bart_checkpoint 函数，传递解析后的参数：fairseq_path, pytorch_dump_folder_path 和 hf_config
```