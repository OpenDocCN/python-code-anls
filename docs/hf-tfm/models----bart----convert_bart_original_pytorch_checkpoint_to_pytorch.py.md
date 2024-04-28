# `.\transformers\models\bart\convert_bart_original_pytorch_checkpoint_to_pytorch.py`

```py
# 导入所需模块和库
import argparse  # 解析命令行参数
import os  # 提供与操作系统交互的功能
from pathlib import Path  # 提供处理文件路径的功能

import fairseq  # 引入 fairseq 库
import torch  # 引入 PyTorch 库
from packaging import version  # 用于处理版本信息
from torch import nn  # 引入 PyTorch 中的神经网络模块

from transformers import (  # 从 transformers 库中导入指定模块
    BartConfig,  # BART 模型配置类
    BartForConditionalGeneration,  # 用于条件生成的 BART 模型类
    BartForSequenceClassification,  # 序列分类任务的 BART 模型类
    BartModel,  # BART 模型类
    BartTokenizer,  # BART 用于分词的 Tokenizer 类
)
from transformers.utils import logging  # 引入日志记录工具

# 定义 fairseq 中的预训练模型列表
FAIRSEQ_MODELS = ["bart.large", "bart.large.mnli", "bart.large.cnn", "bart_xsum/model.pt"]
# 针对特定模型额外提供的架构信息
extra_arch = {"bart.large": BartModel, "bart.large.mnli": BartForSequenceClassification}
# 检查 fairseq 版本是否符合要求
if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")

# 设置日志级别为 info
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger(__name__)

# 示例文本
SAMPLE_TEXT = " Hello world! cécé herlolip"

# 需要重命名的键列表，用于处理 MNLI 模型中的键名
mnli_rename_keys = [
    ("model.classification_heads.mnli.dense.weight", "classification_head.dense.weight"),
    ("model.classification_heads.mnli.dense.bias", "classification_head.dense.bias"),
    ("model.classification_heads.mnli.out_proj.weight", "classification_head.out_proj.weight"),
    ("model.classification_heads.mnli.out_proj.bias", "classification_head.out_proj.bias"),
]

# 移除忽略的键
def remove_ignore_keys_(state_dict):
    # 忽略的键列表
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
    ]
    # 从状态字典中移除忽略的键
    for k in ignore_keys:
        state_dict.pop(k, None)

# 重命名键
def rename_key(dct, old, new):
    # 从字典中弹出旧键对应的值，然后重新以新键加入字典
    val = dct.pop(old)
    dct[new] = val

# 加载 XSum 数据集的检查点
def load_xsum_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    # 加载模型的状态字典
    sd = torch.load(checkpoint_path, map_location="cpu")
    # 使用 PyTorch Hub 加载 BART-Large-CNN 模型
    hub_interface = torch.hub.load("pytorch/fairseq", "bart.large.cnn").eval()
    # 将加载的模型状态设置到 PyTorch Hub 接口中
    hub_interface.model.load_state_dict(sd["model"])
    # 返回加载后的模型接口
    return hub_interface

# 从嵌入层创建线性层
def make_linear_from_emb(emb):
    # 获取嵌入层的词汇大小和嵌入维度
    vocab_size, emb_size = emb.weight.shape
    # 创建一个线性层，输入大小为词汇大小，输出大小为嵌入维度，无偏置
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    # 将线性层的权重设置为嵌入层的权重
    lin_layer.weight.data = emb.weight.data
    # 返回线性层
    return lin_layer

# 转换 BART 检查点
@torch.no_grad()
def convert_bart_checkpoint(checkpoint_path, pytorch_dump_folder_path, hf_checkpoint_name=None):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    # 如果检查点路径不存在，则加载模型
    if not os.path.exists(checkpoint_path):
        bart = torch.hub.load("pytorch/fairseq", checkpoint_path).eval()
    else:
        bart = load_xsum_checkpoint(checkpoint_path)
    # 将当前模型的状态字典升级为 BART 模型的状态字典
    bart.model.upgrade_state_dict(bart.model.state_dict())
    # 如果 hf_checkpoint_name 为 None，则将其设为替换掉点号的 checkpoint_path
    if hf_checkpoint_name is None:
        hf_checkpoint_name = checkpoint_path.replace(".", "-")
    # 从预训练模型加载配置
    config = BartConfig.from_pretrained(hf_checkpoint_name)
    # 对示例文本进行编码并增加维度
    tokens = bart.encode(SAMPLE_TEXT).unsqueeze(0)
    # 使用预训练的分词器对示例文本进行编码并增加维度
    tokens2 = BartTokenizer.from_pretrained(hf_checkpoint_name).encode(SAMPLE_TEXT, return_tensors="pt").unsqueeze(0)
    # 检查 tokens 和 tokens2 是否相等，如果不相等则抛出异常
    if not torch.eq(tokens, tokens2).all():
        raise ValueError(
            f"converted tokenizer and pretrained tokenizer returned different output: {tokens} != {tokens2}"
        )

    # 如果 checkpoint_path 为 "bart.large.mnli"
    if checkpoint_path == "bart.large.mnli":
        # 获取当前模型的状态字典
        state_dict = bart.state_dict()
        # 移除忽略的键
        remove_ignore_keys_(state_dict)
        # 将 "model.decoder.embed_tokens.weight" 复制给 "model.shared.weight"
        state_dict["model.shared.weight"] = state_dict["model.decoder.embed_tokens.weight"]
        # 重命名键
        for src, dest in mnli_rename_keys:
            rename_key(state_dict, src, dest)
        # 创建用于序列分类的 BART 模型并加载状态字典
        model = BartForSequenceClassification(config).eval()
        model.load_state_dict(state_dict)
        # 使用 BART 模型进行预测
        fairseq_output = bart.predict("mnli", tokens, return_logits=True)
        # 获取新模型输出（对数）
        new_model_outputs = model(tokens)[0]
    else:  # 没有需要担心的分类头
        # 获取当前模型的状态字典
        state_dict = bart.model.state_dict()
        # 移除忽略的键
        remove_ignore_keys_(state_dict)
        # 将 "decoder.embed_tokens.weight" 复制给 "shared.weight"
        state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
        # 使用 BART 模型提取特征
        fairseq_output = bart.extract_features(tokens)
        # 如果 hf_checkpoint_name 为 "facebook/bart-large"
        if hf_checkpoint_name == "facebook/bart-large":
            # 创建 BART 模型并加载状态字典
            model = BartModel(config).eval()
            model.load_state_dict(state_dict)
            # 获取新模型输出
            new_model_outputs = model(tokens).model[0]
        else:
            # 创建用于条件生成的 BART 模型并加载状态字典
            model = BartForConditionalGeneration(config).eval()  # an existing summarization ckpt
            model.model.load_state_dict(state_dict)
            # 如果模型具有 "lm_head" 属性，则将其替换为从模型共享层创建的线性层
            if hasattr(model, "lm_head"):
                model.lm_head = make_linear_from_emb(model.model.shared)
            # 获取新模型输出
            new_model_outputs = model.model(tokens)[0]

    # 检查结果
    # 如果 fairseq_output 和 new_model_outputs 的形状不相等，则抛出异常
    if fairseq_output.shape != new_model_outputs.shape:
        raise ValueError(
            f"`fairseq_output` shape and `new_model_output` shape are different: {fairseq_output.shape=}, {new_model_outputs.shape}"
        )
    # 如果 fairseq_output 和 new_model_outputs 中有任何不相等的值，则抛出异常
    if (fairseq_output != new_model_outputs).any().item():
        raise ValueError("Some values in `fairseq_output` are different from `new_model_outputs`")
    # 创建目录（如果不存在）
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 保存预训练模型
    model.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本被直接执行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数
    parser.add_argument(
        "fairseq_path", type=str, help="bart.large, bart.large.cnn or a path to a model.pt on local filesystem."
    )
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--hf_config", default=None, type=str, help="Which huggingface architecture to use: bart-large-xsum"
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将 fairseq 模型转换为 PyTorch 模型
    convert_bart_checkpoint(args.fairseq_path, args.pytorch_dump_folder_path, hf_checkpoint_name=args.hf_config)
```