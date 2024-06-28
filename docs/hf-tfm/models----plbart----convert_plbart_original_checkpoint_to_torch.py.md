# `.\models\plbart\convert_plbart_original_checkpoint_to_torch.py`

```
# 导入必要的库和模块
import argparse  # 导入命令行参数解析模块

import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch的神经网络模块

from transformers import PLBartConfig, PLBartForConditionalGeneration, PLBartForSequenceClassification  # 导入transformers库中的PLBart配置和模型类


def remove_ignore_keys_(state_dict):
    # 定义需要从state_dict中移除的键列表
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "decoder.output_projection.weight",
    ]
    # 遍历并移除state_dict中的特定键
    for k in ignore_keys:
        state_dict.pop(k, None)


def make_linear_from_emb(emb):
    # 获取嵌入层的词汇量和嵌入维度大小
    vocab_size, emb_size = emb.weight.shape
    # 创建一个线性层，其输入大小为词汇量，输出大小为嵌入维度，且没有偏置项
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    # 将线性层的权重初始化为嵌入层的权重
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def convert_fairseq_plbart_checkpoint_from_disk(
    checkpoint_path, hf_config_path="uclanlp/plbart-base", finetuned=False, classification=False
):
    # 从磁盘加载模型的state_dict，使用CPU进行映射
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    # 移除state_dict中的忽略键
    remove_ignore_keys_(state_dict)
    # 获取嵌入层的词汇量大小
    vocab_size = state_dict["encoder.embed_tokens.weight"].shape[0]

    # 根据指定的hf_config_path加载PLBart模型的配置
    plbart_config = PLBartConfig.from_pretrained(hf_config_path, vocab_size=vocab_size)

    # 将state_dict中的"decoder.embed_tokens.weight"复制给"shared.weight"
    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]

    if not classification:
        # 如果不是分类任务，创建条件生成的PLBart模型
        model = PLBartForConditionalGeneration(plbart_config)
        # 加载模型的state_dict
        model.model.load_state_dict(state_dict)
        if finetuned:
            # 如果进行了微调，将lm_head替换为基于嵌入层的线性层
            model.lm_head = make_linear_from_emb(model.model.shared)
    else:
        # 如果是分类任务，初始化分类头部字典
        classification_head = {}
        # 将state_dict中的分类头部相关项移动到classification_head字典中
        for key, value in state_dict.copy().items():
            if key.startswith("classification_heads.sentence_classification_head"):
                classification_head[key.replace("classification_heads.sentence_classification_head.", "")] = value
                state_dict.pop(key)
        # 创建序列分类的PLBart模型
        model = PLBartForSequenceClassification(plbart_config)
        # 加载模型的state_dict和分类头部的state_dict
        model.model.load_state_dict(state_dict)
        model.classification_head.load_state_dict(classification_head)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必需参数：fairseq_path表示本地文件系统上的模型.pt文件
    parser.add_argument("fairseq_path", type=str, help="model.pt on local filesystem.")
    # 可选参数：pytorch_dump_folder_path表示输出PyTorch模型的路径
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--hf_config",
        default="uclanlp/plbart-base",
        type=str,
        help="Which huggingface architecture to use: plbart-base",
    )
    # 添加命令行参数 --hf_config，指定 Huggingface 模型配置的名称，默认为 uclanlp/plbart-base

    parser.add_argument("--finetuned", action="store_true", help="whether the model is a fine-tuned checkpoint")
    # 添加命令行参数 --finetuned，指示模型是否是经过微调的检查点

    parser.add_argument(
        "--classification", action="store_true", help="whether the model is a classification checkpoint"
    )
    # 添加命令行参数 --classification，指示模型是否是一个分类检查点

    args = parser.parse_args()
    # 解析命令行参数并存储到 args 变量中

    model = convert_fairseq_plbart_checkpoint_from_disk(
        args.fairseq_path,
        hf_config_path=args.hf_config,
        finetuned=args.finetuned,
        classification=args.classification,
    )
    # 调用函数 convert_fairseq_plbart_checkpoint_from_disk，从磁盘加载 Fairseq 的 PLBART 检查点，
    # 使用给定的参数来转换为 Huggingface 模型

    model.save_pretrained(args.pytorch_dump_folder_path)
    # 将转换后的 Huggingface 模型保存到指定的 PyTorch dump 文件夹路径中
```