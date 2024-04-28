# `.\transformers\models\plbart\convert_plbart_original_checkpoint_to_torch.py`

```
# 版权声明和许可信息
# 版权声明和许可信息，指定了代码的版权和许可信息
# 从库中导入必要的模块和类
import argparse
import torch
from torch import nn
from transformers import PLBartConfig, PLBartForConditionalGeneration, PLBartForSequenceClassification

# 从状态字典中移除指定的键
def remove_ignore_keys_(state_dict):
    # 定义需要移除的键列表
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "decoder.output_projection.weight",
    ]
    # 遍历并移除指定的键
    for k in ignore_keys:
        state_dict.pop(k, None)

# 从嵌入层创建线性层
def make_linear_from_emb(emb):
    # 获取嵌入层的词汇大小和嵌入维度
    vocab_size, emb_size = emb.weight.shape
    # 创建一个线性层，用于将嵌入层转换为线性层
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer

# 从磁盘中加载 Fairseq PLBart 检查点并转换为 Hugging Face 模型
def convert_fairseq_plbart_checkpoint_from_disk(
    checkpoint_path, hf_config_path="uclanlp/plbart-base", finetuned=False, classification=False
):
    # 加载模型检查点的状态字典
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    # 移除不需要的键
    remove_ignore_keys_(state_dict)
    # 获取词汇大小
    vocab_size = state_dict["encoder.embed_tokens.weight"].shape[0]

    # 根据 Hugging Face 配置路径创建 PLBart 配置
    plbart_config = PLBartConfig.from_pretrained(hf_config_path, vocab_size=vocab_size)

    # 将共享权重设置为解码器的嵌入权重
    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    if not classification:
        # 创建条件生成模型
        model = PLBartForConditionalGeneration(plbart_config)
        model.model.load_state_dict(state_dict)
        if finetuned:
            # 如果进行了微调，则将 lm_head 设置为从共享权重创建的线性层
            model.lm_head = make_linear_from_emb(model.model.shared)

    else:
        # 创建序列分类模型
        classification_head = {}
        for key, value in state_dict.copy().items():
            if key.startswith("classification_heads.sentence_classification_head"):
                classification_head[key.replace("classification_heads.sentence_classification_head.", "")] = value
                state_dict.pop(key)
        model = PLBartForSequenceClassification(plbart_config)
        model.model.load_state_dict(state_dict)
        model.classification_head.load_state_dict(classification_head)

    return model

# 主函数，解析命令行参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必需参数
    parser.add_argument("fairseq_path", type=str, help="model.pt on local filesystem.")
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加一个命令行参数，用于指定 Hugging Face 模型的配置
    parser.add_argument(
        "--hf_config",
        default="uclanlp/plbart-base",
        type=str,
        help="Which huggingface architecture to use: plbart-base",
    )
    # 添加一个命令行参数，用于指定模型是否为微调后的检查点
    parser.add_argument("--finetuned", action="store_true", help="whether the model is a fine-tuned checkpoint")
    # 添加一个命令行参数，用于指定模型是否为分类检查点
    parser.add_argument(
        "--classification", action="store_true", help="whether the model is a classification checkpoint"
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 从磁盘中加载 Fairseq PLBART 检查点并转换为 Hugging Face 模型
    model = convert_fairseq_plbart_checkpoint_from_disk(
        args.fairseq_path,
        hf_config_path=args.hf_config,
        finetuned=args.finetuned,
        classification=args.classification,
    )
    # 将转换后的模型保存到指定路径
    model.save_pretrained(args.pytorch_dump_folder_path)
```