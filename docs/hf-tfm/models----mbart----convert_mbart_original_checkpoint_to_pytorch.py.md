# `.\models\mbart\convert_mbart_original_checkpoint_to_pytorch.py`

```
# 导入必要的库和模块
import argparse  # 用于解析命令行参数

import torch  # PyTorch库，用于机器学习和深度学习任务
from torch import nn  # PyTorch的神经网络模块

from transformers import MBartConfig, MBartForConditionalGeneration  # Hugging Face Transformers库中的MBart配置和生成模型


def remove_ignore_keys_(state_dict):
    # 定义要从状态字典中移除的键列表
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "decoder.output_projection.weight",
    ]
    # 遍历并从状态字典中移除指定的键
    for k in ignore_keys:
        state_dict.pop(k, None)


def make_linear_from_emb(emb):
    # 从嵌入层创建线性层
    vocab_size, emb_size = emb.weight.shape  # 获取嵌入层的词汇量和嵌入维度大小
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)  # 创建一个线性层，没有偏置项
    lin_layer.weight.data = emb.weight.data  # 将嵌入层的权重数据复制到线性层的权重中
    return lin_layer  # 返回创建的线性层


def convert_fairseq_mbart_checkpoint_from_disk(
    checkpoint_path, hf_config_path="facebook/mbart-large-en-ro", finetuned=False, mbart_50=False
):
    # 从磁盘加载Fairseq MBart模型的检查点并转换为Hugging Face MBart模型
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]  # 加载检查点的状态字典
    remove_ignore_keys_(state_dict)  # 移除状态字典中指定的键

    vocab_size = state_dict["encoder.embed_tokens.weight"].shape[0]  # 获取编码器嵌入词汇表大小

    # 根据预训练配置路径加载MBart配置
    mbart_config = MBartConfig.from_pretrained(hf_config_path, vocab_size=vocab_size)
    if mbart_50 and finetuned:
        mbart_config.activation_function = "relu"  # 如果是50层MBart并且是微调模型，则设置激活函数为ReLU

    # 将decoder的嵌入权重设为shared.weight
    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]

    # 创建MBart条件生成模型
    model = MBartForConditionalGeneration(mbart_config)
    model.model.load_state_dict(state_dict)  # 加载状态字典到模型中

    if finetuned:
        model.lm_head = make_linear_from_emb(model.model.shared)  # 如果是微调模型，则使用make_linear_from_emb创建lm_head

    return model  # 返回转换后的Hugging Face MBart模型


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建参数解析器

    # 添加必需的参数
    parser.add_argument(
        "fairseq_path", type=str, help="bart.large, bart.large.cnn or a path to a model.pt on local filesystem."
    )
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")

    # 添加可选参数
    parser.add_argument(
        "--hf_config",
        default="facebook/mbart-large-cc25",
        type=str,
        help="Which huggingface architecture to use: mbart-large",
    )
    parser.add_argument("--mbart_50", action="store_true", help="whether the model is mMART-50 checkpoint")
    parser.add_argument("--finetuned", action="store_true", help="whether the model is a fine-tuned checkpoint")

    args = parser.parse_args()  # 解析命令行参数
    model = convert_fairseq_mbart_checkpoint_from_disk(
        args.fairseq_path, hf_config_path=args.hf_config, finetuned=args.finetuned, mbart_50=args.mbart_50
    )
    # 调用模型对象的save_pretrained方法，将模型保存到指定的路径args.pytorch_dump_folder_path中
    model.save_pretrained(args.pytorch_dump_folder_path)
```