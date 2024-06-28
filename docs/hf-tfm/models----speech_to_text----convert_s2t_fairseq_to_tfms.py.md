# `.\models\speech_to_text\convert_s2t_fairseq_to_tfms.py`

```py
# 导入必要的库
import argparse  # 导入 argparse 库，用于处理命令行参数

import torch  # 导入 PyTorch 库
from torch import nn  # 导入 PyTorch 中的 nn 模块，用于神经网络构建

from transformers import Speech2TextConfig, Speech2TextForConditionalGeneration  # 导入 transformers 库中的 Speech2TextConfig 和 Speech2TextForConditionalGeneration 类

# 定义函数，移除 state_dict 中指定的键
def remove_ignore_keys_(state_dict):
    # 需要移除的键列表
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "decoder.output_projection.weight",
        "_float_tensor",
        "encoder.embed_positions._float_tensor",
        "decoder.embed_positions._float_tensor",
    ]
    # 从 state_dict 中移除指定的键
    for k in ignore_keys:
        state_dict.pop(k, None)

# 定义函数，重命名 state_dict 中的键名
def rename_keys(s_dict):
    keys = list(s_dict.keys())  # 获取 state_dict 的所有键名
    # 遍历键名列表
    for key in keys:
        # 替换包含 "transformer_layers" 的键名为 "layers"
        if "transformer_layers" in key:
            s_dict[key.replace("transformer_layers", "layers")] = s_dict.pop(key)
        # 替换包含 "subsample" 的键名为 "conv"
        elif "subsample" in key:
            s_dict[key.replace("subsample", "conv")] = s_dict.pop(key)

# 定义函数，根据输入的嵌入层创建线性层
def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape  # 获取嵌入层的词汇大小和嵌入维度
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)  # 创建一个无偏置的线性层
    lin_layer.weight.data = emb.weight.data  # 将嵌入层的权重数据复制到线性层的权重中
    return lin_layer

# 定义函数，将 Fairseq 的语音到文本检查点转换为 Transformers 模型
def convert_fairseq_s2t_checkpoint_to_tfms(checkpoint_path, pytorch_dump_folder_path):
    m2m_100 = torch.load(checkpoint_path, map_location="cpu")  # 加载 Fairseq 检查点
    args = m2m_100["args"]  # 获取模型参数
    state_dict = m2m_100["model"]  # 获取模型的 state_dict
    lm_head_weights = state_dict["decoder.output_projection.weight"]  # 获取解码器输出投影层的权重

    remove_ignore_keys_(state_dict)  # 调用函数移除不需要的键名
    rename_keys(state_dict)  # 调用函数重命名键名

    vocab_size = state_dict["decoder.embed_tokens.weight"].shape[0]  # 获取嵌入层的词汇大小

    tie_embeds = args.share_decoder_input_output_embed  # 检查是否共享解码器的输入输出嵌入

    conv_kernel_sizes = [int(i) for i in args.conv_kernel_sizes.split(",")]  # 解析卷积核大小列表
    # 创建一个语音到文本转换模型的配置对象，配置包括词汇大小、最大源和目标位置、编码器和解码器层数、注意力头数等
    config = Speech2TextConfig(
        vocab_size=vocab_size,
        max_source_positions=args.max_source_positions,
        max_target_positions=args.max_target_positions,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        encoder_attention_heads=args.encoder_attention_heads,
        decoder_attention_heads=args.decoder_attention_heads,
        encoder_ffn_dim=args.encoder_ffn_embed_dim,
        decoder_ffn_dim=args.decoder_ffn_embed_dim,
        d_model=args.encoder_embed_dim,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout,
        activation_function="relu",
        num_conv_layers=len(conv_kernel_sizes),
        conv_channels=args.conv_channels,
        conv_kernel_sizes=conv_kernel_sizes,
        input_feat_per_channel=args.input_feat_per_channel,
        input_channels=args.input_channels,
        tie_word_embeddings=tie_embeds,
        num_beams=5,
        max_length=200,
        use_cache=True,
        decoder_start_token_id=2,
        early_stopping=True,
    )

    # 使用上面配置的模型配置对象创建语音到文本转换模型
    model = Speech2TextForConditionalGeneration(config)

    # 加载模型的状态字典，并忽略丢失的一些键，记录下丢失的和不期望的键
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)

    # 如果有丢失的键，并且丢失的键不在预期的键集合内，则抛出值错误异常
    if len(missing) > 0 and not set(missing) <= {
        "encoder.embed_positions.weights",
        "decoder.embed_positions.weights",
    }:
        raise ValueError(
            "Only `encoder.embed_positions.weights` and `decoder.embed_positions.weights` are allowed to be missing,"
            f" but all the following weights are missing {missing}"
        )

    # 如果要求绑定嵌入，则将语言模型头部替换为从嵌入中创建的线性层；否则，直接加载预训练的语言模型头部权重
    if tie_embeds:
        model.lm_head = make_linear_from_emb(model.model.decoder.embed_tokens)
    else:
        model.lm_head.weight.data = lm_head_weights

    # 将模型保存到指定的 PyTorch 模型保存路径
    model.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必选参数
    parser.add_argument("--fairseq_path", type=str, help="Path to the fairseq model (.pt) file.")
    # 添加一个参数选项，指定 fairseq 模型文件的路径，类型为字符串

    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加一个参数选项，指定输出 PyTorch 模型的文件夹路径，默认值为 None，类型为字符串

    args = parser.parse_args()
    # 解析命令行参数，并将其存储在 args 变量中

    convert_fairseq_s2t_checkpoint_to_tfms(args.fairseq_path, args.pytorch_dump_folder_path)
    # 调用函数 convert_fairseq_s2t_checkpoint_to_tfms，传入 fairseq 模型文件路径和 PyTorch 输出文件夹路径作为参数
```