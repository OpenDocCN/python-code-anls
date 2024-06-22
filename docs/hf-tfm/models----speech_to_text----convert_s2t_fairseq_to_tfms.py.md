# `.\transformers\models\speech_to_text\convert_s2t_fairseq_to_tfms.py`

```py
# 导入所需的库
import argparse
import torch
from torch import nn
from transformers import Speech2TextConfig, Speech2TextForConditionalGeneration

# 定义函数用于移除忽略的键
def remove_ignore_keys_(state_dict):
    # 定义需要忽略的键列表
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
    # 遍历忽略的键，从状态字典中移除这些键
    for k in ignore_keys:
        state_dict.pop(k, None)

# 定义函数用于重命名键
def rename_keys(s_dict):
    keys = list(s_dict.keys())
    # 遍历状态字典的键
    for key in keys:
        # 替换包含"transformer_layers"的键名
        if "transformer_layers" in key:
            s_dict[key.replace("transformer_layers", "layers")] = s_dict.pop(key)
        # 替换包含"subsample"的键名
        elif "subsample" in key:
            s_dict[key.replace("subsample", "conv")] = s_dict.pop(key)

# 定义函数用于将嵌入层转换为线性层
def make_linear_from_emb(emb):
    # 获取嵌入层的词汇大小和嵌入维度大小
    vocab_size, emb_size = emb.weight.shape
    # 创建一个线性层，从嵌入层权重初始化线性层权重
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer

# 定义函数用于将fairseq模型的检查点转换为transformers模型的权重
def convert_fairseq_s2t_checkpoint_to_tfms(checkpoint_path, pytorch_dump_folder_path):
    # 加载fairseq模型的检查点
    m2m_100 = torch.load(checkpoint_path, map_location="cpu")
    args = m2m_100["args"]
    state_dict = m2m_100["model"]
    lm_head_weights = state_dict["decoder.output_projection.weight"]

    # 移除忽略的键
    remove_ignore_keys_(state_dict)
    # 重命名键
    rename_keys(state_dict)

    # 获取词汇大小
    vocab_size = state_dict["decoder.embed_tokens.weight"].shape[0]

    # 检查是否共享解码器的输入和输出嵌入
    tie_embeds = args.share_decoder_input_output_embed

    # 解码器卷积核大小列表
    conv_kernel_sizes = [int(i) for i in args.conv_kernel_sizes.split(",")]
    # 创建一个Speech2TextConfig对象，用于配置语音转文本模型
    config = Speech2TextConfig(
        vocab_size=vocab_size,  # 词汇表大小
        max_source_positions=args.max_source_positions,  # 输入序列的最大长度
        max_target_positions=args.max_target_positions,  # 输出序列的最大长度
        encoder_layers=args.encoder_layers,  # 编码器层数
        decoder_layers=args.decoder_layers,  # 解码器层数
        encoder_attention_heads=args.encoder_attention_heads,  # 编码器注意力头数
        decoder_attention_heads=args.decoder_attention_heads,  # 解码器注意力头数
        encoder_ffn_dim=args.encoder_ffn_embed_dim,  # 编码器中FFN层的维度
        decoder_ffn_dim=args.decoder_ffn_embed_dim,  # 解码器中FFN层的维度
        d_model=args.encoder_embed_dim,  # 模型维度
        dropout=args.dropout,  # Dropout率
        attention_dropout=args.attention_dropout,  # 注意力机制中的Dropout率
        activation_dropout=args.activation_dropout,  # 激活函数中的Dropout率
        activation_function="relu",  # 激活函数类型
        num_conv_layers=len(conv_kernel_sizes),  # 卷积层的数量
        conv_channels=args.conv_channels,  # 卷积层的通道数
        conv_kernel_sizes=conv_kernel_sizes,  # 卷积核的尺寸列表
        input_feat_per_channel=args.input_feat_per_channel,  # 每个通道的输入特征数
        input_channels=args.input_channels,  # 输入通道数
        tie_word_embeddings=tie_embeds,  # 是否绑定词嵌入
        num_beams=5,  # Beam搜索中的Beam数
        max_length=200,  # 生成序列的最大长度
        use_cache=True,  # 是否使用缓存
        decoder_start_token_id=2,  # 解码器起始标记的ID
        early_stopping=True,  # 是否启用早停策略
    )
    
    # 创建一个Speech2TextForConditionalGeneration模型对象
    model = Speech2TextForConditionalGeneration(config)
    
    # 加载模型的状态字典，并返回缺失和意外的键
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    
    # 如果有缺失的权重，并且缺失的键不是下列预定义的键集合中的子集，则抛出异常
    if len(missing) > 0 and not set(missing) <= {
        "encoder.embed_positions.weights",
        "decoder.embed_positions.weights",
    }:
        raise ValueError(
            "Only `encoder.embed_positions.weights` and `decoder.embed_positions.weights`  are allowed to be missing,"
            f" but all the following weights are missing {missing}"
        )
    
    # 如果词嵌入是绑定的，则通过解码器的嵌入词嵌入创建一个线性层，并将其分配给模型的语言模型头部
    if tie_embeds:
        model.lm_head = make_linear_from_emb(model.model.decoder.embed_tokens)
    else:
        # 否则，将预训练的语言模型头部的权重数据赋给模型的语言模型头部
        model.lm_head.weight.data = lm_head_weights
    
    # 将模型保存到指定的PyTorch模型保存路径
    model.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本被直接执行（而不是被导入为模块）
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument("--fairseq_path", type=str, help="Path to the fairseq model (.pt) file.")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将 fairseq 模型转换为 PyTorch 模型
    convert_fairseq_s2t_checkpoint_to_tfms(args.fairseq_path, args.pytorch_dump_folder_path)
```