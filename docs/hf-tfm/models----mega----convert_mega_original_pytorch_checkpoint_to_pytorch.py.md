# `.\transformers\models\mega\convert_mega_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的详细信息

"""
将 Mega 预训练检查点转换为 Hugging Face 模型。用于转换位于 https://huggingface.co/mnaylor/mega-wikitext-103 的掩码语言模型检查点

要求：
  - 克隆 Mega 仓库并从中安装 fairseq
    1. git clone https://github.com/facebookresearch/mega.git
    2. cd mega && pip install -e
  - 克隆原始实现的预训练权重从 Hugging Face 仓库
    * 使用此位置作为预训练权重的路径
"""
import argparse

# 导入模型权重和配置文件的实用程序
import os
import pickle as pkl

# PyTorch + 新模型类
import torch
from torch import nn

from transformers import AutoTokenizer, MegaConfig, MegaForMaskedLM


# 导入用于预训练的 EncoderLayer 类
# !! 注意 !! 这需要在安装 Mega 源时构建的 fairseq 版本
try:
    from fairseq.modules.mega_layer import MegaEncoderLayer
except ImportError:
    raise ImportError("您需要安装 Mega 仓库中的 fairseq 版本！")


# 定义用于训练 MLM 的包装类（参见下面的 colab 笔记本）
# https://colab.research.google.com/drive/1qfUO6o5HRdxBblWlw058HVyvaEPhPpH8?usp=sharing
# MegaLM 输出隐藏状态
class MegaLM(nn.Module):
    "我们的 Mega 编码器的基类 - 给定输入 ID，嵌入文本并返回编码器输出"

    def __init__(self, mega_args, depth, vocab_size):
        super().__init__()
        self.mega_args = mega_args
        self.embedding_layer = nn.Embedding(vocab_size, self.mega_args.encoder_embed_dim)
        self.encoders = nn.ModuleList([MegaEncoderLayer(self.mega_args) for _ in range(depth)])
        self.depth = depth
    # 定义一个前向传播函数，接受来自 Hugging Face tokenizer 的 input_ids 和 attention_mask 作为 PyTorch 张量，
    # 返回一个大小为 (batch, n_classes) 的包含分类 logits 的张量
    def forward(self, input_ids, attention_mask, batch_first=True, ignore_mask_value=0):
        
        # Mega 要求嵌入是 (time, batch, embedding size)，但 Hugging Face 返回的是 (batch, time)
        if batch_first:
            # 转置 input_ids，使其变为 (time, batch)
            input_ids = input_ids.T

        # Mega 要求注意力掩码是 (batch, time)，但值为 0 (正常标记) 和 1 (忽略标记)，与 HF 返回的相反
        if ignore_mask_value == 0:
            # 将 attention_mask 反转，使其变为 1 - attention_mask
            attention_mask = 1 - attention_mask

        # 从 ID 中获取标记嵌入
        embeds = self.embedding_layer(input_ids)

        # 通过 Mega 层
        # 输入为 (time, batch, encoder dim)，输出相同
        for encoder in self.encoders:
            embeds = encoder(embeds, attention_mask)

        # 根据指定的形状返回结果
        if batch_first:
            # (T, B, H) --> (B, T, H)
            return torch.transpose(embeds, 0, 1)
        else:
            return embeds
# 将 MegaForMaskedLM 重命名为 OriginalMegaForMaskedLM，以避免与新模块混淆
class OriginalMegaForMaskedLM(nn.Module):
    "A wrapper class for doing masked language modeling with Mega"

    def __init__(self, mega_args, depth, vocab_size):
        super().__init__()
        # 初始化 OriginalMegaForMaskedLM 类，包含 MegaLM 对象和 MLM 头部
        self.mega = MegaLM(mega_args, depth, vocab_size)
        self.mlm_head = nn.Linear(mega_args.encoder_embed_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, batch_first=True, ignore_mask_value=0):
        """
        Perform a forward pass through the Mega encoder and the masked LM head. Returns logits for each vocabulary
        entry.

        If `batch_first` (default to align with Hugging Face tokenizer behavior), output will have the shape (Batch
        size, Sequence length, Vocab size); otherwise (S, B, V)
        """
        # 执行 Mega 编码器和 MLM 头部的前向传播，返回每个词汇条目的 logits
        encoder_output = self.mega(input_ids, attention_mask, batch_first, ignore_mask_value)
        return self.mlm_head(self.dropout(encoder_output))


# 用于将位于用户指定位置的检查点转换为 Huggingface 格式的代码
def convert_checkpoint_to_huggingface(pretrained_checkpoint_path, output_path, includes_tokenizer):
    with open(os.path.join(pretrained_checkpoint_path, "model_args.pkl"), "rb") as f:
        # 从文件中加载 Mega 模型的原始参数
        mega_original_args = pkl.load(f)

    # 加载原始编码器
    original_mlm = OriginalMegaForMaskedLM(**mega_original_args).eval()

    # 加载权重
    print(
        "Original Mega encoder:",
        # 加载编码器权重
        original_mlm.mega.load_state_dict(
            torch.load(os.path.join(pretrained_checkpoint_path, "encoder_weights.pt"), map_location="cpu")
        ),
    )
    print(
        "Original Mega MLM layer:",
        # 加载 MLM 头部权重
        original_mlm.mlm_head.load_state_dict(
            torch.load(os.path.join(pretrained_checkpoint_path, "mlm_head_weights.pt"), map_location="cpu")
        ),
    )

    # 从旧配置创建一个新配置
    # 创建 MegaConfig 对象，用于配置 Mega 模型的参数
    hf_config = MegaConfig(
        num_hidden_layers=mega_original_args["depth"],  # 隐藏层的数量
        vocab_size=mega_original_args["vocab_size"],  # 词汇表的大小
        hidden_size=mega_original_args["mega_args"].encoder_embed_dim,  # 隐藏层的大小
        shared_representation_size=mega_original_args["mega_args"].encoder_z_dim,  # 共享表示的大小
        intermediate_size=mega_original_args["mega_args"].encoder_hidden_dim,  # 中间层的大小
        ema_projection_size=mega_original_args["mega_args"].encoder_n_dim,  # EMA 投影的大小
        dropout_prob=mega_original_args["mega_args"].dropout,  # 丢弃概率
        attention_probs_dropout_prob=mega_original_args["mega_args"].attention_dropout,  # 注意力概率的丢弃概率
        hidden_dropout_prob=mega_original_args["mega_args"].hidden_dropout,  # 隐藏层的丢弃概率
        activation=mega_original_args["mega_args"].activation_fn,  # 激活函数
        attention_activation=mega_original_args["mega_args"].attention_activation_fn,  # 注意力激活函数
        bidirectional=mega_original_args["mega_args"].bidirectional,  # 是否双向
        use_chunking=mega_original_args["mega_args"].encoder_chunk_size > 0,  # 是否使用分块
        chunk_size=mega_original_args["mega_args"].encoder_chunk_size,  # 分块的大小
        truncation=mega_original_args["mega_args"].truncation_length,  # 截断长度
        normalization_type=mega_original_args["mega_args"].normalization_type,  # 归一化类型
        normalize_before_mega=True,  # 在 Mega 之前进行归一化
        norm_affine=True,  # 归一化仿射
        use_feature_dropout=mega_original_args["mega_args"].feature_dropout,  # 是否使用特征丢弃
        relative_positional_bias=mega_original_args["mega_args"].rel_pos_bias,  # 相对位置偏置
        max_positions=mega_original_args["mega_args"].max_source_positions,  # 最大位置
        nffn_hidden_size=mega_original_args["mega_args"].encoder_ffn_embed_dim,  # NFFN 隐藏层的大小
        normalize_before_ffn=mega_original_args["mega_args"].normalize_before,  # 在 NFFN 之前进行归一化
        # 为 HF 实现添加的新参数
        nffn_activation_dropout_prob=0.0,  # NFFN 激活丢弃概率
        add_token_type_embeddings=False,  # 是否添加 token 类型嵌入
        add_lm_hidden_dense_layer=False,  # 是否添加 LM 隐藏密集层
    )

    # 创建 MegaForMaskedLM 模型，并设置为评估模式
    hf_mlm = MegaForMaskedLM(hf_config).eval()

    # 原始检查点只使用 nn.Embedding 进行词嵌入，将原始模型的词嵌入权重赋值�� Mega 模型的词嵌入层
    hf_mlm.mega.embedding_layer.word_embeddings.weight = original_mlm.mega.embedding_layer.weight

    # 修改原始检查点的状态字典，以解决 Hugging Face 生态系统中的命名问题
    # 任何包含 "beta" 或 "gamma" 的名称都不安全，并在 _load_pretrained 时重命名
    # 还重命名以前令人困惑的参数名称
    original_state_dict = original_mlm.mega.encoders.state_dict()
    updated_keys = {}
    # 遍历原始状态字典中的模块名
    for module_name in original_state_dict.keys():
        new_module_name = None
        # 由于 gamma、beta 和 alpha 在原始存储库中在多个模块中使用，需要分别处理它们
        # beta 在 EMA、MovingAverageGatedAttention 和 RotaryRelativePositionalBias 中使用，必须重命名以适应 flax/tf 权重
        # EMA 子层从 "move" 重命名为 "ema_gate" 以提高可读性，因此在此处也进行了相同操作
        if "beta" in module_name:
            # EMA 子层在原始存储库中始终称为 "move.beta"
            if "move.beta" in module_name:
                new_module_name = module_name.replace("move.beta", "ema_gate.ema_expansion_matrix")
            elif "mega_layer.beta" in module_name:
                new_module_name = module_name.replace("beta", "qk_bias")
            else:
                new_module_name = module_name.replace("beta", "b_param")
        # beta 在 EMA 和 MovingAverageGatedAttention 中使用，必须重命名以适应 flax/tf 权重
        elif "gamma" in module_name:
            if "move.gamma" in module_name:
                new_module_name = module_name.replace("move.gamma", "ema_gate.kernel_projection_matrix")
            elif "mega_layer.gamma" in module_name:
                new_module_name = module_name.replace("gamma", "qk_weight")
            else:
                new_module_name = module_name.replace("gamma", "g_param")
        # alpha 在 EMA 和位置偏差中使用；重命名以提高可读性
        elif "move.alpha" in module_name:
            new_module_name = module_name.replace("move.alpha", "ema_gate.decay_factor")
        # delta 仅在 EMA 中使用；重命名以提高可读性
        elif "move.delta" in module_name:
            new_module_name = module_name.replace("move.delta", "ema_gate.damping_factor")
        # omega 仅在 EMA 中使用；重命名以提高可读性
        elif "omega" in module_name:
            new_module_name = module_name.replace("move.omega", "ema_gate.residual_weight")

        # 如果有新的模块名，则更新键
        if new_module_name:
            updated_keys[module_name] = new_module_name

    # 如果有需要重命名的键，则打印这些键
    if len(updated_keys) != 0:
        print(f"Renaming these keys: {updated_keys.keys()}")
    else:
        print("No need to rename state dict entries")
    
    # 将更新后的键值对应关系应用到原始状态字典中
    for old, new in updated_keys.items():
        original_state_dict[new] = original_state_dict.pop(old)

    # 尝试使用更新后的名称加载状态字典
    # 注意，现在我们称之为 `mega.layers` 而不是 `mega.encoders`，以适应 hugging face 风格
    print("HF Mega encoder:", hf_mlm.mega.layers.load_state_dict(original_state_dict))

    # 直接加载 MLM 头权重
    print(
        "HF Mega MLM layer:",
        hf_mlm.mlm_head.load_state_dict(
            torch.load(os.path.join(pretrained_checkpoint_path, "mlm_head_weights.pt"), map_location="cpu")
        ),
    )

    # 在随机生成的输入序列上进行测试
    # 生成一个形状为 (4, 256) 的张量，其中的值在 [0, hf_config.vocab_size) 范围内
    input_ids = torch.randint(0, hf_config.vocab_size, size=(4, 256))
    # 生成一个与 input_ids 相同形状的张量，所有值为1，表示所有 token 都未被 mask
    input_mask = torch.ones_like(input_ids)
    # 将最后10个 token 的 mask 置为0，用于测试 mask 是否正确应用
    input_mask[:, -10:] = 0

    # 运行前向传播
    original_output = original_mlm(input_ids, input_mask, batch_first=True, ignore_mask_value=0)
    # 使用 hf_mlm 进行前向传播
    hf_output = hf_mlm(input_ids, input_mask)[0]

    # 打印输出的形状和差异
    print(f"original output {original_output.shape}")
    print(f"hf output {hf_output.shape}")
    print(f"max diff: {(original_output - hf_output).max()}")  # 0.0
    # 检查两个输出张量是否在给定的误差范围内相等
    success = torch.allclose(original_output, hf_output, atol=1e-3)

    # 如果两个输出张量相等，则打印 "Yay!" 并保存 hf_mlm 的预训练模型
    if success:
        print("Yay!")
        hf_mlm.save_pretrained(output_path)
    # 如果两个输出张量不相等，则抛出异常并打印相关信息
    else:
        raise RuntimeError(f"Something's broken :(\nOriginal:\n{original_output}\n\nHF\n{hf_output}\n{hf_mlm}")

    # 如果需要包含 tokenizer，则加载预训练模型的 tokenizer 并保存
    if includes_tokenizer:
        print("Transferring tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint_path)
        tokenizer.save_pretrained(output_path)
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加预训练模型检查点路径参数
    parser.add_argument(
        "--pretrained_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Point to the directory containing your model weights using the official Mega repo",
    )

    # 添加输出路径参数
    parser.add_argument(
        "--output_path", default=None, type=str, required=True, help="Location to save the Hugging Face version"
    )

    # 添加是否包含分词器参数
    parser.add_argument(
        "--includes_tokenizer",
        action="store_true",
        help="Use this flag if there is a Hugging Face tokenizer in the original checkpoint repo",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数将检查点转换为 Hugging Face 格式
    convert_checkpoint_to_huggingface(args.pretrained_checkpoint_path, args.output_path, args.includes_tokenizer)
```