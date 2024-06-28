# `.\models\mega\convert_mega_original_pytorch_checkpoint_to_pytorch.py`

```py
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert Mega pretrained checkpoint. Built to convert the Masked LM checkpoint located at
https://huggingface.co/mnaylor/mega-wikitext-103

Requirements:
  - clone the Mega repo and install fairseq from there
    1. git clone https://github.com/facebookresearch/mega.git
    2. cd mega && pip install -e
  - clone the pretrained weights for the original implementation from the hugging face repo
    * use this location as the path for pretrained weights
"""
import argparse  # 导入 argparse 模块，用于命令行参数解析

# utilities to import the model weights and config file
import os  # 导入 os 模块，用于操作文件和目录
import pickle as pkl  # 导入 pickle 模块，用于对象序列化和反序列化

# PyTorch + new model classes
import torch  # 导入 PyTorch 库
from torch import nn  # 从 torch 模块导入 nn 模块，用于构建神经网络

from transformers import AutoTokenizer, MegaConfig, MegaForMaskedLM  # 从 transformers 库导入 AutoTokenizer, MegaConfig 和 MegaForMaskedLM 类

# import the EncoderLayer class used to pretrain
# !! NOTE !! this requires the version of fairseq that is built when you install the Mega source
try:
    from fairseq.modules.mega_layer import MegaEncoderLayer  # 尝试从 fairseq 的模块中导入 MegaEncoderLayer 类
except ImportError:
    raise ImportError("You need to install the version of fairseq from the Mega repo!")  # 如果导入失败，抛出 ImportError 异常，提示用户需要安装来自 Mega 仓库的 fairseq 版本

# define the wrapper classes used to train the MLM  (see colab notebook below)
# https://colab.research.google.com/drive/1qfUO6o5HRdxBblWlw058HVyvaEPhPpH8?usp=sharing
# MegaLM outputs hidden states
class MegaLM(nn.Module):
    "The base class for our Mega encoder - given input IDs, embed text and return encoder output"

    def __init__(self, mega_args, depth, vocab_size):
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        self.mega_args = mega_args  # 保存 mega_args 参数
        self.embedding_layer = nn.Embedding(vocab_size, self.mega_args.encoder_embed_dim)  # 创建词嵌入层，词汇表大小为 vocab_size，嵌入维度为 mega_args.encoder_embed_dim
        self.encoders = nn.ModuleList([MegaEncoderLayer(self.mega_args) for _ in range(depth)])  # 创建一个包含指定深度的 MegaEncoderLayer 的 ModuleList
        self.depth = depth  # 保存深度值
    def forward(self, input_ids, attention_mask, batch_first=True, ignore_mask_value=0):
        """
        Code for a forward pass - expects input_ids and attention_mask to come from a Hugging Face tokenizer as PyTorch
        tensors, and returns a tensor of size (batch, n_classes) containing classification logits

        Other options:
          - batch_first: boolean indicating whether the batch dimension is first in input_ids (default: True, which
            aligns with the HF tokenizer behavior)
          - ignore_mask_value: the value in attention_mask that identifies tokens that should be ignored (default: 0,
            which aligns with HF tokenizer)
        """

        # Mega expects embeddings to be (time, batch, embedding size), but
        # Hugging Face returns tokens as (batch, time)
        # 如果 batch_first 为 True，交换 input_ids 的维度，使其变为 (time, batch)
        if batch_first:
            input_ids = input_ids.T

        # to make things more confusing, Mega expects the attention mask to
        # be (batch, time), but with values of 0 (normal token) and 1 (ignore token)
        # which is the opposite of what HF returns
        # 如果 ignore_mask_value 为 0，将 attention_mask 取反，使其与 Mega 要求的格式一致
        if ignore_mask_value == 0:
            attention_mask = 1 - attention_mask

        # get token embeddings from IDs
        # 通过 embedding_layer 获取 token 的嵌入表示
        embeds = self.embedding_layer(input_ids)

        # pass through the Mega layers
        # 通过 Mega layers 进行处理
        # 输入形状为 (time, batch, encoder dim)，输出形状相同
        for encoder in self.encoders:
            embeds = encoder(embeds, attention_mask)

        # return according to the shape specified
        # 根据 batch_first 的值返回不同形状的 embeds
        if batch_first:
            # (T, B, H) --> (B, T, H)
            return torch.transpose(embeds, 0, 1)
        else:
            return embeds
# 将类名从 MegaForMaskedLM 改名为 OriginalMegaForMaskedLM，以避免与新模块混淆
class OriginalMegaForMaskedLM(nn.Module):
    "A wrapper class for doing masked language modeling with Mega"

    def __init__(self, mega_args, depth, vocab_size):
        super().__init__()
        # 初始化 MegaLM 模型作为成员变量
        self.mega = MegaLM(mega_args, depth, vocab_size)
        # 初始化用于 MLM 的线性层，输入维度为 MegaLM 模型的编码器嵌入维度，输出维度为词汇表大小
        self.mlm_head = nn.Linear(mega_args.encoder_embed_dim, vocab_size)
        # 初始化用于 dropout 的层，丢弃概率为 0.1
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, batch_first=True, ignore_mask_value=0):
        """
        执行 Mega 编码器和 MLM 头部的前向传播。返回每个词汇表条目的 logits。

        如果 `batch_first` 为 True（默认与 Hugging Face tokenizer 的行为一致），输出形状为 (Batch size, Sequence length, Vocab size)；
        否则为 (Sequence length, Batch size, Vocab size)。
        """
        # 调用 MegaLM 模型进行编码器的前向传播
        encoder_output = self.mega(input_ids, attention_mask, batch_first, ignore_mask_value)
        # 对编码器输出施加 dropout 后，通过 MLM 头部得到最终输出
        return self.mlm_head(self.dropout(encoder_output))


# 用于将用户指定位置的检查点转换为 Hugging Face 格式的代码
def convert_checkpoint_to_huggingface(pretrained_checkpoint_path, output_path, includes_tokenizer):
    with open(os.path.join(pretrained_checkpoint_path, "model_args.pkl"), "rb") as f:
        # 从文件中加载 MegaLM 模型的原始参数
        mega_original_args = pkl.load(f)

    # 加载原始的 MegaForMaskedLM 模型，用于检查点转换，设为评估模式
    original_mlm = OriginalMegaForMaskedLM(**mega_original_args).eval()

    # 加载原始模型的权重
    print(
        "Original Mega encoder:",
        # 加载编码器的权重
        original_mlm.mega.load_state_dict(
            torch.load(os.path.join(pretrained_checkpoint_path, "encoder_weights.pt"), map_location="cpu")
        ),
    )
    print(
        "Original Mega MLM layer:",
        # 加载 MLM 头部的权重
        original_mlm.mlm_head.load_state_dict(
            torch.load(os.path.join(pretrained_checkpoint_path, "mlm_head_weights.pt"), map_location="cpu")
        ),
    )

    # 从旧配置文件创建一个新的配置
    # 创建 MegaConfig 对象，用于配置 Mega 模型的参数
    hf_config = MegaConfig(
        num_hidden_layers=mega_original_args["depth"],  # 设置隐藏层的数量
        vocab_size=mega_original_args["vocab_size"],  # 设置词汇表大小
        hidden_size=mega_original_args["mega_args"].encoder_embed_dim,  # 设置隐藏层的尺寸
        shared_representation_size=mega_original_args["mega_args"].encoder_z_dim,  # 设置共享表示的尺寸
        intermediate_size=mega_original_args["mega_args"].encoder_hidden_dim,  # 设置中间层尺寸
        ema_projection_size=mega_original_args["mega_args"].encoder_n_dim,  # 设置EMA投影的尺寸
        dropout_prob=mega_original_args["mega_args"].dropout,  # 设置dropout概率
        attention_probs_dropout_prob=mega_original_args["mega_args"].attention_dropout,  # 设置注意力机制的dropout概率
        hidden_dropout_prob=mega_original_args["mega_args"].hidden_dropout,  # 设置隐藏层的dropout概率
        activation=mega_original_args["mega_args"].activation_fn,  # 设置激活函数类型
        attention_activation=mega_original_args["mega_args"].attention_activation_fn,  # 设置注意力激活函数类型
        bidirectional=mega_original_args["mega_args"].bidirectional,  # 设置是否双向
        use_chunking=mega_original_args["mega_args"].encoder_chunk_size > 0,  # 是否使用分块
        chunk_size=mega_original_args["mega_args"].encoder_chunk_size,  # 设置分块的大小
        truncation=mega_original_args["mega_args"].truncation_length,  # 设置截断长度
        normalization_type=mega_original_args["mega_args"].normalization_type,  # 设置归一化类型
        normalize_before_mega=True,  # 在Mega之前是否进行归一化
        norm_affine=True,  # 归一化的affine参数是否开启
        use_feature_dropout=mega_original_args["mega_args"].feature_dropout,  # 是否使用特征dropout
        relative_positional_bias=mega_original_args["mega_args"].rel_pos_bias,  # 设置相对位置偏置
        max_positions=mega_original_args["mega_args"].max_source_positions,  # 设置最大位置
        nffn_hidden_size=mega_original_args["mega_args"].encoder_ffn_embed_dim,  # 设置NFFN隐藏层大小
        normalize_before_ffn=mega_original_args["mega_args"].normalize_before,  # 在FFN之前是否进行归一化
        # 新增的参数用于HF实现
        nffn_activation_dropout_prob=0.0,  # NFFN激活dropout的概率
        add_token_type_embeddings=False,  # 是否添加token类型的嵌入
        add_lm_hidden_dense_layer=False,  # 是否添加LM隐藏层密集层
    )

    # 创建并评估 MegaForMaskedLM 模型
    hf_mlm = MegaForMaskedLM(hf_config).eval()

    # 修改 hf_mlm 模型的嵌入层，使其与 original_mlm 模型的嵌入层权重相同
    hf_mlm.mega.embedding_layer.word_embeddings.weight = original_mlm.mega.embedding_layer.weight

    # 修改 original_mlm 模型的状态字典，以解决在 Hugging Face 生态系统中可能出现的命名问题，
    # 所有包含"beta"或"gamma"的名称在 _load_pretrained 时将被重命名，同时修正之前可能令人困惑的参数名称
    original_state_dict = original_mlm.mega.encoders.state_dict()
    updated_keys = {}
    # 遍历原始状态字典中的每个模块名称
    for module_name in original_state_dict.keys():
        new_module_name = None
        # 需要单独处理 gamma、beta 和 alpha，因为它们在原始代码库中被多个模块使用；
        # beta 在 EMA、MovingAverageGatedAttention 和 RotaryRelativePositionalBias 中使用，必须由于 flax/tf 权重而重命名
        # EMA 子层原始命名为 "move"，已经重命名为 "ema_gate"，这里进行了相应的处理
        if "beta" in module_name:
            # 如果模块名包含 "move.beta"，则进行替换为 "ema_gate.ema_expansion_matrix"
            if "move.beta" in module_name:
                new_module_name = module_name.replace("move.beta", "ema_gate.ema_expansion_matrix")
            # 如果模块名包含 "mega_layer.beta"，则进行替换为 "qk_bias"
            elif "mega_layer.beta" in module_name:
                new_module_name = module_name.replace("beta", "qk_bias")
            # 否则替换为 "b_param"
            else:
                new_module_name = module_name.replace("beta", "b_param")
        # gamma 在 EMA 和 MovingAverageGatedAttention 中使用，必须由于 flax/tf 权重而重命名
        elif "gamma" in module_name:
            # 如果模块名包含 "move.gamma"，则进行替换为 "ema_gate.kernel_projection_matrix"
            if "move.gamma" in module_name:
                new_module_name = module_name.replace("move.gamma", "ema_gate.kernel_projection_matrix")
            # 如果模块名包含 "mega_layer.gamma"，则进行替换为 "qk_weight"
            elif "mega_layer.gamma" in module_name:
                new_module_name = module_name.replace("gamma", "qk_weight")
            # 否则替换为 "g_param"
            else:
                new_module_name = module_name.replace("gamma", "g_param")
        # alpha 在 EMA 和 positional bias 中使用，重命名以提高可读性
        elif "move.alpha" in module_name:
            new_module_name = module_name.replace("move.alpha", "ema_gate.decay_factor")
        # delta 仅在 EMA 中使用，重命名以提高可读性
        elif "move.delta" in module_name:
            new_module_name = module_name.replace("move.delta", "ema_gate.damping_factor")
        # omega 仅在 EMA 中使用，重命名以提高可读性
        elif "omega" in module_name:
            new_module_name = module_name.replace("move.omega", "ema_gate.residual_weight")

        # 如果有新的模块名，则更新键值对
        if new_module_name:
            updated_keys[module_name] = new_module_name

    # 如果有需要重命名的键值对，则打印需要重命名的键的集合
    if len(updated_keys) != 0:
        print(f"Renaming these keys: {updated_keys.keys()}")
    else:
        print("No need to rename state dict entries")

    # 遍历更新后的键值对，将原始状态字典中的旧键替换为新键
    for old, new in updated_keys.items():
        original_state_dict[new] = original_state_dict.pop(old)

    # 尝试使用更新后的名称加载状态字典
    # 注意，现在称为 `mega.layers` 而不是 `mega.encoders`，因为采用了 hugging face 的风格
    print("HF Mega encoder:", hf_mlm.mega.layers.load_state_dict(original_state_dict))

    # 直接加载 MLM 头部权重
    print(
        "HF Mega MLM layer:",
        hf_mlm.mlm_head.load_state_dict(
            torch.load(os.path.join(pretrained_checkpoint_path, "mlm_head_weights.pt"), map_location="cpu")
        ),
    )

    # 在随机生成的输入序列上进行测试
    # 使用 PyTorch 随机生成整数张量作为模型输入的标识符（token IDs），范围在 [0, hf_config.vocab_size) 内
    input_ids = torch.randint(0, hf_config.vocab_size, size=(4, 256))
    # 创建与 input_ids 相同形状的张量，用于指示哪些标记需要被掩码
    input_mask = torch.ones_like(input_ids)
    # 将最后 10 列的掩码值设置为 0，以确保掩码应用正确 :)
    input_mask[:, -10:] = 0

    # 进行前向传播计算
    original_output = original_mlm(input_ids, input_mask, batch_first=True, ignore_mask_value=0)
    # 使用 Hugging Face 模型进行前向传播计算，返回结果的第一个元素
    hf_output = hf_mlm(input_ids, input_mask)[0]

    # 打印输出张量的形状和它们之间的最大差异
    print(f"original output {original_output.shape}")
    print(f"hf output {hf_output.shape}")
    print(f"max diff: {(original_output - hf_output).max()}")  # 0.0
    # 检查两个输出张量是否在指定的绝对误差容限内近似相等
    success = torch.allclose(original_output, hf_output, atol=1e-3)

    if success:
        # 如果成功匹配，则输出 "Yay!"
        print("Yay!")
        # 将 Hugging Face 模型保存到指定的输出路径中
        hf_mlm.save_pretrained(output_path)
    else:
        # 如果匹配失败，则抛出运行时错误，并输出相关信息
        raise RuntimeError(f"Something's broken :(\nOriginal:\n{original_output}\n\nHF\n{hf_output}\n{hf_mlm}")

    # 如果需要包含 tokenizer
    if includes_tokenizer:
        # 打印信息，表示正在传输 tokenizer
        print("Transferring tokenizer")
        # 从预训练的检查点路径加载自动 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint_path)
        # 将 tokenizer 保存到指定的输出路径中
        tokenizer.save_pretrained(output_path)
if __name__ == "__main__":
    # 如果作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建命令行参数解析器对象

    parser.add_argument(
        "--pretrained_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Point to the directory containing your model weights using the official Mega repo",
    )
    # 添加命令行参数：预训练模型检查点路径，必须提供，用于指定官方 Mega 仓库中模型权重的目录路径

    parser.add_argument(
        "--output_path", default=None, type=str, required=True, help="Location to save the Hugging Face version"
    )
    # 添加命令行参数：输出路径，必须提供，用于指定保存 Hugging Face 版本的位置

    parser.add_argument(
        "--includes_tokenizer",
        action="store_true",
        help="Use this flag if there is a Hugging Face tokenizer in the original checkpoint repo",
    )
    # 添加命令行参数：包含 tokenizer 标志，如果原始检查点仓库中包含 Hugging Face 的 tokenizer，则设置此标志

    args = parser.parse_args()
    # 解析命令行参数，并将其存储在 args 变量中

    convert_checkpoint_to_huggingface(args.pretrained_checkpoint_path, args.output_path, args.includes_tokenizer)
    # 调用函数 convert_checkpoint_to_huggingface，将解析得到的参数传递给该函数进行转换操作


这段代码是一个命令行程序的入口，它使用 argparse 库来解析命令行参数，并调用 `convert_checkpoint_to_huggingface` 函数进行预训练模型检查点的转换工作。
```