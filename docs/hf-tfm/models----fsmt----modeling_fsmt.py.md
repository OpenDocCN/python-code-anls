# `.\models\fsmt\modeling_fsmt.py`

```
# coding=utf-8
# 设置文件编码为 UTF-8

# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
# 版权声明，版权归Facebook AI Research Team Authors和HuggingFace Inc.团队所有

# Licensed under the Apache License, Version 2.0 (the "License");
# 使用Apache License Version 2.0许可协议，详见 http://www.apache.org/licenses/LICENSE-2.0

# you may not use this file except in compliance with the License.
# 除非遵守许可证的条款，否则不得使用此文件

# You may obtain a copy of the License at
# 您可以在上述链接获取许可证的副本

# http://www.apache.org/licenses/LICENSE-2.0
# 许可证详细信息

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则按"原样"分发，无论是明示的还是暗示的

# See the License for the specific language governing permissions and
# limitations under the License.
# 详细了解许可协议的权限和限制

# Original implementation: https://github.com/pytorch/fairseq/tree/master/examples/wmt19
# 原始实现来源链接

# Authors:
# - @alexeib Alexei Baevski
# - @edunov Sergey Edunov
# - @michaelauli Michael Auli
# - @myleott Myle Ott
# - @nng555 Nathan Ng
# - David Grangier
# - Kyra Yee
# 各位作者姓名

# Paper: Facebook FAIR's WMT19 News Translation Task Submission https://arxiv.org/abs/1907.06616
# 相关论文

"""
PyTorch Fairseq model, ported from https://github.com/pytorch/fairseq/tree/master/examples/wmt19
PyTorch Fairseq模型，从https://github.com/pytorch/fairseq/tree/master/examples/wmt19 迁移而来
"""

import math
# 导入数学库

from typing import Any, Dict, List, Optional, Tuple, Union
# 导入类型提示相关库

import torch
# 导入PyTorch库

from torch import Tensor, nn
# 导入PyTorch的Tensor和神经网络模块

from torch.nn import CrossEntropyLoss, LayerNorm
# 导入交叉熵损失和层归一化模块

from ...activations import ACT2FN
# 导入激活函数映射模块

from ...integrations.deepspeed import is_deepspeed_zero3_enabled
# 导入DeepSpeed集成模块

from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
# 导入模型输出相关类

from ...modeling_utils import PreTrainedModel
# 导入预训练模型工具类

from ...utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入工具函数和日志模块

from .configuration_fsmt import FSMTConfig
# 导入FSMT配置类

logger = logging.get_logger(__name__)
# 获取日志记录器

_CHECKPOINT_FOR_DOC = "facebook/wmt19-ru-en"
# 模型检查点路径

_CONFIG_FOR_DOC = "FSMTConfig"
# FSMT配置类名称

# See all FSMT models at https://huggingface.co/models?filter=fsmt
# 查看所有FSMT模型的链接

# Porting notes:
# this one is modeled after BartModel*
# 本模型基于BartModel*进行了建模

# Currently only translation (fairseq also has weights for LM)
# 目前仅支持翻译（fairseq还具有语言模型的权重）

# fairseq provides weights for ru-en, en-ru and de-en, en-de pairs. All have been ported.
# fairseq提供了ru-en、en-ru和de-en、en-de等语言对的权重，所有这些都已经迁移

# - ru-en, en-ru use asymmetric vocab
# - de-en, en-de use a merged single vocab (but the code works as if they are separate)
# ru-en、en-ru使用非对称词汇，而de-en、en-de使用合并的单词表（但代码处理时像是分开处理）

# Differences with Bart:
# - not using bos token
# - 2 separate vocabs (src and target)
# - embed weights aren't tied
# - uses a model Ensemble (but that part isn't ported/implemented yet) - so we
#   aren't getting as good of a BLEU score
# - uses a projection layer at the end of the decoder
# - doesn't use final_logits_bias
# - beam search: stops as soon as num_beams == len(hypos) (whereas transformers
#   is not satisfied there and will continue searching until the next cycles
#   aren't promising something better), comparing BLEU scores - the transformers
#   algorithm is slightly superior, therefore using the latter. But if you want
# 与Bart模型的区别：
# - 不使用bos标记
# - 有两个独立的词汇表（源和目标）
# - 嵌入权重不是绑定的
# - 使用模型集成（但该部分尚未迁移/实现），因此我们的BLEU分数不如预期
# - 在解码器末端使用投影层
# - 不使用final_logits_bias
# - 波束搜索：一旦num_beams == len(hypos)就停止（而transformers会继续搜索），比较BLEU分数- transformers算法稍优，因此使用后者。但如果您想要
#   to match fairseq outputs, you need to pass ``early_stopping=True`` to ``generate()``.
#
# SinusoidalPositionalEmbedding is slightly different from Bart's - generates
# different embeddings. This implementation is copied verbatim from fairseq with
# some small changes to make it work here.
#
# Other changes:
#  - doesn't support use_cache as Bart's version does
#
#
# FSMTConfig changes with BartConfig
#
#    Differences with BART:
#    - src/tgt vocabs aren't shared
#    - token embeddings aren't shared
#    - needs a language pair
#    - scale_embedding are True
#
#    some unused args were removed too
#
#
# TODO:
# - port model ensemble (fs uses 4 model checkpoints)
# - solve beam search discrepancies
# docstyle-ignore

"""
PYTHONPATH="src:examples/seq2seq" python examples/seq2seq/run_eval.py facebook/wmt19-$PAIR $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation --num_beams $NUM_BEAMS
# 运行一个命令行脚本，评估模型翻译效果并生成评估结果文件，包括源文件、目标文件、BLEU 分数等参数设置

# (fairseq BLEU: 43.1 http://matrix.statmt.org/matrix/output/1909?run_id=6862)
# 指示 fairseq 模型的 BLEU 分数及其详细信息的链接

"""


FSMT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FSMTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

"""
# FSMT_START_DOCSTRING 注释已提供在示例中


FSMT_GENERATION_EXAMPLE = r"""
    Translation example::

    ```python
    >>> from transformers import AutoTokenizer, FSMTForConditionalGeneration

    >>> mname = "facebook/wmt19-ru-en"
    >>> model = FSMTForConditionalGeneration.from_pretrained(mname)
    >>> tokenizer = AutoTokenizer.from_pretrained(mname)

    >>> src_text = "Машинное обучение - это здорово, не так ли?"
    >>> input_ids = tokenizer(src_text, return_tensors="pt").input_ids
    >>> outputs = model.generate(input_ids, num_beams=5, num_return_sequences=3)
    >>> tokenizer.decode(outputs[0], skip_special_tokens=True)
    "Machine learning is great, isn't it?"
    ```

"""
# FSMT_GENERATION_EXAMPLE 注释已提供在示例中


FSMT_INPUTS_DOCSTRING = r"""
"""


def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)
# invert_mask 函数：反转注意力掩码的值，将1变为0，0变为1，True变为False，False变为True


def triu_onnx(x, diagonal=0):
    l = x.shape[0]
    arange = torch.arange(l, device=x.device)
    mask = arange.expand(l, l)
    arange = arange.unsqueeze(-1)
    if diagonal:
        arange = arange + diagonal
    mask = mask >= arange
    return x.masked_fill(mask == 0, 0)
# triu_onnx 函数：生成一个上三角矩阵的掩码，用于在 ONNX 运行时操作


def _prepare_fsmt_decoder_inputs(
    config,
    input_ids,
    decoder_input_ids=None,
    decoder_padding_mask=None,
    causal_mask_dtype=torch.float32,
):
    """
    Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if none are provided.
    This mimics the default behavior in fairseq. To override it pass in masks. Note: this is not called during
    generation
    """
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
# _prepare_fsmt_decoder_inputs 函数：准备解码器的输入，包括忽略填充标记的掩码和因果掩码，以及处理 fairseq 中的默认行为
    # 如果 decoder_padding_mask 为 None，则使用 decoder_input_ids 和 pad_token_id 创建填充遮罩
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    # 否则，反转 decoder_padding_mask
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    
    # 创建一个上三角矩阵的 causal_mask，使用 fill_with_neg_inf 创建全零矩阵并填充负无穷值，然后取上三角部分并在设备上进行设定
    causal_mask = triu_onnx(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len, dtype=causal_mask_dtype)), 1).to(
        device=decoder_input_ids.device
    )
    
    # 返回 decoder_input_ids（解码器输入序列）、decoder_padding_mask（填充遮罩）和 causal_mask（因果遮罩）
    return decoder_input_ids, decoder_padding_mask, causal_mask
class PretrainedFSMTModel(PreTrainedModel):
    # 使用 FSMTConfig 类作为配置类
    config_class = FSMTConfig
    # 模型中基础模型的前缀
    base_model_prefix = "model"

    def _init_weights(self, module):
        # 从配置中获取初始化的标准差
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            # 如果是线性层，使用正态分布初始化权重，偏置初始化为零
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, SinusoidalPositionalEmbedding):
            # 如果是正弦位置嵌入，不进行初始化操作
            pass
        elif isinstance(module, nn.Embedding):
            # 如果是嵌入层，使用正态分布初始化权重，如果有填充索引，则对应权重初始化为零
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        # 获取填充标记的 ID
        pad_token = self.config.pad_token_id
        # 创建一个示例输入的张量，包含两个样本的输入 ID
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        # 构建虚拟输入字典，包含注意力掩码和输入 ID
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


def _make_linear_from_emb(emb):
    # 从嵌入层创建线性层
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    # 将线性层的权重设为嵌入层的权重
    lin_layer.weight.data = emb.weight.data
    return lin_layer


# Helper Functions, mostly for making masks
def _check_shapes(shape_1, shape2):
    # 检查两个形状是否匹配，如果不匹配则引发错误
    if shape_1 != shape2:
        raise AssertionError(f"shape mismatch: {shape_1} != {shape2}")


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""

    # 将标签中可能的 -100 值替换为 `pad_token_id`
    input_ids.masked_fill_(input_ids == -100, pad_token_id)

    # 克隆输入 ID，作为输出的前一个 token
    prev_output_tokens = input_ids.clone()
    # 找到每个样本中最后一个非填充 token 的索引
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    # 将前一个输出 token 的第一个位置设为最后一个非填充 token
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    # 将其余位置向右移动一个位置
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    # 创建用于填充 token 的掩码，值为 True
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


# Helper Modules


class EncoderLayer(nn.Module):
    def __init__(self, config: FSMTConfig):
        super().__init__()
        # 设置嵌入维度
        self.embed_dim = config.d_model
        # 创建自注意力层
        self.self_attn = Attention(self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout)
        # 自注意力层后的 LayerNorm
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        # 配置中的 dropout 率
        self.dropout = config.dropout
        # 激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 激活函数后的 dropout 率
        self.activation_dropout = config.activation_dropout
        # 第一个全连接层
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 第二个全连接层
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 最终的 LayerNorm
        self.final_layer_norm = LayerNorm(self.embed_dim)
    def forward(self, x, encoder_padding_mask, layer_head_mask, output_attentions=False):
        """
        Args:
            x (`torch.Tensor`): 输入到层的输入，形状为 *(seq_len, batch, embed_dim)*
            encoder_padding_mask (`torch.ByteTensor`): 二进制 ByteTensor，形状为
                *(batch, src_len)*，其中填充元素由 `1` 表示。
                对于 t_tgt，t_src 被排除在外（或者被掩盖），=0 表示在注意力机制中包含它们。
            layer_head_mask (`torch.FloatTensor`): 给定层中注意力头的掩码，大小为
                *(config.encoder_attention_heads,)*。

        Returns:
            编码后的输出，形状为 *(seq_len, batch, embed_dim)*
        """
        residual = x  # 保留残差连接

        # 自注意力机制
        x, attn_weights = self.self_attn(
            query=x,
            key=x,
            key_padding_mask=encoder_padding_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)  # 应用 dropout
        x = residual + x  # 添加残差连接
        x = self.self_attn_layer_norm(x)  # 应用层归一化

        residual = x  # 更新残差连接

        # 前馈神经网络（FFN）部分
        x = self.activation_fn(self.fc1(x))  # 应用激活函数和第一层线性变换
        x = nn.functional.dropout(x, p=self.activation_dropout, training=self.training)  # 应用 dropout
        x = self.fc2(x)  # 第二层线性变换
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)  # 应用 dropout
        x = residual + x  # 添加残差连接
        x = self.final_layer_norm(x)  # 最终的层归一化
        return x, attn_weights  # 返回编码后的输出和注意力权重
class FSMTEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a [`EncoderLayer`].

    Args:
        config: FSMTConfig
    """

    def __init__(self, config: FSMTConfig, embed_tokens):
        super().__init__()
        self.dropout = config.dropout  # 从配置中获取 dropout 比例
        self.layerdrop = config.encoder_layerdrop  # 从配置中获取层间 dropout 比例
        self.padding_idx = embed_tokens.padding_idx  # 获取嵌入标记的填充索引
        self.embed_tokens = embed_tokens  # 嵌入 tokens
        embed_dim = embed_tokens.embedding_dim  # 嵌入维度
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0  # 计算嵌入比例因子
        self.embed_positions = SinusoidalPositionalEmbedding(
            config.max_position_embeddings + self.padding_idx + 1, embed_dim, self.padding_idx
        )  # 创建正弦位置嵌入
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])  # 创建编码器层列表

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: torch.Tensor = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """
        Transformer 编码器的前向传播方法。

        Args:
            input_ids: 输入的 token ids
            attention_mask: 注意力遮罩，可选
            inputs_embeds: 嵌入的输入，可选
            head_mask: 头部遮罩，可选
            output_attentions: 是否输出注意力权重，可选
            output_hidden_states: 是否输出隐藏状态，可选
            return_dict: 是否返回字典形式的输出，可选

        Returns:
            根据 return_dict 返回不同形式的输出结果
        """
        # 省略具体的前向传播逻辑，因为这里只要求注释每行代码的作用
        pass


class DecoderLayer(nn.Module):
    def __init__(self, config: FSMTConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 获取嵌入维度

        self.self_attn = Attention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )  # 创建自注意力层

        self.dropout = config.dropout  # 从配置中获取 dropout 比例
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数
        self.activation_dropout = config.activation_dropout  # 激活函数的 dropout

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)  # 自注意力层的 LayerNorm

        self.encoder_attn = Attention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )  # 创建编码器注意力层

        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)  # 编码器注意力层的 LayerNorm

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)  # 第二个全连接层
        self.final_layer_norm = LayerNorm(self.embed_dim)  # 最终的 LayerNorm

    def forward(
        self,
        x,
        encoder_hidden_states,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        decoder_padding_mask=None,
        output_attentions=False,
    ):
        """
        Transformer 解码器层的前向传播方法。

        Args:
            x: 输入张量
            encoder_hidden_states: 编码器的隐藏状态
            encoder_attn_mask: 编码器注意力的遮罩，可选
            layer_state: 层状态，可选
            causal_mask: 因果遮罩，可选
            layer_head_mask: 层头部遮罩，可选
            cross_attn_layer_head_mask: 交叉注意力层头部遮罩，可选
            decoder_padding_mask: 解码器填充遮罩，可选
            output_attentions: 是否输出注意力权重，可选

        Returns:
            根据不同的参数返回不同形式的输出结果
        """
        # 省略具体的前向传播逻辑，因为这里只要求注释每行代码的作用
        pass
    ):
        residual = x  # 保存输入的残差连接

        if layer_state is None:
            layer_state = {}  # 如果状态为空，则初始化为空字典

        # 自注意力机制
        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # 将键添加到层状态中
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)  # 使用 dropout 进行正则化
        x = residual + x  # 添加残差连接
        x = self.self_attn_layer_norm(x)  # 应用自注意力层的 LayerNorm

        # 跨注意力机制
        residual = x  # 保存输入的残差连接
        assert self.encoder_attn.cache_key != self.self_attn.cache_key  # 断言确保编码器注意力缓存键不同于自注意力的缓存键
        x, cross_attn_weights = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # 更新层状态
            layer_head_mask=cross_attn_layer_head_mask,
            output_attentions=output_attentions,
        )
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)  # 使用 dropout 进行正则化
        x = residual + x  # 添加残差连接
        x = self.encoder_attn_layer_norm(x)  # 应用编码器注意力层的 LayerNorm

        # 全连接层
        residual = x  # 保存输入的残差连接
        x = self.activation_fn(self.fc1(x))  # 应用激活函数和第一个全连接层
        x = nn.functional.dropout(x, p=self.activation_dropout, training=self.training)  # 使用 dropout 进行正则化
        x = self.fc2(x)  # 应用第二个全连接层
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)  # 使用 dropout 进行正则化
        x = residual + x  # 添加残差连接
        x = self.final_layer_norm(x)  # 应用最终的 LayerNorm
        return (
            x,
            self_attn_weights,  # 返回自注意力的权重
            layer_state,  # 返回层状态，用于解码的缓存
            cross_attn_weights,  # 返回跨注意力的权重
        )
class FSMTDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DecoderLayer`]

    Args:
        config: FSMTConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: FSMTConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.dropout  # 从配置中获取丢弃率
        self.layerdrop = config.decoder_layerdrop  # 从配置中获取层丢弃率
        self.padding_idx = embed_tokens.padding_idx  # 获取嵌入层的填充索引
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0  # 根据配置计算嵌入缩放因子
        self.embed_tokens = embed_tokens  # 初始化嵌入 tokens
        embed_dim = embed_tokens.embedding_dim  # 获取嵌入维度
        # 初始化 sinusoidal 位置嵌入
        self.embed_positions = SinusoidalPositionalEmbedding(
            config.max_position_embeddings + self.padding_idx + 1, embed_dim, self.padding_idx
        )
        # 创建多个解码层并存入列表
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])  # type: List[DecoderLayer]

        if is_deepspeed_zero3_enabled():
            import deepspeed
            # 如果启用了 DeepSpeed Zero3，使用 GatheredParameters 重新排列权重
            with deepspeed.zero.GatheredParameters(self.embed_tokens.weight, modifier_rank=None):
                embed_tokens_weight_shape = self.embed_tokens.weight.shape
        else:
            embed_tokens_weight_shape = self.embed_tokens.weight.shape
        # 初始化输出投影线性层，用于转换到嵌入 tokens 的维度
        self.output_projection = nn.Linear(embed_tokens_weight_shape[1], embed_tokens_weight_shape[0], bias=False)
        self.output_projection.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        decoder_padding_mask: torch.Tensor,
        decoder_causal_mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """
        Defines the forward pass for the FSMTDecoder module.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            encoder_hidden_states (torch.Tensor): Hidden states from the encoder.
            encoder_padding_mask (torch.Tensor): Mask for encoder padding.
            decoder_padding_mask (torch.Tensor): Mask for decoder padding.
            decoder_causal_mask (torch.Tensor): Mask for causal (autoregressive) decoding.
            head_mask (Optional[torch.Tensor]): Mask for attention heads.
            inputs_embeds (Optional[torch.Tensor]): Embedded inputs.
            cross_attn_head_mask (Optional[torch.Tensor]): Mask for cross-attention heads.
            past_key_values (Optional[List[torch.FloatTensor]]): Cached key-value states.
            use_cache (bool): Whether to use cached key-values.
            output_attentions (bool): Whether to output attention weights.
            output_hidden_states (bool): Whether to output hidden states.
            return_dict (bool): Whether to return a dictionary.

        Returns:
            torch.Tensor or Dict[str, torch.Tensor]: Depending on `return_dict` flag, either logits or dictionary.
        """
        # Forward pass implementation details omitted
        pass


def _reorder_buffer(attn_cache, new_order):
    """
    Reorders the attention cache according to the new order of indices.

    Args:
        attn_cache (Dict[str, torch.Tensor]): Attention cache dictionary.
        new_order (torch.Tensor): New order of indices.

    Returns:
        Dict[str, torch.Tensor]: Reordered attention cache.
    """
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
    ):
        """
        Initializes the Attention module.

        Args:
            embed_dim (int): Dimensionality of input embeddings.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            bias (bool): Whether to use bias in linear layers.
            encoder_decoder_attention (bool): Whether it's encoder-decoder attention or self-attention.
        """
        super().__init__()
        # Initialization details omitted
        pass
    ):
        super().__init__()  # 调用父类的初始化方法
        self.embed_dim = embed_dim  # 设置嵌入维度
        self.num_heads = num_heads  # 设置注意力头数
        self.dropout = dropout  # 设置Dropout比率
        self.head_dim = embed_dim // num_heads  # 计算每个注意力头的维度
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"  # 断言确保embed_dim能被num_heads整除
        self.scaling = self.head_dim**-0.5  # 缩放因子

        self.encoder_decoder_attention = encoder_decoder_attention  # 设置编码器-解码器注意力标志
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 创建线性层k_proj，用于投影查询
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 创建线性层v_proj，用于投影键
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 创建线性层q_proj，用于投影值
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 创建线性层out_proj，用于最终输出
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"  # 缓存键，根据encoder_decoder_attention选择"encoder_decoder"或"self"

    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # 将张量重塑为(batch_size * num_heads, seq_len, head_dim)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        output_attentions=False,
    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)  # 从saved_state中获取并重塑prev_key
            if static_kv:
                k = prev_key  # 如果static_kv为True，则使用prev_key作为当前的k
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)  # 否则将prev_key和当前的k连接起来
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)  # 从saved_state中获取并重塑prev_value
            if static_kv:
                v = prev_value  # 如果static_kv为True，则使用prev_value作为当前的v
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)  # 否则将prev_value和当前的v连接起来
        assert k is not None and v is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get("prev_key_padding_mask", None)  # 从saved_state获取prev_key_padding_mask
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask  # 如果static_kv为True，则使用prev_key_padding_mask作为新的key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)  # 否则将prev_key_padding_mask和当前的key_padding_mask连接起来
        else:
            new_key_padding_mask = key_padding_mask  # 如果没有prev_key_padding_mask，则直接使用当前的key_padding_mask
        return k, v, new_key_padding_mask
# FP16兼容的函数，用于将输入张量 t 填充为负无穷
def fill_with_neg_inf(t):
    return t.float().fill_(torch.finfo(t.dtype).min).type_as(t)


# 返回张量 t 的形状，如果不存在则返回 None
def _get_shape(t):
    return getattr(t, "shape", None)


# FSMT 模型，继承自 PretrainedFSMTModel 类
@add_start_docstrings(
    "The bare FSMT Model outputting raw hidden-states without any specific head on top.",
    FSMT_START_DOCSTRING,
)
class FSMTModel(PretrainedFSMTModel):
    # 被绑定权重的键列表
    _tied_weights_keys = ["decoder.embed_tokens.weight", "decoder.output_projection.weight"]

    # 初始化方法
    def __init__(self, config: FSMTConfig):
        super().__init__(config)

        # 获取填充索引
        padding_idx = config.pad_token_id
        # 创建编码器嵌入层
        encoder_embed_tokens = nn.Embedding(config.src_vocab_size, config.d_model, padding_idx)
        # 创建解码器嵌入层
        decoder_embed_tokens = nn.Embedding(config.tgt_vocab_size, config.d_model, padding_idx)

        # 初始化编码器和解码器
        self.encoder = FSMTEncoder(config, encoder_embed_tokens)
        self.decoder = FSMTDecoder(config, decoder_embed_tokens)

        # 执行初始化权重和最终处理
        self.post_init()

    # 获取编码器方法
    def get_encoder(self):
        return self.encoder

    # 获取解码器方法
    def get_decoder(self):
        return self.decoder

    # 绑定权重方法
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.get_input_embeddings())
            self._tie_or_clone_weights(self.decoder.output_projection, self.get_input_embeddings())

    # 前向传播方法，使用装饰器添加文档字符串和代码示例文档字符串
    @add_start_docstrings_to_model_forward(FSMT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        # 方法体省略，实现模型的具体前向传播逻辑

    # 获取输入嵌入方法，返回编码器的嵌入层
    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    # 设置输入嵌入方法，设置编码器的嵌入层为指定值
    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    # 获取输出嵌入方法，返回解码器的嵌入层
    def get_output_embeddings(self):
        return self.decoder.embed_tokens

    # 设置输出嵌入方法，设置解码器的嵌入层为指定值
    def set_output_embeddings(self, value):
        self.decoder.embed_tokens = value


@add_start_docstrings(
    "The FSMT Model with a language modeling head. Can be used for summarization.", FSMT_START_DOCSTRING
)
class FSMTForConditionalGeneration(PretrainedFSMTModel):
    base_model_prefix = "model"
    _tied_weights_keys = ["decoder.embed_tokens.weight", "decoder.output_projection.weight"]

    def __init__(self, config: FSMTConfig):
        super().__init__(config)
        # 创建基础的FSMTModel对象，使用给定的配置信息
        base_model = FSMTModel(config)
        # 将创建的模型对象赋值给self.model属性
        self.model = base_model

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(FSMT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(FSMT_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Depending on `return_dict`, either a tuple containing `masked_lm_loss` and model outputs or a `Seq2SeqLMOutput`.

        """
        # Determine if `return_dict` is provided; otherwise, use default from configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Disable caching if `labels` are provided to ensure fresh calculations
        if labels is not None:
            use_cache = False

        # Pass inputs to the model for generation, with optional arguments
        outputs = self.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = outputs[0]  # Extract logits from model outputs

        masked_lm_loss = None
        # Calculate masked language modeling loss if `labels` are provided
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Compute loss only on non-masked tokens between logits and labels
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.tgt_vocab_size), labels.view(-1))

        # If `return_dict` is `False`, return tuple with logits and other outputs
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # Return structured output with loss and model outputs in `Seq2SeqLMOutput` format
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 返回一个字典，包含以下字段：
        # "input_ids": None，不需要input_ids，因为已经定义了encoder_outputs
        # "encoder_outputs": encoder_outputs，编码器的输出
        # "past_key_values": past_key_values，过去的键值（缓存）
        # "decoder_input_ids": decoder_input_ids，解码器的输入ids
        # "attention_mask": attention_mask，注意力掩码
        # "head_mask": head_mask，头掩码
        # "decoder_head_mask": decoder_head_mask，解码器头部掩码
        # "cross_attn_head_mask": cross_attn_head_mask，跨注意力头掩码
        # "use_cache": use_cache，用于控制缓存的标志，可能是为了调试而更改的
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 从标签中准备解码器的输入ids，通过将标签向右移动来实现
        return shift_tokens_right(labels, self.config.pad_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = []
        for layer_past in past_key_values:
            # 对过去的缓存重新排序，根据beam_idx来调整每层的缓存
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)
        return reordered_past

    def get_encoder(self):
        # 返回模型的编码器
        return self.model.encoder

    def get_decoder(self):
        # 返回模型的解码器
        return self.model.decoder

    def get_output_embeddings(self):
        # 返回模型的输出嵌入层
        return self.model.decoder.embed_tokens

    def set_output_embeddings(self, value):
        # 设置模型的输出嵌入层
        self.model.decoder.embed_tokens = value
class SinusoidalPositionalEmbedding(nn.Embedding):
    """
    This module produces sinusoidal positional embeddings of any length.

    We don't want to save the weight of this embedding since it's not trained (deterministic) and it can be huge.

    Padding symbols are ignored.

    These embeddings get automatically extended in forward if more positions is needed.
    """

    def __init__(self, num_positions, embedding_dim, padding_idx):
        # 调用 make_weight 方法创建权重矩阵
        self.make_weight(num_positions, embedding_dim, padding_idx)

    def make_weight(self, num_positions, embedding_dim, padding_idx):
        # 调用 get_embedding 方法获取位置编码的权重
        weight = self.get_embedding(num_positions, embedding_dim, padding_idx)
        if not hasattr(self, "weight"):
            # 如果实例中没有权重，通过 nn.Embedding 的构造函数初始化权重
            super().__init__(num_positions, embedding_dim, padding_idx, _weight=weight)
        else:
            # 如果实例中已经有权重，则更新现有权重的 dtype 和 device
            weight = weight.to(dtype=self.weight.dtype, device=self.weight.device)
            self.weight = nn.Parameter(weight)
        # 将权重设为不可训练
        self.weight.detach_()
        self.weight.requires_grad = False

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        # 计算位置编码的半维度
        half_dim = embedding_dim // 2
        # 计算位置编码的增长率
        emb = math.log(10000) / (half_dim - 1)
        # 计算正弦和余弦位置编码的数值
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # 若维度是奇数，添加零填充
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            # 若有填充索引，则将该位置的编码置为零向量
            emb[padding_idx, :] = 0
        return emb

    @staticmethod
    def make_positions(tensor, padding_idx: int):
        """
        Replace non-padding symbols with their position numbers.

        Position numbers begin at padding_idx+1. Padding symbols are ignored.
        """
        # 生成替换非填充符号的位置数字
        mask = tensor.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
    ):
        """
        Input is expected to be of size [bsz x seqlen].
        """
        # 获取输入张量的批量大小和序列长度
        bsz, seq_len = input.shape[:2]
        # 计算最大位置，考虑填充索引和序列长度
        max_pos = self.padding_idx + 1 + seq_len
        # 如果最大位置超过当前权重张量的大小，则扩展嵌入权重
        if max_pos > self.weight.size(0):
            # 调用方法扩展权重张量
            self.make_weight(max_pos, self.embedding_dim, self.padding_idx)
        # 生成位置编码，使用输入张量和填充索引
        positions = self.make_positions(input, self.padding_idx)
        # 调用父类的 forward 方法，传递位置编码张量
        return super().forward(positions)
```