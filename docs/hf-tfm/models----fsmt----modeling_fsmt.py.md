# `.\models\fsmt\modeling_fsmt.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证，版本 2.0 进行许可
# 除非适用法律要求或书面同意，否则本软件按“原样”分发
# 没有任何形式的保证或条件，明示或暗示
# 有关许可的特定语言，请参阅许可证
# 原始实现来源：https://github.com/pytorch/fairseq/tree/master/examples/wmt19
# 作者:
# - @alexeib Alexei Baevski
# - @edunov Sergey Edunov
# - @michaelauli Michael Auli
# - @myleott Myle Ott
# - @nng555 Nathan Ng
# - David Grangier
# - Kyra Yee
#
# 论文: Facebook FAIR's WMT19 News Translation Task Submission https://arxiv.org/abs/1907.06616
#
"""PyTorch Fairseq 模型，从 https://github.com/pytorch/fairseq/tree/master/examples/wmt19 移植过来"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, LayerNorm

# 从深度加速库中导入是否启用深度速度零3的函数
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
# 从模型输出模块中导入基本模型输出类
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
# 从建模工具中导入预训练模型类
from ...modeling_utils import PreTrainedModel
# 从实用工具中导入函数和日志记录器
from ...utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

# 获取记录器实例
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "facebook/wmt19-ru-en"
_CONFIG_FOR_DOC = "FSMTConfig"

# 查看所有 FSMT 模型：https://huggingface.co/models?filter=fsmt

# 移植注释：
# 该模型基于 BartModel*
#
# 目前仅支持翻译（fairseq 还提供了语言模型的权重）
#
# fairseq 提供了 ru-en、en-ru 和 de-en、en-de 对。所有都已经移植。
# - ru-en、en-ru 使用不对称词汇
# - de-en、en-de 使用合并的单个词汇表（但代码按照它们是分开的来运行）
#
# 与 Bart 的不同之处：
# - 不使用 bos 标记
# - 有 2 个单独的词汇表（源和目标）
# - 嵌入权重不是绑定的
# - 使用模型 Ensemble（但该部分尚未移植/实现）- 所以我们
#   得到的 BLEU 分数不是很好
# - 在解码器末端使用了投影层
# - 不使用 final_logits_bias
# - 波束搜索：一旦 num_beams == len(hypos) 就停止（而 transformers
#   在那里不满足条件，将继续搜索直到下一轮不再有更好的结果），
#   比较 BLEU 分数- transformers 算法略优，因此使用后者。但如果你想
#   to match fairseq outputs, you need to pass ``early_stopping=True`` to ``generate()``.
#   为了与 fairseq 的输出匹配，需要在调用 ``generate()`` 时传递参数 ``early_stopping=True``。

# SinusoidalPositionalEmbedding is slightly different from Bart's - generates
# different embeddings. This implementation is copied verbatim from fairseq with
# some small changes to make it work here.
# SinusoidalPositionalEmbedding 稍有不同于 Bart 的版本 - 生成不同的嵌入。这个实现是直接从 fairseq 复制过来的，做了一些小的更改来使其在这里工作。

# Other changes:
#  - doesn't support use_cache as Bart's version does
#  - 不像 Bart 的版本那样支持 use_cache

# FSMTConfig changes with BartConfig
# FSMTConfig 与 BartConfig 有一些不同
#
#    Differences with BART:
#    - src/tgt vocabs aren't shared
#    - token embeddings aren't shared
#    - needs a language pair
#    - scale_embedding are True
#    BART 的不同之处：
#    - src/tgt 词汇表不共享
#    - 标记嵌入不共享
#    - 需要语言对
#    - scale_embedding 为 True

#    some unused args were removed too
#    一些未使用的参数也被删除了

# TODO:
# - port model ensemble (fs uses 4 model checkpoints)
# - solve beam search discrepancies
# 待办： 
# - 移植模型集合（fs 使用4个模型检查点）
# - 解决束搜索的不一致

# docstyle-ignore
# 忽略代码规范
PYTHONPATH="src:examples/seq2seq" python examples/seq2seq/run_eval.py facebook/wmt19-$PAIR $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation --num_beams $NUM_BEAMS

# 使用给定的环境变量 PYTHONPATH 和指定的命令运行评估脚本，对输入的翻译模型进行评估，并生成翻译结果文件和评分文件
# 参数说明：
# - PYTHONPATH: Python 模块搜索路径
# - examples/seq2seq/run_eval.py: 待运行的评估脚本路径
# - facebook/wmt19-$PAIR: 使用的翻译模型名称
# - $DATA_DIR/val.source: 用于验证的源语言文件路径
# - $SAVE_DIR/test_translations.txt: 生成的翻译结果文件保存路径
# - --reference_path $DATA_DIR/val.target: 参考翻译结果文件路径
# - --score_path $SAVE_DIR/test_bleu.json: 生成的 BLEU 评分结果保存路径
# - --bs $BS: batch size，批处理大小
# - --task translation: 任务类型为翻译
# - --num_beams $NUM_BEAMS: Beam search 中的束搜索大小
# (fairseq BLEU: 43.1 http://matrix.statmt.org/matrix/output/1909?run_id=6862)
# 提供的 BLEU 评分为 43.1，链接指向 BLEU 评分的详细信息页面



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



FSMT_INPUTS_DOCSTRING = r"""
"""



def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    # 反转注意力掩码，将 1 变为 0，0 变为 1，True 变为 False，False 变为 True
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)


def triu_onnx(x, diagonal=0):
    l = x.shape[0]
    arange = torch.arange(l, device=x.device)
    mask = arange.expand(l, l)
    arange = arange.unsqueeze(-1)
    if diagonal:
        arange = arange + diagonal
    mask = mask >= arange
    return x.masked_fill(mask == 0, 0)


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
    # 准备解码器输入所需的掩码，忽略解码器中的填充标记，并为解码器准备一个因果掩码，如果没有提供，则使用默认行为。
    # 这模仿了 fairseq 中的默认行为。要覆盖它，请传入掩码。注意：此函数在生成过程中不会被调用。
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    # 如果decoder_padding_mask为空，则利用decoder_input_ids和pad_token_id创建padding mask
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    # 否则，将decoder_padding_mask反转，并且赋值给decoder_padding_mask
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    # 创建causal_mask，利用fill_with_neg_inf函数创建全零矩阵，并对角线以上的元素赋值为负无穷
    # 将其转换为tensor，使用device参数指定在decoder_input_ids设备上
    causal_mask = triu_onnx(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len, dtype=causal_mask_dtype)), 1).to(
        device=decoder_input_ids.device
    )
    # 返回decoder_input_ids, decoder_padding_mask, causal_mask
    return decoder_input_ids, decoder_padding_mask, causal_mask
class PretrainedFSMTModel(PreTrainedModel):
    # 配置类为FSMTConfig
    config_class = FSMTConfig
    # 基础模型前缀为"model"
    base_model_prefix = "model"

    def _init_weights(self, module):
        # 初始化权重
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            # 如果是线性层，使用正态分布初始化权重，并将偏置项置零
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            # 如果是嵌入层，使用正态分布初始化权重，并将填充索引处的权重置零
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        # 生成虚拟输入的dummy_inputs
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


def _make_linear_from_emb(emb):
    # 从嵌入层创建线性层
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


# Helper Functions, mostly for making masks

def _check_shapes(shape_1, shape2):
    # 检查形状是否匹配
    if shape_1 != shape2:
        raise AssertionError(f"shape mismatch: {shape_1} != {shape2}")


def shift_tokens_right(input_ids, pad_token_id):
    # 将输入向右移动一个标记，并包装最后一个非填充标记（通常是eos）
    # 使用pad_token_id替换标签中的可能的-100值
    input_ids.masked_fill_(input_ids == -100, pad_token_id)

    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def make_padding_mask(input_ids, padding_idx=1):
    # 制作填充掩码，为填充标记返回True
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


# Helper Modules

class EncoderLayer(nn.Module):
    def __init__(self, config: FSMTConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Attention(self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
    # 定义前向传播函数，用于对输入数据进行处理
    def forward(self, x, encoder_padding_mask, layer_head_mask, output_attentions=False):
        """
        Args:
            x (`torch.Tensor`): 输入层的输入数据，形状为 *(seq_len, batch, embed_dim)*
            encoder_padding_mask (`torch.ByteTensor`): 二进制 ByteTensor，形状为
                *(batch, src_len)*，其中填充元素由 `1` 表示。
                对于 t_tgt，t_src 被排除（或屏蔽）之外，=0 表示它包含在注意力中
            layer_head_mask (`torch.FloatTensor`): 给定层中注意力头的掩码，大小为
                *(config.encoder_attention_heads,)*。

        Returns:
            编码输出，形状为 *(seq_len, batch, embed_dim)*
        """
        residual = x  # 保存输入数据，用于后面的残差连接
        # 自注意力计算
        x, attn_weights = self.self_attn(
            query=x,
            key=x,
            key_padding_mask=encoder_padding_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对输出进行 dropout
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x  # 残差连接
        x = self.self_attn_layer_norm(x)  # 对输出进行层归一化

        residual = x  # 保存输出数据，用于后面的残差连接
        x = self.activation_fn(self.fc1(x))  # 使用激活函数处理全连接层输出
        x = nn.functional.dropout(x, p=self.activation_dropout, training=self.training)  # 对输出进行 dropout
        x = self.fc2(x)  # 全连接层处理
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)  # 对输出进行 dropout
        x = residual + x  # 残差连接
        x = self.final_layer_norm(x)  # 对输出进行层归一化
        return x, attn_weights  # 返回处理后的数据及注意力权重
class FSMTEncoder(nn.Module):
    """
    FSMT 编码器，由 *config.encoder_layers* 个自注意力层组成的 Transformer 编码器。

    Args:
        config: FSMTConfig
    """

    def __init__(self, config: FSMTConfig, embed_tokens):
        super().__init__()
        self.dropout = config.dropout  # 初始化丢弃率
        self.layerdrop = config.encoder_layerdrop  # 初始化层丢弃率
        self.padding_idx = embed_tokens.padding_idx  # 嵌入标记的填充索引
        self.embed_tokens = embed_tokens  # 嵌入标记
        embed_dim = embed_tokens.embedding_dim  # 嵌入维度
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0  # 嵌入缩放因子
        self.embed_positions = SinusoidalPositionalEmbedding(
            config.max_position_embeddings + self.padding_idx + 1, embed_dim, self.padding_idx
        )  # 初始化正弦位置嵌入
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])  # type: List[EncoderLayer]  # 编码器层列表

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: torch.Tensor = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,



class DecoderLayer(nn.Module):
    def __init__(self, config: FSMTConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 嵌入维度

        self.self_attn = Attention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )  # 自注意力机制
        self.dropout = config.dropout  # 初始化丢弃率
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数
        self.activation_dropout = config.activation_dropout  # 激活函数的丢弃率

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)  # 自注意力层归一化
        self.encoder_attn = Attention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )  # 编码器注意力机制
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)  # 编码器注意力层归一化
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)  # 全连接层1
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)  # 全连接层2
        self.final_layer_norm = LayerNorm(self.embed_dim)  # 最终层归一化

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
        # 保存输入的残差值
        residual = x

        if layer_state is None:
            # 如果层状态为空，则创建一个空字典
            layer_state = {}

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
        # 对输出进行 Dropout
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        # 交叉注意力机制
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        x, cross_attn_weights = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # 修改层状态
            layer_head_mask=cross_attn_layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对输出进行 Dropout
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.encoder_attn_layer_norm(x)

        # 全连接层
        residual = x
        x = self.activation_fn(self.fc1(x))
        # 对输出进行 Dropout
        x = nn.functional.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        # 对输出进行 Dropout
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
            cross_attn_weights,
        )  # layer_state = cache for decoding
# 定义了一个FSMTDecoder类，表示Transformer解码器，由config.decoder_layers层组成，每一层都是一个DecoderLayer
class FSMTDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DecoderLayer`]

    Args:
        config: FSMTConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: FSMTConfig, embed_tokens: nn.Embedding):
        super().__init__()
        # 设置dropout参数
        self.dropout = config.dropout
        # 设置layerdrop参数
        self.layerdrop = config.decoder_layerdrop
        # 获取embed_tokens中的padding_idx
        self.padding_idx = embed_tokens.padding_idx
        # 设置embed_scale参数
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        # 设置embed_tokens参数
        self.embed_tokens = embed_tokens
        # 获取embed_dim
        embed_dim = embed_tokens.embedding_dim
        # 初始化embed_positions
        self.embed_positions = SinusoidalPositionalEmbedding(
            config.max_position_embeddings + self.padding_idx + 1, embed_dim, self.padding_idx
        )
        # 创建包含config.decoder_layers个DecoderLayer对象的列表
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])  # type: List[DecoderLayer]

        # 检查是否启用了deepspeed zero3，如果是，则处理embed_tokens.weight
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.embed_tokens.weight, modifier_rank=None):
                embed_tokens_weight_shape = self.embed_tokens.weight.shape
        else:
            embed_tokens_weight_shape = self.embed_tokens.weight.shape
        # 定义output_projection层，转换输出
        self.output_projection = nn.Linear(embed_tokens_weight_shape[1], embed_tokens_weight_shape[0], bias=False)
        # 将output_projection的权重与embed_tokens的权重相同
        self.output_projection.weight = self.embed_tokens.weight

    # 定义前向传播函数
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
def _reorder_buffer(attn_cache, new_order):
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
    # 初始化方法，设定模型参数
    def __init__(
        self,
        embed_dim, 
        num_heads, 
        dropout, 
        encoder_decoder_attention, 
        bias=True
    ):
        # 继承父类的初始化方法
        super().__init__()
        # 设定嵌入维度
        self.embed_dim = embed_dim
        # 设定头数
        self.num_heads = num_heads
        # 设定dropout率
        self.dropout = dropout
        # 计算每个头的维度
        self.head_dim = embed_dim // num_heads
        # 确保embed_dim必须可以被num_heads整除
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        # 设定缩放因子
        self.scaling = self.head_dim**-0.5

        # 设定是否为编码器-解码器注意力
        self.encoder_decoder_attention = encoder_decoder_attention
        # 初始化线性转换矩阵
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 根据attention类型设定缓存key
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    # 调整输入张量形状
    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    # 前向传播方法
    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        output_attentions=False,
    # 使用保存的状态进行操作，主要是用于解析缓存的键、值和填充蒙版
    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        # 保存的状态采用(bsz, num_heads, seq_len, head_dim)形状存储
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get("prev_key_padding_mask", None)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = key_padding_mask
        return k, v, new_key_padding_mask
# 用负无穷填充输入张量，适用于FP16
def fill_with_neg_inf(t):
    return t.float().fill_(torch.finfo(t.dtype).min).type_as(t)


# Public API
# 获取张量的形状
def _get_shape(t):
    return getattr(t, "shape", None)


# FSMT 模型的基类
@add_start_docstrings(
    "The bare FSMT Model outputting raw hidden-states without any specific head on top.",
    FSMT_START_DOCSTRING,
)
class FSMTModel(PretrainedFSMTModel):
    # 需要共享权重的键
    _tied_weights_keys = ["decoder.embed_tokens.weight", "decoder.output_projection.weight"]

    # 初始化函数
    def __init__(self, config: FSMTConfig):
        super().__init__(config)

        # 创建编码器的词嵌入层和填充标识
        padding_idx = config.pad_token_id
        encoder_embed_tokens = nn.Embedding(config.src_vocab_size, config.d_model, padding_idx)
        decoder_embed_tokens = nn.Embedding(config.tgt_vocab_size, config.d_model, padding_idx)

        # 创建编码器和解码器
        self.encoder = FSMTEncoder(config, encoder_embed_tokens)
        self.decoder = FSMTDecoder(config, decoder_embed_tokens)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.get_input_embeddings())
            self._tie_or_clone_weights(self.decoder.output_projection, self.get_input_embeddings())

    # 前向传播函数
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
    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.decoder.embed_tokens

    def set_output_embeddings(self, value):
        self.decoder.embed_tokens = value


@add_start_docstrings(
    "The FSMT Model with a language modeling head. Can be used for summarization.", FSMT_START_DOCSTRING
)
# 定义一个用于条件生成的FSMT模型，继承自PretrainedFSMTModel类
class FSMTForConditionalGeneration(PretrainedFSMTModel):
    # 基础模型前缀
    base_model_prefix = "model"
    # 绑定权重的关键字列表
    _tied_weights_keys = ["decoder.embed_tokens.weight", "decoder.output_projection.weight"]

    # 初始化方法
    def __init__(self, config: FSMTConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建一个基础FSMT模型
        base_model = FSMTModel(config)
        # 将基础模型赋值给当前对象的model属性
        self.model = base_model

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
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
            返回值为一个元组，包含一个 torch.Tensor 或 Seq2SeqLMOutput 类型的对象

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False

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
        lm_logits = outputs[0]

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 计算masked语言建模损失
            # TODO(SS): do we need to ignore pad tokens in labels?
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.tgt_vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

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
    # 返回一个包含默认值的字典
    ):
        return {
            "input_ids": None,  # 如果已定义encoder_outputs，则不需要input_ids
            "encoder_outputs": encoder_outputs,  # 返回encoder_outputs
            "past_key_values": past_key_values,  # 返回过去的键值
            "decoder_input_ids": decoder_input_ids,  # 返回decoder输入的id
            "attention_mask": attention_mask,  # 返回attention_mask
            "head_mask": head_mask,  # 返回head_mask
            "decoder_head_mask": decoder_head_mask,  # 返回decoder_head_mask
            "cross_attn_head_mask": cross_attn_head_mask,  # 返回cross_attn_head_mask
            "use_cache": use_cache,  # 更改该值以避免缓存（可能是为了调试）
        }

    # 从标签中准备decoder输入id
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id)

    # 重新排序缓存
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = []
        for layer_past in past_key_values:
            # 根据beam_idx重新排序每层的过去键值
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)
        return reordered_past

    # 获取编码器
    def get_encoder(self):
        return self.model.encoder

    # 获取解码器
    def get_decoder(self):
        return self.model.decoder

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.model.decoder.embed_tokens

    # 设置输出嵌入
    def set_output_embeddings(self, value):
        self.model.decoder.embed_tokens = value
class SinusoidalPositionalEmbedding(nn.Embedding):
    """
    This module produces sinusoidal positional embeddings of any length.

    We don't want to save the weight of this embedding since it's not trained (deterministic) and it can be huge.

    Padding symbols are ignored.

    These embeddings get automatically extended in forward if more positions is needed.
    """

    def __init__(self, num_positions, embedding_dim, padding_idx):
        # 调用make_weight方法创建权重
        self.make_weight(num_positions, embedding_dim, padding_idx)

    def make_weight(self, num_positions, embedding_dim, padding_idx):
        # 调用get_embedding方法获取embedding权重
        weight = self.get_embedding(num_positions, embedding_dim, padding_idx)
        
        if not hasattr(self, "weight"):
            # 如果weight属性不存在，则在初始化中调用父类的初始化方法，并传入权重参数_weight
            super().__init__(num_positions, embedding_dim, padding_idx, _weight=weight)
        else:
            # 如果weight属性存在，将新的权重转换成和当前权重相同的dtype和device，并赋值给weight属性
            weight = weight.to(dtype=self.weight.dtype, device=self.weight.device)
            self.weight = nn.Parameter(weight)
        
        # 设置权重为不可训练
        self.weight.detach_()
        self.weight.requires_grad = False

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        # 计算embedding维度的一半
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        # 计算sinusoidal embedding
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        
        # 若embedding维度为奇数，进行零填充
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        
        # 若存在padding_idx，则将对应位置的embedding置为0
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        
        return emb

    @staticmethod
    def make_positions(tensor, padding_idx: int):
        """
        Replace non-padding symbols with their position numbers.

        Position numbers begin at padding_idx+1. Padding symbols are ignored.
        """
        # 使用padding_idx+1作为起始位置，将非padding符号替换为它们的位置数字
        mask = tensor.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
        # 检查输入的大小是否为 [bsz x seqlen]
        """Input is expected to be of size [bsz x seqlen]."""
        # 获取输入的 batch size 和序列长度
        bsz, seq_len = input.shape[:2]
        # 计算最大位置
        max_pos = self.padding_idx + 1 + seq_len
        # 如果最大位置超出当前权重矩阵的大小
        if max_pos > self.weight.size(0):
            # 如果需要，扩展嵌入
            self.make_weight(max_pos, self.embedding_dim, self.padding_idx)
        # 生成位置编码
        positions = self.make_positions(input, self.padding_idx)
        # 调用父类的 forward 方法，传入位置编码进行计算
        return super().forward(positions)
```