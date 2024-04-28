# `.\transformers\models\big_bird\modeling_big_bird.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 Google Research 和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
# PyTorch BigBird 模型

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_big_bird import BigBirdConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点
_CHECKPOINT_FOR_DOC = "google/bigbird-roberta-base"
# 用于文档的配置
_CONFIG_FOR_DOC = "BigBirdConfig"

# BigBird 预训练模型存档列表
BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/bigbird-roberta-base",
    "google/bigbird-roberta-large",
    "google/bigbird-base-trivia-itc",
    # 查看所有 BigBird 模型：https://huggingface.co/models?filter=big_bird
]

# Trivia QA 映射
_TRIVIA_QA_MAPPING = {
    "big_bird_attention": "attention/self",
    "output_layer_norm": "output/LayerNorm",
    "attention_output": "attention/output/dense",
    "output": "output/dense",
    "self_attention_layer_norm": "attention/output/LayerNorm",
    "intermediate": "intermediate/dense",
    "word_embeddings": "bert/embeddings/word_embeddings",
    "position_embedding": "bert/embeddings/position_embeddings",
    "type_embeddings": "bert/embeddings/token_type_embeddings",
    "embeddings": "bert/embeddings",
    "layer_normalization": "output/LayerNorm",
    "layer_norm": "LayerNorm",
    "trivia_qa_head": "qa_classifier",
    "dense": "intermediate/dense",
    "dense_1": "qa_outputs",
}

# 加载 TensorFlow 检查点到 PyTorch 模型中
def load_tf_weights_in_big_bird(model, tf_checkpoint_path, is_trivia_qa=False):
    """Load tf checkpoints in a pytorch model."""
    # 加载 TensorFlow 模型权重到 PyTorch 模型中
    def load_tf_weights_bert(init_vars, tf_path):
        # 初始化空列表用于存储权重名称
        names = []
        # 初始化空字典用于存储 TensorFlow 权重
        tf_weights = {}
    
        # 遍历 TensorFlow 初始化变量
        for name, shape in init_vars:
            # 从 TensorFlow 路径加载变量值
            array = tf.train.load_variable(tf_path, name)
            # 替换变量名称中的一部分以匹配 PyTorch 模型中的结构
            name = name.replace("bert/encoder/LayerNorm", "bert/embeddings/LayerNorm")
            # 记录加载的 TensorFlow 权重的信息
            logger.info(f"Loading TF weight {name} with shape {shape}")
            # 将变量名称添加到列表中
            names.append(name)
            # 将变量名称及其值添加到 TensorFlow 权重字典中
            tf_weights[name] = array
    
        # 返回 TensorFlow 权重的名称列表和字典
        return names, tf_weights
    
    # 加载 TriviaQA 特定的 TensorFlow 模型权重到 PyTorch 模型中
    def load_tf_weights_trivia_qa(init_vars):
        # 初始化空列表用于存储权重名称
        names = []
        # 初始化空字典用于存储 TensorFlow 权重
        tf_weights = {}
    
        # 遍历 TensorFlow 初始化变量
        for i, var in enumerate(init_vars):
            # 将变量名称按斜杠分割
            name_items = var.name.split("/")
    
            # 如果变量名称中包含特定的字符串，则进行一些重命名以匹配 PyTorch 模型中的结构
            if "transformer_scaffold" in name_items[0]:
                layer_name_items = name_items[0].split("_")
                if len(layer_name_items) < 3:
                    layer_name_items += [0]
                name_items[0] = f"bert/encoder/layer_{layer_name_items[2]}"
    
            # 将变量名称中的部分字符串替换为特定的映射值
            name = "/".join([_TRIVIA_QA_MAPPING[x] if x in _TRIVIA_QA_MAPPING else x for x in name_items])[:-2]
    
            # 如果变量名称包含特定的子字符串，则进行一些重命名以匹配 PyTorch 模型中的结构
            if "self/attention/output" in name:
                name = name.replace("self/attention/output", "output")
    
            # 如果是最后两个变量，则进行一些重命名以匹配 PyTorch 模型中的结构
            if i >= len(init_vars) - 2:
                name = name.replace("intermediate", "output")
    
            # 记录加载的 TensorFlow 权重的信息
            logger.info(f"Loading TF weight {name} with shape {var.shape}")
            # 将 TensorFlow 变量的值转换为 NumPy 数组
            array = var.value().numpy()
            # 将变量名称添加到列表中
            names.append(name)
            # 将变量名称及其值添加到 TensorFlow 权重字典中
            tf_weights[name] = array
    
        # 返回 TensorFlow 权重的名称列表和字典
        return names, tf_weights
    
    # 尝试导入所需的 Python 模块，如果导入失败则抛出 ImportError 异常
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    
    # 获取 TensorFlow 模型的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    # 记录转换 TensorFlow 检查点的信息
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    
    # 从 TensorFlow 模型中加载权重
    init_vars = tf.saved_model.load(tf_path).variables if is_trivia_qa else tf.train.list_variables(tf_path)
    
    # 如果加载的 TensorFlow 变量为空，则抛出 ValueError 异常
    if len(init_vars) <= 0:
        raise ValueError("Loaded trained variables cannot be empty.")
    
    # 获取 PyTorch 模型的权重名称列表
    pt_names = list(model.state_dict().keys())
    
    # 根据标志位选择加载 TensorFlow 模型的函数，并获取加载的权重名称列表和字典
    if is_trivia_qa:
        names, tf_weights = load_tf_weights_trivia_qa(init_vars)
    else:
        names, tf_weights = load_tf_weights_bert(init_vars, tf_path)
    
    # 记录未复制到 PyTorch 模型中的权重信息
    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")
    # 记录在 PyTorch 模型中未初始化的权重信息
    logger.info(f"Weights not initialized in PyTorch model: {', '.join(pt_names)}.")
    # 返回 PyTorch 模型
    return model
class BigBirdEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    # 从配置中构建嵌入层，包括词嵌入、位置嵌入和标记类型嵌入

    # 从transformers.models.bert.modeling_bert.BertEmbeddings.__init__复制而来
    def __init__(self, config):
        super().__init__()
        # 词嵌入层，将词索引映射为隐藏状态空间
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 位置嵌入层，将位置索引映射为隐藏状态空间
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 标记类型嵌入层，将标记类型索引映射为隐藏状态空间
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm 沿用 TensorFlow 模型变量名命名风格，以便能够加载任何 TensorFlow 检查点文件
        # 使用 LayerNorm 进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 采用 dropout 进行正则化
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids（1，长度为最大位置嵌入数）在内存中是连续的，并在序列化时被导出
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册 position_ids 缓冲区，扩展为形状（1，最大位置嵌入数），并设置为非持久性缓冲区
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册 token_type_ids 缓冲区，形状与 position_ids 相同，数据类型为长整型，并设置为非持久性缓冲区
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
        # 复制结束

        # 是否对嵌入进行重新缩放
        self.rescale_embeddings = config.rescale_embeddings
        # 隐藏状态空间的大小
        self.hidden_size = config.hidden_size

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    # 检查输入的输入标识是否存在，若存在则获取其形状
    if input_ids is not None:
        input_shape = input_ids.size()
    # 若不存在输入标识，则获取输入嵌入的形状，排除最后一维
    else:
        input_shape = inputs_embeds.size()[:-1]

    # 获取序列长度
    seq_length = input_shape[1]

    # 如果位置标识不存在，则使用模型中已注册的位置标识，该位置标识通常为全零，在跟踪模型时未传递位置标识时，帮助解决问题 #5664
    if position_ids is None:
        position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

    # 如果标记类型标识不存在，则根据模型中已注册的缓冲区设置为全零，通常在自动生成时，注册缓冲区在跟踪模型时未传递标记类型标识时帮助用户，解决问题 #5664
    if token_type_ids is None:
        if hasattr(self, "token_type_ids"):
            # 获取模型中的标记类型标识缓冲区
            buffered_token_type_ids = self.token_type_ids[:, :seq_length]
            # 扩展标记类型标识以匹配输入形状
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
            # 将标记类型标识设置为扩展后的缓冲区
            token_type_ids = buffered_token_type_ids_expanded
        else:
            # 如果模型中未注册标记类型标识，则创建全零的标记类型标识，与输入形状相同
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

    # 如果输入嵌入不存在，则根据输入标识获取单词嵌入
    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)

    # 如果需要重新缩放嵌入，则将嵌入乘以隐藏大小的平方根
    if self.rescale_embeddings:
        inputs_embeds = inputs_embeds * (self.hidden_size**0.5)

    # 获取标记类型嵌入
    token_type_embeddings = self.token_type_embeddings(token_type_ids)

    # 将输入嵌入与标记类型嵌入相加
    embeddings = inputs_embeds + token_type_embeddings

    # 获取位置嵌入
    position_embeddings = self.position_embeddings(position_ids)
    # 将位置嵌入与之前的嵌入相加
    embeddings += position_embeddings

    # 对嵌入进行 dropout 处理
    embeddings = self.dropout(embeddings)
    # 对嵌入进行 LayerNorm 处理
    embeddings = self.LayerNorm(embeddings)
    # 返回嵌入
    return embeddings
class BigBirdSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 检查隐藏层大小是否能被注意力头数整除，如果不能且配置中没有嵌入大小，则引发错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)

        # 创建丢弃层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        # 重塑张量形状以便进行注意力计算
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,



class BigBirdBlockSparseAttention(nn.Module):
    def __init__(self, config, seed=None):
        super().__init__()

        self.max_seqlen = config.max_position_embeddings
        self.seed = seed

        # 检查隐藏层大小是否能被注意力头数整除，如果不能则引发错误
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.num_random_blocks = config.num_random_blocks
        self.block_size = config.block_size

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)

    def transpose_for_scores(self, x):
        # 重塑张量形状以便进行注意力计算
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
        output_attentions=None,
    ):
        # 当前这个 `class` 不能在解码器中使用。

        # 获取隐藏状态的批大小、序列长度以及维度
        batch_size, seqlen, _ = hidden_states.size()
        to_seq_length = from_seq_length = seqlen
        from_block_size = to_block_size = self.block_size

        # 检查查询侧序列长度是否是块大小的倍数
        if from_seq_length % from_block_size != 0:
            raise ValueError("Query sided sequence length must be multiple of block size")

        # 检查键/值侧序列长度是否是块大小的倍数
        if to_seq_length % to_block_size != 0:
            raise ValueError("Key/Value sided sequence length must be multiple of block size")

        # 将隐藏状态经过查询、键、值的线性变换
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 执行 BigBird 稀疏注意力机制
        context_layer, attention_probs = self.bigbird_block_sparse_attention(
            query_layer,
            key_layer,
            value_layer,
            band_mask,
            from_mask,
            to_mask,
            from_blocked_mask,
            to_blocked_mask,
            self.num_attention_heads,
            self.num_random_blocks,
            self.attention_head_size,
            from_block_size,
            to_block_size,
            batch_size,
            from_seq_length,
            to_seq_length,
            seed=self.seed,
            plan_from_length=None,
            plan_num_rand_blocks=None,
            output_attentions=output_attentions,
        )

        # 将上下文层重塑为(batch_size, from_seq_length, -1)的形状
        context_layer = context_layer.contiguous().view(batch_size, from_seq_length, -1)

        # 如果需要输出注意力矩阵，则将注意力矩阵作为元组的一部分返回，否则只返回上下文层
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    @staticmethod
    def torch_bmm_nd(inp_1, inp_2, ndim=None):
        """快速的nd矩阵乘法"""
        # 替代torch.einsum的更快实现（"bhqk,bhkd->bhqd"）
        return torch.bmm(inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:])).view(
            inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 1])
        )

    @staticmethod
    def torch_bmm_nd_transpose(inp_1, inp_2, ndim=None):
        """带转置的快速nd矩阵乘法"""
        # 替代torch.einsum的更快实现（"bhqd,bhkd->bhqk"）
        return torch.bmm(
            inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:]).transpose(1, 2)
        ).view(inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 2]))

    def bigbird_block_sparse_attention(
        self,
        query_layer,
        key_layer,
        value_layer,
        band_mask,
        from_mask,
        to_mask,
        from_blocked_mask,
        to_blocked_mask,
        n_heads,
        n_rand_blocks,
        attention_head_size,
        from_block_size,
        to_block_size,
        batch_size,
        from_seq_len,
        to_seq_len,
        seed,
        plan_from_length,
        plan_num_rand_blocks,
        output_attentions,
    @staticmethod
    def torch_gather_b2(params, indices):
        # 定义一个函数用于在批次维度为2时实现类似于 tf.gather 的操作

        # 检查参数和索引的前两个维度是否相同
        if params.shape[:2] != indices.shape[:2]:
            raise ValueError(
                "Make sure that the first two dimensions of params and indices are identical,                 but"
                f" they are params: {params.shape[:2]} vs. indices: {indices.shape[:2]}"
            )
        # 计算要收集的索引数量
        num_indices_to_gather = indices.shape[-2] * indices.shape[-1]
        # 获取从 params 中选取的索引数量
        num_indices_to_pick_from = params.shape[2]

        # 生成一个用于偏移索引的张量
        shift = torch.arange(indices.shape[0] * indices.shape[1] * num_indices_to_gather, device=indices.device)
        indices_shift = torch.div(shift, num_indices_to_gather, rounding_mode="floor") * num_indices_to_pick_from

        # 展平索引和参数张量
        flattened_indices = indices.view(-1) + indices_shift
        flattened_params = params.reshape(-1, params.shape[-2], params.shape[-1])

        # 使用索引从 params 中选择元素
        out_flattened = flattened_params.index_select(0, flattened_indices)

        # 将结果重新形状为与原参数相同的形状
        out = out_flattened.reshape(params.shape[:2] + (num_indices_to_gather,) + params.shape[3:])
        return out

    @staticmethod
    def _create_rand_mask_from_inputs(
        from_blocked_mask,
        to_blocked_mask,
        rand_attn,
        num_attention_heads,
        num_rand_blocks,
        batch_size,
        from_seq_length,
        from_block_size,
    ):
        """
        Create 3D attention mask from a 2D tensor mask.

        Args:
            from_blocked_mask: 2D Tensor of shape [batch_size,
            from_seq_length//from_block_size, from_block_size].
            to_blocked_mask: int32 Tensor of shape [batch_size,
            to_seq_length//to_block_size, to_block_size].
            rand_attn: [batch_size, num_attention_heads,
            from_seq_length//from_block_size-2, num_rand_blocks]
            num_attention_heads: int. Number of attention heads.
            num_rand_blocks: int. Number of random chunks per row.
            batch_size: int. Batch size for computation.
            from_seq_length: int. length of from sequence.
            from_block_size: int. size of block in from sequence.

        Returns:
            float Tensor of shape [batch_size, num_attention_heads, from_seq_length//from_block_size-2,
            from_block_size, num_rand_blocks*to_block_size].
        """
        # 计算窗口的数量
        num_windows = from_seq_length // from_block_size - 2
        # 从 to_blocked_mask 中选择随机块，构建随机掩码
        rand_mask = torch.stack([p1[i1.flatten()] for p1, i1 in zip(to_blocked_mask, rand_attn)])
        rand_mask = rand_mask.view(batch_size, num_attention_heads, num_windows, num_rand_blocks * from_block_size)
        # 使用 einsum 将两个掩码合并为最终的随机掩码
        rand_mask = torch.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1], rand_mask)
        return rand_mask

    @staticmethod
    # 获取随机注意力分布计划，确定在哪些位置放置随机注意力

    # 参数:
    #     from_seq_length: int。来自序列的长度。
    #     from_block_size: int。来自序列中的块大小。
    #     num_rand_blocks: int。每行的随机块数。

    # 返回:
    #     plan_from_length: from块计划的结束位置
    #     plan_num_rand_blocks: 每个块的随机结束位置数量
    def _get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
        # 存储计划的from序列长度和随机块数
        plan_from_length = []
        plan_num_rand_blocks = []

        # 如果(2 * num_rand_blocks + 5)小于(from_seq_length // from_block_size)，说明还有足够的空间插入随机块
        if (2 * num_rand_blocks + 5) < (from_seq_length // from_block_size):
            # 添加到计划列表中
            plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks)
            # 添加到计划列表中
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(0)
        # 如果(num_rand_blocks + 5)小于(from_seq_length // from_block_size)，说明空间不够插入所有随机块
        elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
            # 添加到计划列表中
            plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks // 2)
            # 添加到计划列表中
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks // 2))
        else:
            # 添加到计划列表中
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks)

        # 返回计划列表
        return plan_from_length, plan_num_rand_blocks

    # 生成BigBird随机掩码
    def _bigbird_block_rand_mask(
        self, from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, last_idx=-1
        """
        创建随机注意力的邻接列表。

        Args:
            from_seq_length: int。源序列的长度。
            to_seq_length: int。目标序列的长度。
            from_block_size: int。源序列中的块大小。
            to_block_size: int。目标序列中的块大小。
            num_rand_blocks: int。每行随机块的数量。
            last_idx: 如果为-1，则从整个目标序列中选择num_rand_blocks个块，
            如果为正数，则仅选择到last_idx处的num_rand_blocks个块。

        Returns:
            大小为from_seq_length//from_block_size-2乘以num_rand_blocks的邻接列表。
        """
        # 当from_seq_length在[1024, 3072, 4096]范围内时使用此方法

        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError("Error the number of blocks needs to be same!")

        rand_attn = np.zeros((from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
        # 在推理（评估）时不使用随机性
        if not self.training:
            return rand_attn
        middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
        last = to_seq_length // to_block_size - 1
        if last_idx > (2 * to_block_size):
            last = (last_idx // to_block_size) - 1

        r = num_rand_blocks  # 缩写
        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i
            if i == 1:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
            elif i == 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
            elif i == from_seq_length // from_block_size - 3:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # 缺少-3：应该切片到last-3处
            elif i == from_seq_length // from_block_size - 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # 缺少-4：应该切片到last-4处
            else:
                if start > last:
                    start = last
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                elif (end + 1) == last:
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                else:
                    rand_attn[i - 1, :] = np.random.permutation(
                        np.concatenate((middle_seq[:start], middle_seq[end + 1 : last]))
                    )[:r]
        return rand_attn
    # 定义一个方法，用于生成带有头部的大鸟块随机掩码
    def _bigbird_block_rand_mask_with_head(
        self,
        from_seq_length,
        to_seq_length,
        from_block_size,
        to_block_size,
        num_heads,
        plan_from_length,
        plan_num_rand_blocks,
        window_block_left=1,
        window_block_right=1,
        global_block_top=1,
        global_block_bottom=1,
        global_block_left=1,
        global_block_right=1,
    ):
    @staticmethod
    # 定义一个静态方法，用于获取单个块的行注意力
    def _get_single_block_row_attention(
        block_id,
        to_start_block_id,
        to_end_block_id,
        num_rand_blocks,
        window_block_left=1,
        window_block_right=1,
        global_block_left=1,
        global_block_right=1,
    ):
        """
        对于单个行块，获取随机行注意力。

        Args:
            block_id: int. 行的块ID。
            to_start_block_id: int. 随机注意力列的起始ID。
            to_end_block_id: int. 随机注意力列的结束ID。
            num_rand_blocks: int. 要选择的随机块数。
            window_block_left: int. 块左侧窗口中的块数。
            window_block_right: int. 块右侧窗口中的块数。
            global_block_left: int. 左侧全局使用的块数。
            global_block_right: int. 右侧全局使用的块数。

        Returns:
            包含大小为num_rand_blocks的随机注意力向量的行。
        """
        # 生成要选择随机注意力的to_blocks列表
        to_block_list = np.arange(to_start_block_id, to_end_block_id, dtype=np.int32)
        # 对块进行排列
        perm_block = np.random.permutation(to_block_list)

        # 当前块ID的非法块，使用窗口
        illegal_blocks = list(range(block_id - window_block_left, block_id + window_block_right + 1))

        # 在开头和结尾添加块
        illegal_blocks.extend(list(range(global_block_left)))
        illegal_blocks.extend(list(range(to_end_block_id - global_block_right, to_end_block_id)))

        # 第二个from_block不能在倒数第二个to_block上选择随机注意力
        if block_id == 1:
            illegal_blocks.append(to_end_block_id - 2)

        # 倒数第二个from_block不能在第二个to_block上选择随机注意力
        if block_id == to_end_block_id - 2:
            illegal_blocks.append(1)

        selected_random_blokcs = []

        for i in range(to_end_block_id - to_start_block_id):
            if perm_block[i] not in illegal_blocks:
                selected_random_blokcs.append(perm_block[i])
            if len(selected_random_blokcs) == num_rand_blocks:
                break
        return np.array(selected_random_blokcs, dtype=np.int32)
# 从transformers.models.bert.modeling_bert.BertSelfOutput复制代码，并将Bert->BigBird
class BigBirdSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入和输出维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个LayerNorm层，输入维度是config.hidden_size，eps为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将hidden_states输入到全连接层中
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行Dropout
        hidden_states = self.dropout(hidden_states)
        # 对Dropout后的输出进行LayerNorm，并与input_tensor相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BigBirdAttention(nn.Module):
    def __init__(self, config, seed=None):
        super().__init__()
        # 初始化注意力类型、配置和种子
        self.attention_type = config.attention_type
        self.config = config
        self.seed = seed

        # 根据注意力类型选择不同的注意力机制
        if self.config.attention_type == "original_full":
            self.self = BigBirdSelfAttention(config)
        elif self.config.attention_type == "block_sparse":
            self.self = BigBirdBlockSparseAttention(config, seed)
        else:
            raise ValueError(
                f"attention_type can either be original_full or block_sparse, but is {self.config.attention_type}"
            )

        # 初始化输出层
        self.output = BigBirdSelfOutput(config)

    def set_attention_type(self, value: str):
        # 设置注意力类型，只能是"original_full"或"block_sparse"
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # 如果注意力类型已经正确设置，则直接返回
        if value == self.attention_type:
            return

        # 根据新的注意力类型创建新的注意力权重
        self.attention_type = value
        if value == "original_full":
            # 复制所有权重到新的全注意力类
            attn_weights = BigBirdSelfAttention(self.config)
        else:
            # 复制所有权重���新的稀疏注意力类
            attn_weights = BigBirdBlockSparseAttention(self.config, self.seed)

        # 复制权重
        attn_weights.query = self.self.query
        attn_weights.value = self.self.value
        attn_weights.key = self.self.key
        self.self = attn_weights
        self.attention_type = value
        if not self.training:
            self.self.eval()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        # block_sparse config
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
        # fp16 compatibility
        # 如果存在 band_mask，则将其转换为 hidden_states 的数据类型
        if band_mask is not None:
            band_mask = band_mask.to(hidden_states.dtype)
        # 如果存在 from_mask，则将其转换为 hidden_states 的数据类型
        if from_mask is not None:
            from_mask = from_mask.to(hidden_states.dtype)
        # 如果存在 to_mask，则将其转换为 hidden_states 的数据类型
        if to_mask is not None:
            to_mask = to_mask.to(hidden_states.dtype)
        # 如果 attention_type 为 "original_full"，则调用 self.self 方法
        if self.attention_type == "original_full":
            self_outputs = self.self(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
        else:
            # 如果 encoder_hidden_states 不为 None，则抛出异常
            if encoder_hidden_states is not None:
                raise ValueError("BigBird cannot be used as a decoder when config.attention_type != 'original_full'")
            # 调用 self.self 方法
            self_outputs = self.self(
                hidden_states, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, output_attentions
            )

        # 将 self_outputs[0] 和 hidden_states 作为参数传递给 self.output 方法，得到 attention_output
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果输出了 attentions，则将 attentions 添加到 outputs 中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # 返回 outputs
        return outputs
# 从transformers.models.bert.modeling_bert.BertIntermediate复制并将Bert->BigBird
class BigBirdIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入维度为config.hidden_size，输出维度为config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果config.hidden_act是字符串类型，则使用ACT2FN字典中对应的激活函数，否则直接使用config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入hidden_states通过全连接层dense
        hidden_states = self.dense(hidden_states)
        # 将经过全连接层的结果通过激活函数处理
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOutput复制并将Bert->BigBird
class BigBirdOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入维度为config.intermediate_size，输出维度为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建LayerNorm层，输入维度为config.hidden_size，eps为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建Dropout层，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入hidden_states通过全连接层dense
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行Dropout处理
        hidden_states = self.dropout(hidden_states)
        # 将Dropout后的结果与输入tensor相加，然后通过LayerNorm层
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BigBirdLayer(nn.Module):
    def __init__(self, config, seed=None):
        super().__init__()
        self.config = config
        self.attention_type = config.attention_type
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 创建BigBirdAttention层
        self.attention = BigBirdAttention(config, seed=seed)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise TypeError(f"{self} should be used as a decoder model if cross attention is added")
            # 如果添加了跨层Attention，则创建BigBirdAttention层
            self.crossattention = BigBirdAttention(config)
        # 创建BigBirdIntermediate层
        self.intermediate = BigBirdIntermediate(config)
        # 创建BigBirdOutput层
        self.output = BigBirdOutput(config)

    def set_attention_type(self, value: str):
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # 如果value与当前attention_type相同，则直接返回
        if value == self.attention_type:
            return
        # 更新attention_type，并设置对应的Attention类型
        self.attention_type = value
        self.attention.set_attention_type(value)

        if self.add_cross_attention:
            self.crossattention.set_attention_type(value)
    # 前向传播函数，用于模型的前向推理过程
    def forward(
        # 隐藏状态，即模型中各层的输出
        self,
        # 注意力掩码，用于指定哪些位置需要被忽略
        attention_mask=None,
        # 头部掩码，用于指定哪些注意力头部需要被屏蔽
        head_mask=None,
        # 编码器隐藏状态，用于注意力机制中的信息交互
        encoder_hidden_states=None,
        # 编码器注意力掩码，用于指定编码器中哪些位置需要被忽略
        encoder_attention_mask=None,
        # 带状掩码，用于自注意力机制中的信息交互
        band_mask=None,
        # 来源掩码，用于指定输入中哪些位置需要被忽略
        from_mask=None,
        # 目标掩码，用于指定输出中哪些位置需要被忽略
        to_mask=None,
        # 编码器阻塞掩码，用于指定编码器中哪些位置需要被忽略
        blocked_encoder_mask=None,
        # 过去的键值对，用于存储上一步的键值对信息以供后续步骤使用
        past_key_value=None,
        # 是否输出注意力权重，默认为 False
        output_attentions=False,
        # 如果之前有缓存的键/值元组，则decoder单向自注意力缓存的键/值元组在位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
            band_mask=band_mask,
            from_mask=from_mask,
            to_mask=to_mask,
            from_blocked_mask=blocked_encoder_mask,
            to_blocked_mask=blocked_encoder_mask,
        )
        # 获取自注意力计算的输出
        attention_output = self_attention_outputs[0]

        # 如果是decoder，最后的输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果我们输出注意力权重，添加自注意力
        

        cross_attn_present_key_value = None
        # 如果是decoder且有encoder_hidden_states
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果没有交叉注意力层，抛出错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with                    "
                    " cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # 交叉注意力缓存的键/值元组在past_key_value元组的位置3,4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 进行交叉注意力计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力计算的输出
            attention_output = cross_attention_outputs[0]
            # 如果我们输出注意力权重，添加交叉注意力
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力缓存添加到present_key_value元组的位置3,4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 对层输出进行分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        outputs = (layer_output,) + outputs

        # 如果是decoder，则将注意力键/值作为最后的输出
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        # 返回输出
        return outputs
```  
    # 执行神经网络的前向传播，处理注意力输出
    def feed_forward_chunk(self, attention_output):
        # 使用中间层处理注意力输出，得到中间结果
        intermediate_output = self.intermediate(attention_output)
        # 使用输出层处理中间结果和注意力输出，得到最终层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回最终层输出
        return layer_output
# 定义一个 BigBirdEncoder 类，继承自 nn.Module 类
class BigBirdEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化 BigBirdEncoder 的配置
        self.config = config
        # 初始化注意力机制类型
        self.attention_type = config.attention_type

        # 创建包含多个 BigBirdLayer 的列表，并设置其注意力种类
        self.layer = nn.ModuleList(
            [BigBirdLayer(config, seed=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 设置是否启用梯度检查点
        self.gradient_checkpointing = False

    # 设置注意力机制类型
    def set_attention_type(self, value: str):
        # 如果传入的值不是 "original_full" 或 "block_sparse"，则引发 ValueError
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # 如果当前注意力机制类型已经是要设置的类型，则直接返回，无需修改
        if value == self.attention_type:
            return
        # 更新当前注意力机制类型，并将每个 BigBirdLayer 的注意力机制类型设置为相同
        self.attention_type = value
        for layer in self.layer:
            layer.set_attention_type(value)

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        blocked_encoder_mask=None,
        return_dict=True,
```  


# 定义一个 BigBirdPredictionHeadTransform 类，继承自 nn.Module 类
class BigBirdPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，用于转换隐藏状态的维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果配置中的隐藏激活函数是字符串，则从预定义的字典中获取对应的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 创建 LayerNorm 层，用于归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 经过线性转换
        hidden_states = self.dense(hidden_states)
        # 经过激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 经过 LayerNorm 归一化
        hidden_states = self.LayerNorm(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
```  


# 定义一个 BigBirdLMPredictionHead 类，继承自 nn.Module 类
class BigBirdLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个 BigBirdPredictionHeadTransform 实例，用于隐藏状态的转换
        self.transform = BigBirdPredictionHeadTransform(config)

        # 创建一个线性层，用于将隐藏状态映射到词汇表大小的空间
        # 注意：这里的权重与输入嵌入层共享，但每个标记有一个独立的偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 创建一个偏置参数，大小与词汇表相同
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 链接偏置参数和线性层的偏置，以便在调整词嵌入大小时正确调整偏置大小
        self.decoder.bias = self.bias

    # 前向传播函数
    def forward(self, hidden_states):
        # 先经过隐藏状态转换
        hidden_states = self.transform(hidden_states)
        # 再经过线性层得到预测结果
        hidden_states = self.decoder(hidden_states)
        # 返回预测结果
        return hidden_states
```  
# 从 transformers.models.bert.modeling_bert.BertOnlyMLMHead 复制并修改为 BigBirdOnlyMLMHead 类
class BigBirdOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建 BigBirdLMPredictionHead 对象
        self.predictions = BigBirdLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 对序列输出进行 MLM 预测
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# 从 transformers.models.bert.modeling_bert.BertOnlyNSPHead 复制并修改为 BigBirdOnlyNSPHead 类
class BigBirdOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建线性层用于 NSP 预测
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        # 对汇总输出进行 NSP 预测
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


# 从 transformers.models.bert.modeling_bert.BertPreTrainingHeads 复制并修改为 BigBirdPreTrainingHeads 类
class BigBirdPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建 BigBirdLMPredictionHead 和线性层用于 NSP 预测
        self.predictions = BigBirdLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        # 对序列输出进行 MLM 预测和汇总输出进行 NSP 预测
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


# BigBirdPreTrainedModel 类，用于处理权重初始化和下载预训练模型
class BigBirdPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，用于处理权重初始化和简单接口用于下载和加载预训练模型。
    """

    # BigBirdConfig 类
    config_class = BigBirdConfig
    # 加载 TensorFlow 权重的函数
    load_tf_weights = load_tf_weights_in_big_bird
    # 基础模型前缀
    base_model_prefix = "bert"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 稍微不同于 TF 版本，使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# BigBird 模型的文档字符串
BIG_BIRD_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
    Parameters:
        config ([`BigBirdConfig`]): 模型配置类，包含模型的所有参数。
            使用配置文件初始化不会加载与模型关联的权重，只加载配置。查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

# 大鸟预训练模型输入的文档字符串，用于描述输入参数和其形状
BIG_BIRD_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            输入序列标记在词汇表中的索引。
            可以使用 [`AutoTokenizer`] 获得索引。详情参见 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            避免对填充标记索引执行注意力计算的掩码。掩码值在 `[0, 1]` 之间：

            - 对于**未掩码**的标记，值为 1，
            - 对于**已掩码**的标记，值为 0。

            [什么是注意力掩码？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            指示输入的第一部分和第二部分的段标记索引。索引值在 `[0, 1]` 之间：

            - 0 对应*句子 A* 的标记，
            - 1 对应*句子 B* 的标记。

            [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            每个输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.max_position_embeddings - 1]`。

            [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            用于屏蔽自注意力模块的选择头部的掩码。掩码值在 `[0, 1]` 之间：

            - 1 表示头部**未被屏蔽**，
            - 0 表示头部**已被屏蔽**。

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            可选地，您可以选择直接传递嵌入表示而不是传递 `input_ids`。如果您想要对*input_ids*索引如何转换为关联向量
            有更多控制权，则此选项很有用，而不是使用模型的内部嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多细节，请参见返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多细节，请参见返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""


# 大鸟预训练模型的输出类型
@dataclass
class BigBirdForPreTrainingOutput(ModelOutput):
    """
    [`BigBirdForPreTraining`] 的输出类型。
    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义 loss 变量，类型为 torch.FloatTensor，可选参数，默认为 None
    loss: Optional[torch.FloatTensor] = None
    # 定义 prediction_logits 变量，类型为 torch.FloatTensor，默认为 None
    prediction_logits: torch.FloatTensor = None
    # 定义 seq_relationship_logits 变量，类型为 torch.FloatTensor，默认为 None
    seq_relationship_logits: torch.FloatTensor = None
    # 定义 hidden_states 变量，类型为 Tuple[torch.FloatTensor]，可选参数，默认为 None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义 attentions 变量，类型为 Tuple[torch.FloatTensor]，可选参数，默认为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class BigBirdForQuestionAnsweringModelOutput(ModelOutput):
    """
    问题回答模型输出的基类。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *可选*, 当提供 `labels` 时返回):
            总的跨度抽取损失是开始和结束位置的交叉熵之和。
        start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            起始位置的分数（在 SoftMax 之前）。
        end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            结束位置的分数（在 SoftMax 之前）。
        pooler_output (`torch.FloatTensor` of shape `(batch_size, 1)`):
            来自 BigBigModel 的汇集器输出。
        hidden_states (`tuple(torch.FloatTensor)`, *可选*, 当传递 `output_hidden_states=True` 或者 `config.output_hidden_states=True` 时返回):
            `torch.FloatTensor` 元组（一个用于嵌入的输出 + 一个用于每个层的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。

            每个层输出的隐藏状态加上初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *可选*, 当传递 `output_attentions=True` 或者 `config.output_attentions=True` 时返回):
            `torch.FloatTensor` 元组（每个层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            在注意力 SoftMax 之后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@add_start_docstrings(
    "不带特定头部的 BigBird 模型变压器的裸输出。",
    BIG_BIRD_START_DOCSTRING,
)
class BigBirdModel(BigBirdPreTrainedModel):
    """

    该模型既可以作为编码器（仅具有自注意力）也可以作为解码器，此时在自注意力层之间添加了一层交叉注意力，
    遵循[Attention is all you need](https://arxiv.org/abs/1706.03762)一文中描述的架构，
    该文由 Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N. Gomez、Lukasz Kaiser 和 Illia Polosukhin 撰写。

    要作为解码器行为，模型需要使用 `is_decoder` 参数设置为 `True` 进行初始化。要在 Seq2Seq 模型中使用，
    模型需要初始化时将 `is_decoder` 参数和 `add_cross_attention` 设置为 `True`；
    此时预期将 `encoder_hidden_states` 作为输入传递给前向传递。
    """
    # 初始化模型，设置参数和层
    def __init__(self, config, add_pooling_layer=True):
        # 调用父类的初始化方法
        super().__init__(config)
        # 从配置中获取注意力类型和配置信息
        self.attention_type = self.config.attention_type
        self.config = config

        # 设置块大小
        self.block_size = self.config.block_size

        # 初始化嵌入层和编码器
        self.embeddings = BigBirdEmbeddings(config)
        self.encoder = BigBirdEncoder(config)

        # 如果需要添加池化层，则初始化线性层和激活函数
        if add_pooling_layer:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.activation = nn.Tanh()
        else:
            self.pooler = None
            self.activation = None

        # 如果注意力类型不是"original_full"且配置中添加了交叉注意力，则警告并设置注意力类型为"original_full"
        if self.attention_type != "original_full" and config.add_cross_attention:
            logger.warning(
                "When using `BigBirdForCausalLM` as decoder, then `attention_type` must be `original_full`. Setting"
                " `attention_type=original_full`"
            )
            self.set_attention_type("original_full")

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 设置注意力类型
    def set_attention_type(self, value: str):
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # 如果注意力类型已经正确设置，则直接返回
        if value == self.attention_type:
            return
        self.attention_type = value
        self.encoder.set_attention_type(value)

    # 前向传播方法，接受多个输入参数
    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    @staticmethod
    def create_masks_for_block_sparse_attn(attention_mask: torch.Tensor, block_size: int):
        # 获取批量大小和序列长度
        batch_size, seq_length = attention_mask.size()
        # 如果序列长度不是块大小的倍数，则引发值错误
        if seq_length % block_size != 0:
            raise ValueError(
                f"Sequence length must be multiple of block size, but sequence length is {seq_length}, while block"
                f" size is {block_size}."
            )

        def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
            """
            Create 3D attention mask from a 2D tensor mask.

            Args:
                from_blocked_mask: 2D Tensor of shape [batch_size,
                from_seq_length//from_block_size, from_block_size].
                to_blocked_mask: int32 Tensor of shape [batch_size,
                to_seq_length//to_block_size, to_block_size].

            Returns:
                float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4, from_block_size,
                3*to_block_size].
            """
            # 将to_blocked_mask的三个副本拼接在一起以进行后续操作
            exp_blocked_to_pad = torch.cat(
                [to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]], dim=2
            )
            # 通过乘法操作创建带状掩码
            band_mask = torch.einsum("blq,blk->blqk", from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
            # 在第二个维度上添加一个维度
            band_mask.unsqueeze_(1)
            return band_mask

        # 将注意力掩码变形为块形式
        blocked_encoder_mask = attention_mask.view(batch_size, seq_length // block_size, block_size)
        # 创建带状掩码
        band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)

        # 重新变形注意力掩码以匹配需要的形状
        from_mask = attention_mask.view(batch_size, 1, seq_length, 1)
        to_mask = attention_mask.view(batch_size, 1, 1, seq_length)

        return blocked_encoder_mask, band_mask, from_mask, to_mask

    def _pad_to_block_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        pad_token_id: int,
        """A helper function to pad tokens and mask to work with implementation of BigBird block-sparse attention."""
        # 定义一个辅助函数，用于填充标记和掩码，以便与 BigBird 块稀疏注意力的实现配合使用

        # 获取配置中的块大小
        block_size = self.config.block_size

        # 获取输入的形状
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        # 计算需要填充的长度
        padding_len = (block_size - seq_len % block_size) % block_size
        if padding_len > 0:
            # 如果需要填充，则发出警告并进行填充
            logger.warning_once(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.block_size`: {block_size}"
            )
            if input_ids is not None:
                # 对输入标记进行填充
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                # 使用 pad_token_id 对位置标记进行填充，与 modeling_bigbird.BigBirdEmbeddings 中的操作一致
                position_ids = nn.functional.pad(position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                # 创建与填充长度相同的输入标记填充
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config.pad_token_id,
                    dtype=torch.long,
                )
                # 使用填充的输入标记创建填充的嵌入
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                # 将填充的嵌入连接到原始嵌入中
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            # 对注意力掩码进行填充，填充部分不参与注意力计算
            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=False
            )
            # 对标记类型进行填充，填充部分的类型为 0
            token_type_ids = nn.functional.pad(token_type_ids, (0, padding_len), value=0)

        # 返回填充长度和填充后的输入标记、注意力掩码、标记类型、位置标记和输入嵌入
        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds
class BigBirdForPreTraining(BigBirdPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BigBirdModel(config, add_pooling_layer=True)
        self.cls = BigBirdPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BigBirdForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        next_sentence_label: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings("""BigBird Model with a `language modeling` head on top.""", BIG_BIRD_START_DOCSTRING)
class BigBirdForMaskedLM(BigBirdPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BigBirdForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BigBirdModel(config)
        self.cls = BigBirdOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法用于模型的前向传播
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的 token ID
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码
        token_type_ids: Optional[torch.LongTensor] = None,  # token 类型 ID
        position_ids: Optional[torch.LongTensor] = None,  # 位置 ID
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # 编码器隐藏状态
        encoder_attention_mask: Optional[torch.FloatTensor] = None,  # 编码器的注意力掩码
        labels: Optional[torch.LongTensor] = None,  # 标签
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
    # 为生成准备输入的方法
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        # 获取输入的形状
        input_shape = input_ids.shape
        # 获取有效的批次大小
        effective_batch_size = input_shape[0]

        # 添加一个虚拟 token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")
        # 在注意力掩码的最后添加一个全零向量
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        # 创建一个全为 PAD token ID 的虚拟 token
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        # 在输入的 token ID 后面添加虚拟 token
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        # 返回包含输入 token ID 和注意力掩码的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask}
# 添加起始文档字符串，说明 BigBird 模型是带有语言建模头的，用于 CausalLM 微调
@add_start_docstrings(
    """BigBird Model with a `language modeling` head on top for CLM fine-tuning.""", BIG_BIRD_START_DOCSTRING
)
# 定义 BigBirdForCausalLM 类，继承自 BigBirdPreTrainedModel
class BigBirdForCausalLM(BigBirdPreTrainedModel):
    # 定义 _tied_weights_keys 列表，用于表示权重共享的键名
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    # 初始化方法
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)

        # 如果配置不是解码器，则发出警告
        if not config.is_decoder:
            logger.warning("If you want to use `BigBirdForCausalLM` as a standalone, add `is_decoder=True.`")

        # 初始化 BigBirdModel 和 BigBirdOnlyMLMHead
        self.bert = BigBirdModel(config)
        self.cls = BigBirdOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播方法，用于模型推理
    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 前向传播逻辑在此处实现，用于模型推理
        pass

    # 为生成准备输入的方法，用于在生成时预处理输入
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # 获取输入形状
        input_shape = input_ids.shape

        # 如果没有提供注意力遮罩，则创建全1的遮罩
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果使用了过去键值对，则修剪输入的解码器输入序列
        if past_key_values is not None:
            # 获取过去键值对的长度
            past_length = past_key_values[0][0].shape[2]

            # 有些生成方法已经只传递了最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认使用旧的行为：只保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 修剪输入序列
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回准备好的输入字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}
    # 重新排列缓存，以适应束搜索的索引
    def _reorder_cache(self, past_key_values, beam_idx):
        # 初始化重新排列后的缓存
        reordered_past = ()
        # 遍历过去的键值对
        for layer_past in past_key_values:
            # 重新排列每一层的过去状态
            reordered_past += (
                # 对于每一层的过去状态的前两个元素，根据束搜索的索引重新排序，并转移到相同设备上
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                # 保留第三个元素以后的状态
                + layer_past[2:],
            )
        # 返回重新排列后的过去状态
        return reordered_past
class BigBirdClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入大小为隐藏层大小，输出大小为隐藏层大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果分类器的丢弃率不为空，则使用分类器的丢弃率，否则使用隐藏层的丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义一个丢弃层，使用分类器的丢弃率
        self.dropout = nn.Dropout(classifier_dropout)
        # 定义一个全连接层，输入大小为隐藏层大小，输出大小为标签数量
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.config = config

    def forward(self, features, **kwargs):
        # 取出特征的第一个 token，即<CLS> token
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # 对取出的 token 进行丢弃操作
        x = self.dropout(x)
        # 通过全连接层处理 token
        x = self.dense(x)
        # 使用激活函数处理全连接层的输出
        x = ACT2FN[self.config.hidden_act](x)
        # 再次对处理后的 token 进行丢弃操作
        x = self.dropout(x)
        # 通过输出全连接层得到最终结果
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    BigBird Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    BIG_BIRD_START_DOCSTRING,
)
class BigBirdForSequenceClassification(BigBirdPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 标签数量
        self.num_labels = config.num_labels
        self.config = config
        # 初始化 BigBird 模型
        self.bert = BigBirdModel(config)
        # 初始化分类头部
        self.classifier = BigBirdClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 函数参数



@add_start_docstrings(
    """
    BigBird Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    BIG_BIRD_START_DOCSTRING,
)
class BigBirdForMultipleChoice(BigBirdPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 BigBird 模型
        self.bert = BigBirdModel(config)
        # 定义一个丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 定义一个全连接层，输入大小为隐藏层大小，输出大小为1
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    # 添加代码样例的文档字符串，用于自动化文档生成工具
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 指定文档中的检查点
        output_type=MultipleChoiceModelOutput,  # 指定输出类型为多选模型输出
        config_class=_CONFIG_FOR_DOC,  # 指定文档中的配置类
    )
    # 此方法定义了模型的前向传播过程
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的词汇 ID，LongTensor 类型，默认为 None
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力遮罩，FloatTensor 类型，可选，默认为 None
        token_type_ids: Optional[torch.LongTensor] = None,  # 标记类型 ID，LongTensor 类型，可选，默认为 None
        position_ids: Optional[torch.LongTensor] = None,  # 位置 ID，LongTensor 类型，可选，默认为 None
        head_mask: Optional[torch.FloatTensor] = None,  # 注意力头部遮罩，FloatTensor 类型，可选，默认为 None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量，FloatTensor 类型，可选，默认为 None
        labels: Optional[torch.LongTensor] = None,  # 标签，LongTensor 类型，可选，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，布尔类型，可选，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，布尔类型，可选，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，布尔类型，可选，默认为 None
    ) -> Union[MultipleChoiceModelOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确保返回的字典类型正确
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取选择数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 重新构造输入张量形状，使其适合模型输入
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用BERT模型进行前向传播
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取汇总输出
        pooled_output = outputs[1]

        # 使用Dropout进行正则化
        pooled_output = self.dropout(pooled_output)
        # 使用分类器进行分类
        logits = self.classifier(pooled_output)
        # 重塑Logits的形状以匹配输入的选择数量
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 如果提供了标签，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不返回字典，则返回模型输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回带有损失、Logits、隐藏状态和注意力的多项选择模型输出
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```  
# 添加起始文档字符串，描述了 BigBird 模型在顶部带有一个标记分类头的结构，例如用于命名实体识别（NER）任务
# 这个类是 BigBirdForTokenClassification，它继承自 BigBirdPreTrainedModel
class BigBirdForTokenClassification(BigBirdPreTrainedModel):
    # 初始化函数，接受一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 设置模型的标签数量
        self.num_labels = config.num_labels

        # 创建 BigBird 模型
        self.bert = BigBirdModel(config)
        # 设置分类器的 dropout，如果没有指定分类器的 dropout，则使用配置中的隐藏层 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建 dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建线性层作为分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 重写 forward 函数，接受一系列输入参数，包括输入的标记、注意力掩码、标记类型 ID 等等
    # 具体的参数说明在文档字符串中有详细描述
    # forward 函数用于模型的前向传播
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TokenClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果没有指定 return_dict 参数，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 BERT 模型进行处理
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取 BERT 输出的序列表示
        sequence_output = outputs[0]

        # 对序列输出进行 dropout 处理
        sequence_output = self.dropout(sequence_output)
        # 使用分类器对序列输出进行分类
        logits = self.classifier(sequence_output)

        loss = None
        # 如果提供了标签，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不要求返回字典，则构建返回输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 对象，包括损失、预测结果、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class BigBirdForQuestionAnsweringHead(nn.Module):
    """Head for question answering tasks."""

    def __init__(self, config):
        super().__init__()
        # 初始化一个 dropout 层，用于随机断开输入的神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 初始化 BigBirdIntermediate 层，用于处理编码器输出的中间层
        self.intermediate = BigBirdIntermediate(config)
        # 初始化 BigBirdOutput 层，用于处理中间层输出并连接到输出层
        self.output = BigBirdOutput(config)
        # 初始化 qa_outputs 线性层，用于对隐藏状态进行线性变换得到 span 起始和结束的 logit
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, encoder_output):
        # 对编码器输出应用 dropout
        hidden_states = self.dropout(encoder_output)
        # 通过中间层处理隐藏状态
        hidden_states = self.intermediate(hidden_states)
        # 通过输出层处理隐藏状态并与编码器输出连接
        hidden_states = self.output(hidden_states, encoder_output)
        # 将处理后的隐藏状态输入到 qa_outputs 线性层得到 span 起始和结束的 logit
        hidden_states = self.qa_outputs(hidden_states)
        return hidden_states


@add_start_docstrings(
    """
    BigBird Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BIG_BIRD_START_DOCSTRING,
)
class BigBirdForQuestionAnswering(BigBirdPreTrainedModel):
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels
        self.sep_token_id = config.sep_token_id

        # 初始化 BigBirdModel，用于获取编码器的输出
        self.bert = BigBirdModel(config, add_pooling_layer=add_pooling_layer)
        # 初始化问题回答头部
        self.qa_classifier = BigBirdForQuestionAnsweringHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BigBirdForQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        question_lengths: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        ...

    @staticmethod
    def prepare_question_mask(q_lengths: torch.Tensor, maxlen: int):
        # 生成问题长度的掩码
        # q_lengths -> (bz, 1)
        mask = torch.arange(0, maxlen).to(q_lengths.device)
        mask.unsqueeze_(0)  # -> (1, maxlen)
        mask = torch.where(mask < q_lengths, 1, 0)
        return mask
```