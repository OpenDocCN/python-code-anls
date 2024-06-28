# `.\models\big_bird\modeling_big_bird.py`

```py
# 导入必要的库和模块
import math  # 导入数学库，用于数学运算
import os  # 导入操作系统库，用于操作文件路径等操作
from dataclasses import dataclass  # 导入dataclass模块，用于创建数据类
from typing import Optional, Tuple, Union  # 导入类型提示模块，用于类型声明

import numpy as np  # 导入NumPy库，用于数值计算
import torch  # 导入PyTorch库，用于构建和训练神经网络模型
import torch.utils.checkpoint  # 导入PyTorch的checkpoint模块，用于中间结果的保存和恢复
from torch import nn  # 导入PyTorch的神经网络模块，用于构建神经网络层
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入PyTorch的损失函数

from ...activations import ACT2FN  # 导入激活函数，用于神经网络的非线性变换
from ...modeling_outputs import (  # 导入模型输出类，定义了不同任务的输出格式
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类，用于所有预训练模型的基本功能实现
from ...pytorch_utils import apply_chunking_to_forward  # 导入用于分块处理前向传播的工具函数
from ...utils import (  # 导入工具函数，用于日志记录、返回值替换等辅助功能
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_big_bird import BigBirdConfig  # 导入BigBird模型的配置类，用于配置模型参数

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CHECKPOINT_FOR_DOC = "google/bigbird-roberta-base"  # 预训练模型的检查点地址，用于文档示例
_CONFIG_FOR_DOC = "BigBirdConfig"  # BigBird模型的配置信息，用于文档示例

BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST = [  # BigBird预训练模型的地址列表
    "google/bigbird-roberta-base",
    "google/bigbird-roberta-large",
    "google/bigbird-base-trivia-itc",
    # 查看所有BigBird模型地址：https://huggingface.co/models?filter=big_bird
]

_TRIVIA_QA_MAPPING = {  # TriviaQA数据集的映射关系，将TensorFlow模型权重映射到PyTorch模型
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


def load_tf_weights_in_big_bird(model, tf_checkpoint_path, is_trivia_qa=False):
    """Load tf checkpoints in a pytorch model."""
    def load_tf_weights_bert(init_vars, tf_path):
        names = []  # 用于存储变量名的列表
        tf_weights = {}  # 用于存储 TensorFlow 权重的字典

        for name, shape in init_vars:  # 遍历初始化变量的名称和形状
            array = tf.train.load_variable(tf_path, name)  # 加载 TensorFlow 模型中的变量值
            name = name.replace("bert/encoder/LayerNorm", "bert/embeddings/LayerNorm")  # 替换变量名中的特定字符串
            logger.info(f"Loading TF weight {name} with shape {shape}")  # 记录日志，显示加载的 TensorFlow 权重的名称和形状
            names.append(name)  # 将变量名添加到列表中
            tf_weights[name] = array  # 将变量名和对应的数组存储到字典中

        return names, tf_weights  # 返回变量名列表和 TensorFlow 权重字典

    def load_tf_weights_trivia_qa(init_vars):
        names = []  # 用于存储变量名的列表
        tf_weights = {}  # 用于存储 TensorFlow 权重的字典

        for i, var in enumerate(init_vars):  # 遍历初始化变量列表
            name_items = var.name.split("/")  # 使用斜杠分割变量名

            if "transformer_scaffold" in name_items[0]:  # 如果变量名中包含特定字符串
                layer_name_items = name_items[0].split("_")  # 使用下划线分割层名
                if len(layer_name_items) < 3:
                    layer_name_items += [0]  # 如果层名项少于3个，补充一个零

                name_items[0] = f"bert/encoder/layer_{layer_name_items[2]}"  # 格式化为特定的层名格式

            name = "/".join([_TRIVIA_QA_MAPPING[x] if x in _TRIVIA_QA_MAPPING else x for x in name_items])[:-2]
            # 根据映射替换变量名中的部分子串，并删除末尾的":0"

            if "self/attention/output" in name:  # 如果变量名中包含特定子串
                name = name.replace("self/attention/output", "output")  # 替换为指定的新子串

            if i >= len(init_vars) - 2:  # 如果索引超出初始化变量列表长度减2
                name = name.replace("intermediate", "output")  # 替换变量名中的特定子串为另一个

            logger.info(f"Loading TF weight {name} with shape {var.shape}")  # 记录日志，显示加载的 TensorFlow 权重的名称和形状
            array = var.value().numpy()  # 将 TensorFlow 变量的值转换为 NumPy 数组
            names.append(name)  # 将变量名添加到列表中
            tf_weights[name] = array  # 将变量名和对应的数组存储到字典中

        return names, tf_weights  # 返回变量名列表和 TensorFlow 权重字典

    try:
        import re  # 导入正则表达式模块

        import numpy as np  # 导入 NumPy 库
        import tensorflow as tf  # 导入 TensorFlow 库
    except ImportError:  # 如果导入错误
        logger.error(  # 记录错误日志
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise  # 抛出导入错误异常

    tf_path = os.path.abspath(tf_checkpoint_path)  # 获取 TensorFlow 检查点路径的绝对路径
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")  # 记录信息日志，显示正在转换的 TensorFlow 检查点路径

    # Load weights from TF model
    init_vars = tf.saved_model.load(tf_path).variables if is_trivia_qa else tf.train.list_variables(tf_path)
    # 根据条件加载 TensorFlow 模型的变量列表或者变量名列表

    if len(init_vars) <= 0:  # 如果初始化变量列表长度小于等于0
        raise ValueError("Loaded trained variables cannot be empty.")  # 抛出数值错误异常，提示加载的训练变量不能为空

    pt_names = list(model.state_dict().keys())  # 获取 PyTorch 模型状态字典的键列表

    if is_trivia_qa:  # 如果是 TriviaQA 数据集
        names, tf_weights = load_tf_weights_trivia_qa(init_vars)  # 调用加载 TriviaQA 数据集权重的函数
    else:  # 否则
        names, tf_weights = load_tf_weights_bert(init_vars, tf_path)  # 调用加载 BERT 模型权重的函数

    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")
    # 记录信息日志，显示未复制到 PyTorch 模型的权重名称列表

    logger.info(f"Weights not initialized in PyTorch model: {', '.join(pt_names)}.")
    # 记录信息日志，显示未在 PyTorch 模型中初始化的权重名称列表

    return model  # 返回模型
class BigBirdEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        # 创建一个词嵌入层，用于将词的索引映射为词嵌入向量
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建一个位置嵌入层，用于将位置索引映射为位置嵌入向量
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建一个token类型嵌入层，用于将token类型索引映射为token类型嵌入向量
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        # 创建一个LayerNorm层，用于归一化输入向量
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，用于在训练过程中随机丢弃部分输入向量，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # 根据配置创建position_embedding_type，用于指定位置嵌入类型
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册一个持久化的position_ids缓冲区，包含从0到config.max_position_embeddings-1的位置索引
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册一个持久化的token_type_ids缓冲区，其形状与position_ids相同，元素为零，用于token类型嵌入
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
        # End copy

        # 是否对嵌入向量进行重新缩放
        self.rescale_embeddings = config.rescale_embeddings
        # 隐藏层的大小
        self.hidden_size = config.hidden_size

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
        ):
            # 如果传入的 input_ids 不为空，获取其尺寸作为 input_shape
            if input_ids is not None:
                input_shape = input_ids.size()
            else:
                # 否则，获取 inputs_embeds 的尺寸除最后一维之外的部分作为 input_shape
                input_shape = inputs_embeds.size()[:-1]

            # 获取序列的长度
            seq_length = input_shape[1]

            # 如果没有提供 position_ids，则使用预定义的位置 ID，从 self.position_ids 中截取相应长度的部分
            if position_ids is None:
                position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

            # 设置 token_type_ids 为在构造函数中注册的缓冲区，通常情况下是全零，这对于在不传递 token_type_ids 的情况下追踪模型很有帮助，解决了问题 #5664
            if token_type_ids is None:
                if hasattr(self, "token_type_ids"):
                    # 使用已注册的 token_type_ids 缓冲区的部分来填充 token_type_ids，扩展以匹配 input_shape 的第一个维度和 seq_length
                    buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    # 如果未注册 token_type_ids，则创建全零的 tensor 作为 token_type_ids，类型为 long，放置在 self.position_ids 设备上
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

            # 如果 inputs_embeds 为空，则使用 word_embeddings 层来获取对应的 embeddings
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)

            # 如果需要对 embeddings 进行重新缩放，则乘以 sqrt(hidden_size)
            if self.rescale_embeddings:
                inputs_embeds = inputs_embeds * (self.hidden_size**0.5)

            # 根据 token_type_ids 获取 token_type_embeddings
            token_type_embeddings = self.token_type_embeddings(token_type_ids)

            # 将 inputs_embeds 和 token_type_embeddings 相加得到 embeddings
            embeddings = inputs_embeds + token_type_embeddings

            # 根据 position_ids 获取 position_embeddings，并将其加到 embeddings 中
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

            # 对 embeddings 应用 dropout
            embeddings = self.dropout(embeddings)
            
            # 对 embeddings 应用 LayerNorm
            embeddings = self.LayerNorm(embeddings)
            
            # 返回最终的 embeddings
            return embeddings
class BigBirdSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 检查隐藏层大小是否能被注意力头数整除，并且如果没有嵌入大小的属性，抛出错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值的线性层，用于注意力机制
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)

        # 设置 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 标记是否为解码器
        self.is_decoder = config.is_decoder

    # 将输入转换为注意力分数的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数
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

        # 设置最大序列长度和随机数种子
        self.max_seqlen = config.max_position_embeddings
        self.seed = seed

        # 检查隐藏层大小是否能被注意力头数整除，如果不能抛出错误
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 设置注意力头数、随机块数和块大小
        self.num_attention_heads = config.num_attention_heads
        self.num_random_blocks = config.num_random_blocks
        self.block_size = config.block_size

        # 设置注意力头大小和总大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值的线性层，用于注意力机制
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)

    # 将输入转换为注意力分数的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数
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
        # 目前此类无法在解码器中使用。

        # 获取隐藏状态的批量大小、序列长度和最后一个维度的信息
        batch_size, seqlen, _ = hidden_states.size()
        to_seq_length = from_seq_length = seqlen
        from_block_size = to_block_size = self.block_size

        # 检查查询侧序列长度是否是块大小的倍数
        if from_seq_length % from_block_size != 0:
            raise ValueError("Query sided sequence length must be multiple of block size")

        # 检查键/值侧序列长度是否是块大小的倍数
        if to_seq_length % to_block_size != 0:
            raise ValueError("Key/Value sided sequence length must be multiple of block size")

        # 对查询、键和值进行变换以适应注意力矩阵的计算
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 调用大鸟模型的稀疏块注意力计算函数
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

        # 将上下文层展开并重塑形状以匹配输入张量的预期形状
        context_layer = context_layer.contiguous().view(batch_size, from_seq_length, -1)

        # 根据需要返回注意力权重
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    @staticmethod
    def torch_bmm_nd(inp_1, inp_2, ndim=None):
        """快速的多维矩阵乘法"""
        # 使用 torch.bmm 替代 torch.einsum 进行更快的矩阵乘法计算 ("bhqk,bhkd->bhqd")
        return torch.bmm(inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:])).view(
            inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 1])
        )

    @staticmethod
    def torch_bmm_nd_transpose(inp_1, inp_2, ndim=None):
        """带转置的快速多维矩阵乘法"""
        # 使用 torch.bmm 替代 torch.einsum 进行更快的矩阵乘法计算 ("bhqd,bhkd->bhqk")
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
    ):
        # 大鸟模型的稀疏块注意力计算函数，详细实现略过
        pass
    def torch_gather_b2(params, indices):
        # this operation is equivalent to tf.gather when batch_dims=2

        # 检查 params 和 indices 的前两个维度是否相同
        if params.shape[:2] != indices.shape[:2]:
            raise ValueError(
                "Make sure that the first two dimensions of params and indices are identical, "
                f"but they are params: {params.shape[:2]} vs. indices: {indices.shape[:2]}"
            )

        # 计算需要收集的索引数量
        num_indices_to_gather = indices.shape[-2] * indices.shape[-1]
        # 获取 params 中可选择的索引数量
        num_indices_to_pick_from = params.shape[2]

        # 创建偏移量，以便在展平的 indices 上进行选择
        shift = torch.arange(indices.shape[0] * indices.shape[1] * num_indices_to_gather, device=indices.device)
        indices_shift = torch.div(shift, num_indices_to_gather, rounding_mode="floor") * num_indices_to_pick_from

        # 将 indices 展平并添加偏移量，以便在 params 中选择对应数据
        flattened_indices = indices.view(-1) + indices_shift
        flattened_params = params.reshape(-1, params.shape[-2], params.shape[-1])

        # 使用展平后的 indices 在 params 中进行选择
        out_flattened = flattened_params.index_select(0, flattened_indices)

        # 将结果重新形状为原始形状，包括收集的索引数量维度
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
            from_blocked_mask: 2D Tensor of shape [batch_size, from_seq_length//from_block_size, from_block_size].
            to_blocked_mask: int32 Tensor of shape [batch_size, to_seq_length//to_block_size, to_block_size].
            rand_attn: [batch_size, num_attention_heads, from_seq_length//from_block_size-2, num_rand_blocks]
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
        # 从 to_blocked_mask 和 rand_attn 创建随机掩码
        rand_mask = torch.stack([p1[i1.flatten()] for p1, i1 in zip(to_blocked_mask, rand_attn)])
        # 将随机掩码重塑为所需的形状
        rand_mask = rand_mask.view(batch_size, num_attention_heads, num_windows, num_rand_blocks * from_block_size)
        # 使用 einsum 创建最终的随机掩码
        rand_mask = torch.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1], rand_mask)
        return rand_mask
    # 定义一个函数，用于生成随机注意力的分布计划
    def _get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
        """
        Gives the plan of where to put random attention.

        Args:
            from_seq_length: int. length of from sequence. 输入序列的长度
            from_block_size: int. size of block in from sequence. 输入序列中每个块的大小
            num_rand_blocks: int. Number of random chunks per row. 每行的随机块数

        Returns:
            plan_from_length: ending location of from block plan_num_rand_blocks: number of random ending location for
            each block 返回计划的输入序列块的结束位置和每个块的随机结束位置数量
        """

        # 初始化存储计划信息的列表
        plan_from_length = []
        plan_num_rand_blocks = []

        # 根据条件生成计划信息
        if (2 * num_rand_blocks + 5) < (from_seq_length // from_block_size):
            # 如果满足条件，计划从块的长度为 (2 * num_rand_blocks + 5) * from_block_size
            plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks)
            # 输入序列长度为 from_seq_length
            plan_from_length.append(from_seq_length)
            # 随机块数为 0
            plan_num_rand_blocks.append(0)
        elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
            # 否则，如果满足条件，计划从块的长度为 (num_rand_blocks + 5) * from_block_size
            plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
            # 前一半块的随机块数为 num_rand_blocks // 2
            plan_num_rand_blocks.append(num_rand_blocks // 2)
            # 输入序列长度为 from_seq_length
            plan_from_length.append(from_seq_length)
            # 后一半块的随机块数为 num_rand_blocks - (num_rand_blocks // 2)
            plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks // 2))
        else:
            # 否则，输入序列长度为 from_seq_length
            plan_from_length.append(from_seq_length)
            # 随机块数为 num_rand_blocks
            plan_num_rand_blocks.append(num_rand_blocks)

        # 返回生成的计划信息
        return plan_from_length, plan_num_rand_blocks

    # 定义一个函数，用于生成 BigBird 模型的随机掩码
    def _bigbird_block_rand_mask(
        self, from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, last_idx=-1
        ):
    # 创建随机注意力的邻接列表。
    
    Args:
        from_seq_length: int. 来源序列的长度。
        to_seq_length: int. 目标序列的长度。
        from_block_size: int. 来源序列中的块大小。
        to_block_size: int. 目标序列中的块大小。
        num_rand_blocks: int. 每行随机块的数量。
        last_idx: int. 如果为-1，则从目标序列中任意选择 num_rand_blocks 个块；
                  如果为正数，则只选择到 last_idx 为止的 num_rand_blocks 个块。
    
    Returns:
        邻接列表，大小为 from_seq_length//from_block_size-2 行，num_rand_blocks 列。
        表示每个源序列块与随机选择的目标序列块之间的注意力关系。
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
        """
        Generates a random mask for BigBird attention with head information.

        Args:
            from_seq_length: int. Length of the source sequence.
            to_seq_length: int. Length of the target sequence.
            from_block_size: int. Block size of the source sequence.
            to_block_size: int. Block size of the target sequence.
            num_heads: int. Number of attention heads.
            plan_from_length: int. Planned length of the source sequence.
            plan_num_rand_blocks: int. Planned number of random blocks for attention.
            window_block_left: int. Number of blocks in the window to the left of a block.
            window_block_right: int. Number of blocks in the window to the right of a block.
            global_block_top: int. Number of global blocks used at the top.
            global_block_bottom: int. Number of global blocks used at the bottom.
            global_block_left: int. Number of global blocks used to the left.
            global_block_right: int. Number of global blocks used to the right.

        Returns:
            A randomly masked attention matrix with head information.
        """
        # Implementation details for generating a random mask with BigBird constraints
        pass

    @staticmethod
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
        For a single row block, get random row attention.

        Args:
            block_id: int. Block ID of the row.
            to_start_block_id: int. Start ID of the attention column.
            to_end_block_id: int. End ID of the attention column.
            num_rand_blocks: int. Number of random blocks to select.
            window_block_left: int. Number of blocks in the window to the left of a block.
            window_block_right: int. Number of blocks in the window to the right of a block.
            global_block_left: int. Number of global blocks used to the left.
            global_block_right: int. Number of global blocks used to the right.

        Returns:
            Array containing the random attention vector of size num_rand_blocks.
        """
        # List of to_blocks from which to choose random attention
        to_block_list = np.arange(to_start_block_id, to_end_block_id, dtype=np.int32)
        # Permute the blocks
        perm_block = np.random.permutation(to_block_list)

        # Illegal blocks for the current block_id, using window
        illegal_blocks = list(range(block_id - window_block_left, block_id + window_block_right + 1))

        # Add blocks at the start and at the end
        illegal_blocks.extend(list(range(global_block_left)))
        illegal_blocks.extend(list(range(to_end_block_id - global_block_right, to_end_block_id)))

        # The second from_block cannot choose random attention on second last to_block
        if block_id == 1:
            illegal_blocks.append(to_end_block_id - 2)

        # The second last from_block cannot choose random attention on second to_block
        if block_id == to_end_block_id - 2:
            illegal_blocks.append(1)

        selected_random_blocks = []

        for i in range(to_end_block_id - to_start_block_id):
            if perm_block[i] not in illegal_blocks:
                selected_random_blocks.append(perm_block[i])
            if len(selected_random_blocks) == num_rand_blocks:
                break
        return np.array(selected_random_blocks, dtype=np.int32)
# 从transformers.models.bert.modeling_bert.BertSelfOutput中复制代码，并将Bert改为BigBird
class BigBirdSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入特征大小映射为输出特征大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个LayerNorm层，用于归一化输入数据
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，用于随机失活神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态通过全连接层映射到新的空间
        hidden_states = self.dense(hidden_states)
        # 对映射后的结果进行随机失活
        hidden_states = self.dropout(hidden_states)
        # 将映射结果与输入张量进行残差连接，并通过LayerNorm进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回归一化后的结果张量
        return hidden_states


class BigBirdAttention(nn.Module):
    def __init__(self, config, seed=None):
        super().__init__()
        # 初始化注意力类型
        self.attention_type = config.attention_type
        # 存储配置信息
        self.config = config
        # 存储随机种子信息
        self.seed = seed

        # 根据配置选择不同的注意力类型
        if self.config.attention_type == "original_full":
            # 如果是原始全注意力类型，则使用BigBirdSelfAttention
            self.self = BigBirdSelfAttention(config)
        elif self.config.attention_type == "block_sparse":
            # 如果是块稀疏注意力类型，则使用BigBirdBlockSparseAttention
            self.self = BigBirdBlockSparseAttention(config, seed)
        else:
            # 如果配置的注意力类型不在支持范围内，则抛出错误
            raise ValueError(
                f"attention_type can either be original_full or block_sparse, but is {self.config.attention_type}"
            )

        # 创建自定义的输出层
        self.output = BigBirdSelfOutput(config)

    def set_attention_type(self, value: str):
        # 如果设置的注意力类型不在支持的范围内，则抛出错误
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # 如果设置的注意力类型与当前类型一致，则直接返回
        if value == self.attention_type:
            return

        # 更新当前的注意力类型
        self.attention_type = value
        # 根据新的注意力类型重新设置self.self
        if value == "original_full":
            # 复制所有权重到新的全注意力类
            attn_weights = BigBirdSelfAttention(self.config)
        else:
            # 复制所有权重到新的稀疏注意力类
            attn_weights = BigBirdBlockSparseAttention(self.config, self.seed)

        # 将当前的查询、键、值权重复制到新的注意力对象中
        attn_weights.query = self.self.query
        attn_weights.value = self.self.value
        attn_weights.key = self.self.key
        # 更新self.self为新的注意力对象
        self.self = attn_weights
        # 更新注意力类型
        self.attention_type = value
        # 如果不在训练状态下，评估更新后的self.self
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
        # 块稀疏配置
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
        # fp16 compatibility
        # 如果使用了 fp16，需要确保 band_mask、from_mask、to_mask 的数据类型与 hidden_states 相匹配
        if band_mask is not None:
            band_mask = band_mask.to(hidden_states.dtype)
        if from_mask is not None:
            from_mask = from_mask.to(hidden_states.dtype)
        if to_mask is not None:
            to_mask = to_mask.to(hidden_states.dtype)

        # 根据不同的 attention_type 选择不同的 self-attention 计算方式
        if self.attention_type == "original_full":
            # 使用全连接注意力机制进行自注意力计算
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
            # 如果是 BigBird 模型，且作为解码器使用时抛出错误
            if encoder_hidden_states is not None:
                raise ValueError("BigBird cannot be used as a decoder when config.attention_type != 'original_full'")
            # 使用 BigBird 特有的部分连接注意力机制进行自注意力计算
            self_outputs = self.self(
                hidden_states, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, output_attentions
            )

        # 将 self-attention 的输出作为输入，经过输出层处理得到最终的注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)

        # 如果需要输出注意力权重，将它们添加到输出元组中
        outputs = (attention_output,) + self_outputs[1:]  # 如果输出了注意力权重，则添加到输出中
        return outputs
# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制并修改为 BigBirdIntermediate 类
class BigBirdIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入的隐藏状态大小调整为中间状态大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择隐藏层激活函数，如果是字符串则从预定义映射中选择对应函数，否则直接使用配置中的函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态输入全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 应用中间层激活函数到变换后的隐藏状态
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回经过线性变换和激活函数后的隐藏状态
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput 复制并修改为 BigBirdOutput 类
class BigBirdOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将中间状态大小调整为隐藏状态大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，对隐藏状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，用于随机置零隐藏状态中的部分元素
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态输入全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态应用 Dropout 层
        hidden_states = self.dropout(hidden_states)
        # 对加和后的结果进行 LayerNorm 归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回经过线性变换、Dropout 和 LayerNorm 后的隐藏状态
        return hidden_states


# BigBirdLayer 类，定义 BigBird 模型的一个层
class BigBirdLayer(nn.Module):
    def __init__(self, config, seed=None):
        super().__init__()
        self.config = config
        self.attention_type = config.attention_type
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 创建 BigBirdAttention 层，用于处理注意力机制
        self.attention = BigBirdAttention(config, seed=seed)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        # 如果设置了交叉注意力，确保模型是解码器模型，否则抛出错误
        if self.add_cross_attention:
            if not self.is_decoder:
                raise TypeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BigBirdAttention(config)
        # 创建 BigBirdIntermediate 层，用于处理中间层操作
        self.intermediate = BigBirdIntermediate(config)
        # 创建 BigBirdOutput 层，用于处理输出层操作
        self.output = BigBirdOutput(config)

    def set_attention_type(self, value: str):
        # 如果给定的注意力类型不是 'original_full' 或 'block_sparse'，抛出 ValueError
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # 如果当前注意力类型已经正确设置，则直接返回
        if value == self.attention_type:
            return
        # 否则更新注意力类型，并将新类型应用到注意力层和交叉注意力层（如果存在）
        self.attention_type = value
        self.attention.set_attention_type(value)

        if self.add_cross_attention:
            self.crossattention.set_attention_type(value)
    # 定义神经网络的前向传播函数，用于推断或训练过程中的前向计算
    def forward(
        self,
        hidden_states,                    # 输入的隐藏状态张量，通常是模型的输出或前一层的输出
        attention_mask=None,              # 注意力掩码，用于指定哪些位置需要屏蔽，通常用于处理变长序列
        head_mask=None,                   # 头部掩码，用于指定哪些注意力头部需要屏蔽
        encoder_hidden_states=None,       # 编码器的隐藏状态，用于跨层注意力等任务
        encoder_attention_mask=None,      # 编码器的注意力掩码，指定哪些编码器位置需要屏蔽
        band_mask=None,                   # 带状掩码，用于指定注意力矩阵中的带状结构
        from_mask=None,                   # 起始位置掩码，指定从哪些位置开始注意力计算
        to_mask=None,                     # 终止位置掩码，指定到哪些位置结束注意力计算
        blocked_encoder_mask=None,        # 阻塞编码器掩码，用于指定哪些编码器隐藏状态应被屏蔽
        past_key_value=None,              # 过去的键值对，用于缓存前向传播中的注意力权重等信息
        output_attentions=False,          # 是否输出注意力权重信息，默认为不输出

        # 定义神经网络的前向传播函数，用于推断或训练过程中的前向计算
        def forward(
            self,
            hidden_states,                    # 输入的隐藏状态张量，通常是模型的输出或前一层的输出
            attention_mask=None,              # 注意力掩码，用于指定哪些位置需要屏蔽，通常用于处理变长序列
            head_mask=None,                   # 头部掩码，用于指定哪些注意力头部需要屏蔽
            encoder_hidden_states=None,       # 编码器的隐藏状态，用于跨层注意力等任务
            encoder_attention_mask=None,      # 编码器的注意力掩码，指定哪些编码器位置需要屏蔽
            band_mask=None,                   # 带状掩码，用于指定注意力矩阵中的带状结构
            from_mask=None,                   # 起始位置掩码，指定从哪些位置开始注意力计算
            to_mask=None,                     # 终止位置掩码，指定到哪些位置结束注意力计算
            blocked_encoder_mask=None,        # 阻塞编码器掩码，用于指定哪些编码器隐藏状态应被屏蔽
            past_key_value=None,              # 过去的键值对，用于缓存前向传播中的注意力权重等信息
            output_attentions=False,          # 是否输出注意力权重信息，默认为不输出
    ):
        # 如果过去的键/值对存在，则只保留自注意力的缓存的前两个位置（1和2）
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 执行自注意力计算
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

        # 如果模型是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            # 排除最后一个元素，它是自注意力缓存的元组
            outputs = self_attention_outputs[1:-1]
            # 获取自注意力计算的当前键/值对
            present_key_value = self_attention_outputs[-1]
        else:
            # 排除第一个元素，因为我们输出注意力权重时需要添加自注意力
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        # 如果是解码器并且存在编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果模型没有交叉注意力层，则引发错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with                    "
                    " cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # 如果过去的键/值对存在，则只保留交叉注意力缓存的后两个位置（3和4）
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 执行交叉注意力计算
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
            # 添加交叉注意力计算的输出（排除最后一个元素，它是交叉注意力缓存的元组）
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力缓存添加到当前键/值对的第三和第四位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 将注意力输出应用于前向网络的分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        # 组装最终的输出元组
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力的键/值对作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    # 定义一个方法，用于执行神经网络的前向传播，处理给定的注意力输出
    def feed_forward_chunk(self, attention_output):
        # 调用 self.intermediate 方法，对注意力输出进行中间处理
        intermediate_output = self.intermediate(attention_output)
        # 调用 self.output 方法，将中间处理后的结果和注意力输出作为参数，生成最终的层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回最终的层输出作为结果
        return layer_output
# 定义 BigBirdEncoder 类，继承自 nn.Module
class BigBirdEncoder(nn.Module):
    # 初始化方法，接受 config 参数
    def __init__(self, config):
        super().__init__()
        self.config = config  # 将传入的 config 参数保存到实例变量中
        self.attention_type = config.attention_type  # 从 config 中获取 attention_type 参数

        # 创建一个包含多个 BigBirdLayer 实例的列表，每个实例都使用不同的 seed
        self.layer = nn.ModuleList(
            [BigBirdLayer(config, seed=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.gradient_checkpointing = False  # 初始化 gradient_checkpointing 标志为 False

    # 设置 attention_type 的方法，接受一个字符串参数 value
    def set_attention_type(self, value: str):
        # 如果 value 不是 "original_full" 或 "block_sparse"，抛出 ValueError 异常
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # 如果当前 attention_type 已经是要设置的值，则直接返回，不进行更改
        if value == self.attention_type:
            return
        # 更新 attention_type 为新的值
        self.attention_type = value
        # 遍历所有层并设置它们的 attention_type
        for layer in self.layer:
            layer.set_attention_type(value)

    # 前向传播方法定义
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
    ):
        # 这里是 BigBirdEncoder 的前向传播逻辑，具体实现在这里面
        pass  # 在这里应该填写实际的前向传播逻辑，暂时为空



# 定义 BigBirdPredictionHeadTransform 类，继承自 nn.Module
class BigBirdPredictionHeadTransform(nn.Module):
    # 初始化方法，接受 config 参数
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 创建一个线性层
        # 根据 config 中的 hidden_act 字符串或函数设置 transform_act_fn
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 创建 LayerNorm 层

    # 前向传播方法定义，接受输入 hidden_states，并返回输出的 torch.Tensor
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)  # 将输入通过线性层 dense
        hidden_states = self.transform_act_fn(hidden_states)  # 经过激活函数变换
        hidden_states = self.LayerNorm(hidden_states)  # 应用 LayerNorm
        return hidden_states  # 返回变换后的 hidden_states



# 定义 BigBirdLMPredictionHead 类，继承自 nn.Module
class BigBirdLMPredictionHead(nn.Module):
    # 初始化方法，接受 config 参数
    def __init__(self, config):
        super().__init__()
        self.transform = BigBirdPredictionHeadTransform(config)  # 创建 BigBirdPredictionHeadTransform 实例
        # 输出层是一个线性层，将隐藏状态映射到词汇表大小的向量，没有偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))  # 创建偏置参数
        self.decoder.bias = self.bias  # 将偏置参数与 decoder 层关联，以便与 resize_token_embeddings 正确调整大小

    # 前向传播方法定义，接受输入 hidden_states，并返回输出的 torch.Tensor
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)  # 将输入经过 transform 处理
        hidden_states = self.decoder(hidden_states)  # 使用 decoder 进行线性变换
        return hidden_states  # 返回变换后的 hidden_states
# 从 transformers.models.bert.modeling_bert.BertOnlyMLMHead 复制并将 Bert 改为 BigBird
class BigBirdOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化 MLM 头部预测层
        self.predictions = BigBirdLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 调用预测层进行预测
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# 从 transformers.models.bert.modeling_bert.BertOnlyNSPHead 复制并将 Bert 改为 BigBird
class BigBirdOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化 NSP 头部的线性层
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        # 计算序列关系分数
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


# 从 transformers.models.bert.modeling_bert.BertPreTrainingHeads 复制并将 Bert 改为 BigBird
class BigBirdPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化 MLM 预测头部和 NSP 头部
        self.predictions = BigBirdLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        # 分别调用预测层和序列关系层进行计算
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BigBirdPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BigBirdConfig
    load_tf_weights = load_tf_weights_in_big_bird
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化线性层的权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果存在偏置，则将偏置初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化嵌入层的权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果有填充索引，则将填充索引处的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 将 LayerNorm 层的偏置初始化为零，权重初始化为 1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


BIG_BIRD_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
"""
    Parameters:
        config ([`BigBirdConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BIG_BIRD_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@dataclass
class BigBirdForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BigBirdForPreTraining`].
    
    This class defines the output structure for the BigBird model during pre-training.
    """
    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            总损失，由掩码语言建模损失和下一个序列预测（分类）损失之和组成。
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            语言建模头部的预测分数（SoftMax之前的每个词汇标记的分数）。
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            下一个序列预测（分类）头部的预测分数（SoftMax之前的True/False延续的分数）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型在每层输出的隐藏状态的元组，包括初始嵌入输出。
            每个元素是 `torch.FloatTensor`，形状为 `(batch_size, sequence_length, hidden_size)`。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            注意力权重的元组，用于计算自注意力头部中的加权平均值。
            每个元素是 `torch.FloatTensor`，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个数据类，用于存储问题回答模型的输出结果
@dataclass
class BigBirdForQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        pooler_output (`torch.FloatTensor` of shape `(batch_size, 1)`):
            pooler output from BigBirdModel
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

    # 损失值，如果提供了`labels`则返回，表示总的跨度提取损失，由起始和结束位置的交叉熵之和组成
    loss: Optional[torch.FloatTensor] = None
    # 跨度起始得分（SoftMax之前），形状为(batch_size, sequence_length)
    start_logits: torch.FloatTensor = None
    # 跨度结束得分（SoftMax之前），形状为(batch_size, sequence_length)
    end_logits: torch.FloatTensor = None
    # BigBirdModel的汇聚输出，形状为(batch_size, 1)
    pooler_output: torch.FloatTensor = None
    # 隐藏状态，如果`output_hidden_states=True`则返回，是一个元组，包含了每一层的输出，形状为(batch_size, sequence_length, hidden_size)
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力权重，如果`output_attentions=True`则返回，是一个元组，包含了每一层的注意力权重，形状为(batch_size, num_heads, sequence_length, sequence_length)
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# BigBird模型类，继承自BigBirdPreTrainedModel
@add_start_docstrings(
    "The bare BigBird Model transformer outputting raw hidden-states without any specific head on top.",
    BIG_BIRD_START_DOCSTRING,
)
class BigBirdModel(BigBirdPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """
    # 初始化函数，用于初始化 BigBirdForCausalLM 类的实例
    def __init__(self, config, add_pooling_layer=True):
        # 调用父类的初始化函数
        super().__init__(config)
        # 设置注意力机制类型为配置文件中指定的类型
        self.attention_type = self.config.attention_type
        # 保存传入的配置参数
        self.config = config

        # 设置模型的块大小为配置中指定的块大小
        self.block_size = self.config.block_size

        # 初始化嵌入层，使用 BigBirdEmbeddings 类
        self.embeddings = BigBirdEmbeddings(config)
        # 初始化编码器，使用 BigBirdEncoder 类
        self.encoder = BigBirdEncoder(config)

        # 如果需要添加池化层
        if add_pooling_layer:
            # 创建一个线性层用于池化
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            # 激活函数为双曲正切函数
            self.activation = nn.Tanh()
        else:
            # 如果不需要池化层，设置为 None
            self.pooler = None
            self.activation = None

        # 如果注意力类型不是 "original_full" 且配置要求添加交叉注意力
        if self.attention_type != "original_full" and config.add_cross_attention:
            # 发出警告并强制将 attention_type 设为 "original_full"
            logger.warning(
                "When using `BigBirdForCausalLM` as decoder, then `attention_type` must be `original_full`. Setting"
                " `attention_type=original_full`"
            )
            self.set_attention_type("original_full")

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入层的方法
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 设置注意力类型的方法
    def set_attention_type(self, value: str):
        # 如果值不是 "original_full" 或 "block_sparse"，抛出异常
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # 如果当前的 attention_type 已经正确设置，则直接返回
        if value == self.attention_type:
            return
        # 否则更新 attention_type，并通知编码器更新注意力类型
        self.attention_type = value
        self.encoder.set_attention_type(value)

    # 前向传播函数，实现了 BigBirdForCausalLM 类的前向计算过程
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
    ):
        # 此处定义了前向传播的参数和返回值类型，详细文档请参考相应的注释和文档字符串
        pass  # 此处仅为占位符，实际前向传播功能未在此展示
    def create_masks_for_block_sparse_attn(attention_mask: torch.Tensor, block_size: int):
        # 获取批次大小和序列长度
        batch_size, seq_length = attention_mask.size()
        
        # 检查序列长度是否是块大小的整数倍，如果不是则引发异常
        if seq_length % block_size != 0:
            raise ValueError(
                f"Sequence length must be multiple of block size, but sequence length is {seq_length}, while block"
                f" size is {block_size}."
            )

        def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
            """
            从2D张量掩码创建3D注意力掩码。

            Args:
                from_blocked_mask: 形状为[batch_size, from_seq_length//from_block_size, from_block_size]的2D张量。
                to_blocked_mask: 形状为[batch_size, to_seq_length//to_block_size, to_block_size]的int32张量。

            Returns:
                形状为[batch_size, 1, from_seq_length//from_block_size-4, from_block_size, 3*to_block_size]的浮点张量。
            """
            # 从输入的块掩码创建带状掩码
            exp_blocked_to_pad = torch.cat(
                [to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]], dim=2
            )
            band_mask = torch.einsum("blq,blk->blqk", from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
            band_mask.unsqueeze_(1)
            return band_mask

        # 将注意力掩码重塑为块表示形式
        blocked_encoder_mask = attention_mask.view(batch_size, seq_length // block_size, block_size)
        
        # 使用块掩码创建带状掩码
        band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)

        # 为源掩码和目标掩码创建需要的形状
        from_mask = attention_mask.view(batch_size, 1, seq_length, 1)
        to_mask = attention_mask.view(batch_size, 1, 1, seq_length)

        # 返回块掩码、带状掩码、源掩码和目标掩码
        return blocked_encoder_mask, band_mask, from_mask, to_mask

    def _pad_to_block_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        pad_token_id: int,
        """
        A helper function to pad tokens and mask to work with implementation of BigBird block-sparse attention.
        """
        # padding
        block_size = self.config.block_size  # 从模型配置中获取块大小

        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape  # 获取输入的形状
        batch_size, seq_len = input_shape[:2]  # 获取批次大小和序列长度

        padding_len = (block_size - seq_len % block_size) % block_size  # 计算需要填充的长度
        if padding_len > 0:
            logger.warning_once(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.block_size`: {block_size}"
            )  # 发出警告，说明输入的 ids 自动填充以确保长度是 `config.block_size` 的倍数

            if input_ids is not None:
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)  # 对输入 ids 进行填充
            if position_ids is not None:
                # 使用 pad_token_id 填充 position_ids，与 modeling_bigbird.BigBirdEmbeddings 中保持一致
                position_ids = nn.functional.pad(position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                # 创建新的输入 ids 填充，并使用模型的 embeddings 生成对应的嵌入向量
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config.pad_token_id,
                    dtype=torch.long,
                )
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)  # 将填充后的嵌入向量拼接到原始嵌入向量中

            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=False
            )  # 对注意力掩码进行填充，填充部分不计入注意力
            token_type_ids = nn.functional.pad(token_type_ids, (0, padding_len), value=0)  # 使用 token_type_id = 0 进行填充

        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds
# 定义 BigBirdForMaskedLM 类，继承自 BigBirdPreTrainedModel 类
class BigBirdForMaskedLM(BigBirdPreTrainedModel):
    # 定义权重共享的键值对列表
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    # 初始化函数，接收一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果配置要求是 decoder，则发出警告信息
        if config.is_decoder:
            logger.warning(
                "If you want to use `BigBirdForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 使用配置初始化 BigBirdModel，并设置为 self.bert 属性
        self.bert = BigBirdModel(config)
        # 使用配置初始化 BigBirdOnlyMLMHead，并设置为 self.cls 属性
        self.cls = BigBirdOnlyMLMHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    # 返回预测解码器的输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置预测解码器的输出嵌入为新的嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 覆盖模型的 forward 方法，接受多个输入参数，并带有相关的文档字符串和返回值替换
    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
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
    ):
        # 函数体暂时省略
        pass
    # 定义一个方法用于模型的前向传播
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的token ID序列，数据类型为LongTensor
        attention_mask: Optional[torch.FloatTensor] = None,  # 可选的注意力遮罩，数据类型为FloatTensor
        token_type_ids: Optional[torch.LongTensor] = None,  # 可选的token类型ID序列，数据类型为LongTensor
        position_ids: Optional[torch.LongTensor] = None,  # 可选的位置ID序列，数据类型为LongTensor
        head_mask: Optional[torch.FloatTensor] = None,  # 可选的头部遮罩，数据类型为FloatTensor
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 可选的嵌入向量输入，数据类型为FloatTensor
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # 可选的编码器隐藏状态，数据类型为FloatTensor
        encoder_attention_mask: Optional[torch.FloatTensor] = None,  # 可选的编码器注意力遮罩，数据类型为FloatTensor
        labels: Optional[torch.LongTensor] = None,  # 可选的标签，数据类型为LongTensor
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，数据类型为bool
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，数据类型为bool
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果，数据类型为bool
    ):
        # 定义用于生成输入的方法，支持模型生成任务
        def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
            # 获取输入token ID序列的形状信息
            input_shape = input_ids.shape
            # 获取有效的批次大小
            effective_batch_size = input_shape[0]

            # 如果配置中未定义PAD token ID，则抛出数值错误
            if self.config.pad_token_id is None:
                raise ValueError("The PAD token should be defined for generation")
            # 在注意力遮罩的末尾添加一个虚拟token
            attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
            # 创建一个全为PAD token ID的虚拟token张量
            dummy_token = torch.full(
                (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
            )
            # 将虚拟token添加到输入token ID序列的末尾
            input_ids = torch.cat([input_ids, dummy_token], dim=1)

            # 返回输入字典，包括输入token ID序列和更新后的注意力遮罩
            return {"input_ids": input_ids, "attention_mask": attention_mask}
# 使用装饰器添加文档字符串，说明这是一个 BigBird 模型，用于语言建模任务的微调
@add_start_docstrings(
    """BigBird Model with a `language modeling` head on top for CLM fine-tuning.""", BIG_BIRD_START_DOCSTRING
)
# 定义 BigBirdForCausalLM 类，继承自 BigBirdPreTrainedModel
class BigBirdForCausalLM(BigBirdPreTrainedModel):
    # 指定共享权重的键名列表，用于多个权重共享
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    # 初始化方法，接收配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果不是解码器模式，记录警告信息
        if not config.is_decoder:
            logger.warning("If you want to use `BigBirdForCausalLM` as a standalone, add `is_decoder=True.`")

        # 初始化 BigBirdModel 和 BigBirdOnlyMLMHead
        self.bert = BigBirdModel(config)
        self.cls = BigBirdOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回输出嵌入层的方法
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入层的方法
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播方法，接收多个输入参数，返回模型预测结果
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
        # 为生成准备输入的方法，接收输入的 ID、过去的键值对、注意力掩码等参数
        def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
            # 获取输入的形状
            input_shape = input_ids.shape

            # 如果没有提供注意力掩码，创建全为1的注意力掩码
            if attention_mask is None:
                attention_mask = input_ids.new_ones(input_shape)

            # 如果使用过去的键值对，截取输入的 ID
            if past_key_values is not None:
                past_length = past_key_values[0][0].shape[2]

                # 某些生成方法已经只传递了最后一个输入 ID
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # 默认旧的行为：仅保留最后一个 ID
                    remove_prefix_length = input_ids.shape[1] - 1

                input_ids = input_ids[:, remove_prefix_length:]

            # 返回输入字典，包含输入 ID、注意力掩码和过去的键值对
            return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}
    # 重新排序缓存中的过去键值对，以适应束搜索中的索引重排
    def _reorder_cache(self, past_key_values, beam_idx):
        # 初始化重新排序后的过去状态元组
        reordered_past = ()
        # 遍历每一层的过去状态
        for layer_past in past_key_values:
            # 对每一层的过去状态的前两个元素（通常是键和值）进行重新排序
            reordered_past += (
                # 使用beam_idx重新排序，保证与束搜索顺序一致，转移到相同设备上
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                # 将第三个元素及以后的元素保持不变，通常是额外的状态信息
                + layer_past[2:],
            )
        # 返回重新排序后的过去状态元组
        return reordered_past
class BigBirdForMultipleChoice(BigBirdPreTrainedModel):
    """BigBird Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks."""

    def __init__(self, config):
        super().__init__(config)

        # 使用 BigBirdModel 初始化 BERT 部分
        self.bert = BigBirdModel(config)
        # 使用给定的隐藏层丢弃概率初始化 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 使用线性层初始化分类器，输出维度为1，用于多选任务
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(
        BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @replace_return_docstrings(output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
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
    ):
        """
        前向传播方法，执行 BigBird 多选分类任务。

        参数:
            input_ids: 输入 token IDs 张量，形状为 (batch_size, num_choices, sequence_length)
            attention_mask: 注意力掩码张量，形状为 (batch_size, num_choices, sequence_length)
            token_type_ids: token 类型 IDs 张量，形状为 (batch_size, num_choices, sequence_length)
            position_ids: 位置 IDs 张量，形状为 (batch_size, num_choices, sequence_length)
            head_mask: 头部掩码张量，形状为 (num_heads,) 或者 (num_layers, num_heads)
            inputs_embeds: 输入嵌入张量，形状为 (batch_size, num_choices, sequence_length, hidden_size)
            labels: 标签张量，形状为 (batch_size,)，每个值为 0 或 1
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            return_dict: 是否返回字典形式的输出

        返回:
            MultipleChoiceModelOutput: 包含多选模型输出的对象
        """
        # 确保返回字典的选项不为 None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取 BERT 模型的输出
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

        # 提取池化输出（<s> token 对应的输出）
        pooled_output = outputs[1]

        # 应用 Dropout
        pooled_output = self.dropout(pooled_output)

        # 通过分类器获取 logits
        logits = self.classifier(pooled_output)

        if labels is not None:
            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            # 多选分类任务需要将 logits 和 labels 转置
            logits = logits.view(-1, self.num_labels)
            labels = labels.view(-1)
            # 计算损失
            loss = loss_fct(logits, labels)
            # 输出损失和 logits
            return MultipleChoiceModelOutput(loss=loss, logits=logits)

        # 返回 logits
        return logits
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )


    # 使用指定的参数注释添加代码示例的文档字符串
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
    ) -> Union[MultipleChoiceModelOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 初始化返回字典，根据配置确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入数据的选择数目
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将输入张量展平为二维张量，以便于后续处理
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用BERT模型处理输入数据
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

        # 获取BERT模型的池化输出
        pooled_output = outputs[1]

        # 对池化输出应用Dropout操作
        pooled_output = self.dropout(pooled_output)
        # 使用分类器计算logits
        logits = self.classifier(pooled_output)
        # 重新整形logits张量以匹配多选择的形状
        reshaped_logits = logits.view(-1, num_choices)

        # 初始化损失为None
        loss = None
        # 如果有提供标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不需要返回字典，则返回扁平化的输出元组
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 否则，返回带有多选择模型输出格式的字典
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用装饰器添加文档字符串，描述了这是一个在BigBird模型基础上进行标记分类的模型，例如用于命名实体识别（NER）任务
@add_start_docstrings(
    """
    BigBird Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BIG_BIRD_START_DOCSTRING,
)
# 定义 BigBirdForTokenClassification 类，继承自 BigBirdPreTrainedModel 类
class BigBirdForTokenClassification(BigBirdPreTrainedModel):
    # 初始化方法，接受一个配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置类别数为配置对象中的类别数
        self.num_labels = config.num_labels

        # 初始化 BigBirdModel 模型
        self.bert = BigBirdModel(config)
        
        # 根据配置设置分类器的 dropout 概率，如果未设置，则使用隐藏层 dropout 概率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 使用 dropout 模块
        self.dropout = nn.Dropout(classifier_dropout)
        # 分类器层，将隐藏状态映射到类别数上
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 使用装饰器添加文档字符串到 forward 方法，描述输入参数的格式
    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例的文档字符串，指定了检查点、输出类型和配置类
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播方法，接受多个输入参数，输出预测结果或损失
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

        # 函数参数说明：
        # - input_ids: 输入的 token IDs
        # - attention_mask: 注意力掩码，指示哪些位置是填充的
        # - token_type_ids: token 类型 IDs，用于区分不同句子的 token
        # - position_ids: 位置 IDs，标识 token 的位置
        # - head_mask: 头部掩码，用于指定哪些注意力头是有效的
        # - inputs_embeds: 嵌入输入，替代输入的 token IDs
        # - labels: 标签，用于训练时的真实类别
        # - output_attentions: 是否输出注意力权重
        # - output_hidden_states: 是否输出隐藏状态
        # - return_dict: 是否以字典形式返回输出
        
        # 返回 TokenClassifierOutput 类型的对象，包含模型的输出结果
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    ) -> Union[TokenClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 初始化返回字典，如果未指定则使用配置中的返回设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用BERT模型处理输入数据，获取输出结果
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

        # 从BERT模型的输出中获取序列输出（通常是最后一层的隐藏状态）
        sequence_output = outputs[0]

        # 对序列输出进行Dropout处理
        sequence_output = self.dropout(sequence_output)
        
        # 使用分类器对处理后的序列输出进行分类得到logits
        logits = self.classifier(sequence_output)

        # 初始化损失为None
        loss = None
        # 如果提供了标签，计算分类损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不要求返回字典格式，则返回一个元组
        if not return_dict:
            output = (logits,) + outputs[2:]  # 只返回logits和额外的输出（隐藏状态等）
            return ((loss,) + output) if loss is not None else output

        # 返回TokenClassifierOutput格式的对象，包括损失、logits、隐藏状态和注意力权重
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
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intermediate = BigBirdIntermediate(config)  # 初始化中间层对象
        self.output = BigBirdOutput(config)  # 初始化输出层对象
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)  # 初始化线性层

    def forward(self, encoder_output):
        hidden_states = self.dropout(encoder_output)  # 应用 dropout 到编码器输出
        hidden_states = self.intermediate(hidden_states)  # 经过中间层处理
        hidden_states = self.output(hidden_states, encoder_output)  # 经过输出层处理，传入编码器输出
        hidden_states = self.qa_outputs(hidden_states)  # 通过线性层计算最终输出
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

        config.num_labels = 2  # 设置类别数量为2
        self.num_labels = config.num_labels
        self.sep_token_id = config.sep_token_id  # 分隔符 token 的 id

        self.bert = BigBirdModel(config, add_pooling_layer=add_pooling_layer)  # 初始化 BigBird 模型
        self.qa_classifier = BigBirdForQuestionAnsweringHead(config)  # 初始化问题回答头部

        # Initialize weights and apply final processing
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
        # 前向传播函数，详细参数见注释中的说明

    @staticmethod
    def prepare_question_mask(q_lengths: torch.Tensor, maxlen: int):
        # 准备问题掩码，根据问题长度和最大长度生成掩码
        mask = torch.arange(0, maxlen).to(q_lengths.device)
        mask.unsqueeze_(0)  # 增加维度
        mask = torch.where(mask < q_lengths, 1, 0)  # 根据长度生成掩码
        return mask
```