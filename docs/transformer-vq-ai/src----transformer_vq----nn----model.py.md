# `transformer_vq\src\transformer_vq\nn\model.py`

```
# 导入必要的模块
import dataclasses  # 用于创建不可变的类
import chex  # 用于 JAX 的检查工具
import flax.linen as nn  # Flax 提供的神经网络模块
import jax  # 用于自动微分和并行计算
import jax.numpy as jnp  # JAX 的 NumPy 替代品

# 从自定义模块中导入相关类和函数
from transformer_vq.nn.attn import VQAttention  # 导入自定义的注意力机制模块
from transformer_vq.nn.emb import Embeddings  # 导入自定义的嵌入层模块
from transformer_vq.nn.norm import LayerNorm  # 导入自定义的层归一化模块
from transformer_vq.nn.pe import ScaledSin  # 导入自定义的位置编码模块
from transformer_vq.nn.types import TransformerConfig  # 导入自定义的 Transformer 配置类
from transformer_vq.nn.vq import VQSpec  # 导入自定义的向量量化规范类

# 定义一个 TransformerLayer 类，继承自 nn.Module
class TransformerLayer(nn.Module):
    config: TransformerConfig  # 类型注解，指定 config 属性的类型为 TransformerConfig

    # 定义 setup 方法
    def setup(self):
        self.apply_config()  # 调用 apply_config 方法，用于初始化 TransformerLayer
# 定义注意力扫描参数字典
attn_scan_args = dict(
    variable_broadcast="params",  # 参数广播
    split_rngs=dict(
        params=False,  # 参数不分割
        timeless=False,  # 非时态不分割
        ephemeral=True,  # 短暂的分割
    ),
    in_axes=0,  # 输入轴为0
    out_axes=0,  # 输出轴为0，因为指标是零维的，所以必须在轴0上堆叠
)
# 使用定义的注意力扫描参数创建扫描对象
self.scanned_attn1 = nn.scan(VQAttention, **attn_scan_args)(self.config)
self.scanned_attn2 = nn.scan(VQAttention, **attn_scan_args)(self.config)

# 定义丢弃层参数字典
drop_kwargs = dict(
    rng_collection="timeless",  # 随机数集合为非时态
    deterministic=not self.is_train,  # 确定性为非训练状态
    broadcast_dims=(0, 2, 3),  # 在除了批处理之外的所有轴上广播
)
# 使用定义的丢弃层参数创建丢弃层对象
self.droplyr1 = nn.Dropout(self.p_droplyr, **drop_kwargs)
self.droplyr2 = nn.Dropout(self.p_droplyr, **drop_kwargs)
# 应用配置信息到当前对象
def apply_config(self):
    # 遍历配置信息的键值对，使用 setattr 方法将配置信息的值赋值给当前对象的属性
    for k, v in dataclasses.asdict(self.config).items():
        setattr(self, k, v)

# 静态方法，用于初始化状态
@staticmethod
def initial_state(config, batch_size):
    # 返回一个包含两个 VQAttention 初始状态的列表
    return [
        VQAttention.initial_state(config=config, batch_size=batch_size),
        VQAttention.initial_state(config=config, batch_size=batch_size),
    ]

# 调用对象时的方法
def __call__(self, x, doc_ids, state, vq_spec):
    # 获取输入 x 的形状信息
    n_block, batch_size, *_ = x.shape
    # 创建一个包含指定维度信息的 Dimensions 对象
    dims = chex.Dimensions(
        K=n_block,
        B=batch_size,
        L=self.block_len,
        D=self.d_model,
    )
# 将state元组中的两个元素分别赋值给state1和state2
state1, state2 = state

# 检查输入x的形状是否符合预期的维度
chex.assert_shape(x, dims["KBLD"])

# 创建包含输入特征x、文档ID和vq_spec的字典
attn1_input_dict = dict(input_features=x, doc_ids=doc_ids, vq_spec=vq_spec)

# 使用state1和attn1_input_dict作为输入，调用self.scanned_attn1方法，返回attn1_state和attn1_output_dict
attn1_state, attn1_output_dict = self.scanned_attn1(state1, attn1_input_dict)

# 从attn1_output_dict中弹出键为"res"的值，并赋给r1
r1 = attn1_output_dict.pop("res")

# 检查r1的形状是否符合预期的维度
chex.assert_shape(r1, dims["KBLD"])

# 将x与self.droplyr1(r1)相加，并赋给x
x += self.droplyr1(r1)

# 对attn1_output_dict中的值应用jnp.mean函数
attn1_output_dict = jax.tree_util.tree_map(jnp.mean, attn1_output_dict)

# 创建包含输入特征x、文档ID和vq_spec的字典
attn2_input_dict = dict(input_features=x, doc_ids=doc_ids, vq_spec=vq_spec)

# 使用state2和attn2_input_dict作为输入，调用self.scanned_attn2方法，返回attn2_state和attn2_output_dict
attn2_state, attn2_output_dict = self.scanned_attn2(state2, attn2_input_dict)

# 从attn2_output_dict中弹出键为"res"的值，并赋给r2
r2 = attn2_output_dict.pop("res")

# 检查r2的形状是否符合预期的维度
chex.assert_shape(r2, dims["KBLD"])

# 将x与self.droplyr2(r2)相加，并赋给x
x += self.droplyr2(r2)

# 对attn2_output_dict中的值应用jnp.mean函数
attn2_output_dict = jax.tree_util.tree_map(jnp.mean, attn2_output_dict)

# 从attn1_output_dict中弹出键为"l_commit"的值，并赋给l_commit
l_commit = attn1_output_dict.pop("l_commit")

# 将attn2_output_dict中弹出键为"l_commit"的值与l_commit相加，并赋给l_commit
l_commit += attn2_output_dict.pop("l_commit")

# 从attn1_output_dict中弹出键为"l_codebook"的值，并赋给l_codebook
l_codebook = attn1_output_dict.pop("l_codebook")
# 将 attn2_output_dict 中的 "l_codebook" 弹出并添加到 l_codebook 中
l_codebook += attn2_output_dict.pop("l_codebook")

# 使用 tree_map 函数对 attn1_output_dict 和 attn2_output_dict 中的 "metrics" 进行加权平均操作
metric_dict = jax.tree_util.tree_map(
    lambda a, b: (a + b) / 2,
    attn1_output_dict.pop("metrics"),
    attn2_output_dict.pop("metrics"),
)

# 返回包含输出特征、注意力状态、l_commit、l_codebook 和 metrics 的字典
return dict(
    output_features=x,
    attn_state=[attn1_state, attn2_state],
    l_commit=l_commit,
    l_codebook=l_codebook,
    metrics=metric_dict,
)

# 定义一个 Transformer 类，继承自 nn.Module
class Transformer(nn.Module):
    # 定义一个属性 config，类型为 TransformerConfig
    config: TransformerConfig

    # 定义一个方法 setup，用于应用配置
    def setup(self):
        self.apply_config()
# 如果不是无嵌入或者嵌入层被绑定，则创建一个嵌入层对象
if not self.no_emb or self.e_tie:
    self.token_embedder = Embeddings(self.config)

# 如果使用绝对位置编码，则创建一个基于正弦函数的位置编码对象
if self.pe_abs:
    self.position_embedder = ScaledSin(self.config)

# 创建多个TransformerLayer对象，组成transformer_layers列表
self.transformer_layers = [
    nn.remat(TransformerLayer)(self.config) for _ in range(self.n_layer)
]

# 如果使用预层归一化，则创建一个LayerNorm对象
if self.e_preln:
    self.out_ln = LayerNorm(self.d_model, self.param_dtype)

# 如果嵌入层没有被绑定，则创建一个Dense对象作为输出投影层
if not self.e_tie:
    self.out_proj = nn.Dense(
        self.n_vocab,
        use_bias=True,
        kernel_init=self.w_init,
        bias_init=self.b_init,
        param_dtype=self.param_dtype,
        dtype=self.param_dtype,  # always use full-precision logits
    )

# 创建一个Dropout对象作为嵌入层的dropout
drop_kwargs = dict(rng_collection="ephemeral", deterministic=not self.is_train)
self.dropemb = nn.Dropout(self.p_dropemb, **drop_kwargs)
# 应用配置信息到当前对象
def apply_config(self):
    # 遍历配置信息的属性和值，使用 setattr 方法将值赋给当前对象的属性
    for k, v in dataclasses.asdict(self.config).items():
        setattr(self, k, v)

# 返回初始状态列表
@staticmethod
def initial_state(config, batch_size):
    # 使用 TransformerLayer 类的 initial_state 方法初始化状态，返回状态列表
    return [
        TransformerLayer.initial_state(
            config=config,
            batch_size=batch_size,
        )
        for _ in range(config.n_layer)
    ]

# 获取维度信息
def get_chex_dims(self, batch_size, present_len):
    # 返回 Dimensions 对象，包含 B、P、K 三个维度
    return chex.Dimensions(
        B=batch_size,
        P=present_len,
        K=present_len // self.block_len,
    )
    # 初始化模型参数，包括块长度、模型维度和词汇表大小
    def __init__(
        self,
        L=self.block_len,
        D=self.d_model,
        V=self.n_vocab,
    )

    # 从序列中获取块
    def get_blocks_from_sequence(self, x):
        # 获取批处理大小、当前长度和后缀
        batch_size, present_len, *suffix = x.shape
        # 计算块的数量
        n_block = present_len // self.block_len
        # 重新整形输入序列
        x = jnp.reshape(x, [batch_size, n_block, self.block_len, *suffix])
        # 调整维度顺序
        suffix_axes = list(range(3, x.ndim))
        x = jnp.transpose(x, (1, 0, 2, *suffix_axes))
        return x

    # 从块中获取序列
    def get_sequence_from_blocks(self, x):
        # 获取块的数量、批处理大小、块长度和后缀
        num_block, batch_size, block_len, *suffix = x.shape
        # 调整维度顺序
        suffix_axes = list(range(3, x.ndim))
        x = jnp.transpose(x, (1, 0, 2, *suffix_axes))
        # 重新整形输出序列
        x = jnp.reshape(x, [batch_size, num_block * block_len, *suffix])
        return x
# 获取给定 VQ 规范的块
def get_blocks_of_vq_spec(self, vq_spec):
    # 如果 VQ 规范为空，则返回空
    if vq_spec is None:
        return None

    # 检查 VQ 规范的损失掩码的维度是否为2
    chex.assert_rank(vq_spec.loss_mask, 2)
    # 计算块的数量
    n_block = vq_spec.loss_mask.shape[1] // self.block_len

    # 定义一个函数，用于扩展和平铺数组
    def expand_and_tile(array):
        mult = [n_block] + [1 for _ in range(jnp.ndim(array))]
        return jnp.tile(array[None, ...], mult)

    # 创建一个新的 VQ 规范对象，其中包括设备数量、每次更新的块数和重新排列后的损失掩码
    return VQSpec.create(
        n_device=expand_and_tile(vq_spec.n_device),
        n_block_per_update=expand_and_tile(vq_spec.n_block_per_update),
        loss_mask=jnp.transpose(
            jnp.reshape(vq_spec.loss_mask, [-1, n_block, self.block_len]),
            (1, 0, 2),
        ),
    )
# 静态方法，用于可能聚合累加器字典和新字典
@staticmethod
def maybe_aggregate(accumulator_dict, new_dict):
    # 如果累加器字典为空，则返回新字典
    if len(accumulator_dict) == 0:
        return new_dict
    # 如果新字典为空，则返回累加器字典
    if len(new_dict) == 0:
        return accumulator_dict
    # 如果累加器字典和新字典都非空，则对它们进行元素级别的加法操作
    return jax.tree_util.tree_map(lambda a, b: a + b, accumulator_dict, new_dict)

# 静态方法，用于计算层度量的平均值
@staticmethod
def average_layer_metrics(aux, n_layer):
    # 如果辅助字典中没有"metrics"键，则直接返回辅助字典
    if "metrics" not in aux:
        return aux
    # 从辅助字典中弹出"metrics"键对应的值，并对其进行元素级别的除法操作
    metrics = aux.pop("metrics")
    metrics = jax.tree_util.tree_map(lambda y: y / n_layer, metrics)
    # 创建新的辅助字典，包含更新后的"metrics"值和原始辅助字典的其他内容
    new_aux = dict(metrics=metrics, **aux)
    return new_aux
# 定义一个函数，接受输入、文档ID、状态和vq_spec作为参数
def __call__(self, inputs, doc_ids, state, vq_spec):
    # 获取输入的批量大小、展示长度和其他维度信息
    batch_size, present_len, *_ = inputs.shape
    # 获取BP维度的尺寸信息
    dims = self.get_chex_dims(batch_size, present_len)
    # 检查doc_ids的形状是否符合BP维度的要求
    chex.assert_shape(doc_ids, dims["BP"])
    # 初始化新状态和辅助变量
    new_state = []
    aux = {}
    # 将输入赋值给x
    x = inputs
    # 如果不是无嵌入模式，则进行标记嵌入
    if not self.no_emb:
        x = self.token_embedder(x)
    # 如果使用绝对位置编码，则计算偏移量并添加位置编码
    if self.pe_abs:
        offset = state[0][0]["pos_offset"]
        x += self.position_embedder(length=present_len, offset=offset)
    # 对输入进行嵌入层的dropout
    x = self.dropemb(x)
    # 获取序列的块
    x = self.get_blocks_from_sequence(x)
    doc_ids = self.get_blocks_from_sequence(doc_ids)
    vq_spec = self.get_blocks_of_vq_spec(vq_spec)
    # 检查x的形状是否符合KBLD维度的要求
    chex.assert_shape(x, dims["KBLD"])
    # 遍历每个层
    for i in range(self.n_layer):
        # 对每一层进行transformer操作，得到输出字典
        layer_output_dict = self.transformer_layers[i](
            x=x, doc_ids=doc_ids, state=state[i], vq_spec=vq_spec
        )
        # 从 layer_output_dict 中弹出 "attn_state" 并添加到 new_state 列表中
        new_state.append(layer_output_dict.pop("attn_state"))
        # 从 layer_output_dict 中弹出 "output_features" 赋值给 x
        x = layer_output_dict.pop("output_features")
        # 检查 x 的形状是否符合预期的维度
        chex.assert_shape(x, dims["KBLD"])
        # 对辅助信息进行可能的聚合操作
        aux = Transformer.maybe_aggregate(aux, layer_output_dict)
    # 从 blocks 中获取序列信息
    x = self.get_sequence_from_blocks(x)
    # 对辅助信息进行平均层度量
    aux = Transformer.average_layer_metrics(aux, self.n_layer)
    # 如果使用预层归一化，则对 x 进行层归一化处理
    if self.e_preln:
        x = self.out_ln(x)
    # 如果启用了 token embedding 的 logits，则对 x 进行 logits 处理；否则对 x 进行输出投影处理
    x = self.token_embedder.logits(x) if self.e_tie else self.out_proj(x)
    # 对 x 进行缩放处理
    x *= self.e_scale
    # 对 x 进行对数 softmax 处理
    x = jax.nn.log_softmax(x, axis=-1)
    # 检查 x 的形状是否符合预期的维度
    chex.assert_shape(x, dims["BPV"])
    # 返回包含 logprobs, attn_state 和 aux 的字典
    return dict(logprobs=x, attn_state=new_state, **aux)
```