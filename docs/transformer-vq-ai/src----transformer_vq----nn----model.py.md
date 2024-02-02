# `transformer_vq\src\transformer_vq\nn\model.py`

```py
# 导入必要的模块
import dataclasses
import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from transformer_vq.nn.attn import VQAttention
from transformer_vq.nn.emb import Embeddings
from transformer_vq.nn.norm import LayerNorm
from transformer_vq.nn.pe import ScaledSin
from transformer_vq.nn.types import TransformerConfig
from transformer_vq.nn.vq import VQSpec

# 定义一个名为TransformerLayer的类，继承自nn.Module
class TransformerLayer(nn.Module):
    # 定义一个名为config的属性，类型为TransformerConfig
    config: TransformerConfig

    # 初始化方法
    def setup(self):
        # 调用apply_config方法，将config中的属性应用到当前对象中
        self.apply_config()
        # 定义一个名为attn_scan_args的字典，包含一些参数设置
        attn_scan_args = dict(
            variable_broadcast="params",
            split_rngs=dict(
                params=False,
                timeless=False,
                ephemeral=True,
            ),
            in_axes=0,
            out_axes=0,  # metrics are zero-dimensional, so have to stack on axis 0
        )
        # 使用nn.scan函数创建一个扫描VQAttention的对象，并赋值给self.scanned_attn1
        self.scanned_attn1 = nn.scan(VQAttention, **attn_scan_args)(self.config)
        # 使用nn.scan函数创建一个扫描VQAttention的对象，并赋值给self.scanned_attn2
        self.scanned_attn2 = nn.scan(VQAttention, **attn_scan_args)(self.config)

        # 定义一个名为drop_kwargs的字典，包含一些参数设置
        drop_kwargs = dict(
            rng_collection="timeless",
            deterministic=not self.is_train,
            broadcast_dims=(0, 2, 3),  # broadcast over all axes except batch
        )
        # 使用nn.Dropout函数创建一个Dropout层，并赋值给self.droplyr1
        self.droplyr1 = nn.Dropout(self.p_droplyr, **drop_kwargs)
        # 使用nn.Dropout函数创建一个Dropout层，并赋值给self.droplyr2
        self.droplyr2 = nn.Dropout(self.p_droplyr, **drop_kwargs)

    # 定义一个名为apply_config的方法
    def apply_config(self):
        # 遍历config中的属性，并将其值设置到当前对象中
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    # 定义一个静态方法initial_state，接受config和batch_size两个参数
    @staticmethod
    def initial_state(config, batch_size):
        # 返回一个包含两个VQAttention初始状态的列表
        return [
            VQAttention.initial_state(config=config, batch_size=batch_size),
            VQAttention.initial_state(config=config, batch_size=batch_size),
        ]
    # 定义一个方法，接受输入参数 x, doc_ids, state, vq_spec
    def __call__(self, x, doc_ids, state, vq_spec):
        # 获取输入 x 的形状信息
        n_block, batch_size, *_ = x.shape
        # 定义输入 x 的维度信息
        dims = chex.Dimensions(
            K=n_block,
            B=batch_size,
            L=self.block_len,
            D=self.d_model,
        )
        # 将 state 拆分为 state1 和 state2
        state1, state2 = state

        # 检查输入 x 的形状是否符合预期
        chex.assert_shape(x, dims["KBLD"])
        # 构建第一个注意力机制的输入字典
        attn1_input_dict = dict(input_features=x, doc_ids=doc_ids, vq_spec=vq_spec)
        # 使用第一个注意力机制处理输入数据
        attn1_state, attn1_output_dict = self.scanned_attn1(state1, attn1_input_dict)
        # 获取第一个注意力机制的输出结果
        r1 = attn1_output_dict.pop("res")
        # 检查输出结果的形状是否符合预期
        chex.assert_shape(r1, dims["KBLD"])
        # 将输入 x 与第一个注意力机制的输出结果相加
        x += self.droplyr1(r1)
        # 对第一个注意力机制的输出字典进行处理
        attn1_output_dict = jax.tree_util.tree_map(jnp.mean, attn1_output_dict)

        # 构建第二个注意力机制的输入字典
        attn2_input_dict = dict(input_features=x, doc_ids=doc_ids, vq_spec=vq_spec)
        # 使用第二个注意力机制处理输入数据
        attn2_state, attn2_output_dict = self.scanned_attn2(state2, attn2_input_dict)
        # 获取第二个注意力机制的输出结果
        r2 = attn2_output_dict.pop("res")
        # 检查输出结果的形状是否符合预期
        chex.assert_shape(r2, dims["KBLD"])
        # 将输入 x 与第二个注意力机制的输出结果相加
        x += self.droplyr2(r2)
        # 对第二个注意力机制的输出字典进行处理
        attn2_output_dict = jax.tree_util.tree_map(jnp.mean, attn2_output_dict)

        # 合并两个注意力机制的输出 l_commit
        l_commit = attn1_output_dict.pop("l_commit")
        l_commit += attn2_output_dict.pop("l_commit")
        # 合并两个注意力机制的输出 l_codebook
        l_codebook = attn1_output_dict.pop("l_codebook")
        l_codebook += attn2_output_dict.pop("l_codebook")
        # 合并两个注意力机制的输出 metrics
        metric_dict = jax.tree_util.tree_map(
            lambda a, b: (a + b) / 2,
            attn1_output_dict.pop("metrics"),
            attn2_output_dict.pop("metrics"),
        )
        # 返回结果字典
        return dict(
            output_features=x,
            attn_state=[attn1_state, attn2_state],
            l_commit=l_commit,
            l_codebook=l_codebook,
            metrics=metric_dict,
        )
class Transformer(nn.Module):
    config: TransformerConfig  # 定义一个名为config的TransformerConfig类型的属性

    def setup(self):  # 定义一个名为setup的方法
        self.apply_config()  # 调用apply_config方法
        if not self.no_emb or self.e_tie:  # 如果no_emb属性为假或者e_tie属性为真
            self.token_embedder = Embeddings(self.config)  # 创建一个名为token_embedder的Embeddings对象
        if self.pe_abs:  # 如果pe_abs属性为真
            self.position_embedder = ScaledSin(self.config)  # 创建一个名为position_embedder的ScaledSin对象
        self.transformer_layers = [  # 创建一个名为transformer_layers的列表
            nn.remat(TransformerLayer)(self.config) for _ in range(self.n_layer)  # 使用列表推导式创建TransformerLayer对象并添加到列表中
        ]
        if self.e_preln:  # 如果e_preln属性为真
            self.out_ln = LayerNorm(self.d_model, self.param_dtype)  # 创建一个名为out_ln的LayerNorm对象
        if not self.e_tie:  # 如果e_tie属性为假
            self.out_proj = nn.Dense(  # 创建一个名为out_proj的nn.Dense对象
                self.n_vocab,  # 设置n_vocab属性
                use_bias=True,  # 设置use_bias属性为真
                kernel_init=self.w_init,  # 设置kernel_init属性为w_init
                bias_init=self.b_init,  # 设置bias_init属性为b_init
                param_dtype=self.param_dtype,  # 设置param_dtype属性为param_dtype
                dtype=self.param_dtype,  # 设置dtype属性为param_dtype
            )
        drop_kwargs = dict(rng_collection="ephemeral", deterministic=not self.is_train)  # 创建一个名为drop_kwargs的字典
        self.dropemb = nn.Dropout(self.p_dropemb, **drop_kwargs)  # 创建一个名为dropemb的nn.Dropout对象

    def apply_config(self):  # 定义一个名为apply_config的方法
        for k, v in dataclasses.asdict(self.config).items():  # 遍历config属性转换成字典后的键值对
            setattr(self, k, v)  # 设置self对象的属性

    @staticmethod
    def initial_state(config, batch_size):  # 定义一个静态方法initial_state，接受config和batch_size两个参数
        return [  # 返回一个列表
            TransformerLayer.initial_state(  # 调用TransformerLayer的initial_state方法
                config=config,  # 传入config参数
                batch_size=batch_size,  # 传入batch_size参数
            )
            for _ in range(config.n_layer)  # 使用列表推导式创建TransformerLayer对象并添加到列表中
        ]

    def get_chex_dims(self, batch_size, present_len):  # 定义一个名为get_chex_dims的方法，接受batch_size和present_len两个参数
        return chex.Dimensions(  # 返回一个chex.Dimensions对象
            B=batch_size,  # 设置B属性为batch_size
            P=present_len,  # 设置P属性为present_len
            K=present_len // self.block_len,  # 设置K属性为present_len除以block_len的结果
            L=self.block_len,  # 设置L属性为block_len
            D=self.d_model,  # 设置D属性为d_model
            V=self.n_vocab,  # 设置V属性为n_vocab
        )
    # 从序列中获取块，输入参数 x 为序列
    def get_blocks_from_sequence(self, x):
        # 获取批处理大小、当前长度和后缀
        batch_size, present_len, *suffix = x.shape
        # 计算块的数量
        n_block = present_len // self.block_len
        # 重新整形输入序列，将其分成块
        x = jnp.reshape(x, [batch_size, n_block, self.block_len, *suffix])
        # 调整张量的维度顺序
        suffix_axes = list(range(3, x.ndim))
        x = jnp.transpose(x, (1, 0, 2, *suffix_axes))
        # 返回块
        return x

    # 从块中获取序列，输入参数 x 为块
    def get_sequence_from_blocks(self, x):
        # 获取块的数量、批处理大小、块长度和后缀
        num_block, batch_size, block_len, *suffix = x.shape
        # 调整张量的维度顺序
        suffix_axes = list(range(3, x.ndim))
        x = jnp.transpose(x, (1, 0, 2, *suffix_axes))
        # 重新整形块，将其合并成序列
        x = jnp.reshape(x, [batch_size, num_block * block_len, *suffix])
        # 返回序列
        return x

    # 从 VQ 规范中获取块
    def get_blocks_of_vq_spec(self, vq_spec):
        # 如果 VQ 规范为空，则返回空
        if vq_spec is None:
            return None
        # 检查 VQ 规范的 loss_mask 的秩为 2
        chex.assert_rank(vq_spec.loss_mask, 2)
        # 计算块的数量
        n_block = vq_spec.loss_mask.shape[1] // self.block_len

        # 定义扩展和平铺函数
        def expand_and_tile(array):
            mult = [n_block] + [1 for _ in range(jnp.ndim(array))]
            return jnp.tile(array[None, ...], mult)

        # 创建新的 VQ 规范对象
        return VQSpec.create(
            n_device=expand_and_tile(vq_spec.n_device),
            n_block_per_update=expand_and_tile(vq_spec.n_block_per_update),
            loss_mask=jnp.transpose(
                jnp.reshape(vq_spec.loss_mask, [-1, n_block, self.block_len]),
                (1, 0, 2),
            ),
        )

    # 可能进行聚合操作的静态方法
    @staticmethod
    def maybe_aggregate(accumulator_dict, new_dict):
        # 如果旧字典为空，则返回新字典
        if len(accumulator_dict) == 0:
            return new_dict
        # 如果新字典为空，则返回旧字典
        if len(new_dict) == 0:
            return accumulator_dict
        # 如果旧字典和新字典都不为空，则对它们进行聚合操作
        return jax.tree_util.tree_map(lambda a, b: a + b, accumulator_dict, new_dict)

    @staticmethod
    # 计算每一层的平均指标
    def average_layer_metrics(aux, n_layer):
        # 如果辅助信息中没有指标数据，则直接返回辅助信息
        if "metrics" not in aux:
            return aux
        # 从辅助信息中取出指标数据
        metrics = aux.pop("metrics")
        # 将指标数据中的每个值除以层数，得到平均指标数据
        metrics = jax.tree_util.tree_map(lambda y: y / n_layer, metrics)
        # 将平均指标数据和其他辅助信息合并成新的辅助信息
        new_aux = dict(metrics=metrics, **aux)
        return new_aux

    # 模型的调用方法
    def __call__(self, inputs, doc_ids, state, vq_spec):
        # 获取输入数据的形状信息
        batch_size, present_len, *_ = inputs.shape
        # 获取输入数据的维度信息
        dims = self.get_chex_dims(batch_size, present_len)
        # 检查文档 ID 的形状是否符合要求
        chex.assert_shape(doc_ids, dims["BP"])
        # 初始化新的状态和辅助信息
        new_state = []
        aux = {}
        x = inputs
        # 如果不是无嵌入模式，则进行标记嵌入
        if not self.no_emb:
            x = self.token_embedder(x)
        # 如果使用绝对位置编码，则计算偏移量并添加位置编码
        if self.pe_abs:
            offset = state[0][0]["pos_offset"]
            x += self.position_embedder(length=present_len, offset=offset)
        # 对输入数据进行嵌入层的 dropout
        x = self.dropemb(x)
        # 获取序列的块信息
        x = self.get_blocks_from_sequence(x)
        doc_ids = self.get_blocks_from_sequence(doc_ids)
        vq_spec = self.get_blocks_of_vq_spec(vq_spec)
        # 检查输入数据的形状是否符合要求
        chex.assert_shape(x, dims["KBLD"])
        # 遍历每一层的 transformer
        for i in range(self.n_layer):
            # 对输入数据进行 transformer 层的处理
            layer_output_dict = self.transformer_layers[i](
                x=x, doc_ids=doc_ids, state=state[i], vq_spec=vq_spec
            )
            # 将当前层的注意力状态添加到新的状态中
            new_state.append(layer_output_dict.pop("attn_state"))
            # 更新输入数据为当前层的输出特征
            x = layer_output_dict.pop("output_features")
            # 检查输入数据的形状是否符合要求
            chex.assert_shape(x, dims["KBLD"])
            # 将当前层的输出信息合并到辅助信息中
            aux = Transformer.maybe_aggregate(aux, layer_output_dict)
        # 将块信息转换为序列信息
        x = self.get_sequence_from_blocks(x)
        # 计算每一层的平均指标
        aux = Transformer.average_layer_metrics(aux, self.n_layer)
        # 如果使用预层归一化，则对输出进行层归一化
        if self.e_preln:
            x = self.out_ln(x)
        # 如果使用标记嵌入的连接权重，则对输出进行标记嵌入的 logits 计算
        x = self.token_embedder.logits(x) if self.e_tie else self.out_proj(x)
        # 对输出进行缩放
        x *= self.e_scale
        # 对输出进行 log_softmax 处理
        x = jax.nn.log_softmax(x, axis=-1)
        # 检查输出数据的形状是否符合要求
        chex.assert_shape(x, dims["BPV"])
        # 返回结果字典，包括 logprobs、注意力状态和其他辅助信息
        return dict(logprobs=x, attn_state=new_state, **aux)
```