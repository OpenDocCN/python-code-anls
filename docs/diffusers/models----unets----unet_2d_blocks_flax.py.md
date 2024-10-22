# `.\diffusers\models\unets\unet_2d_blocks_flax.py`

```py
# 版权声明，说明该文件的版权信息及相关许可协议
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 许可信息，使用 Apache License 2.0 许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 该文件只能在符合许可证的情况下使用
# you may not use this file except in compliance with the License.
# 提供许可证获取链接
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 免责声明，表明不提供任何形式的保证或条件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 许可证的相关权限及限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入 flax.linen 模块，用于构建神经网络
import flax.linen as nn
# 导入 jax.numpy，用于数值计算
import jax.numpy as jnp

# 从其他模块导入特定的类，用于构建模型的各个组件
from ..attention_flax import FlaxTransformer2DModel
from ..resnet_flax import FlaxDownsample2D, FlaxResnetBlock2D, FlaxUpsample2D


# 定义 FlaxCrossAttnDownBlock2D 类，表示一个 2D 跨注意力下采样模块
class FlaxCrossAttnDownBlock2D(nn.Module):
    r"""
    跨注意力 2D 下采样块 - 原始架构来自 Unet transformers:
    https://arxiv.org/abs/2103.06104

    参数说明：
        in_channels (:obj:`int`):
            输入通道数
        out_channels (:obj:`int`):
            输出通道数
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout 率
        num_layers (:obj:`int`, *optional*, defaults to 1):
            注意力块层数
        num_attention_heads (:obj:`int`, *optional*, defaults to 1):
            每个空间变换块的注意力头数
        add_downsample (:obj:`bool`, *optional*, defaults to `True`):
            是否在每个最终输出之前添加下采样层
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            启用内存高效的注意力 https://arxiv.org/abs/2112.05682
        split_head_dim (`bool`, *optional*, defaults to `False`):
            是否将头维度拆分为一个新的轴进行自注意力计算。在大多数情况下，
            启用此标志应加快 Stable Diffusion 2.x 和 Stable Diffusion XL 的计算速度。
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            参数的数据类型
    """

    # 定义输入通道数
    in_channels: int
    # 定义输出通道数
    out_channels: int
    # 定义 Dropout 率，默认为 0.0
    dropout: float = 0.0
    # 定义注意力块的层数，默认为 1
    num_layers: int = 1
    # 定义注意力头数，默认为 1
    num_attention_heads: int = 1
    # 定义是否添加下采样层，默认为 True
    add_downsample: bool = True
    # 定义是否使用线性投影，默认为 False
    use_linear_projection: bool = False
    # 定义是否仅使用跨注意力，默认为 False
    only_cross_attention: bool = False
    # 定义是否启用内存高效注意力，默认为 False
    use_memory_efficient_attention: bool = False
    # 定义是否拆分头维度，默认为 False
    split_head_dim: bool = False
    # 定义参数的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 定义每个块的变换器层数，默认为 1
    transformer_layers_per_block: int = 1
    # 设置模型的各个组成部分，包括残差块和注意力块
        def setup(self):
            # 初始化残差块列表
            resnets = []
            # 初始化注意力块列表
            attentions = []
    
            # 遍历每一层，构建残差块和注意力块
            for i in range(self.num_layers):
                # 第一层的输入通道为 in_channels，其他层为 out_channels
                in_channels = self.in_channels if i == 0 else self.out_channels
    
                # 创建一个 FlaxResnetBlock2D 实例
                res_block = FlaxResnetBlock2D(
                    in_channels=in_channels,  # 输入通道
                    out_channels=self.out_channels,  # 输出通道
                    dropout_prob=self.dropout,  # 丢弃率
                    dtype=self.dtype,  # 数据类型
                )
                # 将残差块添加到列表中
                resnets.append(res_block)
    
                # 创建一个 FlaxTransformer2DModel 实例
                attn_block = FlaxTransformer2DModel(
                    in_channels=self.out_channels,  # 输入通道
                    n_heads=self.num_attention_heads,  # 注意力头数
                    d_head=self.out_channels // self.num_attention_heads,  # 每个头的维度
                    depth=self.transformer_layers_per_block,  # 每个块的层数
                    use_linear_projection=self.use_linear_projection,  # 是否使用线性投影
                    only_cross_attention=self.only_cross_attention,  # 是否只使用交叉注意力
                    use_memory_efficient_attention=self.use_memory_efficient_attention,  # 是否使用内存高效的注意力
                    split_head_dim=self.split_head_dim,  # 是否拆分头的维度
                    dtype=self.dtype,  # 数据类型
                )
                # 将注意力块添加到列表中
                attentions.append(attn_block)
    
            # 将残差块列表赋值给实例变量
            self.resnets = resnets
            # 将注意力块列表赋值给实例变量
            self.attentions = attentions
    
            # 如果需要下采样，则创建下采样层
            if self.add_downsample:
                self.downsamplers_0 = FlaxDownsample2D(self.out_channels, dtype=self.dtype)
    
        # 定义前向调用方法，处理隐藏状态和编码器隐藏状态
        def __call__(self, hidden_states, temb, encoder_hidden_states, deterministic=True):
            # 初始化输出状态元组
            output_states = ()
    
            # 遍历残差块和注意力块并进行处理
            for resnet, attn in zip(self.resnets, self.attentions):
                # 通过残差块处理隐藏状态
                hidden_states = resnet(hidden_states, temb, deterministic=deterministic)
                # 通过注意力块处理隐藏状态
                hidden_states = attn(hidden_states, encoder_hidden_states, deterministic=deterministic)
                # 将当前隐藏状态添加到输出状态元组中
                output_states += (hidden_states,)
    
            # 如果需要下采样，则进行下采样
            if self.add_downsample:
                hidden_states = self.downsamplers_0(hidden_states)
                # 将下采样后的隐藏状态添加到输出状态元组中
                output_states += (hidden_states,)
    
            # 返回最终的隐藏状态和输出状态元组
            return hidden_states, output_states
# 定义 Flax 2D 降维块类，继承自 nn.Module
class FlaxDownBlock2D(nn.Module):
    r"""
    Flax 2D downsizing block

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        add_downsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add downsampling layer before each final output
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    
    # 声明输入输出通道和其他参数
    in_channels: int
    out_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    add_downsample: bool = True
    dtype: jnp.dtype = jnp.float32

    # 设置方法，用于初始化模型的层
    def setup(self):
        # 创建空列表以存储残差块
        resnets = []

        # 根据层数创建残差块
        for i in range(self.num_layers):
            # 第一个块的输入通道为 in_channels，其余为 out_channels
            in_channels = self.in_channels if i == 0 else self.out_channels

            # 创建残差块实例
            res_block = FlaxResnetBlock2D(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
            # 将残差块添加到列表中
            resnets.append(res_block)
        # 将列表赋值给实例属性
        self.resnets = resnets

        # 如果需要，添加降采样层
        if self.add_downsample:
            self.downsamplers_0 = FlaxDownsample2D(self.out_channels, dtype=self.dtype)

    # 调用方法，执行前向传播
    def __call__(self, hidden_states, temb, deterministic=True):
        # 创建空元组以存储输出状态
        output_states = ()

        # 遍历所有残差块进行前向传播
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, deterministic=deterministic)
            # 将当前隐藏状态添加到输出状态中
            output_states += (hidden_states,)

        # 如果需要，应用降采样层
        if self.add_downsample:
            hidden_states = self.downsamplers_0(hidden_states)
            # 将降采样后的隐藏状态添加到输出状态中
            output_states += (hidden_states,)

        # 返回最终的隐藏状态和输出状态
        return hidden_states, output_states


# 定义 Flax 交叉注意力 2D 上采样块类，继承自 nn.Module
class FlaxCrossAttnUpBlock2D(nn.Module):
    r"""
    Cross Attention 2D Upsampling block - original architecture from Unet transformers:
    https://arxiv.org/abs/2103.06104
    # 定义参数的文档字符串，描述各个参数的用途和类型
        Parameters:
            in_channels (:obj:`int`):  # 输入通道数
                Input channels
            out_channels (:obj:`int`):  # 输出通道数
                Output channels
            dropout (:obj:`float`, *optional*, defaults to 0.0):  # Dropout 率，默认值为 0.0
                Dropout rate
            num_layers (:obj:`int`, *optional*, defaults to 1):  # 注意力块的层数，默认值为 1
                Number of attention blocks layers
            num_attention_heads (:obj:`int`, *optional*, defaults to 1):  # 每个空间变换块的注意力头数量，默认值为 1
                Number of attention heads of each spatial transformer block
            add_upsample (:obj:`bool`, *optional*, defaults to `True`):  # 是否在每个最终输出前添加上采样层，默认值为 True
                Whether to add upsampling layer before each final output
            use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):  # 启用内存高效注意力，默认值为 False
                enable memory efficient attention https://arxiv.org/abs/2112.05682
            split_head_dim (`bool`, *optional*, defaults to `False`):  # 是否将头维度拆分为新轴以进行自注意力计算，默认值为 False
                Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
                enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
            dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):  # 数据类型参数，默认值为 jnp.float32
                Parameters `dtype`
        """
    
        in_channels: int  # 输入通道数的声明
        out_channels: int  # 输出通道数的声明
        prev_output_channel: int  # 前一个输出通道数的声明
        dropout: float = 0.0  # Dropout 率的声明，默认值为 0.0
        num_layers: int = 1  # 注意力层数的声明，默认值为 1
        num_attention_heads: int = 1  # 注意力头数量的声明，默认值为 1
        add_upsample: bool = True  # 是否添加上采样的声明，默认值为 True
        use_linear_projection: bool = False  # 是否使用线性投影的声明，默认值为 False
        only_cross_attention: bool = False  # 是否仅使用交叉注意力的声明，默认值为 False
        use_memory_efficient_attention: bool = False  # 是否启用内存高效注意力的声明，默认值为 False
        split_head_dim: bool = False  # 是否拆分头维度的声明，默认值为 False
        dtype: jnp.dtype = jnp.float32  # 数据类型的声明，默认值为 jnp.float32
        transformer_layers_per_block: int = 1  # 每个块的变换层数的声明，默认值为 1
    # 设置方法，初始化网络结构
    def setup(self):
        # 初始化空列表以存储 ResNet 块
        resnets = []
        # 初始化空列表以存储注意力块
        attentions = []
    
        # 遍历每一层以创建相应的 ResNet 和注意力块
        for i in range(self.num_layers):
            # 设置跳跃连接的通道数，最后一层使用输入通道，否则使用输出通道
            res_skip_channels = self.in_channels if (i == self.num_layers - 1) else self.out_channels
            # 设置当前 ResNet 块的输入通道，第一层使用前一层的输出通道
            resnet_in_channels = self.prev_output_channel if i == 0 else self.out_channels
    
            # 创建 FlaxResnetBlock2D 实例
            res_block = FlaxResnetBlock2D(
                # 设置输入通道为当前 ResNet 块输入通道加跳跃连接通道
                in_channels=resnet_in_channels + res_skip_channels,
                # 设置输出通道为指定的输出通道
                out_channels=self.out_channels,
                # 设置 dropout 概率
                dropout_prob=self.dropout,
                # 设置数据类型
                dtype=self.dtype,
            )
            # 将创建的 ResNet 块添加到列表中
            resnets.append(res_block)
    
            # 创建 FlaxTransformer2DModel 实例
            attn_block = FlaxTransformer2DModel(
                # 设置输入通道为输出通道
                in_channels=self.out_channels,
                # 设置注意力头数
                n_heads=self.num_attention_heads,
                # 设置每个注意力头的维度
                d_head=self.out_channels // self.num_attention_heads,
                # 设置 transformer 块的深度
                depth=self.transformer_layers_per_block,
                # 设置是否使用线性投影
                use_linear_projection=self.use_linear_projection,
                # 设置是否仅使用交叉注意力
                only_cross_attention=self.only_cross_attention,
                # 设置是否使用内存高效的注意力机制
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                # 设置是否分割头部维度
                split_head_dim=self.split_head_dim,
                # 设置数据类型
                dtype=self.dtype,
            )
            # 将创建的注意力块添加到列表中
            attentions.append(attn_block)
    
        # 将 ResNet 列表保存到实例属性
        self.resnets = resnets
        # 将注意力列表保存到实例属性
        self.attentions = attentions
    
        # 如果需要添加上采样层，则创建相应的 FlaxUpsample2D 实例
        if self.add_upsample:
            self.upsamplers_0 = FlaxUpsample2D(self.out_channels, dtype=self.dtype)
    
    # 定义调用方法，接受隐藏状态和其他参数
    def __call__(self, hidden_states, res_hidden_states_tuple, temb, encoder_hidden_states, deterministic=True):
        # 遍历 ResNet 和注意力块
        for resnet, attn in zip(self.resnets, self.attentions):
            # 从跳跃连接的隐藏状态元组中取出最后一个状态
            res_hidden_states = res_hidden_states_tuple[-1]
            # 更新跳跃连接的隐藏状态元组，去掉最后一个状态
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            # 将隐藏状态与跳跃连接的隐藏状态在最后一个轴上拼接
            hidden_states = jnp.concatenate((hidden_states, res_hidden_states), axis=-1)
    
            # 使用当前的 ResNet 块处理隐藏状态
            hidden_states = resnet(hidden_states, temb, deterministic=deterministic)
            # 使用当前的注意力块处理隐藏状态
            hidden_states = attn(hidden_states, encoder_hidden_states, deterministic=deterministic)
    
        # 如果需要添加上采样，则使用上采样层处理隐藏状态
        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states)
    
        # 返回处理后的隐藏状态
        return hidden_states
# 定义一个 2D 上采样块类，继承自 nn.Module
class FlaxUpBlock2D(nn.Module):
    r"""
    Flax 2D upsampling block

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        prev_output_channel (:obj:`int`):
            Output channels from the previous block
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        add_downsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add downsampling layer before each final output
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    # 定义输入输出通道和其他参数
    in_channels: int
    out_channels: int
    prev_output_channel: int
    dropout: float = 0.0
    num_layers: int = 1
    add_upsample: bool = True
    dtype: jnp.dtype = jnp.float32

    # 设置方法用于初始化块的结构
    def setup(self):
        resnets = []  # 创建一个空列表用于存储 ResNet 块

        # 遍历每一层，创建 ResNet 块
        for i in range(self.num_layers):
            # 计算跳跃连接通道数
            res_skip_channels = self.in_channels if (i == self.num_layers - 1) else self.out_channels
            # 设置输入通道数
            resnet_in_channels = self.prev_output_channel if i == 0 else self.out_channels

            # 创建一个新的 FlaxResnetBlock2D 实例
            res_block = FlaxResnetBlock2D(
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
            resnets.append(res_block)  # 将块添加到列表中

        self.resnets = resnets  # 将列表赋值给实例变量

        # 如果需要上采样，初始化上采样层
        if self.add_upsample:
            self.upsamplers_0 = FlaxUpsample2D(self.out_channels, dtype=self.dtype)

    # 定义前向传播方法
    def __call__(self, hidden_states, res_hidden_states_tuple, temb, deterministic=True):
        # 遍历每个 ResNet 块进行前向传播
        for resnet in self.resnets:
            # 从元组中弹出最后的残差隐藏状态
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]  # 更新元组，去掉最后一项
            # 连接当前隐藏状态与残差隐藏状态
            hidden_states = jnp.concatenate((hidden_states, res_hidden_states), axis=-1)

            # 通过 ResNet 块处理隐藏状态
            hidden_states = resnet(hidden_states, temb, deterministic=deterministic)

        # 如果需要上采样，调用上采样层
        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states)

        return hidden_states  # 返回处理后的隐藏状态


# 定义一个 2D 中级交叉注意力块类，继承自 nn.Module
class FlaxUNetMidBlock2DCrossAttn(nn.Module):
    r"""
    Cross Attention 2D Mid-level block - original architecture from Unet transformers: https://arxiv.org/abs/2103.06104
    # 定义参数的文档字符串
    Parameters:
        in_channels (:obj:`int`):  # 输入通道数
            Input channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):  # Dropout比率，默认为0.0
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):  # 注意力层的数量，默认为1
            Number of attention blocks layers
        num_attention_heads (:obj:`int`, *optional*, defaults to 1):  # 每个空间变换块的注意力头数量，默认为1
            Number of attention heads of each spatial transformer block
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):  # 是否启用内存高效的注意力机制，默认为False
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        split_head_dim (`bool`, *optional*, defaults to `False`):  # 是否将头维度分割为新的轴以加速计算，默认为False
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):  # 数据类型参数，默认为jnp.float32
            Parameters `dtype`
    """

    in_channels: int  # 输入通道数的类型
    dropout: float = 0.0  # Dropout比率的默认值
    num_layers: int = 1  # 注意力层数量的默认值
    num_attention_heads: int = 1  # 注意力头数量的默认值
    use_linear_projection: bool = False  # 是否使用线性投影的默认值
    use_memory_efficient_attention: bool = False  # 是否使用内存高效注意力的默认值
    split_head_dim: bool = False  # 是否分割头维度的默认值
    dtype: jnp.dtype = jnp.float32  # 数据类型的默认值
    transformer_layers_per_block: int = 1  # 每个块中的变换层数量的默认值

    def setup(self):  # 设置方法，用于初始化
        # 至少会有一个ResNet块
        resnets = [  # 创建ResNet块列表
            FlaxResnetBlock2D(  # 创建一个ResNet块
                in_channels=self.in_channels,  # 输入通道数
                out_channels=self.in_channels,  # 输出通道数
                dropout_prob=self.dropout,  # Dropout概率
                dtype=self.dtype,  # 数据类型
            )
        ]

        attentions = []  # 初始化注意力块列表

        for _ in range(self.num_layers):  # 遍历指定的注意力层数
            attn_block = FlaxTransformer2DModel(  # 创建一个Transformer块
                in_channels=self.in_channels,  # 输入通道数
                n_heads=self.num_attention_heads,  # 注意力头数量
                d_head=self.in_channels // self.num_attention_heads,  # 每个头的维度
                depth=self.transformer_layers_per_block,  # 变换层深度
                use_linear_projection=self.use_linear_projection,  # 是否使用线性投影
                use_memory_efficient_attention=self.use_memory_efficient_attention,  # 是否使用内存高效注意力
                split_head_dim=self.split_head_dim,  # 是否分割头维度
                dtype=self.dtype,  # 数据类型
            )
            attentions.append(attn_block)  # 将注意力块添加到列表中

            res_block = FlaxResnetBlock2D(  # 创建一个ResNet块
                in_channels=self.in_channels,  # 输入通道数
                out_channels=self.in_channels,  # 输出通道数
                dropout_prob=self.dropout,  # Dropout概率
                dtype=self.dtype,  # 数据类型
            )
            resnets.append(res_block)  # 将ResNet块添加到列表中

        self.resnets = resnets  # 将ResNet块列表赋值给实例属性
        self.attentions = attentions  # 将注意力块列表赋值给实例属性

    def __call__(self, hidden_states, temb, encoder_hidden_states, deterministic=True):  # 调用方法
        hidden_states = self.resnets[0](hidden_states, temb)  # 通过第一个ResNet块处理隐藏状态
        for attn, resnet in zip(self.attentions, self.resnets[1:]):  # 遍历每个注意力块和后续ResNet块
            hidden_states = attn(hidden_states, encoder_hidden_states, deterministic=deterministic)  # 处理隐藏状态
            hidden_states = resnet(hidden_states, temb, deterministic=deterministic)  # 再次处理隐藏状态

        return hidden_states  # 返回处理后的隐藏状态
```