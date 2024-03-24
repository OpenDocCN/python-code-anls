# `.\lucidrains\enformer-tensorflow-sonnet-training-script\train.py`

```
# 版权声明，指明代码的版权归属
# 导入所需的库和模块
import time
import os
import glob
import json
import functools
import inspect
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, Optional, Text, Union, Iterable, List, Sequence

import sonnet as snt
from sonnet.src import base, once, types, utils
from sonnet.src.optimizers import optimizer_utils

import tensorflow as tf
import wandb

# attribute

# 引用 Enformer tensorflow 代码并进行修改以用于分布式训练
# https://github.com/deepmind/deepmind-research/tree/master/enformer

# 引用 Genetic augmentation 代码
# https://github.com/calico/basenji/blob/84c681a4b02f592a3de90799cee7f17d96f81ef8/basenji/archive/augmentation.py

# constants

NUM_CORES_ENFORCE = 64  # 使用 v3-64

SEQUENCE_LENGTH = 196_608
TARGET_LENGTH = 896
BIN_SIZE = 128

# assert TPUs

# 配置 TPU 环境
tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='enformer')
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = snt.distribute.TpuReplicator(tpu)

num_cores = tpu_strategy.num_replicas_in_sync
# 断言核心数与预期值相等
assert num_cores == NUM_CORES_ENFORCE, f'must betraining on {num_cores} cores'

# optimizer

# 实现 Adam 优化器的更新函数
def adam_update(g, alpha, beta_1, beta_2, epsilon, t, m, v):
  """Implements 'Algorithm 1' from :cite:`kingma2014adam`."""
  m = beta_1 * m + (1. - beta_1) * g      # Biased first moment estimate.
  v = beta_2 * v + (1. - beta_2) * g * g  # Biased second raw moment estimate.
  m_hat = m / (1. - tf.pow(beta_1, t))    # Bias corrected 1st moment estimate.
  v_hat = v / (1. - tf.pow(beta_2, t))    # Bias corrected 2nd moment estimate.
  update = alpha * m_hat / (tf.sqrt(v_hat) + epsilon)
  return update, m, v

# 自定义 Adam 优化器类
class Adam(base.Optimizer):
  def __init__(self,
               learning_rate: Union[types.FloatLike, tf.Variable] = 0.001,
               beta1: Union[types.FloatLike, tf.Variable] = 0.9,
               beta2: Union[types.FloatLike, tf.Variable] = 0.999,
               epsilon: Union[types.FloatLike, tf.Variable] = 1e-8,
               weight_decay: Union[types.FloatLike, tf.Variable] = 1e-4,
               name: Optional[str] = None):
    super().__init__(name=name)
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.weight_decay = weight_decay
    # 初始化步数
    self.step = tf.Variable(0, trainable=False, name="t", dtype=tf.int64)
    self.m = []
    self.v = []

  @once.once
  def _initialize(self, parameters: Sequence[tf.Variable]):
    """First and second order moments are initialized to zero."""
    zero_var = lambda p: utils.variable_like(p, trainable=False)
    with tf.name_scope("m"):
      self.m.extend(zero_var(p) for p in parameters)
    with tf.name_scope("v"):
      self.v.extend(zero_var(p) for p in parameters)

  def apply(self, updates: Sequence[types.ParameterUpdate],
            parameters: Sequence[tf.Variable]):
    optimizer_utils.check_distribution_strategy()
    optimizer_utils.check_updates_parameters(updates, parameters)
    self._initialize(parameters)
    self.step.assign_add(1)
    # 使用 zip 函数同时遍历 updates, parameters, self.m, self.v 四个列表中的元素
    for update, param, m_var, v_var in zip(updates, parameters, self.m, self.v):
      # 如果 update 为 None，则跳过当前循环
      if update is None:
        continue

      # 检查 update 和 param 的数据类型是否一致
      optimizer_utils.check_same_dtype(update, param)
      # 将学习率转换为 update 的数据类型
      learning_rate = tf.cast(self.learning_rate, update.dtype)
      # 将 beta1 转换为 update 的数据类型
      beta_1 = tf.cast(self.beta1, update.dtype)
      # 将 beta2 转换为 update 的数据类型
      beta_2 = tf.cast(self.beta2, update.dtype)
      # 将 epsilon 转换为 update 的数据类型
      epsilon = tf.cast(self.epsilon, update.dtype)
      # 将 step 转换为 update 的数据类型
      step = tf.cast(self.step, update.dtype)

      # 使用 adam_update 函数计算更新后的 update, m, v
      update, m, v = adam_update(
        g=update, alpha=learning_rate, beta_1=beta_1, beta_2=beta_2,
        epsilon=epsilon, t=step, m=m_var, v=v_var)

      # 计算权重衰减更新值，排除偏置项
      weight_decay_update = (param * self.weight_decay * learning_rate) if 'w:0' in param.name else tf.zeros_like(param)

      # 更新参数 param
      param.assign_sub(update)
      # 更新参数 param，加入权重衰减项
      param.assign_sub(weight_decay_update)

      # 更新 m_var
      m_var.assign(m)
      # 更新 v_var
      v_var.assign(v)
# 定义一个名为MultiheadAttention的类，用于实现多头注意力机制
class MultiheadAttention(snt.Module):
  """Multi-head attention."""

  def __init__(self,
               value_size: int,
               key_size: int,
               num_heads: int,
               scaling: bool = True,
               attention_dropout_rate: float = 0.1,
               relative_positions: bool = False,
               relative_position_symmetric: bool = False,
               relative_position_functions: Optional[List[str]] = None,
               num_relative_position_features: Optional[int] = None,
               positional_dropout_rate: float = 0.1,
               zero_initialize: bool = True,
               initializer: Optional[snt.initializers.Initializer] = None,
               name: str = None):
    """Creates a MultiheadAttention module.

    Args:
      value_size: 每个头部的值嵌入大小。
      key_size: 每个头部的键和查询嵌入大小。
      num_heads: 每个时间步的独立查询数量。
      scaling: 是否对注意力logits进行缩放。
      attention_dropout_rate: 注意力logits的dropout率。
      relative_positions: 是否使用TransformerXL风格的相对注意力。
      relative_position_symmetric: 如果为True，则使用对称版本的基础函数。
        如果为False，则使用对称和非对称版本。
      relative_position_functions: 用于相对位置偏差的函数名称列表。
      num_relative_position_features: 要计算的相对位置特征数量。
        如果为None，则使用`value_size * num_heads`。
      positional_dropout_rate: 如果使用相对位置，则位置编码的dropout率。
      zero_initialize: 如果为True，则最终的线性层将被初始化为0。
      initializer: 用于投影层的初始化器。如果未指定，则使用VarianceScaling，scale = 2.0。
      name: 模块的名称。
    """
    super().__init__(name=name)
    self._value_size = value_size
    self._key_size = key_size
    self._num_heads = num_heads
    self._attention_dropout_rate = attention_dropout_rate
    self._scaling = scaling
    self._relative_positions = relative_positions
    self._relative_position_symmetric = relative_position_symmetric
    self._relative_position_functions = relative_position_functions
    if num_relative_position_features is None:
      # num_relative_position_features需要能够被相对位置函数数量*2整除（用于对称和非对称版本）。
      divisible_by = 2 * len(self._relative_position_functions)
      self._num_relative_position_features = (
          (self._value_size // divisible_by) * divisible_by)
    else:
      self._num_relative_position_features = num_relative_position_features
    self._positional_dropout_rate = positional_dropout_rate

    self._initializer = initializer
    if self._initializer is None:
      self._initializer = snt.initializers.VarianceScaling(scale=2.0)

    key_proj_size = self._key_size * self._num_heads
    embedding_size = self._value_size * self._num_heads

    # 创建线性层用于查询、键和值的投影
    self._q_layer = snt.Linear(
        key_proj_size,
        name='q_layer',
        with_bias=False,
        w_init=self._initializer)
    self._k_layer = snt.Linear(
        key_proj_size,
        name='k_layer',
        with_bias=False,
        w_init=self._initializer)
    self._v_layer = snt.Linear(
        embedding_size,
        name='v_layer',
        with_bias=False,
        w_init=self._initializer)
    w_init = snt.initializers.Constant(1e-8) if zero_initialize else self._initializer
    # 创建线性层用于嵌入
    self._embedding_layer = snt.Linear(
        embedding_size,
        name='embedding_layer',
        w_init=w_init,
        b_init= snt.initializers.Constant(1e-8))

    # 如果使用相对位置，则创建额外的层
    # 如果存在相对位置信息
    if self._relative_positions:
      # 创建线性层用于处理相对位置信息
      self._r_k_layer = snt.Linear(
          key_proj_size,
          name='r_k_layer',
          with_bias=False,
          w_init=self._initializer)
      # 创建相对位置信息的偏置项
      self._r_w_bias = tf.Variable(
          self._initializer([1, self._num_heads, 1, self._key_size],
                            dtype=tf.float32),
          name='r_w_bias')
      self._r_r_bias = tf.Variable(
          self._initializer([1, self._num_heads, 1, self._key_size],
                            dtype=tf.float32),
          name='r_r_bias')

  def _multihead_output(self, linear, inputs):
    """Applies a standard linear to inputs and returns multihead output."""

    # 对输入应用标准线性变换
    output = snt.BatchApply(linear)(inputs)  # [B, T, H * KV]
    num_kv_channels = output.shape[-1] // self._num_heads
    # 将 H * Channels 分割成不同的轴
    output = snt.reshape(output,
                         output_shape=[-1, self._num_heads, num_kv_channels])
    # [B, T, H, KV] -> [B, H, T, KV]
    return tf.transpose(output, [0, 2, 1, 3])

  def __call__(self,
               inputs,
               is_training=False):
    # 初始化投影层
    embedding_size = self._value_size * self._num_heads
    seq_len = inputs.shape[1]

    # 计算 q, k 和 v 作为输入的多头投影
    q = self._multihead_output(self._q_layer, inputs)  # [B, H, T, K]
    k = self._multihead_output(self._k_layer, inputs)  # [B, H, T, K]
    v = self._multihead_output(self._v_layer, inputs)  # [B, H, T, V]

    # 将查询按照键大小的平方根进行缩放
    if self._scaling:
      q *= self._key_size**-0.5

    if self._relative_positions:
      # 对于相对位置，我们将位置投影以形成相对键
      distances = tf.range(-seq_len + 1, seq_len, dtype=tf.float32)[tf.newaxis]
      positional_encodings = positional_features_all(
          positions=distances,
          feature_size=self._num_relative_position_features,
          seq_length=seq_len,
          feature_functions=self._relative_position_functions,
          symmetric=self._relative_position_symmetric)
      # [1, 2T-1, Cr]

      if is_training:
        positional_encodings = tf.nn.dropout(
            positional_encodings, rate=self._positional_dropout_rate)

      # [1, H, 2T-1, K]
      r_k = self._multihead_output(self._r_k_layer, positional_encodings)

      # 将相对位置的偏移 logits 添加到内容 logits 中
      # [B, H, T', T]
      content_logits = tf.matmul(q + self._r_w_bias, k, transpose_b=True)
      # [B, H, T', 2T-1]
      relative_logits = tf.matmul(
          q + self._r_r_bias, r_k, transpose_b=True)
      #  [B, H, T', T]
      relative_logits = relative_shift(relative_logits)
      logits = content_logits + relative_logits
    else:
      # [B, H, T', T]
      logits = tf.matmul(q, k, transpose_b=True)

    weights = tf.nn.softmax(logits)

    # 在注意力权重上进行 dropout
    if is_training:
      weights = tf.nn.dropout(weights, rate=self._attention_dropout_rate)

    # 转置和重塑输出
    output = tf.matmul(weights, v)  # [B, H, T', V]
    output_transpose = tf.transpose(output, [0, 2, 1, 3])  # [B, T', H, V]

    # 最终线性层
    attended_inputs = snt.reshape(
        output_transpose, output_shape=[embedding_size], preserve_dims=2)
    output = self._embedding_layer(attended_inputs)

    return output
def relative_shift(x):
  """Shift the relative logits like in TransformerXL."""
  # 在最后一个时间尺度维度上添加零
  to_pad = tf.zeros_like(x[..., :1])
  x = tf.concat([to_pad, x], -1)
  _, num_heads, t1, t2 = x.shape
  x = tf.reshape(x, [-1, num_heads, t2, t1])
  x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
  x = tf.reshape(x, [-1, num_heads, t1, t2 - 1])
  x = tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, (t2 + 1) // 2])
  return x

# 可用的特征函数:
def get_positional_feature_function(name):
  """返回位置特征函数。"""
  available = {
      'positional_features_exponential': positional_features_exponential,
      'positional_features_central_mask': positional_features_central_mask,
      'positional_features_gamma': positional_features_gamma
  }
  if name not in available:
    raise ValueError(f'Function {name} not available in {available.keys()}')
  return available[name]


def positional_features_all(positions: tf.Tensor,
                            feature_size: int,
                            seq_length: Optional[int] = None,
                            bin_size: Optional[int] = None,
                            feature_functions: Optional[List[str]] = None,
                            symmetric=False):
  """计算相对位置编码/特征。每个位置特征函数将计算/提供相同比例的特征，组成总特征数为 feature_size。

  Args:
    positions: 任意形状的相对位置张量。
    feature_size: 基函数的总数。
    seq_length: 表示个体位置特征可以使用的特征长度的序列长度。这是必需的，因为输入特征的参数化应该独立于 `positions`，但仍然可能需要使用总特征数。
    bin_size: 用于对序列进行分区的 bin 大小。这可用于计算相对于基因组的绝对尺度上的特征。
    feature_functions: 要使用的不同特征函数的列表。每个函数将以参数形式接受：positions、序列长度和要计算的特征数。
    symmetric: 如果为 True，则生成的特征将在相对位置为 0 时对称（即只有位置的绝对值会影响）。如果为 False，则将使用特征的对称和非对称版本（对称乘以位置的符号）。

  Returns:
    形状为 `positions.shape + (feature_size,)` 的张量。
  """
  if feature_functions is None:
    feature_functions = ['positional_features_exponential',
                         'positional_features_central_mask',
                         'positional_features_gamma']
  num_components = len(feature_functions)  # 每个基函数一个
  if not symmetric:
    num_components = 2 * num_components

  # 目前，我们不允许奇数大小的嵌入。
  if feature_size % num_components != 0:
    raise ValueError(
        f'feature_size 必须能被 {num_components} 整除')

  feature_functions = [get_positional_feature_function(f)
                       for f in feature_functions]
  num_basis_per_class = feature_size // num_components
  embeddings = tf.concat([f(tf.abs(positions), num_basis_per_class,
                            seq_length, bin_size)
                          for f in feature_functions],
                         axis=-1)
  if not symmetric:
    embeddings = tf.concat([embeddings,
                            tf.sign(positions)[..., tf.newaxis] * embeddings],
                           axis=-1)
  tf.TensorShape(embeddings.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
  return embeddings


def _prepend_dims(x, num_dims):
  return tf.reshape(x, shape=[1] * num_dims + x.shape)
def positional_features_exponential(positions: tf.Tensor,
                                    feature_size: int,
                                    seq_length: Optional[int] = None,
                                    bin_size: Optional[int] = None,
                                    min_half_life: Optional[float] = 3.0):
  """Create exponentially decaying positional weights.

  Args:
    positions: Position tensor (arbitrary shape).
    feature_size: Number of basis functions to use.
    seq_length: Sequence length.
    bin_size: (unused). See `positional_features_all`.
    min_half_life: Smallest exponential half life in the grid of half lives.

  Returns:
    A Tensor with shape [2 * seq_length - 1, feature_size].
  """
  # 删除未使用的变量
  del bin_size  # Unused.
  # 如果未提供序列长度，则计算最大位置的绝对值加1作为序列长度
  if seq_length is None:
    seq_length = tf.reduce_max(tf.abs(positions)) + 1
  # 计算最大范围和半衰期
  seq_length = tf.cast(seq_length, dtype=tf.float32)
  max_range = tf.math.log(seq_length) / tf.math.log(2.0)
  half_life = tf.pow(2.0, tf.linspace(min_half_life, max_range, feature_size))
  half_life = _prepend_dims(half_life, positions.shape.rank)
  positions = tf.abs(positions)
  # 计算指数衰减权重
  outputs = tf.exp(-tf.math.log(2.0) / half_life * positions[..., tf.newaxis])
  # 确保输出形状与预期一致
  tf.TensorShape(outputs.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
  return outputs


def positional_features_central_mask(positions: tf.Tensor,
                                     feature_size: int,
                                     seq_length: Optional[int] = None,
                                     bin_size: Optional[int] = None):
  """Positional features using a central mask (allow only central features)."""
  # 删除未使用的变量
  del seq_length  # Unused.
  del bin_size  # Unused.
  # 计算中心掩码的宽度
  center_widths = tf.pow(2.0, tf.range(1, feature_size + 1, dtype=tf.float32))
  center_widths = center_widths - 1
  center_widths = _prepend_dims(center_widths, positions.shape.rank)
  # 创建中心掩码
  outputs = tf.cast(center_widths > tf.abs(positions)[..., tf.newaxis],
                    tf.float32)
  # 确保输出形状与预期一致
  tf.TensorShape(outputs.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
  return outputs


def gamma_pdf(x, concentration, rate):
  """Gamma probability distribution function: p(x|concentration, rate)."""
  # 计算 Gamma 概率分布函数
  log_unnormalized_prob = tf.math.xlogy(concentration - 1., x) - rate * x
  log_normalization = (tf.math.lgamma(concentration) -
                       concentration * tf.math.log(rate))
  return tf.exp(log_unnormalized_prob - log_normalization)


def positional_features_gamma(positions: tf.Tensor,
                              feature_size: int,
                              seq_length: Optional[int] = None,
                              bin_size: Optional[int] = None,
                              stddev=None,
                              start_mean=None):
  """Positional features computed using the gamma distributions."""
  # 删除未使用的变量
  del bin_size  # Unused.
  # 如果未提供序列长度，则计算最大位置的绝对值加1作为序列长度
  if seq_length is None:
    seq_length = tf.reduce_max(tf.abs(positions)) + 1
  # 如果未提供标准差，则使用默认值
  if stddev is None:
    stddev = seq_length / (2 * feature_size)
  # 如果未提供起始均值，则使用默认值
  if start_mean is None:
    start_mean = seq_length / feature_size
  # 计算均值、浓度和速率
  mean = tf.linspace(start_mean, seq_length, num=feature_size)
  mean = _prepend_dims(mean, positions.shape.rank)
  concentration = (mean / stddev)**2
  rate = mean / stddev**2
  # 计算 Gamma 分布概率
  probabilities = gamma_pdf(
      tf.abs(tf.cast(positions, dtype=tf.float32))[..., tf.newaxis],
      concentration, rate)
  probabilities += 1e-8  # 为了确保数值稳定性
  outputs = probabilities / tf.reduce_max(probabilities)
  # 确保输出形状与预期一致
  tf.TensorShape(outputs.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
  return outputs
class Enformer(snt.Module):
  """Main model."""

  def __init__(self,
               channels: int = 1536,
               num_transformer_layers: int = 11,
               num_heads: int = 8,
               pooling_type: str = 'attention',
               use_convnext: bool = False,
               name: str = 'enformer'):
    """Enformer model.

    Args:
      channels: Number of convolutional filters and the overall 'width' of the
        model.
      num_transformer_layers: Number of transformer layers.
      num_heads: Number of attention heads.
      pooling_type: Which pooling function to use. Options: 'attention' or max'.
      name: Name of sonnet module.
    """
    # 初始化 Enformer 模型
    super().__init__(name=name)
    # 定义头部通道数
    heads_channels = {'human': 5313, 'mouse': 1643}
    # 定义丢弃率
    dropout_rate = 0.4
    # 检查通道数是否可以被头部数整除
    assert channels % num_heads == 0, ('channels needs to be divisible '
                                       f'by {num_heads}')
    # 定义整体注意力参数
    whole_attention_kwargs = {
        'attention_dropout_rate': 0.05,
        'initializer': None,
        'key_size': 64,
        'num_heads': num_heads,
        'num_relative_position_features': channels // num_heads,
        'positional_dropout_rate': 0.01,
        'relative_position_functions': [
            'positional_features_exponential',
            'positional_features_central_mask',
            'positional_features_gamma'
        ],
        'relative_positions': True,
        'scaling': True,
        'value_size': channels // num_heads,
        'zero_initialize': True
    }

    # 定义名称作用域
    trunk_name_scope = tf.name_scope('trunk')
    trunk_name_scope.__enter__()
    # 导入 moving_averages 模块

    # 定义卷积块函数
    def conv_block(filters, width=1, w_init=None, name='conv_block', **kwargs):
      with tf.name_scope(name or "batch_norm"):
        moving_mean = moving_averages.ExponentialMovingAverage(
            0.9, name="moving_mean")
        moving_variance = moving_averages.ExponentialMovingAverage(
            0.9, name="moving_variance")
      return Sequential(lambda: [
          snt.distribute.CrossReplicaBatchNorm(create_scale=True,
                        create_offset=True,
                        moving_mean = moving_mean,
                        moving_variance = moving_variance,
                        scale_init=snt.initializers.Ones()),
          gelu,
          snt.Conv1D(filters, width, w_init=w_init, **kwargs)
      ], name=name)

    # 定义 ConvNext 卷积块函数
    def convnext_block(filters, width=1, mult = 4, ds_conv_kernel_size = 7, w_init=None, name='convnext_block', **kwargs):
      return Sequential(lambda: [
          ExpandDims(2),
          snt.DepthwiseConv2D((ds_conv_kernel_size, 1), name ='convnext_ds_conv'),
          Squeeze(2),
          snt.LayerNorm(axis=-1, create_scale=True, create_offset=True),
          snt.Linear(filters * mult, name='convnext_project_in'),
          tf.nn.relu,
          snt.Linear(filters, name='convnext_project_out')
      ], name=name)

    # 根据是否使用 ConvNext 选择不同的卷积块函数
    conv_block_fn = convnext_block if use_convnext else conv_block

    # 定义干部模块
    stem = Sequential(lambda: [
        snt.Conv1D(channels // 2, 15),
        Residual(conv_block(channels // 2, 1, name='pointwise_conv_block')),
        pooling_module(pooling_type, pool_size=2),
    ], name='stem')

    # 定义滤波器列表
    filter_list = exponential_linspace_int(start=channels // 2, end=channels,
                                           num=6, divisible_by=128)
    # 定义卷积塔模块
    conv_tower = Sequential(lambda: [
        Sequential(lambda: [
            conv_block(num_filters, 5),
            Residual(conv_block(num_filters, 1, name='pointwise_conv_block')),
            pooling_module(pooling_type, pool_size=2),
            ],
                   name=f'conv_tower_block_{i}')
        for i, num_filters in enumerate(filter_list)], name='conv_tower')

    # Transformer.
    # 定义一个多层感知机模型
    def transformer_mlp():
      return Sequential(lambda: [
          # 对输入进行 LayerNorm 处理
          snt.LayerNorm(axis=-1, create_scale=True, create_offset=True),
          # 线性变换，将输入维度扩展为 channels * 2
          snt.Linear(channels * 2, name = 'project_in'),
          # 随机失活，防止过拟合
          snt.Dropout(dropout_rate),
          # 激活函数，使用 ReLU
          tf.nn.relu,
          # 线性变换，将输入维度缩减为 channels
          snt.Linear(channels, name = 'project_out'),
          # 随机失活，防止过拟合
          snt.Dropout(dropout_rate)], name='mlp')

    # 定义一个 Transformer 模型
    transformer = Sequential(lambda: [
        Sequential(lambda: [
            # 残差连接，包含 LayerNorm、多头注意力、随机失活
            Residual(Sequential(lambda: [
                snt.LayerNorm(axis=-1,
                              create_scale=True, create_offset=True,
                              scale_init=snt.initializers.Ones()),
                MultiheadAttention(**whole_attention_kwargs,
                                                    name=f'attention_{i}'),
                snt.Dropout(dropout_rate),
            ], name='mha')),
            # 残差连接，包含 MLP 模块
            Residual(transformer_mlp())], name=f'transformer_block_{i}')
        for i in range(num_transformer_layers)], name='transformer')

    # 定义一个目标长度裁剪层
    crop_final = TargetLengthCrop1D(TARGET_LENGTH, name='target_input')

    # 定义一个最终的一维卷积块
    final_pointwise = Sequential(lambda: [
        # 一维卷积块，将输入维度扩展为 channels * 2
        conv_block(channels * 2, 1),
        # 随机失活，防止过拟合
        snt.Dropout(dropout_rate / 8),
        # 激活函数，使用 GELU
        gelu], name='final_pointwise')

    # 构建整个模型的主干部分
    self._trunk = Sequential([stem,
                              conv_tower,
                              transformer,
                              crop_final,
                              final_pointwise],
                             name='trunk')
    trunk_name_scope.__exit__(None, None, None)

    # 构建模型的头部部分
    with tf.name_scope('heads'):
      self._heads = {
          head: Sequential(
              lambda: [snt.Linear(num_channels), tf.nn.softplus],
              name=f'head_{head}')
          for head, num_channels in heads_channels.items()
      }
    # pylint: enable=g-complex-comprehension,g-long-lambda,cell-var-from-loop

  @property
  def trunk(self):
    return self._trunk

  @property
  def heads(self):
    return self._heads

  # 模型的前向传播方法
  def __call__(self, inputs: tf.Tensor,
               is_training: bool) -> Dict[str, tf.Tensor]:
    # 获取主干部分的嵌入表示
    trunk_embedding = self.trunk(inputs, is_training=is_training)
    # 返回各个头部的输出
    return {
        head: head_module(trunk_embedding, is_training=is_training)
        for head, head_module in self.heads.items()
    }

  # 针对输入数据进行预测的方法，用于 SavedModel
  @tf.function(input_signature=[
      tf.TensorSpec([None, SEQUENCE_LENGTH, 4], tf.float32)])
  def predict_on_batch(self, x):
    """Method for SavedModel."""
    return self(x, is_training=False)
class TargetLengthCrop1D(snt.Module):
  """Crop sequence to match the desired target length."""

  def __init__(self, target_length: int, name='target_length_crop'):
    super().__init__(name=name)
    self._target_length = target_length

  def __call__(self, inputs):
    # Calculate the amount to trim from the sequence to match the target length
    trim = (inputs.shape[-2] - self._target_length) // 2
    if trim < 0:
      raise ValueError('inputs longer than target length')

    # Crop the sequence to match the target length
    return inputs[..., trim:-trim, :]

class ExpandDims(snt.Module):

  def __init__(self, dim: int, name='expand_dims'):
    super().__init__(name=name)
    self._dim = dim

  def __call__(self, inputs):
    # Expand the dimensions of the input tensor at the specified dimension
    return tf.expand_dims(inputs, self._dim)

class Squeeze(snt.Module):

  def __init__(self, dim: int, name='squeeze'):
    super().__init__(name=name)
    self._dim = dim

  def __call__(self, inputs):
    # Remove dimensions of size 1 from the input tensor at the specified dimension
    return tf.squeeze(inputs, self._dim)

class Sequential(snt.Module):
  """snt.Sequential automatically passing is_training where it exists."""

  def __init__(self,
               layers: Optional[Union[Callable[[], Iterable[snt.Module]],
                                      Iterable[Callable[..., Any]]]] = None,
               name: Optional[Text] = None):
    super().__init__(name=name)
    if layers is None:
      self._layers = []
    else:
      # layers wrapped in a lambda function to have a common namespace.
      if hasattr(layers, '__call__'):
        with tf.name_scope(name):
          layers = layers()
      self._layers = [layer for layer in layers if layer is not None]

  def __call__(self, inputs: tf.Tensor, is_training: bool, **kwargs):
    outputs = inputs
    for _, mod in enumerate(self._layers):
      if accepts_is_training(mod):
        outputs = mod(outputs, is_training=is_training, **kwargs)
      else:
        outputs = mod(outputs, **kwargs)
    return outputs


def pooling_module(kind, pool_size):
  """Pooling module wrapper."""
  if kind == 'attention':
    return SoftmaxPooling1D(pool_size=pool_size, per_channel=True,
                            w_init_scale=2.0)
  elif kind == 'max':
    return tf.keras.layers.MaxPool1D(pool_size=pool_size, padding='same')
  else:
    raise ValueError(f'Invalid pooling kind: {kind}.')

class SoftmaxPooling1D(snt.Module):
  """Pooling operation with optional weights."""

  def __init__(self,
               pool_size: int = 2,
               per_channel: bool = False,
               w_init_scale: float = 0.0,
               name: str = 'softmax_pooling'):
    """Softmax pooling.

    Args:
      pool_size: Pooling size, same as in Max/AvgPooling.
      per_channel: If True, the logits/softmax weights will be computed for
        each channel separately. If False, same weights will be used across all
        channels.
      w_init_scale: When 0.0 is equivalent to avg pooling, and when
        ~2.0 and `per_channel=False` it's equivalent to max pooling.
      name: Module name.
    """
    super().__init__(name=name)
    self._pool_size = pool_size
    self._per_channel = per_channel
    self._w_init_scale = w_init_scale
    self._logit_linear = None

  @snt.once
  def _initialize(self, num_features):
    # Initialize the linear layer for computing logits
    self._logit_linear = snt.Linear(
        output_size=num_features if self._per_channel else 1,
        with_bias=False,  # Softmax is agnostic to shifts.
        w_init=snt.initializers.Identity(self._w_init_scale))

  def __call__(self, inputs):
    _, length, num_features = inputs.shape
    self._initialize(num_features)
    # Reshape the inputs for pooling operation
    inputs = tf.reshape(
        inputs,
        (-1, length // self._pool_size, self._pool_size, num_features))
    # Perform softmax pooling operation
    return tf.reduce_sum(
        inputs * tf.nn.softmax(self._logit_linear(inputs), axis=-2),
        axis=-2)


class Residual(snt.Module):
  """Residual block."""

  def __init__(self, module: snt.Module, name='residual'):
    super().__init__(name=name)
    self._module = module

  def __call__(self, inputs: tf.Tensor, is_training: bool, *args,
               **kwargs) -> tf.Tensor:
    # 返回输入数据与模块处理后的结果的和
    return inputs + self._module(inputs, is_training, *args, **kwargs)
# 定义 GELU 激活函数，应用高斯误差线性单元激活函数
def gelu(x: tf.Tensor) -> tf.Tensor:
  """Applies the Gaussian error linear unit (GELU) activation function.

  Using approximiation in section 2 of the original paper:
  https://arxiv.org/abs/1606.08415

  Args:
    x: Input tensor to apply gelu activation.
  Returns:
    Tensor with gelu activation applied to it.
  """
  return tf.nn.sigmoid(1.702 * x) * x


# 对序列进行 one-hot 编码
def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value: Any = 0,
                   dtype=np.float32) -> np.ndarray:
  """One-hot encode sequence."""
  # 将字符串转换为 uint8 类型
  def to_uint8(string):
    return np.frombuffer(string.encode('ascii'), dtype=np.uint8)
  # 创建一个零矩阵，用于存储 one-hot 编码结果
  hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
  # 对字母表进行 one-hot 编码
  hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
  hash_table[to_uint8(neutral_alphabet)] = neutral_value
  hash_table = hash_table.astype(dtype)
  return hash_table[to_uint8(sequence)]


# 生成指数增长的整数序列
def exponential_linspace_int(start, end, num, divisible_by=1):
  """Exponentially increasing values of integers."""
  def _round(x):
    return int(np.round(x / divisible_by) * divisible_by)

  base = np.exp(np.log(end / start) / (num - 1))
  return [_round(start * base**i) for i in range(num)]


# 检查模块是否接受 is_training 参数
def accepts_is_training(module):
  return 'is_training' in list(inspect.signature(module.__call__).parameters)


# 获取给定生物体的目标数据
def get_targets(organism):
  targets_txt = f'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_{organism}.txt'
  return pd.read_csv(targets_txt, sep='\t')


# 对批量 one-hot 编码的序列及其标签进行反向互补
def reverse_complement_transform(seq):
  """Reverse complement of batched onehot seq and corresponding label and na."""

  # 反向互补序列
  seq_rc = tf.gather(seq, [3, 2, 1, 0], axis=-1)
  seq_rc = tf.reverse(seq_rc, axis=[0])
  return seq_rc


# 将序列左移或右移指定数量的位置
def shift_sequence(seq, shift_amount, pad_value=0.25):
  """Shift a sequence left or right by shift_amount.
  Args:
    seq: a [batch_size, sequence_length, sequence_depth] sequence to shift
    shift_amount: the signed amount to shift (tf.int32 or int)
    pad_value: value to fill the padding (primitive or scalar tf.Tensor)
  """
  input_shape = seq.shape

  pad = pad_value * tf.ones_like(seq[0:tf.abs(shift_amount), :])

  def _shift_right(_seq):
    sliced_seq = _seq[:-shift_amount:, :]
    return tf.concat([pad, sliced_seq], axis=0)

  def _shift_left(_seq):
    sliced_seq = _seq[-shift_amount:, :]
    return tf.concat([sliced_seq, pad], axis=0)

  output = tf.cond(
      tf.greater(shift_amount, 0), lambda: _shift_right(seq),
      lambda: _shift_left(seq))

  output.set_shape(input_shape)
  return output


# 应用随机移位增强
def augment_stochastic_shifts(seq, augment_shifts):
  """Apply a stochastic shift augmentation.
  Args:
    seq: input sequence of size [batch_size, length, depth]
    augment_shifts: list of int offsets to sample from
  Returns:
    shifted and padded sequence of size [batch_size, length, depth]
  """
  shift_index = tf.random.uniform(shape=[], minval=0,
      maxval=len(augment_shifts), dtype=tf.int64)
  shift_value = tf.gather(tf.constant(augment_shifts), shift_index)

  seq = tf.cond(tf.not_equal(shift_value, 0),
                lambda: shift_sequence(seq, shift_value),
                lambda: seq)

  return seq


# 应用随机移位增强到映射函数
def augment_stochastic_shifts_map_fn(datum):
  augment_shifts = [-2, -1, 0, 1, 2]
  return dict(
    sequence = augment_stochastic_shifts(datum['sequence'], augment_shifts),
    target = datum['target']
  )


# 应用随机反向互补增强到映射函数
def augment_stochastic_rc_map_fn(datum):
  sequence, target = (datum['sequence'], datum['target'])
  augment = tf.random.uniform(shape=[]) > 0.5
  sequence, target = tf.cond(augment, lambda: (sequence[::-1, ::-1], target[::-1, :]),
                              lambda: (sequence, target))
  return dict(sequence = sequence, target = target)


# 获取生物体路径
def organism_path(organism):
    # 返回拼接后的 Google Cloud 存储路径，包含基因组信息
    return os.path.join(f'gs://basenji_barnyard/data', organism)
def get_dataset(organism, subset, num_threads=8, shuffle=True, rotate = 0, augment = False):
  # 获取指定生物的元数据
  metadata = get_metadata(organism)
  # 获取指定生物和数据集子集的 TFRecord 文件列表
  files = tfrecord_files(organism, subset) 
  # 将文件列表按照指定的旋转值重新排序
  files = files[rotate:] + files[:rotate]
  # 创建 TFRecord 数据集对象
  dataset = tf.data.TFRecordDataset(files,
                                    compression_type='ZLIB',
                                    num_parallel_reads=num_threads)
  if shuffle:
    # 如果需要打乱数据集，则重复数据集
    dataset = dataset.repeat()
    # 对数据集进行随机打乱
    dataset = dataset.shuffle(5000, seed = 42)

  # 对数据集中的每个元素进行反序列化操作
  dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                        num_parallel_calls=num_threads)
  if augment:
    # 如果需要数据增强，则对数据集进行增强操作
    dataset = dataset.map(augment_stochastic_shifts_map_fn, num_parallel_calls=num_threads)
    dataset = dataset.map(augment_stochastic_rc_map_fn, num_parallel_calls=num_threads)

  return dataset


def get_metadata(organism):
  # 获取指定生物的元数据
  path = os.path.join(organism_path(organism), 'statistics.json')
  with tf.io.gfile.GFile(path, 'r') as f:
    return json.load(f)


def tfrecord_files(organism, subset):
  # 获取指定生物和数据集子集的 TFRecord 文件列表，并按照文件名中的数字排序
  return sorted(tf.io.gfile.glob(os.path.join(
      organism_path(organism), 'tfrecords', f'{subset}-*.tfr'
  )), key=lambda x: int(x.split('-')[-1].split('.')[0]))


def deserialize(serialized_example, metadata):
  """Deserialize bytes stored in TFRecordFile."""
  # 定义 TFRecord 文件中的特征映射
  feature_map = {
      'sequence': tf.io.FixedLenFeature([], tf.string),
      'target': tf.io.FixedLenFeature([], tf.string),
  }
  # 解析 TFRecord 文件中的序列和目标特征
  example = tf.io.parse_example(serialized_example, feature_map)
  # 解码序列特征并转换为指定形状和数据类型
  sequence = tf.io.decode_raw(example['sequence'], tf.bool)
  sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
  sequence = tf.cast(sequence, tf.float32)

  # 解码目标特征并转换为指定形状和数据类型
  target = tf.io.decode_raw(example['target'], tf.float16)
  target = tf.reshape(target,
                      (metadata['target_length'], metadata['num_targets']))
  target = tf.cast(target, tf.float32)

  return {'sequence': sequence,
          'target': target}

# 新的 get_dataset 函数���用于实际为 196_608 的序列

NEW_TFRECORD_LOCATIONS = dict(
  human = dict(
    train = 'gs://enformer-human-train/',
    valid = 'gs://enformer-human-valid/'
  ),
  mouse = dict(
    train = 'gs://enformer-mouse-train/',
    valid = 'gs://enformer-mouse-valid/'
  )
)

NUM_TRACKS_CONFIG = dict(human = 5313, mouse = 1643)

def new_dataset_map_seq_target(
  element,
  seq_len,
  species,  # 'human' or 'mouse'
  target_length = 896,
  shifts = None,
  augment_rc = False
):
  assert species in NUM_TRACKS_CONFIG, f'{species} not found in config'
  num_tracks = NUM_TRACKS_CONFIG[species]

  num_shifts = 0 if shifts is None else len(list(range(shifts[0], shifts[1] + 1)))

  data = {
    'seq': tf.io.FixedLenFeature([(seq_len + num_shifts) * 4], tf.float32),
    'target': tf.io.FixedLenFeature([target_length * num_tracks], tf.float32),
  }

  content = tf.io.parse_single_example(element, data)

  content['sequence'] = content.pop('seq')
  content['sequence'] = tf.reshape(content['sequence'], (-1, 4))
  content['target'] = tf.reshape(content['target'], (target_length, -1))

  # 处理位移增强

  shifts = tf.pad(tf.random.uniform(shape = [1], minval = 0, maxval = num_shifts, dtype = tf.int64), [[0, 1]])
  content['sequence'] = tf.slice(content['sequence'], shifts, (seq_len, -1))

  if augment_rc:
    content = augment_stochastic_rc_map_fn(content)

  content['sequence'].set_shape(tf.TensorShape([seq_len, 4]))
  content['target'].set_shape(tf.TensorShape([target_length, num_tracks]))

  return content

def get_dataset_new(
  organism,
  datatype,
  shifts = (-2, 2),
  augment_rc = False,
  num_threads = 8
# 获取指定生物和数据类型的 TFRecord 文件路径
gcs_path = NEW_TFRECORD_LOCATIONS[organism][datatype]
# 获取指定路径下所有以 .tfrecord 结尾的文件，并按文件名排序
files = sorted(tf.io.gfile.glob(f'{gcs_path}*.tfrecord'))

# 创建 TFRecord 数据集对象，指定压缩类型为 ZLIB，并行读取线程数为 num_threads
dataset = tf.data.TFRecordDataset(files, compression_type='ZLIB', num_parallel_reads=num_threads)
# 部分应用函数，对数据集中的每个元素进行处理
map_element_fn = partial(new_dataset_map_seq_target, seq_len=SEQUENCE_LENGTH, species=organism, shifts=shifts, augment_rc=augment_rc)
dataset = dataset.map(map_element_fn)
# 返回处理后的数据集
return dataset

# 计算相关系数
def corr_coef(x, y, eps=0):
  # 计算 x 的平方
  x2 = tf.math.square(x)
  # 计算 y 的平方
  y2 = tf.math.square(y)
  # 计算 x 和 y 的乘积
  xy = x * y
  # 计算 x 的均值
  ex = tf.reduce_mean(x, axis=1)
  # 计算 y 的均值
  ey = tf.reduce_mean(y, axis=1)
  # 计算 x 和 y 的乘积的均值
  exy = tf.reduce_mean(xy, axis=1)
  # 计算 x 的平方的均值
  ex2 = tf.reduce_mean(x2, axis=1)
  # 计算 y 的平方的均值
  ey2 = tf.reduce_mean(y2, axis=1)
  # 计算相关系数
  r = (exy - ex * ey) / ((tf.math.sqrt(ex2 - tf.math.square(ex) + eps) * tf.math.sqrt(ey2 - tf.math.square(ey) + eps)) + eps)
  # 返回相关系数的均值
  return tf.reduce_mean(r, axis=-1)

# 创建评估步骤函数
def create_eval_step(model, head):
  @tf.function
  def predict(seq, target):
    # 使用模型进行预测
    pred = model(seq, is_training=False)[head]
    # 返回预测结果与目标值的相关系数
    return corr_coef(pred, target)
  return predict

# 创建训练步骤函数
def create_step_function(model, optimizer, head, clip_grad_norm=1.0, weight_decay=0.0001):

  @tf.function
  def train_step(batch_seq, batch_target):
    with tf.GradientTape() as tape:
      with snt.mixed_precision.scope(tf.float16):
        outputs = model(batch_seq, is_training=True)[head]

      # 计算相关系数损失
      corr_coef_loss = 1 - corr_coef(outputs, batch_target, eps=1e-8)
      # 计算 Poisson 损失
      poisson = tf.reduce_mean(tf.keras.losses.poisson(batch_target, outputs))
      # 总损失为 Poisson 损失
      loss = poisson

    # 计算梯度
    gradients = tape.gradient(loss, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    gradients = [tf.clip_by_norm(grad, clip_grad_norm) for grad in gradients]
    ctx = tf.distribute.get_replica_context()
    gradients = ctx.all_reduce("mean", gradients)
    optimizer.apply(gradients, model.trainable_variables)
    return loss

  return train_step

# 实例化模型和训练/评估函数
with tpu_strategy.scope():
  # 创建 Enformer 模型
  model = Enformer(channels=1536, num_heads=8, num_transformer_layers=11)

  # 创建学习率变量
  learning_rate = tf.Variable(0., trainable=False, name='learning_rate')
  # 创建 Adam 优化器
  optimizer = snt.optimizers.Adam(learning_rate=learning_rate)

  # 创建人类数据集训练步骤函数
  train_step_human = create_step_function(model, optimizer, 'human')
  # 创建小鼠数据集训练步骤函数
  train_step_mouse = create_step_function(model, optimizer, 'mouse')

  # 创建人类数据集评估步骤函数
  eval_step_human = create_eval_step(model, 'human')
  # 创建小鼠数据集评估步骤函数
  eval_step_mouse = create_eval_step(model, 'mouse')

# 实验追踪
wandb.init(project='enformer')
wandb.run.save()

# 训练模型
num_steps = int(2e6)
num_warmup_steps = 5000
target_learning_rate = 5e-4

checkpoint_every = 2500
max_eval_steps = 25
eval_every = 500

# 全局步骤变量
global_step = tf.Variable(0, name='global_step', trainable=False)

# 检查点
checkpoint_root = "gs://enformer/"
checkpoint_name = "enformer"

save_prefix = os.path.join(checkpoint_root, checkpoint_name)

checkpoint = tf.train.Checkpoint(module=model, step=global_step, optimizer=optimizer)

# 如果有最新的检查点，则加载
latest = tf.train.latest_checkpoint(checkpoint_root)
if latest is not None:
  checkpoint.restore(latest)

@tf.function
def step():
  global_step.assign(global_step + 1)

  batch_human, batch_mouse = next(data_it)
  loss_human = tpu_strategy.run(train_step_human, args=(batch_human['sequence'], batch_human['target']))
  loss_mouse = tpu_strategy.run(train_step_mouse, args=(batch_mouse['sequence'], batch_mouse['target']))

  loss_human = tpu_strategy.reduce('mean', loss_human, axis=None)
  loss_mouse = tpu_strategy.reduce('mean', loss_mouse, axis=None)

  learning_rate_frac = tf.math.minimum(1.0, tf.cast(global_step, tf.float32) / tf.math.maximum(1.0, float(num_warmup_steps)))      
  learning_rate.assign(target_learning_rate * learning_rate_frac)

  return loss_human, loss_mouse

@tf.function
# 定义一个函数，用于执行评估步骤
def eval_step():
  # 从验证数据集中获取下一个人类数据批次
  batch_human = next(valid_human_data_it)
  # 从验证数据集中获取下一个老鼠数据批次
  batch_mouse = next(valid_mouse_data_it)
  # 在 TPU 策略下运行人类数据评估步骤
  human_r = tpu_strategy.run(eval_step_human, args = (batch_human['sequence'], batch_human['target']))
  # 在 TPU 策略下运行老鼠数据评估步骤
  mouse_r = tpu_strategy.run(eval_step_mouse, args = (batch_mouse['sequence'], batch_mouse['target']))

  # 对人类数据结果进行均值归约
  human_r = tpu_strategy.reduce('mean', human_r, axis = 0)
  # 对老鼠数据结果进行均值归约
  mouse_r = tpu_strategy.reduce('mean', mouse_r, axis = 0)
  # 返回人类和老鼠数据的评估结果
  return human_r, mouse_r

# 获取全局步数
i = global_step.numpy()

# 计算总老鼠数据量和总人类数据量
total_mice = 114 * 256 + 111
total_human = 132 * 256 + 229
bucket_size = 256
num_seen = i * num_cores
# 计算在人类和老鼠数据中的文件跳过量
human_file_skip = (num_seen % total_human) // bucket_size
mouse_file_skip = (num_seen % total_mice) // bucket_size

# 获取人类和老鼠数据集，并按照指定方式处理
human_dataset = get_dataset('human', 'train', rotate = human_file_skip).batch(num_cores, drop_remainder = True)
mouse_dataset = get_dataset('mouse', 'train', rotate = mouse_file_skip).batch(num_cores, drop_remainder = True)
# 将人类和老鼠数据集进行配对，并预取数据
human_mouse_dataset = tf.data.Dataset.zip((human_dataset, mouse_dataset)).prefetch(2)

# 获取人类和老鼠验证数据集
human_valid_dataset = get_dataset('human', 'valid', shuffle = False).repeat().batch(num_cores)
mouse_valid_dataset = get_dataset('mouse', 'valid', shuffle = False).repeat().batch(num_cores)

# 创建数据集迭代器
data_it = iter(tpu_strategy.experimental_distribute_dataset(human_mouse_dataset))
valid_human_data_it = iter(tpu_strategy.experimental_distribute_dataset(human_valid_dataset))
valid_mouse_data_it = iter(tpu_strategy.experimental_distribute_dataset(mouse_valid_dataset))

# 打印起始步数
print(f'starting from {i}')

# 循环执行训练步骤
while i < num_steps:
  print(f'processing step {i}')
  # 执行训练步骤，获取人类和老鼠数据的损失值
  loss_human, loss_mouse = step()
  loss_human = loss_human.numpy()
  loss_mouse = loss_mouse.numpy()
  learning_rate_numpy = learning_rate.numpy()
  print(f'completed step {i}')
  # 记录损失值和学习率
  log = {
    'loss_human': loss_human,
    'loss_mouse': loss_mouse,
    'learning_rate': learning_rate_numpy
  }

  # 每隔一定步数进行评估
  if i and not i % eval_every:
    print('evaluating')
    # 执行评估步骤，获取人类和老鼠数据的皮尔逊相关系数
    human_pearson_r, mouse_pearson_r = eval_step()
    human_pearson_r = human_pearson_r.numpy()
    mouse_pearson_r = mouse_pearson_r.numpy()
    # 更新记录
    log = {
      **log,
      'human_pearson_r': human_pearson_r,
      'mouse_pearson_r': mouse_pearson_r
    }

  # 将记录写入日志
  wandb.log(log, step = i)

  # 每隔一定步数进行保存模型
  if not i % checkpoint_every:
    print('checkpointing')
    checkpoint.save(save_prefix)

  # 更新步数
  i += 1
```