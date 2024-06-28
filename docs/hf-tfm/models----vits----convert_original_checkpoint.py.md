# `.\models\vits\convert_original_checkpoint.py`

```
# 设置编码格式为 UTF-8

# 版权声明和许可证信息，指定了 Apache License, Version 2.0 的使用条件和限制
# 您可以通过访问指定的 URL 查看许可证的详细内容：http://www.apache.org/licenses/LICENSE-2.0

"""Convert VITS checkpoint."""

# 导入必要的库和模块
import argparse  # 解析命令行参数的库
import json  # 处理 JSON 格式数据的库
import tempfile  # 创建临时文件和目录的库

import torch  # PyTorch 深度学习库
from huggingface_hub import hf_hub_download  # Hugging Face Hub 下载模块

from transformers import VitsConfig, VitsModel, VitsTokenizer, logging  # Hugging Face Transformers 库中的相关模块

# 设置日志的详细程度为 info 级别
logging.set_verbosity_info()

# 获取或创建名为 "transformers.models.vits" 的日志记录器对象
logger = logging.get_logger("transformers.models.vits")

# 将 VITS 模型中文本编码器相关参数的映射定义为字典
MAPPING_TEXT_ENCODER = {
    "enc_p.emb": "text_encoder.embed_tokens",
    "enc_p.encoder.attn_layers.*.conv_k": "text_encoder.encoder.layers.*.attention.k_proj",
    "enc_p.encoder.attn_layers.*.conv_v": "text_encoder.encoder.layers.*.attention.v_proj",
    "enc_p.encoder.attn_layers.*.conv_q": "text_encoder.encoder.layers.*.attention.q_proj",
    "enc_p.encoder.attn_layers.*.conv_o": "text_encoder.encoder.layers.*.attention.out_proj",
    "enc_p.encoder.attn_layers.*.emb_rel_k": "text_encoder.encoder.layers.*.attention.emb_rel_k",
    "enc_p.encoder.attn_layers.*.emb_rel_v": "text_encoder.encoder.layers.*.attention.emb_rel_v",
    "enc_p.encoder.norm_layers_1.*.gamma": "text_encoder.encoder.layers.*.layer_norm.weight",
    "enc_p.encoder.norm_layers_1.*.beta": "text_encoder.encoder.layers.*.layer_norm.bias",
    "enc_p.encoder.ffn_layers.*.conv_1": "text_encoder.encoder.layers.*.feed_forward.conv_1",
    "enc_p.encoder.ffn_layers.*.conv_2": "text_encoder.encoder.layers.*.feed_forward.conv_2",
    "enc_p.encoder.norm_layers_2.*.gamma": "text_encoder.encoder.layers.*.final_layer_norm.weight",
    "enc_p.encoder.norm_layers_2.*.beta": "text_encoder.encoder.layers.*.final_layer_norm.bias",
    "enc_p.proj": "text_encoder.project",
}

# 将 VITS 模型中随机持续时间预测器相关参数的映射定义为字典
MAPPING_STOCHASTIC_DURATION_PREDICTOR = {
    "dp.pre": "duration_predictor.conv_pre",
    "dp.proj": "duration_predictor.conv_proj",
    "dp.convs.convs_sep.*": "duration_predictor.conv_dds.convs_dilated.*",
    "dp.convs.convs_1x1.*": "duration_predictor.conv_dds.convs_pointwise.*",
    "dp.convs.norms_1.*.gamma": "duration_predictor.conv_dds.norms_1.*.weight",
    "dp.convs.norms_1.*.beta": "duration_predictor.conv_dds.norms_1.*.bias",
    "dp.convs.norms_2.*.gamma": "duration_predictor.conv_dds.norms_2.*.weight",
    "dp.convs.norms_2.*.beta": "duration_predictor.conv_dds.norms_2.*.bias",
    "dp.flows.0.logs": "duration_predictor.flows.0.log_scale",
    "dp.flows.0.m": "duration_predictor.flows.0.translate",
    "dp.flows.*.pre": "duration_predictor.flows.*.conv_pre",
}
    # 将模型参数中的路径映射转换为新的路径，用于模型权重加载和迁移
    "dp.flows.*.proj": "duration_predictor.flows.*.conv_proj",
    # 转换卷积层的路径，将原路径映射到新的路径
    "dp.flows.*.convs.convs_1x1.0": "duration_predictor.flows.*.conv_dds.convs_pointwise.0",
    # 转换卷积层的路径，将原路径映射到新的路径
    "dp.flows.*.convs.convs_1x1.1": "duration_predictor.flows.*.conv_dds.convs_pointwise.1",
    # 转换卷积层的路径，将原路径映射到新的路径
    "dp.flows.*.convs.convs_1x1.2": "duration_predictor.flows.*.conv_dds.convs_pointwise.2",
    # 转换分离卷积层的路径，将原路径映射到新的路径
    "dp.flows.*.convs.convs_sep.0": "duration_predictor.flows.*.conv_dds.convs_dilated.0",
    # 转换分离卷积层的路径，将原路径映射到新的路径
    "dp.flows.*.convs.convs_sep.1": "duration_predictor.flows.*.conv_dds.convs_dilated.1",
    # 转换分离卷积层的路径，将原路径映射到新的路径
    "dp.flows.*.convs.convs_sep.2": "duration_predictor.flows.*.conv_dds.convs_dilated.2",
    # 转换归一化层的 gamma 参数路径，将原路径映射到新的路径
    "dp.flows.*.convs.norms_1.0.gamma": "duration_predictor.flows.*.conv_dds.norms_1.0.weight",
    # 转换归一化层的 beta 参数路径，将原路径映射到新的路径
    "dp.flows.*.convs.norms_1.0.beta": "duration_predictor.flows.*.conv_dds.norms_1.0.bias",
    # 转换归一化层的 gamma 参数路径，将原路径映射到新的路径
    "dp.flows.*.convs.norms_1.1.gamma": "duration_predictor.flows.*.conv_dds.norms_1.1.weight",
    # 转换归一化层的 beta 参数路径，将原路径映射到新的路径
    "dp.flows.*.convs.norms_1.1.beta": "duration_predictor.flows.*.conv_dds.norms_1.1.bias",
    # 转换归一化层的 gamma 参数路径，将原路径映射到新的路径
    "dp.flows.*.convs.norms_1.2.gamma": "duration_predictor.flows.*.conv_dds.norms_1.2.weight",
    # 转换归一化层的 beta 参数路径，将原路径映射到新的路径
    "dp.flows.*.convs.norms_1.2.beta": "duration_predictor.flows.*.conv_dds.norms_1.2.bias",
    # 转换归一化层的 gamma 参数路径，将原路径映射到新的路径
    "dp.flows.*.convs.norms_2.0.gamma": "duration_predictor.flows.*.conv_dds.norms_2.0.weight",
    # 转换归一化层的 beta 参数路径，将原路径映射到新的路径
    "dp.flows.*.convs.norms_2.0.beta": "duration_predictor.flows.*.conv_dds.norms_2.0.bias",
    # 转换归一化层的 gamma 参数路径，将原路径映射到新的路径
    "dp.flows.*.convs.norms_2.1.gamma": "duration_predictor.flows.*.conv_dds.norms_2.1.weight",
    # 转换归一化层的 beta 参数路径，将原路径映射到新的路径
    "dp.flows.*.convs.norms_2.1.beta": "duration_predictor.flows.*.conv_dds.norms_2.1.bias",
    # 转换归一化层的 gamma 参数路径，将原路径映射到新的路径
    "dp.flows.*.convs.norms_2.2.gamma": "duration_predictor.flows.*.conv_dds.norms_2.2.weight",
    # 转换归一化层的 beta 参数路径，将原路径映射到新的路径
    "dp.flows.*.convs.norms_2.2.beta": "duration_predictor.flows.*.conv_dds.norms_2.2.bias",
    # 转换后处理阶段的路径，将原路径映射到新的路径
    "dp.post_pre": "duration_predictor.post_conv_pre",
    # 转换后处理阶段的路径，将原路径映射到新的路径
    "dp.post_proj": "duration_predictor.post_conv_proj",
    # 转换后处理阶段的分离卷积层路径，将原路径映射到新的路径
    "dp.post_convs.convs_sep.*": "duration_predictor.post_conv_dds.convs_dilated.*",
    # 转换后处理阶段的 1x1 卷积层路径，将原路径映射到新的路径
    "dp.post_convs.convs_1x1.*": "duration_predictor.post_conv_dds.convs_pointwise.*",
    # 转换后处理阶段的归一化层 gamma 参数路径，将原路径映射到新的路径
    "dp.post_convs.norms_1.*.gamma": "duration_predictor.post_conv_dds.norms_1.*.weight",
    # 转换后处理阶段的归一化层 beta 参数路径，将原路径映射到新的路径
    "dp.post_convs.norms_1.*.beta": "duration_predictor.post_conv_dds.norms_1.*.bias",
    # 转换后处理阶段的归一化层 gamma 参数路径，将原路径映射到新的路径
    "dp.post_convs.norms_2.*.gamma": "duration_predictor.post_conv_dds.norms_2.*.weight",
    # 转换后处理阶段的归一化层 beta 参数路径，将原路径映射到新的路径
    "dp.post_convs.norms_2.*.beta": "duration_predictor.post_conv_dds.norms_2.*.bias",
    # 转换后处理阶段的 logs 参数路径，将原路径映射到新的路径
    "dp.post_flows.0.logs": "duration_predictor.post_flows.0.log_scale",
    # 转换后处理阶段的 m 参数路径，将原路径映射到新的路径
    "dp.post_flows.0.m": "duration_predictor.post_flows.0.translate",
    # 转换后处理阶段的前处理路径，将原路径映射到新的路径
    "dp.post_flows.*.pre": "duration_predictor.post_flows.*.conv_pre",
    # 转换后处理阶段的投影路径，将原路径映射到新的路径
    "dp.post_flows.*.proj": "duration_predictor.post_flows.*.conv_proj",
    # 转换后处理阶段的卷积层路径，将原路径映射到新的路径
    "dp.post_flows.*.convs.convs_1x1.0": "duration_predictor.post_flows.*.conv_dds.convs_pointwise.0",
    # 转换后处理阶段的卷积层路径，将原路径映射到新的路径
    "dp.post_flows.*.convs.convs_1x1.1": "duration_predictor.post_flows.*.conv_dds.convs_pointwise.1",
    # 转换后处理阶段的卷积层路径，将原路径映射到新的路径
    "dp
    # 定义一组映射关系，将源字符串路径映射到目标字符串路径
    "dp.post_flows.*.convs.convs_sep.0": "duration_predictor.post_flows.*.conv_dds.convs_dilated.0",
    "dp.post_flows.*.convs.convs_sep.1": "duration_predictor.post_flows.*.conv_dds.convs_dilated.1",
    "dp.post_flows.*.convs.convs_sep.2": "duration_predictor.post_flows.*.conv_dds.convs_dilated.2",
    # 映射 gamma 参数的路径
    "dp.post_flows.*.convs.norms_1.0.gamma": "duration_predictor.post_flows.*.conv_dds.norms_1.0.weight",
    "dp.post_flows.*.convs.norms_1.0.beta": "duration_predictor.post_flows.*.conv_dds.norms_1.0.bias",
    "dp.post_flows.*.convs.norms_1.1.gamma": "duration_predictor.post_flows.*.conv_dds.norms_1.1.weight",
    "dp.post_flows.*.convs.norms_1.1.beta": "duration_predictor.post_flows.*.conv_dds.norms_1.1.bias",
    "dp.post_flows.*.convs.norms_1.2.gamma": "duration_predictor.post_flows.*.conv_dds.norms_1.2.weight",
    "dp.post_flows.*.convs.norms_1.2.beta": "duration_predictor.post_flows.*.conv_dds.norms_1.2.bias",
    "dp.post_flows.*.convs.norms_2.0.gamma": "duration_predictor.post_flows.*.conv_dds.norms_2.0.weight",
    "dp.post_flows.*.convs.norms_2.0.beta": "duration_predictor.post_flows.*.conv_dds.norms_2.0.bias",
    "dp.post_flows.*.convs.norms_2.1.gamma": "duration_predictor.post_flows.*.conv_dds.norms_2.1.weight",
    "dp.post_flows.*.convs.norms_2.1.beta": "duration_predictor.post_flows.*.conv_dds.norms_2.1.bias",
    "dp.post_flows.*.convs.norms_2.2.gamma": "duration_predictor.post_flows.*.conv_dds.norms_2.2.weight",
    "dp.post_flows.*.convs.norms_2.2.beta": "duration_predictor.post_flows.*.conv_dds.norms_2.2.bias",
    # 映射条件参数路径
    "dp.cond": "duration_predictor.cond",  # num_speakers > 1
```python`
}
# 定义一个映射字典，用于将某些权重键映射到不同的键
MAPPING_FLOW = {
    "flow.flows.*.pre": "flow.flows.*.conv_pre",  # 将 'flow.flows.*.pre' 映射到 'flow.flows.*.conv_pre'
    "flow.flows.*.enc.in_layers.0": "flow.flows.*.wavenet.in_layers.0",  # 将 'flow.flows.*.enc.in_layers.0' 映射到 'flow.flows.*.wavenet.in_layers.0'
    "flow.flows.*.enc.in_layers.1": "flow.flows.*.wavenet.in_layers.1",  # 将 'flow.flows.*.enc.in_layers.1' 映射到 'flow.flows.*.wavenet.in_layers.1'
    "flow.flows.*.enc.in_layers.2": "flow.flows.*.wavenet.in_layers.2",  # 将 'flow.flows.*.enc.in_layers.2' 映射到 'flow.flows.*.wavenet.in_layers.2'
    "flow.flows.*.enc.in_layers.3": "flow.flows.*.wavenet.in_layers.3",  # 将 'flow.flows.*.enc.in_layers.3' 映射到 'flow.flows.*.wavenet.in_layers.3'
    "flow.flows.*.enc.res_skip_layers.0": "flow.flows.*.wavenet.res_skip_layers.0",  # 将 'flow.flows.*.enc.res_skip_layers.0' 映射到 'flow.flows.*.wavenet.res_skip_layers.0'
    "flow.flows.*.enc.res_skip_layers.1": "flow.flows.*.wavenet.res_skip_layers.1",  # 将 'flow.flows.*.enc.res_skip_layers.1' 映射到 'flow.flows.*.wavenet.res_skip_layers.1'
    "flow.flows.*.enc.res_skip_layers.2": "flow.flows.*.wavenet.res_skip_layers.2",  # 将 'flow.flows.*.enc.res_skip_layers.2' 映射到 'flow.flows.*.wavenet.res_skip_layers.2'
    "flow.flows.*.enc.res_skip_layers.3": "flow.flows.*.wavenet.res_skip_layers.3",  # 将 'flow.flows.*.enc.res_skip_layers.3' 映射到 'flow.flows.*.wavenet.res_skip_layers.3'
    "flow.flows.*.enc.cond_layer": "flow.flows.*.wavenet.cond_layer",  # 当 num_speakers > 1 时，将 'flow.flows.*.enc.cond_layer' 映射到 'flow.flows.*.wavenet.cond_layer'
    "flow.flows.*.post": "flow.flows.*.conv_post",  # 将 'flow.flows.*.post' 映射到 'flow.flows.*.conv_post'
}
# 定义一个映射字典，用于将生成器的权重键映射到不同的键
MAPPING_GENERATOR = {
    "dec.conv_pre": "decoder.conv_pre",  # 将 'dec.conv_pre' 映射到 'decoder.conv_pre'
    "dec.ups.0": "decoder.upsampler.0",  # 将 'dec.ups.0' 映射到 'decoder.upsampler.0'
    "dec.ups.1": "decoder.upsampler.1",  # 将 'dec.ups.1' 映射到 'decoder.upsampler.1'
    "dec.ups.2": "decoder.upsampler.2",  # 将 'dec.ups.2' 映射到 'decoder.upsampler.2'
    "dec.ups.3": "decoder.upsampler.3",  # 将 'dec.ups.3' 映射到 'decoder.upsampler.3'
    "dec.resblocks.*.convs1.0": "decoder.resblocks.*.convs1.0",  # 将 'dec.resblocks.*.convs1.0' 映射到 'decoder.resblocks.*.convs1.0'
    "dec.resblocks.*.convs1.1": "decoder.resblocks.*.convs1.1",  # 将 'dec.resblocks.*.convs1.1' 映射到 'decoder.resblocks.*.convs1.1'
    "dec.resblocks.*.convs1.2": "decoder.resblocks.*.convs1.2",  # 将 'dec.resblocks.*.convs1.2' 映射到 'decoder.resblocks.*.convs1.2'
    "dec.resblocks.*.convs2.0": "decoder.resblocks.*.convs2.0",  # 将 'dec.resblocks.*.convs2.0' 映射到 'decoder.resblocks.*.convs2.0'
    "dec.resblocks.*.convs2.1": "decoder.resblocks.*.convs2.1",  # 将 'dec.resblocks.*.convs2.1' 映射到 'decoder.resblocks.*.convs2.1'
    "dec.resblocks.*.convs2.2": "decoder.resblocks.*.convs2.2",  # 将 'dec.resblocks.*.convs2.2' 映射到 'decoder.resblocks.*.convs2.2'
    "dec.conv_post": "decoder.conv_post",  # 将 'dec.conv_post' 映射到 'decoder.conv_post'
    "dec.cond": "decoder.cond",  # 当 num_speakers > 1 时，将 'dec.cond' 映射到 'decoder.cond'
}
# 定义一个映射字典，用于将后验编码器的权重键映射到不同的键
MAPPING_POSTERIOR_ENCODER = {
    "enc_q.pre": "posterior_encoder.conv_pre",  # 将 'enc_q.pre' 映射到 'posterior_encoder.conv_pre'
    "enc_q.enc.in_layers.*": "posterior_encoder.wavenet.in_layers.*",  # 将 'enc_q.enc.in_layers.*' 映射到 'posterior_encoder.wavenet.in_layers.*'
    "enc_q.enc.res_skip_layers.*": "posterior_encoder.wavenet.res_skip_layers.*",  # 将 'enc_q.enc.res_skip_layers.*' 映射到 'posterior_encoder.wavenet.res_skip_layers.*'
    "enc_q.enc.cond_layer": "posterior_encoder.wavenet.cond_layer",  # 当 num_speakers > 1 时，将 'enc_q.enc.cond_layer' 映射到 'posterior_encoder.wavenet.cond_layer'
    "enc_q.proj": "posterior_encoder.conv_proj",  # 将 'enc_q.proj' 映射到 'posterior_encoder.conv_proj'
}
# 合并所有映射字典
MAPPING = {
    **MAPPING_TEXT_ENCODER,  # 将 MAPPING_TEXT_ENCODER 中的键值对加入到 MAPPING 字典中
    **MAPPING_STOCHASTIC_DURATION_PREDICTOR,  # 将 MAPPING_STOCHASTIC_DURATION_PREDICTOR 中的键值对加入到 MAPPING 字典中
    **MAPPING_FLOW,  # 将 MAPPING_FLOW 中的键值对加入到 MAPPING 字典中
    **MAPPING_GENERATOR,  # 将 MAPPING_GENERATOR 中的键值对加入到 MAPPING 字典中
    **MAPPING_POSTERIOR_ENCODER,  # 将 MAPPING_POSTERIOR_ENCODER 中的键值对加入到 MAPPING 字典中
    "emb_g": "embed_speaker",  # 当 num_speakers > 1 时，将 'emb_g' 映射到 'embed_speaker'
}
# 初始化一个空列表，用于存储顶级键
TOP_LEVEL_KEYS = []
# 初始化一个空列表，用于存储忽略的键
IGNORE_KEYS = []


def set_recursively(hf_pointer, key, value, full_name, weight_type):
    # 遍历键，依次获取 hf_pointer 对象的属性
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    # 获取指定权重类型的形状
    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    # 如果键以特定后缀结尾，将 value 的最后一个维度去掉（原始权重是 Conv1d）
    if key.endswith(".k_proj") or key.endswith(".v_proj") or key.endswith(".q_proj") or key.endswith(".out_proj"):
        value = value.squeeze(-1)

    # 检查 hf_shape 和 value.shape 是否匹配，如果不匹配则抛出异常
    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    # 如果 weight_type 是 'weight'，将 hf_pointer 的权重数据```python
}
# 末尾多余的大括号，可能是代码片段复制过程中的错误

MAPPING_FLOW = {
    "flow.flows.*.pre": "flow.flows.*.conv_pre",
    "flow.flows.*.enc.in_layers.0": "flow.flows.*.wavenet.in_layers.0",
    "flow.flows.*.enc.in_layers.1": "flow.flows.*.wavenet.in_layers.1",
    "flow.flows.*.enc.in_layers.2": "flow.flows.*.wavenet.in_layers.2",
    "flow.flows.*.enc.in_layers.3": "flow.flows.*.wavenet.in_layers.3",
    "flow.flows.*.enc.res_skip_layers.0": "flow.flows.*.wavenet.res_skip_layers.0",
    "flow.flows.*.enc.res_skip_layers.1": "flow.flows.*.wavenet.res_skip_layers.1",
    "flow.flows.*.enc.res_skip_layers.2": "flow.flows.*.wavenet.res_skip_layers.2",
    "flow.flows.*.enc.res_skip_layers.3": "flow.flows.*.wavenet.res_skip_layers.3",
    "flow.flows.*.enc.cond_layer": "flow.flows.*.wavenet.cond_layer",  # 当 num_speakers > 1 时使用
    # MAPPING_FLOW 中的映射关系，用于指定流模型的层对应关系
}

MAPPING_GENERATOR = {
    "dec.conv_pre": "decoder.conv_pre",
    "dec.ups.0": "decoder.upsampler.0",
    "dec.ups.1": "decoder.upsampler.1",
    "dec.ups.2": "decoder.upsampler.2",
    "dec.ups.3": "decoder.upsampler.3",
    "dec.resblocks.*.convs1.0": "decoder.resblocks.*.convs1.0",
    "dec.resblocks.*.convs1.1": "decoder.resblocks.*.convs1.1",
    "dec.resblocks.*.convs1.2": "decoder.resblocks.*.convs1.2",
    "dec.resblocks.*.convs2.0": "decoder.resblocks.*.convs2.0",
    "dec.resblocks.*.convs2.1": "decoder.resblocks.*.convs2.1",
    "dec.resblocks.*.convs2.2": "decoder.resblocks.*.convs2.2",
    "dec.conv_post": "decoder.conv_post",
    "dec.cond": "decoder.cond",  # 当 num_speakers > 1 时使用
    # MAPPING_GENERATOR 中```python
}
# 末尾多余的大括号，可能是代码片段复制过程中的错误

MAPPING_FLOW = {
    "flow.flows.*.pre": "flow.flows.*.conv_pre",
    "flow.flows.*.enc.in_layers.0": "flow.flows.*.wavenet.in_layers.0",
    "flow.flows.*.enc.in_layers.1": "flow.flows.*.wavenet.in_layers.1",
    "flow.flows.*.enc.in_layers.2": "flow.flows.*.wavenet.in_layers.2",
    "flow.flows.*.enc.in_layers.3": "flow.flows.*.wavenet.in_layers.3",
    "flow.flows.*.enc.res_skip_layers.0": "flow.flows.*.wavenet.res_skip_layers.0",
    "flow.flows.*.enc.res_skip_layers.1": "flow.flows.*.wavenet.res_skip_layers.1",
    "flow.flows.*.enc.res_skip_layers.2": "flow.flows.*.wavenet.res_skip_layers.2",
    "flow.flows.*.enc.res_skip_layers.3": "flow.flows.*.wavenet.res_skip_layers.3",
    "flow.flows.*.enc.cond_layer": "flow.flows.*.wavenet.cond_layer",  # 当 num_speakers > 1 时使用
    # MAPPING_FLOW 中的映射关系，用于指定流模型的层对应关系
}

MAPPING_GENERATOR = {
    "dec.conv_pre": "decoder.conv_pre",
    "dec.ups.0": "decoder.upsampler.0",
    "dec.ups.1": "decoder.upsampler.1",
    "dec.ups.2": "decoder.upsampler.2",
    "dec.ups.3": "decoder.upsampler.3",
    "dec.resblocks.*.convs1.0": "decoder.resblocks.*.convs1.0",
    "dec.resblocks.*.convs1.1": "decoder.resblocks.*.convs1.1",
    "dec.resblocks.*.convs1.2": "decoder.resblocks.*.convs1.2",
    "dec.resblocks.*.convs2.0": "decoder.resblocks.*.convs2.0",
    "dec.resblocks.*.convs2.1": "decoder.resblocks.*.convs2.1",
    "dec.resblocks.*.convs2.2": "decoder.resblocks.*.convs2.2",
    "dec.conv_post": "decoder.conv_post",
    "dec.cond": "decoder.cond",  # 当 num_speakers > 1 时使用
    # MAPPING_GENERATOR 中的映射关系，用于指定生成器模型的层对应关系
}

MAPPING_POSTERIOR_ENCODER = {
    "enc_q.pre": "posterior_encoder.conv_pre",
    "enc_q.enc.in_layers.*": "posterior_encoder.wavenet.in_layers.*",
    "enc_q.enc.res_skip_layers.*": "posterior_encoder.wavenet.res_skip_layers.*",
    "enc_q.enc.cond_layer": "posterior_encoder.wavenet.cond_layer",  # 当 num_speakers > 1 时使用
    # MAPPING_POSTERIOR_ENCODER 中的映射关系，用于指定后验编码器模型的层对应关系
}

MAPPING = {
    **MAPPING_TEXT_ENCODER,
    **MAPPING_STOCHASTIC_DURATION_PREDICTOR,
    **MAPPING_FLOW,
    **MAPPING_GENERATOR,
    **MAPPING_POSTERIOR_ENCODER,
    "emb_g": "embed_speaker",  # 当 num_speakers > 1 时使用
    # MAPPING 包含了所有模型的映射关系，整合了各个子映射字典
}

TOP_LEVEL_KEYS = []
IGNORE_KEYS = []


def set_recursively(hf_pointer, key, value, full_name, weight_type):
    # 递归设置 hf_pointer 中指定的 key 属性值为 value

    for attribute in key.split("."):
        # 通过循环逐级获取属性，直到达到指定的 key 所在的属性位置
        hf_pointer = getattr(hf_pointer, attribute)

    if weight_type is not None:
        # 如果指定了 weight_type，则获取对应的形状信息
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        # 否则获取整体的形状信息
        hf_shape = hf_pointer.shape

    # 如果 key 以特定字符串结尾，则压缩掉最后的核心维度（原始权重为 Conv1d）
    if key.endswith(".k_proj") or key.endswith(".v_proj") or key.endswith(".q_proj") or key.endswith(".out_proj"):
        value = value.squeeze(-1)

    # 检查值的形状是否与 hf_pointer 的形状相匹配，如果不匹配则抛出 ValueError
    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    if weight_type == "weight":
        # 如果 weight_type 是 'weight'，则将 hf_pointer 的权重数据设为 value
        hf_pointer.weight.data = value
    # 如果权重类型是 "weight_g"，则将值赋给相应的 hf_pointer 对象的 weight_g 属性
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    # 如果权重类型是 "weight_v"，则将值赋给相应的 hf_pointer 对象的 weight_v 属性
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    # 如果权重类型是 "bias"，则将值赋给相应的 hf_pointer 对象的 bias 属性
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    # 如果权重类型是 "running_mean"，则将值赋给相应的 hf_pointer 对象的 running_mean 属性
    elif weight_type == "running_mean":
        hf_pointer.running_mean.data = value
    # 如果权重类型是 "running_var"，则将值赋给相应的 hf_pointer 对象的 running_var 属性
    elif weight_type == "running_var":
        hf_pointer.running_var.data = value
    # 如果权重类型是 "num_batches_tracked"，则将值赋给相应的 hf_pointer 对象的 num_batches_tracked 属性
    elif weight_type == "num_batches_tracked":
        hf_pointer.num_batches_tracked.data = value
    # 如果权重类型不属于以上任何一种情况，则将值直接赋给 hf_pointer 对象的 data 属性
    else:
        hf_pointer.data = value

    # 记录初始化日志信息，描述哪个键的哪种权重类型（如果有）从完整名称 full_name 加载得来
    logger.info(f"{key + ('.' + weight_type if weight_type is not None else '')} was initialized from {full_name}.")
# 检查给定的名称是否应该被忽略，根据忽略规则列表 ignore_keys
def should_ignore(name, ignore_keys):
    for key in ignore_keys:
        # 如果规则以 ".*" 结尾，检查名称是否以去掉最后一个字符的规则开头，如果是则忽略该名称
        if key.endswith(".*"):
            if name.startswith(key[:-1]):
                return True
        # 如果规则中包含 ".*."，则按前缀和后缀进行分割，检查名称中是否同时包含前缀和后缀，如果是则忽略该名称
        elif ".*." in key:
            prefix, suffix = key.split(".*.")
            if prefix in name and suffix in name:
                return True
        # 否则，直接检查名称是否包含规则中指定的字符串，如果是则忽略该名称
        elif key in name:
            return True
    # 如果都不匹配，则不忽略该名称
    return False


# 递归地加载 Fairseq 模型的权重到 Hugging Face 模型中
def recursively_load_weights(fairseq_dict, hf_model):
    unused_weights = []

    # 遍历 Fairseq 模型字典中的每个名称和对应的值
    for name, value in fairseq_dict.items():
        # 检查是否应该忽略该名称的加载
        if should_ignore(name, IGNORE_KEYS):
            # 如果需要忽略，记录日志并继续下一个名称的处理
            logger.info(f"{name} was ignored")
            continue

        is_used = False
        # 遍历映射规则 MAPPING 中的每对键值对
        for key, mapped_key in MAPPING.items():
            # 如果映射规则以 ".*" 结尾，去掉最后一个字符
            if key.endswith(".*"):
                key = key[:-1]
            # 如果映射规则中包含 "*"，按照前缀和后缀进行分割
            elif "*" in key:
                prefix, suffix = key.split(".*.")
                if prefix in name and suffix in name:
                    key = suffix

            # 检查当前名称是否匹配映射规则中的键
            if key in name:
                is_used = True
                # 根据映射规则修改 mapped_key 中的 "*"，用名称中的索引替换
                if mapped_key.endswith(".*"):
                    layer_index = name.split(key)[-1].split(".")[0]
                    mapped_key = mapped_key.replace("*", layer_index)
                elif "*" in mapped_key:
                    layer_index = name.split(key)[0].split(".")[-2]

                    # 根据特定规则重新映射层索引
                    if "flow.flows" in mapped_key:
                        layer_index = str(int(layer_index) // 2)
                    if "duration_predictor.flows" in mapped_key or "duration_predictor.post_flows" in mapped_key:
                        layer_index = str(int(layer_index) // 2 + 1)

                    mapped_key = mapped_key.replace("*", layer_index)
                
                # 根据名称中的标识确定权重类型
                if "weight_g" in name:
                    weight_type = "weight_g"
                elif "weight_v" in name:
                    weight_type = "weight_v"
                elif "bias" in name:
                    weight_type = "bias"
                elif "weight" in name:
                    weight_type = "weight"
                elif "running_mean" in name:
                    weight_type = "running_mean"
                elif "running_var" in name:
                    weight_type = "running_var"
                elif "num_batches_tracked" in name:
                    weight_type = "num_batches_tracked"
                else:
                    weight_type = None
                
                # 使用递归设置函数将值加载到 Hugging Face 模型中的指定位置
                set_recursively(hf_model, mapped_key, value, name, weight_type)
            continue
        # 如果没有匹配的映射规则，则记录为未使用的权重
        if not is_used:
            unused_weights.append(name)

    # 记录未使用的权重信息到日志中
    logger.warning(f"Unused weights: {unused_weights}")


# 使用 Torch 的 no_grad 装饰器，将 PyTorch 模型权重转换为 Transformers 设计的函数
@torch.no_grad()
def convert_checkpoint(
    pytorch_dump_folder_path,
    checkpoint_path=None,
    config_path=None,
    vocab_path=None,
    language=None,
    num_speakers=None,
    sampling_rate=None,
    repo_id=None,
):
    """
    将模型权重从 PyTorch 复制/粘贴/调整到 Transformers 设计中。
    """
    # 如果提供了配置文件路径，则从预训练配置中加载配置
    if config_path is not None:
        config = VitsConfig.from_pretrained(config_path)
    else:
        # 否则创建一个新的 VitsConfig 对象
        config = VitsConfig()

    # 如果提供了说话人数量，则更新配置中的说话人数量和说话人嵌入大小
    if num_speakers:
        config.num_speakers = num_speakers
        config.speaker_embedding_size = 256

    # 如果提供了采样率，则更新配置中的采样率
    if sampling_rate:
        config.sampling_rate = sampling_rate

    # 如果未提供检查点路径，则下载并准备 Facebook MMS-TTS 模型所需的词汇表、配置文件和检查点路径
    if checkpoint_path is None:
        logger.info(f"***Converting model: facebook/mms-tts {language}***")

        # 下载词汇表
        vocab_path = hf_hub_download(
            repo_id="facebook/mms-tts",
            filename="vocab.txt",
            subfolder=f"models/{language}",
        )
        # 下载配置文件
        config_file = hf_hub_download(
            repo_id="facebook/mms-tts",
            filename="config.json",
            subfolder=f"models/{language}",
        )
        # 下载模型检查点
        checkpoint_path = hf_hub_download(
            repo_id="facebook/mms-tts",
            filename="G_100000.pth",
            subfolder=f"models/{language}",
        )

        # 读取并加载配置文件中的超参数
        with open(config_file, "r") as f:
            data = f.read()
            hps = json.loads(data)

        # 检查模型是否针对 uroman 数据集训练，如果是则发出警告
        is_uroman = hps["data"]["training_files"].split(".")[-1] == "uroman"
        if is_uroman:
            logger.warning("For this checkpoint, you should use `uroman` to convert input text before tokenizing it!")
    else:
        # 如果提供了检查点路径，则记录信息并设置 is_uroman 为 False
        logger.info(f"***Converting model: {checkpoint_path}***")
        is_uroman = False

    # 如果词汇表路径为空，则设置默认的符号列表和符号到索引映射关系
    if vocab_path is None:
        _pad = "_"
        _punctuation = ';:,.!?¡¿—…"«»“” '
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        symbols = _pad + _punctuation + _letters + _letters_ipa
        symbol_to_id = {s: i for i, s in enumerate(symbols)}
        phonemize = True
    else:
        # 否则，从给定的词汇表路径读取符号列表，并创建符号到索引映射关系
        symbols = [line.replace("\n", "") for line in open(vocab_path, encoding="utf-8").readlines()]
        symbol_to_id = {s: i for i, s in enumerate(symbols)}
        # MMS-TTS 模型不使用 <pad> 标记，所以将其设置为用于间隔字符的标记
        _pad = symbols[0]
        phonemize = False

    # 创建一个临时文件，将符号到索引映射关系保存为 JSON 格式
    with tempfile.NamedTemporaryFile() as tf:
        with open(tf.name, "w", encoding="utf-8") as f:
            f.write(json.dumps(symbol_to_id, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 根据临时文件中的符号到索引映射关系创建一个 VitsTokenizer 对象
        tokenizer = VitsTokenizer(tf.name, language=language, phonemize=phonemize, is_uroman=is_uroman, pad_token=_pad)

    # 设置配置对象中的词汇表大小
    config.vocab_size = len(symbols)
    
    # 基于配置对象创建 VitsModel 模型
    model = VitsModel(config)

    # 对模型的解码器应用权重归一化
    model.decoder.apply_weight_norm()

    # 加载原始检查点的权重到模型中
    orig_checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    recursively_load_weights(orig_checkpoint["model"], model)

    # 移除模型的解码器上的权重归一化
    model.decoder.remove_weight_norm()

    # 将模型和 tokenizer 的预训练权重和词汇表保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    tokenizer.save_pretrained(pytorch_dump_folder_path)
    # 如果 repo_id 存在（即非空），则执行以下操作
    if repo_id:
        # 打印信息：正在推送到中心库...
        print("Pushing to the hub...")
        # 调用 tokenizer 对象的 push_to_hub 方法，将模型的 tokenizer 推送到指定的 repo_id
        tokenizer.push_to_hub(repo_id)
        # 调用 model 对象的 push_to_hub 方法，将模型本身推送到指定的 repo_id
        model.push_to_hub(repo_id)
# 主程序入口，用于执行脚本的入口点
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数，用于指定原始检查点的本地路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Local path to original checkpoint")
    # 添加命令行参数，用于指定vocab.txt文件的路径
    parser.add_argument("--vocab_path", default=None, type=str, help="Path to vocab.txt")
    # 添加命令行参数，用于指定待转换模型的hf config.json文件的路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加命令行参数，用于指定分词器语言的三字母代码
    parser.add_argument("--language", default=None, type=str, help="Tokenizer language (three-letter code)")
    # 添加命令行参数，用于指定说话者的数量
    parser.add_argument("--num_speakers", default=None, type=int, help="Number of speakers")
    # 添加命令行参数，用于指定模型训练时的采样率
    parser.add_argument(
        "--sampling_rate", default=None, type=int, help="Sampling rate on which the model was trained."
    )
    # 添加命令行参数，必需参数，用于指定输出的PyTorch模型的路径
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    # 添加命令行参数，用于指定转换后模型上传至🤗 hub的位置
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用函数，将指定参数传递给convert_checkpoint函数进行检查点转换
    convert_checkpoint(
        args.pytorch_dump_folder_path,
        args.checkpoint_path,
        args.config_path,
        args.vocab_path,
        args.language,
        args.num_speakers,
        args.sampling_rate,
        args.push_to_hub,
    )
```