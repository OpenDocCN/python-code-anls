# `.\transformers\models\vits\convert_original_checkpoint.py`

```
# 设置编码格式为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本使用此文件
# 除非符合许可证，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据"原样"提供，不提供任何形式的保证
# 无论是明示的还是暗示的，包括但不限于
# 适销性保证和特定用途的适用性保证。
# 查看许可证以了解许可证的特定语言和权限
"""将 VITS 检查点转换为其他格式"""

# 导入必要的库
import argparse  # 用于解析命令行参数
import json  # 用于处理 JSON 格式的数据
import tempfile  # 用于创建临时文件

import torch  # PyTorch 库
from huggingface_hub import hf_hub_download  # 从 huggingface_hub 下载模型

# 导入 VITS 相关模块
from transformers import VitsConfig, VitsModel, VitsTokenizer, logging

# 设置日志记录的详细程度为信息级别
logging.set_verbosity_info()
# 获取记录器
logger = logging.get_logger("transformers.models.vits")

# 定义用于将文本编码器参数映射为 VITS 参数的字典
MAPPING_TEXT_ENCODER = {
    "enc_p.emb": "text_encoder.embed_tokens",  # 文本编码器的嵌入层
    "enc_p.encoder.attn_layers.*.conv_k": "text_encoder.encoder.layers.*.attention.k_proj",  # 注意力层的 k 投影
    "enc_p.encoder.attn_layers.*.conv_v": "text_encoder.encoder.layers.*.attention.v_proj",  # 注意力层的 v 投影
    "enc_p.encoder.attn_layers.*.conv_q": "text_encoder.encoder.layers.*.attention.q_proj",  # 注意力层的 q 投影
    "enc_p.encoder.attn_layers.*.conv_o": "text_encoder.encoder.layers.*.attention.out_proj",  # 注意力层的输出投影
    "enc_p.encoder.attn_layers.*.emb_rel_k": "text_encoder.encoder.layers.*.attention.emb_rel_k",  # 注意力层的相对位置编码（k）
    "enc_p.encoder.attn_layers.*.emb_rel_v": "text_encoder.encoder.layers.*.attention.emb_rel_v",  # 注意力层的相对位置编码（v）
    "enc_p.encoder.norm_layers_1.*.gamma": "text_encoder.encoder.layers.*.layer_norm.weight",  # 归一化层的 gamma 参数
    "enc_p.encoder.norm_layers_1.*.beta": "text_encoder.encoder.layers.*.layer_norm.bias",  # 归一化层的 beta 参数
    "enc_p.encoder.ffn_layers.*.conv_1": "text_encoder.encoder.layers.*.feed_forward.conv_1",  # 前馈网络的第一个卷积层
    "enc_p.encoder.ffn_layers.*.conv_2": "text_encoder.encoder.layers.*.feed_forward.conv_2",  # 前馈网络的第二个卷积层
    "enc_p.encoder.norm_layers_2.*.gamma": "text_encoder.encoder.layers.*.final_layer_norm.weight",  # 最终归一化层的 gamma 参数
    "enc_p.encoder.norm_layers_2.*.beta": "text_encoder.encoder.layers.*.final_layer_norm.bias",  # 最终归一化层的 beta 参数
    "enc_p.proj": "text_encoder.project",  # 文本编码器的投影层
}

# 定义用于将随机持续时间预测器参数映射为 VITS 参数的字典
MAPPING_STOCHASTIC_DURATION_PREDICTOR = {
    "dp.pre": "duration_predictor.conv_pre",  # 持续时间预测器的预处理卷积层
    "dp.proj": "duration_predictor.conv_proj",  # 持续时间预测器的投影卷积层
    "dp.convs.convs_sep.*": "duration_predictor.conv_dds.convs_dilated.*",  # 分离卷积层
    "dp.convs.convs_1x1.*": "duration_predictor.conv_dds.convs_pointwise.*",  # 1x1 卷积层
    "dp.convs.norms_1.*.gamma": "duration_predictor.conv_dds.norms_1.*.weight",  # 第一个归一化层的 gamma 参数
    "dp.convs.norms_1.*.beta": "duration_predictor.conv_dds.norms_1.*.bias",  # 第一个归一化层的 beta 参数
    "dp.convs.norms_2.*.gamma": "duration_predictor.conv_dds.norms_2.*.weight",  # 第二个归一化层的 gamma 参数
    "dp.convs.norms_2.*.beta": "duration_predictor.conv_dds.norms_2.*.bias",  # 第二个归一化层的 beta 参数
    "dp.flows.0.logs": "duration_predictor.flows.0.log_scale",  # 对数尺度参数
    "dp.flows.0.m": "duration_predictor.flows.0.translate",  # 平移参数
    "dp.flows.*.pre": "duration_predictor.flows.*.conv_pre",  # 流的预处理卷积层
}
    # 这些是一系列将键值对映射到新的键值对的语句
    # 它们似乎是用来对模型中的某些参数进行重命名或映射
    # 例如将 "dp.flows.*.proj" 映射到 "duration_predictor.flows.*.conv_proj"
    "dp.flows.*.proj": "duration_predictor.flows.*.conv_proj",
    "dp.flows.*.convs.convs_1x1.0": "duration_predictor.flows.*.conv_dds.convs_pointwise.0",
    "dp.flows.*.convs.convs_1x1.1": "duration_predictor.flows.*.conv_dds.convs_pointwise.1",
    "dp.flows.*.convs.convs_1x1.2": "duration_predictor.flows.*.conv_dds.convs_pointwise.2",
    "dp.flows.*.convs.convs_sep.0": "duration_predictor.flows.*.conv_dds.convs_dilated.0",
    "dp.flows.*.convs.convs_sep.1": "duration_predictor.flows.*.conv_dds.convs_dilated.1",
    "dp.flows.*.convs.convs_sep.2": "duration_predictor.flows.*.conv_dds.convs_dilated.2",
    "dp.flows.*.convs.norms_1.0.gamma": "duration_predictor.flows.*.conv_dds.norms_1.0.weight",
    "dp.flows.*.convs.norms_1.0.beta": "duration_predictor.flows.*.conv_dds.norms_1.0.bias",
    "dp.flows.*.convs.norms_1.1.gamma": "duration_predictor.flows.*.conv_dds.norms_1.1.weight",
    "dp.flows.*.convs.norms_1.1.beta": "duration_predictor.flows.*.conv_dds.norms_1.1.bias",
    "dp.flows.*.convs.norms_1.2.gamma": "duration_predictor.flows.*.conv_dds.norms_1.2.weight",
    "dp.flows.*.convs.norms_1.2.beta": "duration_predictor.flows.*.conv_dds.norms_1.2.bias",
    "dp.flows.*.convs.norms_2.0.gamma": "duration_predictor.flows.*.conv_dds.norms_2.0.weight",
    "dp.flows.*.convs.norms_2.0.beta": "duration_predictor.flows.*.conv_dds.norms_2.0.bias",
    "dp.flows.*.convs.norms_2.1.gamma": "duration_predictor.flows.*.conv_dds.norms_2.1.weight",
    "dp.flows.*.convs.norms_2.1.beta": "duration_predictor.flows.*.conv_dds.norms_2.1.bias",
    "dp.flows.*.convs.norms_2.2.gamma": "duration_predictor.flows.*.conv_dds.norms_2.2.weight",
    "dp.flows.*.convs.norms_2.2.beta": "duration_predictor.flows.*.conv_dds.norms_2.2.bias",
    "dp.post_pre": "duration_predictor.post_conv_pre",
    "dp.post_proj": "duration_predictor.post_conv_proj",
    "dp.post_convs.convs_sep.*": "duration_predictor.post_conv_dds.convs_dilated.*",
    "dp.post_convs.convs_1x1.*": "duration_predictor.post_conv_dds.convs_pointwise.*",
    "dp.post_convs.norms_1.*.gamma": "duration_predictor.post_conv_dds.norms_1.*.weight",
    "dp.post_convs.norms_1.*.beta": "duration_predictor.post_conv_dds.norms_1.*.bias",
    "dp.post_convs.norms_2.*.gamma": "duration_predictor.post_conv_dds.norms_2.*.weight",
    "dp.post_convs.norms_2.*.beta": "duration_predictor.post_conv_dds.norms_2.*.bias",
    "dp.post_flows.0.logs": "duration_predictor.post_flows.0.log_scale",
    "dp.post_flows.0.m": "duration_predictor.post_flows.0.translate",
    "dp.post_flows.*.pre": "duration_predictor.post_flows.*.conv_pre",
    "dp.post_flows.*.proj": "duration_predictor.post_flows.*.conv_proj",
    "dp.post_flows.*.convs.convs_1x1.0": "duration_predictor.post_flows.*.conv_dds.convs_pointwise.0",
    "dp.post_flows.*.convs.convs_1x1.1": "duration_predictor.post_flows.*.conv_dds.convs_pointwise.1",
    "dp.post_flows.*.convs.convs_1x1.2": "duration_predictor.post_flows.*.conv_dds.convs_pointwise.2",
    # 映射dp.post_flows.*.convs.convs_sep.0到duration_predictor.post_flows.*.conv_dds.convs_dilated.0
    "dp.post_flows.*.convs.convs_sep.0": "duration_predictor.post_flows.*.conv_dds.convs_dilated.0",
    # 映射dp.post_flows.*.convs.convs_sep.1到duration_predictor.post_flows.*.conv_dds.convs_dilated.1
    "dp.post_flows.*.convs.convs_sep.1": "duration_predictor.post_flows.*.conv_dds.convs_dilated.1",
    # 映射dp.post_flows.*.convs.convs_sep.2到duration_predictor.post_flows.*.conv_dds.convs_dilated.2
    "dp.post_flows.*.convs.convs_sep.2": "duration_predictor.post_flows.*.conv_dds.convs_dilated.2",
    # 映射dp.post_flows.*.convs.norms_1.0.gamma到duration_predictor.post_flows.*.conv_dds.norms_1.0.weight
    "dp.post_flows.*.convs.norms_1.0.gamma": "duration_predictor.post_flows.*.conv_dds.norms_1.0.weight",
    # 映射dp.post_flows.*.convs.norms_1.0.beta到duration_predictor.post_flows.*.conv_dds.norms_1.0.bias
    "dp.post_flows.*.convs.norms_1.0.beta": "duration_predictor.post_flows.*.conv_dds.norms_1.0.bias",
    # 映射dp.post_flows.*.convs.norms_1.1.gamma到duration_predictor.post_flows.*.conv_dds.norms_1.1.weight
    "dp.post_flows.*.convs.norms_1.1.gamma": "duration_predictor.post_flows.*.conv_dds.norms_1.1.weight",
    # 映射dp.post_flows.*.convs.norms_1.1.beta到duration_predictor.post_flows.*.conv_dds.norms_1.1.bias
    "dp.post_flows.*.convs.norms_1.1.beta": "duration_predictor.post_flows.*.conv_dds.norms_1.1.bias",
    # 映射dp.post_flows.*.convs.norms_1.2.gamma到duration_predictor.post_flows.*.conv_dds.norms_1.2.weight
    "dp.post_flows.*.convs.norms_1.2.gamma": "duration_predictor.post_flows.*.conv_dds.norms_1.2.weight",
    # 映射dp.post_flows.*.convs.norms_1.2.beta到duration_predictor.post_flows.*.conv_dds.norms_1.2.bias
    "dp.post_flows.*.convs.norms_1.2.beta": "duration_predictor.post_flows.*.conv_dds.norms_1.2.bias",
    # 映射dp.post_flows.*.convs.norms_2.0.gamma到duration_predictor.post_flows.*.conv_dds.norms_2.0.weight
    "dp.post_flows.*.convs.norms_2.0.gamma": "duration_predictor.post_flows.*.conv_dds.norms_2.0.weight",
    # 映射dp.post_flows.*.convs.norms_2.0.beta到duration_predictor.post_flows.*.conv_dds.norms_2.0.bias
    "dp.post_flows.*.convs.norms_2.0.beta": "duration_predictor.post_flows.*.conv_dds.norms_2.0.bias",
    # 映射dp.post_flows.*.convs.norms_2.1.gamma到duration_predictor.post_flows.*.conv_dds.norms_2.1.weight
    "dp.post_flows.*.convs.norms_2.1.gamma": "duration_predictor.post_flows.*.conv_dds.norms_2.1.weight",
    # 映射dp.post_flows.*.convs.norms_2.1.beta到duration_predictor.post_flows.*.conv_dds.norms_2.1.bias
    "dp.post_flows.*.convs.norms_2.1.beta": "duration_predictor.post_flows.*.conv_dds.norms_2.1.bias",
    # 映射dp.post_flows.*.convs.norms_2.2.gamma到duration_predictor.post_flows.*.conv_dds.norms_2.2.weight
    "dp.post_flows.*.convs.norms_2.2.gamma": "duration_predictor.post_flows.*.conv_dds.norms_2.2.weight",
    #映射dp.post_flows.*.convs.norms_2.2.beta到duration_predictor.post_flows.*.conv_dds.norms_2.2.bias
    "dp.post_flows.*.convs.norms_2.2.beta": "duration_predictor.post_flows.*.conv_dds.norms_2.2.bias",
    # 映射dp.cond到duration_predictor.cond，当num_speakers > 1时起作用
    "dp.cond": "duration_predictor.cond",  # num_speakers > 1
# 定义一个 MAPPING_FLOW 字典，将 flow.flows.*.pre 映射为 flow.flows.*.conv_pre，以此类推
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
    "flow.flows.*.enc.cond_layer": "flow.flows.*.wavenet.cond_layer",  # 当 num_speakers > 1 时成立
    "flow.flows.*.post": "flow.flows.*.conv_post",
}
# 定义一个 MAPPING_GENERATOR 字典，将 dec.conv_pre 映射为 decoder.conv_pre，以此类推
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
    "dec.cond": "decoder.cond",  # 当 num_speakers > 1 时成立
}
# 定义一个 MAPPING_POSTERIOR_ENCODER 字典，将 enc_q.pre 映射为 posterior_encoder.conv_pre，以此类推
MAPPING_POSTERIOR_ENCODER = {
    "enc_q.pre": "posterior_encoder.conv_pre",
    "enc_q.enc.in_layers.*": "posterior_encoder.wavenet.in_layers.*",
    "enc_q.enc.res_skip_layers.*": "posterior_encoder.wavenet.res_skip_layers.*",
    "enc_q.enc.cond_layer": "posterior_encoder.wavenet.cond_layer",  # 当 num_speakers > 1 时成立
    "enc_q.proj": "posterior_encoder.conv_proj",
}
# 定义一个 MAPPING 字典，将 MAPPING_TEXT_ENCODER、MAPPING_STOCHASTIC_DURATION_PREDICTOR、MAPPING_FLOW、MAPPING_GENERATOR、MAPPING_POSTERIOR_ENCODER、"emb_g" 映射为 "embed_speaker"，当 num_speakers > 1 时成立
MAPPING = {
    **MAPPING_TEXT_ENCODER,
    **MAPPING_STOCHASTIC_DURATION_PREDICTOR,
    **MAPPING_FLOW,
    **MAPPING_GENERATOR,
    **MAPPING_POSTERIOR_ENCODER,
    "emb_g": "embed_speaker",  # 当 num_speakers > 1 时成立
}
# 定义一个空列表 TOP_LEVEL_KEYS
TOP_LEVEL_KEYS = []
# 定义一个空列表 IGNORE_KEYS
IGNORE_KEYS = []

# 定义一个递归设置函数 set_recursively，接收 hf_pointer、key、value、full_name 和 weight_type 等参数
def set_recursively(hf_pointer, key, value, full_name, weight_type):
    # 将 key 按点号分隔后，逐级获取属性值
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    # 如果 weight_type 不为空，获取相应属性的 shape
    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    # 如果 key 以特定字符串结尾，则压缩 value 的最后一个维度
    if key.endswith(".k_proj") or key.endswith(".v_proj") or key.endswith(".q_proj") or key.endswith(".out_proj"):
        value = value.squeeze(-1)

    # 如果 hf_shape 和 value 的 shape 不相同，抛出异常
    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be "
            f"{value.shape} for {full_name}"
        )

    # 如果 weight_type 为 "weight"，则将 hf_pointer 的权重数据设置为 value
    if weight_type == "weight":
        hf_pointer.weight.data = value
    # 根据权重类型设置对应的模型参数值
    elif weight_type == "weight_g":
        # 设置模型参数的 weight_g 属性值为指定的 value
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        # 设置模型参数的 weight_v 属性值为指定的 value
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        # 设置模型参数的 bias 属性值为指定的 value
        hf_pointer.bias.data = value
    elif weight_type == "running_mean":
        # 设置模型参数的 running_mean 属性值为指定的 value
        hf_pointer.running_mean.data = value
    elif weight_type == "running_var":
        # 设置模型参数的 running_var 属性值为指定的 value
        hf_pointer.running_var.data = value
    elif weight_type == "num_batches_tracked":
        # 设置模型参数的 num_batches_tracked 属性值为指定的 value
        hf_pointer.num_batches_tracked.data = value
    else:
        # 若权重类型无法匹配已知的类型，则直接设置模型参数的值为指定的 value
        hf_pointer.data = value
    
    # 记录模型参数初始化的信息，包括参数键（带有权重类型后缀）和参数的完整名称
    logger.info(f"{key + ('.' + weight_type if weight_type is not None else '')} was initialized from {full_name}.")
# 检查名称是否应该被忽略
def should_ignore(name, ignore_keys):
    # 遍历所有需要忽略的关键词
    for key in ignore_keys:
        # 如果关键词以 ".*" 结尾，说明需要检查以该关键词开头的名称
        if key.endswith(".*"):
            if name.startswith(key[:-1]):
                return True
        # 如果关键词包含 ".*."，说明需要检查既包含前缀又包含后缀的名称
        elif ".*." in key:
            prefix, suffix = key.split(".*.")
            if prefix in name and suffix in name:
                return True
        # 如果关键词直接包含在名称中，则应该忽略
        elif key in name:
            return True
    # 如果上述条件都不满足，则不应该忽略
    return False


# 递归加载权重
def recursively_load_weights(fairseq_dict, hf_model):
    # 存储未使用的权重
    unused_weights = []

    # 遍历 fairseq 模型的权重
    for name, value in fairseq_dict.items():
        # 如果应该忽略该权重，则跳过
        if should_ignore(name, IGNORE_KEYS):
            logger.info(f"{name} was ignored")
            continue

        # 标记是否使用了该权重
        is_used = False
        # 遍历权重映射关系
        for key, mapped_key in MAPPING.items():
            # 如果关键词以 ".*" 结尾，则去掉 ".*"
            if key.endswith(".*"):
                key = key[:-1]
            # 如果关键词包含 "*"，则需要解析出层索引
            elif "*" in key:
                prefix, suffix = key.split(".*.")
                if prefix in name and suffix in name:
                    key = suffix

            # 如果名称包含关键词
            if key in name:
                # 标记已使用
                is_used = True
                # 根据映射关系处理层索引
                if mapped_key.endswith(".*"):
                    layer_index = name.split(key)[-1].split(".")[0]
                    mapped_key = mapped_key.replace("*", layer_index)
                elif "*" in mapped_key:
                    layer_index = name.split(key)[0].split(".")[-2]
                    # 对于特定层调整层索引
                    if "flow.flows" in mapped_key:
                        layer_index = str(int(layer_index) // 2)
                    if "duration_predictor.flows" in mapped_key or "duration_predictor.post_flows" in mapped_key:
                        layer_index = str(int(layer_index) // 2 + 1)
                    mapped_key = mapped_key.replace("*", layer_index)
                # 根据名称确定权重类型
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
                # 将权重设置到 HuggingFace 模型中
                set_recursively(hf_model, mapped_key, value, name, weight_type)
            continue
        # 如果该权重没有被使用，则记录下来
        if not is_used:
            unused_weights.append(name)

    # 输出未使用的权重
    logger.warning(f"Unused weights: {unused_weights}")


# 转换检查点
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
    将模型权重拷贝/粘贴/调整到 transformers 设计中。
    """
    # 当配置文件路径不为空时，从预训练模型加载配置
    if config_path is not None:
        config = VitsConfig.from_pretrained(config_path)
    else:
        # 否则使用默认配置
        config = VitsConfig()

    # 如果存在说话者数量，则设置配置中的说话者数量和说话者嵌入大小
    if num_speakers:
        config.num_speakers = num_speakers
        config.speaker_embedding_size = 256

    # 如果存在采样率，则设置配置中的采样率
    if sampling_rate:
        config.sampling_rate = sampling_rate

    # 如果检查点路径为空，则下载模型相关文件并设置相关的路径
    if checkpoint_path is None:
        logger.info(f"***Converting model: facebook/mms-tts {language}***")

        vocab_path = hf_hub_download(
            repo_id="facebook/mms-tts",
            filename="vocab.txt",
            subfolder=f"models/{language}",
        )
        config_file = hf_hub_download(
            repo_id="facebook/mms-tts",
            filename="config.json",
            subfolder=f"models/{language}",
        )
        checkpoint_path = hf_hub_download(
            repo_id="facebook/mms-tts",
            filename="G_100000.pth",
            subfolder=f"models/{language}",
        )

        with open(config_file, "r") as f:
            data = f.read()
            hps = json.loads(data)

        # 检查数据训练文件是否是 uroman 格式
        is_uroman = hps["data"]["training_files"].split(".")[-1] == "uroman"
        if is_uroman:
            logger.warning("For this checkpoint, you should use `uroman` to convert input text before tokenizing it!")
    else:
        # 检查点路径不为空时，设置 is_uroman 为 False
        logger.info(f"***Converting model: {checkpoint_path}***")
        is_uroman = False

    # 如果词汇表路径为空，则设置词汇表相关参数
    if vocab_path is None:
        _pad = "_"   # 设定填充符
        _punctuation = ';:,.!?¡¿—…"«»“” '   # 标点符号
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"   # 大小写字母
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈ... "  # 特殊字符
        symbols = _pad + _punctuation + _letters + _letters_ipa   # 所有字符
        # 根据字符生成对应的编号
        symbol_to_id = {s: i for i, s in enumerate(symbols)}
        phonemize = True   # 是否采用音素标记
    else:
        # 保存词汇表为临时的 JSON 文件
        symbols = [line.replace("\n", "") for line in open(vocab_path, encoding="utf-8").readlines()]
        symbol_to_id = {s: i for i, s in enumerate(symbols)}
        # MMS-TTS 不使用 <pad> 标记，因此设置为用于间隔字符的标记
        _pad = symbols[0]
        phonemize = False   # 不采用音素标记

    # 使用临时文件创建 tokenizer 对象
    with tempfile.NamedTemporaryFile() as tf:
        with open(tf.name, "w", encoding="utf-8") as f:
            f.write(json.dumps(symbol_to_id, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 创建 VitsTokenizer 对象
        tokenizer = VitsTokenizer(tf.name, language=language, phonemize=phonemize, is_uroman=is_uroman, pad_token=_pad)

    # 设置配置中的词汇表大小
    config.vocab_size = len(symbols)
    # 创建 VitsModel 对象
    model = VitsModel(config)

    # 应用权重归一化
    model.decoder.apply_weight_norm()

    # 加载原始检查点，将权重加载到模型中
    orig_checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    recursively_load_weights(orig_checkpoint["model"], model)

    # 移除权重归一化
    model.decoder.remove_weight_norm()

    # 保存预训练模型
    model.save_pretrained(pytorch_dump_folder_path)
    tokenizer.save_pretrained(pytorch_dump_folder_path)
    # 如果存在 repo_id（仓库标识）
    if repo_id:
        # 打印消息，指示正在将内容推送到中心（hub）
        print("Pushing to the hub...")
        # 调用 tokenizer 对象的 push_to_hub 方法，将模型标记器推送到指定的仓库
        tokenizer.push_to_hub(repo_id)
        # 调用 model 对象的 push_to_hub 方法，将模型推送到指定的仓库
        model.push_to_hub(repo_id)
# 如果当前脚本被作为主程序执行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数，用于指定原始检查点的本地路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Local path to original checkpoint")
    # 添加命令行参数，用于指定词汇表文件的路径
    parser.add_argument("--vocab_path", default=None, type=str, help="Path to vocab.txt")
    # 添加命令行参数，用于指定要转换的模型的 hf config.json 文件的路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加命令行参数，用于指定分词器语言（三字母代码）
    parser.add_argument("--language", default=None, type=str, help="Tokenizer language (three-letter code)")
    # 添加命令行参数，用于指定说话者数量
    parser.add_argument("--num_speakers", default=None, type=int, help="Number of speakers")
    # 添加命令行参数，用于指定模型训练的采样率
    parser.add_argument("--sampling_rate", default=None, type=int, help="Sampling rate on which the model was trained.")
    # 添加命令行参数，用于指定输出 PyTorch 模型的路径，此参数是必需的
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    # 添加命令行参数，用于指定转换后模型在 🤗 hub 上的上传位置
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    # 解析命令行参数并将其存储在 args 对象中
    args = parser.parse_args()
    # 调用 convert_checkpoint 函数，传递命令行参数作为参数
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