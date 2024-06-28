# `.\models\hubert\convert_distilhubert_original_s3prl_checkpoint_to_pytorch.py`

```
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert Hubert checkpoint."""


import argparse  # 导入命令行参数解析模块

import torch  # 导入PyTorch库
from s3prl.hub import distilhubert  # 从s3prl库的hub模块导入distilhubert模型

from transformers import HubertConfig, HubertModel, Wav2Vec2FeatureExtractor, logging  # 从transformers库导入Hubert相关类和logging模块


logging.set_verbosity_info()  # 设置日志输出级别为信息级别
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

MAPPING = {
    "post_extract_proj": "feature_projection.projection",
    "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",
    "self_attn.k_proj": "encoder.layers.*.attention.k_proj",
    "self_attn.v_proj": "encoder.layers.*.attention.v_proj",
    "self_attn.q_proj": "encoder.layers.*.attention.q_proj",
    "self_attn.out_proj": "encoder.layers.*.attention.out_proj",
    "self_attn_layer_norm": "encoder.layers.*.layer_norm",
    "fc1": "encoder.layers.*.feed_forward.intermediate_dense",
    "fc2": "encoder.layers.*.feed_forward.output_dense",
    "final_layer_norm": "encoder.layers.*.final_layer_norm",
    "encoder.layer_norm": "encoder.layer_norm",
    "mask_emb": "masked_spec_embed",
}


def set_recursively(hf_pointer, key, value, full_name, weight_type):
    """
    递归设置模型参数的函数。
    
    Args:
        hf_pointer (object): 要设置的模型参数的指针对象。
        key (str): 参数的名称路径。
        value (torch.Tensor): 要设置的参数值。
        full_name (str): 参数的完整名称。
        weight_type (str): 参数类型，如'weight', 'bias'等。

    Raises:
        AssertionError: 如果要设置的参数形状与预期不符合，抛出异常。
    """
    for attribute in key.split("."):  # 按点号分割参数路径并遍历
        hf_pointer = getattr(hf_pointer, attribute)  # 逐级获取属性对象

    if weight_type is not None:  # 如果参数类型不为空
        hf_shape = getattr(hf_pointer, weight_type).shape  # 获取指定类型的参数形状
    else:
        hf_shape = hf_pointer.shape  # 否则获取参数的形状

    assert hf_shape == value.shape, (  # 断言参数形状与值的形状是否一致
        f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
        f" {value.shape} for {full_name}"
    )

    # 根据参数类型设置模型参数的值
    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    else:
        hf_pointer.data = value

    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")


def recursively_load_weights(fairseq_model, hf_model):
    """
    递归加载权重函数。

    Args:
        fairseq_model: Fairseq模型对象。
        hf_model: HuggingFace模型对象。
    """
    unused_weights = []  # 未使用的权重列表
    fairseq_dict = fairseq_model.state_dict()  # 获取Fairseq模型的状态字典

    feature_extractor = hf_model.feature_extractor  # 获取HuggingFace模型的特征提取器
    # 遍历 fairseq_dict 字典中的每个键值对，键是参数名，值是对应的张量值
    for name, value in fairseq_dict.items():
        # 标记当前参数是否被使用的布尔变量，默认为未使用
        is_used = False
        
        # 检查参数名中是否包含 "conv_layers"
        if "conv_layers" in name:
            # 如果包含，则调用 load_conv_layer 函数加载卷积层参数
            load_conv_layer(
                name,  # 参数名
                value,  # 参数值
                feature_extractor,  # 特征提取器对象
                unused_weights,  # 未使用的权重列表
                hf_model.config.feat_extract_norm == "group",  # 特征提取器配置的归一化是否为 "group"
            )
            is_used = True  # 将标记设置为已使用
        
        else:
            # 如果不包含 "conv_layers"，则遍历 MAPPING 字典中的映射关系
            for key, mapped_key in MAPPING.items():
                mapped_key = mapped_key

                # 检查当前参数名中是否包含当前遍历到的 key
                if key in name:
                    is_used = True  # 标记为已使用

                    # 如果 mapped_key 中包含通配符 "*"，则替换为对应的层索引
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)

                    # 根据参数名的后缀确定参数类型，可能是权重的不同类型
                    if "weight_g" in name:
                        weight_type = "weight_g"
                    elif "weight_v" in name:
                        weight_type = "weight_v"
                    elif "weight" in name:
                        weight_type = "weight"
                    elif "bias" in name:
                        weight_type = "bias"
                    else:
                        weight_type = None
                    
                    # 递归设置 hf_model 中的参数值，使用 mapped_key 作为路径
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                
                continue  # 继续下一个映射关系的检查
        
        # 如果参数未被使用，则将参数名添加到 unused_weights 列表中
        if not is_used:
            unused_weights.append(name)

    # 记录未使用的权重参数名到日志中
    logger.warning(f"Unused weights: {unused_weights}")
# 加载卷积层参数的函数，根据给定的全名、数值、特征提取器、未使用的权重列表和是否使用组归一化来操作
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 从完整名称中提取层名称
    name = full_name.split("conv_layers.")[-1]
    # 根据点号分割名称，获取层和类型编号
    items = name.split(".")
    layer_id = int(items[0])  # 提取层编号
    type_id = int(items[1])   # 提取类型编号

    # 如果类型编号为0，处理偏置或权重参数
    if type_id == 0:
        if "bias" in name:
            # 检查并设置偏置参数值，同时记录日志
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            # 检查并设置权重参数值，同时记录日志
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 如果类型编号为2且未使用组归一化，或者类型编号为2且层编号为0且使用了组归一化，则处理层归一化参数
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        if "bias" in name:
            # 检查并设置层归一化偏置参数值，同时记录日志
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape, (
                f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            # 检查并设置层归一化权重参数值，同时记录日志
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        unused_weights.append(full_name)


# 将给定模型转换为Hubert配置
def convert_config(model):
    config = HubertConfig()  # 创建一个Hubert配置对象
    fs_config = model.config  # 获取模型的配置信息

    # 设置Hubert配置的各个参数
    config.activation_dropout = fs_config.activation_dropout
    config.apply_spec_augment = False
    config.attention_dropout = fs_config.attention_dropout
    config.conv_bias = False
    conv_layers = eval(fs_config.extractor_conv_feature_layers)  # 评估提取器的卷积特征层配置
    config.conv_dim = [x[0] for x in conv_layers]  # 提取卷积层维度信息
    config.conv_kernel = [x[1] for x in conv_layers]  # 提取卷积层核大小信息
    config.conv_stride = [x[2] for x in conv_layers]  # 提取卷积层步幅信息
    config.feat_extract_activation = "gelu"  # 设置特征提取激活函数为GELU
    config.feat_extract_norm = "layer" if fs_config.extractor_mode == "layer_norm" else "group"  # 设置特征提取归一化方式
    config.feat_proj_layer_norm = False  # 不使用特征投影层归一化
    # 设置特征投影层的dropout率为0.0
    config.feat_proj_dropout = 0.0
    # 设置最终输出层的dropout率为0.0
    config.final_dropout = 0.0
    # 设置隐藏层激活函数为fs_config.activation_fn
    config.hidden_act = fs_config.activation_fn
    # 设置隐藏层的dropout率为fs_config.dropout
    config.hidden_dropout = fs_config.dropout
    # 设置隐藏层的大小为fs_config.encoder_embed_dim
    config.hidden_size = fs_config.encoder_embed_dim
    # 设置初始化范围为0.02
    config.initializer_range = 0.02
    # 设置中间层的大小为fs_config.encoder_ffn_embed_dim
    config.intermediate_size = fs_config.encoder_ffn_embed_dim
    # 设置层归一化的epsilon值为1e-5
    config.layer_norm_eps = 1e-5
    # 设置层drop的比例为0.0
    config.layerdrop = 0.0
    # 设置注意力头的数量为fs_config.encoder_attention_heads
    config.num_attention_heads = fs_config.encoder_attention_heads
    # 设置卷积位置嵌入的分组数为fs_config.conv_pos_groups
    config.num_conv_pos_embedding_groups = fs_config.conv_pos_groups
    # 设置卷积位置嵌入的数量为fs_config.conv_pos
    config.num_conv_pos_embeddings = fs_config.conv_pos
    # 设置特征提取层的数量为conv_layers列表的长度
    config.num_feat_extract_layers = len(conv_layers)
    # 设置隐藏层的数量为fs_config.encoder_layers
    config.num_hidden_layers = fs_config.encoder_layers
    
    # 返回配置对象config
    return config
# 使用 `torch.no_grad()` 上下文管理器，确保在转换模型检查点期间不计算梯度
@torch.no_grad()
# 定义函数 `convert_hubert_checkpoint`，用于将模型权重从 Hubert 转换到 Transformers 设计
def convert_hubert_checkpoint(pytorch_dump_folder_path, config_path=None):
    # 使用 `distilhubert()` 函数获取 Hubert 模型，并访问其内部的子模型
    model = distilhubert().model.model

    # 如果提供了 `config_path`，从预训练的配置文件加载 HubertConfig
    if config_path is not None:
        config = HubertConfig.from_pretrained(config_path)
    else:
        # 否则，调用 `convert_config` 函数将 Hubert 模型的配置转换为 HubertConfig
        config = convert_config(model)
    # 将模型设置为评估模式（不计算梯度）
    model = model.eval()

    # 创建一个 Wav2Vec2FeatureExtractor 实例，用于音频特征提取
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0,
        do_normalize=False,
        return_attention_mask=False,
    )
    
    # 根据提供的 `config` 创建 HubertModel 实例
    hf_model = HubertModel(config)

    # 递归加载 Hubert 模型的权重到 Transformers 的 HubertModel 中
    recursively_load_weights(model, hf_model)

    # 将特征提取器的状态保存到给定的 `pytorch_dump_folder_path`
    feature_extractor.save_pretrained(pytorch_dump_folder_path)
    # 将转换后的 HubertModel 的状态保存到相同的 `pytorch_dump_folder_path`
    hf_model.save_pretrained(pytorch_dump_folder_path)


# 如果当前脚本作为主程序运行，则执行以下代码
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数 `--pytorch_dump_folder_path`，指定输出 PyTorch 模型的路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加命令行参数 `--config_path`，指定待转换模型的配置文件路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 解析命令行参数
    args = parser.parse_args()
    # 调用 `convert_hubert_checkpoint` 函数，传入解析后的命令行参数
    convert_hubert_checkpoint(args.pytorch_dump_folder_path, args.config_path)
```