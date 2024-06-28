# `.\models\wavlm\convert_wavlm_original_pytorch_checkpoint_to_pytorch.py`

```py
# coding=utf-8
# 声明文件编码格式为UTF-8

# Copyright 2021 The HuggingFace Inc. team.
# 版权声明，指出代码版权属于HuggingFace团队，日期为2021年

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据Apache许可证2.0版（"许可证"）授权使用本文件

# you may not use this file except in compliance with the License.
# 除非遵守许可证，否则不得使用此文件

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证的副本

# http://www.apache.org/licenses/LICENSE-2.0
# http://www.apache.org/licenses/LICENSE-2.0的许可证文本

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是按"原样"分发的，
# 没有任何明示或暗示的担保或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 查看许可证以了解特定语言的权限和限制

"""Convert WavLM checkpoint."""
# 脚本用于转换WavLM检查点

import argparse
# 导入命令行参数解析模块

import torch
# 导入PyTorch库

# Step 1. clone https://github.com/microsoft/unilm
# 步骤1：克隆https://github.com/microsoft/unilm

# Step 2. git checkout to https://github.com/microsoft/unilm/commit/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd
# 步骤2：切换到https://github.com/microsoft/unilm/commit/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd

# Step 3. cd unilm
# 步骤3：进入unilm目录

# Step 4. ln -s $(realpath wavlm/modules.py) ./  # create simlink
# 步骤4：创建符号链接指向realpath wavlm/modules.py

# import classes
# 导入自定义类

from unilm.wavlm.WavLM import WavLM as WavLMOrig
# 从unilm.wavlm.WavLM模块中导入WavLM类，并重命名为WavLMOrig

from unilm.wavlm.WavLM import WavLMConfig as WavLMConfigOrig
# 从unilm.wavlm.WavLM模块中导入WavLMConfig类，并重命名为WavLMConfigOrig

from transformers import WavLMConfig, WavLMModel, logging
# 从transformers库中导入WavLMConfig、WavLMModel类和logging模块

logging.set_verbosity_info()
# 设置日志记录级别为信息级别

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器对象

MAPPING = {
    "post_extract_proj": "feature_projection.projection",
    "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",
    "self_attn.k_proj": "encoder.layers.*.attention.k_proj",
    "self_attn.v_proj": "encoder.layers.*.attention.v_proj",
    "self_attn.q_proj": "encoder.layers.*.attention.q_proj",
    "self_attn.out_proj": "encoder.layers.*.attention.out_proj",
    "self_attn.grep_linear": "encoder.layers.*.attention.gru_rel_pos_linear",
    "self_attn.relative_attention_bias": "encoder.layers.*.attention.rel_attn_embed",
    "self_attn.grep_a": "encoder.layers.*.attention.gru_rel_pos_const",
    "self_attn_layer_norm": "encoder.layers.*.layer_norm",
    "fc1": "encoder.layers.*.feed_forward.intermediate_dense",
    "fc2": "encoder.layers.*.feed_forward.output_dense",
    "final_layer_norm": "encoder.layers.*.final_layer_norm",
    "encoder.layer_norm": "encoder.layer_norm",
    "w2v_model.layer_norm": "feature_projection.layer_norm",
    "quantizer.weight_proj": "quantizer.weight_proj",
    "quantizer.vars": "quantizer.codevectors",
    "project_q": "project_q",
    "final_proj": "project_hid",
    "w2v_encoder.proj": "ctc_proj",
    "mask_emb": "masked_spec_embed",
}
# 映射表，将原始模型的参数路径映射到转换后模型的参数路径

TOP_LEVEL_KEYS = [
    "ctc_proj",
    "quantizer.weight_proj",
    "quantizer.codevectors",
    "project_q",
    "project_hid",
]
# 最顶层的关键字列表，这些关键字需要特殊处理

def set_recursively(hf_pointer, key, value, full_name, weight_type):
    # 递归设置函数，用于设置模型参数

    for attribute in key.split("."):
        # 遍历key中的属性列表
        hf_pointer = getattr(hf_pointer, attribute)
        # 获取hf_pointer对象中的对应属性

    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
        # 如果存在权重类型，则获取该类型的形状
    else:
        hf_shape = hf_pointer.shape
        # 否则获取hf_pointer的形状

    assert hf_shape == value.shape, (
        f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
        f" {value.shape} for {full_name}"
    )
    # 断言，确保hf_pointer的形状与value的形状相匹配，否则输出错误信息
    if weight_type == "weight":
        # 如果权重类型是 "weight"，则将数值赋给模型的权重数据
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        # 如果权重类型是 "weight_g"，则将数值赋给模型的梯度权重数据
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        # 如果权重类型是 "weight_v"，则将数值赋给模型的版本权重数据
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        # 如果权重类型是 "bias"，则将数值赋给模型的偏置数据
        hf_pointer.bias.data = value
    else:
        # 如果没有特定的权重类型匹配，则直接将数值赋给模型指针的数据
        hf_pointer.data = value

    # 记录信息日志，指示哪些参数被初始化，并显示完整的名称路径
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")
# 递归加载权重函数，从Fairseq模型加载到Hugging Face模型
def recursively_load_weights(fairseq_model, hf_model):
    # 未使用的权重列表
    unused_weights = []
    # 获取Fairseq模型的状态字典
    fairseq_dict = fairseq_model.state_dict()

    # 获取Hugging Face模型的特征提取器
    feature_extractor = hf_model.feature_extractor

    # 遍历Fairseq模型状态字典中的每个键值对
    for name, value in fairseq_dict.items():
        # 标记此权重是否被使用过，默认为未使用
        is_used = False

        # 如果名称中包含"conv_layers"
        if "conv_layers" in name:
            # 调用加载卷积层的函数
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            # 标记此权重已被使用
            is_used = True
        else:
            # 遍历映射表中的每个键值对
            for key, mapped_key in MAPPING.items():
                # 如果映射表中的键在名称中或者以"w2v_model."结尾的键在名称的首部
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    # 标记此权重已被使用
                    is_used = True
                    # 如果映射键包含"*"，则替换为层索引
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    # 根据名称确定权重类型
                    if "weight_g" in name:
                        weight_type = "weight_g"
                    elif "weight_v" in name:
                        weight_type = "weight_v"
                    elif "bias" in name and "relative_attention_bias" not in name:
                        weight_type = "bias"
                    elif "weight" in name:
                        weight_type = "weight"
                    else:
                        weight_type = None

                    # 递归设置Hugging Face模型中的权重
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        # 如果未使用此权重，将其名称添加到未使用的权重列表中
        if not is_used:
            unused_weights.append(name)

    # 输出警告，显示未使用的权重列表
    logger.warning(f"Unused weights: {unused_weights}")


# 加载卷积层权重的函数
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 获取卷积层的名称
    name = full_name.split("conv_layers.")[-1]
    # 分割名称为列表
    items = name.split(".")
    # 提取层索引和类型索引
    layer_id = int(items[0])
    type_id = int(items[1])

    # 如果类型索引为0
    if type_id == 0:
        # 如果名称中包含"bias"
        if "bias" in name:
            # 断言当前值的形状与特征提取器中对应卷积层的偏置数据形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            # 将值赋给特征提取器中对应卷积层的偏置数据
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            # 记录日志，显示卷积层的初始化
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 如果名称中包含"weight"
        elif "weight" in name:
            # 断言当前值的形状与特征提取器中对应卷积层的权重数据形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            # 将值赋给特征提取器中对应卷积层的权重数据
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            # 记录日志，显示卷积层的初始化
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 如果 type_id 为 2 并且不使用组归一化，或者 type_id 为 2、layer_id 为 0 并且使用组归一化，则执行以下操作
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 如果名称中包含 "bias"
        if "bias" in name:
            # 断言值的形状与特征提取器中对应卷积层的层归一化偏置数据的形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape, (
                f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            # 将值赋给特征提取器中对应卷积层的层归一化偏置数据
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            # 记录日志，指示层归一化权重已从指定名称的初始化值初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 如果名称中包含 "weight"
        elif "weight" in name:
            # 断言值的形状与特征提取器中对应卷积层的层归一化权重数据的形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            )
            # 将值赋给特征提取器中对应卷积层的层归一化权重数据
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            # 记录日志，指示层归一化权重已从指定名称的初始化值初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    # 如果不满足上述条件，则将未使用的权重名称添加到未使用的权重列表中
    else:
        unused_weights.append(full_name)
@torch.no_grad()
def convert_wavlm_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None):
    # 使用torch.no_grad()装饰器，禁用梯度计算，以节省内存和加快推理速度
    # 载入预训练的检查点文件
    checkpoint = torch.load(checkpoint_path)
    # 使用检查点中的配置信息创建WavLMConfigOrig对象
    cfg = WavLMConfigOrig(checkpoint["cfg"])
    # 使用WavLMOrig类和检查点中的模型状态字典创建模型
    model = WavLMOrig(cfg)
    model.load_state_dict(checkpoint["model"])
    # 将模型设置为评估模式，通常用于推理阶段
    model.eval()

    # 如果提供了配置文件路径，则使用预训练模型的配置创建WavLMConfig对象
    if config_path is not None:
        config = WavLMConfig.from_pretrained(config_path)
    else:
        # 否则创建一个空的配置对象
        config = WavLMConfig()

    # 创建一个新的WavLMModel对象
    hf_wavlm = WavLMModel(config)

    # 递归地加载模型的权重到hf_wavlm中
    recursively_load_weights(model, hf_wavlm)

    # 将转换后的PyTorch模型保存到指定的文件夹路径中
    hf_wavlm.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数，用于指定输出的PyTorch模型路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加命令行参数，用于指定fairseq检查点文件的路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    # 添加命令行参数，用于指定要转换模型的hf配置文件的路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 解析命令行参数
    args = parser.parse_args()
    # 调用convert_wavlm_checkpoint函数，开始模型转换过程
    convert_wavlm_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)
```