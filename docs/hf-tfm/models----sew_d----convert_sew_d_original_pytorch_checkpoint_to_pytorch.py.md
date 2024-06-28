# `.\models\sew_d\convert_sew_d_original_pytorch_checkpoint_to_pytorch.py`

```py
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
"""Convert SEW checkpoint."""


import argparse  # 导入解析命令行参数的模块
import json  # 导入处理 JSON 的模块
import os  # 导入操作系统相关功能的模块

import fairseq  # 导入 fairseq 库
import torch  # 导入 PyTorch 深度学习框架
from fairseq.data import Dictionary  # 从 fairseq 库中导入 Dictionary 类

# Register SEW's fairseq modules
from sew_asapp import tasks  # noqa: F401  # 导入 SEW 的 fairseq 模块，并标记忽略 F401 警告

from transformers import (
    SEWDConfig,  # 从 transformers 库导入 SEWDConfig 类
    SEWDForCTC,  # 从 transformers 库导入 SEWDForCTC 类
    SEWDModel,  # 从 transformers 库导入 SEWDModel 类
    Wav2Vec2CTCTokenizer,  # 从 transformers 库导入 Wav2Vec2CTCTokenizer 类
    Wav2Vec2FeatureExtractor,  # 从 transformers 库导入 Wav2Vec2FeatureExtractor 类
    Wav2Vec2Processor,  # 从 transformers 库导入 Wav2Vec2Processor 类
    logging,  # 导入 logging 模块
)


logging.set_verbosity_info()  # 设置日志记录级别为 info
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

MAPPING = {
    "post_extract_proj": "feature_projection",  # 映射关系的字典，将 SEW 中的 post_extract_proj 映射为 feature_projection
    "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",  # 将 SEW 中的 encoder.pos_conv.0 映射为 encoder.pos_conv_embed.conv
    "attention.self.query_proj": "encoder.encoder.layer.*.attention.self.query_proj",  # 将 SEW 中的 attention.self.query_proj 映射为 encoder.encoder.layer.*.attention.self.query_proj
    "attention.self.key_proj": "encoder.encoder.layer.*.attention.self.key_proj",  # 将 SEW 中的 attention.self.key_proj 映射为 encoder.encoder.layer.*.attention.self.key_proj
    "attention.self.value_proj": "encoder.encoder.layer.*.attention.self.value_proj",  # 将 SEW 中的 attention.self.value_proj 映射为 encoder.encoder.layer.*.attention.self.value_proj
    "attention.output.dense": "encoder.encoder.layer.*.attention.output.dense",  # 将 SEW 中的 attention.output.dense 映射为 encoder.encoder.layer.*.attention.output.dense
    "attention.output.LayerNorm": "encoder.encoder.layer.*.attention.output.LayerNorm",  # 将 SEW 中的 attention.output.LayerNorm 映射为 encoder.encoder.layer.*.attention.output.LayerNorm
    "intermediate.dense": "encoder.encoder.layer.*.intermediate.dense",  # 将 SEW 中的 intermediate.dense 映射为 encoder.encoder.layer.*.intermediate.dense
    "output.dense": "encoder.encoder.layer.*.output.dense",  # 将 SEW 中的 output.dense 映射为 encoder.encoder.layer.*.output.dense
    "output.LayerNorm": "encoder.encoder.layer.*.output.LayerNorm",  # 将 SEW 中的 output.LayerNorm 映射为 encoder.encoder.layer.*.output.LayerNorm
    "encoder.encoder.rel_embeddings": "encoder.encoder.rel_embeddings",  # 将 SEW 中的 encoder.encoder.rel_embeddings 映射为 encoder.encoder.rel_embeddings
    "encoder.encoder.LayerNorm": "encoder.encoder.LayerNorm",  # 将 SEW 中的 encoder.encoder.LayerNorm 映射为 encoder.encoder.LayerNorm
    "encoder.upsample.0": "encoder.upsample.projection",  # 将 SEW 中的 encoder.upsample.0 映射为 encoder.upsample.projection
    "encoder.layer_norm": "encoder.layer_norm",  # 将 SEW 中的 encoder.layer_norm 映射为 encoder.layer_norm
    "w2v_model.layer_norm": "layer_norm",  # 将 SEW 中的 w2v_model.layer_norm 映射为 layer_norm
    "w2v_encoder.proj": "lm_head",  # 将 SEW 中的 w2v_encoder.proj 映射为 lm_head
    "mask_emb": "masked_spec_embed",  # 将 SEW 中的 mask_emb 映射为 masked_spec_embed
}


def set_recursively(hf_pointer, key, value, full_name, weight_type):
    for attribute in key.split("."):  # 遍历 key 字符串按点分割后的列表
        hf_pointer = getattr(hf_pointer, attribute)  # 获取 hf_pointer 对象中的对应属性

    if weight_type is not None:  # 如果 weight_type 不为 None
        hf_shape = getattr(hf_pointer, weight_type).shape  # 获取 hf_pointer 对象中 weight_type 属性的形状
    else:
        hf_shape = hf_pointer.shape  # 否则获取 hf_pointer 对象本身的形状

    assert hf_shape == value.shape, (  # 断言 hf_pointer 对象的形状与 value 的形状相同
        f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
        f" {value.shape} for {full_name}"
    )

    if weight_type == "weight":  # 如果 weight_type 为 "weight"
        hf_pointer.weight.data = value  # 将 hf_pointer 对象的 weight 属性的数据设置为 value
    elif weight_type == "weight_g":  # 如果 weight_type 为 "weight_g"
        hf_pointer.weight_g.data = value  # 将 hf_pointer 对象的 weight_g 属性的数据设置为 value
    elif weight_type == "weight_v":  # 如果 weight_type 为 "weight_v"
        hf_pointer.weight_v.data = value  # 将 hf_pointer 对象的 weight_v 属性的数据设置为 value
    elif weight_type == "bias":  # 如果 weight_type 为 "bias"
        hf_pointer.bias.data = value  # 将 hf_pointer 对象的 bias 属性的数据设置为 value
    else:
        hf_pointer.data = value  # 否则将 hf_pointer 对象的数据设置为 value
    # 使用 logger 对象记录信息，此处使用了格式化字符串，根据条件拼接日志消息
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")
# 递归加载权重函数，用于将 Fairseq 模型的权重加载到 Hugging Face 模型中
def recursively_load_weights(fairseq_model, hf_model, is_finetuned):
    # 未使用的权重列表
    unused_weights = []
    # 获取 Fairseq 模型的状态字典
    fairseq_dict = fairseq_model.state_dict()

    # 根据是否微调选择特征提取器
    feature_extractor = hf_model.sew_d.feature_extractor if is_finetuned else hf_model.feature_extractor

    # 遍历 Fairseq 模型的状态字典
    for name, value in fairseq_dict.items():
        is_used = False
        # 如果权重名称中包含 "conv_layers"
        if "conv_layers" in name:
            # 加载卷积层权重
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        else:
            # 遍历映射字典中的每个键值对
            for key, mapped_key in MAPPING.items():
                # 根据是否微调调整映射键
                mapped_key = "sew_d." + mapped_key if (is_finetuned and mapped_key != "lm_head") else mapped_key

                # 如果权重名称中包含映射字典的键或者符合特定条件
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    is_used = True
                    # 如果映射键包含通配符 "*"
                    if "*" in mapped_key:
                        # 提取层索引号
                        layer_index = name.split(key)[0].split(".")[-2]
                        if not layer_index.isnumeric():
                            continue
                        # 替换通配符为具体层索引
                        mapped_key = mapped_key.replace("*", layer_index)
                    # 确定权重类型
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
                    # 递归设置 Hugging Face 模型的权重
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        # 如果权重未被使用，则加入到未使用的权重列表中
        if not is_used:
            unused_weights.append(name)

    # 记录未使用的权重信息
    logger.warning(f"Unused weights: {unused_weights}")


# 加载卷积层权重的函数
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 提取卷积层名称
    name = full_name.split("conv_layers.")[-1]
    # 分割名称中的各个部分
    items = name.split(".")
    # 提取层和类型 ID
    layer_id = int(items[0])
    type_id = int(items[1])
    # 如果权重类型为0（偏置）：
    if type_id == 0:
        # 如果权重名称中包含"bias"
        if "bias" in name:
            # 断言当前值的形状与特征提取器中指定卷积层的偏置数据形状相匹配，否则抛出异常
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            # 将值赋给特征提取器中指定卷积层的偏置数据
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            # 记录日志，表示第 layer_id 层的卷积层偏置从 full_name 初始化完成
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 如果权重名称中包含"weight"
        elif "weight" in name:
            # 断言当前值的形状与特征提取器中指定卷积层的权重数据形状相匹配，否则抛出异常
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            # 将值赋给特征提取器中指定卷积层的权重数据
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            # 记录日志，表示第 layer_id 层的卷积层权重从 full_name 初始化完成
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    
    # 如果权重类型为2且不使用组规范（group norm），或者权重类型为2且是第一层且使用组规范
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 如果权重名称中包含"bias"
        if "bias" in name:
            # 断言当前值的形状与特征提取器中指定卷积层的层归一化偏置数据形状相匹配，否则抛出异常
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape, (
                f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            # 将值赋给特征提取器中指定卷积层的层归一化偏置数据
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            # 记录日志，表示第 layer_id 层的卷积层层归一化偏置从 full_name 初始化完成
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 如果权重名称中包含"weight"
        elif "weight" in name:
            # 断言当前值的形状与特征提取器中指定卷积层的层归一化权重数据形状相匹配，否则抛出异常
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            )
            # 将值赋给特征提取器中指定卷积层的层归一化权重数据
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            # 记录日志，表示第 layer_id 层的卷积层层归一化权重从 full_name 初始化完成
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    
    # 如果以上条件都不满足，则将未使用的权重名称添加到未使用权重列表中
    else:
        unused_weights.append(full_name)
# 在给定模型和微调标志的情况下，生成SEWDConfig对象，配置转换参数
def convert_config(model, is_finetuned):
    # 创建一个SEWDConfig对象，用于存储配置信息
    config = SEWDConfig()

    # 根据是否微调选择配置信息来源
    if is_finetuned:
        # 如果是微调，从模型的w2v_encoder属性中获取w2v_model的配置信息
        fs_config = model.w2v_encoder.w2v_model.cfg
    else:
        # 如果不是微调，直接使用模型本身的配置信息
        fs_config = model.cfg

    # 将转换后的参数赋值给SEWDConfig对象的各个属性
    config.conv_bias = fs_config.conv_bias
    conv_layers = eval(fs_config.conv_feature_layers)
    config.conv_dim = [x[0] for x in conv_layers]
    config.conv_kernel = [x[1] for x in conv_layers]
    config.conv_stride = [x[2] for x in conv_layers]
    config.feat_extract_activation = "gelu"
    config.feat_extract_norm = "layer" if fs_config.extractor_mode == "layer_norm" else "group"
    config.final_dropout = 0.0
    config.hidden_act = fs_config.activation_fn.name
    config.hidden_size = fs_config.encoder_embed_dim
    config.initializer_range = 0.02
    config.intermediate_size = fs_config.encoder_ffn_embed_dim
    config.layer_norm_eps = 1e-5
    config.layerdrop = fs_config.encoder_layerdrop
    config.num_attention_heads = fs_config.encoder_attention_heads
    config.num_conv_pos_embedding_groups = fs_config.conv_pos_groups
    config.num_conv_pos_embeddings = fs_config.conv_pos
    config.num_feat_extract_layers = len(conv_layers)
    config.num_hidden_layers = fs_config.encoder_layers
    config.squeeze_factor = fs_config.squeeze_factor

    # 针对DeBERTa模型的特定参数设置
    config.max_position_embeddings = fs_config.max_position_embeddings
    config.position_buckets = fs_config.position_buckets
    config.share_att_key = fs_config.share_att_key
    config.relative_attention = fs_config.relative_attention
    config.position_biased_input = fs_config.position_biased_input
    config.pos_att_type = tuple(fs_config.pos_att_type.split("|"))
    config.norm_rel_ebd = fs_config.norm_rel_ebd

    # 对于微调模型，处理可能被Wav2VecCtc模型覆盖的参数
    if is_finetuned:
        fs_config = model.cfg
        config.final_dropout = fs_config.final_dropout
        config.layerdrop = fs_config.layerdrop

    # 设置剩余的配置参数
    config.activation_dropout = fs_config.activation_dropout
    config.apply_spec_augment = fs_config.mask_prob > 0 or fs_config.mask_channel_prob > 0
    config.attention_dropout = fs_config.attention_dropout
    config.feat_proj_dropout = fs_config.dropout_input
    config.hidden_dropout = fs_config.dropout
    config.mask_feature_length = fs_config.mask_channel_length
    config.mask_feature_prob = fs_config.mask_channel_prob
    config.mask_time_length = fs_config.mask_length
    config.mask_time_prob = fs_config.mask_prob

    # 设置特定的特征提取器类型和分词器类
    config.feature_extractor_type = "Wav2Vec2FeatureExtractor"
    config.tokenizer_class = "Wav2Vec2CTCTokenizer"

    # 返回配置对象
    return config


# 使用torch.no_grad()装饰器，确保在此函数内部禁用梯度计算
@torch.no_grad()
def convert_sew_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 在此处实现模型权重复制、粘贴和调整的逻辑，转换为transformers设计
    # 函数体内部逻辑尚未提供，需根据具体需求补充实现
    pass
    # 如果已经微调过模型
    if is_finetuned:
        # 加载模型、任务和参数覆盖，使用指定的检查点路径
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1])}
        )
    else:
        # 加载模型、任务和参数覆盖，使用指定的检查点路径
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])

    # 如果提供了配置路径，从预训练配置加载配置信息
    if config_path is not None:
        config = SEWDConfig.from_pretrained(config_path)
    else:
        # 否则根据模型和微调状态生成配置信息
        config = convert_config(model[0], is_finetuned)

    # 设置模型为评估模式
    model = model[0].eval()

    # 根据配置的特征提取器参数，决定是否返回注意力掩码
    return_attention_mask = True if config.feat_extract_norm == "layer" else False
    # 创建 Wav2Vec2FeatureExtractor 特征提取器对象
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0,
        do_normalize=True,
        return_attention_mask=return_attention_mask,
    )

    # 如果已经微调过模型
    if is_finetuned:
        if dict_path:
            # 加载目标字典并调整特殊标记的索引，以适应当前任务
            target_dict = Dictionary.load(dict_path)
            target_dict.indices[target_dict.bos_word] = target_dict.pad_index
            target_dict.indices[target_dict.pad_word] = target_dict.bos_index
            config.bos_token_id = target_dict.pad_index
            config.pad_token_id = target_dict.bos_index
            config.eos_token_id = target_dict.eos_index
            config.vocab_size = len(target_dict.symbols)
            vocab_path = os.path.join(pytorch_dump_folder_path, "vocab.json")
            # 如果指定的 PyTorch 转储文件夹路径不是目录，则记录错误并返回
            if not os.path.isdir(pytorch_dump_folder_path):
                logger.error("--pytorch_dump_folder_path ({}) should be a directory".format(pytorch_dump_folder_path))
                return
            # 如果目录不存在则创建
            os.makedirs(pytorch_dump_folder_path, exist_ok=True)
            # 将目标字典的索引保存为 JSON 文件
            with open(vocab_path, "w", encoding="utf-8") as vocab_handle:
                json.dump(target_dict.indices, vocab_handle)
            # 创建 Wav2Vec2CTCTokenizer 标记器对象
            tokenizer = Wav2Vec2CTCTokenizer(
                vocab_path,
                unk_token=target_dict.unk_word,
                pad_token=target_dict.pad_word,
                bos_token=target_dict.bos_word,
                eos_token=target_dict.eos_word,
                word_delimiter_token="|",
                do_lower_case=False,
            )
            # 创建 Wav2Vec2Processor 处理器对象
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            # 将处理器保存到指定的 PyTorch 转储文件夹路径
            processor.save_pretrained(pytorch_dump_folder_path)

        # 创建 SEWDForCTC 模型对象
        hf_model = SEWDForCTC(config)
    else:
        # 创建 SEWDModel 模型对象
        hf_model = SEWDModel(config)
        # 将特征提取器保存到指定的 PyTorch 转储文件夹路径
        feature_extractor.save_pretrained(pytorch_dump_folder_path)

    # 递归地加载权重到模型
    recursively_load_weights(model, hf_model, is_finetuned)

    # 将模型保存到指定的 PyTorch 转储文件夹路径
    hf_model.save_pretrained(pytorch_dump_folder_path)
# 如果该脚本作为主程序运行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加参数：输出 PyTorch 模型的路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加参数：fairseq 检查点的路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    # 添加参数：微调模型的字典路径
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    # 添加参数：要转换的模型的 hf config.json 文件路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加参数：标志，指示要转换的模型是否为微调模型
    parser.add_argument(
        "--is_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用 convert_sew_checkpoint 函数，传递解析后的参数
    convert_sew_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path, args.is_finetuned
    )
```