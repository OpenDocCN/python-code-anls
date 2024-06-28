# `.\models\sew\convert_sew_original_pytorch_checkpoint_to_pytorch.py`

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

import argparse  # 导入用于处理命令行参数的模块
import json  # 导入处理JSON格式数据的模块
import os  # 导入操作系统相关功能的模块

import fairseq  # 导入fairseq库
import torch  # 导入PyTorch库
from fairseq.data import Dictionary  # 从fairseq库中导入Dictionary类

# Register SEW's fairseq modules
from sew_asapp import tasks  # noqa: F401  # 导入SEW的fairseq模块，不会引发F401未使用警告

from transformers import (  # 从transformers库中导入多个类和函数
    SEWConfig,  # SEW模型配置类
    SEWForCTC,  # SEW CTC模型类
    SEWModel,  # SEW模型类
    Wav2Vec2CTCTokenizer,  # Wav2Vec2 CTC标记器类
    Wav2Vec2FeatureExtractor,  # Wav2Vec2特征提取器类
    Wav2Vec2Processor,  # Wav2Vec2处理器类
    logging,  # 日志记录模块
)

logging.set_verbosity_info()  # 设置日志级别为info
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

MAPPING = {
    "post_extract_proj": "feature_projection",  # 映射fairseq模型的post_extract_proj到feature_projection
    "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",  # 映射fairseq模型的encoder.pos_conv.0到encoder.pos_conv_embed.conv
    "self_attn.k_proj": "encoder.layers.*.attention.k_proj",  # 映射fairseq模型的self_attn.k_proj到encoder.layers.*.attention.k_proj
    "self_attn.v_proj": "encoder.layers.*.attention.v_proj",  # 映射fairseq模型的self_attn.v_proj到encoder.layers.*.attention.v_proj
    "self_attn.q_proj": "encoder.layers.*.attention.q_proj",  # 映射fairseq模型的self_attn.q_proj到encoder.layers.*.attention.q_proj
    "self_attn.out_proj": "encoder.layers.*.attention.out_proj",  # 映射fairseq模型的self_attn.out_proj到encoder.layers.*.attention.out_proj
    "self_attn_layer_norm": "encoder.layers.*.layer_norm",  # 映射fairseq模型的self_attn_layer_norm到encoder.layers.*.layer_norm
    "fc1": "encoder.layers.*.feed_forward.intermediate_dense",  # 映射fairseq模型的fc1到encoder.layers.*.feed_forward.intermediate_dense
    "fc2": "encoder.layers.*.feed_forward.output_dense",  # 映射fairseq模型的fc2到encoder.layers.*.feed_forward.output_dense
    "final_layer_norm": "encoder.layers.*.final_layer_norm",  # 映射fairseq模型的final_layer_norm到encoder.layers.*.final_layer_norm
    "encoder.upsample.0": "encoder.upsample.projection",  # 映射fairseq模型的encoder.upsample.0到encoder.upsample.projection
    "encoder.layer_norm": "encoder.layer_norm",  # 映射fairseq模型的encoder.layer_norm到encoder.layer_norm
    "w2v_model.layer_norm": "layer_norm",  # 映射fairseq模型的w2v_model.layer_norm到layer_norm
    "w2v_encoder.proj": "lm_head",  # 映射fairseq模型的w2v_encoder.proj到lm_head
    "mask_emb": "masked_spec_embed",  # 映射fairseq模型的mask_emb到masked_spec_embed
}


def set_recursively(hf_pointer, key, value, full_name, weight_type):
    # 递归设置hf_pointer指定路径下的权重或属性值
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    assert hf_shape == value.shape, (
        f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
        f" {value.shape} for {full_name}"
    )

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


def recursively_load_weights(fairseq_model, hf_model, is_finetuned):
    # 递归加载fairseq模型的权重到hf_model中
    unused_weights = []  # 未使用的权重列表
    fairseq_dict = fairseq_model.state_dict()  # 获取fairseq模型的状态字典
    # 根据是否微调选择特征提取器模型
    feature_extractor = hf_model.sew.feature_extractor if is_finetuned else hf_model.feature_extractor

    # 遍历 fairseq_dict 中的每个键值对
    for name, value in fairseq_dict.items():
        # 标记该权重是否被使用的布尔值
        is_used = False
        
        # 检查名字中是否包含 "conv_layers"
        if "conv_layers" in name:
            # 如果包含，则加载卷积层权重
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        else:
            # 遍历 MAPPING 中的每个键值对
            for key, mapped_key in MAPPING.items():
                # 根据微调状态和键映射更新 mapped_key
                mapped_key = "sew." + mapped_key if (is_finetuned and mapped_key != "lm_head") else mapped_key

                # 检查 name 是否包含 key 或者与 key 后面的部分匹配
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    is_used = True
                    
                    # 如果 mapped_key 包含通配符 "*"，则替换为层索引
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    
                    # 根据名字判断权重类型
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
                    
                    # 递归设置 hf_model 中的权重值
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        
        # 如果未使用该权重，则将其添加到未使用权重列表中
        if not is_used:
            unused_weights.append(name)

    # 记录未使用的权重信息到日志中
    logger.warning(f"Unused weights: {unused_weights}")
# 加载卷积层参数的函数，根据完整名称 `full_name`、数值 `value`、特征提取器 `feature_extractor`、未使用的权重列表 `unused_weights` 和是否使用组归一化 `use_group_norm` 进行操作
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 从完整名称中提取出卷积层的名字 `name`
    name = full_name.split("conv_layers.")[-1]
    # 按点号分割名字，得到列表 `items`
    items = name.split(".")
    # 将第一个元素转换为整数，作为层的索引 `layer_id`
    layer_id = int(items[0])
    # 将第二个元素转换为整数，作为类型的索引 `type_id`
    type_id = int(items[1])

    # 根据类型 `type_id` 执行不同的操作
    if type_id == 0:
        # 如果名称中包含 "bias"，则更新对应卷积层的偏置参数
        if "bias" in name:
            # 断言新值 `value` 的形状与目标卷积层的偏置参数形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            # 更新卷积层的偏置参数
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            # 记录日志，指示偏置参数已从 `full_name` 初始化
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 如果名称中包含 "weight"，则更新对应卷积层的权重参数
        elif "weight" in name:
            # 断言新值 `value` 的形状与目标卷积层的权重参数形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            # 更新卷积层的权重参数
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            # 记录日志，指示权重参数已从 `full_name` 初始化
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 如果类型为 2，并且不使用组归一化，或者类型为 2 且是第一层且使用组归一化，则执行以下操作
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 如果名称中包含 "bias"，则更新对应卷积层的组归一化偏置参数
        if "bias" in name:
            # 断言新值 `value` 的形状与目标卷积层的组归一化偏置参数形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape, (
                f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            # 更新卷积层的组归一化偏置参数
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            # 记录日志，指示组归一化偏置参数已从 `full_name` 初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 如果名称中包含 "weight"，则更新对应卷积层的组归一化权重参数
        elif "weight" in name:
            # 断言新值 `value` 的形状与目标卷积层的组归一化权重参数形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            )
            # 更新卷积层的组归一化权重参数
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            # 记录日志，指示组归一化权重参数已从 `full_name` 初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        # 如果不满足上述条件，则将 `full_name` 加入未使用的权重列表
        unused_weights.append(full_name)


# 将模型的配置转换为 SEWConfig 对象的函数，根据是否微调 `is_finetuned` 执行不同的操作
def convert_config(model, is_finetuned):
    # 创建 SEWConfig 对象 `config`
    config = SEWConfig()
    # 根据是否微调选择不同的配置信息
    if is_finetuned:
        # 如果是微调，则从 `model.w2v_encoder.w2v_model.cfg` 中获取配置信息
        fs_config = model.w2v_encoder.w2v_model.cfg
    else:
        # 否则从 `model.cfg` 中获取配置信息
        fs_config = model.cfg

    # 将卷积层的偏置标志 `conv_bias` 设置为 `fs_config` 中的值
    config.conv_bias = fs_config.conv_bias
    # 使用 `eval` 函数解析 `fs_config` 中的卷积层信息，并分别提取卷积层的维度、卷积核、步长等信息
    conv_layers = eval(fs_config.conv_feature_layers)
    config.conv_dim = [x[0] for x in conv_layers]
    config.conv_kernel = [x[1] for x in conv_layers]
    config.conv_stride = [x[2] for x in conv_layers]
    # 设置特征提取的激活函数为 "gelu"
    config.feat_extract_activation = "gelu"
    # 根据 `fs_config` 中的提取器模式设置特征提取的归一化方式为 "layer" 或 "group"
    config.feat_extract_norm = "layer" if fs_config.extractor_mode == "layer_norm" else "group"
    # 最终的 dropout 设为 0.0
    config.final_dropout = 0.0
    # 设置隐藏层激活函数为 `fs_config` 中的激活函数名称
    config.hidden_act = fs_config.activation_fn.name
    # 设置隐藏层大小为预训练模型的编码器嵌入维度
    config.hidden_size = fs_config.encoder_embed_dim
    # 设置初始化范围为0.02
    config.initializer_range = 0.02
    # 设置中间层大小为预训练模型的编码器前馈神经网络嵌入维度
    config.intermediate_size = fs_config.encoder_ffn_embed_dim
    # 设置层归一化的 epsilon 值
    config.layer_norm_eps = 1e-5
    # 设置层丢弃概率为预训练模型的编码器层丢弃率
    config.layerdrop = fs_config.encoder_layerdrop
    # 设置注意力头数为预训练模型的编码器注意力头数
    config.num_attention_heads = fs_config.encoder_attention_heads
    # 设置卷积位置嵌入组数为预训练模型的卷积位置编组数
    config.num_conv_pos_embedding_groups = fs_config.conv_pos_groups
    # 设置卷积位置嵌入数为预训练模型的卷积位置编码数
    config.num_conv_pos_embeddings = fs_config.conv_pos
    # 设置特征提取层数为卷积层列表的长度
    config.num_feat_extract_layers = len(conv_layers)
    # 设置隐藏层数为预训练模型的编码器层数
    config.num_hidden_layers = fs_config.encoder_layers
    # 设置挤压因子为预训练模型的挤压因子
    config.squeeze_factor = fs_config.squeeze_factor

    # 处理被 Wav2VecCtc 模型覆盖的任何参数
    if is_finetuned:
        # 使用模型配置中的配置覆盖当前配置
        fs_config = model.cfg
        # 设置最终丢弃概率为模型配置中的最终丢弃概率
        config.final_dropout = fs_config.final_dropout
        # 设置层丢弃概率为模型配置中的层丢弃概率
        config.layerdrop = fs_config.layerdrop
    # 设置激活丢弃概率为预训练模型的激活丢弃概率
    config.activation_dropout = fs_config.activation_dropout
    # 根据是否应用频谱增强设置配置中的应用特定增强
    config.apply_spec_augment = fs_config.mask_prob > 0 or fs_config.mask_channel_prob > 0
    # 设置注意力丢弃概率为预训练模型的注意力丢弃概率
    config.attention_dropout = fs_config.attention_dropout
    # 设置特征投影丢弃概率为预训练模型的输入丢弃概率
    config.feat_proj_dropout = fs_config.dropout_input
    # 设置隐藏层丢弃概率为预训练模型的丢弃概率
    config.hidden_dropout = fs_config.dropout
    # 设置掩码特征长度为预训练模型的掩码通道长度
    config.mask_feature_length = fs_config.mask_channel_length
    # 设置掩码特征概率为预训练模型的掩码通道概率
    config.mask_feature_prob = fs_config.mask_channel_prob
    # 设置掩码时间长度为预训练模型的掩码长度
    config.mask_time_length = fs_config.mask_length
    # 设置掩码时间概率为预训练模型的掩码概率
    config.mask_time_prob = fs_config.mask_prob

    # 设置特征提取器类型为 "Wav2Vec2FeatureExtractor"
    config.feature_extractor_type = "Wav2Vec2FeatureExtractor"
    # 设置分词器类为 "Wav2Vec2CTCTokenizer"
    config.tokenizer_class = "Wav2Vec2CTCTokenizer"

    # 返回配置对象
    return config
@torch.no_grad()
def convert_sew_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """

    if is_finetuned:
        # 如果是微调模型，使用fairseq加载模型和任务
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1])}
        )
    else:
        # 如果不是微调模型，直接加载模型
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])

    if config_path is not None:
        # 如果提供了配置文件路径，使用SEWConfig从预训练模型加载配置
        config = SEWConfig.from_pretrained(config_path)
    else:
        # 否则，根据模型和微调状态转换配置
        config = convert_config(model[0], is_finetuned)
    # 将模型设置为评估模式
    model = model[0].eval()

    # 根据配置设置是否返回注意力掩码
    return_attention_mask = True if config.feat_extract_norm == "layer" else False
    # 创建Wav2Vec2特征提取器实例
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0,
        do_normalize=True,
        return_attention_mask=return_attention_mask,
    )

    if is_finetuned:
        if dict_path:
            # 如果提供了字典路径，加载字典
            target_dict = Dictionary.load(dict_path)

            # 重要的变化，调整起始和填充符号ID，因为CTC的符号是<pad>而不是<s>
            target_dict.indices[target_dict.bos_word] = target_dict.pad_index
            target_dict.indices[target_dict.pad_word] = target_dict.bos_index
            config.bos_token_id = target_dict.pad_index
            config.pad_token_id = target_dict.bos_index
            config.eos_token_id = target_dict.eos_index
            config.vocab_size = len(target_dict.symbols)
            vocab_path = os.path.join(pytorch_dump_folder_path, "vocab.json")
            if not os.path.isdir(pytorch_dump_folder_path):
                # 如果目标路径不是目录，记录错误并返回
                logger.error("--pytorch_dump_folder_path ({}) should be a directory".format(pytorch_dump_folder_path))
                return
            # 创建目标路径
            os.makedirs(pytorch_dump_folder_path, exist_ok=True)
            # 将字典索引写入JSON文件
            with open(vocab_path, "w", encoding="utf-8") as vocab_handle:
                json.dump(target_dict.indices, vocab_handle)
            # 创建Wav2Vec2CTC标记器实例
            tokenizer = Wav2Vec2CTCTokenizer(
                vocab_path,
                unk_token=target_dict.unk_word,
                pad_token=target_dict.pad_word,
                bos_token=target_dict.bos_word,
                eos_token=target_dict.eos_word,
                word_delimiter_token="|",
                do_lower_case=False,
            )
            # 创建Wav2Vec2处理器实例
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            # 将处理器保存到指定路径
            processor.save_pretrained(pytorch_dump_folder_path)

        # 根据配置创建SEWForCTC模型
        hf_model = SEWForCTC(config)
    else:
        # 根据配置创建SEWModel模型
        hf_model = SEWModel(config)
        # 将特征提取器保存到指定路径
        feature_extractor.save_pretrained(pytorch_dump_folder_path)

    # 递归加载模型权重到预训练模型
    recursively_load_weights(model, hf_model, is_finetuned)

    # 将预训练模型保存到指定路径
    hf_model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 添加命令行参数 `--pytorch_dump_folder_path`，指定输出 PyTorch 模型的路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加命令行参数 `--checkpoint_path`，指定 fairseq 模型的检查点路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    # 添加命令行参数 `--dict_path`，指定经过微调的模型的字典路径
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    # 添加命令行参数 `--config_path`，指定要转换的模型的 hf config.json 路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加命令行参数 `--is_finetuned`，指定要转换的模型是否是经过微调的模型
    parser.add_argument(
        "--is_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    # 解析命令行参数，并将其存储到 args 变量中
    args = parser.parse_args()
    # 调用 convert_sew_checkpoint 函数，传递命令行参数中指定的路径和配置
    convert_sew_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path, args.is_finetuned
    )
```