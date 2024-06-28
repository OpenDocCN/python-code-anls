# `.\models\wav2vec2\convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py`

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
"""Convert Wav2Vec2 checkpoint."""

# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import json  # 用于处理JSON格式的数据
import os  # 用于与操作系统进行交互

import fairseq  # 导入fairseq库
import torch  # 导入PyTorch库
from fairseq.data import Dictionary  # 导入fairseq库中的Dictionary类

# 导入transformers库中的各个组件和模型类
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2ForPreTraining,
    Wav2Vec2Processor,
    logging,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForSequenceClassification

# 设置日志记录的详细程度为INFO级别
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# 定义一个映射字典，用于将旧模型的参数映射到新模型的参数
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
    "adapter_layer": "encoder.layers.*.adapter_layer",
    "w2v_model.layer_norm": "feature_projection.layer_norm",
    "quantizer.weight_proj": "quantizer.weight_proj",
    "quantizer.vars": "quantizer.codevectors",
    "project_q": "project_q",
    "final_proj": "project_hid",
    "w2v_encoder.proj": "lm_head",
    "mask_emb": "masked_spec_embed",
    "pooling_layer.linear": "projector",
    "pooling_layer.projection": "classifier",
}

# 定义顶层键列表，列出需要映射的最高层级参数
TOP_LEVEL_KEYS = [
    "lm_head",
    "quantizer.weight_proj",
    "quantizer.codevectors",
    "project_q",
    "project_hid",
    "projector",
    "classifier",
]

# 定义一个函数，从文本文件中读取内容并存储为字典形式
def read_txt_into_dict(filename):
    result = {}
    with open(filename, "r") as file:
        for line_number, line in enumerate(file):
            line = line.strip()
            if line:
                words = line.split()
                key = line_number
                value = words[0]
                result[key] = value
    return result

# 定义一个递归设置函数，用于根据指定的键路径设置值到相应的属性上
def set_recursively(key, value, full_name, weight_type, hf_pointer):
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    hf_param_name = None  # 暂时未使用的参数名变量
    # 遍历参数映射字典中的所有键
    for param_key in PARAM_MAPPING.keys():
        # 检查完整名称是否以当前参数键结尾
        if full_name.endswith(param_key):
            # 根据参数映射字典获取对应的参数名
            hf_param_name = PARAM_MAPPING[full_name.split(".")[-1]]
            # 设置权重类型为参数类型
            weight_type = "param"

    # 如果权重类型不为空且不为参数类型
    if weight_type is not None and weight_type != "param":
        # 获取指定权重类型属性的形状
        hf_shape = getattr(hf_pointer, weight_type).shape
    # 如果权重类型不为空且为参数类型
    elif weight_type is not None and weight_type == "param":
        # 逐级获取参数名对应的形状指针
        shape_pointer = hf_pointer
        for attribute in hf_param_name.split("."):
            shape_pointer = getattr(shape_pointer, attribute)
        # 获取最终的形状
        hf_shape = shape_pointer.shape

        # 缩减维度，仅保留第一个元素
        value = value[0]
    # 如果以上条件都不满足，获取当前指针的形状
    else:
        hf_shape = hf_pointer.shape

    # 检查获取的形状与值的形状是否相等，如果不相等则抛出异常
    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    # 根据权重类型将值赋给相应的属性
    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    elif weight_type == "param":
        # 逐级获取参数名对应的属性指针，并赋值
        for attribute in hf_param_name.split("."):
            hf_pointer = getattr(hf_pointer, attribute)
        hf_pointer.data = value
    else:
        # 若权重类型为空，则直接将值赋给指针
        hf_pointer.data = value

    # 记录日志，标明哪个权重或参数从哪里初始化
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")
# 定义一个函数，用于根据一组映射规则重命名字典中的键，并更新权重类型
def rename_dict(key, value, full_name, weight_type, hf_dict):
    # 初始化变量hf_param_name
    hf_param_name = None
    # 遍历PARAM_MAPPING字典中的所有键
    for param_key in PARAM_MAPPING.keys():
        # 如果full_name以param_key结尾，确定hf_param_name的值为PARAM_MAPPING中对应的值
        if full_name.endswith(param_key):
            hf_param_name = PARAM_MAPPING[full_name.split(".")[-1]]
            weight_type = "param"

    # 如果weight_type不为None且不等于"param"
    if weight_type is not None and weight_type != "param":
        # 构建完整的键名full_key，格式为key.weight_type
        full_key = ".".join([key, weight_type])
    # 如果weight_type不为None且等于"param"
    elif weight_type is not None and weight_type == "param":
        # 构建完整的键名full_key，格式为key.hf_param_name
        full_key = ".".join([key, hf_param_name])
    else:
        # 否则直接使用key作为完整的键名full_key
        full_key = key

    # 将键值对(full_key, value)添加到hf_dict字典中，如果full_key中包含"lm_head"则只取value的第一个元素
    hf_dict[full_key] = value if "lm_head" in full_key else value[0]


# PARAM_MAPPING字典，用于存储特定键的重命名规则
PARAM_MAPPING = {
    "W_a": "linear_1.weight",
    "W_b": "linear_2.weight",
    "b_a": "linear_1.bias",
    "b_b": "linear_2.bias",
    "ln_W": "norm.weight",
    "ln_b": "norm.bias",
}


# 加载wav2vec2模型的特定层的权重数据
def load_wav2vec2_layer(name, value, hf_model=None, hf_dict=None):
    # 标志变量，指示是否使用了这个权重数据
    is_used = False
    # 遍历MAPPING字典中的所有键值对
    for key, mapped_key in MAPPING.items():
        # 将mapped_key设置为"wav2vec2." + mapped_key，如果mapped_key不在TOP_LEVEL_KEYS中
        mapped_key = "wav2vec2." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key
        # 如果name中包含key或者name按"."分割的第一个部分等于key
        if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
            # 表示这个权重数据被使用了
            is_used = True
            # 如果mapped_key中包含"*"，则替换为name按key分割后的倒数第二部分
            if "*" in mapped_key:
                layer_index = name.split(key)[0].split(".")[-2]
                mapped_key = mapped_key.replace("*", layer_index)
            # 根据name的内容确定权重类型
            if "weight_g" in name:
                weight_type = "weight_g"
            elif "weight_v" in name:
                weight_type = "weight_v"
            elif "bias" in name:
                weight_type = "bias"
            elif "weight" in name:
                # TODO: 不匹配quantizer.weight_proj
                weight_type = "weight"
            else:
                weight_type = None
            # 如果hf_dict不为None，则调用rename_dict函数重命名mapped_key，并将权重数据value存储到hf_dict中
            if hf_dict is not None:
                rename_dict(mapped_key, value, name, weight_type, hf_dict)
            else:
                # 否则调用set_recursively函数递归地设置mapped_key的权重数据value到hf_model中
                set_recursively(mapped_key, value, name, weight_type, hf_model)
            # 返回is_used，表示权重数据被使用了
            return is_used
    # 如果没有使用这个权重数据，则返回False
    return is_used


# 递归加载fairseq模型的权重数据到hf_model中
def recursively_load_weights(fairseq_model, hf_model, is_headless):
    # 未使用的权重数据列表
    unused_weights = []
    # 获取fairseq模型的状态字典
    fairseq_dict = fairseq_model.state_dict()

    # 获取hf_model中的特征提取器
    feature_extractor = hf_model.wav2vec2.feature_extractor

    # 遍历fairseq_dict中的所有权重数据项
    for name, value in fairseq_dict.items():
        # 标志变量，指示这个权重数据是否被使用
        is_used = False
        # 如果name中包含"conv_layers"
        if "conv_layers" in name:
            # 调用load_conv_layer函数加载卷积层的权重数据
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            # 标记这个权重数据被使用了
            is_used = True
        else:
            # 否则调用load_wav2vec2_layer函数加载wav2vec2模型的权重数据
            is_used = load_wav2vec2_layer(name, value, hf_model)
        # 如果这个权重数据没有被使用，则将其添加到未使用的权重数据列表中
        if not is_used:
            unused_weights.append(name)

    # 记录未使用的权重数据到日志中
    logger.warning(f"Unused weights: {unused_weights}")


# 加载卷积层的权重数据到特征提取器中
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 将full_name按"conv_layers."分割，获取卷积层的名称name
    name = full_name.split("conv_layers.")[-1]
    # 将name按"."分割，获取层ID和类型ID
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])
    # 如果权重类型为0（偏置）：
    if type_id == 0:
        # 如果名称中包含"bias"：
        if "bias" in name:
            # 检查传入值的形状是否与对应卷积层的偏置数据形状相同，若不同则引发数值错误异常
            if value.shape != feature_extractor.conv_layers[layer_id].conv.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
                )
            # 将传入的值赋给对应卷积层的偏置数据
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            # 记录日志，表示特征提取器卷积层的偏置数据已从指定来源初始化
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 如果名称中包含"weight"：
        elif "weight" in name:
            # 检查传入值的形状是否与对应卷积层的权重数据形状相同，若不同则引发数值错误异常
            if value.shape != feature_extractor.conv_layers[layer_id].conv.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
                )
            # 将传入的值赋给对应卷积层的权重数据
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            # 记录日志，表示特征提取器卷积层的权重数据已从指定来源初始化
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    
    # 如果权重类型为2且不使用分组归一化，或者权重类型为2且为第一层且使用分组归一化：
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 如果名称中包含"bias"：
        if "bias" in name:
            # 检查传入值的形状是否与对应卷积层的分组归一化偏置数据形状相同，若不同则引发数值错误异常
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape} was found."
                )
            # 将传入的值赋给对应卷积层的分组归一化偏置数据
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            # 记录日志，表示特征提取器卷积层的分组归一化偏置数据已从指定来源初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 如果名称中包含"weight"：
        elif "weight" in name:
            # 检查传入值的形状是否与对应卷积层的分组归一化权重数据形状相同，若不同则引发数值错误异常
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape} was found."
                )
            # 将传入的值赋给对应卷积层的分组归一化权重数据
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            # 记录日志，表示特征提取器卷积层的分组归一化权重数据已从指定来源初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    
    # 如果以上条件都不满足：
    else:
        # 将未使用的权重名称添加到未使用权重列表中
        unused_weights.append(full_name)
@torch.no_grad()
def convert_wav2vec2_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True, is_seq_class=False
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 如果提供了配置文件路径，则从预训练配置加载配置信息
    if config_path is not None:
        config = Wav2Vec2Config.from_pretrained(config_path)
    else:
        # 否则，使用默认配置
        config = Wav2Vec2Config()

    # 如果是序列分类任务
    if is_seq_class:
        # 从文本文件加载字典
        id2label = read_txt_into_dict(dict_path)
        # 将 id 到标签的映射加入配置
        config.id2label = id2label
        # 创建序列分类的 Wav2Vec2 模型
        hf_wav2vec = Wav2Vec2ForSequenceClassification(config)
        # 创建特征提取器对象
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0,
            do_normalize=True,
            return_attention_mask=True,
        )
        # 保存特征提取器配置到指定路径
        feature_extractor.save_pretrained(pytorch_dump_folder_path)

    # 如果是微调模型
    elif is_finetuned:
        if dict_path:
            # 加载目标字典
            target_dict = Dictionary.load(dict_path)

            # 调整配置中的特殊 token id，因为 CTC 符号是 <pad> 而不是 fairseq 中的 <s>
            config.bos_token_id = target_dict.pad_index
            config.pad_token_id = target_dict.bos_index
            config.eos_token_id = target_dict.eos_index
            config.vocab_size = len(target_dict.symbols)
            # 创建词汇表 JSON 文件的路径
            vocab_path = os.path.join(pytorch_dump_folder_path, "vocab.json")
            # 检查目标路径是否是目录
            if not os.path.isdir(pytorch_dump_folder_path):
                logger.error("--pytorch_dump_folder_path ({}) should be a directory".format(pytorch_dump_folder_path))
                return
            # 创建目录（如果不存在）
            os.makedirs(pytorch_dump_folder_path, exist_ok=True)
            # 创建词汇表字典
            vocab_dict = target_dict.indices

            # fairseq 中的 <pad> 和 <s> 需要交换
            vocab_dict["<pad>"] = 0
            vocab_dict["<s>"] = 1
            # 将词汇表字典写入 JSON 文件
            with open(vocab_path, "w", encoding="utf-8") as vocab_handle:
                json.dump(vocab_dict, vocab_handle)
            # 创建 Wav2Vec2CTC tokenizer 对象
            tokenizer = Wav2Vec2CTCTokenizer(
                vocab_path,
                unk_token=target_dict.unk_word,
                pad_token=target_dict.pad_word,
                bos_token=target_dict.bos_word,
                eos_token=target_dict.eos_word,
                word_delimiter_token="|",
                do_lower_case=False,
            )
            # 根据配置选择是否返回注意力掩码
            return_attention_mask = True if config.feat_extract_norm == "layer" else False
            # 创建特征提取器对象
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0,
                do_normalize=True,
                return_attention_mask=return_attention_mask,
            )
            # 创建处理器对象，包括特征提取器和 tokenizer
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            # 保存处理器配置到指定路径
            processor.save_pretrained(pytorch_dump_folder_path)

        # 创建 Wav2Vec2ForCTC 模型对象
        hf_wav2vec = Wav2Vec2ForCTC(config)
    else:
        # 创建预训练模型对象
        hf_wav2vec = Wav2Vec2ForPreTraining(config)
    # 如果模型已经进行了微调或者是用于序列分类，则执行以下操作
    if is_finetuned or is_seq_class:
        # 载入模型集合和任务信息，同时设置数据路径参数
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1])}
        )
    else:
        # 否则，设置音频预训练任务参数
        task_arg = argparse.Namespace(task="audio_pretraining")
        task = fairseq.tasks.setup_task(task_arg)

        # 载入模型集合和任务信息，同时设置任务参数
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path], task=task)

    # 将模型切换到评估模式
    model = model[0].eval()

    # 递归加载权重到模型，使用 hf_wav2vec 的权重，若非微调则反向加载
    recursively_load_weights(model, hf_wav2vec, not is_finetuned)

    # 将 PyTorch 模型保存到指定的转储文件夹路径
    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)
# 如果脚本作为主程序运行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加参数：输出 PyTorch 模型的路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加参数：fairseq 模型的检查点路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    # 添加参数：微调模型的字典路径
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    # 添加参数：待转换模型的 hf config.json 路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加参数：指示待转换模型是否为微调模型的标志
    parser.add_argument(
        "--not_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    # 添加参数：指示待转换模型是否为序列分类模型的标志
    parser.add_argument(
        "--is_seq_class",
        action="store_true",
        help="Whether the model to convert is a fine-tuned sequence classification model or not",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 根据参数判断待转换模型是否为微调模型
    is_finetuned = not args.not_finetuned and not args.is_seq_class
    # 调用函数，将 wav2vec2 模型检查点转换为 PyTorch 模型
    convert_wav2vec2_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path,
        args.dict_path,
        is_finetuned,
        args.is_seq_class,
    )
```