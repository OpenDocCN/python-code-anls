# `.\transformers\models\unispeech\convert_unispeech_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置文件的编码格式为 UTF-8
# 版权声明
# 避免违反许可协议，需要按照 Apache License, Version 2.0 使用该文件
# 可在 http://www.apache.org/licenses/LICENSE-2.0 获得许可协议的副本
# 在适用法律或书面同意的情况下，本软件可根据“原样”分发，无论是明示还是暗示的，不带任何担保或条件
# 请参阅许可协议规定的特定语言，控制权限和限制事项
# 转换 UniSpeech 检查点

# 导入所需的库
import argparse  # 用于解析命令行参数
import json  # 用于处理 JSON 数据的模块
import os  # 用于提供与操作系统交互的功能

import fairseq  # 引入 fairseq 库
import torch  # 引入 torch 库
from fairseq.data import Dictionary  # 从 fairseq 库中引入 Dictionary 类

from transformers import (  # 从 transformers 库中导入以下类和函数
    UniSpeechConfig,  # UniSpeechConfig 类，用于存储 UniSpeech 模型的配置
    UniSpeechForCTC,  # UniSpeechForCTC 类，用于加载预训练的 UniSpeech 语音识别模型
    UniSpeechForPreTraining,  # UniSpeechForPreTraining 类，用于加载预训练的 UniSpeech 语音识别模型
    Wav2Vec2FeatureExtractor,  # Wav2Vec2FeatureExtractor 类，用于提取音频特征
    Wav2Vec2PhonemeCTCTokenizer,  # Wav2Vec2PhonemeCTCTokenizer 类，用于将音频数据转换为 token
    Wav2Vec2Processor,  # Wav2Vec2Processor 类，用于处理音频数据
    logging,  # logging 模块，用于记录日志
)

# 设置日志的详细程度为信息级别
logging.set_verbosity_info()
# 获取记录器对象
logger = logging.get_logger(__name__)

# 定义映射字典
MAPPING = {  # 用于存储键值对，用于映射模型文件中的参数名到 Hugging Face 模型中的参数名
    # ... (此处省略部分键值对)
}

# 定义顶层键列表
TOP_LEVEL_KEYS = [  # 用于存储需要修改的顶层键的列表
    "ctc_proj",  # CTC 投影层
    "quantizer.weight_proj",  # 量化器的权重投影
    "quantizer.codevectors",  # 量化器的码矢
    "project_q",  # 项目 Q
    "project_hid",  # 项目隐藏状态
]

# 递归设置变量值
def set_recursively(hf_pointer, key, value, full_name, weight_type, is_finetuned):
    # 遍历键值对
    for attribute in key.split("."):
        # 如果是微调模型
        if is_finetuned:
            if attribute in ["quantizer", "project_q", "project_hid"]:
                # 这些层只在预训练阶段有用，微调时应该移除
                return

            if attribute == "ctc_proj":
                # 对于微调的音素模型，我们应该将 `ctc_proj` 重命名为 `lm_head`
                attribute = "lm_head"

        # 获取属性的值
        hf_pointer = getattr(hf_pointer, attribute)

    # 如果存在权重类型
    if weight_type is not None:
        # 获取 Hugging Face 模型的形状
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        # 获取 Hugging Face 模型的形状
        hf_shape = hf_pointer.shape

    # 检查形状是否一致
    assert hf_shape == value.shape, (
        f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
        f" {value.shape} for {full_name}"
    )
    if weight_type == "weight":
        # 如果权重类型是"weight"，将value赋值给hf_pointer的weight属性
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        # 如果权重类型是"weight_g"，将value赋值给hf_pointer的weight_g属性
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        # 如果权重类型是"weight_v"，将value赋值给hf_pointer的weight_v属性
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        # 如果权重类型是"bias"，将value赋值给hf_pointer的bias属性
        hf_pointer.bias.data = value
    else:
        # 如果权重类型不是以上任何一种，则将value赋值给hf_pointer的data属性
        hf_pointer.data = value

    # 打印日志信息，说明哪个权重被初始化，并指明其来源
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")
# 递归地加载权重，将 fairseq 模型的权重载入到 HuggingFace 模型中
def recursively_load_weights(fairseq_model, hf_model, is_finetuned):
    # 存储未使用的权重名称
    unused_weights = []
    # 获取 fairseq 模型的状态字典
    fairseq_dict = fairseq_model.state_dict()
    # 获取 HuggingFace 模型的特征提取器
    feature_extractor = hf_model.unispeech.feature_extractor

    # 遍历 fairseq 模型的权重字典
    for name, value in fairseq_dict.items():
        # 标记是否已使用该权重
        is_used = False
        # 如果权重名称包含 "conv_layers"，则加载卷积层的权重
        if "conv_layers" in name:
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        # 否则，尝试将权重映射到 HuggingFace 模型的相应层
        else:
            for key, mapped_key in MAPPING.items():
                # 将 mapped_key 前缀为 "unispeech."，除非它在 TOP_LEVEL_KEYS 中
                mapped_key = "unispeech." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key
                # 如果 key 在权重名称中或者 key 的最后一部分等于权重名称的第一部分
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    is_used = True
                    # 如果 mapped_key 包含 "*"，则用层索引替换它
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    # 根据权重名称确定权重类型
                    if "weight_g" in name:
                        weight_type = "weight_g"
                    elif "weight_v" in name:
                        weight_type = "weight_v"
                    elif "bias" in name:
                        weight_type = "bias"
                    elif "weight" in name:
                        weight_type = "weight"
                    else:
                        weight_type = None
                    # 递归地设置权重
                    set_recursively(hf_model, mapped_key, value, name, weight_type, is_finetuned)
                continue
        # 如果权重未使用，将其添加到未使用的权重列表中
        if not is_used:
            unused_weights.append(name)

    # 记录未使用的权重
    logger.warning(f"Unused weights: {unused_weights}")


# 加载卷积层的权重
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 从完整名称中提取层 ID 和类型 ID
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    # 如果类型 ID 为 0，则加载卷积层的权重和偏差
    if type_id == 0:
        if "bias" in name:
            # 检查偏差的形状是否匹配
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            # 将偏差赋值到 HuggingFace 模型的卷积层
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            # 检查权重的形状是否匹配
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            # 将权重赋值到 HuggingFace 模型的卷积层
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 如果类型ID等于2且不使用组归一化，或者类型ID等于2且层ID等于0且使用组归一化，则执行以下操作
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 如果名字中包含"bias"
        if "bias" in name:
            # 断言值的形状与特征提取器卷积层的层归一化偏差数据的形状相等
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape, (
                f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            # 将特征提取器卷积层的层归一化偏差数据设置为给定值
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            # 记录日志，说明特征提取器层归一化权重已从给定名字中初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 如果名字中包含"weight"
        elif "weight" in name:
            # 断言值的形状与特征提取器卷积层的层归一化权重数据的形状相等
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            )
            # 将特征提取器卷积层的层归一化权重数据设置为给定值
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            # 记录日志，说明特征提取器层归一化权重已从给定名字中初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    # 如果不满足上述条件
    else:
        # 将未使用的权重名字添加到未使用权重列表中
        unused_weights.append(full_name)
pad>"] = 42
            vocab_dict["s>"] = 43
            # 将词典保存为 JSON 文件
            with open(vocab_path, "w", encoding="utf-8") as vocab_handle:
                json.dump(vocab_dict, vocab_handle)
            # 使用词典创建 Wav2Vec2PhonemeCTCTokenizer
            tokenizer = Wav2Vec2PhonemeCTCTokenizer(
                vocab_path,
                unk_token=target_dict.unk_word,
                pad_token=target_dict.pad_word,
                bos_token=target_dict.bos_word,
                eos_token=target_dict.eos_word,
                word_delimiter_token="|",
                do_lower_case=False,
            )
            # 如果 feat_extract_norm 为 "layer"，则返回_attention_mask 为 True
            return_attention_mask = True if config.feat_extract_norm == "layer" else False
            # 创建特征提取器
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0,
                do_normalize=True,
                return_attention_mask=return_attention_mask,
            )
            # 创建处理器
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            # 保存处理器
            processor.save_pretrained(pytorch_dump_folder_path)

        # 创建 UniSpeechForCTC 模型对象
        hf_unispeech = UniSpeechForCTC(config)
    else:
        # 创建 UniSpeechForPreTraining 模型对象
        hf_unispeech = UniSpeechForPreTraining(config)

    # 如果是微调模型
    if is_finetuned:
        # 加载模型及任务
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1]), "w2v_path": checkpoint_path}
        )
    else:
        # 加载模型
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])

    # 设置为评估模式
    model = model[0].eval()

    # 递归加载权重
    recursively_load_weights(model, hf_unispeech, is_finetuned)

    # 保存转换后的模型
    hf_unispeech.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加一个参数，用来指定输出 PyTorch 模型的路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加一个参数，用来指定 fairseq 检查点的路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    # 添加一个参数，用来指定微调模型的字典路径
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    # 添加一个参数，用来指定要转换的模型的 hf config.json 路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加一个参数，用来指定模型是否为微调模型
    parser.add_argument(
        "--not_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将 Unispeech 检查点转换为 PyTorch 模型
    convert_unispeech_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path, not args.not_finetuned
    )
```