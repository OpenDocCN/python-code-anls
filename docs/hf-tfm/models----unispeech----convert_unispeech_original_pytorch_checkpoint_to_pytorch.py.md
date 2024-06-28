# `.\models\unispeech\convert_unispeech_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置编码格式为 UTF-8
# 版权声明和许可协议信息，指明代码版权及使用条款
# 导入 argparse 用于命令行解析，json 和 os 用于文件操作
import argparse
import json
import os

# 导入 fairseq 和 torch 库
import fairseq
import torch
# 从 fairseq 库中导入 Dictionary 类

from fairseq.data import Dictionary

# 从 transformers 库中导入以下模块和函数
from transformers import (
    UniSpeechConfig,                  # 导入 UniSpeechConfig 类
    UniSpeechForCTC,                  # 导入 UniSpeechForCTC 类
    UniSpeechForPreTraining,          # 导入 UniSpeechForPreTraining 类
    Wav2Vec2FeatureExtractor,         # 导入 Wav2Vec2FeatureExtractor 类
    Wav2Vec2PhonemeCTCTokenizer,      # 导入 Wav2Vec2PhonemeCTCTokenizer 类
    Wav2Vec2Processor,                # 导入 Wav2Vec2Processor 类
    logging,                          # 导入 logging 模块
)

# 设置 logging 的详细级别为 info
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义映射关系字典，用于转换 UniSpeech 模型的参数名称
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
    "w2v_model.layer_norm": "feature_projection.layer_norm",
    "quantizer.weight_proj": "quantizer.weight_proj",
    "quantizer.vars": "quantizer.codevectors",
    "project_q": "project_q",
    "final_proj": "project_hid",
    "w2v_encoder.proj": "ctc_proj",
    "mask_emb": "masked_spec_embed",
}

# 定义顶层键列表，列出需要处理的顶层参数名称
TOP_LEVEL_KEYS = [
    "ctc_proj",
    "quantizer.weight_proj",
    "quantizer.codevectors",
    "project_q",
    "project_hid",
]

# 定义函数 set_recursively，用于递归设置参数值
def set_recursively(hf_pointer, key, value, full_name, weight_type, is_finetuned):
    # 根据键名逐级获取 hf_pointer 对象的属性
    for attribute in key.split("."):
        # 如果是微调模型并且当前属性属于需要跳过的层，则直接返回
        if is_finetuned:
            if attribute in ["quantizer", "project_q", "project_hid"]:
                return

            # 对于微调的音素模型，将 `ctc_proj` 重命名为 `lm_head`
            if attribute == "ctc_proj":
                attribute = "lm_head"

        # 获取 hf_pointer 对象的下一级属性
        hf_pointer = getattr(hf_pointer, attribute)

    # 如果 weight_type 不为空，则获取 hf_pointer 对应属性的形状
    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    # 断言判断当前参数的形状与预期值 value.shape 是否相符
    assert hf_shape == value.shape, (
        f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
        f" {value.shape} for {full_name}"
    )
    # 如果 weight_type 是 "weight"，则将 value 赋给 hf_pointer 的权重数据
    if weight_type == "weight":
        hf_pointer.weight.data = value
    # 如果 weight_type 是 "weight_g"，则将 value 赋给 hf_pointer 的权重梯度数据
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    # 如果 weight_type 是 "weight_v"，则将 value 赋给 hf_pointer 的权重版本数据
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    # 如果 weight_type 是 "bias"，则将 value 赋给 hf_pointer 的偏置数据
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    else:
        # 如果 weight_type 为空或未知，则直接将 value 赋给 hf_pointer
        hf_pointer.data = value

    # 记录初始化信息，包括 key、weight_type（如果存在）、和 full_name
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")
# 递归加载权重函数，将 Fairseq 模型的权重加载到 Hugging Face 模型中
def recursively_load_weights(fairseq_model, hf_model, is_finetuned):
    # 存储未使用的权重名称列表
    unused_weights = []
    # 获取 Fairseq 模型的状态字典
    fairseq_dict = fairseq_model.state_dict()

    # 获取 Hugging Face 模型的特征提取器
    feature_extractor = hf_model.unispeech.feature_extractor

    # 遍历 Fairseq 模型的状态字典
    for name, value in fairseq_dict.items():
        # 标记当前权重是否被使用的布尔值
        is_used = False
        
        # 如果权重名称中包含 "conv_layers"，则加载卷积层权重
        if "conv_layers" in name:
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        else:
            # 否则，遍历 MAPPING 字典，查找是否有匹配的键值对应的权重名称
            for key, mapped_key in MAPPING.items():
                # 如果权重名称中包含键值或者与键值相关的名称，则标记为已使用
                mapped_key = "unispeech." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    is_used = True
                    # 如果 mapped_key 中包含通配符 "*"，则替换为层索引
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    # 确定权重类型
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
                    # 递归设置 Hugging Face 模型的权重
                    set_recursively(hf_model, mapped_key, value, name, weight_type, is_finetuned)
                continue
        
        # 如果该权重未被使用，则将其名称添加到未使用的权重列表中
        if not is_used:
            unused_weights.append(name)

    # 输出未使用的权重名称列表的警告信息
    logger.warning(f"Unused weights: {unused_weights}")


# 加载卷积层权重的函数
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 提取卷积层名称
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    # 根据卷积层类型加载权重
    if type_id == 0:
        # 如果权重名称中包含 "bias"，则加载偏置项权重
        if "bias" in name:
            # 断言当前权重的形状与特征提取器中对应卷积层的偏置项数据形状一致
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            # 将权重值赋予特征提取器中对应卷积层的偏置项数据，并输出信息
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 如果权重名称中包含 "weight"，则加载权重矩阵
        elif "weight" in name:
            # 断言当前权重的形状与特征提取器中对应卷积层的权重矩阵数据形状一致
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            # 将权重值赋予特征提取器中对应卷积层的权重矩阵数据，并输出信息
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 如果 type_id 等于 2 并且不使用组规范，或者 type_id 等于 2 且 layer_id 等于 0 并且使用组规范
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 如果变量名中包含 "bias"
        if "bias" in name:
            # 断言当前值的形状与特征提取器中指定卷积层的层归一化偏置数据形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape, (
                f"{full_name} has size {value.shape}, but {feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            # 将当前值赋给特征提取器中指定卷积层的层归一化偏置数据
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            # 记录日志，指示从哪里初始化了特征提取器中指定卷积层的层归一化权重
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 如果变量名中包含 "weight"
        elif "weight" in name:
            # 断言当前值的形状与特征提取器中指定卷积层的层归一化权重数据形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape} was found."
            )
            # 将当前值赋给特征提取器中指定卷积层的层归一化权重数据
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            # 记录日志，指示从哪里初始化了特征提取器中指定卷积层的层归一化权重
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        # 将未使用的权重名称添加到未使用的权重列表中
        unused_weights.append(full_name)
# 使用 torch.no_grad() 装饰器，确保在此函数执行期间不会进行梯度计算
@torch.no_grad()
# 定义函数 convert_unispeech_checkpoint，用于将模型权重从 UniSpeech 转换到 transformers 设计
def convert_unispeech_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 如果提供了 config_path，则从预训练模型加载 UniSpeechConfig
    if config_path is not None:
        config = UniSpeechConfig.from_pretrained(config_path)
    else:
        # 否则创建一个新的 UniSpeechConfig 实例
        config = UniSpeechConfig()

    # 如果是 finetuned 模型
    if is_finetuned:
        # 如果提供了 dict_path，则加载 Dictionary 对象
        if dict_path:
            target_dict = Dictionary.load_from_json(dict_path)

            # 重要的更改：由于 CTC 符号为 <pad> 而非 <s>（与 fairseq 不同），修改 bos & pad token id
            config.bos_token_id = target_dict.pad_index
            config.pad_token_id = target_dict.bos_index
            config.eos_token_id = target_dict.eos_index
            config.vocab_size = len(target_dict.symbols)
            vocab_path = os.path.join(pytorch_dump_folder_path, "vocab.json")

            # 检查 pytorch_dump_folder_path 是否是一个目录，如果不是则记录错误并返回
            if not os.path.isdir(pytorch_dump_folder_path):
                logger.error("--pytorch_dump_folder_path ({}) should be a directory".format(pytorch_dump_folder_path))
                return

            # 创建 pytorch_dump_folder_path 目录（如果不存在）
            os.makedirs(pytorch_dump_folder_path, exist_ok=True)
            vocab_dict = target_dict.indices

            # fairseq 中 <pad> 和 <s> 被交换了，重新设置 vocab_dict
            vocab_dict["<pad>"] = 42
            vocab_dict["<s>"] = 43

            # 将 vocab_dict 写入 vocab_path 文件中
            with open(vocab_path, "w", encoding="utf-8") as vocab_handle:
                json.dump(vocab_dict, vocab_handle)

            # 使用 Wav2Vec2PhonemeCTCTokenizer 初始化 tokenizer
            tokenizer = Wav2Vec2PhonemeCTCTokenizer(
                vocab_path,
                unk_token=target_dict.unk_word,
                pad_token=target_dict.pad_word,
                bos_token=target_dict.bos_word,
                eos_token=target_dict.eos_word,
                word_delimiter_token="|",
                do_lower_case=False,
            )

            # 根据 config 中 feat_extract_norm 的设置确定 return_attention_mask 的值
            return_attention_mask = True if config.feat_extract_norm == "layer" else False

            # 使用 Wav2Vec2FeatureExtractor 初始化 feature_extractor
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0,
                do_normalize=True,
                return_attention_mask=return_attention_mask,
            )

            # 使用初始化的 feature_extractor 和 tokenizer 创建 processor
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

            # 将 processor 的预训练模型保存到 pytorch_dump_folder_path 中
            processor.save_pretrained(pytorch_dump_folder_path)

        # 初始化 hf_unispeech 为 UniSpeechForCTC 实例
        hf_unispeech = UniSpeechForCTC(config)
    else:
        # 初始化 hf_unispeech 为 UniSpeechForPreTraining 实例
        hf_unispeech = UniSpeechForPreTraining(config)

    # 如果是 finetuned 模型
    if is_finetuned:
        # 使用 fairseq.checkpoint_utils.load_model_ensemble_and_task 加载模型及其任务
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1]), "w2v_path": checkpoint_path}
        )
    else:
        # 使用 fairseq.checkpoint_utils.load_model_ensemble_and_task 加载模型及其任务
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])

    # 将模型设置为 evaluation 模式
    model = model[0].eval()

    # 递归加载模型权重到 hf_unispeech
    recursively_load_weights(model, hf_unispeech, is_finetuned)

    # 将 hf_unispeech 的预训练模型保存到 pytorch_dump_folder_path 中
    hf_unispeech.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数，指定输出 PyTorch 模型的路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加命令行参数，指定 fairseq 检查点的路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    # 添加命令行参数，指定微调模型的字典路径
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    # 添加命令行参数，指定要转换的模型的 hf config.json 路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加命令行参数，指定要转换的模型是否为微调模型
    parser.add_argument(
        "--not_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    # 解析命令行参数，并将其存储在 args 对象中
    args = parser.parse_args()
    # 调用函数 convert_unispeech_checkpoint，传递解析后的参数来执行模型转换操作
    convert_unispeech_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path, not args.not_finetuned
    )
```