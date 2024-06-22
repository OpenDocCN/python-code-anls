# `.\models\hubert\convert_hubert_original_pytorch_checkpoint_to_pytorch.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求或书面同意，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制
"""将 Hubert 检查点转换为 HubertConfig、HubertForCTC、HubertModel、Wav2Vec2CTCTokenizer、Wav2Vec2FeatureExtractor、Wav2Vec2Processor"""

# 导入所需库
import argparse
import json
import os

import fairseq
import torch
from fairseq.data import Dictionary

from transformers import (
    HubertConfig,
    HubertForCTC,
    HubertModel,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    logging,
)

# 设置日志级别为信息
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger(__name__)

# 映射关系
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
    "w2v_encoder.proj": "lm_head",
    "mask_emb": "masked_spec_embed",
}

# 递归设置权重
def set_recursively(hf_pointer, key, value, full_name, weight_type):
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

# 递归加载权重
def recursively_load_weights(fairseq_model, hf_model, is_finetuned):
    unused_weights = []
    fairseq_dict = fairseq_model.state_dict()

    feature_extractor = hf_model.hubert.feature_extractor if is_finetuned else hf_model.feature_extractor
    # 遍历 fairseq_dict 字典中的键值对
    for name, value in fairseq_dict.items():
        # 初始化是否使用标志为 False
        is_used = False
        # 如果键名中包含 "conv_layers"
        if "conv_layers" in name:
            # 调用 load_conv_layer 函数加载卷积层
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            # 设置使用标志为 True
            is_used = True
        else:
            # 遍历 MAPPING 字典中的键值对
            for key, mapped_key in MAPPING.items():
                # 根据是否 fine-tuned 和 mapped_key 是否为 "lm_head" 修改 mapped_key
                mapped_key = "hubert." + mapped_key if (is_finetuned and mapped_key != "lm_head") else mapped_key

                # 如果 key 在 name 中，或者 key 的 w2v_model 部分与 name 的第一个部分匹配且未 fine-tuned
                if key in name or (key.split("w2v_model.")[-1] == name.split(".")[0] and not is_finetuned):
                    # 设置使用标志为 True
                    is_used = True
                    # 如果 mapped_key 中包含 "*"，替换为层索引
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    # 根据 name 中的关键词确定权重类型
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
                    # 递归设置 hf_model 中的值
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                # 继续下一个循环
                continue
        # 如果未使用该权重，将其添加到未使用权重列表中
        if not is_used:
            unused_weights.append(name)

    # 输出未使用权重的警告信息
    logger.warning(f"Unused weights: {unused_weights}")
# 加载卷积层的权重，根据全名、值、特征提取器、未使用的权重列表和是否使用组归一化来进行操作
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 根据全名获取层名称
    name = full_name.split("conv_layers.")[-1]
    # 根据层名称获取层编号和类型编号
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    # 如果类型编号为0
    if type_id == 0:
        # 如果名称中包含"bias"
        if "bias" in name:
            # 断言值的形状与特征提取器中卷积层的偏置数据形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            # 将值赋给特征提取器中卷积层的偏置数据
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 如果名称中包含"weight"
        elif "weight" in name:
            # 断言值的形状与特征提取器中卷积层的权重数据形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            # 将值赋给特征提取器中卷积层的权重数据
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 如果类型编号为2且不使用组归一化，或者类型编号为2且层编号为0且使用组归一化
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 如果名称中包含"bias"
        if "bias" in name:
            # 断言值的形状与特征提取器中卷积层的层归一化偏置数据形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape, (
                f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            # 将值赋给特征提取器中卷积层的层归一化偏置数据
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 如果名称中包含"weight"
        elif "weight" in name:
            # 断言值的形状与特征提取器中卷积层的层归一化权重数据形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            )
            # 将值赋给特征提取器中卷积层的层归一化权重数据
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    # 否则
    else:
        # 将全名添加到未使用的权重列表中
        unused_weights.append(full_name)

# 无需梯度计算的装饰器
@torch.no_grad()
# 转换 Hubert 检查点
def convert_hubert_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 如果配置路径不为 None
    if config_path is not None:
        # 从预训练配置路径创建 Hubert 配置
        config = HubertConfig.from_pretrained(config_path)
    else:
        # 创建默认 Hubert 配置
        config = HubertConfig()
    # 如果模型已经微调过
    if is_finetuned:
        # 如果提供了字典路径，则加载字典
        if dict_path:
            target_dict = Dictionary.load(dict_path)

            # 重要更改 bos 和 pad token id，因为 CTC 符号是 <pad> 而不是 fairseq 中的 <s>
            config.bos_token_id = target_dict.pad_index
            config.pad_token_id = target_dict.bos_index
            config.eos_token_id = target_dict.eos_index
            config.vocab_size = len(target_dict.symbols)
            vocab_path = os.path.join(pytorch_dump_folder_path, "vocab.json")
            # 如果 pytorch_dump_folder_path 不是目录，则报错
            if not os.path.isdir(pytorch_dump_folder_path):
                logger.error("--pytorch_dump_folder_path ({}) should be a directory".format(pytorch_dump_folder_path))
                return
            # 创建目录
            os.makedirs(pytorch_dump_folder_path, exist_ok=True)
            # 将字典索引保存到文件中
            with open(vocab_path, "w", encoding="utf-8") as vocab_handle:
                json.dump(target_dict.indices, vocab_handle)
            # 创建 Wav2Vec2CTCTokenizer 对象
            tokenizer = Wav2Vec2CTCTokenizer(
                vocab_path,
                unk_token=target_dict.unk_word,
                pad_token=target_dict.pad_word,
                bos_token=target_dict.bos_word,
                eos_token=target_dict.eos_word,
                word_delimiter_token="|",
                do_lower_case=False,
            )
            # 根据配置创建特征提取器
            return_attention_mask = True if config.feat_extract_norm == "layer" else False
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0,
                do_normalize=True,
                return_attention_mask=return_attention_mask,
            )
            # 创建 Wav2Vec2Processor 对象
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            # 保存微调后的处理器
            processor.save_pretrained(pytorch_dump_folder_path)

        # 创建 HubertForCTC 模型
        hf_wav2vec = HubertForCTC(config)
    else:
        # 创建 HubertModel 模型
        hf_wav2vec = HubertModel(config)

    # 如果模型已经微调过，则加载微调后的模型
    if is_finetuned:
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1])}
        )
    else:
        # 加载模型
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])

    # 将模型设置为评估模式
    model = model[0].eval()

    # 递归加载权重
    recursively_load_weights(model, hf_wav2vec, is_finetuned)

    # 保存微调后的 HubertForCTC 模型
    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加参数--pytorch_dump_folder_path，指定输出 PyTorch 模型的路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加参数--checkpoint_path，指定 fairseq checkpoint 的路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    # 添加参数--dict_path，指定微调模型的字典路径
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    # 添加参数--config_path，指定要转换的模型的 hf config.json 路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加参数--not_finetuned，指定要转换的模型是否是微调模型
    parser.add_argument(
        "--not_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数convert_hubert_checkpoint，将 fairseq checkpoint 转换为 PyTorch 模型
    convert_hubert_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path, not args.not_finetuned
    )
```