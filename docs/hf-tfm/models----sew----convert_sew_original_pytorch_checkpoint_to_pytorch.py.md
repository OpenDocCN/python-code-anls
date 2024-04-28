# `.\transformers\models\sew\convert_sew_original_pytorch_checkpoint_to_pytorch.py`

```
# 导入所需的 Python 库和模块
import argparse
import json
import os

import fairseq
import torch
from fairseq.data import Dictionary

# 导入 SEW 的 fairseq 模块
from sew_asapp import tasks  # noqa: F401

from transformers import (
    SEWConfig,
    SEWForCTC,
    SEWModel,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    logging,
)

# 设置日志级别为信息
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# 定义一个字典,用于映射 fairseq 模型中的层名称到 Transformers 模型中的层名称
MAPPING = {
    "post_extract_proj": "feature_projection",
    "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",
    "self_attn.k_proj": "encoder.layers.*.attention.k_proj",
    "self_attn.v_proj": "encoder.layers.*.attention.v_proj",
    "self_attn.q_proj": "encoder.layers.*.attention.q_proj",
    "self_attn.out_proj": "encoder.layers.*.attention.out_proj",
    "self_attn_layer_norm": "encoder.layers.*.layer_norm",
    "fc1": "encoder.layers.*.feed_forward.intermediate_dense",
    "fc2": "encoder.layers.*.feed_forward.output_dense",
    "final_layer_norm": "encoder.layers.*.final_layer_norm",
    "encoder.upsample.0": "encoder.upsample.projection",
    "encoder.layer_norm": "encoder.layer_norm",
    "w2v_model.layer_norm": "layer_norm",
    "w2v_encoder.proj": "lm_head",
    "mask_emb": "masked_spec_embed",
}

# 定义一个函数,用于递归地设置 Transformers 模型的权重
def set_recursively(hf_pointer, key, value, full_name, weight_type):
    # 按照 key 中的属性递归地设置 hf_pointer
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    # 获取 hf_pointer 的形状
    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    # 检查 hf_pointer 的形状是否与 value 的形状一致
    assert hf_shape == value.shape, (
        f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
        f" {value.shape} for {full_name}"
    )

    # 根据 weight_type 的不同,将 value 赋值给 hf_pointer 的不同属性
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

    # 记录日志
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")

# 定义一个函数,用于递归地加载 fairseq 模型的权重到 Transformers 模型
def recursively_load_weights(fairseq_model, hf_model, is_finetuned):
    unused_weights = []
    fairseq_dict = fairseq_model.state_dict()
    # 如果模型是微调的，则使用 hf_model.sew.feature_extractor 作为特征提取器，否则使用 hf_model.feature_extractor
    feature_extractor = hf_model.sew.feature_extractor if is_finetuned else hf_model.feature_extractor
    
    # 遍历 fairseq 字典中的每个键值对
    for name, value in fairseq_dict.items():
        # 标记是否使用了当前的权重
        is_used = False
        # 如果键名中包含 "conv_layers" 字符串
        if "conv_layers" in name:
            # 载入卷积层权重
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            # 标记为已使用
            is_used = True
        else:
            # 遍历 MAPPING 字典中的每个键值对
            for key, mapped_key in MAPPING.items():
                # 如果模型是微调的并且 mapped_key 不是 "lm_head"，则加上前缀 "sew."，否则保持不变
                mapped_key = "sew." + mapped_key if (is_finetuned and mapped_key != "lm_head") else mapped_key
    
                # 如果键名中包含 key，或者键名的 "w2v_model." 后的部分与 name 的开头匹配
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    # 标记为已使用
                    is_used = True
                    # 如果 mapped_key 中含有 "*"，则替换为权重所在的层索引
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
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
                    # 递归设置模型的对应属性值为当前权重值
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        # 如果当前权重未被使用，则添加到未使用权重列表中
        if not is_used:
            unused_weights.append(name)
    
    # 记录未使用的权重列表到日志中
    logger.warning(f"Unused weights: {unused_weights}")
    ```  
# 加载卷积层参数
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 拆分全名，获取层和类型 ID
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    # 处理卷积层参数
    if type_id == 0:
        # 处理偏置参数
        if "bias" in name:
            # 检查参数形状是否一致，然后赋值
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 处理权重参数
        elif "weight" in name:
            # 检查参数形状是否一致，然后赋值
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 处理层归一化参数
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 处理偏置参数
        if "bias" in name:
            # 检查参数形状是否一致，然后赋值
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape, (
                f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 处理权重参数
        elif "weight" in name:
            # 检查参数形状是否一致，然后赋值
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        # 未使用的权重添加到列表中
        unused_weights.append(full_name)


# 转换模型配置信息
def convert_config(model, is_finetuned):
    # 创建 SEWConfig 配置对象
    config = SEWConfig()
    # 根据是否微调选择不同的配置
    if is_finetuned:
        fs_config = model.w2v_encoder.w2v_model.cfg
    else:
        fs_config = model.cfg

    # 设置卷积层参数
    config.conv_bias = fs_config.conv_bias
    conv_layers = eval(fs_config.conv_feature_layers)
    config.conv_dim = [x[0] for x in conv_layers]
    config.conv_kernel = [x[1] for x in conv_layers]
    config.conv_stride = [x[2] for x in conv_layers]
    config.feat_extract_activation = "gelu"
    # 根据提取模式选择归一化方式
    config.feat_extract_norm = "layer" if fs_config.extractor_mode == "layer_norm" else "group"
    config.final_dropout = 0.0
    config.hidden_act = fs_config.activation_fn.name
    # 将隐藏层的大小设置为fs_config.encoder_embed_dim
    config.hidden_size = fs_config.encoder_embed_dim
    # 设置初始化范围为0.02
    config.initializer_range = 0.02
    # 设置中间大小为fs_config.encoder_ffn_embed_dim
    config.intermediate_size = fs_config.encoder_ffn_embed_dim
    # 设置层标准化的epsilon为1e-5
    config.layer_norm_eps = 1e-5
    # 设置层丢弃率为fs_config.encoder_layerdrop
    config.layerdrop = fs_config.encoder_layerdrop
    # 设置注意力头的数量为fs_config.encoder_attention_heads
    config.num_attention_heads = fs_config.encoder_attention_heads
    # 设置卷积位置嵌入组的数量为fs_config.conv_pos_groups
    config.num_conv_pos_embedding_groups = fs_config.conv_pos_groups
    # 设置卷积位置嵌入数量为fs_config.conv_pos
    config.num_conv_pos_embeddings = fs_config.conv_pos
    # 设置特征提取层的数量为conv_layers的长度
    config.num_feat_extract_layers = len(conv_layers)
    # 设置隐藏层的数量为fs_config.encoder_layers
    config.num_hidden_layers = fs_config.encoder_layers
    # 设置挤压因子为fs_config.squeeze_factor
    config.squeeze_factor = fs_config.squeeze_factor

    # 处理Wav2VecCtc模型覆盖的任何参数
    if is_finetuned:
        # 获取模型的配置
        fs_config = model.cfg
        # 设置最终丢弃率为fs_config.final_dropout
        config.final_dropout = fs_config.final_dropout
        # 设置层丢弃率为fs_config.layerdrop
        config.layerdrop = fs_config.layerdrop
    # 设置激活丢弃率为fs_config.activation_dropout
    config.activation_dropout = fs_config.activation_dropout
    # 检查是否应用特征遮挡
    config.apply_spec_augment = fs_config.mask_prob > 0 or fs_config.mask_channel_prob > 0
    # 设置注意力丢弃率为fs_config.attention_dropout
    config.attention_dropout = fs_config.attention_dropout
    # 设置特征投影丢弃率为fs_config.dropout_input
    config.feat_proj_dropout = fs_config.dropout_input
    # 设置隐藏丢弃率为fs_config.dropout
    config.hidden_dropout = fs_config.dropout
    # 设置特征长度屏蔽为fs_config.mask_channel_length
    config.mask_feature_length = fs_config.mask_channel_length
    # 设置特征概率屏蔽为fs_config.mask_channel_prob
    config.mask_feature_prob = fs_config.mask_channel_prob
    # 设置时间长度屏蔽为fs_config.mask_length
    config.mask_time_length = fs_config.mask_length
    # 设置时间概率屏蔽为fs_config.mask_prob
    config.mask_time_prob = fs_config.mask_prob

    # 设置特征提取器的类型为"Wav2Vec2FeatureExtractor"
    config.feature_extractor_type = "Wav2Vec2FeatureExtractor"
    # 设置标记器类为"Wav2Vec2CTCTokenizer"
    config.tokenizer_class = "Wav2Vec2CTCTokenizer"

    # 返回配置
    return config
# 使用 torch.no_grad() 装饰器，确保在该函数中不会进行梯度计算
@torch.no_grad()
def convert_sew_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """

    # 如果是微调过的模型，则使用 fairseq.checkpoint_utils.load_model_ensemble_and_task 加载模型
    if is_finetuned:
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1])}
        )
    # 如果不是微调过的模型，则直接加载模型
    else:
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])

    # 如果提供了配置文件路径，则使用 SEWConfig.from_pretrained 加载配置，否则根据模型生成配置
    if config_path is not None:
        config = SEWConfig.from_pretrained(config_path)
    else:
        config = convert_config(model[0], is_finetuned)
    # 将模型设置为评估模式
    model = model[0].eval()

    # 根据配置确定是否返回注意力掩码
    return_attention_mask = True if config.feat_extract_norm == "layer" else False
    # 创建 Wav2Vec2FeatureExtractor 对象
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0,
        do_normalize=True,
        return_attention_mask=return_attention_mask,
    )

    # 如果是微调过的模型且提供了字典路径，则加载字典
    if is_finetuned:
        if dict_path:
            target_dict = Dictionary.load(dict_path)

            # 重要更改 bos & pad token id，因为 CTC 符号是 <pad> 而不是 <s> 如在 fairseq 中
            target_dict.indices[target_dict.bos_word] = target_dict.pad_index
            target_dict.indices[target_dict.pad_word] = target_dict.bos_index
            config.bos_token_id = target_dict.pad_index
            config.pad_token_id = target_dict.bos_index
            config.eos_token_id = target_dict.eos_index
            config.vocab_size = len(target_dict.symbols)
            vocab_path = os.path.join(pytorch_dump_folder_path, "vocab.json")
            # 检查 pytorch_dump_folder_path 是否为目录，如果不是则报错
            if not os.path.isdir(pytorch_dump_folder_path):
                logger.error("--pytorch_dump_folder_path ({}) should be a directory".format(pytorch_dump_folder_path))
                return
            # 创建目录
            os.makedirs(pytorch_dump_folder_path, exist_ok=True)
            # 将字典索引保存为 JSON 文件
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
            # 创建 Wav2Vec2Processor 对象
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            # 保存处理器配置
            processor.save_pretrained(pytorch_dump_folder_path)

        # 创建 SEWForCTC 模型
        hf_model = SEWForCTC(config)
    else:
        # 创建 SEWModel 模型
        hf_model = SEWModel(config)
        # 保存特征提取器配置
        feature_extractor.save_pretrained(pytorch_dump_folder_path)

    # 递归加载权重
    recursively_load_weights(model, hf_model, is_finetuned)

    # 保存模型
    hf_model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
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
        "--is_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将 SEW 检查点转换为 PyTorch 模型
    convert_sew_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path, args.is_finetuned
    )
```