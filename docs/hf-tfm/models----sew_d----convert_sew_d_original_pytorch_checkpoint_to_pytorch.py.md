# `.\transformers\models\sew_d\convert_sew_d_original_pytorch_checkpoint_to_pytorch.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本使用此文件
# 只有在遵守许可证的情况下才能使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的
# 没有任何明示或暗示的担保或条件，包括但不限于特定用途的适用性
# 请查看许可证以获取有关权限和限制的详细信息
"""Convert SEW checkpoint."""

# 导入所需的库
import argparse
import json
import os

import fairseq
import torch
from fairseq.data import Dictionary

# 注册 SEW 的 fairseq 模块
from sew_asapp import tasks  # noqa: F401

# 导入所需的 Transformers 库中的模块
from transformers import (
    SEWDConfig,
    SEWDForCTC,
    SEWDModel,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    logging,
)

# 设置日志级别为 info
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义映射关系
MAPPING = {
    "post_extract_proj": "feature_projection",
    "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",
    "attention.self.query_proj": "encoder.encoder.layer.*.attention.self.query_proj",
    "attention.self.key_proj": "encoder.encoder.layer.*.attention.self.key_proj",
    "attention.self.value_proj": "encoder.encoder.layer.*.attention.self.value_proj",
    "attention.output.dense": "encoder.encoder.layer.*.attention.output.dense",
    "attention.output.LayerNorm": "encoder.encoder.layer.*.attention.output.LayerNorm",
    "intermediate.dense": "encoder.encoder.layer.*.intermediate.dense",
    "output.dense": "encoder.encoder.layer.*.output.dense",
    "output.LayerNorm": "encoder.encoder.layer.*.output.LayerNorm",
    "encoder.encoder.rel_embeddings": "encoder.encoder.rel_embeddings",
    "encoder.encoder.LayerNorm": "encoder.encoder.LayerNorm",
    "encoder.upsample.0": "encoder.upsample.projection",
    "encoder.layer_norm": "encoder.layer_norm",
    "w2v_model.layer_norm": "layer_norm",
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
    # 使用日志记录器输出初始化信息，包括键和权重类型（如果有），以及从哪个完整名称初始化
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")
# 递归加载权重到模型中
def recursively_load_weights(fairseq_model, hf_model, is_finetuned):
    # 未使用的权重列表
    unused_weights = []
    # 获取 fairseq 模型的状态字典
    fairseq_dict = fairseq_model.state_dict()

    # 根据是否微调选择特征提取器
    feature_extractor = hf_model.sew_d.feature_extractor if is_finetuned else hf_model.feature_extractor

    # 遍历 fairseq 模型的状态字典
    for name, value in fairseq_dict.items():
        is_used = False
        # 如果名称中包含 "conv_layers"，则加载卷积层
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
            # 遍历映射关系，根据映射关系加载权重
            for key, mapped_key in MAPPING.items():
                mapped_key = "sew_d." + mapped_key if (is_finetuned and mapped_key != "lm_head") else mapped_key

                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    is_used = True
                    # 处理特殊情况，根据映射关系设置权重
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        if not layer_index.isnumeric():
                            continue
                        mapped_key = mapped_key.replace("*", layer_index)
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
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        # 如果未使用，则添加到未使用的权重列表中
        if not is_used:
            unused_weights.append(name)

    # 输出未使用的权重列表
    logger.warning(f"Unused weights: {unused_weights}")


# 加载卷积层
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 获取卷积层名称和编号
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])
    # 如果类型为0
    if type_id == 0:
        # 如果名称中包含"bias"
        if "bias" in name:
            # 断言值的形状与特征提取器卷积层的卷积偏置数据形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            # 将值赋给特征提取器卷积层的卷积偏置数据
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            # 记录日志
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 如果名称中包含"weight"
        elif "weight" in name:
            # 断言值的形状与特征提取器卷积层的卷积权重数据形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            # 将值赋给特征提取器卷积层的卷积权重数据
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            # 记录日志
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 如果类型为2且不使用组归一化，或者类型为2且层ID为0且使用组归一化
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 如果名称中包含"bias"
        if "bias" in name:
            # 断言值的形状与特征提取器卷积层的层归一化偏置数据形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape, (
                f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            # 将值赋给特征提取器卷积层的层归一化偏置数据
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            # 记录日志
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 如果名称中包含"weight"
        elif "weight" in name:
            # 断言值的形状与特征提取器卷积层的层归一化权重数据形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            )
            # 将值赋给特征提取器卷积层的层归一化权重数据
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            # 记录日志
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    # 否则
    else:
        # 将未使用的权重名称添加到列表中
        unused_weights.append(full_name)
# 将模型配置转换为 SEWDConfig 对象
def convert_config(model, is_finetuned):
    # 创建一个 SEWDConfig 对象
    config = SEWDConfig()
    # 根据是否微调选择不同的配置
    if is_finetuned:
        fs_config = model.w2v_encoder.w2v_model.cfg
    else:
        fs_config = model.cfg

    # 设置 SEWDConfig 对象的各项属性值
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
    # DeBERTa-specific parameters:
    config.max_position_embeddings = fs_config.max_position_embeddings
    config.position_buckets = fs_config.position_buckets
    config.share_att_key = fs_config.share_att_key
    config.relative_attention = fs_config.relative_attention
    config.position_biased_input = fs_config.position_biased_input
    config.pos_att_type = tuple(fs_config.pos_att_type.split("|"))
    config.norm_rel_ebd = fs_config.norm_rel_ebd

    # 处理被 Wav2VecCtc 模型覆盖的任何参数
    if is_finetuned:
        fs_config = model.cfg
        config.final_dropout = fs_config.final_dropout
        config.layerdrop = fs_config.layerdrop
    config.activation_dropout = fs_config.activation_dropout
    config.apply_spec_augment = fs_config.mask_prob > 0 or fs_config.mask_channel_prob > 0
    config.attention_dropout = fs_config.attention_dropout
    config.feat_proj_dropout = fs_config.dropout_input
    config.hidden_dropout = fs_config.dropout
    config.mask_feature_length = fs_config.mask_channel_length
    config.mask_feature_prob = fs_config.mask_channel_prob
    config.mask_time_length = fs_config.mask_length
    config.mask_time_prob = fs_config.mask_prob

    config.feature_extractor_type = "Wav2Vec2FeatureExtractor"
    config.tokenizer_class = "Wav2Vec2CTCTokenizer"

    return config

# 无需梯度计算的函数
@torch.no_grad()
def convert_sew_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 如果模型是微调过的，则加载模型集合和任务，同时覆盖参数中的"data"字段
    if is_finetuned:
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1])}
        )
    # 如果模型没有微调，则直接加载模型集合和任务
    else:
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])

    # 如果配置文件路径不为空，则从预训练配置文件中加载配置，否则根据模型和微调状态转换配置
    if config_path is not None:
        config = SEWDConfig.from_pretrained(config_path)
    else:
        config = convert_config(model[0], is_finetuned)
    # 将模型设置为评估模式
    model = model[0].eval()

    # 根据配置中的特征提取规范，确定是否返回注意力掩码
    return_attention_mask = True if config.feat_extract_norm == "layer" else False
    # 创建Wav2Vec2特征提取器
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0,
        do_normalize=True,
        return_attention_mask=return_attention_mask,
    )

    # 如果是微调模型，则加载目标字典并进行一些重要的更改
    if is_finetuned:
        if dict_path:
            target_dict = Dictionary.load(dict_path)

            # 重要更改bos和pad标记ID，因为CTC符号是<pad>而不是<s>，如在fairseq中
            target_dict.indices[target_dict.bos_word] = target_dict.pad_index
            target_dict.indices[target_dict.pad_word] = target_dict.bos_index
            config.bos_token_id = target_dict.pad_index
            config.pad_token_id = target_dict.bos_index
            config.eos_token_id = target_dict.eos_index
            config.vocab_size = len(target_dict.symbols)
            vocab_path = os.path.join(pytorch_dump_folder_path, "vocab.json")
            # 如果pytorch_dump_folder_path不是目录，则记录错误并返回
            if not os.path.isdir(pytorch_dump_folder_path):
                logger.error("--pytorch_dump_folder_path ({}) should be a directory".format(pytorch_dump_folder_path))
                return
            # 创建目录pytorch_dump_folder_path，如果不存在则创建
            os.makedirs(pytorch_dump_folder_path, exist_ok=True)
            # 将目标字典的索引保存到vocab.json文件中
            with open(vocab_path, "w", encoding="utf-8") as vocab_handle:
                json.dump(target_dict.indices, vocab_handle)
            # 创建Wav2Vec2CTCTokenizer
            tokenizer = Wav2Vec2CTCTokenizer(
                vocab_path,
                unk_token=target_dict.unk_word,
                pad_token=target_dict.pad_word,
                bos_token=target_dict.bos_word,
                eos_token=target_dict.eos_word,
                word_delimiter_token="|",
                do_lower_case=False,
            )
            # 创建Wav2Vec2Processor
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            # 保存processor到pytorch_dump_folder_path
            processor.save_pretrained(pytorch_dump_folder_path)

        # 创建SEWDForCTC模型
        hf_model = SEWDForCTC(config)
    else:
        # 创建SEWDModel模型
        hf_model = SEWDModel(config)
        # 保存特征提取器到pytorch_dump_folder_path
        feature_extractor.save_pretrained(pytorch_dump_folder_path)

    # 递归加载权重
    recursively_load_weights(model, hf_model, is_finetuned)

    # 保存hf_model到pytorch_dump_folder_path
    hf_model.save_pretrained(pytorch_dump_folder_path)
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
    # 添加参数--is_finetuned，标志要转换的模型是否是微调模型
    parser.add_argument(
        "--is_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数convert_sew_checkpoint，将参数传递给该函数
    convert_sew_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path, args.is_finetuned
    )
```