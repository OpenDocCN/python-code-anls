# `.\models\unispeech_sat\convert_unispeech_sat_original_pytorch_checkpoint_to_pytorch.py`

```py
# 设置文件编码为 UTF-8

# 引用必要的库和模块
import argparse  # 导入命令行参数解析库 argparse
import fairseq   # 导入 fairseq 库
import torch     # 导入 PyTorch 库

# 从 transformers 库中导入 UniSpeechSatConfig, UniSpeechSatForCTC, UniSpeechSatForPreTraining 和 logging
from transformers import UniSpeechSatConfig, UniSpeechSatForCTC, UniSpeechSatForPreTraining, logging

# 设置日志的详细程度为 info
logging.set_verbosity_info()

# 获取当前模块的 logger 对象
logger = logging.get_logger(__name__)

# 定义一个映射字典，用于将 UniSpeechSat 模型的参数名映射到 HuggingFace 模型的参数名
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
    "encoder.layer_norm_for_extract": "layer_norm_for_extract",
    "w2v_model.layer_norm": "feature_projection.layer_norm",
    "quantizer.weight_proj": "quantizer.weight_proj",
    "quantizer.vars": "quantizer.codevectors",
    "project_q": "project_q",
    "final_proj": "project_hid",
    "w2v_encoder.proj": "lm_head",
    "label_embs_concat": "label_embeddings_concat",
    "mask_emb": "masked_spec_embed",
    "spk_proj": "speaker_proj",
}

# 定义顶层键列表，列出需要逐层设置的顶层参数名
TOP_LEVEL_KEYS = [
    "lm_head",
    "quantizer.weight_proj",
    "quantizer.codevectors",
    "project_q",
    "project_hid",
    "label_embeddings_concat",
    "speaker_proj",
    "layer_norm_for_extract",
]

# 定义一个函数，递归设置 HuggingFace 模型的参数
def set_recursively(hf_pointer, key, value, full_name, weight_type):
    # 按照 key 的路径逐级获取 hf_pointer 的属性
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    # 根据 weight_type 确定需要设置的参数的形状
    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    # 检查待设置参数的形状是否与传入 value 的形状一致，若不一致则抛出异常
    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    # 根据 weight_type 类型设置不同的参数值
    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    # 如果权重类型是偏置（bias），设置模型指针的偏置数据为给定的值
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    # 否则，设置模型指针的数据为给定的值
    else:
        hf_pointer.data = value

    # 记录初始化信息到日志，包括模型中的键（如果存在）、权重类型（如果存在）、以及从哪里初始化的信息
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")
# 递归加载权重到 HF 模型中
def recursively_load_weights(fairseq_model, hf_model):
    # 未使用的权重列表
    unused_weights = []
    # 获取 Fairseq 模型的状态字典
    fairseq_dict = fairseq_model.state_dict()

    # 获取 HF 模型的特征提取器
    feature_extractor = hf_model.unispeech_sat.feature_extractor

    # 遍历 Fairseq 模型的状态字典中的每个键值对
    for name, value in fairseq_dict.items():
        # 是否被使用的标志
        is_used = False
        # 如果名称中包含 "conv_layers"
        if "conv_layers" in name:
            # 调用加载卷积层的函数
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        else:
            # 遍历映射字典 MAPPING 中的每个键值对
            for key, mapped_key in MAPPING.items():
                # 将 mapped_key 添加前缀 "unispeech_sat."，如果它不在 TOP_LEVEL_KEYS 中
                mapped_key = "unispeech_sat." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key
                # 如果名称中包含 key，或者 key 的最后一部分等于名称的第一个部分
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    # 如果名称包含 "layer_norm_for_extract" 并且名称不完全匹配 key，则继续下一个循环
                    if "layer_norm_for_extract" in name and (".".join(name.split(".")[:-1]) != key):
                        continue
                    is_used = True
                    # 如果 mapped_key 包含通配符 "*", 则替换为名称中的层索引
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    # 根据名称的后缀确定权重类型
                    if "weight_g" in name:
                        weight_type = "weight_g"
                    elif "weight_v" in name:
                        weight_type = "weight_v"
                    elif "bias" in name:
                        weight_type = "bias"
                    elif "weight" in name:
                        # TODO: 不匹配 quantizer.weight_proj
                        weight_type = "weight"
                    else:
                        weight_type = None
                    # 递归设置 HF 模型的权重
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        # 如果没有使用，则将名称添加到未使用的权重列表中
        if not is_used:
            unused_weights.append(name)

    # 记录未使用的权重列表
    logger.warning(f"Unused weights: {unused_weights}")


# 加载卷积层
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 获取卷积层名称
    name = full_name.split("conv_layers.")[-1]
    # 将名称拆分为项目列表
    items = name.split(".")
    # 提取层 ID 和类型 ID
    layer_id = int(items[0])
    type_id = int(items[1])
    # 如果类型ID为0
    if type_id == 0:
        # 如果变量名包含"bias"
        if "bias" in name:
            # 检查值的形状是否与卷积层偏置数据的形状相匹配，若不匹配则引发数值错误
            if value.shape != feature_extractor.conv_layers[layer_id].conv.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
                )
            # 将值赋给特征提取器的卷积层偏置数据，并记录日志
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 如果变量名包含"weight"
        elif "weight" in name:
            # 检查值的形状是否与卷积层权重数据的形状相匹配，若不匹配则引发数值错误
            if value.shape != feature_extractor.conv_layers[layer_id].conv.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
                )
            # 将值赋给特征提取器的卷积层权重数据，并记录日志
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    
    # 如果类型ID为2且不使用组规范，或者类型ID为2且为第一层且使用组规范
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 如果变量名包含"bias"
        if "bias" in name:
            # 检查值的形状是否与特征提取器的层归一化偏置数据的形状相匹配，若不匹配则引发数值错误
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape} was found."
                )
            # 将值赋给特征提取器的层归一化偏置数据，并记录日志
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 如果变量名包含"weight"
        elif "weight" in name:
            # 检查值的形状是否与特征提取器的层归一化权重数据的形状相匹配，若不匹配则引发数值错误
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape} was found."
                )
            # 将值赋给特征提取器的层归一化权重数据，并记录日志
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    
    # 如果以上条件都不满足，则将变量名添加到未使用的权重列表中
    else:
        unused_weights.append(full_name)
# 声明一个装饰器，表示在该函数执行时不需要计算梯度信息
@torch.no_grad()
# 定义一个函数，将 UniSpeech 模型的检查点转换为 Transformers 设计
def convert_unispeech_sat_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 如果提供了配置文件路径，则使用预训练配置文件创建 UniSpeechSatConfig 对象
    if config_path is not None:
        config = UniSpeechSatConfig.from_pretrained(config_path)
    else:
        # 否则，创建一个空的 UniSpeechSatConfig 对象
        config = UniSpeechSatConfig()

    # 重置 dict_path 变量为空字符串
    dict_path = ""

    # 根据是否微调标志，选择不同类型的 UniSpeechSat 模型
    if is_finetuned:
        hf_wav2vec = UniSpeechSatForCTC(config)
    else:
        hf_wav2vec = UniSpeechSatForPreTraining(config)

    # 使用 fairseq 提供的工具加载模型集合和任务信息
    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1])}
    )
    # 将加载的模型设置为评估模式，即不计算梯度
    model = model[0].eval()

    # 递归地加载模型的权重到 hf_wav2vec 模型中
    recursively_load_weights(model, hf_wav2vec)

    # 将转换后的 hf_wav2vec 模型保存到指定的 PyTorch 输出文件夹中
    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)


# 如果该脚本作为主程序运行，则执行以下代码
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--not_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    args = parser.parse_args()
    
    # 调用 convert_unispeech_sat_checkpoint 函数，并根据命令行参数决定是否微调模型
    convert_unispeech_sat_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path, not args.not_finetuned
    )
```