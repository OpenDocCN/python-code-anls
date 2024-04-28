# `.\transformers\models\wav2vec2_conformer\convert_wav2vec2_conformer_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置文件编码格式为 utf-8

# 版权声明和许可信息

# 导入需要的库
import argparse  # 用于解析命令行参数
import json  # 用于读取和处理 JSON 格式数据
import os  # 提供操作系统相关的功能

import fairseq  # 导入 fairseq 库
import torch  # 导入 PyTorch 库
from fairseq.data import Dictionary  # 从 fairseq 库中导入 Dictionary 类

from transformers import (  # 从 transformers 库中导入以下模块
    Wav2Vec2ConformerConfig,  # 导入 Wav2Vec2ConformerConfig 类
    Wav2Vec2ConformerForCTC,  # 导入 Wav2Vec2ConformerForCTC 类
    Wav2Vec2ConformerForPreTraining,  # 导入 Wav2Vec2ConformerForPreTraining 类
    Wav2Vec2CTCTokenizer,  # 导入 Wav2Vec2CTCTokenizer 类
    Wav2Vec2FeatureExtractor,  # 导入 Wav2Vec2FeatureExtractor 类
    Wav2Vec2Processor,  # 导入 Wav2Vec2Processor 类
    logging,  # 导入 logging 模块
)

logging.set_verbosity_info()  # 设置日志的输出级别为info
logger = logging.get_logger(__name__)  # 获取当前模块的 logger

MAPPING = {  # 定义一个名为 MAPPING 的字典
    # 一系列字符串匹配映射，用于将 fairseq 模型的参数映射到 Hugging Face 的模型参数上
}

TOP_LEVEL_KEYS = [  # 定义一个名为 TOP_LEVEL_KEYS 的列表
    "lm_head",  # 添加字符串 "lm_head" 到列表
    "quantizer.weight_proj",  # 添加字符串 "quantizer.weight_proj" 到列表
    # 定义了三个字符串，分别为 "quantizer.codevectors"、"project_q" 和 "project_hid"
    "quantizer.codevectors",
    "project_q",
    "project_hid",
# 递归设置指定属性的值
def set_recursively(hf_pointer, key, value, full_name, weight_type):
    # 将 key 按 '.' 分割，逐级获取属性对象
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    # 获取指定权重类型的形状
    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    # 检查形状是否一致，不一致则抛出异常
    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    # 根据权重类型设置值
    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    elif weight_type == "running_mean":
        hf_pointer.running_mean.data = value
    elif weight_type == "running_var":
        hf_pointer.running_var.data = value
    elif weight_type == "num_batches_tracked":
        hf_pointer.num_batches_tracked.data = value
    elif weight_type == "inv_freq":
        hf_pointer.inv_freq.data = value
    else:
        hf_pointer.data = value

    # 记录初始化信息
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")


# 递归加载模型权重
def recursively_load_weights(fairseq_model, hf_model, is_headless):
    unused_weights = []  # 未使用的权重列表
    fairseq_dict = fairseq_model.state_dict()  # 获取 fairseq 模型的状态字典

    # 获取 hf_model 的特征提取器
    feature_extractor = hf_model.wav2vec2_conformer.feature_extractor
    # 遍历 fairseq_dict 字典中的每个键值对
    for name, value in fairseq_dict.items():
        # 初始化是否使用的标志位为 False
        is_used = False
        # 如果键名中包含 "conv_layers"
        if "conv_layers" in name:
            # 调用 load_conv_layer 函数来加载卷积层
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            # 将标志位设置为 True，表示该键名已被使用
            is_used = True
        # 如果键名不包含 "conv_layers"
        else:
            # 遍历 MAPPING 字典中的每个键值对
            for key, mapped_key in MAPPING.items():
                # 如果 mapped_key 不在 TOP_LEVEL_KEYS 中，则添加前缀 "wav2vec2_conformer."
                mapped_key = "wav2vec2_conformer." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key
                # 如果 key 在键名中，或者 key 去除前缀 "w2v_model." 后等于键名去除后缀第一个"."之前的部分
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    # 将标志位设置为 True，表示该键名已被使用
                    is_used = True
                    # 如果 mapped_key 中包含 "*", 替换成相应的层数索引
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    # 判断键名属于哪种类型
                    if "pos_bias_u" in name:
                        weight_type = None
                    elif "pos_bias_v" in name:
                        weight_type = None
                    elif "weight_g" in name:
                        weight_type = "weight_g"
                    elif "weight_v" in name:
                        weight_type = "weight_v"
                    elif "bias" in name:
                        weight_type = "bias"
                    elif "weight" in name:
                        # TODO: 不要匹配 quantizer.weight_proj
                        weight_type = "weight"
                    elif "running_mean" in name:
                        weight_type = "running_mean"
                    elif "inv_freq" in name:
                        weight_type = "inv_freq"
                    elif "running_var" in name:
                        weight_type = "running_var"
                    elif "num_batches_tracked" in name:
                        weight_type = "num_batches_tracked"
                    else:
                        weight_type = None
                    # 递归地设置 hf_model 中的 mapped_key 经过处理后的值
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                # 结束当前循环
                continue
        # 如果标志位为 False，表示该键名未被使用，将其添加到未使用权重列表中
        if not is_used:
            unused_weights.append(name)

    # 输出未使用的权重名称列表
    logger.warning(f"Unused weights: {unused_weights}")
# 从 transformers.models.wav2vec2.convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.load_conv_layer 复制而来的函数
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 从完整名字中提取出层的名字
    name = full_name.split("conv_layers.")[-1]
    # 将层的名字按照"."分割成不同部分
    items = name.split(".")
    # 获取层的 ID
    layer_id = int(items[0])
    # 获取层的类型 ID
    type_id = int(items[1])

    # 如果是卷积层
    if type_id == 0:
        # 如果是偏置项
        if "bias" in name:
            # 检查值的形状是否与模型中相应偏置项的形状匹配
            if value.shape != feature_extractor.conv_layers[layer_id].conv.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
                )
            # 将值赋给模型中相应的偏置项
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 如果是权重项
        elif "weight" in name:
            # 检查值的形状是否与模型中相应权重项的形状匹配
            if value.shape != feature_extractor.conv_layers[layer_id].conv.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
                )
            # 将值赋给模型中相应的权重项
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 如果是层归一化层
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 如果是偏置项
        if "bias" in name:
            # 检查值的形状是否与模型中相应层归一化层的偏置项的形状匹配
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape} was found."
                )
            # 将值赋给模型中相应的层归一化层的偏置项
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 如果是权重项
        elif "weight" in name:
            # 检查值的形状是否与模型中相应层归一化层的权重项的形状匹配
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape} was found."
                )
            # 将值赋给模型中相应的层归一化层的权重项
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        # 如果层既不是卷积层也不是层归一化层，则将其添加到未使用权重列表中
        unused_weights.append(full_name)


# 使用 torch.no_grad() 装饰器，禁用梯度计算
@torch.no_grad()
def convert_wav2vec2_conformer_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 如果提供了配置路径，则从预训练模型加载配置
    if config_path is not None:
        # 从预训练模型加载配置，指定隐藏层激活函数为 swish
        config = Wav2Vec2ConformerConfig.from_pretrained(config_path, hidden_act="swish")
    else:
        # 如果未提供特定配置，则使用默认的 Wav2Vec2ConformerConfig 配置
        config = Wav2Vec2ConformerConfig()

    if "rope" in checkpoint_path:
        # 如果模型路径包含 "rope"，则设定位置嵌入类型为 "rotary"
        config.position_embeddings_type = "rotary"

    if is_finetuned:
        if dict_path:
            # 如果是微调模型并且提供了字典路径，则加载字典
            target_dict = Dictionary.load(dict_path)

            # 将 bos 和 pad 标记设为字典中的 pad_index 和 bos_index
            # 因为 CTC 符号中 <pad> 是 <s>，而不是 fairseq 中的 <s>
            config.bos_token_id = target_dict.pad_index
            config.pad_token_id = target_dict.bos_index
            config.eos_token_id = target_dict.eos_index
            config.vocab_size = len(target_dict.symbols)
            vocab_path = os.path.join(pytorch_dump_folder_path, "vocab.json")

            # 检查路径是否为目录，若不是则报错
            if not os.path.isdir(pytorch_dump_folder_path):
                logger.error("--pytorch_dump_folder_path ({}) 应为目录".format(pytorch_dump_folder_path))
                return

            # 创建目录，如果不存在的话
            os.makedirs(pytorch_dump_folder_path, exist_ok=True)
            vocab_dict = target_dict.indices

            # 在 fairseq 中，
# 如果当前模块是主程序，而非被导入的模块
if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数，指定输出 PyTorch 模型的路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加命令行参数，指定 fairseq checkpoint 的路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    # 添加命令行参数，指定 fine-tuned 模型的字典路径
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    # 添加命令行参数，指定要转换模型的 hf config.json 路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加命令行参数，指定模型是否为 fine-tuned 模型
    parser.add_argument(
        "--not_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 将参数传递给 convert_wav2vec2_conformer_checkpoint 函数，进行模型转换
    convert_wav2vec2_conformer_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path, not args.not_finetuned
    )
```