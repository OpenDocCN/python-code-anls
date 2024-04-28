# `.\transformers\models\wav2vec2\convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置文件的编码格式为 UTF-8
# 版权声明，使用 Apache 2.0 许可证
# 用于转换 Wav2Vec2 检查点

# 导入必要的库
import argparse
import json
import os
import fairseq
import torch
from fairseq.data import Dictionary
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

# 设置日志的详细程度为 info
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# 定义映射关系和顶层键
MAPPING = {
    "post_extract_proj": "feature_projection.projection",
    # ...（其他映射关系）
}
TOP_LEVEL_KEYS = [
    "lm_head",
    "quantizer.weight_proj",
    # ...（其他顶层键）
]

# 读取文本文件内容，转换为字典
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

# 递归设置键和值
def set_recursively(key, value, full_name, weight_type, hf_pointer):
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    hf_param_name = None
    # 遍历PARAM_MAPPING字典的键
    for param_key in PARAM_MAPPING.keys():
        # 如果full_name以param_key结尾
        if full_name.endswith(param_key):
            # 获取PARAM_MAPPING中full_name最后一个点后面的值对应的参数名
            hf_param_name = PARAM_MAPPING[full_name.split(".")[-1]]
            # 设置权重类型为"param"
            weight_type = "param"

    # 如果weight_type不为空且不等于"param"
    if weight_type is not None and weight_type != "param":
        # 获取hf_pointer的weight_type属性的形状
        hf_shape = getattr(hf_pointer, weight_type).shape
    # 如果weight_type不为空且等于"param"
    elif weight_type is not None and weight_type == "param":
        # 设置shape_pointer为hf_pointer
        shape_pointer = hf_pointer
        # 遍历hf_param_name，获取其属性形状
        for attribute in hf_param_name.split("."):
            shape_pointer = getattr(shape_pointer, attribute)
        # 设置hf_shape为shape_pointer的形状
        hf_shape = shape_pointer.shape
        # 减少维度
        value = value[0]
    else:
        # 设置hf_shape为hf_pointer的形状
        hf_shape = hf_pointer.shape

    # 如果hf_shape与value的形状不相等，抛出数值错误
    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    # 根据weight_type的类型，设置hf_pointer的数据值
    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    elif weight_type == "param":
        # 根据hf_param_name的属性，设置hf_pointer的数据值
        for attribute in hf_param_name.split("."):
            hf_pointer = getattr(hf_pointer, attribute)
        hf_pointer.data = value
    else:
        # 设置hf_pointer的数据值为value
        hf_pointer.data = value

    # 记录初始化的信息
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")
# 重命名参数名和值，并将其添加到指定的字典中
def rename_dict(key, value, full_name, weight_type, hf_dict):
    # 初始化 hf_param_name 变量
    hf_param_name = None
    # 遍历 PARAM_MAPPING 字典的键
    for param_key in PARAM_MAPPING.keys():
        # 如果 full_name 以 param_key 结尾
        if full_name.endswith(param_key):
            # 从 PARAM_MAPPING 中获取对应的值并赋给 hf_param_name
            hf_param_name = PARAM_MAPPING[full_name.split(".")[-1]]
            # 将 weight_type 设置为 "param"
            weight_type = "param"

    # 如果 weight_type 不为 None 并且不等于 "param"
    if weight_type is not None and weight_type != "param":
        # 组合 key 和 weight_type，赋值给 full_key
        full_key = ".".join([key, weight_type])
    # 如果 weight_type 不为 None 并且等于 "param"
    elif weight_type is not None and weight_type == "param":
        # 组合 key 和 hf_param_name，赋值给 full_key
        full_key = ".".join([key, hf_param_name])
    else:
        # 否则将 key 赋值给 full_key
        full_key = key

    # 如果 full_key 中包含 "lm_head"
    hf_dict[full_key] = value if "lm_head" in full_key else value[0]


# 参数名映射关系
PARAM_MAPPING = {
    "W_a": "linear_1.weight",
    "W_b": "linear_2.weight",
    "b_a": "linear_1.bias",
    "b_b": "linear_2.bias",
    "ln_W": "norm.weight",
    "ln_b": "norm.bias",
}


# 递归加载 Wav2Vec2 模型的权重
def load_wav2vec2_layer(name, value, hf_model=None, hf_dict=None):
    is_used = False
    # 遍历 MAPPING 字典的键值对
    for key, mapped_key in MAPPING.items():
        # 如果 mapped_key 不在 TOP_LEVEL_KEYS 中，则添加 "wav2vec2."
        mapped_key = "wav2vec2." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key
        # 如果 name 包含 key 或者 name 中 w2v_model. 后的首个部分等于 name 的首个部分
        if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
            is_used = True
            # 如果 mapped_key 中包含 "*"，则替换为 name 中的 layer_index
            if "*" in mapped_key:
                layer_index = name.split(key)[0].split(".")[-2]
                mapped_key = mapped_key.replace("*", layer_index)
            # 根据 name 中的关键词判断 weight_type
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
            # 如果 hf_dict 不为 None，调用 rename_dict 函数
            if hf_dict is not None:
                rename_dict(mapped_key, value, name, weight_type, hf_dict)
            else:
                set_recursively(mapped_key, value, name, weight_type, hf_model)
            # 返回 is_used
            return is_used
    # 返回 is_used
    return is_used


# 递归加载权重
def recursively_load_weights(fairseq_model, hf_model, is_headless):
    # 初始化 unused_weights 列表
    unused_weights = []
    # 获取 fairseq_model 的状态字典
    fairseq_dict = fairseq_model.state_dict()

    # 获取 hf_model 的特征提取器
    feature_extractor = hf_model.wav2vec2.feature_extractor

    # 遍历 fairseq_model 的状态字典
    for name, value in fairseq_dict.items():
        is_used = False
        # 如果 name 中包含 "conv_layers"，调用 load_conv_layer 函数
        if "conv_layers" in name:
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        # 否则调用 load_wav2vec2_layer 函数
        else:
            is_used = load_wav2vec2_layer(name, value, hf_model)
        # 如果未使用则添加到 unused_weights 中
        if not is_used:
            unused_weights.append(name)

    # 输出未使用的权重列表
    logger.warning(f"Unused weights: {unused_weights}")


# 加载卷积层权重
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 获取卷积层名称
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    # 获取层和类型 ID
    layer_id = int(items[0])
    type_id = int(items[1])
    # 如果权重类型为0（表示卷积层）：
    if type_id == 0:
        # 如果权重名称包含“bias”：
        if "bias" in name:
            # 检查权重值的形状是否与特征提取器中对应卷积层的偏置数据形状相匹配，如果不匹配则引发 ValueError 异常
            if value.shape != feature_extractor.conv_layers[layer_id].conv.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
                )
            # 将权重值赋给特征提取器中对应卷积层的偏置数据
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            # 记录日志，表示特征提取器中的卷积层偏置数据已经从给定名称的权重中初始化
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 如果权重名称包含“weight”：
        elif "weight" in name:
            # 检查权重值的形状是否与特征提取器中对应卷积层的权重数据形状相匹配，如果不匹配则引发 ValueError 异常
            if value.shape != feature_extractor.conv_layers[layer_id].conv.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
                )
            # 将权重值赋给特征提取器中对应卷积层的权重数据
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            # 记录日志，表示特征提取器中的卷积层权重数据已经从给定名称的权重中初始化
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 如果权重类型为2（表示层归一化）且不使用组归一化，或者权重类型为2且为第一层且使用组归一化：
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 如果权重名称包含“bias”：
        if "bias" in name:
            # 检查权重值的形状是否与特征提取器中对应层归一化层的偏置数据形状相匹配，如果不匹配则引发 ValueError 异常
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape} was found."
                )
            # 将权重值赋给特征提取器中对应层归一化层的偏置数据
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            # 记录日志，表示特征提取器中的层归一化层偏置数据已经从给定名称的权重中初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 如果权重名称包含“weight”：
        elif "weight" in name:
            # 检查权重值的形状是否与特征提取器中对应层归一化层的权重数据形状相匹配，如果不匹配则引发 ValueError 异常
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape} was found."
                )
            # 将权重值赋给特征提取器中对应层归一化层的权重数据
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            # 记录日志，表示特征提取器中的层归一化层权重数据已经从给定名称的权重中初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    # 否则（其他权重类型）：
    else:
        # 将当前权重名称添加到未使用的权重列表中
        unused_weights.append(full_name)
@torch.no_grad()
# 使用@torch.no_grad()装饰器，在函数运行时取消梯度计算
def convert_wav2vec2_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True, is_seq_class=False
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 如果存在指定的config_path，则使用预训练模型的配置
    if config_path is not None:
        config = Wav2Vec2Config.from_pretrained(config_path)
    else:
        config = Wav2Vec2Config()

    # 如果is_seq_class为True，则为序列分类模型
    if is_seq_class:
        # 读取文件，构建id到label的字典
        id2label = read_txt_into_dict(dict_path)
        config.id2label = id2label
        # 创建Wav2Vec2ForSequenceClassification模型对象
        hf_wav2vec = Wav2Vec2ForSequenceClassification(config)
        # 创建Wav2Vec2FeatureExtractor对象
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0,
            do_normalize=True,
            return_attention_mask=True,
        )
        # 保存feature_extractor
        feature_extractor.save_pretrained(pytorch_dump_folder_path)

    # 如果is_finetuned为True，则为微调模型
    elif is_finetuned:
        if dict_path:
            # 加载字典
            target_dict = Dictionary.load(dict_path)

            # 重要更改bos和pad标记的id，因为CTC的符号是<pad>而不是<s>，和fairseq的不同
            config.bos_token_id = target_dict.pad_index
            config.pad_token_id = target_dict.bos_index
            config.eos_token_id = target_dict.eos_index
            config.vocab_size = len(target_dict.symbols)
            vocab_path = os.path.join(pytorch_dump_folder_path, "vocab.json")
            if not os.path.isdir(pytorch_dump_folder_path):
                logger.error("--pytorch_dump_folder_path ({}) should be a directory".format(pytorch_dump_folder_path))
                return
            os.makedirs(pytorch_dump_folder_path, exist_ok=True)
            vocab_dict = target_dict.indices

            # fairseq有将<pad>和<s>进行了交换
            vocab_dict["<pad>"] = 0
            vocab_dict["<s>"] = 1
            with open(vocab_path, "w", encoding="utf-8") as vocab_handle:
                json.dump(vocab_dict, vocab_handle)
            # 创建Wav2Vec2CTCTokenizer对象
            tokenizer = Wav2Vec2CTCTokenizer(
                vocab_path,
                unk_token=target_dict.unk_word,
                pad_token=target_dict.pad_word,
                bos_token=target_dict.bos_word,
                eos_token=target_dict.eos_word,
                word_delimiter_token="|",
                do_lower_case=False,
            )
            # 根据config特征提取模块的设置，判断是否返回attention mask
            return_attention_mask = True if config.feat_extract_norm == "layer" else False
            # 创建Wav2Vec2FeatureExtractor对象
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0,
                do_normalize=True,
                return_attention_mask=return_attention_mask,
            )
            # 创建Wav2Vec2Processor对象
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            # 保存processor
            processor.save_pretrained(pytorch_dump_folder_path)

        # 创建Wav2Vec2ForCTC模型对象
        hf_wav2vec = Wav2Vec2ForCTC(config)
    else:
        # 创建Wav2Vec2ForPreTraining模型对象
        hf_wav2vec = Wav2Vec2ForPreTraining(config)
    # 如果进行了微调或者是序列分类任务
    if is_finetuned or is_seq_class:
        # 加载模型和任务，使用指定参数覆盖
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1])}
        )
    # 如果不是微调或序列分类任务
    else:
        # 创建音频预训练任务参数
        task_arg = argparse.Namespace(task="audio_pretraining")
        # 设置音频预训练任务
        task = fairseq.tasks.setup_task(task_arg)

        # 加载模型和任务
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path], task=task)

    # 将模型设置为评估模式
    model = model[0].eval()

    # 递归加载权重
    recursively_load_weights(model, hf_wav2vec, not is_finetuned)

    # 保存预训练模型的权重到指定文件夹
    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)
# 判断是否运行在主程序中
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加参数--pytorch_dump_folder_path，设置默认值为None，类型为字符串，帮助信息为输出PyTorch模型的路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加参数--checkpoint_path，设置默认值为None，类型为字符串，帮助信息为fairseq检查点的路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    # 添加参数--dict_path，设置默认值为None，类型为字符串，帮助信息为经过微调的模型的字典路径
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    # 添加参数--config_path，设置默认值为None，类型为字符串，帮助信息为要转换的模型的hf config.json路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加参数--not_finetuned，为布尔类型，设置为True表示模型不是经过微调的模型
    parser.add_argument(
        "--not_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    # 添加参数--is_seq_class，为布尔类型，设置为True表示模型是经过微调的序列分类模型
    parser.add_argument(
        "--is_seq_class",
        action="store_true",
        help="Whether the model to convert is a fine-tuned sequence classification model or not",
    )
    # 解析传递给程序的参数
    args = parser.parse_args()

    # 判断是否模型经过微调，并且不是序列分类模型
    is_finetuned = not args.not_finetuned and not args.is_seq_class
    # 调用函数convert_wav2vec2_checkpoint并传递参数
    convert_wav2vec2_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path,
        args.dict_path,
        is_finetuned,
        args.is_seq_class,
    )
```