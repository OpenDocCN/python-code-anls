# `.\models\hubert\convert_hubert_original_pytorch_checkpoint_to_pytorch.py`

```py
# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import json  # 用于处理 JSON 格式的数据
import os  # 提供与操作系统交互的功能

import fairseq  # 导入 fairseq 库，用于处理 Fairseq 模型
import torch  # 导入 PyTorch 库，用于深度学习任务
from fairseq.data import Dictionary  # 从 fairseq 库中导入 Dictionary 类

# 从 transformers 库中导入相关类和函数
from transformers import (
    HubertConfig,  # 导入 HubertConfig 类，用于配置 Hubert 模型的参数
    HubertForCTC,  # 导入 HubertForCTC 类，用于 Hubert 模型的 CTC（Connectionist Temporal Classification）任务
    HubertModel,  # 导入 HubertModel 类，用于加载 Hubert 模型
    Wav2Vec2CTCTokenizer,  # 导入 Wav2Vec2CTCTokenizer 类，用于 CTC 任务的标记化
    Wav2Vec2FeatureExtractor,  # 导入 Wav2Vec2FeatureExtractor 类，用于音频特征提取
    Wav2Vec2Processor,  # 导入 Wav2Vec2Processor 类，用于处理 Wav2Vec2 模型的输入输出
    logging,  # 导入 logging 模块，用于日志记录
)

logging.set_verbosity_info()  # 设置日志记录级别为 info
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

MAPPING = {
    "post_extract_proj": "feature_projection.projection",  # 映射 Fairseq 中的 post_extract_proj 到 Transformers 中的 feature_projection.projection
    "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",  # 映射 Fairseq 中的 encoder.pos_conv.0 到 Transformers 中的 encoder.pos_conv_embed.conv
    "self_attn.k_proj": "encoder.layers.*.attention.k_proj",  # 映射 Fairseq 中的 self_attn.k_proj 到 Transformers 中的 encoder.layers.*.attention.k_proj
    "self_attn.v_proj": "encoder.layers.*.attention.v_proj",  # 映射 Fairseq 中的 self_attn.v_proj 到 Transformers 中的 encoder.layers.*.attention.v_proj
    "self_attn.q_proj": "encoder.layers.*.attention.q_proj",  # 映射 Fairseq 中的 self_attn.q_proj 到 Transformers 中的 encoder.layers.*.attention.q_proj
    "self_attn.out_proj": "encoder.layers.*.attention.out_proj",  # 映射 Fairseq 中的 self_attn.out_proj 到 Transformers 中的 encoder.layers.*.attention.out_proj
    "self_attn_layer_norm": "encoder.layers.*.layer_norm",  # 映射 Fairseq 中的 self_attn_layer_norm 到 Transformers 中的 encoder.layers.*.layer_norm
    "fc1": "encoder.layers.*.feed_forward.intermediate_dense",  # 映射 Fairseq 中的 fc1 到 Transformers 中的 encoder.layers.*.feed_forward.intermediate_dense
    "fc2": "encoder.layers.*.feed_forward.output_dense",  # 映射 Fairseq 中的 fc2 到 Transformers 中的 encoder.layers.*.feed_forward.output_dense
    "final_layer_norm": "encoder.layers.*.final_layer_norm",  # 映射 Fairseq 中的 final_layer_norm 到 Transformers 中的 encoder.layers.*.final_layer_norm
    "encoder.layer_norm": "encoder.layer_norm",  # 映射 Fairseq 中的 encoder.layer_norm 到 Transformers 中的 encoder.layer_norm
    "w2v_model.layer_norm": "feature_projection.layer_norm",  # 映射 Fairseq 中的 w2v_model.layer_norm 到 Transformers 中的 feature_projection.layer_norm
    "w2v_encoder.proj": "lm_head",  # 映射 Fairseq 中的 w2v_encoder.proj 到 Transformers 中的 lm_head
    "mask_emb": "masked_spec_embed",  # 映射 Fairseq 中的 mask_emb 到 Transformers 中的 masked_spec_embed
}


def set_recursively(hf_pointer, key, value, full_name, weight_type):
    """
    递归设置指针指向的属性值，并记录日志。

    Args:
        hf_pointer (object): Transformers 模型中的属性指针
        key (str): 属性名称，用点分隔表示层次结构
        value (torch.Tensor): 设置的值
        full_name (str): 完整名称，用于日志记录
        weight_type (str): 权重类型，如 'weight', 'bias' 等

    Raises:
        AssertionError: 如果设置的值的形状与预期不符合
    """
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
    """
    递归加载 Fairseq 模型的权重到 Transformers 模型中。

    Args:
        fairseq_model (FairseqModel): Fairseq 模型对象
        hf_model (PreTrainedModel): Transformers 模型对象
        is_finetuned (bool): 是否为微调模型

    Returns:
        None
    """
    unused_weights = []
    fairseq_dict = fairseq_model.state_dict()

    # 根据是否微调选择要加载权重的对象
    feature_extractor = hf_model.hubert.feature_extractor if is_finetuned else hf_model.feature_extractor
    # 遍历 fairseq_dict 字典中的每个键值对
    for name, value in fairseq_dict.items():
        # 初始化是否被使用的标志为 False
        is_used = False
        
        # 检查当前 name 是否包含 "conv_layers" 字符串
        if "conv_layers" in name:
            # 调用 load_conv_layer 函数加载卷积层权重
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            # 标记当前 name 已被使用
            is_used = True
        else:
            # 遍历 MAPPING 字典中的每个键值对
            for key, mapped_key in MAPPING.items():
                # 如果是微调模型并且 mapped_key 不是 "lm_head"，则加上前缀 "hubert."
                mapped_key = "hubert." + mapped_key if (is_finetuned and mapped_key != "lm_head") else mapped_key
                
                # 检查 key 是否在 name 中，或者检查是否不是微调且 name 的第一部分与 key 的最后一部分相同
                if key in name or (key.split("w2v_model.")[-1] == name.split(".")[0] and not is_finetuned):
                    # 标记当前 name 已被使用
                    is_used = True
                    
                    # 如果 mapped_key 中包含通配符 "*", 替换为当前层的索引
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    
                    # 根据 name 的后缀确定权重类型
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
                
                # 继续下一个键值对的检查
                continue
        
        # 如果当前 name 没有被使用，则将其添加到 unused_weights 列表中
        if not is_used:
            unused_weights.append(name)

    # 记录警告信息，显示未使用的权重名称列表
    logger.warning(f"Unused weights: {unused_weights}")
# 定义函数，加载卷积层权重到特征提取器中
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 从完整名称中提取层的简化名称
    name = full_name.split("conv_layers.")[-1]
    # 根据点号分割名称，获取层索引和类型索引
    items = name.split(".")
    layer_id = int(items[0])  # 卷积层索引
    type_id = int(items[1])   # 类型索引

    # 处理卷积层权重和偏置
    if type_id == 0:
        if "bias" in name:
            # 断言确保值的形状与特征提取器中的偏置形状匹配
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            # 将偏置值赋给特征提取器的相应卷积层
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            # 记录初始化日志
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            # 断言确保值的形状与特征提取器中的权重形状匹配
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            # 将权重值赋给特征提取器的相应卷积层
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            # 记录初始化日志
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 处理层归一化参数
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        if "bias" in name:
            # 断言确保值的形状与特征提取器中的层归一化偏置形状匹配
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape, (
                f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            # 将层归一化偏置值赋给特征提取器的相应卷积层
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            # 记录初始化日志
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            # 断言确保值的形状与特征提取器中的层归一化权重形状匹配
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            )
            # 将层归一化权重值赋给特征提取器的相应卷积层
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            # 记录初始化日志
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        # 将未使用的权重名称添加到未使用列表中
        unused_weights.append(full_name)


# 无需计算梯度的上下文管理器装饰器
@torch.no_grad()
# 将Hubert模型的检查点转换为transformers设计的函数
def convert_hubert_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 如果提供了配置文件路径，则从预训练配置中加载配置
    if config_path is not None:
        config = HubertConfig.from_pretrained(config_path)
    else:
        # 否则使用默认配置
        config = HubertConfig()
    # 如果模型是微调过的
    if is_finetuned:
        # 如果提供了字典路径，则加载字典
        if dict_path:
            target_dict = Dictionary.load(dict_path)

            # 重要的更改：调整 bos 和 pad 的标记 ID，因为 CTC 符号是 <pad>，而不是 fairseq 中的 <s>
            config.bos_token_id = target_dict.pad_index
            config.pad_token_id = target_dict.bos_index
            config.eos_token_id = target_dict.eos_index
            config.vocab_size = len(target_dict.symbols)

            # 设置保存词汇表 JSON 文件的路径
            vocab_path = os.path.join(pytorch_dump_folder_path, "vocab.json")
            # 如果 pytorch_dump_folder_path 不是目录，则记录错误并返回
            if not os.path.isdir(pytorch_dump_folder_path):
                logger.error("--pytorch_dump_folder_path ({}) should be a directory".format(pytorch_dump_folder_path))
                return
            # 创建目录 pytorch_dump_folder_path，如果不存在的话
            os.makedirs(pytorch_dump_folder_path, exist_ok=True)
            # 将目标字典的索引保存到 vocab_path 指定的 JSON 文件中
            with open(vocab_path, "w", encoding="utf-8") as vocab_handle:
                json.dump(target_dict.indices, vocab_handle)

            # 使用 Wav2Vec2CTCTokenizer 初始化 tokenizer
            tokenizer = Wav2Vec2CTCTokenizer(
                vocab_path,
                unk_token=target_dict.unk_word,
                pad_token=target_dict.pad_word,
                bos_token=target_dict.bos_word,
                eos_token=target_dict.eos_word,
                word_delimiter_token="|",
                do_lower_case=False,
            )

            # 根据 config 中的 feat_extract_norm 设置 return_attention_mask
            return_attention_mask = True if config.feat_extract_norm == "layer" else False

            # 使用给定参数初始化 Wav2Vec2FeatureExtractor
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0,
                do_normalize=True,
                return_attention_mask=return_attention_mask,
            )

            # 使用 feature_extractor 和 tokenizer 初始化 Wav2Vec2Processor
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

            # 将 processor 的预训练模型保存到 pytorch_dump_folder_path 中
            processor.save_pretrained(pytorch_dump_folder_path)

        # 如果模型是微调过的，则创建 HubertForCTC 模型
        hf_wav2vec = HubertForCTC(config)
    else:
        # 如果模型不是微调过的，则创建 HubertModel 模型
        hf_wav2vec = HubertModel(config)

    # 加载模型的权重并设置为评估模式
    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])

    # 获取模型集合中的第一个模型并设置为评估模式
    model = model[0].eval()

    # 递归加载模型的权重到 hf_wav2vec 模型中
    recursively_load_weights(model, hf_wav2vec, is_finetuned)

    # 将 hf_wav2vec 的预训练模型保存到 pytorch_dump_folder_path 中
    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本被作为主程序运行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数：指向输出 PyTorch 模型的路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加命令行参数：指向 fairseq 检查点的路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    # 添加命令行参数：指向微调模型的字典路径
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    # 添加命令行参数：指向需要转换的模型的 hf config.json 的路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加命令行参数：是否为微调模型的标志，如果存在则设置为 True
    parser.add_argument(
        "--not_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    # 解析命令行参数，并将其存储在 args 对象中
    args = parser.parse_args()
    
    # 调用 convert_hubert_checkpoint 函数，传递命令行参数中的相关路径和标志
    convert_hubert_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path, not args.not_finetuned
    )
```