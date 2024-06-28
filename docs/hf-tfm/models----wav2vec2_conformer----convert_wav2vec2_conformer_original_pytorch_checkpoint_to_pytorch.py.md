# `.\models\wav2vec2_conformer\convert_wav2vec2_conformer_original_pytorch_checkpoint_to_pytorch.py`

```py
# 设置文件编码为 UTF-8

# 版权声明，这里声明代码版权属于 The HuggingFace Inc. 团队

# 导入 argparse 模块，用于命令行参数解析
import argparse

# 导入 json 模块，用于处理 JSON 数据
import json

# 导入 os 模块，提供与操作系统交互的功能
import os

# 导入 fairseq 库，用于序列到序列模型训练
import fairseq

# 导入 torch 库，PyTorch 深度学习框架
import torch

# 从 fairseq.data 模块中导入 Dictionary 类，用于管理词汇表
from fairseq.data import Dictionary

# 从 transformers 库中导入 Wav2Vec2 系列相关的类和函数
from transformers import (
    Wav2Vec2ConformerConfig,
    Wav2Vec2ConformerForCTC,
    Wav2Vec2ConformerForPreTraining,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    logging,
)

# 设置日志输出级别为 INFO
logging.set_verbosity_info()

# 获取当前模块的 logger
logger = logging.get_logger(__name__)

# 定义一个字典，用于将旧模型的参数映射到新模型的参数
MAPPING = {
    "post_extract_proj": "feature_projection.projection",
    "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",
    "self_attn.linear_k": "encoder.layers.*.self_attn.linear_k",
    "self_attn.linear_v": "encoder.layers.*.self_attn.linear_v",
    "self_attn.linear_q": "encoder.layers.*.self_attn.linear_q",
    "self_attn.pos_bias_u": "encoder.layers.*.self_attn.pos_bias_u",
    "self_attn.pos_bias_v": "encoder.layers.*.self_attn.pos_bias_v",
    "self_attn.linear_out": "encoder.layers.*.self_attn.linear_out",
    "self_attn.linear_pos": "encoder.layers.*.self_attn.linear_pos",
    "self_attn.rotary_emb": "encoder.embed_positions",
    "self_attn_layer_norm": "encoder.layers.*.self_attn_layer_norm",
    "conv_module.pointwise_conv1": "encoder.layers.*.conv_module.pointwise_conv1",
    "conv_module.pointwise_conv2": "encoder.layers.*.conv_module.pointwise_conv2",
    "conv_module.depthwise_conv": "encoder.layers.*.conv_module.depthwise_conv",
    "conv_module.batch_norm": "encoder.layers.*.conv_module.batch_norm",
    "conv_module.layer_norm": "encoder.layers.*.conv_module.layer_norm",
    "ffn1.w_1": "encoder.layers.*.ffn1.intermediate_dense",
    "ffn1.w_2": "encoder.layers.*.ffn1.output_dense",
    "ffn1.layer_norm": "encoder.layers.*.ffn1_layer_norm",
    "ffn2.w_1": "encoder.layers.*.ffn2.intermediate_dense",
    "ffn2.w_2": "encoder.layers.*.ffn2.output_dense",
    "ffn2.layer_norm": "encoder.layers.*.ffn2_layer_norm",
    "final_layer_norm": "encoder.layers.*.final_layer_norm",
    "encoder.layer_norm": "encoder.layer_norm",
    "w2v_model.layer_norm": "feature_projection.layer_norm",
    "quantizer.weight_proj": "quantizer.weight_proj",
    "quantizer.vars": "quantizer.codevectors",
    "project_q": "project_q",
    "final_proj": "project_hid",
    "w2v_encoder.proj": "lm_head",
    "mask_emb": "masked_spec_embed",
}

# 定义顶层键列表，用于保存需要转换的顶层模型参数
TOP_LEVEL_KEYS = [
    "lm_head",
    "quantizer.weight_proj",


这段代码是用于转换 Wav2Vec2Conformer 模型的检查点，通过映射旧模型参数到新模型参数来实现模型结构的更新和兼容性保证。
    "quantizer.codevectors",
    # 定义字符串"quantizer.codevectors"，用作后续操作的键值之一
    "project_q",
    # 定义字符串"project_q"，用作后续操作的键值之一
    "project_hid",
    # 定义字符串"project_hid"，用作后续操作的键值之一
# 递归设置模型参数的函数，根据指定的键（key）和值（value）设置深层次对象的属性值
def set_recursively(hf_pointer, key, value, full_name, weight_type):
    # 按照键（key）分割字符串，逐级获取深层次对象的属性指针（hf_pointer）
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    # 根据权重类型（weight_type）确定当前属性的形状（hf_shape）
    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    # 检查当前值（value）的形状是否与目标属性的形状（hf_shape）一致，否则抛出数值错误
    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    # 根据权重类型（weight_type）设置对应属性的数据值
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

    # 记录日志，显示成功初始化的属性路径和权重值来源
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")


# 递归加载Fairseq模型权重到Hugging Face模型中的函数
def recursively_load_weights(fairseq_model, hf_model, is_headless):
    # 初始化未使用的权重列表
    unused_weights = []
    # 获取Fairseq模型的状态字典
    fairseq_dict = fairseq_model.state_dict()

    # 获取Hugging Face模型中的特征提取器
    feature_extractor = hf_model.wav2vec2_conformer.feature_extractor
    # 遍历fairseq_dict中的每个键值对，其中键为权重的名称，值为对应的张量数值
    for name, value in fairseq_dict.items():
        # 初始化一个标志，表示当前权重是否被使用过
        is_used = False
        
        # 如果权重名称中包含"conv_layers"
        if "conv_layers" in name:
            # 调用load_conv_layer函数加载卷积层的权重，并传入相关参数
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            # 标记该权重已被使用
            is_used = True
        
        # 如果权重名称不包含"conv_layers"，进入else分支
        else:
            # 遍历MAPPING字典中的每个键值对
            for key, mapped_key in MAPPING.items():
                # 如果mapped_key不在TOP_LEVEL_KEYS中，则加上"wav2vec2_conformer."
                mapped_key = "wav2vec2_conformer." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key
                
                # 如果key在name中或者key去掉"w2v_model."后与name的第一个分段相同
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    # 标记该权重已被使用
                    is_used = True
                    
                    # 如果mapped_key中包含"*"
                    if "*" in mapped_key:
                        # 获取layer_index作为权重名称中key的前一个分段的倒数第二个元素
                        layer_index = name.split(key)[0].split(".")[-2]
                        # 将"*"替换为layer_index
                        mapped_key = mapped_key.replace("*", layer_index)
                    
                    # 根据权重名称中的特定字符串判断权重类型
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
                        # 对于名为"weight"的权重类型，可能需要进行后续处理，当前标记为"weight"
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
                    
                    # 调用set_recursively函数设置hf_model中mapped_key对应的值为value
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                
                # 继续下一次循环
                continue
        
        # 如果权重未被使用，则将其名称添加到unused_weights列表中
        if not is_used:
            unused_weights.append(name)
    
    # 输出警告日志，记录未使用的权重名称列表unused_weights
    logger.warning(f"Unused weights: {unused_weights}")
# 定义函数 load_conv_layer，加载卷积层的权重或偏置
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 从全名中提取层编号和类型编号
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    # 如果类型为0，表示处理卷积层的权重或偏置
    if type_id == 0:
        # 如果名称中包含 bias，更新卷积层的偏置值
        if "bias" in name:
            # 检查值的形状是否匹配目标卷积层的偏置数据形状
            if value.shape != feature_extractor.conv_layers[layer_id].conv.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 如果名称中包含 weight，更新卷积层的权重值
        elif "weight" in name:
            # 检查值的形状是否匹配目标卷积层的权重数据形状
            if value.shape != feature_extractor.conv_layers[layer_id].conv.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 如果类型为2，并且不使用 GroupNorm 或者是第一层并且使用 GroupNorm
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 如果名称中包含 bias，更新层归一化的偏置值
        if "bias" in name:
            # 检查值的形状是否匹配目标层归一化偏置数据形状
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 如果名称中包含 weight，更新层归一化的权重值
        elif "weight" in name:
            # 检查值的形状是否匹配目标层归一化权重数据形状
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    # 否则，将未使用的权重名称添加到 unused_weights 列表中
    else:
        unused_weights.append(full_name)


# 使用 torch.no_grad 装饰器定义函数 convert_wav2vec2_conformer_checkpoint，不计算梯度
def convert_wav2vec2_conformer_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 如果提供了 config_path，从预训练配置加载配置对象，使用 "swish" 作为隐藏层激活函数
    if config_path is not None:
        config = Wav2Vec2ConformerConfig.from_pretrained(config_path, hidden_act="swish")
    else:
        # 如果未指定配置文件，则使用默认配置
        config = Wav2Vec2ConformerConfig()

    if "rope" in checkpoint_path:
        # 如果模型路径中包含 "rope" 字符串，则设置位置编码类型为 "rotary"
        config.position_embeddings_type = "rotary"

    if is_finetuned:
        if dict_path:
            # 如果模型是在预训练基础上微调的，并且提供了字典路径，则加载目标字典
            target_dict = Dictionary.load(dict_path)

            # 重要变更：修改开始和填充令牌ID，因为CTC符号是 <pad> 而不是 <s>（与fairseq不同）
            config.bos_token_id = target_dict.pad_index
            config.pad_token_id = target_dict.bos_index
            config.eos_token_id = target_dict.eos_index
            config.vocab_size = len(target_dict.symbols)
            vocab_path = os.path.join(pytorch_dump_folder_path, "vocab.json")

            if not os.path.isdir(pytorch_dump_folder_path):
                # 如果指定的目录路径不是一个有效的目录，则记录错误并返回
                logger.error("--pytorch_dump_folder_path ({}) should be a directory".format(pytorch_dump_folder_path))
                return

            # 创建目录（如果不存在）
            os.makedirs(pytorch_dump_folder_path, exist_ok=True)
            vocab_dict = target_dict.indices

            # fairseq 中的 <pad> 和 <s> 被交换了
            vocab_dict["<pad>"] = 0
            vocab_dict["<s>"] = 1

            # 将字典写入到 JSON 文件中
            with open(vocab_path, "w", encoding="utf-8") as vocab_handle:
                json.dump(vocab_dict, vocab_handle)

            # 使用目标字典创建CTC tokenizer
            tokenizer = Wav2Vec2CTCTokenizer(
                vocab_path,
                unk_token=target_dict.unk_word,
                pad_token=target_dict.pad_word,
                bos_token=target_dict.bos_word,
                eos_token=target_dict.eos_word,
                word_delimiter_token="|",
                do_lower_case=False,
            )

            # 根据特征提取器是否返回注意力掩码，设置返回注意力掩码的标志
            return_attention_mask = True if config.feat_extract_norm == "layer" else False

            # 创建特征提取器
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0,
                do_normalize=True,
                return_attention_mask=return_attention_mask,
            )

            # 创建处理器，将特征提取器和tokenizer作为参数传入
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

            # 将处理器保存到指定路径
            processor.save_pretrained(pytorch_dump_folder_path)

        # 根据微调状态选择不同的模型
        hf_wav2vec = Wav2Vec2ConformerForCTC(config)
    else:
        # 如果未微调，则选择预训练模型
        hf_wav2vec = Wav2Vec2ConformerForPreTraining(config)

    # 根据微调状态加载模型
    if is_finetuned:
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1])}
        )
    else:
        task_arg = argparse.Namespace(task="audio_pretraining")
        task = fairseq.tasks.setup_task(task_arg)

        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path], task=task)

    # 将加载的模型设置为评估模式
    model = model[0].eval()

    # 递归地加载权重到模型中
    recursively_load_weights(model, hf_wav2vec, not is_finetuned)

    # 将预训练模型保存到指定路径
    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加参数：输出 PyTorch 模型的路径

    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    # 添加参数：fairseq 检查点的路径

    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    # 添加参数：微调模型的字典路径

    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加参数：要转换的模型的 hf config.json 路径

    parser.add_argument(
        "--not_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    # 添加参数：指示要转换的模型是否为微调模型的标志

    args = parser.parse_args()
    # 解析命令行参数并存储在 args 变量中

    convert_wav2vec2_conformer_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path, not args.not_finetuned
    )
    # 调用函数 convert_wav2vec2_conformer_checkpoint，传递解析后的参数作为函数的输入
```