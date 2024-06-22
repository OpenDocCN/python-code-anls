# `.\models\hubert\convert_distilhubert_original_s3prl_checkpoint_to_pytorch.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""Convert Hubert checkpoint."""

# 导入必要的库
import argparse
import torch
from s3prl.hub import distilhubert
from transformers import HubertConfig, HubertModel, Wav2Vec2FeatureExtractor, logging

# 设置日志级别为信息
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义映射关系，用于将 Fairseq 模型的权重映射到 Hugging Face 模型
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
def recursively_load_weights(fairseq_model, hf_model):
    unused_weights = []
    fairseq_dict = fairseq_model.state_dict()

    feature_extractor = hf_model.feature_extractor
    # 遍历 fairseq_dict 字典中的键值对
    for name, value in fairseq_dict.items():
        # 初始化标志变量，用于标记是否使用了当前权重
        is_used = False
        # 如果键名中包含 "conv_layers"
        if "conv_layers" in name:
            # 调用 load_conv_layer 函数加载卷积层权重
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            # 标记当前权重已被使用
            is_used = True
        else:
            # 遍历 MAPPING 字典中的键值对
            for key, mapped_key in MAPPING.items():
                # 重置 mapped_key 变量
                mapped_key = mapped_key

                # 如果键名中包含 MAPPING 字典中的键
                if key in name:
                    # 标记当前权重已被使用
                    is_used = True
                    # 如果 mapped_key 中包含 "*", 替换为当前层索引
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    # 根据权重名称确定权重类型
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
                    # 递归设置权重到 hf_model 中
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                # 继续下一次循环
                continue
        # 如果当前权重未被使用，则添加到未使用权重列表中
        if not is_used:
            unused_weights.append(name)

    # 输出未使用的权重列表
    logger.warning(f"Unused weights: {unused_weights}")
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 从完整名称中提取层和类型信息
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    # 处理卷积层的权重和偏置
    if type_id == 0:
        if "bias" in name:
            # 检查偏置值的形状是否匹配，并更新特征提取器的偏置值
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            # 检查权重值的形状是否匹配，并更新特征提取器的权重值
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 处理层归一化的权重和偏置
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        if "bias" in name:
            # 检查偏置值的形状是否匹配，并更新特征提取器的层归一化偏置值
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape, (
                f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            # 检查权重值的形状是否匹配，并更新特征提取器的层归一化权重值
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        # 将未使用的权重名称添加到未使用的权重列表中
        unused_weights.append(full_name)


def convert_config(model):
    # 创建一个新的 Hubert 配置对象
    config = HubertConfig()
    fs_config = model.config

    # 将模型的配置信息转换为 Hubert 配置信息
    config.activation_dropout = fs_config.activation_dropout
    config.apply_spec_augment = False
    config.attention_dropout = fs_config.attention_dropout
    config.conv_bias = False
    conv_layers = eval(fs_config.extractor_conv_feature_layers)
    config.conv_dim = [x[0] for x in conv_layers]
    config.conv_kernel = [x[1] for x in conv_layers]
    config.conv_stride = [x[2] for x in conv_layers]
    config.feat_extract_activation = "gelu"
    config.feat_extract_norm = "layer" if fs_config.extractor_mode == "layer_norm" else "group"
    config.feat_proj_layer_norm = False
    # 设置特征投影层的dropout率为0.0
    config.feat_proj_dropout = 0.0
    # 设置最终输出层的dropout率为0.0
    config.final_dropout = 0.0
    # 设置隐藏层激活函数为fs_config中的激活函数
    config.hidden_act = fs_config.activation_fn
    # 设置隐藏层的dropout率为fs_config中的dropout率
    config.hidden_dropout = fs_config.dropout
    # 设置隐藏层的大小为fs_config中的编码器嵌入维度
    config.hidden_size = fs_config.encoder_embed_dim
    # 设置初始化范围为0.02
    config.initializer_range = 0.02
    # 设置中间层的大小为fs_config中的编码器前馈神经网络嵌入维度
    config.intermediate_size = fs_config.encoder_ffn_embed_dim
    # 设置层归一化的epsilon为1e-5
    config.layer_norm_eps = 1e-5
    # 设置层间dropout率为0.0
    config.layerdrop = 0.0
    # 设置注意力头的数量为fs_config中的编码器注意力头数
    config.num_attention_heads = fs_config.encoder_attention_heads
    # 设置卷积位置嵌入的组数为fs_config中的卷积位置组数
    config.num_conv_pos_embedding_groups = fs_config.conv_pos_groups
    # 设置卷积位置嵌入的数量为fs_config中的卷积位置数量
    config.num_conv_pos_embeddings = fs_config.conv_pos
    # 设置特征提取层的数量为卷积层的数量
    config.num_feat_extract_layers = len(conv_layers)
    # 设置隐藏层的数量为fs_config中的编码器层数
    config.num_hidden_layers = fs_config.encoder_layers

    # 返回配置对象
    return config
# 导入 torch 库中的 no_grad 装饰器
@torch.no_grad()
# 定义一个函数，用于将 Hubert 模型的权重转换为 transformers 模型的设计
def convert_hubert_checkpoint(pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 获取 DistilHubert 模型的权重
    model = distilhubert().model.model

    # 如果提供了配置文件路径，则使用该配置文件创建 HubertConfig 对象
    if config_path is not None:
        config = HubertConfig.from_pretrained(config_path)
    else:
        # 否则，根据模型创建配置对象
        config = convert_config(model)
    # 将模型设置为评估模式
    model = model.eval()

    # 创建 Wav2Vec2FeatureExtractor 对象
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0,
        do_normalize=False,
        return_attention_mask=False,
    )
    # 创建 HubertModel 对象
    hf_model = HubertModel(config)

    # 递归加载模型权重
    recursively_load_weights(model, hf_model)

    # 保存特征提取器的权重到指定路径
    feature_extractor.save_pretrained(pytorch_dump_folder_path)
    # 保存 Hubert 模型的权重到指定路径
    hf_model.save_pretrained(pytorch_dump_folder_path)


# 如果该脚本被直接运行
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加参数：输出 PyTorch 模型的路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加参数：转换模型所需的配置文件路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 解析命令行参数
    args = parser.parse_args()
    # 调用 convert_hubert_checkpoint 函数，传入参数路径
    convert_hubert_checkpoint(args.pytorch_dump_folder_path, args.config_path)
```