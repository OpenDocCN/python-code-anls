# `.\transformers\models\siglip\convert_siglip_to_hf.py`

```
# 设置文件编码为 UTF-8
# 版权声明，指明版权归属于 HuggingFace Inc. 团队
# 根据 Apache 许可证 2.0 版本，只有在遵守许可证的情况下才能使用此文件
# 可以在以下链接获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的，没有任何明示或暗示的担保或条件
# 请查看许可证以了解特定语言的权限和限制
# 转换来自原始存储库的 SigLIP 检查点
# URL: https://github.com/google-research/big_vision/tree/main

import argparse  # 导入解析命令行参数的模块
import collections  # 导入 collections 模块
from pathlib import Path  # 从 pathlib 模块导入 Path 类

import numpy as np  # 导入 NumPy 库并重命名为 np
import requests  # 导入 requests 库
import torch  # 导入 PyTorch 库
from huggingface_hub import hf_hub_download  # 从 huggingface_hub 模块导入 hf_hub_download 函数
from numpy import load  # 从 NumPy 导入 load 函数
from PIL import Image  # 从 PIL 库导入 Image 类

from transformers import SiglipConfig, SiglipImageProcessor, SiglipModel, SiglipProcessor, SiglipTokenizer  # 从 transformers 模块导入多个类
from transformers.utils import logging  # 从 transformers.utils 模块导入 logging 模块

logging.set_verbosity_info()  # 设置日志记录级别为 info
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

model_name_to_checkpoint = {  # 定义模型名称到检查点文件路径的映射
    # 基础检查点
    "siglip-base-patch16-224": "/Users/nielsrogge/Documents/SigLIP/webli_en_b16_224_63724782.npz",
    "siglip-base-patch16-256": "/Users/nielsrogge/Documents/SigLIP/webli_en_b16_256_60500360.npz",
    "siglip-base-patch16-384": "/Users/nielsrogge/Documents/SigLIP/webli_en_b16_384_68578854.npz",
    "siglip-base-patch16-512": "/Users/nielsrogge/Documents/SigLIP/webli_en_b16_512_68580893.npz",
    # 大型检查点
    "siglip-large-patch16-256": "/Users/nielsrogge/Documents/SigLIP/webli_en_l16_256_60552751.npz",
    "siglip-large-patch16-384": "/Users/nielsrogge/Documents/SigLIP/webli_en_l16_384_63634585.npz",
    # 多语言检查点
    "siglip-base-patch16-256-i18n": "/Users/nielsrogge/Documents/SigLIP/webli_i18n_b16_256_66117334.npz",
    # so400m 检查点
    "siglip-so400m-patch14-384": "/Users/nielsrogge/Documents/SigLIP/webli_en_so400m_384_58765454.npz",
}

model_name_to_image_size = {  # 定义模型名称到图像大小的映射
    "siglip-base-patch16-224": 224,
    "siglip-base-patch16-256": 256,
    "siglip-base-patch16-384": 384,
    "siglip-base-patch16-512": 512,
    "siglip-large-patch16-256": 256,
    "siglip-large-patch16-384": 384,
    "siglip-base-patch16-256-i18n": 256,
    "siglip-so400m-patch14-384": 384,
}

def get_siglip_config(model_name):  # 定义函数，根据模型名称获取 SigLIP 配置
    config = SiglipConfig()  # 创建 SigLIP 配置对象

    vocab_size = 250000 if "i18n" in model_name else 32000  # 如果模型名称中包含"i18n"，则词汇表大小为 250000，否则为 32000
    image_size = model_name_to_image_size[model_name]  # 获取模型名称对应的图像大小
    patch_size = 16 if "patch16" in model_name else 14  # 如果模型名称中包含"patch16"，则块大小为 16，否则为 14

    # 设置架构的大小
    config.vision_config.image_size = image_size  # 设置图像大小
    config.vision_config.patch_size = patch_size  # 设置块大小
    config.text_config.vocab_size = vocab_size  # 设置词汇表大小

    if "base" in model_name:  # 如果模型名称中包含"base"
        pass  # 什么也不做
    # 如果模型名称中包含"large"，则设置文本配置和视觉配置的参数为大型模型的数值
    elif "large" in model_name:
        config.text_config.hidden_size = 1024
        config.text_config.intermediate_size = 4096
        config.text_config.num_hidden_layers = 24
        config.text_config.num_attention_heads = 16
        config.vision_config.hidden_size = 1024
        config.vision_config.intermediate_size = 4096
        config.vision_config.num_hidden_layers = 24
        config.vision_config.num_attention_heads = 16
    # 如果模型名称中包含"so400m"，则设置文本配置和视觉配置的参数为so400m模型的数值
    elif "so400m" in model_name:
        config.text_config.hidden_size = 1152
        config.text_config.intermediate_size = 4304
        config.text_config.num_hidden_layers = 27
        config.text_config.num_attention_heads = 16
        config.vision_config.hidden_size = 1152
        config.vision_config.intermediate_size = 4304
        config.vision_config.num_hidden_layers = 27
        config.vision_config.num_attention_heads = 16
    # 如果模型名称不符合以上条件，则抛出数值错误
    else:
        raise ValueError("Model not supported")

    # 返回配置对象
    return config
def create_rename_keys(config):
    rename_keys = []
    # fmt: off

    # vision encoder

    # 将指定的参数重命名为新的键，并添加到重命名键列表中
    rename_keys.append(("params/img/embedding/kernel", "vision_model.embeddings.patch_embedding.weight"))
    rename_keys.append(("params/img/embedding/bias", "vision_model.embeddings.patch_embedding.bias"))
    rename_keys.append(("params/img/pos_embedding", "vision_model.embeddings.position_embedding.weight"))
    # 遍历视觉模型的隐藏层，为每一层的参数添加重命名键值对
    for i in range(config.vision_config.num_hidden_layers):
        # 添加重命名键值对，将参数路径映射到视觉模型编码器的第i层的LayerNorm_0的scale参数
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/LayerNorm_0/scale", f"vision_model.encoder.layers.{i}.layer_norm1.weight"))
        # 添加重命名键值对，将参数路径映射到视觉模型编码器的第i层的LayerNorm_0的bias参数
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/LayerNorm_0/bias", f"vision_model.encoder.layers.{i}.layer_norm1.bias"))
        # 添加重命名键值对，将参数路径映射到视觉模型编码器的第i层的LayerNorm_1的scale参数
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/LayerNorm_1/scale", f"vision_model.encoder.layers.{i}.layer_norm2.weight"))
        # 添加重命名键值对，将参数路径映射到视觉模型编码器的第i层的LayerNorm_1的bias参数
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/LayerNorm_1/bias", f"vision_model.encoder.layers.{i}.layer_norm2.bias"))
        # 添加重命名键值对，将参数路径映射到视觉模型编码器的第i层的MlpBlock_0的Dense_0的kernel参数
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_0/kernel", f"vision_model.encoder.layers.{i}.mlp.fc1.weight"))
        # 添加重命名键值对，将参数路径映射到视觉模型编码器的第i层的MlpBlock_0的Dense_0的bias参数
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_0/bias", f"vision_model.encoder.layers.{i}.mlp.fc1.bias"))
        # 添加重命名键值对，将参数路径映射到视觉模型编码器的第i层的MlpBlock_0的Dense_1的kernel参数
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_1/kernel", f"vision_model.encoder.layers.{i}.mlp.fc2.weight"))
        # 添加重命名键值对，将参数路径映射到视觉模型编码器的第i层的MlpBlock_0的Dense_1的bias参数
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_1/bias", f"vision_model.encoder.layers.{i}.mlp.fc2.bias"))
        # 添加重命名键值对，将参数路径映射到视觉模型编码器的第i层的MultiHeadDotProductAttention_0的key的kernel参数
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/key/kernel", f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight"))
        # 添加重命名键值对，将参数路径映射到视觉模型编码器的第i层���MultiHeadDotProductAttention_0的key的bias参数
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/key/bias", f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias"))
        # 添加重命名键值对，将参数路径映射到视觉模型编码器的第i层的MultiHeadDotProductAttention_0的value的kernel参数
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/value/kernel", f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight"))
        # 添加重命名键值对，将参数路径映射到视觉模型编码器的第i层的MultiHeadDotProductAttention_0的value的bias参数
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/value/bias", f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias"))
        # 添加重命名键值对，将参数路径映射到视觉模型编码器的第i层的MultiHeadDotProductAttention_0的query的kernel参数
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/query/kernel", f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight"))
        # 添加重命名键值对，将参数路径映射到视觉模型编码器的第i层的MultiHeadDotProductAttention_0的query的bias参数
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/query/bias", f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias"))
        # 添加重命名键值对，将参数路径映射到视觉模型编码器的第i层的MultiHeadDotProductAttention_0的out的kernel参数
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/out/kernel", f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"))
        # 添加重命名键值对，将参数路径映射到视觉模型编码器的第i层的MultiHeadDotProductAttention_0的out的bias参数
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/out/bias", f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias"))

    # 添加重命名键值对，将参数路径映射到视觉模型编码器的encoder_norm的scale参数
    rename_keys.append(("params/img/Transformer/encoder_norm/scale", "vision_model.post_layernorm.weight"))
    # 添加重命名键值对，将参数路径映射到视觉模型编码器的encoder_norm的bias参数
    rename_keys.append(("params/img/Transformer/encoder_norm/bias", "vision_model.post_layernorm.bias"))

    # 添加重命名键值对，将参数路径映射到视觉模型的MAPHead_0的probe参数
    rename_keys.append(("params/img/MAPHead_0/probe", "vision_model.head.probe"))
    # 将参数重命名并添加到重命名键列表中，用于模型参数加载
    rename_keys.append(("params/img/MAPHead_0/LayerNorm_0/scale", "vision_model.head.layernorm.weight"))
    rename_keys.append(("params/img/MAPHead_0/LayerNorm_0/bias", "vision_model.head.layernorm.bias"))
    rename_keys.append(("params/img/MAPHead_0/MlpBlock_0/Dense_0/kernel", "vision_model.head.mlp.fc1.weight"))
    rename_keys.append(("params/img/MAPHead_0/MlpBlock_0/Dense_0/bias", "vision_model.head.mlp.fc1.bias"))
    rename_keys.append(("params/img/MAPHead_0/MlpBlock_0/Dense_1/kernel", "vision_model.head.mlp.fc2.weight"))
    rename_keys.append(("params/img/MAPHead_0/MlpBlock_0/Dense_1/bias", "vision_model.head.mlp.fc2.bias"))
    rename_keys.append(("params/img/MAPHead_0/MultiHeadDotProductAttention_0/out/kernel", "vision_model.head.attention.out_proj.weight"))
    rename_keys.append(("params/img/MAPHead_0/MultiHeadDotProductAttention_0/out/bias", "vision_model.head.attention.out_proj.bias"))

    # 文本编码器

    # 将参数重命名并添加到重命名键列表中，用于模型参数加载
    rename_keys.append(("params/txt/Embed_0/embedding", "text_model.embeddings.token_embedding.weight"))
    rename_keys.append(("params/txt/pos_embedding", "text_model.embeddings.position_embedding.weight"))
    # 遍历文本模型的隐藏层，将参数重命名并添加到重命名键列表中
    for i in range(config.text_config.num_hidden_layers):
        # 将参数重命名并添加到重命名键列表中
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/LayerNorm_0/scale", f"text_model.encoder.layers.{i}.layer_norm1.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/LayerNorm_0/bias", f"text_model.encoder.layers.{i}.layer_norm1.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/LayerNorm_1/scale", f"text_model.encoder.layers.{i}.layer_norm2.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/LayerNorm_1/bias", f"text_model.encoder.layers.{i}.layer_norm2.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MlpBlock_0/Dense_0/kernel", f"text_model.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MlpBlock_0/Dense_0/bias", f"text_model.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MlpBlock_0/Dense_1/kernel", f"text_model.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MlpBlock_0/Dense_1/bias", f"text_model.encoder.layers.{i}.mlp.fc2.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/key/kernel", f"text_model.encoder.layers.{i}.self_attn.k_proj.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/key/bias", f"text_model.encoder.layers.{i}.self_attn.k_proj.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/value/kernel", f"text_model.encoder.layers.{i}.self_attn.v_proj.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/value/bias", f"text_model.encoder.layers.{i}.self_attn.v_proj.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/query/kernel", f"text_model.encoder.layers.{i}.self_attn.q_proj.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/query/bias", f"text_model.encoder.layers.{i}.self_attn.q_proj.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/out/kernel", f"text_model.encoder.layers.{i}.self_attn.out_proj.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/out/bias", f"text_model.encoder.layers.{i}.self_attn.out_proj.bias"))

    # 将最终层的参数重命名并添加到重命名键列表中
    rename_keys.append(("params/txt/Encoder_0/encoder_norm/scale", "text_model.final_layer_norm.weight"))
    rename_keys.append(("params/txt/Encoder_0/encoder_norm/bias", "text_model.final_layer_norm.bias"))
    rename_keys.append(("params/txt/head/kernel", "text_model.head.weight"))
    rename_keys.append(("params/txt/head/bias", "text_model.head.bias"))

    # 学习的温度和偏差
    # 将("params/t", "logit_scale")添加到重命名键列表中
    rename_keys.append(("params/t", "logit_scale"))
    # 将("params/b", "logit_bias")添加到重命名键列表中
    rename_keys.append(("params/b", "logit_bias"))

    # 格式化代码，结束fmt: off区块
    # fmt: on
    # 返回重命名键列表
    return rename_keys
# 重命名字典中的键，并根据配置对值进行相应的处理
def rename_key(dct, old, new, config):
    # 弹出旧键对应的值
    val = dct.pop(old)

    # 根据新键的特征和配置对值进行不同的reshape操作
    if ("out_proj" in new or "v_proj" in new or "k_proj" in new or "q_proj" in new) and "vision" in new:
        val = val.reshape(-1, config.vision_config.hidden_size)
    if ("out_proj" in new or "v_proj" in new or "k_proj" in new or "q_proj" in new) and "text" in new:
        val = val.reshape(-1, config.text_config.hidden_size)

    # 根据新键的特征对值进行不同的transpose操作
    if "patch_embedding.weight" in new:
        val = val.transpose(3, 2, 0, 1)
    elif new.endswith("weight") and "position_embedding" not in new and "token_embedding" not in new:
        val = val.T

    # 根据新键的特征和配置对值进行不同的reshape操作
    if "position_embedding" in new and "vision" in new:
        val = val.reshape(-1, config.vision_config.hidden_size)
    if "position_embedding" in new and "text" in new:
        val = val.reshape(-1, config.text_config.hidden_size)

    # 根据新键的特征对值进行reshape操作
    if new.endswith("bias"):
        val = val.reshape(-1)

    # 将处理后的值添加到字典中
    dct[new] = torch.from_numpy(val)


# 读取输入投影层的权重和偏置，并将它们添加到状态字典中
def read_in_q_k_v_head(state_dict, config):
    # 读取并处理键、值、查询的投影层权重和偏置
    key_proj_weight = (
        state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/key/kernel")
        .reshape(-1, config.vision_config.hidden_size)
        .T
    )
    key_proj_bias = state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/key/bias").reshape(-1)
    value_proj_weight = (
        state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/value/kernel")
        .reshape(-1, config.vision_config.hidden_size)
        .T
    )
    value_proj_bias = state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/value/bias").reshape(-1)
    query_proj_weight = (
        state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/query/kernel")
        .reshape(-1, config.vision_config.hidden_size)
        .T
    )
    query_proj_bias = state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/query/bias").reshape(-1)

    # 将处理后的权重和偏置合并，并添加到状态字典中
    state_dict["vision_model.head.attention.in_proj_weight"] = torch.from_numpy(
        np.concatenate([query_proj_weight, key_proj_weight, value_proj_weight], axis=0)
    )
    state_dict["vision_model.head.attention.in_proj_bias"] = torch.from_numpy(
        np.concatenate([query_proj_bias, key_proj_bias, value_proj_bias], axis=0)
    )


# 准备一张可爱猫咪的图片
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 从URL获取图片并返回
    image = Image.open(requests.get(url, stream=True).raw)
    return image


# 将嵌套字典展平为一级字典
def flatten_nested_dict(params, parent_key="", sep="/"):
    items = []

    # 遍历字典中的键值对
    for k, v in params.items():
        new_key = parent_key + sep + k if parent_key else k

        # 如果值是字典，则递归展平
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# 禁用梯度计算
@torch.no_grad()
def convert_siglip_checkpoint(model_name, pytorch_dump_folder_path, verify_logits=True, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our SigLIP structure.
    """

    # 定义默认的 SigLIP 配置
    config = get_siglip_config(model_name)

    # 获取检查点
    checkpoint = model_name_to_checkpoint[model_name]

    # 获取词汇文件
    if "i18n" in model_name:
        vocab_file = "/Users/nielsrogge/Documents/SigLIP/multilingual_vocab/sentencepiece.model"
    else:
        vocab_file = "/Users/nielsrogge/Documents/SigLIP/english_vocab/sentencepiece.model"

    # 加载原始状态字典
    data = load(checkpoint)
    state_dict = flatten_nested_dict(data)

    # 删除和重命名一些键
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest, config)

    # 注意力池化头的 qkv 矩阵需要特殊处理
    read_in_q_k_v_head(state_dict, config)

    # 加载 HuggingFace 模型
    model = SiglipModel(config).eval()
    model.load_state_dict(state_dict)

    # 创建处理器
    # 重要: 使令牌化器不返回 attention_mask，因为原始的不需要
    image_size = config.vision_config.image_size
    size = {"height": image_size, "width": image_size}
    image_processor = SiglipImageProcessor(size=size)
    tokenizer = SiglipTokenizer(vocab_file=vocab_file, model_input_names=["input_ids"])
    processor = SiglipProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # 在虚拟图片和文本上进行验证
    url_1 = "https://cdn.openai.com/multimodal-neurons/assets/apple/apple-ipod.jpg"
    image_1 = Image.open(requests.get(url_1, stream=True).raw).convert("RGB")
    url_2 = "https://cdn.openai.com/multimodal-neurons/assets/apple/apple-blank.jpg"
    image_2 = Image.open(requests.get(url_2, stream=True).raw).convert("RGB")
    texts = ["an apple", "a picture of an apple"]

    inputs = processor(images=[image_1, image_2], text=texts, return_tensors="pt", padding="max_length")

    # 针对原始值验证 input_ids
    if image_size == 224:
        filename = "siglip_pixel_values.pt"
    elif image_size == 256:
        filename = "siglip_pixel_values_256.pt"
    elif image_size == 384:
        filename = "siglip_pixel_values_384.pt"
    elif image_size == 512:
        filename = "siglip_pixel_values_512.pt"
    else:
        raise ValueError("Image size not supported")

    filepath = hf_hub_download(repo_id="nielsr/test-image", filename=filename, repo_type="dataset")
    original_pixel_values = torch.load(filepath)
    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="siglip_input_ids.pt", repo_type="dataset")
    original_input_ids = torch.load(filepath)

    if "i18n" not in model_name:
        assert inputs.input_ids.tolist() == original_input_ids.tolist()

    print("Mean of original pixel values:", original_pixel_values.mean())
    # 打印新像素值的平均值
    print("Mean of new pixel values:", inputs.pixel_values.mean())

    # 注意：这里使用原始像素值进行测试，因为我们没有确切的像素值
    # 使用 torch.no_grad() 来禁用梯度计算
    with torch.no_grad():
        # 使用模型进行推断，传入输入文本的 IDs 和原始像素值
        outputs = model(input_ids=inputs.input_ids, pixel_values=original_pixel_values)

    # 打印输出 logits 的前三行三列
    print(outputs.logits_per_image[:3, :3])

    # 计算每个像素值对应的概率
    probs = torch.sigmoid(outputs.logits_per_image)  # 这些是概率值
    # 打印第一张图片是文本0的概率
    print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
    # 打印第一张图片是文本1的概率
    print(f"{probs[0][1]:.1%} that image 0 is '{texts[1]}'")

    # 如果需要验证 logits
    if verify_logits:
        # 根据模型名称设置预期的 slice
        if model_name == "siglip-base-patch16-224":
            expected_slice = torch.tensor(
                [[-2.9621, -2.1672], [-0.2713, 0.2910]],
            )
        elif model_name == "siglip-base-patch16-256":
            expected_slice = torch.tensor(
                [[-3.1146, -1.9894], [-0.7312, 0.6387]],
            )
        elif model_name == "siglip-base-patch16-384":
            expected_slice = torch.tensor(
                [[-2.8098, -2.1891], [-0.4242, 0.4102]],
            )
        elif model_name == "siglip-base-patch16-512":
            expected_slice = torch.tensor(
                [[-2.7899, -2.2668], [-0.4295, -0.0735]],
            )
        elif model_name == "siglip-large-patch16-256":
            expected_slice = torch.tensor(
                [[-1.5827, -0.5801], [-0.9153, 0.1363]],
            )
        elif model_name == "siglip-large-patch16-384":
            expected_slice = torch.tensor(
                [[-2.1523, -0.2899], [-0.2959, 0.7884]],
            )
        elif model_name == "siglip-so400m-patch14-384":
            expected_slice = torch.tensor([[-1.2441, -0.6649], [-0.7060, 0.7374]])
        elif model_name == "siglip-base-patch16-256-i18n":
            expected_slice = torch.tensor(
                [[-0.9064, 0.1073], [-0.0299, 0.5304]],
            )

        # 使用 assert 检查 logits 是否与预期的 slice 接近
        assert torch.allclose(outputs.logits_per_image[:3, :3], expected_slice, atol=1e-4)
        print("Looks ok!")

    # 如果指定了 pytorch_dump_folder_path
    if pytorch_dump_folder_path is not None:
        # 创建目录（如果不存在）
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 打印正在保存模型的消息
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        # 保存模型到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 打印正在保存 processor 的消息
        print(f"Saving processor to {pytorch_dump_folder_path}")
        # 保存 processor 到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub
    if push_to_hub:
        # 将模型推送到 Hub
        model.push_to_hub(f"nielsr/{model_name}")
        # 将 processor 推送到 Hub
        processor.push_to_hub(f"nielsr/{model_name}")
# 如果脚本被直接执行（而不是被导入到其他脚本中），则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--model_name",  # 模型名称参数
        default="siglip-base-patch16-224",  # 默认模型名称
        type=str,  # 参数类型为字符串
        choices=model_name_to_checkpoint.keys(),  # 可选的模型名称列表
        help="Name of the model you'd like to convert.",  # 帮助信息
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",  # PyTorch模型输出目录路径参数
        default=None,  # 默认为None
        type=str,  # 参数类型为字符串
        help="Path to the output PyTorch model directory."  # 帮助信息
    )
    parser.add_argument(
        "--verify_logits",  # 验证logits参数
        action="store_false",  # 设置为False时执行动作
        help="Whether to verify logits against the original implementation."  # 帮助信息
    )
    parser.add_argument(
        "--push_to_hub",  # 推送到hub参数
        action="store_true",  # 设置为True时执行动作
        help="Whether or not to push the converted model to the 🤗 hub."  # 帮助信息
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数以转换模型
    convert_siglip_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.verify_logits, args.push_to_hub)
```