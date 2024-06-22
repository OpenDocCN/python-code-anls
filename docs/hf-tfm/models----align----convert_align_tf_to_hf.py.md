# `.\transformers\models\align\convert_align_tf_to_hf.py`

```py
# 设置脚本编码为 UTF-8

# 版权声明和许可证信息

# 引入模块 argparse 用于命令行参数解析
import argparse
# 引入模块 os 用于操作系统相关功能
import os

# 引入 align 模块
import align
# 引入 numpy 库并使用别名 np
import numpy as np
# 引入 requests 库
import requests
# 引入 tensorflow 库并使用别名 tf
import tensorflow as tf
# 引入 torch 库
import torch
# 引入 PIL 库中的 Image 模块
from PIL import Image
# 从 tokenizer 模块中引入 Tokenizer 类
from tokenizer import Tokenizer

# 从 transformers 模块中引入以下类和函数
from transformers import (
    AlignConfig,  # 用于对齐模型的配置
    AlignModel,  # 对齐模型
    AlignProcessor,  # 对齐处理器
    BertConfig,  # BERT 模型的配置
    BertTokenizer,  # BERT 分词器
    EfficientNetConfig,  # EfficientNet 模型的配置
    EfficientNetImageProcessor,  # EfficientNet 图像处理器
)
# 从 transformers 模块中引入 logging 模块
from transformers.utils import logging

# 设置日志记录级别为 info
logging.set_verbosity_info()
# 获取 logger 对象
logger = logging.get_logger(__name__)


# 图像预处理函数
def preprocess(image):
    # 调整图像大小为 346x346
    image = tf.image.resize(image, (346, 346))
    # 裁剪图像为 289x289
    image = tf.image.crop_to_bounding_box(image, (346 - 289) // 2, (346 - 289) // 2, 289, 289)
    return image


# 获取对齐模型的配置
def get_align_config():
    # 从预训练模型加载 EfficientNet 的配置
    vision_config = EfficientNetConfig.from_pretrained("google/efficientnet-b7")
    # 修改图像尺寸为 289
    vision_config.image_size = 289
    # 修改隐藏层维度为 640
    vision_config.hidden_dim = 640
    # 标签到 ID 的映射
    vision_config.id2label = {"0": "LABEL_0", "1": "LABEL_1"}
    # ID 到标签的映射
    vision_config.label2id = {"LABEL_0": 0, "LABEL_1": 1}
    # 深度可分离卷积填充方式
    vision_config.depthwise_padding = []

    # 创建 BERT 模型配置
    text_config = BertConfig()
    # 从文本和视觉配置创建对齐模型配置
    config = AlignConfig.from_text_vision_configs(
        text_config=text_config, vision_config=vision_config, projection_dim=640
    )
    return config


# 获取图像数据的处理器
def get_processor():
    # 创建 EfficientNet 图像处理器
    image_processor = EfficientNetImageProcessor(
        do_center_crop=True,
        rescale_factor=1 / 127.5,
        rescale_offset=True,
        do_normalize=False,
        include_top=False,
        resample=Image.BILINEAR,
    )
    # 从预训练的 BERT 模型加载分词器
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # 设置分词器的最大模型长度为 64
    tokenizer.model_max_length = 64
    # 创建对齐处理器
    processor = AlignProcessor(image_processor=image_processor, tokenizer=tokenizer)
    return processor


# 列出需要重命名的所有键（左侧为原始名称，右侧为我们的名称）
def rename_keys(original_param_names):
    # 获取所有块的名称
    block_names = [v.split("_")[0].split("block")[1] for v in original_param_names if v.startswith("block")]
    # 去重和排序块名称
    block_names = list(set(block_names))
    block_names = sorted(block_names)
    # 计算块的数量
    num_blocks = len(block_names)
    # 创建块名称映射字典
    block_name_mapping = {b: str(i) for b, i in zip(block_names, range(num_blocks))}

    # 创建要重命名的键的列表
    rename_keys = []
    # 添加重命名键值对，将模型中的参数名映射到新的参数名
    rename_keys.append(("stem_conv/kernel:0", "embeddings.convolution.weight"))
    rename_keys.append(("stem_bn/gamma:0", "embeddings.batchnorm.weight"))
    rename_keys.append(("stem_bn/beta:0", "embeddings.batchnorm.bias"))
    rename_keys.append(("stem_bn/moving_mean:0", "embeddings.batchnorm.running_mean"))
    rename_keys.append(("stem_bn/moving_variance:0", "embeddings.batchnorm.running_var"))

    # 遍历每个块的名称
    for b in block_names:
        # 获取块名称的映射
        hf_b = block_name_mapping[b]
        # 添加重命名键值对，将模型中的参数名映射到新的参数名
        rename_keys.append((f"block{b}_expand_conv/kernel:0", f"encoder.blocks.{hf_b}.expansion.expand_conv.weight"))
        rename_keys.append((f"block{b}_expand_bn/gamma:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.weight"))
        rename_keys.append((f"block{b}_expand_bn/beta:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.bias"))
        rename_keys.append(
            (f"block{b}_expand_bn/moving_mean:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.running_mean")
        )
        rename_keys.append(
            (f"block{b}_expand_bn/moving_variance:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.running_var")
        )
        rename_keys.append(
            (f"block{b}_dwconv/depthwise_kernel:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_conv.weight")
        )
        rename_keys.append((f"block{b}_bn/gamma:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.weight"))
        rename_keys.append((f"block{b}_bn/beta:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.bias"))
        rename_keys.append(
            (f"block{b}_bn/moving_mean:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.running_mean")
        )
        rename_keys.append(
            (f"block{b}_bn/moving_variance:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.running_var")
        )

        rename_keys.append((f"block{b}_se_reduce/kernel:0", f"encoder.blocks.{hf_b}.squeeze_excite.reduce.weight"))
        rename_keys.append((f"block{b}_se_reduce/bias:0", f"encoder.blocks.{hf_b}.squeeze_excite.reduce.bias"))
        rename_keys.append((f"block{b}_se_expand/kernel:0", f"encoder.blocks.{hf_b}.squeeze_excite.expand.weight"))
        rename_keys.append((f"block{b}_se_expand/bias:0", f"encoder.blocks.{hf_b}.squeeze_excite.expand.bias"))
        rename_keys.append(
            (f"block{b}_project_conv/kernel:0", f"encoder.blocks.{hf_b}.projection.project_conv.weight")
        )
        rename_keys.append((f"block{b}_project_bn/gamma:0", f"encoder.blocks.{hf_b}.projection.project_bn.weight"))
        rename_keys.append((f"block{b}_project_bn/beta:0", f"encoder.blocks.{hf_b}.projection.project_bn.bias"))
        rename_keys.append(
            (f"block{b}_project_bn/moving_mean:0", f"encoder.blocks.{hf_b}.projection.project_bn.running_mean")
        )
        rename_keys.append(
            (f"block{b}_project_bn/moving_variance:0", f"encoder.blocks.{hf_b}.projection.project_bn.running_var")
        )

    # 创建空字典用于存储键值对映射
    key_mapping = {}
    # 遍历重命名键值对列表
    for item in rename_keys:
        # 如果原始参数名在原始参数名列表中
        if item[0] in original_param_names:
            # 将原始参数名映射到新的键名，加上特定的前缀
            key_mapping[item[0]] = "vision_model." + item[1]

    # BERT 文本编码器的重命名键值对列表
    rename_keys = []
    # 旧的参数名前缀
    old = "tf_bert_model/bert"
    # 新的参数名前缀
    new = "text_model"
    # 添加重命名键值对到列表中
    rename_keys.append((f"{old}/embeddings/word_embeddings/weight:0", f"{new}.embeddings.word_embeddings.weight"))
    rename_keys.append(
        (f"{old}/embeddings/position_embeddings/embeddings:0", f"{new}.embeddings.position_embeddings.weight")
    )
    rename_keys.append(
        (f"{old}/embeddings/token_type_embeddings/embeddings:0", f"{new}.embeddings.token_type_embeddings.weight")
    )
    rename_keys.append((f"{old}/embeddings/LayerNorm/gamma:0", f"{new}.embeddings.LayerNorm.weight"))
    rename_keys.append((f"{old}/embeddings/LayerNorm/beta:0", f"{new}.embeddings.LayerNorm.bias"))

    rename_keys.append((f"{old}/pooler/dense/kernel:0", f"{new}.pooler.dense.weight"))
    rename_keys.append((f"{old}/pooler/dense/bias:0", f"{new}.pooler.dense.bias"))
    rename_keys.append(("dense/kernel:0", "text_projection.weight"))
    rename_keys.append(("dense/bias:0", "text_projection.bias"))
    rename_keys.append(("dense/bias:0", "text_projection.bias"))
    rename_keys.append(("temperature:0", "temperature"))

    # 再次遍历重命名键值对列表
    for item in rename_keys:
        # 如果原始参数名在原始参数名列表中
        if item[0] in original_param_names:
            # 将原始参数名映射到新的键名
            key_mapping[item[0]] = item[1]
    # 返回参数名映射字典
    return key_mapping
# 定义一个函数，用于将 TensorFlow 模型的参数替换为 HuggingFace 模型的参数
def replace_params(hf_params, tf_params, key_mapping):
    # 获取 TensorFlow 模型参数的键列表，但未做实际操作
    list(hf_params.keys())

    # 遍历 TensorFlow 模型的参数字典
    for key, value in tf_params.items():
        # 如果当前参数不在键映射中，则跳过当前循环
        if key not in key_mapping:
            continue

        # 获取对应的 HuggingFace 模型参数键
        hf_key = key_mapping[key]

        # 根据参数键的特征进行不同的处理
        if "_conv" in key and "kernel" in key:
            # 如果参数键中包含 "_conv" 和 "kernel"，则将其转换为 PyTorch 张量并对维度进行置换
            new_hf_value = torch.from_numpy(value).permute(3, 2, 0, 1)
        elif "embeddings" in key:
            # 如果参数键中包含 "embeddings"，则直接将其转换为 PyTorch 张量
            new_hf_value = torch.from_numpy(value)
        elif "depthwise_kernel" in key:
            # 如果参数键中包含 "depthwise_kernel"，则将其转换为 PyTorch 张量并对维度进行置换
            new_hf_value = torch.from_numpy(value).permute(2, 3, 0, 1)
        elif "kernel" in key:
            # 如果参数键中包含 "kernel"，则将其转置后转换为 PyTorch 张量
            new_hf_value = torch.from_numpy(np.transpose(value))
        elif "temperature" in key:
            # 如果参数键中包含 "temperature"，则直接保留其值
            new_hf_value = value
        elif "bn/gamma" or "bn/beta" in key:
            # 如果参数键中包含 "bn/gamma" 或 "bn/beta"，则将其转置后转换为 PyTorch 张量并挤压到一维
            new_hf_value = torch.from_numpy(np.transpose(value)).squeeze()
        else:
            # 对于其他情况，直接将值转换为 PyTorch 张量
            new_hf_value = torch.from_numpy(value)

        # 将 HuggingFace 模型参数替换为原始 TensorFlow 模型参数
        hf_params[hf_key].copy_(new_hf_value)

# 带有无梯度的上下文管理器的函数，用于将检查点转换为 HuggingFace 模型的 ALIGN 结构
@torch.no_grad()
def convert_align_checkpoint(checkpoint_path, pytorch_dump_folder_path, save_model, push_to_hub):
    """
    Copy/paste/tweak model's weights to our ALIGN structure.
    """
    # 加载原始模型
    seq_length = 64
    tok = Tokenizer(seq_length)
    original_model = align.Align("efficientnet-b7", "bert-base", 640, seq_length, tok.get_vocab_size())
    original_model.compile()
    original_model.load_weights(checkpoint_path)

    # 获取原始模型的可训练和不可训练参数
    tf_params = original_model.trainable_variables
    tf_non_train_params = original_model.non_trainable_variables
    tf_params = {param.name: param.numpy() for param in tf_params}
    for param in tf_non_train_params:
        tf_params[param.name] = param.numpy()
    tf_param_names = list(tf_params.keys())

    # 加载 HuggingFace 模型
    config = get_align_config()
    hf_model = AlignModel(config).eval()
    hf_params = hf_model.state_dict()

    # 创建参数名称映射字典，用于将原始模型参数映射到 HuggingFace 模型参数
    print("Converting parameters...")
    key_mapping = rename_keys(tf_param_names)
    replace_params(hf_params, tf_params, key_mapping)

    # 初始化处理器
    processor = get_processor()
    # 准备输入数据
    inputs = processor(
        images=prepare_img(), text="A picture of a cat", padding="max_length", max_length=64, return_tensors="pt"
    )

    # 进行 HuggingFace 模型的推理
    hf_model.eval()
    with torch.no_grad():
        outputs = hf_model(**inputs)

    # 获取 HuggingFace 模型的图像和文本特征
    hf_image_features = outputs.image_embeds.detach().numpy()
    hf_text_features = outputs.text_embeds.detach().numpy()

    # 进行原始模型的推理
    original_model.trainable = False
    # 初始化 TensorFlow 图像处理器
    tf_image_processor = EfficientNetImageProcessor(
        do_center_crop=True,
        do_rescale=False,
        do_normalize=False,
        include_top=False,
        resample=Image.BILINEAR,
    )
    # 准备输入图像数据
    image = tf_image_processor(images=prepare_img(), return_tensors="tf", data_format="channels_last")["pixel_values"]
    # 对文本进行编码
    text = tok(tf.constant(["A picture of a cat"]))
    # 使用原始模型对图像进行编码，设置为不训练
    image_features = original_model.image_encoder(image, training=False)
    # 使用原始模型对文本进行编码，设置为不训练
    text_features = original_model.text_encoder(text, training=False)

    # 对图像特征进行 L2 归一化，沿着最后一个轴
    image_features = tf.nn.l2_normalize(image_features, axis=-1)
    # 对文本特征进行 L2 归一化，沿着最后一个轴
    text_features = tf.nn.l2_normalize(text_features, axis=-1)

    # 检查原始模型输出与 HF 模型输出是否匹配 -> 使用 np.allclose 函数
    if not np.allclose(image_features, hf_image_features, atol=1e-3):
        # 如果图像特征不匹配，则引发 ValueError
        raise ValueError("The predicted image features are not the same.")
    if not np.allclose(text_features, hf_text_features, atol=1e-3):
        # 如果文本特征不匹配，则引发 ValueError
        raise ValueError("The predicted text features are not the same.")
    # 输出提示信息，表示模型输出匹配
    print("Model outputs match!")

    if save_model:
        # 如果需要保存模型
        # 创建保存模型的文件夹
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)
        # 保存转换后的模型和图像处理器
        hf_model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        # 如果需要推送到 Hub
        # 输出提示信息，表示正在推送转换后的 ALIGN 模型到 Hub
        print("Pushing converted ALIGN to the hub...")
        # 将处理器推送到 Hub
        processor.push_to_hub("align-base")
        # 将 HF 模型推送到 Hub
        hf_model.push_to_hub("align-base")
# 如果脚本被直接执行而非被导入，则执行以下代码
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数
    parser.add_argument(
        "--checkpoint_path",
        default="./weights/model-weights",
        type=str,
        help="Path to the pretrained TF ALIGN checkpoint.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="hf_model",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--save_model", action="store_true", help="Save model to local")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image processor to the hub")

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将 TensorFlow ALIGN 模型转换为 PyTorch 模型
    convert_align_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.save_model, args.push_to_hub)
```