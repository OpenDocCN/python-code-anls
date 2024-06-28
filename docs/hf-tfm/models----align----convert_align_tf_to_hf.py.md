# `.\models\align\convert_align_tf_to_hf.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，指明版权归 HuggingFace Inc. 团队所有，使用 Apache License, Version 2.0 许可
# 详细许可信息可在 http://www.apache.org/licenses/LICENSE-2.0 获取
# 根据适用法律和书面同意，在未经许可的情况下不得使用此文件
"""从原始存储库转换 ALIGN 检查点。"""

# 导入所需的库和模块
import argparse  # 用于解析命令行参数
import os  # 用于操作系统相关功能

import align  # 导入 align 模块
import numpy as np  # 用于数值计算
import requests  # 用于发出 HTTP 请求
import tensorflow as tf  # TensorFlow 深度学习框架
import torch  # PyTorch 深度学习框架
from PIL import Image  # Python Imaging Library，用于图像处理
from tokenizer import Tokenizer  # 导入自定义的 Tokenizer

# 导入 transformers 库的相关模块和函数
from transformers import (
    AlignConfig,  # ALIGN 模型的配置类
    AlignModel,  # ALIGN 模型类
    AlignProcessor,  # ALIGN 模型的处理器类
    BertConfig,  # BERT 模型的配置类
    BertTokenizer,  # BERT 模型的分词器类
    EfficientNetConfig,  # EfficientNet 模型的配置类
    EfficientNetImageProcessor,  # EfficientNet 图像处理器类
)

from transformers.utils import logging  # 导入 transformers 库的日志记录功能

# 设置日志记录的详细程度为 info
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


# 对输入图像进行预处理，将其调整大小并进行裁剪
def preprocess(image):
    image = tf.image.resize(image, (346, 346))  # 调整图像大小为 346x346 像素
    image = tf.image.crop_to_bounding_box(image, (346 - 289) // 2, (346 - 289) // 2, 289, 289)
    return image


# 获取 ALIGN 模型的配置
def get_align_config():
    # 使用预训练的 EfficientNet-B7 配置
    vision_config = EfficientNetConfig.from_pretrained("google/efficientnet-b7")
    vision_config.image_size = 289  # 设置图像输入大小为 289 像素
    vision_config.hidden_dim = 640  # 设置隐藏层维度为 640
    vision_config.id2label = {"0": "LABEL_0", "1": "LABEL_1"}  # 标签映射字典
    vision_config.label2id = {"LABEL_0": 0, "LABEL_1": 1}  # 反向标签映射字典
    vision_config.depthwise_padding = []  # 深度可分卷积填充方式为空列表

    text_config = BertConfig()  # 使用 BERT 配置
    # 根据文本和视觉配置创建 ALIGN 模型的配置对象，投影维度为 640
    config = AlignConfig.from_text_vision_configs(
        text_config=text_config, vision_config=vision_config, projection_dim=640
    )
    return config


# 准备图像数据，使用 COCO 数据集中的一张图像
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)  # 从 URL 打开图像文件
    return im


# 获取处理器，包括图像处理器和分词器
def get_processor():
    # 使用 EfficientNet 图像处理器进行图像预处理
    image_processor = EfficientNetImageProcessor(
        do_center_crop=True,
        rescale_factor=1 / 127.5,
        rescale_offset=True,
        do_normalize=False,
        include_top=False,
        resample=Image.BILINEAR,
    )
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")  # 使用 BERT 分词器
    tokenizer.model_max_length = 64  # 设置最大模型长度为 64
    # 创建 ALIGN 模型处理器，传入图像处理器和分词器
    processor = AlignProcessor(image_processor=image_processor, tokenizer=tokenizer)
    return processor


# 列出需要重命名的所有键（左边是原始名称，右边是我们的名称）
def rename_keys(original_param_names):
    # 获取 EfficientNet 图像编码器的块名称列表
    block_names = [v.split("_")[0].split("block")[1] for v in original_param_names if v.startswith("block")]
    block_names = list(set(block_names))  # 去重
    block_names = sorted(block_names)  # 排序
    num_blocks = len(block_names)  # 获取块的数量
    # 创建块名称到序号的映射字典
    block_name_mapping = {b: str(i) for b, i in zip(block_names, range(num_blocks))}
    # 创建一个空列表，用于存储需要重命名的键值对元组
    rename_keys = []
    # 添加元组到列表，将指定的模型权重名称映射到新的命名结构
    rename_keys.append(("stem_conv/kernel:0", "embeddings.convolution.weight"))
    rename_keys.append(("stem_bn/gamma:0", "embeddings.batchnorm.weight"))
    rename_keys.append(("stem_bn/beta:0", "embeddings.batchnorm.bias"))
    rename_keys.append(("stem_bn/moving_mean:0", "embeddings.batchnorm.running_mean"))
    rename_keys.append(("stem_bn/moving_variance:0", "embeddings.batchnorm.running_var"))

    # 遍历block_names列表中的每个元素，生成重命名后的键值对元组
    for b in block_names:
        hf_b = block_name_mapping[b]
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

    # 创建一个空字典，用于存储键名的映射关系
    key_mapping = {}
    # 遍历重命名键列表中的每个项
    for item in rename_keys:
        # 如果当前项的第一个元素存在于原始参数名列表中
        if item[0] in original_param_names:
            # 将原始参数名映射到新的键，以"vision_model." + 第二个元素作为值
            key_mapping[item[0]] = "vision_model." + item[1]

    # BERT 文本编码器的重命名列表初始化为空
    rename_keys = []
    # 定义旧模型路径
    old = "tf_bert_model/bert"
    # 定义新模型路径
    new = "text_model"
    # 添加重命名对，将旧路径中的特定参数映射到新路径的对应参数上
    rename_keys.append((f"{old}/embeddings/word_embeddings/weight:0", f"{new}.embeddings.word_embeddings.weight"))
    rename_keys.append((f"{old}/embeddings/position_embeddings/embeddings:0", f"{new}.embeddings.position_embeddings.weight"))
    rename_keys.append((f"{old}/embeddings/token_type_embeddings/embeddings:0", f"{new}.embeddings.token_type_embeddings.weight"))
    rename_keys.append((f"{old}/embeddings/LayerNorm/gamma:0", f"{new}.embeddings.LayerNorm.weight"))
    rename_keys.append((f"{old}/embeddings/LayerNorm/beta:0", f"{new}.embeddings.LayerNorm.bias"))

    rename_keys.append((f"{old}/pooler/dense/kernel:0", f"{new}.pooler.dense.weight"))
    rename_keys.append((f"{old}/pooler/dense/bias:0", f"{new}.pooler.dense.bias"))
    rename_keys.append(("dense/kernel:0", "text_projection.weight"))
    rename_keys.append(("dense/bias:0", "text_projection.bias"))
    rename_keys.append(("dense/bias:0", "text_projection.bias"))
    rename_keys.append(("temperature:0", "temperature"))

    # 遍历重命名键列表中的每个项
    for item in rename_keys:
        # 如果当前项的第一个元素存在于原始参数名列表中
        if item[0] in original_param_names:
            # 将原始参数名映射到新的键，以当前项的第二个元素作为值
            key_mapping[item[0]] = item[1]
    # 返回最终的键映射字典
    return key_mapping
# 加载原始模型和相关依赖
seq_length = 64
tok = Tokenizer(seq_length)
original_model = align.Align("efficientnet-b7", "bert-base", 640, seq_length, tok.get_vocab_size())
original_model.compile()
original_model.load_weights(checkpoint_path)

# 获取可训练和不可训练的 TensorFlow 参数
tf_params = original_model.trainable_variables
tf_non_train_params = original_model.non_trainable_variables
tf_params = {param.name: param.numpy() for param in tf_params}
for param in tf_non_train_params:
    tf_params[param.name] = param.numpy()
tf_param_names = list(tf_params.keys())

# 加载 HuggingFace 模型配置和状态字典
config = get_align_config()
hf_model = AlignModel(config).eval()
hf_params = hf_model.state_dict()

# 创建源到目标参数名称映射字典
print("Converting parameters...")
key_mapping = rename_keys(tf_param_names)
replace_params(hf_params, tf_params, key_mapping)

# 初始化处理器
processor = get_processor()

# 准备输入数据
inputs = processor(
    images=prepare_img(),
    text="A picture of a cat",
    padding="max_length",
    max_length=64,
    return_tensors="pt"
)

# 在 HuggingFace 模型上进行推理
hf_model.eval()
with torch.no_grad():
    outputs = hf_model(**inputs)

# 提取 HuggingFace 模型的图像和文本特征
hf_image_features = outputs.image_embeds.detach().numpy()
hf_text_features = outputs.text_embeds.detach().numpy()

# 在原始模型上进行推理
original_model.trainable = False
tf_image_processor = EfficientNetImageProcessor(
    do_center_crop=True,
    do_rescale=False,
    do_normalize=False,
    include_top=False,
    resample=Image.BILINEAR,
)
image = tf_image_processor(images=prepare_img(), return_tensors="tf", data_format="channels_last")["pixel_values"]
text = tok(tf.constant(["A picture of a cat"]))
    # 使用原始模型的图像编码器生成图像特征，设置为不训练状态
    image_features = original_model.image_encoder(image, training=False)
    # 使用原始模型的文本编码器生成文本特征，设置为不训练状态
    text_features = original_model.text_encoder(text, training=False)

    # 对图像特征进行 L2 归一化
    image_features = tf.nn.l2_normalize(image_features, axis=-1)
    # 对文本特征进行 L2 归一化
    text_features = tf.nn.l2_normalize(text_features, axis=-1)

    # 检查原始模型和HF模型的输出是否匹配，使用 np.allclose 函数进行比较，允许的最大误差为 1e-3
    if not np.allclose(image_features, hf_image_features, atol=1e-3):
        raise ValueError("The predicted image features are not the same.")
    if not np.allclose(text_features, hf_text_features, atol=1e-3):
        raise ValueError("The predicted text features are not the same.")
    # 输出匹配成功的消息
    print("Model outputs match!")

    if save_model:
        # 如果需要保存模型，创建保存模型的文件夹
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)
        # 将转换后的 HF 模型和处理器保存到指定路径
        hf_model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        # 如果需要推送到 Hub，打印推送信息
        print("Pushing converted ALIGN to the hub...")
        # 将模型和处理器推送到 Hub 上，使用 "align-base" 作为标识
        processor.push_to_hub("align-base")
        hf_model.push_to_hub("align-base")
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必需的参数
    parser.add_argument(
        "--checkpoint_path",
        default="./weights/model-weights",
        type=str,
        help="Path to the pretrained TF ALIGN checkpoint."
    )
    # 添加名为--checkpoint_path的参数，用于指定预训练的 TF ALIGN 检查点路径，默认为"./weights/model-weights"

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="hf_model",
        type=str,
        help="Path to the output PyTorch model directory."
    )
    # 添加名为--pytorch_dump_folder_path的参数，用于指定输出的 PyTorch 模型目录路径，默认为"hf_model"

    parser.add_argument("--save_model", action="store_true", help="Save model to local")
    # 添加一个标志参数--save_model，如果设置则表示要将模型保存到本地

    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image processor to the hub")
    # 添加一个标志参数--push_to_hub，如果设置则表示要将模型和图像处理器推送到Hub

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数convert_align_checkpoint，传递解析后的参数作为参数传递给函数
    convert_align_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.save_model, args.push_to_hub)
```