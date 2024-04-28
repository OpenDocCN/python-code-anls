# `.\models\efficientnet\convert_efficientnet_to_pytorch.py`

```
# 设置文件编码为 UTF-8
# 版权声明：2023 年由 HuggingFace Inc. 团队所有
# 根据 Apache 许可证版本 2.0 授权使用此文件；
# 除非符合许可证规定，否则您不能使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件根据“原样”分发，
# 没有任何形式的明示或暗示担保或条件。
# 有关详细信息，请参阅许可证。
"""从原始存储库中转换 EfficientNet 检查点。

URL: https://github.com/keras-team/keras/blob/v2.11.0/keras/applications/efficientnet.py
"""

# 导入必要的库
import argparse  # 导入命令行参数解析器
import json  # 导入 JSON 格式数据的处理库
import os  # 导入处理文件和路径的库

import numpy as np  # 导入 NumPy 数组处理库
import PIL  # 导入 Python Imaging Library，用于图像处理
import requests  # 导入用于发送 HTTP 请求的库
import tensorflow.keras.applications.efficientnet as efficientnet  # 导入 TensorFlow 中的 EfficientNet 模型
import torch  # 导入 PyTorch 深度学习库
from huggingface_hub import hf_hub_download  # 导入从 Hugging Face Hub 下载模型的功能
from PIL import Image  # 从 PIL 库中导入 Image 模块，用于图像处理
from tensorflow.keras.preprocessing import image  # 从 TensorFlow 中的图像预处理模块中导入图像处理功能

from transformers import (  # 导入 Transformers 库
    EfficientNetConfig,  # 导入 EfficientNet 配置类
    EfficientNetForImageClassification,  # 导入用于图像分类的 EfficientNet 模型
    EfficientNetImageProcessor,  # 导入 EfficientNet 图像处理器
)
from transformers.utils import logging  # 导入 Transformers 日志记录功能

# 设置日志输出级别为 info
logging.set_verbosity_info()
# 获取记录器实例
logger = logging.get_logger(__name__)

# 定义 EfficientNet 模型类别映射
model_classes = {
    "b0": efficientnet.EfficientNetB0,  # 对应 EfficientNet B0 模型
    "b1": efficientnet.EfficientNetB1,  # 对应 EfficientNet B1 模型
    "b2": efficientnet.EfficientNetB2,  # 对应 EfficientNet B2 模型
    "b3": efficientnet.EfficientNetB3,  # 对应 EfficientNet B3 模型
    "b4": efficientnet.EfficientNetB4,  # 对应 EfficientNet B4 模型
    "b5": efficientnet.EfficientNetB5,  # 对应 EfficientNet B5 模型
    "b6": efficientnet.EfficientNetB6,  # 对应 EfficientNet B6 模型
    "b7": efficientnet.EfficientNetB7,  # 对应 EfficientNet B7 模型
}

# 定义 EfficientNet 模型配置映射
CONFIG_MAP = {
    "b0": {  # EfficientNet B0 配置
        "hidden_dim": 1280,  # 隐藏层维度
        "width_coef": 1.0,  # 宽度系数
        "depth_coef": 1.0,  # 深度系数
        "image_size": 224,  # 图像大小
        "dropout_rate": 0.2,  # Dropout 率
        "dw_padding": [],  # 深度卷积填充
    },
    "b1": {  # EfficientNet B1 配置
        "hidden_dim": 1280,  # 隐藏层维度
        "width_coef": 1.0,  # 宽度系数
        "depth_coef": 1.1,  # 深度系数
        "image_size": 240,  # 图像大小
        "dropout_rate": 0.2,  # Dropout 率
        "dw_padding": [16],  # 深度卷积填充
    },
    "b2": {  # EfficientNet B2 配置
        "hidden_dim": 1408,  # 隐藏层维度
        "width_coef": 1.1,  # 宽度系数
        "depth_coef": 1.2,  # 深度系数
        "image_size": 260,  # 图像大小
        "dropout_rate": 0.3,  # Dropout 率
        "dw_padding": [5, 8, 16],  # 深度卷积填充
    },
    "b3": {  # EfficientNet B3 配置
        "hidden_dim": 1536,  # 隐藏层维度
        "width_coef": 1.2,  # 宽度系数
        "depth_coef": 1.4,  # 深度系数
        "image_size": 300,  # 图像大小
        "dropout_rate": 0.3,  # Dropout 率
        "dw_padding": [5, 18],  # 深度卷积填充
    },
    "b4": {  # EfficientNet B4 配置
        "hidden_dim": 1792,  # 隐藏层维度
        "width_coef": 1.4,  # 宽度系数
        "depth_coef": 1.8,  # 深度系数
        "image_size": 380,  # 图像大小
        "dropout_rate": 0.4,  # Dropout 率
        "dw_padding": [6],  # 深度卷积填充
    },
    "b5": {  # EfficientNet B5 配置
        "hidden_dim": 2048,  # 隐藏层维度
        "width_coef": 1.6,  # 宽度系数
        "depth_coef": 2.2
    # 创建一个字典，键为字符串 "b7"，值为一个包含模型参数的子字典
    "b7": {
        # 隐藏层维度设为2560
        "hidden_dim": 2560,
        # 宽度系数设为2.0，用于计算模型的宽度
        "width_coef": 2.0,
        # 深度系数设为3.1，用于计算模型的深度
        "depth_coef": 3.1,
        # 图像尺寸设为600，用于模型输入的图像大小
        "image_size": 600,
        # 随机失活率设为0.5，用于正则化训练过程中的随机失活
        "dropout_rate": 0.5,
        # 深度可分离卷积填充设为[18]，用于模型中的深度可分离卷积操作的填充参数
        "dw_padding": [18],
    },
# 定义一个函数，用于获取 EfficientNet 模型的配置信息
def get_efficientnet_config(model_name):
    # 创建一个 EfficientNetConfig 实例
    config = EfficientNetConfig()
    # 设置隐藏层维度
    config.hidden_dim = CONFIG_MAP[model_name]["hidden_dim"]
    # 设置宽度系数
    config.width_coefficient = CONFIG_MAP[model_name]["width_coef"]
    # 设置深度系数
    config.depth_coefficient = CONFIG_MAP[model_name]["depth_coef"]
    # 设置图像尺寸
    config.image_size = CONFIG_MAP[model_name]["image_size"]
    # 设置丢弃率
    config.dropout_rate = CONFIG_MAP[model_name]["dropout_rate"]
    # 设置深度可分离卷积填充方式
    config.depthwise_padding = CONFIG_MAP[model_name]["dw_padding"]

    # 设置仓库 ID 和文件名
    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    # 设置类别数量
    config.num_labels = 1000
    # 从 json 文件中加载类别信息
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    # 将类别 ID 和类别名称转换成字典形式
    id2label = {int(k): v for k, v in id2label.items()}

    # 设置类别 ID 到类别名称的映射信息
    config.id2label = id2label
    # 设置类别名称到类别 ID 的映射信息
    config.label2id = {v: k for k, v in id2label.items()}
    # 返回配置信息
    return config


# 准备测试图片
def prepare_img():
    # 图片链接地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 从链接地址中获取图片，并打开为 Image 对象
    im = Image.open(requests.get(url, stream=True).raw)
    return im


# 转换图片处理器
def convert_image_processor(model_name):
    # 获取图像大小
    size = CONFIG_MAP[model_name]["image_size"]
    # 创建 EfficientNetImageProcessor 实例
    preprocessor = EfficientNetImageProcessor(
        size={"height": size, "width": size},
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.47853944, 0.4732864, 0.47434163],
        do_center_crop=False,
    )
    return preprocessor


# 列出需要更名的所有键（左侧为原始名称，右侧为我们设定的名称）
def rename_keys(original_param_names):
    # 从原始参数名称中提取块名称
    block_names = [v.split("_")[0].split("block")[1] for v in original_param_names if v.startswith("block")]
    # 去重并排序块名称
    block_names = sorted(set(block_names))
    # 计算块数量
    num_blocks = len(block_names)
    # 构建块名称映射字典
    block_name_mapping = {b: str(i) for b, i in zip(block_names, range(num_blocks))}

    # 初始化需要更名的键列表
    rename_keys = []
    # 添加需要更名的键-值对
    rename_keys.append(("stem_conv/kernel:0", "embeddings.convolution.weight"))
    rename_keys.append(("stem_bn/gamma:0", "embeddings.batchnorm.weight"))
    rename_keys.append(("stem_bn/beta:0", "embeddings.batchnorm.bias"))
    rename_keys.append(("stem_bn/moving_mean:0", "embeddings.batchnorm.running_mean"))
    rename_keys.append(("stem_bn/moving_variance:0", "embeddings.batchnorm.running_var"))
    # 遍历每个块的名称列表
    for b in block_names:
        # 获取块名称的映射
        hf_b = block_name_mapping[b]
        
        # 更新键名列表，映射扩张卷积的权重和偏置
        rename_keys.append((f"block{b}_expand_conv/kernel:0", f"encoder.blocks.{hf_b}.expansion.expand_conv.weight"))
        rename_keys.append((f"block{b}_expand_bn/gamma:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.weight"))
        rename_keys.append((f"block{b}_expand_bn/beta:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.bias"))
        rename_keys.append(
            (f"block{b}_expand_bn/moving_mean:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.running_mean")
        )
        rename_keys.append(
            (f"block{b}_expand_bn/moving_variance:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.running_var")
        )
        
        # 更新键名列表，映射深度卷积的权重和批标准化参数
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
        
        # 更新键名列表，映射Squeeze-and-Excitation模块的权重和偏置
        rename_keys.append((f"block{b}_se_reduce/kernel:0", f"encoder.blocks.{hf_b}.squeeze_excite.reduce.weight"))
        rename_keys.append((f"block{b}_se_reduce/bias:0", f"encoder.blocks.{hf_b}.squeeze_excite.reduce.bias"))
        rename_keys.append((f"block{b}_se_expand/kernel:0", f"encoder.blocks.{hf_b}.squeeze_excite.expand.weight"))
        rename_keys.append((f"block{b}_se_expand/bias:0", f"encoder.blocks.{hf_b}.squeeze_excite.expand.bias"))
        
        # 更新键名列表，映射投影卷积的权重和批标准化参数
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

    # 更新键名列表，映射顶部卷积的权重和批标准化参数
    rename_keys.append(("top_conv/kernel:0", "encoder.top_conv.weight"))
    rename_keys.append(("top_bn/gamma:0", "encoder.top_bn.weight"))
    rename_keys.append(("top_bn/beta:0", "encoder.top_bn.bias"))
    rename_keys.append(("top_bn/moving_mean:0", "encoder.top_bn.running_mean"))
    rename_keys.append(("top_bn/moving_variance:0", "encoder.top_bn.running_var"))

    # 创建键映射字典
    key_mapping = {}
    # 遍历重命名键值对列表
    for item in rename_keys:
        # 如果重命名键在原始参数名列表中
        if item[0] in original_param_names:
            # 将重命名键和对应值添加到键映射字典中，值为"efficientnet." + item[1]
            key_mapping[item[0]] = "efficientnet." + item[1]

    # 添加额外的键值对到键映射字典中
    key_mapping["predictions/kernel:0"] = "classifier.weight"
    key_mapping["predictions/bias:0"] = "classifier.bias"
    # 返回键映射字典
    return key_mapping
# 替换 HF 模型参数为原始 TF 模型参数
def replace_params(hf_params, tf_params, key_mapping):
    # 遍历 TF 模型参数字典
    for key, value in tf_params.items():
        # 如果参数名中包含 "normalization"，则跳过
        if "normalization" in key:
            continue
        
        # 根据映射字典获取 HF 模型参数名
        hf_key = key_mapping[key]
        # 如果参数名包含 "_conv" 且包含 "kernel"
        if "_conv" in key and "kernel" in key:
            # 调整维度并将值转换为 Torch 张量
            new_hf_value = torch.from_numpy(value).permute(3, 2, 0, 1)
        # 如果参数名包含 "depthwise_kernel"
        elif "depthwise_kernel" in key:
            # 调整维度并将值转换为 Torch 张量
            new_hf_value = torch.from_numpy(value).permute(2, 3, 0, 1)
        # 如果参数名包含 "kernel"
        elif "kernel" in key:
            # 转置值并将其转换为 Torch 张量
            new_hf_value = torch.from_numpy(np.transpose(value))
        # 其它情况下将值转换为 Torch 张量
        else:
            new_hf_value = torch.from_numpy(value)

        # 使用新的 HF 模型参数替换原始参数，并进行形状断言
        assert hf_params[hf_key].shape == new_hf_value.shape
        hf_params[hf_key].copy_(new_hf_value)


# 不需要梯度的上下文管理器
@torch.no_grad()
def convert_efficientnet_checkpoint(model_name, pytorch_dump_folder_path, save_model, push_to_hub):
    """
    复制/粘贴/调整模型权重到我们的 EfficientNet 结构。
    """
    # 加载原始模型
    original_model = model_classes[model_name](
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )

    # 获取可训练参数和不可训练参数
    tf_params = original_model.trainable_variables
    tf_non_train_params = original_model.non_trainable_variables
    # 将所有参数的名称和值存储到字典
    tf_params = {param.name: param.numpy() for param in tf_params}
    for param in tf_non_train_params:
        tf_params[param.name] = param.numpy()
    tf_param_names = list(tf_params.keys())

    # 加载 HuggingFace 模型
    config = get_efficientnet_config(model_name)
    hf_model = EfficientNetForImageClassification(config).eval()
    hf_params = hf_model.state_dict()

    # 创建原始 TF 模型参数名到 HF 模型参数名的映射字典
    print("Converting parameters...")
    key_mapping = rename_keys(tf_param_names)
    replace_params(hf_params, tf_params, key_mapping)

    # 初始化图像预处理器并预处理输入图像
    preprocessor = convert_image_processor(model_name)
    inputs = preprocessor(images=prepare_img(), return_tensors="pt")

    # 进行 HF 模型推理
    hf_model.eval()
    with torch.no_grad():
        outputs = hf_model(**inputs)
    hf_logits = outputs.logits.detach().numpy()

    # 原始模型推理
    original_model.trainable = False
    image_size = CONFIG_MAP[model_name]["image_size"]
    img = prepare_img().resize((image_size, image_size), resample=PIL.Image.NEAREST)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    original_logits = original_model.predict(x)

    # 检查原始模型和 HF 模型输出是否匹配 -> 使用 np.allclose
    assert np.allclose(original_logits, hf_logits, atol=1e-3), "The predicted logits are not the same."
    print("Model outputs match!")
    # 如果需要保存模型
    if save_model:
        # 如果保存模型的文件夹不存在，则创建
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)
        # 保存转换后的模型和图像处理器
        hf_model.save_pretrained(pytorch_dump_folder_path)
        preprocessor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到hub
    if push_to_hub:
        # 打印提示信息
        print(f"Pushing converted {model_name} to the hub...")
        # 修改模型名称
        model_name = f"efficientnet-{model_name}"
        # 将图像处理器推送到hub
        preprocessor.push_to_hub(model_name)
        # 将模型推送到hub
        hf_model.push_to_hub(model_name)
# 如果当前模块是主程序，则执行以下代码
if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--model_name",
        default="b0",
        type=str,
        help="Version name of the EfficientNet model you want to convert, select from [b0, b1, b2, b3, b4, b5, b6, b7].",
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
    # 调用 convert_efficientnet_checkpoint 函数，传入命令行参数
    convert_efficientnet_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.save_model, args.push_to_hub)
```