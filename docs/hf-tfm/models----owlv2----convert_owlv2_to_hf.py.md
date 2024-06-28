# `.\models\owlv2\convert_owlv2_to_hf.py`

```py
# 导入必要的模块和库

import argparse  # 导入命令行参数解析模块
import collections  # 导入collections模块，用于处理嵌套的字典
import os  # 导入操作系统相关的功能模块

import jax  # 导入JAX，用于自动求导和并行计算
import jax.numpy as jnp  # 导入JAX的NumPy接口，命名为jnp
import numpy as np  # 导入NumPy库，命名为np
import torch  # 导入PyTorch库
from flax.training import checkpoints  # 导入Flax的checkpoint模块，用于模型保存和加载
from huggingface_hub import hf_hub_download  # 导入Hugging Face Hub的下载函数
from PIL import Image  # 导入PIL库中的Image模块，用于图像处理

from transformers import (  # 从transformers库中导入多个类和函数
    CLIPTokenizer,  # CLIP模型的tokenizer
    Owlv2Config,  # Owlv2模型的配置类
    Owlv2ForObjectDetection,  # Owlv2模型的对象检测类
    Owlv2ImageProcessor,  # Owlv2模型的图像处理类
    Owlv2Processor,  # Owlv2模型的处理类
    Owlv2TextConfig,  # Owlv2模型的文本配置类
    Owlv2VisionConfig,  # Owlv2模型的视觉配置类
)
from transformers.utils import logging  # 导入transformers库中的logging模块，用于日志记录

# 设置日志记录的详细级别为INFO
logging.set_verbosity_info()

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


def get_owlv2_config(model_name):
    # 根据模型名称选择对应的配置参数
    if "large" in model_name:
        # 如果模型名称中包含"large"
        image_size = 1008  # 图像大小设为1008
        patch_size = 14  # 补丁大小设为14
        vision_hidden_size = 1024  # 视觉模型隐藏层大小设为1024
        vision_intermediate_size = 4096  # 视觉模型中间层大小设为4096
        vision_num_hidden_layers = 24  # 视觉模型隐藏层数设为24
        vision_num_attention_heads = 16  # 视觉模型注意力头数设为16
        projection_dim = 768  # 投影维度设为768
        text_hidden_size = 768  # 文本模型隐藏层大小设为768
        text_intermediate_size = 3072  # 文本模型中间层大小设为3072
        text_num_attention_heads = 12  # 文本模型注意力头数设为12
        text_num_hidden_layers = 12  # 文本模型隐藏层数设为12
    else:
        # 如果模型名称不包含"large"
        image_size = 960  # 图像大小设为960
        patch_size = 16  # 补丁大小设为16
        vision_hidden_size = 768  # 视觉模型隐藏层大小设为768
        vision_intermediate_size = 3072  # 视觉模型中间层大小设为3072
        vision_num_hidden_layers = 12  # 视觉模型隐藏层数设为12
        vision_num_attention_heads = 12  # 视觉模型注意力头数设为12
        projection_dim = 512  # 投影维度设为512
        text_hidden_size = 512  # 文本模型隐藏层大小设为512
        text_intermediate_size = 2048  # 文本模型中间层大小设为2048
        text_num_attention_heads = 8  # 文本模型注意力头数设为8
        text_num_hidden_layers = 12  # 文本模型隐藏层数设为12

    # 创建视觉配置对象
    vision_config = Owlv2VisionConfig(
        patch_size=patch_size,
        image_size=image_size,
        hidden_size=vision_hidden_size,
        num_hidden_layers=vision_num_hidden_layers,
        intermediate_size=vision_intermediate_size,
        num_attention_heads=vision_num_attention_heads,
    )

    # 创建文本配置对象
    text_config = Owlv2TextConfig(
        hidden_size=text_hidden_size,
        intermediate_size=text_intermediate_size,
        num_attention_heads=text_num_attention_heads,
        num_hidden_layers=text_num_hidden_layers,
    )

    # 创建总配置对象
    config = Owlv2Config(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        projection_dim=projection_dim,
    )

    return config


def flatten_nested_dict(params, parent_key="", sep="/"):
    items = []
    # 遍历字典 params 中的键值对
    for k, v in params.items():
        # 如果 parent_key 存在，则将当前键 k 与 parent_key 和分隔符 sep 拼接成新的键 new_key
        # 如果 parent_key 不存在，则直接使用当前键 k 作为新的键 new_key
        new_key = parent_key + sep + k if parent_key else k

        # 检查当前值 v 是否为可变映射（如字典）
        if isinstance(v, collections.MutableMapping):
            # 如果是可变映射，则递归展开其内部结构，并将展开后的结果的键值对添加到 items 列表中
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            # 如果不是可变映射，则将当前键值对作为元组 (new_key, v) 添加到 items 列表中
            items.append((new_key, v))
    
    # 将 items 列表转换为字典并返回
    return dict(items)
# 定义函数，用于创建重命名键列表，根据给定的配置和模型名称
def create_rename_keys(config, model_name):
    # 初始化空的重命名键列表
    rename_keys = []

    # fmt: off
    # CLIP vision encoder
    # 添加重命名键，将原始名称映射为新名称，用于视觉编码器的类嵌入
    rename_keys.append(("backbone/clip/visual/class_embedding", "owlv2.vision_model.embeddings.class_embedding"))
    # 添加重命名键，将原始名称映射为新名称，用于视觉编码器的补丁嵌入权重
    rename_keys.append(("backbone/clip/visual/conv1/kernel", "owlv2.vision_model.embeddings.patch_embedding.weight"))
    # 添加重命名键，将原始名称映射为新名称，用于视觉编码器的位置嵌入权重
    rename_keys.append(("backbone/clip/visual/positional_embedding", "owlv2.vision_model.embeddings.position_embedding.weight"))
    # 添加重命名键，将原始名称映射为新名称，用于视觉编码器的前层归一化权重
    rename_keys.append(("backbone/clip/visual/ln_pre/scale", "owlv2.vision_model.pre_layernorm.weight"))
    # 添加重命名键，将原始名称映射为新名称，用于视觉编码器的前层归一化偏置
    rename_keys.append(("backbone/clip/visual/ln_pre/bias", "owlv2.vision_model.pre_layernorm.bias"))

    # 添加重命名键，将原始名称映射为新名称，用于视觉编码器的后层归一化权重
    rename_keys.append(("backbone/clip/visual/ln_post/scale", "owlv2.vision_model.post_layernorm.weight"))
    # 添加重命名键，将原始名称映射为新名称，用于视觉编码器的后层归一化偏置
    rename_keys.append(("backbone/clip/visual/ln_post/bias", "owlv2.vision_model.post_layernorm.bias"))

    # CLIP text encoder
    # 添加重命名键，将原始名称映射为新名称，用于文本编码器的标记嵌入权重
    rename_keys.append(("backbone/clip/text/token_embedding/embedding", "owlv2.text_model.embeddings.token_embedding.weight"))
    # 添加重命名键，将原始名称映射为新名称，用于文本编码器的位置嵌入权重
    rename_keys.append(("backbone/clip/text/positional_embedding", "owlv2.text_model.embeddings.position_embedding.weight"))

    # 添加重命名键，将原始名称映射为新名称，用于文本编码器的最终层归一化权重
    rename_keys.append(("backbone/clip/text/ln_final/scale", "owlv2.text_model.final_layer_norm.weight"))
    # 添加重命名键，将原始名称映射为新名称，用于文本编码器的最终层归一化偏置
    rename_keys.append(("backbone/clip/text/ln_final/bias", "owlv2.text_model.final_layer_norm.bias"))

    # logit scale
    # 添加重命名键，将原始名称映射为新名称，用于逻辑刻度的权重
    rename_keys.append(("backbone/clip/logit_scale", "owlv2.logit_scale"))

    # projection heads
    # 添加重命名键，将原始名称映射为新名称，用于文本投影头的权重
    rename_keys.append(("backbone/clip/text/text_projection/kernel", "owlv2.text_projection.weight"))

    # class and box heads
    # 添加重命名键，将原始名称映射为新名称，用于合并类令牌的归一化层权重
    rename_keys.append(("backbone/merged_class_token/scale", "layer_norm.weight"))
    # 添加重命名键，将原始名称映射为新名称，用于合并类令牌的归一化层偏置
    rename_keys.append(("backbone/merged_class_token/bias", "layer_norm.bias"))
    # 添加重命名键，将原始名称映射为新名称，用于类头的第一个密集层的权重
    rename_keys.append(("class_head/Dense_0/kernel", "class_head.dense0.weight"))
    # 添加重命名键，将原始名称映射为新名称，用于类头的第一个密集层的偏置
    rename_keys.append(("class_head/Dense_0/bias", "class_head.dense0.bias"))
    # 添加重命名键，将原始名称映射为新名称，用于类头逻辑偏移的权重
    rename_keys.append(("class_head/logit_shift/kernel", "class_head.logit_shift.weight"))
    # 添加重命名键，将原始名称映射为新名称，用于类头逻辑刻度的权重
    rename_keys.append(("class_head/logit_scale/kernel", "class_head.logit_scale.weight"))
    # 添加重命名键，将原始名称映射为新名称，用于类头逻辑刻度的偏置
    rename_keys.append(("class_head/logit_scale/bias", "class_head.logit_scale.bias"))
    # 添加重命名键，将原始名称映射为新名称，用于类头逻辑偏移的偏置
    rename_keys.append(("class_head/logit_shift/bias", "class_head.logit_shift.bias"))
    # 添加重命名键，将原始名称映射为新名称，用于目标框头的第一个密集层的权重
    rename_keys.append(("obj_box_head/Dense_0/kernel", "box_head.dense0.weight"))
    # 添加重命名键，将原始名称映射为新名称，用于目标框头的第一个密集层的偏置
    rename_keys.append(("obj_box_head/Dense_0/bias", "box_head.dense0.bias"))
    # 添加重命名键，将原始名称映射为新名称，用于目标框头的第二个密集层的权重
    rename_keys.append(("obj_box_head/Dense_1/kernel", "box_head.dense1.weight"))
    # 添加重命名键，将原始名称映射为新名称，用于目标框头的第二个密集层的偏置
    rename_keys.append(("obj_box_head/Dense_1/bias", "box_head.dense1.bias"))
    # 添加重命名键，将原始名称映射为新名称，用于目标框头的第三个密集层的权重
    rename_keys.append(("obj_box_head/Dense_2/kernel", "box_head.dense2.weight"))
    # 添加重命名键，将原始名称映射为新名称，用于目标框头的第三个密集层的偏置
    rename_keys.append(("obj_box_head/Dense_2/bias", "box_head.dense2.bias"))
    # objectness head (only for v2)
    # 此处为 v2 特有的目标性头部（暂未提供具体的重命名信息）
    # 如果模型名称包含 "v2"，则执行以下操作
    if "v2" in model_name:
        # 将需要重命名的键值对添加到 rename_keys 列表中
        rename_keys.append(("objectness_head/Dense_0/kernel", "objectness_head.dense0.weight"))
        rename_keys.append(("objectness_head/Dense_0/bias", "objectness_head.dense0.bias"))
        rename_keys.append(("objectness_head/Dense_1/kernel", "objectness_head.dense1.weight"))
        rename_keys.append(("objectness_head/Dense_1/bias", "objectness_head.dense1.bias"))
        rename_keys.append(("objectness_head/Dense_2/kernel", "objectness_head.dense2.weight"))
        rename_keys.append(("objectness_head/Dense_2/bias", "objectness_head.dense2.bias"))

    # 格式化设置关闭，恢复默认的代码格式
    # fmt: on

    # 返回存储重命名键值对的列表 rename_keys
    return rename_keys
# 从字典中弹出旧键对应的值
val = dct.pop(old)

# 如果新键名包含特定字符串并且包含"vision"，则对值进行重新形状，调整为二维数组
if ("out_proj" in new or "v_proj" in new or "k_proj" in new or "q_proj" in new) and "vision" in new:
    val = val.reshape(-1, config.vision_config.hidden_size)
# 如果新键名包含特定字符串并且包含"text"，则对值进行重新形状，调整为二维数组
if ("out_proj" in new or "v_proj" in new or "k_proj" in new or "q_proj" in new) and "text" in new:
    val = val.reshape(-1, config.text_config.hidden_size)

# 如果新键名包含"patch_embedding"，则输出信息并对值进行维度转置
if "patch_embedding" in new:
    print("Reshaping patch embedding... for", new)
    val = val.transpose(3, 2, 0, 1)
# 如果新键名以"weight"结尾并且不包含"position_embedding"和"token_embedding"，则对值进行转置
elif new.endswith("weight") and "position_embedding" not in new and "token_embedding" not in new:
    val = val.T

# 如果新键名以"bias"结尾，则对值进行形状调整，转换为一维数组
if new.endswith("bias"):
    val = val.reshape(-1)

# 将处理过的值转换为NumPy数组，然后转换为PyTorch张量，并将新键与其对应的值加入字典中
dct[new] = torch.from_numpy(np.array(val))
    # 使用给定的文本和图像创建输入对象，返回PyTorch张量格式的输入
    inputs = processor(text=texts, images=image, return_tensors="pt")

    # 如果模型名称中不包含 "large" 字符串，检查像素值是否与原始像素值非常接近
    if "large" not in model_name:
        assert torch.allclose(inputs.pixel_values, original_pixel_values.float(), atol=1e-6)
    # 检查前四个位置的输入标识是否与原始输入标识非常接近
    assert torch.allclose(inputs.input_ids[:4, :], original_input_ids[:4, :], atol=1e-6)

    # 禁用梯度计算的上下文环境，计算模型的输出
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        objectness_logits = outputs.objectness_logits

    # 否则，如果模型转换但未验证 logits，则打印消息
    else:
        print("Model converted without verifying logits")

    # 如果指定了 PyTorch 模型保存路径，则保存模型和处理器到本地
    if pytorch_dump_folder_path is not None:
        print("Saving model and processor locally...")
        # 创建保存模型的文件夹
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)

        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 将处理器保存到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到模型库
    if push_to_hub:
        print(f"Pushing {model_name} to the hub...")
        # 将模型推送到指定的模型库位置
        model.push_to_hub(f"google/{model_name}")
        # 将处理器推送到指定的模型库位置
        processor.push_to_hub(f"google/{model_name}")
if __name__ == "__main__":
    # 如果脚本被直接运行而非被导入，则执行以下代码

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必需参数
    parser.add_argument(
        "--model_name",
        default="owlv2-base-patch16",
        choices=[
            "owlv2-base-patch16",
            "owlv2-base-patch16-finetuned",
            "owlv2-base-patch16-ensemble",
            "owlv2-large-patch14",
            "owlv2-large-patch14-finetuned",
            "owlv2-large-patch14-ensemble",
        ],
        type=str,
        help="Name of the Owlv2 model you'd like to convert from FLAX to PyTorch."
    )
    # 添加一个名为 model_name 的可选参数，用于指定 Owlv2 模型的名称

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the original Flax checkpoint."
    )
    # 添加一个名为 checkpoint_path 的必选参数，用于指定原始 Flax 检查点的路径

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=False,
        help="Path to the output PyTorch model directory."
    )
    # 添加一个名为 pytorch_dump_folder_path 的可选参数，用于指定输出的 PyTorch 模型目录的路径

    parser.add_argument(
        "--verify_logits",
        action="store_false",
        required=False,
        help="Path to the output PyTorch model directory."
    )
    # 添加一个名为 verify_logits 的可选参数，设置为 False，用于验证输出的 logits

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model and image preprocessor to the hub"
    )
    # 添加一个名为 push_to_hub 的可选参数，设置为 True，用于将模型和图像预处理器推送到 hub

    args = parser.parse_args()
    # 解析命令行参数并存储到 args 对象中

    convert_owlv2_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub, args.verify_logits
    )
    # 调用 convert_owlv2_checkpoint 函数，传入解析后的参数进行 Owlv2 模型检查点的转换
```