# `.\transformers\models\vit\convert_vit_timm_to_pytorch.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""从 timm 库转换 ViT 和非蒸馏的 DeiT 检查点。"""

# 导入必要的库
import argparse
from pathlib import Path
import requests
import timm
import torch
from PIL import Image
from timm.data import ImageNetInfo, infer_imagenet_subset
# 导入 Hugging Face Transformers 库中的相关模块
from transformers import DeiTImageProcessor, ViTConfig, ViTForImageClassification, ViTImageProcessor, ViTModel
from transformers.utils import logging

# 设置日志级别为 info
logging.set_verbosity_info()
# 获取 logger 对象
logger = logging.get_logger(__name__)

# 创建要重命名的所有键列表（原始名称在左侧，我们的名称在右侧）
def create_rename_keys(config, base_model=False):
    rename_keys = []
    for i in range(config.num_hidden_layers):
        # 编码器层：输出投影、2 个前馈神经网络和 2 个层归一化
        rename_keys.append((f"blocks.{i}.norm1.weight", f"vit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"vit.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"vit.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"vit.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"vit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"vit.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"vit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"vit.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"vit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"vit.encoder.layer.{i}.output.dense.bias"))

    # 投影层 + 位置嵌入
    rename_keys.extend(
        [
            ("cls_token", "vit.embeddings.cls_token"),
            ("patch_embed.proj.weight", "vit.embeddings.patch_embeddings.projection.weight"),
            ("patch_embed.proj.bias", "vit.embeddings.patch_embeddings.projection.bias"),
            ("pos_embed", "vit.embeddings.position_embeddings"),
        ]
    )
    # 如果存在基础模型，则执行以下操作
    if base_model:
        # 添加需要重命名的键值对到列表中
        rename_keys.extend(
            [
                ("norm.weight", "layernorm.weight"),
                ("norm.bias", "layernorm.bias"),
            ]
        )

        # 如果只有基础模型，应该将所有以"vit"开头的键中的"vit"删除
        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("vit") else pair for pair in rename_keys]
    else:
        # 添加需要重命名的键值对到列表中
        rename_keys.extend(
            [
                ("norm.weight", "vit.layernorm.weight"),
                ("norm.bias", "vit.layernorm.bias"),
                ("head.weight", "classifier.weight"),
                ("head.bias", "classifier.bias"),
            ]
        )

    # 返回重命名后的键值对列表
    return rename_keys
# 将每个编码器层的矩阵拆分为查询（query）、键（keys）和值（values）
def read_in_q_k_v(state_dict, config, base_model=False):
    # 遍历每个编码器层
    for i in range(config.num_hidden_layers):
        # 如果是基础模型，则前缀为空字符串，否则为"vit."
        if base_model:
            prefix = ""
        else:
            prefix = "vit."
        # 读取输入投影层（在timm中，这是一个单矩阵加偏置的操作）
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # 将查询、键和值（按顺序）添加到状态字典中
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


# 从状态字典中移除分类头部
def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


# 重命名键名
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# 在一组可爱猫的图片上验证我们的结果
def prepare_img():
    # 图片地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用requests库获取图片
    im = Image.open(requests.get(url, stream=True).raw)
    return im


# 转换ViT检查点
@torch.no_grad()
def convert_vit_checkpoint(vit_name, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our ViT structure.
    """
    # 定义默认的ViT配置
    config = ViTConfig()
    base_model = False

    # 从timm加载原始模型
    timm_model = timm.create_model(vit_name, pretrained=True)
    timm_model.eval()

    # 检测transformers中不支持的ViT模型
    # 存在fc_norm
    if not isinstance(getattr(timm_model, "fc_norm", None), torch.nn.Identity):
        raise ValueError(f"{vit_name} is not supported in transformers because of the presence of fc_norm.")

    # 使用全局平均池化结合（或不使用）类令牌
    if getattr(timm_model, "global_pool", None) == "avg":
        raise ValueError(f"{vit_name} is not supported in transformers because of use of global average pooling.")

    # 具有norm_pre层的CLIP样式vit
    # 检查是否为 CLIP 风格的 ViT 模型，并且不包含 norm_pre 层，如果是则抛出异常
    if "clip" in vit_name and not isinstance(getattr(timm_model, "norm_pre", None), torch.nn.Identity):
        raise ValueError(
            f"{vit_name} is not supported in transformers because it's a CLIP style ViT with norm_pre layer."
        )

    # 检查是否为 SigLIP 风格的 ViT 模型，并且包含 attn_pool 层，如果是则抛出异常
    if "siglip" in vit_name and getattr(timm_model, "global_pool", None) == "map":
        raise ValueError(
            f"{vit_name} is not supported in transformers because it's a SigLIP style ViT with attn_pool."
        )

    # 检查 ViT 模型块中是否使用了 layer scale，如果是则抛出异常
    if not isinstance(getattr(timm_model.blocks[0], "ls1", None), torch.nn.Identity) or not isinstance(
        getattr(timm_model.blocks[0], "ls2", None), torch.nn.Identity
    ):
        raise ValueError(f"{vit_name} is not supported in transformers because it uses a layer scale in its blocks.")

    # 检查是否为混合 ResNet-ViT 模型，如果是则抛出异常
    if not isinstance(timm_model.patch_embed, timm.layers.PatchEmbed):
        raise ValueError(f"{vit_name} is not supported in transformers because it is a hybrid ResNet-ViT.")

    # 从 patch embedding 子模块中获取 patch 大小和图像大小
    config.patch_size = timm_model.patch_embed.patch_size[0]
    config.image_size = timm_model.patch_embed.img_size[0]

    # 从 timm 模型中检索特定于架构的参数
    config.hidden_size = timm_model.embed_dim
    config.intermediate_size = timm_model.blocks[0].mlp.fc1.out_features
    config.num_hidden_layers = len(timm_model.blocks)
    config.num_attention_heads = timm_model.blocks[0].attn.num_heads

    # 检查模型是否具有分类头
    if timm_model.num_classes != 0:
        config.num_labels = timm_model.num_classes
        # 从 timm 模型中推断 ImageNet 子集
        imagenet_subset = infer_imagenet_subset(timm_model)
        dataset_info = ImageNetInfo(imagenet_subset)
        config.id2label = {i: dataset_info.index_to_label_name(i) for i in range(dataset_info.num_classes())}
        config.label2id = {v: k for k, v in config.id2label.items()}
    else:
        print(f"{vit_name} is going to be converted as a feature extractor only.")
        base_model = True

    # 加载原始模型的 state_dict
    state_dict = timm_model.state_dict()

    # 移除和重命名 state_dict 中的一些键
    if base_model:
        remove_classification_head_(state_dict)
    rename_keys = create_rename_keys(config, base_model)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, base_model)

    # 加载 HuggingFace 模型
    if base_model:
        model = ViTModel(config, add_pooling_layer=False).eval()
    else:
        model = ViTForImageClassification(config).eval()
    model.load_state_dict(state_dict)

    # 在由 ViTImageProcessor/DeiTImageProcessor 准备的图像上检查输出
    if "deit" in vit_name:
        image_processor = DeiTImageProcessor(size=config.image_size)
    # 如果 base_model 为真，则使用 ViTImageProcessor 初始化 image_processor
    else:
        image_processor = ViTImageProcessor(size=config.image_size)
    # 对准备好的图像进行编码，返回张量
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    # 获取像素值
    pixel_values = encoding["pixel_values"]
    # 使用模型处理像素值
    outputs = model(pixel_values)

    # 如果存在 base_model
    if base_model:
        # 使用 timm_model 处理像素值
        timm_pooled_output = timm_model.forward_features(pixel_values)
        # 断言 timm_pooled_output 和 outputs.last_hidden_state 的形状相同
        assert timm_pooled_output.shape == outputs.last_hidden_state.shape
        # 断言 timm_pooled_output 和 outputs.last_hidden_state 在给定的容差范围内相等
        assert torch.allclose(timm_pooled_output, outputs.last_hidden_state, atol=1e-1)
    # 如果不存在 base_model
    else:
        # 使用 timm_model 处理像素值
        timm_logits = timm_model(pixel_values)
        # 断言 timm_logits 和 outputs.logits 的形状相同
        assert timm_logits.shape == outputs.logits.shape
        # 断言 timm_logits 和 outputs.logits 在给定的容差范围内相等
        assert torch.allclose(timm_logits, outputs.logits, atol=1e-3)

    # 创建目录，如果目录已存在则不做任何操作
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印保存模型的信息
    print(f"Saving model {vit_name} to {pytorch_dump_folder_path}")
    # 保存模型到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印保存图像处理器的信息
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 保存图像处理器到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--vit_name",
        default="vit_base_patch16_224",
        type=str,
        help="Name of the ViT timm model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将 ViT 模型转换为 PyTorch 模型
    convert_vit_checkpoint(args.vit_name, args.pytorch_dump_folder_path)
```