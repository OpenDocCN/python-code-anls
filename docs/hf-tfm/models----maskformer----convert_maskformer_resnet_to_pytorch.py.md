# `.\transformers\models\maskformer\convert_maskformer_resnet_to_pytorch.py`

```
# 设置脚本的编码格式为 utf-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 使用 Apache 许可证 2.0 版本，只有在遵守许可证的情况下才能使用该文件
# 可以从以下地址获取许可证副本 http://www.apache.org/licenses/LICENSE-2.0
# 未经适用法律或书面同意，不得使用此文件
# 分发的软件基于"原样"基础分发，没有任何明示或暗示的担保或条件
# 查看许可证以了解特定语言的权限和限制
"""将原始仓库中具有 ResNet 骨干的 MaskFormer 检查点转换为 Hugging Face 模型。URL:
https://github.com/facebookresearch/MaskFormer"""


import argparse
import json
import pickle
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation, MaskFormerImageProcessor, ResNetConfig
from transformers.utils import logging


logging.set_verbosity_info()
# 设置日志记录器
logger = logging.get_logger(__name__)


def get_maskformer_config(model_name: str):
    # 如果模型名称包含"resnet101c"
    if "resnet101c" in model_name:
        # TODO 添加 ResNet-C 骨干的支持，该骨干使用 "deeplab" stem
        raise NotImplementedError("To do")
    # 如果模型名称包含"resnet101"
    elif "resnet101" in model_name:
        # 使用预训练的 ResNet-101 配置，输出特征有["stage1", "stage2", "stage3", "stage4"]
        backbone_config = ResNetConfig.from_pretrained(
            "microsoft/resnet-101", out_features=["stage1", "stage2", "stage3", "stage4"]
        )
    else:
        # 使用预训练的 ResNet-50 配置，输出特征有["stage1", "stage2", "stage3", "stage4"]
        backbone_config = ResNetConfig.from_pretrained(
            "microsoft/resnet-50", out_features=["stage1", "stage2", "stage3", "stage4"]
        )
    # 创建 MaskFormer 模型配置，使用 ResNet 骨干配置
    config = MaskFormerConfig(backbone_config=backbone_config)

    # 获取存储库 ID
    repo_id = "huggingface/label-files"
    # 根据模型名称设置不同的标签数量和文件名
    if "ade20k-full" in model_name:
        config.num_labels = 847
        filename = "maskformer-ade20k-full-id2label.json"
    elif "ade" in model_name:
        config.num_labels = 150
        filename = "ade20k-id2label.json"
    elif "coco-stuff" in model_name:
        config.num_labels = 171
        filename = "maskformer-coco-stuff-id2label.json"
    elif "coco" in model_name:
        # TODO
        config.num_labels = 133
        filename = "coco-panoptic-id2label.json"
    elif "cityscapes" in model_name:
        config.num_labels = 19
        filename = "cityscapes-id2label.json"
    elif "vistas" in model_name:
        config.num_labels = 65
        filename = "mapillary-vistas-id2label.json"

    # 加载 id 到标签的映射关系
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


def create_rename_keys(config):
    # 创建重命名键列表
    rename_keys = []
    # stem
    # fmt: off
    # 添加重命名键对，将原骨干的权重命名为新模型的权重
    rename_keys.append(("backbone.stem.conv1.weight", "model.pixel_level_module.encoder.embedder.embedder.convolution.weight"))
    # 将键值对元组添加到列表中，用于重命名模型参数
    rename_keys.append(("backbone.stem.conv1.norm.weight", "model.pixel_level_module.encoder.embedder.embedder.normalization.weight"))
    rename_keys.append(("backbone.stem.conv1.norm.bias", "model.pixel_level_module.encoder.embedder.embedder.normalization.bias"))
    rename_keys.append(("backbone.stem.conv1.norm.running_mean", "model.pixel_level_module.encoder.embedder.embedder.normalization.running_mean"))
    rename_keys.append(("backbone.stem.conv1.norm.running_var", "model.pixel_level_module.encoder.embedder.embedder.normalization.running_var"))
    # 格式设置：开启格式化
    # stages
    # FPN
    # 格式设置：关闭格式化
    rename_keys.append(("sem_seg_head.layer_4.weight", "model.pixel_level_module.decoder.fpn.stem.0.weight"))
    rename_keys.append(("sem_seg_head.layer_4.norm.weight", "model.pixel_level_module.decoder.fpn.stem.1.weight"))
    rename_keys.append(("sem_seg_head.layer_4.norm.bias", "model.pixel_level_module.decoder.fpn.stem.1.bias"))
    # 使用 zip 函数创建索引范围并同时迭代两个列表，用于重命名模型参数
    for source_index, target_index in zip(range(3, 0, -1), range(0, 3)):
        # 逐层重命名适配器和卷积层参数
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.0.weight"))
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.norm.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.1.weight"))
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.norm.bias", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.1.bias"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.0.weight"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.norm.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.1.weight"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.norm.bias", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.1.bias"))
    # 重命名掩码特征的参数
    rename_keys.append(("sem_seg_head.mask_features.weight", "model.pixel_level_module.decoder.mask_projection.weight"))
    rename_keys.append(("sem_seg_head.mask_features.bias", "model.pixel_level_module.decoder.mask_projection.bias"))
    # 格式设置：开启格式化

    # Transformer decoder
    # 格式设置：关闭格式化
    for idx in range(config.decoder_config.decoder_layers):
        # 针对每一层 decoder，将对应的参数名映射到新的模型参数名
        # self-attention out projection
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.out_proj.weight", f"model.transformer_module.decoder.layers.{idx}.self_attn.out_proj.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.out_proj.bias", f"model.transformer_module.decoder.layers.{idx}.self_attn.out_proj.bias"))
        # cross-attention out projection
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.out_proj.weight", f"model.transformer_module.decoder.layers.{idx}.encoder_attn.out_proj.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.out_proj.bias", f"model.transformer_module.decoder.layers.{idx}.encoder_attn.out_proj.bias"))
        # MLP 1
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear1.weight", f"model.transformer_module.decoder.layers.{idx}.fc1.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear1.bias", f"model.transformer_module.decoder.layers.{idx}.fc1.bias"))
        # MLP 2
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear2.weight", f"model.transformer_module.decoder.layers.{idx}.fc2.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear2.bias", f"model.transformer_module.decoder.layers.{idx}.fc2.bias"))
        # layernorm 1 (self-attention layernorm)
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm1.weight", f"model.transformer_module.decoder.layers.{idx}.self_attn_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm1.bias", f"model.transformer_module.decoder.layers.{idx}.self_attn_layer_norm.bias"))
        # layernorm 2 (cross-attention layernorm)
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm2.weight", f"model.transformer_module.decoder.layers.{idx}.encoder_attn_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm2.bias", f"model.transformer_module.decoder.layers.{idx}.encoder_attn_layer_norm.bias"))
        # layernorm 3 (final layernorm)
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm3.weight", f"model.transformer_module.decoder.layers.{idx}.final_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm3.bias", f"model.transformer_module.decoder.layers.{idx}.final_layer_norm.bias"))

    # 将最后一个映射的参数名映射到新的模型的参数名
    rename_keys.append(("sem_seg_head.predictor.transformer.decoder.norm.weight", "model.transformer_module.decoder.layernorm.weight"))
    # 将键的重命名信息添加到重命名键列表中
    rename_keys.append(("sem_seg_head.predictor.transformer.decoder.norm.bias", "model.transformer_module.decoder.layernorm.bias"))
    # fmt: on
    
    # 开启格式化（fmt: on）
    
    # 头部操作
    # fmt: off
    # 将查询嵌入权重的重命名信息添加到重命名键列表中
    rename_keys.append(("sem_seg_head.predictor.query_embed.weight", "model.transformer_module.queries_embedder.weight"))
    
    # 将输入投影权重的重命名信息添加到重命名键列表中
    rename_keys.append(("sem_seg_head.predictor.input_proj.weight", "model.transformer_module.input_projection.weight"))
    # 将输入投影偏差的重命名信息添加到重命名键列表中
    rename_keys.append(("sem_seg_head.predictor.input_proj.bias", "model.transformer_module.input_projection.bias"))
    
    # 将类别嵌入权重的重命名信息添加到重命名键列表中
    rename_keys.append(("sem_seg_head.predictor.class_embed.weight", "class_predictor.weight"))
    # 将类别嵌入偏差的重命名信息添加到重命名键列表中
    rename_keys.append(("sem_seg_head.predictor.class_embed.bias", "class_predictor.bias"))
    
    # 遍历三个层级，将每个层级的蒙版嵌入权重和偏差的重命名信息添加到重命名键列表中
    for i in range(3):
        rename_keys.append((f"sem_seg_head.predictor.mask_embed.layers.{i}.weight", f"mask_embedder.{i}.0.weight"))
        rename_keys.append((f"sem_seg_head.predictor.mask_embed.layers.{i}.bias", f"mask_embedder.{i}.0.bias"))
    # fmt: on
    
    # 返回重命名键列表
    return rename_keys
# 更改字典中的键名，将旧键名替换为新键名
def rename_key(dct, old, new):
    # 从字典中移除旧键并获取其值
    val = dct.pop(old)
    # 将该值赋给新键名
    dct[new] = val


# 将每层解码器的矩阵拆分为查询（queries）、键（keys）和值（values）
def read_in_decoder_q_k_v(state_dict, config):
    # 关闭代码格式化检查
    # fmt: off
    # 获取解码器的隐藏层大小
    hidden_size = config.decoder_config.hidden_size
    # 遍历解码器中的所有层
    for idx in range(config.decoder_config.decoder_layers):
        # 从状态字典中移除自注意力的输入投影矩阵和偏差
        in_proj_weight = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.in_proj_bias")
        # 在状态字典中添加查询的权重和偏差
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.q_proj.weight"] = in_proj_weight[: hidden_size, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.q_proj.bias"] = in_proj_bias[:config.hidden_size]
        # 在状态字典中添加键的权重和偏差
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.k_proj.weight"] = in_proj_weight[hidden_size : hidden_size * 2, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.k_proj.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
        # 在状态字典中添加值的权重和偏差
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.v_proj.weight"] = in_proj_weight[-hidden_size :, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.v_proj.bias"] = in_proj_bias[-hidden_size :]
        # 从状态字典中移除交叉注意力的输入投影矩阵和偏差
        in_proj_weight = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.in_proj_bias")
        # 在状态字典中添加交叉注意力的查询的权重和偏差
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.q_proj.weight"] = in_proj_weight[: hidden_size, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.q_proj.bias"] = in_proj_bias[:config.hidden_size]
        # 在状态字典中添加交叉注意力的键的权重和偏差
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.k_proj.weight"] = in_proj_weight[hidden_size : hidden_size * 2, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.k_proj.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
        # 在状态字典中添加交叉注意力的值的权重和偏差
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.v_proj.weight"] = in_proj_weight[-hidden_size :, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.v_proj.bias"] = in_proj_bias[-hidden_size :]
    # 重新打开代码格式化检查
    # fmt: on


# 准备一张包含可爱小猫的图像，作为后续步骤的验证输入
def prepare_img() -> torch.Tensor:
    # 定义图片的 URL 地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用 requests 库发出 GET 请求，获取图片数据流，并交给 Image.open() 函数打开图片
    im = Image.open(requests.get(url, stream=True).raw)
    # 返回打开的图片对象
    return im
# 使用 torch.no_grad() 修饰的函数，表示在此函数中的操作将不会计算梯度
@torch.no_grad()
def convert_maskformer_checkpoint(
    model_name: str, checkpoint_path: str, pytorch_dump_folder_path: str, push_to_hub: bool = False
):
    """
    复制/粘贴/调整模型的权重到我们的 MaskFormer 结构中。
    """
    # 获取指定模型的配置信息
    config = get_maskformer_config(model_name)

    # 读取原始 state_dict
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    state_dict = data["model"]

    # 重命名键名
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_decoder_q_k_v(state_dict, config)

    # 更新为 torch 张量
    for key, value in state_dict.items():
        state_dict[key] = torch.from_numpy(value)

    # 加载模型
    model = MaskFormerForInstanceSegmentation(config)
    model.eval()

    # 加载权重
    model.load_state_dict(state_dict)

    # 验证结果
    image = prepare_img()
    if "vistas" in model_name:
        ignore_index = 65
    elif "cityscapes" in model_name:
        ignore_index = 65535
    else:
        ignore_index = 255
    reduce_labels = True if "ade" in model_name else False
    image_processor = MaskFormerImageProcessor(ignore_index=ignore_index, reduce_labels=reduce_labels)

    inputs = image_processor(image, return_tensors="pt")

    outputs = model(**inputs)

    # 根据模型名称设置预期的 logits
    if model_name == "maskformer-resnet50-ade":
        expected_logits = torch.tensor(
            [[6.7710, -0.1452, -3.5687], [1.9165, -1.0010, -1.8614], [3.6209, -0.2950, -1.3813]]
        )
    elif model_name == "maskformer-resnet101-ade":
        expected_logits = torch.tensor(
            [[4.0381, -1.1483, -1.9688], [2.7083, -1.9147, -2.2555], [3.4367, -1.3711, -2.1609]]
        )
    # 省略其他模型的预期 logits
    # 检查输出的 class_queries_logits 是否与预期值接近
    assert torch.allclose(outputs.class_queries_logits[0, :3, :3], expected_logits, atol=1e-4)
    # 输出提示信息，表示检查通过
    print("Looks ok!")
    
    # 如果指定了保存路径
    if pytorch_dump_folder_path is not None:
        # 输出保存模型和图像处理器的信息
        print(f"Saving model and image processor of {model_name} to {pytorch_dump_folder_path}")
        # 创建保存路径
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 保存模型
        model.save_pretrained(pytorch_dump_folder_path)
        # 保存图像处理器
        image_processor.save_pretrained(pytorch_dump_folder_path)
    
    # 如果需要推送到 Hub
    if push_to_hub:
        # 输出推送模型和图像处理器的信息
        print(f"Pushing model and image processor of {model_name} to the hub...")
        # 推送模型到 Hub
        model.push_to_hub(f"facebook/{model_name}")
        # 推送图像处理器到 Hub
        image_processor.push_to_hub(f"facebook/{model_name}")
# 当该脚本作为主程序运行时，执行以下操作
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 定义需要的参数
    # Required parameters
    # 添加 model_name 参数
    # 默认值为 "maskformer-resnet50-ade"
    # 数据类型为字符串，必须提供
    # 可选值包括一系列 MaskFormer 模型名称
    # 添加参数的帮助信息
    parser.add_argument(
        "--model_name",
        default="maskformer-resnet50-ade",
        type=str,
        required=True,
        choices=[
            "maskformer-resnet50-ade",
            "maskformer-resnet101-ade",
            "maskformer-resnet50-coco-stuff",
            "maskformer-resnet101-coco-stuff",
            "maskformer-resnet101-cityscapes",
            "maskformer-resnet50-vistas",
            "maskformer-resnet50-ade20k-full",
            "maskformer-resnet101-ade20k-full",
        ],
        help=("Name of the MaskFormer model you'd like to convert",),
    )
    # 添加 checkpoint_path 参数
    # 数据类型为字符串，必须提供
    # 添加参数的帮助信息
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help=("Path to the original pickle file (.pkl) of the original checkpoint.",),
    )
    # 添加 pytorch_dump_folder_path 参数
    # 数据类型为字符串，默认为 None
    # 添加参数的帮助信息
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加 push_to_hub 参数
    # 如果出现则为 True，否则为 False
    # 添加参数的帮助信息
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用 convert_maskformer_checkpoint 函数
    # 参数为上面定义的参数
    convert_maskformer_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
```