# `.\models\maskformer\convert_maskformer_swin_to_pytorch.py`

```py
# 设置编码格式为 UTF-8
# 版权声明和许可证信息，指定代码使用 Apache License, Version 2.0
# 导入所需模块和库
# 这个脚本用于从原始仓库转换 MaskFormer 模型检查点，详细信息参见 https://github.com/facebookresearch/MaskFormer

import argparse  # 导入命令行参数解析模块
import json  # 导入处理 JSON 格式数据的模块
import pickle  # 导入序列化和反序列化 Python 对象的模块
from pathlib import Path  # 导入处理路径操作的模块

import requests  # 导入发送 HTTP 请求的库
import torch  # 导入 PyTorch 深度学习库
from huggingface_hub import hf_hub_download  # 导入从 Hugging Face Hub 下载资源的函数
from PIL import Image  # 导入 Python Imaging Library，用于图像处理

from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation, MaskFormerImageProcessor, SwinConfig  # 导入 MaskFormer 相关类
from transformers.utils import logging  # 导入日志记录工具

logging.set_verbosity_info()  # 设置日志记录器的详细程度为信息级别
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def get_maskformer_config(model_name: str):
    # 根据预训练的 Swin 模型配置 MaskFormerConfig
    backbone_config = SwinConfig.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224", out_features=["stage1", "stage2", "stage3", "stage4"]
    )
    config = MaskFormerConfig(backbone_config=backbone_config)

    repo_id = "huggingface/label-files"
    if "ade20k-full" in model_name:
        # 设置适用于 ade20k-full 模型的类别数和标签映射文件名
        config.num_labels = 847
        filename = "maskformer-ade20k-full-id2label.json"
    elif "ade" in model_name:
        # 设置适用于 ade 模型的类别数和标签映射文件名
        config.num_labels = 150
        filename = "ade20k-id2label.json"
    elif "coco-stuff" in model_name:
        # 设置适用于 coco-stuff 模型的类别数和标签映射文件名
        config.num_labels = 171
        filename = "maskformer-coco-stuff-id2label.json"
    elif "coco" in model_name:
        # TODO
        config.num_labels = 133
        filename = "coco-panoptic-id2label.json"
    elif "cityscapes" in model_name:
        # 设置适用于 cityscapes 模型的类别数和标签映射文件名
        config.num_labels = 19
        filename = "cityscapes-id2label.json"
    elif "vistas" in model_name:
        # 设置适用于 vistas 模型的类别数和标签映射文件名
        config.num_labels = 65
        filename = "mapillary-vistas-id2label.json"

    # 从 Hugging Face Hub 下载指定文件并加载为字典格式
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    return config


def create_rename_keys(config):
    rename_keys = []
    # 定义需要重命名的键列表
    # stem
    # fmt: off
    rename_keys.append(("backbone.patch_embed.proj.weight", "model.pixel_level_module.encoder.model.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("backbone.patch_embed.proj.bias", "model.pixel_level_module.encoder.model.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("backbone.patch_embed.norm.weight", "model.pixel_level_module.encoder.model.embeddings.norm.weight"))
    # fmt: on
    # 将键值对("backbone.patch_embed.norm.bias", "model.pixel_level_module.encoder.model.embeddings.norm.bias")添加到rename_keys列表中
    rename_keys.append(("backbone.patch_embed.norm.bias", "model.pixel_level_module.encoder.model.embeddings.norm.bias"))

    # 将以下键值对依次添加到rename_keys列表中，用于重命名模型结构中的参数
    rename_keys.append(("sem_seg_head.layer_4.weight", "model.pixel_level_module.decoder.fpn.stem.0.weight"))
    rename_keys.append(("sem_seg_head.layer_4.norm.weight", "model.pixel_level_module.decoder.fpn.stem.1.weight"))
    rename_keys.append(("sem_seg_head.layer_4.norm.bias", "model.pixel_level_module.decoder.fpn.stem.1.bias"))

    # 使用循环将逐个source_index到target_index的适配器和层参数重命名添加到rename_keys列表中
    for source_index, target_index in zip(range(3, 0, -1), range(0, 3)):
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.0.weight"))
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.norm.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.1.weight"))
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.norm.bias", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.1.bias"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.0.weight"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.norm.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.1.weight"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.norm.bias", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.1.bias"))

    # 将键值对("sem_seg_head.mask_features.weight", "model.pixel_level_module.decoder.mask_projection.weight")添加到rename_keys列表中
    rename_keys.append(("sem_seg_head.mask_features.weight", "model.pixel_level_module.decoder.mask_projection.weight"))
    # 将键值对("sem_seg_head.mask_features.bias", "model.pixel_level_module.decoder.mask_projection.bias")添加到rename_keys列表中
    rename_keys.append(("sem_seg_head.mask_features.bias", "model.pixel_level_module.decoder.mask_projection.bias"))
    
    # Transformer解码器部分暂无代码，未进行注释
    # 遍历从配置中获取的解码器层数
    for idx in range(config.decoder_config.decoder_layers):
        # 处理自注意力机制的输出投影层权重和偏置
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.out_proj.weight", f"model.transformer_module.decoder.layers.{idx}.self_attn.out_proj.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.out_proj.bias", f"model.transformer_module.decoder.layers.{idx}.self_attn.out_proj.bias"))
        
        # 处理跨注意力机制的输出投影层权重和偏置
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.out_proj.weight", f"model.transformer_module.decoder.layers.{idx}.encoder_attn.out_proj.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.out_proj.bias", f"model.transformer_module.decoder.layers.{idx}.encoder_attn.out_proj.bias"))
        
        # 处理MLP第一层的权重和偏置
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear1.weight", f"model.transformer_module.decoder.layers.{idx}.fc1.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear1.bias", f"model.transformer_module.decoder.layers.{idx}.fc1.bias"))
        
        # 处理MLP第二层的权重和偏置
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear2.weight", f"model.transformer_module.decoder.layers.{idx}.fc2.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear2.bias", f"model.transformer_module.decoder.layers.{idx}.fc2.bias"))
        
        # 处理自注意力机制的LayerNorm层的权重和偏置
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm1.weight", f"model.transformer_module.decoder.layers.{idx}.self_attn_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm1.bias", f"model.transformer_module.decoder.layers.{idx}.self_attn_layer_norm.bias"))
        
        # 处理跨注意力机制的LayerNorm层的权重和偏置
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm2.weight", f"model.transformer_module.decoder.layers.{idx}.encoder_attn_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm2.bias", f"model.transformer_module.decoder.layers.{idx}.encoder_attn_layer_norm.bias"))
        
        # 处理最终LayerNorm层的权重和偏置
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm3.weight", f"model.transformer_module.decoder.layers.{idx}.final_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm3.bias", f"model.transformer_module.decoder.layers.{idx}.final_layer_norm.bias"))

    # 将最后一个未处理的LayerNorm层的权重和偏置添加到重命名列表中
    rename_keys.append(("sem_seg_head.predictor.transformer.decoder.norm.weight", "model.transformer_module.decoder.layernorm.weight"))
    # 将旧的模型参数名称与新模型参数名称配对并添加到重命名键列表中
    rename_keys.append(("sem_seg_head.predictor.transformer.decoder.norm.bias", "model.transformer_module.decoder.layernorm.bias"))

    # 将旧的模型参数名称与新模型参数名称配对并添加到重命名键列表中，用于顶部的头部模块
    rename_keys.append(("sem_seg_head.predictor.query_embed.weight", "model.transformer_module.queries_embedder.weight"))

    # 将旧的模型参数名称与新模型参数名称配对并添加到重命名键列表中，用于输入投影权重
    rename_keys.append(("sem_seg_head.predictor.input_proj.weight", "model.transformer_module.input_projection.weight"))
    # 将旧的模型参数名称与新模型参数名称配对并添加到重命名键列表中，用于输入投影偏置
    rename_keys.append(("sem_seg_head.predictor.input_proj.bias", "model.transformer_module.input_projection.bias"))

    # 将旧的模型参数名称与新模型参数名称配对并添加到重命名键列表中，用于类别预测权重
    rename_keys.append(("sem_seg_head.predictor.class_embed.weight", "class_predictor.weight"))
    # 将旧的模型参数名称与新模型参数名称配对并添加到重命名键列表中，用于类别预测偏置
    rename_keys.append(("sem_seg_head.predictor.class_embed.bias", "class_predictor.bias"))

    # 循环处理每个掩码嵌入层，将旧的模型参数名称与新模型参数名称配对并添加到重命名键列表中
    for i in range(3):
        rename_keys.append((f"sem_seg_head.predictor.mask_embed.layers.{i}.weight", f"mask_embedder.{i}.0.weight"))
        rename_keys.append((f"sem_seg_head.predictor.mask_embed.layers.{i}.bias", f"mask_embedder.{i}.0.bias"))
    # fmt: on

    # 返回最终的重命名键列表
    return rename_keys
# 重新命名字典 `dct` 中键 `old` 为 `new`
def rename_key(dct, old, new):
    val = dct.pop(old)  # 弹出键为 `old` 的值，并保存到变量 `val`
    dct[new] = val  # 将值 `val` 与新键 `new` 关联并添加到字典中

# we split up the matrix of each encoder layer into queries, keys and values
# 将每个编码器层的矩阵拆分为查询、键和值
def read_in_swin_q_k_v(state_dict, backbone_config):
    num_features = [int(backbone_config.embed_dim * 2**i) for i in range(len(backbone_config.depths))]
    for i in range(len(backbone_config.depths)):
        dim = num_features[i]
        for j in range(backbone_config.depths[i]):
            # fmt: off
            # 读取输入投影层 (in_proj) 的权重和偏置 (在原始实现中，这是一个单独的矩阵加偏置)
            in_proj_weight = state_dict.pop(f"backbone.layers.{i}.blocks.{j}.attn.qkv.weight")
            in_proj_bias = state_dict.pop(f"backbone.layers.{i}.blocks.{j}.attn.qkv.bias")
            # 接下来，按顺序添加查询、键和值到状态字典
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.query.weight"] = in_proj_weight[:dim, :]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.query.bias"] = in_proj_bias[: dim]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.key.weight"] = in_proj_weight[
                dim : dim * 2, :
            ]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.key.bias"] = in_proj_bias[
                dim : dim * 2
            ]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.value.weight"] = in_proj_weight[
                -dim :, :
            ]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.value.bias"] = in_proj_bias[-dim :]
            # fmt: on

# we split up the matrix of each encoder layer into queries, keys and values
# 将每个解码器层的矩阵拆分为查询、键和值
def read_in_decoder_q_k_v(state_dict, config):
    # fmt: off
    hidden_size = config.decoder_config.hidden_size
    # 遍历解码器层次的数量
    for idx in range(config.decoder_config.decoder_layers):
        # 读取自注意力输入投影层的权重和偏置（在原始实现中，这是单独的矩阵和偏置）
        in_proj_weight = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.in_proj_bias")
        
        # 将查询（query）、键（keys）和值（values）依次添加到状态字典中
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.q_proj.weight"] = in_proj_weight[: hidden_size, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.q_proj.bias"] = in_proj_bias[:config.hidden_size]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.k_proj.weight"] = in_proj_weight[hidden_size : hidden_size * 2, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.k_proj.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.v_proj.weight"] = in_proj_weight[-hidden_size :, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.v_proj.bias"] = in_proj_bias[-hidden_size :]
        
        # 读取交叉注意力输入投影层的权重和偏置（在原始实现中，这是单独的矩阵和偏置）
        in_proj_weight = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.in_proj_bias")
        
        # 将查询（query）、键（keys）和值（values）依次添加到状态字典中
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.q_proj.weight"] = in_proj_weight[: hidden_size, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.q_proj.bias"] = in_proj_bias[:config.hidden_size]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.k_proj.weight"] = in_proj_weight[hidden_size : hidden_size * 2, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.k_proj.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.v_proj.weight"] = in_proj_weight[-hidden_size :, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.v_proj.bias"] = in_proj_bias[-hidden_size :]
    
    # 格式化结束
    # fmt: on
# We will verify our results on an image of cute cats
def prepare_img() -> torch.Tensor:
    # 定义图像的 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 通过 HTTP 请求获取图像的原始数据流，并用 PIL 库打开图像
    im = Image.open(requests.get(url, stream=True).raw)
    return im

@torch.no_grad()
def convert_maskformer_checkpoint(
    model_name: str, checkpoint_path: str, pytorch_dump_folder_path: str, push_to_hub: bool = False
):
    """
    Copy/paste/tweak model's weights to our MaskFormer structure.
    """
    # 根据模型名获取 MaskFormer 的配置信息
    config = get_maskformer_config(model_name)

    # 加载原始的状态字典
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    state_dict = data["model"]

    # 打印状态字典中每个键和对应的形状（注释掉的部分）
    # for name, param in state_dict.items():
    #     print(name, param.shape)

    # 根据配置信息创建重命名键列表
    rename_keys = create_rename_keys(config)
    # 对状态字典中的键进行重命名操作
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # 从状态字典中读取 Swin Transformer 的 QKV 参数
    read_in_swin_q_k_v(state_dict, config.backbone_config)
    # 从状态字典中读取解码器的 QKV 参数
    read_in_decoder_q_k_v(state_dict, config)

    # 将所有值转换为 Torch 张量
    for key, value in state_dict.items():
        state_dict[key] = torch.from_numpy(value)

    # 加载 MaskFormer 模型
    model = MaskFormerForInstanceSegmentation(config)
    model.eval()

    # 打印模型中每个参数的名称和形状
    for name, param in model.named_parameters():
        print(name, param.shape)

    # 加载状态字典到模型中，并检查缺失和多余的键
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert missing_keys == [
        "model.pixel_level_module.encoder.model.layernorm.weight",
        "model.pixel_level_module.encoder.model.layernorm.bias",
    ]
    assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"

    # 验证模型在给定图像上的输出结果
    image = prepare_img()
    # 根据模型名设置忽略的索引值
    if "vistas" in model_name:
        ignore_index = 65
    elif "cityscapes" in model_name:
        ignore_index = 65535
    else:
        ignore_index = 255
    # 根据模型名设置是否减少标签数
    reduce_labels = True if "ade" in model_name else False
    # 创建 MaskFormerImageProcessor 实例来处理图像
    image_processor = MaskFormerImageProcessor(ignore_index=ignore_index, reduce_labels=reduce_labels)

    # 对输入图像进行预处理，返回模型所需的输入张量
    inputs = image_processor(image, return_tensors="pt")

    # 在模型上执行前向传播，获取输出
    outputs = model(**inputs)

    # 打印输出张量的一部分内容（Logits）
    print("Logits:", outputs.class_queries_logits[0, :3, :3])

    # 根据模型名设置期望的 Logits 值，用于断言验证
    if model_name == "maskformer-swin-tiny-ade":
        expected_logits = torch.tensor(
            [[3.6353, -4.4770, -2.6065], [0.5081, -4.2394, -3.5343], [2.1909, -5.0353, -1.9323]]
        )
    assert torch.allclose(outputs.class_queries_logits[0, :3, :3], expected_logits, atol=1e-4)
    print("Looks ok!")

    # 如果指定了 pytorch_dump_folder_path，则保存模型和图像处理器
    if pytorch_dump_folder_path is not None:
        print(f"Saving model and image processor to {pytorch_dump_folder_path}")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        image_processor.save_pretrained(pytorch_dump_folder_path)

    # 如果 push_to_hub 为 True，则将模型和图像处理器推送到模型中心
    if push_to_hub:
        print("Pushing model and image processor to the hub...")
        model.push_to_hub(f"nielsr/{model_name}")
        image_processor.push_to_hub(f"nielsr/{model_name}")


if __name__ == "__main__":
    # 主程序入口点，此处不添加任何注释
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    
    # 添加命令行参数：模型名称
    parser.add_argument(
        "--model_name",
        default="maskformer-swin-tiny-ade",
        type=str,
        help=("Name of the MaskFormer model you'd like to convert",),
    )
    
    # 添加命令行参数：检查点路径
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/MaskFormer_checkpoints/MaskFormer-Swin-tiny-ADE20k/model.pkl",
        type=str,
        help="Path to the original state dict (.pth file).",
    )
    
    # 添加命令行参数：PyTorch 模型输出目录路径
    parser.add_argument(
        "--pytorch_dump_folder_path", 
        default=None, 
        type=str, 
        help="Path to the output PyTorch model directory."
    )
    
    # 添加命令行参数：是否推送模型到 🤗 hub
    parser.add_argument(
        "--push_to_hub", 
        action="store_true", 
        help="Whether or not to push the converted model to the 🤗 hub."
    )
    
    # 解析命令行参数，将结果存储在 args 变量中
    args = parser.parse_args()
    
    # 调用函数来转换 MaskFormer 模型的检查点
    convert_maskformer_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
```