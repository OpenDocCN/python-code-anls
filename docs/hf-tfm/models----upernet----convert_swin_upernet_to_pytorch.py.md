# `.\transformers\models\upernet\convert_swin_upernet_to_pytorch.py`

```
# 设置文件编码格式为 utf-8
# 版权声明，引用自 Apache License, Version 2.0
# 该脚本用于将来自 mmsegmentation 仓库的 Swin Transformer + UperNet 检查点转换为 Hugging Face 的格式
# 查看 mmsegmentation 仓库，获取更多信息
import argparse  # 导入解析命令行参数模块
import json  # 导入 json 模块

import requests  # 导入 requests 模块
import torch  # 导入 torch
from huggingface_hub import hf_hub_download  # 从 huggingface_hub 导入 hf_hub_download 函数
from PIL import Image  # 从 PIL 模块导入 Image 类

# 导入 Hugging Face 的图像处理器，SwinConfig 和 UperNetConfig 类，以及 UperNetForSemanticSegmentation 类
from transformers import SegformerImageProcessor, SwinConfig, UperNetConfig, UperNetForSemanticSegmentation


# 定义一个函数，根据给定的模型名称获取 UperNet 的配置信息
def get_upernet_config(model_name):
    auxiliary_in_channels = 384  # 定义辅助输入通道数
    window_size = 7  # 定义窗口大小
    if "tiny" in model_name:  # 如果模型名称中包含 "tiny"
        embed_dim = 96  # 定义嵌入维度
        depths = (2, 2, 6, 2)  # 定义深度
        num_heads = (3, 6, 12, 24)  # 定义头的数量
    elif "small" in model_name:  # 如果模型名称中包含 "small"
        embed_dim = 96  # 定义嵌入维度
        depths = (2, 2, 18, 2)  # 定义深度
        num_heads = (3, 6, 12, 24)  # 定义头的数量
    elif "base" in model_name:  # 如果模型名称中包含 "base"
        embed_dim = 128  # 定义嵌入维度
        depths = (2, 2, 18, 2)  # 定义深度
        num_heads = (4, 8, 16, 32)  # 定义头的数量
        window_size = 12  # 重新定义窗口大小
        auxiliary_in_channels = 512  # 重新定义辅助输入通道数
    elif "large" in model_name:  # 如果模型名称中包含 "large"
        embed_dim = 192  # 定义嵌入维度
        depths = (2, 2, 18, 2)  # 定义深度
        num_heads = (6, 12, 24, 48)  # 定义头的数量
        window_size = 12  # 重新定义窗口大小
        auxiliary_in_channels = 768  # 重新定义辅助输入通道数

    # 设定标签信息
    num_labels = 150  # 定义标签数量
    repo_id = "huggingface/label-files"  # 仓库 ID
    filename = "ade20k-id2label.json"  # 文件名
    # 读取并解析 ade20k-id2label.json，将标签 ID 映射到标签名称
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}  # 转换为整数 ID
    label2id = {v: k for k, v in id2label.items()}  # 获取标签名称映射到 ID 的字典

    # 配置 SwinTransformer 的配置信息
    backbone_config = SwinConfig(
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )
    # 配置 UperNet 的配置信息
    config = UperNetConfig(
        backbone_config=backbone_config,
        auxiliary_in_channels=auxiliary_in_channels,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    return config  # 返回配置信息


# 定义一个函数，创建要重命名的密钥列表
def create_rename_keys(config):
    rename_keys = []  # 初始化重命名列表

    # 注释被暂时关闭
    # stem
    # 将原始名称左侧的密钥名映射到右侧的我们指定的名称
    rename_keys.append(("backbone.patch_embed.projection.weight", "backbone.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("backbone.patch_embed.projection.bias", "backbone.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("backbone.patch_embed.norm.weight", "backbone.embeddings.norm.weight"))
    # 将指定键名重新命名，将("backbone.patch_embed.norm.bias", "backbone.embeddings.norm.bias")添加到重命名列表中
    rename_keys.append(("backbone.patch_embed.norm.bias", "backbone.embeddings.norm.bias"))
    
    # 遍历各个阶段，内部遍历各个块，将不同阶段的权重与偏置进行重命名
    for i in range(len(config.backbone_config.depths)):
        for j in range(config.backbone_config.depths[i]):
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.norm1.weight", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_before.weight"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.norm1.bias", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_before.bias"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.attn.w_msa.relative_position_bias_table", f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.relative_position_bias_table"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.attn.w_msa.relative_position_index", f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.relative_position_index"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.attn.w_msa.proj.weight", f"backbone.encoder.layers.{i}.blocks.{j}.attention.output.dense.weight"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.attn.w_msa.proj.bias", f"backbone.encoder.layers.{i}.blocks.{j}.attention.output.dense.bias"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.norm2.weight", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_after.weight"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.norm2.bias", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_after.bias"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.ffn.layers.0.0.weight", f"backbone.encoder.layers.{i}.blocks.{j}.intermediate.dense.weight"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.ffn.layers.0.0.bias", f"backbone.encoder.layers.{i}.blocks.{j}.intermediate.dense.bias"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.ffn.layers.1.weight", f"backbone.encoder.layers.{i}.blocks.{j}.output.dense.weight"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.ffn.layers.1.bias", f"backbone.encoder.layers.{i}.blocks.{j}.output.dense.bias"))

        # 如果当前阶段小于3，将下采样相关的权重与偏置重命名
        if i < 3:
            rename_keys.append((f"backbone.stages.{i}.downsample.reduction.weight", f"backbone.encoder.layers.{i}.downsample.reduction.weight"))
            rename_keys.append((f"backbone.stages.{i}.downsample.norm.weight", f"backbone.encoder.layers.{i}.downsample.norm.weight"))
            rename_keys.append((f"backbone.stages.{i}.downsample.norm.bias", f"backbone.encoder.layers.{i}.downsample.norm.bias"))
            
        # 将当前阶段的权重与偏置进行重命名
        rename_keys.append((f"backbone.norm{i}.weight", f"backbone.hidden_states_norms.stage{i+1}.weight"))
        rename_keys.append((f"backbone.norm{i}.bias", f"backbone.hidden_states_norms.stage{i+1}.bias"))

    # decode head
    # 将一组元组添加到列表 `rename_keys` 中，用于重命名模型中的特定键
    rename_keys.extend(
        [
            # 将 `decode_head.conv_seg.weight` 重命名为 `decode_head.classifier.weight`
            ("decode_head.conv_seg.weight", "decode_head.classifier.weight"),
            # 将 `decode_head.conv_seg.bias` 重命名为 `decode_head.classifier.bias`
            ("decode_head.conv_seg.bias", "decode_head.classifier.bias"),
            # 将 `auxiliary_head.conv_seg.weight` 重命名为 `auxiliary_head.classifier.weight`
            ("auxiliary_head.conv_seg.weight", "auxiliary_head.classifier.weight"),
            # 将 `auxiliary_head.conv_seg.bias` 重命名为 `auxiliary_head.classifier.bias`
            ("auxiliary_head.conv_seg.bias", "auxiliary_head.classifier.bias"),
        ]
    )
    # fmt: on

    # 返回重命名后的键列表
    return rename_keys
# 将字典 dct 中键名为 old 的键值对弹出，并赋值给变量 val
def rename_key(dct, old, new):
    val = dct.pop(old)
    # 将键名为 new，值为 val 的键值对添加到字典 dct 中

# 根据每个编码器层的矩阵将其划分为查询(query)、键(keys)和值(values)
def read_in_q_k_v(state_dict, backbone_config):
    # 计算每个深度层次的特征数量
    num_features = [int(backbone_config.embed_dim * 2**i) for i in range(len(backbone_config.depths))]
    for i in range(len(backbone_config.depths)):
        dim = num_features[i]
        for j in range(backbone_config.depths[i]):
            # 读取输入投影层的权重 + 偏置（在原始实现中，这是一个单矩阵 + 偏置）
            in_proj_weight = state_dict.pop(f"backbone.stages.{i}.blocks.{j}.attn.w_msa.qkv.weight")
            in_proj_bias = state_dict.pop(f"backbone.stages.{i}.blocks.{j}.attn.w_msa.qkv.bias")
            # 接下来，按顺序（查询、键、值）将权重和偏置添加到状态字典
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.query.weight"] = in_proj_weight[:dim, :]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.query.bias"] = in_proj_bias[: dim]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.key.weight"] = in_proj_weight[
                dim : dim * 2, :
            ]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.key.bias"] = in_proj_bias[
                dim : dim * 2
            ]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.value.weight"] = in_proj_weight[
                -dim :, :
            ]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.value.bias"] = in_proj_bias[-dim :]
            

# 更正反折叠减少顺序
def correct_unfold_reduction_order(x):
    out_channel, in_channel = x.shape
    x = x.reshape(out_channel, 4, in_channel // 4)
    x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
    return x

# 反转正确的反解折减少顺序
def reverse_correct_unfold_reduction_order(x):
    out_channel, in_channel = x.shape
    x = x.reshape(out_channel, in_channel // 4, 4)
    x = x[:, :, [0, 2, 1, 3]].transpose(1, 2).reshape(out_channel, in_channel)
    return x

# 更正反解折规范顺序
def correct_unfold_norm_order(x):
    in_channel = x.shape[0]
    x = x.reshape(4, in_channel // 4)
    x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
    return x

# 解决此版本不兼容性问题，由于新实现使用 nn.Unfold 进行下采样操作。
# 已解决，见此处：https://github.com/open-mmlab/mmdetection/blob/31c84958f54287a8be2b99cbf87a6dcf12e57753/mmdet/models/utils/ckpt_convert.py#L96
def reverse_correct_unfold_norm_order(x):
    in_channel = x.shape[0]
    x = x.reshape(in_channel // 4, 4)
    x = x[:, [0, 2, 1, 3]].transpose(0, 1).reshape(in_channel)
    return x

# 转换 upernet 检查点
def convert_upernet_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    # 定义模型名称到URL的映射
    model_name_to_url = {
        "upernet-swin-tiny": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth",
        "upernet-swin-small": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192015-ee2fff1c.pth",
        "upernet-swin-base": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth",
        "upernet-swin-large": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k_20220318_091743-9ba68901.pth",
    }
    # 获取模型的检查点URL
    checkpoint_url = model_name_to_url[model_name]
    # 从 URL 加载模型的状态字典
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", file_name=model_name)["state_dict"]

    # 打印状态字典中的每个键和对应的参数形状
    for name, param in state_dict.items():
        print(name, param.shape)

    # 获取UperNet的配置
    config = get_upernet_config(model_name)
    # 根据配置创建UperNet模型
    model = UperNetForSemanticSegmentation(config)
    # 设置模型为评估模式
    model.eval()

    # 替换状态字典中的键名，将"bn"替换为"batch_norm"
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if "bn" in key:
            key = key.replace("bn", "batch_norm")
        state_dict[key] = val

    # 重命名状态字典中的键名
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config.backbone_config)

    # 修复下采样参数
    for key, value in state_dict.items():
        if "downsample" in key:
            if "reduction" in key:
                state_dict[key] = reverse_correct_unfold_reduction_order(value)
            if "norm" in key:
                state_dict[key] = reverse_correct_unfold_norm_order(value)

    # 加载模型的状态字典
    model.load_state_dict(state_dict)

    # 在图像上进行验证
    url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
    # 从URL中获取并转换图像为RGB格式
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # 创建Segformer图像处理器并获取像素值张量
    processor = SegformerImageProcessor()
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # 禁用梯度计算，使用模型生成预测结果
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits

    # 打印预测结果的形状和部分值
    print(logits.shape)
    print("First values of logits:", logits[0, 0, :3, :3])
    # 对预测结果进行断言验证
    if model_name == "upernet-swin-tiny":
        expected_slice = torch.tensor(
            [[-7.5958, -7.5958, -7.4302], [-7.5958, -7.5958, -7.4302], [-7.4797, -7.4797, -7.3068]]
        )
    # 如果模型名为 "upernet-swin-small"，则设置期望的切片数据
    elif model_name == "upernet-swin-small":
        expected_slice = torch.tensor(
            [[-7.1921, -7.1921, -6.9532], [-7.1921, -7.1921, -6.9532], [-7.0908, -7.0908, -6.8534]]
        )
    # 如果模型名为 "upernet-swin-base"，则设置期望的切片数据
    elif model_name == "upernet-swin-base":
        expected_slice = torch.tensor(
            [[-6.5851, -6.5851, -6.4330], [-6.5851, -6.5851, -6.4330], [-6.4763, -6.4763, -6.3254]]
        )
    # 如果模型名为 "upernet-swin-large"，则设置期望的切片数据
    elif model_name == "upernet-swin-large":
        expected_slice = torch.tensor(
            [[-7.5297, -7.5297, -7.3802], [-7.5297, -7.5297, -7.3802], [-7.4044, -7.4044, -7.2586]]
        )
    # 打印模型输出的logits的前三行三列数据
    print("Logits:", outputs.logits[0, 0, :3, :3])
    # 检查模型输出的logits的前三行三列数据是否接近期望的切片数据，允许的误差为1e-4
    assert torch.allclose(outputs.logits[0, 0, :3, :3], expected_slice, atol=1e-4)
    # 打印信息，表示判断结果正确
    print("Looks ok!")

    # 如果存在PyTorch模型保存文件夹路径
    if pytorch_dump_folder_path is not None:
        # 打印信息，表示正在保存模型和处理器到指定路径
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        # 保存模型到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 打印信息，表示正在保存处理器到指定路径
        print(f"Saving processor to {pytorch_dump_folder_path}")
        # 保存处理器到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到hub
    if push_to_hub:
        # 打印信息，表示正在推送模型和处理器到hub
        print(f"Pushing model and processor for {model_name} to hub")
        # 将模型推送到hub的openmmlab/{model_name}路径
        model.push_to_hub(f"openmmlab/{model_name}")
        # 将处理器推送到hub的openmmlab/{model_name}路径
        processor.push_to_hub(f"openmmlab/{model_name}")
# 如果脚本作为主程序运行，则执行以下代码
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--model_name",  # 模型名称参数
        default="upernet-swin-tiny",  # 默认值为"upernet-swin-tiny"
        type=str,  # 参数类型为字符串
        choices=[f"upernet-swin-{size}" for size in ["tiny", "small", "base", "large"]],  # 可选值为不同尺寸的模型名称列表
        help="Name of the Swin + UperNet model you'd like to convert.",  # 参数的帮助信息
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",  # PyTorch 模型输出目录参数
        default=None,  # 默认值为 None
        type=str,  # 参数类型为字符串
        help="Path to the output PyTorch model directory."  # 参数的帮助信息
    )
    parser.add_argument(
        "--push_to_hub",  # 推送至 Hub 参数
        action="store_true",  # 如果设置，将该参数值设为 True
        help="Whether or not to push the converted model to the 🤗 hub."  # 参数的帮助信息
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将模型转换为 PyTorch 模型
    convert_upernet_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```