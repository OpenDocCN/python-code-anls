# `.\models\dpt\convert_dpt_swinv2_to_hf.py`

```py
# 设置编码为 UTF-8
# 版权声明
# 授权为 Apache License, Version 2.0
# 创建了一个程序，根据指定的模型名称返回 DPTConfig 和图像大小
import argparse
from pathlib import Path
import requests
import torch
from PIL import Image
# 导入所需的类和函数
from transformers import DPTConfig, DPTForDepthEstimation, DPTImageProcessor, Swinv2Config
from transformers.utils import logging
# 设置日志级别为 info
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义根据模型名称返回 DPTConfig 和图像大小的函数
def get_dpt_config(model_name):
    if "tiny" in model_name:
        embed_dim = 96
        depths = (2, 2, 6, 2)
        num_heads = (3, 6, 12, 24)
        window_size = 16
        pretrained_window_sizes = (0, 0, 0, 0)
    elif "base" in model_name:
        embed_dim = 128
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)
        window_size = 24
        pretrained_window_sizes = (12, 12, 12, 6)
    elif "large" in model_name:
        embed_dim = 192
        depths = (2, 2, 18, 2)
        num_heads = (6, 12, 24, 48)
        window_size = 24
        pretrained_window_sizes = (12, 12, 12, 6)

    # 根据模型名称设置图像大小和其他参数
    if "384" in model_name:
        image_size = 384
    elif "256" in model_name:
        image_size = 256
    else:
        raise ValueError("Model not supported, to do")

    # 配置 Swin Transformer 的属性
    backbone_config = Swinv2Config(
        image_size=image_size,
        embed_dim=embed_dim,
        depths=depths,
        window_size=window_size,
        pretrained_window_sizes=pretrained_window_sizes,
        num_heads=num_heads,
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )

    # 根据模型名称设置颈部隐藏层大小
    if model_name == "dpt-swinv2-tiny-256":
        neck_hidden_sizes = [96, 192, 384, 768]
    elif model_name == "dpt-swinv2-base-384":
        neck_hidden_sizes = [128, 256, 512, 1024]
    elif model_name == "dpt-swinv2-large-384":
        neck_hidden_sizes = [192, 384, 768, 1536]

    # 创建 DPTConfig
    config = DPTConfig(backbone_config=backbone_config, neck_hidden_sizes=neck_hidden_sizes)

    return config, image_size

# 创建用于重命名键的函数
# 在这里列出所有键值对，左侧是原始名称，右侧是我们指定的名称
def create_rename_keys(config):
    rename_keys = []

    # fmt: off
    # stem
    # 添加字典元组到列表，用于重命名模型的键值对
    rename_keys.append(("pretrained.model.patch_embed.proj.weight", "backbone.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("pretrained.model.patch_embed.proj.bias", "backbone.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("pretrained.model.patch_embed.norm.weight", "backbone.embeddings.norm.weight"))
    rename_keys.append(("pretrained.model.patch_embed.norm.bias", "backbone.embeddings.norm.bias"))
    
    # transformer encoder
    # 注意：非 Transformer 的骨干模型（如 Swinv2、LeViT 等）不需要激活后处理（输出投影层 + 尺度调整模块）
    
    # refinenet（这部分比较棘手）
    # 定义映射字典，将旧的键映射到新的键
    mapping = {1:3, 2:2, 3:1, 4:0}
    
    # 循环遍历四个 refinenet 层
    for i in range(1, 5):
        # 根据映射字典确定重命名的索引
        j = mapping[i]
        # 添加字典元组到列表，用于重命名模型的键值对
        rename_keys.append((f"scratch.refinenet{i}.out_conv.weight", f"neck.fusion_stage.layers.{j}.projection.weight"))
        rename_keys.append((f"scratch.refinenet{i}.out_conv.bias", f"neck.fusion_stage.layers.{j}.projection.bias"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit1.conv1.weight", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution1.weight"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit1.conv1.bias", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution1.bias"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit1.conv2.weight", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution2.weight"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit1.conv2.bias", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution2.bias"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit2.conv1.weight", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution1.weight"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit2.conv1.bias", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution1.bias"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit2.conv2.weight", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution2.weight"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit2.conv2.bias", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution2.bias"))
    
    # scratch convolutions
    # 循环遍历四个 scratch 层
    for i in range(4):
        # 添加字典元组到列表，用于重命名模型的键值对
        rename_keys.append((f"scratch.layer{i+1}_rn.weight", f"neck.convs.{i}.weight"))
    
    # head
    # 循环遍历0到5（不包括5）的步长为2的数
    for i in range(0, 5, 2):
        # 添加字典元组到列表，用于重命名模型的键值对
        rename_keys.append((f"scratch.output_conv.{i}.weight", f"head.head.{i}.weight"))
        rename_keys.append((f"scratch.output_conv.{i}.bias", f"head.head.{i}.bias"))
    
    # 返回重命名后的键值对列表
    return rename_keys
# 从状态字典中移除指定的键，如果键不存在则不进行操作
def remove_ignore_keys_(state_dict):
    # 定义要移除的键列表
    ignore_keys = ["pretrained.model.head.weight", "pretrained.model.head.bias"]
    # 遍历要移除的键列表
    for k in ignore_keys:
        # 从状态字典中移除指定键，如果不存在则不进行操作
        state_dict.pop(k, None)


# 将每个编码器层的矩阵分解为查询（queries）、键（keys）和值（values）
def read_in_q_k_v(state_dict, config, model):
    # 遍历编码器层的深度列表
    for i in range(len(config.backbone_config.depths)):
        # 遍历每个深度的层数
        for j in range(config.backbone_config.depths[i]):
            # 获取当前编码器层的查询向量、键向量、值向量的维度
            dim = model.backbone.encoder.layers[i].blocks[j].attention.self.all_head_size
            # 读取输入投影层（在原始实现中，这是一个单矩阵加偏置的层）的权重和偏置
            in_proj_weight = state_dict.pop(f"pretrained.model.layers.{i}.blocks.{j}.attn.qkv.weight")
            # 将查询、键和值（按顺序）添加到状态字典中
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.query.weight"] = in_proj_weight[:dim, :]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.key.weight"] = in_proj_weight[
                dim : dim * 2, :
            ]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.value.weight"] = in_proj_weight[
                -dim:, :
            ]


# 重命名字典中的键
def rename_key(dct, old, new):
    # 弹出旧键对应的值
    val = dct.pop(old)
    # 将值与新键关联起来
    dct[new] = val


# 在一张可爱猫咪的图片上验证结果
def prepare_img():
    # 图片的 URL 地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 通过 URL 获取图片
    im = Image.open(requests.get(url, stream=True).raw)
    # 返回获取的图片
    return im


# 使用 torch.no_grad 装饰器，不对模型进行梯度跟踪
@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, verify_logits, push_to_hub):
    """
    将模型的权重复制/粘贴/调整到我们的 DPT 结构中。
    """

    # DPT 模型名称与对应的 URL 映射关系
    name_to_url = {
        "dpt-swinv2-tiny-256": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt",
        "dpt-swinv2-base-384": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_base_384.pt",
        "dpt-swinv2-large-384": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt",
    }

    # 根据模型名称获取对应的检查点 URL 地址
    checkpoint_url = name_to_url[model_name]
    # 根据模型名称获取 DPT 配置和图片大小
    config, image_size = get_dpt_config(model_name)
    # 从 URL 加载原始状态字典
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")

    # 加载 HuggingFace 模型
    model = DPTForDepthEstimation(config)

    # 移除特定的键
    remove_ignore_keys_(state_dict)
    # 重命名键
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # 读取查询-键-值矩阵
    read_in_q_k_v(state_dict, config, model)

    # 加载模型状态字典，不严格要求匹配键
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # 打印缺失的键和意外的键
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    # 将模型设置为评估模式
    model.eval()

    # 在图片上检查输出
    # 创建 DPTImageProcessor 实例并指定图片大小
    processor = DPTImageProcessor(size={"height": image_size, "width": image_size})

    # 准备图片数据
    image = prepare_img()
    # 将图片数据传入处理器得到处理后的结果
    processor(image, return_tensors="pt")

    # 如果需要验证logits
    if verify_logits:
        # 导入必要的库
        from torchvision import transforms

        # 从 URL 获取图片数据
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        # 对图片进行预处理
        transforms = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        pixel_values = transforms(image).unsqueeze(0)

        # 前向传播
        with torch.no_grad():
            outputs = model(pixel_values)

        predicted_depth = outputs.predicted_depth

        print("Shape of predicted depth:", predicted_depth.shape)
        print("First values of predicted depth:", predicted_depth[0, :3, :3])

        # 断言logits是否符合预期值
        if model_name == "dpt-swinv2-base-384":
            expected_shape = torch.Size([1, 384, 384])
            expected_slice = torch.tensor(
                [
                    [1998.5575, 1997.3887, 2009.2981],
                    [1952.8607, 1979.6488, 2001.0854],
                    [1953.7697, 1961.7711, 1968.8904],
                ],
            )
        elif model_name == "dpt-swinv2-tiny-256":
            expected_shape = torch.Size([1, 256, 256])
            expected_slice = torch.tensor(
                [[978.9163, 976.5215, 978.5349], [974.1859, 971.7249, 975.8046], [971.3419, 970.3118, 971.6830]],
            )
        elif model_name == "dpt-swinv2-large-384":
            expected_shape = torch.Size([1, 384, 384])
            expected_slice = torch.tensor(
                [
                    [1203.7206, 1200.1495, 1197.8234],
                    [1196.2484, 1183.5033, 1186.4640],
                    [1178.8131, 1182.3260, 1174.3975],
                ],
            )

        # 断言预测的深度图的形状和数值是否符合预期
        assert predicted_depth.shape == torch.Size(expected_shape)
        assert torch.allclose(predicted_depth[0, :3, :3], expected_slice)
        print("Looks ok!")

    # 如果指定保存模型和处理器的路径
    if pytorch_dump_folder_path is not None:
        # 创建路径
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 保存模型和处理器到指定路径
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到hub
    if push_to_hub:
        print("Pushing model and processor to hub...")
        # 推送模型和处理器到hub
        model.push_to_hub(repo_id=f"Intel/{model_name}")
        processor.push_to_hub(repo_id=f"Intel/{model_name}")
# 如果当前脚本被直接运行，而不是被导入其他模块，那么执行以下操作
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必选参数
    parser.add_argument(
        "--model_name",
        default="dpt-swinv2-base-384",
        type=str,
        choices=["dpt-swinv2-tiny-256", "dpt-swinv2-base-384", "dpt-swinv2-large-384"],
        help="Name of the model you'd like to convert.",
    )
    # 添加参数：输出 PyTorch 模型目录的路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    # 添加参数：是否在转换后验证对数
    parser.add_argument(
        "--verify_logits",
        action="store_true",
        help="Whether to verify logits after conversion.",
    )
    # 添加参数：是否在转换后将模型推送到 Hub
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub after conversion.",
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用 convert_dpt_checkpoint 函数，传入解析后的参数
    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.verify_logits, args.push_to_hub)
```