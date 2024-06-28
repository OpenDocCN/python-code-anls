# `.\models\dpt\convert_dpt_beit_to_hf.py`

```
# coding=utf-8
# 版权所有 2023 年 HuggingFace Inc. 团队。
#
# 根据 Apache 许可证 2.0 版本许可，除非符合许可协议，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发本软件，
# 无任何明示或暗示的担保或条件。
# 请参阅许可协议了解具体语言的权限和限制。
"""从 MiDaS 仓库转换 DPT 3.1 检查点。URL：https://github.com/isl-org/MiDaS"""

import argparse  # 导入命令行参数解析库
from pathlib import Path  # 导入路径操作库

import requests  # 导入 HTTP 请求库
import torch  # 导入 PyTorch 深度学习库
from PIL import Image  # 导入图像处理库

from transformers import BeitConfig, DPTConfig, DPTForDepthEstimation, DPTImageProcessor  # 导入转换器库的相关组件
from transformers.utils import logging  # 导入转换器库的日志模块


logging.set_verbosity_info()  # 设置日志级别为信息
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def get_dpt_config(model_name):
    hidden_size = 768  # 隐藏层大小设为 768
    num_hidden_layers = 12  # 隐藏层层数设为 12
    num_attention_heads = 12  # 注意力头数设为 12
    intermediate_size = 3072  # 中间层大小设为 3072
    out_features = ["stage3", "stage6", "stage9", "stage12"]  # 输出特征设为 ["stage3", "stage6", "stage9", "stage12"]

    if "large" in model_name:
        hidden_size = 1024  # 如果模型名中包含 "large"，则将隐藏层大小设为 1024
        num_hidden_layers = 24  # 将隐藏层层数设为 24
        num_attention_heads = 16  # 将注意力头数设为 16
        intermediate_size = 4096  # 将中间层大小设为 4096
        out_features = ["stage6", "stage12", "stage18", "stage24"]  # 输出特征设为 ["stage6", "stage12", "stage18", "stage24"]

    if "512" in model_name:
        image_size = 512  # 如果模型名中包含 "512"，则将图像大小设为 512
    elif "384" in model_name:
        image_size = 384  # 如果模型名中包含 "384"，则将图像大小设为 384
    else:
        raise ValueError("Model not supported")  # 如果模型不支持，则引发值错误异常

    # 创建背景配置对象
    backbone_config = BeitConfig(
        image_size=image_size,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        use_relative_position_bias=True,
        reshape_hidden_states=False,
        out_features=out_features,
    )

    # 根据模型名称设置颈部隐藏层大小列表
    neck_hidden_sizes = [256, 512, 1024, 1024] if "large" in model_name else [96, 192, 384, 768]
    
    # 创建 DPT 配置对象
    config = DPTConfig(backbone_config=backbone_config, neck_hidden_sizes=neck_hidden_sizes)

    return config, image_size  # 返回配置对象和图像大小


# 此处列出所有要重命名的键（原始名称在左侧，我们的名称在右侧）
def create_rename_keys(config):
    rename_keys = []  # 初始化空的重命名键列表

    # fmt: off
    # stem
    rename_keys.append(("pretrained.model.cls_token", "backbone.embeddings.cls_token"))  # 添加重命名键：("pretrained.model.cls_token", "backbone.embeddings.cls_token")
    rename_keys.append(("pretrained.model.patch_embed.proj.weight", "backbone.embeddings.patch_embeddings.projection.weight"))  # 添加重命名键：("pretrained.model.patch_embed.proj.weight", "backbone.embeddings.patch_embeddings.projection.weight")
    rename_keys.append(("pretrained.model.patch_embed.proj.bias", "backbone.embeddings.patch_embeddings.projection.bias"))  # 添加重命名键：("pretrained.model.patch_embed.proj.bias", "backbone.embeddings.patch_embeddings.projection.bias")

    # Transfomer encoder
    # fmt: on
    # 遍历从配置中获取的隐藏层数量的范围
    for i in range(config.backbone_config.num_hidden_layers):
        # 添加转换后的键值对，将预训练模型中的参数映射到新的后骨干网络结构中的位置
        rename_keys.append((f"pretrained.model.blocks.{i}.gamma_1", f"backbone.encoder.layer.{i}.lambda_1"))
        rename_keys.append((f"pretrained.model.blocks.{i}.gamma_2", f"backbone.encoder.layer.{i}.lambda_2"))
        rename_keys.append((f"pretrained.model.blocks.{i}.norm1.weight", f"backbone.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"pretrained.model.blocks.{i}.norm1.bias", f"backbone.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append((f"pretrained.model.blocks.{i}.norm2.weight", f"backbone.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"pretrained.model.blocks.{i}.norm2.bias", f"backbone.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"pretrained.model.blocks.{i}.mlp.fc1.weight", f"backbone.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"pretrained.model.blocks.{i}.mlp.fc1.bias", f"backbone.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"pretrained.model.blocks.{i}.mlp.fc2.weight", f"backbone.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"pretrained.model.blocks.{i}.mlp.fc2.bias", f"backbone.encoder.layer.{i}.output.dense.bias"))
        rename_keys.append((f"pretrained.model.blocks.{i}.attn.proj.weight", f"backbone.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"pretrained.model.blocks.{i}.attn.proj.bias", f"backbone.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"pretrained.model.blocks.{i}.attn.relative_position_bias_table", f"backbone.encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_bias_table"))
        rename_keys.append((f"pretrained.model.blocks.{i}.attn.relative_position_index", f"backbone.encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_index"))

    # 激活后处理（读出投影 + 调整块）
    for i in range(4):
        # 读出投影权重和偏置的映射
        rename_keys.append((f"pretrained.act_postprocess{i+1}.0.project.0.weight", f"neck.reassemble_stage.readout_projects.{i}.0.weight"))
        rename_keys.append((f"pretrained.act_postprocess{i+1}.0.project.0.bias", f"neck.reassemble_stage.readout_projects.{i}.0.bias"))

        # 中间层投影权重和偏置的映射
        rename_keys.append((f"pretrained.act_postprocess{i+1}.3.weight", f"neck.reassemble_stage.layers.{i}.projection.weight"))
        rename_keys.append((f"pretrained.act_postprocess{i+1}.3.bias", f"neck.reassemble_stage.layers.{i}.projection.bias"))

        # 如果不是第二个块，映射调整块权重和偏置
        if i != 2:
            rename_keys.append((f"pretrained.act_postprocess{i+1}.4.weight", f"neck.reassemble_stage.layers.{i}.resize.weight"))
            rename_keys.append((f"pretrained.act_postprocess{i+1}.4.bias", f"neck.reassemble_stage.layers.{i}.resize.bias"))

    # RefineNet（这里有点棘手）
    mapping = {1:3, 2:2, 3:1, 4:0}
    # 遍历范围为 1 到 4，根据映射表 mapping 将每个 i 映射到 j
    for i in range(1, 5):
        j = mapping[i]
        # 向 rename_keys 列表中添加元组，将模型参数名从 scratch.refinenet{i} 映射到 neck.fusion_stage.layers.{j}
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

    # 遍历范围为 0 到 4，向 rename_keys 列表中添加元组，映射 scratch.layer{i+1}_rn 到 neck.convs.{i}
    for i in range(4):
        rename_keys.append((f"scratch.layer{i+1}_rn.weight", f"neck.convs.{i}.weight"))

    # 遍历范围为 0 到 5（步长为 2），向 rename_keys 列表中添加元组，映射 scratch.output_conv.{i} 到 head.head.{i}
    for i in range(0, 5, 2):
        rename_keys.append((f"scratch.output_conv.{i}.weight", f"head.head.{i}.weight"))
        rename_keys.append((f"scratch.output_conv.{i}.bias", f"head.head.{i}.bias"))

    # 返回存储了所有模型参数重命名信息的 rename_keys 列表
    return rename_keys
# 从给定的状态字典中移除特定的键
def remove_ignore_keys_(state_dict):
    ignore_keys = ["pretrained.model.head.weight", "pretrained.model.head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)

# 将每个编码器层的矩阵拆分为查询（queries）、键（keys）和值（values）
def read_in_q_k_v(state_dict, config):
    # 获取隐藏层大小
    hidden_size = config.backbone_config.hidden_size
    # 遍历编码器层次的数量
    for i in range(config.backbone_config.num_hidden_layers):
        # 读取输入投影层的权重和偏置（在原始实现中，这是一个单独的矩阵加上偏置）
        in_proj_weight = state_dict.pop(f"pretrained.model.blocks.{i}.attn.qkv.weight")
        q_bias = state_dict.pop(f"pretrained.model.blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"pretrained.model.blocks.{i}.attn.v_bias")
        # 将查询（query）、键（key）和值（value）依次添加到状态字典中
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[hidden_size : hidden_size * 2, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.bias"] = v_bias

# 重命名字典中的键
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val

# 准备一张猫咪图片，用于验证我们的结果
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im

# 将模型的权重复制/粘贴/调整到我们的DPT结构中
@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our DPT structure.
    """

    # 定义基于URL的DPT配置
    name_to_url = {
        "dpt-beit-large-512": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt",
        "dpt-beit-large-384": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_384.pt",
        "dpt-beit-base-384": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_base_384.pt",
    }

    # 根据模型名称选择检查点URL
    checkpoint_url = name_to_url[model_name]
    # 获取DPT配置和图像大小
    config, image_size = get_dpt_config(model_name)
    # 从URL加载原始的状态字典
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    # 移除指定的键
    remove_ignore_keys_(state_dict)
    # 创建重命名键的映射
    rename_keys = create_rename_keys(config)
    # 遍历重命名映射并重命名键
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # 读取qkv矩阵
    read_in_q_k_v(state_dict, config)

    # 加载HuggingFace模型
    model = DPTForDepthEstimation(config)
    # 加载模型的状态字典，允许严格性检查
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # 打印缺失的键
    print("Missing keys:", missing_keys)
    # 打印出意外的键列表
    print("Unexpected keys:", unexpected_keys)
    # 确保缺失的键列表为空
    assert missing_keys == []
    # 将模型设置为评估模式
    model.eval()

    # 创建图像处理器对象，设定图像尺寸和其他参数
    # 这里设置 `keep_aspect_ratio=False`，因为当前的 BEiT 不支持任意窗口大小
    processor = DPTImageProcessor(
        size={"height": image_size, "width": image_size}, keep_aspect_ratio=False, ensure_multiple_of=32
    )

    # 准备图像数据
    image = prepare_img()
    # 使用图像处理器处理图像，返回像素值的张量表示
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # 打印像素值的一些信息
    print("First values of pixel values:", pixel_values[0, 0, :3, :3])
    print("Mean of pixel values:", pixel_values.mean().item())
    print("Shape of pixel values:", pixel_values.shape)

    # 导入必要的库和模块
    import requests
    from PIL import Image
    from torchvision import transforms

    # 从 URL 加载图像并使用 PIL 打开
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # 创建图像转换管道，包括调整大小和转换为张量
    transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    # 对图像进行转换处理，添加一个维度
    pixel_values = transforms(image).unsqueeze(0)

    # 前向传播，关闭梯度计算
    with torch.no_grad():
        outputs = model(pixel_values)

    # 获取预测的深度图
    predicted_depth = outputs.predicted_depth

    # 打印预测深度图的形状和部分值
    print("Shape of predicted depth:", predicted_depth.shape)
    print("First values of predicted depth:", predicted_depth[0, :3, :3])

    # 断言预测深度图的形状和部分值与预期相符
    if model_name == "dpt-beit-large-512":
        expected_shape = torch.Size([1, 512, 512])
        expected_slice = torch.tensor(
            [[2804.6260, 2792.5708, 2812.9263], [2772.0288, 2780.1118, 2796.2529], [2748.1094, 2766.6558, 2766.9834]]
        )
    elif model_name == "dpt-beit-large-384":
        expected_shape = torch.Size([1, 384, 384])
        expected_slice = torch.tensor(
            [[1783.2273, 1780.5729, 1792.6453], [1759.9817, 1765.5359, 1778.5002], [1739.1633, 1754.7903, 1757.1990]],
        )
    elif model_name == "dpt-beit-base-384":
        expected_shape = torch.Size([1, 384, 384])
        expected_slice = torch.tensor(
            [[2898.4482, 2891.3750, 2904.8079], [2858.6685, 2877.2615, 2894.4507], [2842.1235, 2854.1023, 2861.6328]],
        )

    # 断言预测的深度图的形状和部分值与期望的形状和值相等
    assert predicted_depth.shape == torch.Size(expected_shape)
    assert torch.allclose(predicted_depth[0, :3, :3], expected_slice)
    print("Looks ok!")

    # 如果指定了 PyTorch 模型保存文件夹路径，则保存模型和处理器
    if pytorch_dump_folder_path is not None:
        # 确保文件夹存在或创建
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 打印保存路径信息，并保存模型和处理器
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
    # 如果 push_to_hub 变量为真，则执行下面的代码块
    if push_to_hub:
        # 打印消息，指示正在将模型和处理器推送到 Hub
        print("Pushing model and processor to hub...")
        # 调用 model 对象的 push_to_hub 方法，将模型推送到指定的仓库ID
        model.push_to_hub(repo_id=f"nielsr/{model_name}")
        # 调用 processor 对象的 push_to_hub 方法，将处理器推送到指定的仓库ID
        processor.push_to_hub(repo_id=f"nielsr/{model_name}")
if __name__ == "__main__":
    # 如果脚本被直接运行而不是作为模块导入，则执行以下代码块
    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # Required parameters
    parser.add_argument(
        "--model_name",
        default="dpt-beit-large-512",
        type=str,
        choices=["dpt-beit-large-512", "dpt-beit-large-384", "dpt-beit-base-384"],
        help="Name of the model you'd like to convert.",
    )
    # 添加一个必需的参数，用于指定要转换的模型名称，提供默认值和选项列表

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    # 添加一个参数，用于指定输出 PyTorch 模型文件的目录路径，可选，默认为 None

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub after conversion.",
    )
    # 添加一个参数，用于指定是否在转换后将模型推送到指定的 hub 中，这是一个布尔标志参数

    args = parser.parse_args()
    # 解析命令行参数并将其存储在 args 对象中

    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
    # 调用函数 convert_dpt_checkpoint，传入解析得到的参数进行模型转换操作
```