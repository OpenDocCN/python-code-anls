# `.\models\dpt\convert_dpt_beit_to_hf.py`

```py
# 设置编码格式为utf-8
# 版权声明，使用Apache License 2.0授权
# 引入必要的库和模块
# 设置日志输出级别为info
# 获取logger对象

import argparse  # 导入参数解析模块
from pathlib import Path  # 导入路径操作模块
import requests  # 导入requests模块，用于HTTP请求
import torch  # 导入PyTorch
from PIL import Image  # 导入图像处理模块PIL
from transformers import BeitConfig, DPTConfig, DPTForDepthEstimation, DPTImageProcessor  # 导入transformers库中的配置和模型
from transformers.utils import logging  # 导入transformers库中的日志模块

logging.set_verbosity_info()  # 设置日志输出级别为info
logger = logging.get_logger(__name__)  # 获取当前模块的日志输出对象

# 定义函数，根据模型名获取相关配置
def get_dpt_config(model_name):
    hidden_size = 768  # 隐藏层大小
    num_hidden_layers = 12  # 隐藏层层数
    num_attention_heads = 12  # 注意力头数
    intermediate_size = 3072  # 隐层尺寸
    out_features = ["stage3", "stage6", "stage9", "stage12"]  # 输出特征

    if "large" in model_name:  # 如果模型名包含'large'
        hidden_size = 1024  # 隐藏层大小改为1024
        num_hidden_layers = 24  # 隐藏层层数改为24
        num_attention_heads = 16  # 注意力头数改为16
        intermediate_size = 4096  # 隐层尺寸改为4096
        out_features = ["stage6", "stage12", "stage18", "stage24"]  # 输出特征改为该列表

    # 根据模型名选择图像尺寸
    if "512" in model_name:  # 如果模型名包含'512'
        image_size = 512  # 图像尺寸为512
    elif "384" in model_name:  # 如果模型名包含'384'
        image_size = 384  # 图像尺寸为384
    else:
        raise ValueError("Model not supported")  # 模型名不符合预期，引发值错误异常

    # 设置Beit模型的配置
    backbone_config = BeitConfig(
        image_size=image_size,  # 图像尺寸
        num_hidden_layers=num_hidden_layers,  # 隐藏层层数
        hidden_size=hidden_size,  # 隐藏层大小
        intermediate_size=intermediate_size,  # 隐层尺寸
        num_attention_heads=num_attention_heads,  # 注意力头数
        use_relative_position_bias=True,  # 使用相对位置偏置
        reshape_hidden_states=False,  # 不改变隐藏状态形状
        out_features=out_features,  # 输出特征
    )

    # 设置DPT模型的配置
    neck_hidden_sizes = [256, 512, 1024, 1024] if "large" in model_name else [96, 192, 384, 768]  # 设置颈部隐藏层大小
    config = DPTConfig(backbone_config=backbone_config, neck_hidden_sizes=neck_hidden_sizes)  # 使用Beit配置和颈部隐藏层大小构建DPT配置

    return config, image_size  # 返回配置和图像尺寸

# 定义函数，创建重命名键列表
def create_rename_keys(config):
    rename_keys = []  # 初始化重命名键列表

    # fmt: off
    # stem
    rename_keys.append(("pretrained.model.cls_token", "backbone.embeddings.cls_token"))  # 添加键值对到重命名键列表
    rename_keys.append(("pretrained.model.patch_embed.proj.weight", "backbone.embeddings.patch_embeddings.projection.weight"))  # 添加键值对到重命名键列表
    rename_keys.append(("pretrained.model.patch_embed.proj.bias", "backbone.embeddings.patch_embeddings.projection.bias"))  # 添加键值对到重命名键列表

    # Transfomer encoder
    # 循环遍历backbone_config.num_hidden_layers次，每次循环进行以下操作
    for i in range(config.backbone_config.num_hidden_layers):
        # 根据当前循环次数i生成相应的键值对，并添加到rename_keys列表中
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

    # 循环遍历4次，每次循环进行以下操作
    for i in range(4):
        # 根据当前循环次数i生成相应的键值对，并添加到rename_keys列表中
        rename_keys.append((f"pretrained.act_postprocess{i+1}.0.project.0.weight", f"neck.reassemble_stage.readout_projects.{i}.0.weight"))
        rename_keys.append((f"pretrained.act_postprocess{i+1}.0.project.0.bias", f"neck.reassemble_stage.readout_projects.{i}.0.bias"))
        rename_keys.append((f"pretrained.act_postprocess{i+1}.3.weight", f"neck.reassemble_stage.layers.{i}.projection.weight"))
        rename_keys.append((f"pretrained.act_postprocess{i+1}.3.bias", f"neck.reassemble_stage.layers.{i}.projection.bias"))

        # 如果当前循环次数i不等于2
        if i != 2:
            # 根据当前循环次数i生成相应的键值对，并添加到rename_keys列表中
            rename_keys.append((f"pretrained.act_postprocess{i+1}.4.weight", f"neck.reassemble_stage.layers.{i}.resize.weight"))
            rename_keys.append((f"pretrained.act_postprocess{i+1}.4.bias", f"neck.reassemble_stage.layers.{i}.resize.bias"))

    # 创建一个字典mapping
    mapping = {1:3, 2:2, 3:1, 4:0}
    # 遍历范围为1到4的数字序列，根据映射关系生成重命名键值对，用于对模型参数重命名
    for i in range(1, 5):
        # 获取映射关系中的值
        j = mapping[i]
        # 更新重命名键值对列表，将原始参数名与目标参数名对应起来
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

    # 对于scratch convolutions执行重命名
    for i in range(4):
        # 更新重命名键值对列表，将原始参数名与目标参数名对应起来
        rename_keys.append((f"scratch.layer{i+1}_rn.weight", f"neck.convs.{i}.weight"))

    # 处理头部参数的重命名
    for i in range(0, 5, 2):
        # 更新重命名键值对列表，将原始参数名与目标参数名对应起来
        rename_keys.append((f"scratch.output_conv.{i}.weight", f"head.head.{i}.weight"))
        rename_keys.append((f"scratch.output_conv.{i}.bias", f"head.head.{i}.bias"))

    # 返回所有的重命名键值对
    return rename_keys
# 从状态字典中删除特定的键
def remove_ignore_keys_(state_dict):
    # 定义要忽略的键列表
    ignore_keys = ["pretrained.model.head.weight", "pretrained.model.head.bias"]
    # 遍历要忽略的键列表
    for k in ignore_keys:
        # 如果键存在于状态字典中，则删除该键
        state_dict.pop(k, None)


# 将每个编码器层的矩阵拆分为查询、键和值
def read_in_q_k_v(state_dict, config):
    # 获取隐藏层大小
    hidden_size = config.backbone_config.hidden_size
    # 遍历编码器层数
    for i in range(config.backbone_config.num_hidden_layers):
        # 读取输入投影层的权重和偏置（在原始实现中，这是一个单矩阵加偏置）
        in_proj_weight = state_dict.pop(f"pretrained.model.blocks.{i}.attn.qkv.weight")
        q_bias = state_dict.pop(f"pretrained.model.blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"pretrained.model.blocks.{i}.attn.v_bias")
        # 将查询、键和值（按顺序）添加到状态字典中
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.bias"] = v_bias


# 重命名键
def rename_key(dct, old, new):
    # 弹出旧键对应的值
    val = dct.pop(old)
    # 将该值与新键相关联
    dct[new] = val


# 在一张可爱猫咪的图像上验证我们的结果
def prepare_img():
    # 图片的 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 打开图片
    im = Image.open(requests.get(url, stream=True).raw)
    # 返回图片
    return im


# 转换 DPT 检查点
@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    """
    复制/粘贴/调整模型的权重到我们的 DPT 结构。
    """

    # DPT 模型名称到 URL 的映射
    name_to_url = {
        "dpt-beit-large-512": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt",
        "dpt-beit-large-384": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_384.pt",
        "dpt-beit-base-384": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_base_384.pt",
    }

    # 根据模型名称获取检查点的 URL 和配置
    checkpoint_url = name_to_url[model_name]
    config, image_size = get_dpt_config(model_name)
    # 从 URL 加载原始状态字典
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    # 删除特定的键
    remove_ignore_keys_(state_dict)
    # 重命名键
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # 读取 qkv 矩阵
    read_in_q_k_v(state_dict, config)

    # 加载 HuggingFace 模型
    model = DPTForDepthEstimation(config)
    # 加载状态字典到模型，允许缺失键
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # 打印缺失的键
    print("Missing keys:", missing_keys)
    # 输出未预期的键列表
    print("Unexpected keys:", unexpected_keys)
    # 断言确保缺失的键列表为空
    assert missing_keys == []
    # 将模型设置为评估模式
    model.eval()

    # 创建图像处理器对象，设置图像大小，不保持宽高比，确保大小为32的倍数
    processor = DPTImageProcessor(
        size={"height": image_size, "width": image_size}, keep_aspect_ratio=False, ensure_multiple_of=32
    )

    # 准备图像数据
    image = prepare_img()
    # 使用图像处理器处理图像，返回PyTorch张量
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # 输出像素值的前3x3个值
    print("First values of pixel values:", pixel_values[0, 0, :3, :3])
    # 输出像素值的平均值
    print("Mean of pixel values:", pixel_values.mean().item())
    # 输出像素值的形状
    print("Shape of pixel values:", pixel_values.shape)

    # 导入所需的库和模块
    import requests
    from PIL import Image
    from torchvision import transforms

    # 从URL获取图像数据
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # 创建预处理转换管道
    transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    # 对图像进行转换，返回PyTorch张量
    pixel_values = transforms(image).unsqueeze(0)

    # 前向传播
    with torch.no_grad():
        outputs = model(pixel_values)

    # 获取预测的深度值
    predicted_depth = outputs.predicted_depth

    # 输出预测深度的形状和部分值
    print("Shape of predicted depth:", predicted_depth.shape)
    print("First values of predicted depth:", predicted_depth[0, :3, :3])

    # 断言预测深度的形状和部分值与期望值相匹配
    if model_name == "dpt-beit-large-512":
        # 断言大模型的预测深度形状和部分值与预期值相匹配
        expected_shape = torch.Size([1, 512, 512])
        expected_slice = torch.tensor(
            [[2804.6260, 2792.5708, 2812.9263], [2772.0288, 2780.1118, 2796.2529], [2748.1094, 2766.6558, 2766.9834]]
        )
    elif model_name == "dpt-beit-large-384":
        # 断言大384模型的预测深度形状和部分值与预期值相匹配
        expected_shape = torch.Size([1, 384, 384])
        expected_slice = torch.tensor(
            [[1783.2273, 1780.5729, 1792.6453], [1759.9817, 1765.5359, 1778.5002], [1739.1633, 1754.7903, 1757.1990]],
        )
    elif model_name == "dpt-beit-base-384":
        # 断言基础384模型的预测深度形状和部分值与预期值相匹配
        expected_shape = torch.Size([1, 384, 384])
        expected_slice = torch.tensor(
            [[2898.4482, 2891.3750, 2904.8079], [2858.6685, 2877.2615, 2894.4507], [2842.1235, 2854.1023, 2861.6328]],
        )

    assert predicted_depth.shape == torch.Size(expected_shape)
    assert torch.allclose(predicted_depth[0, :3, :3], expected_slice)
    # 输出结果看起来正常
    print("Looks ok!")

    # 如果指定了PyTorch转储路径
    if pytorch_dump_folder_path is not None:
        # 确保路径存在，不存在则创建
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 输出保存模型和处理器到指定路径
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        # 保存模型和处理器到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
    # 如果需要推送到代码库
    if push_to_hub:
        # 打印信息，提示正在将模型和处理器推送到代码库
        print("Pushing model and processor to hub...")
        # 将模型推送到指定的代码库
        model.push_to_hub(repo_id=f"nielsr/{model_name}")
        # 将处理器推送到指定的代码库
        processor.push_to_hub(repo_id=f"nielsr/{model_name}")
# 如果当前脚本被作为主程序运行
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数
    parser.add_argument(
        "--model_name",
        default="dpt-beit-large-512",
        type=str,
        choices=["dpt-beit-large-512", "dpt-beit-large-384", "dpt-beit-base-384"],
        help="Name of the model you'd like to convert.",
    )
    # 添加参数：输出的 PyTorch 模型目录路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    # 添加参数：是否在转换后将模型推送到 hub
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub after conversion.",
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将 DPT 模型检查点转换为 PyTorch 模型
    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```