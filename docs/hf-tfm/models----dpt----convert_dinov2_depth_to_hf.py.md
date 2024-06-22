# `.\models\dpt\convert_dinov2_depth_to_hf.py`

```py
# 设置文件编码为 utf-8
# 版权声明，使用 Apache 许可证 2.0 版本
# 获取完整的许可证文本
# 根据适用法律或协议所需，在符合许可证条件的情况下才能使用该文件
# 在“AS IS”基础上分发软件，没有任何形式的担保或条件，无论是明示的还是暗示的
# 查看适用于特定语言的许可证及限制条件
"""将原始存储库中的 DINOv2 + DPT 检查点转换为 HuggingFace 模型。URL: https://github.com/facebookresearch/dinov2/tree/main"""


import argparse
import itertools
import math
from pathlib import Path

import requests
import torch
from PIL import Image
from torchvision import transforms

from transformers import Dinov2Config, DPTConfig, DPTForDepthEstimation, DPTImageProcessor
from transformers.utils import logging


# 设置日志记录级别为信息
logging.set_verbosity_info()
# 获取 logger 对象
logger = logging.get_logger(__name__)


# 根据模型名称获取 DPT 配置
def get_dpt_config(model_name):
    if "small" in model_name:
        # 创建与 DINOv2 配置兼容的骨干网络配置
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-small", out_indices=[3, 6, 9, 12], apply_layernorm=False, reshape_hidden_states=False
        )
        neck_hidden_sizes = [48, 96, 192, 384]
    elif "base" in model_name:
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-base", out_indices=[3, 6, 9, 12], apply_layernorm=False, reshape_hidden_states=False
        )
        neck_hidden_sizes = [96, 192, 384, 768]
    elif "large" in model_name:
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-large", out_indices=[5, 12, 18, 24], apply_layernorm=False, reshape_hidden_states=False
        )
        neck_hidden_sizes = [128, 256, 512, 1024]
    elif "giant" in model_name:
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-giant", out_indices=[10, 20, 30, 40], apply_layernorm=False, reshape_hidden_states=False
        )
        neck_hidden_sizes = [192, 384, 768, 1536]
    else:
        raise NotImplementedError("To do")

    # 构建 DPT 配置
    config = DPTConfig(
        backbone_config=backbone_config,
        neck_hidden_sizes=neck_hidden_sizes,
        use_bias_in_fusion_residual=False,
        add_projection=True,
    )

    return config


# 列出所有需被重命名的 DPT 键（原始名称在左边，我们的名称在右边）
def create_rename_keys_dpt(config):
    rename_keys = []

    # fmt: off
    # 激活后处理（投影、读出投影 + 调整大小块）
    # 循环处理4次，为指定的键值对列表中添加键值对，用于重命名特定的权重和偏置
    for i in range(4):
        # 更新键值对列表，重命名指定位置的权重和偏置
        rename_keys.append((f"decode_head.reassemble_blocks.projects.{i}.conv.weight", f"neck.reassemble_stage.layers.{i}.projection.weight"))
        rename_keys.append((f"decode_head.reassemble_blocks.projects.{i}.conv.bias", f"neck.reassemble_stage.layers.{i}.projection.bias"))

        rename_keys.append((f"decode_head.reassemble_blocks.readout_projects.{i}.0.weight", f"neck.reassemble_stage.readout_projects.{i}.0.weight"))
        rename_keys.append((f"decode_head.reassemble_blocks.readout_projects.{i}.0.bias", f"neck.reassemble_stage.readout_projects.{i}.0.bias"))

        # 当i不等于2时，更新键值对列表，重命名指定位置的权重和偏置
        if i != 2:
            rename_keys.append((f"decode_head.reassemble_blocks.resize_layers.{i}.weight", f"neck.reassemble_stage.layers.{i}.resize.weight"))
            rename_keys.append((f"decode_head.reassemble_blocks.resize_layers.{i}.bias", f"neck.reassemble_stage.layers.{i}.resize.bias"))

    # fusion layers
    # 循环处理4次，为指定的键值对列表中添加键值对，用于重命名特定的权重和偏置
    for i in range(4):
        rename_keys.append((f"decode_head.fusion_blocks.{i}.project.conv.weight", f"neck.fusion_stage.layers.{i}.projection.weight"))
        rename_keys.append((f"decode_head.fusion_blocks.{i}.project.conv.bias", f"neck.fusion_stage.layers.{i}.projection.bias"))
        if i != 0:
            rename_keys.append((f"decode_head.fusion_blocks.{i}.res_conv_unit1.conv1.conv.weight", f"neck.fusion_stage.layers.{i}.residual_layer1.convolution1.weight"))
            rename_keys.append((f"decode_head.fusion_blocks.{i}.res_conv_unit1.conv2.conv.weight", f"neck.fusion_stage.layers.{i}.residual_layer1.convolution2.weight"))
        rename_keys.append((f"decode_head.fusion_blocks.{i}.res_conv_unit2.conv1.conv.weight", f"neck.fusion_stage.layers.{i}.residual_layer2.convolution1.weight"))
        rename_keys.append((f"decode_head.fusion_blocks.{i}.res_conv_unit2.conv2.conv.weight", f"neck.fusion_stage.layers.{i}.residual_layer2.convolution2.weight"))

    # neck convolutions
    # 循环处理4次，为指定的键值对列表中添加键值对，用于重命名特定的权重
    for i in range(4):
        rename_keys.append((f"decode_head.convs.{i}.conv.weight", f"neck.convs.{i}.weight"))

    # head
    # 更新键值对列表，重命名指定的权重和偏置
    rename_keys.append(("decode_head.project.conv.weight", "head.projection.weight"))
    rename_keys.append(("decode_head.project.conv.bias", "head.projection.bias"))

    # 循环处���5次，从0开始，每次增加2，为指定的键值对列表中添加键值对，用于重命名特定的权重和偏置
    for i in range(0, 5, 2):
        rename_keys.append((f"decode_head.conv_depth.head.{i}.weight", f"head.head.{i}.weight"))
        rename_keys.append((f"decode_head.conv_depth.head.{i}.bias", f"head.head.{i}.bias"))
    # fmt: on

    # 返回重命名后的键值对列表
    return rename_keys
# 列出要重命名的所有骨干键（左侧为原始名称，右侧为我们的名称）
def create_rename_keys_backbone(config):
    # 初始化重命名列表
    rename_keys = []

    # fmt: off
    # 对于补丁嵌入层
    rename_keys.append(("cls_token", "backbone.embeddings.cls_token"))
    rename_keys.append(("mask_token", "backbone.embeddings.mask_token"))
    rename_keys.append(("pos_embed", "backbone.embeddings.position_embeddings"))
    rename_keys.append(("patch_embed.proj.weight", "backbone.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("patch_embed.proj.bias", "backbone.embeddings.patch_embeddings.projection.bias"))

    # Transfomer 编码器
    for i in range(config.backbone_config.num_hidden_layers):
        # 层归一化
        rename_keys.append((f"blocks.{i}.norm1.weight", f"backbone.encoder.layer.{i}.norm1.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"backbone.encoder.layer.{i}.norm1.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"backbone.encoder.layer.{i}.norm2.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"backbone.encoder.layer.{i}.norm2.bias"))
        # MLP
        if config.backbone_config.use_swiglu_ffn:
            rename_keys.append((f"blocks.{i}.mlp.w12.weight", f"backbone.encoder.layer.{i}.mlp.w12.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w12.bias", f"backbone.encoder.layer.{i}.mlp.w12.bias"))
            rename_keys.append((f"blocks.{i}.mlp.w3.weight", f"backbone.encoder.layer.{i}.mlp.w3.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w3.bias", f"backbone.encoder.layer.{i}.mlp.w3.bias"))
        else:
            rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"backbone.encoder.layer.{i}.mlp.fc1.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"backbone.encoder.layer.{i}.mlp.fc1.bias"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"backbone.encoder.layer.{i}.mlp.fc2.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"backbone.encoder.layer.{i}.mlp.fc2.bias"))
        # 层缩放
        rename_keys.append((f"blocks.{i}.ls1.gamma", f"backbone.encoder.layer.{i}.layer_scale1.lambda1"))
        rename_keys.append((f"blocks.{i}.ls2.gamma", f"backbone.encoder.layer.{i}.layer_scale2.lambda1"))
        # 注意力投影层
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"backbone.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"backbone.encoder.layer.{i}.attention.output.dense.bias"))
    # fmt: on

    # 最后添加层归一化权重和偏置的重命名
    rename_keys.append(("norm.weight", "backbone.layernorm.weight"))
    rename_keys.append(("norm.bias", "backbone.layernorm.bias"))

    # 返回重命名列表
    return rename_keys


# 将每个编码器层的矩阵分解为查询、键和值
def read_in_q_k_v(state_dict, config):
        # 遍历配置中指定的主干网络隐藏层数量
        for i in range(config.backbone_config.num_hidden_layers):
            # 弹出state_dict中输入投影层权重和偏置
            in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
            in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
            # 获取隐藏层的大小
            hidden_size = config.backbone_config.hidden_size
            # 接下来，将查询、键和值（按顺序）添加到state_dict中
            state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[:hidden_size, :]
            state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[:hidden_size]
            state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
                hidden_size: hidden_size * 2, :
            ]
            state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
                hidden_size: hidden_size * 2
            ]
            state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-hidden_size:, :]
            state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-hidden_size:]
# 从字典中删除指定键值对，并将该键值对重新加入字典中，完成对键的重命名
def rename_key(dct, old, new):
    val = dct.pop(old)  # 从字典中删除指定键值对，并将其值赋给val
    dct[new] = val  # 将删除的键值对中的值赋给新的键值对


# 准备提取可爱猫咪的图片并对其进行验证
def prepare_img():
    url = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"  # 指定图片的URL
    im = Image.open(requests.get(url, stream=True).raw)  # 使用requests库获取原始图片的流，并使用PIL库打开流中的图片
    return im  # 返回得到的图片对象


# 将模型名称映射到相应的URL
name_to_url = {
    # key: 模型名称, value: 模型对应的URL
    "dpt-dinov2-small-nyu": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_nyu_dpt_head.pth",
    # 其他模型名称和URL的映射
    ...
    "dpt-dinov2-giant-kitti": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_kitti_dpt_head.pth",
}


# 获取图像的原始像素值
def get_original_pixel_values(image):
    # 定义图像的中心填充
    class CenterPadding(object):
        # 初始化函数
        def __init__(self, multiple):
            super().__init__()
            self.multiple = multiple  # 设定填充的倍数

        # 计算填充量
        def _get_pad(self, size):
            # 计算新的尺寸
            new_size = math.ceil(size / self.multiple) * self.multiple
            pad_size = new_size - size
            pad_size_left = pad_size // 2
            pad_size_right = pad_size - pad_size_left
            return pad_size_left, pad_size_right  # 返回填充量

        # 对图像进行填充
        def __call__(self, img):
            pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in img.shape[-2:][::-1]))
            output = torch.nn.functional.pad(img, pads)
            return output  # 返回填充后的图像

        def __repr__(self):
            return self.__class__.__name__ + "()"  # 返回类的字符串表达形式

    # 生成深度变换
    def make_depth_transform() -> transforms.Compose:
        return transforms.Compose(
            [  # 图像转换操作序列
                transforms.ToTensor(),  # 转换为张量
                lambda x: 255.0 * x[:3],  # 丢弃 alpha 通道并按 255 缩放
                transforms.Normalize(  # 归一化
                    mean=(123.675, 116.28, 103.53),  # 均值
                    std=(58.395, 57.12, 57.375),  # 标准差
                ),
                CenterPadding(multiple=14),  # 中心填充
            ]
        )

    transform = make_depth_transform()  # 获取深度变换
    original_pixel_values = transform(image).unsqueeze(0)  # 对图像进行变换并增加一维维度

    return original_pixel_values  # 返回原始像素值


@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, verify_logits):
    """
    Copy/paste/tweak model's weights to our DPT structure.
    """

    # 根据模型名称获取对应的 DPT 配置信息
    checkpoint_url = name_to_url[model_name]  # 从字典中获取对应模型名称的URL
    config = get_dpt_config(model_name)  # 根据模型名称获取 DPT 配置

    # 从URL加载原始 DPT state_dict
    print("URL:", checkpoint_url)  # 打印加载的URL
    # 从给定的 URL 加载模型的预训练权重，并使用 "cpu" 作为设备
    dpt_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["state_dict"]
    # 重命名键值对
    rename_keys = create_rename_keys_dpt(config)
    for src, dest in rename_keys:
        rename_key(dpt_state_dict, src, dest)

    # 从 URL 加载原始骨干模型的 state_dict
    if "small" in model_name:
        original_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    elif "base" in model_name:
        original_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    elif "large" in model_name:
        original_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    elif "giant" in model_name:
        original_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
    else:
        raise NotImplementedError("To do")
    # 设置模型为评估模式
    original_model.eval()
    # 获取骨干模型的 state_dict
    backbone_state_dict = original_model.state_dict()

    # 重命名键值对
    rename_keys = create_rename_keys_backbone(config)
    for src, dest in rename_keys:
        rename_key(backbone_state_dict, src, dest)

    # 读取 qkv 矩阵
    read_in_q_k_v(backbone_state_dict, config)

    # 复制骨干模型 state_dict 并修改键：将 "w12" 改为 "weights_in"，将 "w3" 改为 "weights_out"
    for key, val in backbone_state_dict.copy().items():
        val = backbone_state_dict.pop(key)
        if "w12" in key:
            key = key.replace("w12", "weights_in")
        if "w3" in key:
            key = key.replace("w3", "weights_out")
        backbone_state_dict[key] = val

    # 合并两个 state_dict
    state_dict = {**backbone_state_dict, **dpt_state_dict}

    # 加载 HuggingFace 模型
    model = DPTForDepthEstimation(config)
    # 加载模型 state_dict，strict=False 表示允许有不匹配的键
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # 打印缺失的键和不期望的键
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    assert missing_keys == [
        "neck.fusion_stage.layers.0.residual_layer1.convolution1.weight",
        "neck.fusion_stage.layers.0.residual_layer1.convolution2.weight",
    ]
    # 设置模型为评估模式
    model.eval()

    # 验证图像处理器
    processor = DPTImageProcessor(
        do_resize=False,
        do_rescale=False,
        do_pad=True,
        size_divisor=14,
        do_normalize=True,
        image_mean=(123.675, 116.28, 103.53),
        image_std=(58.395, 57.12, 57.375),
    )

    # 准备图像
    image = prepare_img()
    # 对图像进行处理，���回像素值的张量
    pixel_values = processor(image, return_tensors="pt").pixel_values.float()
    # 获取原始像素值
    original_pixel_values = get_original_pixel_values(image)

    # 检查处理后的像素值和原始像素值是否相等
    assert torch.allclose(pixel_values, original_pixel_values)

    # 验证前向传播
    with torch.no_grad():
        outputs = model(pixel_values)

    # 获取预测的深度信息
    predicted_depth = outputs.predicted_depth

    # 打印预测深度的形状和前几个值
    print("Shape of predicted depth:", predicted_depth.shape)
    print("First values of predicted depth:", predicted_depth[0, :3, :3])

    # 断言日志
    # 如果需要验证logits
    if verify_logits:
        # 如果模型名称为"dpt-dinov2-small-nyu"
        if model_name == "dpt-dinov2-small-nyu":
            # 期望的形状
            expected_shape = torch.Size([1, 576, 736])
            # 期望的切片
            expected_slice = torch.tensor(
                [[3.3576, 3.4741, 3.4345], [3.4324, 3.5012, 3.2775], [3.2560, 3.3563, 3.2354]]
            )

        # 断言预测深度的形状与期望的形状相等
        assert predicted_depth.shape == torch.Size(expected_shape)
        # 断言预测深度的前3行3列与期望的切片接近，允许误差为1e-5
        assert torch.allclose(predicted_depth[0, :3, :3], expected_slice, atol=1e-5)
        # 打印"Looks ok!"表示验证通过

    # 如果需要保存模型和处理器到指定路径
    if pytorch_dump_folder_path is not None:
        # 创建存储路径
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 打印保存模型和处理器的路径
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        # 保存模型到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 保存处理器到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送模型和处理器到hub
    if push_to_hub:
        # 打印推送模型和处理器的信息
        print("Pushing model and processor to hub...")
        # 推送模型到hub的指定资源库
        model.push_to_hub(repo_id=f"facebook/{model_name}")
        # 推送处理器到hub的指定资源库
        processor.push_to_hub(repo_id=f"facebook/{model_name}")
# 主程序入口，判断是否在主模块中运行
if __name__ == "__main__":
    # 创建命令行参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--model_name",
        default="dpt-dinov2-small-nyu",
        type=str,
        choices=name_to_url.keys(),  # 可选值为已知模型名称列表的键集合
        help="Name of the model you'd like to convert.",  # 参数帮助信息
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",  # 参数帮助信息
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",  # 是否设置为 True
        help="Whether to push the model to the hub after conversion.",  # 参数帮助信息
    )
    parser.add_argument(
        "--verify_logits",
        action="store_true",  # 是否设置为 True
        required=False,  # 不是必需的参数
        help="Path to the output PyTorch model directory.",  # 参数帮助信息
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将模型转换为 PyTorch 格式
    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.verify_logits)
```