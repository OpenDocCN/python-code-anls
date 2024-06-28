# `.\models\depth_anything\convert_depth_anything_to_hf.py`

```py
    # coding=utf-8
    # 版权 2024 年 HuggingFace Inc. 团队所有。
    #
    # 根据 Apache 许可证 2.0 版本进行许可；
    # 除非符合许可证的要求，否则不得使用此文件。
    # 您可以在以下网址获取许可证的副本：
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # 除非适用法律要求或书面同意，本软件按"原样"分发，
    # 不作任何明示或暗示的担保或条件。
    # 请参阅许可证了解具体的管理权限和限制。
    """从原始仓库转换 Depth Anything 检查点。URL:
    https://github.com/LiheYoung/Depth-Anything"""

    
    import argparse  # 导入命令行参数解析模块
    from pathlib import Path  # 导入处理文件路径的模块
    
    import requests  # 导入处理 HTTP 请求的模块
    import torch  # 导入 PyTorch 深度学习库
    from huggingface_hub import hf_hub_download  # 导入 Hugging Face Hub 下载模块
    from PIL import Image  # 导入处理图像的模块
    
    from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation, Dinov2Config, DPTImageProcessor
    from transformers.utils import logging  # 导入日志记录模块
    
    logging.set_verbosity_info()  # 设置日志记录级别为信息
    logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器
    
    
    def get_dpt_config(model_name):
        if "small" in model_name:
            # 如果模型名包含 "small"，则使用 Dinov2Config 从预训练模型 "facebook/dinov2-small" 初始化配置
            backbone_config = Dinov2Config.from_pretrained(
                "facebook/dinov2-small", out_indices=[9, 10, 11, 12], apply_layernorm=True, reshape_hidden_states=False
            )
            fusion_hidden_size = 64  # 设置融合隐藏层的大小为 64
            neck_hidden_sizes = [48, 96, 192, 384]  # 设置颈部隐藏层的大小列表
        elif "base" in model_name:
            # 如果模型名包含 "base"，则使用 Dinov2Config 从预训练模型 "facebook/dinov2-base" 初始化配置
            backbone_config = Dinov2Config.from_pretrained(
                "facebook/dinov2-base", out_indices=[9, 10, 11, 12], apply_layernorm=True, reshape_hidden_states=False
            )
            fusion_hidden_size = 128  # 设置融合隐藏层的大小为 128
            neck_hidden_sizes = [96, 192, 384, 768]  # 设置颈部隐藏层的大小列表
        elif "large" in model_name:
            # 如果模型名包含 "large"，则使用 Dinov2Config 从预训练模型 "facebook/dinov2-large" 初始化配置
            backbone_config = Dinov2Config.from_pretrained(
                "facebook/dinov2-large", out_indices=[21, 22, 23, 24], apply_layernorm=True, reshape_hidden_states=False
            )
            fusion_hidden_size = 256  # 设置融合隐藏层的大小为 256
            neck_hidden_sizes = [256, 512, 1024, 1024]  # 设置颈部隐藏层的大小列表
        else:
            raise NotImplementedError("To do")  # 抛出未实现的错误
        
        # 根据给定的配置参数创建 DepthAnythingConfig 对象
        config = DepthAnythingConfig(
            reassemble_hidden_size=backbone_config.hidden_size,
            patch_size=backbone_config.patch_size,
            backbone_config=backbone_config,
            fusion_hidden_size=fusion_hidden_size,
            neck_hidden_sizes=neck_hidden_sizes,
        )
    
        return config  # 返回配置对象
    
    
    def create_rename_keys(config):
        rename_keys = []  # 创建重命名键列表
        
        # fmt: off
        # stem
        # 添加预定义的重命名键对到列表中，格式化关闭
        rename_keys.append(("pretrained.cls_token", "backbone.embeddings.cls_token"))
        rename_keys.append(("pretrained.mask_token", "backbone.embeddings.mask_token"))
        rename_keys.append(("pretrained.pos_embed", "backbone.embeddings.position_embeddings"))
        rename_keys.append(("pretrained.patch_embed.proj.weight", "backbone.embeddings.patch_embeddings.projection.weight"))
        rename_keys.append(("pretrained.patch_embed.proj.bias", "backbone.embeddings.patch_embeddings.projection.bias"))
        
        # Transfomer encoder
    # 遍历预训练模型的隐藏层，生成重命名键值对，将预训练模型参数映射到新的后骨干网络中
    for i in range(config.backbone_config.num_hidden_layers):
        # 重命名预训练模型中的 gamma 参数到对应后骨干网络的 lambda1 参数
        rename_keys.append((f"pretrained.blocks.{i}.ls1.gamma", f"backbone.encoder.layer.{i}.layer_scale1.lambda1"))
        rename_keys.append((f"pretrained.blocks.{i}.ls2.gamma", f"backbone.encoder.layer.{i}.layer_scale2.lambda1"))
        # 重命名预训练模型中的 norm1 和 norm2 参数到对应后骨干网络的 norm1 和 norm2 参数
        rename_keys.append((f"pretrained.blocks.{i}.norm1.weight", f"backbone.encoder.layer.{i}.norm1.weight"))
        rename_keys.append((f"pretrained.blocks.{i}.norm1.bias", f"backbone.encoder.layer.{i}.norm1.bias"))
        rename_keys.append((f"pretrained.blocks.{i}.norm2.weight", f"backbone.encoder.layer.{i}.norm2.weight"))
        rename_keys.append((f"pretrained.blocks.{i}.norm2.bias", f"backbone.encoder.layer.{i}.norm2.bias"))
        # 重命名预训练模型中的 mlp.fc1 和 mlp.fc2 参数到对应后骨干网络的 mlp.fc1 和 mlp.fc2 参数
        rename_keys.append((f"pretrained.blocks.{i}.mlp.fc1.weight", f"backbone.encoder.layer.{i}.mlp.fc1.weight"))
        rename_keys.append((f"pretrained.blocks.{i}.mlp.fc1.bias", f"backbone.encoder.layer.{i}.mlp.fc1.bias"))
        rename_keys.append((f"pretrained.blocks.{i}.mlp.fc2.weight", f"backbone.encoder.layer.{i}.mlp.fc2.weight"))
        rename_keys.append((f"pretrained.blocks.{i}.mlp.fc2.bias", f"backbone.encoder.layer.{i}.mlp.fc2.bias"))
        # 重命名预训练模型中的 attention 参数到对应后骨干网络的 attention 参数
        rename_keys.append((f"pretrained.blocks.{i}.attn.proj.weight", f"backbone.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"pretrained.blocks.{i}.attn.proj.bias", f"backbone.encoder.layer.{i}.attention.output.dense.bias"))

    # 头部部分的重命名
    rename_keys.append(("pretrained.norm.weight", "backbone.layernorm.weight"))
    rename_keys.append(("pretrained.norm.bias", "backbone.layernorm.bias"))

    # 激活后处理（读取投影 + 调整大小块）
    # Depth Anything 不使用 CLS token，因此不需要 readout_projects

    # 遍历深度头部的投影和调整大小层，将其重命名到颈部的重组阶段中
    for i in range(4):
        rename_keys.append((f"depth_head.projects.{i}.weight", f"neck.reassemble_stage.layers.{i}.projection.weight"))
        rename_keys.append((f"depth_head.projects.{i}.bias", f"neck.reassemble_stage.layers.{i}.projection.bias"))

        if i != 2:
            # 对于不是第二个元素的情况，将深度头部的调整大小层重命名到颈部的重组阶段中
            rename_keys.append((f"depth_head.resize_layers.{i}.weight", f"neck.reassemble_stage.layers.{i}.resize.weight"))
            rename_keys.append((f"depth_head.resize_layers.{i}.bias", f"neck.reassemble_stage.layers.{i}.resize.bias"))

    # RefineNet 部分的映射关系
    mapping = {1:3, 2:2, 3:1, 4:0}
    # 遍历范围为 1 到 4 的整数，依次映射到对应的索引值，生成重命名键列表
    for i in range(1, 5):
        j = mapping[i]
        # 添加重命名键对，将深度头部的卷积层权重映射到融合阶段的投影层权重
        rename_keys.append((f"depth_head.scratch.refinenet{i}.out_conv.weight", f"neck.fusion_stage.layers.{j}.projection.weight"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.out_conv.bias", f"neck.fusion_stage.layers.{j}.projection.bias"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit1.conv1.weight", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution1.weight"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit1.conv1.bias", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution1.bias"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit1.conv2.weight", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution2.weight"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit1.conv2.bias", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution2.bias"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit2.conv1.weight", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution1.weight"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit2.conv1.bias", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution1.bias"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit2.conv2.weight", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution2.weight"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit2.conv2.bias", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution2.bias"))

    # 处理 scratch convolutions
    for i in range(4):
        # 将深度头部的 scratch 层权重映射到融合阶段的卷积层权重
        rename_keys.append((f"depth_head.scratch.layer{i+1}_rn.weight", f"neck.convs.{i}.weight"))

    # 处理头部权重
    rename_keys.append(("depth_head.scratch.output_conv1.weight", "head.conv1.weight"))
    rename_keys.append(("depth_head.scratch.output_conv1.bias", "head.conv1.bias"))
    rename_keys.append(("depth_head.scratch.output_conv2.0.weight", "head.conv2.weight"))
    rename_keys.append(("depth_head.scratch.output_conv2.0.bias", "head.conv2.bias"))
    rename_keys.append(("depth_head.scratch.output_conv2.2.weight", "head.conv3.weight"))
    rename_keys.append(("depth_head.scratch.output_conv2.2.bias", "head.conv3.bias"))

    # 返回最终的重命名键列表
    return rename_keys
# 将每个编码器层的权重矩阵分解为查询(query)、键(keys)和值(values)
def read_in_q_k_v(state_dict, config):
    # 从配置中获取隐藏层的大小
    hidden_size = config.backbone_config.hidden_size
    # 遍历每个编码器层
    for i in range(config.backbone_config.num_hidden_layers):
        # 读取输入投影层的权重和偏置（在原始实现中，这是一个单独的矩阵加偏置）
        in_proj_weight = state_dict.pop(f"pretrained.blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"pretrained.blocks.{i}.attn.qkv.bias")
        
        # 将查询(query)、键(keys)和值(values)依次添加到状态字典中
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[:hidden_size]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[hidden_size: hidden_size * 2, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[hidden_size: hidden_size * 2]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-hidden_size:]


# 重命名字典中的键
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# 准备用于验证的可爱猫咪图片
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


# 不使用梯度进行操作的装饰器，用于转换DPT检查点
@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, verify_logits):
    """
    Copy/paste/tweak model's weights to our DPT structure.
    """

    # 获取DPT模型的配置
    config = get_dpt_config(model_name)

    # 定义模型名称到文件名的映射
    model_name_to_filename = {
        "depth-anything-small": "depth_anything_vits14.pth",
        "depth-anything-base": "depth_anything_vitb14.pth",
        "depth-anything-large": "depth_anything_vitl14.pth",
    }

    # 加载原始的state_dict
    filename = model_name_to_filename[model_name]
    # 从HuggingFace Hub下载文件
    filepath = hf_hub_download(
        repo_id="LiheYoung/Depth-Anything", filename=f"checkpoints/{filename}", repo_type="space"
    )
    state_dict = torch.load(filepath, map_location="cpu")

    # 根据配置创建重命名映射
    rename_keys = create_rename_keys(config)
    # 使用重命名映射重命名state_dict中的键
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)

    # 读取qkv矩阵
    read_in_q_k_v(state_dict, config)

    # 加载HuggingFace模型
    model = DepthAnythingForDepthEstimation(config)
    model.load_state_dict(state_dict)
    model.eval()
    # 创建一个图像处理器对象，配置参数包括调整大小、尺寸限制、保持长宽比、重新缩放和归一化处理
    processor = DPTImageProcessor(
        do_resize=True,
        size={"height": 518, "width": 518},
        ensure_multiple_of=14,
        keep_aspect_ratio=True,
        do_rescale=True,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    # 定义一个图像的 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 通过 HTTP 请求获取图像数据并以流的方式打开
    image = Image.open(requests.get(url, stream=True).raw)

    # 使用图像处理器处理图像并返回像素张量
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # 使用无梯度环境，对模型进行前向传播预测深度
    with torch.no_grad():
        outputs = model(pixel_values)
        # 获取预测的深度图
        predicted_depth = outputs.predicted_depth

    # 打印预测深度图的形状
    print("Shape of predicted depth:", predicted_depth.shape)
    # 打印预测深度图的前几个像素值
    print("First values:", predicted_depth[0, :3, :3])

    # 如果需要验证 logits（对数几率），则进行断言验证
    if verify_logits:
        # 定义预期的深度图形状
        expected_shape = torch.Size([1, 518, 686])
        # 根据模型名称选择预期的深度图片段
        if model_name == "depth-anything-small":
            expected_slice = torch.tensor(
                [[8.8204, 8.6468, 8.6195], [8.3313, 8.6027, 8.7526], [8.6526, 8.6866, 8.7453]],
            )
        elif model_name == "depth-anything-base":
            expected_slice = torch.tensor(
                [[26.3997, 26.3004, 26.3928], [26.2260, 26.2092, 26.3427], [26.0719, 26.0483, 26.1254]],
            )
        elif model_name == "depth-anything-large":
            expected_slice = torch.tensor(
                [[87.9968, 87.7493, 88.2704], [87.1927, 87.6611, 87.3640], [86.7789, 86.9469, 86.7991]]
            )
        else:
            # 如果模型名称不受支持，则引发错误
            raise ValueError("Not supported")

        # 断言预测深度图的形状是否符合预期
        assert predicted_depth.shape == torch.Size(expected_shape)
        # 断言预测深度图的前几个像素值是否接近预期值，允许误差为 1e-6
        assert torch.allclose(predicted_depth[0, :3, :3], expected_slice, atol=1e-6)
        # 打印验证通过信息
        print("Looks ok!")

    # 如果指定了 PyTorch 模型保存文件夹路径，则保存模型和处理器
    if pytorch_dump_folder_path is not None:
        # 确保保存路径存在，不存在则创建
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 打印保存模型和处理器的信息
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 将处理器保存到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要将模型和处理器推送到 Hub
    if push_to_hub:
        # 打印推送模型和处理器到 Hub 的信息
        print("Pushing model and processor to hub...")
        # 将模型推送到指定 Hub 仓库
        model.push_to_hub(repo_id=f"LiheYoung/{model_name}-hf")
        # 将处理器推送到指定 Hub 仓库
        processor.push_to_hub(repo_id=f"LiheYoung/{model_name}-hf")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # Required parameters
    parser.add_argument(
        "--model_name",
        default="depth-anything-small",
        type=str,
        choices=name_to_checkpoint.keys(),
        help="Name of the model you'd like to convert.",
    )
    # 添加必需的参数 --model_name，指定要转换的模型名称，必须是预定义的选择之一

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    # 添加参数 --pytorch_dump_folder_path，指定输出的 PyTorch 模型目录的路径

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub after conversion.",
    )
    # 添加参数 --push_to_hub，指定是否在转换后将模型推送到 hub

    parser.add_argument(
        "--verify_logits",
        action="store_false",
        required=False,
        help="Whether to verify the logits after conversion.",
    )
    # 添加参数 --verify_logits，指定是否在转换后验证 logits

    args = parser.parse_args()
    # 解析命令行参数并存储到 args 对象中

    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.verify_logits)
    # 调用 convert_dpt_checkpoint 函数，传递解析后的参数用于模型转换
```