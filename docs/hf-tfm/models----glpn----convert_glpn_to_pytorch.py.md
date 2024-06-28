# `.\models\glpn\convert_glpn_to_pytorch.py`

```py
# 导入必要的模块和库
import argparse  # 导入命令行参数解析模块
from collections import OrderedDict  # 导入有序字典模块
from pathlib import Path  # 导入处理文件路径的模块

import requests  # 导入处理 HTTP 请求的库
import torch  # 导入 PyTorch 深度学习框架
from PIL import Image  # 导入 Python Imaging Library，用于图像处理

# 从transformers库中导入GLPN模型相关的类和函数
from transformers import GLPNConfig, GLPNForDepthEstimation, GLPNImageProcessor
# 从transformers的utils模块中导入日志记录功能
from transformers.utils import logging

# 设置日志级别为信息（info）
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义一个函数，用于重命名模型的state_dict中的键
def rename_keys(state_dict):
    new_state_dict = OrderedDict()
    # 遍历给定的状态字典中的每一个键值对
    for key, value in state_dict.items():
        # 如果键名以 "module.encoder" 开头，将其替换为 "glpn.encoder"
        if key.startswith("module.encoder"):
            key = key.replace("module.encoder", "glpn.encoder")
        # 如果键名以 "module.decoder" 开头，将其替换为 "decoder.stages"
        if key.startswith("module.decoder"):
            key = key.replace("module.decoder", "decoder.stages")
        # 如果键名中包含 "patch_embed"，例如 "patch_embed1"，将其替换为 "patch_embeddings.0"
        if "patch_embed" in key:
            # 获取数字索引
            idx = key[key.find("patch_embed") + len("patch_embed")]
            key = key.replace(f"patch_embed{idx}", f"patch_embeddings.{int(idx)-1}")
        # 如果键名中包含 "norm"，将其替换为 "layer_norm"
        if "norm" in key:
            key = key.replace("norm", "layer_norm")
        # 如果键名中包含 "glpn.encoder.layer_norm"，例如 "glpn.encoder.layer_norm1"，将其替换为 "glpn.encoder.layer_norm.0"
        if "glpn.encoder.layer_norm" in key:
            # 获取数字索引
            idx = key[key.find("glpn.encoder.layer_norm") + len("glpn.encoder.layer_norm")]
            key = key.replace(f"layer_norm{idx}", f"layer_norm.{int(idx)-1}")
        # 如果键名中包含 "layer_norm1"，将其替换为 "layer_norm_1"
        if "layer_norm1" in key:
            key = key.replace("layer_norm1", "layer_norm_1")
        # 如果键名中包含 "layer_norm2"，将其替换为 "layer_norm_2"
        if "layer_norm2" in key:
            key = key.replace("layer_norm2", "layer_norm_2")
        # 如果键名中包含 "block"，例如 "block1"，将其替换为 "block.0"
        if "block" in key:
            # 获取数字索引
            idx = key[key.find("block") + len("block")]
            key = key.replace(f"block{idx}", f"block.{int(idx)-1}")
        # 如果键名中包含 "attn.q"，将其替换为 "attention.self.query"
        if "attn.q" in key:
            key = key.replace("attn.q", "attention.self.query")
        # 如果键名中包含 "attn.proj"，将其替换为 "attention.output.dense"
        if "attn.proj" in key:
            key = key.replace("attn.proj", "attention.output.dense")
        # 如果键名中包含 "attn"，将其替换为 "attention.self"
        if "attn" in key:
            key = key.replace("attn", "attention.self")
        # 如果键名中包含 "fc1"，将其替换为 "dense1"
        if "fc1" in key:
            key = key.replace("fc1", "dense1")
        # 如果键名中包含 "fc2"，将其替换为 "dense2"
        if "fc2" in key:
            key = key.replace("fc2", "dense2")
        # 如果键名中包含 "linear_pred"，将其替换为 "classifier"
        if "linear_pred" in key:
            key = key.replace("linear_pred", "classifier")
        # 如果键名中包含 "linear_fuse"，将其替换为 "linear_fuse" 和 "batch_norm"
        if "linear_fuse" in key:
            key = key.replace("linear_fuse.conv", "linear_fuse")
            key = key.replace("linear_fuse.bn", "batch_norm")
        # 如果键名中包含 "linear_c"，例如 "linear_c4"，将其替换为 "linear_c.3"
        if "linear_c" in key:
            # 获取数字索引
            idx = key[key.find("linear_c") + len("linear_c")]
            key = key.replace(f"linear_c{idx}", f"linear_c.{int(idx)-1}")
        # 如果键名中包含 "bot_conv"，将其替换为 "0.convolution"
        if "bot_conv" in key:
            key = key.replace("bot_conv", "0.convolution")
        # 如果键名中包含 "skip_conv1"，将其替换为 "1.convolution"
        if "skip_conv1" in key:
            key = key.replace("skip_conv1", "1.convolution")
        # 如果键名中包含 "skip_conv2"，将其替换为 "2.convolution"
        if "skip_conv2" in key:
            key = key.replace("skip_conv2", "2.convolution")
        # 如果键名中包含 "fusion1"，将其替换为 "1.fusion"
        if "fusion1" in key:
            key = key.replace("fusion1", "1.fusion")
        # 如果键名中包含 "fusion2"，将其替换为 "2.fusion"
        if "fusion2" in key:
            key = key.replace("fusion2", "2.fusion")
        # 如果键名中包含 "fusion3"，将其替换为 "3.fusion"
        if "fusion3" in key:
            key = key.replace("fusion3", "3.fusion")
        # 如果键名中包含 "fusion" 和 "conv"，将其替换为 "fusion.convolutional_layer"
        if "fusion" in key and "conv" in key:
            key = key.replace("conv", "convolutional_layer")
        # 如果键名以 "module.last_layer_depth" 开头，将其替换为 "head.head"
        if key.startswith("module.last_layer_depth"):
            key = key.replace("module.last_layer_depth", "head.head")
        # 将更新后的键值对存入新的状态字典中
        new_state_dict[key] = value

    # 返回更新后的状态字典
    return new_state_dict
# 读取每个编码器块的键值对权重和偏置
def read_in_k_v(state_dict, config):
    # 遍历每个编码器块
    for i in range(config.num_encoder_blocks):
        # 遍历当前深度下的层数
        for j in range(config.depths[i]):
            # 从状态字典中弹出键和值（在原始实现中，它们是单个矩阵）
            kv_weight = state_dict.pop(f"glpn.encoder.block.{i}.{j}.attention.self.kv.weight")
            kv_bias = state_dict.pop(f"glpn.encoder.block.{i}.{j}.attention.self.kv.bias")
            # 将键和值按顺序添加到状态字典中
            # 键的权重
            state_dict[f"glpn.encoder.block.{i}.{j}.attention.self.key.weight"] = kv_weight[: config.hidden_sizes[i], :]
            # 键的偏置
            state_dict[f"glpn.encoder.block.{i}.{j}.attention.self.key.bias"] = kv_bias[: config.hidden_sizes[i]]
            # 值的权重
            state_dict[f"glpn.encoder.block.{i}.{j}.attention.self.value.weight"] = kv_weight[config.hidden_sizes[i] :, :]
            # 值的偏置
            state_dict[f"glpn.encoder.block.{i}.{j}.attention.self.value.bias"] = kv_bias[config.hidden_sizes[i]:]

# 我们将在 COCO 图像上验证我们的结果
def prepare_img():
    # 定义 COCO 图像的 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用请求获取图像的原始流并打开图像
    image = Image.open(requests.get(url, stream=True).raw)
    # 返回处理后的图像
    return image

@torch.no_grad()
def convert_glpn_checkpoint(checkpoint_path, pytorch_dump_folder_path, push_to_hub=False, model_name=None):
    """
    将模型的权重复制/粘贴/调整到我们的 GLPN 结构中。
    """

    # 加载 GLPN 配置（Segformer-B4 尺寸）
    config = GLPNConfig(hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=64, depths=[3, 8, 27, 3])

    # 加载图像处理器（仅调整大小 + 重新缩放）
    image_processor = GLPNImageProcessor()

    # 准备图像
    image = prepare_img()
    # 使用图像处理器处理图像并获取像素值张量
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

    # 记录日志，指示模型转换过程开始
    logger.info("Converting model...")

    # 加载原始状态字典
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # 重命名键名
    state_dict = rename_keys(state_dict)

    # 处理键和值矩阵需要特殊处理
    read_in_k_v(state_dict, config)

    # 创建 HuggingFace 模型并加载状态字典
    model = GLPNForDepthEstimation(config)
    model.load_state_dict(state_dict)
    model.eval()

    # 前向传播
    outputs = model(pixel_values)
    # 获取预测的深度信息
    predicted_depth = outputs.predicted_depth

    # 验证输出
    # 如果模型名称不是空的话
    if model_name is not None:
        # 如果模型名称包含 "nyu"
        if "nyu" in model_name:
            # 设置预期的切片值为特定的张量
            expected_slice = torch.tensor(
                [[4.4147, 4.0873, 4.0673], [3.7890, 3.2881, 3.1525], [3.7674, 3.5423, 3.4913]]
            )
        # 如果模型名称包含 "kitti"
        elif "kitti" in model_name:
            # 设置预期的切片值为特定的张量
            expected_slice = torch.tensor(
                [[3.4291, 2.7865, 2.5151], [3.2841, 2.7021, 2.3502], [3.1147, 2.4625, 2.2481]]
            )
        else:
            # 如果模型名称既不包含 "nyu" 也不包含 "kitti"，则抛出异常
            raise ValueError(f"Unknown model name: {model_name}")

        # 设置预期的张量形状
        expected_shape = torch.Size([1, 480, 640])

        # 断言预测深度图的形状是否与预期形状相同
        assert predicted_depth.shape == expected_shape
        # 断言预测深度图的前 3x3 部分是否与预期的切片值在指定的误差范围内相近
        assert torch.allclose(predicted_depth[0, :3, :3], expected_slice, atol=1e-4)
        # 打印确认信息
        print("Looks ok!")

    # 最后，如果需要推送到 hub
    if push_to_hub:
        # 记录推送模型和图像处理器到 hub 的信息
        logger.info("Pushing model and image processor to the hub...")
        # 将模型推送到 hub
        model.push_to_hub(
            repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
            organization="nielsr",
            commit_message="Add model",
            use_temp_dir=True,
        )
        # 将图像处理器推送到 hub
        image_processor.push_to_hub(
            repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
            organization="nielsr",
            commit_message="Add image processor",
            use_temp_dir=True,
        )
# 如果当前脚本作为主程序运行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加命令行参数，用于指定原始 PyTorch 检查点文件的路径
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        help="Path to the original PyTorch checkpoint (.pth file).",
    )

    # 添加命令行参数，用于指定输出 PyTorch 模型的文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path", 
        default=None, 
        type=str, 
        help="Path to the folder to output PyTorch model."
    )

    # 添加命令行参数，指定是否将模型上传到 HuggingFace hub
    parser.add_argument(
        "--push_to_hub", 
        action="store_true", 
        help="Whether to upload the model to the HuggingFace hub."
    )

    # 添加命令行参数，用于指定模型名称，上传到 Hub 时会用到
    parser.add_argument(
        "--model_name",
        default="glpn-kitti",
        type=str,
        help="Name of the model in case you're pushing to the hub.",
    )

    # 解析命令行参数，将其存储在 args 变量中
    args = parser.parse_args()

    # 调用函数 convert_glpn_checkpoint，传递解析后的命令行参数作为函数参数
    convert_glpn_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub, args.model_name)
```