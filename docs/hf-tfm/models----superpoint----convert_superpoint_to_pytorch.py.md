# `.\models\superpoint\convert_superpoint_to_pytorch.py`

```
# 导入所需的模块和库
import argparse  # 用于解析命令行参数
import os  # 提供与操作系统交互的功能

import requests  # 发送 HTTP 请求的库
import torch  # 提供深度学习框架
from PIL import Image  # Python Imaging Library，用于图像处理

from transformers import SuperPointConfig, SuperPointForKeypointDetection, SuperPointImageProcessor

# 创建并返回一个 SuperPointConfig 对象，配置模型的各种参数
def get_superpoint_config():
    config = SuperPointConfig(
        encoder_hidden_sizes=[64, 64, 128, 128],  # 编码器各层隐藏单元数
        decoder_hidden_size=256,  # 解码器隐藏层单元数
        keypoint_decoder_dim=65,  # 关键点解码器维度
        descriptor_decoder_dim=256,  # 描述符解码器维度
        keypoint_threshold=0.005,  # 关键点检测阈值
        max_keypoints=-1,  # 最大关键点数目
        nms_radius=4,  # 非最大抑制半径
        border_removal_distance=4,  # 边缘去除距离
        initializer_range=0.02,  # 参数初始化范围
    )
    return config

# 创建并返回一个包含权重重命名信息的列表，用于加载预训练模型
def create_rename_keys(config, state_dict):
    rename_keys = []

    # 编码器权重
    rename_keys.append(("conv1a.weight", "encoder.conv_blocks.0.conv_a.weight"))
    rename_keys.append(("conv1b.weight", "encoder.conv_blocks.0.conv_b.weight"))
    rename_keys.append(("conv2a.weight", "encoder.conv_blocks.1.conv_a.weight"))
    rename_keys.append(("conv2b.weight", "encoder.conv_blocks.1.conv_b.weight"))
    rename_keys.append(("conv3a.weight", "encoder.conv_blocks.2.conv_a.weight"))
    rename_keys.append(("conv3b.weight", "encoder.conv_blocks.2.conv_b.weight"))
    rename_keys.append(("conv4a.weight", "encoder.conv_blocks.3.conv_a.weight"))
    rename_keys.append(("conv4b.weight", "encoder.conv_blocks.3.conv_b.weight"))
    rename_keys.append(("conv1a.bias", "encoder.conv_blocks.0.conv_a.bias"))
    rename_keys.append(("conv1b.bias", "encoder.conv_blocks.0.conv_b.bias"))
    rename_keys.append(("conv2a.bias", "encoder.conv_blocks.1.conv_a.bias"))
    rename_keys.append(("conv2b.bias", "encoder.conv_blocks.1.conv_b.bias"))
    rename_keys.append(("conv3a.bias", "encoder.conv_blocks.2.conv_a.bias"))
    rename_keys.append(("conv3b.bias", "encoder.conv_blocks.2.conv_b.bias"))
    rename_keys.append(("conv4a.bias", "encoder.conv_blocks.3.conv_a.bias"))
    rename_keys.append(("conv4b.bias", "encoder.conv_blocks.3.conv_b.bias"))

    # 关键点解码器权重
    rename_keys.append(("convPa.weight", "keypoint_decoder.conv_score_a.weight"))
    rename_keys.append(("convPb.weight", "keypoint_decoder.conv_score_b.weight"))
    rename_keys.append(("convPa.bias", "keypoint_decoder.conv_score_a.bias"))
    rename_keys.append(("convPb.bias", "keypoint_decoder.conv_score_b.bias"))

    # 描述符解码器权重
    # 将 ("convDa.weight", "descriptor_decoder.conv_descriptor_a.weight") 元组添加到 rename_keys 列表中
    rename_keys.append(("convDa.weight", "descriptor_decoder.conv_descriptor_a.weight"))
    # 将 ("convDb.weight", "descriptor_decoder.conv_descriptor_b.weight") 元组添加到 rename_keys 列表中
    rename_keys.append(("convDb.weight", "descriptor_decoder.conv_descriptor_b.weight"))
    # 将 ("convDa.bias", "descriptor_decoder.conv_descriptor_a.bias") 元组添加到 rename_keys 列表中
    rename_keys.append(("convDa.bias", "descriptor_decoder.conv_descriptor_a.bias"))
    # 将 ("convDb.bias", "descriptor_decoder.conv_descriptor_b.bias") 元组添加到 rename_keys 列表中
    rename_keys.append(("convDb.bias", "descriptor_decoder.conv_descriptor_b.bias"))
    
    # 返回 rename_keys 列表，该列表包含了需要重命名的键值对元组
    return rename_keys
# 重命名字典中的键
def rename_key(dct, old, new):
    # 弹出旧键对应的值
    val = dct.pop(old)
    # 将值与新键关联起来
    dct[new] = val

# 准备图片数据
def prepare_imgs():
    # 第一张图片的 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 获取第一张图片的二进制数据并打开为图像对象
    im1 = Image.open(requests.get(url, stream=True).raw)
    # 第二张图片的 URL
    url = "http://images.cocodataset.org/test-stuff2017/000000004016.jpg"
    # 获取第二张图片的二进制数据并打开为图像对象
    im2 = Image.open(requests.get(url, stream=True).raw)
    # 返回图片对象列表
    return [im1, im2]

# 用于禁用 Torch 的梯度计算
@torch.no_grad()
def convert_superpoint_checkpoint(checkpoint_url, pytorch_dump_folder_path, save_model, push_to_hub, test_mode=False):
    """
    Copy/paste/tweak model's weights to our SuperPoint structure.
    """

    # 打印信息：从检查点下载原始模型
    print("Downloading original model from checkpoint...")
    # 获取 SuperPoint 模型的配置信息
    config = get_superpoint_config()

    # 从 URL 加载原始模型的状态字典
    original_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)

    # 打印信息：转换模型参数
    print("Converting model parameters...")
    
    # 创建重命名键列表
    rename_keys = create_rename_keys(config, original_state_dict)
    # 复制原始状态字典
    new_state_dict = original_state_dict.copy()
    # 遍历重命名键列表，为新状态字典重命名键
    for src, dest in rename_keys:
        rename_key(new_state_dict, src, dest)

    # 加载 HuggingFace 模型
    model = SuperPointForKeypointDetection(config)
    # 加载新的状态字典到模型
    model.load_state_dict(new_state_dict)
    # 设置模型为评估模式
    model.eval()
    # 打印信息：成功加载模型的权重
    print("Successfully loaded weights in the model")

    # 实例化 SuperPoint 图像处理器
    preprocessor = SuperPointImageProcessor()
    # 准备输入数据：使用 prepare_imgs 函数准备的图像数据
    inputs = preprocessor(images=prepare_imgs(), return_tensors="pt")
    # 模型推理：获取模型的输出结果
    outputs = model(**inputs)

    # 如果 test_mode 为 True，则检查模型输出是否与原始结果匹配
    if test_mode:
        # 计算非零值的数量，以确保模型输出与原始结果匹配
        torch.count_nonzero(outputs.mask[0])
        # 期望的关键点形状和分数形状
        expected_keypoints_shape = (2, 830, 2)
        expected_scores_shape = (2, 830)
        expected_descriptors_shape = (2, 830, 256)

        # 期望的关键点、分数和描述子的值
        expected_keypoints_values = torch.tensor([[480.0, 9.0], [494.0, 9.0], [489.0, 16.0]])
        expected_scores_values = torch.tensor([0.0064, 0.0140, 0.0595, 0.0728, 0.5170, 0.0175, 0.1523, 0.2055, 0.0336])
        expected_descriptors_value = torch.tensor(-0.1096)
        
        # 断言：检查模型输出的关键点、分数和描述子是否与预期匹配
        assert outputs.keypoints.shape == expected_keypoints_shape
        assert outputs.scores.shape == expected_scores_shape
        assert outputs.descriptors.shape == expected_descriptors_shape

        assert torch.allclose(outputs.keypoints[0, :3], expected_keypoints_values, atol=1e-3)
        assert torch.allclose(outputs.scores[0, :9], expected_scores_values, atol=1e-3)
        assert torch.allclose(outputs.descriptors[0, 0, 0], expected_descriptors_value, atol=1e-3)
        # 打印信息：模型输出与原始结果匹配
        print("Model outputs match the original results!")
    # 如果需要保存模型
    if save_model:
        # 打印信息：保存模型到本地
        print("Saving model to local...")
        
        # 如果指定的路径不存在文件夹，则创建文件夹
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)

        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 将预处理器保存到指定路径
        preprocessor.save_pretrained(pytorch_dump_folder_path)

        # 设置模型名称为"superpoint"
        model_name = "superpoint"
        
        # 如果需要将模型推送到 hub
        if push_to_hub:
            # 打印信息：推送模型到 hub
            print(f"Pushing {model_name} to the hub...")
            
        # 将模型推送到 hub，并使用模型名称
        model.push_to_hub(model_name)
        # 将预处理器推送到 hub，并使用模型名称
        preprocessor.push_to_hub(model_name)
if __name__ == "__main__":
    # 如果脚本作为主程序执行，开始执行以下代码块
    
    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # 必选参数
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth",
        type=str,
        help="URL of the original SuperPoint checkpoint you'd like to convert.",
    )
    # 添加一个命令行参数，用于指定 SuperPoint 模型的原始检查点的下载地址

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="model",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    # 添加一个命令行参数，用于指定输出 PyTorch 模型的目录路径

    parser.add_argument("--save_model", action="store_true", help="Save model to local")
    # 添加一个命令行参数，指定是否将模型保存到本地

    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image preprocessor to the hub")
    # 添加一个命令行参数，指定是否将模型和图像预处理器推送到某个中心化的平台（比如模型仓库）

    args = parser.parse_args()
    # 解析命令行参数，并将结果存储在 args 变量中

    convert_superpoint_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.save_model, args.push_to_hub
    )
    # 调用函数 convert_superpoint_checkpoint，传递命令行参数中的相关选项作为参数
```