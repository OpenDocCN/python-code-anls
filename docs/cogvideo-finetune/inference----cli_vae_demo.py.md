# `.\cogvideo-finetune\inference\cli_vae_demo.py`

```py
"""
此脚本旨在演示如何使用 CogVideoX-2b VAE 模型进行视频编码和解码。
它允许将视频编码为潜在表示，解码回视频，或顺序执行这两项操作。
在运行脚本之前，请确保克隆了 CogVideoX Hugging Face 模型仓库，并将
`{your local diffusers path}` 参数设置为克隆仓库的路径。

命令 1：编码视频
使用 CogVideoX-5b VAE 模型编码位于 ../resources/videos/1.mp4 的视频。
内存使用量：编码时大约需要 ~18GB 的 GPU 内存。

如果您没有足够的 GPU 内存，我们在资源文件夹中提供了一个预编码的张量文件（encoded.pt），
您仍然可以运行解码命令。

$ python cli_vae_demo.py --model_path {your local diffusers path}/CogVideoX-2b/vae/ --video_path ../resources/videos/1.mp4 --mode encode

命令 2：解码视频

将存储在 encoded.pt 中的潜在表示解码回视频。
内存使用量：解码时大约需要 ~4GB 的 GPU 内存。
$ python cli_vae_demo.py --model_path {your local diffusers path}/CogVideoX-2b/vae/ --encoded_path ./encoded.pt --mode decode

命令 3：编码和解码视频
编码位于 ../resources/videos/1.mp4 的视频，然后立即解码。
内存使用量：编码 需要 34GB + 解码需要 19GB（顺序执行）。
$ python cli_vae_demo.py --model_path {your local diffusers path}/CogVideoX-2b/vae/ --video_path ../resources/videos/1.mp4 --mode both
"""

# 导入 argparse 模块用于处理命令行参数
import argparse
# 导入 torch 库用于深度学习相关的操作
import torch
# 导入 imageio 库用于读取视频文件
import imageio
# 从 diffusers 库导入 AutoencoderKLCogVideoX 模型
from diffusers import AutoencoderKLCogVideoX
# 从 torchvision 库导入 transforms，用于数据转换
from torchvision import transforms
# 导入 numpy 库用于数值计算
import numpy as np


def encode_video(model_path, video_path, dtype, device):
    """
    加载预训练的 AutoencoderKLCogVideoX 模型并编码视频帧。

    参数：
    - model_path (str): 预训练模型的路径。
    - video_path (str): 视频文件的路径。
    - dtype (torch.dtype): 计算所用的数据类型。
    - device (str): 计算所用的设备（例如，"cuda" 或 "cpu"）。

    返回：
    - torch.Tensor: 编码后的视频帧。
    """

    # 从指定路径加载预训练的模型，并将其移动到指定设备
    model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)

    # 启用切片功能以优化内存使用
    model.enable_slicing()
    # 启用平铺功能以处理大图像
    model.enable_tiling()

    # 使用 ffmpeg 读取视频文件
    video_reader = imageio.get_reader(video_path, "ffmpeg")

    # 将视频的每一帧转换为张量并存储在列表中
    frames = [transforms.ToTensor()(frame) for frame in video_reader]
    # 关闭视频读取器
    video_reader.close()

    # 将帧列表转换为张量，调整维度，并将其移动到指定设备和数据类型
    frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(dtype)

    # 在不计算梯度的上下文中进行编码
    with torch.no_grad():
        # 使用模型编码帧，并从中获取样本
        encoded_frames = model.encode(frames_tensor)[0].sample()
    # 返回编码后的帧
    return encoded_frames


def decode_video(model_path, encoded_tensor_path, dtype, device):
    """
    加载预训练的 AutoencoderKLCogVideoX 模型并解码编码的视频帧。

    参数：
    - model_path (str): 预训练模型的路径。
    - encoded_tensor_path (str): 编码张量文件的路径。
    # dtype 参数指定计算时使用的数据类型
        - dtype (torch.dtype): The data type for computation.
        # device 参数指定用于计算的设备（例如，“cuda”或“cpu”）
        - device (str): The device to use for computation (e.g., "cuda" or "cpu").
    
        # 返回解码后的视频帧
        Returns:
        - torch.Tensor: The decoded video frames.
        """
        # 从预训练模型加载 AutoencoderKLCogVideoX，并将其转移到指定设备上，设置数据类型
        model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)
        # 从指定路径加载编码的张量，并将其转移到设备和指定的数据类型上
        encoded_frames = torch.load(encoded_tensor_path, weights_only=True).to(device).to(dtype)
        # 在不计算梯度的上下文中解码编码的帧
        with torch.no_grad():
            # 调用模型解码编码的帧，并获取解码后的样本
            decoded_frames = model.decode(encoded_frames).sample
        # 返回解码后的帧
        return decoded_frames
# 定义一个函数，用于保存视频帧到视频文件
def save_video(tensor, output_path):
    """
    保存视频帧到视频文件。

    参数：
    - tensor (torch.Tensor): 视频帧的张量。
    - output_path (str): 输出视频的保存路径。
    """
    # 将张量转换为浮点32位类型
    tensor = tensor.to(dtype=torch.float32)
    # 将张量的第一个维度去掉，重新排列维度，并转移到 CPU，再转换为 NumPy 数组
    frames = tensor[0].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    # 将帧的值裁剪到 0 到 1 之间，并乘以 255 以转换为像素值
    frames = np.clip(frames, 0, 1) * 255
    # 将帧的数据类型转换为无符号 8 位整数
    frames = frames.astype(np.uint8)
    # 创建一个视频写入对象，设置输出路径和帧率为 8
    writer = imageio.get_writer(output_path + "/output.mp4", fps=8)
    # 遍历每一帧，将其添加到视频写入对象中
    for frame in frames:
        writer.append_data(frame)
    # 关闭视频写入对象，完成写入
    writer.close()


# 如果当前脚本是主程序，则执行以下代码
if __name__ == "__main__":
    # 创建一个参数解析器，用于处理命令行参数
    parser = argparse.ArgumentParser(description="CogVideoX encode/decode demo")
    # 添加一个参数，用于指定模型的路径
    parser.add_argument("--model_path", type=str, required=True, help="The path to the CogVideoX model")
    # 添加一个参数，用于指定视频文件的路径（用于编码）
    parser.add_argument("--video_path", type=str, help="The path to the video file (for encoding)")
    # 添加一个参数，用于指定编码的张量文件的路径（用于解码）
    parser.add_argument("--encoded_path", type=str, help="The path to the encoded tensor file (for decoding)")
    # 添加一个参数，用于指定输出文件的保存路径，默认为当前目录
    parser.add_argument("--output_path", type=str, default=".", help="The path to save the output file")
    # 添加一个参数，指定模式：编码、解码或两者
    parser.add_argument(
        "--mode", type=str, choices=["encode", "decode", "both"], required=True, help="Mode: encode, decode, or both"
    )
    # 添加一个参数，指定计算的数据类型，默认为 'bfloat16'
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    # 添加一个参数，指定用于计算的设备，默认为 'cuda'
    parser.add_argument(
        "--device", type=str, default="cuda", help="The device to use for computation (e.g., 'cuda' or 'cpu')"
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 根据指定的设备创建一个设备对象
    device = torch.device(args.device)
    # 根据指定的数据类型设置数据类型，默认为 bfloat16
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    # 根据模式选择编码、解码或两者的操作
    if args.mode == "encode":
        # 确保提供了视频路径用于编码
        assert args.video_path, "Video path must be provided for encoding."
        # 调用编码函数，将视频编码为张量
        encoded_output = encode_video(args.model_path, args.video_path, dtype, device)
        # 将编码后的张量保存到指定路径
        torch.save(encoded_output, args.output_path + "/encoded.pt")
        # 打印完成编码的消息
        print(f"Finished encoding the video to a tensor, save it to a file at {encoded_output}/encoded.pt")
    elif args.mode == "decode":
        # 确保提供了编码张量的路径用于解码
        assert args.encoded_path, "Encoded tensor path must be provided for decoding."
        # 调用解码函数，将编码的张量解码为视频帧
        decoded_output = decode_video(args.model_path, args.encoded_path, dtype, device)
        # 调用保存视频的函数，将解码后的输出保存为视频文件
        save_video(decoded_output, args.output_path)
        # 打印完成解码的消息
        print(f"Finished decoding the video and saved it to a file at {args.output_path}/output.mp4")
    elif args.mode == "both":
        # 确保提供了视频路径用于编码
        assert args.video_path, "Video path must be provided for encoding."
        # 调用编码函数，将视频编码为张量
        encoded_output = encode_video(args.model_path, args.video_path, dtype, device)
        # 将编码后的张量保存到指定路径
        torch.save(encoded_output, args.output_path + "/encoded.pt")
        # 调用解码函数，将保存的张量解码为视频帧
        decoded_output = decode_video(args.model_path, args.output_path + "/encoded.pt", dtype, device)
        # 调用保存视频的函数，将解码后的输出保存为视频文件
        save_video(decoded_output, args.output_path)
```