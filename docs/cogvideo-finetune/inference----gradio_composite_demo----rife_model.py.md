# `.\cogvideo-finetune\inference\gradio_composite_demo\rife_model.py`

```py
# 导入 PyTorch 库，用于深度学习操作
import torch
# 从 diffusers 库导入 VaeImageProcessor，用于处理图像
from diffusers.image_processor import VaeImageProcessor
# 导入 PyTorch 的函数式 API，主要用于张量操作
from torch.nn import functional as F
# 导入 OpenCV 库，用于图像处理
import cv2
# 导入自定义的 utils 模块，可能包含一些实用函数
import utils
# 从 rife.pytorch_msssim 导入 ssim_matlab，用于计算结构相似性
from rife.pytorch_msssim import ssim_matlab
# 导入 NumPy 库，用于数组操作
import numpy as np
# 导入 logging 模块，用于记录日志
import logging
# 从 skvideo.io 导入用于视频输入输出的库
import skvideo.io
# 从 rife.RIFE_HDv3 导入 Model 类，可能用于帧插值模型
from rife.RIFE_HDv3 import Model
# 从 huggingface_hub 导入下载模型和快照的功能
from huggingface_hub import hf_hub_download, snapshot_download
# 创建一个日志记录器，使用当前模块的名称
logger = logging.getLogger(__name__)

# 检查是否可以使用 GPU，如果可以则设为 'cuda'，否则设为 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义图像填充函数，接受图像和缩放比例作为参数
def pad_image(img, scale):
    # 解构图像形状，获取通道数、高度和宽度
    _, _, h, w = img.shape
    # 计算填充大小，确保是 32 的倍数
    tmp = max(32, int(32 / scale))
    # 计算填充后的高度和宽度
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    # 计算所需的填充边界
    padding = (0,  pw - w, 0, ph - h)
    # 返回填充后的图像和填充参数
    return F.pad(img, padding), padding

# 定义推理函数，接受模型、两帧图像、放大比例和分割数量作为参数
def make_inference(model, I0, I1, upscale_amount, n):
    # 调用模型进行推理，生成中间帧
    middle = model.inference(I0, I1, upscale_amount)
    # 如果分割数量为 1，返回中间帧
    if n == 1:
        return [middle]
    # 递归调用，生成前半部分插值帧
    first_half = make_inference(model, I0, middle, upscale_amount, n=n // 2)
    # 递归调用，生成后半部分插值帧
    second_half = make_inference(model, middle, I1, upscale_amount, n=n // 2)
    # 如果分割数量为奇数，合并结果
    if n % 2:
        return [*first_half, middle, *second_half]
    # 否则直接合并前后两部分
    else:
        return [*first_half, *second_half]

# 使用 PyTorch 的推理模式进行插值操作
@torch.inference_mode()
def ssim_interpolation_rife(model, samples, exp=1, upscale_amount=1, output_device="cpu"):
    # 打印样本数据类型
    print(f"samples dtype:{samples.dtype}")
    # 打印样本形状
    print(f"samples shape:{samples.shape}")
    # 初始化输出列表
    output = []
    # 创建进度条，用于显示推理进度
    pbar = utils.ProgressBar(samples.shape[0], desc="RIFE inference")
    # 样本形状为 [帧数, 通道数, 高度, 宽度]
    # 遍历样本的每一帧
    for b in range(samples.shape[0]):
        # 选取当前帧并增加维度
        frame = samples[b : b + 1]
        # 获取当前帧的高度和宽度
        _, _, h, w = frame.shape
        
        # 将当前帧赋值给 I0
        I0 = samples[b : b + 1]
        # 如果有下一帧，则赋值给 I1，否则使用最后一帧
        I1 = samples[b + 1 : b + 2] if b + 2 < samples.shape[0] else samples[-1:]
         
        # 对 I0 进行填充并返回填充后的图像和填充信息
        I0, padding = pad_image(I0, upscale_amount)
        # 将 I0 转换为浮点数类型
        I0 = I0.to(torch.float)
        # 对 I1 进行填充，第二个返回值不需要
        I1, _ = pad_image(I1, upscale_amount)
        # 将 I1 转换为浮点数类型
        I1 = I1.to(torch.float)
         
        # 将 I0 和 I1 进行双线性插值，缩放至 (32, 32)
        I0_small = F.interpolate(I0, (32, 32), mode="bilinear", align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)

        # 计算 I0_small 和 I1_small 之间的 SSIM 值
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        # 如果 SSIM 值大于 0.996，进行以下操作
        if ssim > 0.996:
            # 将当前帧重新赋值给 I1
            I1 = samples[b : b + 1]
            # print(f'upscale_amount:{upscale_amount}')  # 输出放大倍数（注释掉）
            # print(f'ssim:{upscale_amount}')  # 输出 SSIM 值（注释掉）
            # print(f'I0 shape:{I0.shape}')  # 输出 I0 的形状（注释掉）
            # print(f'I1 shape:{I1.shape}')  # 输出 I1 的形状（注释掉）
            # 对 I1 进行填充并返回填充信息
            I1, padding = pad_image(I1, upscale_amount)
            # print(f'I0 shape:{I0.shape}')  # 输出 I0 的形状（注释掉）
            # print(f'I1 shape:{I1.shape}')  # 输出 I1 的形状（注释掉）
            # 进行推理，使用 I0 和 I1 以及放大倍数
            I1 = make_inference(model, I0, I1, upscale_amount, 1)
            
            # print(f'I0 shape:{I0.shape}')  # 输出 I0 的形状（注释掉）
            # print(f'I1[0] shape:{I1[0].shape}')  # 输出 I1[0] 的形状（注释掉）
            # 取出推理结果的第一张图像
            I1 = I1[0]
            
            # print(f'I1[0] unpadded shape:{I1.shape}')  # 输出 I1 的去填充形状（注释掉） 
            # 将 I1 进行双线性插值，缩放至 (32, 32)
            I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
            # 重新计算 SSIM 值
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            # 根据填充信息确定如何裁剪 frame
            if padding[3] > 0 and padding[1] > 0:
                frame = I1[:, :, : -padding[3],:-padding[1]]
            elif padding[3] > 0:
                frame = I1[:, :, : -padding[3],:]
            elif padding[1] > 0:
                frame = I1[:, :, :,:-padding[1]]
            else:
                frame = I1

        # 初始化临时输出列表
        tmp_output = []
        # 如果 SSIM 值小于 0.2，进行以下操作
        if ssim < 0.2:
            # 根据指数生成多张 I0
            for i in range((2**exp) - 1):
                tmp_output.append(I0)

        else:
            # 如果指数不为零，则进行推理并生成输出
            tmp_output = make_inference(model, I0, I1, upscale_amount, 2**exp - 1) if exp else []

        # 对 frame 进行填充
        frame, _ = pad_image(frame, upscale_amount)
        # print(f'frame shape:{frame.shape}')  # 输出 frame 的形状（注释掉）

        # 将 frame 进行插值，缩放至原始的高度和宽度
        frame = F.interpolate(frame, size=(h, w))
        # 将处理后的 frame 加入输出列表
        output.append(frame.to(output_device))
        # 遍历临时输出并处理
        for i, tmp_frame in enumerate(tmp_output): 
            # tmp_frame, _ = pad_image(tmp_frame, upscale_amount)  # 对 tmp_frame 进行填充（注释掉）
            # 将 tmp_frame 进行插值，缩放至原始的高度和宽度
            tmp_frame = F.interpolate(tmp_frame, size=(h, w))
            # 将处理后的 tmp_frame 加入输出列表
            output.append(tmp_frame.to(output_device))
        # 更新进度条
        pbar.update(1)
    # 返回最终输出
    return output
# 加载 RIFE 模型并返回模型实例
def load_rife_model(model_path):
    # 创建模型实例
    model = Model()
    # 从指定路径加载模型，第二个参数为 -1（表示不使用特定的版本）
    model.load_model(model_path, -1)
    # 将模型设置为评估模式
    model.eval()
    # 返回加载的模型
    return model


# 创建一个生成器，逐帧输出视频帧，类似于 cv2.VideoCapture
def frame_generator(video_capture):
    # 无限循环，直到读取完所有帧
    while True:
        # 从视频捕捉对象中读取一帧，ret 为读取成功标志，frame 为当前帧
        ret, frame = video_capture.read()
        # 如果没有读取到帧，退出循环
        if not ret:
            break
        # 生成当前帧
        yield frame
    # 释放视频捕捉对象
    video_capture.release()


def rife_inference_with_path(model, video_path):
    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)
    # 获取视频的帧率（每秒多少帧）
    fps = video_capture.get(cv2.CAP_PROP_FPS)  
    # 获取视频的总帧数
    tot_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  
    # 存储处理后帧的列表
    pt_frame_data = []
    # 使用 skvideo.io 逐帧读取视频
    pt_frame = skvideo.io.vreader(video_path)
    # 循环读取视频帧
    while video_capture.isOpened():
        # 读取一帧
        ret, frame = video_capture.read()

        # 如果没有读取到帧，退出循环
        if not ret:
            break

        # 将 BGR 格式的帧转换为 RGB 格式
        frame_rgb = frame[..., ::-1]
        # 创建帧的副本
        frame_rgb = frame_rgb.copy()
        # 将 RGB 帧转换为 tensor，并归一化到 [0, 1] 之间
        tensor = torch.from_numpy(frame_rgb).float().to("cpu", non_blocking=True).float() / 255.0
        # 将处理后的 tensor 按 [c, h, w] 格式添加到列表
        pt_frame_data.append(
            tensor.permute(2, 0, 1)
        )  # to [c, h, w,]

    # 将帧数据堆叠为一个 tensor
    pt_frame = torch.from_numpy(np.stack(pt_frame_data))
    # 将 tensor 移动到指定设备
    pt_frame = pt_frame.to(device)
    # 创建进度条，显示处理进度
    pbar = utils.ProgressBar(tot_frame, desc="RIFE inference")
    # 使用 RIFE 模型进行帧插值
    frames = ssim_interpolation_rife(model, pt_frame)
    # 堆叠生成的帧为一个 tensor
    pt_image = torch.stack([frames[i].squeeze(0) for i in range(len(frames))])
    # 将处理后的 tensor 转换为 NumPy 数组
    image_np = VaeImageProcessor.pt_to_numpy(pt_image)  # (to [49, 512, 480, 3])
    # 将 NumPy 数组转换为 PIL 图像
    image_pil = VaeImageProcessor.numpy_to_pil(image_np)
    # 保存处理后的视频，并设置帧率
    video_path = utils.save_video(image_pil, fps=16)
    # 更新进度条
    if pbar:
        pbar.update(1)
    # 返回保存的视频路径
    return video_path


def rife_inference_with_latents(model, latents):
    # 存储 RIFE 处理结果的列表
    rife_results = []
    # 将潜在变量移动到指定设备
    latents = latents.to(device)
    # 遍历每个潜在变量
    for i in range(latents.size(0)):
        # 取出当前的潜在变量
        latent = latents[i]

        # 使用 RIFE 模型进行帧插值
        frames = ssim_interpolation_rife(model, latent)
        # 堆叠生成的帧为一个 tensor
        pt_image = torch.stack([frames[i].squeeze(0) for i in range(len(frames))])  # (to [f, c, w, h])
        # 将处理结果添加到列表中
        rife_results.append(pt_image)

    # 返回所有处理结果的堆叠 tensor
    return torch.stack(rife_results)


# if __name__ == "__main__":
#     # 下载 RIFE 模型快照到指定目录
#     snapshot_download(repo_id="AlexWortega/RIFE", local_dir="model_rife")
#     # 加载 RIFE 模型
#     model = load_rife_model("model_rife")
 
#     # 使用指定视频路径进行 RIFE 推理
#     video_path = rife_inference_with_path(model, "/mnt/ceph/develop/jiawei/CogVideo/output/20241003_130720.mp4")
#     # 打印保存的视频路径
#     print(video_path)
```