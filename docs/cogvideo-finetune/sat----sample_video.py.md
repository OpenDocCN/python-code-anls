# `.\cogvideo-finetune\sat\sample_video.py`

```py
# 导入操作系统相关功能
import os
# 导入数学运算库
import math
# 导入命令行参数解析库
import argparse
# 导入类型注解工具
from typing import List, Union
# 导入进度条库
from tqdm import tqdm
# 导入列表配置类
from omegaconf import ListConfig
# 导入图像输入输出库
import imageio

# 导入PyTorch库
import torch
# 导入NumPy库
import numpy as np
# 导入重排列工具
from einops import rearrange
# 导入图像变换库
import torchvision.transforms as TT

# 从自定义模型库导入获取模型的函数
from sat.model.base_model import get_model
# 从自定义训练库导入加载检查点的函数
from sat.training.model_io import load_checkpoint
# 导入多处理工具
from sat import mpu

# 从扩散视频模块导入视频扩散引擎
from diffusion_video import SATVideoDiffusionEngine
# 从参数模块导入获取参数的函数
from arguments import get_args
# 导入中心裁剪和调整大小功能
from torchvision.transforms.functional import center_crop, resize
# 导入插值模式
from torchvision.transforms import InterpolationMode
# 导入PIL库中的图像模块
from PIL import Image

# 定义从命令行读取输入的生成器函数
def read_from_cli():
    # 初始化计数器
    cnt = 0
    try:
        # 循环直到接收到EOF
        while True:
            # 提示用户输入英文文本
            x = input("Please input English text (Ctrl-D quit): ")
            # 返回处理后的文本和计数器值
            yield x.strip(), cnt
            # 增加计数器
            cnt += 1
    # 捕获EOF错误
    except EOFError as e:
        pass

# 定义从文件读取输入的生成器函数
def read_from_file(p, rank=0, world_size=1):
    # 以只读模式打开文件
    with open(p, "r") as fin:
        # 初始化计数器
        cnt = -1
        # 遍历文件中的每一行
        for l in fin:
            # 增加计数器
            cnt += 1
            # 根据rank和world_size决定是否继续
            if cnt % world_size != rank:
                continue
            # 返回处理后的行和计数器值
            yield l.strip(), cnt

# 定义从条件器获取唯一嵌入器键的函数
def get_unique_embedder_keys_from_conditioner(conditioner):
    # 返回唯一的输入键列表
    return list(set([x.input_key for x in conditioner.embedders]))

# 定义获取批次数据的函数
def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    # 初始化批次字典和未条件化批次字典
    batch = {}
    batch_uc = {}

    # 遍历所有键
    for key in keys:
        # 处理文本键
        if key == "txt":
            # 生成包含提示的批次数据
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            # 生成包含负提示的未条件化批次数据
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        else:
            # 将其他键的值直接添加到批次中
            batch[key] = value_dict[key]

    # 如果T不为None，则添加视频帧数信息
    if T is not None:
        batch["num_video_frames"] = T

    # 遍历批次字典中的所有键
    for key in batch.keys():
        # 如果未条件化字典中没有该键且其为张量，则克隆张量
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    # 返回批次和未条件化批次
    return batch, batch_uc

# 定义将视频保存为网格和MP4格式的函数
def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None):
    # 如果保存路径不存在，则创建它
    os.makedirs(save_path, exist_ok=True)

    # 遍历视频批次
    for i, vid in enumerate(video_batch):
        # 初始化GIF帧列表
        gif_frames = []
        # 遍历每一帧
        for frame in vid:
            # 调整帧的维度顺序
            frame = rearrange(frame, "c h w -> h w c")
            # 将帧的数据转换为0-255的整型
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            # 添加帧到GIF帧列表
            gif_frames.append(frame)
        # 生成当前保存路径
        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")
        # 使用imageio保存视频
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            # 遍历GIF帧并写入视频文件
            for frame in gif_frames:
                writer.append_data(frame)

# 定义调整图像大小以适应矩形裁剪的函数
def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    # 检查输入数组的宽高比与目标宽高比的关系
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        # 按照宽度进行调整大小
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        # 按照高度进行调整大小
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )
    # 获取数组的高度和宽度
        h, w = arr.shape[2], arr.shape[3]
        # 去掉数组的第一个维度，保持其他维度不变
        arr = arr.squeeze(0)
    
        # 计算高度和宽度的差值
        delta_h = h - image_size[0]
        delta_w = w - image_size[1]
    
        # 根据重塑模式确定裁剪的起始位置
        if reshape_mode == "random" or reshape_mode == "none":
            # 随机生成裁剪的顶部和左边位置
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            # 计算中心裁剪的顶部和左边位置
            top, left = delta_h // 2, delta_w // 2
        else:
            # 如果模式不被支持，抛出异常
            raise NotImplementedError
        # 裁剪数组为指定的高度和宽度
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        # 返回裁剪后的数组
        return arr
# 主函数，负责采样过程
def sampling_main(args, model_cls):
    # 检查 model_cls 是否为类型，若是则调用 get_model 函数获取模型
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    # 否则直接使用 model_cls
    else:
        model = model_cls

    # 加载模型的检查点
    load_checkpoint(model, args)
    # 设置模型为评估模式
    model.eval()

    # 根据输入类型读取数据
    if args.input_type == "cli":
        # 从命令行读取数据
        data_iter = read_from_cli()
    elif args.input_type == "txt":
        # 获取当前进程的排名和总进程数
        rank, world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        print("rank and world_size", rank, world_size)
        # 从文件读取数据，带入排名和进程数
        data_iter = read_from_file(args.input_file, rank=rank, world_size=world_size)
    else:
        # 如果输入类型不被支持，抛出错误
        raise NotImplementedError

    # 设置图像大小
    image_size = [480, 720]

    # 如果需要将图像转换为视频
    if args.image2video:
        chained_trainsforms = []
        # 添加将图像转换为张量的变换
        chained_trainsforms.append(TT.ToTensor())
        # 组合变换
        transform = TT.Compose(chained_trainsforms)

    # 获取模型的采样函数
    sample_func = model.sample
    # 定义采样的相关参数
    T, H, W, C, F = args.sampling_num_frames, image_size[0], image_size[1], args.latent_channels, 8
    # 设置样本数量
    num_samples = [1]
    # 定义强制使用的嵌入类型
    force_uc_zero_embeddings = ["txt"]
    # 获取模型所使用的设备
    device = model.device

# 当脚本作为主程序执行时
if __name__ == "__main__":
    # 检查环境变量以获取进程相关信息
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    # 创建命令行参数解析器
    py_parser = argparse.ArgumentParser(add_help=False)
    # 解析已知参数
    known, args_list = py_parser.parse_known_args()

    # 获取完整的命令行参数
    args = get_args(args_list)
    # 将已知参数与其他参数合并
    args = argparse.Namespace(**vars(args), **vars(known))
    # 删除不需要的深度学习配置参数
    del args.deepspeed_config
    # 设置模型配置的检查点大小
    args.model_config.first_stage_config.params.cp_size = 1
    # 设置网络配置的模型并行大小
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    # 关闭检查点激活
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    # 关闭均匀采样
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    # 调用采样主函数
    sampling_main(args, model_cls=SATVideoDiffusionEngine)
```