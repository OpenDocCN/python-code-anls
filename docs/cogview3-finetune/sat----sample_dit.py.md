# `.\cogview3-finetune\sat\sample_dit.py`

```py
# 导入操作系统模块，用于与操作系统交互
import os
# 导入数学模块，提供数学函数和常量
import math
# 导入命令行参数解析模块
import argparse
# 导入类型提示模块
from typing import List, Union
# 导入进度条模块，用于显示进度
from tqdm import tqdm
# 导入 OmegaConf 的 ListConfig 类，用于处理配置
from omegaconf import ListConfig
# 导入图像处理库 PIL
from PIL import Image

# 导入 PyTorch 库
import torch
# 导入 NumPy 库
import numpy as np
# 从 einops 导入 rearrange 和 repeat 函数，用于处理张量
from einops import rearrange, repeat
# 从 torchvision 导入 make_grid 函数，用于生成图像网格
from torchvision.utils import make_grid

# 从自定义模型模块导入获取模型的函数
from sat.model.base_model import get_model
# 从自定义训练模块导入加载检查点的函数
from sat.training.model_io import load_checkpoint

# 从 diffusion 模块导入 SATDiffusionEngine 类
from diffusion import SATDiffusionEngine
# 从 arguments 模块导入获取命令行参数的函数
from arguments import get_args


# 定义从命令行读取输入的生成器函数
def read_from_cli():
    cnt = 0  # 初始化计数器
    try:
        # 无限循环，等待用户输入
        while True:
            # 提示用户输入英文文本，按 Ctrl-D 退出
            x = input("Please input English text (Ctrl-D quit): ")
            # 去除输入文本的前后空格并生成一个元组
            yield x.strip(), cnt
            cnt += 1  # 计数器递增
    except EOFError as e:
        pass  # 捕获文件结束错误，结束循环


# 定义从文件中读取输入的生成器函数
def read_from_file(p, rank=0, world_size=1):
    # 以只读模式打开文件
    with open(p, "r") as fin:
        cnt = -1  # 初始化计数器
        # 遍历文件中的每一行
        for l in fin:
            cnt += 1  # 计数器递增
            # 如果当前计数不是该进程的排名，则跳过
            if cnt % world_size != rank:
                continue
            # 去除行首尾空白并生成一个元组
            yield l.strip(), cnt


# 定义从调节器中获取唯一嵌入器键的函数
def get_unique_embedder_keys_from_conditioner(conditioner):
    # 从嵌入器中提取输入键，去重并转换为列表
    return list(set([x.input_key for x in conditioner.embedders]))


# 定义获取批次的函数
def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}  # 初始化批次字典
    batch_uc = {}  # 初始化无条件批次字典
    # 遍历给定的键列表
    for key in keys:
        # 如果键是 "txt"，处理相关的文本数据
        if key == "txt":
            # 通过重复提示文本构建 batch 中的 "txt" 数据
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            # 通过重复负面提示文本构建 batch_uc 中的 "txt" 数据
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        # 如果键是 "original_size_as_tuple"，处理原始图像大小
        elif key == "original_size_as_tuple":
            # 将原始高度和宽度转换为张量并在设备上重复 N 次
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]]).to(device).repeat(*N, 1)
            )
        # 如果键是 "crop_coords_top_left"，处理裁剪坐标
        elif key == "crop_coords_top_left":
            # 将裁剪坐标转换为张量并在设备上重复 N 次
            batch["crop_coords_top_left"] = (
                torch.tensor([value_dict["crop_coords_top"], value_dict["crop_coords_left"]]).to(device).repeat(*N, 1)
            )
        # 如果键是 "aesthetic_score"，处理美学评分
        elif key == "aesthetic_score":
            # 将美学评分转换为张量并在设备上重复 N 次
            batch["aesthetic_score"] = torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            # 将负面美学评分转换为张量并在 batch_uc 中重复 N 次
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]]).to(device).repeat(*N, 1)
            )

        # 如果键是 "target_size_as_tuple"，处理目标大小
        elif key == "target_size_as_tuple":
            # 将目标高度和宽度转换为张量并在设备上重复 N 次
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]]).to(device).repeat(*N, 1)
            )
        # 如果键是 "fps"，处理帧率
        elif key == "fps":
            # 将帧率转换为张量并在设备上重复 math.prod(N) 次
            batch[key] = torch.tensor([value_dict["fps"]]).to(device).repeat(math.prod(N))
        # 如果键是 "fps_id"，处理帧率 ID
        elif key == "fps_id":
            # 将帧率 ID 转换为张量并在设备上重复 math.prod(N) 次
            batch[key] = torch.tensor([value_dict["fps_id"]]).to(device).repeat(math.prod(N))
        # 如果键是 "motion_bucket_id"，处理运动桶 ID
        elif key == "motion_bucket_id":
            # 将运动桶 ID 转换为张量并在设备上重复 math.prod(N) 次
            batch[key] = torch.tensor([value_dict["motion_bucket_id"]]).to(device).repeat(math.prod(N))
        # 如果键是 "pool_image"，处理图像数据
        elif key == "pool_image":
            # 使用 repeat 函数处理图像数据并在设备上转换数据类型为半精度
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=math.prod(N)).to(device, dtype=torch.half)
        # 如果键是 "cond_aug"，处理条件增强
        elif key == "cond_aug":
            # 将条件增强转换为张量并在 CUDA 设备上重复
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to("cuda"),
                "1 -> b",
                b=math.prod(N),
            )
        # 如果键是 "cond_frames"，处理条件帧
        elif key == "cond_frames":
            # 使用 repeat 函数处理条件帧数据
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        # 如果键是 "cond_frames_without_noise"，处理无噪声条件帧
        elif key == "cond_frames_without_noise":
            # 使用 repeat 函数处理无噪声条件帧数据
            batch[key] = repeat(value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0])
        # 如果键是 "cfg_scale"，处理配置缩放
        elif key == "cfg_scale":
            # 将配置缩放值转换为张量并在设备上重复 math.prod(N) 次
            batch[key] = torch.tensor([value_dict["cfg_scale"]]).to(device).repeat(math.prod(N))
        # 处理其他键，将其值直接赋给 batch
        else:
            batch[key] = value_dict[key]

    # 如果 T 不为 None，添加视频帧数量到 batch 中
    if T is not None:
        batch["num_video_frames"] = T

    # 遍历 batch 中的所有键
    for key in batch.keys():
        # 如果键不在 batch_uc 中且对应值是张量，则进行克隆
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    # 返回 batch 和 batch_uc
    return batch, batch_uc
# 定义一个将样本保存到本地的函数
def perform_save_locally(save_path, samples, grid, only_save_grid=False):
    # 创建保存路径，如果已存在则不报错
    os.makedirs(save_path, exist_ok=True)

    # 如果不只保存网格图像
    if not only_save_grid:
        # 遍历样本列表，获取索引和样本
        for i, sample in enumerate(samples):
            # 将样本转换为 RGB 格式并缩放到 255 范围
            sample = 255.0 * rearrange(sample.numpy(), "c h w -> h w c")
            # 将样本保存为 PNG 图像，命名为索引格式
            Image.fromarray(sample.astype(np.uint8)).save(os.path.join(save_path, f"{i:09}.png"))

    # 如果网格不为空
    if grid is not None:
        # 将网格转换为 RGB 格式并缩放到 255 范围
        grid = 255.0 * rearrange(grid.numpy(), "c h w -> h w c")
        # 将网格保存为 PNG 图像
        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(save_path, f"grid.png"))


# 定义一个主函数用于采样
def sampling_main(args, model_cls):
    # 如果模型类是类型，则获取模型实例
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls

    # 加载模型的检查点
    load_checkpoint(model, args)
    # 设置模型为评估模式
    model.eval()

    # 根据输入类型读取数据
    if args.input_type == "cli":
        data_iter = read_from_cli()
    elif args.input_type == "txt":
        # 获取当前进程的排名和总进程数
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        # 从文件读取数据
        data_iter = read_from_file(args.input_file, rank=rank, world_size=world_size)
    else:
        # 如果输入类型未实现，则抛出异常
        raise NotImplementedError

    # 获取采样图像的尺寸
    image_size_x = args.sampling_image_size_x
    image_size_y = args.sampling_image_size_y
    # 组合成图像大小元组
    image_size = (image_size_x, image_size_y)
    # 获取潜在维度和采样参数
    latent_dim = args.sampling_latent_dim
    f = args.sampling_f

    # 检查图像尺寸是否在有效范围内
    assert (
        image_size_x >= 512 and image_size_y >= 512 and image_size_x <= 2048 and image_size_y <= 2048
    ), "Image size should be between 512 and 2048"
    # 检查图像尺寸是否为 32 的倍数
    assert image_size_x % 32 == 0 and image_size_y % 32 == 0, "Image size should be divisible by 32"

    # 获取模型的采样函数
    sample_func = model.sample

    # 定义图像的高、宽、通道数和采样参数
    H, W, C, F = image_size_x, image_size_y, latent_dim, f
    # 定义样本数量
    num_samples = [args.batch_size]
    # 定义强制使用的嵌入类型
    force_uc_zero_embeddings = ["txt"]
    # 禁用梯度计算，节省内存和加快计算速度
    with torch.no_grad():
        # 遍历数据迭代器中的文本和计数
        for text, cnt in tqdm(data_iter):
            # 创建一个字典来存储生成图像所需的参数
            value_dict = {
                # 提供的提示文本
                "prompt": text,
                # 负提示文本为空
                "negative_prompt": "",
                # 原始图像尺寸以元组形式存储
                "original_size_as_tuple": image_size,
                # 目标图像尺寸以元组形式存储
                "target_size_as_tuple": image_size,
                # 原始图像高度
                "orig_height": image_size_x,
                # 原始图像宽度
                "orig_width": image_size_y,
                # 目标图像高度
                "target_height": image_size_x,
                # 目标图像宽度
                "target_width": image_size_y,
                # 裁剪区域的上边界
                "crop_coords_top": 0,
                # 裁剪区域的左边界
                "crop_coords_left": 0,
            }

            # 获取批量数据和无条件批量数据
            batch, batch_uc = get_batch(
                # 从条件器中获取唯一的嵌入键
                get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples
            )

            # 获取无条件条件的上下文
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                # 传递无条件批量
                batch_uc=batch_uc,
                # 是否强制将无条件嵌入设置为零
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )

            # 遍历条件上下文
            for k in c:
                # 如果不是交叉注意力
                if not k == "crossattn":
                    # 将每个上下文和无条件上下文映射到 CUDA
                    c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))

            # 生成样本
            samples_z = sample_func(
                c,
                uc=uc,
                # 批量大小
                batch_size=args.batch_size,
                # 目标形状
                shape=(C, H // F, W // F),
                # 目标图像尺寸
                target_size=[image_size],
            )

            # 解码生成的样本
            samples_x = model.decode_first_stage(samples_z).to(torch.float32)
            # 将样本标准化到 [0, 1] 范围内，并移到 CPU
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()
            # 获取样本的批量大小
            batch_size = samples.shape[0]
            # 断言确保批量大小能被列数整除
            assert (batch_size // args.grid_num_columns) * args.grid_num_columns == batch_size

            # 如果批量大小为 1，则不生成网格
            if args.batch_size == 1:
                grid = None
            else:
                # 生成样本的网格
                grid = make_grid(samples, nrow=args.grid_num_columns)

            # 生成保存路径
            save_path = os.path.join(args.output_dir, str(cnt) + "_" + text.replace(" ", "_").replace("/", "")[:20])
            # 在本地保存样本和网格
            perform_save_locally(save_path, samples, grid)
# 当脚本作为主程序运行时执行以下代码
if __name__ == "__main__":
    # 创建一个解析命令行参数的解析器，且不自动添加帮助信息
    py_parser = argparse.ArgumentParser(add_help=False)
    # 解析已知参数和剩余参数
    known, args_list = py_parser.parse_known_args()

    # 调用 get_args 函数处理剩余参数，返回结果
    args = get_args(args_list)
    # 将已知参数和处理后的参数合并为一个命名空间对象
    args = argparse.Namespace(**vars(args), **vars(known))

    # 调用 sampling_main 函数，传入参数和模型类
    sampling_main(args, model_cls=SATDiffusionEngine)
```