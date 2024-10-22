# `.\cogview3-finetune\sat\sample_unet.py`

```
# 导入操作系统模块
import os
# 导入数学模块
import math
# 导入命令行参数解析模块
import argparse
# 导入进度条模块
from tqdm import tqdm
# 导入列表和联合类型的类型注解
from typing import List, Union
# 从 OmegaConf 导入列表配置
from omegaconf import ListConfig
# 导入图像处理库
from PIL import Image

# 导入 PyTorch 库
import torch
# 导入 PyTorch 中的功能模块
import torch.nn.functional as functional
# 导入 NumPy 库
import numpy as np
# 从 einops 导入重新排列和重复函数
from einops import rearrange, repeat
# 从 torchvision 导入生成网格的工具
from torchvision.utils import make_grid
# 导入 torchvision 的变换模块
import torchvision.transforms as TT

# 从自定义模型模块导入获取模型的函数
from sat.model.base_model import get_model
# 从自定义训练模块导入加载检查点的函数
from sat.training.model_io import load_checkpoint

# 导入扩散模型引擎
from diffusion import SATDiffusionEngine
# 导入命令行参数获取函数
from arguments import get_args


# 定义从命令行读取输入的生成器函数
def read_from_cli():
    # 初始化计数器
    cnt = 0
    # 尝试读取输入
    try:
        while True:
            # 提示用户输入英文文本，直到 Ctrl-D 结束
            x = input("Please input English text (Ctrl-D quit): ")
            # 去掉输入字符串的前后空白，并生成 (输入字符串, 计数) 元组
            yield x.strip(), cnt
            # 计数器加一
            cnt += 1
    # 捕获 EOFError 异常，表示输入结束
    except EOFError as e:
        pass


# 定义从文件读取输入的生成器函数
def read_from_file(p, rank=0, world_size=1):
    # 打开指定路径的文件，读取模式
    with open(p, "r") as fin:
        # 初始化计数器
        cnt = -1
        # 遍历文件中的每一行
        for l in fin:
            # 计数器加一
            cnt += 1
            # 如果当前计数不符合当前进程的 rank，则跳过该行
            if cnt % world_size != rank:
                continue
            # 去掉行末空白并生成 (行内容, 计数) 元组
            yield l.strip(), cnt


# 定义从条件器中获取唯一嵌入键的函数
def get_unique_embedder_keys_from_conditioner(conditioner):
    # 从条件器的嵌入器中提取输入键，去重并转换为列表
    return list(set([x.input_key for x in conditioner.embedders]))


# 定义获取批次数据的函数
def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    # 初始化批次字典
    batch = {}
    # 初始化无条件批次字典
    batch_uc = {}
    # 遍历指定的键
        for key in keys:
            # 如果键是 "txt"，则处理相关数据
            if key == "txt":
                # 重复 prompt 值，生成指定大小的数组，并转换为列表
                batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
                # 重复 negative_prompt 值，生成指定大小的数组，并转换为列表
                batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            # 如果键是 "original_size_as_tuple"，则处理原始尺寸
            elif key == "original_size_as_tuple":
                # 创建一个张量，包含原始高度和宽度，并在设备上重复
                batch["original_size_as_tuple"] = (
                    torch.tensor([value_dict["orig_height"], value_dict["orig_width"]]).to(device).repeat(*N, 1)
                )
            # 如果键是 "crop_coords_top_left"，则处理裁剪坐标
            elif key == "crop_coords_top_left":
                # 创建一个张量，包含裁剪的顶部和左侧坐标，并在设备上重复
                batch["crop_coords_top_left"] = (
                    torch.tensor([value_dict["crop_coords_top"], value_dict["crop_coords_left"]]).to(device).repeat(*N, 1)
                )
            # 如果键是 "aesthetic_score"，则处理美学评分
            elif key == "aesthetic_score":
                # 创建一个张量，包含美学评分，并在设备上重复
                batch["aesthetic_score"] = torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
                # 创建一个张量，包含负面美学评分，并在设备上重复
                batch_uc["aesthetic_score"] = (
                    torch.tensor([value_dict["negative_aesthetic_score"]]).to(device).repeat(*N, 1)
                )
            # 如果键是 "target_size_as_tuple"，则处理目标尺寸
            elif key == "target_size_as_tuple":
                # 创建一个张量，包含目标高度和宽度，并在设备上重复
                batch["target_size_as_tuple"] = (
                    torch.tensor([value_dict["target_height"], value_dict["target_width"]]).to(device).repeat(*N, 1)
                )
            # 如果键是 "fps"，则处理帧率
            elif key == "fps":
                # 创建一个张量，包含帧率值，并在设备上重复
                batch[key] = torch.tensor([value_dict["fps"]]).to(device).repeat(math.prod(N))
            # 如果键是 "fps_id"，则处理帧率ID
            elif key == "fps_id":
                # 创建一个张量，包含帧率ID值，并在设备上重复
                batch[key] = torch.tensor([value_dict["fps_id"]]).to(device).repeat(math.prod(N))
            # 如果键是 "motion_bucket_id"，则处理运动桶ID
            elif key == "motion_bucket_id":
                # 创建一个张量，包含运动桶ID值，并在设备上重复
                batch[key] = torch.tensor([value_dict["motion_bucket_id"]]).to(device).repeat(math.prod(N))
            # 如果键是 "pool_image"，则处理池图像
            elif key == "pool_image":
                # 重复池图像值，转换维度，并在设备上设置数据类型
                batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=math.prod(N)).to(device, dtype=torch.half)
            # 如果键是 "cond_aug"，则处理条件增强
            elif key == "cond_aug":
                # 创建一个张量，包含条件增强值，并在设备上重复
                batch[key] = repeat(
                    torch.tensor([value_dict["cond_aug"]]).to("cuda"),
                    "1 -> b",
                    b=math.prod(N),
                )
            # 如果键是 "cond_frames"，则处理条件帧
            elif key == "cond_frames":
                # 重复条件帧值，转换维度
                batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
            # 如果键是 "cond_frames_without_noise"，则处理无噪声条件帧
            elif key == "cond_frames_without_noise":
                # 重复无噪声条件帧值，转换维度
                batch[key] = repeat(value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0])
            # 如果键是 "cfg_scale"，则处理配置缩放
            elif key == "cfg_scale":
                # 创建一个张量，包含配置缩放值，并在设备上重复
                batch[key] = torch.tensor([value_dict["cfg_scale"]]).to(device).repeat(math.prod(N))
            # 如果键不在上述条件中，则直接从 value_dict 复制值
            else:
                batch[key] = value_dict[key]
    
        # 如果 T 不是 None，设置视频帧数量
        if T is not None:
            batch["num_video_frames"] = T
    
        # 遍历 batch 中的键
        for key in batch.keys():
            # 如果键不在 batch_uc 中，并且对应的值是张量
            if key not in batch_uc and isinstance(batch[key], torch.Tensor):
                # 克隆张量并存储到 batch_uc 中
                batch_uc[key] = torch.clone(batch[key])
        # 返回处理后的 batch 和 batch_uc
        return batch, batch_uc
# 定义一个将样本保存到本地的函数，参数包括保存路径、样本、网格和是否只保存网格的标志
def perform_save_locally(save_path, samples, grid, only_save_grid=False):
    # 创建保存路径的目录，如果已存在则不报错
    os.makedirs(save_path, exist_ok=True)

    # 如果不只保存网格
    if not only_save_grid:
        # 遍历样本及其索引
        for i, sample in enumerate(samples):
            # 将样本从张量格式转为图片格式，并进行归一化处理
            sample = 255.0 * rearrange(sample.numpy(), "c h w -> h w c")
            # 将处理后的样本保存为 PNG 格式，文件名以索引命名，前面填充零
            Image.fromarray(sample.astype(np.uint8)).save(os.path.join(save_path, f"{i:09}.png"))

    # 如果网格不为 None
    if grid is not None:
        # 将网格从张量格式转为图片格式，并进行归一化处理
        grid = 255.0 * rearrange(grid.numpy(), "c h w -> h w c")
        # 将处理后的网格保存为 PNG 格式，文件名为 grid.png
        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(save_path, f"grid.png"))


# 定义一个主采样函数，参数包括输入参数和模型类
def sampling_main(args, model_cls):
    # 判断 model_cls 是否为类类型
    if isinstance(model_cls, type):
        # 获取模型实例
        model = get_model(args, model_cls)
    else:
        # 如果不是类，直接赋值为模型
        model = model_cls

    # 加载模型的检查点
    load_checkpoint(model, args)
    # 将模型设置为评估模式
    model.eval()

    # 根据输入类型读取数据
    if args.input_type == "cli":
        # 从命令行读取数据
        data_iter = read_from_cli()
    elif args.input_type == "txt":
        # 获取当前进程的排名和总进程数
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        # 从文件中读取数据
        data_iter = read_from_file(args.input_file, rank=rank, world_size=world_size)
    else:
        # 如果输入类型不在支持的范围，抛出未实现错误
        raise NotImplementedError

    # 获取采样图像的尺寸
    image_size = args.sampling_image_size
    input_sample_dirs = None
    # 如果启用中继模型
    if args.relay_model is True:
        # 使用中继采样函数
        sample_func = model.sample_relay
        # 设置图像的高度、宽度和通道数
        H, W, C, F = image_size, image_size, 4, 8
        # 确保输入目录不为 None
        assert args.input_dir is not None
        # 列出输入样本目录
        input_sample_dirs = os.listdir(args.input_dir)
        # 排序目录名称，并提取排名和名称
        input_sample_dirs_and_rank = sorted([(int(name.split("_")[0]), name) for name in input_sample_dirs])
        # 重新构建完整的输入样本目录路径
        input_sample_dirs = [os.path.join(args.input_dir, name) for _, name in input_sample_dirs_and_rank]
    else:
        # 使用常规采样函数
        sample_func = model.sample
        # 获取潜在维度和采样频率
        latent_dim = args.sampling_latent_dim
        f = args.sampling_f
        # 设置图像的高度、宽度、通道数和帧数
        H, W, C, F = image_size, image_size, latent_dim, f
    # 设置样本数量为批量大小
    num_samples = [args.batch_size]
    # 强制将特定嵌入维度设为零
    force_uc_zero_embeddings = ["txt"]
    # 禁用梯度计算，以节省内存和提高计算速度
        with torch.no_grad():
            # 遍历数据迭代器中的文本和计数
            for text, cnt in tqdm(data_iter):
                # 创建一个字典，存储与当前文本相关的参数
                value_dict = {
                    "prompt": text,  # 当前的提示文本
                    "negative_prompt": "",  # 负提示文本，初始为空
                    "original_size_as_tuple": (image_size, image_size),  # 原始图像尺寸
                    "target_size_as_tuple": (image_size, image_size),  # 目标图像尺寸
                    "orig_height": image_size,  # 原始高度
                    "orig_width": image_size,  # 原始宽度
                    "target_height": image_size,  # 目标高度
                    "target_width": image_size,  # 目标宽度
                    "crop_coords_top": 0,  # 裁剪坐标顶部
                    "crop_coords_left": 0,  # 裁剪坐标左侧
                }
    
                # 获取当前批次和无条件的嵌入
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples
                )
                # 获取无条件的条件
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )
    
                # 遍历条件字典，处理每个条件
                for k in c:
                    if not k == "crossattn":  # 如果键不是 "crossattn"
                        # 将条件和无条件条件的相应部分移动到 CUDA
                        c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))
                # 如果参数 relay_model 为真
                if args.relay_model is True:
                    # 获取输入样本目录
                    input_sample_dir = input_sample_dirs[cnt]
                    images = []  # 存储图像的列表
                    # 遍历批次大小
                    for i in range(args.batch_size):
                        # 构建图像文件路径
                        filepath = os.path.join(input_sample_dir, f"{i:09}.png")
                        # 打开图像并转换为 RGB 格式
                        image = Image.open(filepath).convert("RGB")
                        # 将图像转换为张量并标准化
                        image = TT.ToTensor()(image) * 2 - 1
                        # 将处理后的图像添加到列表
                        images.append(image[None, ...])
                    # 将图像列表合并为一个张量
                    images = torch.cat(images, dim=0)
                    # 将图像上采样
                    images = functional.interpolate(images, scale_factor=2, mode="bilinear", align_corners=False)
                    # 转换图像为半精度并移动到 CUDA
                    images = images.to(torch.float16).cuda()
                    # 编码第一阶段的图像
                    images = model.encode_first_stage(images)
                    # 进行采样
                    samples_z = sample_func(images, c, uc=uc, batch_size=args.batch_size, shape=(C, H // F, W // F))
                else:
                    # 直接进行采样
                    samples_z = sample_func(c, uc=uc, batch_size=args.batch_size, shape=(C, H // F, W // F))
                # 解码第一阶段的样本
                samples_x = model.decode_first_stage(samples_z).to(torch.float32)
                # 将样本归一化并转移到 CPU
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()
                # 获取批次大小
                batch_size = samples.shape[0]
                # 确保批次大小能够被列数整除
                assert (batch_size // args.grid_num_columns) * args.grid_num_columns == batch_size
    
                # 如果批次大小为 1，网格设为 None
                if args.batch_size == 1:
                    grid = None
                else:
                    # 创建网格，将样本放置在网格中
                    grid = make_grid(samples, nrow=args.grid_num_columns)
    
                # 构建保存路径
                save_path = os.path.join(args.output_dir, str(cnt) + "_" + text.replace(" ", "_").replace("/", "")[:20])
                # 执行本地保存样本和网格
                perform_save_locally(save_path, samples, grid)
# 当脚本作为主程序运行时执行以下代码
if __name__ == "__main__":
    # 创建一个命令行参数解析器，不自动添加帮助信息
    py_parser = argparse.ArgumentParser(add_help=False)
    # 解析已知参数和位置参数，返回已知参数和剩余参数列表
    known, args_list = py_parser.parse_known_args()

    # 从剩余参数列表中获取自定义参数
    args = get_args(args_list)
    # 将已知参数和自定义参数合并为一个命名空间对象
    args = argparse.Namespace(**vars(args), **vars(known))
    # 调用主函数，传入参数和模型类
    sampling_main(args, model_cls=SATDiffusionEngine)
```