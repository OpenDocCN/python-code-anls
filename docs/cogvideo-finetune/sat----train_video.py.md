# `.\cogvideo-finetune\sat\train_video.py`

```py
# 导入操作系统模块
import os
# 导入命令行参数解析模块
import argparse
# 从 functools 模块导入 partial 函数，用于部分应用
from functools import partial
# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 的分布式模块
import torch.distributed
# 导入 OmegaConf，用于配置管理
from omegaconf import OmegaConf
# 导入 imageio，用于读取和写入视频
import imageio

# 导入 PyTorch 库
import torch

# 从 sat 模块导入 mpu
from sat import mpu
# 从 sat.training.deepspeed_training 导入 training_main
from sat.training.deepspeed_training import training_main

# 从 sgm.util 导入工具函数
from sgm.util import get_obj_from_str, isheatmap

# 从 diffusion_video 导入 SATVideoDiffusionEngine 类
from diffusion_video import SATVideoDiffusionEngine
# 从 arguments 导入获取命令行参数的函数
from arguments import get_args

# 从 einops 导入 rearrange 函数，用于重新排列张量
from einops import rearrange

# 尝试导入 wandb 库，用于实验跟踪
try:
    import wandb
# 如果 wandb 未安装，打印警告信息
except ImportError:
    print("warning: wandb not installed")


# 打印调试信息的函数
def print_debug(args, s):
    # 如果启用了调试模式
    if args.debug:
        # 添加当前进程的排名到输出字符串
        s = f"RANK:[{torch.distributed.get_rank()}]:" + s
        # 打印调试信息
        print(s)


# 保存文本列表到指定目录的函数
def save_texts(texts, save_dir, iterations):
    # 构建输出文件的路径
    output_path = os.path.join(save_dir, f"{str(iterations).zfill(8)}")
    # 打开输出文件，设置编码为 UTF-8
    with open(output_path, "w", encoding="utf-8") as f:
        # 遍历文本列表
        for text in texts:
            # 将每个文本写入文件，每个文本后换行
            f.write(text + "\n")


# 将视频批次保存为网格和 MP4 格式的函数
def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, T: int, fps: int = 5, args=None, key=None):
    # 创建保存路径，如果不存在则创建
    os.makedirs(save_path, exist_ok=True)

    # 遍历视频批次
    for i, vid in enumerate(video_batch):
        gif_frames = []  # 用于存储帧的列表
        # 遍历视频中的每一帧
        for frame in vid:
            # 重新排列帧的维度，从 (c, h, w) 转为 (h, w, c)
            frame = rearrange(frame, "c h w -> h w c")
            # 将帧数据缩放到 [0, 255] 并转换为 uint8 类型
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            # 将处理后的帧添加到列表中
            gif_frames.append(frame)
        # 构建当前视频保存的路径
        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")
        # 使用 imageio 创建视频写入器
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            # 将每一帧写入视频文件
            for frame in gif_frames:
                writer.append_data(frame)
        # 如果 args 存在且 wandb 被启用
        if args is not None and args.wandb:
            # 记录视频到 wandb
            wandb.log(
                {key + f"_video_{i}": wandb.Video(now_save_path, fps=fps, format="mp4")}, step=args.iteration + 1
            )


# 日志视频的函数
def log_video(batch, model, args, only_log_video_latents=False):
    # 获取文本数据
    texts = batch["txt"]
    # 构建保存文本的目录
    text_save_dir = os.path.join(args.save, "video_texts")
    # 创建保存文本目录
    os.makedirs(text_save_dir, exist_ok=True)
    # 保存文本数据到文件
    save_texts(texts, text_save_dir, args.iteration)

    # 准备 GPU 自动混合精度的参数
    gpu_autocast_kwargs = {
        "enabled": torch.is_autocast_enabled(),
        "dtype": torch.get_autocast_gpu_dtype(),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }
    # 在不计算梯度的上下文中，启用自动混合精度
    with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
        # 使用模型记录视频，是否只记录视频的潜在表示
        videos = model.log_video(batch, only_log_video_latents=only_log_video_latents)
    # 检查当前进程是否是主进程（rank 为 0）
    if torch.distributed.get_rank() == 0:
        # 设置视频保存的根目录
        root = os.path.join(args.save, "video")

        # 如果只记录视频潜在变量
        if only_log_video_latents:
            # 创建潜在变量的子目录
            root = os.path.join(root, "latents")
            # 格式化文件名，包含当前迭代次数
            filename = "{}_gs-{:06}".format("latents", args.iteration)
            # 生成完整的文件路径
            path = os.path.join(root, filename)
            # 创建保存路径的目录（如果不存在的话）
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            # 创建文件夹（如果不存在的话）
            os.makedirs(path, exist_ok=True)
            # 保存潜在变量数据到指定路径
            torch.save(videos["latents"], os.path.join(path, "latent.pt"))
        else:
            # 遍历所有视频数据
            for k in videos:
                # 获取当前视频的帧数
                N = videos[k].shape[0]
                # 如果不是热图，裁剪视频数据
                if not isheatmap(videos[k]):
                    videos[k] = videos[k][:N]
                # 如果视频数据是张量
                if isinstance(videos[k], torch.Tensor):
                    # 将张量分离、转换为浮点数并移动到 CPU
                    videos[k] = videos[k].detach().float().cpu()
                    # 如果不是热图，限制张量值范围
                    if not isheatmap(videos[k]):
                        videos[k] = torch.clamp(videos[k], -1.0, 1.0)

            # 获取当前批次的视频帧数
            num_frames = batch["num_frames"][0]
            # 获取当前批次的帧率
            fps = batch["fps"][0].cpu().item()
            # 如果只记录视频潜在变量
            if only_log_video_latents:
                # 创建潜在变量的子目录
                root = os.path.join(root, "latents")
                # 格式化文件名，包含当前迭代次数
                filename = "{}_gs-{:06}".format("latents", args.iteration)
                # 生成完整的文件路径
                path = os.path.join(root, filename)
                # 创建保存路径的目录（如果不存在的话）
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                # 创建文件夹（如果不存在的话）
                os.makedirs(path, exist_ok=True)
                # 保存潜在变量数据到指定路径
                torch.save(videos["latents"], os.path.join(path, "latents.pt"))
            else:
                # 遍历所有视频数据
                for k in videos:
                    # 将视频数据标准化到 [0, 1] 范围
                    samples = (videos[k] + 1.0) / 2.0
                    # 格式化文件名，包含当前迭代次数
                    filename = "{}_gs-{:06}".format(k, args.iteration)

                    # 生成完整的文件路径
                    path = os.path.join(root, filename)
                    # 创建保存路径的目录（如果不存在的话）
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    # 保存视频为网格和 MP4 格式
                    save_video_as_grid_and_mp4(samples, path, num_frames // fps, fps, args, k)
# 定义广播批处理函数
def broad_cast_batch(batch):
    # 获取模型并行世界的大小
    mp_size = mpu.get_model_parallel_world_size()
    # 获取全局进程的排名
    global_rank = torch.distributed.get_rank() // mp_size
    # 计算源进程的索引
    src = global_rank * mp_size

    # 如果批次中存在 mp4 数据，则获取各数据的形状
    if batch["mp4"] is not None:
        broadcast_shape = [batch["mp4"].shape, batch["fps"].shape, batch["num_frames"].shape]
    else:
        # 否则设置为 None
        broadcast_shape = None

    # 准备要广播的对象列表，包括文本和形状信息
    txt = [batch["txt"], broadcast_shape]
    # 广播对象列表到指定源
    torch.distributed.broadcast_object_list(txt, src=src, group=mpu.get_model_parallel_group())
    # 从广播结果中提取文本数据
    batch["txt"] = txt[0]

    # 获取广播后各数据的形状
    mp4_shape = txt[1][0]
    fps_shape = txt[1][1]
    num_frames_shape = txt[1][2]

    # 如果当前模型并行进程不是 0，则初始化对应的张量
    if mpu.get_model_parallel_rank() != 0:
        batch["mp4"] = torch.zeros(mp4_shape, device="cuda")
        batch["fps"] = torch.zeros(fps_shape, device="cuda", dtype=torch.long)
        batch["num_frames"] = torch.zeros(num_frames_shape, device="cuda", dtype=torch.long)

    # 广播 mp4 数据
    torch.distributed.broadcast(batch["mp4"], src=src, group=mpu.get_model_parallel_group())
    # 广播 fps 数据
    torch.distributed.broadcast(batch["fps"], src=src, group=mpu.get_model_parallel_group())
    # 广播 num_frames 数据
    torch.distributed.broadcast(batch["num_frames"], src=src, group=mpu.get_model_parallel_group())
    # 返回处理后的批次数据
    return batch


# 定义前向评估步骤函数
def forward_step_eval(data_iterator, model, args, timers, only_log_video_latents=False, data_class=None):
    # 如果当前模型并行进程是 0，开始数据加载
    if mpu.get_model_parallel_rank() == 0:
        timers("data loader").start()  # 启动计时器
        batch_video = next(data_iterator)  # 获取下一个批次数据
        timers("data loader").stop()  # 停止计时器

        # 如果 mp4 数据的维度为 6，重新调整其形状
        if len(batch_video["mp4"].shape) == 6:
            b, v = batch_video["mp4"].shape[:2]  # 提取批次和视频维度
            batch_video["mp4"] = batch_video["mp4"].view(-1, *batch_video["mp4"].shape[2:])  # 扁平化 mp4 数据
            txt = []  # 初始化文本列表
            # 遍历批次和视频维度，构建文本列表
            for i in range(b):
                for j in range(v):
                    txt.append(batch_video["txt"][j][i])
            batch_video["txt"] = txt  # 更新批次中的文本数据

        # 将批次中的每个张量转移到 GPU
        for key in batch_video:
            if isinstance(batch_video[key], torch.Tensor):
                batch_video[key] = batch_video[key].cuda()  # 转移张量到 CUDA 设备
    else:
        # 如果当前进程不是 0，初始化空的批次数据
        batch_video = {"mp4": None, "fps": None, "num_frames": None, "txt": None}
    # 调用广播函数以同步批次数据
    broad_cast_batch(batch_video)
    # 如果数据并行进程是 0，记录视频数据
    if mpu.get_data_parallel_rank() == 0:
        log_video(batch_video, model, args, only_log_video_latents=only_log_video_latents)

    # 在批次中添加全局步骤信息
    batch_video["global_step"] = args.iteration
    # 进行共享步骤并计算损失
    loss, loss_dict = model.shared_step(batch_video)
    # 将损失字典中的 bfloat16 类型转换为 float32
    for k in loss_dict:
        if loss_dict[k].dtype == torch.bfloat16:
            loss_dict[k] = loss_dict[k].to(torch.float32)  # 转换数据类型
    return loss, loss_dict  # 返回损失和损失字典


# 定义前向步骤函数
def forward_step(data_iterator, model, args, timers, data_class=None):
    # 检查当前模型并行进程的排名是否为0
    if mpu.get_model_parallel_rank() == 0:
        # 启动计时器以记录数据加载时间
        timers("data loader").start()
        # 从数据迭代器中获取下一个批次的数据
        batch = next(data_iterator)
        # 停止计时器
        timers("data loader").stop()
        # 遍历批次中的每个键
        for key in batch:
            # 检查当前键对应的值是否为 PyTorch 张量
            if isinstance(batch[key], torch.Tensor):
                # 将张量移到 GPU 上
                batch[key] = batch[key].cuda()

        # 检查当前进程的分布式排名是否为0
        if torch.distributed.get_rank() == 0:
            # 检查保存目录下是否存在训练配置文件
            if not os.path.exists(os.path.join(args.save, "training_config.yaml")):
                # 加载基础配置文件，并将其存储在列表中
                configs = [OmegaConf.load(cfg) for cfg in args.base]
                # 合并所有基础配置
                config = OmegaConf.merge(*configs)
                # 创建保存目录（如果不存在）
                os.makedirs(args.save, exist_ok=True)
                # 将合并后的配置保存为 YAML 文件
                OmegaConf.save(config=config, f=os.path.join(args.save, "training_config.yaml"))
    else:
        # 如果当前不是模型并行进程0，则创建一个空的批次字典
        batch = {"mp4": None, "fps": None, "num_frames": None, "txt": None}

    # 在批次字典中添加全局步数
    batch["global_step"] = args.iteration

    # 广播批次数据到所有进程
    broad_cast_batch(batch)

    # 执行模型的共享步骤，计算损失和损失字典
    loss, loss_dict = model.shared_step(batch)

    # 返回计算的损失和损失字典
    return loss, loss_dict
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 检查环境变量中是否存在 OMPI_COMM_WORLD_LOCAL_RANK
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        # 将 OMPI_COMM_WORLD_LOCAL_RANK 的值赋给 LOCAL_RANK 环境变量
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        # 将 OMPI_COMM_WORLD_SIZE 的值赋给 WORLD_SIZE 环境变量
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        # 将 OMPI_COMM_WORLD_RANK 的值赋给 RANK 环境变量
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

    # 创建一个不带帮助信息的命令行参数解析器
    py_parser = argparse.ArgumentParser(add_help=False)
    # 解析已知和未知的命令行参数
    known, args_list = py_parser.parse_known_args()
    # 根据解析得到的参数列表获取具体参数
    args = get_args(args_list)
    # 将已知参数和具体参数合并到一个命名空间中
    args = argparse.Namespace(**vars(args), **vars(known))

    # 根据目标字符串获取数据类对象
    data_class = get_obj_from_str(args.data_config["target"])
    # 使用部分应用来创建数据集函数
    create_dataset_function = partial(data_class.create_dataset_function, **args.data_config["params"])

    # 导入 YAML 库
    import yaml

    # 初始化配置列表
    configs = []
    # 遍历基础配置文件列表
    for config in args.base:
        # 以只读模式打开基础配置文件
        with open(config, "r") as f:
            # 安全加载 YAML 文件内容为字典
            base_config = yaml.safe_load(f)
        # 将加载的基础配置添加到配置列表中
        configs.append(base_config)
    # 将加载的配置列表赋给 args.log_config
    args.log_config = configs

    # 调用训练主函数，传入相关参数和函数
    training_main(
        args,
        model_cls=SATVideoDiffusionEngine,
        forward_step_function=partial(forward_step, data_class=data_class),
        forward_step_eval=partial(
            forward_step_eval, data_class=data_class, only_log_video_latents=args.only_log_video_latents
        ),
        create_dataset_function=create_dataset_function,
    )
```