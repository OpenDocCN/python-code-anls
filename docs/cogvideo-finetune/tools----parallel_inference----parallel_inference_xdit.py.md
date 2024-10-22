# `.\cogvideo-finetune\tools\parallel_inference\parallel_inference_xdit.py`

```py
# 这是一个用于 CogVideo 的并行推理脚本，原始脚本来源于 xDiT 项目
"""
This is a parallel inference script for CogVideo. The original script
can be found from the xDiT project at

https://github.com/xdit-project/xDiT/blob/main/examples/cogvideox_example.py

By using this code, the inference process is parallelized on multiple GPUs,
and thus speeded up.

Usage:
1. pip install xfuser
2. mkdir results
3. run the following command to generate video
torchrun --nproc_per_node=4 parallel_inference_xdit.py \
    --model <cogvideox-model-path> --ulysses_degree 1 --ring_degree 2 \
    --use_cfg_parallel --height 480 --width 720 --num_frames 9 \
    --prompt 'A small dog.'

You can also use the run.sh file in the same folder to automate running this
code for batch generation of videos, by running:

sh ./run.sh

"""

# 导入必要的库
import time
import torch
import torch.distributed
from diffusers import AutoencoderKLTemporalDecoder
from xfuser import xFuserCogVideoXPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
)
from diffusers.utils import export_to_video

# 主函数
def main():
    # 创建参数解析器并描述用途
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    # 添加命令行参数并解析
    args = xFuserArgs.add_cli_args(parser).parse_args()
    # 从命令行参数创建引擎配置
    engine_args = xFuserArgs.from_cli_args(args)

    # 检查 ulysses_degree 是否有效
    num_heads = 30
    # 如果 ulysses_degree 大于 0 且不是 num_heads 的因子，则引发错误
    if engine_args.ulysses_degree > 0 and num_heads % engine_args.ulysses_degree != 0:
        raise ValueError(
            f"ulysses_degree ({engine_args.ulysses_degree}) must be a divisor of the number of heads ({num_heads})"
        )

    # 创建引擎和输入配置
    engine_config, input_config = engine_args.create_config()
    # 获取本地进程的排名
    local_rank = get_world_group().local_rank

    # 从预训练模型加载管道
    pipe = xFuserCogVideoXPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )
    # 如果启用 CPU 离线，进行相应设置
    if args.enable_sequential_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
        pipe.vae.enable_tiling()
    else:
        # 将管道移动到指定的 GPU 设备
        device = torch.device(f"cuda:{local_rank}")
        pipe = pipe.to(device)

    # 重置 GPU 的峰值内存统计
    torch.cuda.reset_peak_memory_stats()
    # 记录开始时间
    start_time = time.time()

    # 执行推理，生成视频帧
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        guidance_scale=6,
    ).frames[0]

    # 记录结束时间
    end_time = time.time()
    # 计算推理耗时
    elapsed_time = end_time - start_time
    # 获取当前设备的峰值内存使用量
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
    # 构建包含各种并行配置参数的字符串，用于输出文件名
        parallel_info = (
            f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
            f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
            f"tp{engine_args.tensor_parallel_degree}_"
            f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
        )
        # 检查是否为数据并行的最后一组
        if is_dp_last_group():
            # 获取数据并行的全局大小
            world_size = get_data_parallel_world_size()
            # 根据输入配置构建分辨率字符串
            resolution = f"{input_config.width}x{input_config.height}"
            # 生成输出文件名，包含并行信息和分辨率
            output_filename = f"results/cogvideox_{parallel_info}_{resolution}.mp4"
            # 将输出内容导出为视频文件
            export_to_video(output, output_filename, fps=8)
            # 打印保存的输出文件名
            print(f"output saved to {output_filename}")
    
        # 检查当前进程是否为最后一个进程
        if get_world_group().rank == get_world_group().world_size - 1:
            # 打印当前周期的耗时和内存使用情况
            print(f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB")
        # 销毁分布式环境的运行时状态
        get_runtime_state().destory_distributed_env()
# 判断当前脚本是否为主程序
if __name__ == "__main__":
    # 调用主函数
    main()
```