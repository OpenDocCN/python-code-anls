# `.\marker\convert.py`

```py
# 导入必要的库
import argparse
import os
from typing import Dict, Optional

import ray
from tqdm import tqdm
import math

# 导入自定义模块
from marker.convert import convert_single_pdf, get_length_of_text
from marker.models import load_all_models
from marker.settings import settings
from marker.logger import configure_logging
import traceback
import json

# 配置日志记录
configure_logging()

# 定义一个远程函数，用于处理单个 PDF 文件
@ray.remote(num_cpus=settings.RAY_CORES_PER_WORKER, num_gpus=.05 if settings.CUDA else 0)
def process_single_pdf(fname: str, out_folder: str, model_refs, metadata: Optional[Dict] = None, min_length: Optional[int] = None):
    # 构建输出文件名和元数据文件名
    out_filename = fname.rsplit(".", 1)[0] + ".md"
    out_filename = os.path.join(out_folder, os.path.basename(out_filename))
    out_meta_filename = out_filename.rsplit(".", 1)[0] + "_meta.json"
    
    # 如果输出文件已存在，则直接返回
    if os.path.exists(out_filename):
        return
    
    try:
        # 如果指定了最小文本长度，检查文件文本长度是否符合要求
        if min_length:
            length = get_length_of_text(fname)
            if length < min_length:
                return
        
        # 转换 PDF 文件为 Markdown 格式，并获取转换后的文本和元数据
        full_text, out_metadata = convert_single_pdf(fname, model_refs, metadata=metadata)
        
        # 如果转换后的文本不为空，则写入到文件中
        if len(full_text.strip()) > 0:
            with open(out_filename, "w+", encoding='utf-8') as f:
                f.write(full_text)
            with open(out_meta_filename, "w+") as f:
                f.write(json.dumps(out_metadata, indent=4))
        else:
            print(f"Empty file: {fname}.  Could not convert.")
    except Exception as e:
        # 捕获异常并打印错误信息
        print(f"Error converting {fname}: {e}")
        print(traceback.format_exc())

# 主函数
def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Convert multiple pdfs to markdown.")
    
    # 添加输入文件夹和输出文件夹参数
    parser.add_argument("in_folder", help="Input folder with pdfs.")
    parser.add_argument("out_folder", help="Output folder")
    # 添加命令行参数，指定要转换的块索引
    parser.add_argument("--chunk_idx", type=int, default=0, help="Chunk index to convert")
    # 添加命令行参数，指定并行处理的块数
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks being processed in parallel")
    # 添加命令行参数，指定要转换的最大 pdf 数量
    parser.add_argument("--max", type=int, default=None, help="Maximum number of pdfs to convert")
    # 添加命令行参数，指定要使用的工作进程数
    parser.add_argument("--workers", type=int, default=5, help="Number of worker processes to use")
    # 添加命令行参数，指定要使用的元数据 json 文件进行过滤
    parser.add_argument("--metadata_file", type=str, default=None, help="Metadata json file to use for filtering")
    # 添加命令行参数，指定要转换的 pdf 的最小长度
    parser.add_argument("--min_length", type=int, default=None, help="Minimum length of pdf to convert")

    # 解析命令行参数
    args = parser.parse_args()

    # 获取输入文件夹的绝对路径
    in_folder = os.path.abspath(args.in_folder)
    # 获取输出文件夹的绝对路径
    out_folder = os.path.abspath(args.out_folder)
    # 获取输入文件夹中所有文件的路径列表
    files = [os.path.join(in_folder, f) for f in os.listdir(in_folder)]
    # 如果输出文件夹不存在，则创建输出文件夹
    os.makedirs(out_folder, exist_ok=True)

    # 处理并行处理时的块
    # 确保将所有文件放入一个块中
    chunk_size = math.ceil(len(files) / args.num_chunks)
    start_idx = args.chunk_idx * chunk_size
    end_idx = start_idx + chunk_size
    files_to_convert = files[start_idx:end_idx]

    # 如果需要，限制要转换的文件数量
    if args.max:
        files_to_convert = files_to_convert[:args.max]

    metadata = {}
    # 如果指定了元数据文件，则加载元数据
    if args.metadata_file:
        metadata_file = os.path.abspath(args.metadata_file)
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

    # 确定要使用的进程数
    total_processes = min(len(files_to_convert), args.workers)

    # 初始化 Ray，设置 CPU 和 GPU 数量，存储路径等参数
    ray.init(
        num_cpus=total_processes,
        num_gpus=1 if settings.CUDA else 0,
        storage=settings.RAY_CACHE_PATH,
        _temp_dir=settings.RAY_CACHE_PATH,
        log_to_driver=settings.DEBUG
    )

    # 加载所有模型
    model_lst = load_all_models()
    # 将模型列表放入 Ray 中
    model_refs = ray.put(model_lst)

    # 根据 GPU 内存动态设置每个任务的 GPU 分配比例
    gpu_frac = settings.VRAM_PER_TASK / settings.INFERENCE_RAM if settings.CUDA else 0
    # 打印正在转换的 PDF 文件数量、当前处理的块索引、总块数、使用的进程数以及输出文件夹路径
    print(f"Converting {len(files_to_convert)} pdfs in chunk {args.chunk_idx + 1}/{args.num_chunks} with {total_processes} processes, and storing in {out_folder}")
    
    # 为每个需要转换的 PDF 文件创建一个 Ray 任务，并指定使用的 GPU 分数
    futures = [
        process_single_pdf.options(num_gpus=gpu_frac).remote(
            filename,
            out_folder,
            model_refs,
            metadata=metadata.get(os.path.basename(filename)),
            min_length=args.min_length
        ) for filename in files_to_convert
    ]

    # 运行所有的 Ray 转换任务
    progress_bar = tqdm(total=len(futures))
    while len(futures) > 0:
        # 等待所有任务完成，超时时间为 7 秒
        finished, futures = ray.wait(
            futures, timeout=7.0
        )
        finished_lst = ray.get(finished)
        # 更新进度条
        if isinstance(finished_lst, list):
            progress_bar.update(len(finished_lst))
        else:
            progress_bar.update(1)

    # 关闭 Ray 以释放资源
    ray.shutdown()
# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```