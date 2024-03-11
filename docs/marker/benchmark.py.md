# `.\marker\benchmark.py`

```py
import argparse
import tempfile
import time
from collections import defaultdict

from tqdm import tqdm

from marker.convert import convert_single_pdf
from marker.logger import configure_logging
from marker.models import load_all_models
from marker.benchmark.scoring import score_text
from marker.extract_text import naive_get_text
import json
import os
import subprocess
import shutil
import fitz as pymupdf
from tabulate import tabulate

# 配置日志记录
configure_logging()

# 定义函数，使用 Nougat 进行预测
def nougat_prediction(pdf_filename, batch_size=1):
    # 创建临时目录
    out_dir = tempfile.mkdtemp()
    # 运行 Nougat 命令行工具进行预测
    subprocess.run(["nougat", pdf_filename, "-o", out_dir, "--no-skipping", "--recompute", "--batchsize", str(batch_size)], check=True)
    # 获取生成的 Markdown 文件
    md_file = os.listdir(out_dir)[0]
    with open(os.path.join(out_dir, md_file), "r") as f:
        data = f.read()
    # 删除临时目录
    shutil.rmtree(out_dir)
    return data

# 主函数
def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Benchmark PDF to MD conversion.  Needs source pdfs, and a refernece folder with the correct markdown.")
    # 添加参数：输入 PDF 文件夹
    parser.add_argument("in_folder", help="Input PDF files")
    # 添加参数：参考 Markdown 文件夹
    parser.add_argument("reference_folder", help="Reference folder with reference markdown files")
    # 添加参数：输出文件名
    parser.add_argument("out_file", help="Output filename")
    # 添加参数：是否运行 Nougat 并比较
    parser.add_argument("--nougat", action="store_true", help="Run nougat and compare", default=False)
    # 添加参数：Nougat 批处理大小，默认为 1
    parser.add_argument("--nougat_batch_size", type=int, default=1, help="Batch size to use for nougat when making predictions.")
    # 添加参数：Marker 并行因子，默认为 1
    parser.add_argument("--marker_parallel_factor", type=int, default=1, help="How much to multiply default parallel OCR workers and model batch sizes by.")
    # 添加参数：生成的 Markdown 文件输出路径
    parser.add_argument("--md_out_path", type=str, default=None, help="Output path for generated markdown files")
    # 解析参数
    args = parser.parse_args()

    # 定义方法列表
    methods = ["naive", "marker"]
    if args.nougat:
        methods.append("nougat")

    # 加载所有模型
    model_lst = load_all_models()

    # 初始化得分字典
    scores = defaultdict(dict)
    # 获取指定文件夹中的所有文件列表
    benchmark_files = os.listdir(args.in_folder)
    # 筛选出以".pdf"结尾的文件列表
    benchmark_files = [b for b in benchmark_files if b.endswith(".pdf")]
    # 初始化存储时间信息的字典
    times = defaultdict(dict)
    # 初始化存储页数信息的字典
    pages = defaultdict(int)

    # 遍历每个 PDF 文件
    for fname in tqdm(benchmark_files):
        # 生成对应的 markdown 文件名
        md_filename = fname.rsplit(".", 1)[0] + ".md"

        # 获取参考文件的路径并读取内容
        reference_filename = os.path.join(args.reference_folder, md_filename)
        with open(reference_filename, "r") as f:
            reference = f.read()

        # 获取 PDF 文件的路径并打开
        pdf_filename = os.path.join(args.in_folder, fname)
        doc = pymupdf.open(pdf_filename)
        # 记录该 PDF 文件的页数
        pages[fname] = len(doc)

        # 遍历不同的方法
        for method in methods:
            start = time.time()
            # 根据不同方法进行处理
            if method == "marker":
                full_text, out_meta = convert_single_pdf(pdf_filename, model_lst, parallel_factor=args.marker_parallel_factor)
            elif method == "nougat":
                full_text = nougat_prediction(pdf_filename, batch_size=args.nougat_batch_size)
            elif method == "naive":
                full_text = naive_get_text(doc)
            else:
                raise ValueError(f"Unknown method {method}")

            # 计算处理时间并记录
            times[method][fname] = time.time() - start

            # 计算得分并记录
            score = score_text(full_text, reference)
            scores[method][fname] = score

            # 如果指定了 markdown 输出路径，则将处理结果写入文件
            if args.md_out_path:
                md_out_filename = f"{method}_{md_filename}"
                with open(os.path.join(args.md_out_path, md_out_filename), "w+") as f:
                    f.write(full_text)

    # 计算总页数
    total_pages = sum(pages.values())
    # 打开输出文件，以写入模式打开，如果文件不存在则创建
    with open(args.out_file, "w+") as f:
        # 创建一个默认字典，用于存储数据
        write_data = defaultdict(dict)
        # 遍历每个方法
        for method in methods:
            # 计算每个方法的总时间
            total_time = sum(times[method].values())
            # 为每个文件创建统计信息字典
            file_stats = {
                fname:
                {
                    "time": times[method][fname],
                    "score": scores[method][fname],
                    "pages": pages[fname]
                }
                for fname in benchmark_files
            }
            # 将文件统计信息和方法的平均分数、每页时间、每个文档时间存储到 write_data 中
            write_data[method] = {
                "files": file_stats,
                "avg_score": sum(scores[method].values()) / len(scores[method]),
                "time_per_page": total_time / total_pages,
                "time_per_doc": total_time / len(scores[method])
            }

        # 将 write_data 写入到输出文件中，格式化为 JSON 格式，缩进为 4
        json.dump(write_data, f, indent=4)

    # 创建两个空列表用于存储汇总表和分数表
    summary_table = []
    score_table = []
    # 分数表的表头为 benchmark_files
    score_headers = benchmark_files
    # 遍历每个方法
    for method in methods:
        # 将方法、平均分数、每页时间、每个文档时间添加到汇总表中
        summary_table.append([method, write_data[method]["avg_score"], write_data[method]["time_per_page"], write_data[method]["time_per_doc"]])
        # 将方法和每个文件的分数添加到分数表中
        score_table.append([method, *[write_data[method]["files"][h]["score"] for h in score_headers]])

    # 打印汇总表，包括方法、平均分数、每页时间、每个文档时间
    print(tabulate(summary_table, headers=["Method", "Average Score", "Time per page", "Time per document"]))
    print("")
    print("Scores by file")
    # 打印分数表，包括方法和每个文件的分数
    print(tabulate(score_table, headers=["Method", *score_headers]))
# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```