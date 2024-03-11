# `.\marker\convert_single.py`

```py
# 导入必要的模块
import argparse  # 用于解析命令行参数
from marker.convert import convert_single_pdf  # 导入 convert_single_pdf 函数
from marker.logger import configure_logging  # 导入 configure_logging 函数
from marker.models import load_all_models  # 导入 load_all_models 函数
import json  # 导入 json 模块

# 配置日志记录
configure_logging()

# 主函数
def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument("filename", help="PDF file to parse")  # PDF 文件名
    parser.add_argument("output", help="Output file name")  # 输出文件名
    parser.add_argument("--max_pages", type=int, default=None, help="Maximum number of pages to parse")  # 最大解析页数
    parser.add_argument("--parallel_factor", type=int, default=1, help="How much to multiply default parallel OCR workers and model batch sizes by.")  # 并行因子
    # 解析命令行参数
    args = parser.parse_args()

    # 获取文件名
    fname = args.filename
    # 加载所有模型
    model_lst = load_all_models()
    # 调用 convert_single_pdf 函数，解析 PDF 文件并返回全文和元数据
    full_text, out_meta = convert_single_pdf(fname, model_lst, max_pages=args.max_pages, parallel_factor=args.parallel_factor)

    # 将全文写入输出文件
    with open(args.output, "w+", encoding='utf-8') as f:
        f.write(full_text)

    # 生成元数据文件名
    out_meta_filename = args.output.rsplit(".", 1)[0] + "_meta.json"
    # 将元数据写入元数据文件
    with open(out_meta_filename, "w+") as f:
        f.write(json.dumps(out_meta, indent=4))

# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```