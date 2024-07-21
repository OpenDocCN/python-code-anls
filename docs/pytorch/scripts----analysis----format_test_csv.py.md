# `.\pytorch\scripts\analysis\format_test_csv.py`

```
"""
This script takes a pytest CSV file produced by pytest --csv foo.csv
and summarizes it into a more minimal CSV that is good for uploading
to Google Sheets.  We have been using this with dynamic shapes to
understand how many tests fail when we turn on dynamic shapes.  If
you have a test suite with a lot of skips or xfails, if force the
tests to run anyway, this can help you understand what the actual
errors things are failing with are.

The resulting csv is written to stdout.  An easy way to get the csv
onto your local file system is to send it to GitHub Gist:

    $ python scripts/analysis/format_test_csv.py foo.csv | gh gist create -

See also scripts/analysis/run_test_csv.sh
"""

# 导入必要的模块
import argparse  # 导入处理命令行参数的模块
import csv       # 导入处理 CSV 文件的模块
import subprocess    # 导入执行外部命令的模块
import sys       # 导入系统相关的模块

# 解析命令行参数
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--log-url", type=str, default="", help="URL of raw logs")
parser.add_argument("file", help="pytest CSV file to format")
args = parser.parse_args()

# 将输出结果写入标准输出流，使用 Excel 方式的 CSV 格式
out = csv.writer(sys.stdout, dialect="excel")

# 获取当前 Git 仓库的提交哈希值作为第一列
hash = subprocess.check_output(
    "git rev-parse HEAD".split(" "), encoding="utf-8"
).rstrip()
out.writerow([hash, args.log_url, ""])

# 打开指定的 pytest CSV 文件进行读取
with open(args.file) as f:
    reader = csv.DictReader(f)  # 使用字典方式读取 CSV 文件内容
    for row in reader:
        if row["status"] not in {"failed", "error"}:
            continue
        
        # 提取消息内容，并处理特定的字符串替换操作
        msg = row["message"].split("\n")[0]
        msg.replace(
            " - erroring out! It's likely that this is caused by data-dependent control flow or similar.",
            "",
        )
        msg.replace("\t", " ")

        # 对测试名称进行处理，删除指定的前缀以清理输出结果
        # 可以根据需要自行修改这部分代码，目的是清理生成的电子表格输出
        name = row["name"].replace("test_make_fx_symbolic_exhaustive_", "")
        out.writerow([name, msg, ""])
```