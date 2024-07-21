# `.\pytorch\scripts\release_notes\namespace_check.py`

```py
# 导入 argparse 模块，用于处理命令行参数
import argparse
# 导入 json 模块，用于 JSON 数据的读写操作
import json
# 从 os 模块中导入 path 函数，用于处理文件路径
from os import path

# 导入 torch 模块
import torch

# 导入所有的 utils 模块，以便下面的 getattr 能够找到它们

# 定义了包含所有子模块的列表
all_submod_list = [
    "",
    "nn",
    "nn.functional",
    "nn.init",
    "optim",
    "autograd",
    "cuda",
    "sparse",
    "distributions",
    "fft",
    "linalg",
    "jit",
    "distributed",
    "futures",
    "onnx",
    "random",
    "utils.bottleneck",
    "utils.checkpoint",
    "utils.data",
    "utils.model_zoo",
]

# 根据子模块名称获取其内容列表
def get_content(submod):
    mod = torch
    if submod:
        submod = submod.split(".")
        for name in submod:
            mod = getattr(mod, name)
    content = dir(mod)
    return content

# 过滤掉私有成员后返回数据
def namespace_filter(data):
    out = {d for d in data if d[0] != "_"}
    return out

# 主函数，根据命令行参数运行不同的逻辑
def run(args, submod):
    # 打印正在处理的 torch 下的子模块
    print(f"## Processing torch.{submod}")
    # 定义前一个版本和新版本数据的文件名
    prev_filename = f"prev_data_{submod}.json"
    new_filename = f"new_data_{submod}.json"

    # 如果需要保存前一个版本的数据
    if args.prev_version:
        # 获取指定子模块的内容列表
        content = get_content(submod)
        # 将内容列表保存到 JSON 文件中
        with open(prev_filename, "w") as f:
            json.dump(content, f)
        print("Data saved for previous version.")
    
    # 如果需要保存新版本的数据
    elif args.new_version:
        # 获取指定子模块的内容列表
        content = get_content(submod)
        # 将内容列表保存到 JSON 文件中
        with open(new_filename, "w") as f:
            json.dump(content, f)
        print("Data saved for new version.")
    
    # 如果需要比较两个版本的数据
    else:
        # 确保之前版本的数据文件存在
        if not path.exists(prev_filename):
            raise RuntimeError("Previous version data not collected")
        
        # 确保新版本的数据文件存在
        if not path.exists(new_filename):
            raise RuntimeError("New version data not collected")
        
        # 读取前一个版本数据文件的内容，并转为集合形式
        with open(prev_filename) as f:
            prev_content = set(json.load(f))
        
        # 读取新版本数据文件的内容，并转为集合形式
        with open(new_filename) as f:
            new_content = set(json.load(f))
        
        # 如果不展示所有的差异，只展示公共 API
        if not args.show_all:
            prev_content = namespace_filter(prev_content)
            new_content = namespace_filter(new_content)
        
        # 比较前一个版本和新版本的内容
        if new_content == prev_content:
            print("Nothing changed.")
            print("")
        else:
            print("Things that were added:")
            print(new_content - prev_content)
            print("")

            print("Things that were removed:")
            print(prev_content - new_content)
            print("")

# 主程序入口
def main():
    # 创建命令行解析对象
    parser = argparse.ArgumentParser(
        description="Tool to check namespace content changes"
    )

    # 创建互斥参数组，用于确定运行模式
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prev-version", action="store_true")
    group.add_argument("--new-version", action="store_true")
    group.add_argument("--compare", action="store_true")

    # 创建互斥参数组，用于指定子模块
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--submod", default="", help="part of the submodule to check")
    group.add_argument(
        "--all-submod",
        action="store_true",
        help="collects data for all main submodules",
    )

    # 添加参数，指定是否显示所有差异而不仅仅是公共 API
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="show all the diff, not just public APIs",
    )
    # 解析命令行参数并存储到 args 变量中
    args = parser.parse_args()
    
    # 如果命令行参数中指定了 all_submod 选项，则使用所有子模块列表
    if args.all_submod:
        submods = all_submod_list
    # 否则，只使用命令行参数中指定的单个子模块
    else:
        submods = [args.submod]
    
    # 遍历所有子模块并依次执行指定的程序
    for mod in submods:
        run(args, mod)
# 如果当前脚本作为主程序运行，则执行 main() 函数
if __name__ == "__main__":
    main()
```