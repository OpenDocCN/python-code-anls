# `.\pytorch\test\forward_backward_compatibility\dump_all_function_schemas.py`

```
# 导入 argparse 模块，用于处理命令行参数解析
import argparse

# 导入 torch 模块
import torch


# 定义 dump 函数，用于将 Torch 的 JIT schemas 写入指定文件中
def dump(filename):
    # 获取所有 Torch JIT schemas
    schemas = torch._C._jit_get_all_schemas()
    # 获取所有自定义类的 JIT schemas 并添加到 schemas 列表中
    schemas += torch._C._jit_get_custom_class_schemas()
    
    # 打开文件准备写入
    with open(filename, "w") as f:
        # 遍历 schemas 列表中的每个 schema
        for s in schemas:
            # 将 schema 转换为字符串写入文件
            f.write(str(s))
            # 写入换行符
            f.write("\n")


# 主程序入口，判断是否处于主程序执行环境下
if __name__ == "__main__":
    # 创建 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser(description="Process some integers.")
    
    # 添加命令行参数 -f 和 --filename，用于指定要输出 schemas 的文件名
    parser.add_argument(
        "-f",
        "--filename",
        help="filename to dump the schemas",
        type=str,
        default="schemas.txt",
    )
    
    # 解析命令行参数，并将结果存储在 args 对象中
    args = parser.parse_args()
    
    # 调用 dump 函数，将 schemas 写入指定的文件
    dump(args.filename)
```