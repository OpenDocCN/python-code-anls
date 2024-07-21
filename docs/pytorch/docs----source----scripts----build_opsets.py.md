# `.\pytorch\docs\source\scripts\build_opsets.py`

```py
# 导入必要的库
import os  # 导入操作系统相关的功能
from collections import OrderedDict  # 导入有序字典功能
from pathlib import Path  # 导入处理路径的功能

import torch  # 导入 PyTorch 库
import torch._prims as prims  # 导入 PyTorch 内部的 _prims 模块

from torchgen.gen import parse_native_yaml  # 从 torchgen.gen 模块导入 parse_native_yaml 函数

# 定义根目录为当前文件的上上上级目录
ROOT = Path(__file__).absolute().parent.parent.parent.parent

# 定义常量：原生函数 YAML 文件路径
NATIVE_FUNCTION_YAML_PATH = ROOT / Path("aten/src/ATen/native/native_functions.yaml")

# 定义常量：标签 YAML 文件路径
TAGS_YAML_PATH = ROOT / Path("aten/src/ATen/native/tags.yaml")

# 定义常量：构建目录路径
BUILD_DIR = "build/ir"

# 定义常量：ATen 操作 CSV 文件名
ATEN_OPS_CSV_FILE = "aten_ops.csv"

# 定义常量：Prims 操作 CSV 文件名
PRIMS_OPS_CSV_FILE = "prims_ops.csv"


# 函数：获取 ATen 操作列表
def get_aten():
    # 解析原生 YAML 文件和标签 YAML 文件，返回解析结果
    parsed_yaml = parse_native_yaml(NATIVE_FUNCTION_YAML_PATH, TAGS_YAML_PATH)
    native_functions = parsed_yaml.native_functions

    # 使用有序字典存储 ATen 操作
    aten_ops = OrderedDict()
    for function in native_functions:
        if "core" in function.tags:  # 如果操作包含 'core' 标签
            op_name = str(function.func.name)
            aten_ops[op_name] = function

    # 存储操作名称和模式的元组列表
    op_schema_pairs = []
    for key, op in sorted(aten_ops.items()):  # 对 ATen 操作按键名排序
        op_name = f"aten.{key}"
        schema = str(op.func).replace("*", r"\*")  # 获取操作模式并替换 '*' 字符

        op_schema_pairs.append((op_name, schema))

    return op_schema_pairs


# 函数：获取 Prims 操作列表
def get_prims():
    # 存储操作名称和模式的元组列表
    op_schema_pairs = []
    for op_name in prims.__all__:  # 遍历 prims 模块的所有成员
        op_overload = getattr(prims, op_name, None)

        if not isinstance(op_overload, torch._ops.OpOverload):  # 如果不是 OpOverload 类型则跳过
            continue

        op_overloadpacket = op_overload.overloadpacket

        op_name = str(op_overload).replace(".default", "")  # 获取操作名并移除 '.default' 后缀
        schema = op_overloadpacket.schema.replace("*", r"\*")  # 获取操作模式并替换 '*' 字符

        op_schema_pairs.append((op_name, schema))

    return op_schema_pairs


# 主函数：程序入口
def main():
    # 获取 ATen 操作列表和 Prims 操作列表
    aten_ops_list = get_aten()
    prims_ops_list = get_prims()

    # 创建构建目录（如果不存在）
    os.makedirs(BUILD_DIR, exist_ok=True)

    # 写入 ATen 操作列表到 CSV 文件
    with open(os.path.join(BUILD_DIR, ATEN_OPS_CSV_FILE), "w") as f:
        f.write("Operator,Schema\n")
        for name, schema in aten_ops_list:
            f.write(f'"``{name}``","{schema}"\n')

    # 写入 Prims 操作列表到 CSV 文件
    with open(os.path.join(BUILD_DIR, PRIMS_OPS_CSV_FILE), "w") as f:
        f.write("Operator,Schema\n")
        for name, schema in prims_ops_list:
            f.write(f'"``{name}``","{schema}"\n')


# 如果该脚本作为主程序运行，则执行主函数
if __name__ == "__main__":
    main()
```