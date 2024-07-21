# `.\pytorch\torchgen\shape_functions\gen_jit_shape_functions.py`

```py
#!/usr/bin/env python3
import os  # 导入标准库 os
import sys  # 导入标准库 sys
from importlib.util import module_from_spec, spec_from_file_location  # 从 importlib.util 中导入模块加载函数
from itertools import chain  # 从 itertools 中导入链式迭代函数 chain
from pathlib import Path  # 从 pathlib 中导入路径处理模块 Path


# 手动导入 shape function 模块，基于当前目录而不是直接使用 torch 的导入，
# 避免在运行脚本前需要重新编译 PyTorch。

file_path = Path.cwd() / "torch" / "jit" / "_shape_functions.py"  # 设置文件路径
module_name = "torch.jit._shape_functions"  # 设置模块名

err_msg = """Could not find shape functions file, please make sure
you are in the root directory of the Pytorch git repo"""  # 文件不存在时的错误消息
if not file_path.exists():  # 如果文件路径不存在
    raise Exception(err_msg)  # 抛出异常，输出错误消息

spec = spec_from_file_location(module_name, file_path)  # 从文件路径创建模块规范
assert spec is not None  # 断言模块规范不为空
module = module_from_spec(spec)  # 根据模块规范创建模块对象
sys.modules[module_name] = module  # 将模块对象添加到 sys.modules 中，使其可导入
assert spec.loader is not None  # 断言模块规范的加载器不为空
assert module is not None  # 断言模块不为空
spec.loader.exec_module(module)  # 执行模块加载器，加载模块内容

bounded_compute_graph_mapping = module.bounded_compute_graph_mapping  # 获取 bounded_compute_graph_mapping 对象
shape_compute_graph_mapping = module.shape_compute_graph_mapping  # 获取 shape_compute_graph_mapping 对象


SHAPE_HEADER = r"""
/**
 * @generated
 * This is an auto-generated file. Please do not modify it by hand.
 * To re-generate, please run:
 * cd ~/pytorch && python
 * torchgen/shape_functions/gen_jit_shape_functions.py
 */
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/serialized_shape_function_registry.h>

// clang-format off

namespace torch {
namespace jit {


std::string shape_funcs = ""
"""

# SHAPE_HEADER 注释说明了这是一个自动生成的文件，不应手动修改，指示了重新生成的方法和路径。包含了一些必要的头文件和命名空间声明。


DECOMP_CENTER = r"""


const std::string& GetSerializedShapeFunctions() {
  return shape_funcs;
}

"""

# DECOMP_CENTER 包含 GetSerializedShapeFunctions 函数的定义，返回 shape_funcs 的引用。


DECOMP_END = r"""
// clang-format on

} // namespace jit
} // namespace torch
"""

# DECOMP_END 结束了命名空间 torch::jit 的声明。


SERIALIZED_SHAPE_UTIL_FILE_NAME = "serialized_shape_function_registry.cpp"  # 定义序列化形状函数的文件名


def gen_serialized_decompisitions() -> str:
    already_serialized_names = set()  # 创建已序列化函数名称的集合
    unique_funcs = []  # 创建存放唯一函数的列表
    all_funcs = chain(  # 合并 shape_compute_graph_mapping 和 bounded_compute_graph_mapping 的所有函数
        shape_compute_graph_mapping.values(), *bounded_compute_graph_mapping.values()
    )
    for scripted_func in all_funcs:  # 遍历所有函数
        if scripted_func.name in already_serialized_names:  # 如果函数名已经在集合中
            continue  # 继续下一个循环
        already_serialized_names.add(scripted_func.name)  # 将函数名添加到已序列化集合中
        unique_funcs.append(scripted_func)  # 将函数添加到唯一函数列表中

    output_strs = []  # 创建输出字符串列表
    curr_str = ""  # 初始化当前字符串为空字符串
    for scripted_func in unique_funcs:  # 遍历唯一函数列表
        serialized_code = scripted_func.code  # 获取函数的序列化代码
        # 最大 Microsoft 字符串长度
        MAX_MSFT_STR_LEN = 2000
        if len(curr_str) + len(serialized_code) <= MAX_MSFT_STR_LEN:  # 如果当前字符串长度加上序列化代码长度不超过最大长度
            curr_str += "\n" + serialized_code  # 将序列化代码添加到当前字符串中
        else:  # 如果超过了最大长度
            output_strs.append(curr_str)  # 将当前字符串添加到输出字符串列表中
            curr_str = scripted_func.code  # 更新当前字符串为当前函数的序列化代码
    output_strs.append(curr_str)  # 将最后的当前字符串添加到输出字符串列表中

    final_output = ""  # 初始化最终输出字符串为空字符串
    # Windows 编译器不能正确处理相邻的字符串文字
    # 对于输出字符串列表中的每一个字符串，进行处理
    for output_str in output_strs:
        # 构造起始字符串，使用原始字符串（raw string）语法
        start = '+ std::string(R"=====('
        # 构造结束字符串，包含换行符的原始字符串（raw string）语法
        end = '\n)=====")\n'
        # 将起始字符串、输出字符串、结束字符串连接起来，并添加到最终输出字符串中
        final_output += start + output_str + end
    # 在最终输出字符串末尾添加一个分号
    final_output += ";"
    # 返回最终的输出字符串
    return final_output
# 定义常量，包含生成形状函数映射的起始部分
SHAPE_SCHEMA_START = r"""
const OperatorMap<std::string>& GetShapeFunctionMappings() {
 static const OperatorMap<std::string> shape_mappings {
"""

# 定义常量，包含生成形状函数映射的结束部分
SHAPE_SCHEMA_END = r"""
  };

  return shape_mappings;
}
"""

# 定义函数，生成形状函数的映射并返回为字符串
def gen_shape_mappings() -> str:
    shape_mappings = []
    # 遍历形状计算图映射字典中的每个项
    for schema, scripted_func in shape_compute_graph_mapping.items():
        # 构造每个映射条目的字符串形式，加入到列表中
        shape_mappings.append('    {"' + schema + '", "' + scripted_func.name + '"},')
    # 拼接起始部分、映射列表和结束部分，返回生成的字符串
    return SHAPE_SCHEMA_START + "\n".join(shape_mappings) + SHAPE_SCHEMA_END


# 定义常量，包含生成有界形状映射的起始部分
BOUNDED_SCHEMA_START = r"""
const OperatorMap<std::pair<std::string, std::string>>& GetBoundedShapeMappings() {
 static const OperatorMap<std::pair<std::string, std::string>> shape_mappings {
"""

# 定义函数，生成有界形状映射并返回为字符串
def gen_bounded_mappings() -> str:
    bounded_mappings = []
    # 遍历有界形状计算图映射字典中的每个项
    for schema, (lower_func, upper_func) in bounded_compute_graph_mapping.items():
        # 构造每个映射条目的字符串形式，加入到列表中
        map_str = (
            '    {"'
            + schema
            + '", {"'
            + lower_func.name
            + '", "'
            + upper_func.name
            + '"}},'
        )
        bounded_mappings.append(map_str)
    # 拼接起始部分、映射列表和结束部分，返回生成的字符串
    return BOUNDED_SCHEMA_START + "\n".join(bounded_mappings) + SHAPE_SCHEMA_END


# 定义函数，将生成的序列化分解字符串、形状映射字符串和有界映射字符串写入文件
def write_decomposition_util_file(path: str) -> None:
    # 生成序列化分解字符串
    decomposition_str = gen_serialized_decompisitions()
    # 生成形状映射字符串
    shape_mappings = gen_shape_mappings()
    # 生成有界映射字符串
    bounded_mappings = gen_bounded_mappings()
    # 将所有组成部分放入文件组件列表中
    file_components = [
        SHAPE_HEADER,
        decomposition_str,
        DECOMP_CENTER,
        shape_mappings,
        bounded_mappings,
        DECOMP_END,
    ]
    # 打印写入文件的路径和文件名
    print("writing file to : ", path + "/" + SERIALIZED_SHAPE_UTIL_FILE_NAME)
    # 打开文件并写入最终的输出内容
    with open(os.path.join(path, SERIALIZED_SHAPE_UTIL_FILE_NAME), "wb") as out_file:
        final_output = "".join(file_components)
        out_file.write(final_output.encode("utf-8"))


# 定义主函数，获取 PyTorch 目录并将生成的文件写入相应位置
def main() -> None:
    pytorch_dir = Path(__file__).resolve().parents[2]
    upgrader_path = pytorch_dir / "torch" / "csrc" / "jit" / "runtime"
    write_decomposition_util_file(str(upgrader_path))


# 如果脚本作为主程序运行，则调用主函数
if __name__ == "__main__":
    main()
```