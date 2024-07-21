# `.\pytorch\torchgen\decompositions\gen_jit_decompositions.py`

```
#!/usr/bin/env python3
import os
from pathlib import Path

# 导入用于自动化生成的 Torch 模块代码的相关工具
from torch.jit._decompositions import decomposition_table

# 生成的文件头部注释，指出此文件为自动生成，不要手动修改
DECOMP_HEADER = r"""
/**
 * @generated
 * This is an auto-generated file. Please do not modify it by hand.
 * To re-generate, please run:
 * cd ~/pytorch && python torchgen/decompositions/gen_jit_decompositions.py
 */
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/decomposition_registry_util.h>

namespace torch {
namespace jit {

// 保存序列化后的分解函数字符串
const std::string decomp_funcs =
R"("""


# 生成文件主体部分的字符串注释
DECOMP_CENTER = r"""
)";

// 返回序列化后的分解函数字符串
const std::string& GetSerializedDecompositions() {
  return decomp_funcs;
}

// 返回分解函数映射表
const OperatorMap<std::string>& GetDecompositionMapping() {
  // clang-format off
 static const OperatorMap<std::string> decomposition_mapping {
"""

# 生成文件尾部的字符串注释
DECOMP_END = r"""
  };
  // clang-format on

  return decomposition_mapping;
}

} // namespace jit
} // namespace torch
"""

# 定义生成的工具文件名
DECOMPOSITION_UTIL_FILE_NAME = "decomposition_registry_util.cpp"

# 生成序列化的分解函数字符串
def gen_serialized_decompisitions() -> str:
    return "\n".join(
        [scripted_func.code for scripted_func in decomposition_table.values()]  # type: ignore[misc]
    )

# 生成分解函数的映射表字符串
def gen_decomposition_mappings() -> str:
    decomposition_mappings = []
    for schema, scripted_func in decomposition_table.items():
        decomposition_mappings.append(
            '    {"' + schema + '", "' + scripted_func.name + '"},'  # type: ignore[operator]
        )
    return "\n".join(decomposition_mappings)

# 写入生成的分解函数工具文件
def write_decomposition_util_file(path: str) -> None:
    decomposition_str = gen_serialized_decompisitions()
    decomposition_mappings = gen_decomposition_mappings()
    file_components = [
        DECOMP_HEADER,
        decomposition_str,
        DECOMP_CENTER,
        decomposition_mappings,
        DECOMP_END,
    ]
    print("writing file to : ", path + "/" + DECOMPOSITION_UTIL_FILE_NAME)
    with open(os.path.join(path, DECOMPOSITION_UTIL_FILE_NAME), "wb") as out_file:
        final_output = "".join(file_components)
        out_file.write(final_output.encode("utf-8"))

# 主函数，获取 PyTorch 目录并生成分解函数工具文件
def main() -> None:
    pytorch_dir = Path(__file__).resolve().parents[3]
    upgrader_path = pytorch_dir / "torch" / "csrc" / "jit" / "runtime"
    write_decomposition_util_file(str(upgrader_path))

# 如果作为主程序运行，则执行主函数
if __name__ == "__main__":
    main()
```