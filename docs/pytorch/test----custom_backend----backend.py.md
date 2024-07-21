# `.\pytorch\test\custom_backend\backend.py`

```
# 导入必要的库和模块
import argparse  # 用于解析命令行参数的模块
import os.path  # 用于处理文件路径的模块
import sys  # 提供了与 Python 解释器相关的系统信息和功能

import torch  # 引入 PyTorch 库


def get_custom_backend_library_path():
    """
    获取包含自定义后端的库的路径。

    Return:
        包含自定义后端对象的路径，根据平台定制。
    """
    # 根据操作系统平台选择自定义后端库的文件名
    if sys.platform.startswith("win32"):
        library_filename = "custom_backend.dll"
    elif sys.platform.startswith("darwin"):
        library_filename = "libcustom_backend.dylib"
    else:
        library_filename = "libcustom_backend.so"
    # 构建自定义后端库的绝对路径
    path = os.path.abspath(f"build/{library_filename}")
    # 断言路径存在，否则抛出异常
    assert os.path.exists(path), path
    return path


def to_custom_backend(module):
    """
    这是一个辅助函数，用于将模块包装到自定义后端，并使用空的编译规范仅编译 forward 方法。

    Args:
        module: 输入的 ScriptModule。

    Returns:
        降低后的模块，使其可以在 TestBackend 上运行。
    """
    # 使用 torch._C._jit_to_backend 将模块编译到自定义后端
    lowered_module = torch._C._jit_to_backend(
        "custom_backend", module, {"forward": {"": ""}}
    )
    return lowered_module


class Model(torch.nn.Module):
    """
    用于测试 to_backend API 是否支持在 C++ 中保存、加载和执行的简单模型。
    """

    def forward(self, a, b):
        return (a + b, a - b)


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Lower a Module to a custom backend")
    # 添加必需的命令行参数
    parser.add_argument("--export-module-to", required=True)
    # 解析命令行参数
    options = parser.parse_args()

    # 加载包含自定义后端的库
    library_path = get_custom_backend_library_path()
    torch.ops.load_library(library_path)
    # 断言加载的库路径在 torch.ops.loaded_libraries 中
    assert library_path in torch.ops.loaded_libraries

    # 将 Model 的实例降低到自定义后端，并将其导出到指定位置
    lowered_module = to_custom_backend(torch.jit.script(Model()))
    torch.jit.save(lowered_module, options.export_module_to)


if __name__ == "__main__":
    main()
```