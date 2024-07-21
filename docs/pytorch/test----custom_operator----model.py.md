# `.\pytorch\test\custom_operator\model.py`

```py
# 导入 argparse 模块，用于命令行参数解析
import argparse
# 导入 os.path 模块，用于操作系统路径操作
import os.path
# 导入 sys 模块，用于访问与 Python 解释器交互的变量和函数
import sys

# 导入 PyTorch 库
import torch


# 定义函数，获取自定义操作库文件的路径
def get_custom_op_library_path():
    # 根据操作系统平台确定自定义操作库文件名
    if sys.platform.startswith("win32"):
        library_filename = "custom_ops.dll"
    elif sys.platform.startswith("darwin"):
        library_filename = "libcustom_ops.dylib"
    else:
        library_filename = "libcustom_ops.so"
    # 构建自定义操作库文件的绝对路径
    path = os.path.abspath(f"build/{library_filename}")
    # 断言路径存在，否则抛出异常
    assert os.path.exists(path), path
    return path


# 定义模型类，继承自 torch.jit.ScriptModule
class Model(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()
        # 创建一个形状为 5x5 的单位矩阵，并将其封装为 PyTorch 参数
        self.p = torch.nn.Parameter(torch.eye(5))

    @torch.jit.script_method
    # 定义前向传播方法，使用自定义操作
    def forward(self, input):
        # 调用自定义操作，并对结果进行加一操作后返回
        return torch.ops.custom.op_with_defaults(input)[0] + 1


# 定义主函数
def main():
    # 创建参数解析器对象
    parser = argparse.ArgumentParser(
        description="Serialize a script module with custom ops"
    )
    # 添加命令行参数，指定导出脚本模块的路径，参数为必需
    parser.add_argument("--export-script-module-to", required=True)
    # 解析命令行参数
    options = parser.parse_args()

    # 加载自定义操作库
    torch.ops.load_library(get_custom_op_library_path())

    # 创建模型对象
    model = Model()
    # 将模型保存到指定路径
    model.save(options.export_script_module_to)


# 如果当前脚本为主程序，则执行主函数
if __name__ == "__main__":
    main()
```