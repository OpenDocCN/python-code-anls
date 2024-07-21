# `.\pytorch\test\package\generate_bc_packages.py`

```
# 从 pathlib 库中导入 Path 类，用于处理文件路径
from pathlib import Path

# 导入 torch 库及其子模块
import torch
# 导入 torch.fx 模块中的 symbolic_trace 函数
from torch.fx import symbolic_trace
# 导入 torch.package 模块中的 PackageExporter 类
from torch.package import PackageExporter
# 导入 torch.testing._internal.common_utils 模块中的 IS_FBCODE 和 IS_SANDCASTLE 变量
from torch.testing._internal.common_utils import IS_FBCODE, IS_SANDCASTLE

# 设置打包目录为当前文件的父目录下的 package_bc 子目录
packaging_directory = f"{Path(__file__).parent}/package_bc"
# 设置 torch.package.package_exporter 模块中的 _gate_torchscript_serialization 属性为 False
torch.package.package_exporter._gate_torchscript_serialization = False


def generate_bc_packages():
    """用于创建用于测试向后兼容性的包的函数"""
    # 如果不是在 FBCODE 环境下或者在 SANDCASTLE 环境下
    if not IS_FBCODE or IS_SANDCASTLE:
        # 从 package_a.test_nn_module 模块中导入 TestNnModule 类
        from package_a.test_nn_module import TestNnModule

        # 创建 TestNnModule 类的实例对象
        test_nn_module = TestNnModule()
        # 对 TestNnModule 类进行 TorchScript 脚本化
        test_torchscript_module = torch.jit.script(TestNnModule())
        # 对 TestNnModule 类进行符号化追踪，生成 torch.fx.GraphModule 类的对象
        test_fx_module: torch.fx.GraphModule = symbolic_trace(TestNnModule())

        # 使用 PackageExporter 类创建包，保存 TestNnModule 实例对象
        with PackageExporter(f"{packaging_directory}/test_nn_module.pt") as pe1:
            # 将包中的所有内容进行内部化
            pe1.intern("**")
            # 将 nn_module 对象保存为 pickle 格式的文件 nn_module.pkl
            pe1.save_pickle("nn_module", "nn_module.pkl", test_nn_module)

        # 使用 PackageExporter 类创建包，保存 TorchScript 化后的 TestNnModule 对象
        with PackageExporter(
            f"{packaging_directory}/test_torchscript_module.pt"
        ) as pe2:
            # 将包中的所有内容进行内部化
            pe2.intern("**")
            # 将 torchscript_module 对象保存为 pickle 格式的文件 torchscript_module.pkl
            pe2.save_pickle(
                "torchscript_module", "torchscript_module.pkl", test_torchscript_module
            )

        # 使用 PackageExporter 类创建包，保存符号化追踪后的 TestNnModule 对象
        with PackageExporter(f"{packaging_directory}/test_fx_module.pt") as pe3:
            # 将包中的所有内容进行内部化
            pe3.intern("**")
            # 将 fx_module 对象保存为 pickle 格式的文件 fx_module.pkl
            pe3.save_pickle("fx_module", "fx_module.pkl", test_fx_module)


# 如果当前脚本被作为主程序执行，则调用 generate_bc_packages 函数
if __name__ == "__main__":
    generate_bc_packages()
```