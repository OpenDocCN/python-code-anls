# `.\pytorch\test\jit_hooks\model.py`

```py
# 导入 argparse 模块，用于解析命令行参数
import argparse
# 导入 os 模块，提供与操作系统交互的功能
import os
# 导入 sys 模块，提供对 Python 解释器的访问
import sys

# 导入 torch 模块，用于 PyTorch 相关操作
import torch

# 从 test_jit_hooks.cpp 中导入测试模块
# 获取 pytorch_test_dir 的路径
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# 将路径添加到 sys.path 中，以便找到相应的模块
sys.path.append(pytorch_test_dir)
# 从 jit.test_hooks_modules 模块中导入多个函数
from jit.test_hooks_modules import (
    create_forward_tuple_input,
    create_module_forward_multiple_inputs,
    create_module_forward_single_input,
    create_module_hook_return_nothing,
    create_module_multiple_hooks_multiple_inputs,
    create_module_multiple_hooks_single_input,
    create_module_no_forward_input,
    create_module_same_hook_repeated,
    create_submodule_forward_multiple_inputs,
    create_submodule_forward_single_input,
    create_submodule_hook_return_nothing,
    create_submodule_multiple_hooks_multiple_inputs,
    create_submodule_multiple_hooks_single_input,
    create_submodule_same_hook_repeated,
    create_submodule_to_call_directly_with_hooks,
)

# 定义主函数
def main():
    # 创建 argparse.ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser(
        description="Serialize a script modules with hooks attached"
    )
    # 添加命令行参数 --export-script-module-to，要求参数必须提供
    parser.add_argument("--export-script-module-to", required=True)
    # 解析命令行参数
    options = parser.parse_args()
    # 定义全局变量 save_name，用于保存脚本模块的输出文件名前缀
    global save_name
    save_name = options.export_script_module_to + "_"

    # 定义测试用例列表
    tests = [
        # 测试子模块的单输入前向传播
        ("test_submodule_forward_single_input", create_submodule_forward_single_input()),
        # 测试子模块的多输入前向传播
        ("test_submodule_forward_multiple_inputs", create_submodule_forward_multiple_inputs()),
        # 测试子模块的单输入多钩子
        ("test_submodule_multiple_hooks_single_input", create_submodule_multiple_hooks_single_input()),
        # 测试子模块的多输入多钩子
        ("test_submodule_multiple_hooks_multiple_inputs", create_submodule_multiple_hooks_multiple_inputs()),
        # 测试子模块的返回空钩子
        ("test_submodule_hook_return_nothing", create_submodule_hook_return_nothing()),
        # 测试子模块的重复钩子
        ("test_submodule_same_hook_repeated", create_submodule_same_hook_repeated()),
        # 测试模块的单输入前向传播
        ("test_module_forward_single_input", create_module_forward_single_input()),
        # 测试模块的多输入前向传播
        ("test_module_forward_multiple_inputs", create_module_forward_multiple_inputs()),
        # 测试模块的单输入多钩子
        ("test_module_multiple_hooks_single_input", create_module_multiple_hooks_single_input()),
        # 测试模块的多输入多钩子
        ("test_module_multiple_hooks_multiple_inputs", create_module_multiple_hooks_multiple_inputs()),
        # 测试模块的返回空钩子
        ("test_module_hook_return_nothing", create_module_hook_return_nothing()),
        # 测试模块的重复钩子
        ("test_module_same_hook_repeated", create_module_same_hook_repeated()),
        # 测试模块的无前向传播输入
        ("test_module_no_forward_input", create_module_no_forward_input()),
        # 测试前向元组输入
        ("test_forward_tuple_input", create_forward_tuple_input()),
        # 测试直接调用带钩子的子模块
        ("test_submodule_to_call_directly_with_hooks", create_submodule_to_call_directly_with_hooks()),
    ]
    # 遍历 tests 列表中的每个元组，元组包含模型名称和模型本身
    for name, model in tests:
        # 使用 torch.jit.script 方法对模型进行脚本化
        m_scripted = torch.jit.script(model)
        # 拼接保存文件的名称，将模型名称和 ".pt" 后缀组合成文件名
        filename = save_name + name + ".pt"
        # 使用 torch.jit.save 方法将脚本化后的模型保存到文件中
        torch.jit.save(m_scripted, filename)

    # 打印保存操作完成的消息
    print("OK: completed saving modules with hooks!")
# 如果当前脚本被直接执行（而不是被导入到其他模块中执行），则执行以下代码块
if __name__ == "__main__":
    # 调用名为 main() 的函数，通常用来执行程序的主要逻辑
    main()
```