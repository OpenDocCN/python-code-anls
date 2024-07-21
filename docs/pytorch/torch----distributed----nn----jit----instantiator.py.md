# `.\pytorch\torch\distributed\nn\jit\instantiator.py`

```
#!/usr/bin/python3
# mypy: allow-untyped-defs
# 导入必要的库和模块
import importlib  # 导入模块动态加载功能
import logging  # 导入日志记录模块
import os  # 导入操作系统相关功能
import sys  # 导入系统相关功能
import tempfile  # 导入临时文件夹管理功能
from typing import Optional  # 引入类型提示功能

import torch  # 导入PyTorch深度学习库
from torch.distributed.nn.jit.templates.remote_module_template import (
    get_remote_module_template,  # 从远程模板中获取模板代码函数
)

# 设置日志记录器
logger = logging.getLogger(__name__)

# 定义常量
_FILE_PREFIX = "_remote_module_"
_TEMP_DIR = tempfile.TemporaryDirectory()  # 创建临时目录对象
INSTANTIATED_TEMPLATE_DIR_PATH = _TEMP_DIR.name  # 获取临时目录路径作为模板生成目录路径
logger.info("Created a temporary directory at %s", INSTANTIATED_TEMPLATE_DIR_PATH)  # 记录临时目录创建信息到日志
sys.path.append(INSTANTIATED_TEMPLATE_DIR_PATH)  # 将临时目录路径添加到系统路径中，以便动态加载模块使用


def get_arg_return_types_from_interface(module_interface):
    # 断言输入的模块接口是经过 @torch.jit.interface 装饰的 TorchScript 类接口
    assert getattr(
        module_interface, "__torch_script_interface__", False
    ), "Expect a TorchScript class interface decorated by @torch.jit.interface."
    qualified_name = torch._jit_internal._qualified_name(module_interface)
    cu = torch.jit._state._python_cu
    module_interface_c = cu.get_interface(qualified_name)  # 获取模块接口在编译单元中的接口对象
    assert (
        "forward" in module_interface_c.getMethodNames()
    ), f"Expect forward in interface methods, while it has {module_interface_c.getMethodNames()}"
    method_schema = module_interface_c.getMethod("forward")  # 获取 forward 方法的模式信息

    arg_str_list = []
    arg_type_str_list = []
    assert method_schema is not None
    for argument in method_schema.arguments:
        arg_str_list.append(argument.name)

        if argument.has_default_value():
            default_value_str = f" = {argument.default_value}"
        else:
            default_value_str = ""
        arg_type_str = f"{argument.name}: {argument.type}{default_value_str}"
        arg_type_str_list.append(arg_type_str)

    arg_str_list = arg_str_list[1:]  # 移除 "self"
    args_str = ", ".join(arg_str_list)

    arg_type_str_list = arg_type_str_list[1:]  # 移除 "self"
    arg_types_str = ", ".join(arg_type_str_list)

    assert len(method_schema.returns) == 1
    argument = method_schema.returns[0]
    return_type_str = str(argument.type)

    return args_str, arg_types_str, return_type_str


def _write(out_path, text):
    old_text: Optional[str]
    try:
        with open(out_path) as f:
            old_text = f.read()  # 尝试读取已存在的文件内容
    except OSError:
        old_text = None
    if old_text != text:
        with open(out_path, "w") as f:
            logger.info("Writing %s", out_path)  # 记录写操作的文件路径到日志
            f.write(text)  # 写入新的文本内容到文件
    else:
        logger.info("Skipped writing %s", out_path)  # 如果内容相同则记录跳过写入操作到日志


def _do_instantiate_remote_module_template(
    generated_module_name, str_dict, enable_moving_cpu_tensors_to_cuda
):
    generated_code_text = get_remote_module_template(
        enable_moving_cpu_tensors_to_cuda
    ).format(**str_dict)  # 生成远程模板代码文本
    out_path = os.path.join(
        INSTANTIATED_TEMPLATE_DIR_PATH, f"{generated_module_name}.py"
    )  # 构建生成文件的完整路径
    _write(out_path, generated_code_text)  # 调用写入函数，将生成的代码文本写入文件

    # 以下注释来自 importlib 文档：
    # > If you are dynamically importing a module that was created since
    # the interpreter began execution (e.g., created a Python source file),
    # 调用 invalidate_caches() 方法来使导入系统注意到新生成的模块
    importlib.invalidate_caches()
    # 动态导入指定名称的模块
    generated_module = importlib.import_module(f"{generated_module_name}")
    # 返回导入的生成模块对象
    return generated_module
def instantiate_scriptable_remote_module_template(
    module_interface_cls, enable_moving_cpu_tensors_to_cuda=True
):
    # 检查 module_interface_cls 是否被 @torch.jit.interface 装饰，否则抛出异常
    if not getattr(module_interface_cls, "__torch_script_interface__", False):
        raise ValueError(
            f"module_interface_cls {module_interface_cls} must be a type object decorated by "
            "@torch.jit.interface"
        )

    # 生成模板实例的名称
    module_interface_cls_name = torch._jit_internal._qualified_name(
        module_interface_cls
    ).replace(".", "_")
    generated_module_name = f"{_FILE_PREFIX}{module_interface_cls_name}"

    # 生成类型注解的字符串
    assign_module_interface_cls_str = (
        f"from {module_interface_cls.__module__} import "
        f"{module_interface_cls.__name__} as module_interface_cls"
    )
    args_str, arg_types_str, return_type_str = get_arg_return_types_from_interface(
        module_interface_cls
    )
    kwargs_str = ""
    arrow_and_return_type_str = f" -> {return_type_str}"
    arrow_and_future_return_type_str = f" -> Future[{return_type_str}]"

    # 构建字符串字典
    str_dict = dict(
        assign_module_interface_cls=assign_module_interface_cls_str,
        arg_types=arg_types_str,
        arrow_and_return_type=arrow_and_return_type_str,
        arrow_and_future_return_type=arrow_and_future_return_type_str,
        args=args_str,
        kwargs=kwargs_str,
        jit_script_decorator="@torch.jit.script",
    )
    # 调用 _do_instantiate_remote_module_template 函数进行模板实例化
    return _do_instantiate_remote_module_template(
        generated_module_name, str_dict, enable_moving_cpu_tensors_to_cuda
    )


def instantiate_non_scriptable_remote_module_template():
    # 为非可脚本化模板生成模板实例名称
    generated_module_name = f"{_FILE_PREFIX}non_scriptable"
    # 构建字符串字典，用于非可脚本化模板
    str_dict = dict(
        assign_module_interface_cls="module_interface_cls = None",
        args="*args",
        kwargs="**kwargs",
        arg_types="*args, **kwargs",
        arrow_and_return_type="",
        arrow_and_future_return_type="",
        jit_script_decorator="",
    )
    # 对于非可脚本化模板，总是启用将 CPU 张量移动到 CUDA 设备的功能
    return _do_instantiate_remote_module_template(generated_module_name, str_dict, True)
```