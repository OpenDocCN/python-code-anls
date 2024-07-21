# `.\pytorch\torch\jit\mobile\__init__.py`

```
# 指定允许未类型化的函数定义，用于类型检查工具 mypy
mypy: allow-untyped-defs

# 导入操作系统模块
import os

# 导入 PyTorch 模块
import torch

# 导入 Torch JIT 序列化模块中的位置验证函数
from torch.jit._serialization import validate_map_location


def _load_for_lite_interpreter(f, map_location=None):
    r"""
    使用 torch.jit._save_for_lite_interpreter 保存的 LiteScriptModule 加载一个文件。

    Args:
        f: 文件类对象（必须实现 read、readline、tell 和 seek 方法），或者包含文件名的字符串
        map_location: 一个字符串或 torch.device，用于动态重映射存储到另一组设备。

    Returns:
        一个 LiteScriptModule 对象。

    Example:

    .. testcode::

        import torch
        import io

        # 从保存的文件路径加载 LiteScriptModule
        torch.jit._load_for_lite_interpreter('lite_script_module.pt')

        # 从 io.BytesIO 对象加载 LiteScriptModule
        with open('lite_script_module.pt', 'rb') as f:
            buffer = io.BytesIO(f.read())

        # 将所有张量加载到原始设备
        torch.jit.mobile._load_for_lite_interpreter(buffer)
    """
    # 如果 f 是字符串或者路径对象
    if isinstance(f, (str, os.PathLike)):
        # 检查文件是否存在
        if not os.path.exists(f):
            raise ValueError(f"提供的文件名 {f} 不存在")
        # 检查是否为目录
        if os.path.isdir(f):
            raise ValueError(f"提供的文件名 {f} 是一个目录")

    # 验证 map_location 参数，并确保其为 torch.device 对象
    map_location = validate_map_location(map_location)

    # 如果 f 是字符串或者路径对象
    if isinstance(f, (str, os.PathLike)):
        # 使用 torch._C._load_for_lite_interpreter 从文件加载 LiteScriptModule
        cpp_module = torch._C._load_for_lite_interpreter(os.fspath(f), map_location)
    else:
        # 使用 torch._C._load_for_lite_interpreter_from_buffer 从缓冲区加载 LiteScriptModule
        cpp_module = torch._C._load_for_lite_interpreter_from_buffer(
            f.read(), map_location
        )

    # 返回一个包含 cpp_module 的 LiteScriptModule 对象
    return LiteScriptModule(cpp_module)


class LiteScriptModule:
    def __init__(self, cpp_module):
        # 将 C++ 模块作为参数初始化 LiteScriptModule 对象
        self._c = cpp_module
        super().__init__()

    def __call__(self, *input):
        # 将输入参数传递给底层 C++ 模块的 forward 方法
        return self._c.forward(input)

    def find_method(self, method_name):
        # 查找指定方法名在底层 C++ 模块中的方法
        return self._c.find_method(method_name)

    def forward(self, *input):
        # 调用底层 C++ 模块的 forward 方法，传递输入参数
        return self._c.forward(input)

    def run_method(self, method_name, *input):
        # 在底层 C++ 模块上运行指定方法名的方法，传递输入参数
        return self._c.run_method(method_name, input)


def _export_operator_list(module: LiteScriptModule):
    r"""返回此移动模块中任何方法使用的根操作符名称（带有重载名称）的集合。"""
    return torch._C._export_operator_list(module._c)


def _get_model_bytecode_version(f_input) -> int:
    r"""接受文件类对象并返回一个整数。

    Args:
        f_input: 文件类对象（必须实现 read、readline、tell 和 seek 方法），或者包含文件名的字符串

    Returns:
        version: 一个整数。如果整数为 -1，则版本无效。日志中将显示警告。

    Example:

    .. testcode::

        from torch.jit.mobile import _get_model_bytecode_version

        # 从保存的文件路径获取字节码版本
        version = _get_model_bytecode_version("path/to/model.ptl")
    """
    # 如果 f_input 是字符串或者路径类型的对象
    if isinstance(f_input, (str, os.PathLike)):
        # 如果 f_input 指定的文件不存在，则抛出 ValueError 异常
        if not os.path.exists(f_input):
            raise ValueError(f"The provided filename {f_input} does not exist")
        # 如果 f_input 指定的路径是一个目录，则抛出 ValueError 异常
        if os.path.isdir(f_input):
            raise ValueError(f"The provided filename {f_input} is a directory")
    
    # 如果 f_input 是字符串或者路径类型的对象
    if isinstance(f_input, (str, os.PathLike)):
        # 返回指定模型文件的字节码版本
        return torch._C._get_model_bytecode_version(os.fspath(f_input))
    else:
        # 从缓冲区读取模型数据，并返回其字节码版本
        return torch._C._get_model_bytecode_version_from_buffer(f_input.read())
# 从给定的文件或文件名读取移动模型中包含的类型集合并返回一个集合对象

def _get_mobile_model_contained_types(f_input) -> int:
    r"""Take a file-like object and return a set of string, like ("int", "Optional").

    Args:
        f_input: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name

    Returns:
        type_list: A set of string, like ("int", "Optional"). These are types used in bytecode.

    Example:

    .. testcode::

        from torch.jit.mobile import _get_mobile_model_contained_types

        # Get type list from a saved file path
        type_list = _get_mobile_model_contained_types("path/to/model.ptl")

    """
    # 如果输入是字符串或者路径对象
    if isinstance(f_input, (str, os.PathLike)):
        # 检查文件是否存在
        if not os.path.exists(f_input):
            raise ValueError(f"The provided filename {f_input} does not exist")
        # 检查输入是否是一个目录
        if os.path.isdir(f_input):
            raise ValueError(f"The provided filename {f_input} is a directory")

    # 如果输入是字符串或者路径对象，调用 torch._C._get_mobile_model_contained_types 函数
    if isinstance(f_input, (str, os.PathLike)):
        return torch._C._get_mobile_model_contained_types(os.fspath(f_input))
    else:
        # 否则调用 torch._C._get_mobile_model_contained_types_from_buffer 函数
        return torch._C._get_mobile_model_contained_types_from_buffer(f_input.read())


# 将输入的文件或文件名的移动模型回退到指定版本，并将结果保存到新的文件位置
def _backport_for_mobile(f_input, f_output, to_version):
    r"""Take a input string containing a file name (file-like object) and a new destination to return a boolean.

    Args:
        f_input: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name
        f_output: path to new model destination
        to_version: the expected output model bytecode version

    Returns:
        success: A boolean. If backport success, return true, otherwise false
    """
    # 如果输入是字符串或者路径对象
    if isinstance(f_input, (str, os.PathLike)):
        # 检查文件是否存在
        if not os.path.exists(f_input):
            raise ValueError(f"The provided filename {f_input} does not exist")
        # 检查输入是否是一个目录
        if os.path.isdir(f_input):
            raise ValueError(f"The provided filename {f_input} is a directory")

    # 如果输入和输出都是字符串或者路径对象，调用 torch._C._backport_for_mobile 函数
    if (isinstance(f_input, (str, os.PathLike))) and (
        isinstance(f_output, (str, os.PathLike))
    ):
        return torch._C._backport_for_mobile(
            os.fspath(f_input), os.fspath(f_output), to_version
        )
    else:
        # 否则调用 torch._C._backport_for_mobile_from_buffer 函数
        return torch._C._backport_for_mobile_from_buffer(
            f_input.read(), str(f_output), to_version
        )


# 将输入的文件或文件名的移动模型回退到指定版本，但是结果保存在缓冲区中而不是文件中
def _backport_for_mobile_to_buffer(f_input, to_version):
    r"""Take a string containing a file name (file-like object).

    Args:
        f_input: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name

    """
    # 如果输入是字符串或者路径对象
    if isinstance(f_input, (str, os.PathLike)):
        # 检查文件是否存在
        if not os.path.exists(f_input):
            raise ValueError(f"The provided filename {f_input} does not exist")
        # 检查输入是否是一个目录
        if os.path.isdir(f_input):
            raise ValueError(f"The provided filename {f_input} is a directory")
    # 检查 f_input 是否是字符串类型或者 os.PathLike 对象
    if isinstance(f_input, (str, os.PathLike)):
        # 如果是字符串或者 PathLike 对象，将其转换为文件路径字符串，然后执行移动设备兼容性的后向转换操作
        return torch._C._backport_for_mobile_to_buffer(os.fspath(f_input), to_version)
    else:
        # 如果 f_input 不是字符串或者 PathLike 对象，假设其是一个类文件对象，读取其内容并执行移动设备兼容性的前向转换操作
        return torch._C._backport_for_mobile_from_buffer_to_buffer(
            f_input.read(), to_version
        )
# 定义一个函数，用于获取模型的根操作符及其对应的兼容性信息
def _get_model_ops_and_info(f_input):
    r"""Retrieve the root (top level) operators of a model and their corresponding compatibility info.

    These root operators can call other operators within them (traced ops), and
    a root op can call many different traced ops depending on internal code paths in the root op.
    These traced ops are not returned by this function. Those operators are abstracted into the
    runtime as an implementation detail (and the traced ops themselves can also call other operators)
    making retrieving them difficult and their value from this api negligible since they will differ
    between which runtime version the model is run on. Because of this, there is a false positive this
    api can't prevent in a compatibility usecase. All the root ops of a model are present in a
    target runtime, but not all the traced ops are which prevents a model from being able to run.
    
    Args:
        f_input: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name
    
    Returns:
        Operators and info: A Dictionary mapping strings (the qualified names of the root operators)
        of the model to their OperatorInfo structs.
    
    Example:
    
    .. testcode::
    
        from torch.jit.mobile import _get_model_ops_and_info
    
        # Get bytecode version from a saved file path
        ops_and_info = _get_model_ops_and_info("path/to/model.ptl")
    """

    # 如果 f_input 是字符串或者类似路径的对象
    if isinstance(f_input, (str, os.PathLike)):
        # 检查文件是否存在
        if not os.path.exists(f_input):
            raise ValueError(f"The provided filename {f_input} does not exist")
        # 如果路径是一个目录，则抛出异常
        if os.path.isdir(f_input):
            raise ValueError(f"The provided filename {f_input} is a directory")
    
    # 如果 f_input 是字符串或者类似路径的对象
    if isinstance(f_input, (str, os.PathLike)):
        # 使用文件路径调用 torch._C._get_model_ops_and_info 函数
        return torch._C._get_model_ops_and_info(os.fspath(f_input))
    else:
        # 否则，假设 f_input 是类文件对象，直接调用 torch._C._get_model_ops_and_info 函数
        return torch._C._get_model_ops_and_info(f_input.read())
```