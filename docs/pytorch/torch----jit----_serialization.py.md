# `.\pytorch\torch\jit\_serialization.py`

```py
# 根据给定的 TorchScript 模块 `m` 和文件或文件类对象 `f`，保存模块的离线版本
def save(m, f, _extra_files=None):
    # 记录 TorchScript 使用情况日志
    log_torchscript_usage("save")
    # 如果未提供额外文件，则初始化为空字典
    if _extra_files is None:
        _extra_files = {}
    # 如果 `f` 是字符串或类似路径的对象
    if isinstance(f, (str, os.PathLike)):
        # 调用模块的 save 方法，将模块保存到文件 `f` 中，并包含额外的文件 `_extra_files`
        m.save(f, _extra_files=_extra_files)
    else:
        # 如果 `f` 是文件类对象，则调用模块的 save_to_buffer 方法，将模块保存到缓冲区中，并写入到 `f`
        ret = m.save_to_buffer(_extra_files=_extra_files)
        f.write(ret)
def load(f, map_location=None, _extra_files=None, _restore_shapes=False):
    r"""
    Load a :class:`ScriptModule` or :class:`ScriptFunction` previously saved with :func:`torch.jit.save <torch.jit.save>`.

    All previously saved modules, no matter their device, are first loaded onto CPU,
    and then are moved to the devices they were saved from. If this fails (e.g.
    because the run time system doesn't have certain devices), an exception is
    raised.

    Args:
        f: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name
        map_location (string or torch.device): A simplified version of
            ``map_location`` in `torch.jit.save` used to dynamically remap
            storages to an alternative set of devices.
        _extra_files (dictionary of filename to content): The extra
            filenames given in the map would be loaded and their content
            would be stored in the provided map.
        _restore_shapes (bool): Whether or not to retrace the module on load using stored inputs

    Returns:
        A :class:`ScriptModule` object.

    Example:
    .. testcode::

        import torch
        import io

        torch.jit.load('scriptmodule.pt')

        # Load ScriptModule from io.BytesIO object
        with open('scriptmodule.pt', 'rb') as f:
            buffer = io.BytesIO(f.read())

        # Load all tensors to the original device
        torch.jit.load(buffer)

        # Load all tensors onto CPU, using a device
        buffer.seek(0)
        torch.jit.load(buffer, map_location=torch.device('cpu'))

        # Load all tensors onto CPU, using a string
        buffer.seek(0)
        torch.jit.load(buffer, map_location='cpu')

        # Load with extra files.
        extra_files = {'foo.txt': ''}  # values will be replaced with data
        torch.jit.load('scriptmodule.pt', _extra_files=extra_files)
        print(extra_files['foo.txt'])

    .. testoutput::
        :hide:

        ...

    .. testcleanup::

        import os
        os.remove("scriptmodule.pt")
    """
    log_torchscript_usage("load")  # 记录 TorchScript 加载的使用情况

    # 检查参数 f 是否为字符串或路径类型，如果是字符串则检查文件是否存在，如果是目录则抛出异常
    if isinstance(f, (str, os.PathLike)):
        if not os.path.exists(f):  # 如果文件不存在则抛出异常
            raise ValueError(f"The provided filename {f} does not exist")  # 抛出文件不存在的异常
        if os.path.isdir(f):  # 如果 f 是目录则抛出异常
            raise ValueError(f"The provided filename {f} is a directory")  # 抛出文件是目录的异常

    # 校验并转换 map_location 参数为 torch.device 类型
    map_location = validate_map_location(map_location)

    # 如果 _extra_files 参数为 None，则初始化为空字典
    if _extra_files is None:
        _extra_files = {}

    # 创建一个 torch._C.CompilationUnit 对象
    cu = torch._C.CompilationUnit()

    # 根据参数 f 的类型，选择不同的导入方式，导入 TorchScript 模块
    if isinstance(f, (str, os.PathLike)):
        cpp_module = torch._C.import_ir_module(cu, os.fspath(f), map_location, _extra_files, _restore_shapes)  # 从文件导入 TorchScript 模块
    else:
        cpp_module = torch._C.import_ir_module_from_buffer(
            cu, f.read(), map_location, _extra_files, _restore_shapes
        )  # 从缓冲区导入 TorchScript 模块
    # TODO: Pretty sure this approach loses ConstSequential status and such
    # 返回使用 wrap_cpp_module 包装后的 cpp_module
    return wrap_cpp_module(cpp_module)
# 验证和规范化映射位置参数
def validate_map_location(map_location=None):
    # 如果 map_location 是字符串，则转换为 torch.device 对象
    if isinstance(map_location, str):
        map_location = torch.device(map_location)
    # 如果 map_location 不为 None 且不是 torch.device 对象，则引发异常
    elif not (map_location is None or isinstance(map_location, torch.device)):
        raise ValueError(
            "map_location should be either None, string or torch.device, "
            "but got type: " + str(type(map_location))
        )

    # 如果 map_location 是以 "cuda" 开头的字符串，则验证 CUDA 设备
    if str(map_location).startswith("cuda"):
        validate_cuda_device(map_location)

    # 返回验证后的 map_location
    return map_location


# 从 flatbuffer 格式的文件加载 JIT 模块
def jit_module_from_flatbuffer(f):
    # 如果 f 是字符串或者类似路径对象，则转换为文件路径字符串
    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)
        # 调用 C++ API 加载文件中的 JIT 模块，并使用 wrap_cpp_module 封装返回结果
        return wrap_cpp_module(torch._C._load_jit_module_from_file(f))
    else:
        # 使用 f.read() 读取字节流，并加载其中的 JIT 模块，再使用 wrap_cpp_module 封装返回结果
        return wrap_cpp_module(torch._C._load_jit_module_from_bytes(f.read()))


# 将 JIT 模块保存为 flatbuffer 格式的文件
def save_jit_module_to_flatbuffer(m, f, _extra_files=None):
    """
    将此模块的离线版本保存以供在单独进程中使用。

    保存的模块序列化此模块的所有方法、子模块、参数和属性。
    它可以使用 ``torch::jit::load_jit_module_from_file(filename)`` 在 C++ API 中加载，
    或者使用 :func:`torch.jit.jit_module_from_flatbuffer<torch.jit.jit_module_from_flatbuffer>` 在 Python API 中加载。

    要能够保存模块，它不能调用任何本地 Python 函数。
    这意味着所有子模块必须是 :class:`ScriptModule` 的子类。

    .. DANGER::
        所有模块，无论其设备如何，始终在加载时加载到 CPU 上。
        这与 :func:`torch.load` 的语义不同，可能会在将来更改。

    Args:
        m: 要保存的 :class:`ScriptModule`。
        f: 文件路径的字符串

    Example:
    .. testcode::

        import torch
        import io

        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + 10

        m = torch.jit.script(MyModule())

        # 保存到文件
        torch.jit.save_jit_module_to_flatbuffer(m, 'scriptmodule.ff')
    """
    # 如果 _extra_files 为 None，则初始化为一个空字典
    extra_files = _extra_files
    if extra_files is None:
        extra_files = {}

    # 如果 f 是字符串或者类似路径对象，则转换为文件路径字符串并保存 JIT 模块
    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)
        torch._C._save_jit_module(m._c, f, extra_files)
    else:
        # 将 JIT 模块序列化为字节流并写入 f
        s = torch._C._save_jit_module_to_bytes(m._c, extra_files)
        f.write(s)


# 获取 flatbuffer 格式模型文件的一些信息
def get_flatbuffer_module_info(path_or_file):
    r"""获取 flatbuffer 格式模型文件的一些信息。

    Args:
        path_or_file: str、Path 或类似文件对象（BytesIO 也可以）。
            如果是 str 或 Path，则读取该路径引用的文件作为字节。

    """
    # 如果传入的参数是字符串或者类似于路径的对象
    if isinstance(path_or_file, (str, os.PathLike)):
        # 打开文件，并以二进制模式读取所有内容
        with open(path_or_file, "rb") as f:
            all_bytes = f.read()
    # 否则，假设传入的是一个文件对象，直接读取其内容
    else:
        all_bytes = path_or_file.read()
    # 调用 torch._C._get_module_info_from_flatbuffer 函数，传入所有读取的字节数据，并返回其结果
    return torch._C._get_module_info_from_flatbuffer(all_bytes)
```