# `.\numpy\numpy\_build_utils\__init__.py`

```
# 不要使用已弃用的 NumPy C API。将此定义为固定版本而不是 NPY_API_VERSION，以避免在 NumPy 引入新的弃用功能时破坏已发布的 SciPy 版本的编译。在 setup.py 中使用::
#
#   config.add_extension('_name', sources=['source_fname'], **numpy_nodepr_api)
#
# 定义一个字典 numpy_nodepr_api，包含以下内容：
#   - define_macros 键为 [("NPY_NO_DEPRECATED_API", "NPY_1_9_API_VERSION")]，用于设置 NumPy 不使用已弃用的 API 版本。
#
numpy_nodepr_api = dict(
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_9_API_VERSION")]
)


def import_file(folder, module_name):
    """Import a file directly, avoiding importing scipy"""
    # 导入必要的库：importlib 用于动态导入模块，pathlib 用于处理文件路径
    import importlib
    import pathlib

    # 构建文件路径：folder 目录下的 module_name.py 文件
    fname = pathlib.Path(folder) / f'{module_name}.py'
    
    # 使用 importlib 函数 spec_from_file_location 创建导入模块的规范
    spec = importlib.util.spec_from_file_location(module_name, str(fname))
    
    # 使用 spec.loader.exec_module 执行模块的加载
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # 返回导入的模块对象
    return module
```