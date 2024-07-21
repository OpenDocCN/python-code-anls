# `.\pytorch\test\quantization\ao_migration\common.py`

```
import importlib  # 导入 importlib 模块，用于动态导入模块
from typing import List, Optional  # 导入 List 和 Optional 类型提示

from torch.testing._internal.common_utils import TestCase  # 导入 TestCase 类

class AOMigrationTestCase(TestCase):
    def _test_function_import(
        self,
        package_name: str,
        function_list: List[str],
        base: Optional[str] = None,
        new_package_name: Optional[str] = None,
    ):
        r"""Tests individual function list import by comparing the functions
        and their hashes."""
        if base is None:
            base = "quantization"
        old_base = "torch." + base  # 构建旧基础模块路径
        new_base = "torch.ao." + base  # 构建新基础模块路径
        if new_package_name is None:
            new_package_name = package_name  # 如果新包名称未指定，则使用原始包名称
        old_location = importlib.import_module(f"{old_base}.{package_name}")  # 动态导入旧模块
        new_location = importlib.import_module(f"{new_base}.{new_package_name}")  # 动态导入新模块
        for fn_name in function_list:
            old_function = getattr(old_location, fn_name)  # 获取旧模块中的函数对象
            new_function = getattr(new_location, fn_name)  # 获取新模块中的函数对象
            assert old_function == new_function, f"Functions don't match: {fn_name}"  # 断言函数对象相等
            assert hash(old_function) == hash(new_function), (
                f"Hashes don't match: {old_function}({hash(old_function)}) vs. "
                f"{new_function}({hash(new_function)})"
            )  # 断言函数对象的哈希值相等

    def _test_dict_import(
        self, package_name: str, dict_list: List[str], base: Optional[str] = None
    ):
        r"""Tests individual function list import by comparing the functions
        and their hashes."""
        if base is None:
            base = "quantization"
        old_base = "torch." + base  # 构建旧基础模块路径
        new_base = "torch.ao." + base  # 构建新基础模块路径
        old_location = importlib.import_module(f"{old_base}.{package_name}")  # 动态导入旧模块
        new_location = importlib.import_module(f"{new_base}.{package_name}")  # 动态导入新模块
        for dict_name in dict_list:
            old_dict = getattr(old_location, dict_name)  # 获取旧模块中的字典对象
            new_dict = getattr(new_location, dict_name)  # 获取新模块中的字典对象
            assert old_dict == new_dict, f"Dicts don't match: {dict_name}"  # 断言字典对象相等
            for key in new_dict.keys():
                assert (
                    old_dict[key] == new_dict[key]
                ), f"Dicts don't match: {dict_name} for key {key}"  # 断言字典中对应键的值相等
```