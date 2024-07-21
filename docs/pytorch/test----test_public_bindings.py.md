# `.\pytorch\test\test_public_bindings.py`

```
# Owner(s): ["module: autograd"]

# 导入必要的模块和函数
import importlib
import inspect
import json
import os
import pkgutil
import unittest
from itertools import chain
from pathlib import Path
from typing import Callable

import torch
from torch._utils_internal import get_file_path_2
from torch.testing._internal.common_utils import (
    IS_JETSON,
    IS_MACOS,
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)

# 查找给定包中的所有可导入项并按顺序返回
def _find_all_importables(pkg):
    """Find all importables in the project.

    Return them in order.
    """
    return sorted(
        set(
            chain.from_iterable(
                _discover_path_importables(Path(p), pkg.__name__)
                for p in pkg.__path__
            ),
        ),
    )

# 在给定路径和包名下生成所有可导入的项
def _discover_path_importables(pkg_pth, pkg_name):
    """Yield all importables under a given path and package.

    This is like pkgutil.walk_packages, but does *not* skip over namespace
    packages. Taken from https://stackoverflow.com/questions/41203765/init-py-required-for-pkgutil-walk-packages-in-python3
    """
    for dir_path, _d, file_names in os.walk(pkg_pth):
        pkg_dir_path = Path(dir_path)

        # 跳过 __pycache__ 文件夹
        if pkg_dir_path.parts[-1] == '__pycache__':
            continue

        # 跳过不包含 .py 文件的文件夹
        if all(Path(_).suffix != '.py' for _ in file_names):
            continue

        # 计算相对路径和包前缀
        rel_pt = pkg_dir_path.relative_to(pkg_pth)
        pkg_pref = '.'.join((pkg_name, ) + rel_pt.parts)

        # 使用 pkgutil.walk_packages 生成模块路径并返回
        yield from (
            pkg_path
            for _, pkg_path, _ in pkgutil.walk_packages(
                (str(pkg_dir_path), ), prefix=f'{pkg_pref}.',
            )
        )

# 测试类，用于检查 torch 模块中不应以 _ 开头的重新导出的可调用项
class TestPublicBindings(TestCase):
    def test_no_new_reexport_callables(self):
        """
        This test aims to stop the introduction of new re-exported callables into
        torch whose names do not start with _. Such callables are made available as
        torch.XXX, which may not be desirable.
        """
        # 获取所有不以 'torch' 开头的可调用项，并排序
        reexported_callables = sorted(
            k
            for k, v in vars(torch).items()
            if callable(v) and not v.__module__.startswith('torch')
        )
        # 断言所有可调用项的名称都以 '_' 开头
        self.assertTrue(all(k.startswith('_') for k in reexported_callables), reexported_callables)

    @staticmethod
    # 检查给定模块名是否为公共模块（不以 '_' 开头）
    def _is_mod_public(modname):
        split_strs = modname.split('.')
        for elem in split_strs:
            if elem.startswith("_"):
                return False
        return True

    # 根据平台跳过测试：Windows、macOS 平台下会跳过
    @unittest.skipIf(IS_WINDOWS or IS_MACOS, "Inductor/Distributed modules hard fail on windows and macos")
    # 根据条件跳过测试：当 IS_JETSON 或 IS_WINDOWS 或 IS_MACOS 为真时跳过
    @skipIfTorchDynamo("Broken and not relevant for now")
    @unittest.skipIf(IS_WINDOWS or IS_JETSON or IS_MACOS, "Distributed Attribute Error")
    @skipIfTorchDynamo("Broken and not relevant for now")
# 如果是主程序，则运行测试
if __name__ == '__main__':
    run_tests()
```