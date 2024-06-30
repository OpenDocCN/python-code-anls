# `D:\src\scipysrc\sympy\sympy\testing\tests\test_runtests_pytest.py`

```
import pathlib  # 导入处理路径的模块
from typing import List  # 导入类型提示中的 List 类型

import pytest  # 导入 pytest 测试框架

from sympy.testing.runtests_pytest import (  # 从 sympy.testing.runtests_pytest 模块导入以下函数
    make_absolute_path,  # 导入 make_absolute_path 函数
    sympy_dir,  # 导入 sympy_dir 函数
    update_args_with_paths,  # 导入 update_args_with_paths 函数
    update_args_with_rootdir,  # 导入 update_args_with_rootdir 函数
)


def test_update_args_with_rootdir():
    """`--rootdir` and directory three above this added as arguments."""
    # 调用 update_args_with_rootdir 函数，返回结果与预期列表比较
    args = update_args_with_rootdir([])
    assert args == ['--rootdir', str(pathlib.Path(__file__).parents[3])]


class TestMakeAbsolutePath:

    @staticmethod
    @pytest.mark.parametrize(
        'partial_path', ['sympy', 'sympy/core', 'sympy/nonexistant_directory'],
    )
    def test_valid_partial_path(partial_path: str):
        """Paths that start with `sympy` are valid."""
        _ = make_absolute_path(partial_path)

    @staticmethod
    @pytest.mark.parametrize(
        'partial_path', ['not_sympy', 'also/not/sympy'],
    )
    def test_invalid_partial_path_raises_value_error(partial_path: str):
        """A `ValueError` is raises on paths that don't start with `sympy`."""
        # 使用 pytest.raises 检查 make_absolute_path 对于不以 'sympy' 开头的路径是否引发 ValueError 异常
        with pytest.raises(ValueError):
            _ = make_absolute_path(partial_path)


class TestUpdateArgsWithPaths:

    @staticmethod
    def test_no_paths():
        """If no paths are passed, only `sympy` and `doc/src` are appended.

        `sympy` and `doc/src` are the `testpaths` stated in `pytest.ini`. They
        need to be manually added as if any path-related arguments are passed
        to `pytest.main` then the settings in `pytest.ini` may be ignored.

        """
        # 当 paths 为空列表时，调用 update_args_with_paths 函数，验证返回的参数列表是否符合预期
        paths = []
        args = update_args_with_paths(paths=paths, keywords=None, args=[])
        expected = [
            str(pathlib.Path(sympy_dir(), 'sympy')),  # 使用 sympy_dir 函数获取路径并拼接 'sympy'
            str(pathlib.Path(sympy_dir(), 'doc/src')),  # 使用 sympy_dir 函数获取路径并拼接 'doc/src'
        ]
        assert args == expected

    @staticmethod
    @pytest.mark.parametrize(
        'path',
        ['sympy/core/tests/test_basic.py', '_basic']
    )
    def test_one_file(path: str):
        """Single files/paths, full or partial, are matched correctly."""
        # 测试传入单个文件路径时，update_args_with_paths 函数返回的参数列表是否符合预期
        args = update_args_with_paths(paths=[path], keywords=None, args=[])
        expected = [
            str(pathlib.Path(sympy_dir(), 'sympy/core/tests/test_basic.py')),  # 使用 sympy_dir 函数获取路径并拼接具体文件路径
        ]
        assert args == expected

    @staticmethod
    def test_partial_path_from_root():
        """Partial paths from the root directly are matched correctly."""
        # 测试从根目录直接传入部分路径时，update_args_with_paths 函数返回的参数列表是否符合预期
        args = update_args_with_paths(paths=['sympy/functions'], keywords=None, args=[])
        expected = [str(pathlib.Path(sympy_dir(), 'sympy/functions'))]  # 使用 sympy_dir 函数获取路径并拼接部分路径
        assert args == expected

    @staticmethod
    def test_multiple_paths_from_root():
        """测试从根目录开始的多个路径匹配是否正确。"""
        # 定义要测试的路径列表
        paths = ['sympy/core/tests/test_basic.py', 'sympy/functions']
        # 调用函数，更新路径参数，并获取返回值
        args = update_args_with_paths(paths=paths, keywords=None, args=[])
        # 期望的结果列表，包含完整路径
        expected = [
            str(pathlib.Path(sympy_dir(), 'sympy/core/tests/test_basic.py')),
            str(pathlib.Path(sympy_dir(), 'sympy/functions')),
        ]
        # 断言返回的结果与期望的结果相等
        assert args == expected

    @staticmethod
    @pytest.mark.parametrize(
        'paths, expected_paths',
        [
            (
                ['/core', '/util'],
                [
                    'doc/src/modules/utilities',
                    'doc/src/reference/public/utilities',
                    'sympy/core',
                    'sympy/logic/utilities',
                    'sympy/utilities',
                ]
            ),
        ]
    )
    def test_multiple_paths_from_non_root(paths: List[str], expected_paths: List[str]):
        """测试从非根路径开始的多个路径匹配是否正确。"""
        # 调用函数，更新路径参数，并获取返回值
        args = update_args_with_paths(paths=paths, keywords=None, args=[])
        # 断言返回的结果数量与期望的路径数量相等
        assert len(args) == len(expected_paths)
        # 检查每个返回的路径是否包含在期望的路径列表中
        for arg, expected in zip(sorted(args), expected_paths):
            assert expected in arg

    @staticmethod
    @pytest.mark.parametrize(
        'paths',
        [
            [],  # 空路径列表的测试
            ['sympy/physics'],  # 包含单个路径的测试
            ['sympy/physics/mechanics'],  # 包含深层路径的测试
            ['sympy/physics/mechanics/tests'],  # 包含更深层路径的测试
            ['sympy/physics/mechanics/tests/test_kane3.py'],  # 包含具体文件的测试
        ]
    )
    def test_string_as_keyword(paths: List[str]):
        """测试字符串关键字的匹配是否正确。"""
        # 定义关键字列表
        keywords = ('bicycle', )
        # 调用函数，更新路径参数，并获取返回值
        args = update_args_with_paths(paths=paths, keywords=keywords, args=[])
        # 期望的结果列表，包含特定格式的路径与关键字
        expected_args = ['sympy/physics/mechanics/tests/test_kane3.py::test_bicycle']
        # 断言返回的结果数量与期望的结果数量相等
        assert len(args) == len(expected_args)
        # 检查每个返回的结果是否包含在期望的结果列表中
        for arg, expected in zip(sorted(args), expected_args):
            assert expected in arg

    @staticmethod
    @pytest.mark.parametrize(
        'paths',
        [
            [],  # 空路径列表的测试
            ['sympy/core'],  # 包含单个路径的测试
            ['sympy/core/tests'],  # 包含深层路径的测试
            ['sympy/core/tests/test_sympify.py'],  # 包含具体文件的测试
        ]
    )
    def test_integer_as_keyword(paths: List[str]):
        """测试整数关键字的匹配是否正确。"""
        # 定义关键字列表
        keywords = ('3538', )
        # 调用函数，更新路径参数，并获取返回值
        args = update_args_with_paths(paths=paths, keywords=keywords, args=[])
        # 期望的结果列表，包含特定格式的路径与关键字
        expected_args = ['sympy/core/tests/test_sympify.py::test_issue_3538']
        # 断言返回的结果数量与期望的结果数量相等
        assert len(args) == len(expected_args)
        # 检查每个返回的结果是否包含在期望的结果列表中
        for arg, expected in zip(sorted(args), expected_args):
            assert expected in arg
    def test_multiple_keywords():
        """测试多个关键词是否正确匹配。"""
        # 定义关键词列表
        keywords = ('bicycle', '3538')
        # 调用函数更新参数列表
        args = update_args_with_paths(paths=[], keywords=keywords, args=[])
        # 预期的参数列表
        expected_args = [
            'sympy/core/tests/test_sympify.py::test_issue_3538',
            'sympy/physics/mechanics/tests/test_kane3.py::test_bicycle',
        ]
        # 断言实际参数列表的长度与预期相同
        assert len(args) == len(expected_args)
        # 检查每个实际参数是否包含在预期参数列表中
        for arg, expected in zip(sorted(args), expected_args):
            assert expected in arg

    @staticmethod
    def test_keyword_match_in_multiple_files():
        """测试关键词是否跨多个文件正确匹配。"""
        # 定义关键词列表
        keywords = ('1130', )
        # 调用函数更新参数列表
        args = update_args_with_paths(paths=[], keywords=keywords, args=[])
        # 预期的参数列表
        expected_args = [
            'sympy/integrals/tests/test_heurisch.py::test_heurisch_symbolic_coeffs_1130',
            'sympy/utilities/tests/test_lambdify.py::test_python_div_zero_issue_11306',
        ]
        # 断言实际参数列表的长度与预期相同
        assert len(args) == len(expected_args)
        # 检查每个实际参数是否包含在预期参数列表中
        for arg, expected in zip(sorted(args), expected_args):
            assert expected in arg
```