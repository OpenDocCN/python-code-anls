# `D:\src\scipysrc\pandas\scripts\tests\test_validate_docstrings.py`

```
import io  # 导入 io 模块，用于处理文件流
import textwrap  # 导入 textwrap 模块，用于格式化文本

import pytest  # 导入 pytest 模块，用于编写和运行测试用例

from scripts import validate_docstrings  # 从 scripts 包中导入 validate_docstrings 模块

class BadDocstrings:
    """Everything here has a bad docstring"""

    def private_classes(self) -> None:
        """
        This mentions NDFrame, which is not correct.
        """

    def prefix_pandas(self) -> None:
        """
        Have `pandas` prefix in See Also section.

        See Also
        --------
        pandas.Series.rename : Alter Series index labels or name.
        DataFrame.head : The first `n` rows of the caller object.
        """

    def redundant_import(self, paramx=None, paramy=None) -> None:
        """
        A sample DataFrame method.

        Should not import numpy and pandas.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> df = pd.DataFrame(np.ones((3, 3)),
        ...                   columns=('a', 'b', 'c'))
        >>> df.all(axis=1)
        0    True
        1    True
        2    True
        dtype: bool
        >>> df.all(bool_only=True)
        Series([], dtype: bool)
        """

    def unused_import(self) -> None:
        """
        Examples
        --------
        >>> import pandas as pdf
        >>> df = pd.DataFrame(np.ones((3, 3)), columns=('a', 'b', 'c'))
        """

    def missing_whitespace_around_arithmetic_operator(self) -> None:
        """
        Examples
        --------
        >>> 2+5
        7
        """

    def indentation_is_not_a_multiple_of_four(self) -> None:
        """
        Examples
        --------
        >>> if 2 + 5:
        ...   pass
        """

    def missing_whitespace_after_comma(self) -> None:
        """
        Examples
        --------
        >>> df = pd.DataFrame(np.ones((3,3)),columns=('a','b', 'c'))
        """

    def write_array_like_with_hyphen_not_underscore(self) -> None:
        """
        In docstrings, use array-like over array_like
        """

    def leftover_files(self) -> None:
        """
        Examples
        --------
        >>> import pathlib
        >>> pathlib.Path("foo.txt").touch()
        """


class TestValidator:
    def _import_path(self, klass=None, func=None):
        """
        Build the required import path for tests in this module.

        Parameters
        ----------
        klass : str
            Class name of object in module.
        func : str
            Function name of object in module.

        Returns
        -------
        str
            Import path of specified object in this module
        """
        base_path = "scripts.tests.test_validate_docstrings"

        if klass:
            base_path = f"{base_path}.{klass}"

        if func:
            base_path = f"{base_path}.{func}"

        return base_path

    def test_bad_class(self, capsys) -> None:
        """
        Test case for validating bad docstrings in a class.

        Parameters
        ----------
        capsys : pytest fixture
            Captures stdout and stderr during test execution.

        Raises
        ------
        AssertionError
            If errors list is not returned from validation function.
        """
        # Validate bad docstrings in BadDocstrings class
        errors = validate_docstrings.pandas_validate(
            self._import_path(klass="BadDocstrings")
        )["errors"]
        assert isinstance(errors, list)  # 断言 errors 是一个列表
        assert errors  # 断言 errors 非空
    # 使用 pytest 的参数化装饰器来定义多组测试参数：klass 类名、func 函数名、msgs 错误信息
    @pytest.mark.parametrize(
        "klass,func,msgs",
        [
            (
                "BadDocstrings",
                "private_classes",
                (
                    "Private classes (NDFrame) should not be mentioned in public "
                    "docstrings",
                ),
            ),
            (
                "BadDocstrings",
                "prefix_pandas",
                (
                    "pandas.Series.rename in `See Also` section "
                    "does not need `pandas` prefix",
                ),
            ),
            # 示例测试
            (
                "BadDocstrings",
                "redundant_import",
                ("Do not import numpy, as it is imported automatically",),
            ),
            (
                "BadDocstrings",
                "redundant_import",
                ("Do not import pandas, as it is imported automatically",),
            ),
            (
                "BadDocstrings",
                "unused_import",
                (
                    "flake8 error: line 1, col 1: F401 'pandas as pdf' "
                    "imported but unused",
                ),
            ),
            (
                "BadDocstrings",
                "missing_whitespace_around_arithmetic_operator",
                (
                    "flake8 error: line 1, col 2: "
                    "E226 missing whitespace around arithmetic operator",
                ),
            ),
            (
                "BadDocstrings",
                "indentation_is_not_a_multiple_of_four",
                # 对于 flake8 3.9.0，该消息以四个空格结束；早期版本中以 "four" 结束
                (
                    "flake8 error: line 2, col 3: E111 indentation is not a "
                    "multiple of 4",
                ),
            ),
            (
                "BadDocstrings",
                "missing_whitespace_after_comma",
                ("flake8 error: line 1, col 33: E231 missing whitespace after ','",),
            ),
            (
                "BadDocstrings",
                "write_array_like_with_hyphen_not_underscore",
                ("Use 'array-like' rather than 'array_like' in docstrings",),
            ),
        ],
    )
    # 定义测试函数 test_bad_docstrings，使用 capsys 捕获输出，参数包括 klass 类名、func 函数名、msgs 错误信息
    def test_bad_docstrings(self, capsys, klass, func, msgs) -> None:
        # 调用 validate_docstrings.pandas_validate 函数，传入类和函数名，返回验证结果
        result = validate_docstrings.pandas_validate(
            self._import_path(klass=klass, func=func)
        )
        # 遍历错误信息 msgs，断言每条消息在验证结果的错误列表中
        for msg in msgs:
            assert msg in " ".join([err[1] for err in result["errors"]])
    # 定义一个测试函数，验证在忽略过时警告的情况下的行为
    def test_validate_all_ignore_deprecated(self, monkeypatch) -> None:
        # 使用 monkeypatch 修改 validate_docstrings 中的 pandas_validate 方法，使其返回模拟数据
        monkeypatch.setattr(
            validate_docstrings,
            "pandas_validate",
            lambda func_name: {
                "docstring": "docstring1",
                "errors": [
                    ("ER01", "err desc"),
                    ("ER02", "err desc"),
                    ("ER03", "err desc"),
                ],
                "warnings": [],
                "examples_errors": "",
                "deprecated": True,
            },
        )
        # 调用 validate_all 方法进行验证，预期结果是没有错误
        result = validate_docstrings.validate_all(prefix=None, ignore_deprecated=True)
        assert len(result) == 0

    # 定义一个测试函数，验证在忽略特定错误的情况下的行为
    def test_validate_all_ignore_errors(self, monkeypatch):
        # 使用 monkeypatch 修改 validate_docstrings 中的 pandas_validate 方法，使其返回模拟数据
        monkeypatch.setattr(
            validate_docstrings,
            "pandas_validate",
            lambda func_name: {
                "docstring": "docstring1",
                "errors": [
                    ("ER01", "err desc"),
                    ("ER02", "err desc"),
                    ("ER03", "err desc")
                ],
                "warnings": [],
                "examples_errors": "",
                "deprecated": True,
                "file": "file1",
                "file_line": "file_line1"
            },
        )
        # 使用 monkeypatch 修改 validate_docstrings 中的 get_all_api_items 方法，使其返回模拟数据
        monkeypatch.setattr(
            validate_docstrings,
            "get_all_api_items",
            lambda: [
                (
                    "pandas.DataFrame.align",
                    "func",
                    "current_section",
                    "current_subsection",
                ),
                (
                    "pandas.Index.all",
                    "func",
                    "current_section",
                    "current_subsection",
                ),
            ],
        )

        # 调用 print_validate_all_results 方法，输出默认格式的验证结果，预期不忽略任何错误
        exit_status = validate_docstrings.print_validate_all_results(
            output_format="default",
            prefix=None,
            ignore_deprecated=False,
            ignore_errors={None: {"ER03"}},
        )
        # 预期结果为两个函数各有两个未被忽略的错误
        assert exit_status == 2 * 2

        # 调用 print_validate_all_results 方法，输出默认格式的验证结果，忽略特定错误
        exit_status = validate_docstrings.print_validate_all_results(
            output_format="default",
            prefix=None,
            ignore_deprecated=False,
            ignore_errors={
                None: {"ER03"},
                "pandas.DataFrame.align": {"ER01"},
                # 忽略未请求的错误不应该产生影响
                "pandas.Index.all": {"ER03"}
            }
        )
        # 预期结果为两个函数各有两个未被全局忽略的错误，但一个函数的错误被忽略
        assert exit_status == 2 * 2 - 1
# 测试类，用于测试 API 文档中的各种项目
class TestApiItems:
    
    # 定义属性 api_doc，返回一个包含 API 文档内容的 String IO 对象
    @property
    def api_doc(self):
        return io.StringIO(
            textwrap.dedent(
                """
            .. currentmodule:: itertools

            Itertools
            ---------

            Infinite
            ~~~~~~~~

            .. autosummary::

                cycle
                count

            Finite
            ~~~~~~

            .. autosummary::

                chain

            .. currentmodule:: random

            Random
            ------

            All
            ~~~

            .. autosummary::

                seed
                randint
            """
            )
        )

    # 参数化测试，验证从 API 文档中提取项目名称是否正确
    @pytest.mark.parametrize(
        "idx,name",
        [
            (0, "itertools.cycle"),
            (1, "itertools.count"),
            (2, "itertools.chain"),
            (3, "random.seed"),
            (4, "random.randint"),
        ],
    )
    def test_item_name(self, idx, name) -> None:
        result = list(validate_docstrings.get_api_items(self.api_doc))
        assert result[idx][0] == name

    # 参数化测试，验证从 API 文档中提取函数对象是否可调用且名称正确
    @pytest.mark.parametrize(
        "idx,func",
        [(0, "cycle"), (1, "count"), (2, "chain"), (3, "seed"), (4, "randint")],
    )
    def test_item_function(self, idx, func) -> None:
        result = list(validate_docstrings.get_api_items(self.api_doc))
        assert callable(result[idx][1])
        assert result[idx][1].__name__ == func

    # 参数化测试，验证从 API 文档中提取项目的章节信息是否正确
    @pytest.mark.parametrize(
        "idx,section",
        [
            (0, "Itertools"),
            (1, "Itertools"),
            (2, "Itertools"),
            (3, "Random"),
            (4, "Random"),
        ],
    )
    def test_item_section(self, idx, section) -> None:
        result = list(validate_docstrings.get_api_items(self.api_doc))
        assert result[idx][2] == section

    # 参数化测试，验证从 API 文档中提取项目的子章节信息是否正确
    @pytest.mark.parametrize(
        "idx,subsection",
        [(0, "Infinite"), (1, "Infinite"), (2, "Finite"), (3, "All"), (4, "All")],
    )
    def test_item_subsection(self, idx, subsection) -> None:
        result = list(validate_docstrings.get_api_items(self.api_doc))
        assert result[idx][3] == subsection


# 测试 PandasDocstring 类的文档字符串验证功能
class TestPandasDocstringClass:
    
    # 参数化测试，验证给定 Pandas API 名称的文档字符串是否符合 PEP8
    @pytest.mark.parametrize(
        "name", ["pandas.Series.str.isdecimal", "pandas.Series.str.islower"]
    )
    def test_encode_content_write_to_file(self, name) -> None:
        # 创建 PandasDocstring 对象并验证其符合 PEP8
        docstr = validate_docstrings.PandasDocstring(name).validate_pep8()
        # 确保 PEP8 错误列表为空
        assert not list(docstr)


# 测试主函数入口
class TestMainFunction:
    # 定义一个测试函数，验证主程序的退出状态是否正确
    def test_exit_status_for_main(self, monkeypatch) -> None:
        # 使用 monkeypatch 替换 validate_docstrings 模块中的 pandas_validate 函数
        monkeypatch.setattr(
            validate_docstrings,
            "pandas_validate",
            # 定义一个匿名函数，模拟 pandas_validate 的返回结果
            lambda func_name: {
                "docstring": "docstring1",
                "errors": [
                    ("ER01", "err desc"),
                    ("ER02", "err desc"),
                    ("ER03", "err desc"),
                ],
                "examples_errs": "",
            },
        )
        # 调用主程序 main 函数，验证程序的退出状态
        exit_status = validate_docstrings.main(
            func_name="docstring1",
            prefix=None,
            output_format="default",
            ignore_deprecated=False,
            ignore_errors={},
        )
        # 断言程序的退出状态是否为 3
        assert exit_status == 3

    # 定义一个测试函数，验证 validate_all 函数的错误退出状态是否正确
    def test_exit_status_errors_for_validate_all(self, monkeypatch) -> None:
        # 使用 monkeypatch 替换 validate_docstrings 模块中的 validate_all 函数
        monkeypatch.setattr(
            validate_docstrings,
            "validate_all",
            # 定义一个匿名函数，模拟 validate_all 的返回结果
            lambda prefix, ignore_deprecated=False, ignore_functions=None: {
                "docstring1": {
                    "errors": [
                        ("ER01", "err desc"),
                        ("ER02", "err desc"),
                        ("ER03", "err desc"),
                    ],
                    "file": "module1.py",
                    "file_line": 23,
                },
                "docstring2": {
                    "errors": [("ER04", "err desc"), ("ER05", "err desc")],
                    "file": "module2.py",
                    "file_line": 925,
                },
            },
        )
        # 调用主程序 main 函数，验证程序的退出状态
        exit_status = validate_docstrings.main(
            func_name=None,
            prefix=None,
            output_format="default",
            ignore_deprecated=False,
            ignore_errors={},
        )
        # 断言程序的退出状态是否为 5
        assert exit_status == 5

    # 定义一个测试函数，验证 validate_all 函数在没有错误时的退出状态是否正确
    def test_no_exit_status_noerrors_for_validate_all(self, monkeypatch) -> None:
        # 使用 monkeypatch 替换 validate_docstrings 模块中的 validate_all 函数
        monkeypatch.setattr(
            validate_docstrings,
            "validate_all",
            # 定义一个匿名函数，模拟 validate_all 的返回结果，其中不包含错误
            lambda prefix, ignore_deprecated=False, ignore_functions=None: {
                "docstring1": {"errors": [], "warnings": [("WN01", "warn desc")]},
                "docstring2": {"errors": []},
            },
        )
        # 调用主程序 main 函数，验证程序的退出状态
        exit_status = validate_docstrings.main(
            func_name=None,
            output_format="default",
            prefix=None,
            ignore_deprecated=False,
            ignore_errors={},
        )
        # 断言程序的退出状态是否为 0
        assert exit_status == 0
    # 定义一个测试方法，验证在使用 monkeypatch 修改 validate_docstrings.validate_all 的行为后，对返回的 JSON 结果进行检查
    def test_exit_status_for_validate_all_json(self, monkeypatch) -> None:
        # 使用 monkeypatch 设置 validate_docstrings.validate_all 的返回值为一个包含两个文档字符串检查结果的字典
        monkeypatch.setattr(
            validate_docstrings,
            "validate_all",
            lambda prefix, ignore_deprecated=False, ignore_functions=None: {
                "docstring1": {
                    "errors": [
                        ("ER01", "err desc"),
                        ("ER02", "err desc"),
                        ("ER03", "err desc"),
                    ]
                },
                "docstring2": {"errors": [("ER04", "err desc"), ("ER05", "err desc")]},
            },
        )
        # 调用 validate_docstrings.main 方法，检查输出格式为 JSON，不忽略已弃用的函数，无忽略的错误
        exit_status = validate_docstrings.main(
            func_name=None,
            output_format="json",
            prefix=None,
            ignore_deprecated=False,
            ignore_errors={},
        )
        # 断言退出状态为 0，即检查通过
        assert exit_status == 0

    # 定义另一个测试方法，验证在使用 monkeypatch 设置特定错误码忽略规则后，validate_docstrings.main 的行为
    def test_errors_param_filters_errors(self, monkeypatch) -> None:
        # 使用 monkeypatch 设置 validate_docstrings.validate_all 的返回值为一个包含三个函数文档字符串检查结果的字典
        monkeypatch.setattr(
            validate_docstrings,
            "validate_all",
            lambda prefix, ignore_deprecated=False, ignore_functions=None: {
                "Series.foo": {
                    "errors": [
                        ("ER01", "err desc"),
                        ("ER02", "err desc"),
                        ("ER03", "err desc"),
                    ],
                    "file": "series.py",
                    "file_line": 142,
                },
                "DataFrame.bar": {
                    "errors": [("ER01", "err desc"), ("ER02", "err desc")],
                    "file": "frame.py",
                    "file_line": 598,
                },
                "Series.foobar": {
                    "errors": [("ER01", "err desc")],
                    "file": "series.py",
                    "file_line": 279,
                },
            },
        )
        # 使用 monkeypatch 设置 validate_docstrings.ERROR_MSGS 为一个错误码到错误描述的映射字典
        monkeypatch.setattr(
            validate_docstrings,
            "ERROR_MSGS",
            {
                "ER01": "err desc",
                "ER02": "err desc",
                "ER03": "err desc",
            },
        )
        # 调用 validate_docstrings.main 方法，检查输出格式为默认，不忽略已弃用的函数，忽略指定的错误码集合
        exit_status = validate_docstrings.main(
            func_name=None,
            output_format="default",
            prefix=None,
            ignore_deprecated=False,
            ignore_errors={None: {"ER02", "ER03"}},
        )
        # 断言退出状态为 3，表示有三处错误被检测到但被忽略
        assert exit_status == 3

        # 再次调用 validate_docstrings.main 方法，忽略不同的错误码集合
        exit_status = validate_docstrings.main(
            func_name=None,
            output_format="default",
            prefix=None,
            ignore_deprecated=False,
            ignore_errors={None: {"ER01", "ER02"}},
        )
        # 断言退出状态为 1，表示有一处错误被检测到但被忽略
        assert exit_status == 1
```