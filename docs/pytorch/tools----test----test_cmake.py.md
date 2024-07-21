# `.\pytorch\tools\test\test_cmake.py`

```
    # 引入未来的注释，允许在当前版本的 Python 中使用来自未来版本的特性
    from __future__ import annotations

    # 引入上下文管理模块
    import contextlib
    # 引入操作系统相关的功能模块
    import os
    # 引入类型提示相关的模块
    import typing
    # 引入单元测试框架
    import unittest
    # 引入用于模拟测试对象的模块
    import unittest.mock
    # 引入迭代器类型
    from typing import Iterator, Sequence

    # 引入自定义的 CMake 设置辅助模块
    import tools.setup_helpers.cmake
    # 引入环境设置辅助模块，并标记为忽略 F401 警告，因为它虽然未使用但解决了循环导入问题
    import tools.setup_helpers.env  # noqa: F401 unused but resolves circular import


    # 定义一个类型变量 T，用于类型提示
    T = typing.TypeVar("T")


    # 定义一个测试类 TestCMake，继承自 unittest.TestCase
    class TestCMake(unittest.TestCase):

        # 测试方法 test_build_jobs，用于测试构建作业的数量是否正确
        @unittest.mock.patch("multiprocessing.cpu_count")
        def test_build_jobs(self, mock_cpu_count: unittest.mock.MagicMock) -> None:
            """Tests that the number of build jobs comes out correctly."""
            # 模拟 CPU 核心数量为 13
            mock_cpu_count.return_value = 13

            # 定义测试用例，包括输入和期望输出
            cases = [
                # MAX_JOBS, USE_NINJA, IS_WINDOWS,         want
                (("8", True, False), ["-j", "8"]),  # noqa: E201,E241
                ((None, True, False), None),  # noqa: E201,E241
                (("7", False, False), ["-j", "7"]),  # noqa: E201,E241
                ((None, False, False), ["-j", "13"]),  # noqa: E201,E241
                (("6", True, True), ["-j", "6"]),  # noqa: E201,E241
                ((None, True, True), None),  # noqa: E201,E241
                (("11", False, True), ["/p:CL_MPCount=11"]),  # noqa: E201,E241
                ((None, False, True), ["/p:CL_MPCount=13"]),  # noqa: E201,E241
            ]

            # 遍历测试用例
            for (max_jobs, use_ninja, is_windows), want in cases:
                # 使用子测试进行测试，传入 MAX_JOBS、USE_NINJA 和 IS_WINDOWS 作为子测试名称
                with self.subTest(
                    MAX_JOBS=max_jobs, USE_NINJA=use_ninja, IS_WINDOWS=is_windows
                ):
                    # 使用上下文管理器进行环境设置
                    with contextlib.ExitStack() as stack:
                        # 设置环境变量 MAX_JOBS
                        stack.enter_context(env_var("MAX_JOBS", max_jobs))
                        # 使用 mock 对象模拟 USE_NINJA 的值
                        stack.enter_context(
                            unittest.mock.patch.object(
                                tools.setup_helpers.cmake, "USE_NINJA", use_ninja
                            )
                        )
                        # 使用 mock 对象模拟 IS_WINDOWS 的值
                        stack.enter_context(
                            unittest.mock.patch.object(
                                tools.setup_helpers.cmake, "IS_WINDOWS", is_windows
                            )
                        )

                        # 创建 CMake 对象
                        cmake = tools.setup_helpers.cmake.CMake()

                        # 使用 mock 对象模拟 cmake.run 方法
                        with unittest.mock.patch.object(cmake, "run") as cmake_run:
                            # 调用 CMake 对象的 build 方法
                            cmake.build({})

                        # 断言 cmake.run 方法仅调用一次
                        cmake_run.assert_called_once()
                        (call,) = cmake_run.mock_calls
                        build_args, _ = call.args

                    # 如果期望输出为 None，则断言 build_args 不包含 "-j"
                    if want is None:
                        self.assertNotIn("-j", build_args)
                    # 否则，断言 build_args 包含期望的输出 want
                    else:
                        self.assert_contains_sequence(build_args, want)

        # 静态方法 assert_contains_sequence，用于断言一个序列包含另一个序列
        @staticmethod
        def assert_contains_sequence(
            sequence: Sequence[T], subsequence: Sequence[T]
        ) -> None:
    ) -> None:
        """
        如果子序列不包含在序列中，则引发断言错误。
        """
        if len(subsequence) == 0:
            return  # 如果子序列为空，则所有序列都包含空子序列

        # 遍历所有长度为 len(subsequence) 的窗口，如果找到匹配的窗口则停止。
        for i in range(len(sequence) - len(subsequence) + 1):
            candidate = sequence[i : i + len(subsequence)]
            assert len(candidate) == len(subsequence)  # 检查长度是否一致，作为健全性检查
            if candidate == subsequence:
                return  # 找到匹配的子序列

        # 如果没有找到匹配的子序列，则引发断言错误，显示未找到的子序列和完整序列。
        raise AssertionError(f"{subsequence} not found in {sequence}")
# 创建一个上下文管理器，用于设置或清除 Python 上下文中的环境变量
@contextlib.contextmanager
def env_var(key: str, value: str | None) -> Iterator[None]:
    """Sets/clears an environment variable within a Python context."""
    # 获取当前环境变量的先前值
    previous_value = os.environ.get(key)
    # 设置环境变量为指定值
    set_env_var(key, value)
    try:
        # 执行上下文中的代码块
        yield
    finally:
        # 恢复环境变量为先前的值
        set_env_var(key, previous_value)

# 设置或清除指定的环境变量
def set_env_var(key: str, value: str | None) -> None:
    """Sets/clears an environment variable."""
    # 如果值为 None，则从环境变量中删除键
    if value is None:
        os.environ.pop(key, None)
    else:
        # 否则，将键设置为指定的值
        os.environ[key] = value

# 当脚本作为主程序运行时，执行单元测试
if __name__ == "__main__":
    unittest.main()
```