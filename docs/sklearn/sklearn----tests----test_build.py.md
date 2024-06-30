# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_build.py`

```
import os
import textwrap

import pytest

from sklearn import __version__  # 导入 scikit-learn 版本信息
from sklearn.utils._openmp_helpers import _openmp_parallelism_enabled  # 导入检查 OpenMP 并行性的函数


def test_openmp_parallelism_enabled():
    # 检查 sklearn 是否使用基于 OpenMP 的并行化功能构建。
    # 可以通过设置环境变量 SKLEARN_SKIP_OPENMP_TEST 来跳过此测试。
    if os.getenv("SKLEARN_SKIP_OPENMP_TEST"):
        pytest.skip("test explicitly skipped (SKLEARN_SKIP_OPENMP_TEST)")

    # 根据 scikit-learn 的版本号结尾是否为 ".dev0" 来确定基础 URL
    base_url = "dev" if __version__.endswith(".dev0") else "stable"

    # 构建错误消息，使用 textwrap.dedent 来移除多余的缩进
    err_msg = textwrap.dedent(
        """
        This test fails because scikit-learn has been built without OpenMP.
        This is not recommended since some estimators will run in sequential
        mode instead of leveraging thread-based parallelism.

        You can find instructions to build scikit-learn with OpenMP at this
        address:

            https://scikit-learn.org/{}/developers/advanced_installation.html

        You can skip this test by setting the environment variable
        SKLEARN_SKIP_OPENMP_TEST to any value.
        """
    ).format(base_url)

    # 断言检查是否启用了 OpenMP 并行性，如果未启用则输出错误消息
    assert _openmp_parallelism_enabled(), err_msg
```