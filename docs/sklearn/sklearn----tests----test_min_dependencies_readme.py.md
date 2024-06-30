# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_min_dependencies_readme.py`

```
"""Tests for the minimum dependencies in README.rst and pyproject.toml"""

import os  # 导入操作系统功能
import re  # 导入正则表达式模块
from collections import defaultdict  # 导入默认字典模块
from pathlib import Path  # 导入路径操作模块

import pytest  # 导入 pytest 测试框架

import sklearn  # 导入 scikit-learn 库
from sklearn._min_dependencies import dependent_packages  # 从 scikit-learn 中导入依赖包信息
from sklearn.utils.fixes import parse_version  # 导入版本解析函数

# 创建一个默认字典，用于存储不同额外依赖下的最小依赖包列表
min_depencies_tag_to_packages_without_version = defaultdict(list)
for package, (min_version, extras) in dependent_packages.items():
    for extra in extras.split(", "):
        min_depencies_tag_to_packages_without_version[extra].append(package)

# 创建一个字典，将最小依赖标签映射到 pyproject.toml 文件中的相应部分
min_dependencies_tag_to_pyproject_section = {
    "build": "build-system.requires",
    "install": "project.dependencies",
}
for tag in min_depencies_tag_to_packages_without_version:
    min_dependencies_tag_to_pyproject_section[tag] = (
        f"project.optional-dependencies.{tag}"
    )


def test_min_dependencies_readme():
    # 测试 README.rst 文件中的最小依赖是否与 sklearn/_min_dependencies.py 中定义的一致

    pattern = re.compile(
        r"(\.\. \|)"
        + r"(([A-Za-z]+\-?)+)"
        + r"(MinVersion\| replace::)"
        + r"( [0-9]+\.[0-9]+(\.[0-9]+)?)"
    )

    # 获取 README.rst 文件的路径
    readme_path = Path(sklearn.__file__).parent.parent
    readme_file = readme_path / "README.rst"

    if not os.path.exists(readme_file):
        # 如果 README.rst 文件不存在，则跳过测试
        # 例如，在从 wheels 安装 scikit-learn 时可能会出现这种情况
        pytest.skip("The README.rst file is not available.")

    with readme_file.open("r") as f:
        for line in f:
            matched = pattern.match(line)

            if not matched:
                continue

            # 提取匹配到的包名和版本号
            package, version = matched.group(2), matched.group(5)
            package = package.lower()

            if package in dependent_packages:
                # 解析版本号并进行断言，确保与最小版本匹配
                version = parse_version(version)
                min_version = parse_version(dependent_packages[package][0])

                assert version == min_version, f"{package} has a mismatched version"


def check_pyproject_section(
    pyproject_section, min_dependencies_tag, skip_version_check_for=None
):
    # 导入 tomllib 库并在 Python 3.11 中检查其可用性
    tomllib = pytest.importorskip("tomllib")

    if skip_version_check_for is None:
        skip_version_check_for = []

    # 获取期望检查的包列表
    expected_packages = min_depencies_tag_to_packages_without_version[
        min_dependencies_tag
    ]

    # 获取 scikit-learn 根目录
    root_directory = Path(sklearn.__file__).parent.parent
    pyproject_toml_path = root_directory / "pyproject.toml"

    if not pyproject_toml_path.exists():
        # 如果 pyproject.toml 文件不存在，则跳过测试
        # 例如，在从 wheels 安装 scikit-learn 时可能会出现这种情况
        pytest.skip("pyproject.toml is not available.")

    with pyproject_toml_path.open("rb") as f:
        pyproject_toml = tomllib.load(f)

    # 将 pyproject 部分路径拆分为列表
    pyproject_section_keys = pyproject_section.split(".")
    # 从加载的 pyproject.toml 中提取相关信息
    info = pyproject_toml
    # 遍历 pyproject_section_keys 中的键，逐级获取 info 字典的值
    for key in pyproject_section_keys:
        info = info[key]

    # 初始化一个空字典，用于存储每个依赖包的最低版本要求
    pyproject_build_min_versions = {}

    # 遍历 info 中的每个依赖项
    for requirement in info:
        # 检查依赖项中的版本要求符号，并分割出包名和版本号
        if ">=" in requirement:
            package, version = requirement.split(">=")
        elif "==" in requirement:
            package, version = requirement.split("==")
        else:
            # 如果版本要求不支持 >= 或 ==，抛出 NotImplementedError 异常
            raise NotImplementedError(
                f"{requirement} not supported yet in this test. "
                "Only >= and == are supported for version requirements"
            )

        # 将包名和版本号添加到 pyproject_build_min_versions 字典中
        pyproject_build_min_versions[package] = version

    # 断言：排序后的 pyproject_build_min_versions 的键与 expected_packages 的键相同
    assert sorted(pyproject_build_min_versions) == sorted(expected_packages)

    # 遍历 pyproject_build_min_versions 中的包名及其版本号
    for package, version in pyproject_build_min_versions.items():
        # 解析版本号为标准格式
        version = parse_version(version)
        # 解析依赖包中的预期最低版本为标准格式
        expected_min_version = parse_version(dependent_packages[package][0])
        
        # 如果 package 在 skip_version_check_for 中，跳过版本检查
        if package in skip_version_check_for:
            continue

        # 断言：当前 package 的版本与预期的最低版本相匹配
        assert version == expected_min_version, f"{package} has a mismatched version"
# 使用 pytest 的 parametrize 装饰器来为测试函数提供多组参数化输入
@pytest.mark.parametrize(
    "min_dependencies_tag, pyproject_section",
    min_dependencies_tag_to_pyproject_section.items(),
)
# 定义一个测试函数，用于检查 pyproject.toml 文件中的依赖版本与 _min_dependencies 是否一致
def test_min_dependencies_pyproject_toml(pyproject_section, min_dependencies_tag):
    """Check versions in pyproject.toml is consistent with _min_dependencies."""
    # 如果当前的最小依赖标签是 "build"，则跳过对 NumPy 版本一致性的检查
    skip_version_check_for = ["numpy"] if min_dependencies_tag == "build" else None
    # 调用函数检查 pyproject.toml 中的特定部分与最小依赖标签是否一致，可能跳过指定的版本检查
    check_pyproject_section(
        pyproject_section,
        min_dependencies_tag,
        skip_version_check_for=skip_version_check_for,
    )
```