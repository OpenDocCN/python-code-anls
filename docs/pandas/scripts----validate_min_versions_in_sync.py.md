# `D:\src\scipysrc\pandas\scripts\validate_min_versions_in_sync.py`

```
#!/usr/bin/env python3
"""
Check pandas required and optional dependencies are synced across:

ci/deps/actions-.*-minimum_versions.yaml
pandas/compat/_optional.py
setup.cfg

TODO: doc/source/getting_started/install.rst

This is meant to be run as a pre-commit hook - to run it manually, you can do:

    pre-commit run validate-min-versions-in-sync --all-files
"""
from __future__ import annotations

import pathlib  # 导入处理路径的模块
import sys  # 导入系统相关的模块

import yaml  # 导入处理 YAML 格式的模块

if sys.version_info >= (3, 11):
    import tomllib  # 导入 tomllib 库，用于 Python 版本大于等于 3.11
else:
    import tomli as tomllib  # 导入 tomli 库，用于 Python 版本小于 3.11

from typing import Any  # 导入 Any 类型，用于支持动态类型

from scripts.generate_pip_deps_from_conda import CONDA_TO_PIP  # 导入从 Conda 生成 Pip 依赖的函数

DOC_PATH = pathlib.Path("doc/source/getting_started/install.rst").resolve()  # 获取解析后的安装文档路径
CI_PATH = next(  # 获取解析后的 CI 最小版本配置文件路径
    pathlib.Path("ci/deps").absolute().glob("actions-*-minimum_versions.yaml")
)
CODE_PATH = pathlib.Path("pandas/compat/_optional.py").resolve()  # 获取解析后的 Pandas 兼容性模块路径
SETUP_PATH = pathlib.Path("pyproject.toml").resolve()  # 获取解析后的项目配置文件路径
YAML_PATH = pathlib.Path("ci/deps")  # 获取解析后的 YAML 文件路径
ENV_PATH = pathlib.Path("environment.yml")  # 获取解析后的环境配置文件路径
EXCLUDE_DEPS = {"tzdata", "blosc", "pyqt", "pyqt5"}  # 定义需排除的依赖集合
EXCLUSION_LIST = frozenset(["python=3.8[build=*_pypy]"])  # 冻结的排除列表，指定 Python 版本和构建类型
# pandas package is not available
# in pre-commit environment
sys.path.append("pandas/compat")  # 将 Pandas 兼容性模块路径添加到系统路径中
sys.path.append("pandas/util")  # 将 Pandas 实用工具模块路径添加到系统路径中
import _exceptions  # 导入 Pandas 的异常处理模块
import version  # 导入版本模块

sys.modules["pandas.util.version"] = version  # 在系统模块中注册 Pandas 版本模块
sys.modules["pandas.util._exceptions"] = _exceptions  # 在系统模块中注册 Pandas 异常处理模块
import _optional  # 导入 Pandas 可选模块


def pin_min_versions_to_ci_deps() -> int:
    """
    Pin minimum versions to CI dependencies.

    Pip dependencies are not pinned.
    """
    all_yaml_files = list(YAML_PATH.iterdir())  # 获取 YAML 文件夹中所有文件的列表
    all_yaml_files.append(ENV_PATH)  # 将环境配置文件路径添加到列表末尾
    toml_dependencies = {}  # 初始化空字典，存储来自 Toml 文件的依赖信息
    with open(SETUP_PATH, "rb") as toml_f:
        toml_dependencies = tomllib.load(toml_f)  # 加载并解析 Toml 文件，存储在 toml_dependencies 中
    ret = 0  # 初始化返回值为 0
    for curr_file in all_yaml_files:  # 遍历所有的 YAML 文件和环境配置文件
        with open(curr_file, encoding="utf-8") as yaml_f:
            yaml_start_data = yaml_f.read()  # 读取 YAML 文件的内容为字符串
        yaml_file = yaml.safe_load(yaml_start_data)  # 安全加载 YAML 文件内容为 Python 对象
        yaml_dependencies = yaml_file["dependencies"]  # 获取 YAML 文件中的依赖项
        yaml_map = get_yaml_map_from(yaml_dependencies)  # 调用函数获取 YAML 的依赖映射表
        toml_map = get_toml_map_from(toml_dependencies)  # 调用函数获取 Toml 的依赖映射表
        yaml_result_data = pin_min_versions_to_yaml_file(
            yaml_map, toml_map, yaml_start_data
        )  # 调用函数更新 YAML 文件中的最小版本信息，并获取更新后的内容
        if yaml_result_data != yaml_start_data:  # 检查更新后的 YAML 内容是否有变化
            with open(curr_file, "w", encoding="utf-8") as f:
                f.write(yaml_result_data)  # 如果有变化，则写入更新后的 YAML 内容到文件中
            ret |= 1  # 设置返回值的相应位为 1
    return ret  # 返回结果


def get_toml_map_from(toml_dic: dict[str, Any]) -> dict[str, str]:
    """
    Extracts a dictionary of package names mapped to their minimum required versions from a TOML dictionary.

    Args:
        toml_dic (dict[str, Any]): TOML dictionary containing project optional dependencies.

    Returns:
        dict[str, str]: A dictionary mapping package names to their minimum required versions.
    """
    toml_deps = {}  # 初始化空字典，存储从 TOML 文件中提取的依赖信息
    toml_dependencies = set(toml_dic["project"]["optional-dependencies"]["all"])  # 获取 TOML 文件中的所有可选依赖项
    for dependency in toml_dependencies:  # 遍历 TOML 文件中的每个依赖项
        toml_package, toml_version = dependency.strip().split(">=")  # 拆分依赖项，获取包名和最小版本号
        toml_deps[toml_package] = toml_version  # 将包名和版本号添加到字典中
    return toml_deps  # 返回 TOML 文件中提取的依赖映射表


def get_operator_from(dependency: str) -> str | None:
    """
    Extracts the comparison operator from a dependency string.

    Args:
        dependency (str): Dependency string containing version constraints.

    Returns:
        str | None: Comparison operator ('<=', '>=', '=', '>') or None if not found.
    """
    if "<=" in dependency:  # 如果依赖字符串包含 "<="
        operator = "<="  # 设置比较运算符为 "<="
    elif ">=" in dependency:  # 如果依赖字符串包含 ">="
        operator = ">="  # 设置比较运算符为 ">="
    elif "=" in dependency:  # 如果依赖字符串包含 "="
        operator = "="  # 设置比较运算符为 "="
    elif ">" in dependency:  # 如果依赖字符串包含 ">"
        operator = ">"  # 设置比较运算符为 ">"
    else:
        operator = None  # 否则设置比较运算符为 None
    return operator  # 返回比较运算符
    elif "<" in dependency:
        # 如果依赖关系中包含 "<" 符号，则设置操作符为 "<"
        operator = "<"
    else:
        # 如果依赖关系中不包含 "<" 符号，则设置操作符为 None
        operator = None
    # 返回确定的操作符（可以是 "<" 或 None）
    return operator
# 从 yaml_dic 中获取 YAML 映射，返回一个字典，键为字符串，值为字符串列表或空值
def get_yaml_map_from(
    yaml_dic: list[str | dict[str, list[str]]]
) -> dict[str, list[str] | None]:
    # 初始化一个空的 YAML 映射字典
    yaml_map: dict[str, list[str] | None] = {}

    # 遍历 yaml_dic 中的每个依赖项
    for dependency in yaml_dic:
        # 如果依赖项是字典类型，或者在排除列表 EXCLUSION_LIST 中，或者已经在 yaml_map 中存在，则跳过
        if (
            isinstance(dependency, dict)
            or dependency in EXCLUSION_LIST
            or dependency in yaml_map
        ):
            continue

        # 将依赖项转换为字符串
        search_text = str(dependency)
        
        # 从搜索文本中获取操作符
        operator = get_operator_from(search_text)

        # 如果依赖项包含逗号，则解析出 YAML 包和版本号
        if "," in dependency:
            yaml_dependency, yaml_version1 = search_text.split(",")
            operator = get_operator_from(yaml_dependency)
            assert operator is not None
            yaml_package, yaml_version2 = yaml_dependency.split(operator)
            yaml_version2 = operator + yaml_version2
            yaml_map[yaml_package] = [yaml_version1, yaml_version2]
        
        # 如果依赖项包含特定的构建说明 "[build=*_pypy]"，则处理该情况
        elif "[build=*_pypy]" in dependency:
            search_text = search_text.replace("[build=*_pypy]", "")
            yaml_package, yaml_version = search_text.split(operator)
            yaml_version = operator + yaml_version
            yaml_map[yaml_package] = [yaml_version]
        
        # 如果依赖项包含操作符，则解析出 YAML 包和版本号
        elif operator is not None:
            yaml_package, yaml_version = search_text.split(operator)
            yaml_version = operator + yaml_version
            yaml_map[yaml_package] = [yaml_version]
        
        # 否则，将依赖项作为 YAML 包名，并将版本号设为 None
        else:
            yaml_package, yaml_version = search_text.strip(), None
            yaml_map[yaml_package] = yaml_version
    
    # 返回构建好的 YAML 映射字典
    return yaml_map


# 清理 YAML 版本列表，根据指定的 toml_version 版本
def clean_version_list(
    yaml_versions: list[str], toml_version: version.Version
) -> list[str]:
    # 遍历 YAML 版本列表中的每个版本
    for i in range(len(yaml_versions)):
        # 去除版本号前后的空白字符
        yaml_version = yaml_versions[i].strip()
        
        # 获取版本号的操作符
        operator = get_operator_from(yaml_version)
        assert operator is not None
        
        # 根据操作符调整版本号的格式
        if "<=" in operator or ">=" in operator:
            yaml_version = yaml_version[2:]
        else:
            yaml_version = yaml_version[1:]
        
        # 解析版本号为 Version 对象
        yaml_version = version.parse(yaml_version)
        
        # 根据比较结果，更新版本列表中的版本号
        if yaml_version < toml_version:
            yaml_versions[i] = "-" + str(yaml_version)
        elif yaml_version >= toml_version:
            if ">" in operator:
                yaml_versions[i] = "-" + str(yaml_version)
    
    # 返回更新后的 YAML 版本列表
    return yaml_versions


# 将最小版本固定到 YAML 文件数据中
def pin_min_versions_to_yaml_file(
    yaml_map: dict[str, list[str] | None], toml_map: dict[str, str], yaml_file_data: str
) -> str:
    data = yaml_file_data
    for yaml_package, yaml_versions in yaml_map.items():
        # 遍历yaml_map中的每个包和其版本列表
        if yaml_package in EXCLUSION_LIST:
            # 如果包在排除列表中，则跳过当前循环，处理下一个包
            continue
        old_dep = yaml_package
        # 将当前包名存储为旧依赖，用于后续替换操作
        if yaml_versions is not None:
            # 如果版本列表不为空
            old_dep = old_dep + ", ".join(yaml_versions)
            # 将版本列表转换为字符串，与包名连接作为旧依赖
        if CONDA_TO_PIP.get(yaml_package, yaml_package) in toml_map:
            # 如果yaml_package在CONDA_TO_PIP映射中，则使用映射后的值作为键查找toml_map
            min_dep = toml_map[CONDA_TO_PIP.get(yaml_package, yaml_package)]
            # 设置最小依赖为找到的toml_map中的值
        elif yaml_package in toml_map:
            # 如果yaml_package直接在toml_map中
            min_dep = toml_map[yaml_package]
            # 设置最小依赖为toml_map中的值
        else:
            # 如果都没有找到匹配的最小依赖，则跳过当前循环，处理下一个包
            continue
        if yaml_versions is None:
            # 如果版本列表为空
            new_dep = old_dep + ">=" + min_dep
            # 设置新依赖为旧依赖加上最小版本限制
            data = data.replace(old_dep, new_dep, 1)
            # 在data中进行一次旧依赖到新依赖的替换，只替换第一次出现的旧依赖
            continue
        toml_version = version.parse(min_dep)
        # 解析最小依赖的版本号
        yaml_versions_list = clean_version_list(yaml_versions, toml_version)
        # 清理版本列表，只保留符合最小依赖版本要求的版本号
        cleaned_yaml_versions = [x for x in yaml_versions_list if "-" not in x]
        # 过滤掉包含"-"的版本号，生成干净的版本号列表
        new_dep = yaml_package
        # 设置新依赖为当前包名
        for clean_yaml_version in cleaned_yaml_versions:
            # 遍历清理后的版本号列表
            new_dep += clean_yaml_version + ", "
            # 将每个版本号连接到新依赖字符串末尾
        operator = get_operator_from(new_dep)
        # 获取新依赖中的操作符
        if operator != "=":
            # 如果操作符不是等号
            new_dep += ">=" + min_dep
            # 在新依赖末尾添加最小依赖的大于等于限制
        else:
            # 如果操作符是等号
            new_dep = new_dep[:-2]
            # 去除新依赖末尾的", "，表示精确版本限制
        data = data.replace(old_dep, new_dep)
        # 在data中进行一次旧依赖到新依赖的替换
    return data
# 定义一个函数，从代码中获取最小版本以进行 pandas 代码的检查
def get_versions_from_code() -> dict[str, str]:
    """Min versions for checking within pandas code."""
    # 获取安装映射
    install_map = _optional.INSTALL_MAPPING
    # 创建反向的安装映射，用来从值找到键
    inverse_install_map = {v: k for k, v in install_map.items()}
    # 获取版本字典
    versions = _optional.VERSIONS
    # 遍历需要排除的依赖项
    for item in EXCLUDE_DEPS:
        # 使用反向映射获取逆向的依赖项
        item = inverse_install_map.get(item, item)
        # 如果存在，则移除版本字典中的依赖项
        versions.pop(item, None)
    # 返回包含安装映射中键值的小写形式和版本的字典
    return {install_map.get(k, k).casefold(): v for k, v in versions.items()}


# 定义一个函数，从 CI 作业中获取测试所有可选依赖项的最小版本
def get_versions_from_ci(content: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Min versions in CI job for testing all optional dependencies."""
    # 不要使用 pyyaml 解析，因为它会忽略我们正在寻找的注释
    seen_required = False
    seen_optional = False
    seen_test = False
    required_deps = {}
    optional_deps = {}
    # 遍历内容列表中的每一行
    for line in content:
        if "# test dependencies" in line:
            seen_test = True
        elif seen_test and "- pytest>=" in line:
            # 只获取 pytest 的依赖项
            package, version = line.strip().split(">=")
            package = package[2:]  # 去除开头的 "- "
            optional_deps[package.casefold()] = version
        elif "# required dependencies" in line:
            seen_required = True
        elif "# optional dependencies" in line:
            seen_optional = True
        elif "- pip:" in line:
            continue
        elif seen_required and line.strip():
            if "==" in line:
                package, version = line.strip().split("==", maxsplit=1)
            else:
                package, version = line.strip().split("=", maxsplit=1)
            package = package.split()[-1]
            if package in EXCLUDE_DEPS:
                continue
            if not seen_optional:
                required_deps[package.casefold()] = version
            else:
                optional_deps[package.casefold()] = version
    # 返回必需依赖和可选依赖的字典元组
    return required_deps, optional_deps


# 定义一个函数，从 pyproject.toml 文件中获取 pip install pandas[extra] 的最小版本
def get_versions_from_toml() -> dict[str, str]:
    """Min versions in pyproject.toml for pip install pandas[extra]."""
    # 获取安装映射
    install_map = _optional.INSTALL_MAPPING
    optional_dependencies = {}
    # 打开 pyproject.toml 文件进行读取
    with open(SETUP_PATH, "rb") as pyproject_f:
        # 加载 pyproject.toml 文件
        pyproject_toml = tomllib.load(pyproject_f)
        # 获取可选依赖项
        opt_deps = pyproject_toml["project"]["optional-dependencies"]
        dependencies = set(opt_deps["all"])

        # 移除 pytest 插件依赖项
        pytest_plugins = {dep for dep in opt_deps["test"] if dep.startswith("pytest-")}
        dependencies = dependencies.difference(pytest_plugins)

    # 遍历依赖项集合
    for dependency in dependencies:
        # 分割依赖项和版本号
        package, version = dependency.strip().split(">=")
        optional_dependencies[install_map.get(package, package).casefold()] = version

    # 移除需要排除的依赖项
    for item in EXCLUDE_DEPS:
        optional_dependencies.pop(item, None)
    # 返回可选依赖项的字典
    return optional_dependencies


# 主函数，返回整数结果
def main() -> int:
    ret = 0
    # 使用函数 pin_min_versions_to_ci_deps() 的返回结果更新 ret
    ret |= pin_min_versions_to_ci_deps()
    # 打开 CI_PATH 文件进行读取，获取必需依赖和可选依赖的字典
    with open(CI_PATH, encoding="utf-8") as f:
        _, ci_optional = get_versions_from_ci(f.readlines())
    # 从代码中获取版本信息
    code_optional = get_versions_from_code()
    # 从 TOML 文件中获取版本信息
    setup_optional = get_versions_from_toml()

    # 计算三个版本信息集合的对称差集，即出现在任何一个集合中但不同时出现在所有集合中的元素
    diff = (ci_optional.items() | code_optional.items() | setup_optional.items()) - (
        ci_optional.items() & code_optional.items() & setup_optional.items()
    )

    # 如果存在版本差异
    if diff:
        # 提取出现在差异集合中的包名
        packages = {package for package, _ in diff}
        out = sys.stdout
        # 输出版本差异的提示信息，包括路径变量的值
        out.write(
            f"The follow minimum version differences were found between  "
            f"{CI_PATH}, {CODE_PATH} AND {SETUP_PATH}. "
            f"Please ensure these are aligned: \n\n"
        )

        # 遍历每个出现在差异集合中的包名，输出其在各个路径变量指定位置的版本信息
        for package in packages:
            out.write(
                f"{package}\n"
                f"{CI_PATH}: {ci_optional.get(package, 'Not specified')}\n"
                f"{CODE_PATH}: {code_optional.get(package, 'Not specified')}\n"
                f"{SETUP_PATH}: {setup_optional.get(package, 'Not specified')}\n\n"
            )
        # 设置返回值的位或运算结果，用于指示发现了版本不一致
        ret |= 1
    # 返回最终的返回值
    return ret
# 如果当前脚本作为主程序执行（而不是被导入到其他模块），则执行以下代码块
if __name__ == "__main__":
    # 调用名为 main() 的函数，并退出脚本，返回其退出状态码
    sys.exit(main())
```