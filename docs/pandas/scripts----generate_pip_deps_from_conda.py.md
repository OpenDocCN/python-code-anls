# `D:\src\scipysrc\pandas\scripts\generate_pip_deps_from_conda.py`

```
# 解释脚本文件头部，指定使用 Python3 环境运行
#!/usr/bin/env python3

"""
脚本目的是将 conda 的 environment.yml 转换为 pip 的 requirements-dev.txt，
或者检查它们是否具有相同的软件包（用于持续集成）。

用法：
生成 `requirements-dev.txt`：
$ python scripts/generate_pip_deps_from_conda.py

比较并在未生成 `requirements-dev.txt` 时失败（退出状态不等于0）：
$ python scripts/generate_pip_deps_from_conda.py --compare
"""

# 导入必要的库
import argparse  # 用于解析命令行参数
import pathlib   # 提供处理文件路径的类和方法
import re        # 用于正则表达式操作
import sys       # 提供对解释器相关配置的访问

# 根据解释器版本导入不同的库
if sys.version_info >= (3, 11):
    import tomllib   # 版本大于等于 3.11 使用 tomllib
else:
    import tomli as tomllib   # 版本小于 3.11 使用 tomli

import yaml   # 用于处理 YAML 格式文件的库

# 需要排除的 conda 软件包
EXCLUDE = {"python", "c-compiler", "cxx-compiler"}

# 部分软件包需要重命名的映射表
REMAP_VERSION = {"tzdata": "2022.7"}

# 部分 conda 软件包到 pip 软件包的转换映射表
CONDA_TO_PIP = {
    "pytables": "tables",
    "psycopg2": "psycopg2-binary",
    "dask-core": "dask",
    "seaborn-base": "seaborn",
    "sqlalchemy": "SQLAlchemy",
    "pyqt": "PyQt5",
}

def conda_package_to_pip(package: str):
    """
    将 conda 的软件包名称转换为对应的 pip 软件包名称。

    大多数情况下它们相同，以下是特例：
    - 需要排除的软件包（在 `EXCLUDE` 中）
    - 需要重命名的软件包（在 `CONDA_TO_PIP` 中）
    - 在 conda 中使用单个等号指定特定版本（例如 `pandas=1.0`），
      在 pip 中使用两个等号（例如 `pandas==1.0`）
    """
    package = re.sub("(?<=[^<>~])=", "==", package).strip()

    for compare in ("<=", ">=", "=="):
        if compare in package:
            pkg, version = package.split(compare)
            if pkg in EXCLUDE:
                return
            if pkg in REMAP_VERSION:
                return "".join((pkg, compare, REMAP_VERSION[pkg]))
            if pkg in CONDA_TO_PIP:
                return "".join((CONDA_TO_PIP[pkg], compare, version))

    if package in EXCLUDE:
        return

    if package in CONDA_TO_PIP:
        return CONDA_TO_PIP[package]

    return package

def generate_pip_from_conda(
    conda_path: pathlib.Path, pip_path: pathlib.Path, compare: bool = False
) -> bool:
    """
    从 conda 文件生成 pip 依赖文件，或者比较生成的文件与 conda 文件是否同步（``compare=True``）。

    Parameters
    ----------
    conda_path : pathlib.Path
        conda 文件的路径（例如 `environment.yml`）。
    pip_path : pathlib.Path
        pip 文件的路径（例如 `requirements-dev.txt`）。
    compare : bool, 默认为 False
        是否生成 pip 文件（``False``），或者比较生成的 pip 文件与 conda 文件的最新版本（``True``）。

    Returns
    -------
    bool
        如果比较失败返回 True，否则返回 False。
    """
    with conda_path.open() as file:
        deps = yaml.safe_load(file)["dependencies"]

    pip_deps = []
    # 遍历依赖列表 deps
    for dep in deps:
        # 如果依赖是字符串类型，将其转换为 pip 格式，并添加到 pip_deps 列表中
        if isinstance(dep, str):
            conda_dep = conda_package_to_pip(dep)
            # 如果成功转换为 pip 格式，则添加到 pip_deps 列表中
            if conda_dep:
                pip_deps.append(conda_dep)
        # 如果依赖是字典类型且只有一个键 "pip"，则将其值（列表）扩展到 pip_deps 列表中
        elif isinstance(dep, dict) and len(dep) == 1 and "pip" in dep:
            pip_deps.extend(dep["pip"])
        else:
            # 若依赖既不是字符串，也不是符合预期的字典类型，则引发 ValueError 异常
            raise ValueError(f"Unexpected dependency {dep}")

    # 构建文件头部信息，包含自动生成的文件名和注释
    header = (
        f"# This file is auto-generated from {conda_path.name}, do not modify.\n"
        "# See that file for comments about the need/usage of each dependency.\n\n"
    )
    # 将 header 和 pip_deps 列表中的内容拼接成完整的 pip 内容字符串
    pip_content = header + "\n".join(pip_deps) + "\n"

    # 向 requirements-dev.txt 添加 setuptools
    with open(pathlib.Path(conda_path.parent, "pyproject.toml"), "rb") as fd:
        # 从 pyproject.toml 文件中加载元数据
        meta = tomllib.load(fd)
    # 遍历构建系统的依赖列表，如果发现 setuptools，则添加到 pip_content 中
    for requirement in meta["build-system"]["requires"]:
        if "setuptools" in requirement:
            pip_content += requirement
            pip_content += "\n"

    # 如果需要进行比较，则打开 pip 文件并比较内容
    if compare:
        with pip_path.open() as file:
            return pip_content != file.read()

    # 否则，将生成的 pip_content 写入 pip 文件中
    with pip_path.open("w") as file:
        file.write(pip_content)
    # 返回 False 表示文件未更改
    return False
if __name__ == "__main__":
    # 如果当前脚本被直接执行，则执行以下代码块

    argparser = argparse.ArgumentParser(
        description="convert (or compare) conda file to pip"
    )
    # 创建参数解析器对象，描述为将conda文件转换（或比较）为pip文件

    argparser.add_argument(
        "--compare",
        action="store_true",
        help="compare whether the two files are equivalent",
    )
    # 添加一个名为--compare的可选参数，如果存在则设定为True，用于指示是否进行文件比较

    args = argparser.parse_args()
    # 解析命令行参数并存储在args变量中

    conda_fname = "environment.yml"
    # 定义conda文件名为environment.yml

    pip_fname = "requirements-dev.txt"
    # 定义pip文件名为requirements-dev.txt

    repo_path = pathlib.Path(__file__).parent.parent.absolute()
    # 获取当前脚本文件的父目录的父目录的绝对路径，即repo_path为当前项目的根目录路径

    res = generate_pip_from_conda(
        pathlib.Path(repo_path, conda_fname),
        pathlib.Path(repo_path, pip_fname),
        compare=args.compare,
    )
    # 调用generate_pip_from_conda函数，传入conda文件路径、pip文件路径以及是否进行比较的参数，将结果存储在res变量中

    if res:
        msg = (
            f"`{pip_fname}` has to be generated with `{__file__}` after "
            f"`{conda_fname}` is modified.\n"
        )
        # 如果res为True，生成提示消息，说明需在修改conda文件后使用当前文件生成pip文件

        sys.stderr.write(msg)
        # 将提示消息写入标准错误流

    sys.exit(res)
    # 使用res的值退出程序，0表示成功，1表示失败
```