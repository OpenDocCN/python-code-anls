# `D:\src\scipysrc\scikit-learn\build_tools\update_environments_and_lock_files.py`

```
"""Script to update CI environment files and associated lock files.

To run it you need to be in the root folder of the scikit-learn repo:
python build_tools/update_environments_and_lock_files.py

Two scenarios where this script can be useful:
- make sure that the latest versions of all the dependencies are used in the CI.
  There is a scheduled workflow that does this, see
  .github/workflows/update-lock-files.yml. This is still useful to run this
  script when when the automated PR fails and for example some packages need to
  be pinned. You can add the pins to this script, run it, and open a PR with
  the changes.
- bump minimum dependencies in sklearn/_min_dependencies.py. Running this
  script will update both the CI environment files and associated lock files.
  You can then open a PR with the changes.
- pin some packages to an older version by adding them to the
  default_package_constraints variable. This is useful when regressions are
  introduced in our dependencies, this has happened for example with pytest 7
  and coverage 6.3.

Environments are conda environment.yml or pip requirements.txt. Lock files are
conda-lock lock files or pip-compile requirements.txt.

pip requirements.txt are used when we install some dependencies (e.g. numpy and
scipy) with apt-get and the rest of the dependencies (e.g. pytest and joblib)
with pip.

To run this script you need:
- conda-lock. The version should match the one used in the CI in
  sklearn/_min_dependencies.py
- pip-tools

To only update the environment and lock files for specific builds, you can use
the command line argument `--select-build` which will take a regex. For example,
to only update the documentation builds you can use:
`python build_tools/update_environments_and_lock_files.py --select-build doc`
"""

import json  # 导入处理 JSON 格式数据的模块
import logging  # 导入日志记录模块
import re  # 导入正则表达式模块
import subprocess  # 导入执行外部命令的模块
import sys  # 导入系统相关功能的模块
from importlib.metadata import version  # 导入用于获取包版本的函数
from pathlib import Path  # 导入处理路径的模块

import click  # 导入用于创建命令行界面的模块
from jinja2 import Environment  # 导入用于处理模板的模块
from packaging.version import Version  # 导入处理版本号的模块

logger = logging.getLogger(__name__)  # 获取当前脚本的日志记录器
logger.setLevel(logging.INFO)  # 设置日志级别为 INFO
handler = logging.StreamHandler()  # 创建一个输出到控制台的日志处理器
logger.addHandler(handler)  # 将日志处理器添加到日志记录器中

TRACE = logging.DEBUG - 5  # 设置 TRACE 级别的日志信息

common_dependencies_without_coverage = [  # 定义不包含测试覆盖率的常见依赖项列表
    "python",
    "numpy",
    "blas",
    "scipy",
    "cython",
    "joblib",
    "threadpoolctl",
    "matplotlib",
    "pandas",
    "pyamg",
    "pytest",
    "pytest-xdist",
    "pillow",
    "pip",
    "ninja",
    "meson-python",
]

common_dependencies = common_dependencies_without_coverage + [  # 将测试覆盖率相关的依赖项加入到常见依赖项列表中
    "pytest-cov",
    "coverage",
]

docstring_test_dependencies = ["sphinx", "numpydoc"]  # 文档测试所需的依赖项列表

default_package_constraints = {}  # 默认的包约束为空字典

def remove_from(alist, to_remove):
    return [each for each in alist if each not in to_remove]

build_metadata_list = [
    {
        "name": "pylatest_conda_forge_cuda_array-api_linux-64",
        # 定义软件包的名称
        "type": "conda",
        # 指定包类型为 conda
        "tag": "cuda",
        # 标记为 CUDA 版本
        "folder": "build_tools/github",
        # 指定构建文件夹路径
        "platform": "linux-64",
        # 指定目标平台为 Linux 64 位
        "channels": ["conda-forge", "pytorch", "nvidia"],
        # 指定使用的软件源频道列表
        "conda_dependencies": common_dependencies
        + [
            "ccache",
            # 添加 ccache 作为依赖项
            # 确保 pytorch 来自 pytorch 频道而非 conda-forge
            "pytorch::pytorch",
            "pytorch-cuda",
            "polars",
            "pyarrow",
            "cupy",
            "array-api-compat",
            "array-api-strict",
        ],
    },
    {
        "name": "pylatest_conda_forge_mkl_linux-64",
        # 定义软件包的名称
        "type": "conda",
        # 指定包类型为 conda
        "tag": "main-ci",
        # 标记为主要的持续集成版本
        "folder": "build_tools/azure",
        # 指定构建文件夹路径
        "platform": "linux-64",
        # 指定目标平台为 Linux 64 位
        "channels": ["conda-forge"],
        # 指定使用的软件源频道列表
        "conda_dependencies": common_dependencies
        + [
            "ccache",
            "pytorch",
            "pytorch-cpu",
            "polars",
            "pyarrow",
            "array-api-compat",
            "array-api-strict",
        ],
        "package_constraints": {
            "blas": "[build=mkl]",
            # 设置 BLAS 库的约束为使用 MKL 版本
            "pytorch": "1.13",
            # 设置 pytorch 的版本要求为 1.13
        },
    },
    {
        "name": "pylatest_conda_forge_mkl_osx-64",
        # 定义软件包的名称
        "type": "conda",
        # 指定包类型为 conda
        "tag": "main-ci",
        # 标记为主要的持续集成版本
        "folder": "build_tools/azure",
        # 指定构建文件夹路径
        "platform": "osx-64",
        # 指定目标平台为 macOS 64 位
        "channels": ["conda-forge"],
        # 指定使用的软件源频道列表
        "conda_dependencies": common_dependencies
        + [
            "ccache",
            "compilers",
            "llvm-openmp",
        ],
        "package_constraints": {
            "blas": "[build=mkl]",
            # 设置 BLAS 库的约束为使用 MKL 版本
        },
    },
    {
        "name": "pylatest_conda_mkl_no_openmp",
        # 定义软件包的名称
        "type": "conda",
        # 指定包类型为 conda
        "tag": "main-ci",
        # 标记为主要的持续集成版本
        "folder": "build_tools/azure",
        # 指定构建文件夹路径
        "platform": "osx-64",
        # 指定目标平台为 macOS 64 位
        "channels": ["defaults"],
        # 指定使用的软件源频道列表
        "conda_dependencies": remove_from(
            common_dependencies, ["cython", "threadpoolctl", "meson-python"]
        )
        + ["ccache"],
        "package_constraints": {
            "blas": "[build=mkl]",
            # 设置 BLAS 库的约束为使用 MKL 版本
            # scipy 1.12.x 在此平台上崩溃（https://github.com/scipy/scipy/pull/20086）
            # TODO: 当 1.13 版本在 "defaults" 频道中可用时，释放 scipy 约束
            "scipy": "<1.12",
        },
        # TODO: 当所需版本在主要频道中可用时，将 cython、threadpoolctl 和 meson-python
        # 放回 conda 依赖项中
        "pip_dependencies": ["cython", "threadpoolctl", "meson-python"],
    },
    {
        "name": "pymin_conda_defaults_openblas",
        "type": "conda",
        "tag": "main-ci",
        "folder": "build_tools/azure",
        "platform": "linux-64",
        "channels": ["defaults"],
        "conda_dependencies": remove_from(
            common_dependencies,  # 从通用依赖中移除以下项目
            ["pandas", "threadpoolctl", "pip", "ninja", "meson-python"],  # 移除的项目列表
        )
        + ["ccache"],  # 添加额外的依赖项目
        "package_constraints": {
            "python": "3.9",  # Python 版本要求为 3.9
            "blas": "[build=openblas]",  # BLAS 库的版本约束
            "numpy": "1.21",  # NumPy 的最小版本，无法在 defaults 渠道上找到
            "scipy": "1.7",  # SciPy 的版本约束，某些低级崩溃问题
            "matplotlib": "min",  # Matplotlib 的最小版本
            "cython": "min",  # Cython 的最小版本
            "joblib": "min",  # Joblib 的最小版本
            "threadpoolctl": "min",  # threadpoolctl 的最小版本
        },
        # TODO: 当所需版本在 defaults 渠道上可用时，将 pip 依赖项放回 conda 依赖项中
        "pip_dependencies": ["threadpoolctl"],  # pip 依赖项列表
    },
    {
        "name": "pymin_conda_forge_openblas_ubuntu_2204",
        "type": "conda",
        "tag": "main-ci",
        "folder": "build_tools/azure",
        "platform": "linux-64",
        "channels": ["conda-forge"],
        "conda_dependencies": (
            common_dependencies_without_coverage  # 不包括代码覆盖率的通用依赖项
            + docstring_test_dependencies  # 文档字符串测试的依赖项
            + ["ccache"]  # 添加额外的依赖项目
        ),
        "package_constraints": {
            "python": "3.9",  # Python 版本要求为 3.9
            "blas": "[build=openblas]",  # BLAS 库的版本约束
        },
    },
    {
        "name": "pylatest_pip_openblas_pandas",
        "type": "conda",
        "tag": "main-ci",
        "folder": "build_tools/azure",
        "platform": "linux-64",
        "channels": ["defaults"],
        "conda_dependencies": ["python", "ccache"],  # conda 依赖项列表
        "pip_dependencies": (
            remove_from(common_dependencies, ["python", "blas", "pip"])  # 从通用依赖中移除指定项目
            + docstring_test_dependencies  # 文档字符串测试的依赖项
            + ["lightgbm", "scikit-image"]  # 添加额外的 pip 依赖项目
        ),
        "package_constraints": {
            "python": "3.9",  # Python 版本要求为 3.9
        },
    },
    {
        "name": "pylatest_pip_scipy_dev",
        "type": "conda",
        "tag": "scipy-dev",
        "folder": "build_tools/azure",
        "platform": "linux-64",
        "channels": ["defaults"],
        "conda_dependencies": ["python", "ccache"],
        "pip_dependencies": (
            remove_from(
                common_dependencies,
                [
                    "python",
                    "blas",
                    "matplotlib",
                    "pyamg",
                    # 所有以下依赖项在 CI 中已安装开发版本，因此可以从环境.yml中移除
                    # 从环境.yml中移除以下所有依赖项，因为它们在 CI 中已安装开发版本
                    "numpy",
                    "scipy",
                    "pandas",
                    "cython",
                    "joblib",
                    "pillow",
                ],
            )
            + ["pooch"]
            + docstring_test_dependencies
            # python-dateutil 是 pandas 的依赖项，而 pandas 已从环境.yml中移除
            # 添加 python-dateutil 以确保其版本被固定
            + ["python-dateutil"]
        ),
    },
    {
        "name": "pymin_conda_forge_mkl",
        "type": "conda",
        "tag": "main-ci",
        "folder": "build_tools/azure",
        "platform": "win-64",
        "channels": ["conda-forge"],
        "conda_dependencies": remove_from(common_dependencies, ["pandas", "pyamg"])
        + [
            "wheel",
            "pip",
        ],
        "package_constraints": {
            "python": "3.9",
            "blas": "[build=mkl]",
        },
    },
    {
        "name": "doc_min_dependencies",
        "type": "conda",
        "tag": "main-ci",
        "folder": "build_tools/circle",
        "platform": "linux-64",
        "channels": ["conda-forge"],
        "conda_dependencies": common_dependencies_without_coverage
        + [
            "scikit-image",
            "seaborn",
            "memory_profiler",
            "compilers",
            "sphinx",
            "sphinx-gallery",
            "sphinx-copybutton",
            "numpydoc",
            "sphinx-prompt",
            "plotly",
            "polars",
            "pooch",
            "sphinx-remove-toctrees",
            "sphinx-design",
            "pydata-sphinx-theme",
        ],
        "pip_dependencies": [
            "sphinxext-opengraph",
            "sphinxcontrib-sass",
        ],
        "package_constraints": {
            "python": "3.9",
            "numpy": "min",
            "scipy": "min",
            "matplotlib": "min",
            "cython": "min",
            "scikit-image": "min",
            "sphinx": "min",
            "pandas": "min",
            "sphinx-gallery": "min",
            "sphinx-copybutton": "min",
            "numpydoc": "min",
            "sphinx-prompt": "min",
            "sphinxext-opengraph": "min",
            "plotly": "min",
            "polars": "min",
            "pooch": "min",
            "sphinx-design": "min",
            "sphinxcontrib-sass": "min",
            "sphinx-remove-toctrees": "min",
            "pydata-sphinx-theme": "min",
        }
    }
    
    
    注释：
    
    {
        "name": "doc",
        "type": "conda",
        "tag": "main-ci",
        "folder": "build_tools/circle",
        "platform": "linux-64",
        "channels": ["conda-forge"],
        "conda_dependencies": common_dependencies_without_coverage
        + [
            "scikit-image",
            "seaborn",
            "memory_profiler",
            "compilers",
            "sphinx",
            "sphinx-gallery",
            "sphinx-copybutton",
            "numpydoc",
            "sphinx-prompt",
            "plotly",
            "polars",
            "pooch",
            "sphinxext-opengraph",
            "sphinx-remove-toctrees",
            "sphinx-design",
            "pydata-sphinx-theme",
        ],
        "pip_dependencies": [
            "jupyterlite-sphinx",
            "jupyterlite-pyodide-kernel",
            "sphinxcontrib-sass",
        ],
        "package_constraints": {
            "python": "3.9",
        }
    }
    
    
    注释：
    
    {
        "name": "pymin_conda_forge",
        "type": "conda",
        "tag": "arm",
        "folder": "build_tools/cirrus",
        "platform": "linux-aarch64",
        "channels": ["conda-forge"],
        "conda_dependencies": remove_from(
            common_dependencies_without_coverage, ["pandas", "pyamg"]
        )
        + ["pip", "ccache"],
        "package_constraints": {
            "python": "3.9",
        }
    }
    {
        "name": "debian_atlas_32bit",
        # 软件包名称
        "type": "pip",
        # 类型为 Python 包管理器 pip
        "tag": "main-ci",
        # 标签为主要持续集成
        "folder": "build_tools/azure",
        # 文件夹路径为 build_tools/azure
        "pip_dependencies": [
            "cython",            # 依赖项：cython
            "joblib",            # 依赖项：joblib
            "threadpoolctl",     # 依赖项：threadpoolctl
            "pytest",            # 依赖项：pytest
            "pytest-cov",        # 依赖项：pytest-cov
            "ninja",             # 依赖项：ninja
            "meson-python",      # 依赖项：meson-python
        ],
        "package_constraints": {
            "joblib": "min",         # 对 joblib 的版本限制为最小版本
            "threadpoolctl": "3.1.0", # 对 threadpoolctl 的版本限制为 3.1.0
            "pytest": "min",         # 对 pytest 的版本限制为最小版本
            "pytest-cov": "min",     # 对 pytest-cov 的版本限制为最小版本
            # 由于在32位系统上出现问题，不包括 pytest-xdist
            "cython": "min",         # 对 cython 的版本限制为最小版本
        },
        # Python 版本与 debian-32 构建中的相同
        "python_version": "3.9.2",
    },
    {
        "name": "ubuntu_atlas",
        # 软件包名称
        "type": "pip",
        # 类型为 Python 包管理器 pip
        "tag": "main-ci",
        # 标签为主要持续集成
        "folder": "build_tools/azure",
        # 文件夹路径为 build_tools/azure
        "pip_dependencies": [
            "cython",            # 依赖项：cython
            "joblib",            # 依赖项：joblib
            "threadpoolctl",     # 依赖项：threadpoolctl
            "pytest",            # 依赖项：pytest
            "pytest-xdist",      # 依赖项：pytest-xdist
            "ninja",             # 依赖项：ninja
            "meson-python",      # 依赖项：meson-python
        ],
        "package_constraints": {
            "joblib": "min",         # 对 joblib 的版本限制为最小版本
            "threadpoolctl": "min",  # 对 threadpoolctl 的版本限制为最小版本
            "cython": "min",         # 对 cython 的版本限制为最小版本
        },
        # Python 版本为 3.10.4
        "python_version": "3.10.4",
    },
# 执行给定的命令列表，并记录到调试日志中
def execute_command(command_list):
    logger.debug(" ".join(command_list))
    # 启动一个子进程来执行指定的命令列表，捕获其标准输出和标准错误输出
    proc = subprocess.Popen(
        command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    out, err = proc.communicate()
    out, err = out.decode(errors="replace"), err.decode(errors="replace")

    # 如果子进程返回非零退出码，抛出运行时错误，显示相关信息
    if proc.returncode != 0:
        command_str = " ".join(command_list)
        raise RuntimeError(
            "Command exited with non-zero exit code.\n"
            "Exit code: {}\n"
            "Command:\n{}\n"
            "stdout:\n{}\n"
            "stderr:\n{}\n".format(proc.returncode, command_str, out, err)
        )
    # 记录标准输出到跟踪日志
    logger.log(TRACE, out)
    return out


# 获取指定包名及其约束条件，根据构建元数据获取包约束，可选择使用 pip
def get_package_with_constraint(package_name, build_metadata, uses_pip=False):
    # 获取构建元数据中的包约束
    build_package_constraints = build_metadata.get("package_constraints")
    if build_package_constraints is None:
        constraint = None
    else:
        constraint = build_package_constraints.get(package_name)

    # 如果构建元数据中未指定该包的约束条件，则使用默认约束条件
    constraint = constraint or default_package_constraints.get(package_name)

    # 如果没有约束条件，直接返回包名
    if constraint is None:
        return package_name

    comment = ""
    # 如果约束条件为 "min"，则执行相应命令获取最小依赖版本，并添加注释
    if constraint == "min":
        constraint = execute_command(
            [sys.executable, "sklearn/_min_dependencies.py", package_name]
        ).strip()
        comment = "  # min"

    # 如果约束条件为数字开头的版本号，根据 uses_pip 是否使用 pip 决定添加的等号类型
    if re.match(r"\d[.\d]*", constraint):
        equality = "==" if uses_pip else "="
        constraint = equality + constraint

    # 返回包名及其约束条件（带有注释）
    return f"{package_name}{constraint}{comment}"


# 创建 Jinja2 环境对象，用于生成 Conda 环境文件内容
environment = Environment(trim_blocks=True, lstrip_blocks=True)
# 注册自定义过滤器函数以在模板中使用
environment.filters["get_package_with_constraint"] = get_package_with_constraint


# 根据构建元数据生成 Conda 环境文件内容的字符串表示
def get_conda_environment_content(build_metadata):
    template = environment.from_string(
        """
# DO NOT EDIT: this file is generated from the specification found in the
# following script to centralize the configuration for CI builds:
# build_tools/update_environments_and_lock_files.py
channels:
  {% for channel in build_metadata['channels'] %}
  - {{ channel }}
  {% endfor %}
dependencies:
  {% for conda_dep in build_metadata['conda_dependencies'] %}
  - {{ conda_dep | get_package_with_constraint(build_metadata) }}
  {% endfor %}
  {% if build_metadata['pip_dependencies'] %}
  - pip
  - pip:
  {% for pip_dep in build_metadata.get('pip_dependencies', []) %}
    - {{ pip_dep | get_package_with_constraint(build_metadata, uses_pip=True) }}
  {% endfor %}
  {% endif %}""".strip()
    )
    return template.render(build_metadata=build_metadata)


# 根据构建元数据生成 Conda 环境文件并写入到文件中
def write_conda_environment(build_metadata):
    # 获取 Conda 环境文件内容字符串
    content = get_conda_environment_content(build_metadata)
    build_name = build_metadata["name"]
    folder_path = Path(build_metadata["folder"])
    # 指定输出文件路径
    output_path = folder_path / f"{build_name}_environment.yml"
    logger.debug(output_path)
    # 将生成的内容写入到文件中
    output_path.write_text(content)


# 根据构建元数据列表生成所有 Conda 环境文件并写入到文件中
def write_all_conda_environments(build_metadata_list):
    for build_metadata in build_metadata_list:
        write_conda_environment(build_metadata)
# 使用 conda-lock 工具生成指定环境的 Conda 锁文件
def conda_lock(environment_path, lock_file_path, platform):
    # 执行 conda-lock 命令来锁定环境依赖
    execute_command(
        [
            "conda-lock",
            "lock",
            "--mamba",
            "--kind",
            "explicit",
            "--platform",
            platform,
            "--file",
            str(environment_path),
            "--filename-template",
            str(lock_file_path),
        ]
    )


# 根据构建元数据创建 Conda 锁文件
def create_conda_lock_file(build_metadata):
    # 获取构建名称和文件夹路径
    build_name = build_metadata["name"]
    folder_path = Path(build_metadata["folder"])
    # 构建环境文件路径
    environment_path = folder_path / f"{build_name}_environment.yml"
    platform = build_metadata["platform"]
    # 根据平台确定锁文件基础名
    lock_file_basename = build_name
    if not lock_file_basename.endswith(platform):
        lock_file_basename = f"{lock_file_basename}_{platform}"
    # 构建 Conda 锁文件路径
    lock_file_path = folder_path / f"{lock_file_basename}_conda.lock"
    # 调用 conda_lock 函数生成 Conda 锁文件
    conda_lock(environment_path, lock_file_path, platform)


# 遍历构建元数据列表，为每个构建生成 Conda 锁文件
def write_all_conda_lock_files(build_metadata_list):
    for build_metadata in build_metadata_list:
        # 记录信息：锁定依赖项
        logger.info(f"# Locking dependencies for {build_metadata['name']}")
        # 创建 Conda 锁文件
        create_conda_lock_file(build_metadata)


# 根据构建元数据获取 Pip 依赖内容模板
def get_pip_requirements_content(build_metadata):
    # 使用 Jinja2 模板生成 Pip 依赖内容
    template = environment.from_string(
        """
# DO NOT EDIT: this file is generated from the specification found in the
# following script to centralize the configuration for CI builds:
# build_tools/update_environments_and_lock_files.py
{% for pip_dep in build_metadata['pip_dependencies'] %}
{{ pip_dep | get_package_with_constraint(build_metadata, uses_pip=True) }}
{% endfor %}""".strip()
    )
    return template.render(build_metadata=build_metadata)


# 根据构建元数据生成 Pip 依赖文件
def write_pip_requirements(build_metadata):
    # 获取构建名称和 Pip 依赖内容
    build_name = build_metadata["name"]
    content = get_pip_requirements_content(build_metadata)
    folder_path = Path(build_metadata["folder"])
    # 构建输出路径
    output_path = folder_path / f"{build_name}_requirements.txt"
    # 记录调试信息：输出路径
    logger.debug(output_path)
    # 写入 Pip 依赖内容到文件
    output_path.write_text(content)


# 遍历构建元数据列表，为每个构建生成 Pip 依赖文件
def write_all_pip_requirements(build_metadata_list):
    for build_metadata in build_metadata_list:
        # 创建 Pip 依赖文件
        write_pip_requirements(build_metadata)


# 使用 pip-compile 工具生成 Pip 锁文件
def pip_compile(pip_compile_path, requirements_path, lock_file_path):
    # 执行 pip-compile 命令来锁定 Pip 依赖
    execute_command(
        [
            str(pip_compile_path),
            "--upgrade",
            str(requirements_path),
            "-o",
            str(lock_file_path),
        ]
    )


# 根据构建元数据生成 Pip 锁文件
def write_pip_lock_file(build_metadata):
    build_name = build_metadata["name"]
    python_version = build_metadata["python_version"]
    environment_name = f"pip-tools-python{python_version}"
    # 为确保生成 Pip 锁文件的 Python 版本与使用锁文件的 CI 构建中的版本一致，
    # 首先创建一个包含正确 Python 版本的 Conda 环境，并在此环境中运行 pip-compile
    # 执行命令，使用指定的 Python 版本创建 conda 环境，并安装 pip-tools
    execute_command(
        [
            "conda",                            # 使用 conda 包管理器
            "create",                           # 创建新环境
            "-c",                               # 指定 conda 源为 conda-forge
            "conda-forge",                      
            "-n",                               # 指定环境名称
            f"pip-tools-python{python_version}", # 使用特定 Python 版本命名环境
            f"python={python_version}",          # 安装指定版本的 Python
            "pip-tools",                        # 安装 pip-tools
            "-y",                               # 自动确认安装
        ]
    )
    
    # 执行命令获取当前 conda 环境信息的 JSON 输出
    json_output = execute_command(["conda", "info", "--json"])
    
    # 将 JSON 输出解析为 Python 对象
    conda_info = json.loads(json_output)
    
    # 在 conda 环境路径列表中找到以指定环境名称结尾的环境路径
    environment_folder = [
        each for each in conda_info["envs"] if each.endswith(environment_name)
    ][0]
    
    # 将环境路径字符串转换为 Path 对象
    environment_path = Path(environment_folder)
    
    # 拼接路径，获取 pip-compile 的完整路径
    pip_compile_path = environment_path / "bin" / "pip-compile"
    
    # 从构建元数据中获取文件夹路径，并转换为 Path 对象
    folder_path = Path(build_metadata["folder"])
    
    # 根据构建名称拼接出 requirements.txt 文件的路径
    requirement_path = folder_path / f"{build_name}_requirements.txt"
    
    # 根据构建名称拼接出 lock.txt 文件的路径
    lock_file_path = folder_path / f"{build_name}_lock.txt"
    
    # 调用 pip-compile 函数，生成 requirements.txt 文件和 lock.txt 文件
    pip_compile(pip_compile_path, requirement_path, lock_file_path)
def write_all_pip_lock_files(build_metadata_list):
    # 遍历构建元数据列表，为每个构建元数据记录依赖信息
    for build_metadata in build_metadata_list:
        # 记录日志，指示正在处理哪个构建的依赖
        logger.info(f"# Locking dependencies for {build_metadata['name']}")
        # 调用函数写入该构建的依赖信息到pip lock文件
        write_pip_lock_file(build_metadata)


def check_conda_lock_version():
    # 检查已安装的 conda-lock 版本是否与 _min_dependencies 中的版本一致
    expected_conda_lock_version = execute_command(
        [sys.executable, "sklearn/_min_dependencies.py", "conda-lock"]
    ).strip()

    installed_conda_lock_version = version("conda-lock")
    # 如果安装的 conda-lock 版本与预期版本不一致，则抛出运行时错误
    if installed_conda_lock_version != expected_conda_lock_version:
        raise RuntimeError(
            f"Expected conda-lock version: {expected_conda_lock_version}, got:"
            f" {installed_conda_lock_version}"
        )


def check_conda_version():
    # 检查 conda 版本以避免与特定问题相关的虚拟包问题
    conda_info_output = execute_command(["conda", "info", "--json"])

    conda_info = json.loads(conda_info_output)
    conda_version = Version(conda_info["conda_version"])

    # 如果 conda 版本在 22.9.0 到 23.7 之间，则抛出运行时错误
    if Version("22.9.0") < conda_version < Version("23.7"):
        raise RuntimeError(
            f"conda version should be <= 22.9.0 or >= 23.7 got: {conda_version}"
        )


@click.command()
@click.option(
    "--select-build",
    default="",
    help=(
        "Regex to filter the builds we want to update environment and lock files. By"
        " default all the builds are selected."
    ),
)
@click.option(
    "--skip-build",
    default=None,
    help="Regex to skip some builds from the builds selected by --select-build",
)
@click.option(
    "--select-tag",
    default=None,
    help=(
        "Tag to filter the builds, e.g. 'main-ci' or 'scipy-dev'. "
        "This is an additional filtering on top of --select-build."
    ),
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Print commands executed by the script",
)
@click.option(
    "-vv",
    "--very-verbose",
    is_flag=True,
    help="Print output of commands executed by the script",
)
def main(select_build, skip_build, select_tag, verbose, very_verbose):
    if verbose:
        # 如果指定了 verbose 标志，则设置日志级别为 DEBUG
        logger.setLevel(logging.DEBUG)
    if very_verbose:
        # 如果指定了 very_verbose 标志，则设置日志级别为 TRACE
        logger.setLevel(TRACE)
        handler.setLevel(TRACE)
    
    # 检查 conda-lock 版本是否符合预期
    check_conda_lock_version()
    # 检查 conda 版本是否符合要求
    check_conda_version()

    # 根据选择条件过滤构建元数据列表
    filtered_build_metadata_list = [
        each for each in build_metadata_list if re.search(select_build, each["name"])
    ]
    if select_tag is not None:
        # 如果指定了 select_tag，则进一步按标签过滤构建元数据列表
        filtered_build_metadata_list = [
            each for each in build_metadata_list if each["tag"] == select_tag
        ]
    if skip_build is not None:
        # 如果指定了 skip_build，则排除匹配 skip_build 的构建元数据
        filtered_build_metadata_list = [
            each
            for each in filtered_build_metadata_list
            if not re.search(skip_build, each["name"])
        ]
    # 将筛选后的构建元数据列表中的每个元素格式化成字符串，包括名称、类型和标签，以列表形式连接成一个多行字符串
    selected_build_info = "\n".join(
        f"  - {each['name']}, type: {each['type']}, tag: {each['tag']}"
        for each in filtered_build_metadata_list
    )
    # 创建包含选定构建数量和详细信息的消息字符串
    selected_build_message = (
        f"# {len(filtered_build_metadata_list)} selected builds\n{selected_build_info}"
    )
    # 记录信息消息到日志
    logger.info(selected_build_message)

    # 筛选出类型为 'conda' 的构建元数据列表
    filtered_conda_build_metadata_list = [
        each for each in filtered_build_metadata_list if each["type"] == "conda"
    ]

    # 如果存在 'conda' 类型的构建元数据列表
    if filtered_conda_build_metadata_list:
        # 记录信息到日志，指示开始写入 conda 环境文件
        logger.info("# Writing conda environments")
        # 调用函数，将所有 'conda' 类型的构建元数据列表写入 conda 环境文件
        write_all_conda_environments(filtered_conda_build_metadata_list)
        # 记录信息到日志，指示开始写入 conda 锁定文件
        logger.info("# Writing conda lock files")
        # 调用函数，将所有 'conda' 类型的构建元数据列表写入 conda 锁定文件
        write_all_conda_lock_files(filtered_conda_build_metadata_list)

    # 筛选出类型为 'pip' 的构建元数据列表
    filtered_pip_build_metadata_list = [
        each for each in filtered_build_metadata_list if each["type"] == "pip"
    ]
    # 如果存在 'pip' 类型的构建元数据列表
    if filtered_pip_build_metadata_list:
        # 记录信息到日志，指示开始写入 pip 要求文件
        logger.info("# Writing pip requirements")
        # 调用函数，将所有 'pip' 类型的构建元数据列表写入 pip 要求文件
        write_all_pip_requirements(filtered_pip_build_metadata_list)
        # 记录信息到日志，指示开始写入 pip 锁定文件
        logger.info("# Writing pip lock files")
        # 调用函数，将所有 'pip' 类型的构建元数据列表写入 pip 锁定文件
        write_all_pip_lock_files(filtered_pip_build_metadata_list)
# 如果这个脚本是直接运行的（而不是作为模块被导入），那么执行 main 函数
if __name__ == "__main__":
    main()
```