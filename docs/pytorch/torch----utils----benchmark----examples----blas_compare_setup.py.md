# `.\pytorch\torch\utils\benchmark\examples\blas_compare_setup.py`

```py
# mypy: allow-untyped-defs
import collections  # 导入 collections 模块，用于创建命名元组等数据结构
import os  # 导入 os 模块，提供了与操作系统交互的功能
import shutil  # 导入 shutil 模块，用于高级文件操作，如复制、移动、删除文件等
import subprocess  # 导入 subprocess 模块，用于执行外部命令和获取它们的输出

try:
    # 尝试导入 conda 命令行接口相关模块
    import conda.cli.python_api  # type: ignore[import]
    from conda.cli.python_api import Commands as conda_commands
except ImportError:
    # 如果导入失败，说明不在 conda 环境中，无需处理异常
    pass

# 定义工作根目录
WORKING_ROOT = "/tmp/pytorch_blas_compare_environments"
# 定义常量表示不同的 MKL 版本和 BLAS 实现
MKL_2020_3 = "mkl_2020_3"
MKL_2020_0 = "mkl_2020_0"
OPEN_BLAS = "open_blas"
EIGEN = "eigen"

# 定义一些通用的环境变量
GENERIC_ENV_VARS = ("USE_CUDA=0", "USE_ROCM=0")
# 定义基础包依赖列表
BASE_PKG_DEPS = (
    "cmake",
    "hypothesis",
    "ninja",
    "numpy",
    "pyyaml",
    "setuptools",
    "typing_extensions",
)

# 定义命名元组 SubEnvSpec，用于描述子环境的详细规格
SubEnvSpec = collections.namedtuple(
    "SubEnvSpec", (
        "generic_installs",  # 通用安装包列表
        "special_installs",  # 特殊安装包列表
        "environment_variables",  # 环境变量列表

        # 验证安装的 BLAS 符号
        "expected_blas_symbols",
        # 期望的 MKL 版本
        "expected_mkl_version",
    ))

# 定义子环境的详细规格字典
SUB_ENVS = {
    MKL_2020_3: SubEnvSpec(
        generic_installs=(),  # 无通用安装包
        special_installs=("intel", ("mkl=2020.3", "mkl-include=2020.3")),  # Intel 和指定 MKL 版本的安装包
        environment_variables=("BLAS=MKL",) + GENERIC_ENV_VARS,  # 环境变量设置，指定 BLAS 使用 MKL
        expected_blas_symbols=("mkl_blas_sgemm",),  # 期望的 BLAS 符号
        expected_mkl_version="2020.0.3",  # 期望的 MKL 版本
    ),

    MKL_2020_0: SubEnvSpec(
        generic_installs=(),  # 无通用安装包
        special_installs=("intel", ("mkl=2020.0", "mkl-include=2020.0")),  # Intel 和指定 MKL 版本的安装包
        environment_variables=("BLAS=MKL",) + GENERIC_ENV_VARS,  # 环境变量设置，指定 BLAS 使用 MKL
        expected_blas_symbols=("mkl_blas_sgemm",),  # 期望的 BLAS 符号
        expected_mkl_version="2020.0.0",  # 期望的 MKL 版本
    ),

    OPEN_BLAS: SubEnvSpec(
        generic_installs=("openblas",),  # 安装 OpenBLAS
        special_installs=(),  # 无特殊安装包
        environment_variables=("BLAS=OpenBLAS",) + GENERIC_ENV_VARS,  # 环境变量设置，指定 BLAS 使用 OpenBLAS
        expected_blas_symbols=("exec_blas",),  # 期望的 BLAS 符号
        expected_mkl_version=None,  # 不涉及 MKL 版本
    ),

    # EIGEN: SubEnvSpec(
    #     generic_installs=(),
    #     special_installs=(),
    #     environment_variables=("BLAS=Eigen",) + GENERIC_ENV_VARS,
    #     expected_blas_symbols=(),
    # ),
}


def conda_run(*args):
    """Convenience method to run a conda command."""
    stdout, stderr, retcode = conda.cli.python_api.run_command(*args)  # 执行 conda 命令
    if retcode:
        raise OSError(f"conda error: {str(args)}  retcode: {retcode}\n{stderr}")  # 如果返回码非零，抛出异常

    return stdout  # 返回命令执行的标准输出


def main():
    if os.path.exists(WORKING_ROOT):
        print("Cleaning: removing old working root.")  # 输出信息，清理旧的工作根目录
        shutil.rmtree(WORKING_ROOT)  # 递归删除工作根目录
    os.makedirs(WORKING_ROOT)  # 创建工作根目录

    git_root = subprocess.check_output(
        "git rev-parse --show-toplevel",
        shell=True,
        cwd=os.path.dirname(os.path.realpath(__file__))
    ).decode("utf-8").strip()  # 获取 Git 仓库的根目录路径，并转为字符串格式

if __name__ == "__main__":
    main()  # 如果作为脚本直接运行，则调用主函数 main()
```