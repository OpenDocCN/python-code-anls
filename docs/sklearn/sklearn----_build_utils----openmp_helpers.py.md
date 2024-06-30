# `D:\src\scipysrc\scikit-learn\sklearn\_build_utils\openmp_helpers.py`

```
# 辅助工具，用于在构建过程中支持 OpenMP。

# 这段代码在很大程度上改编自 astropy 的 OpenMP 辅助工具，
# 可以在 https://github.com/astropy/extension-helpers/blob/master/extension_helpers/_openmp_helpers.py 找到

import os
import sys
import textwrap
import warnings

from .pre_build_helpers import compile_test_program


def get_openmp_flag():
    # 根据不同的操作系统平台返回相应的 OpenMP 编译标志
    if sys.platform == "win32":
        return ["/openmp"]
    elif sys.platform == "darwin" and "openmp" in os.getenv("CPPFLAGS", ""):
        # 当使用 Apple-clang 时，无法将 -fopenmp 作为编译标志传递。
        # 必须在预处理过程中启用 OpenMP 支持。
        #
        # 例如，我们的 macOS 轮子构建作业使用以下环境变量来使用 Apple-clang 和 brew 安装的 "libomp"：
        #
        # export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
        # export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
        # export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
        # export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/opt/libomp/lib
        #                          -L/usr/local/opt/libomp/lib -lomp"
        return []
    # 默认情况下适用于 GCC 和 clang：
    return ["-fopenmp"]


def check_openmp_support():
    """检查是否可以编译和运行 OpenMP 测试代码"""
    if "PYODIDE" in os.environ:
        # Pyodide 不支持 OpenMP
        return False

    # 要测试的 C 代码，打印当前线程数目
    code = textwrap.dedent(
        """\
        #include <omp.h>
        #include <stdio.h>
        int main(void) {
        #pragma omp parallel
        printf("nthreads=%d\\n", omp_get_num_threads());
        return 0;
        }
        """
    )

    extra_preargs = os.getenv("LDFLAGS", None)
    if extra_preargs is not None:
        extra_preargs = extra_preargs.strip().split(" ")
        # FIXME: 暂时的修复，用于在 Linux 上链接到系统库
        # 应该删除 "-Wl,--sysroot=/" 
        extra_preargs = [
            flag
            for flag in extra_preargs
            if flag.startswith(("-L", "-Wl,-rpath", "-l", "-Wl,--sysroot=/"))
        ]

    extra_postargs = get_openmp_flag()

    openmp_exception = None
    try:
        # 编译测试程序，并获取输出结果
        output = compile_test_program(
            code, extra_preargs=extra_preargs, extra_postargs=extra_postargs
        )

        if output and "nthreads=" in output[0]:
            nthreads = int(output[0].strip().split("=")[1])
            openmp_supported = len(output) == nthreads
        elif "PYTHON_CROSSENV" in os.environ:
            # 由于交叉编译时无法运行测试程序，假定如果可以编译程序，则支持 OpenMP
            openmp_supported = True
        else:
            openmp_supported = False
    except Exception as exception:
        # 捕获所有异常，以便进行处理
        # 可以更具体地捕获 CompileError、LinkError 和 subprocess.CalledProcessError。
        # setuptools 引入了 CompileError 和 LinkError，但需要版本 61.1。即使是最新版本的 Ubuntu（22.04LTS），也只有 59.6。
        # 因此，目前我们捕获所有异常，并重新引发一个带有原始错误消息的通用异常：
        openmp_supported = False
        openmp_exception = exception

    if not openmp_supported:
        # 如果不支持 OpenMP
        if os.getenv("SKLEARN_FAIL_NO_OPENMP"):
            # 如果环境变量 SKLEARN_FAIL_NO_OPENMP 存在，则抛出异常
            raise Exception(
                "Failed to build scikit-learn with OpenMP support"
            ) from openmp_exception
        else:
            # 否则，生成警告消息
            message = textwrap.dedent(
                """

                                ***********
                                * WARNING *
                                ***********

                It seems that scikit-learn cannot be built with OpenMP.

                - Make sure you have followed the installation instructions:

                    https://scikit-learn.org/dev/developers/advanced_installation.html

                - If your compiler supports OpenMP but you still see this
                  message, please submit a bug report at:

                    https://github.com/scikit-learn/scikit-learn/issues

                - The build will continue with OpenMP-based parallelism
                  disabled. Note however that some estimators will run in
                  sequential mode instead of leveraging thread-based
                  parallelism.

                                    ***
                """
            )
            # 发出警告
            warnings.warn(message)

    return openmp_supported
```