# `D:\src\scipysrc\scikit-learn\setup.py`

```
# 设置脚本的解释器为Python，指定在环境变量PATH中寻找
#! /usr/bin/env python

# 作者信息：scikit-learn的开发团队
# 授权信息：采用三条款BSD许可证
# Authors: The scikit-learn developers
# License: 3-clause BSD

# 导入必要的模块
import importlib  # 导入模块动态加载功能的库
import os  # 导入操作系统相关功能的库
import platform  # 导入获取平台信息的库
import shutil  # 导入文件和目录管理功能的库
import sys  # 导入系统相关功能的库
import traceback  # 导入追踪异常信息的库
from os.path import join  # 从os.path模块中导入join函数

# 从setuptools模块中导入Command和Extension类，以及setup函数
from setuptools import Command, Extension, setup
# 从setuptools.command.build_ext模块中导入build_ext类
from setuptools.command.build_ext import build_ext

try:
    import builtins  # 尝试导入Python内置模块
except ImportError:
    # 兼容Python 2：用__builtin__作为builtins模块的别名
    import __builtin__ as builtins

# 这里是一个略微（非常！）巧妙的处理：设置一个全局变量，以便于主要的
# sklearn __init__模块能够检测是否被设置程序加载，以避免尝试加载尚未构建的组件。
# TODO: 这个处理是否可以简化或者在切换到setuptools后移除？因为不再使用numpy.distutils？
builtins.__SKLEARN_SETUP__ = True

# 设置项目的元信息
DISTNAME = "scikit-learn"  # 项目名称
DESCRIPTION = "A set of python modules for machine learning and data mining"  # 项目描述
# 从README.rst文件中读取长描述信息
with open("README.rst") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "scikit-learn developers"  # 维护者信息
MAINTAINER_EMAIL = "scikit-learn@python.org"  # 维护者邮箱
URL = "https://scikit-learn.org"  # 项目的URL
DOWNLOAD_URL = "https://pypi.org/project/scikit-learn/#files"  # 下载链接
LICENSE = "new BSD"  # 采用的许可证类型
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/scikit-learn/scikit-learn/issues",  # Bug追踪页面链接
    "Documentation": "https://scikit-learn.org/stable/documentation.html",  # 文档链接
    "Source Code": "https://github.com/scikit-learn/scikit-learn",  # 源代码链接
}

# 实际上我们可以导入一个精简版本的sklearn，它不需要编译后的代码
import sklearn  # 导入sklearn库，用于机器学习任务
import sklearn._min_dependencies as min_deps  # 导入sklearn的最小依赖
from sklearn._build_utils import _check_cython_version  # 导入检查Cython版本的函数
from sklearn.externals._packaging.version import parse as parse_version  # 导入解析版本号的函数

# 获取sklearn的版本号
VERSION = sklearn.__version__

# 自定义的清理命令，用于从源码树中删除构建产物
class CleanCommand(Command):
    description = "Remove build artifacts from the source tree"  # 清理命令的描述信息

    user_options = []  # 用户选项为空列表

    def initialize_options(self):
        pass  # 初始化选项方法为空

    def finalize_options(self):
        pass  # 最终化选项方法为空
    def run(self):
        # 获取当前文件所在目录的绝对路径
        cwd = os.path.abspath(os.path.dirname(__file__))
        # 判断是否在一个 sdist 打包环境之外，以决定是否移除 .c 文件
        remove_c_files = not os.path.exists(os.path.join(cwd, "PKG-INFO"))
        # 如果需要移除 .c 文件，则打印提示信息
        if remove_c_files:
            print("Will remove generated .c files")
        # 如果存在 'build' 目录，则递归删除该目录及其内容
        if os.path.exists("build"):
            shutil.rmtree("build")
        # 遍历 'sklearn' 目录及其子目录中的文件和目录
        for dirpath, dirnames, filenames in os.walk("sklearn"):
            # 遍历当前目录下的所有文件名
            for filename in filenames:
                # 分离文件名和扩展名
                root, extension = os.path.splitext(filename)

                # 如果文件扩展名为 .so, .pyd, .dll, .pyc，则删除该文件
                if extension in [".so", ".pyd", ".dll", ".pyc"]:
                    os.unlink(os.path.join(dirpath, filename))

                # 如果需要移除 .c 文件，并且文件扩展名为 .c 或 .cpp，则尝试删除该文件
                if remove_c_files and extension in [".c", ".cpp"]:
                    # 构造与当前文件相同但扩展名为 .pyx 的文件名
                    pyx_file = str.replace(filename, extension, ".pyx")
                    # 如果存在相应的 .pyx 文件，则删除 .c 或 .cpp 文件
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))

                # 如果需要移除 .c 文件，并且文件扩展名为 .tp，则尝试删除与文件名相同的目录
                if remove_c_files and extension == ".tp":
                    if os.path.exists(os.path.join(dirpath, root)):
                        os.unlink(os.path.join(dirpath, root))

            # 遍历当前目录下的所有子目录名
            for dirname in dirnames:
                # 如果子目录名为 '__pycache__'，则递归删除该目录及其内容
                if dirname == "__pycache__":
                    shutil.rmtree(os.path.join(dirpath, dirname))
# 自定义 build_ext 命令，根据操作系统和编译器设置 OpenMP 编译标志，并通过环境变量设置并行级别（用于构建 Wheel 包的 CI 中非常有用）。
# 需要在导入 setuptools 后导入 build_ext。
class build_ext_subclass(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        if self.parallel is None:
            # 如果未通过命令行标志（--parallel 或 -j）定义 self.parallel，则尝试从环境变量 SKLEARN_BUILD_PARALLEL 获取并行级别。
            parallel = os.environ.get("SKLEARN_BUILD_PARALLEL")
            if parallel:
                self.parallel = int(parallel)
        if self.parallel:
            print("setting parallel=%d " % self.parallel)

    def build_extensions(self):
        from sklearn._build_utils.openmp_helpers import get_openmp_flag

        # 对于所有编译的扩展，始终使用 NumPy 1.7 C API。
        # 参考：https://numpy.org/doc/stable/reference/c-api/deprecations.html
        DEFINE_MACRO_NUMPY_C_API = (
            "NPY_NO_DEPRECATED_API",
            "NPY_1_7_API_VERSION",
        )
        for ext in self.extensions:
            ext.define_macros.append(DEFINE_MACRO_NUMPY_C_API)

        # 如果支持 OpenMP，则获取 OpenMP 标志并为每个扩展添加编译和链接参数。
        if sklearn._OPENMP_SUPPORTED:
            openmp_flag = get_openmp_flag()

            for e in self.extensions:
                e.extra_compile_args += openmp_flag
                e.extra_link_args += openmp_flag

        # 调用父类的 build_extensions 方法进行实际的扩展构建。
        build_ext.build_extensions(self)

    def run(self):
        # 指定 `build_clib` 以允许从新克隆的代码库中完全运行 `python setup.py develop`。
        self.run_command("build_clib")
        build_ext.run(self)


# cmdclass 字典，包含 clean 和 build_ext 的自定义命令类。
cmdclass = {
    "clean": CleanCommand,
    "build_ext": build_ext_subclass,
}


def check_package_status(package, min_version):
    """
    返回一个包含布尔值的字典，指明给定包是否为最新版本，以及版本字符串（如果未安装则为空字符串）。
    """
    package_status = {}
    try:
        module = importlib.import_module(package)
        package_version = module.__version__
        package_status["up_to_date"] = parse_version(package_version) >= parse_version(
            min_version
        )
        package_status["version"] = package_version
    except ImportError:
        traceback.print_exc()
        package_status["up_to_date"] = False
        package_status["version"] = ""

    req_str = "scikit-learn 需要 {} >= {}。\n".format(package, min_version)

    instructions = (
        "安装说明请参阅 scikit-learn 网站："
        "https://scikit-learn.org/stable/install.html\n"
    )
    # 如果包的状态标志 "up_to_date" 是 False，表示包不是最新的
    if package_status["up_to_date"] is False:
        # 如果包的版本信息存在
        if package_status["version"]:
            # 抛出 ImportError 异常，提示包的安装版本过时
            raise ImportError(
                "Your installation of {} {} is out-of-date.\n{}{}".format(
                    package, package_status["version"], req_str, instructions
                )
            )
        else:
            # 抛出 ImportError 异常，提示包未安装
            raise ImportError(
                "{} is not installed.\n{}{}".format(package, req_str, instructions)
            )
extension_config = {
    "__check_build": [
        {"sources": ["_check_build.pyx"]},  # 定义 __check_build 扩展模块的源文件为 _check_build.pyx
    ],
    "": [
        {"sources": ["_isotonic.pyx"]},  # 定义空字符串键扩展模块的源文件为 _isotonic.pyx
    ],
    "_loss": [
        {"sources": ["_loss.pyx.tp"]},  # 定义 _loss 扩展模块的源文件为 _loss.pyx.tp
    ],
    "cluster": [
        {"sources": ["_dbscan_inner.pyx"], "language": "c++"},  # 定义 cluster 模块的 _dbscan_inner.pyx 源文件，使用 C++ 编写
        {"sources": ["_hierarchical_fast.pyx"], "language": "c++", "include_np": True},  # 定义 _hierarchical_fast.pyx 源文件，使用 C++ 编写，包含 NumPy
        {"sources": ["_k_means_common.pyx"], "include_np": True},  # 定义 _k_means_common.pyx 源文件，包含 NumPy
        {"sources": ["_k_means_lloyd.pyx"], "include_np": True},  # 定义 _k_means_lloyd.pyx 源文件，包含 NumPy
        {"sources": ["_k_means_elkan.pyx"], "include_np": True},  # 定义 _k_means_elkan.pyx 源文件，包含 NumPy
        {"sources": ["_k_means_minibatch.pyx"], "include_np": True},  # 定义 _k_means_minibatch.pyx 源文件，包含 NumPy
    ],
    "cluster._hdbscan": [
        {"sources": ["_linkage.pyx"], "include_np": True},  # 定义 cluster._hdbscan 模块的 _linkage.pyx 源文件，包含 NumPy
        {"sources": ["_reachability.pyx"], "include_np": True},  # 定义 _reachability.pyx 源文件，包含 NumPy
        {"sources": ["_tree.pyx"], "include_np": True},  # 定义 _tree.pyx 源文件，包含 NumPy
    ],
    "datasets": [
        {
            "sources": ["_svmlight_format_fast.pyx"],  # 定义 datasets 模块的 _svmlight_format_fast.pyx 源文件
            "include_np": True,  # 该模块包含 NumPy
            "compile_for_pypy": False,  # 不针对 PyPy 编译
        }
    ],
    "decomposition": [
        {"sources": ["_online_lda_fast.pyx"]},  # 定义 decomposition 模块的 _online_lda_fast.pyx 源文件
        {"sources": ["_cdnmf_fast.pyx"], "include_np": True},  # 定义 _cdnmf_fast.pyx 源文件，包含 NumPy
    ],
    "ensemble": [
        {"sources": ["_gradient_boosting.pyx"], "include_np": True},  # 定义 ensemble 模块的 _gradient_boosting.pyx 源文件，包含 NumPy
    ],
    "ensemble._hist_gradient_boosting": [
        {"sources": ["_gradient_boosting.pyx"]},  # 定义 ensemble._hist_gradient_boosting 模块的 _gradient_boosting.pyx 源文件
        {"sources": ["histogram.pyx"]},  # 定义 histogram.pyx 源文件
        {"sources": ["splitting.pyx"]},  # 定义 splitting.pyx 源文件
        {"sources": ["_binning.pyx"]},  # 定义 _binning.pyx 源文件
        {"sources": ["_predictor.pyx"]},  # 定义 _predictor.pyx 源文件
        {"sources": ["_bitset.pyx"]},  # 定义 _bitset.pyx 源文件
        {"sources": ["common.pyx"]},  # 定义 common.pyx 源文件
    ],
    "feature_extraction": [
        {"sources": ["_hashing_fast.pyx"], "language": "c++", "include_np": True},  # 定义 feature_extraction 模块的 _hashing_fast.pyx 源文件，使用 C++ 编写，包含 NumPy
    ],
    "linear_model": [
        {"sources": ["_cd_fast.pyx"]},  # 定义 linear_model 模块的 _cd_fast.pyx 源文件
        {"sources": ["_sgd_fast.pyx.tp"]},  # 定义 _sgd_fast.pyx 源文件
        {"sources": ["_sag_fast.pyx.tp"]},  # 定义 _sag_fast.pyx 源文件
    ],
    "manifold": [
        {"sources": ["_utils.pyx"]},  # 定义 manifold 模块的 _utils.pyx 源文件
        {"sources": ["_barnes_hut_tsne.pyx"], "include_np": True},  # 定义 _barnes_hut_tsne.pyx 源文件，包含 NumPy
    ],
    "metrics": [
        {"sources": ["_pairwise_fast.pyx"]},  # 定义 metrics 模块的 _pairwise_fast.pyx 源文件
        {
            "sources": ["_dist_metrics.pyx.tp", "_dist_metrics.pxd.tp"],  # 定义 _dist_metrics.pyx.tp 和 _dist_metrics.pxd.tp 源文件
            "include_np": True,  # 该模块包含 NumPy
        },
    ],
    "metrics.cluster": [
        {"sources": ["_expected_mutual_info_fast.pyx"]},  # 定义 metrics.cluster 模块的 _expected_mutual_info_fast.pyx 源文件
    ],
}
    # 定义一个名为 "metrics._pairwise_distances_reduction" 的列表，其中包含多个字典，每个字典描述了一个编译源文件的任务
    
    {
        "sources": ["_datasets_pair.pyx.tp", "_datasets_pair.pxd.tp"],
        "language": "c++",
        "include_np": True,
        "extra_compile_args": ["-std=c++11"],
    },
    {
        "sources": ["_middle_term_computer.pyx.tp", "_middle_term_computer.pxd.tp"],
        "language": "c++",
        "extra_compile_args": ["-std=c++11"],
    },
    {
        "sources": ["_base.pyx.tp", "_base.pxd.tp"],
        "language": "c++",
        "include_np": True,
        "extra_compile_args": ["-std=c++11"],
    },
    {
        "sources": ["_argkmin.pyx.tp", "_argkmin.pxd.tp"],
        "language": "c++",
        "include_np": True,
        "extra_compile_args": ["-std=c++11"],
    },
    {
        "sources": ["_argkmin_classmode.pyx.tp"],
        "language": "c++",
        "include_np": True,
        "extra_compile_args": ["-std=c++11"],
    },
    {
        "sources": ["_radius_neighbors.pyx.tp", "_radius_neighbors.pxd.tp"],
        "language": "c++",
        "include_np": True,
        "extra_compile_args": ["-std=c++11"],
    },
    {
        "sources": ["_radius_neighbors_classmode.pyx.tp"],
        "language": "c++",
        "include_np": True,
        "extra_compile_args": ["-std=c++11"],
    },
    
    # 定义一个名为 "preprocessing" 的列表，包含两个字典，描述了预处理模块的编译任务
    
    {
        "sources": ["_csr_polynomial_expansion.pyx"]
    },
    {
        "sources": ["_target_encoder_fast.pyx"],
        "language": "c++",
        "extra_compile_args": ["-std=c++11"],
    },
    
    # 定义一个名为 "neighbors" 的列表，包含多个字典，描述了邻居模块的编译任务
    
    {
        "sources": ["_binary_tree.pxi.tp"],
        "include_np": True
    },
    {
        "sources": ["_ball_tree.pyx.tp"],
        "include_np": True
    },
    {
        "sources": ["_kd_tree.pyx.tp"],
        "include_np": True
    },
    {
        "sources": ["_partition_nodes.pyx"],
        "language": "c++",
        "include_np": True
    },
    {
        "sources": ["_quad_tree.pyx"],
        "include_np": True
    }
    {
        "svm": [
            {
                "sources": ["_newrand.pyx"],
                "include_dirs": [join("src", "newrand")],
                "language": "c++",
                # 使用 C++11 的随机数生成器修复
                "extra_compile_args": ["-std=c++11"],
            },
            {
                "sources": ["_libsvm.pyx"],
                "depends": [
                    join("src", "libsvm", "libsvm_helper.c"),
                    join("src", "libsvm", "libsvm_template.cpp"),
                    join("src", "libsvm", "svm.cpp"),
                    join("src", "libsvm", "svm.h"),
                    join("src", "newrand", "newrand.h"),
                ],
                "include_dirs": [
                    join("src", "libsvm"),
                    join("src", "newrand"),
                ],
                "libraries": ["libsvm-skl"],
                "extra_link_args": ["-lstdc++"],
            },
            {
                "sources": ["_liblinear.pyx"],
                "libraries": ["liblinear-skl"],
                "include_dirs": [
                    join("src", "liblinear"),
                    join("src", "newrand"),
                    join("..", "utils"),
                ],
                "depends": [
                    join("src", "liblinear", "tron.h"),
                    join("src", "liblinear", "linear.h"),
                    join("src", "liblinear", "liblinear_helper.c"),
                    join("src", "newrand", "newrand.h"),
                ],
                "extra_link_args": ["-lstdc++"],
            },
            {
                "sources": ["_libsvm_sparse.pyx"],
                "libraries": ["libsvm-skl"],
                "include_dirs": [
                    join("src", "libsvm"),
                    join("src", "newrand"),
                ],
                "depends": [
                    join("src", "libsvm", "svm.h"),
                    join("src", "newrand", "newrand.h"),
                    join("src", "libsvm", "libsvm_sparse_helper.c"),
                ],
                "extra_link_args": ["-lstdc++"],
            },
        ],
        "tree": [
            {
                "sources": ["_tree.pyx"],
                "language": "c++",
                "include_np": True,
                "optimization_level": "O3",
            },
            {"sources": ["_splitter.pyx"], "include_np": True, "optimization_level": "O3"},
            {"sources": ["_criterion.pyx"], "include_np": True, "optimization_level": "O3"},
            {"sources": ["_utils.pyx"], "include_np": True, "optimization_level": "O3"},
        ],
    }
    
    
    
    # SVM部分的配置：包含不同的编译和链接选项以支持SVM相关的Python扩展模块
    "svm": [
        {
            "sources": ["_newrand.pyx"],
            "include_dirs": [join("src", "newrand")],
            "language": "c++",
            # 使用 C++11 的随机数生成器修复
            "extra_compile_args": ["-std=c++11"],
        },
        {
            "sources": ["_libsvm.pyx"],
            "depends": [
                join("src", "libsvm", "libsvm_helper.c"),
                join("src", "libsvm", "libsvm_template.cpp"),
                join("src", "libsvm", "svm.cpp"),
                join("src", "libsvm", "svm.h"),
                join("src", "newrand", "newrand.h"),
            ],
            "include_dirs": [
                join("src", "libsvm"),
                join("src", "newrand"),
            ],
            "libraries": ["libsvm-skl"],
            "extra_link_args": ["-lstdc++"],
        },
        {
            "sources": ["_liblinear.pyx"],
            "libraries": ["liblinear-skl"],
            "include_dirs": [
                join("src", "liblinear"),
                join("src", "newrand"),
                join("..", "utils"),
            ],
            "depends": [
                join("src", "liblinear", "tron.h"),
                join("src", "liblinear", "linear.h"),
                join("src", "liblinear", "liblinear_helper.c"),
                join("src", "newrand", "newrand.h"),
            ],
            "extra_link_args": ["-lstdc++"],
        },
        {
            "sources": ["_libsvm_sparse.pyx"],
            "libraries": ["libsvm-skl"],
            "include_dirs": [
                join("src", "libsvm"),
                join("src", "newrand"),
            ],
            "depends": [
                join("src", "libsvm", "svm.h"),
                join("src", "newrand", "newrand.h"),
                join("src", "libsvm", "libsvm_sparse_helper.c"),
            ],
            "extra_link_args": ["-lstdc++"],
        },
    ],
    
    # Tree部分的配置：包含不同的编译选项以支持决策树相关的Python扩展模块
    "tree": [
        {
            "sources": ["_tree.pyx"],
            "language": "c++",
            "include_np": True,
            "optimization_level": "O3",
        },
        {"sources": ["_splitter.pyx"], "include_np": True, "optimization_level": "O3"},
        {"sources": ["_criterion.pyx"], "include_np": True, "optimization_level": "O3"},
        {"sources": ["_utils.pyx"], "include_np": True, "optimization_level": "O3"},
    ]
    "utils": [
        {"sources": ["sparsefuncs_fast.pyx"]},  # 定义一个包含单一源文件的字典
        {"sources": ["_cython_blas.pyx"]},  # 定义一个包含单一源文件的字典
        {"sources": ["arrayfuncs.pyx"]},  # 定义一个包含单一源文件的字典
        {
            "sources": ["murmurhash.pyx", join("src", "MurmurHash3.cpp")],  # 定义一个包含两个源文件及一个包含目录的字典
            "include_dirs": ["src"],  # 指定源文件包含目录
        },
        {"sources": ["_fast_dict.pyx"], "language": "c++"},  # 定义一个包含单一源文件及语言选项的字典
        {"sources": ["_openmp_helpers.pyx"]},  # 定义一个包含单一源文件的字典
        {"sources": ["_seq_dataset.pyx.tp", "_seq_dataset.pxd.tp"]},  # 定义一个包含两个源文件的字典
        {"sources": ["_weight_vector.pyx.tp", "_weight_vector.pxd.tp"]},  # 定义一个包含两个源文件的字典
        {"sources": ["_random.pyx"]},  # 定义一个包含单一源文件的字典
        {"sources": ["_typedefs.pyx"]},  # 定义一个包含单一源文件的字典
        {"sources": ["_heap.pyx"]},  # 定义一个包含单一源文件的字典
        {"sources": ["_sorting.pyx"]},  # 定义一个包含单一源文件的字典
        {
            "sources": ["_vector_sentinel.pyx"],  # 定义一个包含单一源文件的字典
            "language": "c++",  # 指定语言选项为 C++
            "include_np": True,  # 指定包含 NumPy 头文件
        },
        {"sources": ["_isfinite.pyx"]},  # 定义一个包含单一源文件的字典
    ],
}

# `libraries` 中的路径必须相对于根目录，因为 `libraries` 直接传递给 `setup`
libraries = [
    (
        "libsvm-skl",
        {
            "sources": [
                join("sklearn", "svm", "src", "libsvm", "libsvm_template.cpp"),
            ],
            "depends": [
                join("sklearn", "svm", "src", "libsvm", "svm.cpp"),
                join("sklearn", "svm", "src", "libsvm", "svm.h"),
                join("sklearn", "svm", "src", "newrand", "newrand.h"),
            ],
            # 使用 C++11 以修复随机数生成器问题
            "extra_compiler_args": ["-std=c++11"],
            "extra_link_args": ["-lstdc++"],
        },
    ),
    (
        "liblinear-skl",
        {
            "sources": [
                join("sklearn", "svm", "src", "liblinear", "linear.cpp"),
                join("sklearn", "svm", "src", "liblinear", "tron.cpp"),
            ],
            "depends": [
                join("sklearn", "svm", "src", "liblinear", "linear.h"),
                join("sklearn", "svm", "src", "liblinear", "tron.h"),
                join("sklearn", "svm", "src", "newrand", "newrand.h"),
            ],
            # 使用 C++11 以修复随机数生成器问题
            "extra_compiler_args": ["-std=c++11"],
            "extra_link_args": ["-lstdc++"],
        },
    ),
]


def configure_extension_modules():
    # 如果 `sdist` 在 sys.argv 中或者 `--help` 在 sys.argv 中，则跳过 Cython 编译
    # 因为我们不希望将生成的 C/C++ 文件包含在发布的 tar 包中，因为它们未必与未来版本的 Python 兼容
    if "sdist" in sys.argv or "--help" in sys.argv:
        return []

    import numpy

    from sklearn._build_utils import cythonize_extensions, gen_from_templates

    is_pypy = platform.python_implementation() == "PyPy"
    np_include = numpy.get_include()
    default_optimization_level = "O2"

    if os.name == "posix":
        default_libraries = ["m"]
    else:
        default_libraries = []

    default_extra_compile_args = []
    build_with_debug_symbols = (
        os.environ.get("SKLEARN_BUILD_ENABLE_DEBUG_SYMBOLS", "0") != "0"
    )
    if os.name == "posix":
        if build_with_debug_symbols:
            default_extra_compile_args.append("-g")
        else:
            # 设置 -g0 将会去除符号信息，从而减小扩展的二进制大小
            default_extra_compile_args.append("-g0")

    cython_exts = []
    return cythonize_extensions(cython_exts)


def setup_package():
    python_requires = ">=3.9"
    required_python_version = (3, 9)
    metadata = dict(  # 创建一个元数据字典，用于描述软件包的基本信息
        name=DISTNAME,  # 软件包的名称
        maintainer=MAINTAINER,  # 维护者的姓名
        maintainer_email=MAINTAINER_EMAIL,  # 维护者的电子邮件地址
        description=DESCRIPTION,  # 软件包的简短描述
        license=LICENSE,  # 软件包的许可证信息
        url=URL,  # 软件包的主页 URL
        download_url=DOWNLOAD_URL,  # 软件包的下载 URL
        project_urls=PROJECT_URLS,  # 与项目相关的其他 URL
        version=VERSION,  # 软件包的版本号
        long_description=LONG_DESCRIPTION,  # 软件包的详细描述
        classifiers=[  # 软件包的分类器列表，描述软件包适用的环境和类型
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: C",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Development Status :: 5 - Production/Stable",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: Implementation :: CPython",
        ],
        cmdclass=cmdclass,  # 定义在构建过程中用于自定义命令的类
        python_requires=python_requires,  # 指定软件包所需的 Python 版本范围
        install_requires=min_deps.tag_to_packages["install"],  # 指定安装软件包所需的依赖项
        package_data={  # 指定软件包在安装过程中需要包含的额外数据文件
            "": ["*.csv", "*.gz", "*.txt", "*.pxd", "*.rst", "*.jpg", "*.css"]
        },
        zip_safe=False,  # 禁用将软件包安装为 .egg 文件，确保可以直接运行
        extras_require={  # 指定不同用途下的额外依赖项
            key: min_deps.tag_to_packages[key]
            for key in ["examples", "docs", "tests", "benchmark"]
        },
    )

    commands = [arg for arg in sys.argv[1:] if not arg.startswith("-")]  # 获取命令行参数列表，排除以 '-' 开头的参数
    if not all(  # 检查是否所有的命令行参数都属于预定义的一些操作
        command in ("egg_info", "dist_info", "clean", "check") for command in commands
    ):
        if sys.version_info < required_python_version:  # 如果当前 Python 版本低于要求的版本
            required_version = "%d.%d" % required_python_version  # 将要求的 Python 版本转换为字符串格式
            raise RuntimeError(  # 抛出运行时异常，指出 Python 版本不符合要求
                "Scikit-learn requires Python %s or later. The current"
                " Python version is %s installed in %s."
                % (required_version, platform.python_version(), sys.executable)
            )

        check_package_status("numpy", min_deps.NUMPY_MIN_VERSION)  # 检查 numpy 包是否满足最低版本要求
        check_package_status("scipy", min_deps.SCIPY_MIN_VERSION)  # 检查 scipy 包是否满足最低版本要求

        _check_cython_version()  # 检查 Cython 的版本

        metadata["ext_modules"] = configure_extension_modules()  # 配置扩展模块
        metadata["libraries"] = libraries  # 设置软件包需要链接的外部库

    setup(**metadata)  # 调用 setup 函数，传入元数据字典来安装软件包
# 如果当前脚本作为主程序执行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 调用 setup_package 函数来执行必要的初始化和配置
    setup_package()
```