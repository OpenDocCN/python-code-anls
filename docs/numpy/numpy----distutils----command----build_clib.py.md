# `.\numpy\numpy\distutils\command\build_clib.py`

```py
# 修改了构建库的版本，使其能处理 Fortran 源文件
import os  # 导入 os 模块
from glob import glob  # 从 glob 模块导入 glob 函数
import shutil  # 导入 shutil 模块
from distutils.command.build_clib import build_clib as old_build_clib  # 从 distutils.command.build_clib 模块中导入 build_clib 类，并重命名为 old_build_clib
from distutils.errors import DistutilsSetupError, DistutilsError, \
    DistutilsFileError  # 导入异常类型

from numpy.distutils import log  # 导入 numpy.distutils 模块中的 log
from distutils.dep_util import newer_group  # 从 distutils.dep_util 模块中导入 newer_group 函数
from numpy.distutils.misc_util import (
    filter_sources, get_lib_source_files, get_numpy_include_dirs,
    has_cxx_sources, has_f_sources, is_sequence
)  # 从 numpy.distutils.misc_util 模块中导入多个函数
from numpy.distutils.ccompiler_opt import new_ccompiler_opt  # 从 numpy.distutils.ccompiler_opt 模块中导入 new_ccompiler_opt 函数，重命名为 new_ccompiler_opt

# 修复 Python distutils 的 bug sf #1718574：
_l = old_build_clib.user_options  # 获取 old_build_clib 的 user_options 属性，赋值给 _l
for _i in range(len(_l)):  # 遍历 _l
    if _l[_i][0] in ['build-clib', 'build-temp']:  # 如果 _l[_i][0] 在 ['build-clib', 'build-temp'] 中
        _l[_i] = (_l[_i][0] + '=',) + _l[_i][1:]  # 将 _l[_i][0] 赋值为 (_l[_i][0] + '=',) + _l[_i][1:]

class build_clib(old_build_clib):  # 定义 build_clib 类，继承于 old_build_clib 类

    description = "build C/C++/F libraries used by Python extensions"  # 定义 description 描述

    user_options = old_build_clib.user_options + [  # 将 old_build_clib.user_options 添加到 user_options 中
        ('fcompiler=', None,  # 定义 ('fcompiler=', None) 的元组
         "specify the Fortran compiler type"),  # 指定 Fortran 编译器类型
        ('inplace', 'i', 'Build in-place'),  # 定义 ('inplace', 'i', 'Build in-place') 的元组
        ('parallel=', 'j',  # 定义 ('parallel=', 'j') 的元组
         "number of parallel jobs"),  # 指定并行作业的数量
        ('warn-error', None,  # 定义 ('warn-error', None) 的元组
         "turn all warnings into errors (-Werror)"),  # 将所有警告转换为错误 (-Werror)
        ('cpu-baseline=', None,  # 定义 ('cpu-baseline=', None) 的元组
         "specify a list of enabled baseline CPU optimizations"),  # 指定启用的基准 CPU 优化列表
        ('cpu-dispatch=', None,  # 定义 ('cpu-dispatch=', None) 的元组
         "specify a list of dispatched CPU optimizations"),  # 指定分派的 CPU 优化列表
        ('disable-optimization', None,  # 定义 ('disable-optimization', None) 的元组
         "disable CPU optimized code(dispatch,simd,fast...)"),  # 禁用 CPU 优化代码(dispatch,simd,fast...)
    ]

    boolean_options = old_build_clib.boolean_options + \
    ['inplace', 'warn-error', 'disable-optimization']  # 将 old_build_clib.boolean_options 和 ['inplace', 'warn-error', 'disable-optimization'] 相加

    def initialize_options(self):  # 定义 initialize_options 方法
        old_build_clib.initialize_options(self)  # 调用 old_build_clib 的 initialize_options 方法
        self.fcompiler = None  # 初始化 self.fcompiler 为 None
        self.inplace = 0  # 初始化 self.inplace 为 0
        self.parallel = None  # 初始化 self.parallel 为 None
        self.warn_error = None  # 初始化 self.warn_error 为 None
        self.cpu_baseline = None  # 初始化 self.cpu_baseline 为 None
        self.cpu_dispatch = None  # 初始化 self.cpu_dispatch 为 None
        self.disable_optimization = None  # 初始化 self.disable_optimization 为 None

    def finalize_options(self):  # 定义 finalize_options 方法
        if self.parallel:  # 如果 self.parallel 存在
            try:  # 尝试执行下面代码
                self.parallel = int(self.parallel)  # 将 self.parallel 转换为整数
            except ValueError as e:  # 捕获 ValueError 异常，并赋值给 e
                raise ValueError("--parallel/-j argument must be an integer") from e  # 抛出 ValueError 异常
        old_build_clib.finalize_options(self)  # 调用 old_build_clib 的 finalize_options 方法
        self.set_undefined_options('build',  # 设置未定义的选项
                                        ('parallel', 'parallel'),  # （'parallel', 'parallel') 选项
                                        ('warn_error', 'warn_error'),  # （'warn_error', 'warn_error') 选项
                                        ('cpu_baseline', 'cpu_baseline'),  # （'cpu_baseline', 'cpu_baseline') 选项
                                        ('cpu_dispatch', 'cpu_dispatch'),  # （'cpu_dispatch', 'cpu_dispatch') 选项
                                        ('disable_optimization', 'disable_optimization')  # （'disable_optimization', 'disable_optimization') 选项
                                  )

    def have_f_sources(self):  # 定义 have_f_sources 方法
        for (lib_name, build_info) in self.libraries:  # 遍历 self.libraries
            if has_f_sources(build_info.get('sources', [])):  # 如果 build_info 中有 'sources'，则调用 has_f_sources 函数
                return True  # 返回 True
        return False  # 返回 False
    # 检查是否有 C++ 源文件
    def have_cxx_sources(self):
        # 遍历库列表中的每个库
        for (lib_name, build_info) in self.libraries:
            # 如果库中有 C++ 源文件，则返回 True
            if has_cxx_sources(build_info.get('sources', [])):
                return True
        # 如果所有库都没有 C++ 源文件，返回 False
        return False

    # 获取所有源文件
    def get_source_files(self):
        # 检查库列表
        self.check_library_list(self.libraries)
        filenames = []
        # 遍历所有库，并获取每个库的源文件
        for lib in self.libraries:
            filenames.extend(get_lib_source_files(lib))
        return filenames

    # 编译库
    def build_libraries(self, libraries):
        # 遍历所有库，针对每个库调用编译函数
        for (lib_name, build_info) in libraries:
            self.build_a_library(build_info, lib_name, libraries)

    # 组装标志
    def assemble_flags(self, in_flags):
        """ 从标志列表中组装标志

        Parameters
        ----------
        in_flags : None or sequence
            None 对应空列表。序列元素可以是字符串，也可以是返回字符串列表的可调用对象。可调用对象以 `self` 作为单个参数。

        Returns
        -------
        out_flags : list
        """
        # 如果输入为空，则返回空列表
        if in_flags is None:
            return []
        out_flags = []
        for in_flag in in_flags:
            # 如果输入是可调用对象，则调用它并获取返回的标志列表
            if callable(in_flag):
                out_flags += in_flag(self)
            else:
                # 否则直接添加至输出标志列表
                out_flags.append(in_flag)
        return out_flags
```