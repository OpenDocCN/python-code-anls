# `.\numpy\numpy\f2py\_backends\_distutils.py`

```
# 从 _backend 模块导入 Backend 类
from ._backend import Backend

# 导入 numpy.distutils.core 模块中的 setup 和 Extension 类
from numpy.distutils.core import setup, Extension
# 导入 numpy.distutils.system_info 模块中的 get_info 函数
from numpy.distutils.system_info import get_info
# 导入 numpy.distutils.misc_util 模块中的 dict_append 函数
from numpy.distutils.misc_util import dict_append
# 导入 numpy.exceptions 中的 VisibleDeprecationWarning 异常
from numpy.exceptions import VisibleDeprecationWarning

# 导入标准库中的 os、sys、shutil、warnings 模块
import os
import sys
import shutil
import warnings

# 定义 DistutilsBackend 类，继承自 Backend 类
class DistutilsBackend(Backend):
    # 初始化方法
    def __init__(sef, *args, **kwargs):
        # 发出 VisibleDeprecationWarning 警告，提示 distutils 已自 NumPy 1.26.x 起废弃
        warnings.warn(
            "\ndistutils has been deprecated since NumPy 1.26.x\n"
            "Use the Meson backend instead, or generate wrappers"
            " without -c and use a custom build script",
            VisibleDeprecationWarning,
            stacklevel=2,
        )
        # 调用父类 Backend 的初始化方法
        super().__init__(*args, **kwargs)

    # 编译方法
    def compile(self):
        # 初始化 num_info 字典为空
        num_info = {}
        # 如果 num_info 不为空，则将其 include_dirs 属性添加到 self.include_dirs 中
        if num_info:
            self.include_dirs.extend(num_info.get("include_dirs", []))
        
        # 构建 Extension 对象的参数字典 ext_args
        ext_args = {
            "name": self.modulename,
            "sources": self.sources,
            "include_dirs": self.include_dirs,
            "library_dirs": self.library_dirs,
            "libraries": self.libraries,
            "define_macros": self.define_macros,
            "undef_macros": self.undef_macros,
            "extra_objects": self.extra_objects,
            "f2py_options": self.f2py_flags,
        }

        # 如果 self.sysinfo_flags 不为空，则遍历 self.sysinfo_flags 列表
        if self.sysinfo_flags:
            for n in self.sysinfo_flags:
                # 调用 get_info 函数获取名称为 n 的系统信息并赋值给 i
                i = get_info(n)
                # 如果 i 为空，则打印相关信息提示找不到资源
                if not i:
                    print(
                        f"No {repr(n)} resources found"
                        "in system (try `f2py --help-link`)"
                    )
                # 将获取的系统信息 i 添加到 ext_args 字典中
                dict_append(ext_args, **i)

        # 创建 Extension 对象 ext
        ext = Extension(**ext_args)

        # 设置 sys.argv 的参数以便进行 setup 操作
        sys.argv = [sys.argv[0]] + self.setup_flags
        sys.argv.extend(
            [
                "build",
                "--build-temp",
                self.build_dir,
                "--build-base",
                self.build_dir,
                "--build-platlib",
                ".",
                "--disable-optimization",
            ]
        )

        # 如果 self.fc_flags 不为空，则将其添加到 sys.argv 中
        if self.fc_flags:
            sys.argv.extend(["config_fc"] + self.fc_flags)
        # 如果 self.flib_flags 不为空，则将其添加到 sys.argv 中
        if self.flib_flags:
            sys.argv.extend(["build_ext"] + self.flib_flags)

        # 调用 setup 函数进行模块的扩展编译，使用之前构建的 Extension 对象 ext
        setup(ext_modules=[ext])

        # 如果设置了 remove_build_dir 为 True，并且 self.build_dir 存在，则删除该目录
        if self.remove_build_dir and os.path.exists(self.build_dir):
            print(f"Removing build directory {self.build_dir}")
            shutil.rmtree(self.build_dir)
```