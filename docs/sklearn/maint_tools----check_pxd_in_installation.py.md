# `D:\src\scipysrc\scikit-learn\maint_tools\check_pxd_in_installation.py`

```
# 导入必要的库和模块
import os                      # 导入操作系统相关的功能
import pathlib                 # 导入处理路径相关的功能
import subprocess              # 导入执行外部命令的功能
import sys                     # 导入与 Python 解释器交互的功能
import tempfile                # 导入创建临时文件和目录的功能
import textwrap                # 导入文本格式化的功能

# 从命令行参数中获取 scikit-learn 安装目录路径
sklearn_dir = pathlib.Path(sys.argv[1])

# 查找所有的 .pxd 文件并存储在列表中
pxd_files = list(sklearn_dir.glob("**/*.pxd"))

# 打印找到的 .pxd 文件列表
print("> Found pxd files:")
for pxd_file in pxd_files:
    print(" -", pxd_file)

# 尝试编译一个 Cython 扩展，该扩展 cimport 所有对应的模块
print("\n> Trying to compile a cython extension cimporting all corresponding modules\n")

# 使用临时目录进行操作
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = pathlib.Path(tmpdir)

    # 创建一个 Cython 测试文件，该文件 cimport 所有对应的 .pxd 文件中的模块
    with open(tmpdir / "tst.pyx", "w") as f:
        for pxd_file in pxd_files:
            to_import = str(pxd_file.relative_to(sklearn_dir))
            to_import = to_import.replace(os.path.sep, ".")
            to_import = to_import.replace(".pxd", "")
            f.write("cimport sklearn." + to_import + "\n")

    # 创建一个基本的 setup 文件用于构建测试文件
    # 我们设置语言为 c++，并且使用 numpy.get_include() 来包含必要的模块
    with open(tmpdir / "setup_tst.py", "w") as f:
        f.write(
            textwrap.dedent(
                """
                from setuptools import setup, Extension
                from Cython.Build import cythonize
                import numpy

                extensions = [Extension("tst",
                                        sources=["tst.pyx"],
                                        language="c++",
                                        include_dirs=[numpy.get_include()])]

                setup(ext_modules=cythonize(extensions))
                """
            )
        )

    # 在临时目录中运行 setup.py 构建扩展模块
    subprocess.run(
        ["python", "setup_tst.py", "build_ext", "-i"], check=True, cwd=tmpdir
    )

    # 打印编译成功的信息
    print("\n> Compilation succeeded !")
```