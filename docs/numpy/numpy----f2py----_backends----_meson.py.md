# `.\numpy\numpy\f2py\_backends\_meson.py`

```
    # 从 __future__ 导入 annotations 特性，用于支持注解类型提示
    from __future__ import annotations

    # 导入操作系统相关模块
    import os
    # 导入 errno 模块，用于处理错误码
    import errno
    # 导入 shutil 模块，用于文件操作
    import shutil
    # 导入 subprocess 模块，用于执行外部命令
    import subprocess
    # 导入 sys 模块，提供对 Python 解释器的访问
    import sys
    # 导入 re 模块，用于正则表达式操作
    import re
    # 导入 pathlib 中的 Path 类
    from pathlib import Path

    # 从 _backend 模块导入 Backend 类
    from ._backend import Backend
    # 导入 string 模块中的 Template 类，用于字符串模板替换
    from string import Template
    # 导入 itertools 模块中的 chain 函数，用于迭代工具函数
    from itertools import chain

    # 导入 warnings 模块，用于处理警告信息

    # MesonTemplate 类，用于生成 Meson 构建文件模板
    class MesonTemplate:
        """Template meson build file generation class."""

        # 初始化方法，接收多个参数用于设置模板的各项属性
        def __init__(
            self,
            modulename: str,         # 模块名
            sources: list[Path],     # 源文件路径列表
            deps: list[str],         # 依赖模块列表
            libraries: list[str],    # 库列表
            library_dirs: list[Path],# 库目录列表
            include_dirs: list[Path],# 包含目录列表
            object_files: list[Path],# 目标文件列表
            linker_args: list[str],  # 链接器参数列表
            fortran_args: list[str], # Fortran 编译器参数列表
            build_type: str,         # 构建类型
            python_exe: str,         # Python 解释器路径
        ):
            # 初始化各属性
            self.modulename = modulename
            # 设置构建模板文件路径
            self.build_template_path = (
                Path(__file__).parent.absolute() / "meson.build.template"
            )
            self.sources = sources          # 源文件列表
            self.deps = deps                # 依赖模块列表
            self.libraries = libraries      # 库列表
            self.library_dirs = library_dirs# 库目录列表
            # 如果 include_dirs 不为 None，则使用给定列表，否则为空列表
            if include_dirs is not None:
                self.include_dirs = include_dirs
            else:
                self.include_dirs = []
            self.substitutions = {}         # 模板替换字典
            self.objects = object_files     # 目标文件列表

            # 将 Fortran 编译器参数转换为 Meson 需要的格式
            self.fortran_args = [
                f"'{x}'" if not (x.startswith("'") and x.endswith("'")) else x
                for x in fortran_args
            ]

            # 定义模板处理管道，按顺序调用各个替换方法
            self.pipeline = [
                self.initialize_template,
                self.sources_substitution,
                self.deps_substitution,
                self.include_substitution,
                self.libraries_substitution,
                self.fortran_args_substitution,
            ]

            self.build_type = build_type     # 构建类型
            self.python_exe = python_exe     # Python 解释器路径
            self.indent = " " * 21           # 缩进字符串

        # 生成 Meson 构建模板文件内容的方法，返回字符串
        def meson_build_template(self) -> str:
            # 检查构建模板文件是否存在，如果不存在则抛出 FileNotFoundError 异常
            if not self.build_template_path.is_file():
                raise FileNotFoundError(
                    errno.ENOENT,
                    "Meson build template"
                    f" {self.build_template_path.absolute()}"
                    " does not exist.",
                )
            # 读取并返回构建模板文件的文本内容
            return self.build_template_path.read_text()

        # 初始化模板替换字典的方法
        def initialize_template(self) -> None:
            self.substitutions["modulename"] = self.modulename  # 设置模块名替换项
            self.substitutions["buildtype"] = self.build_type   # 设置构建类型替换项
            self.substitutions["python"] = self.python_exe      # 设置 Python 解释器路径替换项

        # 替换源文件列表的方法
        def sources_substitution(self) -> None:
            self.substitutions["source_list"] = ",\n".join(
                [f"{self.indent}'''{source}'''," for source in self.sources]
            )

        # 替换依赖模块列表的方法
        def deps_substitution(self) -> None:
            self.substitutions["dep_list"] = f",\n{self.indent}".join(
                [f"{self.indent}dependency('{dep}')," for dep in self.deps]
            )
    # 定义一个方法用于处理库目录的替换
    def libraries_substitution(self) -> None:
        # 将每个库目录生成声明依赖的字符串，并连接成一个换行分隔的字符串
        self.substitutions["lib_dir_declarations"] = "\n".join(
            [
                f"lib_dir_{i} = declare_dependency(link_args : ['''-L{lib_dir}'''])"
                for i, lib_dir in enumerate(self.library_dirs)
            ]
        )

        # 将每个库生成声明依赖的字符串，并连接成一个换行分隔的字符串
        self.substitutions["lib_declarations"] = "\n".join(
            [
                f"{lib.replace('.','_')} = declare_dependency(link_args : ['-l{lib}'])"
                for lib in self.libraries
            ]
        )

        # 生成库列表的字符串，每个库名替换点号为下划线，并以缩进开始每行
        self.substitutions["lib_list"] = f"\n{self.indent}".join(
            [f"{self.indent}{lib.replace('.','_')}," for lib in self.libraries]
        )
        
        # 生成库目录列表的字符串，以缩进开始每行
        self.substitutions["lib_dir_list"] = f"\n{self.indent}".join(
            [f"{self.indent}lib_dir_{i}," for i in range(len(self.library_dirs))]
        )

    # 定义一个方法用于处理包含文件路径的替换
    def include_substitution(self) -> None:
        # 生成包含文件路径列表的字符串，每个路径以三引号括起来，并以缩进开始每行
        self.substitutions["inc_list"] = f",\n{self.indent}".join(
            [f"{self.indent}'''{inc}'''," for inc in self.include_dirs]
        )

    # 定义一个方法用于处理 Fortran 编译参数的替换
    def fortran_args_substitution(self) -> None:
        # 如果有 Fortran 编译参数，生成对应的字符串；否则为空字符串
        if self.fortran_args:
            self.substitutions["fortran_args"] = (
                f"{self.indent}fortran_args: [{', '.join([arg for arg in self.fortran_args])}],"
            )
        else:
            self.substitutions["fortran_args"] = ""

    # 定义一个方法用于生成 Meson 构建文件
    def generate_meson_build(self):
        # 执行管道中每个节点的操作
        for node in self.pipeline:
            node()
        
        # 使用 Meson 构建模板创建模板对象
        template = Template(self.meson_build_template())
        # 根据替换字典替换模板中的变量
        meson_build = template.substitute(self.substitutions)
        # 去除生成字符串中可能存在的连续逗号
        meson_build = re.sub(r",,", ",", meson_build)
        # 返回最终生成的 Meson 构建文件内容
        return meson_build
# 定义 MesonBackend 类，继承自 Backend 类
class MesonBackend(Backend):
    # 初始化方法，接收任意位置参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用父类 Backend 的初始化方法
        super().__init__(*args, **kwargs)
        # 从 extra_dat 中获取依赖列表，默认为空列表
        self.dependencies = self.extra_dat.get("dependencies", [])
        # 设置 Meson 的构建目录为 "bbdir"
        self.meson_build_dir = "bbdir"
        # 确定构建类型是 "debug" 还是 "release"，根据 self.fc_flags 中是否包含 "debug" 标志决定
        self.build_type = (
            "debug" if any("debug" in flag for flag in self.fc_flags) else "release"
        )
        # 处理编译标志，调用 _get_flags 函数对 self.fc_flags 进行处理
        self.fc_flags = _get_flags(self.fc_flags)

    # 将生成的可执行文件移动到根目录下的私有方法
    def _move_exec_to_root(self, build_dir: Path):
        # 设置遍历目录为 build_dir 下的 self.meson_build_dir 目录
        walk_dir = Path(build_dir) / self.meson_build_dir
        # 构建文件路径迭代器，包含所有与 self.modulename 匹配的 .so 和 .pyd 文件
        path_objects = chain(
            walk_dir.glob(f"{self.modulename}*.so"),
            walk_dir.glob(f"{self.modulename}*.pyd"),
        )
        # 遍历所有文件路径对象
        for path_object in path_objects:
            # 设置目标路径为当前工作目录下的 path_object 的文件名
            dest_path = Path.cwd() / path_object.name
            # 如果目标路径存在，则删除
            if dest_path.exists():
                dest_path.unlink()
            # 复制 path_object 到 dest_path
            shutil.copy2(path_object, dest_path)
            # 删除原始文件 path_object
            os.remove(path_object)

    # 写入 Meson 构建文件的方法，接收构建目录参数 build_dir，并返回 None
    def write_meson_build(self, build_dir: Path) -> None:
        """Writes the meson build file at specified location"""
        # 创建 MesonTemplate 对象，生成 Meson 构建文件内容
        meson_template = MesonTemplate(
            self.modulename,
            self.sources,
            self.dependencies,
            self.libraries,
            self.library_dirs,
            self.include_dirs,
            self.extra_objects,
            self.flib_flags,
            self.fc_flags,
            self.build_type,
            sys.executable,
        )
        # 创建构建目录 build_dir（如果不存在），确保父目录存在
        Path(build_dir).mkdir(parents=True, exist_ok=True)
        # 构建 Meson 构建文件路径
        meson_build_file = Path(build_dir) / "meson.build"
        # 将生成的 Meson 构建文件内容写入到 meson_build_file
        meson_build_file.write_text(src)
        # 返回生成的 Meson 构建文件路径 meson_build_file
        return meson_build_file

    # 运行子进程命令的私有方法，接收命令和工作目录参数
    def _run_subprocess_command(self, command, cwd):
        # 在指定工作目录 cwd 下运行命令 subprocess.run，检查返回状态
        subprocess.run(command, cwd=cwd, check=True)

    # 运行 Meson 构建系统的方法，接收构建目录参数 build_dir
    def run_meson(self, build_dir: Path):
        # 设置 Meson 的 setup 命令，初始化 self.meson_build_dir 目录
        setup_command = ["meson", "setup", self.meson_build_dir]
        # 调用 _run_subprocess_command 方法运行 setup_command，设置工作目录为 build_dir
        self._run_subprocess_command(setup_command, build_dir)
        # 设置 Meson 的 compile 命令，编译 self.meson_build_dir 目录下的项目
        compile_command = ["meson", "compile", "-C", self.meson_build_dir]
        # 调用 _run_subprocess_command 方法运行 compile_command，设置工作目录为 build_dir
        self._run_subprocess_command(compile_command, build_dir)

    # 编译方法，返回 None
    def compile(self) -> None:
        # 准备源文件列表，调用 _prepare_sources 函数，更新 self.sources
        self.sources = _prepare_sources(self.modulename, self.sources, self.build_dir)
        # 写入 Meson 构建文件到 self.build_dir
        self.write_meson_build(self.build_dir)
        # 运行 Meson 构建系统，设置工作目录为 self.build_dir
        self.run_meson(self.build_dir)
        # 移动生成的可执行文件到根目录
        self._move_exec_to_root(self.build_dir)


# 准备源文件的函数，接收模块名 mname、源文件列表 sources 和构建目录 bdir
def _prepare_sources(mname, sources, bdir):
    # 复制原始源文件列表到 extended_sources
    extended_sources = sources.copy()
    # 创建构建目录 bdir（如果不存在），确保父目录存在
    Path(bdir).mkdir(parents=True, exist_ok=True)
    # 遍历源文件列表 sources
    for source in sources:
        # 如果源文件路径 source 存在且为文件
        if Path(source).exists() and Path(source).is_file():
            # 将源文件复制到构建目录 bdir
            shutil.copy(source, bdir)
    # 生成的源文件列表
    generated_sources = [
        Path(f"{mname}module.c"),
        Path(f"{mname}-f2pywrappers2.f90"),
        Path(f"{mname}-f2pywrappers.f"),
    ]
    # 设置构建目录为 Path 对象 bdir
    bdir = Path(bdir)
    # 遍历生成的源文件列表
    for generated_source in generated_sources:
        # 检查生成的源文件是否存在
        if generated_source.exists():
            # 将生成的源文件复制到目标目录中，保留原文件名
            shutil.copy(generated_source, bdir / generated_source.name)
            # 将复制后的文件名添加到扩展源文件列表中
            extended_sources.append(generated_source.name)
            # 删除原始的生成源文件
            generated_source.unlink()

    # 从扩展源文件列表中提取文件名，同时过滤掉后缀为“.pyf”的文件
    extended_sources = [
        Path(source).name
        for source in extended_sources
        if not Path(source).suffix == ".pyf"
    ]

    # 返回经过筛选和处理的扩展源文件名列表
    return extended_sources
# 定义一个函数 _get_flags，用于从给定的 fc_flags 列表中提取标志值并返回一个去重后的列表
def _get_flags(fc_flags):
    # 初始化一个空列表，用于存储提取出的标志值
    flag_values = []
    
    # 定义一个正则表达式模式，用于匹配形如 "--f77flags=xxx" 或 "--f90flags=xxx" 的标志
    flag_pattern = re.compile(r"--f(77|90)flags=(.*)")
    
    # 遍历 fc_flags 列表中的每一个标志
    for flag in fc_flags:
        # 尝试在当前标志中匹配 flag_pattern
        match_result = flag_pattern.match(flag)
        
        # 如果匹配成功
        if match_result:
            # 获取匹配结果中第二个分组（即标志值部分），去除首尾空格后按空格分割，然后去除每个元素的引号
            values = match_result.group(2).strip().split()
            values = [val.strip("'\"") for val in values]
            
            # 将处理后的标志值列表合并到 flag_values 中
            flag_values.extend(values)
    
    # 使用字典的方式来去除重复的标志值，从而保持标志值的顺序
    unique_flags = list(dict.fromkeys(flag_values))
    
    # 返回去重后的标志值列表
    return unique_flags
```