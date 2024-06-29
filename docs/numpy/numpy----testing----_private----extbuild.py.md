# `.\numpy\numpy\testing\_private\extbuild.py`

```py
# 导入需要的库和模块
import os
import pathlib
import subprocess
import sys
import sysconfig
import textwrap

# 定义模块中公开的函数和变量列表
__all__ = ['build_and_import_extension', 'compile_extension_module']

# 定义函数，用于构建并导入 C 扩展模块
def build_and_import_extension(
        modname, functions, *, prologue="", build_dir=None,
        include_dirs=[], more_init=""):
    """
    Build and imports a c-extension module `modname` from a list of function
    fragments `functions`.

    Parameters
    ----------
    functions : list of fragments
        Each fragment is a sequence of func_name, calling convention, snippet.
    prologue : string
        Code to precede the rest, usually extra ``#include`` or ``#define``
        macros.
    build_dir : pathlib.Path
        Where to build the module, usually a temporary directory
    include_dirs : list
        Extra directories to find include files when compiling
    more_init : string
        Code to appear in the module PyMODINIT_FUNC

    Returns
    -------
    out: module
        The module will have been loaded and is ready for use

    Examples
    --------
    >>> functions = [("test_bytes", "METH_O", \"\"\"
        if ( !PyBytesCheck(args)) {
            Py_RETURN_FALSE;
        }
        Py_RETURN_TRUE;
    \"\"\")]
    >>> mod = build_and_import_extension("testme", functions)
    >>> assert not mod.test_bytes('abc')
    >>> assert mod.test_bytes(b'abc')
    """
    # 构建函数体
    body = prologue + _make_methods(functions, modname)
    # 构建初始化代码
    init = """PyObject *mod = PyModule_Create(&moduledef);
           """
    # 如果未指定构建目录，使用当前目录
    if not build_dir:
        build_dir = pathlib.Path('.')
    # 如果有额外的初始化代码，添加到初始化部分
    if more_init:
        init += """#define INITERROR return NULL
                """
        init += more_init
    init += "\nreturn mod;"
    # 生成源代码字符串
    source_string = _make_source(modname, init, body)
    
    try:
        # 编译扩展模块
        mod_so = compile_extension_module(
            modname, build_dir, include_dirs, source_string)
    except Exception as e:
        # 编译失败时抛出异常
        raise RuntimeError(f"could not compile in {build_dir}:") from e
    
    # 导入编译后的模块
    import importlib.util
    spec = importlib.util.spec_from_file_location(modname, mod_so)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo


def compile_extension_module(
        name, builddir, include_dirs,
        source_string, libraries=[], library_dirs=[]):
    """
    Build an extension module and return the filename of the resulting
    native code file.

    Parameters
    ----------
    name : string
        name of the module, possibly including dots if it is a module inside a
        package.
    builddir : pathlib.Path
        Where to build the module, usually a temporary directory
    include_dirs : list
        Extra directories to find include files when compiling
    libraries : list
        Libraries to link into the extension module
    """
    # 定义一个文档字符串，描述了变量 `library_dirs` 的含义
    library_dirs: list
        Where to find the libraries, ``-L`` passed to the linker
    """
    # 根据模块名获取最后一个点号后的部分作为模块名
    modname = name.split('.')[-1]
    # 使用 `builddir` 和模块名构建目录路径，并确保该目录存在
    dirname = builddir / name
    dirname.mkdir(exist_ok=True)
    # 将源代码字符串转换为文件，并获取返回的文件对象
    cfile = _convert_str_to_file(source_string, dirname)
    # 将系统 Python 配置的包含目录添加到 `include_dirs` 列表中
    include_dirs = include_dirs + [sysconfig.get_config_var('INCLUDEPY')]

    # 调用 `_c_compile` 函数编译 C 文件
    return _c_compile(
        cfile, outputfilename=dirname / modname,
        include_dirs=include_dirs, libraries=[], library_dirs=[],
        )
# 创建一个名为 `source.c` 的文件，其中包含了参数 `source` 中的字符串内容，存放在 `dirname` 目录下。返回文件名。
def _convert_str_to_file(source, dirname):
    filename = dirname / 'source.c'  # 构建文件名路径对象
    with filename.open('w') as f:
        f.write(str(source))  # 将字符串写入文件
    return filename  # 返回生成的文件名路径对象


# 将给定的函数列表 `functions` 中的函数名、标志、代码转换为完整的函数，并在 `methods_table` 中列出。然后将 `methods_table` 转换为 `PyMethodDef` 结构体，返回准备好编译的代码片段。
def _make_methods(functions, modname):
    methods_table = []  # 用于存放方法表的列表
    codes = []  # 用于存放函数代码的列表
    for funcname, flags, code in functions:
        cfuncname = "%s_%s" % (modname, funcname)  # 构建C风格的函数名
        if 'METH_KEYWORDS' in flags:
            signature = '(PyObject *self, PyObject *args, PyObject *kwargs)'  # 方法签名，包含关键字参数
        else:
            signature = '(PyObject *self, PyObject *args)'  # 方法签名，不包含关键字参数
        methods_table.append(
            "{\"%s\", (PyCFunction)%s, %s}," % (funcname, cfuncname, flags))  # 将方法信息添加到方法表
        func_code = """
        static PyObject* {cfuncname}{signature}
        {{
        {code}
        }}
        """.format(cfuncname=cfuncname, signature=signature, code=code)  # 构建函数的C代码
        codes.append(func_code)  # 将函数代码添加到列表中

    # 构建方法表的C代码片段，并包含在最终的C代码中
    body = "\n".join(codes) + """
    static PyMethodDef methods[] = {
    %(methods)s
    { NULL }
    };
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "%(modname)s",  /* m_name */
        NULL,           /* m_doc */
        -1,             /* m_size */
        methods,        /* m_methods */
    };
    """ % dict(methods='\n'.join(methods_table), modname=modname)
    return body  # 返回构建好的C代码片段


# 将各个代码片段组合成准备编译的源代码。
def _make_source(name, init, body):
    code = """
    #include <Python.h>

    %(body)s

    PyMODINIT_FUNC
    PyInit_%(name)s(void) {
    %(init)s
    }
    """ % dict(
        name=name, init=init, body=body,
    )  # 构建源代码的C代码

    return code  # 返回准备编译的源代码


# 编译C文件 `cfile`，并将输出文件命名为 `outputfilename`，支持指定的包含目录 `include_dirs`、库文件 `libraries`、库目录 `library_dirs`。
def _c_compile(cfile, outputfilename, include_dirs=[], libraries=[],
               library_dirs=[]):
    if sys.platform == 'win32':
        compile_extra = ["/we4013"]  # 针对Windows平台的额外编译选项
        link_extra = ["/LIBPATH:" + os.path.join(sys.base_prefix, 'libs')]  # 针对Windows平台的额外链接选项
    elif sys.platform.startswith('linux'):
        compile_extra = [
            "-O0", "-g", "-Werror=implicit-function-declaration", "-fPIC"]  # 针对Linux平台的额外编译选项
        link_extra = []  # 针对Linux平台的额外链接选项
    else:
        compile_extra = link_extra = []  # 其他平台的编译和链接选项为空
        pass

    if sys.platform == 'win32':
        link_extra = link_extra + ['/DEBUG']  # 如果是Windows平台，则生成.pdb文件的调试信息
    if sys.platform == 'darwin':
        # 支持Fink和Darwinports的额外处理
        for s in ('/sw/', '/opt/local/'):
            if (s + 'include' not in include_dirs
                    and os.path.exists(s + 'include')):
                include_dirs.append(s + 'include')
            if s + 'lib' not in library_dirs and os.path.exists(s + 'lib'):
                library_dirs.append(s + 'lib')
    # 将输出文件名更改为带有操作系统特定后缀的新文件名
    outputfilename = outputfilename.with_suffix(get_so_suffix())
    
    # 调用 build 函数编译生成目标文件
    build(
        cfile, outputfilename,  # 编译的源文件和目标输出文件名
        compile_extra, link_extra,  # 编译和链接的额外参数
        include_dirs, libraries, library_dirs  # 包含目录、库文件和库目录
    )
    
    # 返回更新后的输出文件名作为函数的结果
    return outputfilename
# 使用 meson 工具编译构建一个模块
def build(cfile, outputfilename, compile_extra, link_extra,
          include_dirs, libraries, library_dirs):
    "use meson to build"

    # 创建构建目录，如果目录不存在则创建
    build_dir = cfile.parent / "build"
    os.makedirs(build_dir, exist_ok=True)

    # 获取输出文件名的最后部分，作为共享对象的名称
    so_name = outputfilename.parts[-1]

    # 在 cfile 的父目录下创建 meson.build 文件，用于配置 meson 构建系统
    with open(cfile.parent / "meson.build", "wt") as fid:
        # 将 include_dirs 转换成 -I 参数形式的列表
        includes = ['-I' + d for d in include_dirs]
        # 将 library_dirs 转换成 -L 参数形式的列表
        link_dirs = ['-L' + d for d in library_dirs]
        # 写入 meson.build 文件内容，使用 textwrap.dedent 进行格式化
        fid.write(textwrap.dedent(f"""\
            project('foo', 'c')
            shared_module('{so_name}', '{cfile.parts[-1]}',
                c_args: {includes} + {compile_extra},
                link_args: {link_dirs} + {link_extra},
                link_with: {libraries},
                name_prefix: '',
                name_suffix: 'dummy',
            )
        """))

    # 根据操作系统类型执行不同的构建命令
    if sys.platform == "win32":
        subprocess.check_call(["meson", "setup",
                               "--buildtype=release", 
                               "--vsenv", ".."],
                              cwd=build_dir,
                              )
    else:
        subprocess.check_call(["meson", "setup", "--vsenv", ".."],
                              cwd=build_dir
                              )

    # 在构建目录中执行编译命令
    subprocess.check_call(["meson", "compile"], cwd=build_dir)

    # 重命名生成的共享对象文件，去掉 .dummy 后缀
    os.rename(str(build_dir / so_name) + ".dummy", cfile.parent / so_name)
        
# 获取共享对象文件的后缀名
def get_so_suffix():
    ret = sysconfig.get_config_var('EXT_SUFFIX')
    assert ret
    return ret
```