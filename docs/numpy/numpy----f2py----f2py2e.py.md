# `.\numpy\numpy\f2py\f2py2e.py`

```
#!/usr/bin/env python3
"""

f2py2e - Fortran to Python C/API generator. 2nd Edition.
         See __usage__ below.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
# 导入必要的标准库模块
import sys              # 导入 sys 模块，提供对系统参数和功能的访问
import os               # 导入 os 模块，提供了对操作系统进行调用的接口
import pprint           # 导入 pprint 模块，用于打印 Python 数据结构，美观打印字典和列表等
import re               # 导入 re 模块，提供正则表达式的支持
from pathlib import Path   # 导入 pathlib 中的 Path 类，用于处理文件和目录路径
from itertools import dropwhile   # 导入 itertools 中的 dropwhile 函数，用于迭代直到条件为假
import argparse         # 导入 argparse 模块，用于命令行参数解析
import copy             # 导入 copy 模块，用于对象的浅拷贝和深拷贝

# 导入自定义模块
from . import crackfortran
from . import rules
from . import cb_rules
from . import auxfuncs
from . import cfuncs
from . import f90mod_rules
from . import __version__
from . import capi_maps
from numpy.f2py._backends import f2py_build_generator

# 设置 f2py_version 和 numpy_version 变量为当前版本
f2py_version = __version__.version
numpy_version = __version__.version

# 定义错误信息输出函数
errmess = sys.stderr.write
# 定义显示函数，使用 pprint.pprint 实现
show = pprint.pprint
# 定义标准输出信息函数
outmess = auxfuncs.outmess

# 判断 Python 版本是否符合要求，设置 MESON_ONLY_VER 为布尔值
MESON_ONLY_VER = (sys.version_info >= (3, 12))

# 定义程序用法说明字符串
__usage__ =\
f"""Usage:

1) To construct extension module sources:

      f2py [<options>] <fortran files> [[[only:]||[skip:]] \\
                                        <fortran functions> ] \\
                                       [: <fortran files> ...]

2) To compile fortran files and build extension modules:

      f2py -c [<options>, <build_flib options>, <extra options>] <fortran files>

3) To generate signature files:

      f2py -h <filename.pyf> ...< same options as in (1) >

Description: This program generates a Python C/API file (<modulename>module.c)
             that contains wrappers for given fortran functions so that they
             can be called from Python. With the -c option the corresponding
             extension modules are built.
"""
Options:

# -h <filename>    Write signatures of the fortran routines to file <filename>
#                  and exit. You can then edit <filename> and use it instead
#                  of <fortran files>. If <filename>==stdout then the
#                  signatures are printed to stdout.

# <fortran functions>  Names of fortran routines for which Python C/API
#                      functions will be generated. Default is all that are found
#                      in <fortran files>.

# <fortran files>  Paths to fortran/signature files that will be scanned for
#                  <fortran functions> in order to determine their signatures.

# skip:            Ignore fortran functions that follow until `:'.

# only:            Use only fortran functions that follow until `:'.

# :                Get back to <fortran files> mode.

# -m <modulename>  Name of the module; f2py generates a Python/C API
#                  file <modulename>module.c or extension module <modulename>.
#                  Default is 'untitled'.

# '-include<header>'  Writes additional headers in the C wrapper, can be passed
#                     multiple times, generates #include <header> each time.

# --[no-]lower     Do [not] lower the cases in <fortran files>. By default,
#                  --lower is assumed with -h key, and --no-lower without -h key.

# --build-dir <dirname>  All f2py generated files are created in <dirname>.
#                  Default is tempfile.mkdtemp().

# --overwrite-signature  Overwrite existing signature file.

# --[no-]latex-doc Create (or not) <modulename>module.tex.
#                  Default is --no-latex-doc.

# --short-latex    Create 'incomplete' LaTeX document (without commands
#                  \\documentclass, \\tableofcontents, and \\begin{{document}},
#                  \\end{{document}}).

# --[no-]rest-doc Create (or not) <modulename>module.rst.
#                  Default is --no-rest-doc.

# --debug-capi     Create C/API code that reports the state of the wrappers
#                  during runtime. Useful for debugging.

# --[no-]wrap-functions    Create Fortran subroutine wrappers to Fortran 77
#                  functions. --wrap-functions is default because it ensures
#                  maximum portability/compiler independence.

# --include-paths <path1>:<path2>:...   Search include files from the given
#                  directories.

# --help-link [..] List system resources found by system_info.py. See also
#                  --link-<resource> switch below. [..] is optional list
#                  of resources names. E.g. try 'f2py --help-link lapack_opt'.

# --f2cmap <filename>  Load Fortran-to-Python KIND specification from the given
#                  file. Default: .f2py_f2cmap in current directory.

# --quiet          Run quietly.

# --verbose        Run with extra verbosity.

# --skip-empty-wrappers   Only generate wrapper files when needed.

# -v               Print f2py version ID and exit.
# 构建后端选项（仅对使用 -c 有效）
# [NO_MESON] 用于指示不应与 meson 后端或 Python 3.12 以上版本一起使用的选项：

# --fcompiler=         指定 Fortran 编译器类型按供应商分类 [NO_MESON]
# --compiler=          指定 distutils C 编译器类型 [NO_MESON]

# --help-fcompiler     列出可用的 Fortran 编译器并退出 [NO_MESON]
# --f77exec=           指定 F77 编译器的路径 [NO_MESON]
# --f90exec=           指定 F90 编译器的路径 [NO_MESON]
# --f77flags=          指定 F77 编译器的编译选项
# --f90flags=          指定 F90 编译器的编译选项
# --opt=               指定优化选项 [NO_MESON]
# --arch=              指定与体系结构相关的优化选项 [NO_MESON]
# --noopt              编译时不使用优化 [NO_MESON]
# --noarch             编译时不使用与体系结构相关的优化 [NO_MESON]
# --debug              使用调试信息进行编译

# --dep                <dependency>
#                      为模块指定 meson 依赖项。可以多次传递此选项以指定多个依赖项。
#                      依赖项将存储在一个列表中以供进一步处理。

#                      示例: --dep lapack --dep scalapack
#                      这将识别 "lapack" 和 "scalapack" 作为依赖项，并从 argv 中删除它们，留下一个包含 ["lapack", "scalapack"] 的依赖项列表。

# --backend            <backend_type>
#                      指定编译过程的构建后端类型。支持的后端包括 'meson' 和 'distutils'。
#                      如果未指定，默认为 'distutils'。在 Python 3.12 或更高版本中，默认为 'meson'。

# 额外选项（仅在使用 -c 有效）：

# --link-<resource>    将扩展模块与 numpy.distutils/system_info.py 中定义的资源链接起来。
#                      例如，要链接优化的 LAPACK 库（在 MacOSX 上是 vecLib，在其他地方是 ATLAS），使用 --link-lapack_opt。
#                      参见 --help-link 开关。 [NO_MESON]

# -L/path/to/lib/ -l<libname>
# -D<define> -U<name>
# -I/path/to/include/
# <filename>.o <filename>.so <filename>.a

# 对于非 gcc Fortran 编译器，可能需要使用以下宏：
#   -DPREPEND_FORTRAN -DNO_APPEND_FORTRAN -DUPPERCASE_FORTRAN

# 使用 -DF2PY_REPORT_ATEXIT 时，在退出时打印 F2PY 接口的性能报告（平台：Linux）。

# 使用 -DF2PY_REPORT_ON_ARRAY_COPY=<int> 时，每当 F2PY 接口复制数组时，将向 stderr 发送一条消息。
# 整数 <int> 设置了在应显示消息的数组大小阈值。

Version:     {f2py_version}
numpy Version: {numpy_version}
License:     NumPy license (see LICENSE.txt in the NumPy source code)
Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
# 定义函数 scaninputline，用于处理输入行，返回多个变量的初始值
def scaninputline(inputline):
    # 初始化空列表，用于存储文件名、跳过函数名、仅包含函数名、调试信息
    files, skipfuncs, onlyfuncs, debug = [], [], [], []
    # 初始化多个标志变量和计数变量
    f, f2, f3, f5, f6, f8, f9, f10 = 1, 0, 0, 0, 0, 0, 0, 0
    # 设定详细输出标志为1
    verbose = 1
    # 设定空生成器标志为真
    emptygen = True
    # 设定 dolc 变量为 -1
    dolc = -1
    # 设定 dolatexdoc 和 dorestdoc 变量为 0
    dolatexdoc = 0
    dorestdoc = 0
    # 设定 wrapfuncs 变量为 1
    wrapfuncs = 1
    # 设定 buildpath 变量为当前路径字符串
    buildpath = '.'
    # 调用 get_includes 函数，获取包含路径列表和更新的 inputline
    include_paths, inputline = get_includes(inputline)
    # 初始化 signsfile 和 modulename 变量为 None
    signsfile, modulename = None, None
    # 初始化 options 字典，包含 buildpath 键和空值，coutput 键和空值，f2py_wrapper_output 键和空值
    options = {'buildpath': buildpath,
               'coutput': None,
               'f2py_wrapper_output': None}
    # 遍历输入的每一行
    for l in inputline:
        # 如果当前行为空字符串，则跳过
        if l == '':
            pass
        # 如果当前行为'only:'，设置标志位f为0
        elif l == 'only:':
            f = 0
        # 如果当前行为'skip:'，设置标志位f为-1
        elif l == 'skip:':
            f = -1
        # 如果当前行为':'，设置标志位f为1
        elif l == ':':
            f = 1
        # 如果当前行以'--debug-'开头，将后面的内容添加到debug列表中
        elif l[:8] == '--debug-':
            debug.append(l[8:])
        # 如果当前行为'--lower'，设置dolc标志为1
        elif l == '--lower':
            dolc = 1
        # 如果当前行为'--build-dir'，设置f6标志为1
        elif l == '--build-dir':
            f6 = 1
        # 如果当前行为'--no-lower'，设置dolc标志为0
        elif l == '--no-lower':
            dolc = 0
        # 如果当前行为'--quiet'，设置verbose标志为0
        elif l == '--quiet':
            verbose = 0
        # 如果当前行为'--verbose'，增加verbose计数
        elif l == '--verbose':
            verbose += 1
        # 如果当前行为'--latex-doc'，设置dolatexdoc标志为1
        elif l == '--latex-doc':
            dolatexdoc = 1
        # 如果当前行为'--no-latex-doc'，设置dolatexdoc标志为0
        elif l == '--no-latex-doc':
            dolatexdoc = 0
        # 如果当前行为'--rest-doc'，设置dorestdoc标志为1
        elif l == '--rest-doc':
            dorestdoc = 1
        # 如果当前行为'--no-rest-doc'，设置dorestdoc标志为0
        elif l == '--no-rest-doc':
            dorestdoc = 0
        # 如果当前行为'--wrap-functions'，设置wrapfuncs标志为1
        elif l == '--wrap-functions':
            wrapfuncs = 1
        # 如果当前行为'--no-wrap-functions'，设置wrapfuncs标志为0
        elif l == '--no-wrap-functions':
            wrapfuncs = 0
        # 如果当前行为'--short-latex'，设置options字典中的'shortlatex'为1
        elif l == '--short-latex':
            options['shortlatex'] = 1
        # 如果当前行为'--coutput'，设置f8标志为1
        elif l == '--coutput':
            f8 = 1
        # 如果当前行为'--f2py-wrapper-output'，设置f9标志为1
        elif l == '--f2py-wrapper-output':
            f9 = 1
        # 如果当前行为'--f2cmap'，设置f10标志为1
        elif l == '--f2cmap':
            f10 = 1
        # 如果当前行为'--overwrite-signature'，设置options字典中的'h-overwrite'为1
        elif l == '--overwrite-signature':
            options['h-overwrite'] = 1
        # 如果当前行为'-h'，设置f2标志为1
        elif l == '-h':
            f2 = 1
        # 如果当前行为'-m'，设置f3标志为1
        elif l == '-m':
            f3 = 1
        # 如果当前行以'-v'开头，打印f2py_version并退出程序
        elif l[:2] == '-v':
            print(f2py_version)
            sys.exit()
        # 如果当前行为'--show-compilers'，设置f5标志为1
        elif l == '--show-compilers':
            f5 = 1
        # 如果当前行以'-include'开头，将该行内容添加到相应列表和字典中
        elif l[:8] == '-include':
            cfuncs.outneeds['userincludes'].append(l[9:-1])
            cfuncs.userincludes[l[9:-1]] = '#include ' + l[8:]
        # 如果当前行为'--skip-empty-wrappers'，设置emptygen为False
        elif l == '--skip-empty-wrappers':
            emptygen = False
        # 如果当前行以'-'开头但不符合以上任何规则，打印错误消息并退出程序
        elif l[0] == '-':
            errmess('Unknown option %s\n' % repr(l))
            sys.exit()
        # 如果f2标志为真，将当前行作为文件名存入files列表，f2标志置为0
        elif f2:
            f2 = 0
            signsfile = l
        # 如果f3标志为真，将当前行作为模块名存入modulename，f3标志置为0
        elif f3:
            f3 = 0
            modulename = l
        # 如果f6标志为真，将当前行作为构建路径存入buildpath，f6标志置为0
        elif f6:
            f6 = 0
            buildpath = l
        # 如果f8标志为真，将当前行作为选项字典中的'coutput'存入options，f8标志置为0
        elif f8:
            f8 = 0
            options["coutput"] = l
        # 如果f9标志为真，将当前行作为选项字典中的'f2py_wrapper_output'存入options，f9标志置为0
        elif f9:
            f9 = 0
            options["f2py_wrapper_output"] = l
        # 如果f10标志为真，将当前行作为选项字典中的'f2cmap_file'存入options，f10标志置为0
        elif f10:
            f10 = 0
            options["f2cmap_file"] = l
        # 如果f为1，尝试打开当前行对应的文件并将其加入files列表，如果出现OSError则打印错误消息
        elif f == 1:
            try:
                with open(l):
                    pass
                files.append(l)
            except OSError as detail:
                errmess(f'OSError: {detail!s}. Skipping file "{l!s}".\n')
        # 如果f为-1，将当前行加入skipfuncs列表
        elif f == -1:
            skipfuncs.append(l)
    # 如果f5标志为假且files和modulename都为空，则打印__usage__并退出程序
    if not f5 and not files and not modulename:
        print(__usage__)
        sys.exit()
    # 如果buildpath不是一个目录，且verbose标志为假，则输出创建buildpath的信息
    if not os.path.isdir(buildpath):
        if not verbose:
            outmess('Creating build directory %s\n' % (buildpath))
        # 创建buildpath目录
        os.mkdir(buildpath)
    # 如果signsfile有值，则将其路径设置为buildpath和signsfile的组合路径
    if signsfile:
        signsfile = os.path.join(buildpath, signsfile)
    # 检查是否提供了签名文件路径，并且该路径指向一个文件，同时选项中没有 'h-overwrite'
    if signsfile and os.path.isfile(signsfile) and 'h-overwrite' not in options:
        # 如果满足条件，则输出错误信息并退出程序
        errmess(
            'Signature file "%s" exists!!! Use --overwrite-signature to overwrite.\n' % (signsfile))
        sys.exit()

    # 将函数选项传递给选项字典
    options['emptygen'] = emptygen
    options['debug'] = debug
    options['verbose'] = verbose

    # 根据 dolc 变量的值设置 'do-lower' 选项，如果 dolc 为 -1 并且没有提供签名文件，则设置为 0
    if dolc == -1 and not signsfile:
        options['do-lower'] = 0
    else:
        options['do-lower'] = dolc

    # 如果提供了模块名称，则将其添加到选项字典中的 'module' 键
    if modulename:
        options['module'] = modulename

    # 如果提供了签名文件路径，则将其添加到选项字典中的 'signsfile' 键
    if signsfile:
        options['signsfile'] = signsfile

    # 如果提供了 onlyfuncs，则将其添加到选项字典中的 'onlyfuncs' 键
    if onlyfuncs:
        options['onlyfuncs'] = onlyfuncs

    # 如果提供了 skipfuncs，则将其添加到选项字典中的 'skipfuncs' 键
    if skipfuncs:
        options['skipfuncs'] = skipfuncs

    # 将 dolatexdoc 的值设置为选项字典中的 'dolatexdoc' 键
    options['dolatexdoc'] = dolatexdoc

    # 将 dorestdoc 的值设置为选项字典中的 'dorestdoc' 键
    options['dorestdoc'] = dorestdoc

    # 将 wrapfuncs 的值设置为选项字典中的 'wrapfuncs' 键
    options['wrapfuncs'] = wrapfuncs

    # 将 buildpath 的值设置为选项字典中的 'buildpath' 键
    options['buildpath'] = buildpath

    # 将 include_paths 的值设置为选项字典中的 'include_paths' 键
    options['include_paths'] = include_paths

    # 设置 'f2cmap_file' 键的默认值为 None，如果已经存在则不改变
    options.setdefault('f2cmap_file', None)

    # 返回 files 和 options 作为函数的结果
    return files, options
# 定义一个函数 `callcrackfortran`，用于调用 crackfortran 进行处理，接受文件列表和选项参数。
def callcrackfortran(files, options):
    # 设置全局变量 rules.options 为传入的选项参数
    rules.options = options
    # 设置 crackfortran 的调试模式为 options 中的 debug 值
    crackfortran.debug = options['debug']
    # 设置 crackfortran 的详细输出模式为 options 中的 verbose 值
    crackfortran.verbose = options['verbose']
    # 如果 options 中包含 'module'，则设置 crackfortran.f77modulename 为对应值
    if 'module' in options:
        crackfortran.f77modulename = options['module']
    # 如果 options 中包含 'skipfuncs'，则设置 crackfortran.skipfuncs 为对应值
    if 'skipfuncs' in options:
        crackfortran.skipfuncs = options['skipfuncs']
    # 如果 options 中包含 'onlyfuncs'，则设置 crackfortran.onlyfuncs 为对应值
    if 'onlyfuncs' in options:
        crackfortran.onlyfuncs = options['onlyfuncs']
    # 设置 crackfortran.include_paths 为 options 中的 include_paths 列表
    crackfortran.include_paths[:] = options['include_paths']
    # 设置 crackfortran.dolowercase 为 options 中的 do-lower 值
    crackfortran.dolowercase = options['do-lower']
    # 调用 crackfortran.crackfortran 处理文件列表，返回处理后的列表 postlist
    postlist = crackfortran.crackfortran(files)
    # 如果 options 中包含 'signsfile'
    if 'signsfile' in options:
        # 输出保存签名信息到文件的提示信息
        outmess('Saving signatures to file "%s"\n' % (options['signsfile']))
        # 调用 crackfortran.crack2fortran 处理 postlist，返回处理后的结果 pyf
        pyf = crackfortran.crack2fortran(postlist)
        # 如果 options['signsfile'] 的后六个字符为 'stdout'
        if options['signsfile'][-6:] == 'stdout':
            # 将处理结果 pyf 输出到标准输出
            sys.stdout.write(pyf)
        else:
            # 将处理结果 pyf 写入到 options['signsfile'] 指定的文件中
            with open(options['signsfile'], 'w') as f:
                f.write(pyf)
    # 如果 options["coutput"] 为 None
    if options["coutput"] is None:
        # 遍历 postlist 中的每个模块 mod
        for mod in postlist:
            # 设置 mod 的 "coutput" 属性为 "%smodule.c" % mod["name"]
            mod["coutput"] = "%smodule.c" % mod["name"]
    else:
        # 否则，遍历 postlist 中的每个模块 mod
        for mod in postlist:
            # 设置 mod 的 "coutput" 属性为 options["coutput"]
            mod["coutput"] = options["coutput"]
    # 如果 options["f2py_wrapper_output"] 为 None
    if options["f2py_wrapper_output"] is None:
        # 遍历 postlist 中的每个模块 mod
        for mod in postlist:
            # 设置 mod 的 "f2py_wrapper_output" 属性为 "%s-f2pywrappers.f" % mod["name"]
            mod["f2py_wrapper_output"] = "%s-f2pywrappers.f" % mod["name"]
    else:
        # 否则，遍历 postlist 中的每个模块 mod
        for mod in postlist:
            # 设置 mod 的 "f2py_wrapper_output" 属性为 options["f2py_wrapper_output"]
            mod["f2py_wrapper_output"] = options["f2py_wrapper_output"]
    # 返回处理后的模块列表 postlist
    return postlist


# 定义一个函数 buildmodules，用于构建模块，接受模块列表作为参数 lst
def buildmodules(lst):
    # 调用 cfuncs 模块的 buildcfuncs 函数
    cfuncs.buildcfuncs()
    # 输出建立模块的提示信息
    outmess('Building modules...\n')
    # 初始化模块、模块名称列表和使用关系字典
    modules, mnames, isusedby = [], [], {}
    # 遍历传入的模块列表 lst 中的每个元素 item
    for item in lst:
        # 如果 item 的 'name' 属性中包含 '__user__'
        if '__user__' in item['name']:
            # 调用 cb_rules 模块的 buildcallbacks 函数处理该 item
            cb_rules.buildcallbacks(item)
        else:
            # 否则，如果 item 中包含 'use' 属性
            if 'use' in item:
                # 遍历 item['use'] 字典的键 u
                for u in item['use'].keys():
                    # 如果 u 不在 isusedby 字典中，则将其初始化为一个空列表
                    if u not in isusedby:
                        isusedby[u] = []
                    # 将 item['name'] 加入到 u 在 isusedby 中对应的列表中
                    isusedby[u].append(item['name'])
            # 将 item 加入到模块列表 modules 中
            modules.append(item)
            # 将 item['name'] 加入到模块名称列表 mnames 中
            mnames.append(item['name'])
    # 初始化返回字典 ret
    ret = {}
    # 遍历模块列表 modules 和模块名称列表 mnames 中对应位置的元素 module 和 name
    for module, name in zip(modules, mnames):
        # 如果 name 在 isusedby 中
        if name in isusedby:
            # 输出跳过模块 name 的提示信息，说明它被 isusedby[name] 使用
            outmess('\tSkipping module "%s" which is used by %s.\n' % (
                name, ','.join('"%s"' % s for s in isusedby[name])))
        else:
            # 否则，创建一个空列表 um
            um = []
            # 如果 module 中包含 'use' 属性
            if 'use' in module:
                # 遍历 module['use'] 字典的键 u
                for u in module['use'].keys():
                    # 如果 u 在 isusedby 中，并且 u 在 mnames 中
                    if u in isusedby and u in mnames:
                        # 将 modules[mnames.index(u)] 加入到 um 中
                        um.append(modules[mnames.index(u)])
                    else:
                        # 否则，输出模块 name 使用了不存在的 u 的提示信息
                        outmess(
                            f'\tModule "{name}" uses nonexisting "{u}" '
                            'which will be ignored.\n')
            # 将 rules.buildmodule 处理 module 和 um 后的结果添加到 ret[name] 中
            ret[name] = {}
            dict_append(ret[name], rules.buildmodule(module, um))
    # 返回处理后的模块字典 ret
    return ret


# 定义一个函数 dict_append，用于向目标字典 d_out 中追加源字典 d_in 的内容
def dict_append(d_out, d_in):
    # 遍历源字典 d_in 的键值对 (k, v)
    for (k, v) in d_in.items():
        # 如果目标字典 d_out 中不存在键 k，则初始化一个空列表
        if k not in d_out:
            d_out[k] = []
        # 如果 v 是列表，则将其与 d_out[k] 合并
        if isinstance(v, list):
            d_out[k] = d_out[k] + v
        else:
            # 否则，将 v 添加到 d_out[k] 中
            d_out[k].append(v)


# 定义一个函数 run_main，用于模拟运行 f2py 命令，接受命令行参数列表 comline_list
def run_main(comline_list):
    """
    Equivalent to running::

        f2py <args>
    """
    crackfortran.reset_global_f2py_vars()
    # 重置全局的 f2py 变量，清理之前的状态

    f2pydir = os.path.dirname(os.path.abspath(cfuncs.__file__))
    # 获取当前文件 cfuncs 的绝对路径所在目录名

    fobjhsrc = os.path.join(f2pydir, 'src', 'fortranobject.h')
    # 构建头文件 fortranobject.h 的路径，位于 f2pydir/src 目录下

    fobjcsrc = os.path.join(f2pydir, 'src', 'fortranobject.c')
    # 构建源文件 fortranobject.c 的路径，位于 f2pydir/src 目录下

    # gh-22819 -- begin
    parser = make_f2py_compile_parser()
    # 创建 f2py 编译的解析器对象

    args, comline_list = parser.parse_known_args(comline_list)
    # 使用解析器解析传入的命令行参数列表 comline_list，并获取解析后的参数和剩余的命令行列表

    pyf_files, _ = filter_files("", "[.]pyf([.]src|)", comline_list)
    # 过滤出 comline_list 中的 ".pyf" 文件，并忽略匹配的扩展名 ".src"

    # Checks that no existing modulename is defined in a pyf file
    # TODO: Remove all this when scaninputline is replaced

    if args.module_name:
        if "-h" in comline_list:
            modname = (
                args.module_name
            )  # 当命令行包含 "-h" 选项时，直接使用 args 中的模块名
        else:
            modname = validate_modulename(
                pyf_files, args.module_name
            )  # 当命令行不包含 "-h" 选项时，验证模块名的有效性
        comline_list += ['-m', modname]  # 添加模块名参数到命令行列表，供后续扫描输入行使用

    # gh-22819 -- end

    files, options = scaninputline(comline_list)
    # 使用扫描输入行函数处理命令行列表，获取文件列表和选项字典

    auxfuncs.options = options
    # 设置辅助函数的选项参数

    capi_maps.load_f2cmap_file(options['f2cmap_file'])
    # 加载指定的 f2c 映射文件

    postlist = callcrackfortran(files, options)
    # 调用 crackfortran 处理文件列表和选项，返回处理后的列表

    isusedby = {}
    # 初始化用于存储模块使用信息的字典

    for plist in postlist:
        if 'use' in plist:
            for u in plist['use'].keys():
                if u not in isusedby:
                    isusedby[u] = []
                isusedby[u].append(plist['name'])
    # 遍历处理后的列表，记录模块的使用关系

    for plist in postlist:
        if plist['block'] == 'python module' and '__user__' in plist['name']:
            if plist['name'] in isusedby:
                # if not quiet:
                outmess(
                    f'Skipping Makefile build for module "{plist["name"]}" '
                    'which is used by {}\n'.format(
                        ','.join(f'"{s}"' for s in isusedby[plist['name']])))
    # 遍历处理后的列表，对于 Python 模块且被其他模块使用的情况，输出跳过构建的信息

    if 'signsfile' in options:
        if options['verbose'] > 1:
            outmess(
                'Stopping. Edit the signature file and then run f2py on the signature file: ')
            outmess('%s %s\n' %
                    (os.path.basename(sys.argv[0]), options['signsfile']))
        return
    # 如果选项中包含签名文件，且详细模式大于 1，则输出停止信息，并返回
    # 对于 postlist 中的每个 plist 元素进行迭代
    for plist in postlist:
        # 如果 plist 的 'block' 属性不是 'python module'
        if plist['block'] != 'python module':
            # 如果 options 中不包含 'python module' 选项
            if 'python module' not in options:
                # 打印错误消息，提示用户如何处理 Fortran 源代码
                errmess('Tip: If your original code is Fortran source then you must use -m option.\n')
            # 抛出类型错误，说明所有的块必须是 'python module' 块，但得到了不同的类型
            raise TypeError('All blocks must be python module blocks but got %s' % (repr(plist['block'])))

    # 设置 auxfuncs.debugoptions 为 options 中的 'debug' 值
    auxfuncs.debugoptions = options['debug']
    # 设置 f90mod_rules.options 为 options 对象
    f90mod_rules.options = options
    # 设置 auxfuncs.wrapfuncs 为 options 中的 'wrapfuncs' 值
    auxfuncs.wrapfuncs = options['wrapfuncs']

    # 调用 buildmodules 函数，将 postlist 作为参数，获取返回值并赋给 ret
    ret = buildmodules(postlist)

    # 对于 ret 中的每个模块名 mn，向其对应的字典中添加 'csrc' 和 'h' 两个键，值分别为 fobjcsrc 和 fobjhsrc
    for mn in ret.keys():
        dict_append(ret[mn], {'csrc': fobjcsrc, 'h': fobjhsrc})
    
    # 返回 ret 字典
    return ret
def filter_files(prefix, suffix, files, remove_prefix=None):
    """
    Filter files by prefix and suffix.

    prefix: str, prefix string to match at the beginning of filenames.
    suffix: str, suffix string to match at the end of filenames.
    files: list of str, list of filenames to filter.
    remove_prefix: str or None, if provided, the prefix to remove from filtered filenames.

    Returns:
    filtered: list of str, filenames that match the prefix and suffix criteria.
    rest: list of str, filenames that do not match the criteria.
    """
    filtered, rest = [], []
    # Regular expression pattern to match filenames starting with `prefix` and ending with `suffix`
    match = re.compile(prefix + r'.*' + suffix + r'\Z').match
    if remove_prefix:
        ind = len(prefix)
    else:
        ind = 0
    for file in [x.strip() for x in files]:
        if match(file):
            # Append the filename without the prefix (if `remove_prefix` is specified)
            filtered.append(file[ind:])
        else:
            # Append filenames that do not match the pattern
            rest.append(file)
    return filtered, rest


def get_prefix(module):
    """
    Retrieve the parent directory of the parent directory of the module file.

    module: module object, the module whose parent directories need to be retrieved.

    Returns:
    p: str, parent directory path.
    """
    p = os.path.dirname(os.path.dirname(module.__file__))
    return p


class CombineIncludePaths(argparse.Action):
    """
    Custom action to combine multiple include paths into a set.

    Appends each new include path to the existing set of include paths.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        include_paths_set = set(getattr(namespace, 'include_paths', []) or [])
        if option_string == "--include_paths":
            outmess("Use --include-paths or -I instead of --include_paths which will be removed")
        if option_string == "--include-paths" or option_string == "--include_paths":
            include_paths_set.update(values.split(':'))
        else:
            include_paths_set.add(values)
        setattr(namespace, 'include_paths', list(include_paths_set))


def include_parser():
    """
    Create an ArgumentParser for handling include paths.

    Returns:
    parser: ArgumentParser object configured for include paths.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-I", dest="include_paths", action=CombineIncludePaths)
    parser.add_argument("--include-paths", dest="include_paths", action=CombineIncludePaths)
    parser.add_argument("--include_paths", dest="include_paths", action=CombineIncludePaths)
    return parser


def get_includes(iline):
    """
    Parse a list of command-line arguments to extract include paths.

    iline: list of str, command-line arguments.

    Returns:
    ipaths: list of str, include paths extracted from the command-line.
    remain: list of str, remaining command-line arguments after parsing include paths.
    """
    iline = (' '.join(iline)).split()
    parser = include_parser()
    args, remain = parser.parse_known_args(iline)
    ipaths = args.include_paths
    if args.include_paths is None:
        ipaths = []
    return ipaths, remain


def make_f2py_compile_parser():
    """
    Create an ArgumentParser for parsing f2py compilation options.

    Returns:
    parser: ArgumentParser object configured for f2py compilation options.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dep", action="append", dest="dependencies")
    parser.add_argument("--backend", choices=['meson', 'distutils'], default='distutils')
    parser.add_argument("-m", dest="module_name")
    return parser


def preparse_sysargv():
    """
    Prepare sys.argv for f2py compilation.

    Parses sys.argv using a predefined parser and adjusts it accordingly.
    """
    # To keep backwards bug compatibility, newer flags are handled by argparse,
    # and `sys.argv` is passed to the rest of `f2py` as is.
    parser = make_f2py_compile_parser()

    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv

    backend_key = args.backend
    if MESON_ONLY_VER and backend_key == 'distutils':
        outmess("Cannot use distutils backend with Python>=3.12,"
                " using meson backend instead.\n")
        backend_key = "meson"

    return {
        "dependencies": args.dependencies or [],
        "backend": backend_key,
        "modulename": args.module_name,
    }


def run_compile():
    """
    Perform all necessary actions for f2py compilation in one function call.

    Collects dependency flags, preprocesses sys.argv, and retrieves the module name.
    """
    import tempfile

    # Collect dependency flags, preprocess sys.argv
    argy = preparse_sysargv()
    modulename = argy["modulename"]
    # 如果模块名为 None，则将其设置为 'untitled'
    if modulename is None:
        modulename = 'untitled'

    # 从 argy 字典中获取依赖项列表
    dependencies = argy["dependencies"]

    # 从 argy 字典中获取后端关键字
    backend_key = argy["backend"]

    # 使用 backend_key 调用 f2py_build_generator 函数，生成构建后端对象
    build_backend = f2py_build_generator(backend_key)

    # 找到并删除命令行参数列表中的 '-c' 参数
    i = sys.argv.index('-c')
    del sys.argv[i]

    # 初始化 remove_build_dir 变量为 0
    remove_build_dir = 0

    # 尝试找到 '--build-dir' 参数在命令行参数列表中的位置
    try:
        i = sys.argv.index('--build-dir')
    except ValueError:
        i = None

    # 根据 '--build-dir' 参数的存在与否，设置 build_dir 和相应的命令行参数
    if i is not None:
        build_dir = sys.argv[i + 1]
        del sys.argv[i + 1]
        del sys.argv[i]
    else:
        # 如果未指定 '--build-dir' 参数，则创建临时目录并设置为 build_dir
        remove_build_dir = 1
        build_dir = tempfile.mkdtemp()

    # 使用正则表达式 _reg1 匹配 '--link-' 开头的参数，存储在 sysinfo_flags 中
    _reg1 = re.compile(r'--link-')
    sysinfo_flags = [_m for _m in sys.argv[1:] if _reg1.match(_m)]

    # 从 sys.argv 中移除 sysinfo_flags 中包含的参数
    sys.argv = [_m for _m in sys.argv if _m not in sysinfo_flags]

    # 如果 sysinfo_flags 不为空，去除每个元素中的 '--link-' 前缀
    if sysinfo_flags:
        sysinfo_flags = [f[7:] for f in sysinfo_flags]

    # 使用正则表达式 _reg2 匹配一系列参数，存储在 f2py_flags 中，并从 sys.argv 中移除它们
    _reg2 = re.compile(
        r'--((no-|)(wrap-functions|lower)|debug-capi|quiet|skip-empty-wrappers)|-include')
    f2py_flags = [_m for _m in sys.argv[1:] if _reg2.match(_m)]
    sys.argv = [_m for _m in sys.argv if _m not in f2py_flags]

    # 初始化 f2py_flags2 和 fl 变量
    f2py_flags2 = []
    fl = 0

    # 遍历 sys.argv 中的每个参数
    for a in sys.argv[1:]:
        # 如果 a 在 ['only:', 'skip:'] 中，则将 fl 设置为 1
        if a in ['only:', 'skip:']:
            fl = 1
        # 如果 a 为 ':', 则将 fl 设置为 0
        elif a == ':':
            fl = 0
        # 如果 fl 为真或者 a 为 ':', 则将 a 添加到 f2py_flags2 列表中
        if fl or a == ':':
            f2py_flags2.append(a)

    # 如果 f2py_flags2 不为空且最后一个元素不是 ':', 则添加 ':'
    if f2py_flags2 and f2py_flags2[-1] != ':':
        f2py_flags2.append(':')

    # 将 f2py_flags2 的内容添加到 f2py_flags 中
    f2py_flags.extend(f2py_flags2)

    # 从 sys.argv 中移除 f2py_flags2 中包含的参数
    sys.argv = [_m for _m in sys.argv if _m not in f2py_flags2]

    # 使用正则表达式 _reg3 匹配一系列参数，存储在 flib_flags 中，并从 sys.argv 中移除它们
    _reg3 = re.compile(
        r'--((f(90)?compiler(-exec|)|compiler)=|help-compiler)')
    flib_flags = [_m for _m in sys.argv[1:] if _reg3.match(_m)]
    sys.argv = [_m for _m in sys.argv if _m not in flib_flags]

    # 定义正则表达式以匹配特定的 f77 和 f90 参数
    reg_f77_f90_flags = re.compile(r'--f(77|90)flags=')
    reg_distutils_flags = re.compile(r'--((f(77|90)exec|opt|arch)=|(debug|noopt|noarch|help-fcompiler))')

    # 从 sys.argv 中筛选出 fc_flags 和 distutils_flags
    fc_flags = [_m for _m in sys.argv[1:] if reg_f77_f90_flags.match(_m)]
    distutils_flags = [_m for _m in sys.argv[1:] if reg_distutils_flags.match(_m)]

    # 如果 MESON_ONLY_VER 为假且 backend_key 不为 'meson'，则将 distutils_flags 添加到 fc_flags 中
    if not (MESON_ONLY_VER or backend_key == 'meson'):
        fc_flags.extend(distutils_flags)

    # 从 sys.argv 中移除 fc_flags 和 distutils_flags
    sys.argv = [_m for _m in sys.argv if _m not in (fc_flags + distutils_flags)]

    # 初始化 del_list 列表，用于存储待删除的参数
    del_list = []
    for s in flib_flags:
        # 定义用于识别编译器选项的前缀字符串
        v = '--fcompiler='
        # 如果当前字符串 s 的开头是 v，则进入条件
        if s[:len(v)] == v:
            # 如果 MESON_ONLY_VER 为真或者 backend_key 为 'meson'，则输出提示信息
            if MESON_ONLY_VER or backend_key == 'meson':
                outmess(
                    "--fcompiler cannot be used with meson,"
                    "set compiler with the FC environment variable\n"
                    )
            else:
                # 导入 numpy.distutils 中的 fcompiler 模块
                from numpy.distutils import fcompiler
                # 加载所有的编译器类
                fcompiler.load_all_fcompiler_classes()
                # 获取所有已知编译器的键列表
                allowed_keys = list(fcompiler.fcompiler_class.keys())
                # 将 s 后面的部分转换为小写，并赋值给 nv 和 ov
                nv = ov = s[len(v):].lower()
                # 如果 ov 不在允许的键列表中
                if ov not in allowed_keys:
                    vmap = {}  # XXX
                    # 尝试使用 vmap 进行映射转换
                    try:
                        nv = vmap[ov]
                    except KeyError:
                        # 如果 ov 不在 vmap 的值列表中，则打印未知厂商信息
                        if ov not in vmap.values():
                            print('Unknown vendor: "%s"' % (s[len(v):]))
                    # 若没有找到映射，则 nv 保持 ov 不变
                    nv = ov
                # 找到 s 在 flib_flags 中的索引 i
                i = flib_flags.index(s)
                # 更新 flib_flags 中的元素为新的编译器选项
                flib_flags[i] = '--fcompiler=' + nv
                # 继续处理下一个 flib_flags 元素
                continue
    for s in del_list:
        # 找到 s 在 flib_flags 中的索引 i
        i = flib_flags.index(s)
        # 从 flib_flags 中删除该元素
        del flib_flags[i]
    # 断言 flib_flags 的长度不超过 2，用于验证条件
    assert len(flib_flags) <= 2, repr(flib_flags)

    _reg5 = re.compile(r'--(verbose)')
    # 从 sys.argv 中筛选出符合正则表达式 _reg5 的参数，放入 setup_flags
    setup_flags = [_m for _m in sys.argv[1:] if _reg5.match(_m)]
    # 从 sys.argv 中移除 setup_flags 中的参数
    sys.argv = [_m for _m in sys.argv if _m not in setup_flags]

    if '--quiet' in f2py_flags:
        # 如果 '--quiet' 在 f2py_flags 中，则添加到 setup_flags
        setup_flags.append('--quiet')

    # 丑陋的过滤器，用于移除除了源文件外的所有内容
    sources = sys.argv[1:]
    f2cmapopt = '--f2cmap'
    # 如果 f2cmapopt 在 sys.argv 中
    if f2cmapopt in sys.argv:
        # 找到 f2cmapopt 在 sys.argv 中的索引 i
        i = sys.argv.index(f2cmapopt)
        # 将 f2py_flags 扩展为包含 f2cmapopt 及其后续参数
        f2py_flags.extend(sys.argv[i:i + 2])
        # 删除 sys.argv 中的 f2cmapopt 及其后续参数
        del sys.argv[i + 1], sys.argv[i]
        # 更新 sources 为处理后的 sys.argv 中的源文件列表
        sources = sys.argv[1:]

    # 过滤出 pyf 文件和其他源文件
    pyf_files, _sources = filter_files("", "[.]pyf([.]src|)", sources)
    # 合并 pyf_files 和 _sources 到 sources
    sources = pyf_files + _sources
    # 验证模块名是否有效，返回有效的模块名
    modulename = validate_modulename(pyf_files, modulename)
    # 过滤出额外的目标文件和源文件
    extra_objects, sources = filter_files('', '[.](o|a|so|dylib)', sources)
    # 过滤出库目录和源文件
    library_dirs, sources = filter_files('-L', '', sources, remove_prefix=1)
    # 过滤出库名称和源文件
    libraries, sources = filter_files('-l', '', sources, remove_prefix=1)
    # 过滤出未定义的宏和源文件
    undef_macros, sources = filter_files('-U', '', sources, remove_prefix=1)
    # 过滤出定义的宏和源文件
    define_macros, sources = filter_files('-D', '', sources, remove_prefix=1)
    # 遍历所有的定义宏，将形如 name=value 的字符串拆分成元组
    for i in range(len(define_macros)):
        name_value = define_macros[i].split('=', 1)
        # 如果只有名字没有值，则将值设为 None
        if len(name_value) == 1:
            name_value.append(None)
        # 如果有名字和值，则将其作为元组赋值给 define_macros[i]
        if len(name_value) == 2:
            define_macros[i] = tuple(name_value)
        else:
            print('Invalid use of -D:', name_value)

    # 构建包装器、签名或其他所需内容
    if backend_key == 'meson':
        # 如果使用 meson 后端
        if not pyf_files:
            # 如果没有 pyf 文件，则输出使用 meson 后端的信息并设置 f2py_flags
            outmess('Using meson backend\nWill pass --lower to f2py\nSee https://numpy.org/doc/stable/f2py/buildtools/meson.html\n')
            f2py_flags.append('--lower')
            # 运行主函数，传递适当的参数给 f2py
            run_main(f" {' '.join(f2py_flags)} -m {modulename} {' '.join(sources)}".split())
        else:
            # 如果有 pyf 文件，则运行主函数，传递适当的参数给 f2py
            run_main(f" {' '.join(f2py_flags)} {' '.join(pyf_files)}".split())
    # 获取源文件的包含目录和源文件列表，用于后续构建
    include_dirs, sources = get_includes(sources)
    
    # 使用获取到的参数创建后端构建器对象
    builder = build_backend(
        modulename,           # 模块名称
        sources,              # 源文件列表
        extra_objects,        # 额外的对象文件列表
        build_dir,            # 构建目录
        include_dirs,         # 包含目录列表
        library_dirs,         # 库文件目录列表
        libraries,            # 库文件列表
        define_macros,        # 宏定义列表
        undef_macros,         # 宏未定义列表
        f2py_flags,           # f2py 标志
        sysinfo_flags,        # 系统信息标志
        fc_flags,             # fc 标志
        flib_flags,           # flib 标志
        setup_flags,          # 设置标志
        remove_build_dir,     # 是否移除构建目录的标志
        {"dependencies": dependencies},  # 依赖项字典
    )
    
    # 编译构建器对象
    builder.compile()
# 验证模块名称是否有效，可以根据需要重置模块名称
def validate_modulename(pyf_files, modulename='untitled'):
    # 如果传入的 .pyf 文件数量大于1，抛出数值错误异常
    if len(pyf_files) > 1:
        raise ValueError("Only one .pyf file per call")
    
    # 如果有 .pyf 文件
    if pyf_files:
        # 获取第一个 .pyf 文件的路径
        pyff = pyf_files[0]
        # 调用 auxfuncs 模块中的 get_f2py_modulename 函数获取模块名
        pyf_modname = auxfuncs.get_f2py_modulename(pyff)
        
        # 如果指定的模块名与 .pyf 文件中获取的模块名不一致
        if modulename != pyf_modname:
            # 输出警告信息，指出正在忽略指定的模块名，并显示 .pyf 文件中的实际模块名
            outmess(
                f"Ignoring -m {modulename}.\n"
                f"{pyff} defines {pyf_modname} to be the modulename.\n"
            )
            # 使用 .pyf 文件中获取的模块名作为新的模块名
            modulename = pyf_modname
    
    # 返回最终确认的模块名
    return modulename

# 主函数，根据命令行参数执行不同的操作
def main():
    # 如果命令行参数中包含 '--help-link'
    if '--help-link' in sys.argv[1:]:
        # 移除 '--help-link' 参数
        sys.argv.remove('--help-link')
        
        # 如果 MESON_ONLY_VER 为真，输出提示信息告知使用 '--dep' 用于 Meson 构建
        if MESON_ONLY_VER:
            outmess("Use --dep for meson builds\n")
        else:
            # 否则导入 numpy.distutils.system_info 模块，并调用 show_all 函数显示所有系统信息
            from numpy.distutils.system_info import show_all
            show_all()
        
        # 函数返回，结束执行
        return
    
    # 如果命令行参数中包含 '-c'
    if '-c' in sys.argv[1:]:
        # 运行编译函数
        run_compile()
    else:
        # 否则运行主函数，传入剩余的命令行参数（除了程序名以外的参数）
        run_main(sys.argv[1:])
```