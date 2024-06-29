# `.\numpy\numpy\linalg\lapack_lite\make_lite.py`

```
#!/usr/bin/env python2.7
# WARNING! This a Python 2 script. Read README.rst for rationale.
"""
Usage: make_lite.py <wrapped_routines_file> <lapack_dir>

Typical invocation:

    make_lite.py wrapped_routines /tmp/lapack-3.x.x

Requires the following to be on the path:
 * f2c
 * patch

"""
import sys  # 导入 sys 模块，用于访问命令行参数和系统相关功能
import os  # 导入 os 模块，提供了对操作系统进行调用的接口
import re  # 导入 re 模块，用于处理正则表达式的匹配和操作
import subprocess  # 导入 subprocess 模块，用于执行外部命令
import shutil  # 导入 shutil 模块，提供了一些高级的文件操作功能

import fortran  # 导入自定义的 fortran 模块，用于处理 Fortran 相关操作
import clapack_scrub  # 导入自定义的 clapack_scrub 模块，用于处理 Clapack 相关操作

try:
    from distutils.spawn import find_executable as which  # 尝试从 distutils.spawn 模块导入 find_executable 函数（Python 2）
except ImportError:
    from shutil import which  # 如果导入失败，则从 shutil 模块导入 which 函数（Python 3）

# Arguments to pass to f2c. You'll always want -A for ANSI C prototypes
# Others of interest: -a to not make variables static by default
#                     -C to check array subscripts
F2C_ARGS = ['-A', '-Nx800']  # 设置传递给 f2c 的参数列表

# The header to add to the top of the f2c_*.c file. Note that dlamch_() calls
# will be replaced by the macros below by clapack_scrub.scrub_source()
HEADER_BLURB = '''\
/*
 * NOTE: This is generated code. Look in numpy/linalg/lapack_lite for
 *       information on remaking this file.
 */
'''

HEADER = HEADER_BLURB + '''\
#include "f2c.h"

#ifdef HAVE_CONFIG
#include "config.h"
#else
extern doublereal dlamch_(char *);
#define EPSILON dlamch_("Epsilon")
#define SAFEMINIMUM dlamch_("Safe minimum")
#define PRECISION dlamch_("Precision")
#define BASE dlamch_("Base")
#endif

extern doublereal dlapy2_(doublereal *x, doublereal *y);

/*
f2c knows the exact rules for precedence, and so omits parentheses where not
strictly necessary. Since this is generated code, we don't really care if
it's readable, and we know what is written is correct. So don't warn about
them.
*/
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wparentheses"
#endif
'''

class FortranRoutine:
    """Wrapper for a Fortran routine in a file.
    """
    type = 'generic'

    def __init__(self, name=None, filename=None):
        self.filename = filename
        if name is None:
            root, ext = os.path.splitext(filename)
            name = root
        self.name = name
        self._dependencies = None

    def dependencies(self):
        if self._dependencies is None:
            deps = fortran.getDependencies(self.filename)
            self._dependencies = [d.lower() for d in deps]
        return self._dependencies

    def __repr__(self):
        return "FortranRoutine({!r}, filename={!r})".format(self.name,
                                                            self.filename)

class UnknownFortranRoutine(FortranRoutine):
    """Wrapper for a Fortran routine for which the corresponding file
    is not known.
    """
    type = 'unknown'

    def __init__(self, name):
        FortranRoutine.__init__(self, name=name, filename='<unknown>')

    def dependencies(self):
        return []

class FortranLibrary:
    """Container for a bunch of Fortran routines.
    """
    def __init__(self, src_dirs):
        self._src_dirs = src_dirs
        self.names_to_routines = {}
    def _findRoutine(self, rname):
        # 将输入的例程名转换为小写
        rname = rname.lower()
        # 遍历源目录列表中的每一个目录
        for s in self._src_dirs:
            # 构建可能的Fortran例程文件名
            ffilename = os.path.join(s, rname + '.f')
            # 如果文件存在，返回新创建的Fortran例程对象
            if os.path.exists(ffilename):
                return self._newFortranRoutine(rname, ffilename)
        # 如果未找到文件，返回一个未知的Fortran例程对象
        return UnknownFortranRoutine(rname)

    def _newFortranRoutine(self, rname, filename):
        # 创建并返回一个新的Fortran例程对象
        return FortranRoutine(rname, filename)

    def addIgnorableRoutine(self, rname):
        """Add a routine that we don't want to consider when looking at
        dependencies.
        """
        # 将例程名转换为小写
        rname = rname.lower()
        # 创建一个未知的Fortran例程对象
        routine = UnknownFortranRoutine(rname)
        # 将例程名和对象存储到类属性中
        self.names_to_routines[rname] = routine

    def addRoutine(self, rname):
        """Add a routine to the library.
        """
        # 调用getRoutine方法以确保例程存在于库中
        self.getRoutine(rname)

    def getRoutine(self, rname):
        """Get a routine from the library. Will add if it's not found.
        """
        # 创建一个唯一标识符列表
        unique = []
        # 将例程名转换为小写
        rname = rname.lower()
        # 从类属性中获取对应例程名的例程对象，若不存在则返回唯一标识符列表
        routine = self.names_to_routines.get(rname, unique)
        # 如果获取的是唯一标识符列表，则调用_findRoutine方法寻找并添加例程
        if routine is unique:
            routine = self._findRoutine(rname)
            self.names_to_routines[rname] = routine
        # 返回找到或者新添加的例程对象
        return routine

    def allRoutineNames(self):
        """Return the names of all the routines.
        """
        # 返回类属性中所有例程名组成的列表
        return list(self.names_to_routines.keys())

    def allRoutines(self):
        """Return all the routines.
        """
        # 返回类属性中所有例程对象组成的列表
        return list(self.names_to_routines.values())

    def resolveAllDependencies(self):
        """Try to add routines to the library to satisfy all the dependencies
        for each routine in the library.

        Returns a set of routine names that have the dependencies unresolved.
        """
        # 已处理的例程名集合
        done_this = set()
        # 上次循环未解决的例程名集合
        last_todo = set()
        while True:
            # 获取所有未处理的例程名集合
            todo = set(self.allRoutineNames()) - done_this
            # 如果当前未处理集合与上次相同，则退出循环
            if todo == last_todo:
                break
            # 遍历每一个未处理的例程名
            for rn in todo:
                # 获取该例程名对应的例程对象
                r = self.getRoutine(rn)
                # 获取该例程对象的所有依赖例程名
                deps = r.dependencies()
                # 为每一个依赖例程名添加到库中
                for d in deps:
                    self.addRoutine(d)
                # 将该例程名标记为已处理
                done_this.add(rn)
            # 更新上次未处理的例程名集合
            last_todo = todo
        # 返回未解决依赖的例程名集合
        return todo
# LapackLibrary 类，继承自 FortranLibrary 类
class LapackLibrary(FortranLibrary):
    
    # 重写父类方法 _newFortranRoutine
    def _newFortranRoutine(self, rname, filename):
        # 调用父类方法创建新的 FortranRoutine 对象
        routine = FortranLibrary._newFortranRoutine(self, rname, filename)
        
        # 根据文件名和函数名设置 routine 的 type 属性
        if 'blas' in filename.lower():
            routine.type = 'blas'
        elif 'install' in filename.lower():
            routine.type = 'config'
        elif rname.startswith('z'):
            routine.type = 'z_lapack'
        elif rname.startswith('c'):
            routine.type = 'c_lapack'
        elif rname.startswith('s'):
            routine.type = 's_lapack'
        elif rname.startswith('d'):
            routine.type = 'd_lapack'
        else:
            routine.type = 'lapack'
        
        return routine
    
# 打印描述和给定类型的所有 routines 名称
def printRoutineNames(desc, routines):
    print(desc)
    for r in routines:
        print('\t%s' % r.name)

# 根据给定的 wrapped_routines、ignores 和 lapack_dir 创建 LapackLibrary 对象
def getLapackRoutines(wrapped_routines, ignores, lapack_dir):
    # 确定 BLAS 和 LAPACK 源码目录
    blas_src_dir = os.path.join(lapack_dir, 'BLAS', 'SRC')
    if not os.path.exists(blas_src_dir):
        blas_src_dir = os.path.join(lapack_dir, 'blas', 'src')
    lapack_src_dir = os.path.join(lapack_dir, 'SRC')
    if not os.path.exists(lapack_src_dir):
        lapack_src_dir = os.path.join(lapack_dir, 'src')
    install_src_dir = os.path.join(lapack_dir, 'INSTALL')
    if not os.path.exists(install_src_dir):
        install_src_dir = os.path.join(lapack_dir, 'install')
    
    # 创建 LapackLibrary 对象
    library = LapackLibrary([install_src_dir, blas_src_dir, lapack_src_dir])
    
    # 将 ignores 中的 routine 添加到忽略列表中
    for r in ignores:
        library.addIgnorableRoutine(r)
    
    # 将 wrapped_routines 中的 routine 添加到 library 中
    for w in wrapped_routines:
        library.addRoutine(w)
    
    # 解析所有依赖关系
    library.resolveAllDependencies()
    
    return library

# 从 wrapped_routines_file 中获取 wrapped routines 和 ignores
def getWrappedRoutineNames(wrapped_routines_file):
    routines = []
    ignores = []
    with open(wrapped_routines_file) as fo:
        for line in fo:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('IGNORE:'):
                line = line[7:].strip()
                ig = line.split()
                ignores.extend(ig)
            else:
                routines.append(line)
    return routines, ignores

# 类型集合，包含了可能的 routine 类型
types = {'blas', 'lapack', 'd_lapack', 's_lapack', 'z_lapack', 'c_lapack', 'config'}

# 将 library 中每种类型的 routines 名称和依赖关系写入输出目录中的文件
def dumpRoutineNames(library, output_dir):
    for typename in {'unknown'} | types:
        routines = library.allRoutinesByType(typename)
        filename = os.path.join(output_dir, typename + '_routines.lst')
        with open(filename, 'w') as fo:
            for r in routines:
                deps = r.dependencies()
                fo.write('%s: %s\n' % (r.name, ' '.join(deps)))

# 将 routines 中的所有源文件内容合并并写入 output_file
def concatenateRoutines(routines, output_file):
    with open(output_file, 'w') as output_fo:
        for r in routines:
            with open(r.filename) as fo:
                source = fo.read()
            output_fo.write(source)
class F2CError(Exception):
    pass

# 将 Fortran 文件名中的反斜杠替换为正斜杠，以适应不同操作系统路径格式
def runF2C(fortran_filename, output_dir):
    fortran_filename = fortran_filename.replace('\\', '/')
    try:
        # 使用 subprocess 调用 f2c 转换 Fortran 文件为 C 文件，并指定输出目录
        subprocess.check_call(
            ["f2c"] + F2C_ARGS + ['-d', output_dir, fortran_filename]
        )
    except subprocess.CalledProcessError:
        # 如果调用出错，则抛出自定义异常 F2CError
        raise F2CError

# 清理指定 C 文件中的源码，通过 clapack_scrub 模块进行处理
def scrubF2CSource(c_file):
    with open(c_file) as fo:
        source = fo.read()
    # 调用 clapack_scrub 模块的 scrubSource 函数清理源码，并输出详细信息
    source = clapack_scrub.scrubSource(source, verbose=True)
    with open(c_file, 'w') as fo:
        # 将清理后的源码写回原文件，并添加标准头部 HEADER
        fo.write(HEADER)
        fo.write(source)

# 确保指定的可执行文件名在系统路径中可找到，否则抛出 SystemExit 异常
def ensure_executable(name):
    try:
        which(name)
    except Exception:
        raise SystemExit(name + ' not found')

# 创建 LAPACK 符号重命名头文件，以避免符号冲突，并加入 BLAS/LAPACK 和 f2c 的重命名
def create_name_header(output_dir):
    routine_re = re.compile(r'^      (subroutine|.* function)\s+(\w+)\(.*$',
                            re.I)
    extern_re = re.compile(r'^extern [a-z]+ ([a-z0-9_]+)\(.*$')

    # BLAS/LAPACK 符号集合
    symbols = set(['xerbla'])
    for fn in os.listdir(output_dir):
        fn = os.path.join(output_dir, fn)

        if not fn.endswith('.f'):
            continue

        with open(fn) as f:
            for line in f:
                m = routine_re.match(line)
                if m:
                    # 提取并添加符号到集合中
                    symbols.add(m.group(2).lower())

    # f2c 符号集合
    f2c_symbols = set()
    with open('f2c.h') as f:
        for line in f:
            m = extern_re.match(line)
            if m:
                # 提取并添加 f2c 符号到集合中
                f2c_symbols.add(m.group(1))

    with open(os.path.join(output_dir, 'lapack_lite_names.h'), 'w') as f:
        f.write(HEADER_BLURB)
        f.write(
            "/*\n"
            " * This file renames all BLAS/LAPACK and f2c symbols to avoid\n"
            " * dynamic symbol name conflicts, in cases where e.g.\n"
            " * integer sizes do not match with 'standard' ABI.\n"
            " */\n")

        # 重命名 BLAS/LAPACK 符号并写入文件
        for name in sorted(symbols):
            f.write("#define %s_ BLAS_FUNC(%s)\n" % (name, name))

        # 也重命名 f2c 导出的符号并写入文件
        f.write("\n"
                "/* Symbols exported by f2c.c */\n")
        for name in sorted(f2c_symbols):
            f.write("#define %s numpy_lapack_lite_%s\n" % (name, name))

# 主程序入口，执行 LAPACK 转换和符号重命名操作
def main():
    if len(sys.argv) != 3:
        # 如果命令行参数不为3个，打印帮助文档并返回
        print(__doc__)
        return
    # 确保系统能找到 patch 和 f2c 可执行文件
    ensure_executable('f2c')
    ensure_executable('patch')

    # 读取命令行参数
    wrapped_routines_file = sys.argv[1]
    lapack_src_dir = sys.argv[2]
    output_dir = os.path.join(os.path.dirname(__file__), 'build')

    # 删除旧的输出目录并创建新的输出目录
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)

    # 获取需要转换的 LAPACK 函数列表和忽略列表
    wrapped_routines, ignores = getWrappedRoutineNames(wrapped_routines_file)
    # 获取 LAPACK 函数库
    library = getLapackRoutines(wrapped_routines, ignores, lapack_src_dir)

    # 将 LAPACK 函数信息导出到指定的输出目录
    dumpRoutineNames(library, output_dir)
    # 遍历类型列表中的每个类型名
    for typename in types:
        # 构建对应的 Fortran 文件路径
        fortran_file = os.path.join(output_dir, 'f2c_%s.f' % typename)
        # 根据 Fortran 文件路径生成对应的 C 文件路径
        c_file = fortran_file[:-2] + '.c'
        # 打印正在创建的 C 文件名
        print('creating %s ...' % c_file)
        # 获取特定类型的所有例程
        routines = library.allRoutinesByType(typename)
        # 将所有例程连接起来并写入 Fortran 文件
        concatenateRoutines(routines, fortran_file)

        # 应用补丁文件
        patch_file = os.path.basename(fortran_file) + '.patch'
        # 如果补丁文件存在，则使用 subprocess 调用 patch 命令进行打补丁操作
        if os.path.exists(patch_file):
            subprocess.check_call(['patch', '-u', fortran_file, patch_file])
            print("Patched {}".format(fortran_file))

        try:
            # 尝试运行 f2c 工具转换 Fortran 文件为 C 文件
            runF2C(fortran_file, output_dir)
        except F2CError:
            # 如果转换失败，则打印错误信息
            print('f2c failed on %s' % fortran_file)
            break

        # 清理生成的 C 文件的源代码
        scrubF2CSource(c_file)

        # 检查并应用 C 文件的补丁
        c_patch_file = os.path.basename(c_file) + '.patch'
        if os.path.exists(c_patch_file):
            subprocess.check_call(['patch', '-u', c_file, c_patch_file])

        # 打印空行，用于分隔不同文件处理的输出信息
        print()

    # 创建名称头文件
    create_name_header(output_dir)

    # 遍历输出目录中的文件
    for fname in os.listdir(output_dir):
        # 复制所有以 '.c' 结尾的文件或特定的头文件 'lapack_lite_names.h'
        if fname.endswith('.c') or fname == 'lapack_lite_names.h':
            print('Copying ' + fname)
            # 将文件复制到当前脚本所在目录
            shutil.copy(
                os.path.join(output_dir, fname),
                os.path.abspath(os.path.dirname(__file__)),
            )
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 调用主函数 main()
    main()
```