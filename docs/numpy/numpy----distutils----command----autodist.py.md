# `.\numpy\numpy\distutils\command\autodist.py`

```py
"""This module implements additional tests ala autoconf which can be useful.

"""
import textwrap  # 导入 textwrap 模块，用于文本缩进处理

# We put them here since they could be easily reused outside numpy.distutils

def check_inline(cmd):
    """Return the inline identifier (may be empty)."""
    cmd._check_compiler()  # 调用 cmd 对象的 _check_compiler 方法
    body = textwrap.dedent("""
        #ifndef __cplusplus
        static %(inline)s int static_func (void)
        {
            return 0;
        }
        %(inline)s int nostatic_func (void)
        {
            return 0;
        }
        #endif""")
    
    # 尝试不同的关键字来编译 body 中的代码段
    for kw in ['inline', '__inline__', '__inline']:
        st = cmd.try_compile(body % {'inline': kw}, None, None)
        if st:
            return kw

    return ''  # 如果未找到合适的关键字，则返回空字符串


def check_restrict(cmd):
    """Return the restrict identifier (may be empty)."""
    cmd._check_compiler()  # 调用 cmd 对象的 _check_compiler 方法
    body = textwrap.dedent("""
        static int static_func (char * %(restrict)s a)
        {
            return 0;
        }
        """)
    
    # 尝试不同的关键字来编译 body 中的代码段
    for kw in ['restrict', '__restrict__', '__restrict']:
        st = cmd.try_compile(body % {'restrict': kw}, None, None)
        if st:
            return kw

    return ''  # 如果未找到合适的关键字，则返回空字符串


def check_compiler_gcc(cmd):
    """Check if the compiler is GCC."""
    cmd._check_compiler()  # 调用 cmd 对象的 _check_compiler 方法
    body = textwrap.dedent("""
        int
        main()
        {
        #if (! defined __GNUC__)
        #error gcc required
        #endif
            return 0;
        }
        """)
    
    # 尝试编译 body 中的代码段来检查是否为 GCC 编译器
    return cmd.try_compile(body, None, None)


def check_gcc_version_at_least(cmd, major, minor=0, patchlevel=0):
    """
    Check that the gcc version is at least the specified version."""
    cmd._check_compiler()  # 调用 cmd 对象的 _check_compiler 方法
    version = '.'.join([str(major), str(minor), str(patchlevel)])
    body = textwrap.dedent("""
        int
        main()
        {
        #if (! defined __GNUC__) || (__GNUC__ < %(major)d) || \\
                (__GNUC_MINOR__ < %(minor)d) || \\
                (__GNUC_PATCHLEVEL__ < %(patchlevel)d)
        #error gcc >= %(version)s required
        #endif
            return 0;
        }
        """)
    kw = {'version': version, 'major': major, 'minor': minor,
          'patchlevel': patchlevel}
    
    # 尝试编译 body 中的代码段来检查 GCC 的版本是否符合要求
    return cmd.try_compile(body % kw, None, None)


def check_gcc_function_attribute(cmd, attribute, name):
    """Return True if the given function attribute is supported."""
    cmd._check_compiler()  # 调用 cmd 对象的 _check_compiler 方法
    body = textwrap.dedent("""
        #pragma GCC diagnostic error "-Wattributes"
        #pragma clang diagnostic error "-Wattributes"

        int %s %s(void* unused)
        {
            return 0;
        }

        int
        main()
        {
            return 0;
        }
        """) % (attribute, name)
    
    # 尝试编译 body 中的代码段来检查给定的函数属性是否受支持
    return cmd.try_compile(body, None, None) != 0


def check_gcc_function_attribute_with_intrinsics(cmd, attribute, name, code,
                                                include):
    """Return True if the given function attribute is supported with
    intrinsics."""
    cmd._check_compiler()  # 调用 cmd 对象的 _check_compiler 方法
    # 将字符串格式化为C语言代码，包括头文件、函数名、属性、代码
    body = textwrap.dedent("""
        #include<%s>            # 包含指定的头文件
        int %s %s(void)         # 定义名为name的函数，返回类型为attribute
        {
            %s;                 # 函数体内的代码
            return 0;           # 返回0
        }

        int                     # 定义整型函数
        main()                  # 主函数
        {
            return 0;           # 返回0
        }
        """) % (include, attribute, name, code)   # 格式化字符串，传入include, attribute, name, code
    # 使用C语言编译器尝试编译代码，返回结果是否为0
    return cmd.try_compile(body, None, None) != 0
# 检查给定的变量属性是否受支持，如果受支持则返回True
def check_gcc_variable_attribute(cmd, attribute):
    """Return True if the given variable attribute is supported."""
    # 检查编译器是否被设置
    cmd._check_compiler()
    # 定义一个包含特定变量属性的代码块
    body = textwrap.dedent("""
        #pragma GCC diagnostic error "-Wattributes"
        #pragma clang diagnostic error "-Wattributes"

        int %s foo;

        int
        main()
        {
            return 0;
        }
        """) % (attribute, )
    # 尝试编译代码块，如果编译成功返回0，否则返回非0
    return cmd.try_compile(body, None, None) != 0
```