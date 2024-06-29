# `.\numpy\numpy\distutils\ccompiler_opt.py`

```py
# 导入必要的模块
import atexit  # 导入 atexit 模块，用于注册退出函数
import inspect  # 导入 inspect 模块，用于检查对象信息
import os  # 导入 os 模块，提供操作系统相关的功能
import pprint  # 导入 pprint 模块，用于美观打印数据结构
import re  # 导入 re 模块，提供正则表达式操作支持
import subprocess  # 导入 subprocess 模块，用于执行外部命令
import textwrap  # 导入 textwrap 模块，用于格式化文本段落

class _Config:
    """An abstract class holds all configurable attributes of `CCompilerOpt`,
    these class attributes can be used to change the default behavior
    of `CCompilerOpt` in order to fit other requirements.
    
    Attributes
    ----------
    conf_nocache : bool
        Set True to disable memory and file cache.
        Default is False.
    
    conf_noopt : bool
        Set True to forces the optimization to be disabled,
        in this case `CCompilerOpt` tends to generate all
        expected headers in order to 'not' break the build.
        Default is False.
    
    conf_cache_factors : list
        Add extra factors to the primary caching factors. The caching factors
        are utilized to determine if there are changes had happened that
        requires to discard the cache and re-updating it. The primary factors
        are the arguments of `CCompilerOpt` and `CCompiler`'s properties(type, flags, etc).
        Default is list of two items, containing the time of last modification
        of `ccompiler_opt` and value of attribute "conf_noopt"
    
    conf_tmp_path : str,
        The path of temporary directory. Default is auto-created
        temporary directory via ``tempfile.mkdtemp()``.
    
    conf_check_path : str
        The path of testing files. Each added CPU feature must have a
        **C** source file contains at least one intrinsic or instruction that
        related to this feature, so it can be tested against the compiler.
        Default is ``./distutils/checks``.
    
    conf_target_groups : dict
        Extra tokens that can be reached from dispatch-able sources through
        the special mark ``@targets``. Default is an empty dictionary.
        
        **Notes**:
            - case-insensitive for tokens and group names
            - sign '#' must stick in the begin of group name and only within ``@targets``
        
        **Example**:
            .. code-block:: console
        
                $ "@targets #avx_group other_tokens" > group_inside.c
        
            >>> CCompilerOpt.conf_target_groups["avx_group"] = \\
            "$werror $maxopt avx2 avx512f avx512_skx"
            >>> cco = CCompilerOpt(cc_instance)
            >>> cco.try_dispatch(["group_inside.c"])
    
    conf_c_prefix : str
        The prefix of public C definitions. Default is ``"NPY_"``.
"""
    # 定义内部 C 定义的前缀字符串。默认为 "NPY__"。
    conf_c_prefix_ : str
        The prefix of internal C definitions. Default is ``"NPY__"``.

    # 定义多个编译器标志的嵌套字典，用于链接到一些主要函数。主键表示编译器名称，子键表示标志名称。
    # 默认情况下已经包含所有支持的 C 编译器。
    conf_cc_flags : dict
        Nested dictionaries defining several compiler flags
        that linked to some major functions, the main key
        represent the compiler name and sub-keys represent
        flags names. Default is already covers all supported
        **C** compilers.

        # 子键说明如下：
        "native": str or None
            used by argument option `native`, to detect the current
            machine support via the compiler.
        "werror": str or None
            utilized to treat warning as errors during testing CPU features
            against the compiler and also for target's policy `$werror`
            via dispatch-able sources.
        "maxopt": str or None
            utilized for target's policy '$maxopt' and the value should
            contains the maximum acceptable optimization by the compiler.
            e.g. in gcc ``'-O3'``

        **Notes**:
            * case-sensitive for compiler names and flags
            * use space to separate multiple flags
            * any flag will tested against the compiler and it will skipped
              if it's not applicable.

    # 定义用于参数选项 `'min'` 的 CPU 特性字典，键表示 CPU 架构名称，例如 `'x86'`。
    # 默认值在广泛的用户平台上提供最佳支持。
    # 注意：架构名称区分大小写。
    conf_min_features : dict
        A dictionary defines the used CPU features for
        argument option ``'min'``, the key represent the CPU architecture
        name e.g. ``'x86'``. Default values provide the best effort
        on wide range of users platforms.

        **Note**: case-sensitive for architecture names.

    """
    # 禁用缓存的配置选项，默认为 False。
    conf_nocache = False
    # 禁用优化的配置选项，默认为 False。
    conf_noopt = False
    # 缓存因子的配置选项，默认为 None。
    conf_cache_factors = None
    # 临时路径的配置选项，默认为 None。
    conf_tmp_path = None
    # 检查路径的配置选项，设置为当前文件的 "checks" 子目录。
    conf_check_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "checks"
    )
    # 目标组的配置选项，初始化为空字典。
    conf_target_groups = {}
    # C 前缀的配置选项，默认为 'NPY_'。
    conf_c_prefix = 'NPY_'
    # 重新定义 C 前缀的配置选项，此处为 'NPY__'。
    conf_c_prefix_ = 'NPY__'
    conf_cc_flags = dict(
        gcc = dict(
            # 对于 arm 和 ppc64 平台，native 应该总是失败，
            # native 通常只在 x86 平台有效
            native = '-march=native',
            opt = '-O3',
            werror = '-Werror',
        ),
        clang = dict(
            native = '-march=native',
            opt = "-O3",
            # 为了保证测试过程的健全性，Clang 需要适用以下某一个标志，
            # 然而，在某些情况下，由于“未使用的参数”警告，-Werror 在可用性测试中被跳过。
            # 参见 https://github.com/numpy/numpy/issues/19624
            werror = '-Werror=switch -Werror',
        ),
        icc = dict(
            native = '-xHost',
            opt = '-O3',
            werror = '-Werror',
        ),
        iccw = dict(
            native = '/QxHost',
            opt = '/O3',
            werror = '/Werror',
        ),
        msvc = dict(
            native = None,
            opt = '/O2',
            werror = '/WX',
        ),
        fcc = dict(
            native = '-mcpu=a64fx',
            opt = None,
            werror = None,
        )
    )
    conf_min_features = dict(
        x86 = "SSE SSE2",
        x64 = "SSE SSE2 SSE3",
        ppc64 = '', # 保守起见
        ppc64le = "VSX VSX2",
        s390x = '',
        armhf = '', # 保守起见
        aarch64 = "NEON NEON_FP16 NEON_VFPV4 ASIMD"
    )
    def __init__(self):
        # 如果临时路径为 None，则创建一个临时目录，并在程序退出时自动删除
        if self.conf_tmp_path is None:
            import shutil
            import tempfile
            tmp = tempfile.mkdtemp()
            def rm_temp():
                try:
                    shutil.rmtree(tmp)
                except OSError:
                    pass
            atexit.register(rm_temp)
            self.conf_tmp_path = tmp

        # 如果缓存因子为 None，则设置默认的缓存因子列表
        if self.conf_cache_factors is None:
            self.conf_cache_factors = [
                os.path.getmtime(__file__),
                self.conf_nocache
            ]
class _Distutils:
    """A helper class that provides a collection of fundamental methods
    implemented on top of Python and NumPy Distutils.

    The idea behind this class is to gather all methods that may
    need to be overridden in case 'CCompilerOpt' is reused in an environment
    different from what NumPy has.

    Parameters
    ----------
    ccompiler : `CCompiler`
        The generated instance returned from `distutils.ccompiler.new_compiler()`.
    """
    def __init__(self, ccompiler):
        # 初始化方法，将传入的编译器实例保存在属性中
        self._ccompiler = ccompiler

    def dist_compile(self, sources, flags, ccompiler=None, **kwargs):
        """Wrap CCompiler.compile()"""
        # 断言参数类型为列表
        assert(isinstance(sources, list))
        assert(isinstance(flags, list))
        # 将额外的编译标志合并到 flags 中
        flags = kwargs.pop("extra_postargs", []) + flags
        # 如果未提供 ccompiler，则使用初始化时传入的编译器实例
        if not ccompiler:
            ccompiler = self._ccompiler

        # 调用编译器实例的 compile 方法进行编译
        return ccompiler.compile(sources, extra_postargs=flags, **kwargs)

    def dist_test(self, source, flags, macros=[]):
        """Return True if 'CCompiler.compile()' is able to compile
        a source file with certain flags.
        """
        # 断言参数类型为字符串
        assert(isinstance(source, str))
        # 导入 CompileError 类
        from distutils.errors import CompileError
        # 获取当前保存的编译器实例
        cc = self._ccompiler;
        # 备份原始的 spawn 方法
        bk_spawn = getattr(cc, 'spawn', None)
        # 根据编译器类型设置不同的 spawn 方法
        if bk_spawn:
            cc_type = getattr(self._ccompiler, "compiler_type", "")
            if cc_type in ("msvc",):
                setattr(cc, 'spawn', self._dist_test_spawn_paths)
            else:
                setattr(cc, 'spawn', self._dist_test_spawn)
        # 默认测试结果为 False
        test = False
        try:
            # 调用 dist_compile 方法尝试编译源文件
            self.dist_compile(
                [source], flags, macros=macros, output_dir=self.conf_tmp_path
            )
            # 如果成功编译，则设置测试结果为 True
            test = True
        except CompileError as e:
            # 捕获编译异常，记录错误信息
            self.dist_log(str(e), stderr=True)
        # 恢复原始的 spawn 方法
        if bk_spawn:
            setattr(cc, 'spawn', bk_spawn)
        # 返回测试结果
        return test
    def dist_info(self):
        """
        Return a tuple containing info about (platform, compiler, extra_args),
        required by the abstract class '_CCompiler' for discovering the
        platform environment. This is also used as a cache factor in order
        to detect any changes happening from outside.
        """
        # 如果已经计算过并缓存了信息，则直接返回缓存的结果
        if hasattr(self, "_dist_info"):
            return self._dist_info

        # 获取当前编译器类型
        cc_type = getattr(self._ccompiler, "compiler_type", '')
        # 根据编译器类型确定平台
        if cc_type in ("intelem", "intelemw"):
            platform = "x86_64"
        elif cc_type in ("intel", "intelw", "intele"):
            platform = "x86"
        else:
            # 如果是 Unix 系统，通过 distutils 获取平台信息
            from distutils.util import get_platform
            platform = get_platform()

        # 获取编译器信息
        cc_info = getattr(self._ccompiler, "compiler", getattr(self._ccompiler, "compiler_so", ''))
        # 如果编译器类型为空或者是 Unix 系统，则处理编译器信息
        if not cc_type or cc_type == "unix":
            if hasattr(cc_info, "__iter__"):
                compiler = cc_info[0]
            else:
                compiler = str(cc_info)
        else:
            compiler = cc_type

        # 获取额外的编译参数
        if hasattr(cc_info, "__iter__") and len(cc_info) > 1:
            extra_args = ' '.join(cc_info[1:])
        else:
            extra_args = os.environ.get("CFLAGS", "")
            extra_args += os.environ.get("CPPFLAGS", "")

        # 缓存计算结果并返回
        self._dist_info = (platform, compiler, extra_args)
        return self._dist_info

    @staticmethod
    def dist_error(*args):
        """Raise a compiler error"""
        # 抛出编译错误异常
        from distutils.errors import CompileError
        raise CompileError(_Distutils._dist_str(*args))

    @staticmethod
    def dist_fatal(*args):
        """Raise a distutils error"""
        # 抛出 distutils 错误异常
        from distutils.errors import DistutilsError
        raise DistutilsError(_Distutils._dist_str(*args))

    @staticmethod
    def dist_log(*args, stderr=False):
        """Print a console message"""
        # 打印控制台消息，根据 stderr 参数选择打印级别
        from numpy.distutils import log
        out = _Distutils._dist_str(*args)
        if stderr:
            log.warn(out)
        else:
            log.info(out)

    @staticmethod
    def dist_load_module(name, path):
        """Load a module from file, required by the abstract class '_Cache'."""
        # 从文件加载模块，用于抽象类 '_Cache' 所需
        from .misc_util import exec_mod_from_location
        try:
            return exec_mod_from_location(name, path)
        except Exception as e:
            # 记录加载模块时出现的异常
            _Distutils.dist_log(e, stderr=True)
        return None

    @staticmethod
    def _dist_str(*args):
        """Return a string to print by log and errors."""
        # 生成用于日志和错误打印的字符串
        def to_str(arg):
            if not isinstance(arg, str) and hasattr(arg, '__iter__'):
                ret = []
                for a in arg:
                    ret.append(to_str(a))
                return '('+ ' '.join(ret) + ')'
            return str(arg)

        # 获取调用栈信息，生成打印起始信息
        stack = inspect.stack()[2]
        start = "CCompilerOpt.%s[%d] : " % (stack.function, stack.lineno)
        # 将所有参数转换为字符串并拼接
        out = ' '.join([
            to_str(a)
            for a in (*args,)
        ])
        return start + out
    def _dist_test_spawn_paths(self, cmd, display=None):
        """
        Fix msvc SDK ENV path same as distutils do
        without it we get c1: fatal error C1356: unable to find mspdbcore.dll
        """
        # 检查 self._ccompiler 是否有 "_paths" 属性，若没有则调用 self._dist_test_spawn(cmd) 并返回
        if not hasattr(self._ccompiler, "_paths"):
            self._dist_test_spawn(cmd)
            return
        # 保存当前环境变量中的 PATH 到 old_path
        old_path = os.getenv("path")
        try:
            # 设置环境变量中的 PATH 为 self._ccompiler._paths
            os.environ["path"] = self._ccompiler._paths
            # 调用 self._dist_test_spawn(cmd) 执行命令
            self._dist_test_spawn(cmd)
        finally:
            # 恢复原来的环境变量中的 PATH
            os.environ["path"] = old_path

    _dist_warn_regex = re.compile(
        # 编译警告正则表达式，匹配 Intel 和 MSVC 编译器的警告信息
        ".*("
        "warning D9002|"  # MSVC，应该适用于任何语言。
        "invalid argument for option" # Intel
        ").*"
    )
    @staticmethod
    def _dist_test_spawn(cmd, display=None):
        try:
            # 执行命令 cmd，捕获输出到 o，将 stderr 合并到 stdout
            o = subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                        text=True)
            # 如果输出 o 存在，并且匹配 _Distutils._dist_warn_regex 中定义的警告模式
            if o and re.match(_Distutils._dist_warn_regex, o):
                # 调用 _Distutils.dist_error 输出错误信息，指示编译器不支持命令中的标志
                _Distutils.dist_error(
                    "Flags in command", cmd ,"aren't supported by the compiler"
                    ", output -> \n%s" % o
                )
        except subprocess.CalledProcessError as exc:
            # 捕获 subprocess 执行命令时的异常，将输出和返回码保存到 o 和 s
            o = exc.output
            s = exc.returncode
        except OSError as e:
            # 捕获 OS 错误，将错误信息保存到 o，返回码设置为 127
            o = e
            s = 127
        else:
            # 没有异常发生时返回 None
            return None
        # 调用 _Distutils.dist_error 输出命令执行失败的错误信息，包括返回码和输出内容
        _Distutils.dist_error(
            "Command", cmd, "failed with exit status %d output -> \n%s" % (
            s, o
        ))
# 共享缓存，用于存储所有实例对象的缓存数据
_share_cache = {}

# 缓存类，处理缓存功能，提供内存和文件两级缓存
class _Cache:
    """An abstract class handles caching functionality, provides two
    levels of caching, in-memory by share instances attributes among
    each other and by store attributes into files.

    **Note**:
        any attributes that start with ``_`` or ``conf_`` will be ignored.

    Parameters
    ----------
    cache_path : str or None
        The path of cache file, if None then cache in file will disabled.

    *factors :
        The caching factors that need to utilize next to `conf_cache_factors`.

    Attributes
    ----------
    cache_private : set
        Hold the attributes that need be skipped from "in-memory cache".

    cache_infile : bool
        Utilized during initializing this class, to determine if the cache was able
        to loaded from the specified cache path in 'cache_path'.
    """

    # 正则表达式，用于忽略不需要缓存的属性
    _cache_ignore = re.compile("^(_|conf_)")

    def __init__(self, cache_path=None, *factors):
        # 初始化内存缓存字典
        self.cache_me = {}
        # 初始化需要跳过的私有缓存集合
        self.cache_private = set()
        # 初始化文件缓存加载状态
        self.cache_infile = False
        # 缓存文件路径
        self._cache_path = None

        # 如果禁用了缓存，则记录日志并返回
        if self.conf_nocache:
            self.dist_log("cache is disabled by `Config`")
            return

        # 计算缓存哈希值
        self._cache_hash = self.cache_hash(*factors, *self.conf_cache_factors)
        self._cache_path = cache_path

        # 如果指定了缓存文件路径
        if cache_path:
            # 如果缓存文件存在
            if os.path.exists(cache_path):
                self.dist_log("load cache from file ->", cache_path)
                # 加载缓存文件作为模块
                cache_mod = self.dist_load_module("cache", cache_path)
                if not cache_mod:
                    self.dist_log(
                        "unable to load the cache file as a module",
                        stderr=True
                    )
                # 如果缓存模块缺少必要的属性
                elif not hasattr(cache_mod, "hash") or \
                     not hasattr(cache_mod, "data"):
                    self.dist_log("invalid cache file", stderr=True)
                # 如果缓存哈希匹配成功
                elif self._cache_hash == cache_mod.hash:
                    self.dist_log("hit the file cache")
                    # 将缓存模块的数据项设置为当前实例的属性
                    for attr, val in cache_mod.data.items():
                        setattr(self, attr, val)
                    # 标记文件缓存已命中
                    self.cache_infile = True
                else:
                    self.dist_log("miss the file cache")

        # 如果文件缓存未命中，则尝试从共享缓存中获取
        if not self.cache_infile:
            other_cache = _share_cache.get(self._cache_hash)
            if other_cache:
                self.dist_log("hit the memory cache")
                # 将共享缓存对象的属性设置为当前实例的属性
                for attr, val in other_cache.__dict__.items():
                    if attr in other_cache.cache_private or \
                               re.match(self._cache_ignore, attr):
                        continue
                    setattr(self, attr, val)

        # 将当前实例加入共享缓存
        _share_cache[self._cache_hash] = self
        # 注册析构函数，在程序退出时刷新缓存
        atexit.register(self.cache_flush)

    def __del__(self):
        # 在实例销毁时，从共享缓存中移除当前实例
        for h, o in _share_cache.items():
            if o == self:
                _share_cache.pop(h)
                break
    def cache_flush(self):
        """
        Force update the cache.
        """
        # 如果缓存路径为空，则直接返回，不执行更新操作
        if not self._cache_path:
            return
        # 输出缓存写入路径日志信息
        self.dist_log("write cache to path ->", self._cache_path)
        # 复制对象的字典表示，以准备写入缓存
        cdict = self.__dict__.copy()
        # 遍历对象字典的所有键
        for attr in self.__dict__.keys():
            # 如果属性名匹配缓存忽略规则，则从复制的字典中删除该属性
            if re.match(self._cache_ignore, attr):
                cdict.pop(attr)

        # 获取缓存文件的父目录路径，如果不存在则创建
        d = os.path.dirname(self._cache_path)
        if not os.path.exists(d):
            os.makedirs(d)

        # 将复制后的字典转换为紧凑格式的字符串表示
        repr_dict = pprint.pformat(cdict, compact=True)
        # 打开缓存文件，写入缓存头部信息和数据
        with open(self._cache_path, "w") as f:
            f.write(textwrap.dedent("""\
            # AUTOGENERATED DON'T EDIT
            # Please make changes to the code generator (distutils/ccompiler_opt.py)
            hash = {}
            data = \\
            """).format(self._cache_hash))
            f.write(repr_dict)

    def cache_hash(self, *factors):
        # 计算给定因子的散列值
        chash = 0
        for f in factors:
            for char in str(f):
                chash  = ord(char) + (chash << 6) + (chash << 16) - chash
                chash &= 0xFFFFFFFF
        return chash

    @staticmethod
    def me(cb):
        """
        A static method that can be treated as a decorator to
        dynamically cache certain methods.
        """
        def cache_wrap_me(self, *args, **kwargs):
            # 生成用于缓存查找的唯一键
            cache_key = str((
                cb.__name__, *args, *kwargs.keys(), *kwargs.values()
            ))
            # 如果键已存在于缓存中，则直接返回缓存值
            if cache_key in self.cache_me:
                return self.cache_me[cache_key]
            # 否则，调用原始方法生成结果，并存储在缓存中
            ccb = cb(self, *args, **kwargs)
            self.cache_me[cache_key] = ccb
            return ccb
        return cache_wrap_me
class _CCompiler:
    """A helper class for `CCompilerOpt` containing all utilities that
    related to the fundamental compiler's functions.

    Attributes
    ----------
    cc_on_x86 : bool
        True when the target architecture is 32-bit x86
    cc_on_x64 : bool
        True when the target architecture is 64-bit x86
    cc_on_ppc64 : bool
        True when the target architecture is 64-bit big-endian powerpc
    cc_on_ppc64le : bool
        True when the target architecture is 64-bit little-endian powerpc
    cc_on_s390x : bool
        True when the target architecture is IBM/ZARCH on linux
    cc_on_armhf : bool
        True when the target architecture is 32-bit ARMv7+
    cc_on_aarch64 : bool
        True when the target architecture is 64-bit Armv8-a+
    cc_on_noarch : bool
        True when the target architecture is unknown or not supported
    cc_is_gcc : bool
        True if the compiler is GNU or
        if the compiler is unknown
    cc_is_clang : bool
        True if the compiler is Clang
    cc_is_icc : bool
        True if the compiler is Intel compiler (unix like)
    cc_is_iccw : bool
        True if the compiler is Intel compiler (msvc like)
    cc_is_nocc : bool
        True if the compiler isn't supported directly,
        Note: that cause a fail-back to gcc
    cc_has_debug : bool
        True if the compiler has debug flags
    cc_has_native : bool
        True if the compiler has native flags
    cc_noopt : bool
        True if the compiler has definition 'DISABLE_OPT*',
        or 'cc_on_noarch' is True
    cc_march : str
        The target architecture name, or "unknown" if
        the architecture isn't supported
    cc_name : str
        The compiler name, or "unknown" if the compiler isn't supported
    cc_flags : dict
        Dictionary containing the initialized flags of `_Config.conf_cc_flags`
    """

    @_Cache.me
    def cc_test_flags(self, flags):
        """
        Returns True if the compiler supports 'flags'.
        """
        assert(isinstance(flags, list))  # 断言，确保flags是一个列表
        self.dist_log("testing flags", flags)  # 记录测试的标志
        test_path = os.path.join(self.conf_check_path, "test_flags.c")  # 构建测试文件路径
        test = self.dist_test(test_path, flags)  # 进行测试
        if not test:
            self.dist_log("testing failed", stderr=True)  # 如果测试失败，则记录错误日志
        return test  # 返回测试结果

    @_Cache.me
    def cc_test_cexpr(self, cexpr, flags=[]):
        """
        Same as the above but supports compile-time expressions.
        """
        self.dist_log("testing compiler expression", cexpr)  # 记录测试的编译器表达式
        test_path = os.path.join(self.conf_tmp_path, "npy_dist_test_cexpr.c")  # 构建测试文件路径
        with open(test_path, "w") as fd:
            fd.write(textwrap.dedent(f"""\
               #if !({cexpr})
                   #error "unsupported expression"
               #endif
               int dummy;
            """))  # 写入测试文件内容，用于测试编译器表达式
        test = self.dist_test(test_path, flags)  # 进行测试
        if not test:
            self.dist_log("testing failed", stderr=True)  # 如果测试失败，则记录错误日志
        return test  # 返回测试结果
    # 定义一个方法用于规范化编译器标志，处理由收集到的隐含特性标志所引起的冲突。

    def cc_normalize_flags(self, flags):
        """
        Remove the conflicts that caused due gathering implied features flags.

        Parameters
        ----------
        'flags' list, compiler flags
            flags should be sorted from the lowest to the highest interest.

        Returns
        -------
        list, filtered from any conflicts.

        Examples
        --------
        >>> self.cc_normalize_flags(['-march=armv8.2-a+fp16', '-march=armv8.2-a+dotprod'])
        ['armv8.2-a+fp16+dotprod']

        >>> self.cc_normalize_flags(
            ['-msse', '-msse2', '-msse3', '-mssse3', '-msse4.1', '-msse4.2', '-mavx', '-march=core-avx2']
        )
        ['-march=core-avx2']
        """
        # 断言确保 flags 是一个列表
        assert(isinstance(flags, list))
        # 如果是 GCC、Clang 或 ICC 编译器，调用 Unix 系统下的标志规范化方法
        if self.cc_is_gcc or self.cc_is_clang or self.cc_is_icc:
            return self._cc_normalize_unix(flags)

        # 如果是 MSVC 或 ICCW 编译器，调用 Windows 系统下的标志规范化方法
        if self.cc_is_msvc or self.cc_is_iccw:
            return self._cc_normalize_win(flags)
        
        # 如果以上条件都不符合，则直接返回原始的 flags
        return flags

    # Unix 系统下的正则表达式模式，用于匹配和处理不同类型的编译器标志
    _cc_normalize_unix_mrgx = re.compile(
        # 匹配以 -mcpu=、-march= 或 -x[A-Z0-9\-] 开头的标志
        r"^(-mcpu=|-march=|-x[A-Z0-9\-])"
    )

    # Unix 系统下的正则表达式模式，用于排除特定类型的编译器标志
    _cc_normalize_unix_frgx = re.compile(
        # 匹配不以 -mcpu=、-march=、-x[A-Z0-9\-] 或 -m[a-z0-9\-\.]*.$ 开头的标志，并排除 -mzvector
        r"^(?!(-mcpu=|-march=|-x[A-Z0-9\-]|-m[a-z0-9\-\.]*.$))|"
        r"(?:-mzvector)"
    )

    # Unix 系统下的正则表达式模式，用于保留 -mfpu 和 -mtune 类型的编译器标志
    _cc_normalize_unix_krgx = re.compile(
        r"^(-mfpu|-mtune)"
    )

    # 匹配版本号中的数字和小数点，用于提取和处理架构版本信息
    _cc_normalize_arch_ver = re.compile(
        r"[0-9.]"
    )
    def _cc_normalize_unix(self, flags):
        def ver_flags(f):
            # 解析版本相关的标志
            # -march=armv8.2-a+fp16fml
            tokens = f.split('+')
            # 提取架构版本号并转换为浮点数
            ver = float('0' + ''.join(
                re.findall(self._cc_normalize_arch_ver, tokens[0])
            ))
            return ver, tokens[0], tokens[1:]

        if len(flags) <= 1:
            return flags
        # 获取最高匹配的标志
        for i, cur_flag in enumerate(reversed(flags)):
            if not re.match(self._cc_normalize_unix_mrgx, cur_flag):
                continue
            lower_flags = flags[:-(i+1)]
            upper_flags = flags[-i:]
            # 过滤掉与 _cc_normalize_unix_frgx 不匹配的标志
            filtered = list(filter(
                self._cc_normalize_unix_frgx.search, lower_flags
            ))
            # 获取版本号、架构和子标志
            ver, arch, subflags = ver_flags(cur_flag)
            if ver > 0 and len(subflags) > 0:
                for xflag in lower_flags:
                    xver, _, xsubflags = ver_flags(xflag)
                    if ver == xver:
                        subflags = xsubflags + subflags
                cur_flag = arch + '+' + '+'.join(subflags)

            flags = filtered + [cur_flag]
            if i > 0:
                flags += upper_flags
            break

        # 移除可以被覆盖的标志
        final_flags = []
        matched = set()
        for f in reversed(flags):
            match = re.match(self._cc_normalize_unix_krgx, f)
            if not match:
                pass
            elif match[0] in matched:
                continue
            else:
                matched.add(match[0])
            final_flags.insert(0, f)
        return final_flags

    _cc_normalize_win_frgx = re.compile(
        r"^(?!(/arch\:|/Qx\:))"
    )
    _cc_normalize_win_mrgx = re.compile(
        r"^(/arch|/Qx:)"
    )
    def _cc_normalize_win(self, flags):
        for i, f in enumerate(reversed(flags)):
            if not re.match(self._cc_normalize_win_mrgx, f):
                continue
            i += 1
            # 返回匹配 _cc_normalize_win_frgx 的标志
            return list(filter(
                self._cc_normalize_win_frgx.search, flags[:-i]
            )) + flags[-i:]
        return flags
# 定义一个辅助类`_Feature`，用于管理 CPU 功能
class _Feature:
    """A helper class for `CCompilerOpt` that managing CPU features.

    Attributes
    ----------
    feature_supported : dict
        Dictionary containing all CPU features that supported
        by the platform, according to the specified values in attribute
        `_Config.conf_features` and `_Config.conf_features_partial()`

    feature_min : set
        The minimum support of CPU features, according to
        the specified values in attribute `_Config.conf_min_features`.
    """
    def __init__(self):
        # 如果已经有了`feature_is_cached`属性，则直接返回，不再执行下面的代码
        if hasattr(self, "feature_is_cached"):
            return
        # 获取所有受平台支持的 CPU 功能，根据属性`_Config.conf_features`和`_Config.conf_features_partial()`
        self.feature_supported = pfeatures = self.conf_features_partial()
        # 遍历每个 CPU 功能
        for feature_name in list(pfeatures.keys()):
            # 获取 CPU 功能的详细信息
            feature  = pfeatures[feature_name]
            # 获取针对当前 CPU 功能的配置信息
            cfeature = self.conf_features[feature_name]
            # 将配置信息中没有的部分添加到 CPU 功能的详细信息中
            feature.update({
                k:v for k,v in cfeature.items() if k not in feature
            })
            # 检查当前 CPU 功能是否被禁用
            disabled = feature.get("disable")
            if disabled is not None:
                # 如果被禁用，从受支持的 CPU 功能中移除，同时记录日志
                pfeatures.pop(feature_name)
                self.dist_log(
                    "feature '%s' is disabled," % feature_name,
                    disabled, stderr=True
                )
                continue
            # 内部使用列表的选项
            for option in (
                "implies", "group", "detect", "headers", "flags", "extra_checks"
            ) :
                # 将字符串类型的选项转换为列表
                oval = feature.get(option)
                if isinstance(oval, str):
                    feature[option] = oval.split()

        # 初始化最小支持的 CPU 功能集合
        self.feature_min = set()
        # 获取最小支持的 CPU 功能
        min_f = self.conf_min_features.get(self.cc_march, "")
        # 将最小支持的 CPU 功能转换为大写并分割成集合
        for F in min_f.upper().split():
            if F in self.feature_supported:
                self.feature_min.add(F)

        # 标记属性`feature_is_cached`已经被设置
        self.feature_is_cached = True
    def feature_names(self, names=None, force_flags=None, macros=[]):
        """
        返回平台和 **C** 编译器支持的一组 CPU 特性名称

        Parameters
        ----------
        names : sequence or None, optional
            指定要测试的特定 CPU 特性，以便与 **C** 编译器进行测试。
            如果为 None（默认），将测试所有当前支持的特性。
            **注意**: 特性名称必须是大写。

        force_flags : list or None, optional
            如果为 None（默认），将在测试期间使用每个 CPU 特性的默认编译器标志

        macros : list of tuples, optional
            一个 C 宏定义的列表。
        """
        assert(
            names is None or (
                not isinstance(names, str) and
                hasattr(names, "__iter__")
            )
        )
        assert(force_flags is None or isinstance(force_flags, list))
        if names is None:
            names = self.feature_supported.keys()
        supported_names = set()
        for f in names:
            if self.feature_is_supported(
                f, force_flags=force_flags, macros=macros
            ):
                supported_names.add(f)
        return supported_names

    def feature_is_exist(self, name):
        """
        如果某个特性存在且在 ``_Config.conf_features`` 中有覆盖则返回 True。

        Parameters
        ----------
        'name': str
            大写特性名称。
        """
        assert(name.isupper())
        return name in self.conf_features

    def feature_sorted(self, names, reverse=False):
        """
        按照最低兴趣排序 CPU 特性列表。

        Parameters
        ----------
        'names': sequence
            大写支持的特性名称序列。
        'reverse': bool, optional
            如果为真，则倒序排列特性。（兴趣最高）

        Returns
        -------
        list, 排序后的 CPU 特性列表
        """
        def sort_cb(k):
            if isinstance(k, str):
                return self.feature_supported[k]["interest"]
            # 多个特性
            rank = max([self.feature_supported[f]["interest"] for f in k])
            # FIXME: 这不是增加多个目标的等级的安全方法
            rank += len(k) -1
            return rank
        return sorted(names, reverse=reverse, key=sort_cb)
    # 定义一个方法，用于获取由给定CPU特性名字所暗示的一组CPU特性
    def feature_implies(self, names, keep_origins=False):
        """
        Return a set of CPU features that implied by 'names'

        Parameters
        ----------
        names : str or sequence of str
            CPU feature name(s) in uppercase.

        keep_origins : bool
            if False(default) then the returned set will not contain any
            features from 'names'. This case happens only when two features
            imply each other.

        Examples
        --------
        >>> self.feature_implies("SSE3")
        {'SSE', 'SSE2'}
        >>> self.feature_implies("SSE2")
        {'SSE'}
        >>> self.feature_implies("SSE2", keep_origins=True)
        # 'SSE2' found here since 'SSE' and 'SSE2' imply each other
        {'SSE', 'SSE2'}
        """
        # 定义一个内部方法，用于获取暗示的CPU特性
        def get_implies(name, _caller=set()):
            implies = set()
            d = self.feature_supported[name]
            for i in d.get("implies", []):
                # 添加暗示的CPU特性到结果集合
                implies.add(i)
                if i in _caller:
                    # 由于特性可以互相暗示，需要防止无限递归
                    continue
                # 将当前特性加入调用堆栈，用于检查递归
                _caller.add(name)
                # 递归获取暗示的CPU特性并合并到结果集合
                implies = implies.union(get_implies(i, _caller))
            return implies

        # 判断输入的特性名是否是字符串
        if isinstance(names, str):
            # 获取特性暗示的结果集合
            implies = get_implies(names)
            # 转换为列表方便后续操作
            names = [names]
        else:
            # 如果输入是一个可迭代对象，则遍历获取特性暗示的结果集合
            assert(hasattr(names, "__iter__"))
            implies = set()
            for n in names:
                # 将每个特性暗示的结果集合合并到一起
                implies = implies.union(get_implies(n))
        # 如果不需要保留原始特性，从结果集合中删去输入的特性
        if not keep_origins:
            implies.difference_update(names)
        # 返回最终的特性暗示结果集合
        return implies

    # 定义另一个方法，与上述方法类似，但是会将输入的特性名字集合合并后再获取暗示的结果集合
    def feature_implies_c(self, names):
        """same as feature_implies() but combining 'names'"""
        if isinstance(names, str):
            names = set((names,))
        else:
            names = set(names)
        # 返回合并后的特性暗示结果集合
        return names.union(self.feature_implies(names))
    # 定义一个方法，用于返回在给定名称中删除任何暗示特性后并保留原始特性的特性列表
    def feature_ahead(self, names):
        """
        Return list of features in 'names' after remove any
        implied features and keep the origins.

        Parameters
        ----------
        'names': sequence
            sequence of CPU feature names in uppercase.

        Returns
        -------
        list of CPU features sorted as-is 'names'

        Examples
        --------
        >>> self.feature_ahead(["SSE2", "SSE3", "SSE41"])
        ["SSE41"]
        # assume AVX2 and FMA3 implies each other and AVX2
        # is the highest interest
        >>> self.feature_ahead(["SSE2", "SSE3", "SSE41", "AVX2", "FMA3"])
        ["AVX2"]
        # assume AVX2 and FMA3 don't implies each other
        >>> self.feature_ahead(["SSE2", "SSE3", "SSE41", "AVX2", "FMA3"])
        ["AVX2", "FMA3"]
        """
        # 检查输入是否为字符串，并且是否可迭代
        assert(
            not isinstance(names, str)
            and hasattr(names, '__iter__')
        )
        # 获取暗示的特性，保留原始特性
        implies = self.feature_implies(names, keep_origins=True)
        # 获取不暗示的特性
        ahead = [n for n in names if n not in implies]
        if len(ahead) == 0:
            # 如果所有特性都互相暗示，则返回最感兴趣的特性
            ahead = self.feature_sorted(names, reverse=True)[:1]
        return ahead

    # 定义一个方法，与'feature_ahead()'相同，但如果两个特性互相暗示，保留最感兴趣的特性
    def feature_untied(self, names):
        """
        same as 'feature_ahead()' but if both features implied each other and keep the highest interest.

        Parameters
        ----------
        'names': sequence
            sequence of CPU feature names in uppercase.

        Returns
        -------
        list of CPU features sorted as-is 'names'

        Examples
        --------
        >>> self.feature_untied(["SSE2", "SSE3", "SSE41"])
        ["SSE2", "SSE3", "SSE41"]
        # assume AVX2 and FMA3 implies each other
        >>> self.feature_untied(["SSE2", "SSE3", "SSE41", "FMA3", "AVX2"])
        ["SSE2", "SSE3", "SSE41", "AVX2"]
        """
        # 检查输入是否为字符串，并且是否可迭代
        assert(
            not isinstance(names, str)
            and hasattr(names, '__iter__')
        )
        # 最终结果列表
        final = []
        for n in names:
            # 获取暗示的特性
            implies = self.feature_implies(n)
            tied = [
                nn for nn in final
                if nn in implies and n in self.feature_implies(nn)
            ]
            if tied:
                # 根据最感兴趣的顺序排序
                tied = self.feature_sorted(tied + [n])
                # 如果n不在除第一个特性外的列表中，则继续下一个循环
                if n not in tied[1:]:
                    continue
                # 移除最感兴趣的特性
                final.remove(tied[:1][0])
            # 添加特性到最终结果列表
            final.append(n)
        return final
    def feature_get_til(self, names, keyisfalse):
        """
        same as `feature_implies_c()` but stop collecting implied
        features when feature's option that provided through
        parameter 'keyisfalse' is False, also sorting the returned
        features.
        """
        def til(tnames):
            # 调用 feature_implies_c() 函数获取所有可能的特征
            tnames = self.feature_implies_c(tnames)
            # 根据兴趣从高到低对特征进行排序
            tnames = self.feature_sorted(tnames, reverse=True)
            # 如果 keyisfalse 参数对应的特征选项为 False，则截断列表
            for i, n in enumerate(tnames):
                if not self.feature_supported[n].get(keyisfalse, True):
                    tnames = tnames[:i+1]
                    break
            return tnames

        if isinstance(names, str) or len(names) <= 1:
            # 对单个特征名称或短列表进行处理
            names = til(names)
            # 对排序进行归一化
            names.reverse()
            return names

        # 处理包含多个特征名称的列表
        names = self.feature_ahead(names)
        # 获取所有特征名称的完整集合，并返回排序后的结果
        names = {t for n in names for t in til(n)}
        return self.feature_sorted(names)

    def feature_detect(self, names):
        """
        Return a list of CPU features that required to be detected
        sorted from the lowest to highest interest.
        """
        # 获取需要进行检测的特征列表，按照兴趣从低到高排序
        names = self.feature_get_til(names, "implies_detect")
        detect = []
        for n in names:
            # 获取特征 n 的支持信息字典
            d = self.feature_supported[n]
            # 将检测所需的特征添加到 detect 列表中
            detect += d.get("detect", d.get("group", [n]))
        return detect

    @_Cache.me
    def feature_flags(self, names):
        """
        Return a list of CPU features flags sorted from the lowest
        to highest interest.
        """
        # 对特征列表进行排序，并且获取所有可能的特征
        names = self.feature_sorted(self.feature_implies_c(names))
        flags = []
        for n in names:
            # 获取特征 n 的支持信息字典
            d = self.feature_supported[n]
            # 获取特征的标志（flags），如果为空或不满足 cc_test_flags() 的条件则跳过
            f = d.get("flags", [])
            if not f or not self.cc_test_flags(f):
                continue
            # 将特征的标志添加到 flags 列表中
            flags += f
        return self.cc_normalize_flags(flags)

    @_Cache.me
    # 测试特定的 CPU 功能在编译器中的支持情况，通过其自身的检查文件
    def feature_test(self, name, force_flags=None, macros=[]):
        """
        Test a certain CPU feature against the compiler through its own
        check file.

        Parameters
        ----------
        name : str
            Supported CPU feature name.

        force_flags : list or None, optional
            If None(default), the returned flags from `feature_flags()`
            will be used.

        macros : list of tuples, optional
            A list of C macro definitions.
        """
        # 如果 force_flags 为 None，则使用 feature_flags() 返回的标志
        if force_flags is None:
            force_flags = self.feature_flags(name)

        # 记录日志，测试特性与其使用的标志
        self.dist_log(
            "testing feature '%s' with flags (%s)" % (
            name, ' '.join(force_flags)
        ))

        # 每个 CPU 功能必须有包含至少一个与该功能相关的指令的 C 源代码
        test_path = os.path.join(
            self.conf_check_path, "cpu_%s.c" % name.lower()
        )
        # 检查测试文件是否存在
        if not os.path.exists(test_path):
            self.dist_fatal("feature test file is not exist", test_path)

        # 进行测试
        test = self.dist_test(
            test_path, force_flags + self.cc_flags["werror"], macros=macros
        )
        # 如果测试失败则记录日志
        if not test:
            self.dist_log("testing failed", stderr=True)
        return test

    @_Cache.me
    def feature_is_supported(self, name, force_flags=None, macros=[]):
        """
        Check if a certain CPU feature is supported by the platform and compiler.

        Parameters
        ----------
        name : str
            CPU feature name in uppercase.

        force_flags : list or None, optional
            If None(default), default compiler flags for every CPU feature will
            be used during test.

        macros : list of tuples, optional
            A list of C macro definitions.
        """
        # 断言 CPU 功能名为大写
        assert(name.isupper())
        assert(force_flags is None or isinstance(force_flags, list))

        # 检查特定 CPU 功能是否在平台和编译器中受支持
        supported = name in self.feature_supported
        if supported:
            # 对于每个实现，检查其依赖的功能
            for impl in self.feature_implies(name):
                if not self.feature_test(impl, force_flags, macros=macros):
                    return False
            # 检查该功能
            if not self.feature_test(name, force_flags, macros=macros):
                return False
        return supported

    @_Cache.me
    def feature_can_autovec(self, name):
        """
        check if the feature can be auto-vectorized by the compiler
        """
        # 断言参数为字符串
        assert(isinstance(name, str))
        d = self.feature_supported[name]
        can = d.get("autovec", None)
        if can is None:
            # 检查是否有有效的标志支持自动向量化
            valid_flags = [
                self.cc_test_flags([f]) for f in d.get("flags", [])
            ]
            can = valid_flags and any(valid_flags)
        return can

    @_Cache.me
    def feature_extra_checks(self, name):
        """
        Return a list of supported extra checks after testing them against
        the compiler.

        Parameters
        ----------
        names : str
            CPU feature name in uppercase.
        """
        # 确保参数 name 是一个字符串
        assert isinstance(name, str)
        # 获取特定特性的字典
        d = self.feature_supported[name]
        # 获取额外检查的列表
        extra_checks = d.get("extra_checks", [])
        # 如果额外检查列表为空，返回空列表
        if not extra_checks:
            return []

        # 执行额外检查的测试，并记录日志
        self.dist_log("Testing extra checks for feature '%s'" % name, extra_checks)
        # 获取特定特性的编译器标志
        flags = self.feature_flags(name)
        # 存储支持的额外检查
        available = []
        # 存储不支持的额外检查
        not_available = []
        # 遍历额外检查列表
        for chk in extra_checks:
            # 构建额外检查文件的路径
            test_path = os.path.join(
                self.conf_check_path, "extra_%s.c" % chk.lower()
            )
            # 如果文件不存在，记录致命错误
            if not os.path.exists(test_path):
                self.dist_fatal("extra check file does not exist", test_path)

            # 进行额外检查的测试，并判断是否支持
            is_supported = self.dist_test(test_path, flags + self.cc_flags["werror"])
            if is_supported:
                available.append(chk)
            else:
                not_available.append(chk)

        # 如果有不支持的额外检查，记录日志
        if not_available:
            self.dist_log("testing failed for checks", not_available, stderr=True)
        # 返回支持的额外检查列表
        return available


    def feature_c_preprocessor(self, feature_name, tabs=0):
        """
        Generate C preprocessor definitions and include headers of a CPU feature.

        Parameters
        ----------
        'feature_name': str
            CPU feature name in uppercase.
        'tabs': int
            if > 0, align the generated strings to the right depend on number of tabs.

        Returns
        -------
        str, generated C preprocessor

        Examples
        --------
        >>> self.feature_c_preprocessor("SSE3")
        /** SSE3 **/
        #define NPY_HAVE_SSE3 1
        #include <pmmintrin.h>
        """
        # 确保特性名称是大写字符串
        assert(feature_name.isupper())
        # 获取特定特性的信息字典
        feature = self.feature_supported.get(feature_name)
        # 确保特性信息存在
        assert(feature is not None)

        # 初始化预处理器定义列表
        prepr = [
            "/** %s **/" % feature_name,
            "#define %sHAVE_%s 1" % (self.conf_c_prefix, feature_name)
        ]
        # 添加特性所需的头文件包含
        prepr += [
            "#include <%s>" % h for h in feature.get("headers", [])
        ]

        # 获取特性的额外定义组并添加到预处理器定义列表
        extra_defs = feature.get("group", [])
        extra_defs += self.feature_extra_checks(feature_name)
        for edef in extra_defs:
            # 防止额外定义与其他特性冲突，进行宏定义保护
            prepr += [
                "#ifndef %sHAVE_%s" % (self.conf_c_prefix, edef),
                "\t#define %sHAVE_%s 1" % (self.conf_c_prefix, edef),
                "#endif",
            ]

        # 如果需要，根据制表符数量进行右侧对齐
        if tabs > 0:
            prepr = [('\t'*tabs) + l for l in prepr]
        # 返回生成的 C 预处理器定义字符串
        return '\n'.join(prepr)
# 定义一个帮助类，用于解析`CCompilerOpt`的主要参数，同时解析可调度源中的配置语句
class _Parse:
    """A helper class that parsing main arguments of `CCompilerOpt`,
    also parsing configuration statements in dispatch-able sources.
    一个帮助类，用于解析`CCompilerOpt`的主要参数，同时解析可调度源中的配置语句。

    Parameters
    ----------
    cpu_baseline : str or None
        minimal set of required CPU features or special options.
        最小的所需 CPU 特性或特殊选项。

    cpu_dispatch : str or None
        dispatched set of additional CPU features or special options.
        额外的 CPU 特性或特殊选项。

    Special options can be:
        - **MIN**: Enables the minimum CPU features that utilized via `_Config.conf_min_features`
        - **MAX**: Enables all supported CPU features by the Compiler and platform.
        - **NATIVE**: Enables all CPU features that supported by the current machine.
        - **NONE**: Enables nothing
        - **Operand +/-**: remove or add features, useful with options **MAX**, **MIN** and **NATIVE**.
            NOTE: operand + is only added for nominal reason.
    特殊选项可以是：
        - **MIN**：启用通过`_Config.conf_min_features`使用的最小 CPU 特性。
        - **MAX**：启用编译器和平台支持的所有 CPU 特性。
        - **NATIVE**：启用当前计算机支持的所有 CPU 特性。
        - **NONE**：不启用任何特性。
        - **操作符 +/-**：移除或添加特性，与 **MAX**，**MIN** 和**NATIVE** 选项一起使用时非常有用。
            注意：操作符+仅添加了名义上的原因。

    NOTES:
        - Case-insensitive among all CPU features and special options.
        - Comma or space can be used as a separator.
        - If the CPU feature is not supported by the user platform or compiler,
          it will be skipped rather than raising a fatal error.
        - Any specified CPU features to 'cpu_dispatch' will be skipped if its part of CPU baseline features
        - 'cpu_baseline' force enables implied features.
    注意：
        - 与所有 CPU 特性和特殊选项大小写不敏感。
        - 逗号或空格可以用作分隔符。
        - 如果 CPU 特性不受用户平台或编译器支持，它将被跳过而不是引发致命错误。
        - 如果指定的任何 CPU 特性是`cpu_baseline`的一部分，则它指定是`cpu_dispatch`的将被跳过。
        - `cpu_baseline` 强制启用隐含的特性。

    Attributes
    ----------
    parse_baseline_names : list
        Final CPU baseline's feature names(sorted from low to high)
        最终 CPU 基线的特性名称（从低到高排序）。

    parse_baseline_flags : list
        Compiler flags of baseline features
        基线特性的编译器标志。

    parse_dispatch_names : list
        Final CPU dispatch-able feature names(sorted from low to high)
        最终 CPU 可调度特性的特性名称（从低到高排序）。

    parse_target_groups : dict
        Dictionary containing initialized target groups that configured
        through class attribute `conf_target_groups`.

        The key is represent the group name and value is a tuple
        contains three items :
            - bool, True if group has the 'baseline' option.
            - list, list of CPU features.
            - list, list of extra compiler flags.
    parse_target_groups : dict
        包含配置好的初始化目标组的字典，通过类属性`conf_target_groups`进行配置。

        键代表组名，值是一个包含三个项的元组：
            - bool，如果组具有`baseline`选项，则为True。
            - list，CPU 特性列表。
            - list，额外的编译器标志列表。
    """
    # 解析目标 CPU 功能的配置语句
    def parse_targets(self, source):
        """
        Fetch and parse configuration statements that required for
        defining the targeted CPU features, statements should be declared
        in the top of source in between **C** comment and start
        with a special mark **@targets**.
    
        Configuration statements are sort of keywords representing
        CPU features names, group of statements and policies, combined
        together to determine the required optimization.
    
        Parameters
        ----------
        source : str
            the path of **C** source file.
    
        Returns
        -------
        - bool, True if group has the 'baseline' option
        - list, list of CPU features
        - list, list of extra compiler flags
        """
        # 输出日志，查找 '@targets' 关键字
        self.dist_log("looking for '@targets' inside -> ", source)
        # 打开源文件
        with open(source) as fd:
            tokens = ""  # 存放目标 CPU 功能配置语句
            max_to_reach = 1000  # 最大行数
            start_with = "@targets"  # 开始标记
            start_pos = -1  # 开始位置
            end_with = "*/"  # 结束标记
            end_pos = -1  # 结束位置
            for current_line, line in enumerate(fd):
                if current_line == max_to_reach:
                    self.dist_fatal("reached the max of lines")
                    break
                if start_pos == -1:
                    # 查找开始标记位置
                    start_pos = line.find(start_with)
                    if start_pos == -1:
                        continue
                    start_pos += len(start_with)
                tokens += line  # 存储当前行内容
                end_pos = line.find(end_with)  # 查找结束标记位置
                if end_pos != -1:
                    end_pos += len(tokens) - len(line)
                    break
    
        if start_pos == -1:
            self.dist_fatal("expected to find '%s' within a C comment" % start_with)
        if end_pos == -1:
            self.dist_fatal("expected to end with '%s'" % end_with)
    
        tokens = tokens[start_pos:end_pos]  # 截取目标 CPU 功能配置语句
        return self._parse_target_tokens(tokens)
    # 定义正则表达式，用于分割目标 CPU 功能配置语句
    _parse_regex_arg = re.compile(r'\s|,|([+-])')
    # 解析参数特性，验证参数是否为字符串类型
    def _parse_arg_features(self, arg_name, req_features):
        if not isinstance(req_features, str):
            self.dist_fatal("expected a string in '%s'" % arg_name)

        final_features = set()
        # 使用空格和逗号作为分隔符，将字符串分割成列表
        tokens = list(filter(None, re.split(self._parse_regex_arg, req_features)))
        append = True # 默认是追加操作
        for tok in tokens:
            # 检查是否以 '#' 或 '$' 开头，如果是则报错
            if tok[0] in ("#", "$"):
                self.dist_fatal(
                    arg_name, "target groups and policies "
                    "aren't allowed from arguments, "
                    "only from dispatch-able sources"
                )
            # 如果是 '+'，则设置为追加操作并继续下一个循环
            if tok == '+':
                append = True
                continue
            # 如果是 '-'，则设置为不追加操作并继续下一个循环
            if tok == '-':
                append = False
                continue

            TOK = tok.upper() # 内部使用大写
            features_to = set()
            # 如果是 "NONE"，不做任何操作
            if TOK == "NONE":
                pass
            # 如果是 "NATIVE"，则获取本地支持的特性
            elif TOK == "NATIVE":
                native = self.cc_flags["native"]
                if not native:
                    self.dist_fatal(arg_name,
                        "native option isn't supported by the compiler"
                    )
                features_to = self.feature_names(
                    force_flags=native, macros=[("DETECT_FEATURES", 1)]
                )
            # 如果是 "MAX"，则获取所有支持的特性
            elif TOK == "MAX":
                features_to = self.feature_supported.keys()
            # 如果是 "MIN"，则获取最低要求的特性
            elif TOK == "MIN":
                features_to = self.feature_min
            else:
                # 如果是已知的特性，则加入到特性集合中；否则报错
                if TOK in self.feature_supported:
                    features_to.add(TOK)
                else:
                    if not self.feature_is_exist(TOK):
                        self.dist_fatal(arg_name,
                            ", '%s' isn't a known feature or option" % tok
                        )
            # 根据追加操作，合并或移除最终的特性集合
            if append:
                final_features = final_features.union(features_to)
            else:
                final_features = final_features.difference(features_to)

            append = True # 恢复默认值

        return final_features

    # 编译正则表达式，用于解析目标
    _parse_regex_target = re.compile(r'\s|[*,/]|([()])')
    # 解析策略令牌
    def _parse_token_policy(self, token):
        """validate policy token"""
        # 检查策略名称的有效性
        if len(token) <= 1 or token[-1:] == token[0]:
            self.dist_fatal("'$' must stuck in the begin of policy name")
        token = token[1:] # 去掉开头的 '$'
        # 如果策略名称不在已知的策略集合中，则报错
        if token not in self._parse_policies:
            self.dist_fatal(
                "'%s' is an invalid policy name, available policies are" % token,
                self._parse_policies.keys()
            )
        return token
    # 验证并解析组合标记
    def _parse_token_group(self, token, has_baseline, final_targets, extra_flags):
        """validate group token"""
        # 如果标记长度小于等于1，或者最后一个字符与第一个字符相等，抛出错误
        if len(token) <= 1 or token[-1:] == token[0]:
            self.dist_fatal("'#' must stuck in the begin of group name")
    
        # 去掉标记的第一个字符
        token = token[1:]
        # 获取标记对应的目标组、目标、额外标志
        ghas_baseline, gtargets, gextra_flags = self.parse_target_groups.get(
            token, (False, None, [])
        )
        # 如果目标为空，则抛出错误，并列出所有可用目标组
        if gtargets is None:
            self.dist_fatal(
                "'%s' is an invalid target group name, " % token + \
                "available target groups are",
                self.parse_target_groups.keys()
            )
        # 如果该组有基准线，则设置 has_baseline 为真
        if ghas_baseline:
            has_baseline = True
        # 将组内目标加入最终目标列表，保持原有排序
        final_targets += [f for f in gtargets if f not in final_targets]
        # 将组内额外标志加入额外标志列表，保持原有排序
        extra_flags += [f for f in gextra_flags if f not in extra_flags]
        # 返回处理后的结果
        return has_baseline, final_targets, extra_flags
    
    # 验证被括号包围的多个目标
    def _parse_multi_target(self, targets):
        """validate multi targets that defined between parentheses()"""
        # 移除任何暗含的特征，保留原始特征
        if not targets:
            self.dist_fatal("empty multi-target '()'")
        # 如果目标列表中存在无效目标，则抛出错误
        if not all([
            self.feature_is_exist(tar) for tar in targets
        ]) :
            self.dist_fatal("invalid target name in multi-target", targets)
        # 如果目标列表中存在不是基准线或分发线的特征，则返回空
        if not all([
            (
                tar in self.parse_baseline_names or
                tar in self.parse_dispatch_names
            )
            for tar in targets
        ]) :
            return None
        # 将目标列表排序，使之可比较
        targets = self.feature_ahead(targets)
        # 如果目标列表为空，则返回空
        if not targets:
            return None
        # 强制排序多个目标，使之可比较
        targets = self.feature_sorted(targets)
        targets = tuple(targets) # 可散列
        return targets
    
    # 跳过所有基准线特征
    def _parse_policy_not_keepbase(self, has_baseline, final_targets, extra_flags):
        """skip all baseline features"""
        skipped = []
        for tar in final_targets[:]:
            is_base = False
            if isinstance(tar, str):
                is_base = tar in self.parse_baseline_names
            else:
                # 多个目标
                is_base = all([
                    f in self.parse_baseline_names
                    for f in tar
                ])
            if is_base:
                skipped.append(tar)
                final_targets.remove(tar)
    
        # 如果有跳过的基准线特征，则记录日志
        if skipped:
            self.dist_log("skip baseline features", skipped)
    
        # 返回处理后的结果
        return has_baseline, final_targets, extra_flags
    # 解析保持排序策略，将通知记录在日志中，然后返回处理结果
    def _parse_policy_keepsort(self, has_baseline, final_targets, extra_flags):
        """leave a notice that $keep_sort is on"""
        self.dist_log(
            "policy 'keep_sort' is on, dispatch-able targets", final_targets, "\n"
            "are 'not' sorted depend on the highest interest but"
            "as specified in the dispatch-able source or the extra group"
        )
        return has_baseline, final_targets, extra_flags

    # 解析不保持排序策略，根据最高兴趣度对最终目标进行排序，然后返回处理结果
    def _parse_policy_not_keepsort(self, has_baseline, final_targets, extra_flags):
        """sorted depend on the highest interest"""
        final_targets = self.feature_sorted(final_targets, reverse=True)
        return has_baseline, final_targets, extra_flags

    # 解析最大优化策略，尝试追加编译器优化标志，最后返回处理结果
    def _parse_policy_maxopt(self, has_baseline, final_targets, extra_flags):
        """append the compiler optimization flags"""
        if self.cc_has_debug:
            self.dist_log("debug mode is detected, policy 'maxopt' is skipped.")
        elif self.cc_noopt:
            self.dist_log("optimization is disabled, policy 'maxopt' is skipped.")
        else:
            flags = self.cc_flags["opt"]
            if not flags:
                self.dist_log(
                    "current compiler doesn't support optimization flags, "
                    "policy 'maxopt' is skipped", stderr=True
                )
            else:
                extra_flags += flags
        return has_baseline, final_targets, extra_flags

    # 解析错误作为警告处理策略，尝试追加编译器错误作为警告标志，最后返回处理结果
    def _parse_policy_werror(self, has_baseline, final_targets, extra_flags):
        """force warnings to treated as errors"""
        flags = self.cc_flags["werror"]
        if not flags:
            self.dist_log(
                "current compiler doesn't support werror flags, "
                "warnings will 'not' treated as errors", stderr=True
            )
        else:
            self.dist_log("compiler warnings are treated as errors")
            extra_flags += flags
        return has_baseline, final_targets, extra_flags

    # 解析自动向量化支持策略，跳过编译器不支持的特性，最后返回处理结果
    def _parse_policy_autovec(self, has_baseline, final_targets, extra_flags):
        """skip features that has no auto-vectorized support by compiler"""
        skipped = []
        for tar in final_targets[:]:
            if isinstance(tar, str):
                can = self.feature_can_autovec(tar)
            else: # multiple target
                can = all([
                    self.feature_can_autovec(t)
                    for t in tar
                ])
            if not can:
                final_targets.remove(tar)
                skipped.append(tar)

        if skipped:
            self.dist_log("skip non auto-vectorized features", skipped)

        return has_baseline, final_targets, extra_flags
# 定义名为CCompilerOpt的类，它集成了_Config、_Distutils、_Cache、_CCompiler、_Feature和_Parse类
class CCompilerOpt(_Config, _Distutils, _Cache, _CCompiler, _Feature, _Parse):
    """
    A helper class for `CCompiler` aims to provide extra build options
    to effectively control of compiler optimizations that are directly
    related to CPU features.
    """
    # 初始化方法，接受ccompiler、cpu_baseline、cpu_dispatch和cache_path这几个参数
    def __init__(self, ccompiler, cpu_baseline="min", cpu_dispatch="max", cache_path=None):
        # 调用_Config类的初始化方法
        _Config.__init__(self)
        # 调用_Distutils类的初始化方法，传入ccompiler参数
        _Distutils.__init__(self, ccompiler)
        # 调用_Cache类的初始化方法，传入cache_path、dist_info()、cpu_baseline和cpu_dispatch参数
        _Cache.__init__(self, cache_path, self.dist_info(), cpu_baseline, cpu_dispatch)
        # 调用_CCompiler类的初始化方法
        _CCompiler.__init__(self)
        # 调用_Feature类的初始化方法
        _Feature.__init__(self)
        # 如果cc_noopt为假且cc_has_native为真，输出警告日志
        if not self.cc_noopt and self.cc_has_native:
            self.dist_log(
                "native flag is specified through environment variables. "
                "force cpu-baseline='native'"
            )
            cpu_baseline = "native"
        # 调用_Parse类的初始化方法，传入cpu_baseline和cpu_dispatch参数
        _Parse.__init__(self, cpu_baseline, cpu_dispatch)
        # 保存请求的基线特性和分发特性，用于后续报告和跟踪目的
        self._requested_baseline = cpu_baseline
        self._requested_dispatch = cpu_dispatch
        # 创建一个字典，key为可分配的源，value为包含两个项（has_baseline[布尔值]，dispatched-features[列表]）的元组
        self.sources_status = getattr(self, "sources_status", {})
        # 每个实例应该有自己独立的cache_private成员
        self.cache_private.add("sources_status")
        # 在初始化类之后设置它，确保在初始化后进行cache写入
        self.hit_cache = hasattr(self, "hit_cache")

    # 判断是否从缓存文件中加载了该类，返回True或False
    def is_cached(self):
        """
        Returns True if the class loaded from the cache file
        """
        return self.cache_infile and self.hit_cache

    # 返回最终的CPU基线编译器标志列表
    def cpu_baseline_flags(self):
        """
        Returns a list of final CPU baseline compiler flags
        """
        return self.parse_baseline_flags

    # 返回最终的CPU基线特性名称列表
    def cpu_baseline_names(self):
        """
        return a list of final CPU baseline feature names
        """
        return self.parse_baseline_names

    # 返回最终的CPU分发特性名称列表
    def cpu_dispatch_names(self):
        """
        return a list of final CPU dispatch feature names
        """
        return self.parse_dispatch_names
        # 将输出目录、源文件路径、目标文件名进行包装
        def _wrap_target(self, output_dir, dispatch_src, target, nochange=False):
            # 断言，确保目标参数是字符串或元组类型
            assert(isinstance(target, (str, tuple)))
            # 如果目标参数是字符串
            if isinstance(target, str):
                ext_name = target_name = target
            else:
                # 如果目标参数是元组，表示多个目标
                ext_name = '.'.join(target)
                target_name = '__'.join(target)

            # 构造包装后的路径名
            wrap_path = os.path.join(output_dir, os.path.basename(dispatch_src))
            wrap_path = "{0}.{2}{1}".format(*os.path.splitext(wrap_path), ext_name.lower())
            # 如果不改变并且包装后的文件已存在，则直接返回路径
            if nochange and os.path.exists(wrap_path):
                return wrap_path

            # 在日志中记录包装后的目标路径
            self.dist_log("wrap dispatch-able target -> ", wrap_path)
            # 对特性进行排序以便阅读
            features = self.feature_sorted(self.feature_implies_c(target))
            target_join = "#define %sCPU_TARGET_" % self.conf_c_prefix_
            target_defs = [target_join + f for f in features]
            target_defs = '\n'.join(target_defs)

            # 打开包装后的路径，写入自动生成的代码
            with open(wrap_path, "w") as fd:
                fd.write(textwrap.dedent("""\
                /**
                 * AUTOGENERATED DON'T EDIT
                 * Please make changes to the code generator (distutils/ccompiler_opt.py)
                 */
                #define {pfx}CPU_TARGET_MODE
                #define {pfx}CPU_TARGET_CURRENT {target_name}
                {target_defs}
                #include "{path}"
                """).format(
                    pfx=self.conf_c_prefix_, target_name=target_name,
                    path=os.path.abspath(dispatch_src), target_defs=target_defs
                ))
            # 返回包装后的路径
            return wrap_path
    def _generate_config(self, output_dir, dispatch_src, targets, has_baseline=False):
        # 从 dispatch_src 中提取配置文件名，替换后缀为 '.h'，然后与 output_dir 组合成完整路径
        config_path = os.path.basename(dispatch_src)
        config_path = os.path.splitext(config_path)[0] + '.h'
        config_path = os.path.join(output_dir, config_path)
        
        # 计算当前 targets 和 has_baseline 的缓存哈希值
        cache_hash = self.cache_hash(targets, has_baseline)
        
        try:
            # 尝试打开配置文件，读取其中的 cache_hash 值，如果匹配则返回 True
            with open(config_path) as f:
                last_hash = f.readline().split("cache_hash:")
                if len(last_hash) == 2 and int(last_hash[1]) == cache_hash:
                    return True
        except OSError:
            pass
        
        # 如果文件不存在，创建文件所在的目录
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # 输出生成的配置文件路径到日志
        self.dist_log("generate dispatched config -> ", config_path)
        
        dispatch_calls = []
        for tar in targets:
            if isinstance(tar, str):
                target_name = tar
            else:  # 多目标情况下，将目标名称用双下划线连接
                target_name = '__'.join([t for t in tar])
            
            # 执行特征检测，生成检测条件字符串
            req_detect = self.feature_detect(tar)
            req_detect = '&&'.join([
                "CHK(%s)" % f for f in req_detect
            ])
            
            # 构建 dispatch_calls 列表，格式化为需要的宏定义形式
            dispatch_calls.append(
                "\t%sCPU_DISPATCH_EXPAND_(CB((%s), %s, __VA_ARGS__))" % (
                self.conf_c_prefix_, req_detect, target_name
            ))
        
        # 将 dispatch_calls 列表合并为字符串，用换行符连接
        dispatch_calls = ' \\\n'.join(dispatch_calls)
        
        # 根据 has_baseline 决定是否设置 baseline_calls
        if has_baseline:
            baseline_calls = (
                "\t%sCPU_DISPATCH_EXPAND_(CB(__VA_ARGS__))"
            ) % self.conf_c_prefix_
        else:
            baseline_calls = ''
        
        # 将生成的配置内容写入到 config_path 文件中
        with open(config_path, "w") as fd:
            fd.write(textwrap.dedent("""\
            // cache_hash:{cache_hash}
            /**
             * AUTOGENERATED DON'T EDIT
             * Please make changes to the code generator (distutils/ccompiler_opt.py)
             */
            #ifndef {pfx}CPU_DISPATCH_EXPAND_
                #define {pfx}CPU_DISPATCH_EXPAND_(X) X
            #endif
            #undef {pfx}CPU_DISPATCH_BASELINE_CALL
            #undef {pfx}CPU_DISPATCH_CALL
            #define {pfx}CPU_DISPATCH_BASELINE_CALL(CB, ...) \\
            {baseline_calls}
            #define {pfx}CPU_DISPATCH_CALL(CHK, CB, ...) \\
            {dispatch_calls}
            """).format(
                pfx=self.conf_c_prefix_, baseline_calls=baseline_calls,
                dispatch_calls=dispatch_calls, cache_hash=cache_hash
            ))
        
        # 返回 False 表示生成配置文件过程中未复用现有文件
        return False
# 创建一个新的 CCompilerOpt 实例，并生成分发头文件
def new_ccompiler_opt(compiler, dispatch_hpath, **kwargs):
    """
    Create a new instance of 'CCompilerOpt' and generate the dispatch header
    which contains the #definitions and headers of platform-specific instruction-sets for
    the enabled CPU baseline and dispatch-able features.

    Parameters
    ----------
    compiler : CCompiler instance
        编译器实例，用于编译优化
    dispatch_hpath : str
        分发头文件的路径，用于存储生成的平台特定指令集定义和头文件
    **kwargs: passed as-is to `CCompilerOpt(...)`
        其余的参数传递给 `CCompilerOpt` 构造函数

    Returns
    -------
    new instance of CCompilerOpt
        返回一个新的 CCompilerOpt 实例
    """
    # 使用传入的编译器实例和其他参数，创建一个 CCompilerOpt 对象
    opt = CCompilerOpt(compiler, **kwargs)
    
    # 检查分发头文件是否已经存在，或者是否需要重新生成
    if not os.path.exists(dispatch_hpath) or not opt.is_cached():
        # 如果分发头文件不存在，或者 CCompilerOpt 缓存无效，则生成新的分发头文件
        opt.generate_dispatch_header(dispatch_hpath)
    
    # 返回创建的 CCompilerOpt 对象
    return opt
```