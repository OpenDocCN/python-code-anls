# `.\numpy\numpy\distutils\tests\test_ccompiler_opt.py`

```
import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError

# 确定是否独立运行
is_standalone = __name__ == '__main__' and __package__ is None

# 如果是独立运行，导入必要的模块和函数，并添加路径
if is_standalone:
    import unittest, contextlib, tempfile, shutil
    sys.path.append(path.abspath(path.join(path.dirname(__file__), "..")))
    from ccompiler_opt import CCompilerOpt

    # 定义临时目录上下文管理器，用于创建临时目录并在结束时清理
    @contextlib.contextmanager
    def tempdir(*args, **kwargs):
        tmpdir = tempfile.mkdtemp(*args, **kwargs)
        try:
            yield tmpdir
        finally:
            shutil.rmtree(tmpdir)

    # 断言函数，如果表达式不成立则抛出断言错误
    def assert_(expr, msg=''):
        if not expr:
            raise AssertionError(msg)

# 如果不是独立运行，则从相应的模块导入必要的函数和类
else:
    from numpy.distutils.ccompiler_opt import CCompilerOpt
    from numpy.testing import assert_, tempdir

# 定义要测试的不同架构和编译器的字典
arch_compilers = dict(
    x86 = ("gcc", "clang", "icc", "iccw", "msvc"),
    x64 = ("gcc", "clang", "icc", "iccw", "msvc"),
    ppc64 = ("gcc", "clang"),
    ppc64le = ("gcc", "clang"),
    armhf = ("gcc", "clang"),
    aarch64 = ("gcc", "clang", "fcc"),
    s390x = ("gcc", "clang"),
    noarch = ("gcc",)
)

# 定义一个虚拟的C编译器优化类，继承自CCompilerOpt
class FakeCCompilerOpt(CCompilerOpt):
    fake_info = ""

    # 初始化方法，接受陷阱文件和陷阱标志作为参数
    def __init__(self, trap_files="", trap_flags="", *args, **kwargs):
        self.fake_trap_files = trap_files
        self.fake_trap_flags = trap_flags
        CCompilerOpt.__init__(self, None, **kwargs)

    # 返回虚假的编译器优化信息的字符串表示
    def __repr__(self):
        return textwrap.dedent("""\
            <<<<
            march    : {}
            compiler : {}
            ----------------
            {}
            >>>>
        """).format(self.cc_march, self.cc_name, self.report())

    # 模拟编译方法，接受源文件列表和标志列表作为参数，并返回假对象
    def dist_compile(self, sources, flags, **kwargs):
        assert(isinstance(sources, list))
        assert(isinstance(flags, list))
        # 如果设置了假陷阱文件，检查源文件是否匹配，如果匹配则报错
        if self.fake_trap_files:
            for src in sources:
                if re.match(self.fake_trap_files, src):
                    self.dist_error("source is trapped by a fake interface")
        # 如果设置了假陷阱标志，检查标志是否匹配，如果匹配则报错
        if self.fake_trap_flags:
            for f in flags:
                if re.match(self.fake_trap_flags, f):
                    self.dist_error("flag is trapped by a fake interface")
        # 返回源文件和标志的zip对象
        return zip(sources, [' '.join(flags)] * len(sources))

    # 返回虚假的编译器优化信息
    def dist_info(self):
        return FakeCCompilerOpt.fake_info

    # 静态方法，用于记录编译器优化的信息，这里不做任何操作
    @staticmethod
    def dist_log(*args, stderr=False):
        pass

# 测试CCompilerOpt类的辅助类
class _Test_CCompilerOpt:
    arch = None  # 架构，默认为x86_64
    cc   = None  # 编译器，默认为gcc

    # 设置类的初始化方法，设置FakeCCompilerOpt的conf_nocache为True
    def setup_class(self):
        FakeCCompilerOpt.conf_nocache = True
        self._opt = None

    # 返回虚拟编译器优化实例的方法，接受任意参数
    def nopt(self, *args, **kwargs):
        FakeCCompilerOpt.fake_info = (self.arch, self.cc, "")
        return FakeCCompilerOpt(*args, **kwargs)

    # 返回虚拟编译器优化实例的方法，如果实例不存在则创建一个
    def opt(self):
        if not self._opt:
            self._opt = self.nopt()
        return self._opt

    # 返回当前虚拟编译器优化实例的架构信息
    def march(self):
        return self.opt().cc_march

    # 返回当前虚拟编译器优化实例的编译器名称信息
    def cc_name(self):
        return self.opt().cc_name
    # 设置编译器的目标和组，作为 FakeCCompilerOpt 类的类属性
    FakeCCompilerOpt.conf_target_groups = groups
    # 初始化编译器选项对象 opt，设置各种参数
    opt = self.nopt(
        cpu_baseline=kwargs.get("baseline", "min"),  # 设置 CPU 基准，默认为 "min"
        cpu_dispatch=kwargs.get("dispatch", "max"),  # 设置 CPU 分发，默认为 "max"
        trap_files=kwargs.get("trap_files", ""),  # 设置陷阱文件，默认为空字符串
        trap_flags=kwargs.get("trap_flags", "")  # 设置陷阱标志，默认为空字符串
    )
    # 在临时目录下创建文件 'test_targets.c'，写入参数 targets 的内容
    with tempdir() as tmpdir:
        file = os.path.join(tmpdir, "test_targets.c")
        with open(file, 'w') as f:
            f.write(targets)
        # 初始化 gtargets 和 gflags 列表和字典
        gtargets = []
        gflags = {}
        # 使用 opt 对象尝试编译 file 文件，获取编译结果 fake_objects
        fake_objects = opt.try_dispatch([file])
        # 遍历 fake_objects 中的每个源文件和标志
        for source, flags in fake_objects:
            # 从源文件路径中提取目标名称，转换为大写形式
            gtar = path.basename(source).split('.')[1:-1]
            glen = len(gtar)
            if glen == 0:
                gtar = "baseline"  # 若目标名称为空，则使用 "baseline"
            elif glen == 1:
                gtar = gtar[0].upper()  # 若目标名称为一个，则转换为大写形式
            else:
                # 将多个目标名称转换为括号形式的字符串，转换为大写形式
                gtar = ('('+' '.join(gtar)+')').upper()
            gtargets.append(gtar)  # 将目标名称添加到 gtargets 列表
            gflags[gtar] = flags  # 将目标名称与其对应的标志添加到 gflags 字典

    # 获取编译器返回的文件是否包含基准版本和编译的目标列表
    has_baseline, targets = opt.sources_status[file]
    # 若存在基准版本，则将 "baseline" 添加到目标列表中
    targets = targets + ["baseline"] if has_baseline else targets
    # 将目标列表中的元组表示的多个目标转换为括号形式的字符串
    targets = [
        '('+' '.join(tar)+')' if isinstance(tar, tuple) else tar
        for tar in targets
    ]
    # 如果编译返回的目标列表与 gtargets 不一致，抛出 AssertionError 异常
    if len(targets) != len(gtargets) or not all(t in gtargets for t in targets):
        raise AssertionError(
            "'sources_status' returns different targets than the compiled targets\n"
            "%s != %s" % (targets, gtargets)
        )
    # 返回编译器返回的目标列表和 gflags 字典
    return targets, gflags

# 根据参数设置的架构和编译器名称映射，返回与之对应的正则表达式
def arg_regex(self, **kwargs):
    map2origin = dict(
        x64 = "x86",  # 将 'x64' 映射为 'x86'
        ppc64le = "ppc64",  # 将 'ppc64le' 映射为 'ppc64'
        aarch64 = "armhf",  # 将 'aarch64' 映射为 'armhf'
        clang = "gcc",  # 将 'clang' 映射为 'gcc'
    )
    # 调用 self.march() 和 self.cc_name() 方法获取架构和编译器名称
    march = self.march(); cc_name = self.cc_name()
    # 根据 map2origin 映射表获取架构和编译器名称的原始名称
    map_march = map2origin.get(march, march)
    map_cc = map2origin.get(cc_name, cc_name)
    # 遍历指定的架构和编译器名称及其映射组合，查找 kwargs 中匹配的正则表达式
    for key in (
        march, cc_name, map_march, map_cc,
        march + '_' + cc_name,
        map_march + '_' + cc_name,
        march + '_' + map_cc,
        map_march + '_' + map_cc,
    ) :
        regex = kwargs.pop(key, None)  # 从 kwargs 中获取指定的正则表达式参数
        if regex is not None:
            break
    if regex:
        if isinstance(regex, dict):
            # 如果 regex 是字典类型，则遍历每个键值对，确保值以 '$' 结尾
            for k, v in regex.items():
                if v[-1:] not in ')}$?\\.+*':
                    regex[k] = v + '$'
        else:
            assert(isinstance(regex, str))
            # 如果 regex 是字符串类型，则确保其以 '$' 结尾
            if regex[-1:] not in ')}$?\\.+*':
                regex += '$'
    return regex  # 返回匹配的正则表达式
    # 定义一个方法用于验证期望的 CPU 分发特性是否符合预期
    def expect(self, dispatch, baseline="", **kwargs):
        # 从参数中获取匹配模式
        match = self.arg_regex(**kwargs)
        # 如果没有匹配模式，则返回
        if match is None:
            return
        
        # 创建一个特性选项对象，传入基准线和分发特性，以及可能的陷阱文件和标志
        opt = self.nopt(
            cpu_baseline=baseline, cpu_dispatch=dispatch,
            trap_files=kwargs.get("trap_files", ""),
            trap_flags=kwargs.get("trap_flags", "")
        )
        
        # 获取 CPU 分发特性的名称，并用空格连接起来
        features = ' '.join(opt.cpu_dispatch_names())
        
        # 如果没有匹配模式，则判断特性列表是否为空，若不为空则抛出断言错误
        if not match:
            if len(features) != 0:
                raise AssertionError(
                    'expected empty features, not "%s"' % features
                )
            return
        
        # 使用正则表达式检查 CPU 分发特性列表是否符合期望的模式
        if not re.match(match, features, re.IGNORECASE):
            raise AssertionError(
                'dispatch features "%s" not match "%s"' % (features, match)
            )

    # 定义一个方法用于验证期望的 CPU 基准线特性是否符合预期
    def expect_baseline(self, baseline, dispatch="", **kwargs):
        # 从参数中获取匹配模式
        match = self.arg_regex(**kwargs)
        # 如果没有匹配模式，则返回
        if match is None:
            return
        
        # 创建一个特性选项对象，传入基准线和分发特性，以及可能的陷阱文件和标志
        opt = self.nopt(
            cpu_baseline=baseline, cpu_dispatch=dispatch,
            trap_files=kwargs.get("trap_files", ""),
            trap_flags=kwargs.get("trap_flags", "")
        )
        
        # 获取 CPU 基准线特性的名称，并用空格连接起来
        features = ' '.join(opt.cpu_baseline_names())
        
        # 如果没有匹配模式，则判断特性列表是否为空，若不为空则抛出断言错误
        if not match:
            if len(features) != 0:
                raise AssertionError(
                    'expected empty features, not "%s"' % features
                )
            return
        
        # 使用正则表达式检查 CPU 基准线特性列表是否符合期望的模式
        if not re.match(match, features, re.IGNORECASE):
            raise AssertionError(
                'baseline features "%s" not match "%s"' % (features, match)
            )

    # 定义一个方法用于验证期望的 CPU 标志是否符合预期
    def expect_flags(self, baseline, dispatch="", **kwargs):
        # 从参数中获取匹配模式
        match = self.arg_regex(**kwargs)
        # 如果没有匹配模式，则返回
        if match is None:
            return
        
        # 创建一个特性选项对象，传入基准线和分发特性，以及可能的陷阱文件和标志
        opt = self.nopt(
            cpu_baseline=baseline, cpu_dispatch=dispatch,
            trap_files=kwargs.get("trap_files", ""),
            trap_flags=kwargs.get("trap_flags", "")
        )
        
        # 获取 CPU 基准线标志的名称，并用空格连接起来
        flags = ' '.join(opt.cpu_baseline_flags())
        
        # 如果没有匹配模式，则判断标志列表是否为空，若不为空则抛出断言错误
        if not match:
            if len(flags) != 0:
                raise AssertionError(
                    'expected empty flags not "%s"' % flags
                )
            return
        
        # 使用正则表达式检查 CPU 基准线标志列表是否符合期望的模式
        if not re.match(match, flags):
            raise AssertionError(
                'flags "%s" not match "%s"' % (flags, match)
            )

    # 定义一个方法用于验证期望的目标是否符合预期
    def expect_targets(self, targets, groups={}, **kwargs):
        # 从参数中获取匹配模式
        match = self.arg_regex(**kwargs)
        # 如果没有匹配模式，则返回
        if match is None:
            return
        
        # 获取目标列表和其他相关数据
        targets, _ = self.get_targets(targets=targets, groups=groups, **kwargs)
        # 将目标列表用空格连接起来
        targets = ' '.join(targets)
        
        # 如果没有匹配模式，则判断目标列表是否为空，若不为空则抛出断言错误
        if not match:
            if len(targets) != 0:
                raise AssertionError(
                    'expected empty targets, not "%s"' % targets
                )
            return
        
        # 使用正则表达式检查目标列表是否符合期望的模式
        if not re.match(match, targets, re.IGNORECASE):
            raise AssertionError(
                'targets "%s" not match "%s"' % (targets, match)
            )
    # 检查预期的目标标志是否存在于给定的目标列表中，并验证其是否匹配预期的正则表达式模式
    def expect_target_flags(self, targets, groups={}, **kwargs):
        # 使用传入的关键字参数生成匹配字典
        match_dict = self.arg_regex(**kwargs)
        # 如果生成的匹配字典为空，则直接返回
        if match_dict is None:
            return
        # 断言生成的匹配字典确实是一个字典对象
        assert(isinstance(match_dict, dict))
        # 获取目标和对应的标志信息
        _, tar_flags = self.get_targets(targets=targets, groups=groups)

        # 遍历匹配字典中的每个目标和其对应的标志
        for match_tar, match_flags in match_dict.items():
            # 如果匹配的目标不在目标标志中，则抛出断言错误
            if match_tar not in tar_flags:
                raise AssertionError(
                    'expected to find target "%s"' % match_tar
                )
            # 获取目标对应的实际标志信息
            flags = tar_flags[match_tar]
            # 如果预期的标志为空，但实际标志不为空，则抛出断言错误
            if not match_flags:
                if len(flags) != 0:
                    raise AssertionError(
                        'expected to find empty flags in target "%s"' % match_tar
                    )
            # 如果预期的标志与实际标志不匹配，则抛出断言错误
            if not re.match(match_flags, flags):
                raise AssertionError(
                    '"%s" flags "%s" not match "%s"' % (match_tar, flags, match_flags)
                )

    # 测试接口函数，验证特定的属性和条件是否为真
    def test_interface(self):
        # 根据当前架构判断错误的架构名称
        wrong_arch = "ppc64" if self.arch != "ppc64" else "x86"
        # 根据当前编译器判断错误的编译器名称
        wrong_cc   = "clang" if self.cc   != "clang" else "icc"
        # 调用opt函数获取优化选项对象
        opt = self.opt()
        # 断言特定属性存在
        assert_(getattr(opt, "cc_on_" + self.arch))
        # 断言特定属性不存在
        assert_(not getattr(opt, "cc_on_" + wrong_arch))
        # 断言特定属性存在
        assert_(getattr(opt, "cc_is_" + self.cc))
        # 断言特定属性不存在
        assert_(not getattr(opt, "cc_is_" + wrong_cc))

    # 测试参数为空的情况
    def test_args_empty(self):
        # 遍历预定义的基线和调度组合
        for baseline, dispatch in (
            ("", "none"),
            (None, ""),
            ("none +none", "none - none"),
            ("none -max", "min - max"),
            ("+vsx2 -VSX2", "vsx avx2 avx512f -max"),
            ("max -vsx - avx + avx512f neon -MAX ",
             "min -min + max -max -vsx + avx2 -avx2 +NONE")
        ) :
            # 调用nopt函数生成优化选项对象
            opt = self.nopt(cpu_baseline=baseline, cpu_dispatch=dispatch)
            # 断言CPU基线名称列表为空
            assert(len(opt.cpu_baseline_names()) == 0)
            # 断言CPU调度名称列表为空
            assert(len(opt.cpu_dispatch_names()) == 0)

    # 测试参数验证的情况
    def test_args_validation(self):
        # 如果当前架构为"unknown"，则直接返回，不进行参数验证
        if self.march() == "unknown":
            return
        # 遍历预定义的基线和调度组合
        for baseline, dispatch in (
            ("unkown_feature - max +min", "unknown max min"), # unknowing features
            ("#avx2", "$vsx") # groups and polices aren't acceptable
        ) :
            try:
                # 尝试调用nopt函数生成优化选项对象，预期会抛出DistutilsError异常
                self.nopt(cpu_baseline=baseline, cpu_dispatch=dispatch)
                # 如果未抛出异常，则抛出断言错误，提示预期会得到一个无效参数的异常
                raise AssertionError("excepted an exception for invalid arguments")
            except DistutilsError:
                pass
    def test_skip(self):
        # 只使用平台支持的特性，忽略不支持的特性，不抛出异常
        self.expect(
            "sse vsx neon",
            x86="sse", ppc64="vsx", armhf="neon", unknown=""
        )
        self.expect(
            "sse41 avx avx2 vsx2 vsx3 neon_vfpv4 asimd",
            x86   = "sse41 avx avx2",
            ppc64 = "vsx2 vsx3",
            armhf = "neon_vfpv4 asimd",
            unknown = ""
        )
        # 如果特性包含在基线特性中，则在cpu_dispatch中的任何特性都必须被忽略
        self.expect(
            "sse neon vsx", baseline="sse neon vsx",
            x86="", ppc64="", armhf=""
        )
        self.expect(
            "avx2 vsx3 asimdhp", baseline="avx2 vsx3 asimdhp",
            x86="", ppc64="", armhf=""
        )

    def test_implies(self):
        # 基线特性结合了隐含的特性，我们依赖它而不直接测试'feature_implies()'
        self.expect_baseline(
            "fma3 avx2 asimd vsx3",
            # .* 在两个空格之间可以验证中间的特性
            x86   = "sse .* sse41 .* fma3.*avx2",
            ppc64 = "vsx vsx2 vsx3",
            armhf = "neon neon_fp16 neon_vfpv4 asimd"
        )
        """
        special cases
        """
        # 在icc和msvc中，FMA3和AVX2不能分开
        # 它们需要互相隐含，AVX512F和CD同理
        for f0, f1 in (
            ("fma3",    "avx2"),
            ("avx512f", "avx512cd"),
        ):
            diff = ".* sse42 .* %s .*%s$" % (f0, f1)
            self.expect_baseline(f0,
                x86_gcc=".* sse42 .* %s$" % f0,
                x86_icc=diff, x86_iccw=diff
            )
            self.expect_baseline(f1,
                x86_gcc=".* avx .* %s$" % f1,
                x86_icc=diff, x86_iccw=diff
            )
        # 在msvc中，以下特性也不能分开
        for f in (("fma3", "avx2"), ("avx512f", "avx512cd", "avx512_skx")):
            for ff in f:
                self.expect_baseline(ff,
                    x86_msvc=".*%s" % ' '.join(f)
                )

        # 在ppc64le中，VSX和VSX2不能分开
        self.expect_baseline("vsx", ppc64le="vsx vsx2")
        # 在aarch64中，以下特性也不能分开
        for f in ("neon", "neon_fp16", "neon_vfpv4", "asimd"):
            self.expect_baseline(f, aarch64="neon neon_fp16 neon_vfpv4 asimd")
    def test_args_options(self):
        # 定义一个测试方法，用于测试不同的参数选项

        # 对于参数选项 "max" 和 "native"
        for o in ("max", "native"):
            # 如果 o 是 "native" 并且当前编译器是 "msvc"，则跳过本次循环
            if o == "native" and self.cc_name() == "msvc":
                continue
            
            # 调用 self.expect 方法，期望满足以下条件
            self.expect(o,
                trap_files=".*cpu_(sse|vsx|neon|vx).c",
                x86="", ppc64="", armhf="", s390x=""
            )
            # 再次调用 self.expect 方法，期望满足以下条件
            self.expect(o,
                trap_files=".*cpu_(sse3|vsx2|neon_vfpv4|vxe).c",
                x86="sse sse2", ppc64="vsx", armhf="neon neon_fp16",
                aarch64="", ppc64le="", s390x="vx"
            )
            # 再次调用 self.expect 方法，期望满足以下条件
            self.expect(o,
                trap_files=".*cpu_(popcnt|vsx3).c",
                x86="sse .* sse41", ppc64="vsx vsx2",
                armhf="neon neon_fp16 .* asimd .*",
                s390x="vx vxe vxe2"
            )
            # 再次调用 self.expect 方法，期望满足以下条件
            self.expect(o,
                x86_gcc=".* xop fma4 .* avx512f .* avx512_knl avx512_knm avx512_skx .*",
                # 在 icc 中，不支持 xop 和 fma4
                x86_icc=".* avx512f .* avx512_knl avx512_knm avx512_skx .*",
                x86_iccw=".* avx512f .* avx512_knl avx512_knm avx512_skx .*",
                # 在 msvc 中，不支持 avx512_knl 和 avx512_knm
                x86_msvc=".* xop fma4 .* avx512f .* avx512_skx .*",
                armhf=".* asimd asimdhp asimddp .*",
                ppc64="vsx vsx2 vsx3 vsx4.*",
                s390x="vx vxe vxe2.*"
            )
        
        # 对于参数选项 "min"
        self.expect("min",
            x86="sse sse2", x64="sse sse2 sse3",
            armhf="", aarch64="neon neon_fp16 .* asimd",
            ppc64="", ppc64le="vsx vsx2", s390x=""
        )
        # 再次调用 self.expect 方法，期望满足以下条件
        self.expect(
            "min", trap_files=".*cpu_(sse2|vsx2).c",
            x86="", ppc64le=""
        )
        
        # 当启用 "native" 标志但不支持时，必须触发异常
        # 通过参数激活选项 "native" 时，必须触发异常
        try:
            self.expect("native",
                trap_flags=".*(-march=native|-xHost|/QxHost|-mcpu=a64fx).*",
                x86=".*", ppc64=".*", armhf=".*", s390x=".*", aarch64=".*",
            )
            # 如果 self.march() 不是 "unknown"，则引发 AssertionError 异常
            if self.march() != "unknown":
                raise AssertionError(
                    "excepted an exception for %s" % self.march()
                )
        except DistutilsError:
            # 如果 self.march() 是 "unknown"，则引发 AssertionError 异常
            if self.march() == "unknown":
                raise AssertionError("excepted no exceptions")
    # 定义测试方法 test_flags，用于测试不同体系结构下的编译标志
    def test_flags(self):
        # 预期返回不同体系结构下的编译标志，参数为支持的体系结构及其相应标志
        self.expect_flags(
            "sse sse2 vsx vsx2 neon neon_fp16 vx vxe",
            x86_gcc="-msse -msse2", x86_icc="-msse -msse2",
            x86_iccw="/arch:SSE2",
            x86_msvc="/arch:SSE2" if self.march() == "x86" else "",
            ppc64_gcc= "-mcpu=power8",
            ppc64_clang="-mcpu=power8",
            armhf_gcc="-mfpu=neon-fp16 -mfp16-format=ieee",
            aarch64="",
            s390x="-mzvector -march=arch12"
        )
        
        # 再次测试 normalize -march 的用法
        self.expect_flags(
            "asimd",
            aarch64="",
            armhf_gcc=r"-mfp16-format=ieee -mfpu=neon-fp-armv8 -march=armv8-a\+simd"
        )
        
        # 测试 asimdhp 的编译标志
        self.expect_flags(
            "asimdhp",
            aarch64_gcc=r"-march=armv8.2-a\+fp16",
            armhf_gcc=r"-mfp16-format=ieee -mfpu=neon-fp-armv8 -march=armv8.2-a\+fp16"
        )
        
        # 测试 asimddp 的编译标志，仅适用于 aarch64_gcc
        self.expect_flags(
            "asimddp", aarch64_gcc=r"-march=armv8.2-a\+dotprod"
        )
        
        # 测试 asimdfhm 的编译标志，依赖于 asimdhp
        self.expect_flags(
            "asimdfhm", aarch64_gcc=r"-march=armv8.2-a\+fp16\+fp16fml"
        )
        
        # 测试同时使用 asimddp、asimdhp 和 asimdfhm 的编译标志
        self.expect_flags(
            "asimddp asimdhp asimdfhm",
            aarch64_gcc=r"-march=armv8.2-a\+dotprod\+fp16\+fp16fml"
        )
        
        # 测试 vx、vxe 和 vxe2 的编译标志，仅适用于 s390x
        self.expect_flags(
            "vx vxe vxe2",
            s390x=r"-mzvector -march=arch13"
        )

    # 定义测试方法 test_targets_exceptions，用于测试异常情况下的目标处理
    def test_targets_exceptions(self):
        # 遍历不同的异常目标情况
        for targets in (
            "bla bla", "/*@targets",
            "/*@targets */",
            "/*@targets unknown */",
            "/*@targets $unknown_policy avx2 */",
            "/*@targets #unknown_group avx2 */",
            "/*@targets $ */",
            "/*@targets # vsx */",
            "/*@targets #$ vsx */",
            "/*@targets vsx avx2 ) */",
            "/*@targets vsx avx2 (avx2 */",
            "/*@targets vsx avx2 () */",
            "/*@targets vsx avx2 ($autovec) */", # no features
            "/*@targets vsx avx2 (xxx) */",
            "/*@targets vsx avx2 (baseline) */",
        ):
            try:
                # 预期对给定异常目标抛出异常，参数为各体系结构的预期空字符串
                self.expect_targets(
                    targets,
                    x86="", armhf="", ppc64="", s390x=""
                )
                # 如果当前体系结构不是 unknown，则抛出断言错误
                if self.march() != "unknown":
                    raise AssertionError(
                        "excepted an exception for %s" % self.march()
                    )
            except DistutilsError:
                # 如果当前体系结构是 unknown，则抛出断言错误
                if self.march() == "unknown":
                    raise AssertionError("excepted no exceptions")
    # 定义一个测试方法，用于测试目标语法的多个变体
    def test_targets_syntax(self):
        # 遍历多个目标语法的字符串
        for targets in (
            "/*@targets $keep_baseline sse vsx neon vx*/",
            "/*@targets,$keep_baseline,sse,vsx,neon vx*/",
            "/*@targets*$keep_baseline*sse*vsx*neon*vx*/",
            """
            /*
            ** @targets
            ** $keep_baseline, sse vsx,neon, vx
            */
            """,
            """
            /*
            ************@targets****************
            ** $keep_baseline, sse vsx, neon, vx
            ************************************
            */
            """,
            """
            /*
            /////////////@targets/////////////////
            //$keep_baseline//sse//vsx//neon//vx
            /////////////////////////////////////
            */
            """,
            """
            /*
            @targets
            $keep_baseline
            SSE VSX NEON VX*/
            """
        ):
            # 调用测试方法，验证目标语法的解析情况
            self.expect_targets(targets,
                x86="sse", ppc64="vsx", armhf="neon", s390x="vx", unknown=""
            )
    def test_targets(self):
        # 测试跳过基线特性
        self.expect_targets(
            """
            /*@targets
                sse sse2 sse41 avx avx2 avx512f
                vsx vsx2 vsx3 vsx4
                neon neon_fp16 asimdhp asimddp
                vx vxe vxe2
            */
            """,
            baseline="avx vsx2 asimd vx vxe",  # 设置基线特性
            x86="avx512f avx2",  # x86 平台支持的特性
            armhf="asimddp asimdhp",  # ARM 平台支持的特性
            ppc64="vsx4 vsx3",  # PPC64 平台支持的特性
            s390x="vxe2"  # s390x 平台支持的特性
        )
        # 测试跳过非分发特性
        self.expect_targets(
            """
            /*@targets
                sse41 avx avx2 avx512f
                vsx2 vsx3 vsx4
                asimd asimdhp asimddp
                vx vxe vxe2
            */
            """,
            baseline="",  # 设置基线特性为空
            dispatch="sse41 avx2 vsx2 asimd asimddp vxe2",  # 设置分发特性
            x86="avx2 sse41",  # x86 平台支持的特性
            armhf="asimddp asimd",  # ARM 平台支持的特性
            ppc64="vsx2",  # PPC64 平台支持的特性
            s390x="vxe2"  # s390x 平台支持的特性
        )
        # 测试跳过不支持的特性
        self.expect_targets(
            """
            /*@targets
                sse2 sse41 avx2 avx512f
                vsx2 vsx3 vsx4
                neon asimdhp asimddp
                vx vxe vxe2
            */
            """,
            baseline="",  # 设置基线特性为空
            trap_files=".*(avx2|avx512f|vsx3|vsx4|asimddp|vxe2).c",  # 设置陷阱文件规则
            x86="sse41 sse2",  # x86 平台支持的特性
            ppc64="vsx2",  # PPC64 平台支持的特性
            armhf="asimdhp neon",  # ARM 平台支持的特性
            s390x="vxe vx"  # s390x 平台支持的特性
        )
        # 测试跳过互斥的特性
        self.expect_targets(
            """
            /*@targets
                sse sse2 avx fma3 avx2 avx512f avx512cd
                vsx vsx2 vsx3
                neon neon_vfpv4 neon_fp16 neon_fp16 asimd asimdhp
                asimddp asimdfhm
            */
            """,
            baseline="",  # 设置基线特性为空
            x86_gcc="avx512cd avx512f avx2 fma3 avx sse2",  # x86 平台 GCC 编译器支持的特性
            x86_msvc="avx512cd avx2 avx sse2",  # x86 平台 MSVC 编译器支持的特性
            x86_icc="avx512cd avx2 avx sse2",  # x86 平台 ICC 编译器支持的特性
            x86_iccw="avx512cd avx2 avx sse2",  # x86 平台 ICCW 编译器支持的特性
            ppc64="vsx3 vsx2 vsx",  # PPC64 平台支持的特性
            ppc64le="vsx3 vsx2",  # PPC64LE 平台支持的特性
            armhf="asimdfhm asimddp asimdhp asimd neon_vfpv4 neon_fp16 neon",  # ARM 平台支持的特性
            aarch64="asimdfhm asimddp asimdhp asimd"  # AArch64 平台支持的特性
        )
    def test_targets_policies(self):
        # 定义测试方法 test_targets_policies
        self.expect_targets(
            """
            /*@targets
                $keep_baseline
                sse2 sse42 avx2 avx512f
                vsx2 vsx3
                neon neon_vfpv4 asimd asimddp
                vx vxe vxe2
            */
            """,
            # 调用 expect_targets 方法，验证目标与策略
            baseline="sse41 avx2 vsx2 asimd vsx3 vxe",
            x86="avx512f avx2 sse42 sse2",
            ppc64="vsx3 vsx2",
            armhf="asimddp asimd neon_vfpv4 neon",
            # neon, neon_vfpv4, asimd 互为推论
            aarch64="asimddp asimd",
            s390x="vxe2 vxe vx"
        )
        # 'keep_sort', 保留排序不变
        self.expect_targets(
            """
            /*@targets
                $keep_baseline $keep_sort
                avx512f sse42 avx2 sse2
                vsx2 vsx3
                asimd neon neon_vfpv4 asimddp
                vxe vxe2
            */
            """,
            # 调用 expect_targets 方法，验证目标与策略
            x86="avx512f sse42 avx2 sse2",
            ppc64="vsx2 vsx3",
            armhf="asimd neon neon_vfpv4 asimddp",
            # neon, neon_vfpv4, asimd 互为推论
            aarch64="asimd asimddp",
            s390x="vxe vxe2"
        )
        # 'autovec', 跳过无法由编译器向量化的特性
        self.expect_targets(
            """
            /*@targets
                $keep_baseline $keep_sort $autovec
                avx512f avx2 sse42 sse41 sse2
                vsx3 vsx2
                asimddp asimd neon_vfpv4 neon
            */
            """,
            # 调用 expect_targets 方法，验证目标与策略
            x86_gcc="avx512f avx2 sse42 sse41 sse2",
            x86_icc="avx512f avx2 sse42 sse41 sse2",
            x86_iccw="avx512f avx2 sse42 sse41 sse2",
            x86_msvc="avx512f avx2 sse2" if self.march() == 'x86' else "avx512f avx2",
            ppc64="vsx3 vsx2",
            armhf="asimddp asimd neon_vfpv4 neon",
            # neon, neon_vfpv4, asimd 互为推论
            aarch64="asimddp asimd"
        )
        for policy in ("$maxopt", "$autovec"):
            # 'maxopt' 和 'autovec' 设置最大可接受的优化标志
            self.expect_target_flags(
                "/*@targets baseline %s */" % policy,
                gcc={"baseline":".*-O3.*"}, icc={"baseline":".*-O3.*"},
                iccw={"baseline":".*/O3.*"}, msvc={"baseline":".*/O2.*"},
                unknown={"baseline":".*"}
            )

        # 'werror', 强制编译器将警告视为错误
        self.expect_target_flags(
            "/*@targets baseline $werror */",
            gcc={"baseline":".*-Werror.*"}, icc={"baseline":".*-Werror.*"},
            iccw={"baseline":".*/Werror.*"}, msvc={"baseline":".*/WX.*"},
            unknown={"baseline":".*"}
        )
    # 定义测试方法：验证目标和组
    def test_targets_groups(self):
        # 预期的目标设定和组定义
        self.expect_targets(
            """
            /*@targets $keep_baseline baseline #test_group */
            """,
            # 定义组，并给出其成员
            groups=dict(
                test_group=("""
                    $keep_baseline
                    asimddp sse2 vsx2 avx2 vsx3
                    avx512f asimdhp
                """)
            ),
            # 各架构的默认目标设定
            x86="avx512f avx2 sse2 baseline",
            ppc64="vsx3 vsx2 baseline",
            armhf="asimddp asimdhp baseline"
        )
        
        # 测试跳过重复项和排序功能
        self.expect_targets(
            """
            /*@targets
             * sse42 avx avx512f
             * #test_group_1
             * vsx2
             * #test_group_2
             * asimddp asimdfhm
            */
            """,
            # 定义多个组，每个组包含不同的目标设定
            groups=dict(
                test_group_1=("""
                    VSX2 vsx3 asimd avx2 SSE41
                """),
                test_group_2=("""
                    vsx2 vsx3 asImd aVx2 sse41
                """)
            ),
            # 各架构的默认目标设定
            x86="avx512f avx2 avx sse42 sse41",
            ppc64="vsx3 vsx2",
            # ppc64le 的默认基线包含 vsx2
            ppc64le="vsx3",
            armhf="asimdfhm asimddp asimd",
            # aarch64 的默认基线包含 asimd
            aarch64="asimdfhm asimddp"
        )

    # 测试多目标设定
    def test_targets_multi(self):
        self.expect_targets(
            """
            /*@targets
                (avx512_clx avx512_cnl) (asimdhp asimddp)
            */
            """,
            # x86 和 armhf 架构的目标设定
            x86=r"\(avx512_clx avx512_cnl\)",
            armhf=r"\(asimdhp asimddp\)",
        )
        
        # 测试跳过隐含特性并自动排序
        self.expect_targets(
            """
            /*@targets
                f16c (sse41 avx sse42) (sse3 avx2 avx512f)
                vsx2 (vsx vsx3 vsx2)
                (neon neon_vfpv4 asimd asimdhp asimddp)
            */
            """,
            # 各架构的目标设定
            x86="avx512f f16c avx",
            ppc64="vsx3 vsx2",
            # ppc64le 的默认基线包含 vsx2
            ppc64le="vsx3",
            armhf=r"\(asimdhp asimddp\)",
        )
        
        # 测试跳过隐含特性并保持排序
        self.expect_targets(
            """
            /*@targets $keep_sort
                (sse41 avx sse42) (sse3 avx2 avx512f)
                (vsx vsx3 vsx2)
                (asimddp neon neon_vfpv4 asimd asimdhp)
                (vx vxe vxe2)
            */
            """,
            # 各架构的目标设定
            x86="avx avx512f",
            ppc64="vsx3",
            armhf=r"\(asimdhp asimddp\)",
            s390x="vxe2"
        )
        
        # 测试编译器的多样性和避免重复
        self.expect_targets(
            """
            /*@targets $keep_sort
                fma3 avx2 (fma3 avx2) (avx2 fma3) avx2 fma3
            */
            """,
            # 不同编译器下的目标设定
            x86_gcc=r"fma3 avx2 \(fma3 avx2\)",
            x86_icc="avx2", x86_iccw="avx2",
            x86_msvc="avx2"
        )
# 定义函数 new_test，用于生成测试类的字符串表示
def new_test(arch, cc):
    # 检查是否是独立运行模式，如果是直接返回一个字符串模板，包含类定义和初始化方法
    if is_standalone: return textwrap.dedent("""\
    class TestCCompilerOpt_{class_name}(_Test_CCompilerOpt, unittest.TestCase):
        arch = '{arch}'
        cc   = '{cc}'
        def __init__(self, methodName="runTest"):
            unittest.TestCase.__init__(self, methodName)
            self.setup_class()
    """).format(
        class_name=arch + '_' + cc, arch=arch, cc=cc
    )
    # 如果不是独立运行模式，返回另一种字符串模板，只包含类定义
    return textwrap.dedent("""\
    class TestCCompilerOpt_{class_name}(_Test_CCompilerOpt):
        arch = '{arch}'
        cc   = '{cc}'
    """).format(
        class_name=arch + '_' + cc, arch=arch, cc=cc
    )

# 如果条件为真（1 and is_standalone），执行以下代码块
if 1 and is_standalone:
    # 设置 FakeCCompilerOpt 类的 fake_info 属性为 "x86_icc"
    FakeCCompilerOpt.fake_info = "x86_icc"
    # 创建 FakeCCompilerOpt 实例 cco，使用 None 和 "avx2" 作为参数
    cco = FakeCCompilerOpt(None, cpu_baseline="avx2")
    # 输出 cco 的 cpu_baseline_names 方法返回的结果，以空格分隔
    print(' '.join(cco.cpu_baseline_names()))
    # 输出 cco 的 cpu_baseline_flags 方法返回的结果
    print(cco.cpu_baseline_flags())
    # 运行单元测试框架的主函数
    unittest.main()
    # 系统退出
    sys.exit()

# 遍历 arch_compilers 字典的键值对
for arch, compilers in arch_compilers.items():
    # 遍历 compilers 列表中的每个编译器 cc
    for cc in compilers:
        # 使用 exec 函数执行 new_test 函数返回的字符串，生成对应的测试类
        exec(new_test(arch, cc))

# 如果是独立运行模式，运行单元测试框架的主函数
if is_standalone:
    unittest.main()
```